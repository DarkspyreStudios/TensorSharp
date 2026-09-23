// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Whole-graph Qwen-Image-2.1 regression and explicitly synthetic timing probe.
// Backend selection is process-wide: --write-reference on CPU produces the
// explicit-attention reference consumed by CUDA or Metal --reference.
// These small synthetic weights test numerical/lifetime behavior, not image
// quality or the throughput of a downloaded, quantized full model.
#include "ggml_ops_qwen_image21.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
const char* TSGgml_GetLastError();
int TSGgml_IsBackendAvailable(int backend_type);
int TSGgml_QwenImage21Forward(const TSGQi21Desc* desc);
void TSGgml_QwenImage21ResetForwardCache();
void TSGgml_ClearHostBufferCache();
void TSGgml_InvalidateHostBuffer(void* ptr);
void TSGgml_ReleaseReuseComputeBuffers();
void TSGgml_SetDeviceCopyBudget(std::int64_t bytes);
std::int64_t TSGgml_DeviceCopyCacheResidentBytes();
void TSGgml_Shutdown();
}

namespace {
constexpr int head_dim = 128;
constexpr int channels = 16;
constexpr int text_dim = 96;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void env(const char* name, const char* value) {
#ifdef _WIN32
    require(_putenv_s(name, value) == 0, "cannot set environment variable");
#else
    require(setenv(name, value, 1) == 0, "cannot set environment variable");
#endif
}
struct Rng {
    std::uint64_t state;
    explicit Rng(std::uint64_t seed = 0x9e3779b97f4a7c15ull) : state(seed) {}
    float next(float scale) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        return (static_cast<float>((state >> 40) & 0xffff) / 32768.0f - 1.0f) * scale;
    }
};
struct Arena {
    std::vector<std::vector<float>> buffers;
    float* allocate(std::size_t n, Rng& rng, float scale, float offset = 0.f) {
        buffers.emplace_back(n);
        for (float& v : buffers.back()) v = offset + rng.next(scale);
        return buffers.back().data();
    }
    TSGQi21Weight weight(int in, int out, Rng& rng) {
        TSGQi21Weight w{};
        w.data = allocate(static_cast<std::size_t>(in) * out, rng, 0.8f / std::sqrt(float(in)));
        w.type = 0; // GGML_TYPE_F32. The benchmark prints this limitation.
        w.ne0 = in; w.ne1 = out; w.bytes = std::int64_t(in) * out * sizeof(float);
        return w;
    }
};
struct Model {
    Arena arena;
    TSGQi21Desc base{};
    std::vector<TSGQi21Block> fused, separate;
    Model(int dim, int ff, int layers) {
        Rng rng;
        base.struct_bytes = sizeof(base);
        base.dim = dim; base.heads = dim / head_dim; base.head_dim = head_dim;
        base.channels = channels; base.text_dim = text_dim; base.num_layers = layers;
        base.eps = 1e-6f;
        base.image_in = arena.weight(channels, dim, rng);
        base.text_in = arena.weight(text_dim, dim, rng);
        base.text_out = arena.weight(dim, dim, rng);
        base.time_in = arena.weight(256, dim, rng);
        base.time_out = arena.weight(dim, dim, rng);
        base.modulation = arena.weight(dim, dim * 4, rng);
        // Keep attention/text/time contributions large enough that a stale
        // dynamic upload cannot hide under the backend precision tolerance.
        auto modulation = static_cast<float*>(base.modulation.data);
        for (std::int64_t i = 0; i < base.modulation.ne0 * base.modulation.ne1; ++i) modulation[i] *= 8.f;
        base.norm_out = arena.weight(dim, dim, rng);
        base.proj_out = arena.weight(dim, channels, rng);
        base.text_norm = arena.allocate(text_dim, rng, 0.15f);
        fused.resize(layers);
        for (auto& b : fused) {
            b.q = arena.weight(dim, dim, rng); b.k = arena.weight(dim, dim, rng);
            b.v = arena.weight(dim, dim, rng); b.out = arena.weight(dim, dim, rng);
            b.gate = arena.weight(dim, ff * 2, rng); b.down = arena.weight(ff, dim, rng);
            b.norm_q = arena.allocate(head_dim, rng, 0.2f, 1.f);
            b.norm_k = arena.allocate(head_dim, rng, 0.2f, 1.f);
        }
        // Same exact matrix entries in [gate; up] and separate descriptors.
        // This also tests graph-cache keys when the host base pointer is equal
        // but its descriptor shape and byte count change.
        separate = fused;
        for (auto& b : separate) {
            b.gate.ne1 = ff; b.gate.bytes /= 2;
            b.up = b.gate;
            b.up.data = static_cast<float*>(b.gate.data) + std::size_t(dim) * ff;
        }
    }
};
struct Shape {
    const char* name;
    int image_seq, text_seq;
    std::vector<TSGQi21Segment> segments;
};
struct Inputs {
    std::vector<float> images, text, time, cos, sin, output;
    explicit Inputs(const Shape& shape, int variant) {
        const int total = shape.segments.back().end;
        const int target = shape.segments.back().end - shape.segments.back().start;
        images.resize(std::size_t(channels) * shape.image_seq);
        text.resize(std::size_t(text_dim) * shape.text_seq);
        time.resize(512);
        cos.resize(std::size_t(head_dim / 2) * total);
        sin.resize(cos.size()); output.resize(std::size_t(channels) * target);
        Rng rng(0x123456789abcdefull + std::uint64_t(variant) * 0x100001ull);
        for (auto& v : images) v = rng.next(1.f);
        for (auto& v : text) v = rng.next(0.75f);
        for (int branch = 0; branch < 2; ++branch)
            for (int i = 0; i < 128; ++i) {
                const float t = branch ? 0.f : 400.f + variant * 137.f;
                const float angle = t * std::pow(10000.f, -float(i) / 128.f);
                time[branch * 256 + i] = std::cos(angle);
                time[branch * 256 + 128 + i] = std::sin(angle);
            }
        for (int t = 0; t < total; ++t)
            for (int i = 0; i < head_dim / 2; ++i) {
                const float angle = (t + variant * 0.25f) * std::pow(10000.f, -float(2 * i) / head_dim);
                cos[std::size_t(t) * (head_dim / 2) + i] = std::cos(angle);
                sin[std::size_t(t) * (head_dim / 2) + i] = std::sin(angle);
            }
    }
    TSGQi21Desc descriptor(Model& model, const Shape& shape, bool packed) {
        auto d = model.base;
        d.images = images.data(); d.text = text.data(); d.time_embedding = time.data();
        d.cos = cos.data(); d.sin = sin.data(); d.output = output.data();
        d.blocks = packed ? model.fused.data() : model.separate.data();
        d.segments = shape.segments.data(); d.num_segments = int(shape.segments.size());
        d.image_seq = shape.image_seq; d.text_seq = shape.text_seq;
        d.total_seq = shape.segments.back().end; d.prefix_seq = shape.segments.back().start;
        return d;
    }
    void copy_values_from(const Inputs& source) {
        // Preserve host allocation addresses, as the sampler updates its
        // latent and timestep buffers in place between denoising steps.
        require(images.size() == source.images.size() && text.size() == source.text.size() &&
            time.size() == source.time.size() && cos.size() == source.cos.size(), "input shape mismatch");
        std::copy(source.images.begin(), source.images.end(), images.begin());
        std::copy(source.text.begin(), source.text.end(), text.begin());
        std::copy(source.time.begin(), source.time.end(), time.begin());
        std::copy(source.cos.begin(), source.cos.end(), cos.begin());
        std::copy(source.sin.begin(), source.sin.end(), sin.begin());
    }
};
void forward(Model& model, const Shape& shape, Inputs& input, bool packed) {
    if (std::getenv("TS_QWEN21_TEST_TRACE")) {
        std::fprintf(stderr, "[qwen21-test] forward shape=%s packed=%d reuse=%s flash=%s\n",
            shape.name, int(packed), std::getenv("TS_QWEN21_GRAPH_REUSE"), std::getenv("TS_QWEN21_FLASH"));
        std::fflush(stderr);
    }
    std::fill(input.output.begin(), input.output.end(), std::numeric_limits<float>::quiet_NaN());
    const auto d = input.descriptor(model, shape, packed);
    if (!TSGgml_QwenImage21Forward(&d))
        throw std::runtime_error(std::string("forward failed: ") + TSGgml_GetLastError());
    for (float v : input.output) require(std::isfinite(v), "forward produced a nonfinite output");
}
struct Error {
    double max_absolute = 0, normalized_max = 0, relative_rms = 0;
};
Error compare(const std::vector<float>& actual, const std::vector<float>& expected,
              const std::string& label, double tolerance) {
    require(actual.size() == expected.size() && !actual.empty(), label + ": size mismatch");
    Error error;
    double square_error = 0, square_reference = 0, scale = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), label + ": nonfinite output");
        double delta = double(actual[i]) - expected[i];
        error.max_absolute = std::max(error.max_absolute, std::abs(delta));
        scale = std::max(scale, std::abs(double(expected[i])));
        square_error += delta * delta; square_reference += double(expected[i]) * expected[i];
    }
    error.normalized_max = error.max_absolute / std::max(1e-6, scale);
    error.relative_rms = std::sqrt(square_error / std::max(1e-12, square_reference));
    require(error.normalized_max <= tolerance && error.relative_rms <= tolerance,
        label + ": numerical mismatch, normalized_max=" + std::to_string(error.normalized_max) +
        " relative_rms=" + std::to_string(error.relative_rms));
    return error;
}
void changed(const std::vector<float>& a, const std::vector<float>& b) {
    double delta = 0, scale = 0;
    require(a.size() == b.size(), "changed-input comparison size mismatch");
    for (std::size_t i = 0; i < a.size(); ++i) {
        delta = std::max(delta, std::abs(double(a[i]) - b[i]));
        scale = std::max(scale, std::abs(double(a[i])));
    }
    require(delta > 1e-3 * std::max(scale, 1e-6), "changed inputs returned effectively unchanged output");
}
using Reference = std::vector<std::vector<float>>;
void write_reference(const std::string& path, const Reference& data) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream stream(path, std::ios::binary);
    const std::uint32_t header[] = {0x51323152u, 1u, static_cast<std::uint32_t>(data.size())};
    stream.write(reinterpret_cast<const char*>(header), sizeof(header));
    for (const auto& row : data) {
        auto count = static_cast<std::uint64_t>(row.size());
        stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
        stream.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }
    require(bool(stream), "cannot write reference " + path);
}
Reference read_reference(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    std::uint32_t header[3]{};
    stream.read(reinterpret_cast<char*>(header), sizeof(header));
    require(bool(stream) && header[0] == 0x51323152u && header[1] == 1 && header[2] < 100,
        "invalid reference file " + path);
    Reference data(header[2]);
    for (auto& row : data) {
        std::uint64_t count = 0;
        stream.read(reinterpret_cast<char*>(&count), sizeof(count));
        require(bool(stream) && count > 0 && count < 10000000, "invalid reference size");
        row.resize(static_cast<std::size_t>(count));
        stream.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
    }
    require(bool(stream) && stream.peek() == std::char_traits<char>::eof(), "truncated or trailing reference data");
    return data;
}

struct Options {
    int backend = 2;
    bool benchmark = false;
    int dim = 256, ff = 512, layers = 2, image_seq = 80, text_seq = 17, iterations = 10;
    std::string reference, write_reference;
};
Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "cpu") options.backend = 2;
        else if (arg == "cuda") options.backend = 3;
        else if (arg == "metal") options.backend = 1;
        else if (arg == "--benchmark") options.benchmark = true;
        else {
            require(i + 1 < argc, "missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--reference") options.reference = value;
            else if (arg == "--write-reference") options.write_reference = value;
            else if (arg == "--dim") options.dim = std::stoi(value);
            else if (arg == "--ff") options.ff = std::stoi(value);
            else if (arg == "--layers") options.layers = std::stoi(value);
            else if (arg == "--image-seq") options.image_seq = std::stoi(value);
            else if (arg == "--text-seq") options.text_seq = std::stoi(value);
            else if (arg == "--iterations") options.iterations = std::stoi(value);
            else throw std::runtime_error("unknown argument " + arg);
        }
    }
    require(options.dim > 0 && options.dim % head_dim == 0 && options.dim <= 8192 &&
        options.ff > 0 && options.ff <= 32768 && options.layers > 0 && options.layers <= 64 &&
        options.image_seq > 0 && options.image_seq <= 16384 && options.text_seq > 0 &&
        options.text_seq <= 4096 && options.iterations > 0 && options.iterations <= 1000, "invalid dimensions/counts");
    require(options.backend == 2 || options.write_reference.empty(), "CPU must produce the explicit-attention reference");
    require(options.benchmark || (options.dim == 256 && options.ff == 512 && options.layers == 2),
        "custom model dimensions require --benchmark");
    return options;
}

void regression(const Options& options) {
    Model model(options.dim, options.ff, options.layers);
    const std::vector<Shape> shapes = {
        {"text17-image80", 80, 17, {{0,17,0,0}, {17,97,0,1}}},
        {"text-image-text-image", 61, 17, {{0,11,0,0}, {11,24,0,1}, {24,30,11,0}, {30,78,13,1}}},
        {"changed-shape", 31, 9, {{0,9,0,0}, {9,40,0,1}}},
        {"return-original-shape", 80, 17, {{0,17,0,0}, {17,97,0,1}}},
        {"same-total-new-segments", 80, 17, {{0,9,0,0}, {9,17,0,1}, {17,25,9,0}, {25,97,8,1}}},
        {"zero-prefix", 33, 9, {{0,33,0,1}}},
    };
    Reference reference = options.reference.empty() ? Reference{} : read_reference(options.reference);
    require(reference.empty() || reference.size() == shapes.size() * 2, "reference case count mismatch");
    Reference computed;
    double worst_relative = 0, worst_max = 0;
    int calls = 0;
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        const auto& shape = shapes[index];
        Inputs a(shape, 0), b(shape, 1);
        const Inputs pristine = a;
        env("TS_QWEN21_GRAPH_REUSE", "0"); env("TS_QWEN21_FLASH", "0");
        env("TS_QWEN21_PAD_MASK", "1");
        forward(model, shape, a, true); forward(model, shape, b, true); calls += 2;
        computed.push_back(a.output); computed.push_back(b.output);
        const auto expected_a = reference.empty() ? a.output : reference[index * 2];
        const auto expected_b = reference.empty() ? b.output : reference[index * 2 + 1];
        changed(expected_a, expected_b);
        compare(a.output, expected_a, std::string(shape.name) + " baseline A", 0.003);
        compare(b.output, expected_b, std::string(shape.name) + " baseline B", 0.003);
        env("TS_QWEN21_GRAPH_REUSE", "1"); env("TS_QWEN21_PAD_MASK", "0");
        for (bool flash : {false, true}) for (bool packed : {true, false}) {
            env("TS_QWEN21_FLASH", flash ? "1" : "0");
            const std::string label = std::string(shape.name) + (flash ? " flash" : " explicit") +
                (packed ? " packed" : " separate");
            forward(model, shape, a, packed); ++calls;
            auto error = compare(a.output, expected_a, label + " A", 0.003);
            worst_max = std::max(worst_max, error.normalized_max);
            worst_relative = std::max(worst_relative, error.relative_rms);
            const auto first = a.output;
            forward(model, shape, a, packed); ++calls;
            compare(a.output, first, label + " deterministic repeat", 1e-6);
            a.copy_values_from(b);
            forward(model, shape, a, packed); ++calls;
            compare(a.output, expected_b, label + " in-place mutation", 0.003);
            changed(first, a.output);
            // Keep both allocations alive: a cache must refresh its upload
            // pointers as well as its values when a CFG branch changes.
            forward(model, shape, b, packed); ++calls;
            error = compare(b.output, expected_b, label + " B", 0.003);
            worst_max = std::max(worst_max, error.normalized_max);
            worst_relative = std::max(worst_relative, error.relative_rms);
            a.copy_values_from(pristine);
            Inputs relocated = a;
            forward(model, shape, relocated, packed); ++calls;
            compare(relocated.output, expected_a, label + " relocated A", 0.003);
        }
        // Cached graph references may outlive the resident weight buffers.
        // Both the broad clear and single-key invalidation must be safe.
        TSGgml_ClearHostBufferCache();
        forward(model, shape, a, true); ++calls;
        compare(a.output, expected_a, std::string(shape.name) + " after cache clear", 0.003);
        TSGgml_InvalidateHostBuffer(model.fused[0].q.data);
        forward(model, shape, a, true); ++calls;
        compare(a.output, expected_a, std::string(shape.name) + " after weight invalidation", 0.003);
        TSGgml_QwenImage21ResetForwardCache();
        forward(model, shape, a, true); ++calls;
        compare(a.output, expected_a, std::string(shape.name) + " after graph reset", 0.003);
        TSGgml_ReleaseReuseComputeBuffers();
        forward(model, shape, a, true); ++calls;
        compare(a.output, expected_a, std::string(shape.name) + " after compute-buffer release", 0.003);
        if (index == 0) {
            auto reject = [](const TSGQi21Desc* bad) {
                require(TSGgml_QwenImage21Forward(bad) == 0, "invalid descriptor was accepted");
                const char* error = TSGgml_GetLastError();
                require(error && *error, "invalid descriptor supplied no error");
            };
            reject(nullptr);
            auto bad = a.descriptor(model, shape, true);
            --bad.struct_bytes; reject(&bad);
            bad = a.descriptor(model, shape, true);
            bad.eps = std::numeric_limits<float>::quiet_NaN(); reject(&bad);
            bad = a.descriptor(model, shape, true);
            auto bad_segments = shape.segments;
            ++bad_segments.back().end;
            bad.segments = bad_segments.data(); reject(&bad);
            forward(model, shape, a, true); ++calls;
            compare(a.output, expected_a, "recovery after invalid descriptors", 0.003);
            // The final projection is linear, so scaling its host weights
            // provides an exact changed-weight oracle without a second graph.
            // Invalidation must retire both its resident copy and baked graph.
            auto projection = static_cast<float*>(model.base.proj_out.data);
            const auto count = static_cast<std::size_t>(model.base.proj_out.ne0 * model.base.proj_out.ne1);
            const std::vector<float> original(projection, projection + count);
            auto scaled_expected = expected_a;
            for (std::size_t i = 0; i < count; ++i) projection[i] *= 1.125f;
            for (auto& v : scaled_expected) v *= 1.125f;
            TSGgml_InvalidateHostBuffer(projection);
            forward(model, shape, a, true); ++calls;
            compare(a.output, scaled_expected, "changed projection after invalidation", 0.003);
            changed(expected_a, a.output);
            std::copy(original.begin(), original.end(), projection);
            TSGgml_InvalidateHostBuffer(projection);
            forward(model, shape, a, true); ++calls;
            compare(a.output, expected_a, "restored projection after invalidation", 0.003);
        }
        std::printf("PASS %-25s total=%d prefix=%d\n", shape.name,
            shape.segments.back().end, shape.segments.back().start);
    }
    if (options.backend != 1) {
        // Force the graph-owned constant path. A zero budget disables the cap;
        // one byte positively refuses every model weight in the device cache.
        // Metal maps host weights directly, so a device-copy cap cannot force
        // this path there; do not report that scenario as Metal coverage.
        // INPUT alone is insufficient: gallocr can reuse a constant's slot
        // after its last consumer and corrupt later executions of the graph.
        TSGgml_ClearHostBufferCache();
        TSGgml_ReleaseReuseComputeBuffers();
        TSGgml_SetDeviceCopyBudget(1);
        env("TS_QWEN21_GRAPH_REUSE", "1"); env("TS_QWEN21_FLASH", "1");
        env("TS_QWEN21_PAD_MASK", "0");
        const auto& expected = reference.empty() ? computed : reference;
        Inputs a(shapes[0], 0), b(shapes[0], 1), c(shapes[1], 0), d(shapes[1], 1);
        struct Replay { std::size_t shape, reference; Inputs* input; };
        const Replay replays[] = {
            {0,0,&a}, {0,1,&b}, {0,0,&a}, {1,2,&c},
            {1,3,&d}, {1,2,&c}, {0,0,&a}, {1,2,&c}
        };
        for (const auto& replay : replays) {
            forward(model, shapes[replay.shape], *replay.input, true); ++calls;
            compare(replay.input->output, expected[replay.reference], "streamed constants/masks replay", 0.003);
            require(TSGgml_DeviceCopyCacheResidentBytes() == 0,
                "tiny budget did not force graph-owned model constants");
        }
        TSGgml_ReleaseReuseComputeBuffers();
        forward(model, shapes[0], a, true); ++calls;
        compare(a.output, expected[0], "streamed constants after compute-buffer release", 0.003);
        auto projection = static_cast<float*>(model.base.proj_out.data);
        const auto count = static_cast<std::size_t>(model.base.proj_out.ne0 * model.base.proj_out.ne1);
        const std::vector<float> original(projection, projection + count);
        auto scaled_expected = expected[0];
        for (std::size_t i = 0; i < count; ++i) projection[i] *= 1.125f;
        for (auto& v : scaled_expected) v *= 1.125f;
        // This invalidation must also retire graph-owned copies even though
        // there is no corresponding resident device-cache entry to remove.
        TSGgml_InvalidateHostBuffer(projection);
        forward(model, shapes[0], a, true); ++calls;
        compare(a.output, scaled_expected, "streamed changed weight after invalidation", 0.003);
        std::copy(original.begin(), original.end(), projection);
        TSGgml_InvalidateHostBuffer(projection);
        forward(model, shapes[0], a, true); ++calls;
        compare(a.output, expected[0], "streamed restored weight after invalidation", 0.003);
        require(TSGgml_DeviceCopyCacheResidentBytes() == 0, "streamed replay unexpectedly populated device cache");
        TSGgml_ClearHostBufferCache();
        TSGgml_SetDeviceCopyBudget(0);
        forward(model, shapes[0], a, true); ++calls;
        compare(a.output, expected[0], "return to resident weights", 0.003);
        std::printf("PASS forced one-byte weight budget: 12 forwards, constants/masks survive replay, invalidation and release\n");
    }
    if (!options.write_reference.empty()) write_reference(options.write_reference, computed);
    std::printf("PASS %d whole-graph forwards; max normalized error %.6g, relative RMS %.6g; CPU reference=%s\n",
        calls, worst_max, worst_relative, options.backend == 2 ? "generated" : reference.empty() ? "NOT SUPPLIED" : "verified");
    TSGgml_QwenImage21ResetForwardCache();
    TSGgml_ClearHostBufferCache();
}

void benchmark(const Options& options) {
    Model model(options.dim, options.ff, options.layers);
    Shape shape{"synthetic-benchmark", options.image_seq, options.text_seq,
        {{0,options.text_seq,0,0}, {options.text_seq,options.text_seq + options.image_seq,0,1}}};
    Inputs input(shape, 0);
    std::printf("SYNTHETIC F32 WEIGHTS; dim=%d ff=%d layers=%d head_dim=%d image=%d text=%d iterations=%d\n",
        options.dim, options.ff, options.layers, head_dim, options.image_seq, options.text_seq, options.iterations);
    std::printf("Times include host uploads, graph compute/synchronization and output download; exclude weight generation.\n");
    std::vector<float> expected;
    for (bool reuse : {false, true}) {
        env("TS_QWEN21_GRAPH_REUSE", reuse ? "1" : "0");
        env("TS_QWEN21_FLASH", "1"); env("TS_QWEN21_PAD_MASK", "0");
        TSGgml_QwenImage21ResetForwardCache();
        // Charge only the current mode's activation arena, even when the
        // baseline generic allocator previously reserved a large buffer.
        TSGgml_ReleaseReuseComputeBuffers();
        auto begin = std::chrono::steady_clock::now();
        forward(model, shape, input, true);
        const double cold = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        if (expected.empty()) expected = input.output;
        else compare(input.output, expected, "benchmark reuse vs rebuild", 1e-6);
        // Several warmups let upstream CUDA capture/instantiate its executable.
        for (int i = 0; i < 4; ++i) forward(model, shape, input, true);
        std::vector<double> times;
        for (int i = 0; i < options.iterations; ++i) {
            begin = std::chrono::steady_clock::now();
            forward(model, shape, input, true);
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
            compare(input.output, expected, "benchmark repeat", 1e-6);
        }
        std::sort(times.begin(), times.end());
        double sum = 0; for (double t : times) sum += t;
        const double median = times.size() % 2 ? times[times.size() / 2] :
            (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2;
        std::printf("reuse=%d first_ms=%.3f mean_ms=%.3f median_ms=%.3f min_ms=%.3f max_ms=%.3f\n",
            int(reuse), cold, sum / times.size(), median, times.front(), times.back());
    }
    TSGgml_QwenImage21ResetForwardCache();
    TSGgml_ClearHostBufferCache();
}
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const Options options = parse(argc, argv);
        const char* backend_name = options.backend == 1 ? "metal" : options.backend == 3 ? "cuda" : "cpu";
        if (!TSGgml_IsBackendAvailable(options.backend)) {
            std::printf("SKIP: %s backend unavailable: %s\n", backend_name, TSGgml_GetLastError());
            TSGgml_Shutdown();
            return 77;
        }
        std::printf("backend=%s\n", backend_name);
        if (options.benchmark) benchmark(options);
        else regression(options);
        TSGgml_Shutdown();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        TSGgml_Shutdown();
        return 1;
    }
}
