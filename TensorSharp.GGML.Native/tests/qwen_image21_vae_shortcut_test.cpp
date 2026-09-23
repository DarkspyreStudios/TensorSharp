// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Whole-interpreter checks for Qwen-Image-2.1's temporal/spatial VAE shortcuts.
// The independent reference materializes packed channels, including the zero
// temporal prefix, instead of using native reshape/permute/reduction operations.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

extern "C" {
struct TSGVaeWeightRef { void* data; std::int64_t bytes; };
struct TSGVaeOp {
    std::int32_t kind, w, b, oc, ic, kh, kw, sh, sw, pt, pb, pl, pr, src, dst, aux;
};
struct TSGgmlQwenVaeDesc {
    void* input; std::int32_t in_w, in_h, in_c;
    void* output; std::int64_t out_len;
    const TSGVaeOp* ops; std::int32_t num_ops;
    const TSGVaeWeightRef* weights; std::int32_t num_weights;
    std::int32_t struct_bytes;
};
struct TSGgmlConv2dDesc {
    void* input; std::int32_t W, H, C;
    void* weight; std::int32_t wtype, KW, KH, IC, OC;
    std::int64_t weight_bytes;
    void* bias;
    void* output;
    std::int32_t strideW, strideH, padL, padR, padT, padB;
    std::int32_t struct_bytes;
};
int TSGgml_IsBackendAvailable(int backend_type);
int TSGgml_QwenVaeRun(const TSGgmlQwenVaeDesc* desc);
int TSGgml_Conv2dF32(const TSGgmlConv2dDesc* desc);
const char* TSGgml_GetLastError();
void TSGgml_ReleaseReuseComputeBuffers();
void TSGgml_Shutdown();
int TSGgml_RecreateBackend();
}

namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
struct Feature {
    int c, h, w;
    std::vector<float> values;
    Feature(int channels, int height, int width) : c(channels), h(height), w(width),
        values(static_cast<std::size_t>(channels) * height * width) {}
    float& at(int channel, int y, int x) { return values[(std::size_t(channel) * h + y) * w + x]; }
    float at(int channel, int y, int x) const { return values[(std::size_t(channel) * h + y) * w + x]; }
};
Feature down_reference(const Feature& input, int output_channels, int time, int spatial) {
    Feature output(output_channels, input.h / spatial, input.w / spatial);
    const int packed_channels = input.c * time * spatial * spatial;
    const int group = packed_channels / output_channels;
    std::vector<float> packed(packed_channels);
    for (int y = 0; y < output.h; ++y) for (int x = 0; x < output.w; ++x) {
        int p = 0;
        for (int c = 0; c < input.c; ++c) for (int t = 0; t < time; ++t)
            for (int dy = 0; dy < spatial; ++dy) for (int dx = 0; dx < spatial; ++dx)
                packed[p++] = t == time - 1 ? input.at(c, y * spatial + dy, x * spatial + dx) : 0.f;
        for (int c = 0; c < output_channels; ++c) {
            double sum = 0;
            for (int g = 0; g < group; ++g) sum += packed[c * group + g];
            output.at(c, y, x) = static_cast<float>(sum / group);
        }
    }
    return output;
}
Feature up_reference(const Feature& input, int output_channels, int time, int spatial) {
    Feature output(output_channels, input.h * spatial, input.w * spatial);
    const int packed_channels = output_channels * time * spatial * spatial;
    const int repeats = packed_channels / input.c;
    std::vector<float> repeated(packed_channels);
    for (int y = 0; y < input.h; ++y) for (int x = 0; x < input.w; ++x) {
        int p = 0;
        for (int c = 0; c < input.c; ++c) for (int r = 0; r < repeats; ++r)
            repeated[p++] = input.at(c, y, x);
        p = 0;
        for (int c = 0; c < output_channels; ++c) for (int t = 0; t < time; ++t)
            for (int dy = 0; dy < spatial; ++dy) for (int dx = 0; dx < spatial; ++dx) {
                if (t == time - 1) output.at(c, y * spatial + dy, x * spatial + dx) = repeated[p];
                ++p;
            }
    }
    return output;
}
TSGVaeOp shortcut(int kind, int channels, int time, int spatial) {
    TSGVaeOp op{};
    op.kind = kind; op.oc = channels; op.kh = time; op.kw = spatial;
    op.w = op.b = -1;
    return op;
}
TSGgmlQwenVaeDesc descriptor(Feature& input, std::vector<float>& output, const std::vector<TSGVaeOp>& ops) {
    TSGgmlQwenVaeDesc desc{};
    desc.input = input.values.data(); desc.in_w = input.w; desc.in_h = input.h; desc.in_c = input.c;
    desc.output = output.data(); desc.out_len = output.size();
    desc.ops = ops.data(); desc.num_ops = static_cast<int>(ops.size());
    desc.struct_bytes = sizeof(desc);
    return desc;
}
void check(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& label) {
    require(actual.size() == expected.size(), label + ": size mismatch");
    double maximum = 0, scale = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), label + ": nonfinite output");
        maximum = std::max(maximum, std::abs(double(actual[i]) - expected[i]));
        scale = std::max(scale, std::abs(double(expected[i])));
    }
    require(maximum <= 2e-6 * std::max(1.0, scale), label + ": max error=" + std::to_string(maximum));
}
void run(Feature& input, const Feature& expected, const std::vector<TSGVaeOp>& ops, const std::string& label,
         const std::vector<TSGVaeWeightRef>& weights = {}) {
    std::vector<float> output(expected.values.size(), std::numeric_limits<float>::quiet_NaN());
    auto desc = descriptor(input, output, ops);
    desc.weights = weights.data(); desc.num_weights = static_cast<int>(weights.size());
    if (!TSGgml_QwenVaeRun(&desc))
        throw std::runtime_error(label + ": " + TSGgml_GetLastError());
    check(output, expected.values, label);
}
void literal_cases() {
    Feature down(2, 2, 2);
    down.values = {1,2,3,4,5,6,7,8};
    Feature down_expected(4, 1, 1);
    down_expected.values = {0,2.5f,0,6.5f};
    run(down, down_expected, {shortcut(7,4,2,2)}, "literal zero temporal prefix");
    Feature up(4, 1, 1);
    up.values = {10,20,30,40};
    Feature up_expected(2, 2, 2);
    up_expected.values = {20,20,20,20,40,40,40,40};
    run(up, up_expected, {shortcut(8,2,2,2)}, "literal final temporal sample");
    Feature shuffle(8, 1, 1);
    shuffle.values = {1,2,3,4,5,6,7,8};
    Feature shuffled(2, 2, 2);
    shuffled.values = shuffle.values;
    run(shuffle, shuffled, {shortcut(8,2,1,2)}, "literal spatial channel shuffle");
}
void synthetic_cases() {
    struct Case { int kind, c, h, w, oc, tf, sf; };
    const Case cases[] = {
        {7,3,6,10,6,2,2}, {7,8,4,6,4,1,2}, {7,5,3,7,5,1,1},
        {7,6,5,3,3,2,1}, {7,3,4,6,4,1,2},
        {8,12,3,5,3,1,2}, {8,8,3,5,2,2,2}, {8,3,5,7,6,1,1},
        {8,8,5,3,4,2,2}, {8,8,3,5,4,1,2},
    };
    for (const auto& c : cases) {
        Feature input(c.c, c.h, c.w);
        for (int variant = 0; variant < 2; ++variant) {
            for (std::size_t i = 0; i < input.values.size(); ++i)
                input.values[i] = (static_cast<int>((i * 37 + variant * 53) % 257) - 128) / 16.f;
            auto expected = c.kind == 7 ? down_reference(input,c.oc,c.tf,c.sf) : up_reference(input,c.oc,c.tf,c.sf);
            run(input, expected, {shortcut(c.kind,c.oc,c.tf,c.sf)}, "synthetic shortcut");
        }
    }
    Feature input(4,6,10);
    for (std::size_t i = 0; i < input.values.size(); ++i) input.values[i] = float(i % 131) / 8.f;
    auto expected = up_reference(down_reference(input,4,1,2),4,1,2);
    for (std::size_t i = 0; i < expected.values.size(); ++i) expected.values[i] += input.values[i];
    TSGVaeOp save{}; save.kind = 4; save.dst = 1;
    TSGVaeOp add{}; add.kind = 5; add.aux = 1;
    run(input, expected, {save,shortcut(7,4,1,2),shortcut(8,4,1,2),add}, "saved residual plus down/up composition");
}

void high_range_convolution_cases() {
    // Real Qwen21 VAE intermediates exceed 65504 before channel normalization.
    // F16 im2col overflows before GEMM's accumulation mode has any effect.
    // Preserve both signs and unequal channel magnitudes: clipping the input
    // would produce a different normalized result, even if it stayed finite.
    Feature input(2,3,5), expected(2,3,5);
    std::vector<float> kernel(2 * 2 * 3 * 3, 0.f);
    kernel[(0 * 2 + 0) * 9 + 4] = 1.5f;
    kernel[(0 * 2 + 1) * 9 + 5] = 0.25f;
    kernel[(1 * 2 + 1) * 9 + 4] = -0.75f;
    kernel[(1 * 2 + 0) * 9 + 1] = -0.125f;
    std::vector<float> bias{1024.f,-2048.f}, gamma{1.f,1.25f};
    std::vector<TSGVaeWeightRef> weights{
        {kernel.data(),static_cast<std::int64_t>(kernel.size() * sizeof(float))},
        {bias.data(),static_cast<std::int64_t>(bias.size() * sizeof(float))},
        {gamma.data(),static_cast<std::int64_t>(gamma.size() * sizeof(float))},
    };
    TSGVaeOp conv{};
    conv.kind = 0; conv.w = 0; conv.b = 1; conv.oc = conv.ic = 2;
    conv.kh = conv.kw = 3; conv.sh = conv.sw = 1;
    conv.pt = conv.pb = conv.pl = conv.pr = 1;
    conv.aux = 1; // Qwen21's scoped F32 convolution contract.
    TSGVaeOp norm{}; norm.kind = 1; norm.w = 2;
    for (int variant = 0; variant < 2; ++variant) {
        for (int y = 0; y < input.h; ++y) for (int x = 0; x < input.w; ++x) {
            const int position = y * input.w + x;
            input.at(0,y,x) = 70000.f + position * 1024.f + variant * 4096.f;
            input.at(1,y,x) = -140000.f + position * 512.f - variant * 2048.f;
        }
        // Independent scalar spatial convolution, accumulating in double.
        for (int oc = 0; oc < 2; ++oc) for (int y = 0; y < input.h; ++y) for (int x = 0; x < input.w; ++x) {
            double sum = bias[oc];
            for (int ic = 0; ic < 2; ++ic) for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
                const int sy = y + ky - 1, sx = x + kx - 1;
                if (sy >= 0 && sy < input.h && sx >= 0 && sx < input.w)
                    sum += double(input.at(ic,sy,sx)) * kernel[(oc * 2 + ic) * 9 + ky * 3 + kx];
            }
            expected.at(oc,y,x) = static_cast<float>(sum);
        }
        std::vector<float> output(expected.values.size(),std::numeric_limits<float>::quiet_NaN());
        TSGgmlConv2dDesc desc{};
        desc.input = input.values.data(); desc.W = input.w; desc.H = input.h; desc.C = input.c;
        desc.weight = kernel.data(); desc.wtype = 0; desc.KW = desc.KH = 3; desc.IC = desc.OC = 2;
        desc.weight_bytes = kernel.size() * sizeof(float); desc.bias = bias.data(); desc.output = output.data();
        desc.strideW = desc.strideH = 1; desc.padL = desc.padR = desc.padT = desc.padB = 1;
        desc.struct_bytes = sizeof(desc);
        if (!TSGgml_Conv2dF32(&desc)) throw std::runtime_error(std::string("F32 convolution: ") + TSGgml_GetLastError());
        check(output,expected.values,"standalone convolution preserves values above F16 range");
        require(*std::max_element(output.begin(),output.end()) > 65504.f, "convolution output was clipped to F16 range");
        run(input,expected,{conv},"fused convolution preserves values above F16 range",weights);
        auto normalized = expected;
        for (int y = 0; y < input.h; ++y) for (int x = 0; x < input.w; ++x) {
            const double a = expected.at(0,y,x), b = expected.at(1,y,x);
            const double inverse_rms = std::sqrt(2.0 / (a * a + b * b + 1e-12));
            normalized.at(0,y,x) = static_cast<float>(a * inverse_rms * gamma[0]);
            normalized.at(1,y,x) = static_cast<float>(b * inverse_rms * gamma[1]);
        }
        run(input,normalized,{conv,norm},"large convolution output followed by channel normalization",weights);
    }
    // Start with ordinary inputs, create >65504 activations in one projection,
    // consume them in another, then normalize. Input-only range checks cannot
    // protect this all-device graph from a narrowing intermediate conversion.
    Feature small(2,2,3), normalized(2,2,3);
    std::vector<float> amplify{4096.f,0.f,0.f,2048.f}, identity{1.f,0.f,0.f,1.f};
    std::vector<float> ones{1.f,1.f};
    weights = {{amplify.data(),16},{identity.data(),16},{ones.data(),8}};
    auto first = conv; first.kh = first.kw = 1; first.pt = first.pb = first.pl = first.pr = 0; first.b = -1;
    auto second = first; second.w = 1;
    for (int y = 0; y < small.h; ++y) for (int x = 0; x < small.w; ++x) {
        small.at(0,y,x) = 20.f + y * small.w + x;
        small.at(1,y,x) = -48.f - 2 * (y * small.w + x);
        const double a = small.at(0,y,x) * 4096.0, b = small.at(1,y,x) * 2048.0;
        const double inverse_rms = std::sqrt(2.0 / (a * a + b * b + 1e-12));
        normalized.at(0,y,x) = static_cast<float>(a * inverse_rms);
        normalized.at(1,y,x) = static_cast<float>(b * inverse_rms);
    }
    run(small,normalized,{first,second,norm},"internal large activation survives next convolution and normalization",weights);
}
void invalid_cases() {
    Feature input(3,4,6);
    const std::vector<TSGVaeOp> invalid = {
        shortcut(7,3,0,2), shortcut(7,3,1,0), shortcut(7,0,1,2), shortcut(7,5,1,2),
        shortcut(8,3,0,2), shortcut(8,3,1,0), shortcut(8,0,1,2), shortcut(8,2,1,2),
    };
    for (const auto& op : invalid) {
        std::vector<float> output(256);
        std::vector<TSGVaeOp> ops{op};
        auto desc = descriptor(input,output,ops);
        require(TSGgml_QwenVaeRun(&desc) == 0, "invalid shortcut parameters accepted");
        require(TSGgml_GetLastError() && *TSGgml_GetLastError(), "invalid shortcut supplied no error");
    }
    Feature odd(3,3,5);
    std::vector<float> output(256);
    std::vector<TSGVaeOp> ops{shortcut(7,3,1,2)};
    auto desc = descriptor(odd,output,ops);
    require(TSGgml_QwenVaeRun(&desc) == 0, "nondivisible spatial downsample accepted");
    // Failure may not poison the next graph or leave reused storage invalid.
    literal_cases();
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool cuda = argc >= 2 && std::string(argv[1]) == "cuda";
        const bool missing_cudnn = argc == 3 && std::string(argv[2]) == "--missing-cudnn-runtime";
        require(argc <= 3 && (argc == 1 || cuda || std::string(argv[1]) == "cpu") &&
            (argc < 3 || (cuda && missing_cudnn)), "usage: test [cpu|cuda] [--missing-cudnn-runtime]");
        if (!TSGgml_IsBackendAvailable(cuda ? 3 : 2)) {
            std::printf("SKIP: %s backend unavailable: %s\n", cuda ? "cuda" : "cpu", TSGgml_GetLastError());
            TSGgml_Shutdown(); return 77;
        }
        literal_cases(); synthetic_cases(); invalid_cases(); high_range_convolution_cases();
        if (missing_cudnn) {
#if defined(_WIN32)
            // Confirm this process actually probed our incomplete runtime.
            // Otherwise a missing/misplaced fixture could silently test the
            // installed vendor library instead of the intended failure path.
            const HMODULE fixture = GetModuleHandleW(L"cudnn64_9.dll");
            require(fixture && GetProcAddress(fixture,"TensorSharpMissingCudnnFixture") &&
                !GetProcAddress(fixture,"cudnnCreate"), "missing-symbol cuDNN fixture was not probed");
            std::puts("PASS missing-symbol cuDNN fixture: F32 ggml fallback verified");
#else
            throw std::runtime_error("missing-cuDNN fixture requires Windows");
#endif
        }
        TSGgml_ReleaseReuseComputeBuffers();
        literal_cases();
        TSGgml_Shutdown();
        TSGgml_Shutdown(); // teardown must also be safe with no cuDNN handle
        require(TSGgml_RecreateBackend() != 0, "backend recreation failed");
        high_range_convolution_cases(); // recreate vendor handle after teardown
        TSGgml_Shutdown();
        std::printf("PASS %s: VAE2.1 shortcuts, shape/reuse recovery and F32 convolution above the F16 range\n",
            cuda ? "cuda" : "cpu");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        TSGgml_Shutdown(); return 1;
    }
}
