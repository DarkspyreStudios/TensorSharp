// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "../dsv41_engram_advice.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

static unsigned checks = 0;
static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
    ++checks;
}

static const void * address(uintptr_t value) {
    // Planner inputs only; these synthetic addresses are never dereferenced.
    return reinterpret_cast<const void *>(value);
}

static void planner_and_policy() {
    using namespace tsg_dsv41;
    constexpr uintptr_t base = 0x10000;
    constexpr size_t page = 4096, extent = 4 * page + 37;
    mapped_advice_range range{};
    auto plan = [&](uintptr_t data, size_t bytes) {
        return plan_mapped_advice(address(base), extent, address(data), bytes, page, range);
    };
    require(plan(base, page), "Aligned tensor range rejected");
    require(range.address == address(base) && range.bytes == page, "Aligned range changed");
    require(plan(base + 13, page + 23), "Unaligned tensor range rejected");
    require(range.address == address(base) && range.bytes == page + 36,
            "Unaligned range did not cover its leading bytes and exact end");
    require(plan(base + page + 13, page + 17), "Interior range rejected");
    require(range.address == address(base + page) && range.bytes == page + 30,
            "Interior range escaped its first page or rounded its end");
    require(plan(base + 3 * page + 32, page + 5), "Exact unaligned mapping end rejected");
    require(range.address == address(base + 3 * page) && range.bytes == page + 37,
            "Final partial mapping page was misplanned");
    require(plan(base + extent - 1, 1), "Final mapped byte rejected");
    require(range.address == address(base + 4 * page) && range.bytes == 37,
            "Final-byte plan exceeded the mapping extent");
    require(plan(base + page, 0) && range.bytes == 0, "Empty interior range was not empty");
    require(plan(base + extent, 0) && range.bytes == 0, "Empty end range rejected");
    require(!plan(base - 1, 1), "Range before mapping accepted");
    require(!plan(base + extent, 1), "Nonempty range at mapping end accepted");
    require(!plan(base + extent + 1, 0), "Empty range beyond mapping accepted");
    require(!plan(base + extent - 1, 2), "Range exceeding mapping by one byte accepted");
    require(!plan(base + 13, std::numeric_limits<size_t>::max()), "Tensor-length overflow accepted");
    require(!plan_mapped_advice(nullptr, extent, address(base), 1, page, range), "Null mapping accepted");
    require(!plan_mapped_advice(address(base), extent, nullptr, 0, page, range), "Null data accepted");
    require(!plan_mapped_advice(address(base + 1), extent, address(base + 1), 1, page, range),
            "Unaligned mapping base accepted");
    require(!plan_mapped_advice(address(base), extent, address(base), 1, 0, range), "Zero page size accepted");
    require(!plan_mapped_advice(address(base), std::numeric_limits<size_t>::max(), address(base), 1, page, range),
            "Mapping-length overflow accepted");
    const uintptr_t last_page = std::numeric_limits<uintptr_t>::max() & ~uintptr_t(page - 1);
    require(!plan_mapped_advice(address(last_page), page, address(last_page), 1, page, range),
            "Mapping end wrapping the address space accepted");
    require(!plan_mapped_advice(address(last_page), page - 1,
                address(std::numeric_limits<uintptr_t>::max() - 3), 8, page, range),
            "Tensor end wrapping the address space accepted");

    for (unsigned threads : {1u, 2u, 16u, 32u}) {
        require(resolve_engram_random(nullptr, threads) == (threads > 1), "Automatic advice policy changed");
        require(!resolve_engram_random("0", threads), "Explicit advice disable ignored");
        require(resolve_engram_random("1", threads), "Explicit advice enable ignored");
    }
    for (const char * value : {"", "true", "false", " 1", "1 ", "+1", "01", "2", "-1", "1junk"}) {
        bool rejected = false;
        try { (void) resolve_engram_random(value, 16); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "Malformed advice option accepted");
    }
}

#if defined(__linux__)
static std::optional<bool> random_vm_flag(const void * data) {
    std::ifstream file("/proc/self/smaps");
    if (!file) return std::nullopt;
    const uintptr_t target = reinterpret_cast<uintptr_t>(data);
    bool selected = false;
    std::string line;
    while (std::getline(file, line)) {
        unsigned long long begin, end;
        if (std::sscanf(line.c_str(), "%llx-%llx", &begin, &end) == 2) {
            selected = target >= begin && target < end;
        } else if (selected && line.compare(0, 8, "VmFlags:") == 0) {
            std::istringstream words(line.substr(8));
            std::string word;
            while (words >> word) if (word == "rr") return true;
            return false;
        }
    }
    return std::nullopt;
}

static void linux_mapping() {
    using namespace tsg_dsv41;
    const long detected_page = sysconf(_SC_PAGESIZE);
    require(detected_page > 0, "Cannot determine system page size");
    const size_t page = size_t(detected_page), bytes = 4 * page + 37;
    struct file_owner {
        FILE * file = std::tmpfile();
        ~file_owner() { if (file) std::fclose(file); }
    } owner;
    require(owner.file != nullptr, "Cannot create owned scratch file");
    std::vector<unsigned char> expected(bytes);
    for (size_t i = 0; i < bytes; ++i) expected[i] = static_cast<unsigned char>((i * 37 + i / 17) % 251);
    require(std::fwrite(expected.data(), 1, bytes, owner.file) == bytes && std::fflush(owner.file) == 0,
            "Cannot initialize owned scratch file");
    void * mapped = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fileno(owner.file), 0);
    require(mapped != MAP_FAILED, "Cannot map owned scratch file");
    struct mapping_owner {
        void * data; size_t bytes;
        ~mapping_owner() { if (data != MAP_FAILED) munmap(data, bytes); }
    } mapping{mapped, bytes};
    auto * data = static_cast<unsigned char *>(mapped);
    const auto initial = random_vm_flag(data + page);
    if (initial) require(!*initial, "Fresh scratch mapping unexpectedly has random-read advice");

    const auto invalid = advise_engram_random(mapped, bytes, data + bytes, 1);
    require(invalid.status == mapped_advice_status::invalid_range, "Out-of-bounds advice reached the kernel");
    const auto empty = advise_engram_random(mapped, bytes, data + bytes, 0);
    require(empty.status == mapped_advice_status::empty && empty.bytes == 0 && empty.error == 0,
            "Empty range did not return an empty result");
    const auto result = advise_engram_random(mapped, bytes, data + page + 13, page + 17);
    require(result.status == mapped_advice_status::applied && result.error == 0,
            "Owned mapping did not accept random-read advice");
    require(result.bytes == page + 30, "Advice result did not preserve planned byte length");
    require(std::memcmp(mapped, expected.data(), bytes) == 0, "Advice modified mapped file contents");

    const auto first = random_vm_flag(data);
    const auto advised = random_vm_flag(data + page);
    const auto second = random_vm_flag(data + 2 * page);
    const auto after = random_vm_flag(data + 3 * page);
    const auto tail = random_vm_flag(data + 4 * page);
    if (initial && first && advised && second && after && tail) {
        require(*advised && *second, "Kernel VmFlags did not show random-read advice on both covered pages");
        require(!*first && !*after && !*tail, "Advice escaped the covered pages into another mapped region");
        std::puts("Linux smaps confirms rr on the two advised pages only");
    } else {
        std::puts("Linux smaps unavailable; kernel return and unchanged bytes verified without a VmFlags claim");
    }
    // This same valid address range must report the OS error once our mapping
    // is gone. There is no other allocation between unmap and the advice call.
    const void * old_data = data + page + 13;
    require(munmap(mapping.data, mapping.bytes) == 0, "Cannot unmap owned scratch file");
    mapping.data = MAP_FAILED;
    const auto absent = advise_engram_random(mapped, bytes, old_data, page + 17);
    require(absent.status == mapped_advice_status::system_error && absent.error != 0,
            "OS failure advising an unmapped range was lost");
    std::rewind(owner.file);
    std::vector<unsigned char> actual(bytes);
    require(std::fread(actual.data(), 1, bytes, owner.file) == bytes && actual == expected,
            "Advice modified the owned file bytes");
}
#endif

int main() try {
    planner_and_policy();
#if defined(__linux__)
    linux_mapping();
#else
    const auto result = tsg_dsv41::advise_engram_random(address(0x10000), 8192, address(0x1000d), 16);
    require(result.status == tsg_dsv41::mapped_advice_status::unsupported,
            "Non-Linux advice did not report unsupported");
    std::puts("Non-Linux: range/policy checks complete; OS advice explicitly unsupported");
#endif
    std::printf("Passed %u mapped Engram advice checks\n", checks);
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "Mapped Engram advice test failed after %u checks: %s\n", checks, error.what());
    return 1;
}
