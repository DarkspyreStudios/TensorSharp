#!/usr/bin/env python3
"""Exercise the actual Windows reader-constructor branch with POSIX I/O adapters.

This is a portable extracted-source resource-lifetime check, not execution on
Windows. It does not build TensorSharp or change an existing native library.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parents[2] /
                        "TensorSharp.GGML.Native/ggml_ops_deepseek41_tp.cpp")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    constructor = source[source.index("explicit reader("):source.index("    ~reader()")]
    branch = constructor.split("#else", 1)[1].split("#endif", 1)[0]
    # Instrument only opening/closing; adapters still use the real libc FILE.
    branch = branch.replace("std::fopen", "open_record").replace("std::fclose", "close_record")
    prefix = r'''
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
static int opened_fd = -1, close_count = 0;
static bool fail_seek = false, fail_tell = false;
static FILE * open_record(const char * path, const char * mode) {
    FILE * value = std::fopen(path, mode);
    opened_fd = value ? fileno(value) : -1;
    return value;
}
static int close_record(FILE * file) { ++close_count; return std::fclose(file); }
static int _fseeki64(FILE * file, long long offset, int mode) {
    return fail_seek ? -1 : fseeko(file, offset, mode);
}
static long long _ftelli64(FILE * file) { return fail_tell ? -1 : ftello(file); }
static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
struct source { std::string path; size_t offset; };
struct reader {
    FILE * file = nullptr;
    explicit reader(const source & src, size_t bytes) {
'''
    suffix = r'''
    }
    ~reader() { if (file) close_record(file); }
};
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    int failed = 0;
    for (int scenario = 0; scenario < 6; ++scenario) {
        opened_fd = -1; close_count = 0;
        fail_seek = scenario == 3; fail_tell = scenario == 4;
        bool constructed = false;
        try {
            reader value({scenario == 5 ? std::string(argv[1]) + ".missing" : argv[1],
                          size_t(scenario == 2 ? 65 : 4)}, scenario == 1 ? 128 : 16);
            constructed = true;
        } catch (const std::exception &) {}
        const bool closed = opened_fd < 0 || fcntl(opened_fd, F_GETFD) == -1;
        const bool passed = constructed == (scenario == 0) && closed && close_count == (scenario == 5 ? 0 : 1);
        std::cout << "{\"scenario\":" << scenario << ",\"constructed\":" << (constructed ? "true" : "false")
                  << ",\"file_closed\":" << (closed ? "true" : "false") << ",\"close_count\":" << close_count
                  << ",\"passed\":" << (passed ? "true" : "false") << "}\n";
        if (!passed) ++failed;
        // Close leaked control descriptors so subsequent cases remain isolated.
        if (!closed && opened_fd >= 0) close(opened_fd);
    }
    return failed ? 1 : 0;
}
'''
    names = ["valid", "truncated", "offset_past_end", "seek_failed", "tell_failed", "open_failed"]
    with tempfile.TemporaryDirectory(prefix="dsv41-windows-reader-") as temporary:
        root = Path(temporary)
        cpp, executable = root / "reader.cpp", root / "reader"
        cpp.write_text(prefix + branch + suffix)
        (root / "weights.bin").write_bytes(bytes(64))
        compile_command = [os.environ.get("CXX", "c++"), "-std=c++17", str(cpp), "-o", str(executable)]
        compiled = subprocess.run(compile_command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(compiled.stderr)
        result = subprocess.run([str(executable), str(root / "weights.bin")], capture_output=True, text=True)
        checks = [dict(name=names[row["scenario"]], **row)
                  for row in (json.loads(line) for line in result.stdout.splitlines())]
        report = dict(scope=__doc__, source=str(args.source), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      extracted_harness_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
                      compile_command=compile_command, exit_code=result.returncode, stderr=result.stderr, checks=checks)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Passed {sum(row['passed'] for row in checks)}/{len(checks)} extracted Windows-branch checks")
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
