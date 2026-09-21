"""A partial model read failure must leave failed, reviewable verification evidence."""
import errno
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("dsv41_verify", Path(__file__).parents[1] / "dsv41-verify-download.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class DownloadVerificationTests(unittest.TestCase):
    def run_reader(self, reader):
        payload = b"modeldata"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            name = "model.gguf"
            (directory / name).write_bytes(payload)
            entry = dict(path=name, size=len(payload), lfs=dict(oid=hashlib.sha256(payload).hexdigest()))
            with patch.object(Path, "open", return_value=reader), patch.object(verify.time, "sleep") as sleep, patch("sys.stdout", io.StringIO()):
                result = verify.verify_shard(directory, entry)
            return result, sleep

    def test_partial_read_then_transient_error_resumes_exact_verified_offset(self):
        for error_number in (errno.EINTR, errno.EAGAIN, errno.ENOMEM):
            with self.subTest(errno=error_number):
                class RecoveringRead(io.BytesIO):
                    positions = []
                    failed = False

                    def read(self, size=-1):
                        self.positions.append(self.tell())
                        if not self.tell():
                            return super().read(4)
                        if not self.failed:
                            self.failed = True
                            # Model a read that advanced the fd before raising.
                            super().read(2)
                            raise OSError(error_number, "injected transient read failure")
                        return super().read(size)

                reader = RecoveringRead(b"modeldata")
                result, sleep = self.run_reader(reader)
                self.assertTrue(result["passed"])
                self.assertEqual(result["actual_sha256"], hashlib.sha256(b"modeldata").hexdigest())
                self.assertEqual(reader.positions, [0, 4, 4])
                self.assertEqual((result["bytes_read"], result["retry_count"], result["short_read_count"]), (9, 1, 1))
                self.assertEqual(len(result["read_errors"]), 1)
                self.assertEqual(result["read_errors"][0]["offset"], 4)
                self.assertEqual(result["read_errors"][0]["errno"], error_number)
                self.assertTrue(result["read_errors"][0]["retried"])
                sleep.assert_called_once_with(0.05)

    def test_retry_exhaustion_and_nontransient_errors_fail_without_hash(self):
        for error_number, retries in ((errno.ENOMEM, 5), (errno.EIO, 0), (errno.EPERM, 0)):
            with self.subTest(errno=error_number):
                class FailedRead(io.BytesIO):
                    def read(self, size=-1):
                        if self.tell():
                            raise OSError(error_number, "injected read failure")
                        return super().read(4)

                result, sleep = self.run_reader(FailedRead(b"modeldata"))
                self.assertFalse(result["passed"])
                self.assertEqual(result["bytes_read"], 4)
                self.assertNotIn("actual_sha256", result)
                self.assertEqual(result["retry_count"], retries)
                self.assertEqual(len(result["read_errors"]), retries + 1)
                self.assertFalse(result["read_errors"][-1]["retried"])
                self.assertTrue(all(event["offset"] == 4 for event in result["read_errors"]))
                self.assertEqual(sleep.call_count, retries)

    def test_none_read_retries_as_eagain_with_recovery_and_exhaustion(self):
        for blocked_reads in (1, 6):
            with self.subTest(blocked_reads=blocked_reads):
                class WouldBlockRead(io.BytesIO):
                    remaining = blocked_reads
                    positions = []

                    def read(self, size=-1):
                        self.positions.append(self.tell())
                        if not self.tell():
                            return super().read(4)
                        if self.remaining:
                            self.remaining -= 1
                            return None
                        return super().read(size)

                reader = WouldBlockRead(b"modeldata")
                result, sleep = self.run_reader(reader)
                self.assertEqual(result["passed"], blocked_reads == 1)
                self.assertEqual(reader.positions, [0] + [4] * (2 if blocked_reads == 1 else 6))
                self.assertEqual(result["retry_count"], min(blocked_reads, 5))
                self.assertEqual(sleep.call_count, min(blocked_reads, 5))
                self.assertEqual(len(result["read_errors"]), blocked_reads)
                self.assertTrue(all(event["errno"] == errno.EAGAIN and event["offset"] == 4
                                    for event in result["read_errors"]))
                if blocked_reads == 1:
                    self.assertEqual(result["actual_sha256"], hashlib.sha256(b"modeldata").hexdigest())
                    self.assertEqual(result["bytes_read"], 9)
                else:
                    self.assertNotIn("actual_sha256", result)
                    self.assertEqual(result["bytes_read"], 4)
                    self.assertFalse(result["read_errors"][-1]["retried"])
                    self.assertIn("would block", result["error"])

    def test_unexpected_eof_is_never_retried(self):
        result, sleep = self.run_reader(io.BytesIO(b"mode"))
        self.assertFalse(result["passed"])
        self.assertEqual(result["bytes_read"], 4)
        self.assertEqual(result["retry_count"], 0)
        self.assertNotIn("actual_sha256", result)
        self.assertIn("Unexpected EOF at byte 4 of 9", result["error"])
        sleep.assert_not_called()

    def test_partial_read_error_preserves_all_shard_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = b"modeldata"
            entries = []
            for index in range(1, 8):
                name = f"DeepSeek-V4.1-Flash-Q2_K-{index:05}-of-00007.gguf"
                (directory / name).write_bytes(payload)
                entries.append(dict(type="file", path=name, size=len(payload),
                                    lfs=dict(oid=hashlib.sha256(payload).hexdigest())))
            original_open = Path.open

            class FailedRead(io.BytesIO):
                def read(self, size=-1):
                    if self.tell():
                        raise OSError(12, "Cannot allocate memory")
                    return super().read(4)

            def open_file(path, *args, **kwargs):
                if path.name == entries[0]["path"] and args == ("rb",):
                    return FailedRead(payload)
                return original_open(path, *args, **kwargs)

            report_path = directory / "verification.json"
            with patch.object(Path, "open", open_file), patch.object(verify.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(entries).encode())), patch("sys.argv", ["verify", str(directory), "--workers", "1", "--report", str(report_path)]), patch("sys.stdout", io.StringIO()), patch.object(verify.time, "sleep"):
                with self.assertRaises(SystemExit) as failure:
                    verify.main()
            self.assertEqual(failure.exception.code, 1)
            report = json.loads(report_path.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(report["shards"]), 7)
            self.assertEqual(report["shards"][0]["bytes_read"], 4)
            self.assertIn("Cannot allocate memory", report["shards"][0]["error"])
            self.assertNotIn("actual_sha256", report["shards"][0])
            self.assertEqual(report["shards"][0]["retry_count"], 5)
            self.assertEqual(len(report["shards"][0]["read_errors"]), 6)
            self.assertTrue(all(item["passed"] for item in report["shards"][1:]))


if __name__ == "__main__":
    unittest.main()
