import importlib.util
import io
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("qwen35_compare", Path(__file__).resolve().parents[1] / "compare-qwen35-checkpoints.py")
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def fixture(last_row=2.0, rotated=False):
    result = io.BytesIO()
    result.write(struct.pack("<Ii", 0x51354B43, 2))
    # Exercise the .NET seven-bit length encoding with a multi-byte length.
    result.write(bytes((160 | 128, 1)) + b"x" * 160)
    result.write(struct.pack("<7i", 2, 2, 0, 1, 32, 4, 2))
    result.write(b"\0")
    for _ in range(2):
        result.write(struct.pack("<iq", 1, 64))
        result.write(np.array([[1] * 32, [last_row] * 32], dtype="<f2").tobytes())
    result.write(b"\1")
    conv = np.arange(6, dtype="<f4").reshape(3, 2)
    if rotated:
        conv = np.roll(conv, 1, axis=0)
    result.write(struct.pack("<i", 6) + conv.tobytes())
    result.write(struct.pack("<iq", int(rotated), 16))
    result.write(np.arange(4, dtype="<f4").tobytes())
    return result.getvalue()


class Qwen35CheckpointCompareTests(unittest.TestCase):
    def test_new_kv_row_is_separate_from_prefix_and_ring_offset_is_canonicalized(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            a.write_bytes(fixture())
            b.write_bytes(fixture(last_row=3, rotated=True))
            result = COMPARE.compare(a, b)
        attn, gdn = result["layers"]
        self.assertEqual(attn["components"]["k_prefix"]["max_absolute"], 0)
        self.assertEqual(attn["components"]["k_last_row"]["max_absolute"], 1)
        self.assertEqual(gdn["components"]["conv"]["max_absolute"], 0)
        self.assertEqual(gdn["components"]["delta"]["max_absolute"], 0)

    def test_q8_rows_decode_scale_and_signed_values(self):
        block = np.float16(0.5).tobytes() + np.arange(-16, 16, dtype=np.int8).tobytes()
        reader = COMPARE.Reader(io.BytesIO(struct.pack("<iq", 1, 34) + block))
        values, dtype = reader.kv({"kv_heads": 1, "rows": 1, "head_dim": 32})
        self.assertEqual(dtype, "Q8_0")
        np.testing.assert_array_equal(values.ravel(), np.arange(-16, 16) * 0.5)

    def test_q4_rows_decode_ggml_low_then_high_nibbles(self):
        block = np.float16(2).tobytes() + bytes(range(16))
        reader = COMPARE.Reader(io.BytesIO(struct.pack("<iq", 1, 18) + block))
        values, dtype = reader.kv({"kv_heads": 1, "rows": 1, "head_dim": 32})
        self.assertEqual(dtype, "Q4_0")
        np.testing.assert_array_equal(values.ravel(), np.concatenate((np.arange(-8, 8) * 2, np.full(16, -16))))

    def test_truncated_checkpoint_fails(self):
        with self.assertRaisesRegex(ValueError, "Truncated"):
            COMPARE.Reader(io.BytesIO(fixture()[:20])).header()


if __name__ == "__main__":
    unittest.main()
