import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import engines


class BackendLaunchTests(unittest.TestCase):
    def test_explicit_tensor_shard_count_matches_requested_gpu_count(self):
        spec = engines.config.BackendSpec(
            backend_id="test_tp", display="test", kind="gpu", ts_backend="ggml_cuda",
            ts_tp=True, ts_env={"TS_DSV41_TP": "{tp}", "TS_DSV4_UBATCH": "256"})
        model = SimpleNamespace(gguf=Path("/tmp/model.gguf"), mmproj=None,
                                is_image_edit=False, is_diffusion=False)
        server = engines.TensorSharpServer(model, "test_tp", Path("/tmp/unused.log"), tp=4)
        with patch.object(engines, "_port_open", return_value=False), \
             patch.object(engines.config, "BACKENDS", {"test_tp": spec}), \
             patch.object(engines.config, "TENSORSHARP_SERVER_DLL", Path("/tmp/server.dll")), \
             patch.object(engines.config, "tp_device_env", return_value={"CUDA_VISIBLE_DEVICES": "1,2,5,7"}), \
             patch.object(server, "_spawn") as spawn:
            server.start()
        arguments, kwargs = spawn.call_args
        command = arguments[0]
        self.assertEqual(command[command.index("--tp") + 1], "4")
        self.assertEqual(kwargs["env"]["TS_DSV41_TP"], "4")
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1,2,5,7")
        self.assertEqual(kwargs["env"]["TS_DSV4_UBATCH"], "256")
