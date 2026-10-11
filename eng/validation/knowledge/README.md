# Knowledge HTTP and agent probes

Run these from the repository root against the real local TensorAgent host.
Models must already be installed; the tools do not download weights. All mutable
host state and generated evidence belong under ignored `artifacts/` or
`docs/validation/`.

Start the host in a separate terminal, replacing the model paths:

```sh
TS_KNOWLEDGE_EMBEDDING_MODEL=/absolute/path/to/embedding.gguf \
dotnet run --project eng/validation/TensorAgentHost -- \
  --root artifacts/knowledge/live-host --skills TensorAgent/skills \
  --weights /absolute/path/to/chat-model.gguf \
  --mmproj /absolute/path/to/multimodal-projector.gguf \
  --backend ggml_metal --context 8192 --max-tokens 768 \
  --stop-file artifacts/knowledge/live-host/stop
```

Use a supported backend and model for the machine. A previously created `stop`
file must be removed before restarting. Wait for `connection.json` to be written;
it holds this validation host's loopback authentication token.

```sh
python3 eng/validation/knowledge/http_probe.py \
  --connection artifacts/knowledge/live-host/connection.json \
  --output artifacts/knowledge/http-probe
python3 eng/validation/knowledge/agent_probe.py \
  --connection artifacts/knowledge/live-host/connection.json \
  --output artifacts/knowledge/agent-probe
```

The HTTP probe reuses committed synthetic attachment fixtures, verifies original
bytes, extraction status, visual/audio ranking, a private fact answer, authoritative
source downloads, final answer citation URLs, and empty-scope isolation. It records
single-sample transcript word error rate. `--skip-media` explicitly omits image,
audio, and video checks. A generated wrong answer or invalid citation fails the
probe; successful infrastructure checks remain separately visible in its report.
The library collection is retained in the validation host for inspection.

The agent probe creates a deterministic 500-row CSV and requires actual successful
`knowledge_open` and `shell` events, exact Python stdout over all rows, and the exact
final answer. It deletes its own collection afterward. `--retry` permits one fresh
session retry while preserving the failed attempt. This is an explicit tool-use
scenario, not a general measure of autonomous planning quality.

Stop the host with `touch artifacts/knowledge/live-host/stop`. Keep connection
files, logs, JSON reports, SSE streams, and generated datasets out of Git. These
small correctness probes do not establish broad OCR/ASR/video understanding,
multi-client throughput, or mobile/Windows runtime support. In-process retrieval
performance is measured separately by [KnowledgeBench](../KnowledgeBench/README.md).
