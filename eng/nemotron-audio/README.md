# Nemotron audio companion conversion

These scripts extract the audio tower from NVIDIA's
`Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` checkpoint at revision
`e5e9932441de940c9a62185c870ea5bcd4cd24e2`. Keep `audio-header.json`,
`config.json`, and `summary.json` beside the scripts: they contain the pinned
tensor metadata and model configuration needed for conversion.

Run from the repository root with Python 3 (standard library only). The scripts
download byte ranges from Hugging Face and write to the fixed directory
`/workspace/models/nemotron-omni/audio-e5e9932`, so its parent must exist and be
writable. Allow space for both the downloaded BF16 audio weights and the GGUF.

```sh
mkdir -p /workspace/models/nemotron-omni
python3 eng/nemotron-audio/prepare_f32.py
```

This produces `mmproj-audio-bf16-f32compute.gguf`, retaining BF16 weights while
setting `nemotron.audio.compute_bf16=false`. `prepare.py` instead produces
`mmproj-audio-bf16.gguf` with BF16 compute enabled. Each writes a checksum
manifest alongside its output.

The scripts and metadata were preserved unchanged when historical validation
artifacts were removed from Git. Conversion and model inference were not rerun
as part of that cleanup. The previously recorded CPU check matched the trained
companion only with F32 compute; the BF16 compute comparison failed. Speech
answer quality, GPU execution, and latency remain unvalidated. See the
[model documentation](../../docs/models/nemotron.md#47-audio-tower-nemotronaudioencoder-companion-gguf)
for loading and test coverage.
