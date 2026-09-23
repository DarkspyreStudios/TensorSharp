# vLLM structured diffusion

`JevCompiler.cs` adapts prompt text and the answer-template token-resolution
approach from the vLLM project's structured diffusion example:

- Copyright contributors to the vLLM project
- License: Apache License, Version 2.0 (see `Apache-2.0.txt` alongside this notice)
- Source: https://github.com/vllm-project/vllm/blob/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/examples/features/structured_diffusion/structured_server.py

TensorSharp rewrites the compiler in C#, resolves tokens from the loaded GGUF,
uses TensorSharp-owned model execution and selected-label projection, limits the
endpoint to text-only one-step reads, and reports conditional-label entropy.
The implementation is modified from the referenced example; it is not a vLLM
runtime dependency. TensorSharp contributions remain under the repository's
BSD-3-Clause license.
