# Persistence-backed model sources

The Darkspyre branch can load GGUF and safetensors artifacts through
`Darkspyre.Persistence.IPersistenceStore` without adding filesystem paths to that storage contract.
Applications identify an artifact with `PersistenceFileReference`, which contains the store, store
name, and logical asset name.

```csharp
var model = new PersistenceFileReference(store, "models", "qwen/model.gguf");
await modelService.LoadModelAsync(model, mmProj: null, "ggml_metal", cancellationToken);
```

`GgufFile.OpenAsync`, `SafetensorsFile.OpenAsync`, and `TorchStateDictionaryFile.OpenAsync` expose
the same source boundary for lower-level readers.

`TorchStateDictionaryFile` supports the narrow PyTorch ZIP state-dictionary shape needed by model
publishers that do not provide GGUF or safetensors weights. It is intentionally not a general pickle
runtime: metadata is interpreted without Python, and unknown opcodes, globals, reducers, devices,
storage types, and non-contiguous tensor views are rejected. The accepted surface is dense CPU F32,
BF16, and I64 tensors rebuilt through `torch._utils._rebuild_tensor_v2`.

## Lifetime and projection

- When `OpenReadAsync` returns a `FileStream`, TensorSharp uses that file in place and retains the
  stream for the reader or loaded model lifetime. No second cache copy is created.
- For any other readable stream, TensorSharp copies the bytes into its own temporary projection
  because its memory-mapped and native loaders require a path. TensorSharp deletes that projection
  on unload, disposal, cancellation, or failed loading.
- A failed model replacement retains the previous model's lease until the existing rollback attempt
  has finished. A successful replacement releases the previous lease after the new model is loaded.
- Applications continue to use only `IPersistenceStore`; no public path-resolution extension is
  required or exposed.

## Current scope

The persistence factory currently represents one artifact. Single-file GGUF and safetensors models,
plus a separately identified multimodal projector, are supported. A non-file store cannot yet supply
sibling files discovered implicitly by split-GGUF naming, a safetensors index, environment-configured
draft heads, or other path-discovered sidecars. File-backed stores continue to support those existing
path discovery behaviors when the related files share the same physical directory.

Callers must dispose `GgufFile`, `SafetensorsFile`, `TorchStateDictionaryFile`, or `ModelService`;
disposal is the ownership boundary that releases retained streams and temporary projections.
