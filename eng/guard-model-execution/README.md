# Model Execution Admission Inventory

The tool parses TensorSharp-owned model source with the SDK's Roslyn assemblies. It loads no
native library. It inventories mapped execution, cache, resource, speculative and media
entrypoints, including explicit interface implementations and model-associated children.
It rejects unclassified public/internal model methods and missing unconditional entry guards.
The worker check also verifies admission before receive and before direct dispatch outside
the provider exception handlers. The inventory does not qualify actual model or device execution.

From the repository root:

```sh
dotnet run --project eng/guard-model-execution -- --self-test
dotnet run --project eng/guard-model-execution -- --verify "$PWD"
dotnet run --project eng/guard-model-execution -- --inventory "$PWD"
```

Metadata, managed configuration/sampling/timing and existing shared guarded model disposal are
classified separately. Independent static helpers and private executor internals are outside
model admission. GLM vision associates its owning model internally; standalone encoders have
no model association. The checks do not add a backend-wide ban or an in-flight drain mechanism.
