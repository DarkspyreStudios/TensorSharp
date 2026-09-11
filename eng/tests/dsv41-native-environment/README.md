# Managed/native environment probe

This standalone diagnostic compares `Environment.GetEnvironmentVariable` with the platform C runtime's `getenv`. It uses a unique process-local variable, verifies `getenv` with a native setter as a positive control, and observes `MoeCpuOffloadConfig.SetCpuThreads(9)` and reset to zero. Only the probe variable and sanitized `TS_CPU_MOE_THREADS` values are printed. It does not load model weights or execute an inference backend.

Build without rebuilding TensorSharp:

```sh
dotnet build eng/tests/dsv41-native-environment/dsv41-native-environment.csproj \
  -c Release --output /tmp/dsv41-native-environment
dotnet /tmp/dsv41-native-environment/dsv41-native-environment.dll \
  /absolute/path/to/existing/host /tmp/native-environment.json
```

The three files needed on another .NET 10 platform are the `.dll`, `.deps.json`, and `.runtimeconfig.json`; omit the platform-specific extensionless apphost. The supplied host directory must contain its existing TensorSharp assemblies and matching native GgmlOps library. Host files are read without modification.

For an explicit inherited-value control, launch a separate process with `TS_CPU_MOE_THREADS=48`; for the unset control, use `env -u TS_CPU_MOE_THREADS` on POSIX. Observations do not reveal the native setter's private atomic override or an already loaded model's CPU thread count. In particular, a native environment value unchanged by the managed setter does not mean the generic native MoE setter failed.
