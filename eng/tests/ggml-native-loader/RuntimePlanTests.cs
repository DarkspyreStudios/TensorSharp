using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using TensorSharp.GGML;
using Xunit;

namespace TensorSharp.NativeLoader.Tests;

public sealed class RuntimePlanTests
{
    [Fact]
    public async Task EmptyExplicitPlanIsImmutableUnavailableAndNeverBindsAmbient()
    {
        using var owner = new IsolatedOwner();
        Assert.Equal("Unconfigured", owner.State);
        owner.Configure(owner.Plan());
        Assert.Equal("Configured", owner.State);
        Task first = owner.Initialize();
        Task second = owner.Initialize();
        Assert.Same(first, second);
        await first;
        object result = first.GetType().GetProperty("Result")!.GetValue(first)!;
        Assert.Equal("Unavailable", result.GetType().GetProperty("State")!.GetValue(result)!.ToString());
        Assert.Equal("Unavailable", owner.State);
        Assert.Null(owner.Loader.GetProperty("Current")!.GetValue(null)!.GetType().GetProperty("LibraryPath")!.GetValue(owner.Loader.GetProperty("Current")!.GetValue(null)));
        owner.Configure(owner.Plan());
        Assert.Equal("Configured", owner.State);
    }

    [Theory]
    [InlineData("Rid", "wrong-rid")]
    [InlineData("TensorSharpBuild", "wrong-build")]
    [InlineData("NativeAbi", "0000000000000000000000000000000000000000000000000000000000000000")]
    public async Task IncompatiblePlanIsUnsupportedWithoutNativeUse(string field, string value)
    {
        using var owner = new IsolatedOwner();
        owner.Configure(owner.Plan(field, value));
        Task task = owner.Initialize();
        await task;
        object result = task.GetType().GetProperty("Result")!.GetValue(task)!;
        Assert.Equal("Unsupported", result.GetType().GetProperty("State")!.GetValue(result)!.ToString());
        owner.Configure(owner.Plan());
        Assert.Equal("Configured", owner.State);
    }

    [Fact]
    public void DefaultProbingMustBeExplicitAndCannotCoexistWithCandidates()
    {
        using var owner = new IsolatedOwner();
        Assert.IsType<ArgumentException>(Assert.Throws<TargetInvocationException>(() => owner.Configure(owner.Plan(nullCandidates: true))).InnerException);
        Assert.IsType<ArgumentException>(Assert.Throws<TargetInvocationException>(() => owner.Configure(owner.Plan(defaultProbing: true))).InnerException);
        owner.Configure(owner.Plan(nullCandidates: true, defaultProbing: true));
        Assert.Equal("Configured", owner.State);
    }

    [Fact]
    public async Task NoLoadShutdownIsTerminalAndIdempotent()
    {
        using var owner = new IsolatedOwner();
        object first = owner.Loader.GetMethod("Shutdown")!.Invoke(null, null)!;
        Assert.True((bool)first.GetType().GetProperty("Released")!.GetValue(first)!);
        Assert.Same(first, owner.Loader.GetMethod("Shutdown")!.Invoke(null, null));
        Assert.Equal("Stopped", owner.State);
        Assert.IsType<InvalidOperationException>(Assert.Throws<TargetInvocationException>(() => owner.Configure(owner.Plan())).InnerException);
        Assert.IsType<InvalidOperationException>((await Assert.ThrowsAsync<TargetInvocationException>(async () => await owner.Initialize())).InnerException);
    }

    [Fact]
    public void IdempotentOwnedHookRegistrationIsRemovedBySuccessfulNoLoadShutdown()
    {
        WeakReference[] roots = RegisterHookShutdownAndUnload();
        for (int attempt = 0; attempt < 20 && roots.Any(root => root.IsAlive); attempt++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }
        Assert.All(roots, root => Assert.False(root.IsAlive));
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference[] RegisterHookShutdownAndUnload()
    {
        var context = new AssemblyLoadContext("owned-hook-no-native-test", isCollectible: true);
        Assembly assembly = context.LoadFromAssemblyPath(typeof(GgmlNativeLoader).Assembly.Location);
        Type loader = assembly.GetType(typeof(GgmlNativeLoader).FullName!)!;
        MethodInfo register = loader.GetMethod("RegisterProcessExitHook", BindingFlags.NonPublic | BindingFlags.Static)!;
        register.Invoke(null, null);
        register.Invoke(null, null);
        FieldInfo hooked = loader.GetField("s_processExitHooked", BindingFlags.NonPublic | BindingFlags.Static)!;
        Assert.True((bool)hooked.GetValue(null)!);
        object result = loader.GetMethod("Shutdown")!.Invoke(null, null)!;
        Assert.True((bool)result.GetType().GetProperty("Released")!.GetValue(result)!);
        Assert.False((bool)hooked.GetValue(null)!);
        WeakReference[] roots = [new(context), new(assembly)];
        context.Unload();
        return roots;
    }

    private sealed class IsolatedOwner : IDisposable
    {
        private readonly AssemblyLoadContext context = new("native-owner-test-" + Guid.NewGuid(), isCollectible: true);
        private readonly Assembly assembly;
        internal Type Loader { get; }
        internal string State => Loader.GetProperty("State")!.GetValue(null)!.ToString()!;
        internal IsolatedOwner()
        {
            assembly = context.LoadFromAssemblyPath(typeof(GgmlNativeLoader).Assembly.Location);
            Loader = assembly.GetType(typeof(GgmlNativeLoader).FullName!)!;
        }
        internal object Plan(string? field = null, string? value = null, bool nullCandidates = false, bool defaultProbing = false)
        {
            object Identity(string property) => field == property ? value! : Loader.GetProperty(property == "Rid" ? "RuntimeIdentifier" : property)!.GetValue(null)!;
            return Activator.CreateInstance(assembly.GetType(typeof(GgmlRuntimePlan).FullName!)!,
                Identity("TensorSharpBuild"), Identity("NativeAbi"), Identity("Rid"),
                nullCandidates ? null : Array.CreateInstance(assembly.GetType(typeof(GgmlNativeCandidate).FullName!)!, 0),
                defaultProbing, Enum.Parse(assembly.GetType(typeof(GgmlBackendType).FullName!)!, "Cpu"))!;
        }
        internal void Configure(object plan) => Loader.GetMethod("Configure")!.Invoke(null, [plan]);
        internal Task Initialize() => (Task)Loader.GetMethod("InitializeAsync")!.Invoke(null, [CancellationToken.None])!;
        public void Dispose() => context.Unload();
    }
}
