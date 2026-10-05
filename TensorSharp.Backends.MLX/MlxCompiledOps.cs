using System;
using System.Threading;

namespace TensorSharp.MLX
{
    // Compiled (fused) activation kernels. Each one wraps a small graph of
    // MLX ops in mlx_compile so MLX collapses the chain into a single Metal
    // kernel and avoids materializing intermediates. Mirrors the pattern
    // used in ollama/x/mlxrunner/mlx/act.go.
    //
    // Each kernel is lazily compiled on first use under a guard so the
    // initialization is thread-safe. Cache clearing releases its closure references.
    // MLX caches per (shape, dtype) tuple
    // internally when the closure is non-shapeless, or specializes once when
    // shapeless. We use shapeless because the FFN tile shapes vary per
    // sequence length during prefill.
    internal static class MlxCompiledOps
    {
        // Global kill switch. Set TS_MLX_DISABLE_COMPILE=1 to skip the
        // compile path and fall back to the original eager op chains (used
        // for A/B benchmarking and as a safety hatch).
        internal static readonly bool Disabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_DISABLE_COMPILE"), "1", StringComparison.Ordinal);

        private sealed class ClosureSlot
        {
            internal MlxNative.CompiledClosure Closure;
        }

        private sealed class CompiledKernelCache
        {
            internal readonly ClosureSlot siluClosure = new();
            internal readonly ClosureSlot geluTanhClosure = new();
            internal readonly ClosureSlot swiGluClosure = new();
            internal readonly ClosureSlot geGluClosure = new();
            internal readonly ClosureSlot sigmoidMulClosure = new();
            internal readonly ClosureSlot addScaledClosure = new();
            internal readonly ClosureSlot rmsNormScaledClosure = new();
            internal readonly NativeOwnerRegistration NativeOwner;

            internal CompiledKernelCache()
            {
                NativeOwner = NativeQuarantineAuthority.Register(this, NativeOwnerRole.Graph);
                NativeOwner.AttachMlxSharedRuntime();
            }

            internal void ReleaseAfterSynchronization()
            {
                using NativeMlxReleaseReservation reservation = NativeOwner.ReserveMlxRelease(this);
                MlxWorker.Shared.Invoke(() =>
                {
                    using (NativeEffectLease validation = reservation.EnterEffect())
                        validation.ValidateMlxRelease(this, reservation);
                    lock (initLock)
                    {
                        ReleaseSlot(siluClosure);
                        ReleaseSlot(geluTanhClosure);
                        ReleaseSlot(swiGluClosure);
                        ReleaseSlot(geGluClosure);
                        ReleaseSlot(sigmoidMulClosure);
                        ReleaseSlot(addScaledClosure);
                        ReleaseSlot(rmsNormScaledClosure);
                    }
                });
            }

            internal void RetireReleasedOwner()
            {
                using NativeMlxReleaseReservation reservation = NativeOwner.ReserveMlxRelease(this);
                using NativeEffectLease effect = reservation.EnterEffect();
                effect.ValidateMlxRelease(this, reservation);
                lock (initLock)
                {
                    if (siluClosure.Closure != null || geluTanhClosure.Closure != null
                        || swiGluClosure.Closure != null || geGluClosure.Closure != null
                        || sigmoidMulClosure.Closure != null || addScaledClosure.Closure != null
                        || rmsNormScaledClosure.Closure != null)
                        throw new InvalidOperationException("MLX compiled cache still owns closure references at worker retirement.");
                    effect.CompleteSafeRelease(this);
                }
            }

            private static void ReleaseSlot(ClosureSlot slot)
            {
                if (slot.Closure == null) return;
                MlxNative.FreeCompiledClosure(slot.Closure);
                Volatile.Write(ref slot.Closure, null);
            }
        }

        private sealed class ClosurePublicationResources(CompiledKernelCache owner) : MlxNativeResources
        {
            internal MlxNative.CompiledClosure Closure;
            internal bool Published;
            internal override bool HasNativeResources => !Published && Closure != null && !Closure.SafelyReleased;
            internal override bool CleanupRequiresOrdinaryEffect => false;

            internal void Release()
            {
                if (!Published)
                    MlxNative.FreeCompiledClosure(Closure);
                GC.KeepAlive(owner);
            }
        }

        private static readonly Lazy<CompiledKernelCache> Cache = new(() => new CompiledKernelCache());
        private static CompiledKernelCache cache => Cache.Value;
        private static readonly object initLock = new();

        internal static void ReleaseAfterSynchronization()
        {
            if (Cache.IsValueCreated)
                cache.ReleaseAfterSynchronization();
        }

        internal static void RetireReleasedOwner()
        {
            if (Cache.IsValueCreated)
                cache.RetireReleasedOwner();
        }

        private static MlxNative.CompiledClosure EnsureCompiled(ClosureSlot slot, MlxNative.TraceFunc trace)
        {
            var resources = new ClosurePublicationResources(cache);
            return MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                cache.NativeOwner.ThrowIfQuarantined();
                lock (initLock)
                {
                    var existing = Volatile.Read(ref slot.Closure);
                    if (existing != null) return existing;
                    resources.Closure = MlxNative.NewClosure(trace, shapeless: true);
                    using NativeEffectLease effect = cache.NativeOwner.EnterEffect();
                    try
                    {
                        Volatile.Write(ref slot.Closure, resources.Closure);
                        resources.Published = true;
                    }
                    catch (Exception original)
                    {
                        Exception error = original;
                        try { effect.PublishFailure(cache, original, NativeRuntimeFailureStage.CacheRelease); }
                        catch (Exception publication) { error = MlxNative.JoinNativeErrors(error, publication); }
                        System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(error).Throw();
                    }
                    return resources.Closure;
                }
            }, resources.Release);
        }

        // silu(x) = x * sigmoid(x). One fused kernel instead of two-op chain.
        public static MlxNative.MlxArray SiLU(MlxNative.MlxArray x)
            => MlxWorker.Shared.Invoke(() => RunSiLU(x));

        private static MlxNative.MlxArray RunSiLU(MlxNative.MlxArray x)
        {
            var c = EnsureCompiled(cache.siluClosure, inputs =>
            {
                MlxNative.MlxArray inp = inputs[0];
                var resources = new MlxArrayResources(2, inputs);
                return MlxWorker.Shared.InvokeWithResources(resources, () =>
                {
                    ref MlxNative.MlxArray sig = ref resources.Arrays[0];
                    ref MlxNative.MlxArray result = ref resources.Arrays[1];
                    sig = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, inp);
                    result = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, inp, sig);
                    MlxNative.MlxArray[] outputs = new[] { result };
                    resources.ReturnedIndex = 1;
                    return outputs;
                }, resources.Release);
            });
            return MlxNative.ApplyClosure1(c, x);
        }

        // GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // Currently produced eagerly via 13 op calls; once compiled, MLX
        // fuses into a single kernel.
        public static MlxNative.MlxArray GeluTanh(MlxNative.MlxArray x)
            => MlxWorker.Shared.Invoke(() => RunGeluTanh(x));

        private static MlxNative.MlxArray RunGeluTanh(MlxNative.MlxArray x)
        {
            var c = EnsureCompiled(cache.geluTanhClosure, inputs =>
            {
                var resources = new MlxArrayResources(1, inputs);
                return MlxWorker.Shared.InvokeWithResources(resources, () =>
                {
                    ref MlxNative.MlxArray result = ref resources.Arrays[0];
                    result = GeluTanhTrace(inputs[0]);
                    MlxNative.MlxArray[] outputs = new[] { result };
                    resources.ReturnedIndex = 0;
                    return outputs;
                }, resources.Release);
            });
            return MlxNative.ApplyClosure1(c, x);
        }

        private static MlxNative.MlxArray GeluTanhTrace(MlxNative.MlxArray input)
        {
            // Allocate scalars inside the trace so they participate in the
            // compiled graph. The MLX compiler folds them as constants.
            var resources = new MlxArrayResources(13, new[] { input });
            return MlxWorker.Shared.InvokeWithResources(resources, () =>
            {
                ref MlxNative.MlxArray coeffCubic = ref resources.Arrays[0];
                ref MlxNative.MlxArray coeffInner = ref resources.Arrays[1];
                ref MlxNative.MlxArray one = ref resources.Arrays[2];
                ref MlxNative.MlxArray half = ref resources.Arrays[3];
                ref MlxNative.MlxArray squared = ref resources.Arrays[4];
                ref MlxNative.MlxArray cubed = ref resources.Arrays[5];
                ref MlxNative.MlxArray scaledCubic = ref resources.Arrays[6];
                ref MlxNative.MlxArray inner = ref resources.Arrays[7];
                ref MlxNative.MlxArray scaledInner = ref resources.Arrays[8];
                ref MlxNative.MlxArray tanh = ref resources.Arrays[9];
                ref MlxNative.MlxArray onePlusTanh = ref resources.Arrays[10];
                ref MlxNative.MlxArray halfInput = ref resources.Arrays[11];
                ref MlxNative.MlxArray output = ref resources.Arrays[12];
                coeffCubic = MlxNative.NewScalar(0.044715f);
                coeffInner = MlxNative.NewScalar(0.7978845608f);
                one = MlxNative.NewScalar(1.0f);
                half = MlxNative.NewScalar(0.5f);

                squared = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, input, input);
                cubed = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, squared, input);
                scaledCubic = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, cubed, coeffCubic);
                inner = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, input, scaledCubic);
                scaledInner = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, inner, coeffInner);
                tanh = MlxNative.Unary(MlxNative.MlxUnaryOp.Tanh, scaledInner);
                onePlusTanh = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, one, tanh);
                halfInput = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, input, half);
                output = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, halfInput, onePlusTanh);
                resources.ReturnedIndex = 12;
                return output;
            }, resources.Release);
        }

        // SwiGLU = silu(gate) * up. The common LLaMA/Qwen FFN activation.
        public static MlxNative.MlxArray SwiGLU(MlxNative.MlxArray gate, MlxNative.MlxArray up)
            => MlxWorker.Shared.Invoke(() => RunSwiGLU(gate, up));

        private static MlxNative.MlxArray RunSwiGLU(MlxNative.MlxArray gate, MlxNative.MlxArray up)
        {
            var c = EnsureCompiled(cache.swiGluClosure, inputs =>
            {
                MlxNative.MlxArray g = inputs[0];
                MlxNative.MlxArray u = inputs[1];
                var resources = new MlxArrayResources(3, inputs);
                return MlxWorker.Shared.InvokeWithResources(resources, () =>
                {
                    ref MlxNative.MlxArray sig = ref resources.Arrays[0];
                    ref MlxNative.MlxArray silu = ref resources.Arrays[1];
                    ref MlxNative.MlxArray result = ref resources.Arrays[2];
                    sig = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, g);
                    silu = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, g, sig);
                    result = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, silu, u);
                    MlxNative.MlxArray[] outputs = new[] { result };
                    resources.ReturnedIndex = 2;
                    return outputs;
                }, resources.Release);
            });
            return MlxNative.ApplyClosure2(c, gate, up);
        }

        // GeGLU = gelu(gate) * up. Used by Gemma family MLP and MoE paths.
        public static MlxNative.MlxArray GeGLU(MlxNative.MlxArray gate, MlxNative.MlxArray up)
            => MlxWorker.Shared.Invoke(() => RunGeGLU(gate, up));

        private static MlxNative.MlxArray RunGeGLU(MlxNative.MlxArray gate, MlxNative.MlxArray up)
        {
            var c = EnsureCompiled(cache.geGluClosure, inputs =>
            {
                MlxNative.MlxArray g = inputs[0];
                MlxNative.MlxArray u = inputs[1];
                var resources = new MlxArrayResources(2, inputs);
                return MlxWorker.Shared.InvokeWithResources(resources, () =>
                {
                    ref MlxNative.MlxArray gelu = ref resources.Arrays[0];
                    ref MlxNative.MlxArray result = ref resources.Arrays[1];
                    gelu = GeluTanhTrace(g);
                    result = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, gelu, u);
                    MlxNative.MlxArray[] outputs = new[] { result };
                    resources.ReturnedIndex = 1;
                    return outputs;
                }, resources.Release);
            });
            return MlxNative.ApplyClosure2(c, gate, up);
        }

        // SigmoidMul = x * sigmoid(gate). Variant used by some attention
        // gating paths.
        public static MlxNative.MlxArray SigmoidMul(MlxNative.MlxArray x, MlxNative.MlxArray gate)
            => MlxWorker.Shared.Invoke(() => RunSigmoidMul(x, gate));

        private static MlxNative.MlxArray RunSigmoidMul(MlxNative.MlxArray x, MlxNative.MlxArray gate)
        {
            var c = EnsureCompiled(cache.sigmoidMulClosure, inputs =>
            {
                MlxNative.MlxArray xi = inputs[0];
                MlxNative.MlxArray gi = inputs[1];
                var resources = new MlxArrayResources(2, inputs);
                return MlxWorker.Shared.InvokeWithResources(resources, () =>
                {
                    ref MlxNative.MlxArray sig = ref resources.Arrays[0];
                    ref MlxNative.MlxArray result = ref resources.Arrays[1];
                    sig = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, gi);
                    result = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, xi, sig);
                    MlxNative.MlxArray[] outputs = new[] { result };
                    resources.ReturnedIndex = 1;
                    return outputs;
                }, resources.Release);
            });
            return MlxNative.ApplyClosure2(c, x, gate);
        }

        // RmsNormScaled = fast_rms_norm(x, weight, eps) * scalar. Fuses the
        // norm + scalar multiply that the Qwen3.5 GDN path does for Q and K
        // (`q * (1/dim)`, `k * (1/sqrt(dim))`) into a single Metal kernel.
        // Saves 2 kernel launches per GDN layer × 30 GDN layers/forward.
        public static MlxNative.MlxArray RmsNormScaled(
            MlxNative.MlxArray x,
            MlxNative.MlxArray weight,
            MlxNative.MlxArray scalar,
            float eps)
            => MlxWorker.Shared.Invoke(() => RunRmsNormScaled(x, weight, scalar, eps));

        private static MlxNative.MlxArray RunRmsNormScaled(
            MlxNative.MlxArray x, MlxNative.MlxArray weight, MlxNative.MlxArray scalar, float eps)
        {
            var c = EnsureCompiled(cache.rmsNormScaledClosure, inputs =>
            {
                MlxNative.MlxArray xi = inputs[0];
                MlxNative.MlxArray wi = inputs[1];
                MlxNative.MlxArray si = inputs[2];
                // Use a closure-time-captured eps. Since eps is constant for
                // the Qwen35 GDN call site (1e-6f) and traces once shapelessly,
                // hard-coding it here is fine — the compiled graph specializes
                // to this eps value.
                var resources = new MlxArrayResources(2, inputs);
                return MlxWorker.Shared.InvokeWithResources(resources, () =>
                {
                    ref MlxNative.MlxArray normed = ref resources.Arrays[0];
                    ref MlxNative.MlxArray result = ref resources.Arrays[1];
                    normed = MlxNative.FastRmsNorm(xi, wi, 1e-6f);
                    result = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, normed, si);
                    MlxNative.MlxArray[] outputs = new[] { result };
                    resources.ReturnedIndex = 1;
                    return outputs;
                }, resources.Release);
            });
            // Note: eps is captured by the closure trace; if a different eps
            // is ever needed, add it as a 4th input or compile a separate slot.
            _ = eps; // suppress unused param warning; documented above
            return MlxNative.ApplyClosure(c, new[] { x, weight, scalar })[0];
        }

        // AddScaled = output + scalar * src. Fuses mulv + addt into a single
        // Metal kernel. Used by the MoE decode accumulator where we run this
        // 8 times per layer × 60 MoE layers — going from 2 kernels to 1 saves
        // ~480 kernel launches per decode token.
        // The scalar is passed as a 0-D MLX array input so the same compiled
        // closure works regardless of its value (no per-call recompile).
        public static MlxNative.MlxArray AddScaled(MlxNative.MlxArray output, MlxNative.MlxArray src, MlxNative.MlxArray scalar)
            => MlxWorker.Shared.Invoke(() => RunAddScaled(output, src, scalar));

        private static MlxNative.MlxArray RunAddScaled(MlxNative.MlxArray output, MlxNative.MlxArray src, MlxNative.MlxArray scalar)
        {
            var c = EnsureCompiled(cache.addScaledClosure, inputs =>
            {
                MlxNative.MlxArray o = inputs[0];
                MlxNative.MlxArray s = inputs[1];
                MlxNative.MlxArray k = inputs[2];
                var resources = new MlxArrayResources(2, inputs);
                return MlxWorker.Shared.InvokeWithResources(resources, () =>
                {
                    ref MlxNative.MlxArray scaled = ref resources.Arrays[0];
                    ref MlxNative.MlxArray result = ref resources.Arrays[1];
                    scaled = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, s, k);
                    result = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, o, scaled);
                    MlxNative.MlxArray[] outputs = new[] { result };
                    resources.ReturnedIndex = 1;
                    return outputs;
                }, resources.Release);
            });
            return MlxNative.ApplyClosure(c, new[] { output, src, scalar })[0];
        }
    }
}
