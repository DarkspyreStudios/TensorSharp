using System;

namespace TensorSharp.MLX;

/// <summary>Stateless mapped dense weights and BF16 boundaries for LTX's Gemma encoder.</summary>
public static class MlxLtx25PromptOps
{
    public static bool TryMappedDense(Tensor output, Tensor input, IntPtr weights, int ggmlType, int inner, int columns,
        bool bf16Matmul = false)
    {
        if (weights == IntPtr.Zero || ggmlType is not (0 or 1 or 30) || inner < 1 || columns < 1 ||
            output?.Storage is not MlxStorage target || input?.Storage is not MlxStorage source ||
            !ReferenceEquals(target.AllocatorImpl, source.AllocatorImpl) || ReferenceEquals(target, source) ||
            output.ElementType != DType.Float32 || input.ElementType != DType.Float32 ||
            !output.IsContiguous() || !input.IsContiguous() || input.DimensionCount != 2 || output.DimensionCount != 2 ||
            input.Sizes[1] != inner || output.Sizes[1] != columns || output.Sizes[0] != input.Sizes[0] ||
            (bf16Matmul && ggmlType != 30)) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x = default, result = default;
            try
            {
                x = source.CreateArrayView(input);
                result = MlxNative.Ltx25MappedDense(x, weights, ggmlType, inner, columns, bf16Matmul);
                // Materialize before dropping the mapped descriptor. This bounds
                // residency to this projection and never expands the full encoder.
                MlxNative.Eval(result);
                if (output.StorageOffset == 0 && output.ElementCount() == output.Storage.ElementCount)
                { target.ReplaceDeviceArray(result); result = default; }
                else target.UpdateDeviceSlice(output, result);
                return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(result); }
        });
    }

    public static bool TryRoundBf16(Tensor tensor)
    {
        if (tensor?.Storage is not MlxStorage storage || tensor.ElementType != DType.Float32 || !tensor.IsContiguous()) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray input = default, output = default;
            try
            {
                input = storage.CreateArrayView(tensor); output = MlxNative.Ltx25RoundBf16(input);
                if (tensor.StorageOffset == 0 && tensor.ElementCount() == tensor.Storage.ElementCount)
                { storage.ReplaceDeviceArray(output); output = default; }
                else storage.UpdateDeviceSlice(tensor, output);
                return true;
            }
            finally { MlxNative.FreeArray(input); MlxNative.FreeArray(output); }
        });
    }

    /// <summary>Apply Gemma split-half RoPE with a BF16 boundary after every
    /// product and sum. Trigonometric tables retain the caller's MathF values.</summary>
    public static bool TryGemmaRope(Tensor input, Tensor cosine, Tensor sine, int heads, int dim, int rotated)
    {
        if (input?.Storage is not MlxStorage storage || cosine?.Storage is not MlxStorage cs ||
            sine?.Storage is not MlxStorage ss || !ReferenceEquals(storage.AllocatorImpl, cs.AllocatorImpl) ||
            !ReferenceEquals(storage.AllocatorImpl, ss.AllocatorImpl) || ReferenceEquals(storage, cs) || ReferenceEquals(storage, ss) ||
            input.ElementType != DType.Float32 || cosine.ElementType != DType.Float32 || sine.ElementType != DType.Float32 ||
            !input.IsContiguous() || !cosine.IsContiguous() || !sine.IsContiguous() || input.DimensionCount != 2 ||
            heads <= 0 || dim <= 0 || dim % 2 != 0 || rotated <= 0 || rotated > dim / 2 ||
            input.Sizes[0] <= 0 || input.ElementCount() > int.MaxValue || input.Sizes[1] != (long)heads * dim ||
            cosine.ElementCount() != input.Sizes[0] * rotated || sine.ElementCount() != cosine.ElementCount()) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x = default, c = default, s = default, output = default;
            try
            {
                x = storage.CreateArrayView(input); c = cs.CreateArrayView(cosine); s = ss.CreateArrayView(sine);
                output = MlxNative.Ltx25GemmaRope(x, c, s, checked((int)input.Sizes[0]), heads, dim, rotated);
                if (input.StorageOffset == 0 && input.ElementCount() == input.Storage.ElementCount)
                { storage.ReplaceDeviceArray(output); output = default; }
                else storage.UpdateDeviceSlice(input, output);
                return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(c); MlxNative.FreeArray(s); MlxNative.FreeArray(output); }
        });
    }
}

internal static partial class MlxNative
{
    internal static MlxArray Ltx25MappedDense(MlxArray input, IntPtr weights, int type, int inner, int columns,
        bool bf16Matmul)
    {
        // mlx-c's public dtype enumeration: float16=9, float32=10, bfloat16=12.
        MlxArray weight = mlx_array_new_data(weights, [columns, inner], 2, type == 30 ? 12 : type == 1 ? 9 : 10);
        MlxArray transposed = default, x = default, product = default;
        try
        {
            if (!weight.IsValid) throw new InvalidOperationException("Could not create mapped LTX dense weights.");
            transposed = Transpose(weight, [1, 0]);
            if (bf16Matmul)
            {
                Check(mlx_astype(out x, input, 12, DefaultStream()), "casting LTX BF16 matmul input");
                Check(mlx_matmul(out product, x, transposed, DefaultStream()), "multiplying LTX BF16 dense weights");
                return Astype(product, DType.Float32);
            }
            Check(mlx_matmul(out product, input, transposed, DefaultStream()), "multiplying LTX dense weights");
            MlxArray result = product; product = default; return result;
        }
        finally { FreeArray(transposed); FreeArray(weight); FreeArray(x); FreeArray(product); }
    }
    internal static MlxArray Ltx25RoundBf16(MlxArray input)
    {
        Check(mlx_astype(out MlxArray bf16, input, 12, DefaultStream()), "rounding LTX BF16 activations");
        try { return Astype(bf16, DType.Float32); }
        finally { FreeArray(bf16); }
    }

    private static MlxFastMetalKernel ltx25GemmaRopeKernel;
    private const string Ltx25GemmaRopeHeader = """
        inline float ts_ltx25_prompt_bf16(float value) {
            uint bits = as_type<uint>(value);
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16) & 1u);
            return as_type<float>(bits & 0xffff0000u);
        }
        """;
    private const string Ltx25GemmaRopeSource = """
        const uint gid = thread_position_in_grid.x;
        constexpr int Half = Dim / 2;
        const int total = int(x_shape[0]) * Heads * Half;
        if (int(gid) >= total) return;
        const int pair = int(gid) % Half, tokenHead = int(gid) / Half;
        const int a = tokenHead * Dim + pair, b = a + Half;
        const float first = x[a], second = x[b];
        if (pair >= Rotated) { y[a] = first; y[b] = second; return; }
        const int table = (tokenHead / Heads) * Rotated + pair;
        const float co = c[table], si = s[table];
        const float fc = ts_ltx25_prompt_bf16(first * co);
        const float ss = ts_ltx25_prompt_bf16(second * si);
        const float sc = ts_ltx25_prompt_bf16(second * co);
        const float fs = ts_ltx25_prompt_bf16(first * si);
        y[a] = ts_ltx25_prompt_bf16(fc - ss);
        y[b] = ts_ltx25_prompt_bf16(sc + fs);
        """;

    internal static MlxArray Ltx25GemmaRope(MlxArray input, MlxArray cosine, MlxArray sine,
        int tokens, int heads, int dim, int rotated)
    {
        lock (fastKernelSync)
            if (!ltx25GemmaRopeKernel.IsValid)
                ltx25GemmaRopeKernel = CreateFastMetalKernel("tensorsharp_ltx25_gemma_rope_bf16",
                    ["x", "c", "s"], ["y"], Ltx25GemmaRopeSource, Ltx25GemmaRopeHeader);
        var config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, outputs = default;
        try
        {
            AddTemplateInt(config, "Heads", heads); AddTemplateInt(config, "Dim", dim); AddTemplateInt(config, "Rotated", rotated);
            int[] shape = [tokens, checked(heads * dim)];
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring LTX Gemma RoPE");
            Check(mlx_fast_metal_kernel_config_set_grid(config, checked(tokens * heads * (dim / 2)), 1, 1), "configuring LTX Gemma RoPE grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring LTX Gemma RoPE threads");
            inputs = CreateVectorArray(input, cosine, sine); outputs = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs, ltx25GemmaRopeKernel, inputs, config, DefaultStream()), "running LTX Gemma RoPE");
            Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading LTX Gemma RoPE");
            return result;
        }
        finally
        {
            if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
            if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
            if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
        }
    }
}
