using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Validation;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[Collection("PrefixCacheModelConformance")]
public sealed class Gemma4HeterogeneousCacheTests(ITestOutputHelper output)
{
    [ModelFact("TS_TEST_GEMMA4_MODEL")]
    public void CompactedClonesAndGrowingHolders_AllExecuteFusedDecodeAndPreserveLogits()
    {
        using var environment = new EnvScope();
        environment.Set("MAX_CONTEXT", "16384");
        environment.Set("TS_KV_INITIAL_TOKENS", "0");
        using var model = ModelBase.Create(Environment.GetEnvironmentVariable("TS_TEST_GEMMA4_MODEL")!, TestGates.PinnedGgmlBackend);
        var gemma = Assert.IsType<Gemma4Model>(model);
        foreach (int width in new[] { 2, 3, 4 })
        {
            CacheProbeComparison result = Gemma4CacheProbe.Compare(gemma, width, 8, batchedFirst: width == 3, requireFused: true);
            Assert.Equal(8, result.Batched.FusedSteps);
            Assert.Equal(0, result.Batched.FallbackSteps);
            Assert.Equal(width - 2, result.Batched.GrowthRows);
            Assert.Equal(width * 8 - result.Batched.GrowthRows, result.Batched.NativeBatchedTokens);
            Assert.NotNull(result.UniformBatchedReference);
            Assert.Equal(0, result.UniformBatchedReference.TopTokenDifferences);
            output.WriteLine($"width={width} initial={string.Join(',', result.Batched.InitialCapacities)} final={string.Join(',', result.Batched.FinalCapacities)} fused=8/8 reference={result.UniformBatchedReference}; serial_difference: cosine={result.MinCosine:G9} rmse={result.MaxRmse:G9} max_abs={result.MaxAbsoluteError:G9} top_token_differences={result.TopTokenDifferences}");
        }
    }
}
