using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;
using TensorSharp.Models.MiniMaxH3;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3MlxModulationTests
{
    [MlxTheory]
    [InlineData(17, 33, false)]
    [InlineData(17, 33, true)]
    [InlineData(1776, 2688, false)]
    public void PackedModulationMatchesPreviousExpressionExactly(int rows,int width,bool offsetView)
        => MlxWorker.Shared.Invoke(() =>
        {
            using var allocator=new MlxAllocator();
            using var context=new DirectContext(allocator);
            using var operations=new MiniMaxH3TensorOps(context);
            int storageRows=offsetView ? rows+2 : rows;
            float[] input=Values(storageRows*width,313), updates=Values(rows*width,751), mods=Values(18*width,911);
            using var update=context.FromFloats(updates,rows,width);
            using var modulation=context.FromFloats(mods,18,width);
            H3DitSegment[] segments=[
                new() { Start=1,End=rows/3,Col=12 },
                new() { Start=rows/3+1,End=2*rows/3,Col=0 },
                new() { Start=2*rows/3,End=rows-1,Col=6 },
                new() { Start=rows,End=rows,Col=int.MaxValue }];
            foreach (var (first,second,gated) in new[] { (0,1,false),(3,4,false),(2,2,true),(5,5,true) })
            {
                // Exercise same-content cache hits and in-place segment-array
                // mutation, without changing its shape or array identity.
                for (int iteration=0; iteration<3; iteration++)
                {
                    if (iteration == 2) segments[0].Col=(segments[0].Col+6)%18;
                    using var expectedStorage=context.FromFloats(input,storageRows,width);
                    using var actualStorage=context.FromFloats(input,storageRows,width);
                    using var expected=expectedStorage.Narrow(0,offsetView ? 1 : 0,rows);
                    using var actual=actualStorage.Narrow(0,offsetView ? 1 : 0,rows);
                    foreach (var segment in segments)
                    {
                        if (segment.End <= segment.Start) continue;
                        using var firstSlice=modulation.Narrow(0,segment.Col+first,1);
                        using var firstRow=firstSlice.View(width);
                        if (gated) DirectOps.GateAddRows(context,expected,update,firstRow,segment.Start,segment.End-segment.Start);
                        else
                        {
                            using var secondSlice=modulation.Narrow(0,segment.Col+second,1);
                            using var secondRow=secondSlice.View(width);
                            DirectOps.ModulateRows(context,expected,expected,firstRow,secondRow,segment.Start,segment.End-segment.Start);
                        }
                    }
                    float[] reference=DirectOps.ToArray(expectedStorage);
                    Assert.True(operations.TryModulatePacked(actual,update,modulation,segments,first,second,gated));
                    EqualBits(reference,DirectOps.ToArray(actualStorage),$"packed operation {first}/{second}/{gated}, iteration {iteration}");
                }
            }
            EqualBits(updates,DirectOps.ToArray(update),"update input");
            EqualBits(mods,DirectOps.ToArray(modulation),"modulation input");
        });

    [MlxFact]
    public void PackedModulationPreservesSeparateInputAndRejectsOverlappingSegments()
        => MlxWorker.Shared.Invoke(() =>
        {
            using var allocator=new MlxAllocator();
            using var context=new DirectContext(allocator);
            using var operations=new MiniMaxH3TensorOps(context);
            float[] input=Values(35,17), mods=Values(42,23);
            using var source=context.FromFloats(input,5,7);
            using var output=context.NewF32(5,7); Ops.Fill(output,37);
            using var modulation=context.FromFloats(mods,6,7);
            using var indices=Tensor.FromArray(allocator,new[] { 0,-1,0,-1,0 });
            Assert.True(MlxMiniMaxH3Ops.TryPackedModulation(output,source,null,modulation,indices,0,1,false));
            EqualBits(input,DirectOps.ToArray(source),"separate input");
            float[] actual=DirectOps.ToArray(output);
            EqualBits(input.AsSpan(7,7).ToArray(),actual.AsSpan(7,7).ToArray(),"first inactive row");
            EqualBits(input.AsSpan(21,7).ToArray(),actual.AsSpan(21,7).ToArray(),"second inactive row");
            H3DitSegment[] overlapping=[new() {Start=0,End=3,Col=0},new() {Start=2,End=5,Col=0}];
            Assert.False(operations.TryModulatePacked(source,null,modulation,overlapping,0,1,false));
            EqualBits(input,DirectOps.ToArray(source),"rejected overlapping input");
            using var wrongIndices=Tensor.FromArray(allocator,new[] { 0,0,0,0 });
            Assert.False(MlxMiniMaxH3Ops.TryPackedModulation(output,source,null,modulation,wrongIndices,0,1,false));
            Assert.False(MlxMiniMaxH3Ops.TryPackedModulation(output,source,null,modulation,indices,-1,1,false));
            Assert.False(MlxMiniMaxH3Ops.TryPackedModulation(output,source,null,modulation,indices,0,6,false));
            EqualBits(actual,DirectOps.ToArray(output),"rejected output");
        });

    private static float[] Values(int count,int seed)
    {
        float[] extremes=[0f,-0f,float.Epsilon,-float.Epsilon,1e-30f,-1e-30f,
            1f,-1f,80000f,-80000f,1e30f,-1e30f,float.MaxValue,float.MinValue];
        var random=new Random(seed);
        var result=new float[count];
        for (int i=0; i<count; i++) result[i]=i%37 < extremes.Length
            ? extremes[(i+seed)%extremes.Length] : (float)(random.NextDouble()*200-100);
        return result;
    }

    private static void EqualBits(float[] expected,float[] actual,string label)
        => Assert.True(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.AsSpan())),
            $"H3 MLX changed float bits in {label}.");
}
