using System;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3NativeGraphAbiTests
{
    [Fact]
    public void DitDescriptor_MatchesNativeGraphArenaLayout()
    {
        // The paired native static_assert validates the C++ side independently.
        Assert.Equal(8, IntPtr.Size);
        Assert.Equal(544, Marshal.SizeOf<H3DitForwardArgs>());
        Assert.Equal(528, Marshal.OffsetOf<H3DitForwardArgs>(nameof(H3DitForwardArgs.GraphArena)).ToInt32());
        Assert.Equal(536, Marshal.OffsetOf<H3DitForwardArgs>(nameof(H3DitForwardArgs.GraphArenaBytes)).ToInt32());
    }
}
