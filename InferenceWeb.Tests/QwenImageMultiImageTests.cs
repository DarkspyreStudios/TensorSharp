// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Model-free unit tests for the multi-image Qwen-Image-2.1 conditioning logic:
// the Qwen3-VL M-RoPE position layout (get_rope_index) for several vision spans.
using TensorSharp.Models.QwenImage;
using Xunit;

namespace InferenceWeb.Tests
{
    public class QwenImageMultiImageTests
    {
        // ---- M-RoPE positions (get_rope_index) --------------------------------

        [Fact]
        public void BuildPositions_TextOnly_AllAxesSequential()
        {
            int[] pos = QwenImageTextEncoder.BuildPositions(4, null);
            for (int s = 0; s < 4; s++)
            {
                Assert.Equal(s, pos[s]);          // t
                Assert.Equal(s, pos[4 + s]);      // h
                Assert.Equal(s, pos[8 + s]);      // w
            }
        }

        [Fact]
        public void BuildPositions_TwoImageSpans_EachGetsItsOwnGridAndAdvance()
        {
            // layout: [txt, txt, img0(2x2 grid -> 4 tokens), txt, img1(1x2 -> 2 tokens), txt]
            var imgs = new[]
            {
                new ImageCond { Start = 2, Count = 4, GridH = 4, GridW = 4 },   // llm grid 2x2
                new ImageCond { Start = 7, Count = 2, GridH = 2, GridW = 4 },   // llm grid 1x2
            };
            int seq = 10;
            int[] pos = QwenImageTextEncoder.BuildPositions(seq, imgs);

            int T(int s) => pos[s]; int H(int s) => pos[seq + s]; int W(int s) => pos[2 * seq + s];

            // leading text: 0,1
            Assert.Equal(0, T(0)); Assert.Equal(1, T(1));
            // image 0 at cur=2: t=2 for all, h=2+row, w=2+col over a 2x2 grid
            Assert.Equal(2, T(2)); Assert.Equal(2, H(2)); Assert.Equal(2, W(2));   // (0,0)
            Assert.Equal(2, T(3)); Assert.Equal(2, H(3)); Assert.Equal(3, W(3));   // (0,1)
            Assert.Equal(2, T(4)); Assert.Equal(3, H(4)); Assert.Equal(2, W(4));   // (1,0)
            Assert.Equal(2, T(5)); Assert.Equal(3, H(5)); Assert.Equal(3, W(5));   // (1,1)
            // after image 0: cur advanced by max(2,2)=2 -> text at 4
            Assert.Equal(4, T(6)); Assert.Equal(4, H(6)); Assert.Equal(4, W(6));
            // image 1 at cur=5: 1x2 grid
            Assert.Equal(5, T(7)); Assert.Equal(5, H(7)); Assert.Equal(5, W(7));   // (0,0)
            Assert.Equal(5, T(8)); Assert.Equal(5, H(8)); Assert.Equal(6, W(8));   // (0,1)
            // after image 1: cur advanced by max(1,2)=2 -> text at 7
            Assert.Equal(7, T(9)); Assert.Equal(7, H(9)); Assert.Equal(7, W(9));
        }
    }
}
