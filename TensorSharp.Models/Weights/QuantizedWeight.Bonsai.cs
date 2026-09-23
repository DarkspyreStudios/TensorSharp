// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Runtime.ExceptionServices;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class QuantizedWeight
{
    private float[] _bonsaiSigns;
    private int _bonsaiBlockSize;
    private bool _bonsaiInverse;
    private int _bonsaiPermutationHeadDim, _bonsaiPermutationKeyHeads, _bonsaiPermutationRepeat;
    private HashSet<IntPtr> _bonsaiRegisteredKeys;

    internal void SetBonsaiTransform(float[] signs, int blockSize, bool inverse,
        int permutationHeadDim = 0, int permutationKeyHeads = 0, int permutationRepeat = 1)
    {
        _bonsaiSigns = signs;
        _bonsaiBlockSize = blockSize;
        _bonsaiInverse = inverse;
        _bonsaiPermutationHeadDim = permutationHeadDim;
        _bonsaiPermutationKeyHeads = permutationKeyHeads;
        _bonsaiPermutationRepeat = permutationRepeat;
        RegisterBonsaiCacheKey(Data);
        RegisterBonsaiCacheKey(CacheKey);
    }

    private void RegisterBonsaiCacheKey(IntPtr key)
    {
        if (_bonsaiSigns == null || key == IntPtr.Zero)
            return;
        _bonsaiRegisteredKeys ??= new();
        // Reserve managed ownership before the native mutation so an allocation
        // failure in HashSet.Add cannot leave an untracked native registration.
        if (!_bonsaiRegisteredKeys.Add(key))
            return;
        try
        {
            GgmlBonsai.RegisterWeight(key, _bonsaiSigns, _bonsaiBlockSize, _bonsaiInverse,
                _bonsaiPermutationHeadDim, _bonsaiPermutationKeyHeads, _bonsaiPermutationRepeat);
        }
        catch
        {
            _bonsaiRegisteredKeys.Remove(key);
            throw;
        }
    }

    private void UnregisterBonsaiWeights()
    {
        if (_bonsaiRegisteredKeys == null)
            return;
        Exception firstError = null;
        foreach (IntPtr key in _bonsaiRegisteredKeys)
        {
            try { GgmlBonsai.UnregisterWeight(key); }
            catch (Exception error) { firstError ??= error; }
        }
        _bonsaiRegisteredKeys.Clear();
        if (firstError != null)
            ExceptionDispatchInfo.Capture(firstError).Throw();
    }

    private void UnregisterBonsaiCacheKey(IntPtr key)
    {
        if (_bonsaiRegisteredKeys?.Remove(key) == true)
            GgmlBonsai.UnregisterWeight(key);
    }

    internal void InverseBonsaiEmbeddingRow(Span<float> row)
    {
        if (_bonsaiInverse)
            BonsaiHadamardMetadata.InverseTransform(row, _bonsaiSigns, _bonsaiBlockSize);
    }
}
