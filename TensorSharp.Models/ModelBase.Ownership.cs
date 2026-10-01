// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Models;

public abstract partial class ModelBase
{
    // The callback transfers ownership as its final operation.
    private void TransferOwnedResource<T>(T resource, Action<T> transfer) where T : IDisposable
    {
        try { transfer(resource); }
        catch (Exception original)
        {
            RollBackLocalResource(resource, original);
            throw;
        }
    }

    private void RollBackLocalResource(IDisposable resource, Exception original)
    {
        try { resource.Dispose(); }
        catch (Exception cleanupError)
        {
            RetainFailedModelOwnership(resource, cleanupError);
            throw new AggregateException("Model work and local ownership rollback both failed.", original, cleanupError);
        }
    }

    private void RetireOwnedResource(IDisposable resource)
    {
        try { resource.Dispose(); }
        catch (Exception cleanupError)
        {
            RetainFailedModelOwnership(resource, cleanupError);
            throw;
        }
    }
}
