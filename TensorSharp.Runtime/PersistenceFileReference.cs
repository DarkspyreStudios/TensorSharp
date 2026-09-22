// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using Darkspyre.Persistence;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Identifies one model artifact through an application-owned persistence store.
    /// TensorSharp resolves the artifact for the lifetime of the reader or loaded model;
    /// callers do not need to expose a filesystem path.
    /// </summary>
    public sealed class PersistenceFileReference
    {
        public PersistenceFileReference(IPersistenceStore store, string storeName, string assetName)
        {
            Store = store ?? throw new ArgumentNullException(nameof(store));
            ArgumentException.ThrowIfNullOrWhiteSpace(storeName);
            ArgumentException.ThrowIfNullOrWhiteSpace(assetName);
            StoreName = storeName;
            AssetName = assetName;
        }

        public IPersistenceStore Store { get; }

        public string StoreName { get; }

        public string AssetName { get; }

        public override string ToString() => $"{StoreName}/{AssetName}";
    }
}
