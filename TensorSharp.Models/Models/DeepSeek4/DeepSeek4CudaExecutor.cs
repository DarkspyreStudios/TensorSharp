// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DeepSeek V4 (Flash) executor for the direct-CUDA backend.
//
// The model-side half of the DSV4 direct-CUDA path: opens the split-GGUF
// shards, parses the deepseek4 hyper-parameters, dequantizes the small
// tensors (norms, gates, sinks, APE tables, the router) to F32 host arrays,
// precomputes the raw/compress RoPE cos-sin tables (same YaRN math as
// DeepSeek4CpuExecutor.RopeCacheInit), and hands everything to
// TensorSharp.Cuda.Dsv4CudaEngine, which runs the forward pass with
// driver-API kernels — fully independent of ggml.
//
// The bulk weights are never staged in host RAM: each one is described to the
// engine as (shard, file offset, length) and the engine's loader pool streams
// it through pinned chunks straight into VRAM (layer-split across the visible
// GPUs). A 150 GiB model therefore loads with a ~2 GB host footprint.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Cuda;

namespace TensorSharp.Models
{
    internal sealed unsafe class DeepSeek4CudaExecutor : IDisposable, IDsv41EngramSource
    {
        private const int CsaRatio = 4;
        private const int HcaRatio = 128;

        private readonly List<GgufFile> _shards = new List<GgufFile>();
        private readonly List<string> _shardPaths = new List<string>();
        private readonly Dictionary<string, (GgufFile File, GgufTensorInfo Info)> _tensorMap
            = new Dictionary<string, (GgufFile, GgufTensorInfo)>(StringComparer.Ordinal);
        private readonly List<IntPtr> _ownedBuffers = new List<IntPtr>();
        private ShardSource[] _shardSources;
        private Dictionary<string, IntPtr> _prefetched;
        private readonly Dictionary<GgufFile, int> _shardIndexOf = new Dictionary<GgufFile, int>();

        /// <summary>
        /// Positional-read view over one GGUF shard, handed to the CUDA engine
        /// so big weights go file -> pinned chunk -> VRAM without a host-RAM
        /// copy of the whole model.
        /// </summary>
        /// <remarks>
        /// One descriptor per reader thread, deliberately. pread(2) on a shared
        /// handle is correct but on FUSE filesystems it is also *serialized*:
        /// on MooseFS, 16 threads sharing one descriptor read at 0.69 GB/s
        /// while the same 16 threads on their own descriptors read at 2.4 GB/s,
        /// independent of the access pattern.
        /// </remarks>
        private sealed class ShardSource : IDsv4WeightSource, IDsv4MappedWeightSource, IDisposable
        {
            private readonly string _path;
            private readonly ThreadLocal<Microsoft.Win32.SafeHandles.SafeFileHandle> _handles;

            // Whole-file read-only mapping, created lazily by TryMapRange. The
            // engine borrows raw pointers into it for as long as it lives, so
            // the mapping survives DisposeReaders (the post-load cleanup) and
            // only Dispose — called after the engine is gone — releases it.
            private readonly object _mapLock = new object();
            private System.IO.MemoryMappedFiles.MemoryMappedFile _map;
            private System.IO.MemoryMappedFiles.MemoryMappedViewAccessor _view;
            private byte* _mapBase;
            private long _mapLength;

            public ShardSource(string path)
            {
                _path = path;
                _handles = new ThreadLocal<Microsoft.Win32.SafeHandles.SafeFileHandle>(
                    () => File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read),
                    trackAllValues: true);
            }

            public void Read(long offset, IntPtr dst, long bytes)
            {
                var handle = _handles.Value;
                byte* p = (byte*)dst;
                long done = 0;
                while (done < bytes)
                {
                    int want = (int)Math.Min(1 << 30, bytes - done);
                    int got = RandomAccess.Read(handle, new Span<byte>(p + done, want), offset + done);
                    if (got <= 0)
                        throw new IOException($"[dsv4-cuda] short read at offset {offset + done} in {_path}");
                    done += got;
                }
            }

            public bool HasMapping => _mapBase != null;

            public bool TryMapRange(long offset, long bytes, out IntPtr ptr)
            {
                ptr = IntPtr.Zero;
                if (offset < 0 || bytes <= 0)
                    return false;
                lock (_mapLock)
                {
                    if (_mapBase == null)
                    {
                        try
                        {
                            long length = new FileInfo(_path).Length;
                            var map = System.IO.MemoryMappedFiles.MemoryMappedFile.CreateFromFile(
                                _path, FileMode.Open, mapName: null, 0,
                                System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);
                            var view = map.CreateViewAccessor(0, 0,
                                System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);
                            byte* b = null;
                            view.SafeMemoryMappedViewHandle.AcquirePointer(ref b);
                            _map = map;
                            _view = view;
                            _mapBase = b;
                            _mapLength = length;
                        }
                        catch
                        {
                            return false; // caller copies instead
                        }
                    }
                    if (offset + bytes > _mapLength)
                        return false;
                    ptr = (IntPtr)(_mapBase + offset);
                    return true;
                }
            }

            /// <summary>Drops the per-thread read handles (only needed while the
            /// loader streams weights) but keeps any mapping the engine borrowed
            /// pointers into.</summary>
            public void DisposeReaders()
            {
                foreach (var h in _handles.Values)
                    h?.Dispose();
                _handles.Dispose();
            }

            public void Dispose()
            {
                try { DisposeReaders(); } catch (ObjectDisposedException) { }
                lock (_mapLock)
                {
                    if (_mapBase != null)
                    {
                        _view.SafeMemoryMappedViewHandle.ReleasePointer();
                        _view.Dispose();
                        _map.Dispose();
                        _view = null;
                        _map = null;
                        _mapBase = null;
                        _mapLength = 0;
                    }
                }
            }
        }

        // hparams
        private int _nLayer, _nEmbd, _nHead, _nVocab, _headDim, _nRot, _qLoraRank, _oGroups, _oLoraRank, _nSwa;
        private float _rmsEps;
        private int _nExpert, _nExpertUsed, _nFfExp, _hashLayerCount;
        private float _expertWeightsScale;
        private bool _expertWeightsNorm;
        private float[] _swigluClampExp = Array.Empty<float>();
        private float[] _swigluClampShexp = Array.Empty<float>();
        private int _idxNHead, _idxHeadSize, _idxTopK;
        private int[] _compressRatios = Array.Empty<int>();
        private float _compressRopeBase = 10000f, _ropeFreqBase = 10000f;
        private float _yarnFreqScale = 1f, _yarnExtFactor;
        private float _yarnBetaFast = 32f, _yarnBetaSlow = 1f;
        private int _nCtxOrig;
        private int _hcSinkhornIters = 20;
        private float _hcEps = 1e-6f;
        private float _compCorr0, _compCorr1;

        private Dsv4CudaEngine _engine;

        public int VocabSize => _nVocab;
        public int NPast => _engine.NPast;

        private GgufFile _dsparkGguf;

        /// <summary>
        /// The engine's host-side quantized matmul for <c>--n-cpu-moe</c>
        /// layers. It lives here because the managed quantized kernels are in
        /// this assembly, which TensorSharp.Backends.Cuda cannot reference (the
        /// dependency runs the other way) — same inversion as
        /// <c>IDsv4WeightSource</c>.
        /// </summary>
        private sealed class HostMatMul : IDsv4HostMatMul
        {
            public static readonly HostMatMul Instance = new HostMatMul();

            public bool TryMatMulBatch(int ggmlType, int inDim, int inputRowStride,
                ReadOnlySpan<IDsv4HostMatMul.Job> jobs)
            {
                if (jobs.Length == 0)
                    return true;
                var mapped = new ManagedQuantizedOps.QuantMatMulJob[jobs.Length];
                for (int i = 0; i < jobs.Length; i++)
                    mapped[i] = new ManagedQuantizedOps.QuantMatMulJob(
                        jobs[i].Weights, jobs[i].Input, jobs[i].Output,
                        jobs[i].OutDim, jobs[i].RowCount, jobs[i].OutputRowStride);
                return ManagedQuantizedOps.TryAddmmQuantizedBatch(ggmlType, inDim, inputRowStride, mapped);
            }
        }

        /// <param name="nCpuMoe">Routed-expert CPU offload policy: 0 none (the
        /// default — offload is opt-in, and a model that does not fit is refused
        /// with the number of layers that would make it fit), N the first N
        /// layers, <see cref="int.MaxValue"/> every layer, -1 auto (the fewest
        /// leading layers that make the model fit; opt-in only).</param>
        public DeepSeek4CudaExecutor(string ggufPath, int maxContext, int nUbatch, int nGpu, string dsparkPath = null,
            int nCpuMoe = 0)
        {
            var sw = Stopwatch.StartNew();
            bool stats = ParseEnvInt("TS_DSV4_LOAD_STATS", 0) != 0;
            void Mark(string phase)
            {
                if (stats)
                    Console.Error.WriteLine($"[dsv4-cuda]   +{sw.Elapsed.TotalSeconds,6:F1}s {phase}");
            }

            OpenShards(ggufPath, dsparkPath);
            ParseHparams();
            if (_isV41)
                LoadEngramSidecar(ggufPath);
            Mark("shards opened / hparams parsed");

            // Large weights are read exactly once, on their way to VRAM, so the
            // engine streams them straight from the shards through pinned
            // chunks instead of staging the whole (hundreds of GB) model in
            // host RAM first. TS_DSV4_MMAP=1 opts back into the mmap path.
            bool stream = ParseEnvInt("TS_DSV4_MMAP", 0) == 0;
            if (stream)
            {
                _shardSources = new ShardSource[_shardPaths.Count];
                for (int s = 0; s < _shardPaths.Count; s++)
                    _shardSources[s] = new ShardSource(_shardPaths[s]);
                PrefetchSmallTensors();
                Mark("small tensors prefetched");
            }

            int nCtx = maxContext > 0 ? maxContext : 16384;
            int ubatch = nUbatch > 0 ? nUbatch : 1024;

            var desc = BuildModelDesc(nCtx, ubatch);
            Mark("model desc built");
            _engine = new Dsv4CudaEngine(desc, nGpu, nCpuMoe);
            Mark("engine ready");

            // Everything lives in VRAM now; drop the host-side scraps.
            _prefetched = null;
            foreach (var p in _ownedBuffers)
                Marshal.FreeHGlobal(p);
            _ownedBuffers.Clear();
            ReleaseShardReaders();

            Console.Error.WriteLine($"[dsv4-cuda] model ready in {sw.Elapsed.TotalSeconds:F1}s");
        }

        private static int ParseEnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) ? v : fallback;
        }

        public void Forward(int[] tokens, float[] logitsOut) => _engine.Forward(tokens, logitsOut);

        public void Reset() => _engine.Reset();

        // ---- DSpark speculative decoding (no-ops without a drafter) ----

        public bool HasDspark => _engine.DsparkBlockSize > 0;

        public int DsparkBlockSize => _engine.DsparkBlockSize;

        /// <summary>Row width of the target features the drafter consumes.</summary>
        public int DsparkFeatureSize => _engine.DsparkFeatureSize;

        public int UBatch => _engine.UBatch;

        public void ForwardSpec(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
            => _engine.ForwardSpec(tokens, hAllOut, logitsOut, allLogitsRows);

        public void DsparkCatchUp(float[] hRows, int rows, int firstPos)
            => _engine.DsparkCatchUp(hRows, rows, firstPos);

        public int DsparkDraft(int anchorToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
            => _engine.DsparkDraft(anchorToken, hPrev, position, draftOut, confOut);

        public void Rewind(int nPast) => _engine.Rewind(nPast);

        // -------------------------------------------------------------------
        // Loading (mirrors DeepSeek4CpuExecutor's split-shard resolver)
        // -------------------------------------------------------------------

        private void OpenShards(string firstPath, string dsparkPath = null)
        {
            // Standalone, like the rest of the loop below: this executor
            // enumerates the shards itself from split.count, so letting shard 1
            // also pull in its siblings would build a second, duplicate set of
            // GgufFile objects -- a second mmap and mlock pass over the whole
            // checkpoint, and a merged tensor table this code does not use.
            var first = GgufFile.OpenWithoutSiblingShards(firstPath);
            _shards.Add(first);
            _shardPaths.Add(firstPath);

            int splitCount = (int)first.GetUint32("split.count", 1);
            if (splitCount > 1)
            {
                const string marker = "-00001-of-";
                int pos = firstPath.IndexOf(marker, StringComparison.Ordinal);
                if (pos >= 0)
                {
                    for (int i = 2; i <= splitCount; i++)
                    {
                        string path = firstPath.Substring(0, pos) + $"-{i:D5}-of-" + firstPath.Substring(pos + marker.Length);
                        // This loop IS the shard enumeration, so each shard is opened
                        // for itself. Letting it expand its siblings again would open
                        // every file N times and attribute every tensor to whichever
                        // shard happened to be opened last.
                        _shards.Add(GgufFile.OpenWithoutSiblingShards(path));
                        _shardPaths.Add(path);
                    }
                }
            }

            // The DSpark drafter is a separate GGUF whose tensors are all
            // mtp.*-prefixed, so it can share the shard table (and therefore the
            // streaming loader) with the target model's shards.
            if (!string.IsNullOrEmpty(dsparkPath))
            {
                _dsparkGguf = new GgufFile(dsparkPath);
                _shards.Add(_dsparkGguf);
                _shardPaths.Add(dsparkPath);
            }

            // Before the split sizes anything: a shard cut short by an
            // interrupted download would otherwise fail as a short read well
            // into the upload, with the weight buffers already committed.
            foreach (var shard in _shards)
                shard.ThrowIfTruncated();

            for (int s = 0; s < _shards.Count; s++)
            {
                _shardIndexOf[_shards[s]] = s;
                foreach (var kv in _shards[s].Tensors)
                    _tensorMap[kv.Key] = (_shards[s], kv.Value);
            }
        }

        /// <summary>
        /// Reads the tensors the host itself has to look at (norms, gates,
        /// sinks, APE tables, the router, tid2eid) up front and in parallel.
        /// Individually they are tiny, but there are hundreds of them and a
        /// serial read of each costs a full round-trip on a network filesystem.
        /// Anything not prefetched still resolves through GetRaw's direct read.
        /// </summary>
        private void PrefetchSmallTensors()
        {
            var names = new List<string>();
            foreach (var kv in _tensorMap)
            {
                var type = kv.Value.Info.Type;
                if (type == GgmlTensorType.F32 || type == GgmlTensorType.I32)
                    names.Add(kv.Key);
            }
            if (names.Count == 0)
                return;

            var slots = new IntPtr[names.Count];
            Parallel.For(0, names.Count, new ParallelOptions { MaxDegreeOfParallelism = 16 }, i =>
            {
                var entry = _tensorMap[names[i]];
                long bytes = entry.File.GetTensorByteCount(entry.Info);
                IntPtr buf = Marshal.AllocHGlobal((nint)bytes);
                _shardSources[_shardIndexOf[entry.File]]
                    .Read(entry.File.DataOffset + (long)entry.Info.Offset, buf, bytes);
                slots[i] = buf;
            });

            _prefetched = new Dictionary<string, IntPtr>(names.Count, StringComparer.Ordinal);
            for (int i = 0; i < names.Count; i++)
            {
                _prefetched[names[i]] = slots[i];
                _ownedBuffers.Add(slots[i]);
            }
        }

        /// <summary>
        /// Post-load cleanup: the per-thread pread handles only serve the load,
        /// but a source whose mapping the engine borrowed (<c>--n-cpu-moe</c>
        /// experts point straight into it) must stay alive until
        /// <see cref="Dispose"/>.
        /// </summary>
        private void ReleaseShardReaders()
        {
            if (_shardSources == null)
                return;
            bool anyMapped = false;
            foreach (var s in _shardSources)
            {
                if (s == null)
                    continue;
                s.DisposeReaders();
                anyMapped |= s.HasMapping;
            }
            if (!anyMapped)
                _shardSources = null;
        }

        private void DisposeShardSources()
        {
            if (_shardSources == null)
                return;
            foreach (var s in _shardSources)
                s?.Dispose();
            _shardSources = null;
        }

        private void ParseHparams()
        {
            GgufFile g = _shards[0];
            // V4 and V4.1 are separate architectures with separate key prefixes.
            string arch = g.GetString("general.architecture", "deepseek4");
            if (arch != "deepseek4" && arch != "deepseek41")
                throw new NotSupportedException(
                    $"The DeepSeek CUDA executor requires the deepseek4 or deepseek41 architecture, got '{arch}'.");
            _isV41 = arch == "deepseek41";
            string a = arch;
            _nLayer = (int)g.GetUint32($"{a}.block_count");
            _nEmbd = (int)g.GetUint32($"{a}.embedding_length");
            _nHead = (int)g.GetUint32($"{a}.attention.head_count");
            _headDim = (int)g.GetUint32($"{a}.attention.key_length");
            _nRot = (int)g.GetUint32($"{a}.rope.dimension_count");
            _qLoraRank = (int)g.GetUint32($"{a}.attention.q_lora_rank");
            _oGroups = (int)g.GetUint32($"{a}.attention.output_group_count");
            _oLoraRank = (int)g.GetUint32($"{a}.attention.output_lora_rank");
            _nSwa = (int)g.GetUint32($"{a}.attention.sliding_window");
            _rmsEps = g.GetFloat32($"{a}.attention.layer_norm_rms_epsilon", 1e-6f);
            _nExpert = (int)g.GetUint32($"{a}.expert_count");
            _nExpertUsed = (int)g.GetUint32($"{a}.expert_used_count");
            _nFfExp = (int)g.GetUint32($"{a}.expert_feed_forward_length");
            _expertWeightsScale = g.GetFloat32($"{a}.expert_weights_scale", 1.0f);
            _expertWeightsNorm = g.GetBool($"{a}.expert_weights_norm");
            _hashLayerCount = (int)g.GetUint32($"{a}.hash_layer_count");
            _idxNHead = (int)g.GetUint32($"{a}.attention.indexer.head_count");
            _idxHeadSize = (int)g.GetUint32($"{a}.attention.indexer.key_length");
            _idxTopK = (int)g.GetUint32($"{a}.attention.indexer.top_k");
            _compressRatios = g.GetInt32Array($"{a}.attention.compress_ratios") ?? Array.Empty<int>();
            _compressRopeBase = g.GetFloat32($"{a}.attention.compress_rope_freq_base", 10000f);
            _hcSinkhornIters = (int)g.GetUint32($"{a}.hyper_connection.sinkhorn_iterations", 20);
            _hcEps = g.GetFloat32($"{a}.hyper_connection.epsilon", 1e-6f);
            _swigluClampExp = g.GetFloatArray($"{a}.swiglu_clamp_exp") ?? Array.Empty<float>();
            _swigluClampShexp = g.GetFloatArray($"{a}.swiglu_clamp_shexp") ?? _swigluClampExp;
            _ropeFreqBase = g.GetFloat32($"{a}.rope.freq_base", 10000f);

            float yarnFactor = g.GetFloat32($"{a}.rope.scaling.factor", 0f);
            if (yarnFactor > 0f)
            {
                _yarnFreqScale = 1.0f / yarnFactor;
                _yarnExtFactor = 1.0f;
            }
            _nCtxOrig = (int)g.GetUint32($"{a}.rope.scaling.original_context_length");
            _yarnBetaFast = g.GetFloat32($"{a}.rope.scaling.yarn_beta_fast", 32f);
            _yarnBetaSlow = g.GetFloat32($"{a}.rope.scaling.yarn_beta_slow", 1f);

            int hc = (int)g.GetUint32($"{a}.hyper_connection.count", 4);
            if (hc != 4)
                throw new NotSupportedException($"DeepSeek4 CUDA executor supports hyper_connection.count == 4, got {hc}.");
            if (_nLayer <= 0 || _compressRatios.Length < _nLayer)
                throw new InvalidOperationException("Missing or invalid deepseek4 GGUF metadata.");

            _compCorr0 = MathF.Max(0f, MathF.Floor(YarnCorrDim(_nRot, _nCtxOrig, _yarnBetaFast, _compressRopeBase)));
            _compCorr1 = MathF.Min(_nRot - 1, MathF.Ceiling(YarnCorrDim(_nRot, _nCtxOrig, _yarnBetaSlow, _compressRopeBase)));
        }

        /// <summary>deepseek41 rather than deepseek4.</summary>
        private bool _isV41;
        private Dsv41EngramData _engram;
        private int[] _v41KvSource, _v41IndexSource;
        private int[] _engramHistory;
        private int _engramHistoryLength;
        private int[] _engramHashes;
        private int _engramUbatchTokens;

        /// <summary>
        /// Reads <c>deepseek41.engram.bin</c> from beside the checkpoint and
        /// derives this checkpoint's cache-sharing topology from it. The GGUF
        /// conversion keeps neither the compressed token map nor the bucket
        /// layout, so V4.1 cannot address an Engram row without the sidecar.
        /// </summary>
        private void LoadEngramSidecar(string ggufPath)
        {
            string[] tokens = _shards[0].GetStringArray("tokenizer.ggml.tokens")
                ?? throw new InvalidOperationException("DeepSeek V4.1 tokenizer metadata is missing");
            ulong fingerprint = 14695981039346656037UL;
            foreach (string token in tokens)
                fingerprint = Dsv41EngramData.FingerprintToken(fingerprint, token);

            string directory = Path.GetDirectoryName(Path.GetFullPath(ggufPath));
            string sidecar = Path.Combine(directory ?? string.Empty, "deepseek41.engram.bin");
            _engram = Dsv41EngramData.Load(sidecar, (uint)tokens.Length, fingerprint);

            if (_engram.Layers[^1].Id >= _nLayer ||
                _engram.KvSourceLayerIds[^1] >= _nLayer || _engram.IndexSourceLayerIds[^1] >= _nLayer)
                throw new InvalidOperationException("DeepSeek V4.1 sidecar names a layer beyond the layer count");

            _v41KvSource = new int[_nLayer];
            _v41IndexSource = new int[_nLayer];
            int kv = -1, index = -1;
            for (int il = 0; il < _nLayer; il++)
            {
                if (Array.IndexOf(_engram.KvSourceLayerIds, il) >= 0) kv = il;
                if (Array.IndexOf(_engram.IndexSourceLayerIds, il) >= 0) index = il;
                int ratio = _compressRatios[il];
                if (ratio < 0 || ratio > 2)
                    throw new NotSupportedException($"Invalid DeepSeek V4.1 compression ratio {ratio} on layer {il}.");
                if (ratio != 0 && (kv < 0 || index < 0 ||
                    _compressRatios[kv] != ratio || _compressRatios[index] != ratio))
                    throw new InvalidOperationException(
                        "DeepSeek V4.1 cache-sharing topology does not match the compression ratios");
                _v41KvSource[il] = ratio != 0 ? kv : -1;
                _v41IndexSource[il] = ratio != 0 ? index : -1;
            }
        }

        // -------------------------------------------------------------------
        // IDsv41EngramSource: the host side of the Engram lookup
        //
        // An Engram table is hundreds of millions of rows and tens to hundreds
        // of GiB. It stays a host mapping and only the rows a token actually
        // selects are dequantized and handed to the engine, which uploads them.
        // -------------------------------------------------------------------

        private struct EngramTable
        {
            public byte* Base;
            public GgmlTensorType Type;
            public long RowBytes;
            public long Rows;
        }

        private EngramTable[] _engramTables;

        public int HashColumns => (int)_engram.HashColumns;

        public int HeadDim => (int)_engram.HeadDim;

        public void BeginEngramUbatch(ReadOnlySpan<int> tokens, int startPos)
        {
            _engramHashes = _engram.HashTokens(tokens, startPos, ref _engramHistory, ref _engramHistoryLength);
            _engramUbatchTokens = tokens.Length;
        }

        public void ResetEngram() => _engramHistoryLength = 0;

        public void GatherEngramRows(int engramIndex, int count, float* dst)
        {
            if (_engramHashes == null || count > _engramUbatchTokens)
                throw new InvalidOperationException("[dsv4-cuda] Engram rows requested before the ubatch was hashed");

            EngramTable table = _engramTables[engramIndex];
            int columns = HashColumns, headDim = HeadDim;
            int rowValues = columns * headDim;
            fixed (int* hashes = _engramHashes)
            {
                int* mine = hashes + (long)engramIndex * _engramUbatchTokens * columns;
                // Each (token, column) is an independent random row: the one place
                // this executor touches the big table at all.
                Parallel.For(0, count, t =>
                {
                    float* rows = dst + (long)t * rowValues;
                    int* ids = mine + (long)t * columns;
                    for (int c = 0; c < columns; c++)
                    {
                        int row = ids[c];
                        if ((uint)row >= (uint)table.Rows)
                            throw new InvalidOperationException("[dsv4-cuda] Engram lookup is out of bounds");
                        ManagedQuantizedOps.DequantizeRowToFloat32(
                            (int)table.Type, (IntPtr)(table.Base + row * table.RowBytes),
                            rows + c * headDim, headDim);
                    }
                });
            }
        }

        /// <summary>Maps one Engram table read-only. Falls back to a full staged
        /// copy only when the platform cannot map, which for a table this size
        /// would mean RAM the box does not have -- so say so rather than
        /// silently allocating.</summary>
        private EngramTable MapEngramTable(string name)
        {
            if (!_tensorMap.TryGetValue(name, out var entry))
                throw new InvalidOperationException($"[dsv4-cuda] missing tensor: {name}");
            GgufTensorInfo info = entry.Info;
            long bytes = entry.File.GetTensorByteCount(info);
            long offset = entry.File.DataOffset + (long)info.Offset;

            IntPtr mapped = IntPtr.Zero;
            if (_shardSources != null)
                _shardSources[_shardIndexOf[entry.File]].TryMapRange(offset, bytes, out mapped);
            if (mapped == IntPtr.Zero && !entry.File.TryGetTensorDataPointer(info, out mapped))
                throw new InvalidOperationException(
                    $"[dsv4-cuda] cannot memory-map {name} ({bytes / (1024.0 * 1024 * 1024):F1} GiB). " +
                    "The Engram tables are read row by row and are far too large to stage in RAM.");

            return new EngramTable
            {
                Base = (byte*)mapped,
                Type = info.Type,
                RowBytes = ManagedQuantizedOps.RowSize((int)info.Type, (int)info.Shape[0]),
                Rows = info.Shape.Length > 1 ? (long)info.Shape[1] : 1,
            };
        }

        private static float YarnCorrDim(int nDims, int nCtxOrig, float nRot, float freqBase)
        {
            return nDims * MathF.Log(nCtxOrig / (nRot * 2f * MathF.PI)) / (2f * MathF.Log(freqBase));
        }

        /// <summary>
        /// Materializes a tensor in host memory. Only used for the small
        /// tensors the host actually has to look at (norms, gates, sinks, APE
        /// tables, the router, tid2eid); the bulk weights never come here —
        /// they stream from the shard directly into VRAM.
        /// </summary>
        private (IntPtr Ptr, GgufTensorInfo Info) GetRaw(string name, bool required = true)
        {
            if (!_tensorMap.TryGetValue(name, out var entry))
            {
                if (required)
                    throw new InvalidOperationException($"[dsv4-cuda] missing tensor: {name}");
                return (IntPtr.Zero, null);
            }

            GgufTensorInfo info = entry.Info;
            if (_prefetched != null && _prefetched.TryGetValue(name, out IntPtr cached))
                return (cached, info);

            long bytes = entry.File.GetTensorByteCount(info);
            if (_shardSources != null)
            {
                IntPtr staged = Marshal.AllocHGlobal((nint)bytes);
                _ownedBuffers.Add(staged);
                _shardSources[_shardIndexOf[entry.File]]
                    .Read(entry.File.DataOffset + (long)info.Offset, staged, bytes);
                return (staged, info);
            }

            if (entry.File.TryGetTensorDataPointer(info, out IntPtr mapped))
                return (mapped, info);

            IntPtr buf = Marshal.AllocHGlobal((nint)bytes);
            _ownedBuffers.Add(buf);
            entry.File.ReadTensorDataToNative(info, buf, bytes);
            return (buf, info);
        }

        /// <summary>
        /// Describes a bulk weight to the engine. In the default (streaming)
        /// mode this touches no tensor data at all: it hands over the shard and
        /// the file offset, and the engine's loader pool moves the bytes.
        /// </summary>
        private Dsv4CudaEngine.QuantWeightDesc GetQW(string name, bool required = true)
        {
            if (!_tensorMap.TryGetValue(name, out var entry))
            {
                if (required)
                    throw new InvalidOperationException($"[dsv4-cuda] missing tensor: {name}");
                return default;
            }

            GgufTensorInfo info = entry.Info;
            var desc = new Dsv4CudaEngine.QuantWeightDesc
            {
                GgmlType = (int)info.Type,
                Ne0 = (int)info.Shape[0],
                Ne1 = info.Shape.Length > 1 ? (int)info.Shape[1] : 1,
                Ne2 = info.Shape.Length > 2 ? (int)info.Shape[2] : 1,
                RowBytes = ManagedQuantizedOps.RowSize((int)info.Type, (int)info.Shape[0]),
                Name = name,
            };

            if (_shardSources != null)
            {
                desc.Source = _shardSources[_shardIndexOf[entry.File]];
                desc.SourceOffset = entry.File.DataOffset + (long)info.Offset;
            }
            else
            {
                var (ptr, _) = GetRaw(name, required);
                if (ptr == IntPtr.Zero)
                    return default;
                desc.HostPtr = ptr;
            }
            return desc;
        }

        private float[] GetF32(string name, bool required = true)
        {
            var (ptr, info) = GetRaw(name, required);
            if (ptr == IntPtr.Zero)
                return null;
            long n = info.NumElements;
            var arr = new float[n];
            ManagedQuantizedOps.DequantizeToFloat32((int)info.Type, ptr, arr, 0, n);
            return arr;
        }

        private int[] GetI32(string name)
        {
            var (ptr, info) = GetRaw(name);
            if (info.Type != GgmlTensorType.I32)
                throw new InvalidOperationException($"[dsv4-cuda] {name}: expected I32, got {info.Type}");
            long n = info.NumElements;
            var arr = new int[n];
            Marshal.Copy(ptr, arr, 0, (int)n);
            return arr;
        }

        // -------------------------------------------------------------------
        // Descriptor construction
        // -------------------------------------------------------------------

        private Dsv4CudaEngine.ModelDesc BuildModelDesc(int nCtx, int nUbatch)
        {
            var tokEmbd = GetQW("token_embd.weight");
            _nVocab = tokEmbd.Ne1;
            int tokType = tokEmbd.GgmlType;
            // ts_dsv4_embed_f32 decodes these row layouts directly.
            if (tokType != 8 && tokType != 1 && tokType != 0 && tokType != 30)
                throw new NotSupportedException($"[dsv4-cuda] token_embd type {tokType} unsupported (Q8_0/F16/BF16/F32).");

            var m = new Dsv4CudaEngine.ModelDesc
            {
                NLayer = _nLayer,
                NEmbd = _nEmbd,
                NHead = _nHead,
                HeadDim = _headDim,
                NRot = _nRot,
                QLoraRank = _qLoraRank,
                OGroups = _oGroups,
                OLoraRank = _oLoraRank,
                NSwa = _nSwa,
                NVocab = _nVocab,
                NExpert = _nExpert,
                NExpertUsed = _nExpertUsed,
                NFfExp = _nFfExp,
                HashLayerCount = _hashLayerCount,
                IdxNHead = _idxNHead,
                IdxHeadSize = _idxHeadSize,
                IdxTopK = _idxTopK,
                HcSinkhornIters = _hcSinkhornIters,
                RmsEps = _rmsEps,
                HcEps = _hcEps,
                ExpertWeightsScale = _expertWeightsScale,
                ExpertWeightsNorm = _expertWeightsNorm,
                NCtx = nCtx,
                NUbatch = nUbatch,
                TokEmbd = tokEmbd,
                Output = GetQW("output.weight"),
                OutputNorm = GetF32("output_norm.weight"),
                // V4.1 collapses the streams for the head with the LAST layer's
                // FFN gates, so it ships no output_hc_* tensors at all.
                HcHeadFn = GetF32("output_hc_fn.weight", required: !_isV41),
                HcHeadScale = GetF32("output_hc_scale.weight", required: !_isV41),
                HcHeadBase = GetF32("output_hc_base.weight", required: !_isV41),
                V41 = _isV41,
                CandidateSource = _isV41 ? _engram.CandidateSourceLayerId : -1,
                CandidateTopk = _isV41 ? (int)_engram.CandidateTopkBlocks : 0,
                CandidateBlock = _isV41 ? (int)_engram.CandidateBlockSize : 0,
                Engram = _isV41 ? this : null,
                RopeRawTable = BuildRopeTable(nCtx, comp: false),
                RopeCompTable = BuildRopeTable(nCtx, comp: true),
                Layers = new Dsv4CudaEngine.LayerDesc[_nLayer],
                Dspark = BuildDsparkDesc(),
                HostMatMul = HostMatMul.Instance,
            };

            for (int il = 0; il < _nLayer; il++)
            {
                string p = $"blk.{il}.";
                var L = new Dsv4CudaEngine.LayerDesc
                {
                    Ratio = _compressRatios[il],
                    ClampExp = il < _swigluClampExp.Length ? _swigluClampExp[il] : 0f,
                    ClampShexp = il < _swigluClampShexp.Length ? _swigluClampShexp[il] : 0f,
                    AttnNorm = GetF32(p + "attn_norm.weight"),
                    Sinks = GetF32(p + "attn_sinks.weight"),
                    WqA = GetQW(p + "attn_q_a.weight"),
                    QANorm = GetF32(p + "attn_q_a_norm.weight"),
                    WqB = GetQW(p + "attn_q_b.weight"),
                    Wkv = GetQW(p + "attn_kv.weight"),
                    KvNorm = GetF32(p + "attn_kv_a_norm.weight"),
                    WoA = GetQW(p + "attn_output_a.weight"),
                    WoB = GetQW(p + "attn_output_b.weight"),
                    HcAttnFn = GetF32(p + "hc_attn_fn.weight"),
                    HcAttnScale = GetF32(p + "hc_attn_scale.weight"),
                    HcAttnBase = GetF32(p + "hc_attn_base.weight"),
                    HcFfnFn = GetF32(p + "hc_ffn_fn.weight"),
                    HcFfnScale = GetF32(p + "hc_ffn_scale.weight"),
                    HcFfnBase = GetF32(p + "hc_ffn_base.weight"),
                    GateInp = GetF32(p + "ffn_gate_inp.weight"),
                    FfnNorm = GetF32(p + "ffn_norm.weight"),
                    GateExps = GetQW(p + "ffn_gate_exps.weight"),
                    DownExps = GetQW(p + "ffn_down_exps.weight"),
                    UpExps = GetQW(p + "ffn_up_exps.weight"),
                    GateShexp = GetQW(p + "ffn_gate_shexp.weight"),
                    DownShexp = GetQW(p + "ffn_down_shexp.weight"),
                    UpShexp = GetQW(p + "ffn_up_shexp.weight"),
                };

                if (_isV41)
                {
                    // Only the per-ratio source layers carry compressor and
                    // indexer-query tensors; the rest of their group reads the
                    // caches those layers build.
                    L.KvSource = _v41KvSource[il];
                    L.IndexSource = _v41IndexSource[il];
                    if (L.KvSource == il)
                    {
                        L.CompWkv = GetQW(p + "attn_compressor_kv.weight");
                        L.CompNorm = GetF32(p + "attn_compressor_norm.weight");
                        if (L.Ratio > 1)
                            L.CompWgate = GetQW(p + "attn_compressor_gate.weight");
                        L.IndexerK = GetQW(p + "indexer.attn_k.weight");
                        L.IndexerKNorm = GetF32(p + "indexer.k_norm.weight");
                    }
                    if (L.IndexSource == il)
                    {
                        L.IdxProj = GetQW(p + "indexer.proj.weight");
                        L.IdxQB = GetQW(p + "indexer.attn_q_b.weight");
                    }
                    for (int t = 0; t < _engram.Layers.Length; t++)
                    {
                        if (_engram.Layers[t].Id != il)
                            continue;
                        L.EngramIndex = t;
                        L.EngramWkv = GetQW(p + "engram_wkv.weight");
                        L.EngramQ = GetF32(p + "engram_q.weight");
                        L.EngramK = GetF32(p + "engram_k.weight");
                        _engramTables ??= new EngramTable[_engram.Layers.Length];
                        _engramTables[t] = MapEngramTable(p + "engram_embd.weight");
                        break;
                    }
                }
                else if (L.Ratio != 0)
                {
                    L.CompWkv = GetQW(p + "attn_compressor_kv.weight");
                    L.CompWgate = GetQW(p + "attn_compressor_gate.weight");
                    L.CompApe = GetF32(p + "attn_compressor_ape.weight");
                    L.CompNorm = GetF32(p + "attn_compressor_norm.weight");
                    if (L.Ratio == CsaRatio)
                    {
                        L.IdxProj = GetQW(p + "indexer.proj.weight");
                        L.IdxQB = GetQW(p + "indexer.attn_q_b.weight");
                        L.IdxCompWkv = GetQW(p + "indexer_compressor_kv.weight");
                        L.IdxCompWgate = GetQW(p + "indexer_compressor_gate.weight");
                        L.IdxCompApe = GetF32(p + "indexer_compressor_ape.weight");
                        L.IdxCompNorm = GetF32(p + "indexer_compressor_norm.weight");
                    }
                }

                if (il < _hashLayerCount)
                    L.Tid2Eid = GetI32(p + "ffn_gate_tid2eid.weight");
                else
                    L.ExpProbsBias = GetF32(p + "exp_probs_b.bias");

                m.Layers[il] = L;
            }

            return m;
        }

        /// <summary>
        /// Describes the DSpark drafter (a separate GGUF: three DSV4 blocks with
        /// compress_ratio 0, plus the Markov and confidence heads) for the
        /// engine. Returns null when no drafter was supplied.
        /// </summary>
        private Dsv4CudaEngine.DsparkDesc BuildDsparkDesc()
        {
            if (_dsparkGguf == null)
                return null;

            // Published DSpark drafters carry the same weights under three
            // naming schemes (the ds4 builder's `mtp.*`, and two `dspark.*`
            // variants that differ in the metadata prefix), so resolve both the
            // keys and the tensor names by trying each spelling.
            string arch = _dsparkGguf.GetString("general.architecture") ?? string.Empty;
            if (arch != "deepseek4-dspark" && arch != "deepseek_v4_flash_dspark_draft")
            {
                throw new InvalidOperationException(
                    $"[dsv4-cuda] draft model architecture '{arch}' is not a DeepSeek V4 DSpark drafter " +
                    "(expected deepseek4-dspark or deepseek_v4_flash_dspark_draft). DSpark drafters for other " +
                    "architectures such as Gemma 4 use a different drafter design and are not supported.");
            }

            int DsUint(params string[] keys)
            {
                foreach (string k in keys)
                {
                    uint v = _dsparkGguf.GetUint32(k, 0);
                    if (v != 0)
                        return (int)v;
                }
                return 0;
            }

            int nStages = DsUint("dspark.n_layers", "dspark.stage_count", "dspark.layer_count",
                                 "deepseek4.dspark.n_layers", "deepseek4.dspark.layer_count");
            int blockSize = DsUint("dspark.block_size", "deepseek4.dspark.block_size");
            int markovRank = DsUint("dspark.markov_rank", "deepseek4.dspark.markov_rank");
            int noiseToken = DsUint("dspark.noise_token_id", "deepseek4.dspark.noise_token_id");
            int[] targetLayers = _dsparkGguf.GetInt32Array("dspark.target_layer_ids")
                ?? _dsparkGguf.GetInt32Array("dspark.target_layers")
                ?? _dsparkGguf.GetInt32Array("deepseek4.dspark.target_layer_ids")
                ?? _dsparkGguf.GetInt32Array("deepseek4.dspark.target_layers")
                ?? Array.Empty<int>();
            if (nStages <= 0 || blockSize <= 0 || markovRank <= 0 || targetLayers.Length == 0)
                throw new InvalidOperationException("[dsv4-cuda] draft model is missing dspark.* metadata");

            // The drafter has no per-layer swiglu clamp of its own; the module is
            // trained with the target's single swiglu_limit.
            float clamp = _swigluClampExp.Length > 0 ? _swigluClampExp[_swigluClampExp.Length - 1] : 0f;
            float clampSh = _swigluClampShexp.Length > 0 ? _swigluClampShexp[_swigluClampShexp.Length - 1] : clamp;

            // First spelling that exists in the drafter file wins.
            string Pick(params string[] names)
            {
                foreach (string n in names)
                    if (_tensorMap.ContainsKey(n))
                        return n;
                return names[0];
            }

            var stages = new Dsv4CudaEngine.LayerDesc[nStages];
            for (int s = 0; s < nStages; s++)
            {
                string p = _tensorMap.ContainsKey($"mtp.{s}.attn_norm.weight") ? $"mtp.{s}." : $"dspark.{s}.";
                stages[s] = new Dsv4CudaEngine.LayerDesc
                {
                    Ratio = 0,
                    ClampExp = clamp,
                    ClampShexp = clampSh,
                    AttnNorm = GetF32(p + "attn_norm.weight"),
                    Sinks = GetF32(p + "attn_sinks.weight"),
                    WqA = GetQW(p + "attn_q_a.weight"),
                    QANorm = GetF32(p + "attn_q_a_norm.weight"),
                    WqB = GetQW(p + "attn_q_b.weight"),
                    Wkv = GetQW(p + "attn_kv.weight"),
                    KvNorm = GetF32(p + "attn_kv_a_norm.weight"),
                    WoA = GetQW(p + "attn_output_a.weight"),
                    WoB = GetQW(p + "attn_output_b.weight"),
                    HcAttnFn = GetF32(p + "hc_attn_fn.weight"),
                    HcAttnScale = GetF32(p + "hc_attn_scale.weight"),
                    HcAttnBase = GetF32(p + "hc_attn_base.weight"),
                    HcFfnFn = GetF32(p + "hc_ffn_fn.weight"),
                    HcFfnScale = GetF32(p + "hc_ffn_scale.weight"),
                    HcFfnBase = GetF32(p + "hc_ffn_base.weight"),
                    GateInp = GetF32(p + "ffn_gate_inp.weight"),
                    ExpProbsBias = GetF32(p + "exp_probs_b.bias", required: false),
                    FfnNorm = GetF32(p + "ffn_norm.weight"),
                    GateExps = GetQW(p + "ffn_gate_exps.weight"),
                    DownExps = GetQW(p + "ffn_down_exps.weight"),
                    UpExps = GetQW(p + "ffn_up_exps.weight"),
                    GateShexp = GetQW(p + "ffn_gate_shexp.weight"),
                    DownShexp = GetQW(p + "ffn_down_shexp.weight"),
                    UpShexp = GetQW(p + "ffn_up_shexp.weight"),
                };
            }

            string first = "mtp.0.";
            string lastS = $"mtp.{nStages - 1}.";
            return new Dsv4CudaEngine.DsparkDesc
            {
                BlockSize = blockSize,
                NoiseTokenId = noiseToken,
                MarkovRank = markovRank,
                TargetLayerIds = targetLayers,
                Stages = stages,
                MainProj = GetQW(Pick(first + "main_proj.weight", "dspark.main_proj.weight")),
                MainNorm = GetF32(Pick(first + "main_norm.weight", "dspark.main_norm.weight")),
                Norm = GetF32(Pick(lastS + "norm.weight", "dspark.norm.weight")),
                HcHeadFn = GetF32(Pick(lastS + "hc_head_fn.weight", "dspark.hc_head_fn.weight")),
                HcHeadScale = GetF32(Pick(lastS + "hc_head_scale.weight", "dspark.hc_head_scale.weight")),
                HcHeadBase = GetF32(Pick(lastS + "hc_head_base.weight", "dspark.hc_head_base.weight")),
                MarkovW1 = GetF32(Pick(lastS + "markov_head.markov_w1.weight", "dspark.markov_w1.weight")),
                MarkovW2 = GetQW(Pick(lastS + "markov_head.markov_w2.weight", "dspark.markov_w2.weight")),
                ConfProj = GetF32(Pick(lastS + "confidence_head.proj.weight",
                                       "dspark.conf_proj.weight", "dspark.confidence_head.weight")),
            };
        }

        /// <summary>
        /// Interleaved (cos, sin) per position with the YaRN attention factor
        /// folded in — the table-driven form of DeepSeek4CpuExecutor.RopeCacheInit.
        /// </summary>
        private float[] BuildRopeTable(int nCtx, bool comp)
        {
            int nDims = _nRot;
            float freqBase = comp ? _compressRopeBase : _ropeFreqBase;
            float freqScale = comp ? _yarnFreqScale : 1f;
            float extFactor = comp ? _yarnExtFactor : 0f;
            float attnFactor = extFactor == 0f ? 1f : 1f / (1f + 0.1f * MathF.Log(1f / freqScale));
            float thetaScale = MathF.Pow(freqBase, -2f / nDims);

            var table = new float[(long)nCtx * nDims];
            Parallel.For(0, nCtx, pos =>
            {
                float theta = pos;
                long baseIdx = (long)pos * nDims;
                for (int i0 = 0; i0 < nDims; i0 += 2)
                {
                    float thetaInterp = freqScale * theta;
                    float th = thetaInterp;
                    float mscale = attnFactor;
                    if (extFactor != 0f)
                    {
                        float y = (i0 / 2 - _compCorr0) / MathF.Max(0.001f, _compCorr1 - _compCorr0);
                        float ramp = (1f - MathF.Min(1f, MathF.Max(0f, y))) * extFactor;
                        th = thetaInterp * (1f - ramp) + theta * ramp;
                        mscale *= 1f + 0.1f * MathF.Log(1f / freqScale);
                    }
                    table[baseIdx + i0 + 0] = MathF.Cos(th) * mscale;
                    table[baseIdx + i0 + 1] = MathF.Sin(th) * mscale;
                    theta *= thetaScale;
                }
            });
            return table;
        }

        public void Dispose()
        {
            _engine?.Dispose();
            _engine = null;
            DisposeShardSources();
            foreach (var p in _ownedBuffers)
                Marshal.FreeHGlobal(p);
            _ownedBuffers.Clear();
            foreach (var s in _shards)
                s.Dispose();
            _shards.Clear();
        }
    }
}
