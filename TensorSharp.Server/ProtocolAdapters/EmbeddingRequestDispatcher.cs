// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Models.Embeddings;

namespace TensorSharp.Server.ProtocolAdapters;

/// <summary>Combines requests already waiting for the resident encoder.</summary>
internal sealed class EmbeddingRequestDispatcher(IEmbeddingModel model) : IDisposable, IAsyncDisposable
{
    internal const int MaxBatchSequences = 64;
    internal const int MaxBatchTokens = 4096;
    private static readonly AsyncLocal<EmbeddingRequestDispatcher?> Executing = new();
    private readonly object _sync = new();
    private readonly Queue<Request> _pending = new();
    private bool _running, _disposed;
    private Batch? _active;
    private TaskCompletionSource _idle = Completed();

    private static TaskCompletionSource Completed()
    {
        var completion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        completion.SetResult();
        return completion;
    }

    private sealed class Request(int[][] inputs, int tokens, CancellationToken cancellation)
    {
        internal readonly int[][] Inputs = inputs;
        internal readonly int Tokens = tokens;
        internal readonly CancellationToken Cancellation = cancellation;
        internal readonly TaskCompletionSource<EmbeddingBatchResult> Completion = new(TaskCreationOptions.RunContinuationsAsynchronously);
        internal CancellationTokenRegistration Registration;
        internal Batch? Batch;
        internal bool CancellationCounted;
    }

    private sealed class Batch(List<Request> requests)
    {
        internal readonly List<Request> Requests = requests;
        internal int LiveRequests = requests.Count;
        // Preserve the caller's token for an isolated cancellable request.
        // Shared work gets its own source: one caller cannot cancel its peers.
        internal readonly CancellationTokenSource? Source = requests.Count == 1 && requests[0].Cancellation.CanBeCanceled
            ? null : new CancellationTokenSource();
        internal CancellationToken Token => Source?.Token ?? Requests[0].Cancellation;
    }

    internal Task<EmbeddingBatchResult> EmbedAsync(IReadOnlyList<int[]> inputs, CancellationToken cancellationToken)
    {
        if (Executing.Value == this)
            throw new InvalidOperationException("An embedding encoder cannot submit recursively to its own dispatcher.");
        ArgumentNullException.ThrowIfNull(inputs);
        cancellationToken.ThrowIfCancellationRequested();
        var owned = new int[inputs.Count][];
        int tokens = 0;
        for (int i = 0; i < inputs.Count; ++i)
        {
            // The adapter owns the token arrays; retain their immutable rows
            // while taking our own copy of the outer request list.
            owned[i] = inputs[i] ?? throw new ArgumentException("Input sequences must not be null.", nameof(inputs));
            tokens = checked(tokens + owned[i].Length);
        }
        var request = new Request(owned, tokens, cancellationToken);
        request.Registration = cancellationToken.UnsafeRegister(_ => Cancel(request), null);
        bool start = false, disposed;
        lock (_sync)
        {
            disposed = _disposed;
            if (!disposed)
            {
                _pending.Enqueue(request);
                if (!_running)
                {
                    _running = start = true;
                    _idle = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
                }
            }
        }
        if (disposed)
        {
            request.Registration.Dispose();
            throw new ObjectDisposedException(nameof(EmbeddingRequestDispatcher));
        }
        // Begin synchronously: the first request gets no batching timer or
        // extra Task.Run. The model already schedules its physical inference.
        if (start) _ = PumpAsync();
        return request.Completion.Task;
    }

    private void Cancel(Request request)
    {
        request.Completion.TrySetCanceled(request.Cancellation);
        CancellationTokenSource? source = null;
        lock (_sync)
        {
            if (request.Batch is { } batch && !request.CancellationCounted)
            {
                request.CancellationCounted = true;
                if (--batch.LiveRequests == 0) source = batch.Source;
            }
        }
        CancelPhysical(source);
    }

    private static void CancelPhysical(CancellationTokenSource? source)
    {
        try { source?.Cancel(); }
        catch (ObjectDisposedException) { } // Physical completion won the race.
        catch (AggregateException) { } // A custom encoder's cancellation callback must not prevent caller completion.
    }

    private async Task PumpAsync()
    {
        var previous = Executing.Value;
        Executing.Value = this;
        try
        {
            while (true)
            {
                var discarded = new List<Request>();
                Batch? batch;
                lock (_sync)
                {
                    var selected = new List<Request>();
                    int sequences = 0, tokens = 0;
                    while (_pending.TryPeek(out var next))
                    {
                        if (next.Completion.Task.IsCompleted)
                        {
                            discarded.Add(_pending.Dequeue());
                            continue;
                        }
                        // An oversized HTTP request runs alone. The model's
                        // existing microbatching handles its complete sequences.
                        if (selected.Count != 0 && (sequences + next.Inputs.Length > MaxBatchSequences ||
                            (long)tokens + next.Tokens > MaxBatchTokens)) break;
                        selected.Add(_pending.Dequeue());
                        sequences += next.Inputs.Length;
                        tokens += next.Tokens;
                    }
                    batch = selected.Count == 0 ? null : new Batch(selected);
                    _active = batch;
                    if (batch != null)
                        foreach (var request in batch.Requests) request.Batch = batch;
                    else
                    {
                        _running = false;
                        _idle.TrySetResult();
                    }
                }
                foreach (var request in discarded) request.Registration.Dispose();
                if (batch == null) return;
                try
                {
                    // A callback may have completed its caller immediately
                    // before acquiring _sync; account for that cancellation now.
                    foreach (var request in batch.Requests)
                        if (request.Cancellation.IsCancellationRequested) Cancel(request);
                    int[][] inputs;
                    if (batch.Requests.Count == 1) inputs = batch.Requests[0].Inputs;
                    else
                    {
                        int count = 0;
                        foreach (var request in batch.Requests) count += request.Inputs.Length;
                        inputs = new int[count][];
                        int offset = 0;
                        foreach (var request in batch.Requests)
                        {
                            request.Inputs.CopyTo(inputs, offset);
                            offset += request.Inputs.Length;
                        }
                    }
                    var result = await model.EmbedTokensAsync(inputs, batch.Token).ConfigureAwait(false);
                    if (result.Embeddings.Length != inputs.Length)
                        throw new InvalidOperationException("Embedding result count does not match input count.");
                    int first = 0;
                    foreach (var request in batch.Requests)
                    {
                        if (request.Cancellation.IsCancellationRequested)
                            request.Completion.TrySetCanceled(request.Cancellation);
                        else if (batch.Requests.Count == 1)
                            request.Completion.TrySetResult(result);
                        else
                        {
                            var embeddings = new float[request.Inputs.Length][];
                            Array.Copy(result.Embeddings, first, embeddings, 0, embeddings.Length);
                            request.Completion.TrySetResult(new EmbeddingBatchResult(embeddings, request.Tokens));
                        }
                        first += request.Inputs.Length;
                    }
                }
                catch (Exception error)
                {
                    foreach (var request in batch.Requests)
                        if (request.Cancellation.IsCancellationRequested)
                            request.Completion.TrySetCanceled(request.Cancellation);
                        else request.Completion.TrySetException(error);
                }
                finally
                {
                    lock (_sync)
                    {
                        foreach (var request in batch.Requests) request.Batch = null;
                        _active = null;
                    }
                    foreach (var request in batch.Requests) request.Registration.Dispose();
                    batch.Source?.Dispose();
                }
            }
        }
        finally { Executing.Value = previous; }
    }

    private Task Stop()
    {
        if (Executing.Value == this)
            throw new InvalidOperationException("An embedding encoder cannot dispose its own dispatcher during inference.");
        List<Request> stopped;
        CancellationTokenSource? source;
        Task idle;
        lock (_sync)
        {
            _disposed = true;
            stopped = new List<Request>(_pending);
            _pending.Clear();
            if (_active != null) stopped.AddRange(_active.Requests);
            source = _active?.Source;
            idle = _idle.Task;
        }
        foreach (var request in stopped)
            request.Completion.TrySetException(new ObjectDisposedException(nameof(EmbeddingRequestDispatcher)));
        CancelPhysical(source);
        // Wait outside _sync; callbacks and the inference continuation acquire it.
        foreach (var request in stopped) request.Registration.Dispose();
        return idle;
    }

    public void Dispose() => Stop().GetAwaiter().GetResult();
    public ValueTask DisposeAsync() => new(Stop());
}
