using System.Runtime.ExceptionServices;

namespace TensorSharp.Server.Host.Hosting;

internal static class HostRuntimeRetirement
{
    internal static void Run(Action runHost, Action? shutdownRuntime)
    {
        Exception? failure = null;
        try { runHost(); }
        catch (Exception error) { failure = error; }

        try { shutdownRuntime?.Invoke(); }
        catch (Exception cleanup)
        {
            if (failure is not null) throw new AggregateException(failure, cleanup);
            throw;
        }
        if (failure is not null) ExceptionDispatchInfo.Capture(failure).Throw();
    }
}
