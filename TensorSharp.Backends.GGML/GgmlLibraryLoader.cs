#nullable enable
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

internal static partial class GgmlLibraryLoader
{
    internal static IntPtr Load(string absolutePath)
    {
        if (!Path.IsPathFullyQualified(absolutePath)) throw new ArgumentException("Native bridge paths must be absolute.", nameof(absolutePath));
        if (!OperatingSystem.IsWindows()) return NativeLibrary.Load(absolutePath);
        // Dependencies come only from the chosen package directory and System32, never PATH/CWD.
        IntPtr handle = LoadLibraryExW(absolutePath, IntPtr.Zero, 0x00000100 | 0x00000800);
        if (handle == IntPtr.Zero)
        {
            int error = Marshal.GetLastPInvokeError();
            throw new DllNotFoundException($"Secure native load failed (Win32 {error}): {new Win32Exception(error).Message}");
        }
        return handle;
    }

    [LibraryImport("kernel32.dll", StringMarshalling = StringMarshalling.Utf16, SetLastError = true)]
    private static partial IntPtr LoadLibraryExW(string fileName, IntPtr file, uint flags);
}
