using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using IronPython.Runtime;
using IronPython.Modules;
using Microsoft.Scripting;
using NumpyDotNet;



namespace NumpyDotNet
{
    /// <summary>
    /// Provides access to the core for ufunc and numeric functions.
    /// </summary>
    internal static class UMathModule
    {


        //
        // Core-provided loops.
        //
        [DllImport("ndarray", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "npy_BYTE_add")]
        internal static extern void npy_BYTE_add(IntPtr array, IntPtr newDescr);

        static UMathModule() {
            IntPtr x = Marshal.GetFunctionPointerForDelegate(npy_BYTE_add);
            npy_BYTE_add(x, IntPtr.Zero);
        }

    }
}
