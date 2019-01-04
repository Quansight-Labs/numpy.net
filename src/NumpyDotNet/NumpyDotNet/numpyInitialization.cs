/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018-2019
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumpyLib;

namespace NumpyDotNet
{
    public partial class numpy
    {
        public static string __version__ = "0.0.1";
        public static bool _isNumpyLibraryInitialized = false;

        public static bool InitializeNumpyLibrary()
        {
            if (_isNumpyLibraryInitialized == true)
                return true;

            NpyArray_FunctionDefs functionDefs = null;

            NpyInterface_WrapperFuncs wrapperFuncs = new NpyInterface_WrapperFuncs()
            {
                array_new_wrapper = numpy_interface_array_new_wrapper,
                iter_new_wrapper = numpy_interface_iter_new_wrapper,
                multi_iter_new_wrapper = numpy_interface_multi_iter_new_wrapper,
                neighbor_iter_new_wrapper = numpy_interface_neighbor_iter_new_wrapper,
                descr_new_from_type = numpy_interface_descr_new_from_type,
                descr_new_from_wrapper = numpy_interface_descr_new_from_wrapper,
                ufunc_new_wrapper = numpy_interface_ufunc_new_wrapper,
            };

            npy_tp_error_set error_set = ErrorSet_handler;
            npy_tp_error_occurred error_occurred = ErrorOccurred_handler;
            npy_tp_error_clear error_clear = ErrorClear_handler;
            npy_tp_cmp_priority cmp_priority = numpy_tp_cmp_priority;
            npy_interface_incref incref = numpy_interface_incref;
            npy_interface_decref decref = numpy_interface_decref;
            enable_threads et = enable_threads;
            disable_threads dt = disable_threads;

            numpyAPI.npy_initlib(functionDefs, wrapperFuncs, error_set, error_occurred, error_clear, cmp_priority, incref, decref, et, dt);

            _isNumpyLibraryInitialized = true;
            return true;
        }


        public class NumpyExceptionInfo
        {
            public string FunctionName;
            public npyexc_type exctype;
            public string error;
        }

        public static List<NumpyExceptionInfo> NumpyErrors = new List<NumpyExceptionInfo>();

        static void ErrorSet_handler(string FunctionName, npyexc_type et, string error)
        {
            if (et == npyexc_type.NpyExc_DotNetException)
            {
                throw new Exception("Got an unexpected .NET exception");
            }
            NumpyErrors.Add(new NumpyExceptionInfo() { FunctionName = FunctionName, exctype = et, error = error });

            throw new Exception(string.Format("({0}) {1}: {2}", et, FunctionName, error));
            return;
        }

        static bool ErrorOccurred_handler(string FunctionName)
        {
            return false;
        }


        static void ErrorClear_handler(string FunctionName)
        {

        }

        static bool numpy_interface_array_new_wrapper(NpyArray newArray, bool ensureArray, bool customStrides, object subtype, object interfaceData, ref object interfaceRet)
        {
            return true;
        }
        static bool numpy_interface_iter_new_wrapper(NpyArrayIterObject iter, ref object interfaceRet)
        {
            return true;
        }
        static bool numpy_interface_multi_iter_new_wrapper(NpyArrayMultiIterObject iter, ref object interfaceRet)
        {
            return true;
        }
        static bool numpy_interface_neighbor_iter_new_wrapper(NpyArrayNeighborhoodIterObject iter, ref object interfaceRet)
        {
            return true;
        }
        static bool numpy_interface_descr_new_from_type(int type, NpyArray_Descr descr, ref object interfaceRet)
        {
            return true;
        }
        static bool numpy_interface_descr_new_from_wrapper(object _base, NpyArray_Descr descr, ref object interfaceRet)
        {
            return true;
        }
        static bool numpy_interface_ufunc_new_wrapper(object _base, ref object interfaceRet)
        {
            return true;
        }

        static int numpy_tp_cmp_priority(object o1, object o2)
        {
            return 0;
        }

        static object numpy_interface_incref(object o1, ref object o2)
        {
            return null;
        }
        static object numpy_interface_decref(object o1, ref object o2)
        {
            return null;
        }
        static object enable_threads()
        {
            return null;
        }
        static object disable_threads(object o1)
        {
            return null;
        }

    }
}
