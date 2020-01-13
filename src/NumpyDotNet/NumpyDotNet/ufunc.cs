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
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using NumpyDotNet;
using NumpyLib;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{
    public partial class ufunc
    {
        private static String[] ufuncArgNames = { "extobj", "sig" };
        protected NpyUFuncObject core;

        internal ufunc(NpyUFuncObject corePtr)
        {
            core = corePtr;
        }


        ~ufunc()
        {

        }

        internal NpyUFuncObject UFunc
        {
            get { return core; }
        }




        /// <summary>
        /// Named arguments for reduce & accumulate.
        /// </summary>
        private static string[] ReduceArgNames = new String[] {
            "array", "axis", "dtype", "out" };


        private static string[] ReduceAtArgNames = new String[] {
            "array", "indices", "axis", "dtype", "out" };

        #region Python interface


        public int nin
        {
            get
            {
                CheckValid();
                return core.nin;
            }
        }

        public int nout
        {
            get
            {
                CheckValid();
                return core.nout;
            }
        }

        public int nargs
        {
            get
            {
                CheckValid();
                return core.nargs;
            }
        }

        public bool CoreEnabled
        {
            get
            {
                CheckValid();
                return core.core_enabled != 0;
            }
        }

        // TODO: Implement 'types'
        public override string ToString()
        {
            return __name__;
        }

        public string __name__
        {
            get
            {
                CheckValid();
                return core.name;
            }
        }

        public string signature()
        {
            CheckValid();
            return core.core_signature;
        }


        #endregion


        /// <summary>
        /// Simply checks to verify that the object was correctly initialized and hasn't
        /// already been disposed before we go accessing native memory.
        /// </summary>
        private void CheckValid()
        {
            if (core == null)
                throw new InvalidOperationException("UFunc object is invalid or already disposed.");
        }



        /// <summary>
        /// Performs a generic reduce or accumulate operation on an input array.
        /// A reduce operation reduces the number of dimensions of the input array
        /// by one where accumulate does not.  Accumulate stores in incremental
        /// accumulated values in the extra dimension.
        /// </summary>
        /// <param name="arr">Input array</param>
        /// <param name="indices">Used only for reduceat</param>
        /// <param name="axis">Axis to reduce</param>
        /// <param name="otype">Output type of the array</param>
        /// <param name="outArr">Optional output array</param>
        /// <param name="operation">Reduce/accumulate operation to perform</param>
        /// <returns>Resulting array, either outArr or a new array</returns>
        private Object GenericReduce(ndarray arr, ndarray indices, int axis,
            dtype otype, ndarray outArr, GenericReductionOp operation)
        {

            if (signature() != null)
            {
                throw new RuntimeException("Reduction is not defined on ufunc's with signatures");
            }
            if (nin != 2)
            {
                throw new ArgumentException("Reduce/accumulate only supported for binary functions");
            }
            if (nout != 1)
            {
                throw new ArgumentException("Reduce/accumulate only supported for functions returning a single value");
            }

            if (arr.ndim == 0)
            {
                throw new ArgumentTypeException("Cannot reduce/accumulate a scalar");
            }

            if (arr.IsFlexible || (otype != null && NpyDefs.IsFlexible(otype.TypeNum)))
            {
                throw new ArgumentTypeException("Cannot perform reduce/accumulate with flexible type");
            }

            return NpyCoreApi.GenericReduction(this, arr, indices,
                outArr, axis, otype, operation);
        }

        class WithFunc
        {
            public object arg;
            public object func;
        }




        private bool IsStringType(string sig)
        {
            int pos = sig.IndexOf("->");
            if (pos == -1)
            {
                return false;
            }
            else
            {
                int n = sig.Length - 2;
                if (pos != nin || n - 2 != nout)
                {
                    throw new ArgumentException(
                        String.Format("a type-string for {0}, requires {1} typecode(s) before and {2} after the -> sign",
                                      this, nin, nout));
                }
                return true;
            }
        }

        /// <summary>
        /// Converts args to arrays and return an array nargs long containing the arrays and
        /// nulls.
        /// </summary>
        /// <param name="args"></param>
        /// <returns></returns>
        private ndarray[] ConvertArgs(object[] args)
        {
            if (args.Length < nin || args.Length > nargs)
            {
                throw new ArgumentException("invalid number of arguments");
            }
            ndarray[] result = new ndarray[nargs];
            for (int i = 0; i < nin; i++)
            {
                // TODO: Add check for scalars
                object arg = args[i];
                object context = null;
                if (!(arg is ndarray) && !(arg is ScalarGeneric))
                {
                    object[] contextArray = null;
                    contextArray = new object[] { this, new PythonTuple(args), i };
                    context = new PythonTuple(contextArray);
                }
                result[i] = np.FromAny(arg, context: context);
            }

            for (int i = nin; i < nargs; i++)
            {
                if (i >= args.Length || args[i] == null)
                {
                    result[i] = null;
                }
                else if (args[i] is ndarray)
                {
                    result[i] = (ndarray)args[i];
                }
                else if (args[i] is flatiter)
                {
                    // TODO What this code needs to do... Is flatiter the right equiv to PyArrayIter?
                    //PyObject *new = PyObject_CallMethod(obj, "__array__", NULL);
                    //if (new == NULL) {
                    //    result = -1;
                    //    goto fail;
                    //} else if (!PyArray_Check(new)) {
                    //    PyErr_SetString(PyExc_TypeError,
                    //                    "__array__ must return an array.");
                    //    Py_DECREF(new);
                    //    result = -1;
                    //    goto fail;
                    //} else {
                    //    mps[i] = (PyArrayObject *)new;
                    //}
                    throw new NotImplementedException("Calling __array__ method on flatiter (PyArrayIter) is not yet implemented.");
                }
                else
                {
                    throw new ArgumentTypeException("return arrays must be of array type");
                }
            }

            return result;
        }
    }
}
