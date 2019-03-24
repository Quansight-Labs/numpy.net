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
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Reflection;
using NumpyLib;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet {
    /// <summary>
    /// ModuleMethods implements the module-level numpy functions.
    /// </summary>
    public static class ModuleMethods {
        private static String[] arrayKwds = { "object", "dtype", "copy", "order", "subok", "ndmin" };


        /// <summary>
        /// Ick. See ScalarGeneric.Initialized.
        /// </summary>
        public static void InitializeScalars() {
            ScalarGeneric.Initialized = true;
        }

  

        public static void putmask(ndarray arr, object mask, object values) {
            ndarray aMask;
            ndarray aValues;

            aMask = (mask as ndarray);
            if (aMask == null) {
                aMask = np.FromAny(mask, NpyCoreApi.DescrFromType(NPY_TYPES.NPY_BOOL),
                    0, 0, NPYARRAYFLAGS.NPY_CARRAY | NPYARRAYFLAGS.NPY_FORCECAST, null);
            }

            aValues = (values as ndarray);
            if (aValues == null) {
                aValues = np.FromAny(values, arr.Dtype, 0, 0, NPYARRAYFLAGS.NPY_CARRAY, null);
            }

            arr.PutMask(aValues, aMask);
        }


        public static ndarray lexsort(object keysObj, int axis = -1) {
            IEnumerable<object> keys = keysObj as IEnumerable<object>;
            if (keys == null || keys.Count() == 0) {
                throw new ArgumentTypeException("Need sequence of keys with len > 0 in lexsort");
            }

            int n = keys.Count();
            ndarray[] arrays = new ndarray[n];
            int i = 0;
            foreach (object k in keys) {
                arrays[i++] = np.FromAny(k);
            }
            return ndarray.LexSort(arrays, axis);
        }


        public static ndarray concatenate(IEnumerable<ndarray> seq, int axis = 0) {
            return np.Concatenate(seq, axis);
        }

        public static object inner(object o1, object o2) {
            return ndarray.ArrayReturn(np.inner(o1, o2));
        }

    

        public static object dot(object o1, object o2)
        {
            return ndarray.ArrayReturn(np.MatrixProduct(o1, o2));
        }


        public static object _fastCopyAndTranspose(object a) {
            ndarray arr = np.FromAny(a, flags: NPYARRAYFLAGS.NPY_CARRAY);
            return NpyCoreApi.CopyAndTranspose(arr);
        }

  
        public static string format_longfloat(object x, int precision) {
            if (x is ScalarFloat64) {
                return NpyCoreApi.FormatLongFloat((double)(ScalarFloat64)x, precision);
            } else if (x is double) {
                return NpyCoreApi.FormatLongFloat((double)x, precision);
            }
            throw new NotImplementedException(
                String.Format("Unhandled long float type '{0}'", x.GetType().Name));
        }


        public static object compare_chararrays(object a1, object a2, string cmp, object rstrip) {
            bool rstripFlag = NpyUtil_ArgProcessing.BoolConverter(rstrip);

            NpyDefs.NPY_COMPARE_OP cmpOp;

            if (cmp == "==") cmpOp = NpyDefs.NPY_COMPARE_OP.NPY_EQ;
            else if (cmp == "!=") cmpOp = NpyDefs.NPY_COMPARE_OP.NPY_NE;
            else if (cmp == "<") cmpOp = NpyDefs.NPY_COMPARE_OP.NPY_LT;
            else if (cmp == "<=") cmpOp = NpyDefs.NPY_COMPARE_OP.NPY_LE;
            else if (cmp == ">=") cmpOp = NpyDefs.NPY_COMPARE_OP.NPY_GE;
            else if (cmp == ">") cmpOp = NpyDefs.NPY_COMPARE_OP.NPY_GT;
            else throw new ArgumentException("comparison must be '==', '!=', '<', '>', '<=', or '>='");

            ndarray arr1 = np.FromAny(a1);
            ndarray arr2 = np.FromAny(a2);
            if (arr1 == null || arr2 == null) {
                return null;
            }

            ndarray res;
            if (arr1.IsString && arr2.IsString) {
                if (arr1.Dtype.TypeNum == NPY_TYPES.NPY_STRING &&
                    arr2.Dtype.TypeNum == NPY_TYPES.NPY_UNICODE) {
                    arr1 = PromoteToUnicode(arr1, arr2);
                } else if (arr2.Dtype.TypeNum == NPY_TYPES.NPY_STRING &&
                    arr1.Dtype.TypeNum == NPY_TYPES.NPY_UNICODE) {
                    arr2 = PromoteToUnicode(arr2, arr1);
                }
                res = NpyCoreApi.CompareStringArrays(arr1, arr2, cmpOp, rstripFlag);
            } else {
                throw new ArgumentTypeException("comparison of non-string arrays");
            }
            return res;
        }

        private static ndarray PromoteToUnicode(ndarray arr, ndarray src) {
            dtype unicode = new dtype(src.Dtype);
            unicode.ElementSize = src.itemsize * 4;
            return np.FromAny(arr, unicode);
        }

     
    }
}
