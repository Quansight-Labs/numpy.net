/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018-2021
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
using System.Runtime.InteropServices;
using System.Text;
using System.Numerics;
using NumpyLib;

#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif


namespace NumpyDotNet {

    public static partial class np
    {
        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules to the arguments.
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <param name="type_suggestion"></param>
        /// <returns></returns>
        internal static dtype result_type(object a1, object a2, object type_suggestion = null)
        {
            var arr1 = asanyarray(a1);
            var arr2 = asanyarray(a2);

            if (arr1.IsDecimal || arr2.IsDecimal)
                return np.Decimal;

            if (arr1.IsComplex || arr2.IsComplex)
                return np.Complex;

            return np.Float64;
        }
        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules to the arguments.
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        /// <param name="type_suggestion"></param>
        /// <returns></returns>
        internal static dtype result_type(dtype a1, dtype a2, object type_suggestion = null)
        {
            if (a1.IsDecimal || a2.IsDecimal)
                return np.Decimal;

            if (a1.IsComplex || a1.IsComplex)
                return np.Complex;

            return np.Float64;
        }
        /// <summary>
        /// Returns the type that results from applying the NumPy type promotion rules to the arguments.
        /// </summary>
        /// <param name="type_num"></param>
        /// <returns></returns>
        internal static dtype result_type(NPY_TYPES type_num)
        {
            var return_type = DefaultArrayHandlers.GetArrayHandler(type_num).MathOpFloatingType(UFuncOperation.divide);
            dtype result_dtype = NpyCoreApi.DescrFromType(return_type);
            return result_dtype;
        }

        /// <summary>
        /// Returns the data type with the smallest size and smallest scalar kind to which both type1 and type2
        /// </summary>
        /// <param name="type1"></param>
        /// <param name="type2"></param>
        /// <returns></returns>
        internal static dtype promote_types(dtype type1, dtype type2)
        {
            if (type1.TypeNum < type2.TypeNum)
                return type2;
            else
                return type1;
        }


        private static bool can_cast(object from_, dtype to, string casting = "safe")
        {
            dtype from = null;

            if (from_ is dtype)
            {
                from = from_ as dtype;
            }
            else
            {
                try
                {
                    var arr = asanyarray(from_);
                    if (arr != null)
                        from = arr.Dtype;
                }
                catch (Exception ex)
                {
                    return false;
                }
            }

            return NpyCoreApi.CanCastTo(from, to);
        }

    }
      


    internal static class NpyUtil_ArgProcessing {

        internal static bool BoolConverter(Object o) {
            return IntConverter(o) != 0;
        }


        internal static npy_intp IntPConverter(Object o)
        {
            if (o == null) return 0;
            else if (o is int) return (int)o;
            else return ConvertToIntP(o);
        }

        internal static npy_intp ConvertToIntP(object o)
        {
#if NPY_INTP_64
            return Convert.ToInt64(o);
#else
            return Convert.ToInt32(o);
#endif
        }

        internal static int IntConverter(Object o)
        {
            if (o == null) return 0;
            else if (o is int) return (int)o;
            else return Convert.ToInt32(o);
        }

        internal static int AxisConverter(object o, int dflt = NpyDefs.NPY_MAXDIMS)
        {
            if (o == null)
            {
                return dflt;
            }
            else if (o is int)
            {
                return (int)o;
            }
            else if (o is IConvertible)
            {
                return ((IConvertible)o).ToInt32(null);
            }
            else
            {
                throw new NotImplementedException(
                    String.Format("Type '{0}' is not supported for an axis.", o.GetType().Name));
            }
        }
    

        internal static char ByteorderConverter(string s) {
            if (s == null) {
                return 's';
            } else {
                if (s.Length == 0) {
                    throw new ArgumentException("Byteorder string must be at least length 1");
                }
                switch (s[0]) {
                    case '>':
                    case 'b':
                    case 'B':
                        return '>';
                    case '<':
                    case 'l':
                    case 'L':
                        return '<';
                    case '=':
                    case 'n':
                    case 'N':
                        return '=';
                    case 's':
                    case 'S':
                        return 's';
                    default:
                        throw new ArgumentException(String.Format("{0} is an unrecognized byte order"));
                }
            }
        }


        internal static ndarray[] ConvertToCommonType(IEnumerable<object> objs) {
            // Determine the type and size;
            // TODO: Handle scalars correctly.
            long n = 0;
            dtype intype = null;
            foreach (object o in objs) {
                intype = np.FindArrayType(o, intype, NpyDefs.NPY_MAXDIMS);
                ++n;
            }

            if (n == 0) {
                throw new ArgumentException("0-length sequence");
            }

            // Convert items to array objects
            return objs.Select(x => np.FromAny(x, intype, 0, 0, NPYARRAYFLAGS.NPY_CARRAY)).ToArray();
        }
    }


    internal static class NpyUtil_IndexProcessing
    {
        internal static void PureIndexConverter(ndarray arr, Int64[] indexArgs, NpyIndexes indexes)
        {
            int index = 0;
            // This is the simple case. Just convert each arg.
            if (indexArgs.Length > NpyDefs.NPY_MAXDIMS)
            {
                throw new IndexOutOfRangeException("Too many indices");
            }
            foreach (object arg in indexArgs)
            {
                ConvertSingleIndex(arr, arg, indexes, index++);
            }
        }

        internal static void IndexConverter(ndarray arr, Object[] indexArgs, NpyIndexes indexes)
        {
            int index = 0;

            if (indexArgs.Length != 1)
            {
                // This is the simple case. Just convert each arg.
                if (indexArgs.Length > NpyDefs.NPY_MAXDIMS)
                {
                    throw new IndexOutOfRangeException("Too many indices");
                }
                foreach (object arg in indexArgs)
                {
                    ConvertSingleIndex(arr, arg, indexes, index++);
                }
            }
            else
            {
                // Single index.
                object arg = indexArgs[0];
                if (arg is ndarray)
                {
                    ConvertSingleIndex(arr, arg, indexes, index++);
                }
                else if (arg is string)
                {
                    ConvertSingleIndex(arr, arg, indexes, index++);
                }
                else if (arg is IEnumerable<object> && SequenceTuple((IEnumerable<object>)arg))
                {
                    foreach (object sub in (IEnumerable<object>)arg)
                    {
                        ConvertSingleIndex(arr, sub, indexes,index++);
                    }
                }
                else
                {
                    ConvertSingleIndex(arr, arg, indexes, index++);
                }
            }
        }

        /// <summary>
        /// Determines whether or not to treat the sequence as multiple indexes
        /// We do this unless it looks like a sequence of indexes.
        /// </summary>
        private static bool SequenceTuple(IEnumerable<object> seq)
        {
            if (seq.Count() > NpyDefs.NPY_MAXDIMS)
                return false;

            foreach (object arg in seq)
            {
                if (arg == null ||
                    arg is Ellipsis ||
                    arg is ISlice ||
                    arg is IEnumerable<object>)
                    return true;
            }
            return false;
        }

        private static void ConvertSingleIndex(ndarray arr, Object arg, NpyIndexes indexes, int index)
        {
            if (arg == null)
            {
                indexes.AddNewAxis();
            }
            else if (arg is string && (string)arg == "...")
            {
                indexes.AddEllipsis();
            }
            else if (arg is Ellipsis)
            {
                indexes.AddEllipsis();
            }
            else if (arg is CSharpTuple)
            {
                indexes.AddCSharpTuple((CSharpTuple)arg);
            }
            else if (arg is bool)
            {
                indexes.AddIndex((bool)arg);
            }
            else if (arg is int)
            {
                indexes.AddIndex((int)arg);
            }
            else if (arg is Int64)
            {
                indexes.AddIndex((Int64)arg);
            }
            else if (arg is BigInteger)
            {
                BigInteger bi = (BigInteger)arg;
                npy_intp lval = (npy_intp)bi;
                indexes.AddIndex(lval);
            }
            else if (arg is ISlice)
            {
                indexes.AddIndex((ISlice)arg);
            }
            else if (arg is string)
            {
                indexes.AddIndex(arr, (string)arg, index);
            }
            else
            {
                ndarray array_arg = null;

                if (arg.GetType().IsArray)
                {
                    if (arg is ndarray[])
                    {

                    }
                    else
                    {
                        if (arg is Int64[])
                        {
                            array_arg = np.array(new VoidPtr(arg as Int64[]), null);
                        }
                        else
                        if (arg is Int32[])
                        {
                            array_arg = np.array(new VoidPtr(arg as Int32[]), null);
                        }
                        else
                        {
                            throw new Exception("Unexpected index type");
                        }
                    }
                }
                else
                {
                    array_arg = arg as ndarray;
                }


                if (array_arg == null && arg is IEnumerable<object>)
                {
                    array_arg = np.FromIEnumerable((IEnumerable<object>)arg, null, false, 0, 0);
                }
                if (array_arg == null && arg is IEnumerable<npy_intp>)
                {
                    var arr1 = arg as IEnumerable<npy_intp>;
                    array_arg = np.array(arr1.ToArray(), null);
                }

                // Boolean scalars
                if (array_arg != null &&
                    array_arg.ndim == 0 &&
                    NpyDefs.IsBool(array_arg.TypeNum))
                {
                    indexes.AddIndex(Convert.ToBoolean(array_arg[0]));
                }
                // Integer scalars
                else if (array_arg != null &&
                    array_arg.ndim == 0 &&
                    NpyDefs.IsInteger(array_arg.TypeNum))
                {
                    try
                    {
                        indexes.AddIndex((npy_intp)Convert.ToInt64(array_arg[0]));
                    }
                    catch (Exception e)
                    {
                        throw new IndexOutOfRangeException(e.Message);
                    }
                }
                else if (array_arg != null)
                {
                    // Arrays must be either boolean or integer.
                    if (NpyDefs.IsInteger(array_arg.TypeNum))
                    {
                        indexes.AddIntpArray(array_arg);
                    }
                    else if (NpyDefs.IsBool(array_arg.TypeNum))
                    {
                        indexes.AddBoolArray(array_arg);
                    }
                    else
                    {
                        throw new IndexOutOfRangeException("arrays used as indices must be of integer (or boolean) type.");
                    }
                }
                else if (arg is IEnumerable<Object>)
                {
                    // Other sequences we convert to an intp array
                    indexes.AddIntpArray(arg);
                }
                else if (arg is IConvertible)
                {
#if NPY_INTP_64
                    indexes.AddIndex((npy_intp)Convert.ToInt64(arg));
#else
                    indexes.AddIndex((npy_intp)Convert.ToInt32(arg));
#endif
                }
                else
                {
                    throw new ArgumentException(String.Format("Argument '{0}' is not a valid index.", arg));
                }
            }
        }
    }
}
