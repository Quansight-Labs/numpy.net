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
        // todo: big task to reimplement this.
        public static dtype result_type(params object []p)
        {
            return np.Float64;
        }

        // todo: big task to reimplement this.
        public static dtype promote_types(dtype type1, dtype type2)
        {
            if (type1.TypeNum < type2.TypeNum)
                return type1;
            else
                return type2;
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


    /// <summary>
    /// Package of extension methods.
    /// </summary>
    internal static class NpyUtils_Extensions {

        /// <summary>
        /// Applies function f to all elements in 'input'. Same as Select() but
        /// with no result.
        /// </summary>
        /// <typeparam name="Tin">Element type</typeparam>
        /// <param name="input">Input sequence</param>
        /// <param name="f">Function to be applied</param>
        public static void Iter<Tin>(this IEnumerable<Tin> input, Action<Tin> f) {
            foreach (Tin x in input) {
                f(x);
            }
        }

        /// <summary>
        /// Applies function f to all elements in 'input' plus the index of each
        /// element.
        /// </summary>
        /// <typeparam name="Tin">Type of input elements</typeparam>
        /// <param name="input">Input sequence</param>
        /// <param name="f">Function to be applied</param>
        public static void Iteri<Tin>(this IEnumerable<Tin> input, Action<Tin, int> f) {
            int i = 0;
            foreach (Tin x in input) {
                f(x, i);
                i++;
            }
        }
    }


    /// <summary>
    /// A package of utilities for dealing with Python
    /// </summary>
    internal static class NpyUtil_Python
    {
        internal static bool IsIntegerScalar(object o) {
            return (o is int || o is BigInteger || o is ScalarInteger);
        }

        internal static bool IsTupleOfIntegers(object o) {
            PythonTuple t = o as PythonTuple;
            if (t == null) {
                return false;
            }
            foreach (object item in t) {
                if (!IsIntegerScalar(item)) {
                    return false;
                }
            }
            return true;
        }

        internal static PythonTuple ToPythonTuple(npy_intp[] array) {
            int n = array.Length;
            object[] vals = new object[n];
            // Convert to Python types
            for (int i = 0; i < n; i++) {
                long v = array[i];
                if (v < int.MinValue || v > int.MaxValue) {
                    vals[i] = new BigInteger(v);
                } else {
                    vals[i] = (int)v;
                }
            }
            // Make the tuple
            return new PythonTuple(vals);
        }

        internal static object ToPython(long l) {
            if (l < int.MinValue || l > int.MaxValue) {
                return new BigInteger(l);
            } else {
                return (int)l;
            }
        }

        internal static object ToPython(UInt64 l) {
            if (l > int.MaxValue) {
                return new BigInteger(l);
            } else {
                return (int)l;
            }
        }

        internal static object ToPython(UInt32 l) {
            if (l > int.MaxValue) {
                return new BigInteger(l);
            } else {
                return (int)l;
            }
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
            else return Convert.ToInt64(o);
        }

        internal static int IntConverter(Object o)
        {
            if (o == null) return 0;
            else if (o is int) return (int)o;
            else return Convert.ToInt32(o);
        }

        internal static long LongConverter(Object o) {
            if (o == null) return 0;
            else return Convert.ToInt64(o);
        }

        /// <summary>
        /// Converts an input sequence or scalar to a long[].  Equivalent to
        /// PyArray_IntpFromSequence.
        /// </summary>
        /// <param name="o">Sequence or scalar integer value</param>
        /// <returns>Array of long values</returns>
        internal static npy_intp[] IntArrConverter(Object o)
        {
            if (o == null) return null;
            else if (o is IEnumerable<Object>)
            {
                return ((IEnumerable<Object>)o).Select(x => IntPConverter(x)).ToArray();
            }
            else
            {
                return new npy_intp[1] { IntConverter(o) };
            }
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

        internal static NPY_CLIPMODE ClipmodeConverter(object o) {
            if (o == null) return NPY_CLIPMODE.NPY_RAISE;
            else if (o is string) {
                string s = (string)o;
                switch (s[0]) {
                    case 'C':
                    case 'c':
                        return NPY_CLIPMODE.NPY_CLIP;
                    case 'W':
                    case 'w':
                        return NPY_CLIPMODE.NPY_WRAP;
                    case 'r':
                    case 'R':
                        return NPY_CLIPMODE.NPY_RAISE;
                    default:
                        throw new ArgumentTypeException("clipmode not understood");
                }
            } else {
                int i = IntConverter(o);
                if (i < (int)NPY_CLIPMODE.NPY_CLIP || i > (int)NPY_CLIPMODE.NPY_RAISE) {
                    throw new ArgumentTypeException("clipmode not understood");
                }
                return (NPY_CLIPMODE)i;
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


        internal static npy_intp[] IntListConverter(IEnumerable<object> args)
        {
            npy_intp[] result = new npy_intp[args.Count()];
            int i = 0;
            foreach (object arg in args)
            {
                result[i++] = IntConverter(arg);
            }
            return result;
        }


        internal static Object[] BuildArgsArray(Object[] posArgs, String[] kwds, IDictionary<object, object> namedArgs) {
            // For some reason the name of the attribute can only be access via ToString
            // and not as a key so we fix that here.
            if (namedArgs == null) {
                return new Object[kwds.Length];
            }

            Dictionary<String, Object> argsDict = namedArgs
                .Select(kvPair => new KeyValuePair<String, Object>(kvPair.Key.ToString(), kvPair.Value))
                .ToDictionary((kvPair => kvPair.Key), (kvPair => kvPair.Value));

            // The result, filled in as we go.
            Object[] args = new Object[kwds.Length];
            int i;

            // Copy in the position arguments.
            for (i = 0; i < posArgs.Length; i++) {
                if (argsDict.ContainsKey(kwds[i])) {
                    throw new ArgumentException(String.Format("Argument '{0}' is specified both positionally and by name.", kwds[i]));
                }
                args[i] = posArgs[i];
            }

            // Now insert any named arguments into the correct position.
            for (i = posArgs.Length; i < kwds.Length; i++) {
                if (argsDict.TryGetValue(kwds[i], out args[i])) {
                    argsDict.Remove(kwds[i]);
                } else {
                    args[i] = null;
                }
            }
            if (argsDict.Count > 0) {
                throw new ArgumentTypeException("Unknown named arguments were specified.");
            }
            return args;
        }


        internal static NPY_SORTKIND SortkindConverter(string kind) {
            if (kind == null) {
                return NPY_SORTKIND.NPY_QUICKSORT;
            }
            if (kind.Length < 1) {
                throw new ArgumentException("Sort kind string must be at least length 1");
            }
            switch (kind[0]) {
                case 'q':
                case 'Q':
                    return NPY_SORTKIND.NPY_QUICKSORT;
                case 'h':
                case 'H':
                    return NPY_SORTKIND.NPY_HEAPSORT;
                case 'm':
                case 'M':
                    return NPY_SORTKIND.NPY_MERGESORT;
                default:
                    throw new ArgumentException(String.Format("{0} is an unrecognized kind of SortedDictionary", kind));
            }
        }

        internal static NPY_SEARCHSIDE SearchsideConverter(string side) {
            if (side == null) {
                return NPY_SEARCHSIDE.NPY_SEARCHLEFT;
            }
            if (side.Length < 1) {
                throw new ArgumentException("Expected nonexpty string for keyword 'side'");
            }
            switch (side[0]) {
                case 'l':
                case 'L':
                    return NPY_SEARCHSIDE.NPY_SEARCHLEFT;
                case 'r':
                case 'R':
                    return NPY_SEARCHSIDE.NPY_SEARCHRIGHT;
                default:
                    throw new ArgumentException(String.Format("'{0}' is an InvalidCastException value for keyword 'side'", side));
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
        public static void IndexConverter(ndarray arr, Object[] indexArgs, NpyIndexes indexes)
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
            else if (arg is ScalarInt16)
            {
                indexes.AddIndex((int)(ScalarInt16)arg);
            }
            else if (arg is ScalarInt32)
            {
                indexes.AddIndex((int)(ScalarInt32)arg);
            }
            else if (arg is ScalarInt64)
            {
                indexes.AddIndex((int)(ScalarInt64)arg);
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
                    dynamic arr1 = arg;
                    if (arr1[0] is ndarray)
                    {

                    }
                    else
                    {
                        array_arg = np.array(new VoidPtr(arr1), null);
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
                    NpyDefs.IsBool(array_arg.Dtype.TypeNum))
                {
                    indexes.AddIndex(Convert.ToBoolean(array_arg[0]));
                }
                // Integer scalars
                else if (array_arg != null &&
                    array_arg.ndim == 0 &&
                    NpyDefs.IsInteger(array_arg.Dtype.TypeNum))
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
                    if (NpyDefs.IsInteger(array_arg.Dtype.TypeNum))
                    {
                        indexes.AddIntpArray(array_arg);
                    }
                    else if (NpyDefs.IsBool(array_arg.Dtype.TypeNum))
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
