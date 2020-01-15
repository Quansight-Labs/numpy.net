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
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Reflection;
using System.Numerics;
using NumpyLib;

#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{
    /// <summary>
    /// Implements the Numpy python 'ndarray' object and acts as an interface to
    /// the core NpyArray data structure.  Npy_INTERFACE(NpyArray *) points an
    /// instance of this class.
    /// </summary>
    public partial class ndarray : IEnumerable<object>, NumpyDotNet.IArray
    {
        public NpyArray core;
        public ndarray() {
        }

        public ndarray(NpyArray a)
        {
            if (a == null)
            {
                throw new Exception("Attempt to create ndarray with null object");
            }
            core = a;
        }


        #region Public interfaces (must match CPython)

        private static Func<ndarray, string> reprFunction;
        private static Func<ndarray, string> strFunction;

        /// <summary>
        /// Sets a function to be triggered for the repr() operator or null to default to the
        /// built-in version.
        /// </summary>
        public static Func<ndarray, string> ReprFunction {
            get { return reprFunction; }
            internal set { reprFunction = (value != null) ? value : x => x.BuildStringRepr(true); }
        }

        /// <summary>
        /// Sets a function to be triggered on the str() operator or ToString() method. Null defaults to
        /// the built-in version.
        /// </summary>
        public static Func<ndarray, string> StrFunction {
            get { return strFunction; }
            internal set { strFunction = (value != null) ? value : x => x.BuildStringRepr(false); }
        }

        static ndarray() {
            ReprFunction = null;
            StrFunction = null;
        }


        #region Operators

        internal static ndarray BinaryOp(ndarray a, object b, ufunc f)
        {
            return NpyCoreApi.PerformNumericOp(a, f.UFunc.ops, np.asanyarray(b), false);
        }

        internal static object BinaryOpInPlace(ndarray a, object b, ufunc f, ndarray ret)
        {
            ndarray numericOpResult = NpyCoreApi.PerformNumericOp(a, f.UFunc.ops, np.asanyarray(b), true);
            if (numericOpResult != null && ret != null)
            {
                NpyCoreApi.CopyAnyInto(ret, numericOpResult);
            }

            return numericOpResult;
        }

        internal static ndarray BinaryOp(ndarray a, object b, NpyArray_Ops op)
        {
            ufunc f = NpyCoreApi.GetNumericOp(op);
            return BinaryOp(a, b, f);
        }

        public static object BinaryOpInPlace(ndarray a, object b, NpyArray_Ops op, ndarray ret)
        {
            ufunc f = NpyCoreApi.GetNumericOp(op);
            return BinaryOpInPlace(a, b, f, ret);
        }

        internal static object UnaryOp(ndarray a, NpyArray_Ops op)
        {
            return NpyCoreApi.PerformNumericOp(a, op, 0, false);
        }


        internal static object UnaryOpInPlace(ndarray a, NpyArray_Ops op, ndarray ret)
        {
            ndarray numericOpResult = NpyCoreApi.PerformNumericOp(a, op, 0, true);

            if (numericOpResult != null && ret != null)
            {
                NpyCoreApi.CopyAnyInto(ret, numericOpResult);
            }

            return numericOpResult;
        }



        public static object operator +(ndarray a) {
            return a;
        }

        public static ndarray operator +(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_add, operand);
        }

        public static ndarray operator +(object operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_add, operand);
        }

        public static ndarray operator +(ndarray a, ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_add, b);
        }

        [SpecialName]
        public ndarray InPlaceAdd(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, NpyArray_Ops.npy_op_add, b, true);
        }

        public static ndarray operator -(ndarray a, object operand) {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_subtract, operand);
        }
        public static ndarray operator -(object operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_subtract, operand);
        }

        public static ndarray operator -(ndarray a, ndarray b) {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_subtract, b);
        }


        [SpecialName]
        public ndarray InPlaceSubtract(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, NpyArray_Ops.npy_op_subtract, b, true);
        }

        public static ndarray operator -(ndarray a) {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_negative, 0);
        }

        public static ndarray operator *(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_multiply, operand);
        }

        public static ndarray operator *(object operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_multiply, operand);
        }

        public static ndarray operator *(ndarray a, ndarray b) {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_multiply, b);
        }

        [SpecialName]
        public ndarray InPlaceMultiply(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, NpyArray_Ops.npy_op_multiply, b, true);
        }


        public static ndarray operator /(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_divide, operand);
        }
        public static ndarray operator /(object operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_divide, operand);
        }


        public static object operator /(ndarray a, ndarray b) {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_divide, b);
        }

        [SpecialName]
        public ndarray InPlaceDivide(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, NpyArray_Ops.npy_op_divide, b, true);
        }

        [SpecialName]
        public ndarray InPlaceTrueDivide(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, NpyArray_Ops.npy_op_true_divide, b, true);
        }

        [SpecialName]
        public ndarray InPlaceFloorDivide(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, NpyArray_Ops.npy_op_floor_divide, b, true);
        }

        public static ndarray operator %(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_remainder, operand);
        }
        public static ndarray operator %(object operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_remainder, operand);
        }
        public static ndarray operator %(ndarray a, ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_remainder, b);
        }



        public static ndarray operator &(ndarray a, Int64 operand) {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_bitwise_and, operand);
        }
        public static ndarray operator &(Int64 operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_bitwise_and, operand);
        }

        public static ndarray operator &(ndarray a, ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_bitwise_and, b);
        }

        [SpecialName]
        public ndarray InPlaceBitwiseAnd(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, NpyArray_Ops.npy_op_bitwise_and, b, true);
        }

        public static ndarray operator |(ndarray a, Int64 operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_bitwise_or, operand);
        }
        public static ndarray operator |(Int64 operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_bitwise_or, operand);
        }

        public static ndarray operator |(ndarray a, ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_bitwise_or, b);
        }

        [SpecialName]
        public ndarray InPlaceBitwiseOr(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, NpyArray_Ops.npy_op_bitwise_or, b, true);
        }

        public static ndarray operator ^(ndarray a, Int64 operand) {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_bitwise_xor, operand);
        }
        public static ndarray operator ^(Int64 operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_bitwise_xor, operand);
        }

        public static object operator ^(ndarray a, ndarray b) {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_bitwise_xor, b);
        }

        public static ndarray operator <<(ndarray a, int shift) {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_left_shift, shift);
        }

        public static ndarray operator >>(ndarray a, int shift) {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_right_shift, shift);
        }

        public static ndarray operator <(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_less, operand);
        }
        public static ndarray operator <=(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_less_equal, operand);
        }
        public static ndarray operator >(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_greater, operand);
        }
        public static ndarray operator >=(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_greater_equal, operand);
        }
        public static ndarray operator ==(ndarray a, double operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_equal, operand);
        }
        public static ndarray operator ==(ndarray a, float operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_equal, operand);
        }
        public static ndarray operator ==(ndarray a, Int64 operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_equal, operand);
        }
        public static ndarray operator ==(ndarray a, Int32 operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_equal, operand);
        }
        public static ndarray operator ==(ndarray a, bool operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_equal, operand);
        }
        public static ndarray operator ==(ndarray a, decimal operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_equal, operand);
        }
        public static ndarray operator ==(ndarray a, Complex operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_equal, operand);
        }
        public static ndarray operator ==(ndarray a, BigInteger operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_equal, operand);
        }
        [SpecialName]
        public ndarray Equals(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, NpyArray_Ops.npy_op_equal, b);
        }
        public static ndarray operator !=(ndarray a, double operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, float operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, Int64 operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, Int32 operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, bool operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, decimal operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, Complex operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, BigInteger operand)
        {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_not_equal, operand);
        }
        [SpecialName]
        public ndarray NotEquals(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, NpyArray_Ops.npy_op_not_equal, b);
        }

        [SpecialName]
        public ndarray InPlaceExclusiveOr(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, NpyArray_Ops.npy_op_bitwise_xor, b, true);
        }

        public static ndarray operator ~(ndarray a) {
            return NpyCoreApi.PerformNumericOp(a, NpyArray_Ops.npy_op_invert, 0);
        }

        public static implicit operator String(ndarray a) {
            return StrFunction(a);
        }



        public static explicit operator bool(ndarray arr)
        {
            int val = NpyCoreApi.ArrayBool(arr);
            if (val < 0)
            {
                NpyCoreApi.CheckError();
                return false;
            }
            else
            {
                return val != 0;
            }
        }



        #endregion

        #region indexing

        public object this[int index]
        {
            get
            {
                return GetArrayItem((npy_intp)index);
            }
            set
            {
                SetArrayItem((npy_intp)index, value);
            }
        }

        public object this[long index]
        {
            get
            {
                return GetArrayItem((npy_intp)index);
            }
            set
            {
                SetArrayItem((npy_intp)index, value);
            }
        }

        public object this[BigInteger index]
        {
            get
            {
                npy_intp lIndex = (npy_intp)index;
                return GetArrayItem(lIndex);
            }
            set
            {
                npy_intp lIndex = (npy_intp)index;
                SetArrayItem(lIndex, value);
            }
        }

        public Object SliceMe(params object[] args)
        {
            return this[args];
        }

        public ndarray A(params object[] args)
        {
            ndarray ret = this[args] as ndarray;
            if (ret == null)
            {
                throw new Exception("This operation did not result in an expected ndarray");
            }

            return ret;
        }

        public Object this[params object[] args]
        {
            get
            {
                if (args == null)
                {
                    args = new object[] { null };
                }
                //else
                //{
                //    if (args.Length == 1 && args[0] is PythonTuple)
                //    {
                //        args = ((IEnumerable<object>)args[0]).ToArray();
                //    }

                //    if (args.Length == 1 && args[0] is string)
                //    {
                //        string field = (string)args[0];
                //        return ArrayReturn(NpyCoreApi.GetField(this, field));
                //    }
                //}
                NpyIndexes indexes = new NpyIndexes();
                {
                    NpyUtil_IndexProcessing.IndexConverter(this, args, indexes);
                    if (indexes.IsSingleItem(ndim))
                    {
                        // Optimization for single item index.
                        long offset = 0;
                        npy_intp[] dims = this.dims;
                        npy_intp[] s = strides;
                        for (int i = 0; i < ndim; i++)
                        {
                            long d = dims[i];
                            long val = indexes.GetIntP(i);
                            if (val < 0)
                            {
                                val += d;
                            }
                            if (val < 0 || val >= d)
                            {
                                throw new IndexOutOfRangeException();
                            }
                            offset += val * s[i];
                        }
                        return Dtype.ToScalar(this, offset);
                    }
                    else if (indexes.IsMultiField)
                    {
                        throw new Exception("Don't currenty support multi-fields");
                    }

                    ndarray result = null;

                    if (indexes.IsAdvancedIndexing)
                    {

                        // advanced subscript case.
                        NpyCoreApi.Incref(Array);
                        //var newDType = NpyCoreApi.DescrFromType(this.Array.ItemType);
                        //NpyCoreApi.DescrReplaceSubarray(newDType, this.Dtype, result.dims);
                        result = new ndarray(NpyCoreApi.ArraySubscript(this, indexes));
                        result = NpyCoreApi.FromArray(result, null, NPYARRAYFLAGS.NPY_ENSURECOPY);

                        NpyCoreApi.Decref(Array);
                    }
                    else
                    {
                        // General subscript case.
                        NpyCoreApi.Incref(Array);
                        result = new ndarray(NpyCoreApi.ArraySubscript(this, indexes));
                        NpyCoreApi.Decref(Array);
                    }



                    if (result.ndim == 0)
                    {
                        // We only want to return a scalar if there are not elipses
                        bool noelipses = true;
                        int n = indexes.NumIndexes;
                        for (int i = 0; i < n; i++)
                        {
                            NpyIndexType t = indexes.IndexType(i);
                            if (t == NpyIndexType.NPY_INDEX_ELLIPSIS ||
                                t == NpyIndexType.NPY_INDEX_STRING ||
                                t == NpyIndexType.NPY_INDEX_BOOL)
                            {
                                noelipses = false;
                                break;
                            }
                        }
                        if (noelipses)
                        {
                            return result.Dtype.ToScalar(this);
                        }
                    }

                    return result;
                }
            }
            set
            {
                if (!ChkFlags(NPYARRAYFLAGS.NPY_WRITEABLE))
                {
                    throw new RuntimeException("array is not writeable.");
                }

                if (args == null)
                {
                    args = new object[] { null };
                }
                //else
                //{
                //    if (args.Length == 1 && args[0] is PythonTuple)
                //    {
                //        PythonTuple pt = (PythonTuple)args[0];
                //        args = pt.ToArray();
                //    }

                //    if (args.Length == 1 && args[0] is string)
                //    {
                //        string field = (string)args[0];
                //        if (!ChkFlags(NPYARRAYFLAGS.NPY_WRITEABLE))
                //        {
                //            throw new RuntimeException("array is not writeable.");
                //        }
                //        NpyArray_Descr descr = null;
                //        int offset = NpyCoreApi.GetFieldOffset(Dtype, field, ref descr);
                //        if (offset < 0)
                //        {
                //            throw new ArgumentException(String.Format("field name '{0}' not found.", field));
                //        }
                //        np.SetField(this, descr, offset, value);
                //        return;
                //    }
                //}


                NpyIndexes indexes = new NpyIndexes();
                {
                    NpyUtil_IndexProcessing.IndexConverter(this, args, indexes);

                    // Special case for boolean on 0-d arrays.
                    if (ndim == 0 && indexes.NumIndexes == 1 && indexes.IndexType(0) == NpyIndexType.NPY_INDEX_BOOL)
                    {
                        if (indexes.GetBool(0))
                        {
                            SetItem(value, 0);
                        }
                        return;
                    }

                    // Special case for single assignment.
                    long single_offset = indexes.SingleAssignOffset(this);
                    if (single_offset >= 0 && np.IsNumericType(value))
                    {
                        // This is a single item assignment. Use SetItem.
                        SetItem(value, single_offset / this.ItemSize);
                        return;
                    }

                    if (indexes.IsSimple)
                    {
                        ndarray view = null;
                        try
                        {
                            if (GetType() == typeof(ndarray))
                            {
                                view = NpyCoreApi.IndexSimple(this, indexes);
                            }
                            else
                            {
                                throw new Exception("not an ndarray");
                            }

                            if (view != null)
                            {
                                np.CopyObject(view, value);
                            }
                        }
                        finally
                        {
                            if (view != null)
                            {

                            }
                        }
                    }
                    else
                    {
                        ndarray array_value = np.FromAny(value, Dtype, 0, 0, NPYARRAYFLAGS.NPY_FORCECAST, null);
                        try
                        {
                            // KM: this hack lets the IndexFancyAssign work
                            if (array_value.Array.nd == 0)
                            {
                                array_value.Array.nd = 1;
                                array_value.Array.dimensions = new npy_intp[1] { 1 };
                                array_value.Array.strides = new npy_intp[1] { array_value.ItemSize };
                            }
                            // KM: this hack lets the IndexFancyAssign work

                            NpyCoreApi.Incref(array_value.Array);
                            if (NpyCoreApi.IndexFancyAssign(this, indexes, array_value) < 0)
                            {
                                NpyCoreApi.CheckError();
                            }
                        }
                        finally
                        {
                            NpyCoreApi.Decref(array_value.Array);
                        }
                    }
                }
            }
        }

        #endregion

        #region properties

        /// <summary>
        /// Number of dimensions in the array
        /// </summary>
        public int ndim {

            get { return core.nd; }
        }

        /// <summary>
        /// Returns the size of each dimension as a tuple.
        /// </summary>
        public shape shape
        {
            get
            {
                return new shape(this.dims, this.ndim);
            }
        }


        /// <summary>
        /// Total number of elements in the array.
        /// </summary>
        public npy_intp size {
            get { return NpyCoreApi.ArraySize(this); }
        }

        public VoidPtr rawdata(npy_intp index = 0)
        {
            var flattened = this.flatten();
            return numpyAPI.NpyArray_Index2Ptr(flattened.Array, index);
        }

        /// <summary>
        /// Returns the reference count of the core array object.  Used for debugging only.
        /// </summary>
        public uint __coreRefCount__ { get { return Array.RefCount; } }

        public ndarray __array_wrap__(ndarray a)
        {
            return a;
        }

        /// <summary>
        /// The type descriptor object for this array
        /// </summary>
        public dtype Dtype {
            get {
                if (core == null) return null;
                return new dtype(core.descr);
            }
            set {
                NpyCoreApi.ArraySetDescr(this, value);
            }
        }


        /// <summary>
        /// The type descriptor object for this array
        /// </summary>
        public object dtype {
            get {
                return this.Dtype;
            }
            //set {
            //    dtype descr = value as dtype;
            //    if (descr == null) {
            //        descr = NpyDescr.DescrConverter(NpyUtil_Python.DefaultContext, value);
            //    }
            //    NpyCoreApi.ArraySetDescr(this, descr);
            //}
        }

        /// <summary>
        /// Flags for this array
        /// </summary>
        public flagsobj flags {
            get {
                return new flagsobj(this);
            }
        }

  
        public object flat {
            get {
                return NpyCoreApi.IterNew(this);
            }
            set {
                // Assing like a.flat[:] = value
                flatiter it = NpyCoreApi.IterNew(this);
                it[new Slice(null)] = value;
            }
        }

        public object @base {
            get {
                // TODO: Handle non-array bases
                return BaseArray;
            }
        }

        public int ItemSize {
            get {

                if (core == null)
                    return 0;

                return core.descr.elsize;
            }
        }

        public NPY_TYPES TypeNum
        {
            get
            {
                if (core == null)
                    return NPY_TYPES.NPY_VOID;

                return core.descr.type_num;
            }
        }

        public long nbytes {
            get {
                return ItemSize * Size;
            }
        }

        public ndarray T {
            get {
                return this.Transpose();
            }
        }


        #endregion

        #region methods


        public ndarray astype(dtype dtype = null, string order = "K", string casting = "unsafe", bool subok = true, bool copy = true)
        {
            if (dtype == this.Dtype)
            {
                return this;
            }
            if (this.Dtype.HasNames)
            {
                // CastToType doesn't work properly for
                // record arrays, so we use FromArray.
                NPYARRAYFLAGS flags = NPYARRAYFLAGS.NPY_FORCECAST;
                if (IsFortran)
                {
                    flags |= NPYARRAYFLAGS.NPY_FORTRAN;
                }
                return NpyCoreApi.FromArray(this, dtype, flags);
            }
            return NpyCoreApi.CastToType(this, dtype, this.IsFortran);
        }

        public ndarray byteswap(bool inplace = false) {
            return NpyCoreApi.Byteswap(this, inplace);
        }

        private static string[] chooseArgNames = { "out", "mode" };

        public object copy(NPY_ORDER order = NPY_ORDER.NPY_CORDER) {
            return ArrayReturn(Copy(order));
        }

        public ndarray Copy(NPY_ORDER order = NPY_ORDER.NPY_CORDER) {
            return NpyCoreApi.NewCopy(this, order);
        }


        public ndarray dot(object other)
        {
            return np.MatrixProduct(this, other);
        }

        public void fill(object scalar) {
            FillWithScalar(scalar);
        }

        public ndarray flatten(NPY_ORDER order = NPY_ORDER.NPY_CORDER) {
            return this.Flatten(order);
        }


        public object item(params object[] args)
        {
            if (args != null && args.Length == 1 && args[0] is PythonTuple)
            {
                PythonTuple t = (PythonTuple)args[0];
                args = t.ToArray();
            }
            if (args == null || args.Length == 0)
            {
                if (ndim == 0 || Size == 1)
                {
                    return GetItem(0);
                }
                else
                {
                    throw new ArgumentException("can only convert an array of size 1 to a Python scalar");
                }
            }
            else
            {
                NpyIndexes indexes = new NpyIndexes();
                {
                    NpyUtil_IndexProcessing.IndexConverter(this, args, indexes);
                    if (args.Length == 1)
                    {
                        if (indexes.IndexType(0) != NpyIndexType.NPY_INDEX_INTP)
                        {
                            throw new ArgumentException("invalid integer");
                        }
                        // Do flat indexing
                        return Flat.Get(indexes.GetIntP(0));
                    }
                    else
                    {
                        if (indexes.IsSingleItem(ndim))
                        {
                            long offset = indexes.SingleAssignOffset(this);
                            return GetItem(offset / this.ItemSize);
                        }
                        else
                        {
                            throw new ArgumentException("Incorrect number of indices for the array");
                        }
                    }
                }
            }
        }

        public void itemset(params object[] args)
        {
            // Convert args to value and args
            if (args == null || args.Length == 0)
            {
                throw new ArgumentException("itemset must have at least one argument");
            }
            object value = args.Last();
            args = args.Take(args.Length - 1).ToArray();

            if (args.Length == 1 && args[0] is PythonTuple)
            {
                PythonTuple t = (PythonTuple)args[0];
                args = t.ToArray();
            }
            if (args.Length == 0)
            {
                if (ndim == 0 || Size == 1)
                {
                    SetItem(value, 0);
                }
                else
                {
                    throw new ArgumentException("can only convert an array of size 1 to a Python scalar");
                }
            }
            else
            {
                NpyIndexes indexes = new NpyIndexes();
                {
                    NpyUtil_IndexProcessing.IndexConverter(this, args, indexes);
                    if (args.Length == 1)
                    {
                        if (indexes.IndexType(0) != NpyIndexType.NPY_INDEX_INTP)
                        {
                            throw new ArgumentException("invalid integer");
                        }
                        // Do flat indexing
                        Flat.SingleAssign(indexes.GetIntP(0), value);
                    }
                    else
                    {
                        if (indexes.IsSingleItem(ndim))
                        {
                            long offset = indexes.SingleAssignOffset(this);
                            SetItem(value, offset / this.ItemSize);
                        }
                        else
                        {
                            throw new ArgumentException("Incorrect number of indices for the array");
                        }
                    }
                }
            }
        }


        public ndarray newbyteorder(string endian = null) {
            dtype newtype = NpyCoreApi.DescrNewByteorder(Dtype, NpyUtil_ArgProcessing.ByteorderConverter(endian));
            return NpyCoreApi.View(this, newtype, null);
        }


        public int put(object indices, object values, object mode = null)
        {
            return np.put(this, indices, values, mode);
        }

        public ndarray ravel(NPY_ORDER order = NPY_ORDER.NPY_CORDER) {
            return this.Ravel(order);
        }

        public ndarray reshape(IEnumerable<npy_intp> shape, NPY_ORDER order = NPY_ORDER.NPY_ANYORDER)
        {
            npy_intp[] newshape = shape.Select(x => (npy_intp)x).ToArray();

            return NpyCoreApi.Newshape(this, newshape, order);
        }

        public ndarray reshape(int shape, NPY_ORDER order = NPY_ORDER.NPY_ANYORDER)
        {
            npy_intp[] newshape = new npy_intp[] { shape };

            return NpyCoreApi.Newshape(this, newshape, order);
        }
        private static string[] resizeKeywords = { "refcheck" };


        public void setflags(object write = null, object align = null, object uic = null)
        {
            NPYARRAYFLAGS flags = RawFlags;
            if (align != null)
            {
                bool bAlign = NpyUtil_ArgProcessing.BoolConverter(align);
                if (bAlign)
                {
                    flags |= NPYARRAYFLAGS.NPY_ALIGNED;
                }
                else
                {
                    if (!NpyCoreApi.IsAligned(this))
                    {
                        throw new ArgumentException("cannot set aligned flag of mis-aligned array to True");
                    }
                    flags &= ~NPYARRAYFLAGS.NPY_ALIGNED;
                }
            }
            if (uic != null)
            {
                bool bUic = NpyUtil_ArgProcessing.BoolConverter(uic);
                if (bUic)
                {
                    throw new ArgumentException("cannot set UPDATEIFCOPY flag to True");
                }
                else
                {
                    NpyCoreApi.ClearUPDATEIFCOPY(this);
                }
            }
            if (write != null)
            {
                bool bWrite = NpyUtil_ArgProcessing.BoolConverter(write);
                if (bWrite)
                {
                    if (!NpyCoreApi.IsWriteable(this))
                    {
                        throw new ArgumentException("cannot set WRITEABLE flag to true on this array");
                    }
                    flags |= NPYARRAYFLAGS.NPY_WRITEABLE;
                }
                else
                {
                    flags &= ~NPYARRAYFLAGS.NPY_WRITEABLE;
                }
            }
            RawFlags = flags;
        }

        public object take(object indices, object axis = null, ndarray @out = null, object mode = null)
        {
            ndarray aIndices;
            int iAxis;
            NPY_CLIPMODE cMode;

            aIndices = (indices as ndarray);
            if (aIndices == null)
            {
                aIndices = np.FromAny(indices, NpyCoreApi.DescrFromType(NpyDefs.NPY_INTP),
                    1, 0, NPYARRAYFLAGS.NPY_CONTIGUOUS, null);
            }
            iAxis = NpyUtil_ArgProcessing.AxisConverter(axis);
            cMode = NpyUtil_ArgProcessing.ClipmodeConverter(mode);
            return ArrayReturn(TakeFrom(aIndices, iAxis, @out, cMode));
        }




        public byte[] tobytes(NPY_ORDER order = NPY_ORDER.NPY_ANYORDER) {
            return ToString(order);
        }


        #endregion

        #endregion


        public long Size {
            get { return NpyCoreApi.ArraySize(this); }
        }

        public ndarray Real {
            get { return NpyCoreApi.GetReal(this); }
        }

        public ndarray Imag {
            get { return NpyCoreApi.GetImag(this); }
        }

        public override string ToString() {
            return StrFunction(this);
        }
  

        public flatiter Flat {
            get {
                return NpyCoreApi.IterNew(this);
            }
        }


        public ndarray NewCopy(NPY_ORDER order = NPY_ORDER.NPY_CORDER) {
            return NpyCoreApi.NewCopy(this, order);
        }


        /// <summary>
        /// Directly accesses the array memory and returns the object at that
        /// offset.  No checks are made, caller can easily crash the program
        /// or retrieve garbage data.
        /// </summary>
        /// <param name="offset">Offset into data array in bytes</param>
        /// <returns>Contents of the location</returns>
        public object GetItem(npy_intp offset)
        {
            return numpyAPI.GetItem(this.Array, offset);
        }


        /// <summary>
        /// Directly sets a given location in the data array.  No checks are
        /// made to make sure the offset is sensible or the data is valid in
        /// anyway -- caller beware.
        /// 'internal' because this is a security vulnerability.
        /// </summary>
        /// <param name="src">Value to write</param>
        /// <param name="offset">Offset into array in bytes</param>
        public void SetItem(object src, npy_intp offset)
        {
            numpyAPI.SetItem(this.Array, offset, src);
        }


        /// <summary>
        /// Handle to the core representation.
        /// </summary>
        public NpyArray Array {
            get { return core; }
        }


        /// <summary>
        /// Base address of the array data memory. Use with caution.
        /// </summary>
        internal VoidPtr DataAddress {
            get { return core.data; }
        }

        /// <summary>
        /// Returns an array of the sizes of each dimension. This property allocates
        /// a new array with each call and must make a managed-to-native call so it's
        /// worth caching the results if used in a loop.
        /// </summary>
 
        public npy_intp[] dims
        {
            get { return NpyCoreApi.GetArrayDimsOrStrides(this, true); }
        }

        /// <summary>
        /// Returns an array of the stride of each dimension.
        /// </summary>

        public npy_intp[] strides
        {
            get { return NpyCoreApi.GetArrayDimsOrStrides(this, false); }
        }

        /// <summary>
        /// Returns the stride of a given dimension. For looping over all dimensions,
        /// use 'strides'.  This is more efficient if only one dimension is of interest.
        /// </summary>
        /// <param name="dimension">Dimension to query</param>
        /// <returns>Data stride in bytes</returns>
        public long Dim(int dimension)
        {
            return this.Array.dimensions[dimension];
        }


        /// <summary>
        /// Returns the stride of a given dimension. For looping over all dimensions,
        /// use 'strides'.  This is more efficient if only one dimension is of interest.
        /// </summary>
        /// <param name="dimension">Dimension to query</param>
        /// <returns>Data stride in bytes</returns>
        public long Stride(int dimension)
        {
            return this.Array.strides[dimension];
        }


        /// <summary>
        /// True if memory layout of array is contiguous
        /// </summary>
        public bool IsContiguous {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_CONTIGUOUS); }
        }

        public bool IsOneSegment {
            get { return ndim == 0 || ChkFlags(NPYARRAYFLAGS.NPY_FORTRAN) || ChkFlags(NPYARRAYFLAGS.NPY_CARRAY); }
        }

        public bool IsASlice
        {
            get { return BaseArray != null; }
        }

        public bool IsAScalar
        {
            get { return Array.IsScalar; }
        }

        /// <summary>
        /// True if memory layout is Fortran order, false implies C order
        /// </summary>
        public bool IsFortran {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_FORTRAN) && ndim > 1; }
        }

        public bool IsNotSwapped {
            get { return Dtype.IsNativeByteOrder; }
        }

        public bool IsByteSwapped {
            get { return !IsNotSwapped; }
        }

        public bool IsCArray {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_CARRAY) && IsNotSwapped; }
        }

        public bool IsCArray_RO {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_CARRAY_RO) && IsNotSwapped; }
        }

        public bool IsFArray {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_FARRAY) && IsNotSwapped; }
        }

        public bool IsFArray_RO {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_FARRAY_RO) && IsNotSwapped; }
        }

        public bool IsBehaved {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_BEHAVED) && IsNotSwapped; }
        }

        public bool IsBehaved_RO {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_ALIGNED) && IsNotSwapped; }
        }

        internal bool IsComplex {
            get { return NpyDefs.IsComplex(TypeNum); }
        }
               
        internal bool IsBigInt
        {
            get { return NpyDefs.IsBigInt(TypeNum); }
        }

        internal bool IsDecimal
        {
            get { return NpyDefs.IsDecimal(TypeNum); }
        }

        internal bool IsInteger {
            get { return NpyDefs.IsInteger(TypeNum); }
        }

        internal bool IsFloatingPoint
        {
            get { return NpyDefs.IsFloat(TypeNum); }
        }

        internal bool IsInexact
        {
            get { return IsFloatingPoint || IsComplex; }
        }

        public bool IsFlexible {
            get { return NpyDefs.IsFlexible(TypeNum); }
        }

        public bool IsMathFunctionCapable
        {
            get
            {
                switch (TypeNum)
                {
                    case NPY_TYPES.NPY_OBJECT:
                    case NPY_TYPES.NPY_STRING:
                    case NPY_TYPES.NPY_DATETIME:
                    case NPY_TYPES.NPY_TIMEDELTA:
                    case NPY_TYPES.NPY_VOID:
                        return false;
                    default:
                        return true;
                }
            }
        }


        internal bool IsMatrix
        {
            get { return false; }
        }

        public bool IsWriteable {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_WRITEABLE); }
        }

        public bool IsString {
            get { return TypeNum == NPY_TYPES.NPY_STRING; }
        }


        /// <summary>
        /// TODO: What does this return?
        /// </summary>
        public int ElementStrides {
            get { return NpyCoreApi.ElementStrides(this); }
        }

        public bool StridingOk(NPY_ORDER order) {
            return order == NPY_ORDER.NPY_ANYORDER ||
                order == NPY_ORDER.NPY_CORDER && IsContiguous ||
                order == NPY_ORDER.NPY_FORTRANORDER && IsFortran;
        }

        private bool ChkFlags(NPYARRAYFLAGS flag) {
            return ((RawFlags & flag) == flag);
        }

        // These operators are useful from other C# code and also turn into the
        // appropriate Python functions (+ goes to __add__, etc).

        #region IEnumerable<object> interface

        public IEnumerator<object> GetEnumerator() {
            return new ndarray_Enumerator(this);
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() {
            return new ndarray_Enumerator(this);
        }

        #endregion

        #region Internal methods

        internal long Length {
            get {
                return Dim(0);
            }
        }

        public static object ArrayReturn(ndarray a) {
            if (a.ndim == 0) {
                return a.Dtype.ToScalar(a);
            } else {
                return a;
            }
        }
        private string BuildStringRepr(bool repr)
        {
            // Equivalent to array_repr_builtin (arrayobject.c)
            List<string> sb = new List<string>();
            if (repr)
                sb.Append("array(");

            DumpData.DumpArray(sb, this.Array, repr);

            if (repr)
            {
                if (NpyDefs.IsExtended(this.TypeNum))
                {
                    sb.Add(String.Format(", '{0}{1}')", (char)Dtype.Type, this.ItemSize));
                }
                else
                {
                    sb.Add(String.Format(", '{0}')", (char)Dtype.Type));
                }
            }

            StringBuilder sb1 = new StringBuilder();
            foreach (var s in sb)
            {
                sb1.Append(s);
            }
            return sb1.ToString();
        }


        /// <summary>
        /// Indexes an array by a single long and returns either an item or a sub-array.
        /// </summary>
        /// <param name="index">The index into the array</param>
        object GetArrayItem(npy_intp index)
        {
            if (ndim == 1)
            {
                if (Math.Abs(index) >= this.shape.iDims[0])
                {
                    throw new Exception("index exceeds size of array");
                }

                if (index < 0)
                {
                    index += this.shape.iDims[0];
                }
                return numpyAPI.GetItem(this.Array, index);
            }
            else
            {
                return NpyCoreApi.ArrayItem(this, index);
            }
        }

        /// <summary>
        /// Indexes an array by a single long and returns either an item or a sub-array.
        /// </summary>
        /// <param name="index">The index into the array</param>
        void SetArrayItem(npy_intp index, object value)
        {
            if (ndim == 1)
            {
                numpyAPI.SetItem(this.Array, index, value);
            }
            else
            {
                var item = NpyCoreApi.ArrayItem(this, index);

                if (item is ndarray)
                {
                    try
                    {
                        ndarray itemarr = item as ndarray;

                        np.copyto(itemarr, value);

                    }
                    catch (Exception ex)
                    {

                    }

            
                }

                // todo: set the array items with the value??
            }
        }

        internal NPYARRAYFLAGS RawFlags
        {
            get
            {
                return Array.flags;
            }
            set
            {
                Array.flags = value;
            }
        }

        internal static dtype GetTypeDouble(dtype dtype1, dtype dtype2)
        {
            if (dtype2 != null)
            {
                return dtype2;
            }
            if (dtype1.TypeNum < NPY_TYPES.NPY_FLOAT)
            {
                return NpyCoreApi.DescrFromType(NPY_TYPES.NPY_DOUBLE);
            }
            else
            {
                return dtype1;
            }
        }

        private static bool IsNdarraySubtype(object type)
        {
            return false;
        }


        internal ndarray BaseArray {
            get {
                if (core.base_arr == null)
                    return null;
                return new ndarray(core.base_arr);
            }
            set {
                lock (this) {
                    core.base_arr = value.core;
                    NpyCoreApi.Decref(value.core);
                 }
            }
        }

        /// <summary>
        /// Copies data into the array from 'data'.  Offset is the offset into this
        /// array's data space in bytes.  The number of bytes copied is based on the
        /// element size of the array's dtype.
        /// </summary>
        /// <param name="offset">Offset into this array's data (bytes)</param>
        /// <param name="data">Memory address to copy the data from</param>
        /// <param name="swap">If true data is byte-swapped during copy</param>
        internal void CopySwapIn(long offset, VoidPtr data, bool swap) {
            NpyCoreApi.CopySwapIn(this, offset, data, swap);
        }

        /// <summary>
        /// Copies data out of the array into 'data'. Offset is the offset into this
        /// array's data space in bytes. Number of bytes copied is based on the
        /// element size of the array's dtype.
        /// </summary>
        /// <param name="offset">Offset into array's data in bytes</param>
        /// <param name="data">Memory address to copy the data to</param>
        /// <param name="swap">If true, results are byte-swapped from the array's image</param>
        internal void CopySwapOut(long offset, VoidPtr data, bool swap) {
            NpyCoreApi.CopySwapOut(this, offset, data, swap);
        }

        #endregion

      
    }

    internal class ndarray_Enumerator : IEnumerator<object>
    {
        public ndarray_Enumerator(ndarray a) {
            arr = a;
            index = -1;
        }

        public object Current
        {
            get
            {
                //if (arr.BaseArray != null)
                //{
                //    return arr[index * arr.Strides[0]];
                //}
                //else
                {
                    return arr[index];
                }
            }
        }

        public void Dispose() {
            arr = null;
        }


        public bool MoveNext() {
            index += 1;
            return (index < arr.dims[0]);
        }

        public void Reset() {
            index = -1;
        }

        private ndarray arr;
        private long index;
    }

    public class CSharpTuple
    {
        public long? index1 = null;
        public long? index2 = null;
        public long? index3 = null;
        public CSharpTuple(long index1)
        {
            this.index1 = index1;
        }
        public CSharpTuple(long index1, long index2)
        {
            this.index1 = index1;
            this.index2 = index2;
        }
        public CSharpTuple(long index1, long index2, long index3)
        {
            this.index1 = index1;
            this.index2 = index2;
            this.index3 = index3;
        }
    }
}
