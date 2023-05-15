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
    public class ndarray_serializable
    {
        public string Name
        {
            get; set;
        }

        public bool[] bool_array;
        public byte[] byte_array;
        public sbyte[] sbyte_array;
        public Int16[] int16_array;
        public UInt16[] uint16_array;
        public Int32[] int32_array;
        public UInt32[] uint32_array;
        public Int64[] int64_array;
        public UInt64[] uint64_array;
        public float[] float_array;
        public double[] double_array;
        public decimal[] decimal_array;
        public System.Numerics.Complex[] complex_array;
        public System.Numerics.BigInteger[] bigint_array;
        public System.String[] string_array;
        public System.Object[] object_array;

        public npy_intp data_offset;

        public int nd;                /* number of dimensions, also called ndim */

        public npy_intp[] dimensions; /* size in each dimension */
        public npy_intp[] strides;    /* bytes to jump to get to next element in each dimension */

        public ndarray_serializable base_array;

        public NpyArray_Descr_serializable descr;  /* Pointer to type structure */
        public NPYARRAYFLAGS flags;   /* Flags describing array -- see below */

        public bool IsScalar = false;
    }


    /// <summary>
    /// Implements the Numpy python 'ndarray' class
    /// </summary>
    public partial class ndarray : IEnumerable<object>, NumpyDotNet.IArray
    {
        #region Explicit conversion to single element data types
        /// special case for bools. Tries to convert all data types to true/false value
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
        public static explicit operator byte(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_UBYTE)
                throw new Exception("ndarray does not contain bytes");

            CheckElementCount(nd);

            return (byte)nd.GetItem(0);
        }
        public static explicit operator sbyte(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_BYTE)
                throw new Exception("ndarray does not contain sbytes");

            CheckElementCount(nd);

            return (sbyte)nd.GetItem(0);
        }
        public static explicit operator UInt16(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_UINT16)
                throw new Exception("ndarray does not contain UInt16s");

            CheckElementCount(nd);

            return (UInt16)nd.GetItem(0);
        }
        public static explicit operator Int16(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_INT16)
                throw new Exception("ndarray does not contain Int16s");

            CheckElementCount(nd);

            return (Int16)nd.GetItem(0);
        }
        public static explicit operator UInt32(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_UINT32)
                throw new Exception("ndarray does not contain UInt32s");

            CheckElementCount(nd);

            return (UInt32)nd.GetItem(0);
        }
        public static explicit operator Int32(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_INT32)
                throw new Exception("ndarray does not contain Int32s");

            CheckElementCount(nd);

            return (Int32)nd.GetItem(0);
        }
        public static explicit operator UInt64(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_UINT64)
                throw new Exception("ndarray does not contain UInt64s");

            CheckElementCount(nd);

            return (UInt64)nd.GetItem(0);
        }
        public static explicit operator Int64(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_INT64)
                throw new Exception("ndarray does not contain Int64s");

            CheckElementCount(nd);

            return (Int64)nd.GetItem(0);
        }
        public static explicit operator float(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_FLOAT)
                throw new Exception("ndarray does not contain floats");

            CheckElementCount(nd);

            return (float)nd.GetItem(0);
        }
        public static explicit operator double(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_DOUBLE)
                throw new Exception("ndarray does not contain doubles");

            CheckElementCount(nd);

            return (double)nd.GetItem(0);
        }
        public static explicit operator decimal(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_DECIMAL)
                throw new Exception("ndarray does not contain decimals");

            CheckElementCount(nd);

            return (decimal)nd.GetItem(0);
        }
        public static explicit operator System.Numerics.Complex(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_COMPLEX)
                throw new Exception("ndarray does not contain complex numbers");

            CheckElementCount(nd);

            return (System.Numerics.Complex)nd.GetItem(0);
        }
        public static explicit operator System.Numerics.BigInteger(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_BIGINT)
                throw new Exception("ndarray does not contain BigIntegers");

            CheckElementCount(nd);

            return (System.Numerics.BigInteger)nd.GetItem(0);
        }

        //public static explicit operator System.Object(ndarray nd)
        //{
        //    if (nd.TypeNum != NPY_TYPES.NPY_OBJECT)
        //        throw new Exception("ndarray does not contain Objects");

        //    CheckElementCount(nd);

        //    return (System.Object)nd.GetItem(0);
        //}

        public static explicit operator System.String(ndarray nd)
        {
            if (nd.TypeNum != NPY_TYPES.NPY_STRING)
                throw new Exception("ndarray does not contain Strings");

            CheckElementCount(nd);

            return (System.String)nd.GetItem(0);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CheckElementCount(ndarray nd)
        {
            //if (nd.Array.data.datap.GetElementCount() != 1)
            //    throw new Exception("ndarray is not a single element array");
        }

        #endregion

        internal NpyArray core;
        public ndarray() {
        }

        internal ndarray(NpyArray a)
        {
            if (a == null)
            {
                throw new Exception("Attempt to create ndarray with null object");
            }
            core = a;
        }

        public ndarray(ndarray_serializable serializable)
        {
            List<ndarray_serializable> NestedArrays = new List<ndarray_serializable>();

            NestedArrays.Add(serializable);

            ndarray_serializable temp = serializable;
            while (true)
            {
                if (temp.base_array != null)
                {
                    NestedArrays.Add(temp.base_array);
                    temp = temp.base_array;
                }
                else
                {
                    break;
                }
            }

            ndarray_serializable FirstArray = NestedArrays[NestedArrays.Count - 1];
            ndarray First;


            switch (FirstArray.descr.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    First = np.array(serializable.bool_array);
                    break;

                case NPY_TYPES.NPY_BYTE:
                    First = np.array(serializable.sbyte_array);
                    break;

                case NPY_TYPES.NPY_UBYTE:
                    First = np.array(serializable.byte_array);
                    break;

                case NPY_TYPES.NPY_INT16:
                    First = np.array(serializable.int16_array);
                    break;

                case NPY_TYPES.NPY_UINT16:
                    First = np.array(serializable.uint16_array);
                    break;

                case NPY_TYPES.NPY_INT32:
                    First = np.array(serializable.int32_array);
                    break;

                case NPY_TYPES.NPY_UINT32:
                    First = np.array(serializable.uint32_array);
                    break;

                case NPY_TYPES.NPY_INT64:
                    First = np.array(serializable.int64_array);
                    break;

                case NPY_TYPES.NPY_UINT64:
                    First = np.array(serializable.uint64_array);
                    break;

                case NPY_TYPES.NPY_FLOAT:
                    First = np.array(serializable.float_array);
                    break;

                case NPY_TYPES.NPY_DOUBLE:
                    First = np.array(serializable.double_array);
                    break;

                case NPY_TYPES.NPY_DECIMAL:
                    First = np.array(serializable.decimal_array);
                    break;

                case NPY_TYPES.NPY_COMPLEX:
                    First = np.array(serializable.complex_array);
                    break;

                case NPY_TYPES.NPY_BIGINT:
                    First = np.array(serializable.bigint_array);
                    break;

                case NPY_TYPES.NPY_OBJECT:
                    First = np.array(serializable.object_array);
                    break;

                case NPY_TYPES.NPY_STRING:
                    First = np.array(serializable.string_array);
                    break;

                default:
                    throw new Exception("Attempt to deserialize unrecognized data type");
            }

            if (FirstArray.nd > 1)
            {
                First = First.reshape(FirstArray.dimensions);
            }

            First.Name = FirstArray.Name;
            First.core.descr = NpyArray_Descr.FromSerializable(FirstArray.descr);

            if (NestedArrays.Count > 1)
            {
                for (int i = NestedArrays.Count - 2; i >=0; i--)
                {
                    First = First.reshape(NestedArrays[i].dimensions);
                    First.Name = NestedArrays[i].Name;
                    First.core.descr = NpyArray_Descr.FromSerializable(NestedArrays[i].descr);
                }
            }

            core = First.core;
            return;
        }

        public ndarray_serializable ToSerializable(bool SerializeDataArray = true)
        {
            ndarray_serializable serializable = new ndarray_serializable();
            serializable.Name = this.Name;

            serializable.data_offset = this.DataAddress.data_offset;

            if (SerializeDataArray)
            {
                switch (this.TypeNum)
                {
                    case NPY_TYPES.NPY_BOOL:
                        serializable.bool_array = this.DataAddress.datap as bool[];
                        break;

                    case NPY_TYPES.NPY_BYTE:
                        serializable.sbyte_array = this.DataAddress.datap as sbyte[];
                        break;

                    case NPY_TYPES.NPY_UBYTE:
                        serializable.byte_array = this.DataAddress.datap as byte[];
                        break;

                    case NPY_TYPES.NPY_INT16:
                        serializable.int16_array = this.DataAddress.datap as Int16[];
                        break;

                    case NPY_TYPES.NPY_UINT16:
                        serializable.uint16_array = this.DataAddress.datap as UInt16[];
                        break;

                    case NPY_TYPES.NPY_INT32:
                        serializable.int32_array = this.DataAddress.datap as Int32[];
                        break;

                    case NPY_TYPES.NPY_UINT32:
                        serializable.uint32_array = this.DataAddress.datap as UInt32[];
                        break;

                    case NPY_TYPES.NPY_INT64:
                        serializable.int64_array = this.DataAddress.datap as Int64[];
                        break;

                    case NPY_TYPES.NPY_UINT64:
                        serializable.uint64_array = this.DataAddress.datap as UInt64[];
                        break;

                    case NPY_TYPES.NPY_FLOAT:
                        serializable.float_array = this.DataAddress.datap as float[];
                        break;

                    case NPY_TYPES.NPY_DOUBLE:
                        serializable.double_array = this.DataAddress.datap as double[];
                        break;

                    case NPY_TYPES.NPY_DECIMAL:
                        serializable.decimal_array = this.DataAddress.datap as decimal[];
                        break;

                    case NPY_TYPES.NPY_COMPLEX:
                        serializable.complex_array = this.DataAddress.datap as System.Numerics.Complex[];
                        break;

                    case NPY_TYPES.NPY_BIGINT:
                        serializable.bigint_array = this.DataAddress.datap as System.Numerics.BigInteger[];
                        break;

                    case NPY_TYPES.NPY_OBJECT:
                        serializable.object_array = this.DataAddress.datap as System.Object[];
                        break;

                    case NPY_TYPES.NPY_STRING:
                        serializable.string_array = this.DataAddress.datap as System.String[];
                        break;

                    default:
                        throw new Exception("Attempt to serialize unrecognized data type");
                }
            }
   
            serializable.nd = this.ndim;
            serializable.dimensions = this.dims;
            serializable.strides = this.strides;

            if (this.BaseArray != null)
            {
                serializable.base_array = this.BaseArray.ToSerializable(SerializeDataArray=false);
            }

            serializable.descr = this.core.descr.ToSerializable();
            serializable.flags = this.core.flags;
            serializable.IsScalar = this.core.IsScalar;

            return serializable;
        }

        #region Public interfaces (must match CPython)

        private static Func<ndarray, string> reprFunction;
        private static Func<ndarray, string> strFunction;

        /// <summary>
        /// User assigned name for allocated ndarray
        /// </summary>
        public string Name
        {
            get
            {
                return core.Name;
            }
            set
            {
                core.Name = value;
            }
        }

        /// <summary>
        /// Sets a function to be triggered for the repr() operator or null to default to the
        /// built-in version.
        /// </summary>
        internal static Func<ndarray, string> ReprFunction {
            get { return reprFunction; }
            private set { reprFunction = (value != null) ? value : x => x.BuildStringRepr(true); }
        }

        /// <summary>
        /// Sets a function to be triggered on the str() operator or ToString() method. Null defaults to
        /// the built-in version.
        /// </summary>
        internal static Func<ndarray, string> StrFunction {
            get { return strFunction; }
            private set { strFunction = (value != null) ? value : x => x.BuildStringRepr(false); }
        }

        static ndarray() {
            ReprFunction = null;
            StrFunction = null;
        }


        #region Operators

        internal static ndarray BinaryOp(ndarray a, object b, NpyUFuncObject UFunc)
        {
            return NpyCoreApi.PerformNumericOp(a, UFunc.ops, np.asanyarray(b), false);
        }

        internal static ndarray BinaryOpInPlace(ndarray a, object b, NpyUFuncObject UFunc, ndarray ret)
        {
            ndarray numericOpResult = NpyCoreApi.PerformNumericOp(a, UFunc.ops, np.asanyarray(b), true);
            if (numericOpResult != null && ret != null)
            {
                NpyCoreApi.CopyAnyInto(ret, numericOpResult);
            }

            return numericOpResult;
        }

        internal static ndarray BinaryOp(ndarray a, object b, UFuncOperation op)
        {
            var f = NpyCoreApi.GetNumericOp(op);
            return BinaryOp(a, b, f);
        }

        public static ndarray BinaryOpInPlace(ndarray a, object b, UFuncOperation op, ndarray ret)
        {
            var f = NpyCoreApi.GetNumericOp(op);
            return BinaryOpInPlace(a, b, f, ret);
        }

        internal static ndarray UnaryOp(ndarray a, UFuncOperation op)
        {
            return NpyCoreApi.PerformNumericOp(a, op, 0, false);
        }


        internal static ndarray UnaryOpInPlace(ndarray a, UFuncOperation op, ndarray ret)
        {
            ndarray numericOpResult = NpyCoreApi.PerformNumericOp(a, op, 0, true);

            if (numericOpResult != null && ret != null)
            {
                NpyCoreApi.CopyAnyInto(ret, numericOpResult);
            }

            return numericOpResult;
        }



        public static ndarray operator +(ndarray a) {
            return a;
        }

        public static ndarray operator +(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.add, operand);
        }

        public static ndarray operator +(object a, ndarray operand)
        {
            return NpyCoreApi.PerformNumericOp(np.asanyarray(a), UFuncOperation.add, operand);
        }

        public static ndarray operator +(ndarray a, ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.add, b);
        }

        [SpecialName]
        public ndarray InPlaceAdd(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, UFuncOperation.add, b, true);
        }

        public static ndarray operator -(ndarray a, object operand) {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.subtract, operand);
        }
        public static ndarray operator -(object a, ndarray operand)
        {
            return NpyCoreApi.PerformNumericOp(np.asanyarray(a), UFuncOperation.subtract, operand);
        }

        public static ndarray operator -(ndarray a, ndarray b) {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.subtract, b);
        }


        [SpecialName]
        public ndarray InPlaceSubtract(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, UFuncOperation.subtract, b, true);
        }

        public static ndarray operator -(ndarray a) {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.negative, 0);
        }

        public static ndarray operator *(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.multiply, operand);
        }

        public static ndarray operator *(object a, ndarray operand)
        {
            return NpyCoreApi.PerformNumericOp(np.asanyarray(a), UFuncOperation.multiply, operand);
        }

        public static ndarray operator *(ndarray a, ndarray b) {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.multiply, b);
        }

        [SpecialName]
        public ndarray InPlaceMultiply(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, UFuncOperation.multiply, b, true);
        }


        public static ndarray operator /(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.divide, operand);
        }
        public static ndarray operator /(object a, ndarray operand)
        {
            return NpyCoreApi.PerformNumericOp(np.asanyarray(a), UFuncOperation.divide, operand);
        }


        public static ndarray operator /(ndarray a, ndarray b) {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.divide, b);
        }

        [SpecialName]
        public ndarray InPlaceDivide(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, UFuncOperation.divide, b, true);
        }

        [SpecialName]
        public ndarray InPlaceTrueDivide(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, UFuncOperation.true_divide, b, true);
        }

        [SpecialName]
        public ndarray InPlaceFloorDivide(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, UFuncOperation.floor_divide, b, true);
        }

        public static ndarray operator %(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.remainder, operand);
        }
        public static ndarray operator %(object a, ndarray operand)
        {
            return NpyCoreApi.PerformNumericOp(np.asanyarray(a), UFuncOperation.remainder, operand);
        }
        public static ndarray operator %(ndarray a, ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.remainder, b);
        }



        public static ndarray operator &(ndarray a, Int64 operand) {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.bitwise_and, operand);
        }
        public static ndarray operator &(Int64 operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.bitwise_and, operand);
        }

        public static ndarray operator &(ndarray a, ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.bitwise_and, b);
        }

        [SpecialName]
        public ndarray InPlaceBitwiseAnd(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, UFuncOperation.bitwise_and, b, true);
        }

        public static ndarray operator |(ndarray a, Int64 operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.bitwise_or, operand);
        }
        public static ndarray operator |(Int64 operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.bitwise_or, operand);
        }

        public static ndarray operator |(ndarray a, ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.bitwise_or, b);
        }

        [SpecialName]
        public ndarray InPlaceBitwiseOr(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, UFuncOperation.bitwise_or, b, true);
        }

        public static ndarray operator ^(ndarray a, Int64 operand) {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.bitwise_xor, operand);
        }
        public static ndarray operator ^(Int64 operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.bitwise_xor, operand);
        }

        public static ndarray operator ^(ndarray a, ndarray b) {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.bitwise_xor, b);
        }

        public static ndarray operator <<(ndarray a, int shift) {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.left_shift, shift);
        }

        public static ndarray operator >>(ndarray a, int shift) {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.right_shift, shift);
        }

        public static ndarray operator <(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.less, operand);
        }
        public static ndarray operator <(ndarray a, ndarray operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.less, operand);
        }
        public static ndarray operator <=(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.less_equal, operand);
        }
        public static ndarray operator <=(ndarray a, ndarray operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.less_equal, operand);
        }
        public static ndarray operator >(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.greater, operand);
        }
        public static ndarray operator >(ndarray a, ndarray operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.greater, operand);
        }
        public static ndarray operator < (object operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.greater, operand);
        }

        public static ndarray operator > (object operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.less, operand);
        }

        public static ndarray operator <= (object operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.greater_equal, operand);
        }

        public static ndarray operator >= (object operand, ndarray a)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.less_equal, operand);
        }

        public static ndarray operator >=(ndarray a, object operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.greater_equal, operand);
        }

        public static ndarray operator >=(ndarray a, ndarray operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.greater_equal, operand);
        }
        public static ndarray operator ==(ndarray a, double operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.equal, operand);
        }
        public static ndarray operator ==(ndarray a, float operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.equal, operand);
        }
        public static ndarray operator ==(ndarray a, Int64 operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.equal, operand);
        }
        public static ndarray operator ==(ndarray a, Int32 operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.equal, operand);
        }
        public static ndarray operator ==(ndarray a, bool operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.equal, operand);
        }
        public static ndarray operator ==(ndarray a, decimal operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.equal, operand);
        }
        public static ndarray operator ==(ndarray a, Complex operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.equal, operand);
        }
        public static ndarray operator ==(ndarray a, BigInteger operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.equal, operand);
        }
 
        [SpecialName]
        public ndarray Equals(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, UFuncOperation.equal, b);
        }
        [SpecialName]
        public ndarray Equals(string b)
        {
            return NpyCoreApi.PerformNumericOp(this, UFuncOperation.equal, b);
        }
        public static ndarray operator !=(ndarray a, double operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, float operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, Int64 operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, Int32 operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, bool operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, decimal operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, Complex operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.not_equal, operand);
        }
        public static ndarray operator !=(ndarray a, BigInteger operand)
        {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.not_equal, operand);
        }

        [SpecialName]
        public ndarray NotEquals(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, UFuncOperation.not_equal, b);
        }
        [SpecialName]
        public ndarray NotEquals(string b)
        {
            return NpyCoreApi.PerformNumericOp(this, UFuncOperation.not_equal, b);
        }
        [SpecialName]
        public ndarray InPlaceExclusiveOr(ndarray b)
        {
            return NpyCoreApi.PerformNumericOp(this, UFuncOperation.bitwise_xor, b, true);
        }

        public static ndarray operator ~(ndarray a) {
            return NpyCoreApi.PerformNumericOp(a, UFuncOperation.invert, 0);
        }

        //public static implicit operator String(ndarray a) {
        //    return StrFunction(a);
        //}

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
        /// <summary>
        /// slicing/indexing function to set a breakpoint in
        /// </summary>
        /// <param name="args"></param>
        /// <returns></returns>
        public Object SliceMe(params object[] args)
        {
            return this[args];
        }
        /// <summary>
        /// sliced/indexed array cast to ndarray.  Throws exception if result is not ndarray. Maybe better than casting to ndarray everwhere.
        /// </summary>
        /// <param name="args"></param>
        /// <returns></returns>
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
  
                NpyIndexes indexes = new NpyIndexes();
                {
                    NpyUtil_IndexProcessing.IndexConverter(this, args, indexes);
                    if (indexes.IsSingleItem(ndim))
                    {
                        npy_intp offset = indexes.SingleAssignOffset(this);
                        offset += this.DataAddress.data_offset;
                        return numpyAPI.GetItem(RootArray(this.Array), offset >> this.ItemSizeDiv);
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
                            return result.GetItem(0);
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
                    npy_intp single_offset = indexes.SingleAssignOffset(this);
                    if (single_offset >= 0 && np.IsNumericType(value))
                    {
                        // This is a single item assignment. Use SetItem.
                        single_offset += this.DataAddress.data_offset;
                        numpyAPI.SetItem(RootArray(this.Array), single_offset >> this.ItemSizeDiv, value);
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

        private NpyArray RootArray(NpyArray srcArray)
        {
            NpyArray t = srcArray;

            while (t.base_arr != null)
            {
                t = t.base_arr;
            }
            return t;
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
        /// the shape of the array
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
        /// <summary>
        /// returns a raw pointer of the ndarray data.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public VoidPtr rawdata(npy_intp index = 0)
        {
            var flattened = this.ravel();
            return numpyAPI.NpyArray_Index2Ptr(flattened.Array, index);
        }
        
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

        /// <summary>
        /// size in bytes of the data items stored in this array
        /// </summary>
        public int ItemSize {
            get {

                if (core == null)
                    return 0;

                return core.descr.elsize;
            }
        }
        /// <summary>
        /// value used to  convert data_offset to an index.  "data_offset >> ItemSizeDiv"
        /// </summary>
        public int ItemSizeDiv
        {
            get
            {

                if (core == null)
                    return 0;

                return core.descr.eldivshift;
            }
        }
        /// <summary>
        /// The data type of this ndarray
        /// </summary>
        public NPY_TYPES TypeNum
        {
            get
            {
                if (core == null)
                    return NPY_TYPES.NPY_OBJECT;

                return core.descr.type_num;
            }
        }
        /// <summary>
        ///  total number of bytes in the ndarray 
        /// </summary>
        public long nbytes {
            get {
                return ItemSize * Size;
            }
        }
        /// <summary>
        /// transpose this array
        /// </summary>
        public ndarray T {
            get {
                return this.Transpose();
            }
        }


        #endregion

        #region methods

        /// <summary>
        /// Copy of the array, cast to a specified type.
        /// </summary>
        /// <param name="dtype">data type to cast to</param>
        /// <param name="copy"></param>
        /// <returns></returns>
        public ndarray astype(dtype dtype = null, bool copy = true)
        {
            if (dtype == this.Dtype && this.BaseArray == null)
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
        /// <summary>
        /// Swap the bytes of the array elements
        /// </summary>
        /// <param name="inplace">If True, swap bytes in-place, default is False.</param>
        /// <returns></returns>
        public ndarray byteswap(bool inplace = false)
        {
            return NpyCoreApi.Byteswap(this, inplace);
        }

        /// <summary>
        /// Return an array copy of the given object.
        /// </summary>
        /// <param name="order">{‘C’, ‘F’, ‘A’, ‘K’}, optional</param>
        /// <returns></returns>
        public ndarray Copy(NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            return NpyCoreApi.NewCopy(this, order);
        }

        /// <summary>
        /// Dot product of two arrays.
        /// </summary>
        /// <param name="other">array to calculate dot product with</param>
        /// <returns></returns>
        public ndarray dot(object other)
        {
            return np.MatrixProduct(this, other);
        }
        /// <summary>
        /// Fill the array with a scalar value.
        /// </summary>
        /// <param name="scalar">value to file array with</param>
        public void fill(object scalar)
        {
            FillWithScalar(scalar);
        }
        /// <summary>
        /// Return a copy of the array collapsed into one dimension.
        /// </summary>
        /// <param name="order">{‘C’, ‘F’, ‘A’, ‘K’}, optional</param>
        /// <returns></returns>
        public ndarray flatten(NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            return this.Flatten(order);
        }

        /// <summary>
        /// This function will allow specifying index with an array.
        /// Instead of specifying and item like a[1,2,3] can use a[new int[] {1,2,3}];
        /// </summary>
        /// <param name="args">an array with index values to select specific items</param>
        /// <returns></returns>
        public object item_byindex(Int32[] args)
        {
            Int64[] args64 = new Int64[args.Length];
            for (int i = 0; i < args.Length; i++)
                args64[i] = args[i];
            return item_byindex(args64);
        }

        /// <summary>
        /// This function will allow specifying index with an array.
        /// Instead of specifying and item like a[1,2,3] can use a[new int[] {1,2,3}];
        /// </summary>
        /// <param name="args">an array with index values to select specific items</param>
        /// <returns></returns>
        public object item_byindex(Int64[] args)
        {
            if (args == null || args.Length == 0)
                throw new ArgumentException("invalid index specified.  Must be not null and greater than 0 length");

            NpyIndexes indexes = new NpyIndexes();
            {
                NpyUtil_IndexProcessing.PureIndexConverter(this, args, indexes);
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
                        npy_intp offset = indexes.SingleAssignOffset(this);
                        offset += this.DataAddress.data_offset;
                        return numpyAPI.GetItem(RootArray(this.Array), offset >> this.ItemSizeDiv);
                    }
                    else
                    {
                        throw new ArgumentException("Incorrect number of indices for the array");
                    }
                }
            }
        }


        /// <summary>
        /// This function will allow specifying items via collection of index types.
        /// </summary>
        /// <param name="args">a collection if index items</param>
        /// <returns></returns>
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
                            npy_intp offset = indexes.SingleAssignOffset(this);
                            offset += this.DataAddress.data_offset;
                            return numpyAPI.GetItem(RootArray(this.Array), offset >> this.ItemSizeDiv);
                        }
                        else
                        {
                            throw new ArgumentException("Incorrect number of indices for the array");
                        }
                    }
                }
            }
        }

        public void itemset_byindex(Int32 [] args, object value)
        {
            Int64[] args64 = new Int64[args.Length];
            for (int i = 0; i < args.Length; i++)
                args64[i] = args[i];
            itemset_byindex(args64, value);
        }

        public void itemset_byindex(Int64[] args, object value)
        {
            if (args == null || args.Length == 0)
                throw new ArgumentException("invalid index specified.  Must be not null and greater than 0 length");

            NpyIndexes indexes = new NpyIndexes();
            {
                NpyUtil_IndexProcessing.PureIndexConverter(this, args, indexes);
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
                        npy_intp offset = indexes.SingleAssignOffset(this);
                        offset += this.DataAddress.data_offset;
                        numpyAPI.SetItem(RootArray(this.Array), offset >> this.ItemSizeDiv, value);
                    }
                    else
                    {
                        throw new ArgumentException("Incorrect number of indices for the array");
                    }
                }
            }
        }

        /// <summary>
        /// This function will allow specifying items via collection of index types.
        /// </summary>
        /// <param name="args">a collection if index items</param>
        /// <returns></returns>
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
                            npy_intp offset = indexes.SingleAssignOffset(this);
                            offset += this.DataAddress.data_offset;
                            numpyAPI.SetItem(RootArray(this.Array), offset >> this.ItemSizeDiv, value);
                        }
                        else
                        {
                            throw new ArgumentException("Incorrect number of indices for the array");
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Return the array with the same data viewed with a different byte order.
        /// </summary>
        /// <param name="new_order"></param>
        /// <returns></returns>
        public ndarray newbyteorder(string new_order = null)
        {
            dtype newtype = NpyCoreApi.DescrNewByteorder(Dtype, NpyUtil_ArgProcessing.ByteorderConverter(new_order));
            return NpyCoreApi.View(this, newtype, null);
        }

        /// <summary>
        /// Replaces specified elements of an array with given values.
        /// </summary>
        /// <param name="indices">Target indices, interpreted as integers.</param>
        /// <param name="values">Values to place in a at target indices. </param>
        /// <param name="mode">{‘raise’, ‘wrap’, ‘clip’}, optional</param>
        /// <returns></returns>
        public int put(object indices, object values,  NPY_CLIPMODE mode = NPY_CLIPMODE.NPY_RAISE)
        {
            return np.put(this, indices, values, mode);
        }
        /// <summary>
        /// Return a flattened array.
        /// </summary>
        /// <param name="order">{‘C’,’F’, ‘A’, ‘K’}, optional</param>
        /// <returns></returns>
        public ndarray ravel(NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
            return this.Ravel(order);
        }
        /// <summary>
        /// Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="shape">The new shape should be compatible with the original shape.</param>
        /// <param name="order">{‘C’, ‘F’, ‘A’}, optional</param>
        /// <returns></returns>
        public ndarray reshape(IEnumerable<npy_intp> shape, NPY_ORDER order = NPY_ORDER.NPY_ANYORDER)
        {
            npy_intp[] newshape = shape.Select(x => (npy_intp)x).ToArray();

            return NpyCoreApi.Newshape(this, newshape, order);
        }
        /// <summary>
        /// Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="shape">The new shape should be compatible with the original shape.</param>
        /// <param name="order">{‘C’, ‘F’, ‘A’}, optional</param>
        /// <returns></returns>
        public ndarray reshape(int shape, NPY_ORDER order = NPY_ORDER.NPY_ANYORDER)
        {
            npy_intp[] newshape = new npy_intp[] { shape };

            return NpyCoreApi.Newshape(this, newshape, order);
        }
        /// <summary>
        /// Set array flags WRITEABLE, ALIGNED, (WRITEBACKIFCOPY and UPDATEIFCOPY), respectively.
        /// </summary>
        /// <param name="write"></param>
        /// <param name="align"></param>
        /// <param name="uic"></param>
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

        /// <summary>
        /// copies array data into byte[].  Can change the ordering.
        /// </summary>
        /// <param name="order">{‘C’, ‘F’, ‘A’}, optional</param>
        /// <returns></returns>
        public byte[] tobytes(NPY_ORDER order = NPY_ORDER.NPY_ANYORDER)
        {
            switch (order)
            {
                case NPY_ORDER.NPY_CORDER:
                case NPY_ORDER.NPY_FORTRANORDER:
                case NPY_ORDER.NPY_ANYORDER:
                    break;
                default:
                    throw new Exception("order parameter must be 'C', 'F' or 'A'");
            }

            return ToString(order);
        }


        #endregion

        #endregion

        /// <summary>
        /// Number of elements in the array.
        /// </summary>
        public npy_intp Size {
            get { return NpyCoreApi.ArraySize(this); }
        }
        /// <summary>
        /// Return the real part of the complex argument.
        /// </summary>
        public ndarray Real {
            get { return NpyCoreApi.GetReal(this); }
        }
        /// <summary>
        /// Return the imaginary part of the complex argument.
        /// </summary>
        public ndarray Imag {
            get { return NpyCoreApi.GetImag(this); }
        }
        /// <summary>
        /// returns printable string representation of ndarray
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return StrFunction(this);
        }

        /// <summary>
        /// A 1-D iterator over the array.
        /// </summary>
        public flatiter Flat
        {
            get
            {
                return NpyCoreApi.IterNew(this);
            }
        }



        internal ndarray NewCopy(NPY_ORDER order = NPY_ORDER.NPY_CORDER)
        {
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
        internal NpyArray Array {
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
        /// the strides of the array.
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
        public npy_intp Dim(int dimension)
        {
            return this.Array.dimensions[dimension];
        }


        /// <summary>
        /// Returns the stride of a given dimension. For looping over all dimensions,
        /// use 'strides'.  This is more efficient if only one dimension is of interest.
        /// </summary>
        /// <param name="dimension">Dimension to query</param>
        /// <returns>Data stride in bytes</returns>
        public npy_intp Stride(int dimension)
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
        /// <summary>
        /// true of array is a slice/view into another array.
        /// </summary>
        public bool IsASlice
        {
            get { return BaseArray != null; }
        }
        /// <summary>
        /// true if array is a single element array
        /// </summary>
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
        /// <summary>
        /// true of byte order is not swapped
        /// </summary>
        public bool IsNotSwapped {
            get { return Dtype.IsNativeByteOrder; }
        }
        /// <summary>
        /// true if byte order is swapped
        /// </summary>
        public bool IsByteSwapped {
            get { return !IsNotSwapped; }
        }
        /// <summary>
        /// true of array is ordered in "C" format
        /// </summary>
        public bool IsCArray
        {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_CARRAY) && IsNotSwapped; }
        }
        /// <summary>
        /// true if array is ordered in "C" formation and Read Only
        /// </summary>
        public bool IsCArray_RO {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_CARRAY_RO) && IsNotSwapped; }
        }
        /// <summary>
        /// returns true of array is ordered in "F"ortran order
        /// </summary>
        public bool IsFArray
        {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_FARRAY) && IsNotSwapped; }
        }
        /// <summary>
        /// returns true of array is ordered in "F"ortran order and Read Only
        /// </summary>
        public bool IsFArray_RO
        {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_FARRAY_RO) && IsNotSwapped; }
        }
        /// <summary>
        /// returns true of data type is aligned, writable and machine byte-order
        /// </summary>
        public bool IsBehaved
        {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_BEHAVED) && IsNotSwapped; }
        }
        /// <summary>
        /// returns true of data type is aligned and machine byte-order
        /// </summary>
        public bool IsBehaved_RO {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_ALIGNED) && IsNotSwapped; }
        }

        /// <summary>
        /// return true if data type is a complex number
        /// </summary>
        public bool IsBool
        {
            get { return NpyDefs.IsBool(TypeNum); }
        }

        /// <summary>
        /// return true if data type is a complex number
        /// </summary>
        public bool IsComplex
        {
            get { return NpyDefs.IsComplex(TypeNum); }
        }
        /// <summary>
        /// returns true if data type is a "BigInteger"
        /// </summary>
        public bool IsBigInt
        {
            get { return NpyDefs.IsBigInt(TypeNum); }
        }
        /// <summary>
        /// returns true if data type is Decimal
        /// </summary>
        public bool IsDecimal
        {
            get { return NpyDefs.IsDecimal(TypeNum); }
        }
        /// <summary>
        /// returns true if data type is an integer value
        /// </summary>
        public bool IsInteger
        {
            get { return NpyDefs.IsInteger(TypeNum); }
        }
        /// <summary>
        /// returns true if data type is a signed integer value
        /// </summary>
        public bool IsSignedInteger
        {
            get { return NpyDefs.IsSigned(TypeNum); }
        }

        /// <summary>
        /// returns true if data type is a signed integer value
        /// </summary>
        public bool IsUnsignedInteger
        {
            get { return NpyDefs.IsUnsigned(TypeNum); }
        }

        /// <summary>
        /// returns true if data type is a floating point value
        /// </summary>
        public bool IsFloatingPoint
        {
            get { return NpyDefs.IsFloat(TypeNum); }
        }

        /// <summary>
        /// returns true if data type is numeric value
        /// </summary>
        public bool IsNumber
        {
            get { return NpyDefs.IsNumber(TypeNum); }
        }

        /// <summary>
        /// returns true if data type is inexact (i.e. floating point or complex)
        /// </summary>
        public bool IsInexact
        {
            get { return IsFloatingPoint || IsComplex; }
        }
        /// <summary>
        /// returns true of data type is string
        /// </summary>
        public bool IsFlexible
        {
            get { return NpyDefs.IsFlexible(TypeNum); }
        }
        /// <summary>
        /// returns true of internal math functions can be operated on the data type
        /// </summary>
        public bool IsMathFunctionCapable
        {
            get
            {
                switch (TypeNum)
                {
                    case NPY_TYPES.NPY_OBJECT:
                    case NPY_TYPES.NPY_STRING:
                        return false;
                    default:
                        return true;
                }
            }
        }

        /// <summary>
        /// always false since matrix types obsolete and not supported
        /// </summary>
        internal bool IsMatrix
        {
            get { return false; }
        }
        /// <summary>
        /// true if array is not Read Only
        /// </summary>
        public bool IsWriteable
        {
            get { return ChkFlags(NPYARRAYFLAGS.NPY_WRITEABLE); }
        }
        /// <summary>
        /// return true if data type is a string
        /// </summary>
        public bool IsString
        {
            get { return TypeNum == NPY_TYPES.NPY_STRING; }
        }


        /// <summary>
        /// TODO: What does this return?
        /// </summary>
        internal int ElementStrides {
            get { return NpyCoreApi.ElementStrides(this); }
        }

        /// <summary>
        /// check for ndarray flag == true
        /// </summary>
        /// <param name="flag"></param>
        /// <returns></returns>
        private bool ChkFlags(NPYARRAYFLAGS flag)
        {
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
            sb1.Append(string.Format("shape={0}, ", this.shape.ToString()));
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
            else if (ndim == 0)
            {
                return numpyAPI.GetItem(this.Array, 0);
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


        internal ndarray BaseArray
        {
            get
            {
                if (core.base_arr == null)
                    return null;
                return new ndarray(core.base_arr);
            }
            //set
            //{
            //    lock (this)
            //    {
            //        core.SetBase(value.core);
            //        NpyCoreApi.Decref(value.core);
            //    }
            //}
        }


        #endregion


    }

    internal class ndarray_Enumerator : IEnumerator<object>
    {
        public ndarray_Enumerator(ndarray a) {
            arr = a;
            index = -1;

            if (arr.ndim <= 1)
            {
                UseLocalCache = true;
            }
        }

        public object Current
        {
            get
            {
                if (UseLocalCache)
                {
                    if (LocalCacheIndex >= LocalCacheLength)
                    {
                        ReLoadLocalCache();
                    }
                    return LocalCache[LocalCacheIndex++];
                }

                return arr[index];
            }
        }

        public void Dispose() {
            arr = null;
            LocalCache = null;
        }


        public bool MoveNext() {
            index += 1;

            return (index < arr.dims[0]);
        }

        public void Reset() {
            index = -1;
            LocalCache = null;
        }

        private void ReLoadLocalCache()
        {
            LocalCacheLength = Math.Min(arr.size - index, MaxCacheSize);
            if (LocalCache == null)
            {
                LocalCache = new object[LocalCacheLength];
            }

            NpyCoreApi.GetItems(arr, LocalCache, index, LocalCacheLength);

            LocalCacheIndex = 0;
        }

        private ndarray arr;
        private npy_intp index;

        bool UseLocalCache = false;
        npy_intp MaxCacheSize = 10000;
        npy_intp LocalCacheLength = 0;
        object[] LocalCache = null;
        int LocalCacheIndex = 0;
    }

    internal class CSharpTuple
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
