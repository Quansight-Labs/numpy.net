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
using System.Numerics;
using NumpyLib;

#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{
    public class ScalarGeneric : IArray, IConvertible
    {
        internal virtual ndarray ToArray() {
            return null;
        }

        public virtual object Value {
            get {
                throw new NotImplementedException(
                    String.Format("Internal error: Value has not been overridden for scalar type '{0}'", GetType().Name));
            }
        }

        /// <summary>
        /// Indicates whether the scalars have been "initialized" or not.  This is an
        /// unpleasant hack that mimicks that CPython behavior whereby the tp_new field for
        /// the scalar types is modified in the middle of initialization.
        /// </summary>
        static internal bool Initialized { get; set; }


        /// <summary>
        /// Fill the value with the value from the 0-d array
        /// </summary>
        /// <param name="arr"></param>
        internal virtual ScalarGeneric FillData(ndarray arr, long offset, bool nativeByteOrder) {
            throw new NotImplementedException();
        }

        internal virtual ScalarGeneric FillData(VoidPtr dataPtr, int size, bool nativeByteOrder) {
            throw new NotImplementedException();
        }

   
        #region IArray interface

 
        public object @base {
            get { return null; }
        }

        public ndarray byteswap(bool inplace = false) {
            if (inplace) {
                throw new ArgumentException("cannot byteswap a scalar inplace");
            } else {
                // TODO: Fix to return a scalar
                return ToArray().byteswap(false);
            }
        }

        public ndarray conj(ndarray @out = null) {
            return ToArray().conj(@out);
        }

        public ndarray conjugate(ndarray @out = null) {
            return ToArray().conjugate(@out);
        }

        public object copy(NPY_ORDER order = NPY_ORDER.NPY_KORDER) {
            return ToArray().copy(order);
        }

        public virtual object dtype {
            get {
                return NpyCoreApi.DescrFromType(NPY_TYPES.NPY_VOID);
            }
            set {
                throw new ArgumentTypeException("array-scalars are immutable");
            }
        }

        public void fill(object scalar) {
            // TODO: This doesn't make any sense but is the same for CPython
            ToArray().fill(scalar);
        }

        public flagsobj flags {
            get { return new flagsobj(null); }
        }

        public object flat {
            get {
                return ToArray().flat;
            }
            set {
                throw new ArgumentTypeException("array-scalars are immutable");
            }
        }

        public ndarray flatten(NPY_ORDER order = NPY_ORDER.NPY_CORDER) {
            return ToArray().flatten(order);
        }


        public virtual object imag {
            get {
                return ndarray.ArrayReturn((ndarray)ToArray().imag);
            }
            set {
                throw new ArgumentTypeException("array-scalars are immutable");
            }
        }

        public object item(params object[] args) {
            return ToArray().item(args:args);
        }

        public void itemset(params object[] args) {
            throw new ArgumentTypeException("array-scalars are immutable");
        }

        public int itemsize {
            get { return ((dtype)dtype).itemsize; }
        }


        /// <summary>
        /// Size of the object in bytes
        /// </summary>
        public object nbytes {
            get { return this.itemsize; }
        }

        public int ndim {
            get {
                return 0;
            }
        }

        public ndarray newbyteorder(string endian = null) {
            return ToArray().newbyteorder(endian);
        }

        public ndarray ravel(NPY_ORDER order = NPY_ORDER.NPY_CORDER) {
            return ToArray().ravel(order);
        }

        public virtual object real {
            get {
                return ndarray.ArrayReturn((ndarray)ToArray().real);
            }
            set {
                throw new ArgumentTypeException("array-scalars are immutable");
            }
        }
 

        public void setflags(object write = null, object align = null, object uic = null) {
            // CPython implementation simply does nothing, so we will too.
        }

        public shape shape
        {
            get { return new shape(0); }
        }

        public npy_intp size {
            get { return 1; }
        }

        public npy_intp[] strides {
            get { return new npy_intp[0]; }
        }

   
        public ndarray swapaxes(int a1, int a2) {
            return ToArray().SwapAxes(a1, a2);
        }

        public ndarray swapaxes(object a1, object a2) {
            return ToArray().SwapAxes(Convert.ToInt32(a1), Convert.ToInt32(a2));
        }

        public static object Power(Object a, Object b)
        {
            if (a is double && b is double)
            {
                return Math.Pow((double)a, (double)b);
            }
            else if (a is double && b is ScalarFloat64)
            {
                return Math.Pow((double)a, (double)((ScalarFloat64)b).Value);
            }
            else
            {
                return np.power(np.FromAny(a), np.FromAny(b));
            }
        }

        /// <summary>
        /// Returns the transpose of this object, for scalars there is no change.
        /// </summary>
        public object T {
            get { return this; }
        }


        public object take(object indices, object axis = null, ndarray @out = null, object mode = null) {
            return ToArray().take(indices, axis, @out, mode);
        }

        public object this[params object[] args] {
            get {
                return ToArray()[args: args];
            }
            set {
                throw new ArgumentTypeException("array-scalars are immutable");
            }
        }

        public virtual object this[int index] {
            get {
                return ToArray()[index];
            }
        }

        public virtual object this[long index] {
            get {
                return ToArray()[index];
            }
        }

 

        public virtual object this[System.Numerics.BigInteger index] {
            get {
                return ToArray()[index];
            }
        }


        public byte[] tobytes(NPY_ORDER order = NPY_ORDER.NPY_ANYORDER) {
            return ToArray().tobytes(order);
        }



        #endregion

        #region operators

   

        public static object operator +(ScalarGeneric a) {
            return a;
        }


        #endregion

        internal static dtype GetDtype(int size, NPY_TYPECHAR typechar) {
            if (typechar == NPY_TYPECHAR.NPY_UNICODELTR) {
                dtype d = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_UNICODE);
                d = NpyCoreApi.DescrNew(d);
                d.ElementSize = size * 4;
                return d;
            } else if (typechar ==  NPY_TYPECHAR.NPY_STRINGLTR) {
                dtype d = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_STRING);
                d = NpyCoreApi.DescrNew(d);
                d.ElementSize = size;
                return d;
            } else {
                NPY_TYPES t = NpyCoreApi.TypestrConvert(size, typechar);
                return NpyCoreApi.DescrFromType(t);
            }
        }

        internal static object ScalarFromData(dtype type, VoidPtr data, int size) {
            return type.ToScalar(data, size);
        }

        #region IConvertible

        public virtual bool ToBoolean(IFormatProvider fp=null) {
            throw new NotImplementedException();
        }

        public virtual byte ToByte(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual char ToChar(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual DateTime ToDateTime(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Decimal ToDecimal(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Double ToDouble(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Int16 ToInt16(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Int32 ToInt32(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Int64 ToInt64(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual SByte ToSByte(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Single ToSingle(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual String ToString(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual Object ToType(Type t, IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual UInt16 ToUInt16(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual UInt32 ToUInt32(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual UInt64 ToUInt64(IFormatProvider fp = null) {
            throw new NotImplementedException();
        }

        public virtual TypeCode GetTypeCode() {
            throw new NotImplementedException();
        }

        #endregion
    }

    public class ScalarBool : ScalarGeneric
    {
  

        public ScalarBool() {
            value = false;
        }

        public ScalarBool(bool val) {
            value = val;
        }

        public override object Value { get { return value; } }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_BOOL);
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            numpyAPI.SetItem(result.Array, 0, value);
            return result;
        }

  
        public static implicit operator bool(ScalarBool s) {
            return s.value;
        }

  

        #region IConvertible

        public override bool ToBoolean(IFormatProvider fp=null) {
            return value;
        }

        public override Int16 ToInt16(IFormatProvider fp = null) {
            return value ? (short)1 : (short)0;
        }

        public override Int32 ToInt32(IFormatProvider fp = null) {
            return value ? 1 : 0;
        }

        public override Int64 ToInt64(IFormatProvider fp = null) {
            return value ? 1 : 0;
        }

        public override UInt16 ToUInt16(IFormatProvider fp = null) {
            return value ? (UInt16)1 : (UInt16)0;
        }

        public override UInt32 ToUInt32(IFormatProvider fp = null) {
            return value ? 1u : 0u;
        }

        public override UInt64 ToUInt64(IFormatProvider fp = null) {
            return value ? 1U : 0U;
        }


        public override String ToString(IFormatProvider fp = null) {
            return value.ToString();
        }

        #endregion

        private bool value;
        static private dtype dtype_;

        static private readonly ScalarBool FALSE = new ScalarBool(false);
        static private readonly ScalarBool TRUE = new ScalarBool(true);
    }

    public class ScalarNumber : ScalarGeneric
    {
        public override object dtype {
            get {
                return NpyCoreApi.DescrFromType(NPY_TYPES.NPY_DOUBLE);
            }
        }
    }

    public class ScalarInteger : ScalarNumber
    {
        public override object dtype {
            get {
                return NpyCoreApi.DescrFromType(NPY_TYPES.NPY_LONG);
            }
        }
    }

    public class ScalarSignedInteger : ScalarInteger {  }

    public class ScalarIntegerImpl<T> : ScalarInteger where T : IConvertible, IComparable<T>
    {
        protected T value;

        public override object Value { get { return value; } }

        #region IConvertible

        public override bool ToBoolean(IFormatProvider fp = null) {
            return value.ToBoolean(fp);
        }

        public override byte ToByte(IFormatProvider fp = null) {
            return value.ToByte(fp);
        }

        public override char ToChar(IFormatProvider fp = null) {
            return value.ToChar(fp);
        }

        public override Decimal ToDecimal(IFormatProvider fp = null) {
            return value.ToDecimal(fp);
        }

        public override Double ToDouble(IFormatProvider fp = null) {
            return value.ToDouble(fp);
        }

        public override Int16 ToInt16(IFormatProvider fp = null) {
            return value.ToInt16(fp);
        }

        public override Int32 ToInt32(IFormatProvider fp = null) {
            return value.ToInt32(fp);
        }

        public override Int64 ToInt64(IFormatProvider fp = null) {
            return value.ToInt64(fp);
        }

        public override SByte ToSByte(IFormatProvider fp = null) {
            return value.ToSByte(fp);
        }

        public override Single ToSingle(IFormatProvider fp = null) {
            return value.ToSingle(fp);
        }

        public override UInt16 ToUInt16(IFormatProvider fp = null) {
            return value.ToUInt16(fp);
        }

        public override UInt32 ToUInt32(IFormatProvider fp = null) {
            return value.ToUInt32(fp);
        }

        public override UInt64 ToUInt64(IFormatProvider fp = null) {
            return value.ToUInt64(fp);
        }

        #endregion

  

    }

    public class ScalarInt8 : ScalarIntegerImpl<sbyte>
    {
        public ScalarInt8() {
            value = 0;
        }

        public ScalarInt8(sbyte value) {
            this.value = value;
        }

        public ScalarInt8(IConvertible value) {
            this.value = Convert.ToSByte(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_BYTE);
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            numpyAPI.SetItem(result.Array, 0, value);
            return result;
        }

  
        public static implicit operator int(ScalarInt8 i) {
            return i.value;
        }

        public static implicit operator BigInteger(ScalarInt8 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarInt8 i) {
            return i.value;
        }

  

        public static explicit operator bool(ScalarInt8 s) {
            return s.value != 0;
        }

    

        static private dtype dtype_;

        internal static readonly int MinValue = sbyte.MinValue;
        internal static readonly int MaxValue = sbyte.MaxValue;
    }

    public class ScalarInt16 : ScalarIntegerImpl<Int16>
    {
        public ScalarInt16() {
            value = 0;
        }

        public ScalarInt16(Int16 value) {
            this.value = value;
        }

        public ScalarInt16(string value, int @base = 10) {
            this.value = Convert.ToInt16(value, @base);
        }


        public ScalarInt16(IConvertible value) {
            this.value = Convert.ToInt16(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(2, NPY_TYPECHAR.NPY_INTLTR);
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            numpyAPI.SetItem(result.Array, 0, value);
            return result;
        }

        public static implicit operator int(ScalarInt16 i) {
            return i.value;
        }

        public static implicit operator BigInteger(ScalarInt16 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarInt16 i) {
            return i.value;
        }

 
        public static explicit operator bool(ScalarInt16 s) {
            return s.value != 0;
        }

        public object __index__() {
            return value;
        }

   

        static private dtype dtype_;

        internal static readonly int MinValue = Int16.MinValue;
        internal static readonly int MaxValue = Int16.MaxValue;
    }

    public class ScalarInt32 : ScalarIntegerImpl<Int32>
    {
        public ScalarInt32() {
            value = 0;
        }

        public ScalarInt32(Int32 value) {
            this.value = value;
        }

        public ScalarInt32(string value, int @base = 10) {
            this.value = Convert.ToInt32(value, @base);
        }

        public ScalarInt32(IConvertible value) {
            this.value = Convert.ToInt32(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NpyCoreApi.TypeOf_Int32);
                        }
                    }
                }
                return dtype_;
            }
        }

        public override object Value { get { return value; } }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            numpyAPI.SetItem(result.Array, 0, value);
            return result;
        }

        public static implicit operator int(ScalarInt32 i) {
            return i.value;
        }

        public static implicit operator BigInteger(ScalarInt32 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarInt32 i) {
            return i.value;
        }

        public static explicit operator bool(ScalarInt32 s) {
            return s.value != 0;
        }


 
        public object __index__() {
            return value;
        }

 
        static private dtype dtype_;

        internal static readonly int MinValue = Int32.MinValue;
        internal static readonly int MaxValue = Int32.MaxValue;
    }


    /// <summary>
    /// This is a fairly ugly workaround to an issue with scalars on IronPython.  Each int scalar
    /// represents a specific size integer (8, 16, 32, or 6 bits). However, in the core there are
    /// five types - byte, short, int, long, and longlong with two being the same size based on
    /// platform (32-bit int == long, 64-bit long == longlong). This lets us represent an int were
    /// int and long (int32) are the same size.
    /// </summary>
    public class ScalarIntC : ScalarInt32
    {
        public ScalarIntC() {
            value = 0;
        }

        public ScalarIntC(Int32 value) : base(value) {
        }

        public ScalarIntC(string value, int @base = 10) : base(value, @base) {
        }

        public ScalarIntC(IConvertible value) : base(value) {
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_INT);
                        }
                    }
                }
                return dtype_;
            }
        }

        public static implicit operator int(ScalarIntC i) {
            return i.value;
        }

        public static implicit operator BigInteger(ScalarIntC i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarIntC i) {
            return i.value;
        }

        public static explicit operator bool(ScalarIntC s) {
            return s.value != 0;
        }

        static private dtype dtype_;

        internal static readonly int MinValue = Int32.MinValue;
        internal static readonly int MaxValue = Int32.MaxValue;
    }
    
    public class ScalarInt64 : ScalarIntegerImpl<Int64>
    {
        public ScalarInt64() {
            value = 0;
        }

        public ScalarInt64(Int64 value) {
            this.value = value;
        }

        public ScalarInt64(string value, int @base = 10) {
            this.value = Convert.ToInt64(value, @base);
        }


        public ScalarInt64(IConvertible value) {
            this.value = Convert.ToInt64(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(8, NPY_TYPECHAR.NPY_INTLTR);
                        }
                    }
                }
                return dtype_;
            }
        }

        public override object Value { get { return value; } }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            numpyAPI.SetItem(result.Array, 0, value);
            return result;
        }

   
        public static explicit operator int(ScalarInt64 i) {
            if (i < int.MinValue || i > int.MaxValue) {
                throw new OverflowException();
            }
            return (int)i.value;
        }

        public static implicit operator BigInteger(ScalarInt64 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarInt64 i) {
            return i.value;
        }

        public static explicit operator bool(ScalarInt64 s) {
            return s.value != 0;
        }

        public object __index__() {
            return value;
        }


        static private dtype dtype_;

        internal static readonly BigInteger MinValue = new BigInteger(Int64.MinValue);
        internal static readonly BigInteger MaxValue = new BigInteger(Int64.MaxValue);
    }


    /// <summary>
    /// This is a fairly ugly workaround to an issue with scalars on IronPython.  Each int scalar
    /// represents a specific size integer (8, 16, 32, or 6 bits). However, in the core there are
    /// five types - byte, short, int, long, and longlong with two being the same size based on
    /// platform (32-bit int == long, 64-bit long == longlong). This lets us represent an int were
    /// int and long (int32) are the same size.
    /// </summary>
    public class ScalarLongLong : ScalarInt64
    {
        public ScalarLongLong() {
            value = 0;
        }

        public ScalarLongLong(Int64 value)
            : base(value) {
        }

        public ScalarLongLong(IConvertible value)
            : base(value) {
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_LONG);
                        }
                    }
                }
                return dtype_;
            }
        }

        public static explicit operator int(ScalarLongLong i) {
            if (i < int.MinValue || i > int.MaxValue) {
                throw new OverflowException();
            }
            return (int)i.value;
        }

        public static implicit operator BigInteger(ScalarLongLong i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarLongLong i) {
            return i.value;
        }

        public static explicit operator bool(ScalarLongLong s) {
            return s.value != 0;
        }

        static private dtype dtype_;

        internal static new readonly long MinValue = Int32.MinValue;
        internal static new readonly long MaxValue = Int32.MaxValue;
    }


    public class ScalarUnsignedInteger : ScalarInteger
    {
        public override object dtype {
            get {
                return NpyCoreApi.DescrFromType(NPY_TYPES.NPY_ULONG);
            }
        }
    }

    public class ScalarUnsignedImpl<T> : ScalarUnsignedInteger where T : IConvertible, IComparable<T>
    {
        protected T value;

        public override object Value { get { return value; } }

        #region IConvertible

        public override bool ToBoolean(IFormatProvider fp = null) {
            return value.ToBoolean(fp);
        }

        public override byte ToByte(IFormatProvider fp = null) {
            return value.ToByte(fp);
        }

        public override char ToChar(IFormatProvider fp = null) {
            return value.ToChar(fp);
        }

        public override Decimal ToDecimal(IFormatProvider fp = null) {
            return value.ToDecimal(fp);
        }

        public override Double ToDouble(IFormatProvider fp = null) {
            return value.ToDouble(fp);
        }

        public override Int16 ToInt16(IFormatProvider fp = null) {
            return value.ToInt16(fp);
        }

        public override Int32 ToInt32(IFormatProvider fp = null) {
            return value.ToInt32(fp);
        }

        public override Int64 ToInt64(IFormatProvider fp = null) {
            return value.ToInt64(fp);
        }

        public override SByte ToSByte(IFormatProvider fp = null) {
            return value.ToSByte(fp);
        }

        public override Single ToSingle(IFormatProvider fp = null) {
            return value.ToSingle(fp);
        }

        public override UInt16 ToUInt16(IFormatProvider fp = null) {
            return value.ToUInt16(fp);
        }

        public override UInt32 ToUInt32(IFormatProvider fp = null) {
            return value.ToUInt32(fp);
        }

        public override UInt64 ToUInt64(IFormatProvider fp = null) {
            return value.ToUInt64(fp);
        }

        #endregion

   

    }

    public class ScalarUInt8 : ScalarUnsignedImpl<byte>
    {
        public ScalarUInt8() {
            value = 0;
        }

        public ScalarUInt8(byte value) {
            this.value = value;
        }

        public ScalarUInt8(IConvertible value) {
            try {
                this.value = Convert.ToByte(value);
            } catch (OverflowException) {
                this.value = Byte.MaxValue;
            }
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(1, NPY_TYPECHAR.NPY_UNSIGNEDLTR);
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            numpyAPI.SetItem(result.Array, 0, value);
            return result;
        }

        internal override ScalarGeneric FillData(ndarray arr, long offset, bool isNativeByteOrder) {
            value = (byte)numpyAPI.GetItem(arr.Array, (int)offset);
            return this;
        }

        public static implicit operator int(ScalarUInt8 i) {
            return i.value;
        }

        public static implicit operator BigInteger(ScalarUInt8 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarUInt8 i) {
            return i.value;
        }


        public static explicit operator bool(ScalarUInt8 s) {
            return s.value != 0;
        }

   
        static private dtype dtype_;

        internal static readonly int MinValue = 0;
        internal static readonly int MaxValue = byte.MaxValue;
    }

    public class ScalarUInt16 : ScalarUnsignedImpl<UInt16>
    {
        public ScalarUInt16() {
            value = 0;
        }

        public ScalarUInt16(UInt16 value) {
            this.value = value;
        }

        public ScalarUInt16(int value) {
            this.value = (ushort)(short)value;
        }

        public ScalarUInt16(IConvertible value) {
            this.value = Convert.ToUInt16(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(2, NPY_TYPECHAR.NPY_UNSIGNEDLTR);
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            numpyAPI.SetItem(result.Array, 0, value);
            return result;
        }

  
        public static implicit operator int(ScalarUInt16 i) {
            return i.value;
        }

        public static implicit operator BigInteger(ScalarUInt16 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarUInt16 i) {
            return i.value;
        }

        public static explicit operator bool(ScalarUInt16 s) {
            return s.value != 0;
        }

  
        static private dtype dtype_;

        internal static readonly int MinValue = 0;
        internal static readonly int MaxValue = UInt16.MaxValue;
    }

    public class ScalarUInt32 : ScalarUnsignedImpl<UInt32>
    {
        public ScalarUInt32() {
            value = 0;
        }

        public ScalarUInt32(UInt32 value) {
            this.value = value;
        }

        public ScalarUInt32(int value) {
            this.value = (uint)value;
        }

        public ScalarUInt32(IConvertible value) {
            this.value = Convert.ToUInt32(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NpyCoreApi.TypeOf_UInt32);
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            numpyAPI.SetItem(result.Array, 0, value);
            return result;
        }

 

        static private dtype dtype_;

        internal static readonly int MinValue = 0;
        internal static readonly BigInteger MaxValue = new BigInteger(UInt32.MaxValue);
    }



    /// <summary>
    /// This is a fairly ugly workaround to an issue with scalars on IronPython.  Each int scalar
    /// represents a specific size integer (8, 16, 32, or 6 bits). However, in the core there are
    /// five types - byte, short, int, long, and longlong with two being the same size based on
    /// platform (32-bit int == long, 64-bit long == longlong). This lets us represent an int were
    /// int and long (int32) are the same size.
    /// </summary>
    public class ScalarUIntC : ScalarUInt32
    {
        public ScalarUIntC() {
            value = 0;
        }

        public ScalarUIntC(UInt32 value)
            : base(value) {
        }

        public ScalarUIntC(IConvertible value)
            : base(value) {
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_UINT);
                        }
                    }
                }
                return dtype_;
            }
        }

        public static explicit operator int(ScalarUIntC i) {
            if (i.value > int.MaxValue) {
                throw new OverflowException();
            }
            return (int)i.value;
        }

        public static implicit operator BigInteger(ScalarUIntC i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarUIntC i) {
            return i.value;
        }

        public static explicit operator bool(ScalarUIntC s) {
            return s.value != 0;
        }

        static private dtype dtype_;

        internal static new readonly uint MinValue = UInt32.MinValue;
        internal static new readonly uint MaxValue = UInt32.MaxValue;
    }

    public class ScalarUInt64 : ScalarUnsignedImpl<UInt64>
    {
        public ScalarUInt64() {
            value = 0;
        }

        public ScalarUInt64(UInt64 value) {
            this.value = value;
        }

        public ScalarUInt64(int value) {
            this.value = (ulong)(long)value;    // Cast to signed long then reinterpret bits into ulong so -2 converts to correct (big) value.
        }

        public ScalarUInt64(long value) {
            this.value = (ulong)value;
        }

        public ScalarUInt64(BigInteger value) {
            this.value = (ulong)value;
        }

        public ScalarUInt64(IConvertible value) {
            this.value = Convert.ToUInt64(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NpyCoreApi.TypeOf_UInt64);
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            numpyAPI.SetItem(result.Array, 0, value);
            return result;
        }




        static private dtype dtype_;

        internal static readonly int MinValue = 0;
        internal static readonly BigInteger MaxValue = new BigInteger(UInt64.MaxValue);
    }

    /// <summary>
    /// This is a fairly ugly workaround to an issue with scalars on IronPython.  Each int scalar
    /// represents a specific size integer (8, 16, 32, or 6 bits). However, in the core there are
    /// five types - byte, short, int, long, and longlong with two being the same size based on
    /// platform (32-bit int == long, 64-bit long == longlong). This lets us represent an int were
    /// int and long (int32) are the same size.
    /// </summary>
    public class ScalarULongLong : ScalarUInt64
    {
        public ScalarULongLong() {
            value = 0;
        }

        public ScalarULongLong(UInt64 value)
            : base(value) {
        }

        public ScalarULongLong(IConvertible value)
            : base(value) {
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_ULONG);
                        }
                    }
                }
                return dtype_;
            }
        }

        public static explicit operator int(ScalarULongLong i) {
            if (i.value > int.MaxValue) {
                throw new OverflowException();
            }
            return (int)i.value;
        }

        public static implicit operator BigInteger(ScalarULongLong i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarULongLong i) {
            return i.value;
        }

        public static explicit operator bool(ScalarULongLong s) {
            return s.value != 0;
        }

        static private dtype dtype_;

        internal static new readonly ulong MinValue = UInt64.MinValue;
        internal static new readonly ulong MaxValue = UInt64.MaxValue;
    }



    public class ScalarTimeInteger : ScalarInt64 { }


    public class ScalarInexact : ScalarNumber { }

    public class ScalarFloating : ScalarInexact { }

    public class ScalarFloatingImpl<T> : ScalarFloating where T : IConvertible
    {
        protected T value;

   
        public override object Value { get { return value; } }

        #region IConvertible

        public override bool ToBoolean(IFormatProvider fp = null) {
            return value.ToBoolean(fp);
        }

        public override byte ToByte(IFormatProvider fp = null) {
            return value.ToByte(fp);
        }

        public override char ToChar(IFormatProvider fp = null) {
            return value.ToChar(fp);
        }

        public override Decimal ToDecimal(IFormatProvider fp = null) {
            return value.ToDecimal(fp);
        }

        public override Double ToDouble(IFormatProvider fp = null) {
            return value.ToDouble(fp);
        }

        public override Int16 ToInt16(IFormatProvider fp = null) {
            return value.ToInt16(fp);
        }

        public override Int32 ToInt32(IFormatProvider fp = null) {
            return value.ToInt32(fp);
        }

        public override Int64 ToInt64(IFormatProvider fp = null) {
            return value.ToInt64(fp);
        }

        public override SByte ToSByte(IFormatProvider fp = null) {
            return value.ToSByte(fp);
        }

        public override Single ToSingle(IFormatProvider fp = null) {
            return value.ToSingle(fp);
        }

        public override UInt16 ToUInt16(IFormatProvider fp = null) {
            return value.ToUInt16(fp);
        }

        public override UInt32 ToUInt32(IFormatProvider fp = null) {
            return value.ToUInt32(fp);
        }

        public override UInt64 ToUInt64(IFormatProvider fp = null) {
            return value.ToUInt64(fp);
        }

        #endregion
    }

    public class ScalarFloat32 : ScalarFloatingImpl<Single>
    {
        public ScalarFloat32() {
            value = 0;
        }

        public ScalarFloat32(Single value) {
            this.value = value;
        }

        public ScalarFloat32(IConvertible value) {
            this.value = Convert.ToSingle(value);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(4, NPY_TYPECHAR.NPY_FLOATINGLTR);
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            numpyAPI.SetItem(result.Array, 0, value);
            return result;
        }

        public static explicit operator int(ScalarFloat32 i) {
            if (i.value < int.MinValue || i.value > int.MaxValue) {
                throw new OverflowException();
            }
            return (int)i.value;
        }

        public static implicit operator BigInteger(ScalarFloat32 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator Single(ScalarFloat32 i) {
            return i.value;
        }

        public static implicit operator double(ScalarFloat32 i) {
            return i.value;
        }

        public static implicit operator Complex(ScalarFloat32 x) {
            return new Complex(x.value, 0.0);
        }


        static private dtype dtype_;
    }

    public class ScalarFloat64 : ScalarFloatingImpl<Double>
    {
        public ScalarFloat64() {
            value = 0;
        }

        public ScalarFloat64(Double value) {
            this.value = value;
        }

        public ScalarFloat64(IConvertible value) {
            this.value = Convert.ToDouble(value);
        }

        public ScalarFloat64(ndarray value) {
            if (value.ndim == 0) {
                object v = ((dtype)dtype).ToScalar(value);
                if (v is IConvertible) {
                    this.value = Convert.ToDouble((IConvertible)v);
                } else {
                    throw new ArgumentTypeException(
                        String.Format("Unable to convert array of {0} to double", value.Dtype.ToString()));
                }
            } else {
                throw new ArgumentTypeException("Only 0-d arrays can be converted to scalars");
            }
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(8, NPY_TYPECHAR.NPY_FLOATINGLTR);
                        }
                    }
                }
                return dtype_;
            }
        }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            numpyAPI.SetItem(result.Array, 0, value);
            return result;
        }


        public static explicit operator int(ScalarFloat64 i) {
            if (i.value < int.MinValue || i.value > int.MaxValue) {
                throw new OverflowException();
            }
            return (int)i.value;
        }

        public static implicit operator BigInteger(ScalarFloat64 i) {
            return new BigInteger(i.value);
        }

        public static implicit operator double(ScalarFloat64 i) {
            return i.value;
        }

        public static implicit operator Complex(ScalarFloat64 x) {
            return new Complex(x.value, 0.0);
        }

   
        public static explicit operator bool(ScalarFloat64 s) {
            return s.value != 0;
        }



        static private dtype dtype_;
    }

    public class ScalarComplexFloating : ScalarInexact
    {
        public override object dtype {
            get {
                return NpyCoreApi.DescrFromType(NPY_TYPES.NPY_COMPLEX);
            }
        }

    }


    public class ScalarComplex64 : ScalarComplexFloating
    {
        public ScalarComplex64() {
            value.Real = 0.0f;
            value.Imag = 0.0f;
        }

        public ScalarComplex64(object o) {
            SetFromObj(o);
        }

        public ScalarComplex64(float real, float imag) {
            value.Real = real;
            value.Imag = imag;
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(8, NPY_TYPECHAR.NPY_COMPLEXLTR);
                        }
                    }
                }
                return dtype_;
            }
        }

        public override object Value { get { return new Complex(value.Real, value.Imag); } }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            numpyAPI.SetItem(result.Array, 0, value);
            return result;
        }

  

        public override object imag {
            get {
                return new ScalarFloat32(value.Imag);
            }
        }

        public override object real {
            get {
                return new ScalarFloat32(value.Real);
            }
        }

        public static implicit operator Complex(ScalarComplex64 x) {
            return new Complex(x.value.Real, x.value.Imag);
        }

        public static implicit operator string(ScalarComplex64 x) {
            return x.ToString();
        }


        const string RealNumFormat = "R";
        public override string ToString(IFormatProvider fp = null)
        {
            // Use the Python str() function instead of .NET formatting because the
            // formats are slightly different and cause regression failures.
            if (value.Real == 0.0)
            {
                return String.Format("{0}j", value.Imag.ToString(RealNumFormat));
            }
            else
            {
                return String.Format("({0}+{1}j)", value.Real.ToString(RealNumFormat), value.Imag.ToString(RealNumFormat));
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Data
        {
            internal float Real;
            internal float Imag;
        }


        /// <summary>
        /// Sets the object value from an unknown object type.  If imagOnly is false, then the real
        /// or real and imaginary parts are set.  If imagOnly is set, then only the imaginary part
        /// is set and arguments of complex type are rejected.
        /// </summary>
        /// <param name="o">Value to set</param>
        /// <param name="imagOnly">True only sets imaginary part, false sets real or both</param>
        protected void SetFromObj(object o) {
            if (o == null) real = imag = 0.0f;
            else if (o is int) {
                value.Real = (int)o;
                value.Imag = 0.0f;
            } else if (o is long) {
                value.Real = (long)o;
                value.Imag = 0.0f;
            } else if (o is float) {
                value.Real = (float)o;
                value.Imag = 0.0f;
            } else if (o is double) {
                value.Real = (float)(double)o;
                value.Imag = 0.0f;
            } else if (o is Complex) {
                value.Real = (float)((Complex)o).Real;
                value.Imag = (float)((Complex)o).Imaginary;
            } else if (o is ScalarComplex64) {
                value = ((ScalarComplex64)o).value;
            } else if (o is ScalarComplex128) {
                value.Real = (float)(double)((ScalarComplex128)o).real;
                value.Imag = (float)(double)((ScalarComplex128)o).imag;
            } else if (o is ScalarGeneric) {
                value.Real = (float)(double)((ScalarGeneric)o).real;
                value.Imag = 0.0f;
            } else throw new ArgumentTypeException(
                  String.Format("Unable to construct complex value from type '{0}'.", o.GetType().Name));
        }


        private Data value;
        static private dtype dtype_;
    }

    public class ScalarComplex128 : ScalarComplexFloating
    {
        public ScalarComplex128() {
            value = 0;
        }

        public ScalarComplex128(object o) {
            SetFromObj(o);
        }

        public ScalarComplex128(double real, double imag) {
            value = new Complex(real, imag);
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = GetDtype(16, NPY_TYPECHAR.NPY_COMPLEXLTR);
                        }
                    }
                }
                return dtype_;
            }
        }

        public override object Value { get { return value; } }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            numpyAPI.SetItem(result.Array, 0, value);
            return result;
        }

  
        public object __complex__() {
            return new Complex(value.Real, value.Imaginary);
        }

        public override object imag {
            get {
                return new ScalarFloat64(value.Imaginary);
            }
        }

        public override object real {
            get {
                return new ScalarFloat64(value.Real);
            }
        }
        string RealNumFormat = "R";
        public override string ToString(IFormatProvider fp = null) {
            if (value.Real == 0.0) {
                return String.Format("{0}j",value.Imaginary.ToString(RealNumFormat));
            } else {
                return String.Format("({0}+{1}j)", value.Real.ToString(RealNumFormat),value.Imaginary.ToString("R"));
            }
        }

        public static implicit operator string(ScalarComplex128 x) {
            return x.ToString();
        }

        public static implicit operator Complex(ScalarComplex128 x) {
            return x.value;
        }

        /// <summary>
        /// Sets the object value from an unknown object type.  If imagOnly is false, then the real
        /// or real and imaginary parts are set.  If imagOnly is set, then only the imaginary part
        /// is set and arguments of complex type are rejected.
        /// </summary>
        /// <param name="o">Value to set</param>
        /// <param name="imagOnly">True only sets imaginary part, false sets real or both</param>
        protected void SetFromObj(object o) {
            if (o == null) real = imag = 0.0f;
            else if (o is int) {
                value = new Complex((int)o, 0.0);
            } else if (o is long) {
                value = new Complex((long)o, 0.0);
            } else if (o is float) {
                value = new Complex((float)o, 0.0);
            } else if (o is double) {
                value = new Complex((double)o, 0.0);
            } else if (o is ScalarInt16) {
                value = new Complex((int)(ScalarInt16)o, 0.0);
            } else if (o is ScalarInt16) {
                value = new Complex((int)(ScalarInt32)o, 0.0);
            } else if (o is ScalarInt16) {
                value = new Complex((long)(ScalarInt64)o, 0.0);
            } else if (o is Complex) {
                value = (Complex)o;
            } else if (o is ScalarComplex64) {
                value = new Complex((float)((ScalarComplex64)o).real, (float)((ScalarComplex64)o).imag);
            } else if (o is ScalarComplex128) {
                value = ((ScalarComplex128)o).value;
            } else if (o is ScalarGeneric) {
                value = new Complex((double)((ScalarComplex64)o).real, (float)((ScalarComplex64)o).imag);
            }
            else throw new ArgumentTypeException(
                  String.Format("Unable to construct complex value from type '{0}'.", o.GetType().Name));
        }


        private Complex value;
        static private dtype dtype_;
    }

    public class ScalarFlexible : ScalarGeneric { }

    public class ScalarVoid : ScalarFlexible, IDisposable
    {
  
        private static object FromObject(object val) {
            ndarray arr = np.FromAny(val, NpyCoreApi.DescrFromType(NPY_TYPES.NPY_VOID), flags: NPYARRAYFLAGS.NPY_FORCECAST);
            return ndarray.ArrayReturn(arr);
        }

        public ScalarVoid() {
            dtype_ = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_VOID);
            dataptr =null;
        }

        internal ScalarVoid(int size) {
            AllocData(size);
            dtype_ = new dtype(NpyCoreApi.DescrFromType(NPY_TYPES.NPY_VOID));
            dtype_.ElementSize = size;
        }

        private void AllocData(int size) {
     
        }

        ~ScalarVoid() {
            Dispose(false);
        }

        public void Dispose() {
            Dispose(true);
        }

        private void Dispose(bool disposing) {
   
        }

        public override object dtype {
            get {
                return dtype_;
            }
        }

        public override object Value { get { return this[0]; } }

        internal override ndarray ToArray() {
            ndarray a = NpyCoreApi.NewFromDescr(dtype_, new npy_intp[0], null, dataptr, 0, null);
            //a.BaseObj = this;
            return a;
        }

        internal override ScalarGeneric FillData(ndarray arr, long offset, bool isNativeByteOrder)
        {
            int elsize = arr.ItemSize;

            if (dataptr != null)
            {
                throw new RuntimeException("Unexpected modification to existing scalar object.");
            }


            dtype_ = arr.Dtype;

            if (arr.Dtype.HasNames)
            {
                base_arr = arr;
                dataptr = arr.Array.data + offset;
            }
            else
            {
                base_arr = null;
                AllocData(elsize);

                arr.CopySwapOut(offset, dataptr, !arr.IsNotSwapped);
            }
            return this;
        }
 
        internal override ScalarGeneric FillData(VoidPtr dataPtr, int size, bool isNativeByteOrder) {
            throw new NotImplementedException("Scalar fill operations are not supported for flexible (variable-size) types.");
        }

        public override object this[int index] {
            get {
                return Index(index);
            }
        }

        public override object this[long index] {
            get {
                return Index((int)index);
            }
        }

        public override object this[BigInteger index] {
            get {
                return Index((int)index);
            }
        }

        public object this[string index] {
            get {
                return Index(index);
            }
        }

        private object Index(int index) {
            if (!dtype_.HasNames) {
                throw new IndexOutOfRangeException("cant' index void scalar without fields");
            }
            return Index(dtype_.Names[index]);
        }

        private object Index(string index) {
            return ToArray()[index];
        }

        private dtype dtype_;
        private VoidPtr dataptr;

        /// <summary>
        /// When set this object is sharing memroy with the array below.  This occurs when accessing
        /// elements of record type array.
        /// </summary>
        private ndarray base_arr;
    }

    public class ScalarCharacter : ScalarFlexible
    {
        public override object dtype {
            get {
                return NpyCoreApi.DescrFromType(NPY_TYPES.NPY_STRING);
            }
        }
    }

 


    public class ScalarObject : ScalarGeneric
    {
   

        public ScalarObject() {
            value = null;
        }

        public ScalarObject(object o) {
            value = o;
        }

        public override object dtype {
            get {
                if (dtype_ == null) {
                    lock (GetType()) {
                        if (dtype_ == null) {
                            dtype_ = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_OBJECT);
                        }
                    }
                }
                return dtype_;
            }
        }

        public override object Value { get { return value; } }

        internal override ndarray ToArray() {
            ndarray result = NpyCoreApi.AllocArray((dtype)dtype, 0, null, false);
            result.SetItem(value, 0);
            return result;
        }

        internal override ScalarGeneric FillData(ndarray arr, long offset, bool isNativeByteOrder) {
            value = arr.GetItem(offset);
            return this;
        }


        internal override ScalarGeneric FillData(VoidPtr dataPtr, int size, bool isNativeByteOrder) {
            throw new NotImplementedException("Scalar fill operations are not supported for flexible (variable-size) types.");
        }

        private object value;
        private static dtype dtype_;
    }
}
