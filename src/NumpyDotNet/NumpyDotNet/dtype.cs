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
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using NumpyDotNet;
using NumpyLib;

#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif


namespace NumpyDotNet {

    public class dtype_serializable
    {
        public NpyArray_Descr_serializable core;
    }


    public class dtype  {

        NpyArray_Descr core;


        public dtype_serializable ToSerializable()
        {
            return new dtype_serializable()
            {
                core = this.core.ToSerializable(),
            };
        }


        public dtype(dtype_serializable dtype_Serializable)
        {
            core = NpyArray_Descr.FromSerializable(dtype_Serializable.core);
            funcs = core.f;
        }

        /// <summary>
        /// Constructs a new NpyArray_Descr objet matching the passed one.
        /// Equivalent to NpyAray_DescrNew.
        /// </summary>
        /// <param name="d">Descriptor to duplicate</param>
        internal dtype(dtype d) {
            core = numpyAPI.NpyArray_DescrNew(d.core);
            funcs = core.f;
        }

        /// <summary>
        /// Creates a wrapper for an array created on the native side, such as
        /// the result of a slice operation.
        /// </summary>
        /// <param name="d">Pointer to core NpyArray_Descr structure</param>
        internal dtype(NpyArray_Descr d) {
            core = d;
            funcs = d.f;
        }

        public override string ToString()
        {
            string ret;

            if (this.HasNames)
            {
                Object lst = this.descr;
                ret = (lst != null) ? lst.ToString() : "<err>";
       
                ret = String.Format("('{0}', {1})", this.str, this.descr);
            }
            else if (this.HasSubarray)
            {
                dtype b = @base;
                if (!b.HasNames && !b.HasSubarray)
                {
                    ret = String.Format("('{0}',{1})", b.ToString(), shape);
                }
                else
                {
                    ret = String.Format("({0},{1})", b.ToString(), shape);
                }
            }
            else if (NpyDefs.IsFlexible(this.TypeNum) || !this.IsNativeByteOrder)
            {
                ret = this.str;
            }
            else
            {
                ret = this.name;
            }
            return ret;
        }
  

        #region Python interface

 
        /// <summary>
        /// Returns the name of the underlying data type such as 'int32' or 'object'.
        /// </summary>
        public string name
        {
            get
            {
                string typeName = this.TypeNum.ToString();
                if (NpyDefs.IsUserDefined(this.TypeNum))
                {
                    int i = typeName.LastIndexOf('.');
                    if (i != -1)
                    {
                        typeName = typeName.Substring(i + 1);
                    }
                }
                else
                {
                    int prefixLen = "NPY_".Length;
                    int len = typeName.Length;
                    if (typeName[len - 1] == '_')
                    {
                        len--;
                    }
                    len -= prefixLen;
                    typeName = typeName.Substring(prefixLen, len);
                }

                if (NpyDefs.IsFlexible(this.TypeNum) && this.ElementSize != 0)
                {
                    typeName += this.ElementSize.ToString();
                }
     
                return typeName;
            }
        }

        public string str {
            get {
                char endian = this.ByteOrder;
                int size = this.ElementSize;

                if (endian == '=') {
                    endian = NpyDefs.IsNativeByteOrder('<') ? '<' : '>';
                }

                string ret = String.Format("{0}{1}{2}", (char)endian, (char)this.Kind, size);
     
                return ret;
            }
        }

        public object descr
        {
            get
            {
                if (!this.HasNames)
                {
                    List<PythonTuple> res = new List<PythonTuple>();
                    res.Add(new PythonTuple(new Object[] { "", this.str }));
                    return res;
                }
                return null;
            }
        }

        public dtype @base
        {
            get
            {
                return new dtype(Subarray._base);
            }
        }


        /// <summary>
        /// A tuple describing the size of each dimension of the array.
        /// </summary>
        internal PythonTuple shape
        {
            get
            {
                var subarray = Subarray;
                if (subarray == null)
                {
                    return new PythonTuple();
                }
                else
                {
                    long n = subarray.shape_num_dims;
                    object[] dims = new object[n];
                    for (long i = 0; i < n; i++)
                    {
                        dims[i] = subarray.shape_dims[i];
                    }
                    return new PythonTuple(dims);
                }
            }
        }


        public bool isnative {
            get {
                return NpyCoreApi.DescrIsNative(this);
            }
        }


        internal bool IsComplex
        {
            get { return NpyDefs.IsComplex(this.TypeNum); }
        }

        internal bool IsDecimal
        {
            get { return NpyDefs.IsDecimal(this.TypeNum); }
        }

        internal bool IsInteger
        {
            get { return NpyDefs.IsInteger(this.TypeNum); }
        }

        internal bool IsSignedInt
        {
            get { return NpyDefs.IsSigned(this.TypeNum); }
        }

        internal bool IsUnsignedInt
        {
            get { return NpyDefs.IsUnsigned(this.TypeNum); }
        }

        internal bool IsFloatingPoint
        {
            get { return NpyDefs.IsFloat(this.TypeNum); }
        }

        internal bool IsInexact
        {
            get { return IsFloatingPoint || IsComplex; }
        }

        public bool IsFlexible
        {
            get { return NpyDefs.IsFlexible(this.TypeNum); }
        }


        public object fields { get { return this.GetFieldsDict(); } }

        public Dictionary<string, object> Fields { get { return this.GetFieldsDict(); } }

    
        public int itemsize {
            get { return ElementSize; }
        }

        public object names {
            get {
                var n = Names;
                if (n != null) {
                    return new PythonTuple(n);
                } else {
                    return null;
                }
            }
            set {
                int n = this.Names.Count();
                IEnumerable<object> ival = value as IEnumerable<object>;
                if (ival == null) {
                    throw new ArgumentException(String.Format("Value must be a sequence of {0} strings.", n));
                }
                if (ival.Any(x => !(x is string))) {
                    throw new ArgumentException("All items must be strings.");
                }
                NpyCoreApi.SetNamesList(this, ival.Cast<String>().ToArray());
            }
        }
   

        public string kind { get { return new string((char)this.Kind, 1); } }

  
        public string byteorder { get { return new string((char)this.ByteOrder, 1); } }

        public int alignment { get { return this.Alignment; } }

        internal NpyArray_Descr_Flags flags { get { return this.Flags; } }

        public dtype newbyteorder(string endian = null) {
            return NpyCoreApi.DescrNewByteorder(this, NpyUtil_ArgProcessing.ByteorderConverter(endian));
        }

        public object this[Object idx]
        {
            get
            {
                object result;

                if (!this.HasNames)
                {
                    result = null;
                }
                else if (idx is string)
                {
                    if (!GetFieldsDict().TryGetValue((string)idx, out result))
                    {
                        throw new System.Collections.Generic.KeyNotFoundException(
                            String.Format("Field named \"{0}\" not found.", (string)idx));
                    }
                }
                else if (idx is int || idx is long || idx is BigInteger)
                {
                    int i = Convert.ToInt32(idx);
                    try
                    {
                        result = GetFieldsDict()[Names[i]]; // Names list checks index out of range
                    }
                    catch (ArgumentException e)
                    {
                        // Translate exception type to make test_dtype_keyerrrs test in test_regression.py
                        // happy.
                        throw new IndexOutOfRangeException(e.Message);
                    }
                }
                else
                {
                    throw new ArgumentException("Field key must be an integer, string, or unicode");
                }

                // If result is set, it is a PythonTuple of the dtype and offset. We just want to return
                // the dtype itself.
                if (result != null)
                {
                    result = ((PythonTuple)result)[0];
                }
                return result;
            }
        }

        #endregion

        #region .NET Properties

        internal NpyArray_Descr Descr {
            get { return core; }
        }

        public bool IsNativeByteOrder {
            get { return NpyDefs.IsNativeByteOrder(ByteOrder); }
        }

        public char Kind {
            get {
                return Convert.ToChar(core.kind);
            }
        }

        public char Type {
            get { return (char)core.type; }
        }

        public char ByteOrder {
            get { return core.byteorder; }
            set { core.byteorder = value; }
        }

        internal NpyArray_Descr_Flags Flags {
            get { return core.flags; }
            private set { core.flags = value; }
        }

        internal bool ChkFlags(NpyArray_Descr_Flags flags) {
            return (Flags & flags) == flags;
        }

        public NPY_TYPES TypeNum {
            get { return core.type_num; }
        }

        public int ElementSize {
            get { return core.elsize; }
            internal set { core.elsize = value; }
        }

        public int Alignment {
            get { return core.alignment; }
            internal set { core.alignment = value; }
        }

        public bool HasNames {
            get { return core.names != null && core.names.Count > 0; }
        }

        public int Length {
            get {
                return core.names.Count;
             }
        }


        internal List<string> Names {
            get {
                return core.names;
            }
        }

        internal bool HasSubarray {
            get { return core.subarray != null; }
        }

        internal NpyArray_ArrayDescr Subarray
        {
            get
            {
                return core.subarray;
            }
        }

        internal NpyArray_ArrFuncs f {
            get { return funcs; }
        }

        #endregion


        #region Comparison
        public override bool Equals(object obj) {
            if (obj != null && obj is dtype) return Equals((dtype)obj);
            return false;
        }

        public bool Equals(dtype other) {
            if (other == null) return false;
            return (this.core == other.core || NpyCoreApi.EquivTypes(this, other));
        }

        /// <summary>
        /// Compares two types and returns true if they are equivalent,
        /// including complex types, even if represented by two different
        /// underlying descriptor objects.
        /// </summary>
        /// <param name="t1">Type 1</param>
        /// <param name="t2">Type 2</param>
        /// <returns>True if types are equivalent</returns>
        public static bool operator ==(dtype t1, dtype t2) {
            return System.Object.ReferenceEquals(t1, t2) ||
                (object)t1 != null && (object)t2 != null && t1.Equals(t2);
        }

        public static bool operator !=(dtype t1, dtype t2) {
            return !System.Object.ReferenceEquals(t1, t2) &&
                ((object)t1 == null || (object)t2 == null || !t1.Equals(t2));
        }

        public override int GetHashCode() {
            int hash = 17;
            foreach (object item in HashItems()) {
                hash = hash * 31 + item.GetHashCode();
            }
            return hash;
        }

        private IEnumerable<object> HashItems()
        {
            if (!HasNames && !HasSubarray)
            {
                yield return Kind;

                if (ByteOrder == (byte)'=')
                {
                    yield return NpyCoreApi.NativeByteOrder;
                }
                else
                {
                    yield return ByteOrder;
                }

                yield return TypeNum;
                yield return ElementSize;
                yield return Alignment;
            }
            else
            {
                if (HasNames)
                {
                    foreach (object item in FieldsHashItems())
                    {
                        yield return item;
                    }
                }
                if (HasSubarray)
                {
                    foreach (object item in SubarrayHashItems())
                    {
                        yield return item;
                    }
                }
            }
        }

        private IEnumerable<object> FieldsHashItems()
        {
            if (HasNames)
            {
                yield return core.fields.bucketArray.ToArray();
            }
        }

        private IEnumerable<object> SubarrayHashItems() {
            if (HasSubarray) {
                foreach (object item in @base.HashItems()) {
                    yield return item;
                }
                foreach (object item in shape) {
                    yield return item;
                }
            }
        }

    
        #endregion


        #region Internal data & methods

        private string AppendDateTimeTypestr(string str) {
            // TODO: Fix date time type string. See descriptor.c: _append_to_datetime_typestr
            throw new NotImplementedException("to do ");
        }


        private Dictionary<string, object> GetFieldsDict()
        {
            Dictionary<string, object> ret;

            if (!HasNames)
            {
                ret = null;
            }
            else
            {
                NpyDict_Iter iter = null;
                NpyDict dict = core.fields;
                ret = new Dictionary<string, object>();
                try
                {
                    NpyDict_KVPair KVPair = new NpyDict_KVPair();
                    iter = NpyCoreApi.NpyDict_AllocIter();
                    while (NpyCoreApi.NpyDict_Next(dict, iter, KVPair))
                    {
                        string key = (string)KVPair.key;
                        NpyArray_DescrField value = (NpyArray_DescrField)KVPair.value;
                        PythonTuple t;

                        dtype d = new dtype(value.descr);
                        if (value.title == null)
                        {
                            t = new PythonTuple(new Object[] { d, value.offset });
                        }
                        else
                        {
                            t = new PythonTuple(new Object[] { d, value.offset, value.title });
                        }
                        ret.Add(key, t);
                    }
                }
                finally
                {
                    NpyCoreApi.NpyDict_FreeIter(iter);
                }
            }
            return ret;
        }

        /// <summary>
        /// Type-specific functions
        /// </summary>
        [NonSerialized]
        private readonly NpyArray_ArrFuncs funcs;

        #endregion


    }


}
