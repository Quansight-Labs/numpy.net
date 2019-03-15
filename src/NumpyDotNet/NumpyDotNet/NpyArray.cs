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
using System.Numerics;
using System.Text;
using System.Runtime.InteropServices;
using NumpyLib;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet {
    /// <summary>
    /// Implements array manipulation and construction functionality.  This
    /// class has functionality corresponding to functions in arrayobject.c,
    /// ctors.c, and multiarraymodule.c
    /// </summary>
    public static partial class np
    {     
  
        /// <summary>
        /// Copies the source object into the destination array.  src can be
        /// any type so long as the number of elements matches dest.  In the
        /// case of strings, they will be padded with spaces if needed but
        /// can not be longer than the number of elements in dest.
        /// </summary>
        /// <param name="dest">Destination array</param>
        /// <param name="src">Source object</param>
        public static void CopyObject(ndarray dest, Object src)
        {
            // For char arrays pad the input string.
            if (dest.Dtype.Type == NPY_TYPECHAR.NPY_CHARLTR &&
                dest.ndim > 0 && src is String)
            {
                int ndimNew = (int)dest.Dims[dest.ndim - 1];
                int ndimOld = ((String)src).Length;

                if (ndimNew > ndimOld)
                {
                    src = ((String)src).PadRight(ndimNew, ' ');
                }
            }

            ndarray srcArray;
            if (src is ndarray)
            {
                srcArray = (ndarray)src;
            }
            else if (false)
            {
                // TODO: Not handling scalars.  See arrayobject.c:111
            }
            else
            {
                srcArray = np.FromAny(src, dest.Dtype, 0, dest.ndim, 0, null);
            }
            NpyCoreApi.MoveInto(dest, srcArray);
        }


        internal static void SetField(ndarray dest, NpyArray_Descr descr, int offset, object src)
        {
            // For char arrays pad the input string.
            if (dest.Dtype.Type == NPY_TYPECHAR.NPY_CHARLTR &&
                dest.ndim > 0 && src is String)
            {
                int ndimNew = (int)dest.Dims[dest.ndim - 1];
                int ndimOld = ((String)src).Length;

                if (ndimNew > ndimOld)
                {
                    src = ((String)src).PadRight(ndimNew, ' ');
                }
            }
            ndarray srcArray;
            if (src is ndarray)
            {
                srcArray = (ndarray)src;
            }
            else if (false)
            {
                // TODO: Not handling scalars.  See arrayobject.c:111
            }
            else
            {
                dtype src_dtype =  new dtype(descr);
                srcArray = np.FromAny(src, src_dtype, 0, dest.ndim,  NPYARRAYFLAGS.NPY_CARRAY, null);
            }
            NpyCoreApi.Incref(descr);
            if (numpyAPI.NpyArray_SetField(dest.core, descr, offset, srcArray.core) < 0)
            {
                NpyCoreApi.CheckError();
            }
        }




        /// <summary>
        /// Checks the strides against the shape of the array.  This duplicates
        /// NpyArray_CheckStrides and is only here because we don't currently support
        /// buffers and can simplify this function plus it's much faster to do here
        /// than to pass the arrays into the native world.
        /// </summary>
        /// <param name="elSize">Size of array element in bytes</param>
        /// <param name="shape">Size of each dimension of the array</param>
        /// <param name="strides">Stride of each dimension</param>
        /// <returns>True if strides are ok, false if not</returns>
        public static bool CheckStrides(this NpyArray a, int elSize, npy_intp[] shape, npy_intp[] strides)
        {
            // Product of all dimension sizes * element size in bytes.
            long numbytes = shape.Aggregate(1L, (acc, x) => acc * x) * elSize;
            long end = numbytes - elSize;
            for (int i = 0; i < shape.Length; i++)
            {
                if (strides[i] * (shape[i] - 1) > end) return false;
            }
            return true;
        }


        public static ndarray CheckFromAny(Object src, dtype descr, int minDepth,
            int maxDepth, NPYARRAYFLAGS requires, Object context)
        {

            if ((requires & NPYARRAYFLAGS.NPY_NOTSWAPPED) != 0)
            {
                if (descr == null && src is ndarray &&
                    !((ndarray)src).Dtype.IsNativeByteOrder)
                {
                    descr = new dtype(((ndarray)src).Dtype);
                }
                else if (descr != null && !descr.IsNativeByteOrder)
                {
                    // Descr replace
                }
                if (descr != null)
                {
                    descr.ByteOrder = '=';
                }
            }

            ndarray arr = np.FromAny(src, descr, minDepth, maxDepth, requires, context);

            if (arr != null && (requires & NPYARRAYFLAGS.NPY_ELEMENTSTRIDES) != 0 &&
                arr.ElementStrides == 0)
            {
                arr = arr.NewCopy(NPY_ORDER.NPY_ANYORDER);
            }
            return arr;
        }


        private static Exception UpdateIfCopyError()
        {
            return new ArgumentException("UPDATEIFCOPY used for non-array input.");
        }

        private static ndarray FromAnyReturn(ndarray result, int minDepth, int maxDepth)
        {
            if (minDepth != 0 && result.ndim < minDepth)
            {
                throw new ArgumentException("object of too small depth for desired array");
            }
            if (maxDepth != 0 && result.ndim > maxDepth)
            {
                throw new ArgumentException("object too deep for desired array");
            }
            return result;
        }

        internal static ndarray EnsureArray(this NpyArray a, object o)
        {
            if (o == null)
            {
                return null;
            }
            if (o.GetType() == typeof(ndarray))
            {
                return (ndarray)o;
            }
            if (o is ndarray)
            {
                return NpyCoreApi.FromArray((ndarray)o, null, NPYARRAYFLAGS.NPY_ENSUREARRAY);
            }
            return np.FromAny(o, flags: NPYARRAYFLAGS.NPY_ENSUREARRAY);
        }

        internal static ndarray EnsureAnyArray(object o)
        {
            if (o == null)
            {
                return null;
            }
            if (o is ndarray)
            {
                return (ndarray)o;
            }
            return np.FromAny(o, flags: NPYARRAYFLAGS.NPY_ENSUREARRAY);
        }


        /// <summary>
        /// Constructs a new array from multiple input types, like lists, arrays, etc.
        /// </summary>
        /// <param name="src"></param>
        /// <param name="descr"></param>
        /// <param name="minDepth"></param>
        /// <param name="maxDepth"></param>
        /// <param name="requires"></param>
        /// <param name="context"></param>
        /// <returns></returns>
        public static ndarray FromAny(Object src, dtype descr = null, int minDepth = 0,
            int maxDepth = 0, NPYARRAYFLAGS flags = 0, Object context = null)
        {
            ndarray result = null;

            if (src == null)
            {
                return np.empty(new shape(0), NpyCoreApi.DescrFromType(NPY_TYPES.NPY_OBJECT));
            }

            Type t = src.GetType();

            if (t != typeof(PythonTuple))
            {
                if (src is ndarray)
                {
                    result = NpyCoreApi.FromArray((ndarray)src, descr, flags);
                    return FromAnyReturn(result, minDepth, maxDepth);
                }

                if (t.IsArray)
                {
                    result = asanyarray(src);
                }

                if (src is ScalarGeneric)
                {
                    if ((flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) != 0)
                    {
                        throw UpdateIfCopyError();
                    }
                    result = FromScalar((ScalarGeneric)src, descr);
                    return FromAnyReturn(result, minDepth, maxDepth);
                }

                dtype newtype = (descr ?? FindScalarType(src));
                if (descr == null && newtype != null)
                {
                    if ((flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) != 0)
                    {
                        throw UpdateIfCopyError();
                    }
                    result = FromPythonScalar(src, newtype);
                    return FromAnyReturn(result, minDepth, maxDepth);
                }

                //result = FromScalar(src, descr, context);
                if (result != null)
                {
                    if (descr != null && !NpyCoreApi.EquivTypes(descr, result.Dtype) || flags != 0)
                    {
                        result = NpyCoreApi.FromArray(result, descr, flags);
                        return FromAnyReturn(result, minDepth, maxDepth);
                    }
                }
            }

            bool is_object = false;

            if ((flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) != 0)
            {
                throw UpdateIfCopyError();
            }
            if (descr == null)
            {
                descr = FindArrayType(src, null);
            }
            else if (descr.TypeNum == NPY_TYPES.NPY_OBJECT)
            {
                is_object = true;
            }

            if (result == null)
            {
                // Hack required because in C# strings are enumerations of chars, not objects. 
                // However, we want to keep src as a string if we are building a string or object array.
                if (!is_object &&
                    (descr.TypeNum != NPY_TYPES.NPY_STRING || descr.Type == NPY_TYPECHAR.NPY_CHARLTR) &&
                    descr.TypeNum != NPY_TYPES.NPY_UNICODE && src is string && ((string)src).Length > 1)
                {
                    src = ((string)src).Cast<object>();
                }

                bool seq = false;
                if (src is IEnumerable<object> )
                {
                    try
                    {
                        result = FromIEnumerable((IEnumerable<object>)src, descr, (flags & NPYARRAYFLAGS.NPY_FORTRAN) != 0, minDepth, maxDepth);
                        seq = true;
                    }
                    catch (InsufficientMemoryException)
                    {
                        throw;
                    }
                    catch
                    {
                        if (is_object)
                        {
                            result = FromNestedList(src, descr, (flags & NPYARRAYFLAGS.NPY_FORTRAN) != 0);
                            seq = true;
                        }
                    }
                }
                if (!seq)
                {
                    result = FromScalar(src, descr, null);
                }
            }
            return FromAnyReturn(result, minDepth, maxDepth);
        }

        private static ndarray FromScalar(object src, dtype descr, object context)
        {
            npy_intp[] dims = new npy_intp[1] { 1 };
            ndarray result = NpyCoreApi.AllocArray(descr, 1, dims, false);
            if (result.ndim != 1)
            {
                throw new ArgumentException("shape-mismatch on array construction");
            }

            result.Dtype.f.setitem(0, src, result.Array);
            return result;
        }

        internal static ndarray FromNestedList(object src, dtype descr, bool fortran)
        {
            npy_intp[] dims = new npy_intp[NpyDefs.NPY_MAXDIMS];
  

            int nd = ObjectDepthAndDimension(src, dims, 0, NpyDefs.NPY_MAXDIMS);
            if (nd == 0)
            {
                return FromPythonScalar(src, descr);
            }
            ndarray result = NpyCoreApi.AllocArray(descr, nd, dims, fortran);
            AssignToArray(src, result);
            return result;
        }

        /// <summary>
        /// Walks a set of nested lists (or tuples) to get the dimensions.  The dimensionality must
        /// be consistent for each nesting level. Thus, if one level is a mix of lsits and scalars,
        /// it is truncated and all are assumed to be scalar objects.
        ///
        /// That is, [[1, 2], 3, 4] is a 1-d array of 3 elements.  It just happens that element 0 is
        /// an object that is a list of [1, 2].
        /// </summary>
        /// <param name="src">Input object to talk</param>
        /// <param name="dims">Array of dimensions of size 'max' filled in up to the return value</param>
        /// <param name="idx">Current iteration depth, always start with 0</param>
        /// <param name="max">Size of dims array at the start, then becomes depth so far when !firstElem</param>
        /// <param name="firstElem">True if processing the first element of the list (populates dims), false for subsequent (checks dims)</param>
        /// <returns>Number of dimensions (depth of nesting)</returns>
        internal static int ObjectDepthAndDimension(object src, npy_intp[] dims, int idx, int max, bool firstElem = true)
        {
            int nd = -1;

            // Recursively walk the tree and get the sizes of each dimension. When processing the
            // first element in each sequence, firstElem is true and we populate dims[]. After that,
            // we just verify that dims[] matches for subsequent elements.
            IList<object> list = src as IList<object>;  // List and PythonTuple both implement IList
            if (max < 1 || list == null)
            {
                nd = 0;
            }
            else if (list.Count == 0)
            {
                nd = 0;
            }
            else if (max < 2)
            {
                // On the first pass, populate the dimensions array. One subsequent passes verify
                // that the size is the same or, if not,
                if (firstElem)
                {
                    dims[idx] = list.Count;
                    nd = 1;
                }
                else
                {
                    nd = (dims[idx] == list.Count) ? 1 : 0;
                }
            }
            else if (!firstElem && dims[idx] != list.Count)
            {
                nd = 0;
            }
            else
            {
                // First element we traverse up to max depth and fill in the dims array.
                nd = ObjectDepthAndDimension(list.First(), dims, idx + 1, max - 1, firstElem);

                // Subsequent elements we just check that the size of each dimension is the
                // same as clip the max depth to shallowest depth we have seen thus far.
                nd = list.Skip(1).Aggregate(nd, (ndAcc, elem) =>
                    Math.Min(ndAcc, ObjectDepthAndDimension(elem, dims, idx + 1, ndAcc, false))
                );
                nd += 1;
                dims[idx] = list.Count;
            }
            return nd;
        }

 
        internal static ndarray FromScalar(ScalarGeneric scalar, dtype descr = null)
        {
            ndarray arr = scalar.ToArray();
            if (descr != null && !NpyCoreApi.EquivTypes((dtype)scalar.dtype, descr))
            {
                arr = NpyCoreApi.CastToType(arr, descr, arr.IsFortran);
            }
            return arr;
        }


        internal static ndarray FromPythonScalar(object src, dtype descr)
        {
            int itemsize = descr.ElementSize;
            NPY_TYPES type = descr.TypeNum;

            if (itemsize == 0 && NpyDefs.IsExtended(type))
            {
                int n = PythonOps.Length(src);
                if (type == NPY_TYPES.NPY_UNICODE)
                {
                    n *= 4;
                }
                descr = new dtype(descr);
                descr.ElementSize = n;
            }

            ndarray result = NpyCoreApi.AllocArray(descr, 0, null, false);
            if (result.ndim > 0)
            {
                throw new ArgumentException("shape-mismatch on array construction");
            }

            result.Dtype.f.setitem(0, src, result.Array);
            return result;
        }


        /// <summary>
        /// Builds an array from a sequence of objects.  The elements of the sequence
        /// can also be sequences in which case this function recursively walks the
        /// nested sequences and builds an n dimentional array.
        ///
        /// IronPython tuples and lists work as sequences.
        /// </summary>
        /// <param name="src">Input sequence</param>
        /// <param name="descr">Desired array element type or null to determine automatically</param>
        /// <param name="fortran">True if array should be Fortran layout, false for C</param>
        /// <param name="minDepth"></param>
        /// <param name="maxDepth"></param>
        /// <returns>New array instance</returns>
        internal static ndarray FromIEnumerable(IEnumerable<Object> src, dtype descr,
            bool fortran, int minDepth, int maxDepth)
        {
            ndarray result = null;

 
            if (descr == null)
            {
                descr = FindArrayType(src, null, NpyDefs.NPY_MAXDIMS);
            }

            int itemsize = descr.ElementSize;

            NPY_TYPES type = descr.TypeNum;
            bool checkIt = (descr.Type != NPY_TYPECHAR.NPY_CHARLTR);
            bool stopAtString =
                type != NPY_TYPES.NPY_STRING ||
                descr.Type == NPY_TYPECHAR.NPY_STRINGLTR;
            bool stopAtTuple =
                type == NPY_TYPES.NPY_VOID &&
                (descr.HasNames || descr.HasSubarray);

            int numDim = DiscoverDepth(src, NpyDefs.NPY_MAXDIMS + 1, stopAtString, stopAtTuple);
            if (numDim == 0)
            {
                return FromPythonScalar(src, descr);
            }
            else
            {
                if (maxDepth > 0 && type == NPY_TYPES.NPY_OBJECT &&
                    numDim > maxDepth)
                {
                    numDim = maxDepth;
                }
                if (maxDepth > 0 && numDim > maxDepth ||
                    minDepth > 0 && numDim < minDepth)
                {
                    throw new ArgumentException("Invalid number of dimensions.");
                }

                npy_intp[] dims = new npy_intp[numDim];
                DiscoverDimensions(src, numDim, dims, 0, checkIt);
                if (descr.Type == NPY_TYPECHAR.NPY_CHARLTR &&
                    numDim > 0 && dims[numDim - 1] == 1)
                {
                    numDim--;
                }

                if (itemsize == 0 && NpyDefs.IsExtended(descr.TypeNum))
                {
                    itemsize = DiscoverItemsize(src, numDim, 0);
                    if (descr.TypeNum == NPY_TYPES.NPY_UNICODE)
                    {
                        itemsize *= 4;
                    }
                    descr = new dtype(descr);
                    descr.ElementSize = itemsize;
                }

                result = NpyCoreApi.AllocArray(descr, numDim, dims, fortran);
                AssignToArray(src, result);
            }
            return result;
        }

        internal static ndarray PrependOnes(ndarray arr, int nd, int ndmin)
        {
            npy_intp[] newdims = new npy_intp[ndmin];
            npy_intp[] newstrides = new npy_intp[ndmin];
            int num = ndmin - nd;
            // Set the first num dims and strides for the 1's
            for (int i = 0; i < num; i++)
            {
                newdims[i] = (npy_intp)1;
                newstrides[i] = (npy_intp)arr.Dtype.ElementSize;
            }
            // Copy in the rest of dims and strides
            for (int i = num; i < ndmin; i++)
            {
                int k = i - num;
                newdims[i] = (npy_intp)arr.Dims[k];
                newstrides[i] = (npy_intp)arr.Strides[k];
            }

            return NpyCoreApi.NewView(arr.Dtype, ndmin, newdims, newstrides, arr, 0, false);
        }

        private static dtype FindArrayReturn(dtype chktype, dtype minitype)
        {
            dtype result = NpyCoreApi.SmallType(chktype, minitype);
            if (result.TypeNum == NPY_TYPES.NPY_VOID &&
                minitype.TypeNum != NPY_TYPES.NPY_VOID)
            {
                result = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_OBJECT);
            }
            return result;
        }

        /// <summary>
        /// Given some object and an optional minimum type, returns the appropriate type descriptor.
        /// Equivalent to _array_find_type in common.c of CPython interface.
        /// </summary>
        /// <param name="src">Source object</param>
        /// <param name="minitype">Minimum type, or null if any</param>
        /// <param name="max">Maximum dimensions</param>
        /// <returns>Type descriptor fitting requirements</returns>
        public static dtype FindArrayType(Object src, dtype minitype, int max = NpyDefs.NPY_MAXDIMS)
        {
            dtype chktype = null;

            if (src.GetType().IsArray)
            {
                dynamic arr1 = src;
                if (arr1[0] is ndarray)
                {
                    src = arr1[0];
                }
            }

            if (src is ndarray)
            {
                chktype = ((ndarray)src).Dtype;
                if (minitype == null)
                {
                    return chktype;
                }
                else
                {
                    return FindArrayReturn(chktype, minitype);
                }
            }

            if (src is ScalarGeneric)
            {
                chktype = (dtype)((ScalarGeneric)src).dtype;
                if (minitype == null)
                {
                    return chktype;
                }
                else
                {
                    return FindArrayReturn(chktype, minitype);
                }
            }

            if (minitype == null)
            {
                minitype = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_BOOL);
            }
            if (max < 0)
            {
                chktype = UseDefaultType(src);
                return FindArrayReturn(chktype, minitype);
            }

            chktype = FindScalarType(src);
            if (chktype != null)
            {
                return FindArrayReturn(chktype, minitype);
            }

   
            if (src is String)
            {
                String s = (String)src;
                chktype = new dtype(NpyCoreApi.DescrFromType(NPY_TYPES.NPY_UNICODE));
                chktype.ElementSize = s.Length * 4;
                return FindArrayReturn(chktype, minitype);
            }

            chktype = UseDefaultType(src);
            return FindArrayReturn(chktype, minitype);
        }

        private static dtype UseDefaultType(Object src)
        {
            // TODO: User-defined types are not implemented yet.
            return NpyCoreApi.DescrFromType(NPY_TYPES.NPY_OBJECT);
        }


        /// <summary>
        /// Returns the descriptor for a given native type or null if src is
        /// not a scalar type
        /// </summary>
        /// <param name="src">Object to type</param>
        /// <returns>Descriptor for type of 'src' or null if not scalar</returns>
        internal static dtype FindScalarType(Object src)
        {
            NPY_TYPES type;

            if (src is Double) type = NPY_TYPES.NPY_DOUBLE;
            else if (src is Single) type = NPY_TYPES.NPY_FLOAT;
            else if (src is Boolean) type = NPY_TYPES.NPY_BOOL;
            else if (src is Byte) type = NPY_TYPES.NPY_UBYTE;
            else if (src is SByte) type = NPY_TYPES.NPY_BYTE;
            else if (src is Int16) type = NPY_TYPES.NPY_SHORT;
            else if (src is Int32) type = NpyCoreApi.TypeOf_Int32;
            else if (src is Int64) type = NpyCoreApi.TypeOf_Int64;
            else if (src is UInt16) type = NPY_TYPES.NPY_USHORT;
            else if (src is UInt32) type = NpyCoreApi.TypeOf_UInt32;
            else if (src is UInt64) type = NpyCoreApi.TypeOf_UInt64;
            else if (src is Decimal) type = NpyCoreApi.TypeOf_Decimal;
            else if (src is BigInteger)
            {
                BigInteger bi = (BigInteger)src;
                if (System.Int64.MinValue <= bi && bi <= System.Int64.MaxValue)
                {
                    type = NpyCoreApi.TypeOf_Int64;
                }
                else
                {
                    type = NPY_TYPES.NPY_OBJECT;
                }
            }
            else if (src is Complex) type = NPY_TYPES.NPY_CDOUBLE;
            else type = NPY_TYPES.NPY_NOTYPE;

            return (type != NPY_TYPES.NPY_NOTYPE) ?
                NpyCoreApi.DescrFromType(type) : null;
        }


        /// <summary>
        /// Recursively discovers the nesting depth of a source object.
        /// </summary>
        /// <param name="src">Input object</param>
        /// <param name="max">Max recursive depth</param>
        /// <param name="stopAtString">Stop discovering if string is encounted</param>
        /// <param name="stopAtTuple">Stop discovering if tuple is encounted</param>
        /// <returns>Nesting depth or -1 on error</returns>
        private static int DiscoverDepth(Object src, int max,
            bool stopAtString, bool stopAtTuple)
        {
            int d = 0;

            if (max < 1)
            {
                throw new ArgumentException("invalid input sequence");
            }

            if (stopAtTuple && src is PythonTuple)
            {
                return 0;
            }
            if (src is string || src is IEnumerable<char>)
            {
                return (stopAtString ? 0 : 1);
            }

            if (src is ndarray)
            {
                return ((ndarray)src).ndim;
            }

            if (src is IList<object>)
            {
                IList<object> list = (IList<object>)src;
                if (list.Count == 0)
                {
                    return 1;
                }
                else
                {
                    d = DiscoverDepth(list[0], max - 1, stopAtString, stopAtTuple);
                    return d + 1;
                }
            }

            if (src is IEnumerable<object> && !(src is dtype))
            {
                IEnumerable<object> seq = (IEnumerable<object>)src;
                object first;
                try
                {
                    first = seq.First();
                }
                catch (InvalidOperationException)
                {
                    // Empty sequence
                    return 1;
                }
                d = DiscoverDepth(first, max - 1, stopAtString, stopAtTuple);
                return d + 1;
            }

            // TODO: Not handling __array_struct__ attribute
            // TODO: Not handling __array_interface__ attribute
            return 0;
        }


        /// <summary>
        /// Recursively discovers the size of each dimension given an input object.
        /// </summary>
        /// <param name="src">Input object</param>
        /// <param name="numDim">Number of dimensions</param>
        /// <param name="dims">Uninitialized array of dimension sizes to be filled in</param>
        /// <param name="dimIdx">Current index into dims, incremented recursively</param>
        /// <param name="checkIt">Verify that src is consistent</param>
        private static void DiscoverDimensions(Object src, int numDim,
            npy_intp[] dims, int dimIdx, bool checkIt)
        {
            npy_intp nLowest;

            if (src is ndarray)
            {
                ndarray arr = (ndarray)src;
                if (arr.ndim == 0) dims[dimIdx] = 0;
                else
                {
                    npy_intp[] d = arr.Dims;
                    for (int i = 0; i < numDim; i++)
                    {
                        dims[i + dimIdx] = d[i];
                    }
                }
            }
            else if (src is IList<object>)
            {
                IList<object> seq = (IList<object>)src;

                nLowest = 0;
                dims[dimIdx] = seq.Count();
                if (numDim > 1)
                {
                    foreach (Object o in seq)
                    {
                        DiscoverDimensions(o, numDim - 1, dims, dimIdx + 1, checkIt);
                        if (checkIt && nLowest != 0 && nLowest != dims[dimIdx + 1])
                        {
                            throw new ArgumentException("Inconsistent shape in sequence");
                        }
                        if (dims[dimIdx + 1] > nLowest) nLowest = dims[dimIdx + 1];
                    }
                    dims[dimIdx + 1] = nLowest;
                }
            }
            else if (src is IEnumerable<Object>)
            {
                IEnumerable<Object> seq = (IEnumerable<Object>)src;

                nLowest = 0;
                dims[dimIdx] = seq.Count();
                if (numDim > 1)
                {
                    foreach (Object o in seq)
                    {
                        DiscoverDimensions(o, numDim - 1, dims, dimIdx + 1, checkIt);
                        if (checkIt && nLowest != 0 && nLowest != dims[dimIdx + 1])
                        {
                            throw new ArgumentException("Inconsistent shape in sequence");
                        }
                        if (dims[dimIdx + 1] > nLowest) nLowest = dims[dimIdx + 1];
                    }
                    dims[dimIdx + 1] = nLowest;
                }
            }
            else
            {
                // Scalar condition.
                dims[dimIdx] = 1;
            }
        }

        private static int DiscoverItemsize(object s, int nd, int min)
        {
            if (s is ndarray)
            {
                ndarray a1 = (ndarray)s;
                return Math.Max(min, a1.Dtype.ElementSize);
            }

            throw new Exception("not an ndarray");
  
        }

 
    
        internal static void FillObjects(ndarray arr, object o)
        {
            dtype d = arr.Dtype;
            if (d.IsObject)
            {
                if (d.HasNames)
                {
                    foreach (string name in d.Names)
                    {
                        ndarray view = NpyCoreApi.GetField(arr, name);
                        FillObjects(view, o);
                    }
                }
                else
                {
                    NpyCoreApi.FillWithObject(arr, o);
                }
            }
        }

        internal static void AssignToArray(Object src, ndarray result)
        {
            IEnumerable<object> seq = src as IEnumerable<object>;
            if (seq == null)
            {
                throw new ArgumentException("assignment from non-sequence");
            }
            if (result.ndim == 0)
            {
                throw new ArgumentException("assignment to 0-d array");
            }
            AssignFromSeq(seq, result, 0, 0);
        }

        private static void AssignFromSeq(IEnumerable<Object> seq, ndarray result,
            int dim, long offset)
        {
            if (dim >= result.ndim)
            {
                throw new RuntimeException(String.Format("Source dimensions ({0}) exceeded target array dimensions ({1}).", dim, result.ndim));
            }

            if (seq is ndarray && seq.GetType() != typeof(ndarray))
            {
                // Convert to an array to ensure the dimensionality reduction
                // assumption works.
                ndarray array = NpyCoreApi.FromArray((ndarray)seq, null, NPYARRAYFLAGS.NPY_ENSUREARRAY);
                seq = (IEnumerable<object>)array;
            }

            if (seq.Count() != result.Dims[dim])
            {
                throw new RuntimeException("AssignFromSeq: sequence/array shape mismatch.");
            }

            long stride = result.Stride(dim);
            if (dim < result.ndim - 1)
            {
                // Sequence elements should be additional sequences
                seq.Iteri((o, i) =>
                    AssignFromSeq((IEnumerable<Object>)o, result, dim + 1, offset + stride * i));
            }
            else
            {
                seq.Iteri((o, i) => result.Dtype.f.setitem(offset + i * stride, o, result.Array));
            }
        }

        public static ndarray Concatenate(IEnumerable<ndarray> arrays, int? _axis = 0)
        {
            int i;

            if (!_axis.HasValue)
                _axis = 0;

            int axis = _axis.Value;

            try
            {
                arrays.First();
            }
            catch (InvalidOperationException)
            {
                throw new ArgumentException("concatenation of zero-length sequence is impossible");
            }

            ndarray[] mps = NpyUtil_ArgProcessing.ConvertToCommonType(arrays);
            int n = mps.Length;
            // TODO: Deal with subtypes
            if (axis >= NpyDefs.NPY_MAXDIMS)
            {
                // Flatten the arrays
                for (i = 0; i < n; i++)
                {
                    mps[i] = mps[i].Ravel(NPY_ORDER.NPY_CORDER);
                }
            }
            else if (axis != 0)
            {
                // Swap to make the axis 0
                for (i = 0; i < n; i++)
                {
                    mps[i] = NpyCoreApi.FromArray(mps[i].SwapAxes(axis, 0), null, NPYARRAYFLAGS.NPY_C_CONTIGUOUS);
                }
            }
            npy_intp[] dims = mps[0].Dims;
            if (dims.Length == 0)
            {
                throw new ArgumentException("0-d arrays can't be concatenated");
            }
            npy_intp new_dim = dims[0];
            for (i = 1; i < n; i++)
            {
                npy_intp[] dims2 = mps[i].Dims;
                if (dims.Length != dims2.Length)
                {
                    throw new ArgumentException("arrays must have same number of dimensions");
                }
                bool eq = Enumerable.Zip(dims.Skip(1), dims2.Skip(1), (a1, b) => (a1 == b)).All(x => x);
                if (!eq)
                {
                    throw new ArgumentException("array dimensions do not agree");
                }
                new_dim += dims2[0];
            }
            dims[0] = new_dim;
            ndarray result = NpyCoreApi.AllocArray(mps[0].Dtype, dims.Length, dims, false);
            if (!result.Dtype.IsObject)
            {
                if (NpyCoreApi.CombineInto(result, mps) < 0)
                {
                    throw new ArgumentException("unable to concatenate these arrays");
                }
            }
            else
            {
                // Do a high-level copy to get the references right.
                long j = 0;
                flatiter flat = result.Flat;
                foreach (ndarray a1 in mps)
                {
                    long size = a1.Size;
                    flat[new Slice(j, j + size)] = a1.flat;
                    j += size;
                }
            }
            if (0 < axis && axis < NpyDefs.NPY_MAXDIMS || axis < 0)
            {
                return result.SwapAxes(axis, 0);
            }
            else
            {
                return result;
            }
        }

        public static ndarray inner(object o1, object o2)
        {
            dtype d = FindArrayType(o1, null);
            d = FindArrayType(o2, d);

            ndarray a1 = np.FromAny(o1, d, flags: NPYARRAYFLAGS.NPY_ALIGNED);
            ndarray a2 = np.FromAny(o2, d, flags: NPYARRAYFLAGS.NPY_ALIGNED);
            return NpyCoreApi.InnerProduct(a1, a2, d.TypeNum);
        }

        public static ndarray MatrixProduct(object o1, object o2)
        {
            dtype d = FindArrayType(o1, null);
            d = FindArrayType(o2, d);

            ndarray a1 = np.FromAny(o1, d, flags: NPYARRAYFLAGS.NPY_ALIGNED);
            ndarray a2 = np.FromAny(o2, d, flags: NPYARRAYFLAGS.NPY_ALIGNED);
            if (a1.ndim == 0)
            {
                return EnsureAnyArray((a1.item() as ndarray) * a2);
            }
            else if (a2.ndim == 0)
            {
                return EnsureAnyArray(a1 * (a2.item() as ndarray));
            }
            else
            {
                return NpyCoreApi.MatrixProduct(a1, a2, d.TypeNum);
            }
        }
    }
}

