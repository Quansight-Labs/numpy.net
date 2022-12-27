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

using NumpyLib;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
using npy_ucs4 = System.Int64;
#else
using npy_intp = System.Int32;
using npy_ucs4 = System.Int32;
#endif

namespace NumpyDotNet
{
 
    public static partial class np
    {
        // need this static initializer.  It forces the code below to be initialized in release mode.
        static np()
        {
        }

        private static readonly bool _init = numpy.InitializeNumpyLibrary();


        /// <summary>
        /// Data Type descriptor for bool ndarray
        /// </summary>
        public static readonly dtype Bool = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_BOOL);
        /// <summary>
        /// Data Type descriptor for sbyte ndarray
        /// </summary> 
        public static readonly dtype Int8 = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_BYTE);
        /// <summary>
        /// Data Type descriptor for byte ndarray
        /// </summary>
        public static readonly dtype UInt8 = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_UBYTE);
        /// <summary>
        /// Data Type descriptor for Int16 ndarray
        /// </summary>
        public static readonly dtype Int16 = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_INT16);
        /// <summary>
        /// Data Type descriptor for UInt16 ndarray
        /// </summary>
        public static readonly dtype UInt16 = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_UINT16);
        /// <summary>
        /// Data Type descriptor for Int32 ndarray
        /// </summary>
        public static readonly dtype Int32 = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_INT32);
        /// <summary>
        /// Data Type descriptor for UInt32 ndarray
        /// </summary>
        public static readonly dtype UInt32 = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_UINT32);
        /// <summary>
        /// Data Type descriptor for Int64 ndarray
        /// </summary>
        public static readonly dtype Int64 = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_INT64);
        /// <summary>
        /// Data Type descriptor for UInt64 ndarray
        /// </summary>
        public static readonly dtype UInt64 = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_UINT64);
        /// <summary>
        ///  Data Type descriptor for System.Single/Float ndarray
        /// </summary>
        public static readonly dtype Float32 = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_FLOAT);
        /// <summary>
        ///  Data Type descriptor for System.Double ndarray
        /// </summary>
        public static readonly dtype Float64 = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_DOUBLE);
        /// <summary>
        ///  Data Type descriptor for Decimal ndarray
        /// </summary>
        public static readonly dtype Decimal = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_DECIMAL);
        /// <summary>
        ///  Data Type descriptor for System.Numerics.Complex ndarray
        /// </summary>
        public static readonly dtype Complex = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_COMPLEX);
        /// <summary>
        ///  Data Type descriptor for System.Numerics.BigInteger ndarray
        /// </summary>
        public static readonly dtype BigInt = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_BIGINT);
        /// <summary>
        ///  Data Type descriptor for System.Object ndarray
        /// </summary>
        public static readonly dtype Object = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_OBJECT);
        /// <summary>
        ///  Data Type descriptor for System.String ndarray
        /// </summary>
        public static readonly dtype Strings = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_STRING);

#if NPY_INTP_64
        /// <summary>
        ///  Data Type descriptor for INTP ndarray.  Used for indexing.
        /// </summary>
        public static readonly dtype intp = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_INT64);
#else
        /// <summary>
        /// Data Type descriptor for INTP ndarray.  Used for indexing.
        /// </summary>
        public static readonly dtype intp = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_INT32);
#endif
        /// <summary>
        /// special case
        /// </summary>
        public static readonly dtype None = null;
        /// <summary>
        /// special case
        /// </summary>
        public static readonly object newaxis = null;


        public static readonly bool initialized = true;

        public static bool IsInitialized()
        {
            return initialized;
        }

        #region Assertions

        private static void AssertConvertableToFloating(object a, dtype dtype)
        {
            if (dtype != null)
            {
                switch (dtype.TypeNum)
                {
                    case NPY_TYPES.NPY_OBJECT:
                    case NPY_TYPES.NPY_STRING:
                        throw new Exception(string.Format("This function doesn't support {0} data types", dtype.TypeNum.ToString().Substring(4)));
                }
            }
  
   
        }



        #endregion


        #region array

        /// <summary>
        /// Create ndarray from .net bool array
        /// </summary>
        /// <param name="arr">input bool array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_BOOL</returns>
        public static ndarray array(bool[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_BOOL);
        }
        /// <summary>
        /// Create ndarray from .net byte array
        /// </summary>
        /// <param name="arr">input byte array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_UBYTE</returns>
        public static ndarray array(byte[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_UBYTE);
        }
        /// <summary>
        /// Create ndarray from .net sbyte array
        /// </summary>
        /// <param name="arr">input sbyte array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_BYTE</returns>
        public static ndarray array(sbyte[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_BYTE);
        }
        /// <summary>
        /// Create ndarray from .net Int16 array
        /// </summary>
        /// <param name="arr">input Int16 array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_INT16</returns>
        public static ndarray array(Int16[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_INT16);
        }
        /// <summary>
        /// Create ndarray from .net UInt16 array
        /// </summary>
        /// <param name="arr">input UInt16 array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_UINT16</returns>
        public static ndarray array(UInt16[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_UINT16);
        }
        /// <summary>
        /// Create ndarray from .net Int32 array
        /// </summary>
        /// <param name="arr">input Int32 array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_INT32</returns>
        public static ndarray array(Int32[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_INT32);
        }
        /// <summary>
        /// Create ndarray from .net UInt32 array
        /// </summary>
        /// <param name="arr">input UInt32 array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_UINT32</returns>
        public static ndarray array(UInt32[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_UINT32);
        }
        /// <summary>
        /// Create ndarray from .net Int64 array
        /// </summary>
        /// <param name="arr">input Int64 array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_INT64</returns>
        public static ndarray array(Int64[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_INT64);
        }
        /// <summary>
        /// Create ndarray from .net UInt64 array
        /// </summary>
        /// <param name="arr">input UInt64 array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_UINT64</returns>
        public static ndarray array(UInt64[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_UINT64);
        }
        /// <summary>
        /// Create ndarray from .net float array
        /// </summary>
        /// <param name="arr">input float array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_FLOAT</returns>
        public static ndarray array(float[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_FLOAT);
        }
        /// <summary>
        /// Create ndarray from .net double array
        /// </summary>
        /// <param name="arr">input double array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_DOUBLE</returns>
        public static ndarray array(double[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_DOUBLE);
        }
        /// <summary>
        /// Create ndarray from .net decimal array
        /// </summary>
        /// <param name="arr">input decimal array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_DECIMAL</returns>
        public static ndarray array(decimal[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_DECIMAL);
        }
        /// <summary>
        /// Create ndarray from .net Complex array
        /// </summary>
        /// <param name="arr">input Complex array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_COMPLEX</returns>
        public static ndarray array(System.Numerics.Complex[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_COMPLEX);
        }
        /// <summary>
        /// Create ndarray from .net BigInteger array
        /// </summary>
        /// <param name="arr">input BigInteger array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_BIGINT</returns>
        public static ndarray array(System.Numerics.BigInteger[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_BIGINT);
        }
        /// <summary>
        /// Create ndarray from .net Object array
        /// </summary>
        /// <param name="arr">input Object array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_OBJECT</returns>
        public static ndarray array(System.Object[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_OBJECT);
        }
        /// <summary>
        /// Create ndarray from .net String array
        /// </summary>
        /// <param name="arr">input String array</param>
        /// <returns>ndarray of type NPY_TYPES.NPY_STRING</returns>
        public static ndarray array(System.String[] arr)
        {
            return _array(arr, NPY_TYPES.NPY_STRING);
        }

        private static ndarray _array<T>(T[] arr, NPY_TYPES arrayType)
        {
            dtype dtype = null; bool copy = true; NPY_ORDER order = NPY_ORDER.NPY_ANYORDER; bool subok = false; int ndmin = 0;

            var arrayDescr = numpyAPI.NpyArray_DescrFromType(arrayType);
            if (arrayDescr == null)
            {
                return null;
            }

            VoidPtr data = GetDataPointer(arr, arrayType, copy);
            NPYARRAYFLAGS flags = NPYARRAYFLAGS.NPY_DEFAULT;

            bool ensureArray = true;
            object subtype = null;
            object interfacedata = null;

            int nd = 1;
            npy_intp[] dims = new npy_intp[] { arr.Length };

            var NpyArray = numpyAPI.NpyArray_NewFromDescr(arrayDescr, nd, dims, null, data, flags, ensureArray, subtype, interfacedata);
            if (NpyArray != null)
            {
                ndarray ndArray = new ndarray(NpyArray);
                return array(ndArray, dtype, false, order, subok, ndmin);
            }
            return null;
        }

        /// <summary>
        /// Create ndarray from specified buffer, converting to the specified data type if necessary
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="arr">buffer to create ndarray from</param>
        /// <param name="dtype">data type of created ndarray</param>
        /// <param name="copy">true to copy arr buffer, false to share the arr buffer</param>
        /// <param name="order">allows for specifying the ordering of bytes (fortran/C/K),</param>
        /// <param name="subok">(optional) if true, subclasses will be passed thru</param>
        /// <param name="ndmin">(optional) specifies number of dimensions resulting array should have</param>
        /// <returns></returns>
        public static ndarray array<T>(T[] arr, dtype dtype = null, bool copy = true, NPY_ORDER order = NPY_ORDER.NPY_ANYORDER, bool subok = false, int ndmin = 0)
        {
            NPY_TYPES arrayType = DetermineArrayType(arr, dtype);

            if (arrayType != NPY_TYPES.NPY_OBJECT && arrayType != NPY_TYPES.NPY_STRING)
            {
                if (Get_NPYType(arr) != arrayType)
                {
                    throw new Exception("Mismatch data types between input array and dtype");
                }
            }


            var arrayDescr = numpyAPI.NpyArray_DescrFromType(arrayType);
            if (arrayDescr == null)
            {
                return null;
            }

            VoidPtr data = GetDataPointer(arr, arrayType, copy);
            NPYARRAYFLAGS flags = NPYARRAYFLAGS.NPY_DEFAULT;

            bool ensureArray = true;
            object subtype = null;
            object interfacedata = null;

            int nd = 1;
            npy_intp[] dims = new npy_intp[] { arr.Length };

            var NpyArray = numpyAPI.NpyArray_NewFromDescr(arrayDescr, nd, dims, null, data, flags, ensureArray, subtype, interfacedata);
            if (NpyArray != null)
            {
                ndarray ndArray = new ndarray(NpyArray);
                return array(ndArray, dtype, false, order, subok, ndmin);
            }
            return null;
        }

        /// <summary>
        /// Create ndarray from specified buffer, converting to the specified data type if necessary
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="arr">buffer to create ndarray from</param>
        /// <param name="dtype">data type of created ndarray</param>
        /// <param name="copy">true to copy arr buffer, false to share the arr buffer</param>
        /// <param name="order">allows for specifying the ordering of bytes (fortran/C/K),</param>
        /// <param name="subok">(optional) if true, subclasses will be passed thru</param>
        /// <param name="ndmin">(optional) specifies number of dimensions resulting array should have</param>
        /// <returns></returns>
        public static ndarray array(VoidPtr arr, dtype dtype = null, bool copy = true, NPY_ORDER order = NPY_ORDER.NPY_ANYORDER, bool subok = false, int ndmin = 0)
        {
            switch (arr.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return array(arr.datap as bool[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_BYTE:
                    return array(arr.datap as sbyte[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_UBYTE:
                    return array(arr.datap as byte[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_INT16:
                    return array(arr.datap as Int16[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_UINT16:
                    return array(arr.datap as UInt16[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_INT32:
                    return array(arr.datap as Int32[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_UINT32:
                    return array(arr.datap as UInt32[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_INT64:
                    return array(arr.datap as Int64[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_UINT64:
                    return array(arr.datap as UInt64[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_FLOAT:
                    return array(arr.datap as float[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_DOUBLE:
                    return array(arr.datap as double[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_DECIMAL:
                    return array(arr.datap as decimal[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_COMPLEX:
                    return array(arr.datap as System.Numerics.Complex[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_BIGINT:
                    return array(arr.datap as System.Numerics.BigInteger[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_OBJECT:
                    return array(arr.datap as object[], dtype, copy, order, subok, ndmin);
                case NPY_TYPES.NPY_STRING:
                    return array(arr.datap as string[], dtype, copy, order, subok, ndmin);
            }

            throw new Exception("unrecognized array type");
        }

        private static ndarray array<T>(VoidPtr arr, int numElements, int elsize) 
        {
            T[] data = new T[numElements];
            Array.Copy(arr.datap as T[], arr.data_offset / elsize, data, 0, numElements);
            return array(data);
        }
        /// <summary>
        /// create array from passed data but specifying number of elements
        /// </summary>
        /// <param name="arr">array of data to create array from</param>
        /// <param name="numElements">number of elements from arr to create ndarray from</param>
        /// <returns></returns>
        public static ndarray array(VoidPtr arr, int numElements)
        {
            var ArrayHandler = DefaultArrayHandlers.GetArrayHandler(arr.type_num);
            int startingOffset = (int)(arr.data_offset >> ArrayHandler.ItemDiv);
            return array(ArrayHandler.AllocateAndCopy(arr.datap, startingOffset, numElements));
        }
        /// <summary>
        /// Create ndarray from specified buffer, converting to the specified data type if necessary
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="arr">buffer to create ndarray from</param>
        /// <param name="dtype">data type of created ndarray</param>
        /// <param name="copy">true to copy arr buffer, false to share the arr buffer</param>
        /// <param name="order">allows for specifying the ordering of bytes (fortran/C/K),</param>
        /// <param name="subok">(optional) if true, subclasses will be passed thru</param>
        /// <param name="ndmin">(optional) specifies number of dimensions resulting array should have</param>
        /// <returns></returns>
        public static ndarray array(Object src, dtype dtype = null, bool copy = true, NPY_ORDER order = NPY_ORDER.NPY_ANYORDER, bool subok = false, int ndmin = 0)
        {

            ndarray result = null;
            if (ndmin >= NpyDefs.NPY_MAXDIMS)
            {
                throw new ArgumentException(String.Format("ndmin ({0}) bigger than allowable number of dimension ({1}).",ndmin, NpyDefs.NPY_MAXDIMS - 1));
            }

            if (subok && src is ndarray || !subok && src != null && src.GetType() == typeof(ndarray))
            {
                ndarray arr = (ndarray)src;
                if (dtype == null)
                {
                    if (!copy && StridingOk(arr,order))
                    {
                        result = arr;
                    }
                    else
                    {
                        result = NpyCoreApi.NewCopy(arr, order);
                    }
                }
                else
                {
                    dtype oldtype = arr.Dtype;
                    if (NpyCoreApi.EquivTypes(oldtype, dtype))
                    {
                        if (!copy && StridingOk(arr,order))
                        {
                            result = arr;
                        }
                        else
                        {
                            result = NpyCoreApi.NewCopy(arr, order);
                            if (oldtype != dtype)
                            {
                                result.Dtype = oldtype;
                            }
                        }
                    }
                }
            }

            // If no result has been determined...
            if (result == null)
            {
                NPYARRAYFLAGS flags = 0;

                if (copy) flags = NPYARRAYFLAGS.NPY_ENSURECOPY;
                if (order == NPY_ORDER.NPY_CORDER)
                {
                    flags |= NPYARRAYFLAGS.NPY_CONTIGUOUS;
                }
                else if (order == NPY_ORDER.NPY_FORTRANORDER || src is ndarray && ((ndarray)src).IsFortran)
                {
                    flags |= NPYARRAYFLAGS.NPY_FORTRAN;
                }

                if (!subok) flags |= NPYARRAYFLAGS.NPY_ENSUREARRAY;

                flags |= NPYARRAYFLAGS.NPY_FORCECAST;
                result = np.CheckFromAny(src, dtype, 0, 0, flags, null);
            }

            if (result != null && result.ndim < ndmin)
            {
                result = np.PrependOnes(result, result.ndim, ndmin);
            }
            return result;
        }


        public static bool StridingOk(ndarray arr, NPY_ORDER order)
        {
            return order == NPY_ORDER.NPY_ANYORDER ||
                order == NPY_ORDER.NPY_CORDER && arr.IsContiguous ||
                order == NPY_ORDER.NPY_FORTRANORDER && arr.IsFortran;
        }

        /// <summary>
        /// creates zero filled array with the specified shape and data type
        /// </summary>
        /// <param name="shape">shape of array to create</param>
        /// <param name="dtype">data type of array to create</param>
        /// <returns></returns>
        public static ndarray ndarray(shape shape, dtype dtype)
        {
            return zeros(shape, dtype);
        }

        //public static ndarray array(object input, object where)
        //{
        //    ndarray arr = null;
        //    ndarray wherearr = null;

        //    try
        //    {
        //        arr = asanyarray(input);
        //    }
        //    catch (Exception ex)
        //    {
        //        throw new ValueError("Unable to convert input into an ndarray.");
        //    }

        //    if (where != null)
        //    {
        //        try
        //        {
        //            wherearr = asanyarray(where);
        //        }
        //        catch (Exception ex)
        //        {
        //            throw new ValueError("Unable to convert 'where' into an ndarray.");
        //        }

        //        try
        //        {
        //            arr = arr.A(wherearr);
        //        }
        //        catch (Exception ex)
        //        {
        //            throw new ValueError("input[where] does not result in a valid ndarray.");
        //        }

        //    }


        //    return arr;
        //}
        #endregion

        #region arange
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval. The interval includes this value. The default start value is 0.</param>
        /// <param name="stop">End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.</param>
        /// <param name="step">Spacing between values. For any output out, this is the distance between two adjacent values, out[i+1] - out[i]. The default step size is 1. If step is specified, start must also be given.</param>
        /// <param name="dtype">The type of the output array. If dtype is not given, infer the data type from the other input arguments.</param>
        /// <returns>Array of evenly spaced values</returns>
        /// <notes>For floating point arguments, the length of the result is ceil((stop - start)/step). Because of floating point overflow, this rule may result in the last element of out being greater than stop.</notes>
        /// <notes>When using a non-integer step, such as 0.1, the results will often not be consistent. It is better to use linspace for these cases.</notes>
        public static ndarray arange(Int64 start, Int64? stop = null, Int64? step = null, dtype dtype = null)
        {
            npy_intp[] dims;

            ValidateArangeDtype(dtype);

            // determine what data type it should be if not set,
            if (dtype == null)
            {
                Int64 stop2 = stop != null ? (Int64)stop : 0;
                if (start <= System.Int32.MaxValue && stop2 <= System.Int32.MaxValue)
                {
                    dtype = np.Int32;
                }
                else
                {
                    dtype = np.Int64;
                }

            }

            if (dtype == null)
            {
                dtype = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_INT64);
                dtype = FindArrayType(start, dtype);
                if (stop != null)
                {
                    dtype = FindArrayType(stop, dtype);
                }
                if (step != null)
                {
                    dtype = FindArrayType(step, dtype);
                }

            }
            if (step == null)
            {
                step = 1;
            }
            if (stop == null)
            {
                stop = start;
                start = 0;
            }

            npy_intp len = 0;
            try
            {
                npy_intp ArrayLen = (npy_intp)(stop - start);

                if ((ArrayLen % step) > 0)
                {
                    ArrayLen += (npy_intp)(ArrayLen % step);
                }
                ArrayLen = (npy_intp)(ArrayLen / step);

                len = ArrayLen;
            }
            catch (OverflowException)
            {
                // Translate the error to make test_regression.py happy.
                throw new ArgumentException("step can't be 0");
            }

            if (len < 0)
            {
                dims = new npy_intp[] { 0 };
                return NpyCoreApi.NewFromDescr(dtype, dims, null, 0, null);
            }

            dtype native;
            bool swap;
            if (!dtype.IsNativeByteOrder)
            {
                native = NpyCoreApi.DescrNewByteorder(dtype, '=');
                swap = true;
            }
            else
            {
                native = dtype;
                swap = false;
            }

            dims = new npy_intp[] { len };
            ndarray result = NpyCoreApi.NewFromDescr(native, dims, null, 0, null);

            var CC = BuildFastArrayAccessDataByType(result.Array.data);
            var SetItemFunc = FastSetItemFuncByType(CC, result.Array.data);

            // populate the array
            Int64 _step = (Int64)step;
            Int64 _start = (Int64)start;
            Parallel.For(0, len, i =>
            {
                double value = _start + (i * _step);
                SetItemFunc(i, value);
            });


            if (swap)
            {
                NpyCoreApi.Byteswap(result, true);
                result.Dtype = dtype;
            }
            return result;
        }

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval. The interval includes this value. The default start value is 0.</param>
        /// <param name="stop">End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.</param>
        /// <param name="step">Spacing between values. For any output out, this is the distance between two adjacent values, out[i+1] - out[i]. The default step size is 1. If step is specified, start must also be given.</param>
        /// <param name="dtype">The type of the output array. If dtype is not given, infer the data type from the other input arguments.</param>
        /// <returns>Array of evenly spaced values</returns>
        /// <notes>For floating point arguments, the length of the result is ceil((stop - start)/step). Because of floating point overflow, this rule may result in the last element of out being greater than stop.</notes>
        /// <notes>When using a non-integer step, such as 0.1, the results will often not be consistent. It is better to use linspace for these cases.</notes>
        public static ndarray arange(double start, double? stop = null, double? step = null, dtype dtype = null)
        {
            npy_intp[] dims;

            ValidateArangeDtype(dtype);

            if (stop == null)
            {
                stop = start;
                start = 0;
            }

            // determine what data type it should be if not set,
            if (dtype == null)
            {
                if ((float)start <= System.Single.MaxValue && (float)stop <= System.Single.MaxValue)
                {
                    dtype = np.Float32;
                }
                else
                {
                    dtype = np.Float64;
                }

            }

            if (dtype == null)
            {
                dtype = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_DOUBLE);
                dtype = FindArrayType(start, dtype);
                if (stop != null)
                {
                    dtype = FindArrayType(stop, dtype);
                }
                if (step != null)
                {
                    dtype = FindArrayType(step, dtype);
                }

            }
            if (step == null)
            {
                step = 1;
            }
            if (stop == null)
            {
                stop = start;
                start = 0;
            }

            double len = 0;
            try
            {
                double ArrayLen = (double)(stop - start);

                if ((ArrayLen % step) > 0)
                {
                    ArrayLen += (double)(ArrayLen % step);
                }
                ArrayLen = (double)(ArrayLen / step);

                len = ArrayLen;
            }
            catch (OverflowException)
            {
                // Translate the error to make test_regression.py happy.
                throw new ArgumentException("step can't be 0");
            }

            if (len < 0)
            {
                dims = new npy_intp[] { 0 };
                return NpyCoreApi.NewFromDescr(dtype, dims, null, 0, null);
            }

            dtype native;
            bool swap;
            if (!dtype.IsNativeByteOrder)
            {
                native = NpyCoreApi.DescrNewByteorder(dtype, '=');
                swap = true;
            }
            else
            {
                native = dtype;
                swap = false;
            }

            dims = new npy_intp[] { (npy_intp)len };
            ndarray result = NpyCoreApi.NewFromDescr(native, dims, null, 0, null);

            var CC = BuildFastArrayAccessDataByType(result.Array.data);
            var SetItemFunc = FastSetItemFuncByType(CC, result.Array.data);

            // populate the array
            double _step = (double)step;
            double _start = (double)start;
            Parallel.For(0, (npy_intp)len, i =>
            {
                double value = _start + (i * _step);
                SetItemFunc(i, value);
            });


            if (swap)
            {
                NpyCoreApi.Byteswap(result, true);
                result.Dtype = dtype;
            }
            return result;
        }

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval. The interval includes this value. The default start value is 0.</param>
        /// <param name="stop">End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.</param>
        /// <param name="step">Spacing between values. For any output out, this is the distance between two adjacent values, out[i+1] - out[i]. The default step size is 1. If step is specified, start must also be given.</param>
        /// <param name="dtype">The type of the output array. If dtype is not given, infer the data type from the other input arguments.</param>
        /// <returns>Array of evenly spaced values</returns>
        /// <notes>For floating point arguments, the length of the result is ceil((stop - start)/step). Because of floating point overflow, this rule may result in the last element of out being greater than stop.</notes>
        /// <notes>When using a non-integer step, such as 0.1, the results will often not be consistent. It is better to use linspace for these cases.</notes>
        public static ndarray arange(decimal start, decimal? stop = null, decimal? step = null, dtype dtype = null)
        {
            npy_intp[] dims;

            ValidateArangeDtype(dtype);

            if (stop == null)
            {
                stop = start;
                start = 0m;
            }

            // determine what data type it should be if not set,
            if (dtype == null)
            {
                dtype = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_DECIMAL);
            }

            if (dtype.TypeNum != NPY_TYPES.NPY_DECIMAL)
            {
                throw new ArgumentException("Array type must be Decimal");
            }

            if (step == null)
            {
                step = 1m;
            }
            if (stop == null)
            {
                stop = start;
                start = 0m;
            }

            decimal len = 0m;
            try
            {
                decimal ArrayLen = (decimal)(stop - start);

                if ((ArrayLen % step) > 0)
                {
                    ArrayLen += (decimal)(ArrayLen % step);
                }
                ArrayLen = (decimal)(ArrayLen / step);

                len = ArrayLen;
            }
            catch (OverflowException)
            {
                // Translate the error to make test_regression.py happy.
                throw new ArgumentException("step can't be 0");
            }

            if (len < 0)
            {
                dims = new npy_intp[] { 0 };
                return NpyCoreApi.NewFromDescr(dtype, dims, null, 0, null);
            }

            dtype native;
            bool swap;
            if (!dtype.IsNativeByteOrder)
            {
                native = NpyCoreApi.DescrNewByteorder(dtype, '=');
                swap = true;
            }
            else
            {
                native = dtype;
                swap = false;
            }

            dims = new npy_intp[] { (npy_intp)len };
            ndarray result = NpyCoreApi.NewFromDescr(native, dims, null, 0, null);

            var CC = BuildFastArrayAccessDataByType(result.Array.data);
            var SetItemFunc = FastSetItemFuncByType(CC, result.Array.data);

            // populate the array
            decimal _step = (decimal)step;
            decimal _start = (decimal)start;
            Parallel.For(0, (npy_intp)len, i =>
            {
                decimal value = _start + (i * _step);
                SetItemFunc(i, value);
            });


            if (swap)
            {
                NpyCoreApi.Byteswap(result, true);
                result.Dtype = dtype;
            }
            return result;
        }

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval. The interval includes this value. The default start value is 0.</param>
        /// <param name="stop">End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.</param>
        /// <param name="step">Spacing between values. For any output out, this is the distance between two adjacent values, out[i+1] - out[i]. The default step size is 1. If step is specified, start must also be given.</param>
        /// <param name="dtype">The type of the output array. If dtype is not given, infer the data type from the other input arguments.</param>
        /// <returns>Array of evenly spaced values</returns>
        /// <notes>For floating point arguments, the length of the result is ceil((stop - start)/step). Because of floating point overflow, this rule may result in the last element of out being greater than stop.</notes>
        /// <notes>When using a non-integer step, such as 0.1, the results will often not be consistent. It is better to use linspace for these cases.</notes>
        public static ndarray arange(System.Numerics.Complex start, System.Numerics.Complex? stop = null, System.Numerics.Complex? step = null, dtype dtype = null)
        {
            npy_intp[] dims;

            ValidateArangeDtype(dtype);

            if (stop == null)
            {
                stop = start;
                start = 0;
            }

            // determine what data type it should be if not set,
            if (dtype == null)
            {
                dtype = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_COMPLEX);
            }

            if (dtype.TypeNum != NPY_TYPES.NPY_COMPLEX)
            {
                throw new ArgumentException("Array type must be Complex");
            }

            if (step == null)
            {
                step = 1;
            }
            if (stop == null)
            {
                stop = start;
                start = 0;
            }

            npy_intp len = 0;
            try
            {
                npy_intp ArrayLen = (npy_intp)(stop.Value.Real - start.Real);

                if ((ArrayLen % step.Value.Real) > 0)
                {
                    ArrayLen += (npy_intp)(ArrayLen % step.Value.Real);
                }
                ArrayLen = (npy_intp)(ArrayLen / step.Value.Real);

                len = ArrayLen;
            }
            catch (OverflowException)
            {
                // Translate the error to make test_regression.py happy.
                throw new ArgumentException("step can't be 0");
            }

            if (len < 0)
            {
                dims = new npy_intp[] { 0 };
                return NpyCoreApi.NewFromDescr(dtype, dims, null, 0, null);
            }

            dtype native;
            bool swap;
            if (!dtype.IsNativeByteOrder)
            {
                native = NpyCoreApi.DescrNewByteorder(dtype, '=');
                swap = true;
            }
            else
            {
                native = dtype;
                swap = false;
            }

            dims = new npy_intp[] { (npy_intp)len };
            ndarray result = NpyCoreApi.NewFromDescr(native, dims, null, 0, null);

            var CC = BuildFastArrayAccessDataByType(result.Array.data);
            var SetItemFunc = FastSetItemFuncByType(CC, result.Array.data);

            // populate the array
            System.Numerics.Complex _step = step.Value;
            System.Numerics.Complex _start = start;
            Parallel.For(0, (npy_intp)len, i =>
            {
                System.Numerics.Complex value = _start + (i * _step);
                SetItemFunc(i, value);
            });


            if (swap)
            {
                NpyCoreApi.Byteswap(result, true);
                result.Dtype = dtype;
            }
            return result;
        }

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval. The interval includes this value. The default start value is 0.</param>
        /// <param name="stop">End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.</param>
        /// <param name="step">Spacing between values. For any output out, this is the distance between two adjacent values, out[i+1] - out[i]. The default step size is 1. If step is specified, start must also be given.</param>
        /// <param name="dtype">The type of the output array. If dtype is not given, infer the data type from the other input arguments.</param>
        /// <returns>Array of evenly spaced values</returns>
        /// <notes>For floating point arguments, the length of the result is ceil((stop - start)/step). Because of floating point overflow, this rule may result in the last element of out being greater than stop.</notes>
        /// <notes>When using a non-integer step, such as 0.1, the results will often not be consistent. It is better to use linspace for these cases.</notes>
        public static ndarray arange(System.Numerics.BigInteger start, System.Numerics.BigInteger? stop = null, System.Numerics.BigInteger? step = null, dtype dtype = null)
        {
            npy_intp[] dims;

            ValidateArangeDtype(dtype);

            if (stop == null)
            {
                stop = start;
                start = 0;
            }

            // determine what data type it should be if not set,
            if (dtype == null)
            {
                dtype = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_BIGINT);
            }

            if (dtype.TypeNum != NPY_TYPES.NPY_BIGINT)
            {
                throw new ArgumentException("Array type must be BigInt");
            }

            if (step == null)
            {
                step = 1;
            }
            if (stop == null)
            {
                stop = start;
                start = 0;
            }

            npy_intp len = 0;
            try
            {
                npy_intp ArrayLen = (npy_intp)(stop.Value - start);

                if ((ArrayLen % step.Value) > 0)
                {
                    ArrayLen += (npy_intp)(ArrayLen % step.Value);
                }
                ArrayLen = (npy_intp)(ArrayLen / step.Value);

                len = ArrayLen;
            }
            catch (OverflowException)
            {
                // Translate the error to make test_regression.py happy.
                throw new ArgumentException("step can't be 0");
            }

            if (len < 0)
            {
                dims = new npy_intp[] { 0 };
                return NpyCoreApi.NewFromDescr(dtype, dims, null, 0, null);
            }

            dtype native;
            bool swap;
            if (!dtype.IsNativeByteOrder)
            {
                native = NpyCoreApi.DescrNewByteorder(dtype, '=');
                swap = true;
            }
            else
            {
                native = dtype;
                swap = false;
            }

            dims = new npy_intp[] { (npy_intp)len };
            ndarray result = NpyCoreApi.NewFromDescr(native, dims, null, 0, null);

            var CC = BuildFastArrayAccessDataByType(result.Array.data);
            var SetItemFunc = FastSetItemFuncByType(CC, result.Array.data);

            // populate the array
            System.Numerics.BigInteger _step = step.Value;
            System.Numerics.BigInteger _start = start;
            Parallel.For(0, (npy_intp)len, i =>
            {
                System.Numerics.BigInteger value = _start + (i * _step);
                SetItemFunc(i, value);
            });


            if (swap)
            {
                NpyCoreApi.Byteswap(result, true);
                result.Dtype = dtype;
            }
            return result;
        }

        private static void ValidateArangeDtype(dtype dtype)
        {
            if (dtype != null)
            {
                switch (dtype.TypeNum)
                {
                    case NPY_TYPES.NPY_OBJECT:
                    case NPY_TYPES.NPY_STRING:
                        throw new Exception(string.Format("This function doesn't support {0} data types", dtype.TypeNum.ToString().Substring(4)));
                }
            }
  
        }


        #endregion

        #region FAST ARRAY ACCESS
        private class FastArrayAccessData
        {
            public bool[] BoolArray;
            public byte[] ByteArray;
            public sbyte[] SByteArray;
            public Int16[] Int16Array;
            public UInt16[] UInt16Array;
            public Int32[] Int32Array;
            public UInt32[] UInt32Array;
            public Int64[] Int64Array;
            public UInt64[] UInt64Array;
            public float[] FloatArray;
            public double[] DoubleArray;
            public decimal[] DecimalArray;
            public System.Numerics.Complex[] ComplexArray;
            public System.Numerics.BigInteger[] BigIntArray;

        }


        private static FastArrayAccessData BuildFastArrayAccessDataByType(VoidPtr vp)
        {
            FastArrayAccessData FAData = new FastArrayAccessData();

            switch (vp.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    FAData.BoolArray = vp.datap as bool[];
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    FAData.ByteArray = vp.datap as byte[];
                    break;
                case NPY_TYPES.NPY_BYTE:
                    FAData.SByteArray = vp.datap as sbyte[];
                    break;
                case NPY_TYPES.NPY_UINT16:
                    FAData.UInt16Array = vp.datap as UInt16[];
                    break;
                case NPY_TYPES.NPY_INT16:
                    FAData.Int16Array = vp.datap as Int16[];
                    break;
                case NPY_TYPES.NPY_UINT32:
                    FAData.UInt32Array = vp.datap as UInt32[];
                    break;
                case NPY_TYPES.NPY_INT32:
                    FAData.Int32Array = vp.datap as Int32[];
                    break;
                case NPY_TYPES.NPY_INT64:
                    FAData.Int64Array = vp.datap as Int64[];
                    break;
                case NPY_TYPES.NPY_UINT64:
                    FAData.UInt64Array = vp.datap as UInt64[];
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    FAData.FloatArray = vp.datap as float[];
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    FAData.DoubleArray = vp.datap as double[];
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    FAData.DecimalArray = vp.datap as decimal[];
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    FAData.ComplexArray = vp.datap as System.Numerics.Complex[];
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    FAData.BigIntArray = vp.datap as System.Numerics.BigInteger[];
                    break;
                default:
                    throw new Exception("Unsupported data type");
            }

            return FAData;
        }

        private static Func<npy_intp, object, int> FastSetItemFuncByType(FastArrayAccessData FAData, VoidPtr vp)
        {
            Func<npy_intp, object, int> ret = null;

            switch (vp.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    ret = (index, value) => { FAData.BoolArray[index] = Convert.ToBoolean(value); return 0; };
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    ret = (index, value) => { FAData.ByteArray[index] = Convert.ToByte(value); return 0; };
                    break;
                case NPY_TYPES.NPY_BYTE:
                    ret = (index, value) => { FAData.SByteArray[index] = Convert.ToSByte(value); return 0; };
                    break;
                case NPY_TYPES.NPY_UINT16:
                    ret = (index, value) => { FAData.UInt16Array[index] = Convert.ToUInt16(value); return 0; };
                    break;
                case NPY_TYPES.NPY_INT16:
                    ret = (index, value) => { FAData.Int16Array[index] = Convert.ToInt16(value); return 0; };
                    break;
                case NPY_TYPES.NPY_UINT32:
                    ret = (index, value) => { FAData.UInt32Array[index] = Convert.ToUInt32(value); return 0; };
                    break;
                case NPY_TYPES.NPY_INT32:
                    ret = (index, value) => { FAData.Int32Array[index] = Convert.ToInt32(value); return 0; };
                    break;
                case NPY_TYPES.NPY_INT64:
                    ret = (index, value) => { FAData.Int64Array[index] = Convert.ToInt64(value); return 0; };
                    break;
                case NPY_TYPES.NPY_UINT64:
                    ret = (index, value) => { FAData.UInt64Array[index] = Convert.ToUInt64(value); return 0; };
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    ret = (index, value) => { FAData.FloatArray[index] = Convert.ToSingle(value); return 0; };
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    ret = (index, value) => { FAData.DoubleArray[index] = Convert.ToDouble(value); return 0; };
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    ret = (index, value) => { FAData.DecimalArray[index] = Convert.ToDecimal(value); return 0; };
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    ret = (index, value) => 
                    {
                        if (value is System.Numerics.Complex)
                            FAData.ComplexArray[index] = (System.Numerics.Complex)value;
                        else
                            FAData.ComplexArray[index] = new System.Numerics.Complex(Convert.ToDouble(value), 0);
                        return 0;
                    };
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    ret = (index, value) =>
                    {
                        if (value is System.Numerics.BigInteger)
                            FAData.BigIntArray[index] = (System.Numerics.BigInteger)value;
                        else
                            FAData.BigIntArray[index] = new System.Numerics.BigInteger(Convert.ToDouble(value));
                        return 0;
                    };
                    break;
                default:
                    throw new Exception("Unsupported data type");
            }


            return ret;
        }
        #endregion

        #region linspace

        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// </summary>
        /// <param name="start">The starting value of the sequence</param>
        /// <param name="stop">The end value of the sequence, unless `endpoint` is set to False.</param>
        /// <param name="retstep">return (`samples`, `step`), where `step` is the spacing between samples.</param>
        /// <param name="num">Number of samples to generate.Default is 50.Must be non-negative.</param>
        /// <param name="endpoint">If True, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <param name="dtype">The type of the output array.If `dtype` is not given, infer the data type from the other input arguments.</param>
        /// <returns></returns>
        public static ndarray linspace(Int64 start, Int64 stop, ref double retstep, int num = 50, bool endpoint = true,  dtype dtype = null)
        {
            return linspace(Convert.ToDouble(start), Convert.ToDouble(stop), ref retstep, num, endpoint,  dtype);
        }
        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// </summary>
        /// <param name="start">The starting value of the sequence</param>
        /// <param name="stop">The end value of the sequence, unless `endpoint` is set to False.</param>
        /// <param name="retstep">return (`samples`, `step`), where `step` is the spacing between samples.</param>
        /// <param name="num">Number of samples to generate.Default is 50.Must be non-negative.</param>
        /// <param name="endpoint">If True, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <param name="dtype">The type of the output array.If `dtype` is not given, infer the data type from the other input arguments.</param>
        /// <returns></returns>
        public static ndarray linspace(double start, double stop, ref double retstep, int num = 50, bool endpoint = true,  dtype dtype = null)
        {
            //  Return evenly spaced numbers over a specified interval.

            //  Returns `num` evenly spaced samples, calculated over the
            //  interval[`start`, `stop`].

            //  The endpoint of the interval can optionally be excluded.

            //  Parameters
            //  ----------
            //  start: scalar
            //     The starting value of the sequence.
            // stop : scalar
            //     The end value of the sequence, unless `endpoint` is set to False.
            //      In that case, the sequence consists of all but the last of ``num + 1``
            //      evenly spaced samples, so that `stop` is excluded.Note that the step
            //      size changes when `endpoint` is False.
            //  num : int, optional
            //      Number of samples to generate.Default is 50.Must be non-negative.
            //endpoint : bool, optional
            //      If True, `stop` is the last sample.Otherwise, it is not included.
            //      Default is True.
            //  retstep : bool, optional
            //      If True, return (`samples`, `step`), where `step` is the spacing
            //      between samples.
            //  dtype: dtype, optional
            //     The type of the output array.If `dtype` is not given, infer the data
            //   type from the other input arguments.


            //   ..versionadded:: 1.9.0

            //  Returns
            //  ------ -
            //  samples : ndarray
            //      There are `num` equally spaced samples in the closed interval
            //      ``[start, stop]`` or the half-open interval ``[start, stop)``
            //      (depending on whether `endpoint` is True or False).
            //  step : float, optional
            //      Only returned if `retstep` is True

            //      Size of spacing between samples.


            //  See Also
            //  --------
            //  arange : Similar to `linspace`, but uses a step size (instead of the
            //           number of samples).
            //  logspace : Samples uniformly distributed in log space.

            //  Examples
            //  --------
            //  >>> np.linspace(2.0, 3.0, num= 5)
            //  array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
            //  >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
            //  array([ 2. ,  2.2,  2.4,  2.6,  2.8])
            //  >>> np.linspace(2.0, 3.0, num=5, retstep=True)
            //  (array([2.  , 2.25, 2.5, 2.75, 3.  ]), 0.25)

            //  Graphical illustration:

            //  >>> import matplotlib.pyplot as plt
            //  >>> N = 8
            //  >>> y = np.zeros(N)
            //  >>> x1 = np.linspace(0, 10, N, endpoint = True)
            //  >>> x2 = np.linspace(0, 10, N, endpoint = False)
            //  >>> plt.plot(x1, y, 'o')
            //  [< matplotlib.lines.Line2D object at 0x...>]
            //  >>> plt.plot(x2, y + 0.5, 'o')
            //  [< matplotlib.lines.Line2D object at 0x...>]
            //  >>> plt.ylim([-0.5, 1])
            //  (-0.5, 1)
            //  >>> plt.show()

            ValidateArangeDtype(dtype);


            if (num < 0)
            {
                throw new ValueError(string.Format("Number of samples, {0}, must be non-negative.", num));
            }

            int div = 0;

            if (endpoint)
            {
                div = num - 1;
            }
            else
            {
                div = num;
            }

            // Convert float/complex array scalars to float, gh-3504
            // and make sure one can use variables that have an __array_interface__, gh-6634

            dtype dt = np.Float64; // result_type(start, stop, Convert.ToSingle(num));
            if (dtype == null)
                dtype = dt;

            var y = np.arange(0, num, dtype : dt);

            double delta = stop - start;
            // In-place multiplication y *= delta/div is faster, but prevents the multiplicant
            // from overriding what class is produced, and thus prevents, e.g. use of Quantities,
            // see gh-7142. Hence, we multiply in place only for standard scalar types.

            bool _mult_inplace = np.isscalar(delta);
            double step = 0;
            if (num > 1)
            {
                step = delta / div;
                if (step == 0)
                {
                    // Special handling for denormal numbers, gh-5437
                    y /= div;
                    if (_mult_inplace)
                    {
                        y *= delta;
                    }
                    else
                    {
                        y = y * delta;
                    }
                }
                else
                {
                    if (_mult_inplace)
                    {
                        y *= step;
                    }
                    else
                    {
                        y = y * step;
                    }
                }
    
            }
            else
            {
                // 0 and 1 item long sequences have an undefined step
                step = double.NaN;
                // Multiply with delta to allow possible override of output class.
                y = y * delta;
            }

            y += start;

            if (endpoint && num > 1)
            {
                y["-1"] = stop;
            }

            retstep = step;

            return y.astype(dtype, copy : false);

        }

        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// </summary>
        /// <param name="start">The starting value of the sequence</param>
        /// <param name="stop">The end value of the sequence, unless `endpoint` is set to False.</param>
        /// <param name="retstep">return (`samples`, `step`), where `step` is the spacing between samples.</param>
        /// <param name="num">Number of samples to generate.Default is 50.Must be non-negative.</param>
        /// <param name="endpoint">If True, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <param name="dtype">The type of the output array.If `dtype` is not given, infer the data type from the other input arguments.</param>
        /// <returns></returns>
        public static ndarray linspace(decimal start, decimal stop, ref decimal retstep, int num = 50, bool endpoint = true, dtype dtype = null)
        {
            //  Return evenly spaced numbers over a specified interval.

            //  Returns `num` evenly spaced samples, calculated over the
            //  interval[`start`, `stop`].

            //  The endpoint of the interval can optionally be excluded.

            //  Parameters
            //  ----------
            //  start: scalar
            //     The starting value of the sequence.
            // stop : scalar
            //     The end value of the sequence, unless `endpoint` is set to False.
            //      In that case, the sequence consists of all but the last of ``num + 1``
            //      evenly spaced samples, so that `stop` is excluded.Note that the step
            //      size changes when `endpoint` is False.
            //  num : int, optional
            //      Number of samples to generate.Default is 50.Must be non-negative.
            //endpoint : bool, optional
            //      If True, `stop` is the last sample.Otherwise, it is not included.
            //      Default is True.
            //  retstep : bool, optional
            //      If True, return (`samples`, `step`), where `step` is the spacing
            //      between samples.
            //  dtype: dtype, optional
            //     The type of the output array.If `dtype` is not given, infer the data
            //   type from the other input arguments.


            //   ..versionadded:: 1.9.0

            //  Returns
            //  ------ -
            //  samples : ndarray
            //      There are `num` equally spaced samples in the closed interval
            //      ``[start, stop]`` or the half-open interval ``[start, stop)``
            //      (depending on whether `endpoint` is True or False).
            //  step : float, optional
            //      Only returned if `retstep` is True

            //      Size of spacing between samples.


            //  See Also
            //  --------
            //  arange : Similar to `linspace`, but uses a step size (instead of the
            //           number of samples).
            //  logspace : Samples uniformly distributed in log space.

            //  Examples
            //  --------
            //  >>> np.linspace(2.0, 3.0, num= 5)
            //  array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
            //  >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
            //  array([ 2. ,  2.2,  2.4,  2.6,  2.8])
            //  >>> np.linspace(2.0, 3.0, num=5, retstep=True)
            //  (array([2.  , 2.25, 2.5, 2.75, 3.  ]), 0.25)

            //  Graphical illustration:

            //  >>> import matplotlib.pyplot as plt
            //  >>> N = 8
            //  >>> y = np.zeros(N)
            //  >>> x1 = np.linspace(0, 10, N, endpoint = True)
            //  >>> x2 = np.linspace(0, 10, N, endpoint = False)
            //  >>> plt.plot(x1, y, 'o')
            //  [< matplotlib.lines.Line2D object at 0x...>]
            //  >>> plt.plot(x2, y + 0.5, 'o')
            //  [< matplotlib.lines.Line2D object at 0x...>]
            //  >>> plt.ylim([-0.5, 1])
            //  (-0.5, 1)
            //  >>> plt.show()

            ValidateArangeDtype(dtype);

            if (num < 0)
            {
                throw new ValueError(string.Format("Number of samples, {0}, must be non-negative.", num));
            }

            int div = 0;

            if (endpoint)
            {
                div = num - 1;
            }
            else
            {
                div = num;
            }

            // Convert float/complex array scalars to float, gh-3504
            // and make sure one can use variables that have an __array_interface__, gh-6634

            dtype dt = np.Decimal; // result_type(start, stop, Convert.ToSingle(num));
            if (dtype == null)
                dtype = dt;

            var y = np.arange(0, num, dtype: dt);

            decimal delta = stop - start;
            // In-place multiplication y *= delta/div is faster, but prevents the multiplicant
            // from overriding what class is produced, and thus prevents, e.g. use of Quantities,
            // see gh-7142. Hence, we multiply in place only for standard scalar types.

            bool _mult_inplace = np.isscalar(delta);
            decimal step = 0;
            if (num > 1)
            {
                step = delta / div;
                if (step == 0)
                {
                    // Special handling for denormal numbers, gh-5437
                    y /= div;
                    if (_mult_inplace)
                    {
                        y *= delta;
                    }
                    else
                    {
                        y = y * delta;
                    }
                }
                else
                {
                    if (_mult_inplace)
                    {
                        y *= step;
                    }
                    else
                    {
                        y = y * step;
                    }
                }

            }
            else
            {
                // 0 and 1 item long sequences have an undefined step
                step = -99999999999999;
                // Multiply with delta to allow possible override of output class.
                y = y * delta;
            }

            y += start;

            if (endpoint && num > 1)
            {
                y["-1"] = stop;
            }

            retstep = step;

            return y.astype(dtype, copy: false);

        }

        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// </summary>
        /// <param name="start">The starting value of the sequence</param>
        /// <param name="stop">The end value of the sequence, unless `endpoint` is set to False.</param>
        /// <param name="retstep">return (`samples`, `step`), where `step` is the spacing between samples.</param>
        /// <param name="num">Number of samples to generate.Default is 50.Must be non-negative.</param>
        /// <param name="endpoint">If True, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <param name="dtype">The type of the output array.If `dtype` is not given, infer the data type from the other input arguments.</param>
        /// <returns></returns>
        public static ndarray linspace(System.Numerics.Complex start, System.Numerics.Complex stop, ref System.Numerics.Complex retstep, int num = 50, bool endpoint = true, dtype dtype = null)
        {
            //  Return evenly spaced numbers over a specified interval.

            //  Returns `num` evenly spaced samples, calculated over the
            //  interval[`start`, `stop`].

            //  The endpoint of the interval can optionally be excluded.

            //  Parameters
            //  ----------
            //  start: scalar
            //     The starting value of the sequence.
            // stop : scalar
            //     The end value of the sequence, unless `endpoint` is set to False.
            //      In that case, the sequence consists of all but the last of ``num + 1``
            //      evenly spaced samples, so that `stop` is excluded.Note that the step
            //      size changes when `endpoint` is False.
            //  num : int, optional
            //      Number of samples to generate.Default is 50.Must be non-negative.
            //endpoint : bool, optional
            //      If True, `stop` is the last sample.Otherwise, it is not included.
            //      Default is True.
            //  retstep : bool, optional
            //      If True, return (`samples`, `step`), where `step` is the spacing
            //      between samples.
            //  dtype: dtype, optional
            //     The type of the output array.If `dtype` is not given, infer the data
            //   type from the other input arguments.


            //   ..versionadded:: 1.9.0

            //  Returns
            //  ------ -
            //  samples : ndarray
            //      There are `num` equally spaced samples in the closed interval
            //      ``[start, stop]`` or the half-open interval ``[start, stop)``
            //      (depending on whether `endpoint` is True or False).
            //  step : float, optional
            //      Only returned if `retstep` is True

            //      Size of spacing between samples.


            //  See Also
            //  --------
            //  arange : Similar to `linspace`, but uses a step size (instead of the
            //           number of samples).
            //  logspace : Samples uniformly distributed in log space.

            //  Examples
            //  --------
            //  >>> np.linspace(2.0, 3.0, num= 5)
            //  array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
            //  >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
            //  array([ 2. ,  2.2,  2.4,  2.6,  2.8])
            //  >>> np.linspace(2.0, 3.0, num=5, retstep=True)
            //  (array([2.  , 2.25, 2.5, 2.75, 3.  ]), 0.25)

            //  Graphical illustration:

            //  >>> import matplotlib.pyplot as plt
            //  >>> N = 8
            //  >>> y = np.zeros(N)
            //  >>> x1 = np.linspace(0, 10, N, endpoint = True)
            //  >>> x2 = np.linspace(0, 10, N, endpoint = False)
            //  >>> plt.plot(x1, y, 'o')
            //  [< matplotlib.lines.Line2D object at 0x...>]
            //  >>> plt.plot(x2, y + 0.5, 'o')
            //  [< matplotlib.lines.Line2D object at 0x...>]
            //  >>> plt.ylim([-0.5, 1])
            //  (-0.5, 1)
            //  >>> plt.show()

            ValidateArangeDtype(dtype);

            if (num < 0)
            {
                throw new ValueError(string.Format("Number of samples, {0}, must be non-negative.", num));
            }

            int div = 0;

            if (endpoint)
            {
                div = num - 1;
            }
            else
            {
                div = num;
            }

            // Convert float/complex array scalars to float, gh-3504
            // and make sure one can use variables that have an __array_interface__, gh-6634

            dtype dt = np.Complex; // result_type(start, stop, Convert.ToSingle(num));
            if (dtype == null)
                dtype = dt;

            var y = np.arange(new System.Numerics.Complex(0, 0), num, dtype: dt);

            System.Numerics.Complex delta = stop - start;
            // In-place multiplication y *= delta/div is faster, but prevents the multiplicant
            // from overriding what class is produced, and thus prevents, e.g. use of Quantities,
            // see gh-7142. Hence, we multiply in place only for standard scalar types.

            bool _mult_inplace = np.isscalar(delta);
            System.Numerics.Complex step = 0;
            if (num > 1)
            {
                step = delta / div;
                if (step == 0)
                {
                    // Special handling for denormal numbers, gh-5437
                    y /= div;
                    if (_mult_inplace)
                    {
                        y *= delta;
                    }
                    else
                    {
                        y = y * delta;
                    }
                }
                else
                {
                    if (_mult_inplace)
                    {
                        y *= step;
                    }
                    else
                    {
                        y = y * step;
                    }
                }

            }
            else
            {
                // 0 and 1 item long sequences have an undefined step
                step = -99999999999999;
                // Multiply with delta to allow possible override of output class.
                y = y * delta;
            }

            y += start;

            if (endpoint && num > 1)
            {
                y["-1"] = stop;
            }

            retstep = step;

            return y.astype(dtype, copy: false);

        }

        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// </summary>
        /// <param name="start">The starting value of the sequence</param>
        /// <param name="stop">The end value of the sequence, unless `endpoint` is set to False.</param>
        /// <param name="retstep">return (`samples`, `step`), where `step` is the spacing between samples.</param>
        /// <param name="num">Number of samples to generate.Default is 50.Must be non-negative.</param>
        /// <param name="endpoint">If True, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <param name="dtype">The type of the output array.If `dtype` is not given, infer the data type from the other input arguments.</param>
        /// <returns></returns>
        public static ndarray linspace(System.Numerics.BigInteger start, System.Numerics.BigInteger stop, ref System.Numerics.BigInteger retstep, int num = 50, bool endpoint = true, dtype dtype = null)
        {
            //  Return evenly spaced numbers over a specified interval.

            //  Returns `num` evenly spaced samples, calculated over the
            //  interval[`start`, `stop`].

            //  The endpoint of the interval can optionally be excluded.

            //  Parameters
            //  ----------
            //  start: scalar
            //     The starting value of the sequence.
            // stop : scalar
            //     The end value of the sequence, unless `endpoint` is set to False.
            //      In that case, the sequence consists of all but the last of ``num + 1``
            //      evenly spaced samples, so that `stop` is excluded.Note that the step
            //      size changes when `endpoint` is False.
            //  num : int, optional
            //      Number of samples to generate.Default is 50.Must be non-negative.
            //endpoint : bool, optional
            //      If True, `stop` is the last sample.Otherwise, it is not included.
            //      Default is True.
            //  retstep : bool, optional
            //      If True, return (`samples`, `step`), where `step` is the spacing
            //      between samples.
            //  dtype: dtype, optional
            //     The type of the output array.If `dtype` is not given, infer the data
            //   type from the other input arguments.


            //   ..versionadded:: 1.9.0

            //  Returns
            //  ------ -
            //  samples : ndarray
            //      There are `num` equally spaced samples in the closed interval
            //      ``[start, stop]`` or the half-open interval ``[start, stop)``
            //      (depending on whether `endpoint` is True or False).
            //  step : float, optional
            //      Only returned if `retstep` is True

            //      Size of spacing between samples.


            //  See Also
            //  --------
            //  arange : Similar to `linspace`, but uses a step size (instead of the
            //           number of samples).
            //  logspace : Samples uniformly distributed in log space.

            //  Examples
            //  --------
            //  >>> np.linspace(2.0, 3.0, num= 5)
            //  array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
            //  >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
            //  array([ 2. ,  2.2,  2.4,  2.6,  2.8])
            //  >>> np.linspace(2.0, 3.0, num=5, retstep=True)
            //  (array([2.  , 2.25, 2.5, 2.75, 3.  ]), 0.25)

            //  Graphical illustration:

            //  >>> import matplotlib.pyplot as plt
            //  >>> N = 8
            //  >>> y = np.zeros(N)
            //  >>> x1 = np.linspace(0, 10, N, endpoint = True)
            //  >>> x2 = np.linspace(0, 10, N, endpoint = False)
            //  >>> plt.plot(x1, y, 'o')
            //  [< matplotlib.lines.Line2D object at 0x...>]
            //  >>> plt.plot(x2, y + 0.5, 'o')
            //  [< matplotlib.lines.Line2D object at 0x...>]
            //  >>> plt.ylim([-0.5, 1])
            //  (-0.5, 1)
            //  >>> plt.show()

            ValidateArangeDtype(dtype);

            if (num < 0)
            {
                throw new ValueError(string.Format("Number of samples, {0}, must be non-negative.", num));
            }

            int div = 0;

            if (endpoint)
            {
                div = num - 1;
            }
            else
            {
                div = num;
            }

            // Convert float/complex array scalars to float, gh-3504
            // and make sure one can use variables that have an __array_interface__, gh-6634

            dtype dt = np.BigInt; // result_type(start, stop, Convert.ToSingle(num));
            if (dtype == null)
                dtype = dt;

            var y = np.arange(new System.Numerics.BigInteger(0), num, dtype: dt);

            System.Numerics.BigInteger delta = stop - start;
            // In-place multiplication y *= delta/div is faster, but prevents the multiplicant
            // from overriding what class is produced, and thus prevents, e.g. use of Quantities,
            // see gh-7142. Hence, we multiply in place only for standard scalar types.

            bool _mult_inplace = np.isscalar(delta);
            System.Numerics.BigInteger step = 0;
            if (num > 1)
            {
                step = delta / div;
                if (step == 0)
                {
                    // Special handling for denormal numbers, gh-5437
                    y /= div;
                    if (_mult_inplace)
                    {
                        y *= delta;
                    }
                    else
                    {
                        y = y * delta;
                    }
                }
                else
                {
                    if (_mult_inplace)
                    {
                        y *= step;
                    }
                    else
                    {
                        y = y * step;
                    }
                }

            }
            else
            {
                // 0 and 1 item long sequences have an undefined step
                step = -99999999999999;
                // Multiply with delta to allow possible override of output class.
                y = y * delta;
            }

            y += start;

            if (endpoint && num > 1)
            {
                y["-1"] = stop;
            }

            retstep = step;

            return y.astype(dtype, copy: false);

        }


        #endregion

        #region logspace

        //  Return numbers spaced evenly on a log scale.

        //  In linear space, the sequence starts at ``base * *start``
        //  (`base` to the power of `start`) and ends with ``base * *stop``
        //  (see `endpoint` below).

        //  Parameters
        //  ----------
        //  start: float
        //      ``base * *start`` is the starting value of the sequence.
        //  stop: float
        //      ``base * *stop`` is the final value of the sequence, unless `endpoint`
        //      is False.In that case, ``num + 1`` values are spaced over the
        //    interval in log - space, of which all but the last(a sequence of
        //      length `num`) are returned.
        //  num: integer, optional
        //     Number of samples to generate.  Default is 50.
        // endpoint : boolean, optional
        //     If true, `stop` is the last sample.Otherwise, it is not included.
        //      Default is True.
        //  base : float, optional
        //      The base of the log space. The step size between the elements in
        //      ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        //      Default is 10.0.
        //  dtype : dtype
        //      The type of the output array.If `dtype` is not given, infer the data
        //    type from the other input arguments.

        //Returns
        //------ -
        //samples : ndarray
        //      `num` samples, equally spaced on a log scale.

        //  See Also
        //  --------
        //  arange : Similar to linspace, with the step size specified instead of the
        //           number of samples.Note that, when used with a float endpoint, the
        //           endpoint may or may not be included.
        //  linspace : Similar to logspace, but with the samples uniformly distributed
        //             in linear space, instead of log space.
        //  geomspace : Similar to logspace, but with endpoints specified directly.

        //  Notes
        //  ---- -
        //  Logspace is equivalent to the code

        //  >>> y = np.linspace(start, stop, num = num, endpoint = endpoint)
        //  ... # doctest: +SKIP
        //  >>> power(base, y).astype(dtype)
        //  ... # doctest: +SKIP

        //  Examples
        //  --------
        //  >>> np.logspace(2.0, 3.0, num = 4)
        //  array([100.        , 215.443469, 464.15888336, 1000.        ])
        //  >>> np.logspace(2.0, 3.0, num = 4, endpoint = False)
        //  array([100.        , 177.827941, 316.22776602, 562.34132519])
        //  >>> np.logspace(2.0, 3.0, num = 4, base = 2.0)
        //  array([4.        , 5.0396842, 6.34960421, 8.        ])

        //  Graphical illustration:

        //  >>> import matplotlib.pyplot as plt
        //  >>> N = 10
        //  >>> x1 = np.logspace(0.1, 1, N, endpoint = True)
        //  >>> x2 = np.logspace(0.1, 1, N, endpoint = False)
        //  >>> y = np.zeros(N)
        //  >>> plt.plot(x1, y, 'o')
        //  [< matplotlib.lines.Line2D object at 0x...>]
        //  >>> plt.plot(x2, y + 0.5, 'o')
        //  [< matplotlib.lines.Line2D object at 0x...>]
        //  >>> plt.ylim([-0.5, 1])
        //  (-0.5, 1)
        //  >>> plt.show()


        /// <summary>
        /// Return numbers spaced evenly on a log scale.
        /// </summary>
        /// <param name="start">the starting value of the sequence</param>
        /// <param name="stop">is the final value of the sequence</param>
        /// <param name="num">Number of samples to generate.  Default is 50.</param>
        /// <param name="endpoint">If true, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <param name="_base">The base of the log space.</param>
        /// <param name="dtype">The type of the output array</param>
        /// <returns></returns>                   
        public static ndarray logspace(Int64 start, Int64 stop, int num = 50, bool endpoint = true, double _base = 10.0, dtype dtype = null)
        {
            return logspace(Convert.ToDouble(start), Convert.ToDouble(stop), num, endpoint, _base, dtype);
        }
        /// <summary>
        /// Return numbers spaced evenly on a log scale.
        /// </summary>
        /// <param name="start">the starting value of the sequence</param>
        /// <param name="stop">is the final value of the sequence</param>
        /// <param name="num">Number of samples to generate.  Default is 50.</param>
        /// <param name="endpoint">If true, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <param name="_base">The base of the log space.</param>
        /// <param name="dtype">The type of the output array</param>
        /// <returns></returns>
        public static ndarray logspace(double start, double stop, int num = 50, bool endpoint = true, double _base = 10.0, dtype dtype = null)
        {
            double retstep = 0;
            ndarray y = linspace(start, stop, ref retstep, num: num, endpoint: endpoint);

            // convert to ndarray so we call the right power function
            ndarray pbase = asanyarray(_base);

            if (dtype is null)
                return np.power(pbase, y);
            return np.power(pbase, y).astype(dtype);
        }
        /// <summary>
        /// Return numbers spaced evenly on a log scale.
        /// </summary>
        /// <param name="start">the starting value of the sequence</param>
        /// <param name="stop">is the final value of the sequence</param>
        /// <param name="num">Number of samples to generate.  Default is 50.</param>
        /// <param name="endpoint">If true, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <param name="_base">The base of the log space.</param>
        /// <param name="dtype">The type of the output array</param>
        /// <returns></returns>
        public static ndarray logspace(decimal start, decimal stop, int num = 50, bool endpoint = true, double _base = 10.0, dtype dtype = null)
        {
            decimal retstep = 0;
            ndarray y = linspace(start, stop, ref retstep, num: num, endpoint: endpoint);

            // convert to ndarray so we call the right power function
            ndarray pbase = asanyarray(Convert.ToDecimal(_base));

            if (dtype is null)
                return np.power(pbase, y);
            return np.power(pbase, y).astype(dtype);
        }
        /// <summary>
        /// Return numbers spaced evenly on a log scale.
        /// </summary>
        /// <param name="start">the starting value of the sequence</param>
        /// <param name="stop">is the final value of the sequence</param>
        /// <param name="num">Number of samples to generate.  Default is 50.</param>
        /// <param name="endpoint">If true, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <param name="_base">The base of the log space.</param>
        /// <param name="dtype">The type of the output array</param>
        /// <returns></returns>
        public static ndarray logspace(System.Numerics.Complex start, System.Numerics.Complex stop, int num = 50, bool endpoint = true, double _base = 10.0, dtype dtype = null)
        {
            System.Numerics.Complex retstep = 0;
            ndarray y = linspace(start, stop, ref retstep, num: num, endpoint: endpoint);

            // convert to ndarray so we call the right power function
            ndarray pbase = asanyarray(new System.Numerics.Complex(_base, 0));

            if (dtype is null)
                return np.power(pbase, y);
            return np.power(pbase, y).astype(dtype);
        }
        /// <summary>
        /// Return numbers spaced evenly on a log scale.
        /// </summary>
        /// <param name="start">the starting value of the sequence</param>
        /// <param name="stop">is the final value of the sequence</param>
        /// <param name="num">Number of samples to generate.  Default is 50.</param>
        /// <param name="endpoint">If true, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <param name="_base">The base of the log space.</param>
        /// <param name="dtype">The type of the output array</param>
        /// <returns></returns>
        public static ndarray logspace(System.Numerics.BigInteger start, System.Numerics.BigInteger stop, int num = 50, bool endpoint = true, double _base = 10.0, dtype dtype = null)
        {
            System.Numerics.BigInteger retstep = 0;
            ndarray y = linspace(start, stop, ref retstep, num: num, endpoint: endpoint);

            // convert to ndarray so we call the right power function
            ndarray pbase = asanyarray(new System.Numerics.BigInteger(_base));

            if (dtype is null)
                return np.power(pbase, y);
            return np.power(pbase, y).astype(dtype);
        }

        #endregion

        #region geomspace
        //  Return numbers spaced evenly on a log scale(a geometric progression).

        //  This is similar to `logspace`, but with endpoints specified directly.
        //  Each output sample is a constant multiple of the previous.

        //  Parameters
        //  ----------
        //  start: scalar
        //     The starting value of the sequence.
        // stop : scalar
        //     The final value of the sequence, unless `endpoint` is False.
        //     In that case, ``num + 1`` values are spaced over the
        //     interval in log - space, of which all but the last(a sequence of
        //       length `num`) are returned.
        //  num: integer, optional
        //     Number of samples to generate.  Default is 50.
        // endpoint : boolean, optional
        //     If true, `stop` is the last sample.Otherwise, it is not included.
        //      Default is True.
        //  dtype : dtype
        //      The type of the output array.If `dtype` is not given, infer the data
        //    type from the other input arguments.

        //Returns
        //------ -
        //samples : ndarray
        //      `num` samples, equally spaced on a log scale.

        //  See Also
        //  --------
        //  logspace : Similar to geomspace, but with endpoints specified using log
        //             and base.
        //  linspace : Similar to geomspace, but with arithmetic instead of geometric
        //             progression.
        //  arange : Similar to linspace, with the step size specified instead of the
        //           number of samples.

        //  Notes
        //  ---- -
        //  If the inputs or dtype are complex, the output will follow a logarithmic
        //  spiral in the complex plane.  (There are an infinite number of spirals
        //  passing through two points; the output will follow the shortest such path.)

        //  Examples
        //  --------
        //  >>> np.geomspace(1, 1000, num = 4)
        //  array([1., 10., 100., 1000.])
        //  >>> np.geomspace(1, 1000, num = 3, endpoint = False)
        //  array([1., 10., 100.])
        //  >>> np.geomspace(1, 1000, num = 4, endpoint = False)
        //  array([1.        , 5.62341325, 31.6227766, 177.827941])
        //  >>> np.geomspace(1, 256, num = 9)
        //  array([1., 2., 4., 8., 16., 32., 64., 128., 256.])

        //  Note that the above may not produce exact integers:

        //  >>> np.geomspace(1, 256, num = 9, dtype = int)
        //  array([1, 2, 4, 7, 16, 32, 63, 127, 256])
        //  >>> np.around(np.geomspace(1, 256, num = 9)).astype(int)
        //  array([1, 2, 4, 8, 16, 32, 64, 128, 256])

        //  Negative, decreasing, and complex inputs are allowed:

        //  >>> np.geomspace(1000, 1, num = 4)
        //  array([1000., 100., 10., 1.])
        //  >>> np.geomspace(-1000, -1, num = 4)
        //  array([-1000., -100., -10., -1.])
        //  >>> np.geomspace(1j, 1000j, num = 4)  # Straight line
        //  array([0.   + 1.j, 0.  + 10.j, 0. + 100.j, 0.+ 1000.j])
        //  >>> np.geomspace(-1 + 0j, 1 + 0j, num = 5)  # Circle
        //  array([-1.00000000 + 0.j, -0.70710678 + 0.70710678j,
        //          0.00000000 + 1.j, 0.70710678 + 0.70710678j,
        //          1.00000000 + 0.j])

        //  Graphical illustration of ``endpoint`` parameter:

        //  >>> import matplotlib.pyplot as plt
        //  >>> N = 10
        //  >>> y = np.zeros(N)
        //  >>> plt.semilogx(np.geomspace(1, 1000, N, endpoint = True), y + 1, 'o')
        //  >>> plt.semilogx(np.geomspace(1, 1000, N, endpoint = False), y + 2, 'o')
        //  >>> plt.axis([0.5, 2000, 0, 3])
        //  >>> plt.grid(True, color = '0.7', linestyle = '-', which = 'both', axis = 'both')
        //  >>> plt.show()

        /// <summary>
        /// Return numbers spaced evenly on a log scale(a geometric progression).
        /// </summary>
        /// <param name="start">The starting value of the sequence.</param>
        /// <param name="stop">The final value of the sequence</param>
        /// <param name="num">Number of samples to generate.</param>
        /// <param name="endpoint">If true, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <returns></returns>
        public static ndarray geomspace(Int64 start, Int64 stop, int num = 50, bool endpoint = true, dtype dtype = null)
        {
            return geomspace(Convert.ToDouble(start), Convert.ToDouble(stop), num, endpoint, dtype);
        }
        /// <summary>
        /// Return numbers spaced evenly on a log scale(a geometric progression).
        /// </summary>
        /// <param name="start">The starting value of the sequence.</param>
        /// <param name="stop">The final value of the sequence</param>
        /// <param name="num">Number of samples to generate.</param>
        /// <param name="endpoint">If true, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <returns></returns>
        public static ndarray geomspace(decimal start, decimal stop, int num = 50, bool endpoint = true)
        {
            return geomspace(Convert.ToDouble(start), Convert.ToDouble(stop), num, endpoint, np.Decimal);
        }
        /// <summary>
        /// Return numbers spaced evenly on a log scale(a geometric progression).
        /// </summary>
        /// <param name="start">The starting value of the sequence.</param>
        /// <param name="stop">The final value of the sequence</param>
        /// <param name="num">Number of samples to generate.</param>
        /// <param name="endpoint">If true, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <returns></returns>
        public static ndarray geomspace(double start, double stop, int num = 50, bool endpoint = true, dtype dtype = null)
        {
            ValidateArangeDtype(dtype);

            if (start == 0 || stop == 0)
            {
                throw new ValueError("Geometric sequence cannot include zero");
            }

            dtype dt = np.Float64; // result_type(start, stop, float(num))
            if (dtype == null)
            {
                dtype = dt;
            }

            // Avoid negligible real or imaginary parts in output by rotating to
            // positive real, calculating, then undoing rotation
            double out_sign = 1;
            //if (start.real == stop.real == 0)
            //{
            //    start = start.imag;
            //    stop = stop.imag;
            //    out_sign = 1j* out_sign;
            //}
            if (start < 0 && stop < 0)
            {
                start = -start;
                stop = -stop;
                out_sign = -out_sign;
            }

            // Promote both arguments to the same dtype in case, for instance, one is
            // complex and another is negative and log would produce NaN otherwise
            start = start + (stop - stop);
            stop = stop + (start - start);
            //if (_nx.issubdtype(dtype, _nx.complexfloating))
            //{
            //    start = start + 0j;
            //    stop = stop + 0j;
            //}


            var log_start = Math.Log10(start);
            var log_stop = Math.Log10(stop);

            var result = out_sign * logspace(log_start, log_stop, num: num,
                                         endpoint: endpoint, _base: 10.0, dtype: dtype);

            return result.astype(dtype);
        }

        /// <summary>
        /// Return numbers spaced evenly on a log scale(a geometric progression).
        /// </summary>
        /// <param name="start">The starting value of the sequence.</param>
        /// <param name="stop">The final value of the sequence</param>
        /// <param name="num">Number of samples to generate.</param>
        /// <param name="endpoint">If true, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <returns></returns>
        public static ndarray geomspace(System.Numerics.Complex start, System.Numerics.Complex stop, int num = 50, bool endpoint = true, dtype dtype = null)
        {
            ValidateArangeDtype(dtype);

            if (start == 0 || stop == 0)
            {
                throw new ValueError("Geometric sequence cannot include zero");
            }

            dtype dt = np.Complex; // result_type(start, stop, float(num))
            if (dtype == null)
            {
                dtype = dt;
            }

            // Avoid negligible real or imaginary parts in output by rotating to
            // positive real, calculating, then undoing rotation
            System.Numerics.Complex out_sign = 1;
            if (start.Real == 0 && stop.Real == 0)
            {
                start = start.Imaginary;
                stop = stop.Imaginary;
                out_sign = 1 * out_sign;
            }
            if (start.Real < 0 && stop.Real < 0)
            {
                start = -start;
                stop = -stop;
                out_sign = -out_sign;
            }

            // Promote both arguments to the same dtype in case, for instance, one is
            // complex and another is negative and log would produce NaN otherwise
            start = start + (stop - stop);
            stop = stop + (start - start);
            //if (_nx.issubdtype(dtype, _nx.complexfloating))
            //{
            //    start = start + 0j;
            //    stop = stop + 0j;
            //}


            var log_start = System.Numerics.Complex.Log10(start);
            var log_stop = System.Numerics.Complex.Log10(stop);

            var result = out_sign * logspace(log_start, log_stop, num: num,
                                         endpoint: endpoint, _base: 10.0, dtype: dtype);

            return result.astype(dtype);
        }

        /// <summary>
        /// Return numbers spaced evenly on a log scale(a geometric progression).
        /// </summary>
        /// <param name="start">The starting value of the sequence.</param>
        /// <param name="stop">The final value of the sequence</param>
        /// <param name="num">Number of samples to generate.</param>
        /// <param name="endpoint">If true, `stop` is the last sample.Otherwise, it is not included.</param>
        /// <returns></returns>
        public static ndarray geomspace(System.Numerics.BigInteger start, System.Numerics.BigInteger stop, int num = 50, bool endpoint = true, dtype dtype = null)
        {
            ValidateArangeDtype(dtype);

            if (start == 0 || stop == 0)
            {
                throw new ValueError("Geometric sequence cannot include zero");
            }

            dtype dt = np.BigInt; // result_type(start, stop, float(num))
            if (dtype == null)
            {
                dtype = dt;
            }

            System.Numerics.BigInteger out_sign = 1;
  
            if (start < 0 && stop < 0)
            {
                start = -start;
                stop = -stop;
                out_sign = -out_sign;
            }

            // Promote both arguments to the same dtype in case, for instance, one is
            // complex and another is negative and log would produce NaN otherwise
            start = start + (stop - stop);
            stop = stop + (start - start);
 

            var log_start = System.Numerics.BigInteger.Log10(start);
            var log_stop = System.Numerics.BigInteger.Log10(stop);

            var result = out_sign * logspace(log_start, log_stop, num: num,
                                         endpoint: endpoint, _base: 10.0, dtype: dtype);

            return result.astype(dtype);
        }

 
        #endregion

        #region To/From file/string/stream

        /// <summary>
        /// Reads the contents of a text or binary file and turns the contents into an array. If
        /// 'sep' is specified the file is assumed to be text, other it is assumed binary.
        /// </summary>
        /// <param name="file">file name string</param>
        /// <param name="dtype">Optional type for the resulting array, default is double</param>
        /// <param name="count">Optional number of array elements to read, default reads all elements</param>
        /// <param name="sep">Optional separator for text elements</param>
        /// <returns></returns>
        public static ndarray fromfile(string fileName, dtype dtype = null, int count = -1, string sep = null)
        {
            if (dtype == null)
            {
                dtype = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_FLOAT);
            }


            return NpyCoreApi.ArrayFromFile(fileName, dtype, count, (sep != null) ? sep.ToString() : null);
        }

        /// <summary>
        /// Reads the contents of a text or binary file and turns the contents into an array. If
        /// 'sep' is specified the file is assumed to be text, other it is assumed binary.
        /// </summary>
        /// <param name="fileStream">Stream containing data</param>
        /// <param name="dtype">Optional type for the resulting array, default is double</param>
        /// <param name="count">Optional number of array elements to read, default reads all elements</param>
        /// <param name="sep">Optional separator for text elements</param>
        /// <returns></returns>
        public static ndarray fromfile(Stream fileStream, dtype dtype = null, int count = -1, string sep = null)
        {
            if (dtype == null)
            {
                dtype = NpyCoreApi.DescrFromType(NPY_TYPES.NPY_FLOAT);
            }


            return NpyCoreApi.ArrayFromStream(fileStream, dtype, count, (sep != null) ? sep.ToString() : null);
        }

        /// <summary>
        /// writes the specified array contents to the specified file
        /// </summary>
        /// <param name="arr">array with data to write to file</param>
        /// <param name="fileName">full path name of file to write</param>
        /// <param name="sep">>Optional separator for text elements</param>
        /// <param name="format">Optional output format specifier</param>
        public static void tofile(ndarray arr, string fileName, string sep = null, string format = null)
        {
            NpyCoreApi.ArrayToFile(arr, fileName, sep, format);
        }

        /// <summary>
        /// writes the specified array contents to the specified file
        /// </summary>
        /// <param name="arr">array with data to write to file</param>
        /// <param name="fileStream">.NET stream to write file contents to</param>
        /// <param name="sep">>Optional separator for text elements</param>
        /// <param name="format">Optional output format specifier</param>
        public static void tofile(ndarray arr, Stream fileStream, string sep = null, string format = null)
        {
            NpyCoreApi.ArrayToStream(arr, fileStream, sep, format);
        }
        #endregion

        #region frombuffer

        /// <summary>
        /// converts a byte buffer of data into the specified data type.
        /// </summary>
        /// <param name="buffer">buffer of data to convert</param>
        /// <param name="dtype">type of data to output.  if null, Float64 will be returned</param>
        /// <param name="count">number of bytes to copy</param>
        /// <param name="offset">offset into buffer to start copying.</param>
        /// <returns></returns>
        public static ndarray frombuffer(byte []buffer, dtype dtype = null, int count = -1, int offset = 0)
        {
            if (dtype == null)
                dtype = np.Float64;

            int bytesToCopy = count >= 0 ? count : buffer.Length;

            Array tempBuffer = null;

            switch (dtype.TypeNum)
            {
                case NPY_TYPES.NPY_BOOL:
                    tempBuffer = new bool[bytesToCopy / sizeof(bool)];
                    break;
                case NPY_TYPES.NPY_BYTE:
                    tempBuffer = new sbyte[bytesToCopy / sizeof(sbyte)];
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    tempBuffer = new byte[bytesToCopy / sizeof(byte)];
                    break;
                case NPY_TYPES.NPY_INT16:
                    tempBuffer = new Int16[bytesToCopy / sizeof(Int16)];
                    break;
                case NPY_TYPES.NPY_UINT16:
                    tempBuffer = new UInt16[bytesToCopy / sizeof(UInt16)];
                    break;
                case NPY_TYPES.NPY_INT32:
                    tempBuffer = new Int32[bytesToCopy / sizeof(Int32)];
                    break;
                case NPY_TYPES.NPY_UINT32:
                    tempBuffer = new UInt32[bytesToCopy / sizeof(UInt32)];
                    break;
                case NPY_TYPES.NPY_INT64:
                    tempBuffer = new Int64[bytesToCopy / sizeof(Int64)];
                    break;
                case NPY_TYPES.NPY_UINT64:
                    tempBuffer = new UInt64[bytesToCopy / sizeof(UInt64)];
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    tempBuffer = new float[bytesToCopy / sizeof(float)];
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    tempBuffer = new double[bytesToCopy / sizeof(double)];
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    throw new Exception("np.frombuffer does not support Decimal data types");
                case NPY_TYPES.NPY_COMPLEX:
                    throw new Exception("np.frombuffer does not support Complex data types");
                case NPY_TYPES.NPY_BIGINT:
                    throw new Exception("np.frombuffer does not support BigInt data types");
                case NPY_TYPES.NPY_OBJECT:
                    throw new Exception("np.frombuffer does not support Object data types");
                case NPY_TYPES.NPY_STRING:
                    throw new Exception("np.frombuffer does not support string data types");
            }

            if (tempBuffer != null)
            {
                Buffer.BlockCopy(buffer, offset, tempBuffer, 0, bytesToCopy);
                ndarray c = np.array(tempBuffer, dtype: dtype, order: NumpyLib.NPY_ORDER.NPY_CORDER, copy: true);
                return c;
            }

            return null;
        }
        #endregion

        #region view

        /// <summary>
        /// New view of array with the same data.
        /// </summary>
        /// <param name="arr">array to take new view from</param>
        /// <param name="dtype">Data-type of the returned view</param>
        /// <param name="type">not used</param>
        /// <returns></returns>
        public static ndarray view(ndarray arr, dtype dtype = null, object type = null)
        {
            if (dtype == null)
            {
                dtype = arr.Dtype;
            }

  
            return NpyCoreApi.View(arr, dtype, type);
        }

        #endregion

        #region numeric operations

        /// <summary>
        /// Return the element-wise square of the input.
        /// </summary>
        /// <param name="a">Input data</param>
        /// <returns></returns>
        public static ndarray square(object a)
        {
            return NpyCoreApi.PerformNumericOp(asanyarray(a), UFuncOperation.square, 0);
        }
        /// <summary>
        /// Return the non-negative square-root of an array, element-wise.
        /// </summary>
        /// <param name="a">The values whose square-roots are required.</param>
        /// <returns></returns>
        public static ndarray sqrt(object a)
        {
            return NpyCoreApi.PerformNumericOp(asanyarray(a), UFuncOperation.sqrt, 0);
        }

        /// <summary>
        /// Return the cube-root of an array, element-wise.
        /// </summary>
        /// <param name="a">The values whose cube-roots are required.</param>
        /// <returns></returns>
        public static ndarray cbrt(object a)
        {
            return NpyCoreApi.PerformNumericOp(asanyarray(a), UFuncOperation.power, 1.0/3.0);
        }
        /// <summary>
        /// Calculate the absolute value element-wise.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns></returns>
        public static ndarray absolute(object a)
        {
            return NpyCoreApi.PerformNumericOp(asanyarray(a), UFuncOperation.absolute, 0);
        }
        /// <summary>
        /// Compute the absolute values element-wise.
        /// </summary>
        /// <param name="a">The array of numbers for which the absolute values are required</param>
        /// <returns></returns>
        public static ndarray fabs(object a)
        {
            return NpyCoreApi.PerformNumericOp(asanyarray(a), UFuncOperation.absolute, 0);
        }


        #endregion

        private delegate ndarray WrapDelegate(ndarray a);

        #region concatenate

        /// <summary>
        /// Join a sequence of arrays along an existing axis.
        /// </summary>
        /// <param name="value">sequence of array_like</param>
        /// <param name="axis">The axis along which the arrays will be joined</param>
        /// <returns></returns>
        public static ndarray concatenate(object value, int? axis = 0)
        {
            return concatenate( asanyarray(value), axis: axis);
        }
        /// <summary>
        /// Join a sequence of arrays along an existing axis.
        /// </summary>
        /// <param name="value">sequence of array_like</param>
        /// <param name="axis">The axis along which the arrays will be joined</param>
        /// <returns></returns>
        public static ndarray concatenate(ndarray array, int? axis = 0)
        {
            if (array.ndim <= 1)
            {
                throw new Exception("zero and one dimensional arrays cannot be concatenated");
            }

            var numSequence = array.shape.iDims[0];

            ndarray[] sequenceArray = new ndarray[numSequence];
            for (npy_intp i = 0; i < numSequence; i++)
            {
                sequenceArray[i] = array[i] as ndarray;
            }

            return concatenate(sequenceArray, axis: axis);
        }
        /// <summary>
        /// Join a sequence of arrays along an existing axis.
        /// </summary>
        /// <param name="value">sequence of array_like</param>
        /// <param name="axis">The axis along which the arrays will be joined</param>
        /// <returns></returns>
        public static ndarray concatenate(ValueTuple<object, object> values, int? axis = 0)
        {
            return concatenate(new ndarray[] { asanyarray(values.Item1), asanyarray(values.Item2) }, axis: axis);
        }
        /// <summary>
        /// Join a sequence of arrays along an existing axis.
        /// </summary>
        /// <param name="value">sequence of array_like</param>
        /// <param name="axis">The axis along which the arrays will be joined</param>
        /// <returns></returns>
        public static ndarray concatenate(ValueTuple<object, object, object> values, int? axis = 0)
        {
            return concatenate(new ndarray[] { asanyarray(values.Item1), asanyarray(values.Item2), asanyarray(values.Item3) }, axis: axis);
        }
        /// <summary>
        /// Join a sequence of arrays along an existing axis.
        /// </summary>
        /// <param name="value">sequence of array_like</param>
        /// <param name="axis">The axis along which the arrays will be joined</param>
        /// <returns></returns>
        public static ndarray concatenate(ValueTuple<object, object, object, object> values, int? axis = 0)
        {
            return concatenate(new ndarray[] { asanyarray(values.Item1), asanyarray(values.Item2), asanyarray(values.Item3), asanyarray(values.Item4) }, axis: axis);
        }
        /// <summary>
        /// Join a sequence of arrays along an existing axis.
        /// </summary>
        /// <param name="value">sequence of array_like</param>
        /// <param name="axis">The axis along which the arrays will be joined</param>
        /// <returns></returns>
        public static ndarray concatenate(IEnumerable<ndarray> seq, int? axis = 0)
        {
            return np.Concatenate(seq, axis);
        }
        /// <summary>
        /// Join a sequence of arrays along an existing axis.
        /// </summary>
        /// <param name="value">sequence of array_like</param>
        /// <param name="axis">The axis along which the arrays will be joined</param>
        /// <returns></returns>
        public static ndarray concatenate(ndarray a, ndarray b, int? axis = 0)
        {
            ndarray[] seq = new ndarray[] { a, b };
            return np.Concatenate(seq, axis);
        }

        #endregion

        #region where
        /// <summary>
        /// Return elements chosen from x or y depending on condition.
        /// </summary>
        /// <param name="condition">array_like, bool, Where True, yield x, otherwise yield y.</param>
        /// <param name="x">Values from which to choose.</param>
        /// <param name="y">Values from which to choose.</param>
        /// <returns></returns>
        public static object where(object condition,  object x = null, object y = null)
        {
            var _condition = asanyarray(condition);

            int missing = 0;
            missing += x != null ? 0 : 1;
            missing += y != null ? 0 : 1;

            if (missing == 1)
            {
                throw new Exception("Must provide both 'x' and 'y' or neither.");
            }
            if (missing == 2)
            {
                ndarray aCondition = np.FromAny(_condition, null, 0, 0, 0, null);
                return aCondition.NonZero();
            }

            var _x = asanyarray(x);
            var _y = asanyarray(y);

            ndarray aCondition1 = np.FromAny(_condition, null, 0, 0, 0, null);
            ndarray ret = np.ndarray(aCondition1.shape, _x.Dtype);

            Parallel.For(0, aCondition1.Size, i =>
            {
                bool c = (bool)_GetWhereItem(aCondition1, i);
                if (c)
                {
                    _SetWhereItem(ret, i, _GetWhereItem(_x, i));
                }
                else
                {
                    _SetWhereItem(ret, i, _GetWhereItem(_y, i));
                }
            });

            return ret;
        }

        private static void _SetWhereItem(ndarray a, npy_intp index, object v)
        {
            a.SetItem(v, _SanitizeIndex(a, index));
        }

        private static object _GetWhereItem(ndarray a, npy_intp index)
        {
            return a.GetItem(_SanitizeIndex(a, index));
        }

        private static npy_intp _SanitizeIndex(ndarray a, npy_intp index)
        {
            if (a.Size <= index)
                return index % a.Size;
            return index;
        }

        #endregion

        #region packbits/unpackbits

        /// <summary>
        ///  Packs the elements of a binary-valued array into bits in a uint8 array.
        /// </summary>
        /// <param name="input">An array of integers or booleans whose elements should be packed to bits</param>
        /// <param name="axis">The dimension over which bit-packing is done. 0 implies packing the flattened array</param>
        /// <returns>Array of type uint8 whose elements represent bits corresponding to the logical (0 or nonzero) value of the input elements. 
        /// The shape of packed has the same number of dimensions as the input (unless axis is None, in which case the output is 1-D).</returns>
        public static ndarray packbits(ndarray input, int axis = 0)
        {
            // sanity check input array type
            if (input.TypeNum != NPY_TYPES.NPY_UBYTE)
            {
                throw new Exception(string.Format("Expected input type of uint8, instead got {0}", input.TypeNum));
            }


            // sanity check the axis value first
            if (axis >= 0)
            {
                if (axis >= input.ndim)
                {
                    throw new Exception(string.Format("AxisError(axis {0} is out of bounds for array of dimension {1})", axis, input.ndim));
                }
            }

            // get the raw data from the array object and allocate another array big enough to hold the unpacked bits
            var rawdatavp =  input.rawdata(0);
            byte[] rawdata = (byte [])rawdatavp.datap;
            byte[] packedData = new byte[rawdata.Length / 8];

            // pack the bits
            for (int i = 0; i < rawdata.Length; i++)
            {
                byte Bit = (byte)((rawdata[i] & 0x01) != 0 ? 0x80 : 0x00);
                packedData[i >> 3] |= (byte)(Bit >> (i & 0x07));
            }

            // create a new ndarray with the unpacked bits
            var packedArray = array(packedData);

            // if the user wants to reshape on a different axis
            if (axis >= 0)
            {
                npy_intp[] NewIndices = new npy_intp[input.ndim];
                Array.Copy(input.Array.dimensions, NewIndices, NewIndices.Length);

                NewIndices[axis] = -1;
                return packedArray.reshape(NewIndices);
            }

            // return the unpacked bits
            return packedArray;
        }

        /// <summary>
        /// Unpacks elements of a uint8 array into a binary-valued output array. Each element of myarray represents a bit-field that should 
        /// be unpacked into a binary-valued output array. The shape of the output array is either 1-D (if axis is None) or the same shape 
        /// as the input array with unpacking done along the axis specified.
        /// </summary>
        /// <param name="input">Input array. ndarray, uint8 type</param>
        /// <param name="axis">The dimension over which bit-unpacking is done. 0 implies unpacking the flattened array.</param>
        /// <returns>ndarray, uint8 type. The elements are binary-valued (0 or 1).</returns>
        public static ndarray unpackbits(ndarray input, int axis = 0)
        {
            // sanity check input array type
            if (input.TypeNum != NPY_TYPES.NPY_UBYTE)
            {
                throw new Exception(string.Format("Expected input type of uint8, instead got {0}", input.TypeNum));
            }

            // sanity check the axis value first
            if (axis >= 0)
            {
                if (axis >= input.ndim)
                {
                    throw new Exception(string.Format("AxisError(axis {0} is out of bounds for array of dimension {1})", axis, input.ndim));
                }
            }

            // get the raw data from the array object and allocate another array big enough to hold the unpacked bits
            var rawdatavp = input.rawdata(0);
            byte[] rawdata = (byte[])rawdatavp.datap;
            byte[] unpackedData = new byte[rawdata.Length * 8];

            // unpack the bits
            int upd_offset = 0;
            foreach (byte rd in rawdata)
            {
                byte Mask = 0x80;
                for (int i = 0; i < 8; i++)
                {
                    unpackedData[upd_offset++] = (byte)(((rd & Mask) != 0) ? 0x01 : 0x00);
                    Mask >>= 1;
                }

            }

            // create a new npArray with the unpacked bits
            var unpackedArray = array(unpackedData);

            // if the user wants to reshape on a different axis
            if (axis >= 0)
            {
                npy_intp[] NewIndices = new npy_intp[input.ndim];
                Array.Copy(input.Array.dimensions, NewIndices, NewIndices.Length);

                NewIndices[axis] = -1;
                return unpackedArray.reshape(NewIndices);
            }

            // return the unpacked bits
            return unpackedArray;
        }

        #endregion
  
        #region ToString

        public static string ToString(ndarray a)
        {
            long nbytes = a.Size * a.ItemSize;
            byte[] data = new byte[nbytes];
            NpyCoreApi.GetBytes(a, data, NPY_ORDER.NPY_CORDER);
            return ASCIIEncoding.UTF8.GetString(data);
        }

        public static string ToString(VoidPtr a)
        {
            if (a.type_num != NPY_TYPES.NPY_BYTE)
                return null;

            return ASCIIEncoding.UTF8.GetString(a.datap as byte[]);
        }

        #endregion

        #region bitwise_and
        /// <summary>
        /// Compute the bit-wise AND of two arrays element-wise.
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <param name="out"></param>
        /// <param name="where"></param>
        /// <returns></returns>
        public static ndarray bitwise_and(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.bitwise_and, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region bitwise_or
        /// <summary>
        /// Compute the bit-wise OR of two arrays element-wise.
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <param name="out"></param>
        /// <param name="where"></param>
        /// <returns></returns>
        public static ndarray bitwise_or(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.bitwise_or, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region bitwise_xor
        /// <summary>
        /// Compute the bit-wise XOR of two arrays element-wise.
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <param name="out"></param>
        /// <param name="where"></param>
        /// <returns></returns>
        public static ndarray bitwise_xor(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.bitwise_xor, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region bitwise_not
        /// <summary>
        /// invert the bits of the specified array.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static ndarray bitwise_not(object input)
        {
            return NpyCoreApi.PerformNumericOp(asanyarray(input), UFuncOperation.invert, 0, false);
        }
        #endregion

        #region logical_and
        /// <summary>
        /// Compute the truth value of x1 AND x2 element-wise.
        /// </summary>
        /// <param name="x1">Input array</param>
        /// <param name="x2">Input array</param>
        /// <param name="out">(optional) A location into which the result is stored</param>
        /// <param name="where">(optional) This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray logical_and(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.logical_and, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region logical_or
        /// <summary>
        /// Compute the truth value of x1 OR x2 element-wise.
        /// </summary>
        /// <param name="x1">Input array</param>
        /// <param name="x2">Input array</param>
        /// <param name="out">(optional) A location into which the result is stored</param>
        /// <param name="where">(optional) This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray logical_or(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.logical_or, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region logical_xor
        /// <summary>
        /// Compute the truth value of x1 XOR x2, element-wise.
        /// </summary>
        /// <param name="x1">Input array</param>
        /// <param name="x2">Input array</param>
        /// <param name="out">(optional) A location into which the result is stored</param>
        /// <param name="where">(optional) This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray logical_xor(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.not_equal, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region logical_not
        /// <summary>
        /// Compute the truth value of NOT x element-wise.
        /// </summary>
        /// <param name="input">Input array</param>
        /// <returns></returns>
        public static ndarray logical_not(object input)
        {
            return NpyCoreApi.PerformNumericOp(asanyarray(input), UFuncOperation.equal, 0, false);
        }

        #endregion

        #region greater

        /// <summary>
        /// Return the truth value of (x1 > x2) element-wise.
        /// </summary>
        /// <param name="x1">Input array</param>
        /// <param name="x2">Input array</param>
        /// <param name="out">A location into which the result is stored</param>
        /// <param name="where">At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray greater(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.greater, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region greater_equal
        /// <summary>
        /// Return the truth value of (x1 >= x2) element-wise.
        /// </summary>
        /// <param name="x1">Input array</param>
        /// <param name="x2">Input array</param>
        /// <param name="out">A location into which the result is stored</param>
        /// <param name="where">At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray greater_equal(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.greater_equal, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region less
        /// <summary>
        /// Return the truth value of (x1 < x2) element-wise.
        /// </summary>
        /// <param name="x1">Input array</param>
        /// <param name="x2">Input array</param>
        /// <param name="out">A location into which the result is stored</param>
        /// <param name="where">At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray less(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.less, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region less_equal
        /// <summary>
        /// Return the truth value of (x1 <= x2) element-wise.
        /// </summary>
        /// <param name="x1">Input array</param>
        /// <param name="x2">Input array</param>
        /// <param name="out">A location into which the result is stored</param>
        /// <param name="where">At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray less_equal(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.less_equal, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region equal
        /// <summary>
        /// Return the truth value of (x1 == x2) element-wise.
        /// </summary>
        /// <param name="x1">Input array</param>
        /// <param name="x2">Input array</param>
        /// <param name="out">A location into which the result is stored</param>
        /// <param name="where">At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray equal(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.equal, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region not_equal
        /// <summary>
        /// Return the truth value of (x1 != x2) element-wise.
        /// </summary>
        /// <param name="x1">Input array</param>
        /// <param name="x2">Input array</param>
        /// <param name="out">A location into which the result is stored</param>
        /// <param name="where">At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray not_equal(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.not_equal, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region invert
        /// <summary>
        /// Compute bit-wise inversion, or bit-wise NOT, element-wise.
        /// </summary>
        /// <param name="input">Only integer and boolean types are handled.</param>
        /// <returns></returns>
        public static ndarray invert(object input)
        {
            return NpyCoreApi.PerformNumericOp(asanyarray(input), UFuncOperation.invert, 0, false);
        }

        #endregion

        #region right_shift
        /// <summary>
        /// Shift the bits of an integer to the right.
        /// </summary>
        /// <param name="x1">Input values.</param>
        /// <param name="x2">Number of bits to remove at the right of x1</param>
        /// <param name="out">A location into which the result is stored</param>
        /// <param name="where">At locations where the condition is True, the out array will be set to the ufunc result. </param>
        /// <returns></returns>
        public static ndarray right_shift(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.right_shift, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region left_shift
        /// <summary>
        /// Shift the bits of an integer to the left.
        /// </summary>
        /// <param name="x1">Input values.</param>
        /// <param name="x2">Number of bits to remove at the right of x1</param>
        /// <param name="out">A location into which the result is stored</param>
        /// <param name="where">At locations where the condition is True, the out array will be set to the ufunc result. </param>
        /// <returns></returns>
        public static ndarray left_shift(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.left_shift, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        #endregion

        #region floor
        /// <summary>
        /// Return the floor of the input, element-wise.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <returns></returns>
        public static ndarray floor(object x)
        {
            return NpyCoreApi.Floor(asanyarray(x), null);
        }


        #endregion

        #region isnan
        public static float NaN = float.NaN;
        /// <summary>
        /// Test element-wise for NaN and return result as a boolean array.
        /// </summary>
        /// <param name="input">Input array.</param>
        /// <returns></returns>
        public static ndarray isnan(ndarray input)
        {
            return NpyCoreApi.IsNaN(input);
        }

        #endregion

        #region Infinity
        public static float Inf = float.PositiveInfinity;
        public static float NInf = float.NegativeInfinity;

        /// <summary>
        /// Test element-wise for positive or negative infinity.
        /// </summary>
        /// <param name="o">Input values</param>
        /// <returns></returns>
        public static ndarray isfinite(object o)
        {
            var input = asanyarray(o);
            var output = new bool[input.size];

            var ArrayHandler = DefaultArrayHandlers.GetArrayHandler(input.TypeNum);

            int oindex = 0;
            foreach (var i in input.Flat)
            {
                 if (ArrayHandler.IsNan(i) || ArrayHandler.IsInfinity(i))
                    output[oindex] = false;
                else
                    output[oindex] = true;
                oindex++;
            }

            return array(output).reshape(input.shape);
        }
        /// <summary>
        /// Test element-wise for positive or negative infinity.
        /// </summary>
        /// <param name="o">Input values</param>
        /// <returns></returns>
        public static ndarray isinf(object o)
        {
            var input = asanyarray(o);
            var output = new bool[input.size];

            int oindex = 0;
            foreach (var i in input.Flat)
            {
                float f = Convert.ToSingle(i);
                if (float.IsInfinity(f))
                    output[oindex] = true;
                else
                    output[oindex] = false;
                oindex++;
            }

            return array(output).reshape(input.shape);
        }
        /// <summary>
        /// Test element-wise for negative infinity, return result as bool array.
        /// </summary>
        /// <param name="o">The input array</param>
        /// <returns></returns>
        public static ndarray isneginf(object o)
        {
            var input = asanyarray(o);
            var output = new bool[input.size];

            int oindex = 0;
            foreach (var i in input.Flat)
            {
                float f = Convert.ToSingle(i);
                if (float.IsNegativeInfinity(f))
                    output[oindex] = true;
                else
                    output[oindex] = false;
                oindex++;
            }

            return array(output).reshape(input.shape);
        }
        /// <summary>
        /// Test element-wise for positive infinity, return result as bool array.
        /// </summary>
        /// <param name="o">The input array.</param>
        /// <returns></returns>
        public static ndarray isposinf(object o)
        {
            var input = asanyarray(o);
            var output = new bool[input.size];

            int oindex = 0;
            foreach (var i in input.Flat)
            {
                float f = Convert.ToSingle(i);
                if (float.IsPositiveInfinity(f))
                    output[oindex] = true;
                else
                    output[oindex] = false;
                oindex++;
            }

            return array(output).reshape(input.shape);
        }
        #endregion

        #region IndicesFromAxis
        public static IList<npy_intp> IndicesFromAxis(ndarray a, int axis)
        {
            axis = normalize_axis_index(axis, a.ndim);

            return NpyCoreApi.IndicesFromAxis(a, axis);
        }
        #endregion

        #region ViewFromAxis

        /// <summary>
        /// Creates a view that matches only the specified axis values
        /// </summary>
        /// <param name="a"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static ndarray ViewFromAxis(ndarray a, int axis)
        {
            axis = normalize_axis_index(axis, a.ndim);

            var slices = new object[a.ndim];
            for (int i = 0; i < slices.Length; i++)
            {
                slices[i] = 0;
            }
            slices[axis] = ":";

            return a.A(slices);
        }
        #endregion

        #region correlate
        /// <summary>
        /// Cross-correlation of two 1-dimensional sequences.
        /// </summary>
        /// <param name="o1">Input sequence</param>
        /// <param name="o2">Input sequence</param>
        /// <param name="mode">{‘valid’, ‘same’, ‘full’}, optional</param>
        /// <returns></returns>
        public static ndarray correlate(object o1, object o2, NPY_CONVOLE_MODE mode = NPY_CONVOLE_MODE.NPY_CONVOLVE_VALID)
        {
            //Cross - correlation of two 1 - dimensional sequences.

            //This function computes the correlation as generally defined in signal
            //processing texts::

            //    c_{ av}[k] = sum_n a[n + k] * conj(v[n])

            //with a and v sequences being zero-padded where necessary and conj being
            //the conjugate.

            //Parameters
            //----------
            //a, v : array_like
            //    Input sequences.
            //mode : { 'valid', 'same', 'full'}, optional
            //Refer to the `convolve` docstring.Note that the default
            //    is 'valid', unlike `convolve`, which uses 'full'.
            //old_behavior : bool
            //    `old_behavior` was removed in NumPy 1.10. If you need the old
            //    behavior, use `multiarray.correlate`.

            //Returns
            //-------
            //out : ndarray
            //    Discrete cross-correlation of `a` and `v`.

            //See Also
            //--------
            //convolve : Discrete, linear convolution of two one-dimensional sequences.
            //multiarray.correlate : Old, no conjugate, version of correlate.

            //Notes
            //-----
            //The definition of correlation above is not unique and sometimes correlation
            //may be defined differently.Another common definition is::

            //    c'_{av}[k] = sum_n a[n] conj(v[n+k])

            //which is related to ``c_{av} [k]`` by ``c'_{av}[k] = c_{av}[-k]``.

            //Examples
            //--------
            //>>> np.correlate([1, 2, 3], [0, 1, 0.5])
            //array([3.5])
            //>>> np.correlate([1, 2, 3], [0, 1, 0.5], "same")
            //array([ 2. ,  3.5,  3. ])
            //>>> np.correlate([1, 2, 3], [0, 1, 0.5], "full")
            //array([ 0.5,  2. ,  3.5,  3. ,  0. ])

            //Using complex sequences:

            //>>> np.correlate([1 + 1j, 2, 3 - 1j], [0, 1, 0.5j], 'full')
            //array([ 0.5-0.5j,  1.0+0.j,  1.5-1.5j,  3.0-1.j,  0.0+0.j])

            //Note that you get the time reversed, complex conjugated result
            //when the two input sequences change places, i.e.,
            //``c_{va} [k] = c^{*}_{av}[-k]``:

            //>>> np.correlate([0, 1, 0.5j], [1 + 1j, 2, 3 - 1j], 'full')
            //array([ 0.0+0.j,  3.0+1.j,  1.5+1.5j,  1.0+0.j,  0.5+0.5j])

            dtype type = np.FindArrayType(asanyarray(o1), null);
            type = np.FindArrayType(asanyarray(o2), type);

            ndarray arr1 = np.FromAny(o1, type, 1, 1, NPYARRAYFLAGS.NPY_DEFAULT);
            ndarray arr2 = np.FromAny(o2, type, 1, 1, NPYARRAYFLAGS.NPY_DEFAULT);
            return NpyCoreApi.Correlate(arr1, arr2, type.TypeNum, mode);
        }

        #endregion

        #region maximum/minimum/fmax/fmin

        /// <summary>
        /// Element-wise maximum of array elements.
        /// </summary>
        /// <param name="x1">The arrays holding the elements to be compared.</param>
        /// <param name="x2">The arrays holding the elements to be compared.</param>
        /// <param name="out">A location into which the result is stored.</param>
        /// <param name="where">At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray maximum(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.maximum, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        /// <summary>
        /// Element-wise minimum of array elements.
        /// </summary>
        /// <param name="x1">The arrays holding the elements to be compared.</param>
        /// <param name="x2">The arrays holding the elements to be compared.</param>
        /// <param name="out">A location into which the result is stored.</param>
        /// <param name="where">At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray minimum(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.minimum, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        /// <summary>
        /// Element-wise maximum of array elements.
        /// </summary>
        /// <param name="x1">The arrays holding the elements to be compared.</param>
        /// <param name="x2">The arrays holding the elements to be compared.</param>
        /// <param name="out">A location into which the result is stored.</param>
        /// <param name="where">At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray fmax(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.fmax, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }
        /// <summary>
        /// Element-wise minimum of array elements.
        /// </summary>
        /// <param name="x1">The arrays holding the elements to be compared.</param>
        /// <param name="x2">The arrays holding the elements to be compared.</param>
        /// <param name="out">A location into which the result is stored.</param>
        /// <param name="where">At locations where the condition is True, the out array will be set to the ufunc result.</param>
        /// <returns></returns>
        public static ndarray fmin(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.fmin, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }


        #endregion

        #region heaviside

        /// <summary>
        /// Compute the Heaviside step function.
        /// </summary>
        /// <param name="x1">Input values.</param>
        /// <param name="x2">Input values.</param>
        /// <param name="out">A location into which the result is stored.</param>
        /// <param name="where">At locations where the condition is True, the out array will be set to the ufunc result. </param>
        /// <returns></returns>
        public static ndarray heaviside(object x1, object x2, ndarray @out = null, object where = null)
        {
            return NpyCoreApi.PerformUFUNC(UFuncOperation.heaviside, asanyarray(x1), asanyarray(x2), @out, asanyarray(where));
        }

        #endregion

        public static void putmask(ndarray arr, object mask, object values)
        {
            ndarray aMask;
            ndarray aValues;

            aMask = (mask as ndarray);
            if (aMask == null)
            {
                aMask = np.FromAny(mask, NpyCoreApi.DescrFromType(NPY_TYPES.NPY_BOOL),
                    0, 0, NPYARRAYFLAGS.NPY_CARRAY | NPYARRAYFLAGS.NPY_FORCECAST, null);
            }

            aValues = (values as ndarray);
            if (aValues == null)
            {
                aValues = np.FromAny(values, arr.Dtype, 0, 0, NPYARRAYFLAGS.NPY_CARRAY, null);
            }

            if (NpyCoreApi.PutMask(arr, aValues, aMask) < 0)
            {
                NpyCoreApi.CheckError();
            }

        }

        /// <summary>
        /// Copies values from one array to another, broadcasting as necessary.
        /// </summary>
        /// <param name="dst">The array into which values are copied.</param>
        /// <param name="src">The array from which values are copied.</param>
        /// <param name="casting">{‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional</param>
        /// <param name="where"> A boolean array which is broadcasted to match the dimensions of dst, and selects elements to copy from src to dst wherever it contains the value True.</param>
        public static void copyto(ndarray dst, object src, NPY_CASTING casting = NPY_CASTING.NPY_SAME_KIND_CASTING, object where = null)
        {
            /*
                Copies values from one array to another, broadcasting as necessary.
                Raises a TypeError if the casting rule is violated, and if where is provided, it selects which elements to copy.


                Parameters:
                    dst : ndarray
                        The array into which values are copied.
                    src : array_like
                        The array from which values are copied.
                    casting : {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional
                        Controls what kind of data casting may occur when copying.
                            ‘no’ means the data types should not be cast at all.
                            ‘equiv’ means only byte-order changes are allowed.
                            ‘safe’ means only casts which can preserve values are allowed.
                            ‘same_kind’ means only safe casts or casts within a kind, like float64 to float32, are allowed.
                            ‘unsafe’ means any data conversions may be done.
                    where : array_like of bool, optional
                        A boolean array which is broadcasted to match the dimensions of dst, and selects elements to copy from src to dst 
                        wherever it contains the value True.
             */
            NpyCoreApi.CopyTo(dst, asanyarray(src), casting, asanyarray(where));
        }


        public static ndarray_serializable ToSerializable(ndarray a)
        {
            return a.ToSerializable();
        }
        public static ndarray FromSerializable(ndarray_serializable sa)
        {
            return new ndarray(sa);
        }
        public static dtype_serializable ToSerializable(dtype a)
        {
            return a.ToSerializable();
        }
        public static dtype FromSerializable(dtype_serializable sa)
        {
            return new dtype(sa);
        }


        private static long CalculateNewShapeSize(shape shape)
        {
            if (shape.iDims != null)
            {
                long TotalBytes = 1;
                foreach (var dim in shape.iDims)
                {
                    if (dim < 0)
                    {
                        throw new Exception("new shape specification not properly formatted");
                    }
                    TotalBytes *= dim;
                }
                return TotalBytes;
            }

            throw new Exception("new shape specification not properly formatted");
        }

        private static NPY_ORDER CheckOnlyCorF(NPY_ORDER order)
        {
            NPY_ORDER NpyOrder = NPY_ORDER.NPY_ANYORDER;
            switch (order)
            {
                case NPY_ORDER.NPY_CORDER:
                    NpyOrder = NPY_ORDER.NPY_CORDER;
                    break;
                case NPY_ORDER.NPY_FORTRANORDER:
                    NpyOrder = NPY_ORDER.NPY_FORTRANORDER;
                    break;
                default:
                    throw new ArgumentException("order can only be C or F");

            }

            return NpyOrder;
        }

        private static NPY_ORDER ConvertOrder(ndarray src, NPY_ORDER order)
        {
            NPY_ORDER NpyOrder = NPY_ORDER.NPY_ANYORDER;
            switch (order)
            {
                case NPY_ORDER.NPY_CORDER:
                    NpyOrder = NPY_ORDER.NPY_CORDER;
                    break;
                case NPY_ORDER.NPY_FORTRANORDER:
                    NpyOrder = NPY_ORDER.NPY_FORTRANORDER;
                    break;
                case NPY_ORDER.NPY_ANYORDER:
                case NPY_ORDER.NPY_KORDER:
                    if (CheckNPYARRAYFLAGS(src, NPYARRAYFLAGS.NPY_FORTRAN))
                        NpyOrder = NPY_ORDER.NPY_FORTRANORDER;
                    else
                        NpyOrder = NPY_ORDER.NPY_CORDER;
                    break;
                default:
                    throw new ArgumentException("order can only be C or F, A or K");

            }

            return NpyOrder;
        }

        private static bool CheckNPYARRAYFLAGS(ndarray a, NPYARRAYFLAGS flags)
        {
            if ((a.Array.flags & flags) == flags)
                return true;
            return false;
        }

        private static NPY_TYPES DetermineArrayType<T>(T[] array, dtype dtype)
        {
            NPY_TYPES arrayType = NPY_TYPES.NPY_NOTYPE;

            if (dtype == null)
            {
                arrayType = Get_NPYType(array);
            }
            else
            {
                arrayType = dtype.TypeNum;
            }

            return arrayType;
        }


        private static VoidPtr GetDataPointer<T>(T[] array, NPY_TYPES arrayType, bool copy)
        {
            VoidPtr data = new VoidPtr(array, arrayType);
            if (copy)
            {
                data = GetArrayCopy(data);
            }
            return data;
        }

        private static VoidPtr GetArrayCopy(VoidPtr data)
        {
            return DefaultArrayHandlers.GetArrayHandler(data.type_num).GetArrayCopy(data);
        }

        private static NPY_TYPES Get_NPYTypeFromEmptyArray(System.Array ssrc)
        {
            string typeName = ssrc.GetType().ToString();
            int arrayMarkerIndex = typeName.IndexOf("[");
            if (arrayMarkerIndex >= 0)
            {
                typeName = typeName.Substring(0, arrayMarkerIndex);
            }

            switch (typeName)
            {
                case "System.Boolean":
                    return NPY_TYPES.NPY_BOOL;
                case "System.Byte":
                    return NPY_TYPES.NPY_UBYTE;
                case "System.SByte":
                    return NPY_TYPES.NPY_BYTE;
                case "System.Int16":
                    return NPY_TYPES.NPY_INT16;
                case "System.UInt16":
                    return NPY_TYPES.NPY_UINT16;
                case "System.Int32":
                    return NPY_TYPES.NPY_INT32;
                case "System.UInt32":
                    return NPY_TYPES.NPY_UINT32;
                case "System.Int64":
                    return NPY_TYPES.NPY_INT64;
                case "System.UInt64":
                    return NPY_TYPES.NPY_UINT64;
                case "System.Single":
                    return NPY_TYPES.NPY_FLOAT;
                case "System.Double":
                    return NPY_TYPES.NPY_DOUBLE;
                case "System.Decimal":
                    return NPY_TYPES.NPY_DECIMAL;
                case "System.Numerics.Complex":
                    return NPY_TYPES.NPY_COMPLEX;
                case "System.Numerics.BigInteger":
                    return NPY_TYPES.NPY_BIGINT;
                case "System.Object":
                    return NPY_TYPES.NPY_OBJECT;
                case "System.String":
                    return NPY_TYPES.NPY_STRING;
                default:
                    return NPY_TYPES.NPY_OBJECT;
            }
        }
        private static NPY_TYPES Get_NPYType<T>(T[] _Array)
        {
            if (typeof(T) == typeof(Object))
            {
                return NPY_TYPES.NPY_OBJECT;
            }

            if (_Array.Length == 0)
            {
                return Get_NPYTypeFromEmptyArray(_Array);
            }

            return Get_NPYType(_Array[0]);
        }

        private static NPY_TYPES Get_NPYType(object obj)
        {
            NPY_TYPES NumType = DefaultArrayHandlers.GetArrayType(obj);
            if (NumType != NPY_TYPES.NPY_NOTSET)
            {
                return NumType;
            }
            return NPY_TYPES.NPY_OBJECT; // Our data type registration system is internal anyways, inextensible from outside. We can default to using object
            throw new Exception("This data type is not registered with the numpy system");
        }

        private static int normalize_axis_index(int axis, int ndim)
        {

            if (axis < -ndim || axis >= ndim)
            {
                throw new Exception("AxisError");
            }

            if (axis < 0)
            {
                axis += ndim;
            }
            return axis;
        }

        private static int len(ndarray array)
        {
            return (int)array.Dim(0);
        }
        private static int len(int []arr)
        {
            return arr.Length;
        }
        private static int len(Int64[] arr)
        {
            return arr.Length;
        }

        private static List<npy_intp> list(npy_intp[] array)
        {
            return array.ToList();
        }

        private static object[] BuildSliceArray(Slice Copy, int n)
        {
            object[] result = new object[n];

            for (int i = 0; i < n; i++)
            {
                result[i] = new Slice(Copy.start, Copy.stop, Copy.step);
            }

            return result;
        }
   

        class zip2IntArrays
        {
            public List<Tuple<int, int>> data = new List<Tuple<int, int>>();

            public zip2IntArrays(int[] d, int[] s)
            {
                if (d.Length != s.Length)
                    throw new Exception("Must be same sized arrays");

                for (int i = 0; i < d.Length; i++)
                {
                    data.Add(new Tuple<int, int>(d[i], s[i]));
                }
            }

            public void Sort()
            {
                data.Sort((c1, c2) =>
                {

                    if (c1.Item1 > c2.Item1)
                        return 1;

                    if (c1.Item1 < c2.Item1)
                        return -1;

                    return 0;
                });
            }
        }

    }
}
