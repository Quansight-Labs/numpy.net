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
using System.Threading.Tasks;
using NumpyLib;


#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNetTests
{
    class Common
    {
        public static int GeneratedArrayLength = 16;


        public static void CommonInit()
        {
            NpyArray_FunctionDefs functionDefs = null;

            NpyInterface_WrapperFuncs wrapperFuncs = new NpyInterface_WrapperFuncs()
            {
                array_new_wrapper = numpy_interface_array_new_wrapper,
                iter_new_wrapper = numpy_interface_iter_new_wrapper,
                multi_iter_new_wrapper = numpy_interface_multi_iter_new_wrapper,
                neighbor_iter_new_wrapper = numpy_interface_neighbor_iter_new_wrapper,
                descr_new_from_type = numpy_interface_descr_new_from_type,
                descr_new_from_wrapper = numpy_interface_descr_new_from_wrapper,
            };

            npy_tp_error_set error_set = ErrorSet_handler;
            npy_tp_error_occurred error_occurred = ErrorOccurred_handler;
            npy_tp_error_clear error_clear = ErrorClear_handler;
            npy_tp_cmp_priority cmp_priority = numpy_tp_cmp_priority;
            npy_interface_incref incref = numpy_interface_incref;
            npy_interface_decref decref = numpy_interface_decref;
            enable_threads et = enable_threads;
            disable_threads dt = disable_threads;

            numpyAPI.npy_initlib(functionDefs, wrapperFuncs, error_set, error_occurred, error_clear, cmp_priority, incref, decref, et, dt);
        }

        public class NumpyExceptionInfo
        {
            public string FunctionName;
            public npyexc_type exctype;
            public string error;
        }

        public static List<NumpyExceptionInfo> NumpyErrors = new List<NumpyExceptionInfo>();

        static void ErrorSet_handler(string FunctionName, npyexc_type et, string error)
        {
            if (et == npyexc_type.NpyExc_DotNetException)
            {
                throw new Exception("Got an unexpected .NET exception");
            }
            NumpyErrors.Add(new NumpyExceptionInfo() { FunctionName = FunctionName, exctype = et, error = error });
            return;
        }

        static bool ErrorOccurred_handler(string FunctionName)
        {
            return false;
        }


        static void ErrorClear_handler(string FunctionName)
        {

        }

        static bool numpy_interface_array_new_wrapper(NpyArray newArray, bool ensureArray, bool customStrides, object subtype, object interfaceData, ref object interfaceRet)
        {
            return true;
        }
        static bool numpy_interface_iter_new_wrapper(NpyArrayIterObject iter, ref object interfaceRet)
        {
            return true;
        }
        static bool numpy_interface_multi_iter_new_wrapper(NpyArrayMultiIterObject iter, ref object interfaceRet)
        {
            return true;
        }
        static bool numpy_interface_neighbor_iter_new_wrapper(NpyArrayNeighborhoodIterObject iter, ref object interfaceRet)
        {
            return true;
        }
        static bool numpy_interface_descr_new_from_type(int type, NpyArray_Descr descr, ref object interfaceRet)
        {
            return true;
        }
        static bool numpy_interface_descr_new_from_wrapper(object _base, NpyArray_Descr descr, ref object interfaceRet)
        {
            return true;
        }
        static bool numpy_interface_ufunc_new_wrapper(object _base, ref object interfaceRet)
        {
            return true;
        }

        static int numpy_tp_cmp_priority(object o1, object o2)
        {
            return 0;
        }

        static object numpy_interface_incref(object o1, ref object o2)
        {
            return null;
        }
        static object numpy_interface_decref(object o1, ref object o2)
        {
            return null;
        }
        static object enable_threads()
        {
            return null;
        }
        static object disable_threads(object o1)
        {
            return null;
        }

        public static void DefaultCastFunction(VoidPtr Src, VoidPtr Dest, int srclen, NpyArray srcArray, NpyArray destArray)
        {
            numpyAPI.DefaultCastFunction(Src, Dest, srclen, srcArray, destArray);
            return;
        }


        public static bool MatchError(string ErrorText)
        {
            return NumpyErrors.Any(t => t.error.Contains(ErrorText));
        }

        public static bool MatchError(npyexc_type et, string ErrorText)
        {
            return NumpyErrors.Any(t => t.exctype == et && t.error.Contains(ErrorText));
        }

        public static bool CompareArrays(NpyArray T1, NpyArray T2)
        {
            return CompareArrays(T1.data, T2.data);
        }
        public static bool CompareArrays(VoidPtr T1, VoidPtr T2)
        {
            if (T1.type_num == NPY_TYPES.NPY_BOOL && T2.type_num == NPY_TYPES.NPY_BOOL)
            {
                return CompareArrays(T1.datap as bool[], T2.datap as bool[]);
            }
            if (T1.type_num == NPY_TYPES.NPY_BYTE && T2.type_num == NPY_TYPES.NPY_BYTE)
            {
                return CompareArrays(T1.datap as sbyte[], T2.datap as sbyte[]);
            }
            if (T1.type_num == NPY_TYPES.NPY_UBYTE && T2.type_num == NPY_TYPES.NPY_UBYTE)
            {
                return CompareArrays(T1.datap as byte[], T2.datap as byte[]);
            }
            if (T1.type_num == NPY_TYPES.NPY_UBYTE && T2.type_num == NPY_TYPES.NPY_UBYTE)
            {
                return CompareArrays(T1.datap as byte[], T2.datap as byte[]);
            }
            if (T1.type_num == NPY_TYPES.NPY_INT16 && T2.type_num == NPY_TYPES.NPY_INT16)
            {
                return CompareArrays(T1.datap as Int16[], T2.datap as Int16[]);
            }
            if (T1.type_num == NPY_TYPES.NPY_UINT16 && T2.type_num == NPY_TYPES.NPY_UINT16)
            {
                return CompareArrays(T1.datap as UInt16[], T2.datap as UInt16[]);
            }
            if (T1.type_num == NPY_TYPES.NPY_INT32 && T2.type_num == NPY_TYPES.NPY_INT32)
            {
                return CompareArrays(T1.datap as Int32[], T2.datap as Int32[]);
            }
            if (T1.type_num == NPY_TYPES.NPY_UINT32 && T2.type_num == NPY_TYPES.NPY_UINT32)
            {
                return CompareArrays(T1.datap as UInt32[], T2.datap as UInt32[]);
            }
            if (T1.type_num == NPY_TYPES.NPY_INT64 && T2.type_num == NPY_TYPES.NPY_INT64)
            {
                return CompareArrays(T1.datap as Int64[], T2.datap as Int64[]);
            }
            if (T1.type_num == NPY_TYPES.NPY_UINT64 && T2.type_num == NPY_TYPES.NPY_UINT64)
            {
                return CompareArrays(T1.datap as UInt64[], T2.datap as UInt64[]);
            }
            if (T1.type_num == NPY_TYPES.NPY_FLOAT && T2.type_num == NPY_TYPES.NPY_FLOAT)
            {
                return CompareArrays(T1.datap as float[], T2.datap as float[]);
            }
            if (T1.type_num == NPY_TYPES.NPY_DOUBLE && T2.type_num == NPY_TYPES.NPY_DOUBLE)
            {
                return CompareArrays(T1.datap as double[], T2.datap as double[]);
            }
            if (T1.type_num == NPY_TYPES.NPY_DECIMAL && T2.type_num == NPY_TYPES.NPY_DECIMAL)
            {
                return CompareArrays(T1.datap as decimal[], T2.datap as decimal[]);
            }
            return false;
        }


        public static bool CompareArrays<T>(T[] simpleData, T[] scalarData)
        {
            //if (simpleData.Length != scalarData.Length)
            //    return false;

            for (int i = 0; i < simpleData.Length; i++)
            {
                T s1 = simpleData[i];
                T s2 = scalarData[i];
                if (!s1.Equals(s2))
                    return false;
            }
            return true;
        }


        public static NpyArray GetSimpleArray(NPY_TYPES npy_type, ref int DataSize, UInt64 FillData = 0, bool UseFillData = false, bool UseMaxValue = false)
        {
            int dataLen = GeneratedArrayLength;
            int itemsize = GetDefaultItemSize(npy_type);

            var data = GetArrayOfData(npy_type, dataLen, FillData, UseFillData, UseMaxValue);

            int subtype = 1;

            npy_intp[] dimensions = new npy_intp[] { dataLen };


            var npArray = numpyAPI.NpyArray_New(subtype, dimensions.Length, dimensions, npy_type, null, data, itemsize, NPYARRAYFLAGS.NPY_DEFAULT, null);

            DataSize = dataLen;
            return npArray;
        }

        public static NpyArray GetOneSegmentArray(NPY_TYPES npy_type, ref int DataSize, UInt64 FillData = 0, bool UseFillData = false, bool UseMaxValue = false)
        {
            int dataLen = GeneratedArrayLength;
            int itemsize = GetDefaultItemSize(npy_type);

            var data = GetArrayOfData(npy_type, dataLen, FillData, UseFillData, UseMaxValue);

            int subtype = 1;

            npy_intp[] dimensions = new npy_intp[] { dataLen };


            var npArray = numpyAPI.NpyArray_New(subtype, 0, null, npy_type, null, data, itemsize, NPYARRAYFLAGS.NPY_DEFAULT, null);

            DataSize = dataLen;
            return npArray;
        }


        public static NpyArray GetComplexArray2D(NPY_TYPES npy_type, ref int DataSize, int a, int b, UInt64 FillData = 0, bool UseFillData = false, bool UseMaxValue = false)
        {
            int dataLen = a * b;
            int itemsize = GetDefaultItemSize(npy_type);

            var data = GetArrayOfData(npy_type, dataLen, FillData, UseFillData, UseMaxValue);

            int subtype = 1;

            npy_intp[] dimensions = new npy_intp[] {a, b};
            
            var npArray = numpyAPI.NpyArray_New(subtype, dimensions.Length, dimensions, npy_type, null, data, itemsize, NPYARRAYFLAGS.NPY_DEFAULT, null);

            DataSize = dataLen;
            return npArray;
        }

        public static NpyArray GetComplexArray3D(NPY_TYPES npy_type, ref int DataSize, int a, int b, int c,  UInt64 FillData = 0, bool UseFillData = false, bool UseMaxValue = false)
        {
            int dataLen = a * b * c;
            int itemsize = GetDefaultItemSize(npy_type);

            var data = GetArrayOfData(npy_type, dataLen, FillData, UseFillData, UseMaxValue);

            int subtype = 1;

            npy_intp[] dimensions = new npy_intp[] { a, b, c };


            var npArray = numpyAPI.NpyArray_New(subtype, dimensions.Length, dimensions, npy_type, null, data, itemsize, NPYARRAYFLAGS.NPY_DEFAULT, null);

            DataSize = dataLen;
            return npArray;
        }

        public static NpyArray GetComplexArray4D(NPY_TYPES npy_type, ref int DataSize, int a, int b, int c, int d, UInt64 FillData = 0, bool UseFillData = false, bool UseMaxValue = false)
        {
            int dataLen = a * b * c * d;
            int itemsize = GetDefaultItemSize(npy_type);

            var data = GetArrayOfData(npy_type, dataLen, FillData, UseFillData, UseMaxValue);

            int subtype = 1;

            npy_intp[] dimensions = new npy_intp[] { a, b, c, d };


            var npArray = numpyAPI.NpyArray_New(subtype, dimensions.Length, dimensions, npy_type, null, data, itemsize, NPYARRAYFLAGS.NPY_DEFAULT, null);

            DataSize = dataLen;
            return npArray;
        }

        public static NpyArray GetComplexArray5D(NPY_TYPES npy_type, ref int DataSize, int a, int b, int c, int d, int e, UInt64 FillData = 0, bool UseFillData = false, bool UseMaxValue = false)
        {
            int dataLen = a * b * c * d * e;
            int itemsize = GetDefaultItemSize(npy_type);

            var data = GetArrayOfData(npy_type, dataLen, FillData, UseFillData, UseMaxValue);

            int subtype = 1;

            npy_intp[] dimensions = new npy_intp[] { a, b, c, d, e };


            var npArray = numpyAPI.NpyArray_New(subtype, dimensions.Length, dimensions, npy_type, null, data, itemsize, NPYARRAYFLAGS.NPY_DEFAULT, null);

            DataSize = dataLen;
            return npArray;
        }

        public static int ArrayDataAdjust = 0;
        public static VoidPtr GetArrayOfData(NPY_TYPES item_type, int dataLen, UInt64 FillData = 0, bool UseFillData = false, bool UseMaxValue = false)
        {
            VoidPtr data = null;

            switch (item_type)
            {
                case NPY_TYPES.NPY_BOOL:
                    {
                        bool[] bdata = new bool[dataLen];

                        for (int i = 0; i < bdata.Length; i++)
                        {
                            if (UseFillData)
                            {
                                bdata[i] = i%2 != 0 ? true : false;
                            }
                            else
                            {
                                bdata[i] = i % 2 != 0 ? true : false;
                            }
                        }

                        data = new VoidPtr(bdata);
                        break;
                    }
                case NPY_TYPES.NPY_BYTE:
                    {
                        sbyte[] bdata = new sbyte[dataLen];

                        for (int i = 0; i < bdata.Length; i++)
                        {
                            if (UseFillData)
                            {
                                if (UseMaxValue)
                                {
                                    bdata[i] = sbyte.MaxValue;
                                }
                                else
                                {
                                    bdata[i] = (sbyte)FillData;
                                }
                            }
                            else
                            {
                                bdata[i] = (sbyte)(i + ArrayDataAdjust);
                            }
                        }

                        data = new VoidPtr(bdata);
                        break;
                    }
                case NPY_TYPES.NPY_UBYTE:
                    {
                        byte[] bdata = new byte[dataLen];

                        for (int i = 0; i < bdata.Length; i++)
                        {
                            if (UseFillData)
                            {
                                if (UseMaxValue)
                                {
                                    bdata[i] = byte.MaxValue;
                                }
                                else
                                {
                                    bdata[i] = (byte)FillData;
                                }
                            }
                            else
                            {
                                bdata[i] = (byte)(i + ArrayDataAdjust);
                            }
                        }

                        data = new VoidPtr(bdata);
                        break;
                    }
 

                case NPY_TYPES.NPY_INT16:
                    {
                        Int16[] bdata = new Int16[dataLen];

                        for (int i = 0; i < bdata.Length; i++)
                        {
                            if (UseFillData)
                            {
                                if (UseMaxValue)
                                {
                                    bdata[i] = Int16.MaxValue;
                                }
                                else
                                {
                                    bdata[i] = (Int16)FillData;
                                }
                            }
                            else
                            {
                                bdata[i] = (Int16)(i + ArrayDataAdjust);
                            }
                        }

                        data = new VoidPtr(bdata);
                        break;
                    }
 
                case NPY_TYPES.NPY_UINT16:
                    {
                        UInt16[] bdata = new UInt16[dataLen];

                        for (int i = 0; i < bdata.Length; i++)
                        {
                            if (UseFillData)
                            {
                                if (UseMaxValue)
                                {
                                    bdata[i] = UInt16.MaxValue;
                                }
                                else
                                {
                                    bdata[i] = (UInt16)FillData;
                                }
                            }
                            else
                            {
                                bdata[i] = (UInt16)(i + ArrayDataAdjust);
                            }
                        }

                        data = new VoidPtr(bdata);
                        break;
                    }
                case NPY_TYPES.NPY_INT32:
                    {
                        Int32[] bdata = new Int32[dataLen];

                        for (int i = 0; i < bdata.Length; i++)
                        {
                            if (UseFillData)
                            {
                                if (UseMaxValue)
                                {
                                    bdata[i] = Int32.MaxValue;
                                }
                                else
                                {
                                    bdata[i] = (Int32)FillData;
                                }
                            }
                            else
                            {
                                bdata[i] = (Int32)(i + ArrayDataAdjust);
                            }
                        }

                        data = new VoidPtr(bdata);
                        break;
                    }
                case NPY_TYPES.NPY_UINT32:
                    {
                        UInt32[] bdata = new UInt32[dataLen];

                        for (int i = 0; i < bdata.Length; i++)
                        {
                            if (UseFillData)
                            {
                                if (UseMaxValue)
                                {
                                    bdata[i] = UInt32.MaxValue;
                                }
                                else
                                {
                                    bdata[i] = (UInt32)FillData;
                                }
                            }
                            else
                            {
                                bdata[i] = (UInt32)(i + ArrayDataAdjust);
                            }
                        }

                        data = new VoidPtr(bdata);
                        break;
                    }
                case NPY_TYPES.NPY_INT64:
                    {
                        Int64[] bdata = new Int64[dataLen];

                        for (int i = 0; i < bdata.Length; i++)
                        {
                            if (UseFillData)
                            {
                                if (UseMaxValue)
                                {
                                    bdata[i] = Int64.MaxValue;
                                }
                                else
                                {
                                    bdata[i] = (Int64)FillData;
                                }
                            }
                            else
                            {
                                bdata[i] = (Int64)(i + ArrayDataAdjust);
                            }
                        }

                        data = new VoidPtr(bdata);
                        break;
                    }
                case NPY_TYPES.NPY_UINT64:
                    {
                        UInt64[] bdata = new UInt64[dataLen];

                        for (int i = 0; i < bdata.Length; i++)
                        {
                            if (UseFillData)
                            {
                                if (UseMaxValue)
                                {
                                    bdata[i] = UInt64.MaxValue;
                                }
                                else
                                {
                                    bdata[i] = (UInt64)FillData;
                                }
                            }
                            else
                            {
                                bdata[i] = (UInt64)(i + ArrayDataAdjust);
                            }
                        }

                        data = new VoidPtr(bdata);
                        break;
                    }
                case NPY_TYPES.NPY_FLOAT:
                    {
                        float[] bdata = new float[dataLen];

                        for (int i = 0; i < bdata.Length; i++)
                        {
                            if (UseFillData)
                            {
                                if (UseMaxValue)
                                {
                                    bdata[i] = float.MaxValue;
                                }
                                else
                                {
                                    bdata[i] = (float)FillData;
                                }
                            }
                            else
                            {
                                bdata[i] = (float)(i + ArrayDataAdjust);
                            }
                        }

                        data = new VoidPtr(bdata);
                        break;
                    }
                case NPY_TYPES.NPY_DOUBLE:
                    {
                        double[] bdata = new double[dataLen];

                        for (int i = 0; i < bdata.Length; i++)
                        {
                            if (UseFillData)
                            {
                                if (UseMaxValue)
                                {
                                    bdata[i] = double.MaxValue;
                                }
                                else
                                {
                                    bdata[i] = (double)FillData;
                                }
                            }
                            else
                            {
                                bdata[i] = (double)(i + ArrayDataAdjust);
                            }
                        }

                        data = new VoidPtr(bdata);
                        break;
                    }
                case NPY_TYPES.NPY_DECIMAL:
                    {
                        decimal[] bdata = new decimal[dataLen];

                        for (int i = 0; i < bdata.Length; i++)
                        {
                            if (UseFillData)
                            {
                                if (UseMaxValue)
                                {
                                    bdata[i] = decimal.MaxValue;
                                }
                                else
                                {
                                    bdata[i] = (decimal)FillData;
                                }
                            }
                            else
                            {
                                bdata[i] = (decimal)(i + ArrayDataAdjust);
                            }
                        }

                        data = new VoidPtr(bdata);
                        break;
                    }
  

                case NPY_TYPES.NPY_COMPLEX:
                    {
                        UInt64[] bdata = new UInt64[dataLen];

                        for (int i = 0; i < bdata.Length; i++)
                        {
                            if (UseFillData)
                            {
                                bdata[i] = (UInt64)FillData;
                            }
                            else
                            {
                                bdata[i] = (UInt64)(i + ArrayDataAdjust);
                            }
                        }

                        data = new VoidPtr(bdata);
                        break;
                    }

                default:
                    throw new Exception(string.Format("GetArrayOfData: Unexpected item_type {0}", item_type));

            }

            data.type_num = item_type;
            return data;
        }

        static int NotSupportedSizeYet = -1;


        internal static int GetDefaultItemSize(NPY_TYPES npy_type)
        {
            switch (npy_type)
            {
                case NPY_TYPES.NPY_BOOL:
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_UBYTE:
                    return 1;
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_UINT16:
                    return 2;
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_UINT32:
                    return 4;
                case NPY_TYPES.NPY_INT64:
                case NPY_TYPES.NPY_UINT64:
                    return 8;
                case NPY_TYPES.NPY_FLOAT:
                    return 4;

                case NPY_TYPES.NPY_DOUBLE:
                    return 8;

                case NPY_TYPES.NPY_DECIMAL:
                    return sizeof(decimal);


                case NPY_TYPES.NPY_USERDEF:
                    return NotSupportedSizeYet;
                default:
                    return NotSupportedSizeYet;

            }
    

        }


        internal static void GCClear()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }



    }
}
