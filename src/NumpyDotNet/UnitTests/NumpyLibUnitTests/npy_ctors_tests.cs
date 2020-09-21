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
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyLib;
using System.Collections.Generic;
using System.Linq;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyLibTests
{
    [TestClass]
    public class npy_ctors_tests
    {
        [ClassInitialize]
        public static void CommonInit(TestContext t)
        {
            Common.CommonInit();
        }

        [TestInitialize]
        public void FunctionInit()
        {
            Common.NumpyErrors.Clear();
        }


        [TestMethod]
        public void NpyArray_New__Test1()
        {
            int subtype = 1;

            npy_intp[] dimensions = new npy_intp[1]; // {2,8};
            byte[] data = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

            dimensions[0] = data.Length;
            var npArray = numpyAPI.NpyArray_New(subtype, dimensions.Length, dimensions, NPY_TYPES.NPY_UINT16, null, new VoidPtr(data), 2, NPYARRAYFLAGS.NPY_DEFAULT, null);

            NpyArray_Dims newDims = new NpyArray_Dims();
            newDims.ptr = new npy_intp[] { 4, 4 };
            newDims.len = 2;

            numpyAPI.NpyArray_SetShape(npArray, newDims);

            return;
        }


        [TestMethod]
        public void npy_byte_swap_vector_test1()
        {
            for (int i = 1; i <= 8; i++)
            {
                Int32[] data = new Int32[16];
                VoidPtr p = new VoidPtr();
                p.datap = data;
                p.type_num = NPY_TYPES.NPY_INT32;

                numpyAPI.npy_byte_swap_vector(p, i, i);
            }

        }

        [TestMethod]
        public void npy_byte_swap_vector_test2()
        {
            for (int i = 1; i <= 8; i++)
            {
                byte[] data = new byte[64];
                VoidPtr p = new VoidPtr();
                p.datap = data;
                p.type_num = NPY_TYPES.NPY_BYTE;

                numpyAPI.npy_byte_swap_vector(p, i, i);
            }

        }

        [TestMethod]
        public void npy_byte_swap_vector_test3()
        {
            for (int i = 1; i <= 8; i++)
            {
                Int16[] data = new Int16[32];
                VoidPtr p = new VoidPtr();
                p.datap = data;
                p.type_num = NPY_TYPES.NPY_INT16;

                numpyAPI.npy_byte_swap_vector(p, i, i);
            }

        }

        [TestMethod]
        public void npy_byte_swap_vector_test4()
        {
            for (int i = 1; i <= 8; i++)
            {
                Int64[] data = new Int64[16];
                VoidPtr p = new VoidPtr();
                p.datap = data;
                p.type_num = NPY_TYPES.NPY_INT64;

                numpyAPI.npy_byte_swap_vector(p, i, i);
            }

        }

        private void npy_byte_swap_common(VoidPtr vp, int size)
        {
            numpyAPI.npy_byte_swap_vector(vp, size, size);
        }

        [Ignore] // obsolete.  doesn't include recent data types.  Covered by tests in NumpyDotNetTests
        [TestMethod]
        public void ConvertToDesiredArrayType_Test1()
        {
            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES targetType in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(targetType);
                if (ExpectedSize < 0)
                    continue;

                ConvertToDesiredArrayType_Test(targetType);
            }

            return;
        }

        private void ConvertToDesiredArrayType_Test(NPY_TYPES targetType)
        {
            NpyArray SimpleArray = null;

            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES desiredType in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(desiredType);
                if (ExpectedSize < 0)
                    continue;

                int DataSize = 0;
                SimpleArray = Common.GetSimpleArray(targetType, ref DataSize);

                DataSize *= Common.GetDefaultItemSize(targetType);

                VoidPtr nvp1 = numpyAPI.ConvertToDesiredArrayType(SimpleArray.data, 0, DataSize, desiredType);
                VoidPtr nvp2 = numpyAPI.ConvertToDesiredArrayType(nvp1, 0, DataSize, targetType);

                if (desiredType != NPY_TYPES.NPY_BOOL  && desiredType != NPY_TYPES.NPY_DECIMAL && targetType != NPY_TYPES.NPY_DECIMAL)
                {
                    Assert.IsTrue(Common.CompareArrays(SimpleArray.data, nvp2));
                }
            }

            return;
        }

        [TestMethod]
        public void MemCopy_Test1()
        {

            Int32[] SourceArray = new Int32[16];
            for (int i = 0; i < SourceArray.Length; i++)
            {
                SourceArray[i] = Convert.ToInt32((0x01020304 << (i & 0x3) & 0x7FFFFFFF));
            }

            Int32[] DestArray1 = new Int32[16];

            bool bResult = numpyAPI.MemCpy(new VoidPtr(DestArray1), 0, new VoidPtr(SourceArray), 2, (SourceArray.Length * sizeof(Int32)) - 2);


            return;
        }


        [Ignore] // obsolete.  doesn't handle recent data types
        [TestMethod]
        public void flat_copyinto_OneSegmentArray_Test()
        {
            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES targetType in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(targetType);
                if (ExpectedSize < 0)
                    continue;

                int DataSize = 0;
                var TargetArray = Common.GetOneSegmentArray(targetType, ref DataSize, 0, true, false);

                var SrcArray = Common.GetOneSegmentArray(targetType, ref DataSize);

                int result1 = numpyAPI._flat_copyinto(TargetArray, SrcArray, NPY_ORDER.NPY_ANYORDER);
                Assert.AreEqual(0, result1);
                if (targetType != NPY_TYPES.NPY_BOOL)
                {
                    Assert.IsTrue(Common.CompareArrays(SrcArray, TargetArray));
                }

                int result2 = numpyAPI._flat_copyinto(TargetArray, SrcArray, NPY_ORDER.NPY_CORDER);
                Assert.AreEqual(0, result2);
                if (targetType != NPY_TYPES.NPY_BOOL)
                {
                    Assert.IsTrue(Common.CompareArrays(SrcArray, TargetArray));
                }

                int result3 = numpyAPI._flat_copyinto(TargetArray, SrcArray, NPY_ORDER.NPY_FORTRANORDER);
                Assert.AreEqual(0, result3);
                if (targetType != NPY_TYPES.NPY_BOOL)
                {
                    Assert.IsTrue(Common.CompareArrays(SrcArray, TargetArray));
                }

            }

            return;
        }

        [Ignore] // obsolete.  doesn't handle recent data types
        [TestMethod]
        public void flat_copyinto_SimpleArray_Test()
        {
            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES targetType in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(targetType);
                if (ExpectedSize < 0)
                    continue;

                int DataSize = 0;
                var TargetArray = Common.GetSimpleArray(targetType, ref DataSize, 0, true, false);

                var SrcArray = Common.GetSimpleArray(targetType, ref DataSize);

                int result1 = numpyAPI._flat_copyinto(TargetArray, SrcArray, NPY_ORDER.NPY_ANYORDER);
                Assert.AreEqual(0, result1);
                if (targetType != NPY_TYPES.NPY_BOOL && targetType != NPY_TYPES.NPY_DECIMAL)
                {
                    Assert.IsTrue(Common.CompareArrays(SrcArray, TargetArray));
                }

                int result2 = numpyAPI._flat_copyinto(TargetArray, SrcArray, NPY_ORDER.NPY_CORDER);
                Assert.AreEqual(0, result2);
                if (targetType != NPY_TYPES.NPY_BOOL && targetType != NPY_TYPES.NPY_DECIMAL)
                {
                    Assert.IsTrue(Common.CompareArrays(SrcArray, TargetArray));
                }

                int result3 = numpyAPI._flat_copyinto(TargetArray, SrcArray, NPY_ORDER.NPY_FORTRANORDER);
                Assert.AreEqual(0, result3);
                if (targetType != NPY_TYPES.NPY_BOOL && targetType != NPY_TYPES.NPY_DECIMAL)
                {
                    Assert.IsTrue(Common.CompareArrays(SrcArray, TargetArray));
                }

            }

            return;
        }

        [Ignore] // obsolete.  doesn't handle recent data types
        [TestMethod]
        public void flat_copyinto_ComplexArray2D_Test()
        {
            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES targetType in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(targetType);
                if (ExpectedSize < 0)
                    continue;

                int DataSize = 0;
                var TargetArray = Common.GetSimpleArray(targetType, ref DataSize, 0, true, false);


                int Dim1 = 4;
                var SrcArray = Common.GetComplexArray2D(targetType, ref DataSize, 4,Common.GeneratedArrayLength/ Dim1);

                int result1 = numpyAPI._flat_copyinto(TargetArray, SrcArray, NPY_ORDER.NPY_ANYORDER);
                Assert.AreEqual(0, result1);
                if (targetType != NPY_TYPES.NPY_BOOL && targetType != NPY_TYPES.NPY_DECIMAL)
                {
                    Assert.IsTrue(Common.CompareArrays(SrcArray, TargetArray));
                }

                int result2 = numpyAPI._flat_copyinto(TargetArray, SrcArray, NPY_ORDER.NPY_CORDER);
                Assert.AreEqual(0, result2);
                if (targetType != NPY_TYPES.NPY_BOOL && targetType != NPY_TYPES.NPY_DECIMAL)
                {
                    Assert.IsTrue(Common.CompareArrays(SrcArray, TargetArray));
                }

                int result3 = numpyAPI._flat_copyinto(TargetArray, SrcArray, NPY_ORDER.NPY_FORTRANORDER);
                Assert.AreEqual(0, result3);
                if (targetType != NPY_TYPES.NPY_BOOL && targetType != NPY_TYPES.NPY_DECIMAL)
                {
                    Assert.IsFalse(Common.CompareArrays(SrcArray, TargetArray));
                }

            }

            return;
        }

        [Ignore] // obsolete
        [TestMethod]
        public void NpyArray_CheckFromArray_Test()
        {
            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            NPY_TYPES targetType = NPY_TYPES.NPY_INT32;

            foreach (NPY_TYPES srcType in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(srcType);
                if (ExpectedSize < 0)
                    continue;



                int DataSize = 0;
                var DataArray = Common.GetSimpleArray(srcType, ref DataSize, 0, true, false);


                var srcDescr = numpyAPI.NpyArray_DescrNewFromType(srcType);

                var CheckArray = numpyAPI.NpyArray_CheckFromArray(DataArray, srcDescr, NPYARRAYFLAGS.NPY_CARRAY | NPYARRAYFLAGS.NPY_ENSUREARRAY | NPYARRAYFLAGS.NPY_ENSURECOPY);

                Assert.IsNotNull(CheckArray);

                Assert.AreEqual(srcType, CheckArray.ItemType);

                if (srcType != NPY_TYPES.NPY_BOOL)
                {
                    Assert.IsTrue(Common.CompareArrays(CheckArray, DataArray));
                }


            }

        }




        [TestMethod]
        public void NpyArray_ConversionTest()
        {
            int subtype = 1;

            int[] dimensions = new int[1]; // {2,8};
            byte[] data = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

            var i32Data = ConvertToInt32(data);

            //i32Data[1] = 0x7FFFFFFF;

            var bAgain = ConvertToByte(i32Data);

            i32Data[5] = 99;


            return;
        }

        private static int[] ConvertToInt32(byte[] data)
        {
            var AdjustedArray = Array.ConvertAll<byte, int>(data, Convert.ToInt32);
            return AdjustedArray;
        }

        private static byte[] ConvertToByte(Int32[] data)
        {
            var AdjustedArray = Array.ConvertAll<Int32, byte>(data, Convert.ToByte);
            return AdjustedArray;
        }

    }
}
