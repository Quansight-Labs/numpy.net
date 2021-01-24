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

namespace NumpyLibTests
{
    [TestClass]
    public class npy_convert_tests
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
            Common.ArrayDataAdjust = 0;
        }

        [TestMethod]
        public void NpyArray_View_Test1()
        {
            int DataSize = 0;

            var SimpleArray = Common.GetSimpleArray(NPY_TYPES.NPY_INT32, ref DataSize);
            Assert.AreEqual(NPY_TYPES.NPY_INT32,SimpleArray.ItemType);

            var newDescr = numpyAPI.NpyArray_DescrNewFromType(NPY_TYPES.NPY_BYTE);
 

            var newArray = numpyAPI.NpyArray_View(SimpleArray, newDescr, 1);
            Assert.AreEqual(NPY_TYPES.NPY_BYTE, newArray.ItemType);
            Assert.AreEqual(Common.GetDefaultItemSize(SimpleArray.data.type_num), 4);
            Assert.AreEqual(Common.GetDefaultItemSize(newArray.data.type_num), 4);

        }



        [TestMethod]
        public void NpyArray_NewCopy_Test1()
        {
            int DataSize = 0;
            
            var SimpleArray = Common.GetSimpleArray(NPY_TYPES.NPY_INT32, ref DataSize);
            Assert.AreEqual(NPY_TYPES.NPY_INT32, SimpleArray.ItemType);

            var newArray = numpyAPI.NpyArray_NewCopy(SimpleArray, NPY_ORDER.NPY_ANYORDER);
            Assert.AreEqual(NPY_TYPES.NPY_INT32, newArray.ItemType);

            Assert.AreEqual(Common.GetDefaultItemSize(SimpleArray.data.type_num), 4);
            Assert.AreEqual(Common.GetDefaultItemSize(newArray.data.type_num), 4);
        }

        [TestMethod]
        public void NpyArray_NewCopy_Test2()
        {
            int DataSize = 0;

            var SimpleArray = Common.GetSimpleArray(NPY_TYPES.NPY_BYTE, ref DataSize);
            Assert.AreEqual(NPY_TYPES.NPY_BYTE, SimpleArray.ItemType);

            var newArray = numpyAPI.NpyArray_NewCopy(SimpleArray, NPY_ORDER.NPY_ANYORDER);
            Assert.AreEqual(NPY_TYPES.NPY_BYTE, newArray.ItemType);

            Assert.AreEqual(Common.GetDefaultItemSize(SimpleArray.data.type_num), 1);
            Assert.AreEqual(Common.GetDefaultItemSize(newArray.data.type_num), 1);
        }

        [TestMethod]
        public void NpyArray_NewCopy_Test_Float()
        {
            int DataSize = 0;

            var SimpleArray = Common.GetSimpleArray(NPY_TYPES.NPY_FLOAT, ref DataSize);
            Assert.AreEqual(NPY_TYPES.NPY_FLOAT, SimpleArray.ItemType);

            var newArray = numpyAPI.NpyArray_NewCopy(SimpleArray, NPY_ORDER.NPY_ANYORDER);
            Assert.AreEqual(NPY_TYPES.NPY_FLOAT, newArray.ItemType);

            Assert.AreEqual(Common.GetDefaultItemSize(SimpleArray.data.type_num), 4);
            Assert.AreEqual(Common.GetDefaultItemSize(newArray.data.type_num), 4);
        }


        [TestMethod]
        public void NpyArray_NewCopy_TestAll()
        {
            int DataSize = 0;

            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES type in NPY_TYPES_Values)
            {
                var type2 = NPY_TYPES.NPY_INT16;


                int ExpectedSize = Common.GetDefaultItemSize(type2);
                if (ExpectedSize < 0)
                    continue;

                var SimpleArray = Common.GetSimpleArray(type2, ref DataSize);
                Assert.AreEqual(type2, SimpleArray.ItemType);


                var newArray = numpyAPI.NpyArray_NewCopy(SimpleArray, NPY_ORDER.NPY_ANYORDER);
                var array_size = numpyAPI.NpyArray_Size(newArray);

                Assert.AreEqual(type2, newArray.ItemType);

                Assert.AreEqual(DataSize, array_size);
                Assert.AreEqual(Common.GetDefaultItemSize(type2), SimpleArray.ItemSize);
                Assert.AreEqual(type2, SimpleArray.ItemType);
            }


        }

  
       [Ignore]
        [TestMethod]
        public void NpyArray_FillWithScalar_TestByte()
        {
            int DataSize = 0;


            NPY_TYPES num_type = NPY_TYPES.NPY_BYTE;

            var SimpleArray = Common.GetSimpleArray(num_type, ref DataSize);
            Assert.AreEqual(num_type, SimpleArray.ItemType);

            Common.ArrayDataAdjust = 5;
            var ScalarArray = Common.GetSimpleArray(num_type, ref DataSize);
            Assert.AreEqual(num_type, ScalarArray.ItemType);

            var result = numpyAPI.NpyArray_FillWithScalar(SimpleArray, ScalarArray);
            Assert.AreEqual(0, result);


            byte[] simpleData = (byte[])SimpleArray.data.datap;
            byte[] scalarData = (byte[])ScalarArray.data.datap;

            Assert.IsTrue(Common.CompareArrays(simpleData, scalarData));

        }

        [Ignore]
        [TestMethod]
        public void NpyArray_FillWithScalar_TestInt64()
        {
            int DataSize = 0;


            NPY_TYPES num_type = NPY_TYPES.NPY_INT64;

            var SimpleArray = Common.GetSimpleArray(num_type, ref DataSize);
            Assert.AreEqual(num_type, SimpleArray.ItemType);

            Common.ArrayDataAdjust = 5;
            var ScalarArray = Common.GetSimpleArray(num_type, ref DataSize);
            Assert.AreEqual(num_type, ScalarArray.ItemType);

            var result = numpyAPI.NpyArray_FillWithScalar(SimpleArray, ScalarArray);
            Assert.AreEqual(0, result);


            Int64[] simpleData = (Int64[])SimpleArray.data.datap;
            Int64[] scalarData = (Int64[])ScalarArray.data.datap;

            Assert.IsTrue(Common.CompareArrays(simpleData, scalarData));

        }


        [Ignore]
        [TestMethod]
        public void NpyArray_FillWithScalar_TestUInt64()
        {
            int DataSize = 0;


            NPY_TYPES num_type = NPY_TYPES.NPY_UINT64;

            var SimpleArray = Common.GetSimpleArray(num_type, ref DataSize);
            Assert.AreEqual(num_type, SimpleArray.ItemType);

            Common.ArrayDataAdjust = 5;
            var ScalarArray = Common.GetSimpleArray(num_type, ref DataSize);
            Assert.AreEqual(num_type, ScalarArray.ItemType);

            var result = numpyAPI.NpyArray_FillWithScalar(SimpleArray, ScalarArray);
            Assert.AreEqual(0, result);


            UInt64[] simpleData = (UInt64[])SimpleArray.data.datap;
            UInt64[] scalarData = (UInt64[])ScalarArray.data.datap;

            Assert.IsTrue(Common.CompareArrays(simpleData, scalarData));

        }


        [Ignore]
        [TestMethod]
        public void NpyArray_FillWithScalar_TestFloat()
        {
            int DataSize = 0;


            NPY_TYPES num_type = NPY_TYPES.NPY_FLOAT;

            var SimpleArray = Common.GetSimpleArray(num_type, ref DataSize);
            Assert.AreEqual(num_type, SimpleArray.ItemType);

            Common.ArrayDataAdjust = 5;
            var ScalarArray = Common.GetSimpleArray(num_type, ref DataSize);
            Assert.AreEqual(num_type, ScalarArray.ItemType);

            var result = numpyAPI.NpyArray_FillWithScalar(SimpleArray, ScalarArray);
            Assert.AreEqual(0, result);


            float[] simpleData = (float[])SimpleArray.data.datap;
            float[] scalarData = (float[])ScalarArray.data.datap;

            Assert.IsTrue(Common.CompareArrays(simpleData, scalarData));

        }


        [Ignore]
        [TestMethod]
        public void NpyArray_FillWithScalar_TestDouble()
        {
            int DataSize = 0;


            NPY_TYPES num_type = NPY_TYPES.NPY_DOUBLE;

            var SimpleArray = Common.GetSimpleArray(num_type, ref DataSize);
            Assert.AreEqual(num_type, SimpleArray.ItemType);

            Common.ArrayDataAdjust = 5;
            var ScalarArray = Common.GetSimpleArray(num_type, ref DataSize);
            Assert.AreEqual(num_type, ScalarArray.ItemType);

            var result = numpyAPI.NpyArray_FillWithScalar(SimpleArray, ScalarArray);
            Assert.AreEqual(0, result);


            double[] simpleData = (double[])SimpleArray.data.datap;
            double[] scalarData = (double[])ScalarArray.data.datap;

            Assert.IsTrue(Common.CompareArrays(simpleData, scalarData));

        }


    }
}
