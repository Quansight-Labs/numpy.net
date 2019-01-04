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
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyLibTests
{
    [TestClass]
    public class npy_arrayobject_tests
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
        public void NpyArray_Size_SimpleArray_Test()
        {
            NpyArray SimpleArray = null;
            npy_intp array_size;

            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES type in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(type);
                if (ExpectedSize < 0)
                    continue;

                int DataSize = 0;
                SimpleArray = Common.GetSimpleArray(type, ref DataSize);
                array_size = numpyAPI.NpyArray_Size(SimpleArray);

                Assert.AreEqual(DataSize, array_size);
                Assert.AreEqual(Common.GetDefaultItemSize(type), SimpleArray.ItemSize);
                Assert.AreEqual(type, SimpleArray.ItemType);
            }

        }

        [TestMethod]
        public void NpyArray_CompareUCS4_Test()
        {
            npy_intp[] a1 = new npy_intp[] { 1, 2, 3, 4 };
            npy_intp[] b1 = new npy_intp[] { 1, 2, 3, 4 };

            Assert.AreEqual(0, numpyAPI.NpyArray_CompareUCS4(a1, b1, a1.Length));

            npy_intp[] a2 = new npy_intp[] { 1, 2, 3, 4 };
            npy_intp[] b2 = new npy_intp[] { 1, 2, 99, 4 };

            Assert.AreEqual(-1, numpyAPI.NpyArray_CompareUCS4(a2, b2, a1.Length));

            npy_intp[] a3 = new npy_intp[] { 1, 99, 3, 4 };
            npy_intp[] b3 = new npy_intp[] { 1, 2, 3, 4 };

            Assert.AreEqual(1, numpyAPI.NpyArray_CompareUCS4(a3, b3, a1.Length));

            npy_intp[] a4 = new npy_intp[] { 1, 2, 3, 4 };
            npy_intp[] b4 = new npy_intp[] { 1, 2, 3 };

            Assert.AreEqual(-2, numpyAPI.NpyArray_CompareUCS4(a4, b4, a1.Length));
            Assert.IsTrue(Common.MatchError(npyexc_type.NpyExc_IndexError, "NpyArray_CompareUCS4:"));
        }


        [TestMethod]
        //[ExpectedException(typeof(Exception))]
        public void NpyArray_CompareString_Test()
        {
            string a1 = "Numpy.net rocks";
            string b1 = "Numpy.net rocks";

            Assert.AreEqual(0, numpyAPI.NpyArray_CompareString(a1, b1, a1.Length));

            string a2 = "Numpy.net rocks";
            string b2 = "Numpy.net socks";

            Assert.AreEqual(-1, numpyAPI.NpyArray_CompareString(a2, b2, a1.Length-1));

            string a3 = "Numpy.net socks";
            string b3 = "Numpy.net rocks";

            Assert.AreEqual(1, numpyAPI.NpyArray_CompareString(a3, b3, a1.Length));

            string a4 = "Numpy.net rocks";
            string b4 = "Numpy.net rocks";

            int ret = numpyAPI.NpyArray_CompareString(a4, b4, a1.Length + 1);
            Assert.AreEqual(-2, ret);
            Assert.IsTrue(Common.MatchError(npyexc_type.NpyExc_IndexError, "NpyArray_CompareString:"));
        }

        [TestMethod]
        public void NpyArray_ElementStrides_2DTest()
        {
            NpyArray ComplexArray = null;
            npy_intp array_size;

            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES type in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(type);
                if (ExpectedSize < 0)
                    continue;

                int DataSize = 0;
                ComplexArray = Common.GetComplexArray2D(type, ref DataSize, 2+(int)type, 3+(int)type);

                array_size = numpyAPI.NpyArray_Size(ComplexArray);
                int result = numpyAPI.NpyArray_ElementStrides(ComplexArray);

                Assert.AreEqual(DataSize, array_size);
                Assert.AreEqual(Common.GetDefaultItemSize(type), ComplexArray.ItemSize);
                Assert.AreEqual(type, ComplexArray.ItemType);

                Assert.AreEqual(1, result);
            }
        }

        [TestMethod]
        public void NpyArray_ElementStrides_3DTest()
        {
            NpyArray ComplexArray = null;
            npy_intp array_size;

            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES type in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(type);
                if (ExpectedSize < 0)
                    continue;

                int DataSize = 0;

                int a = 2 + (int)type;
                int b = 3 + (int)type;
                int c = 4 + (int)type;
                ComplexArray = Common.GetComplexArray3D(type, ref DataSize, a, b, c);

                array_size = numpyAPI.NpyArray_Size(ComplexArray);
                int result = numpyAPI.NpyArray_ElementStrides(ComplexArray);

                Assert.AreEqual(DataSize, array_size);
                Assert.AreEqual(Common.GetDefaultItemSize(type), ComplexArray.ItemSize);
                Assert.AreEqual(type, ComplexArray.ItemType);

                Assert.AreEqual(1, result);
            }
        }

        [TestMethod]
        public void NpyArray_ElementStrides_4DTest()
        {
            NpyArray ComplexArray = null;
            npy_intp array_size;

            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES type in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(type);
                if (ExpectedSize < 0)
                    continue;

                int DataSize = 0;

                int a = 2 + (int)type;
                int b = 3 + (int)type;
                int c = 4 + (int)type;
                int d = 5 + (int)type;
                ComplexArray = Common.GetComplexArray4D(type, ref DataSize, a, b, c, d);

                array_size = numpyAPI.NpyArray_Size(ComplexArray);
                int result = numpyAPI.NpyArray_ElementStrides(ComplexArray);

                Assert.AreEqual(DataSize, array_size);
                Assert.AreEqual(Common.GetDefaultItemSize(type), ComplexArray.ItemSize);
                Assert.AreEqual(type, ComplexArray.ItemType);

                Assert.AreEqual(1, result);
            }
        }

        [TestMethod]
        public void NpyArray_ElementStrides_5DTest()
        {
            NpyArray ComplexArray = null;
            npy_intp array_size;

            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES type in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(type);
                if (ExpectedSize < 0)
                    continue;

                int DataSize = 0;

                int a = 2 + (int)type;
                int b = 3 + (int)type;
                int c = 4 + (int)type;
                int d = 5 + (int)type;
                int e = 6 + (int)type;

                ComplexArray = Common.GetComplexArray5D(type, ref DataSize, a, b, c, d, e);

                array_size = numpyAPI.NpyArray_Size(ComplexArray);
                int result = numpyAPI.NpyArray_ElementStrides(ComplexArray);

                Assert.AreEqual(DataSize, array_size);
                Assert.AreEqual(Common.GetDefaultItemSize(type), ComplexArray.ItemSize);
                Assert.AreEqual(type, ComplexArray.ItemType);

                Assert.AreEqual(1, result);
            }
        }

        [TestMethod]
        public void NpyArray_CheckStrides_Test()
        {
            int elsize = 2;
            int nd = 3;
            int numbytes = 0;  // 0 indicates unknown, will be calculated
            npy_intp[] dims = new npy_intp[] { 5, 2, 10 };
            int offset = 10;
            npy_intp[] newstrides = new npy_intp[] { 2, 3, 5 };


            bool result = numpyAPI.NpyArray_CheckStrides(elsize, nd, numbytes, offset, dims, newstrides);
            Assert.IsTrue(result);

            offset = 190;
            result = numpyAPI.NpyArray_CheckStrides(elsize, nd, numbytes, offset, dims, newstrides);
            Assert.IsFalse(result);

        }

        [TestMethod]
        public void NpyArray_CompareStringArrays_Test()
        {

        }


    }
}
