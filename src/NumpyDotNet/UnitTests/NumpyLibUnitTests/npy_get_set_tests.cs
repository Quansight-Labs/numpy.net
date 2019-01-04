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
    public class npy_get_set_tests
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
        public void NpyArray_SetShape_Test1()
        {
            int DataSize = 0;


            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES desiredType in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(desiredType);
                if (ExpectedSize < 0)
                    continue;

                var BaseArray = Common.GetSimpleArray(desiredType, ref DataSize);
                var TestArray = Common.GetSimpleArray(desiredType, ref DataSize);

                NpyArray_Dims NewDims = new NpyArray_Dims()
                {
                    ptr = new npy_intp[] { 2, -1, 2 },
                    len = 3,
                };

                int result = numpyAPI.NpyArray_SetShape(TestArray, NewDims);
                Assert.AreEqual(0, result);

                if (desiredType != NPY_TYPES.NPY_BOOL)
                {
                    Assert.IsTrue(Common.CompareArrays(BaseArray, TestArray));
                }

                Assert.AreEqual(TestArray.dimensions[1], Common.GeneratedArrayLength / 4);

            }

        }

        [TestMethod]
        public void NpyArray_SetShape_Test2()
        {
            int DataSize = 0;


            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES desiredType in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(desiredType);
                if (ExpectedSize < 0)
                    continue;

                var BaseArray = Common.GetSimpleArray(desiredType, ref DataSize);
                var TestArray = Common.GetSimpleArray(desiredType, ref DataSize);

                NpyArray_Dims NewDims = new NpyArray_Dims()
                {
                    ptr = new npy_intp[] { 2, 2, -1 },
                    len = 3,
                };

                int result = numpyAPI.NpyArray_SetShape(TestArray, NewDims);
                Assert.AreEqual(0, result);

                if (desiredType != NPY_TYPES.NPY_BOOL)
                {
                    Assert.IsTrue(Common.CompareArrays(BaseArray, TestArray));
                }

                Assert.AreEqual(TestArray.dimensions[2], Common.GeneratedArrayLength / 4);

            }
        }



        [TestMethod]
        public void NpyArray_SetShape_Test3()
        {
            int DataSize = 0;


            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES desiredType in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(desiredType);
                if (ExpectedSize < 0)
                    continue;

                var BaseArray = Common.GetSimpleArray(desiredType, ref DataSize);
                var TestArray = Common.GetSimpleArray(desiredType, ref DataSize);

                NpyArray_Dims NewDims = new NpyArray_Dims()
                {
                    ptr = new npy_intp[] { -1, 4, 1 },
                    len = 3,
                };

                int result = numpyAPI.NpyArray_SetShape(TestArray, NewDims);
                Assert.AreEqual(0, result);

                if (desiredType != NPY_TYPES.NPY_BOOL)
                {
                    Assert.IsTrue(Common.CompareArrays(BaseArray, TestArray));
                }

                Assert.AreEqual(TestArray.dimensions[0], Common.GeneratedArrayLength / 4);

            }

        }

        [Ignore]  // not sure how to put array into right format.  base_arr needs to be set.
        [TestMethod]
        public void NpyArray_SetStrides_Test1()
        {
            int DataSize = 0;


            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES desiredType in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(desiredType);
                if (ExpectedSize < 0)
                    continue;

                var BaseArray = Common.GetSimpleArray(desiredType, ref DataSize);
                var TestArray = Common.GetSimpleArray(desiredType, ref DataSize);

                NpyArray_Dims NewDims = new NpyArray_Dims()
                {
                    ptr = new npy_intp[] { -1, 4, 1 },
                    len = 3,
                };

                int result = numpyAPI.NpyArray_SetShape(TestArray, NewDims);

                NpyArray_Dims NewStrides = new NpyArray_Dims()
                {
                    ptr = new npy_intp[] { 2, Common.GeneratedArrayLength / 4, 2 },
                    len = 3,
                };

                result = numpyAPI.NpyArray_SetStrides(TestArray, NewDims);
                Assert.AreEqual(0, result);

                if (desiredType != NPY_TYPES.NPY_BOOL)
                {
                    Assert.IsTrue(Common.CompareArrays(BaseArray, TestArray));
                }


            }

        }

    }
}
