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
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;

namespace NumpyLibTests
{
    [TestClass]
    public class npy_convert_datatype_tests
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
        public void NpyArray_GetCastFunc_Test1()
        {
            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));

            foreach (NPY_TYPES type in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(type);
                if (ExpectedSize < 0)
                    continue;

                NpyArray_VectorUnaryFunc castfunc = null;
                var newDescr = numpyAPI.NpyArray_DescrNewFromType(type);
                castfunc = numpyAPI.NpyArray_GetCastFunc(newDescr, NPY_TYPES.NPY_INT32);
                Assert.IsTrue(castfunc != Common.DefaultCastFunction);

                int result = numpyAPI.NpyArray_RegisterCastFunc(newDescr, NPY_TYPES.NPY_INT32, Common.DefaultCastFunction);
                Assert.AreEqual(0, result);
                castfunc = numpyAPI.NpyArray_GetCastFunc(newDescr, NPY_TYPES.NPY_INT32);
                Assert.IsTrue(castfunc == Common.DefaultCastFunction);

            }

        }

        [TestMethod]
        public void NpyArray_CastTo_Bool()
        {
            NpyArray_CastTo_SpecifiedType(NPY_TYPES.NPY_BOOL);
            return;
        }
        [TestMethod]
        public void NpyArray_CastTo_Bytes()
        {
            NpyArray_CastTo_SpecifiedType(NPY_TYPES.NPY_BYTE);
            return;
        }
        [TestMethod]
        public void NpyArray_CastTo_UBytes()
        {
            NpyArray_CastTo_SpecifiedType(NPY_TYPES.NPY_UBYTE);
            return;
        }
        [TestMethod]
        public void NpyArray_CastTo_Int16()
        {
            NpyArray_CastTo_SpecifiedType(NPY_TYPES.NPY_INT16);
            return;
        }
        [TestMethod]
        public void NpyArray_CastTo_UInt16()
        {
            NpyArray_CastTo_SpecifiedType(NPY_TYPES.NPY_UINT16);
            return;
        }
        [TestMethod]
        public void NpyArray_CastTo_Int32()
        {
            NpyArray_CastTo_SpecifiedType(NPY_TYPES.NPY_INT32);
            return;
        }
        [TestMethod]
        public void NpyArray_CastTo_UInt32()
        {
            NpyArray_CastTo_SpecifiedType(NPY_TYPES.NPY_UINT32);
            return;
        }
        [TestMethod]
        public void NpyArray_CastTo_Int64()
        {
            NpyArray_CastTo_SpecifiedType(NPY_TYPES.NPY_INT64);
            return;
        }
        [TestMethod]
        public void NpyArray_CastTo_UInt64()
        {
            NpyArray_CastTo_SpecifiedType(NPY_TYPES.NPY_UINT64);
            return;
        }

        [TestMethod]
        public void NpyArray_CastTo_Float()
        {
            NpyArray_CastTo_SpecifiedType(NPY_TYPES.NPY_FLOAT);
            return;
        }
        [TestMethod]
        public void NpyArray_CastTo_Double()
        {
            NpyArray_CastTo_SpecifiedType(NPY_TYPES.NPY_DOUBLE);
            return;
        }
        [Ignore] //obsolete
        [TestMethod]
        public void NpyArray_CastTo_Decimal()
        {
            NpyArray_CastTo_SpecifiedType(NPY_TYPES.NPY_DECIMAL);
            return;
        }
        private void NpyArray_CastTo_SpecifiedType(NPY_TYPES specifiedType)
        {
            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));


            foreach (NPY_TYPES type in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(type);
                if (ExpectedSize < 0)
                    continue;

                int DataSize = 0;
                var SrcArray = Common.GetSimpleArray(type, ref DataSize, 0, true, true);
                var DestArray = Common.GetSimpleArray(specifiedType, ref DataSize, 0, true);

                int castToResult = numpyAPI.NpyArray_CastTo(DestArray, SrcArray);
                Assert.AreEqual(0, castToResult);
            }

            return;
        }

        [Ignore]
        [TestMethod]
        public void NpyArray_CastTo_Byte_2D()
        {
            NPY_TYPES type = NPY_TYPES.NPY_INT32;
            NPY_TYPES specifiedType = NPY_TYPES.NPY_BYTE;

            int DataSize = 0;
            var SrcArray = Common.GetComplexArray2D(type, ref DataSize, 10, 5, 0, false, false);
            var DestArray = Common.GetComplexArray2D(specifiedType, ref DataSize, 10, 20, 0, true, false);

            int castToResult = numpyAPI.NpyArray_CastTo(DestArray, SrcArray);
            Assert.AreEqual(0, castToResult);
        }

        [TestMethod]
        public void NpyArray_CastToType_Test1()
        {
            NPY_TYPES type = NPY_TYPES.NPY_INT32;
            NPY_TYPES specifiedType = NPY_TYPES.NPY_BYTE;

            int DataSize = 0;
            var SrcArray = Common.GetSimpleArray(type, ref DataSize, 0, false, false);
            var Sample = Common.GetSimpleArray(specifiedType, ref DataSize, 0, true, false);

            var DestArray = numpyAPI.NpyArray_CastToType(SrcArray, Sample.descr, true);
            Assert.AreNotEqual(null, DestArray);

            DestArray = numpyAPI.NpyArray_CastToType(SrcArray, Sample.descr, false);
            Assert.AreNotEqual(null, DestArray);

        }


        [TestMethod]
        public void NpyArray_CastToAny_Test1()
        {
            NPY_TYPES type = NPY_TYPES.NPY_INT32;
            NPY_TYPES specifiedType = NPY_TYPES.NPY_UBYTE;

            int DataSize = 0;
            var SrcArray = Common.GetSimpleArray(type, ref DataSize, 0, false, false);
            var DestArray = Common.GetSimpleArray(specifiedType, ref DataSize, 0, true, false);

            int result = numpyAPI.NpyArray_CastAnyTo(DestArray, SrcArray);
            Assert.AreEqual(0, result);
        }

        [Ignore]
        [TestMethod]
        public void NpyArray_CastToAny_Test2()
        {
            NPY_TYPES type = NPY_TYPES.NPY_INT32;
            NPY_TYPES specifiedType = NPY_TYPES.NPY_UBYTE;

            int DataSize = 0;
            var SrcArray = Common.GetSimpleArray(type, ref DataSize, 0, false, false);
            var DestArray = Common.GetComplexArray2D(specifiedType, ref DataSize, 10, Common.GeneratedArrayLength / 10, 0, true, false);

            int result = numpyAPI.NpyArray_CastAnyTo(DestArray, SrcArray);
            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void NpyArray_CanCastSafely_Test1()
        {
 
            bool bResult = numpyAPI.NpyArray_CanCastSafely(NPY_TYPES.NPY_BYTE, NPY_TYPES.NPY_INT32);
            Assert.AreEqual(true, bResult);
        }

        [TestMethod]
        public void NpyArray_CanCastTo_Test1()
        {
            NPY_TYPES type = NPY_TYPES.NPY_BYTE;
            NPY_TYPES specifiedType = NPY_TYPES.NPY_INT32;

            int DataSize = 0;
            var SrcArray = Common.GetSimpleArray(type, ref DataSize, 0, false, false);
            var DestArray = Common.GetComplexArray2D(specifiedType, ref DataSize, 10, Common.GeneratedArrayLength / 10, 0, true, false);

            bool bResult = numpyAPI.NpyArray_CanCastTo(SrcArray.descr, DestArray.descr);
            Assert.AreEqual(true, bResult);

            bResult = numpyAPI.NpyArray_CanCastTo(DestArray.descr, SrcArray.descr);
            Assert.AreEqual(false, bResult);
        }

        [TestMethod]
        public void NpyArray_ValidType_Test1()
        {
            var NPY_TYPES_Values = Enum.GetValues(typeof(NPY_TYPES));


            foreach (NPY_TYPES type in NPY_TYPES_Values)
            {
                int ExpectedSize = Common.GetDefaultItemSize(type);
                if (ExpectedSize < 0)
                    continue;

                bool bResult = numpyAPI.NpyArray_ValidType(type);
                Assert.AreEqual(true, bResult);
            }


            bool result = numpyAPI.NpyArray_ValidType((NPY_TYPES)999);
            Assert.AreEqual(false, result);

            return;
        }

        [Ignore]
        [TestMethod]
        public void NpyArray_CastTo_Performance()
        {
            const int rounds = 100, n = 1000;

            int DataSize = 64;
            var SrcArray = Common.GetSimpleArray(NPY_TYPES.NPY_BYTE, ref DataSize);
            var DestArray = Common.GetSimpleArray(NPY_TYPES.NPY_INT32, ref DataSize, 0, true);


            int byteResult;
            byteResult = numpyAPI.NpyArray_RegisterCastFunc(SrcArray.descr, NPY_TYPES.NPY_BYTE, Common.DefaultCastFunction);
            byteResult = numpyAPI.NpyArray_RegisterCastFunc(SrcArray.descr, NPY_TYPES.NPY_INT32, Common.DefaultCastFunction);

            byteResult = numpyAPI.NpyArray_RegisterCastFunc(DestArray.descr, NPY_TYPES.NPY_BYTE, Common.DefaultCastFunction);
            byteResult = numpyAPI.NpyArray_RegisterCastFunc(DestArray.descr, NPY_TYPES.NPY_INT32, Common.DefaultCastFunction);

            var fList = new List<TimeSpan>();
            var f2List = new List<TimeSpan>();

            for (int i = 0; i < rounds; i++)
            {

                //CastFunctions.UseGenerics = true;
                // Test generic
                Common.GCClear();
                var sw = new Stopwatch();
                sw.Start();

                for (int j = 0; j < n; j++)
                {
                    numpyAPI.NpyArray_CastTo(DestArray, SrcArray);
                }


                sw.Stop();
                fList.Add(sw.Elapsed);

                //CastFunctions.UseGenerics = false;

                // Test not-generic
                Common.GCClear();
                var sw2 = new Stopwatch();
                sw2.Start();

                for (int j = 0; j < n; j++)
                {
                    numpyAPI.NpyArray_CastTo(DestArray, SrcArray);
                }

                sw2.Stop();
                f2List.Add(sw2.Elapsed);
            }

            double f1AverageTicks = fList.Average(ts => ts.Ticks);
            Console.WriteLine("Elapsed for GENERICS = {0} \t ticks = {1}", fList.Average(ts => ts.TotalMilliseconds), f1AverageTicks);
            double f2AverageTicks = f2List.Average(ts => ts.Ticks);
            Console.WriteLine("Elapsed for NON GENERICS = {0} \t ticks = {1}", f2List.Average(ts => ts.TotalMilliseconds),  f2AverageTicks);
            Console.WriteLine("Not-generic method is {0} times faster, or on {1}%", f1AverageTicks / f2AverageTicks, (f1AverageTicks / f2AverageTicks - 1) * 100);
            //Console.ReadKey();



        }


        [TestMethod]
        public void NpyArray_Combine_Test1()
        {
            int DataSize = 0;
            var arr1 = Common.GetSimpleArray(NPY_TYPES.NPY_INT32, ref DataSize, 0, false, false);
            var arr2 = Common.GetSimpleArray(NPY_TYPES.NPY_INT32, ref DataSize, 0, false, false);

            var DestArray = numpyAPI.NpyArray_Combine(arr1, arr2);
            Assert.AreNotEqual(null, DestArray);

        }

        [TestMethod]
        public void NpyArray_Combine_Test2()
        {
            int DataSize = 0;
            var arr1 = Common.GetSimpleArray(NPY_TYPES.NPY_INT32, ref DataSize, 0, false, false);
            var arr2 = Common.GetSimpleArray(NPY_TYPES.NPY_BYTE, ref DataSize, 0, false, false);

            var DestArray = numpyAPI.NpyArray_Combine(arr1, arr2);
            Assert.AreEqual(null, DestArray);

        }

    }
}
