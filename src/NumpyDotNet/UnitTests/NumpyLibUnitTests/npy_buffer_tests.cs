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
using System.Runtime.InteropServices;

namespace NumpyLibTests
{
    [TestClass]
    public class npy_buffer_tests
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
        public void npy_voidptr_Test1()
        {
    
            Int32[] TestData = new int[500000];

            VoidPtr vp1 = new VoidPtr(TestData);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();
            for (int i = 0; i < 251375223; i++)
            {
                vp1 += 1;
            }
            sw.Stop();

            long Test1MS = sw.ElapsedMilliseconds;


            vp1 = new VoidPtr(TestData);
            sw.Reset();
            sw.Start();
            for (int i = 0; i < 251375223; i++)
            {
                vp1.data_offset += 1;
            }
            sw.Stop();
            long Test2MS = sw.ElapsedMilliseconds;


            Console.WriteLine("{0} vs {1}", Test1MS, Test2MS);


        }

        [TestMethod]
        public unsafe void npy_memcpy_Test1()
        {

            byte[] TestData = new byte[500000];
            byte[] TestData2 = new byte[500000];

            int copySize = 4;
            int loopCnt = 50000000;

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < loopCnt; i++)
            {
                Tests.FastCopy.Copy(TestData, TestData2, 0, (uint)copySize);
            }
            sw.Stop();

            long Test1MS = sw.ElapsedMilliseconds;


            sw.Reset();
            sw.Start();
            for (int i = 0; i < loopCnt; i++)
            {
                Array.Copy(TestData, TestData2, copySize);
            }
            sw.Stop();
            long Test2MS = sw.ElapsedMilliseconds;


            sw.Reset();
            sw.Start();
            for (int i = 0; i < loopCnt; i++)
            {
                Buffer.BlockCopy(TestData, 0, TestData2, 0, copySize);
            }
            sw.Stop();
            long Test3MS = sw.ElapsedMilliseconds;

            sw.Reset();
            sw.Start();
            for (int i = 0; i < loopCnt; i++)
            {
                //GCHandle handle1 = GCHandle.Alloc(TestData2);
                //IntPtr parameter = (IntPtr)handle1;
                //// call WinAPi and pass the parameter here
                //// then free the handle when not needed:
                //Marshal.Copy(TestData, 0, parameter, copySize);

                //handle1.Free();
            }
            sw.Stop();
            long Test4MS = 0;


            sw.Reset();
            sw.Start();
            for (int i = 0; i < loopCnt; i++)
            {
                for (int j = 0; j < copySize; j++)
                {
                    TestData2[j] = TestData[j];
                }
            }
            sw.Stop();
            long Test5MS = sw.ElapsedMilliseconds;


            sw.Reset();
            sw.Start();
            for (int i = 0; i < loopCnt; i++)
            {
                fixed (void* pSrc = TestData2, pDst = TestData)
                {
                        CustomCopy(pSrc, pDst, copySize);
                }
            }
            sw.Stop();
            long Test6MS = sw.ElapsedMilliseconds;



            Console.WriteLine("{0} vs {1} vs {2} vs {3} vs {4} vs {5}", Test1MS, Test2MS, Test3MS, Test4MS, Test5MS, Test6MS);


        }

        static unsafe void CustomCopy(void* dest, void* src, int count)
        {
            int block;

            block = count >> 3;

            long* pDest = (long*)dest;
            long* pSrc = (long*)src;

            for (int i = 0; i < block; i++)
            {
                *pDest = *pSrc; pDest++; pSrc++;
            }
            dest = pDest;
            src = pSrc;
            count = count - (block << 3);

            if (count > 0)
            {
                byte* pDestB = (byte*)dest;
                byte* pSrcB = (byte*)src;
                for (int i = 0; i < count; i++)
                {
                    *pDestB = *pSrcB; pDestB++; pSrcB++;
                }
            }
        }



        [TestMethod]
        public void npy_array_getsegcount_Test1()
        {
            NPY_TYPES type = NPY_TYPES.NPY_INT32; 

            int DataSize = 0;
            var ComplexArray = Common.GetComplexArray2D(type, ref DataSize, 2 + (int)type, 3 + (int)type);

            ulong temp = 0;

            var segment_count = numpyAPI.npy_array_getsegcount(ComplexArray, ref temp);

            Assert.AreEqual(1, segment_count);

        }

        [TestMethod]
        public void npy_array_getreadbuf_Test1()
        {
            NPY_TYPES type = NPY_TYPES.NPY_BYTE;

            int DataSize = 0;
            var OneSegmentArray = Common.GetOneSegmentArray(type, ref DataSize);

            VoidPtr retPtr = null;

            var byte_count = numpyAPI.npy_array_getreadbuf(OneSegmentArray,0, ref retPtr);

            int itemsize = Common.GetDefaultItemSize(type);
           // Assert.AreEqual(DataSize * itemsize, byte_count);

            byte_count = numpyAPI.npy_array_getwritebuf(OneSegmentArray, 0, ref retPtr);
            //Assert.AreEqual(DataSize * itemsize, byte_count);
                    

        }


        [TestMethod]
        public void Npy_IsWriteable_Test1()
        {
            NPY_TYPES type = NPY_TYPES.NPY_INT32;

            int DataSize = 0;

            bool b1 = numpyAPI.Npy_IsWriteable(Common.GetOneSegmentArray(type, ref DataSize));
            bool b2 = numpyAPI.Npy_IsWriteable(Common.GetSimpleArray(type, ref DataSize));

        }


        [TestMethod]
        public void NpyArray_Index2Ptr_Test1()
        {

            int DataSize = 0;

            var SimpleArray = Common.GetSimpleArray(NPY_TYPES.NPY_INT32, ref DataSize);
            VoidPtr dp1 = numpyAPI.NpyArray_Index2Ptr(SimpleArray, 0);

            var ComplexArray = Common.GetComplexArray2D(NPY_TYPES.NPY_INT32, ref DataSize, 3,5);
            VoidPtr dp2 = numpyAPI.NpyArray_Index2Ptr(ComplexArray, 0);
            VoidPtr dp3 = numpyAPI.NpyArray_Index2Ptr(ComplexArray, 1);
            VoidPtr dp4 = numpyAPI.NpyArray_Index2Ptr(ComplexArray, 2);

        }

        
    }
}
