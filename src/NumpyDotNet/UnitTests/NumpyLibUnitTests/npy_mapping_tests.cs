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
    public class npy_mapping_tests
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
        public void NpyArray_Subscript_Test1()
        {
            int DataSize = 0;
            NPY_TYPES desiredType = NPY_TYPES.NPY_INT32;

            var BaseArray = Common.GetSimpleArray(desiredType, ref DataSize);


            NpyIndex[] indexes = new NpyIndex[1];
            indexes[0] = new NpyIndex();
            indexes[0].slice = new NpyIndexSlice() { start = 1, stop = 5, step = 2 };

            var MappedArray = numpyAPI.NpyArray_Subscript(BaseArray, indexes, 1);

            return;

        }


        [TestMethod]
        public void NpyArray_Subscript_Test2()
        {
            int DataSize = 0;
            NPY_TYPES desiredType = NPY_TYPES.NPY_INT16;


            var BaseArray = Common.GetComplexArray3D(desiredType, ref DataSize, 5, 6, 8);
            //Console.WriteLine(DumpData.DumpArray(BaseArray, true));

            NpyIndex[] indexes = new NpyIndex[3];
            indexes[0] = new NpyIndex();
            indexes[0].type = NpyIndexType.NPY_INDEX_SLICE;
            indexes[0].slice = new NpyIndexSlice() { start = 1, stop = 5, step = 1 };

            indexes[1] = new NpyIndex();
            indexes[1].type = NpyIndexType.NPY_INDEX_SLICE;
            indexes[1].slice = new NpyIndexSlice() { start = 1, stop = 2, step = 1 };

            indexes[2] = new NpyIndex();
            indexes[2].type = NpyIndexType.NPY_INDEX_SLICE;
            indexes[2].slice = new NpyIndexSlice() { start = 1, stop = 2, step = 1 };

            var MappedArray = numpyAPI.NpyArray_Subscript(BaseArray, indexes, 3);

            //Console.WriteLine(DumpData.DumpArray(MappedArray, true));

            npy_intp Sum = 1;
            for (int i = 0; i < MappedArray.nd; i++)
            {
                Sum *= MappedArray.dimensions[i];
            }

            //MappedArray.descr.f.setitem(Sum, 99, MappedArray);

            Console.WriteLine(DumpData.DumpArray(BaseArray, false));
            Console.WriteLine(DumpData.DumpArray(MappedArray, false));

            //todo: this throws an exception because the index is out of range.  Possibly our stride problem

            return;

        }
    }
}
