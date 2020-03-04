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

using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using NumpyLib;
using System;
using System.Linq;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNetTests
{
    [TestClass]
    public class LargeArrayTests : TestBaseClass
    {

        [TestMethod]
        public void KEVIN_matmul_DOUBLE()
        {
            float scaling = 5.0f;
            int width = 256;
            int height = 256;
            double ret_step = 0;

            var x_range = np.linspace(-1 * scaling, scaling, ref ret_step, width, dtype: np.Float32);

            var x_mat = np.matmul(np.ones(new shape(height, 1)), x_range.reshape(1, width));
            //print(x_mat);

            var sum = np.sum(x_mat);
            return;

        }



        [TestMethod]
        public void test_maxtrix_99_BROKEN()
        {
            double ret_step = 0;

            var a = np.linspace(0.0, 1.0, ref ret_step, num: 32).reshape(1, 32);
            print(a);

            var b = np.reshape(a, new shape(1, 1, 32)) * np.ones((65536, 1)); // * 1;
            //print(b);
            var c = np.sum(b);
            print(c);
        }


        [TestMethod]
        public void test_maxtrix_100_BROKEN()
        {
            var a = np.arange(00, 32).reshape(1, 32);
            print(a);

            var b = np.reshape(a, new shape(1, 1, 32)) * np.ones((65536, 1)); // * 1;
            //print(b);
            var c = np.sum(b);
            print(c);
        }


        [TestMethod]
        public void test_maxtrix_101_BROKEN()
        {
            var a = np.arange(00, 32).reshape(1, 32);
            print(a);

            var b = np.full((1, 1, 32), 2) * np.full((65536, 1), 3); // * 1;
            //print(b);

            var d = np.where(b != 6);

            var kevin = b.AsDoubleArray();

            var c = np.sum(b);
            print(c);
        }

    }
}
