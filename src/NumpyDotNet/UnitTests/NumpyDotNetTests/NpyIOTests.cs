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

using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using MathNet.Numerics;

namespace NumpyDotNetTests
{
    [TestClass]
    public class NpyIOTests : TestBaseClass
    {
        //[Ignore]
        [TestMethod]
        public void test_load_1()
        {

            ndarray t1 = np.load("c:/temp/t1.npy");
            print(t1);

            ndarray tf = t1.reshape((2, 3), NumpyLib.NPY_ORDER.NPY_FORTRANORDER);
            ndarray tc = t1.reshape((2, 3), NumpyLib.NPY_ORDER.NPY_CORDER);

            ndarray t2 = np.load("c:/temp/t2.npy");
            print(t2);

            ndarray t3 = np.load("c:/temp/t3.npy");
            print(t3);


            return;
        }

        [TestMethod]
        public void test_load_decimal_1()
        {

            ndarray d1 = np.arange(0, 32, dtype: np.Decimal).reshape(4,8);
            print(d1);

            np.save("c:/temp/d1.npy", d1);

            ndarray r1 = np.load("c:/temp/d1.npy");
            print(r1);

            ndarray v1 = np.equal(d1,r1);
            Assert.IsTrue(np.allb(v1));

            return;
        }

        [TestMethod]
        public void test_load_complex_1()
        {

            ndarray d1 = np.arange(0, 32, dtype: np.Complex).reshape(4, 8);
            print(d1);

            np.save("c:/temp/c1.npy", d1);

            ndarray r1 = np.load("c:/temp/c1.npy");
            print(r1);

            ndarray v1 = np.equal(d1, r1);
            Assert.IsTrue(np.allb(v1));

            return;
        }


        [TestMethod]
        public void test_load_1_fortran()
        {

            ndarray tf = np.load("c:/temp/tf.npy");
            ndarray tc = np.load("c:/temp/tc.npy");


            print(tf);
            print(tc);


            return;
        }

        [TestMethod]
        public void test_save_1()
        {
            ndarray sb = np.arange(0,120, dtype: np.Float64).reshape(6, 2, 10);


            ndarray s1 = (ndarray)sb["2::", "1::", "5::"];

            s1["..."] = s1 * 10.0;

            print(sb);
            print(s1);

            np.save("c:/temp/tx.npy", s1);

            ndarray t1 = np.load("c:/temp/tx.npy");
            print(t1);

     
            return;
        }

    }
}
