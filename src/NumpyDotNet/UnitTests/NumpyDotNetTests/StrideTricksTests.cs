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
using NumpyDotNet;
using NumpyLib;

namespace NumpyDotNetTests
{
    [TestClass]
    public class StrideTricksTests : TestBaseClass
    {
        [TestMethod]
        public void test_broadcast_1()
        {
            var x = np.array(new int[,] { { 11 }, { 2 }, { 3 } });
            var y = np.array(new int[] { 4, 5, 6 });
            var b = np.broadcast(x, y);
            Assert.AreEqual(b.shape.iDims.Length, 2);
            Assert.AreEqual(b.shape.iDims[0], 3);
            Assert.AreEqual(b.shape.iDims[1], 3);
            print(b.shape);

            Assert.AreEqual(b.index, 0);
            print(b.index);

            foreach (var uv in b)
            {
                print(uv);
            }
            Assert.AreEqual(b.index, 9);
            print(b.index);

        }

        [TestMethod]
        public void test_broadcast_2()
        {
            var x = np.array(new int[,] { { 11 }, { 2 }, { 3 } });
            var y = np.array(new int[] { 4, 5, 6, 7,8, 9 });
            var b = np.broadcast(x, y);
            Assert.AreEqual(b.shape.iDims.Length, 2);
            Assert.AreEqual(b.shape.iDims[0], 3);
            Assert.AreEqual(b.shape.iDims[1], 6);
            print(b.shape);
            Assert.AreEqual(b.size, 18);
            print(b.size);

            Assert.AreEqual(b.index, 0);
            print(b.index);

            foreach (var uv in b)
            {
                print(uv);
            }
            Assert.AreEqual(b.index, 18);
            print(b.index);

        }

        [TestMethod]
        public void test_broadcast_3()
        {
            var x = np.array(new int[,] { { 11 }, { 2 }, { 3 } });
            var y = np.array(new int[] { 4, 5, 6, 7, 8, 9 });
            var z = np.array(new int[,] { { 21 }, { 22 }, { 23 } });
            var b = np.broadcast(x, y, z);
            Assert.AreEqual(b.shape.iDims.Length, 2);
            Assert.AreEqual(b.shape.iDims[0], 3);
            Assert.AreEqual(b.shape.iDims[1], 6);
            print(b.shape);
            Assert.AreEqual(b.size, 18);
            print(b.size);

            Assert.AreEqual(b.index, 0);
            print(b.index);

            foreach (var uv in b)
            {
                print(uv);
            }
            Assert.AreEqual(b.index, 18);
            print(b.index);

        }

        [TestMethod]
        public void test_broadcast_to_1()
        {
            var a = np.broadcast_to(5, (4, 4));
            AssertArray(a, new int[,] { { 5,5,5,5}, { 5, 5, 5, 5 }, { 5, 5, 5, 5 }, { 5, 5, 5, 5 }});
            AssertStrides(a, 0, 0);
            print(a);
            print(a.shape);
            print(a.strides);
            print("*************");
 

            var b = np.broadcast_to(new int[] { 1, 2, 3 }, (3, 3));
            AssertArray(b, new int[,] { { 1, 2, 3 }, { 1, 2, 3 }, { 1, 2, 3 } });
            AssertStrides(b, 0, 4);
            print(b);
            print(b.shape);
            print(b.strides);
            print("*************");


        }

        [TestMethod]
        public void test_broadcast_to_2()
        {
            var x = np.array(new int[,] { { 1, 2, 3 } });
            //print(x);
            //print(x.shape);
            //print(x.strides);
            //print("*************");

            var b = np.broadcast_to(x, (4, 3));
            AssertArray(b, new int[,] { { 1, 2, 3 }, { 1, 2, 3 }, { 1, 2, 3 }, { 1, 2, 3 } });
            AssertStrides(b, 0, 4);
            print(b);
            print(b.shape);
            print(b.strides);
            print("*************");
        }

        [TestMethod]
        public void test_broadcast_to_3()
        {
            try
            {
                var a = np.array(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }).reshape((2, 2, 3));
                var b = np.broadcast_to(a, (4, 2, 3));
            }
            catch (Exception ex)
            {
                return;
            }

            Assert.Fail("Should have caught an exception");
     
        }

        [TestMethod]
        public void test_broadcast_arrays_1()
        {
            var x = np.array(new int[,] { { 1, 2, 3 } });
            var y = np.array(new int[,] { { 4 }, { 5 } });
            var z = np.broadcast_arrays(false, new ndarray[] { x, y });

            print(z);

        }

        [TestMethod]
        public void test_as_strided_1()
        {
            var y = np.zeros((10, 10));
            AssertStrides(y, 80, 8);
            print(y.strides);

            var n = 1000;
            var a = np.arange(n, dtype: np.UInt64);

            var b = np.as_strided(a, (n, n), (0, 8));

            //print(b);

            Assert.AreEqual(1000000, b.size);
            print(b.size);
            AssertShape(b, 1000, 1000);
            print(b.shape);
            AssertStrides(b, 0, 8);
            print(b.strides);
            Assert.AreEqual(8000000, b.nbytes);
            print(b.nbytes);

        }

        [TestMethod]
        public void test_as_strided_2()
        {
            var y = np.zeros((2, 2));
            AssertStrides(y, 16, 8);
            print(y.strides);

            var n = 4;
            var a = np.arange(n);
            AssertArray(a, new int[] { 0, 1, 2, 3 });
            print(a);

            var b = np.as_strided(a, (n, n), (0, 4));
            AssertArray(b, new int[,] { { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, { 0, 1, 2, 3 } });
            print(b);

            Assert.AreEqual(16, b.size);
            print(b.size);
            AssertShape(b, 4, 4);
            print(b.shape);
            AssertStrides(b, 0, 4);
            print(b.strides);
            Assert.AreEqual(64, b.nbytes);
            print(b.nbytes);

        }

    }
}
