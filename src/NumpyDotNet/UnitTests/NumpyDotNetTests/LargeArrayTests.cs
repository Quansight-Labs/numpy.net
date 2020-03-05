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
using System;
#if NPY_INTP_64
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNetTests
{
    [TestClass]
    public class LargeArrayTests : TestBaseClass
    {

        [TestMethod]
        public void test_largearray_matmul_INT64_1()
        {
            int width = 1024;
            int height = 1024;

            var x_range = np.arange(0, width, 1, dtype : np.Int64);
            var y_range = np.arange(0, height * 2, 2, dtype : np.Int64);

            var x_mat = np.matmul(x_range.reshape(width, 1), y_range.reshape(1, height));
            var z = np.sum(x_mat);
            print(z);

            Assert.AreEqual(548682596352, z.GetItem(0));

            return;

        }

        [TestMethod]
        public void test_largearray_matmul_INT64_2()
        {
            int width = 1024;
            int height = 1024;

            var x_range = np.arange(0, width, 1, dtype: np.Int64);
            var y_range = np.arange(0, height * 2, 2, dtype: np.Int64);

            var x_mat = np.matmul(x_range.reshape(width, 1), y_range.reshape(1, height));

            var z = np.sum(x_mat, axis: 0);
            var z1 = np.sum(z);
            print(z1);
            Assert.AreEqual(548682596352, z1.GetItem(0));

            z = np.sum(x_mat, axis: 1);
            z1 = np.sum(z);
            print(z1);
            Assert.AreEqual(548682596352, z1.GetItem(0));

            return;

        }


        [TestMethod]
        public void test_largearray_add_INT64_1()
        {
            int width = 1024;
            int height = 1024;

            var x_range = np.arange(0, width, 1, dtype: np.Int64);
            var y_range = np.arange(0, height * 2, 2, dtype: np.Int64);

            var x_mat = np.add(x_range.reshape(width, 1), y_range.reshape(1, height));

            var z = np.sum(x_mat, axis: 0);
            var z1 = np.sum(z);
            print(z1);
            Assert.AreEqual((Int64)1609039872, z1.GetItem(0));

            z = np.sum(x_mat, axis: 1);
            z1 = np.sum(z);
            print(z1);
            Assert.AreEqual((Int64)1609039872, z1.GetItem(0));

            return;

        }


        [TestMethod]
        public void test_largearray_add_INT64_2()
        {
            int width = 1024;
            int height = 1024;

            var x_range = np.arange(0, width, 1, dtype: np.Int64);
            var y_range = np.arange(0, height * 2, 2, dtype: np.Int64);

            var x_mat = np.add(x_range.reshape(width, 1), y_range.reshape(1, height));
            x_mat = np.expand_dims(x_mat, 0);

            var z = np.sum(x_mat, axis: 0);
            var z1 = np.sum(z);
            print(z1);
            Assert.AreEqual((Int64)1609039872, z1.GetItem(0));

            z = np.sum(x_mat, axis: 1);
            z1 = np.sum(z);
            print(z1);
            Assert.AreEqual((Int64)1609039872, z1.GetItem(0));

            z = np.sum(x_mat, axis: 2);
            z1 = np.sum(z);
            print(z1);
            Assert.AreEqual((Int64)1609039872, z1.GetItem(0));

            return;

        }


        [TestMethod]
        public void test_largearray_multiply_BigInt_1()
        {
            int width = 2048;
            int height = 2048;

            var x_range = np.arange(0, width, 1, dtype: np.Int64).astype(np.BigInt);
            var y_range = np.arange(0, height * 2, 2, dtype: np.Int64).astype(np.BigInt);

            var x_mat = np.multiply(x_range.reshape(width, 1), y_range.reshape(1, height));

            var z = np.sum(x_mat, axis: 0);
            var z1 = np.sum(z);
            print(z1);
            Assert.AreEqual((System.Numerics.BigInteger)8787505184768, z1.GetItem(0));

            z = np.sum(x_mat, axis: 1);
            z1 = np.sum(z);
            print(z1);
            Assert.AreEqual((System.Numerics.BigInteger)8787505184768, z1.GetItem(0));

            return;

        }


        [TestMethod]
        public void test_largearray_add_BigInt_2()
        {
            int width = 4096;
            int height = 4096;

            var x_range = np.arange(0, width, 1, dtype: np.Int64).astype(np.BigInt);
            var y_range = np.arange(0, height * 2, 2, dtype: np.Int64).astype(np.BigInt);

            var x_mat = np.multiply(x_range.reshape(1, width), y_range.reshape(height, 1));
            x_mat = np.expand_dims(x_mat, 0);

            var z = np.sum(x_mat, axis: 0);
            var z1 = np.sum(z);
            print(z1);
            Assert.AreEqual((System.Numerics.BigInteger)140668777267200, z1.GetItem(0));

            z = np.sum(x_mat, axis: 1);
            z1 = np.sum(z);
            print(z1);
            Assert.AreEqual((System.Numerics.BigInteger)140668777267200, z1.GetItem(0));

            z = np.sum(x_mat, axis: 2);
            z1 = np.sum(z);
            print(z1);
            Assert.AreEqual((System.Numerics.BigInteger)140668777267200, z1.GetItem(0));

            return;

        }

        [TestMethod]
        public void test_largearray_copy_int64_1()
        {
            int length = (Int32.MaxValue) / sizeof(double) - 20;
            var x = np.arange(0, length, 1, dtype: np.Int64);
            Assert.AreEqual(length, x.size);

            var z = np.sum(x);
            Assert.AreEqual(36028791247601895, z.GetItem(0));

            var y = x.Copy();
            z = np.sum(x);
            Assert.AreEqual(36028791247601895, z.GetItem(0));

            return;
        }


        [TestMethod]
        public void test_largearray_copy_int64_2()
        {
            int length = (Int32.MaxValue) / sizeof(double) - 21;
            var x = np.arange(0, length, 1, dtype: np.Int64).reshape(2,-1);
            Assert.AreEqual(length, x.size);

            var z = np.sum(x, axis: 0);
            z = np.sum(z);
            Assert.AreEqual(36028790979166461, z.GetItem(0));

            var y = x.Copy();
            z = np.sum(x, axis: 1);
            z = np.sum(z);
            Assert.AreEqual(36028790979166461, z.GetItem(0));

            return;
        }

        [TestMethod]
        public void test_largearray_meshgrid_int64_2()
        {
            int length = 100 * 100;

            var x = np.arange(0, length, 1, dtype: np.Int64);

            ndarray[] xv = np.meshgrid(new ndarray[] { x, x });

            var s1 = np.sum(xv[0]);
            Assert.AreEqual(499950000000, s1.GetItem(0));
            var s2 = np.sum(xv[1]);
            Assert.AreEqual(499950000000, s2.GetItem(0));


            return;
        }

        [TestMethod]
        public void test_largearray_checkerboard_1()
        {
            var x = np.zeros((2048, 2048), dtype: np.Int32);
            x["1::2", "::2"] = 1;
            x["::2", "1::2"] = 1;

     
            AssertShape(x, 2048, 2048);
            AssertStrides(x, sizeof(Int32) * 2048, sizeof(Int32));

            Assert.AreEqual(2097152, np.sum(x).GetItem(0));

            return;
        }

        [TestMethod]
        public void test_largearray_byteswap_int64_2()
        {
            var length = 1024 * 1024 * 32; // (Int32.MaxValue) / sizeof(double) - 21;
            var x = np.arange(0, length, 1, dtype : np.Int64).reshape(2, -1);
            var y = x.byteswap();

            var z = np.sum(y, axis : 0);
            z = np.sum(z);
            print(z);
            Assert.AreEqual(72057594037927936, z.GetItem(0));

            z = np.sum(y, axis: 1);
            z = np.sum(z);
            print(z);
            Assert.AreEqual(72057594037927936, z.GetItem(0));

            return;
        }

        [TestMethod]
        public void test_largearray_unique_INT32()
        {
            var matrix = np.arange(16000000, dtype: np.Int32).reshape(40, -1);

            matrix = matrix["1:40:2", "1:-2:1"] as ndarray;

            var result = np.unique(matrix, return_counts: true, return_index: true, return_inverse: true);

            Assert.AreEqual(-1830479044, np.sum(result.data).GetItem(0));
            Assert.AreEqual(31999516001830, np.sum(result.indices).GetItem(0));
            Assert.AreEqual(31999516001830, np.sum(result.inverse).GetItem(0));
            Assert.AreEqual((Int64)7999940, np.sum(result.counts).GetItem(0));

        }

        [TestMethod]
        public void test_largearray_where_INT32()
        {
            var matrix = np.arange(16000000, dtype: np.Int32).reshape(40, -1);
            Assert.AreEqual(1376644608, np.sum(matrix).GetItem(0));

            var indices = np.where(matrix % 2 == 0);

            var m1 = matrix[indices] as ndarray;
            Assert.AreEqual(684322304, np.sum(m1).GetItem(0));

        }

        [TestMethod]
        public void test_largearray_insert_BIGINT()
        {
            var matrix = np.arange(16000000, dtype: np.BigInt).reshape(40, -1);
            Assert.AreEqual((System.Numerics.BigInteger)127999992000000, np.sum(matrix).GetItem(0));

            ndarray m1 = np.insert(matrix, 0, new System.Numerics.BigInteger[] { 999, 100, 101 });
            Assert.AreEqual((System.Numerics.BigInteger)127999992001200, np.sum(m1).GetItem(0));

        }

        [TestMethod]
        public void test_largearray_append_BIGINT()
        {
            var matrix = np.arange(16000000, dtype: np.BigInt).reshape(40, -1);
            Assert.AreEqual((System.Numerics.BigInteger)127999992000000, np.sum(matrix).GetItem(0));

            ndarray m1 = np.append(matrix, new System.Numerics.BigInteger[] { 999, 100, 101 });
            Assert.AreEqual((System.Numerics.BigInteger)127999992001200, np.sum(m1).GetItem(0));
        }


        [TestMethod]
        public void test_largearray_concatenate_INT64()
        {
            var a = np.arange(16000000, dtype: np.Int64).reshape(40, -1);
            var b = np.arange(1,16000001, dtype: np.Int64).reshape(40, -1);

            var c = np.concatenate((a, b), axis: 0);
            Assert.AreEqual(256000000000000, np.sum(c).GetItem(0));

            var e = np.concatenate((a, b), axis: null);
            Assert.AreEqual(256000000000000, np.sum(e).GetItem(0));
        }

        [TestMethod]
        public void test_largearray_min_INT64()
        {
            var a = np.arange(16000000, dtype: np.Int64).reshape(40, -1);

            var b = np.amin(a) as ndarray;
            Assert.AreEqual((Int64)0, np.sum(b).GetItem(0));

            b = np.amin(a, axis:0) as ndarray;
            Assert.AreEqual((Int64)79999800000, np.sum(b).GetItem(0));

            b = np.amin(a, axis: 1) as ndarray;
            Assert.AreEqual((Int64)312000000, np.sum(b).GetItem(0));

       }

        [TestMethod]
        public void test_largearray_max_INT64()
        {
            var a = np.arange(16000000, dtype: np.Int64).reshape(40, -1);

            var b = np.amax(a) as ndarray;
            Assert.AreEqual((Int64)15999999, np.sum(b).GetItem(0));

            b = np.amax(a, axis: 0) as ndarray;
            Assert.AreEqual((Int64)6319999800000, np.sum(b).GetItem(0));

            b = np.amax(a, axis: 1) as ndarray;
            Assert.AreEqual((Int64)327999960, np.sum(b).GetItem(0));

        }


    }
}
