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
using System.Collections.Generic;
using System.Text;
using NumpyLib;
using System.Linq;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNetTests
{
    [TestClass]
    public class ArrayCreationTests : TestBaseClass
    {
        [ClassInitialize]
        public static new void CommonInit(TestContext t)
        {
            //Common.CommonInit();
        }

        [TestInitialize]
        public new void FunctionInit()
        {
            Common.NumpyErrors.Clear();
        }

        [TestMethod]
        public void test_simpleShape_1()
        {
            var a = np.array(new int[] { 1, 2, 3 });

            var x = a.shape.iDims[0];
            Assert.AreEqual(x, 3);
            print(x);

            var y = a.shape[0];
            Assert.AreEqual(y, 3);
            print(y);

            var b = np.arange(0,100).reshape(4,-1);

            var x1 = b.shape.iDims[0];
            var x2 = b.shape.iDims[1];

            Assert.AreEqual(x1, 4);
            Assert.AreEqual(x2, 25);

            var y1 = b.shape[0];
            var y2 = b.shape[1];

            Assert.AreEqual(y1, 4);
            Assert.AreEqual(y2, 25);

            return;
        }


        [TestMethod]
        public void test_asfarray_1()
        {
            var a = np.asfarray(new int[] { 2, 3 });
            AssertArray(a, new double[] { 2, 3 });
            print(a);

            var b = np.asfarray(new int[] { 2, 3 }, dtype: np.Float32);
            AssertArray(b, new float[] { 2, 3 });
            print(b);

            var c = np.asfarray(new int[] { 2, 3 }, dtype: np.Int8);
            AssertArray(c, new double[] { 2, 3 });
            print(c);


            return;
        }

        [TestMethod]
        public void test_copy_1()
        {
            var x = np.array(new int[] { 1, 2, 3 });
            var y = x;

            var z = np.copy(x);

            // Note that, when we modify x, y changes, but not z:

            x[0] = 10;

            Assert.AreEqual(10, y[0]);

            Assert.AreEqual(1, z[0]);

            return;
        }

        [TestMethod]
        public void test_linspace_1()
        {
            double retstep = 0;

            var a = np.linspace(2.0, 3.0, ref retstep, num: 5);
            AssertArray(a, new double[] { 2.0, 2.25, 2.5, 2.75, 3.0 });
            print(a);

            var b = np.linspace(2.0, 3.0, ref retstep, num: 5, endpoint: false);
            AssertArray(b, new double[] { 2.0, 2.2, 2.4, 2.6, 2.8 });
            print(b);

            var c = np.linspace(2.0, 3.0, ref retstep, num: 5);
            AssertArray(c, new double[] { 2.0, 2.25, 2.5, 2.75, 3.0 });
            print(c);
        }

        [TestMethod]
        public void test_logspace_1()
        {
            var a = np.logspace(2.0, 3.0, num: 4);
            AssertArray(a, new double[] { 100.0, 215.443469, 464.15888336, 1000.0 });
            print(a);

            var b = np.logspace(2.0, 3.0, num: 4, endpoint: false);
            AssertArray(b, new double[] { 100.0, 177.827941, 316.22776602, 562.34132519 });
            print(b);

            var c = np.logspace(2.0, 3.0, num: 4, _base: 2.0);
            AssertArray(c, new double[] { 4.0, 5.0396842, 6.34960421, 8.0 });
            print(c);
        }

        [TestMethod]
        public void test_geomspace_1()
        {
            var a = np.geomspace(1, 1000, num: 4);
            AssertArray(a, new double[] { 1.0, 10.0, 100.0, 1000.0 });
            print(a);

            var b = np.geomspace(1, 1000, num: 3, endpoint: false);
            AssertArray(b, new double[] { 1.0, 10.0, 100.0 });
            print(b);

            var c = np.geomspace(1, 1000, num: 4, endpoint: false);
            AssertArray(c, new double[] { 1.0, 5.62341325, 31.6227766, 177.827941 });
            print(c);

            var d = np.geomspace(1, 256, num: 9);
            AssertArray(d, new double[] { 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0 });
            print(d);
        }

        [TestMethod]
        public void test_meshgrid_1()
        {
            int nx = 3;
            int ny = 2;

            double ret = 0;

            var x = np.linspace(0, 1, ref ret, nx);
            var y = np.linspace(0, 1, ref ret, ny);

            ndarray[] xv = np.meshgrid(new ndarray[] { x });
            AssertArray(xv[0], new double[] { 0.0, 0.5, 1.0 });
            print(xv[0]);

            print("************");

            ndarray[] xyv = np.meshgrid(new ndarray[] { x, y });
            AssertArray(xyv[0], new double[,] { { 0.0, 0.5, 1.0 }, { 0.0, 0.5, 1.0 } });
            AssertArray(xyv[1], new double[,] { { 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 } });

            print(xyv[0]);
            print(xyv[1]);

            print("************");

            xyv = np.meshgrid(new ndarray[] { x, y }, sparse: true);
            AssertArray(xyv[0], new double[,] { { 0.0, 0.5, 1.0 } });
            AssertArray(xyv[1], new double[,] { { 0.0 }, { 1.0 } });

            print(xyv[0]);
            print(xyv[1]);

            print("************");

            x = np.arange(-5, 5, 1);
            y = np.arange(-5, 5, 1);
            xyv = np.meshgrid(new ndarray[] { x, y }, sparse: true);
            AssertArray(xyv[0], new Int32[,] { { -5, -4, -3, -2, -1, 0, 1, 2, 3, 4 } });
            AssertArray(xyv[1], new Int32[,] { { -5 }, { -4 }, { -3 }, { -2 }, { -1 }, { 0 }, { 1 }, { 2 }, { 3 }, { 4 } });
            print(xyv[0]);
            print(xyv[1]);

            print("************");
        }

        [TestMethod]
        public void test_meshgrid_2()
        {
            int nx = 3;
            int ny = 2;
            int nz = 2;

            double ret = 0;

            var x = np.linspace(0, 1, ref ret, nx);
            var y = np.linspace(4, 5, ref ret, ny);
            var z = np.linspace(8, 9, ref ret, nz);

            ndarray[] xyzv = np.meshgrid(new ndarray[] { x, y, z }, indexing: "ij");
            AssertArray(xyzv[0], new double[,,] { { { 0.0, 0.0 }, { 0.0, 0.0 } }, { { 0.5, 0.5 }, { 0.5, 0.5 } }, { { 1.0, 1.0 }, { 1.0, 1.0 } } });
            AssertArray(xyzv[1], new double[,,] { { { 4.0, 4.0 }, { 5.0, 5.0 } }, { { 4.0, 4.0 }, { 5.0, 5.0 } }, { { 4.0, 4.0 }, { 5.0, 5.0 } } });
            AssertArray(xyzv[2], new double[,,] { { { 8.0, 9.0 }, { 8.0, 9.0 } }, { { 8.0, 9.0 }, { 8.0, 9.0 } }, { { 8.0, 9.0 }, { 8.0, 9.0 } } });

            print(xyzv[0]);
            print(xyzv[1]);
            print(xyzv[2]);

            print("************");

            xyzv = np.meshgrid(new ndarray[] { x, y, z }, sparse: true);
            AssertArray(xyzv[0], new double[,,] { { { 0.0 }, { 0.5 }, { 1.0 } } });
            AssertArray(xyzv[1], new double[,,] { { { 4.0 } }, { { 5.0 } } });
            AssertArray(xyzv[2], new double[,,] { { { 8.0, 9.0 } } });

            print(xyzv[0]);
            print(xyzv[1]);
            print(xyzv[2]);

            print("************");



            x = np.arange(-2, 2, 1);
            y = np.arange(-2, 2, 1);
            z = np.arange(2, -2, -1);

            xyzv = np.meshgrid(new ndarray[] { x, y, z }, copy: true);

            var ExpectedData1 = new Int32[,,]
            {{{-2, -2, -2, -2}, {-1, -1, -1, -1},{ 0, 0,  0,  0}, { 1,  1,  1,  1}},
             {{-2, -2, -2, -2}, {-1, -1, -1, -1},{ 0, 0,  0,  0}, { 1,  1,  1,  1}},
             {{-2, -2, -2, -2}, {-1, -1, -1, -1},{ 0, 0,  0,  0}, { 1,  1,  1,  1}},
             {{-2, -2, -2, -2}, {-1, -1, -1, -1},{ 0, 0,  0,  0}, { 1,  1,  1,  1}}};
            AssertArray(xyzv[0], ExpectedData1);


            print(xyzv[0]);
            print(xyzv[1]);
            print(xyzv[2]);

            print("************");
        }

        [TestMethod]
        public void test_OneDimensionalArray()
        {
            double[] l = new double[] { 12.23f, 13.32f, 100f, 36.32f };
            print("Original List:", l);
            var a = np.array(l);
            print("One-dimensional numpy array: ", a);
            print(a.shape);
            print(a.strides);

            AssertArray(a, l);
            AssertShape(a, 4);
            AssertStrides(a, sizeof(double));
        }

        [TestMethod]
        public void test_arange_2to11()
        {
            var a = np.arange(2, 11, 1, dtype: np.Int8);
            print(a);

            print(a.shape);
            print(a.strides);

            AssertArray(a, new sbyte[] { 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            AssertShape(a, 9);
            AssertStrides(a, sizeof(sbyte));
        }

        [TestMethod]
        public void test_arange_2to11_double()
        {
            var a = np.arange(2.5, 11.5, 2, np.Float64);
            print(a);

            print(a.shape);
            print(a.strides);


            AssertArray(a, new double[] { 2.5, 4.5, 6.5, 8.5, 10.5 });
            AssertShape(a, 5);
            AssertStrides(a, sizeof(double));
        }

        [TestMethod]
        public void test_arange_2to11_float()
        {
            var a = np.arange(2.5, 37.7, 2.2, dtype: np.Float32);
            print(a);

            print(a.shape);
            print(a.strides);

            AssertArray(a, new float[] { 2.5f, 4.7f, 6.9f, 9.1f, 11.3f, 13.5f, 15.7f, 17.9f, 20.1f, 22.3f, 24.5f, 26.7f, 28.9f, 31.1f, 33.3f, 35.5f });
            AssertShape(a, 16);
            AssertStrides(a, sizeof(float));
        }

        [TestMethod]
        public void test_arange_reshape_33()
        {
            var a = np.arange(2, 11).reshape(new shape(3, 3));
            print(a);

            print(a.shape);
            print(a.strides);

            AssertArray(a, new Int32[3, 3] { { 2, 3, 4 }, { 5, 6, 7 }, { 8, 9, 10 } });
            AssertShape(a, 3, 3);
            AssertStrides(a, sizeof(Int32) * 3, sizeof(Int32));

        }

        [TestMethod]
        public void test_reshape_maxnd_1()
        {
            var a = np.arange(2, 11).reshape(new shape(3, 3));
            a = a.reshape(new long[] { 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
            Assert.AreEqual(32, a.ndim);
            var dimsarray = np.array(a.dims);
            var stridearray = np.array(a.strides);
            AssertArray(dimsarray, new npy_intp[] { 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
            AssertArray(stridearray, new npy_intp[] { 12, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 });

            a = a.reshape(new int[] { 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
            Assert.AreEqual(32, a.ndim);
            dimsarray = np.array(a.dims);
            stridearray = np.array(a.strides);
            AssertArray(dimsarray, new npy_intp[] { 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
            AssertArray(stridearray, new npy_intp[] { 12, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 });

            try
            {
                a = a.reshape(new long[] { 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
                Assert.Fail("This should have thrown an exception");
            }
            catch (Exception ex)
            {
                print(ex.Message);
            }

            try
            {
                a = a.reshape(new int[] { 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
                Assert.Fail("This should have thrown an exception");
            }
            catch (Exception ex)
            {
                print(ex.Message);
            }

        }

        [TestMethod]
        public void test_reshape_maxnd_2()
        {
            var a = np.arange(2, 11).reshape(new shape(3, 3));
            a = a.reshape((long)3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 );
            Assert.AreEqual(32, a.ndim);
            var dimsarray = np.array(a.dims);
            var stridearray = np.array(a.strides);
            AssertArray(dimsarray, new npy_intp[] { 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
            AssertArray(stridearray, new npy_intp[] { 12, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 });

            a = a.reshape(3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
            Assert.AreEqual(32, a.ndim);
            dimsarray = np.array(a.dims);
            stridearray = np.array(a.strides);
            AssertArray(dimsarray, new npy_intp[] { 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
            AssertArray(stridearray, new npy_intp[] { 12, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 });

            try
            {
                a = a.reshape( (long)3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 );
                Assert.Fail("This should have thrown an exception");
            }
            catch (Exception ex)
            {
                print(ex.Message);
            }

            try
            {
                a = a.reshape( 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 );
                Assert.Fail("This should have thrown an exception");
            }
            catch (Exception ex)
            {
                print(ex.Message);
            }

        }


        [TestMethod]
        public void test_arange_reshape_53()
        {
            var a = np.arange(0, 15).reshape(new shape(5, 3));
            print(a);

            print(a.shape);
            print(a.strides);


            AssertArray(a, new Int32[5, 3] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 }, { 9, 10, 11 }, { 12, 13, 14 } });
            AssertShape(a, 5, 3);
            AssertStrides(a, sizeof(Int32) * 3, sizeof(Int32));
        }

        [TestMethod]
        public void test_reverse_array()
        {
            var x = np.arange(0, 40);
            print("Original array:");
            print(x);
            print("Reverse array:");
            //x = (ndarray)x[new Slice(null, null, -1)];
            x = (ndarray)x["::-1"];
            print(x);

            AssertArray(x, new Int32[] { 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 });
            AssertShape(x, 40);
            AssertStrides(x, -sizeof(float));

            var y = x + 100;
            print(y);

            var z = x.reshape((5, -1));
            print(z);
        }

        //[Ignore] // throws an error in python code
        [TestMethod]
        public void xxx_test_1_OnBorder_0Inside()
        {
            var x = np.ones(new shape(15, 15));
            print("Original array:");
            print(x);
            print(x.shape);
            print(x.strides);
            print("1 on the border and 0 inside in the array");
            //x[new Slice(1, -1, 1), new Slice(1, -1, 1)] = 0;
            x["1:-1", "1:-1"] = 0;
            print(x);
            print(x.shape);
            print(x.strides);

            var ExpectedData = new double[15, 15]

                {{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                 { 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                 { 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                 { 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                 { 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                 { 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                 { 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                 { 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                 { 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                 { 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                 { 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                 { 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                 { 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                 { 1.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                 { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0} };

            AssertArray(x, ExpectedData);
            AssertShape(x, 15, 15);
            AssertStrides(x, sizeof(double) * 15, sizeof(double));

        }

        [TestMethod]
        public void test_1_OnBorder_0Inside_2()
        {
            var x = np.arange(0, 225, dtype: np.Float64).reshape(new shape(15, 15));
            print("Original array:");
            print(x);
            print(x.shape);
            print(x.strides);
            print("1 on the border and 0 inside in the array");
            //x = (ndarray)x[new Slice(1, -1, 1), new Slice(1, -1, 1)];
            x = (ndarray)x["1:-1", "1:-1"];
            print(x);
            print(x.shape);
            print(x.strides);

            var ExpectedData = new double[13, 13]

                {{ 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0},
                 { 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0},
                 { 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0},
                 { 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0},
                 { 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0},
                 { 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0},
                 { 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, },
                 { 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, },
                 { 136.0, 137.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0, 148.0, },
                 { 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0, },
                 { 166.0, 167.0, 168.0, 169.0, 170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, },
                 { 181.0, 182.0, 183.0, 184.0, 185.0, 186.0, 187.0, 188.0, 189.0, 190.0, 191.0, 192.0, 193.0, },
                 { 196.0, 197.0, 198.0, 199.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,} };

            AssertArray(x, ExpectedData);
            AssertShape(x, 13, 13);
            AssertStrides(x, sizeof(double) * 15, sizeof(double));


        }

        [TestMethod]
        public void test_checkerboard_1()
        {
            var x = np.ones((3, 3));
            print("Checkerboard pattern:");
            x = np.zeros((8, 8), dtype: np.Int32);
            x["1::2", "::2"] = 1;
            x["::2", "1::2"] = 1;
            print(x);

            var ExpectedData = new Int32[8, 8]
            {
                 { 0, 1, 0, 1, 0, 1, 0, 1 },
                 { 1, 0, 1, 0, 1, 0, 1, 0 },
                 { 0, 1, 0, 1, 0, 1, 0, 1 },
                 { 1, 0, 1, 0, 1, 0, 1, 0 },
                 { 0, 1, 0, 1, 0, 1, 0, 1, },
                 { 1, 0, 1, 0, 1, 0, 1, 0, },
                 { 0, 1, 0, 1, 0, 1, 0, 1, },
                 { 1, 0, 1, 0, 1, 0, 1, 0, },
            };

            AssertArray(x, ExpectedData);
            AssertShape(x, 8, 8);
            AssertStrides(x, sizeof(Int32) * 8, sizeof(Int32));

        }

        [TestMethod]
        public void test_F2C_1()
        {
            float[] fvalues = new float[] { 0, 12, 45.21f, 34, 99.91f };
            ndarray F = (ndarray)np.array(fvalues);
            print("Values in Fahrenheit degrees:");
            print(F);
            print("Values in  Centigrade degrees:");

            ndarray C = 5 * F / 9 - 5 * 32 / 9;
            print(C);

            AssertArray(C, new float[] { -17.0f, -10.333334f, 8.116665f, 1.88888931f, 38.505558f });

        }

        [TestMethod]
        public void test_RealImage_float_1()
        {
            var x = np.array(new float[] { 1.0f, 1.5f, 1.74f, 1.9f });
            var y = np.array(new float[] { 0.1f, 0.3f, 0.5f, 0.7f });
            print("Original array:x ", x);
            print("Original array:y ", y);
            print("Real part of the array:");
            print(x.Real);
            print(y.Real);
            print("Imaginary part of the array:");
            print(x.Imag);
            print(y.Imag);

        }

        [TestMethod]
        public void test_ArrayStats_1()
        {
            var x = np.array(new double[] { 1, 2, 3 }, dtype: np.Float64);
            print("Size of the array: ", x.size);
            print("Length of one array element in bytes: ", x.ItemSize);
            print("Total bytes consumed by the elements of the array: ", x.nbytes);

            Assert.AreEqual(3, x.size);
            Assert.AreEqual(8, x.ItemSize);
            Assert.AreEqual(24, x.nbytes);

        }

        [TestMethod]
        public void test_tofile_fromfile_text()
        {
            var x = np.arange(0.73, 25.73, dtype: np.Float64).reshape(new shape(5, 5));

            var filename = "numpyDOTNETToFileTest.txt";
            x.tofile(filename, sep: ", ");
            var y = np.fromfile(filename, sep: ",");
            print(y);

            AssertArray(y, new Single[] { 0.73f, 1.73f, 2.73f, 3.73f, 4.73f, 5.73f, 6.73f, 7.73f, 8.73f, 9.73f,
                                         10.73f, 11.73f, 12.73f, 13.73f, 14.73f, 15.73f, 16.73f, 17.73f, 18.73f,
                                         19.73f, 20.73f, 21.73f, 22.73f, 23.73f, 24.73f });
        }

        [TestMethod]
        public void test_tofile_fromfile_binary()
        {
            var x = np.arange(0.73, 25.73, dtype: np.Float64).reshape(new shape(5, 5));

            var filename = "numpyDOTNETToFileTest.bin";
            x.tofile(filename);
            var y = np.fromfile(filename, dtype: np.Float64);
            print(y);

            x.tofile("numpyDotNetToFileTest.bin");

            AssertArray(y, new double[] { 0.73, 1.73, 2.73, 3.73, 4.73, 5.73, 6.73, 7.73, 8.73, 9.73,
                                         10.73, 11.73, 12.73, 13.73, 14.73, 15.73, 16.73, 17.73, 18.73,
                                         19.73, 20.73, 21.73, 22.73, 23.73, 24.73 });
        }


        [TestMethod]
        public void test_tofile_fromfile_binary_2()
        {
            var x = np.arange(2, 27, dtype: np.UInt8).reshape(new shape(5, 5));

            var filename = "numpyDOTNETToFileTest.bin";
            x.tofile(filename);
            var y = np.fromfile(filename, dtype: np.UInt8);
            print(y);

            x.tofile("numpyDotNetToFileTest.bin");

            AssertArray(y, new byte[] {  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26});
        }

        [TestMethod]
        public void test_tofile_fromfile_binary_3()
        {
            var x = np.array(new System.Numerics.Complex[] { new Complex(1, 2), new Complex(3, 4), new Complex(5, 6) });

            var filename = "numpyDOTNETToFileTest.bin";
            x.tofile(filename);
            var y = np.fromfile(filename, dtype: np.Complex);
            print(y);

            x.tofile("numpyDotNetToFileTest.bin");

            AssertArray(y, new System.Numerics.Complex[] { new Complex(1, 2), new Complex(3, 4), new Complex(5, 6) });
        }

   

        [TestMethod]
        public void test_ndarray_fill()
        {
            var x = np.arange(0, 9, dtype: np.Float64).reshape(new shape(3, 3));
            x.fill(33);
            print(x);
            AssertArray(x, new double[,] { { 33.0, 33.0, 33.0 }, { 33.0, 33.0, 33.0 }, { 33.0, 33.0, 33.0 } });

            var y = x["1:2", "1:2"] as ndarray;
            print(y);

            y.fill(44);
            print(x);

            // y.fill("AAAA");
            // print(x);

            x = np.arange(0, 9, dtype: np.Complex).reshape(new shape(3, 3));
            x.fill(123);
            print(x);

            x = np.arange(0, 9, dtype: np.Decimal).reshape(new shape(3, 3));
            x.fill(166);
            print(x);

            //y.fill(new Complex(11, 11));
            //print(x);

        }

        [TestMethod]
        public void test_ndarray_flatten()
        {
            var x = np.arange(0.73, 25.73, dtype: np.Float64).reshape(new shape(5, 5));
            var y = x.flatten();
            print(x);
            print(y);

            AssertArray(y, new double[] { 0.73, 1.73, 2.73, 3.73, 4.73, 5.73, 6.73, 7.73, 8.73, 9.73,
                                         10.73, 11.73, 12.73, 13.73, 14.73, 15.73, 16.73, 17.73, 18.73,
                                         19.73, 20.73, 21.73, 22.73, 23.73, 24.73 });

            y = x.flatten(order: NPY_ORDER.NPY_FORTRANORDER);
            print(y);

            AssertArray(y, new double[] { 0.73, 5.73, 10.73, 15.73, 20.73,  1.73, 6.73, 11.73, 16.73,
                                         21.73, 2.73,  7.73, 12.73, 17.73, 22.73, 3.73, 8.73, 13.73, 18.73,
                                         23.73, 4.73,  9.73, 14.73, 19.73, 24.73 });

            y = x.flatten(order: NPY_ORDER.NPY_KORDER);
            print(y);

            AssertArray(y, new double[] { 0.73, 1.73, 2.73, 3.73, 4.73, 5.73, 6.73, 7.73, 8.73, 9.73,
                                         10.73, 11.73, 12.73, 13.73, 14.73, 15.73, 16.73, 17.73, 18.73,
                                         19.73, 20.73, 21.73, 22.73, 23.73, 24.73 });
        }

        [TestMethod]
        public void test_ndarray_byteswap()
        {
            var x = np.arange(32, 64, dtype: np.Int16);
            print(x);
            var y = x.byteswap(true);
            print(y);

            AssertArray(y, new Int16[] { 8192,  8448,  8704,  8960,  9216,  9472,  9728,  9984,
                                        10240, 10496, 10752, 11008, 11264, 11520, 11776,
                                        12032, 12288, 12544, 12800, 13056, 13312, 13568, 13824, 14080,
                                        14336, 14592, 14848, 15104, 15360, 15616, 15872, 16128 });

            x = np.arange(32, 64, dtype: np.Int32);
            print(x);
            y = x.byteswap(true);
            print(y);

            AssertArray(y, new Int32[] { 536870912,  553648128,  570425344,  587202560,  603979776,
                                         620756992,  637534208,  654311424,  671088640,  687865856,
                                         704643072,  721420288,  738197504,  754974720,  771751936,
                                         788529152,  805306368,  822083584,  838860800,  855638016,
                                         872415232,  889192448,  905969664,  922746880,  939524096,
                                         956301312,  973078528,  989855744, 1006632960, 1023410176,
                                         1040187392, 1056964608});

            x = np.arange(32, 64, dtype: np.Int64);
            print(x);
            y = x.byteswap(true);
            print(y);

            AssertArray(y, new Int64[] { 2305843009213693952, 2377900603251621888, 2449958197289549824,
                                         2522015791327477760, 2594073385365405696, 2666130979403333632,
                                         2738188573441261568, 2810246167479189504, 2882303761517117440,
                                         2954361355555045376, 3026418949592973312, 3098476543630901248,
                                         3170534137668829184, 3242591731706757120, 3314649325744685056,
                                         3386706919782612992, 3458764513820540928, 3530822107858468864,
                                         3602879701896396800, 3674937295934324736, 3746994889972252672,
                                         3819052484010180608, 3891110078048108544, 3963167672086036480,
                                         4035225266123964416, 4107282860161892352, 4179340454199820288,
                                         4251398048237748224, 4323455642275676160, 4395513236313604096,
                                         4467570830351532032, 4539628424389459968});

        }

        [TestMethod]
        public void test_ndarray_view()
        {
            var x = np.arange(256 + 32, 256 + 64, dtype: np.Int16);
            print(x);
            print(x.shape);
            print(x.Dtype);

            AssertArray(x, new Int16[] { 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
                                         299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
                                         310, 311, 312, 313, 314, 315, 316, 317, 318, 319});


            var y = x.view(np.Int8);
            print(y);
            print(y.shape);
            print(y.Dtype);

            AssertArray(y, new sbyte[] { 32,  1, 33,  1, 34,  1, 35,  1, 36,  1, 37,  1, 38,  1,
                                         39,  1, 40,  1, 41,  1, 42,  1, 43,  1, 44,  1, 45,  1,
                                         46,  1, 47,  1, 48,  1, 49,  1, 50,  1, 51,  1, 52,  1,
                                         53,  1, 54,  1, 55,  1, 56,  1, 57,  1, 58,  1, 59,  1,
                                         60,  1, 61,  1, 62,  1, 63,  1});


            print("modifying data");
            y[1] = 99;
            print(x);

            AssertArray(x, new Int16[] { 25376, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
                                         299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
                                         310, 311, 312, 313, 314, 315, 316, 317, 318, 319});

        }

        [TestMethod]
        public void test_ndarray_view_1()
        {
            var x = np.arange(0, 32, dtype: np.Int16).reshape(new shape(2, -1, 4));

            print("X");
            print(x);
            print(x.shape);

            var ExpectedDataX = new Int16[2, 4, 4]
            {
                {
                    { 0, 1, 2, 3},
                    { 4, 5, 6, 7},
                    { 8, 9, 10, 11 },
                    { 12, 13, 14, 15},
                },
                {
                    { 16, 17, 18, 19},
                    { 20, 21, 22, 23},
                    { 24, 25, 26, 27},
                    { 28, 29, 30, 31},
                },
            };

            AssertArray(x, ExpectedDataX);
            AssertShape(x, 2, 4, 4);


            var y = x.T;


            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Int16[4, 4, 2]
            {
                {
                    { 0, 16},
                    { 4, 20},
                    { 8, 24},
                    { 12, 28},
                },
                {
                    { 1, 17},
                    { 5, 21},
                    { 9, 25},
                    { 13, 29},
                },
                {
                    { 2, 18},
                    { 6, 22},
                    { 10, 26},
                    { 14, 30},
                },
                {
                    { 3, 19},
                    { 7, 23},
                    { 11, 27},
                    { 15, 31},
                },
            };
            AssertArray(y, ExpectedDataY);
            AssertShape(y, 4, 4, 2);

            var z = y.view();
            z[0] = 99;
            print("Z");
            print(z);
            print(z.shape);

            var ExpectedDataZ = new Int16[4, 4, 2]
            {
                {
                    { 99, 99},
                    { 99, 99},
                    { 99, 99},
                    { 99, 99},
                },
                {
                    { 1, 17},
                    { 5, 21},
                    { 9, 25},
                    { 13, 29},
                },
                {
                    { 2, 18},
                    { 6, 22},
                    { 10, 26},
                    { 14, 30},
                },
                {
                    { 3, 19},
                    { 7, 23},
                    { 11, 27},
                    { 15, 31},
                },
            };
            AssertArray(z, ExpectedDataZ);
            AssertShape(z, 4, 4, 2);


            print("X");
            print(x);

            ExpectedDataX = new Int16[2, 4, 4]
            {
                {
                    { 99, 1, 2, 3},
                    { 99, 5, 6, 7},
                    { 99, 9, 10, 11 },
                    { 99, 13, 14, 15},
                },
                {
                    { 99, 17, 18, 19},
                    { 99, 21, 22, 23},
                    { 99, 25, 26, 27},
                    { 99, 29, 30, 31},
                },
            };

            AssertArray(x, ExpectedDataX);
            AssertShape(x, 2, 4, 4);


            print("Y");
            print(y);

            ExpectedDataY = new Int16[4, 4, 2]
            {
                {
                    { 99, 99},
                    { 99, 99},
                    { 99, 99},
                    { 99, 99},
                },
                {
                    { 1, 17},
                    { 5, 21},
                    { 9, 25},
                    { 13, 29},
                },
                {
                    { 2, 18},
                    { 6, 22},
                    { 10, 26},
                    { 14, 30},
                },
                {
                    { 3, 19},
                    { 7, 23},
                    { 11, 27},
                    { 15, 31},
                },
            };
            AssertArray(y, ExpectedDataY);
            AssertShape(y, 4, 4, 2);

        }

        [TestMethod]
        public void test_ndarray_view2()
        {
            var x = np.arange(256 + 32, 256 + 64, dtype: np.Int16);
            print(x);
            print(x.shape);
            print(x.Dtype);

            AssertArray(x, new Int16[] { 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
                                         299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
                                         310, 311, 312, 313, 314, 315, 316, 317, 318, 319});

            var y = x.view(np.Int32);
            print(y);
            print(x.shape);
            print(x.Dtype);

            AssertArray(y, new Int32[] { 18940192, 19071266, 19202340, 19333414, 19464488, 19595562,
                                         19726636, 19857710, 19988784, 20119858, 20250932, 20382006,
                                         20513080, 20644154, 20775228, 20906302});


            print("modifying data");
            y[1] = 99;
            y[5] = 88;
            print(y);

            AssertArray(y, new Int32[] { 18940192, 99, 19202340, 19333414, 19464488, 88,
                                         19726636, 19857710, 19988784, 20119858, 20250932, 20382006,
                                         20513080, 20644154, 20775228, 20906302});

            print(x);

            AssertArray(x, new Int16[] { 288, 289, 99, 0, 292, 293, 294, 295, 296, 297, 88,
                                         0, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
                                         310, 311, 312, 313, 314, 315, 316, 317, 318, 319});

        }

        [TestMethod]
        public void test_ndarray_view2_reshape()
        {
            var x = np.arange(65470 + 32, 65470 + 64, dtype: np.UInt16).reshape(new shape(2, 2, -1));
            print(x);
            print(x.shape);
            print(x.Dtype);

            var ExpectedDataX = new UInt16[2, 2, 8]
            {
                {
                    { 65502, 65503, 65504, 65505, 65506, 65507, 65508, 65509},
                    { 65510, 65511, 65512, 65513, 65514, 65515, 65516, 65517},
                },
                {
                    { 65518, 65519, 65520, 65521, 65522, 65523, 65524, 65525},
                    { 65526, 65527, 65528, 65529, 65530, 65531, 65532, 65533},
                },
            };

            AssertArray(x, ExpectedDataX);

            var z = (ndarray)x[":", ":", "[2]"];
            print(z);

            var ExpectedDataZ = new UInt16[2, 2, 1]
            {
                {
                    { 65504 },
                    { 65512 },
                },
                {
                    { 65520 },
                    { 65528 },
                },
            };
            AssertArray(z, ExpectedDataZ);


            var y = z.view().reshape(-1);
            print(y);
            print(x.shape);
            print(x.Dtype);

            var ExpectedDataY = new UInt16[4] { 65504, 65512, 65520, 65528 };
            AssertArray(y, ExpectedDataY);
        }

        [TestMethod]
        public void test_ndarray_view3()
        {
            var x = np.arange(256 + 32, 256 + 64, dtype: np.Int16);
            print(x);
            print(x.shape);
            print(x.Dtype);

            AssertArray(x, new Int16[] { 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
                                         299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
                                         310, 311, 312, 313, 314, 315, 316, 317, 318, 319});

            var y = x.view(np.Int64);
            print(y);
            print(x.shape);
            print(x.Dtype);

            AssertArray(y, new Int64[] { 81910463782256928, 83036380869230884, 84162297956204840,
                                         85288215043178796, 86414132130152752, 87540049217126708,
                                         88665966304100664, 89791883391074620});


            print("modifying data");
            y[1] = 99;
            y[5] = 88;
            print(y);

            AssertArray(y, new Int64[] { 81910463782256928, 99, 84162297956204840,
                                         85288215043178796, 86414132130152752, 88,
                                         88665966304100664, 89791883391074620});


            print(x);


            AssertArray(x, new Int16[] { 288, 289, 290, 291, 99, 0, 0, 0, 296, 297, 298,
                                         299, 300, 301, 302, 303, 304, 305, 306, 307, 88, 0,
                                         0, 0, 312, 313, 314, 315, 316, 317, 318, 319});


        }

        [TestMethod]
        public void test_ndarray_delete1()
        {
            var x = np.arange(0, 32, dtype: np.Int16).reshape(new shape(8, 4));
            print("X");
            print(x);
            print(x.shape);

            var ExpectedDataX = new Int16[8, 4]
            {
                    { 0, 1, 2, 3},
                    { 4, 5, 6, 7},
                    { 8, 9, 10, 11 },
                    { 12, 13, 14, 15},
                    { 16, 17, 18, 19},
                    { 20, 21, 22, 23},
                    { 24, 25, 26, 27},
                    { 28, 29, 30, 31},
            };

            AssertArray(x, ExpectedDataX);
            AssertShape(x, 8, 4);

            var y = np.delete(x, 0, axis:1);
            y[1] = 99;
            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Int16[8, 3]
            {
                    { 1, 2, 3},
                    { 99, 99, 99},
                    { 9, 10, 11 },
                    { 13, 14, 15},
                    { 17, 18, 19},
                    { 21, 22, 23},
                    { 25, 26, 27},
                    { 29, 30, 31},
            };

            AssertArray(y, ExpectedDataY);
            AssertShape(y, 8, 3);

            print("X");
            print(x);


            AssertArray(x, ExpectedDataX);
            AssertShape(x, 8, 4);
        }

        [TestMethod]
        public void test_ndarray_delete2()
        {
            var x = np.arange(0, 32, dtype: np.Int16);
            print("X");
            print(x);
            print(x.shape);

            var ExpectedDataX = new Int16[] {0,  1,  2,  3,  4,  5,  6,  7,
                                             8,  9,  10, 11, 12, 13, 14, 15,
                                             16, 17, 18, 19, 20, 21, 22, 23,
                                             24, 25, 26, 27, 28, 29, 30, 31 };
            AssertArray(x, ExpectedDataX);
            AssertShape(x, 32);

            var y = np.delete(x, 1, 0);
            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Int16[] {0,  2,  3,  4,  5,  6,  7,
                                             8,  9,  10, 11, 12, 13, 14, 15,
                                             16, 17, 18, 19, 20, 21, 22, 23,
                                             24, 25, 26, 27, 28, 29, 30, 31 };
            AssertArray(y, ExpectedDataY);
            AssertShape(y, 31);

            print("X");
            print(x);

            AssertArray(x, ExpectedDataX);
        }

        [TestMethod]
        public void test_ndarray_delete3()
        {
            var x = np.arange(0, 32, dtype: np.Int16).reshape(new shape(8, 4));
            print("X");
            print(x);
            print(x.shape);

            var ExpectedDataX = new Int16[8, 4]
            {
                { 0, 1, 2, 3},
                { 4, 5, 6, 7},
                { 8, 9, 10, 11 },
                { 12, 13, 14, 15},
                { 16, 17, 18, 19},
                { 20, 21, 22, 23},
                { 24, 25, 26, 27},
                { 28, 29, 30, 31},
            };

            AssertArray(x, ExpectedDataX);
            AssertShape(x, 8, 4);

            var mask = np.ones_like(x, dtype: np.Bool);
            mask[new Slice(null), 0] = false;
            print(mask);

            var ExpectedDataMask = new bool[8, 4]
            {
                { false, true, true, true },
                { false, true, true, true },
                { false, true, true, true },
                { false, true, true, true },
                { false, true, true, true },
                { false, true, true, true },
                { false, true, true, true },
                { false, true, true, true },
            };

            AssertArray(mask, ExpectedDataMask);
            AssertShape(mask, 8, 4);

            var y = ((ndarray)(x[mask])).reshape(new shape(8, 3));

            print("Y");
            print(y);

            var ExpectedDataY = new Int16[8, 3]
            {
                { 1, 2, 3},
                { 5, 6, 7},
                { 9, 10, 11 },
                { 13, 14, 15},
                { 17, 18, 19},
                { 21, 22, 23},
                { 25, 26, 27},
                { 29, 30, 31},
            };

            AssertArray(y, ExpectedDataY);
            AssertShape(y, 8, 3);


            print("X");
            print(x);

            AssertArray(x, ExpectedDataX);
            AssertShape(x, 8, 4);
        }

        [TestMethod]
        public void test_pythonindexing_1()
        {
            var x = np.arange(0, 32, dtype: np.Int16).reshape(new shape(2, -1, 4));

            print("X");
            print(x);
            print(x.shape);

            var y = (ndarray)x["1:"];
            var z = (ndarray)x[new Slice(1, null)];

            // Y and Z should be the same
            print("Y");
            print(y);
            print(y.shape);

            print("Z");
            print(z);
            print(z.shape);

            var ExpectedDataY = new Int16[1, 4, 4]
            {
                {
                    { 16, 17, 18, 19},
                    { 20, 21, 22, 23},
                    { 24, 25, 26, 27},
                    { 28, 29, 30, 31},
                },
            };

            AssertArray(y, ExpectedDataY);
            AssertShape(y, 1, 4, 4);

            AssertArray(z, ExpectedDataY);
            AssertShape(z, 1, 4, 4);

        }

        [TestMethod]
        public void test_ndarray_unique_1()
        {
            var x = np.array(new Int32[] { 1, 2, 3, 1, 3, 4, 5, 4, 4 });

            print("X");
            print(x);

            var result = np.unique(x, return_counts: true, return_index: true, return_inverse: true);
            var uvalues = result.data;
            var indexes = result.indices;
            var inverse = result.inverse;
            var counts = result.counts;

            print("uvalues");
            print(uvalues);
            AssertArray(uvalues, new Int32[] { 1, 2, 3, 4, 5 });

            print("indexes");
            print(indexes);
            AssertArray(indexes, new npy_intp[] { 0, 1, 2, 5, 6 });

            print("inverse");
            print(inverse);
            AssertArray(inverse, new npy_intp[] { 0, 1, 2, 0, 2, 3, 4, 3, 3 });

            print("counts");
            print(counts);
            AssertArray(counts, new npy_intp[] { 2, 1, 2, 3, 1 });


        }

        [TestMethod]
        public void test_ndarray_where_1()
        {
            var x = np.array(new Int32[] { 1, 2, 3, 1, 3, 4, 5, 4, 4 }).reshape(new shape(3, 3));

            print("X");
            print(x);

            ndarray[] y = (ndarray[])np.where(x == 3);
            print("Y");
            print(y);


        }

        [TestMethod]
        public void test_ndarray_where_2()
        {
            var x = np.array(new UInt32[] { 1, 2, 3, 1, 3, 4, 5, 4, 4 }).reshape(new shape(3, 3));

            print("X");
            print(x);

            ndarray[] y = (ndarray[])np.where(x == 3);
            print("Y");
            print(y);

            Assert.AreEqual(2, y.Length);
            AssertArray(y[0], new npy_intp[] { 0, 1 });
            AssertArray(y[1], new npy_intp[] { 2, 1 });

            var z = x.SliceMe(y) as ndarray;
            print("Z");
            print(z);
            AssertArray(z, new UInt32[] { 3, 3 });
        }

        [TestMethod]
        public void test_ndarray_where_3()
        {
            var x = np.arange(0, 1000).reshape(new shape(-1, 10));

            //print("X");
            //print(x);

            ndarray[] y = (ndarray[])np.where(x % 10 == 0);
            print("Y");
            print(y);

            var z = x[y] as ndarray;
            print("Z");
            print(z);

            var ExpectedDataZ = new Int32[]
            {
                0,  10,  20,  30,  40,  50,  60,  70,  80,
                90, 100, 110, 120, 130, 140, 150, 160, 170,
                180, 190, 200, 210, 220, 230, 240, 250, 260,
                270, 280, 290, 300, 310, 320, 330, 340, 350,
                360, 370, 380, 390, 400, 410, 420, 430, 440,
                450, 460, 470, 480, 490, 500, 510, 520, 530,
                540, 550, 560, 570, 580, 590, 600, 610, 620,
                630, 640, 650, 660, 670, 680, 690, 700, 710,
                720, 730, 740, 750, 760, 770, 780, 790, 800,
                810, 820, 830, 840, 850, 860, 870, 880, 890,
                900, 910, 920, 930, 940, 950, 960, 970, 980, 990
            };

            AssertArray(z, ExpectedDataZ);

        }

        [TestMethod]
        public void test_ndarray_where_4()
        {
            var x = np.arange(0, 3000000, dtype: np.Int32);

            var y = np.where(x % 7 == 0);
            //print("Y");
            //print(y);

            var z = x[y] as ndarray;
            var m = np.mean(z);
            print("M");
            Assert.AreEqual(1499998.5, m.GetItem(0));
            print(m);

            return;
        }

        [TestMethod]
        public void test_ndarray_where_5()
        {
            var a = np.arange(10);

            var b = np.where(a < 5, a, 10 * a) as ndarray;
            AssertArray(b, new int[] { 0, 1, 2, 3, 4, 50, 60, 70, 80, 90 });
            print(b);

            a = np.array(new int[,] { { 0, 1, 2 }, { 0, 2, 4 }, { 0, 3, 6 } });
            b = np.where(a < 4, a, -1) as ndarray;  // -1 is broadcast
            AssertArray(b, new int[,] { { 0, 1, 2 }, { 0, 2, -1 }, { 0, 3, -1 } });
            print(b);

            var c = np.where(new bool[,] { { true, false }, { true, true } },
                                    new int[,] { { 1, 2 }, { 3, 4 } },
                                    new int[,] { { 9, 8 }, { 7, 6 } }) as ndarray;

            AssertArray(c, new int[,] { { 1, 8 }, { 3, 4 } });

            print(c);

            return;
        }

        [TestMethod]
        public void test_ndarray_unpackbits_1()
        {
            var x = np.arange(0, 12, dtype: np.UInt8).reshape(new shape(3, -1));

            print("X");
            print(x);

            var ExpectedDataX = new byte[3, 4]
            {
                { 0, 1, 2, 3},
                { 4, 5, 6, 7},
                { 8, 9, 10, 11 },
            };

            AssertArray(x, ExpectedDataX);
            AssertShape(x, 3, 4);

            var y = (ndarray)np.unpackbits(x, 1);

            print("Y");
            print(y);

            var ExpectedDataY = new byte[3, 32]
            {
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1},
                { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1},
                { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1 },
            };
            AssertArray(y, ExpectedDataY);
            AssertShape(y, 3, 32);

            var z = (ndarray)np.packbits(y, 1);

            print("Z");
            print(z);

            AssertArray(z, ExpectedDataX);
            AssertShape(z, 3, 4);

        }

        [TestMethod]
        public void test_arange_slice_1()
        {
            var a = np.arange(0, 1024, dtype: np.UInt16).reshape(new shape(2, 4, -1));

            print("A");
            // print(a);
            print(a.shape);
            print(a.strides);

            AssertShape(a, 2, 4, 128);
            AssertStrides(a, 1024, 256, 2);

            var b = (ndarray)a[":", ":", 122];
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new UInt16[2, 4]
            {
                { 122, 250, 378, 506},
                { 634, 762, 890, 1018 },
            };

            AssertArray(b, ExpectedDataB);
            AssertShape(b, 2, 4);
            AssertStrides(b, 1024, 256);

            var c = (ndarray)a.A(":", ":", new Int64[] { 122 });
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            var ExpectedDataC = new UInt16[2, 4, 1]
            {
                {
                    { 122 },
                    { 250 },
                    { 378 },
                    { 506 },
                },
                {
                    { 634 },
                    { 762 },
                    { 890 },
                    { 1018 },
                },

            };

            AssertArray(c, ExpectedDataC);
            AssertShape(c, 2, 4, 1);
            AssertStrides(c, 8, 2, 16);

            var d = (ndarray)a.A(":", ":", new Int64[] { 122, 123 });
            print("D");
            print(d);
            print(d.shape);
            print(d.strides);

            var ExpectedDataD = new UInt16[2, 4, 2]
            {
                {
                    { 122, 123 },
                    { 250, 251 },
                    { 378, 379 },
                    { 506, 507 },
                },
                {
                    { 634, 635 },
                    { 762, 763 },
                    { 890, 891 },
                    { 1018, 1019 },
                },

            };

            AssertArray(d, ExpectedDataD);
            AssertShape(d, 2, 4, 2);
            AssertStrides(d, 8, 2, 16);

        }

        [TestMethod]
        public void test_arange_slice_2()
        {
            var a = np.arange(0, 32, dtype: np.UInt16).reshape(new shape(2, 4, -1));

            print("A");
            // print(a);
            print(a.shape);
            print(a.strides);

            var b = (ndarray)a[":", ":", new int[] { 2 }];
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new UInt16[2, 4, 1]
            {
                {
                    { 2 },
                    { 6 },
                    { 10 },
                    { 14 },
                },
                {
                    { 18 },
                    { 22 },
                    { 26 },
                    { 30 },
                },

            };

            AssertArray(b, ExpectedDataB);
            AssertShape(b, 2, 4, 1);
            AssertStrides(b, 8, 2, 16);

        }

        [TestMethod]
        public void test_arange_slice_2A()
        {
            var a = np.arange(0, 32, dtype: np.UInt16).reshape(new shape(2, 4, -1));

            print("A");
            // print(a);
            print(a.shape);
            print(a.strides);

            var b = (ndarray)a[":", ":", np.where(a > 20)];
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new UInt16[,,,]
                {{{{1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
                   {1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3},
                   {1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3}},

                  {{5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5},
                   {5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7},
                   {5,  6,  7,  4,  5,  6,  7,  4,  5,  6,  7}},

                  {{9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9},
                   {9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11},
                   {9, 10, 11,  8,  9, 10, 11,  8,  9, 10, 11}},

                  {{13, 13 ,13, 13, 13, 13, 13, 13, 13, 13, 13},
                   {13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15},
                   {13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15}}},

                 {{{17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17},
                   {17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19},
                   {17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19}},

                  {{21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21},
                   {21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23},
                   {21, 22, 23, 20, 21, 22, 23, 20, 21, 22, 23}},

                  {{25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25},
                   {25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27},
                   {25, 26, 27, 24, 25, 26, 27, 24, 25, 26, 27}},

                  {{29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29},
                   {29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31},
                   {29, 30, 31, 28, 29, 30, 31, 28, 29, 30, 31}}}};

            AssertArray(b, ExpectedDataB);
            AssertStrides(b, 8, 2, 176, 16);
        }

        [TestMethod]
        public void test_arange_slice_2B()
        {
            var a = np.arange(0, 32, dtype: np.UInt16).reshape(new shape(2, 4, -1));
            var b = np.arange(100, 132, dtype: np.UInt16).reshape(new shape(2, 4, -1));
            print("A");
            // print(a);
            print(a.shape);
            print(a.strides);

            AssertShape(a, 2, 4, 4);
            AssertStrides(a, 32, 8, 2);

            b[":", ":", "[2]"] = a[":", ":", "[2]"];
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new UInt16[2, 4, 4]
            {
                {
                    { 100, 101, 2, 103 },
                    { 104, 105, 6, 107 },
                    { 108, 109, 10, 111 },
                    { 112, 113, 14, 115 },
                },
                {
                    { 116, 117, 18, 119 },
                    { 120, 121, 22, 123 },
                    { 124, 125, 26, 127 },
                    { 128, 129, 30, 131 },
                },

            };

            AssertArray(b, ExpectedDataB);
            AssertShape(b, 2, 4, 4);
            AssertStrides(b, 32, 8, 2);

        }

        [TestMethod]
        public void test_arange_slice_2C2()
        {
            var a = np.arange(0, 32, dtype: np.UInt16).reshape(new shape(2, 4, -1));
            var b = np.arange(100, 132, dtype: np.UInt16).reshape(new shape(2, 4, -1));
            print("A");
            // print(a);
            print(a.shape);
            print(a.strides);

            AssertShape(a, 2, 4, 4);
            AssertStrides(a, 32, 8, 2);

            // b has unexpected strides.  If a copy from A is made first
            ndarray aarray = (ndarray)a[":", ":", new int[] { 2 }];
            ndarray barray = (ndarray)b[":", ":", new int[] { 2 }];
            ndarray carray = barray | aarray;
            print("B");
            print(carray);
            print(carray.shape);
            print(carray.strides);

            var ExpectedDataC = new UInt16[2, 4, 1]
            {
                {
                    { 102 },
                    { 110 },
                    { 110 },
                    { 126 },
                },
                {
                    { 118 },
                    { 126 },
                    { 126 },
                    { 158 },
                },

            };

            AssertArray(carray, ExpectedDataC);
            AssertShape(carray, 2, 4, 1);
            AssertStrides(carray, 8, 2, 2);
        }

        [TestMethod]
        public void test_ndarray_NAN()
        {

            Int32 _max = 5;
            var output = np.ndarray(new shape(_max), dtype: np.Float32);
            output[":"] = np.NaN;

            print(output);
            print(output.shape);

            AssertArrayNAN(output, new float[] { np.NaN, np.NaN, np.NaN, np.NaN, np.NaN });
            AssertShape(output, 5);


        }

        [TestMethod]
        public void test_insert_1()
        {
            Int32[,] TestData = new int[,] { { 1, 1 }, { 2, 2 }, { 3, 3 } };
            ndarray a = np.array(TestData, dtype: np.Int32);
            ndarray b = np.insert(a, 1, 5);
            ndarray c = np.insert(a, 0, new int[] { 999, 100, 101 });

            print(a);
            print(a.shape);

            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new int[] { 1, 5, 1, 2, 2, 3, 3 });
            AssertShape(b, 7);
            AssertStrides(b, 4);

            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            AssertArray(c, new int[] { 999, 100, 101, 1, 1, 2, 2, 3, 3 });
            AssertShape(c, 9);
            AssertStrides(c, 4);
        }

        [TestMethod]
        public void test_insert_2()
        {
            Int32[] TestData1 = new int[] { 1, 1, 2, 2, 3, 3 };
            Int32[] TestData2 = new int[] { 90, 91, 92, 92, 93, 93 };

            ndarray a = np.array(TestData1, dtype: np.Int32);
            ndarray b = np.array(TestData2, dtype: np.Int32);
            ndarray c = np.insert(a, new Slice(null), b);

            print(a);
            print(a.shape);

            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new int[] { 90, 91, 92, 92, 93, 93 });
            AssertShape(b, 6);
            AssertStrides(b, 4);

            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            AssertArray(c, new int[] { 90, 1, 91, 1, 92, 2, 92, 2, 93, 3, 93, 3 });
            AssertShape(c, 12);
            AssertStrides(c, 4);

        }

        [TestMethod]
        public void test_append_1()
        {
            Int32[] TestData = new int[] { 1, 1, 2, 2, 3, 3 };
            ndarray a = np.array(TestData, dtype: np.Int32);
            ndarray b = np.append(a, (Int32)1);

            print(a);
            print(a.shape);

            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new int[] { 1, 1, 2, 2, 3, 3, 1 });
            AssertShape(b, 7);
            AssertStrides(b, 4);
        }

        [TestMethod]
        public void test_append_1d()
        {
            Decimal[] TestData = new Decimal[] { 5, 6, 2, 2, 3, 3 };
            ndarray a = np.array(TestData, dtype: np.Decimal);
            ndarray b = np.append(a, (Decimal)1);

            print(a);
            print(a.shape);

            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new Decimal[] { 5, 6, 2, 2, 3, 3, 1 });
            AssertShape(b, 7);
            AssertStrides(b, 16);
        }

        [TestMethod]
        public void test_append_2()
        {
            Int32[] TestData = new int[] { 1, 1, 2, 2, 3, 3 };
            Int32[] TestData2 = new int[] { 4, 4 };
            ndarray a = np.array(TestData, dtype: np.Int32);
            ndarray b = np.append(a, TestData2);

            print(a);
            print(a.shape);

            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new int[] { 1, 1, 2, 2, 3, 3, 4, 4 });
            AssertShape(b, 8);
            AssertStrides(b, 4);
        }

        [TestMethod]
        public void test_append_3()
        {
            Int32[] TestData1 = new int[] { 1, 1, 2, 2, 3, 3 };
            Int32[] TestData2 = new int[] { 4, 4, 5, 5, 6, 6 };
            ndarray a = np.array(TestData1, dtype: np.Int32);
            ndarray b = np.array(TestData2, dtype: np.Int32);

            ndarray c = np.append(a, b);

            print(a);
            print(a.shape);

            print(b);
            print(b.shape);

            print(c);
            print(c.shape);
            print(c.strides);

            AssertArray(c, new int[] { 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6 });
            AssertShape(c, 12);
            AssertStrides(c, 4);
        }

        [TestMethod]
        public void test_append_4()
        {
            Int32[] TestData1 = new int[] { 1, 1, 2, 2, 3, 3 };
            Int32[] TestData2 = new int[] { 4, 4, 5, 5, 6, 6 };
            ndarray a = np.array(TestData1, dtype: np.Int32).reshape((2, -1));
            ndarray b = np.array(TestData2, dtype: np.Int32).reshape((2, -1));

            ndarray c = np.append(a, b, axis: 1);

            print(a);
            print(a.shape);
            print("");

            print(b);
            print(b.shape);
            print("");

            print(c);
            print(c.shape);
            print(c.strides);
            print("");

            var ExpectedDataC = new Int32[,]
            {
                { 1, 1, 2, 4, 4, 5 },
                { 2, 3, 3, 5, 6, 6 },
            };

            AssertArray(c, ExpectedDataC);
            AssertShape(c, 2, 6);
            //AssertStrides(c, 24, 4); 

        }
   
        [TestMethod]
        public void test_flat_1()
        {
            var x = np.arange(10, 16).reshape(new shape(2, 3));
            print(x);

            x.Flat[3] = 9;
            print(x);

            var ExpectedDataX = new Int32[2, 3]
            {
                { 10, 11, 12 },
                { 9, 14, 15 },
            };
            AssertArray(x, ExpectedDataX);
            AssertShape(x, 2, 3);
            AssertStrides(x, 12, 4);



            var y = x.Flat.FlatView()[3];
            print(y);

            var z = x.Flat[3];
            print(z);
            print("");
            print("indexes");
            print("");

            List<Int32> indexes = new List<int>();
            foreach (var zz in x.Flat)
            {
                print(zz);
                indexes.Add((Int32)zz);
            }

            ndarray c = np.array(indexes.ToArray());
            AssertArray(c, new Int32[] { 10, 11, 12, 9, 14, 15 });
            AssertShape(c, 6);
            AssertStrides(c, 4);

        }

        [TestMethod]
        public void test_flat_2()
        {
            var x = np.arange(1, 7).reshape((2, 3));
            print(x);

            Assert.AreEqual(4, x.Flat[3]);
            print(x.Flat[3]);

            print(x.T);
            Assert.AreEqual(5, x.T.Flat[3]);
            print(x.T.Flat[3]);

            x.flat = 3;
            AssertArray(x, new int[,] { { 3, 3, 3 }, { 3, 3, 3 } });
            print(x);

            x.Flat[new int[] { 1, 4 }] = 1;
            AssertArray(x, new int[,] { { 3, 1, 3 }, { 3, 1, 3 } });
            print(x);
        }

        [TestMethod]
        public void test_intersect1d_1()
        {
            ndarray a = np.array(new int[] { 1, 3, 4, 3 });
            ndarray b = np.array(new int[] { 3, 1, 2, 1 });

            ndarray c = np.intersect1d(a, b);
            print(c);

            AssertArray(c, new Int32[] { 1, 3 });
            AssertShape(c, 2);
            AssertStrides(c, 4);

        }

        [TestMethod]
        public void test_setxor1d_1()
        {
            ndarray a = np.array(new int[] { 1, 2, 3, 2, 4 });
            ndarray b = np.array(new int[] { 2, 3, 5, 7, 5 });

            ndarray c = np.setxor1d(a, b);
            print(c);

            AssertArray(c, new Int32[] { 1, 4, 5, 7 });
            AssertShape(c, 4);
            AssertStrides(c, 4);
        }

        [TestMethod]
        public void test_in1d_1()
        {
            ndarray test = np.array(new int[] { 0, 1, 2, 5, 0 });
            ndarray states = np.array(new int[] { 0, 2 });

            ndarray mask = np.in1d(test, states);
            print(mask);
            print(test[mask]);

            AssertArray(mask, new bool[] { true, false, true, false, true });
            AssertShape(mask, 5);
            AssertStrides(mask, 1);

            ndarray a = test[mask] as ndarray;
            AssertArray(a, new Int32[] { 0, 2, 0 });
            AssertShape(a, 3);
            AssertStrides(a, 4);

            mask = np.in1d(test, states, invert: true);
            print(mask);
            print(test[mask]);

            AssertArray(mask, new bool[] { false, true, false, true, false });
            AssertShape(mask, 5);
            AssertStrides(mask, 1);

            ndarray b = test[mask] as ndarray;
            AssertArray(b, new Int32[] { 1, 5 });
            AssertShape(b, 2);
            AssertStrides(b, 4);

        }

        [TestMethod]
        public void test_isin_1()
        {
            ndarray element = 2 * np.arange(4).reshape(new shape(2, 2));
            print(element);

            ndarray test_elements = np.array(new int[] { 1, 2, 4, 8 });
            ndarray mask = np.isin(element, test_elements);
            print(mask);
            print(element[mask]);

            ndarray a = element[mask] as ndarray;

            var ExpectedDataMask = new bool[2, 2]
            {
                { false, true },
                { true, false },
            };

            AssertArray(mask, ExpectedDataMask);
            AssertShape(mask, 2, 2);
            AssertStrides(mask, 2, 1);

            AssertArray(a, new Int32[] { 2, 4 });
            AssertShape(a, 2);
            AssertStrides(a, 4);

            print("***********");

            mask = np.isin(element, test_elements, invert: true);
            print(mask);
            print(element[mask]);

            a = element[mask] as ndarray;


            ExpectedDataMask = new bool[2, 2]
            {
                { true, false },
                { false, true },
            };


            AssertArray(mask, ExpectedDataMask);
            AssertShape(mask, 2, 2);
            AssertStrides(mask, 2, 1);

            AssertArray(a, new Int32[] { 0, 6 });
            AssertShape(a, 2);
            AssertStrides(a, 4);
        }

        [TestMethod]
        public void test_union1d_1()
        {
            ndarray a1 = np.array(new int[] { -1, 0, 1 });
            ndarray a2 = np.array(new int[] { -2, 0, 2 });

            ndarray a = np.union1d(a1, a2);
            print(a);

            AssertArray(a, new Int32[] { -2, -1, 0, 1, 2 });
            AssertShape(a, 5);
            AssertStrides(a, 4);


        }

        [TestMethod]
        public void test_Ellipsis_indexing_1()
        {
            var a = np.array(new double[] { 10.0, 7, 4, 3, 2, 1 });

            var b = a.A("...", -1);
            Assert.AreEqual((double)1.0, b.GetItem(0));
            print(b);
            print("********");


            a = np.array(new double[,] { { 10.0, 7, 4 }, { 3, 2, 1 } });
            var c = a.A("...", -1);
            AssertArray(c, new double[] { 4.0, 1.0 });
            print(c);
            print("********");

            var TestData = new UInt32[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            a = np.array(TestData, dtype: np.UInt32).reshape((1, 3, 2, -1, 1));
            var d = a["...", -1] as ndarray;
            AssertArray(d, new UInt32[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } } });
            print(d);
            print("********");

            var e = a[0, "...", -1] as ndarray;
            AssertArray(e, new UInt32[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
            print(e);
            print("********");

            var f = a[0, ":", ":", ":", -1] as ndarray;
            AssertArray(f, new UInt32[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
            print(f);
            print("********");

            var g = a.A(0, 1, "...", -1);
            AssertArray(g, new UInt32[,] { { 5, 6 }, { 7, 8 } });
            print(g);
            print("********");

            var h = a.A(0, 2, 1, "...", -1);
            AssertArray(h, new UInt32[] { 11, 12 });
            print(h);
            print("********");

            var i = a[":", 2, 1, 1, "..."] as ndarray;
            AssertArray(i, new UInt32[,] { { 12 } });
            print(i);
        }

        [TestMethod]
        public void test_Ellipsis_indexing_2()
        {
            var a = np.array(new double[] { 10.0, 7, 4, 3, 2, 1 });

            var b = a.A(new Ellipsis(), -1);
            Assert.AreEqual((double)1.0, b.GetItem(0));
            print(b);
            print("********");


            a = np.array(new double[,] { { 10.0, 7, 4 }, { 3, 2, 1 } });
            var c = a.A(new Ellipsis(), -1);
            AssertArray(c, new double[] { 4.0, 1.0 });
            print(c);
            print("********");

            var TestData = new UInt32[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            a = np.array(TestData, dtype: np.UInt32).reshape((1, 3, 2, -1, 1));
            var d = a[new Ellipsis(), -1] as ndarray;
            AssertArray(d, new UInt32[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } } });
            print(d);
            print("********");

            var e = a[0, new Ellipsis(), -1] as ndarray;
            AssertArray(e, new UInt32[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
            print(e);
            print("********");

            var f = a[0, ":", ":", ":", -1] as ndarray;
            AssertArray(f, new UInt32[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
            print(f);
            print("********");

            var g = a.A(0, 1, new Ellipsis(), -1);
            AssertArray(g, new UInt32[,] { { 5, 6 }, { 7, 8 } });
            print(g);
            print("********");

            var h = a.A(0, 2, 1, new Ellipsis(), -1);
            AssertArray(h, new UInt32[] { 11, 12 });
            print(h);
            print("********");

            var i = a[":", 2, 1, 1, new Ellipsis()] as ndarray;
            AssertArray(i, new UInt32[,] { { 12 } });
            print(i);
        }

        [TestMethod]
        public void test_concatenate_1()
        {

            var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            var b = np.array(new int[,] { { 5, 6 } });
            var c = np.concatenate((a, b), axis: 0);
            AssertArray(c, new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            print(c);

            var d = np.concatenate((a, b.T), axis: 1);
            AssertArray(d, new int[,] { { 1, 2, 5 }, { 3, 4, 6 } });
            print(d);

            var e = np.concatenate((a, b), axis: null);
            AssertArray(e, new int[] { 1, 2, 3, 4, 5, 6 });
            print(e);

            var f = np.concatenate((np.eye(2), np.ones((2, 2))), axis: 0);
            AssertArray(f, new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 }, { 1, 1 }, });
            print(f);

            var g = np.concatenate((np.eye(2), np.ones((2, 2))), axis: 1);
            AssertArray(g, new double[,] { { 1, 0, 1, 1 }, { 0, 1, 1, 1 } });
            print(g);
        }

        [TestMethod]
        public void test_concatenate_2()
        {
            //var aa = np.array(new int[]  { 1, 2 ,  3, 4 ,  5, 6 });
            //var ca = np.concatenate(aa, axis: 0);

            var a = np.array(new int[,,,] { { { { 1, 2 }, { 3, 4 }, { 5, 6 } } } });
            var c = np.concatenate(a, axis: 0);
            AssertArray(c, new int[,,] { { { 1, 2 }, { 3, 4 }, { 5, 6 } } });
            print(c);

            var d = np.concatenate(a, axis: 1);
            AssertArray(d, new int[,,] { { { 1, 2 }, { 3, 4 }, { 5, 6 } } });
            print(d);

            var e = np.concatenate(a, axis: null);
            AssertArray(e, new int[] { 1, 2, 3, 4, 5, 6 });
            print(e);

        }

        [TestMethod]
        public void test_concatenate_3()
        {

            var a = np.array(new int[,,,] { { { { 1, 2 }, { 3, 4 }, { 5, 6 } } } });
            var c = np.concatenate(a, axis: -1);
            AssertArray(c, new int[,,] { { { 1, 2 }, { 3, 4 }, { 5, 6 } } });
            print(c);

            var d = np.concatenate(a, axis: -2);
            AssertArray(d, new int[,,] { { { 1, 2 }, { 3, 4 }, { 5, 6 } } });
            print(d);

            c = np.concatenate((a, a, a), axis: -1);
            AssertArray(c, new int[,,,] { { { { 1, 2, 1, 2, 1, 2 }, { 3, 4, 3, 4, 3, 4 }, { 5, 6, 5, 6, 5, 6 } } } });
            print(c);

            d = np.concatenate((a, a, a), axis: -2);
            AssertArray(d, new int[,,,] { { { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 1, 2 }, { 3, 4 }, { 5, 6 }, { 1, 2 }, { 3, 4 }, { 5, 6 } } } });
            print(d);


        }

        [TestMethod]
        public void test_multi_index_selection()
        {
            var x = np.arange(10, dtype: np.Int32);

            var y = x.reshape(new shape(2, 5));
            print(y);
            Assert.AreEqual(3, (int)y[0, 3]);
            Assert.AreEqual(8, (int)y[1, 3]);

            x = np.arange(20, dtype: np.Int32);
            y = x.reshape(new shape(2, 2, 5));
            print(y);
            Assert.AreEqual(3, (int)y[0, 0, 3]);
            Assert.AreEqual(8, (int)y[0, 1, 3]);

            Assert.AreEqual(13, (int)y[1, 0, 3]);
            Assert.AreEqual(18, (int)y[1, 1, 3]);

        }

        [TestMethod]
        public void test_multi_index_setting()
        {
            var x = np.arange(10, dtype: np.Int32);

            var y = x.reshape(new shape(2, 5));

            y[0, 3] = 55;
            y[1, 3] = 66;

            Assert.AreEqual(55, (int)y[0, 3]);
            Assert.AreEqual(66, (int)y[1, 3]);

            x = np.arange(20, dtype: np.Int32);
            y = x.reshape(new shape(2, 2, 5));

            y[1, 0, 3] = 55;
            y[1, 1, 3] = 66;

            Assert.AreEqual(55, (int)y[1, 0, 3]);
            Assert.AreEqual(66, (int)y[1, 1, 3]);

        }


        [TestMethod]
        public void test_newaxis_ufunc_1()
        {
            var x = np.arange(0, 4, 1, dtype: np.Float64);
            var y = x.reshape((4, 1)) * 4;
            var z = x + y;
            print(z.shape);
            print(z);
            print("*************");

            AssertArray(z, new double[,] { { 0.0, 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0, 7.0 },
                                           { 8.0, 9.0, 10.0, 11.0 }, { 12.0, 13.0, 14.0, 15.0 } });

            y = x.reshape((4, 1, 1)) * 4;
            z = x + y;
            print(z.shape);
            print(z);
            print("*************");

            AssertArray(z, new double[,,] { { { 0, 1, 2, 3, } }, { { 4, 5, 6, 7, } }, { { 8, 9, 10, 11, } }, { { 12, 13, 14, 15, } } });

            y = x.reshape((4, 1, 1, 1)) * 4;
            z = x + y;
            print(z.shape);
            print(z);
            print("*************");

            AssertArray(z, new double[,,,] { { { { 0, 1, 2, 3, } } }, { { { 4, 5, 6, 7, } } }, { { { 8, 9, 10, 11, } } }, { { { 12, 13, 14, 15, } } } });
            y = x.reshape((1, 4, 1)) * 4;
            z = x + y;
            print(z.shape);
            print(z);
            print("*************");
            AssertArray(z, new double[,,] { { { 0, 1, 2, 3, }, { 4, 5, 6, 7, }, { 8, 9, 10, 11, }, { 12, 13, 14, 15, } } });

            y = x.reshape((1, 4, 1, 1)) * 4;
            z = x + y;
            print(z.shape);
            print(z);
            print("*************");

            AssertArray(z, new double[,,,] { { { { 0, 1, 2, 3, } }, { { 4, 5, 6, 7, } }, { { 8, 9, 10, 11, } }, { { 12, 13, 14, 15, } } } });

        }

        [TestMethod]
        public void test_newaxis_ufunc_2()
        {
            var x = np.arange(0, 4, 1, dtype: np.Float32).reshape((2, 2));
            var y = x.reshape((2, 2, 1)) * 4;
            var z = x + y;
            print(z.shape);
            print(z);
            print("*************");
            AssertArray(z, new float[,,] { { { 0, 1, }, { 6, 7, } }, { { 8, 9, }, { 14, 15, } } });

            y = x.reshape((2, 2, 1, 1)) * 4;
            z = x + y;
            print(z.shape);
            print(z);
            print("*************");
            AssertArray(z, new float[,,,] { { { { 0, 1, }, { 2, 3, } }, { { 4, 5, }, { 6, 7, } } }, { { { 8, 9, }, { 10, 11, } }, { { 12, 13, }, { 14, 15, } } } });

            y = x.reshape((2, 2, 1, 1, 1)) * 4;
            z = x + y;
            print(z.shape);
            print(z);
            print("*************");
            AssertArray(z, new float[,,,,] { { { { { 0, 1, }, { 2, 3, } } }, { { { 4, 5, }, { 6, 7, } } } }, { { { { 8, 9, }, { 10, 11, } } }, { { { 12, 13, }, { 14, 15, } } } } });

            y = x.reshape((1, 2, 2, 1)) * 4;
            z = x + y;
            print(z.shape);
            print(z);
            print("*************");
            AssertArray(z, new float[,,,] { { { { 0, 1, }, { 6, 7, } }, { { 8, 9, }, { 14, 15, } } } });

            y = x.reshape((1, 2, 2, 1, 1)) * 4;
            z = x + y;
            print(z.shape);
            print(z);
            print("*************");
            AssertArray(z, new float[,,,,] { { { { { 0, 1, }, { 2, 3, } }, { { 4, 5, }, { 6, 7, } } }, { { { 8, 9, }, { 10, 11, } }, { { 12, 13, }, { 14, 15, } } } } });

        }


        [TestMethod]
        public void test_newaxis_ufunc_3()
        {
            var x1 = np.array(new Int32[] { 1, 2, 3, 4, 5 });
            var x2 = np.array(new Int32[] { 5, 4, 3 });

            var x1_new = (ndarray)x1[":", np.newaxis];
            var z = x1_new + x2;
            print(z);
            AssertArray(z, new Int32[,] { { 6, 5, 4 }, { 7, 6, 5 }, { 8, 7, 6 }, { 9, 8, 7 }, { 10, 9, 8 } });
            print("*************");

            x1_new = (ndarray)x1[np.newaxis, ":", np.newaxis];
            z = x1_new + x2;
            print(z);
            AssertArray(z, new Int32[,,] { { { 6, 5, 4 }, { 7, 6, 5 }, { 8, 7, 6 }, { 9, 8, 7 }, { 10, 9, 8 } } });
            print("*************");



            var x2_new = (ndarray)x2[":", np.newaxis];
            z = x1 + x2_new;
            print(z);
            AssertArray(z, new Int32[,] { { 6, 7, 8, 9, 10 }, { 5, 6, 7, 8, 9 }, { 4, 5, 6, 7, 8 } });
            print("*************");

            x2_new = (ndarray)x2[":", np.newaxis, np.newaxis];
            z = x1 + x2_new;
            print(z);
            AssertArray(z, new Int32[,,] { { { 6, 7, 8, 9, 10 } }, { { 5, 6, 7, 8, 9 } }, { { 4, 5, 6, 7, 8 } } });
            print("*************");


        }


        [TestMethod]
        public void test_newaxis_ufunc_4()
        {
            var x1 = np.array(new Int32[] { 1, 2, 3, 4, 5 });
            var x2 = np.array(new Int32[] { 5, 4, 3 });

            var x2_new = (ndarray)x2[":", np.newaxis];
            var z = x2_new + x1;
            print(z);
            AssertArray(z, new Int32[,] { { 6, 7, 8, 9, 10 }, { 5, 6, 7, 8, 9 }, { 4, 5, 6, 7, 8 } });
            print("*************");

            x2_new = (ndarray)x2[np.newaxis, ":", np.newaxis];
            z = x2_new + x1;
            print(z);
            AssertArray(z, new Int32[,,] { { { 6, 7, 8, 9, 10 }, { 5, 6, 7, 8, 9 }, { 4, 5, 6, 7, 8 } } });
            print("*************");



            x2_new = (ndarray)x2[":", np.newaxis];
            z = x2_new + x1;
            print(z);
            AssertArray(z, new Int32[,] { { 6, 7, 8, 9, 10 }, { 5, 6, 7, 8, 9 }, { 4, 5, 6, 7, 8 } });
            print("*************");

            x2_new = (ndarray)x2[":", np.newaxis, np.newaxis];
            z = x2_new + x1;
            print(z);
            AssertArray(z, new Int32[,,] {{{ 6,  7,  8,  9, 10}}, {{ 5,  6,  7,  8,  9}}, {{ 4,  5,  6,  7,  8}}});
            print("*************");


        }

        [TestMethod]
        public void test_ndarray_copy_parameter()
        {
            var BoolTestData = new bool[] { false, false, false, false };

            var a = np.array(BoolTestData);
            a[2] = true;
            Assert.AreEqual(BoolTestData[2], false);

            a = np.array(BoolTestData, copy: false);
            a[2] = true;
            Assert.AreEqual(BoolTestData[2], true);

            ////////////////////////

            var ByteTestData = new byte[] { 0, 1, 2, 3 };

            a = np.array(ByteTestData);
            a[2] = 10;
            Assert.AreEqual(ByteTestData[2], 2);

            a = np.array(ByteTestData, copy: false);
            a[2] = 10;
            Assert.AreEqual(ByteTestData[2], 10);

            ////////////////////////

            var SByteTestData = new sbyte[] { 0, 1, 2, 3 };

            a = np.array(SByteTestData);
            a[2] = 20;
            Assert.AreEqual(SByteTestData[2], 2);

            a = np.array(SByteTestData, copy: false);
            a[2] = 20;
            Assert.AreEqual(SByteTestData[2], 20);

            ////////////////////////

            var Int16TestData = new Int16[] { 0, 1, 2, 3 };

            a = np.array(Int16TestData);
            a[2] = 30;
            Assert.AreEqual(Int16TestData[2], 2);

            a = np.array(Int16TestData, copy: false);
            a[2] = 30;
            Assert.AreEqual(Int16TestData[2], 30);

            ////////////////////////

            var UInt16TestData = new UInt16[] { 0, 1, 2, 3 };

            a = np.array(UInt16TestData);
            a[2] = 40;
            Assert.AreEqual(UInt16TestData[2], 2);

            a = np.array(UInt16TestData, copy: false);
            a[2] = 40;
            Assert.AreEqual(UInt16TestData[2], 40);

            ////////////////////////


            var Int32TestData = new Int32[] { 0, 1, 2, 3 };

            a = np.array(Int32TestData);
            a[2] = 50;
            Assert.AreEqual(Int32TestData[2], 2);

            a = np.array(Int32TestData, copy: false);
            a[2] = 50;
            Assert.AreEqual(Int32TestData[2], 50);

            ////////////////////////

            var UInt32TestData = new UInt32[] { 0, 1, 2, 3 };

            a = np.array(UInt32TestData);
            a[2] = 60;
            Assert.AreEqual(UInt32TestData[2], (UInt32)2);

            a = np.array(UInt32TestData, copy: false);
            a[2] = 60;
            Assert.AreEqual(UInt32TestData[2], (UInt32)60);

            ////////////////////////

            var Int64TestData = new Int64[] { 0, 1, 2, 3 };

            a = np.array(Int64TestData);
            a[2] = 70;
            Assert.AreEqual(Int64TestData[2], 2);

            a = np.array(Int64TestData, copy: false);
            a[2] = 70;
            Assert.AreEqual(Int64TestData[2], 70);

            ////////////////////////

            var UInt64TestData = new UInt64[] { 0, 1, 2, 3 };

            a = np.array(UInt64TestData);
            a[2] = 80;
            Assert.AreEqual(UInt64TestData[2], (UInt64)2);

            a = np.array(UInt64TestData, copy: false);
            a[2] = 80;
            Assert.AreEqual(UInt64TestData[2], (UInt64)80);

            ////////////////////////

            var FloatTestData = new float[] { 0, 1, 2, 3 };

            a = np.array(FloatTestData);
            a[2] = 90;
            Assert.AreEqual(FloatTestData[2], (float)2);

            a = np.array(FloatTestData, copy: false);
            a[2] = 90;
            Assert.AreEqual(FloatTestData[2], (float)90);

            ////////////////////////

            var DoubleTestData = new double[] { 0, 1, 2, 3 };

            a = np.array(DoubleTestData);
            a[2] = 100;
            Assert.AreEqual(DoubleTestData[2], (double)2);

            a = np.array(DoubleTestData, copy: false);
            a[2] = 100;
            Assert.AreEqual(DoubleTestData[2], (double)100);

            ////////////////////////

            var DecimalTestData = new decimal[] { 0, 1, 2, 3 };

            a = np.array(DecimalTestData);
            a[2] = 100;
            Assert.AreEqual(DecimalTestData[2], (decimal)2);

            a = np.array(DecimalTestData, copy: false);
            a[2] = 100;
            Assert.AreEqual(DecimalTestData[2], (decimal)100);

            ////////////////////////

            var ComplexTestData = new System.Numerics.Complex[] { 0, 1, 2, 3 };

            a = np.array(ComplexTestData);
            a[2] = 110;
            Assert.AreEqual(ComplexTestData[2], (System.Numerics.Complex)2);

            a = np.array(ComplexTestData, copy: false);
            a[2] = 110;
            Assert.AreEqual(ComplexTestData[2], (System.Numerics.Complex)110);

            ////////////////////////

            var BigIntTestData = new System.Numerics.BigInteger[] { 0, 1, 2, 3 };

            a = np.array(BigIntTestData);
            a[2] = 120;
            Assert.AreEqual(BigIntTestData[2], (System.Numerics.BigInteger)2);

            a = np.array(BigIntTestData, copy: false);
            a[2] = 120;
            Assert.AreEqual(BigIntTestData[2], (System.Numerics.BigInteger)120);

            ////////////////////////

            var ObjectTestData = new System.Object[] { 0, 1, 2, 3 };

            a = np.array(ObjectTestData);
            a[2] = 130;
            Assert.AreEqual(ObjectTestData[2], 2);

            a = np.array(ObjectTestData, copy: false);
            a[2] = 130;
            Assert.AreEqual(ObjectTestData[2], 130);

            ////////////////////////

            var StringTestData = new String[] { "0", "1", "2", "3" };

            a = np.array(StringTestData);
            a[2] = "A";
            Assert.AreEqual(StringTestData[2], "2");

            a = np.array(StringTestData, copy: false);
            a[2] = "A";
            Assert.AreEqual(StringTestData[2], "A");

        }


        //[TestMethod]
        //public void test_fields_1()
        //{
        //    var data = np.zeros(4);
        //    //var kk = data.

        //    List<string> TestData = new List<string>();
        //    TestData.Add("Test1");
        //    TestData.Add("Test2");

        //    numpyAPI.NpyArray_DescrSetNames(data.Dtype.Descr, TestData);
        //    var tt = data.Dtype.Names;


        //    var data2 = data.reshape((2, 2));
        //    var t2 = data2.Dtype.Names;

        //    return;

        //}

        [TestMethod]
        public void test_explicit_cast_1()
        {
            var boolData = np.array(new bool[] { true });
            Assert.AreEqual(true, (bool)boolData);

            var byteData = np.array(new byte[] { 24 });
            Assert.AreEqual((byte)24, (byte)byteData);

            var sbyteData = np.array(new sbyte[] { -24 });
            Assert.AreEqual((sbyte)-24, (sbyte)sbyteData);

            var uint16Data = np.array(new UInt16[] { 689 });
            Assert.AreEqual((UInt16)689, (UInt16)uint16Data);

            var int16Data = np.array(new Int16[] { -689 });
            Assert.AreEqual((Int16)(-689), (Int16)int16Data);

            var uint32Data = np.array(new UInt32[] { 6899 });
            Assert.AreEqual((UInt32)6899, (UInt32)uint32Data);

            var int32Data = np.array(new Int32[] { -6899 });
            Assert.AreEqual((Int32)(-6899), (Int32)int32Data);

            var uint64Data = np.array(new UInt64[] { 68999 });
            Assert.AreEqual((UInt64)68999, (UInt64)uint64Data);

            var int64Data = np.array(new Int64[] { -68999 });
            Assert.AreEqual((Int64)(-68999), (Int64)int64Data);

            var floatData = np.array(new float[] { 68999 });
            Assert.AreEqual((float)68999, (float)floatData);

            var doubleData = np.array(new double[] { -68999 });
            Assert.AreEqual((double)(-68999), (double)doubleData);

            var decimalData = np.array(new decimal[] { -68999 });
            Assert.AreEqual((decimal)(-68999), (decimal)decimalData);

            var complexData = np.array(new System.Numerics.Complex[] { -68999 });
            Assert.AreEqual((System.Numerics.Complex)(-68999), (System.Numerics.Complex)complexData);

            var bigintData = np.array(new System.Numerics.BigInteger[] { -68999 });
            Assert.AreEqual((System.Numerics.BigInteger)(-68999), (System.Numerics.BigInteger)bigintData);

            var stringData = np.array(new string[] {"Numpy.NET" });
            Assert.AreEqual("Numpy.NET", (string)stringData);


            return;

        }
        [TestMethod]
        public void test_othertypes_char()
        {
            // copy: false
            Assert.AreEqual('2', (char)np.max(np.array(new[] { '1', '2' })).Flat[0]);
            Assert.AreEqual('2', (char)np.array(new[] { '1', '2', '3' }, np.Object, false).Flat[1]);
            Assert.AreEqual('3', (char)np.array(new[,] { { '1', '2' }, { '3', '4' } }, np.Object, false).Flat[2]);
            // copy: true
            Assert.AreEqual('2', (char)np.max(np.array(new[] { '1', '2' }, copy:true)).Flat[0]);
            Assert.AreEqual('2', (char)np.array(new[] { '1', '2', '3' }, np.Object, true).Flat[1]);
            Assert.AreEqual('3', (char)np.array(new[,] { { '1', '2' }, { '3', '4' } }, np.Object, true).Flat[2]);
        }
        public class Custom { public string s; public static Custom operator *(Custom a, Custom b) => new Custom { s = a.s + "*" + b.s }; }
        [TestMethod]
        public void test_othertypes_custom()
        {
            // 1d
            Assert.AreEqual("a*b", ((Custom)np.prod(np.array(new[] { new Custom { s = "a" }, new Custom { s = "b" } })).Flat[0]).s);
            Assert.AreEqual("a*b", ((Custom)np.prod(np.array(new[] { new Custom { s = "a" }, new Custom { s = "b" } }, np.Object)).Flat[0]).s);
            Assert.AreEqual("a*b", ((Custom)np.prod(np.array(new[] { new Custom { s = "a" }, new Custom { s = "b" } }, np.Object, false)).Flat[0]).s);
            Assert.AreEqual("a*b", ((Custom)np.prod(np.array(new[] { new Custom { s = "a" }, new Custom { s = "b" } }, np.Object, true)).Flat[0]).s);
            // 2d
            Assert.AreEqual("a*b", ((Custom)np.prod(np.array(new[,] { { new Custom { s = "a" } }, { new Custom { s = "b" } } })).Flat[0]).s);
            Assert.AreEqual("a*b", ((Custom)np.prod(np.array(new[,] { { new Custom { s = "a" } }, { new Custom { s = "b" } } }, np.Object)).Flat[0]).s);
            Assert.AreEqual("a*b", ((Custom)np.prod(np.array(new[,] { { new Custom { s = "a" } }, { new Custom { s = "b" } } }, np.Object, false)).Flat[0]).s);
            Assert.AreEqual("a*b", ((Custom)np.prod(np.array(new[,] { { new Custom { s = "a" } }, { new Custom { s = "b" } } }, np.Object, true)).Flat[0]).s);
        }

        [TestMethod]
        public void test_allnull()
        {
            var a = np.array(new string[,] { { null, null }, { null, null } });
            Assert.AreEqual(np.Strings, a.Dtype);
            Assert.AreEqual(null, (string)a[0, 0]);
            a[0, 0] = "a";
            Assert.AreEqual("a", (string)a[0, 0]);
            var b = np.array(new object[,] { { null, null }, { null, null } });
            Assert.AreEqual(np.Object, b.Dtype);
            Assert.AreEqual(null, b[0, 0]);
            b[0, 0] = "b";
            Assert.AreEqual("b", (string)b[0, 0]);
            var c = np.array(new System.Text.StringBuilder[,] { { null, null }, { null, null } });
            Assert.AreEqual(np.Object, c.Dtype);
            Assert.AreEqual(null, c[0, 0]);
            c[0, 0] = new System.Text.StringBuilder();
            Assert.IsNotNull(c[0, 0]);
        }
    }

}
