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
    public class FromNumericTests : TestBaseClass
    {
        [TestMethod]
        public void test_take_1()
        {
            var a = np.array(new Int32[] { 4, 3, 5, 7, 6, 8, 9, 12, 14, 16, 18, 20, 22, 24, 26, 28 });
            var indices = np.array(new Int32[] { 0, 1, 4 });
            ndarray b = np.take(a, indices);
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new Int32[] { 4, 3, 6 });
            AssertShape(b, 3);
            AssertStrides(b, sizeof(Int32));


            a = np.array(new Int32[] { 4, 3, 5, 7, 6, 8, 9, 12, 14, 16, 18, 20, 22, 24, 26, 28 });
            indices = np.array(new Int32[,] { { 0, 1 }, { 2, 3 } });
            ndarray c = np.take(a, indices);
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            var ExpectedDataC = new Int32[2, 2]
            {
                { 4, 3 },
                { 5, 7 },
            };
            AssertArray(c, ExpectedDataC);
            AssertShape(c, 2, 2);
            AssertStrides(c, sizeof(Int32) * 2, sizeof(Int32));

            ndarray d = np.take(a.reshape(new shape(4, -1)), indices, axis: 0);
            print("D");
            print(d);
            print(d.shape);
            print(d.strides);

            var ExpectedDataD = new Int32[2, 2, 4]
            {
                {
                    { 4, 3, 5, 7 },
                    { 6, 8, 9, 12 },
                },
                {
                    { 14, 16, 18, 20 },
                    { 22, 24, 26, 28 },
                },
      
            };
            AssertArray(d, ExpectedDataD);
            AssertShape(d, 2, 2, 4);
            AssertStrides(d, 32,16,4);

            ndarray e = np.take(a.reshape(new shape(4, -1)), indices, axis: 1);
            print("E");
            print(e);
            print(e.shape);
            print(e.strides);

            var ExpectedDataE = new Int32[4, 2, 2]
            {
                {
                    { 4, 3 },
                    { 5, 7 },
                },
                {
                    { 6, 8 },
                    { 9, 12 },
                },
                {
                    { 14, 16 },
                    { 18, 20 },
                },
                {
                    { 22, 24 },
                    { 26, 28 },
                },

            };

            AssertArray(e, ExpectedDataE);
            AssertShape(e, 4, 2, 2);
            AssertStrides(e, 16, 8, 4);

        }

        [Ignore]
        [TestMethod]
        public void test_take_along_axis_1()
        {

        }

        [TestMethod]
        public void test_reshape_1()
        {
            ndarray a = np.arange(6).reshape(new shape(3, 2));
            print(a);
            print("");

            ndarray b = np.reshape(a, new shape(2, 3));    // C-like index ordering
            print(b);

            var ExpectedDataB = new Int32[2, 3]
            {
                { 0, 1, 2 },
                { 3, 4, 5 },
            };
            AssertArray(b, ExpectedDataB);


            print("");
            ndarray c = np.reshape(np.ravel(a), new shape(2, 3)); // equivalent to C ravel then C reshape
            print(c);

            var ExpectedDataC = new Int32[2, 3]
            {
                { 0, 1, 2 },
                { 3, 4, 5 },
            };
            AssertArray(c, ExpectedDataC);

            print("");
            ndarray d = np.reshape(a, new shape(2, 3), NPY_ORDER.NPY_FORTRANORDER); // Fortran-like index ordering
            print(d);

            var ExpectedDataD = new Int32[2, 3]
            {
                { 0, 4, 3 },
                { 2, 1, 5 },
            };
            AssertArray(d, ExpectedDataD);

            print("");
            ndarray e = np.reshape(np.ravel(a, NPY_ORDER.NPY_FORTRANORDER), new shape(2, 3), NPY_ORDER.NPY_FORTRANORDER);
            print(e);

            var ExpectedDataE = new Int32[2, 3]
            {
                { 0, 4, 3 },
                { 2, 1, 5 },
            };
            AssertArray(e, ExpectedDataE);

        }

        [TestMethod]
        public void test_ravel_1()
        {
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var b = np.ravel(a);
            AssertArray(b, new int[] { 1, 2, 3, 4, 5, 6 });
            print(b);

            var c = a.reshape(-1);
            AssertArray(c, new int[] { 1, 2, 3, 4, 5, 6 });
            print(c);

            var d = np.ravel(a, order: NPY_ORDER.NPY_FORTRANORDER);
            AssertArray(d, new int[] { 1, 4, 2, 5, 3, 6 });
            print(d);

            // When order is 'A', it will preserve the array's 'C' or 'F' ordering:
            var e = np.ravel(a.T);
            AssertArray(e, new int[] { 1, 4, 2, 5, 3, 6 });
            print(e);

            var f = np.ravel(a.T, order: NPY_ORDER.NPY_ANYORDER);
            AssertArray(f, new int[] { 1, 2, 3, 4, 5, 6 });
            print(f);
        }

        [TestMethod]
        public void test_ravel_2()
        {
            // When order is 'K', it will preserve orderings that are neither 'C' nor 'F', but won't reverse axes:

            var a = np.arange(3)["::-1"] as ndarray;
            AssertArray(a, new int[] { 2, 1, 0 });
            print(a);

            var b = a.ravel(order: NPY_ORDER.NPY_CORDER);
            AssertArray(b, new int[] { 2, 1, 0 });
            print(b);

            var c = a.ravel(order : NPY_ORDER.NPY_ANYORDER);
            AssertArray(c, new int[] { 2, 1, 0 });
            print(c);
        }

        [TestMethod]
        public void test_ravel_3()
        {
            var a = np.arange(12).reshape((2, 3, 2)).SwapAxes(1, 2);
            AssertArray(a, new int[,,] { { { 0, 2, 4 }, { 1, 3, 5 } }, { { 6, 8, 10 }, { 7, 9, 11 } } } );

            print(a);

            var b = a.ravel(order: NPY_ORDER.NPY_CORDER);
            AssertArray(b, new int[] { 0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11 });
            print(b);

            var c = a.ravel(order: "K");
            // AssertArray(c, new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }); // todo: order K does not produce expected result
            print(c);
        }


        [TestMethod]
        public void test_choose_1()
        {
            ndarray choice1 = np.array(new Int32[] { 0, 1, 2, 3 });
            ndarray choice2 = np.array(new Int32[] { 10, 11, 12, 13 });
            ndarray choice3 = np.array(new Int32[] { 20, 21, 22, 23 });
            ndarray choice4 = np.array(new Int32[] { 30, 31, 32, 33 });

            ndarray[] choices = new ndarray[] { choice1, choice2, choice3, choice4 };

            ndarray a = np.choose(np.array(new Int32[] { 2, 3, 1, 0 }), choices);

            print(a);

            AssertArray(a, new Int32[] { 20, 31, 12, 3 });
        }


        [TestMethod]
        public void test_choose_2()
        {
            ndarray choice1 = np.array(new Int32[] { 0, 1, 2, 3 });
            ndarray choice2 = np.array(new Int32[] { 10, 11, 12, 13 });
            ndarray choice3 = np.array(new Int32[] { 20, 21, 22, 23 });
            ndarray choice4 = np.array(new Int32[] { 30, 31, 32, 33 });

            ndarray[] choices = new ndarray[] { choice1, choice2, choice3, choice4 };

            ndarray a = np.choose(np.array(new Int32[] { 2, 4, 1, 0 }), choices, mode: NPY_CLIPMODE.NPY_CLIP);
            print(a);
            AssertArray(a, new Int32[] { 20, 31, 12, 3 });

            a = np.choose(np.array(new Int32[] { 2, 4, 1, 0 }), choices, mode: NPY_CLIPMODE.NPY_WRAP);
            print(a);
            AssertArray(a, new Int32[] { 20, 1, 12, 3 });

            try
            {
                a = np.choose(np.array(new Int32[] { 2, 4, 1, 0 }), choices, mode: NPY_CLIPMODE.NPY_RAISE);
                print(a);
                AssertArray(a, new Int32[] { 20, 1, 12, 3 });
            }
            catch (Exception ex)
            {
                if (ex.Message.Contains("invalid entry in choice array"))
                    return;
            }
            Assert.Fail("Should have caught exception from np.choose");


        }

        [TestMethod]
        public void test_choose_3()
        {
            ndarray a = np.array(new Int32[,] { { 1, 0, 1 }, { 0, 1, 0 }, { 1, 0, 1 } });
            ndarray choice1 = np.array(new Int32[] { -10 });
            ndarray choice2 = np.array(new Int32[] { 10 });
            ndarray[] choices = new ndarray[] { choice1, choice2 };

            ndarray b = np.choose(a, choices );
            print(b);

            var ExpectedDataB = new Int32[,]
                {{10, -10, 10},
                 {-10, 10,-10},
                 {10, -10, 10}};
            AssertArray(b, ExpectedDataB);
        }

        [TestMethod]
        public void test_choose_4()
        {
            ndarray a = np.array(new Int32[] { 0,1 }).reshape(new shape(2,1,1));
            ndarray c1 = np.array(new Int32[] { 1,2,3 }).reshape(new shape(1,3,1));
            ndarray c2 = np.array(new Int32[] { -1,-2,-3,-4,-5 }).reshape(new shape(1,1,5));
            ndarray[] choices = new ndarray[] { c1, c2 };

            ndarray b = np.choose(a, choices);
            print(b);

            var ExpectedDataB = new Int32[,,]
                {{{1,  1,  1,  1,  1},
                  {2,  2,  2,  2,  2},
                  {3,  3,  3,  3,  3}},

                 {{-1, -2, -3, -4, -5},
                  {-1, -2, -3, -4, -5},
                  {-1, -2, -3, -4, -5}}};
     
            AssertArray(b, ExpectedDataB);
        }

        [TestMethod]
        public void test_select_1()
        {
            var x = np.arange(10);
            var condlist = new ndarray[] { x < 3, x > 5 };
            var choicelist = new ndarray[] { x,  np.array(np.power(x, 2), dtype: np.Int32) };
            var y = np.select(condlist, choicelist);

            AssertArray(y, new int[] { 0,  1,  2,  0,  0,  0, 36, 49, 64, 81 });
            print(y);
        }


        [TestMethod]
        public void test_repeat_1()
        {
            ndarray x = np.array(new Int32[] { 1, 2, 3, 4 }).reshape(new shape(2, 2));
            var y = new Int32[] { 2 };

            ndarray z = np.repeat(x, y);
            print(z);
            print("");
            AssertArray(z, new Int32[] { 1, 1, 2, 2, 3, 3, 4, 4 });

            z = np.repeat(3, 4);
            print(z);
            print("");
            AssertArray(z, new Int32[] { 3, 3, 3, 3 });

            z = np.repeat(x, 3, axis: 0);
            print(z);
            print("");

            var ExpectedData1 = new Int32[6, 2]
            {
                { 1, 2 },
                { 1, 2 },
                { 1, 2 },
                { 3, 4 },
                { 3, 4 },
                { 3, 4 },
            };

            AssertArray(z, ExpectedData1);
            AssertShape(z, 6, 2);

            z = np.repeat(x, 3, axis: 1);
            print(z);
            print("");

            var ExpectedData2 = new Int32[2, 6]
            {
                { 1, 1, 1, 2, 2, 2 },
                { 3, 3, 3, 4, 4, 4 },
            };

            AssertArray(z, ExpectedData2);
            AssertShape(z, 2, 6);


            np.array(new int[] { 1, 2 });

            z = np.repeat(x, new int[] { 1, 2 }, axis: 0);
            print(z);

            var ExpectedData3 = new Int32[3, 2]
            {
                { 1, 2 },
                { 3, 4 },
                { 3, 4 },
            };

            AssertArray(z, ExpectedData3);
            AssertShape(z, 3, 2);
        }

        [TestMethod]
        public void test_put_1()
        {
            ndarray a = np.arange(5);
            np.put(a, new int[] { 0, 2 }, new int[] { -44, -55 });
            print(a);
            AssertArray(a, new int[] {-44, 1,-55, 3, 4 });

            a = np.arange(5);
            np.put(a, 22, -5, mode: NPY_CLIPMODE.NPY_CLIP);
            print(a);
            AssertArray(a, new int[] { 0,  1,  2,  3, -5 });

            a = np.arange(5);
            np.put(a, 22, -5, mode: NPY_CLIPMODE.NPY_WRAP);
            print(a);
            AssertArray(a, new int[] { 0,  1, -5,  3,  4 });

            try
            {
                a = np.arange(5);
                np.put(a, 22, -5, mode: NPY_CLIPMODE.NPY_RAISE);
                print(a);
            }
            catch (Exception ex)
            {
                return;
            }
            throw new Exception("this should have caught an exception");

        }

        [TestMethod]
        public void test_put_2()
        {
            ndarray a = np.arange(15);
            np.put(a.A(":5"), new int[] { 0, 2 }, new int[] { -44, -55 });
            print(a);
            AssertArray(a, new int[] { -44, 1, -55, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 });

            a = np.arange(15);
            np.put(a.A(":5"), 22, -5, mode: NPY_CLIPMODE.NPY_CLIP);
            print(a);
            AssertArray(a, new int[] { 0, 1, 2, 3, -5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 });

            a = np.arange(15);
            np.put(a.A(":5"), 22, -5, mode: NPY_CLIPMODE.NPY_WRAP);
            print(a);
            AssertArray(a, new int[] { 0, 1, -5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 });

            try
            {
                a = np.arange(15);
                np.put(a.A(":5"), 22, -5, mode: NPY_CLIPMODE.NPY_RAISE);
                print(a);
            }
            catch (Exception ex)
            {
                return;
            }
            throw new Exception("this should have caught an exception");

        }

        [Ignore]
        [TestMethod]
        public void test_put_along_axis_1()
        {

        }

        [Ignore]
        [TestMethod]
        public void test_put_mask_1()
        {

        }

        [TestMethod]
        public void test_swapaxes_1()
        {
            ndarray x = np.array(new Int32[,] { { 1, 2, 3 } });
            print(x);
            print("********");

            ndarray y = np.swapaxes(x, 0, 1);
            print(y);
            AssertArray(y, new Int32[3, 1] { { 1 }, { 2 }, { 3 } });
            print("********");

            x = np.array(new Int32[,,]{{{0, 1},{2, 3}},{{4,5},{6,7}}});
            print(x);

            var ExpectedDataX = new Int32[2, 2, 2]
            {
                {
                    { 0,1 },
                    { 2,3 },
                },
                {
                    { 4,5 },
                    { 6,7 },
                },
            };
            AssertArray(x, ExpectedDataX);

            print("********");

            y = np.swapaxes(x, 0, 2);
            print(y);

            var ExpectedDataY = new Int32[2, 2, 2]
            {
                {
                    { 0,4 },
                    { 2,6 },
                },
                {
                    { 1,5 },
                    { 3,7 },
                },
            };
            AssertArray(y, ExpectedDataY);
        }


        [TestMethod]
        public void test_ndarray_T_1()
        {
            var x = np.arange(0, 32, dtype: np.Int16).reshape(new shape(8, 4));
            print("X");
            print(x);
            print(x.shape);

            var y = x.T;

            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Int16[4, 8]
            {
                { 0, 4,  8, 12, 16, 20, 24, 28 },
                { 1, 5,  9, 13, 17, 21, 25, 29 },
                { 2, 6, 10, 14, 18, 22, 26, 30 },
                { 3, 7, 11, 15, 19, 23, 27, 31 },
            };

            AssertArray(y, ExpectedDataY);

        }

        [TestMethod]
        public void test_ndarray_T_2()
        {
            var x = np.arange(0, 32, dtype: np.Int16);
            print("X");
            print(x);
            print(x.shape);

            var y = x.T;

            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Int16[32]
                { 0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                 16, 17, 18,19, 20, 21, 22,23,24,25,26,27,28,29, 30, 31};

            AssertArray(y, ExpectedDataY);
        }

        [TestMethod]
        public void test_ndarray_T_3()
        {
            var x = np.arange(0, 32, dtype: np.Int16).reshape(new shape(2, -1, 4));
            print("X");
            print(x);
            print(x.shape);

            var y = x.T;

            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Int16[4, 4, 2]
            {
                {
                    { 0, 16 },
                    { 4, 20 },
                    { 8, 24 },
                    { 12, 28 },
                },
                {
                    { 1, 17 },
                    { 5, 21 },
                    { 9, 25 },
                    { 13, 29 },
                },
                {
                    { 2, 18 },
                    { 6, 22 },
                    { 10, 26 },
                    { 14, 30 },
                },
                {
                    { 3, 19 },
                    { 7, 23 },
                    { 11, 27 },
                    { 15, 31 },
                },
            };

            AssertArray(y, ExpectedDataY);
        }

        [TestMethod]
        public void test_ndarray_T_4()
        {
            var x = np.arange(0, 64, dtype: np.Int16).reshape(new shape(2, 4, -1, 4));
            print("X");
            print(x);
            print(x.shape);

            var y = x.T;

            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Int16[4, 2, 4, 2]
                {{{ {0, 32},
                    {8, 40},
                    {16, 48},
                    {24, 56}},
                   {{ 4, 36},
                    {12, 44},
                    {20, 52},
                    {28, 60}}},
                  {{{ 1, 33},
                    { 9, 41},
                    {17, 49},
                    {25, 57}},
                   {{ 5, 37},
                    {13, 45},
                    {21, 53},
                    {29, 61}}},
                  {{{ 2, 34},
                    {10, 42},
                    {18, 50},
                    {26, 58}},
                   {{ 6, 38},
                    {14, 46},
                    {22, 54},
                    {30, 62}}},
                  {{{ 3, 35},
                    {11, 43},
                    {19, 51},
                    {27, 59}},
                   {{ 7, 39},
                    {15, 47},
                    {23, 55},
                    {31, 63}}}};

            AssertArray(y, ExpectedDataY);

        }

        [TestMethod]
        public void test_ndarray_transpose_1()
        {
            var x = np.arange(0, 64, dtype: np.Int16).reshape(new shape(2, 4, -1, 4));
            print("X");
            print(x);
            print(x.shape);

            var y = np.transpose(x, new long[] { 1, 2, 3, 0 });

            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Int16[4, 2, 4, 2]
                {{{ {0, 32},
                    {1, 33},
                    {2, 34},
                    {3, 35}},
                   {{4, 36},
                    {5, 37},
                    {6, 38},
                    {7, 39}}},
                  {{{8, 40},
                    {9, 41},
                    {10, 42},
                    {11, 43}},
                   {{12, 44},
                    {13, 45},
                    {14, 46},
                    {15, 47}}},
                  {{{16, 48},
                    {17, 49},
                    {18, 50},
                    {19, 51}},
                   {{20, 52},
                    {21, 53},
                    {22, 54},
                    {23, 55}}},
                  {{{24, 56},
                    {25, 57},
                    {26, 58},
                    {27, 59}},
                   {{28, 60},
                    {29, 61},
                    {30, 62},
                    {31, 63}}}};

            AssertArray(y, ExpectedDataY);

        }

        [TestMethod]
        public void test_ndarray_transpose_2()
        {
            var x = np.arange(0, 64, dtype: np.Int16).reshape(new shape(2, 4, -1, 4));
            print("X");
            print(x);
            print(x.shape);

            var y = np.transpose(x, new long[] { 3, 2, 1, 0 });

            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Int16[4, 2, 4, 2]
                {{{ {0, 32},
                    {8, 40},
                    {16, 48},
                    {24, 56}},
                   {{ 4, 36},
                    {12, 44},
                    {20, 52},
                    {28, 60}}},
                  {{{ 1, 33},
                    { 9, 41},
                    {17, 49},
                    {25, 57}},
                   {{ 5, 37},
                    {13, 45},
                    {21, 53},
                    {29, 61}}},
                  {{{ 2, 34},
                    {10, 42},
                    {18, 50},
                    {26, 58}},
                   {{ 6, 38},
                    {14, 46},
                    {22, 54},
                    {30, 62}}},
                  {{{ 3, 35},
                    {11, 43},
                    {19, 51},
                    {27, 59}},
                   {{ 7, 39},
                    {15, 47},
                    {23, 55},
                    {31, 63}}}};

            AssertArray(y, ExpectedDataY);
        }

        [Ignore] // not implemented yet
        [TestMethod]
        public void xxx_test_partition_1()
        {
            var a = np.array(new int[] { 3, 4, 2, 1 });
            ndarray b = np.partition(a, 3);
            print(b);

            print("********");
            ndarray c = np.partition(a, new Int32[] { 1, 3 });
            print(c);
        }

        [Ignore] // not implemented yet
        [TestMethod]
        public void xxx_test_argpartition_1()
        {
            var a = np.array(new int[] { 3, 4, 2, 1 });
            ndarray b = np.argpartition(a, 3);
            print(b);

            print("********");
            ndarray c = np.argpartition(a, new Int32[] { 1, 3 });
            print(c);
        }

        [TestMethod]
        public void test_sort_1()
        {
            var a = np.array(new int[,] { { 1, 4 }, { 3, 1 } });
            ndarray b = np.sort(a);                 // sort along the last axis
            print(b);
            AssertArray(b, new int[,] { { 1, 4 }, { 1, 3 } });

            ndarray c = np.sort(a, axis: null);     // sort the flattened array
            print(c);
            print("********");
            AssertArray(c, new int[] { 1, 1, 3, 4 });

            ndarray d = np.sort(a, axis: 0);        // sort along the first axis
            print(d);
            AssertArray(d, new int[,] { { 1, 1 }, { 3, 4 } });
            print("********");

        }

        [TestMethod]
        public void test_sort_2()
        {
            var InputData = new double[]
                {32.2, 31.2, 30.2, 29.2, 28.2, 27.2, 26.2, 25.2, 24.2, 23.2, 22.2, 21.2, 20.2, 19.2, 18.2, 17.2,
                 16.2, 15.2, 14.2, 13.2, 12.2, 11.2, 10.2, 9.2,  8.2,  7.2,  6.2,  5.2,  4.2,  3.2,  2.2,  1.2};

            var a = np.array(InputData).reshape(new shape(8, 4));
            ndarray b = np.sort(a);                 // sort along the last axis
            print(b);

            var ExpectedDataB = new double[8, 4]
            {
             {29.2, 30.2, 31.2, 32.2},
             {25.2, 26.2, 27.2, 28.2},
             {21.2, 22.2, 23.2, 24.2},
             {17.2, 18.2, 19.2, 20.2},
             {13.2, 14.2, 15.2, 16.2},
             {9.2, 10.2, 11.2, 12.2},
             {5.2,  6.2,  7.2,  8.2},
             {1.2,  2.2,  3.2,  4.2},
            };

            AssertArray(b, ExpectedDataB);

            ndarray c = np.sort(a, axis: null);     // sort the flattened array
            print(c);
            print("********");

            var ExpectedDataC = new double[]
            {1.2,  2.2,  3.2,  4.2,  5.2,  6.2,  7.2,  8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 16.2,
            17.2, 18.2, 19.2, 20.2, 21.2, 22.2, 23.2, 24.2, 25.2, 26.2, 27.2, 28.2, 29.2, 30.2, 31.2, 32.2};

            AssertArray(c, ExpectedDataC);

            ndarray d = np.sort(a, axis: 0);        // sort along the first axis
            print(d);

            var ExpectedDataD = new double[8, 4]
            {
                {4.2,  3.2,  2.2,  1.2},
                {8.2,  7.2,  6.2,  5.2},
                {12.2, 11.2, 10.2,  9.2},
                {16.2, 15.2, 14.2, 13.2},
                {20.2, 19.2, 18.2, 17.2},
                {24.2, 23.2, 22.2, 21.2},
                {28.2, 27.2, 26.2, 25.2},
                {32.2, 31.2, 30.2, 29.2},
            };

            AssertArray(d, ExpectedDataD);
            print("********");

        }

        [TestMethod]
        public void test_ndarray_argsort_1()
        {
            var x = np.array(new Int32[] { 1, 2, 3, 1, 3, 4, 5, 4, 4 });
            var ar = np.array(new Int32[] { 3, 2, 1 });

            ndarray perm1 = np.argsort(x, kind: NPY_SORTKIND.NPY_MERGESORT);
            ndarray perm2 = np.argsort(ar, kind: NPY_SORTKIND.NPY_QUICKSORT);
            ndarray perm3 = np.argsort(ar);

            print(perm1);
            AssertArray(perm1, new Int64[] { 0, 3, 1, 2, 4, 5, 7, 8, 6 });

            print(perm2);
            AssertArray(perm2, new Int64[] { 2,1,0 });

            print(perm3);
            AssertArray(perm3, new Int64[] { 2, 1, 0 });

        }

        [TestMethod]
        public void test_ndarray_argsort_2()
        {
            var ar = np.array(new Int32[] { 1, 2, 3, 1, 3, 4, 5, 4, 4, 1, 9, 6, 9, 11, 23, 9, 5, 0, 11, 12 }).reshape(new shape(5, 4));

            ndarray perm1 = np.argsort(ar, kind: NPY_SORTKIND.NPY_MERGESORT);
            ndarray perm2 = np.argsort(ar, kind: NPY_SORTKIND.NPY_QUICKSORT);
            ndarray perm3 = np.argsort(ar);

            print(perm1);

            var Perm1Expected = new Int64[,]
            {{0, 3, 1, 2},
             {0, 1, 3, 2},
             {1, 0, 3, 2},
             {0, 3, 1, 2},
             {1, 0, 2, 3}};
            AssertArray(perm1, Perm1Expected);

            print(perm2);
            var Perm2Expected = new Int64[,]
            {{0, 3, 1, 2},
             {0, 1, 3, 2},
             {1, 0, 3, 2},
             {0, 3, 1, 2},
             {1, 0, 2, 3}};
            AssertArray(perm2, Perm2Expected);


            print(perm3);
            var Perm3Expected = new Int64[,]
            {{0, 3, 1, 2},
             {0, 1, 3, 2},
             {1, 0, 3, 2},
             {0, 3, 1, 2},
             {1, 0, 2, 3}};
            AssertArray(perm3, Perm3Expected);
        }

        [TestMethod]
        public void test_argmax_1()
        {
            ndarray a = np.array(new int[] { 32, 33, 45, 98, 11, 02 }).reshape(new shape(2, 3));
            print(a);
            ndarray b = np.argmax(a);
            print(b);
            Assert.AreEqual(b.GetItem(0), (Int64)3);
            print("********");

            ndarray c = np.argmax(a, axis: 0);
            print(c);
            AssertArray(c, new Int64[] { 1, 0, 0 });
            print("********");

            ndarray d = np.argmax(a, axis: 1);
            print(d);
            AssertArray(d, new Int64[] { 2, 0 });
            print("********");

        }

        [TestMethod]
        public void test_argmin_1()
        {
            ndarray a = np.array(new int[] { 32, 33, 45, 98, 11, 02 }).reshape(new shape(2, 3));
            print(a);

            ndarray b = np.argmin(a);
            print(b);
            Assert.AreEqual(b.GetItem(0), (Int64)5);
            print("********");

            ndarray c = np.argmin(a, axis: 0);
            print(c);
            AssertArray(c, new Int64[] { 0, 1, 1 });
            print("********");

            ndarray d = np.argmin(a, axis: 1);
            print(d);
            AssertArray(d, new Int64[] { 0, 2 });
            print("********");

        }

        [TestMethod]
        public void test_searchsorted_1()
        {
            ndarray arr = np.array(new Int32[] { 1, 2, 3, 4, 5 });
            ndarray a = np.searchsorted(arr, 3);
            print(a);
            Assert.AreEqual(a.GetItem(0), (Int64)2);


            ndarray b = np.searchsorted(arr, 3, side: NPY_SEARCHSIDE.NPY_SEARCHRIGHT);
            print(b);
            Assert.AreEqual(b.GetItem(0), (Int64)3);


            ndarray c = np.searchsorted(arr, new Int32[] { -10, 10, 2, 3 });
            print(c);
            AssertArray(c, new Int64[] { 0, 5, 1, 2 });


            ndarray d = np.searchsorted(np.array(new Int32[] { 15, 14, 13, 12, 11 }), 13);
            print(d);
            Assert.AreEqual(d.GetItem(0), (Int64)0);


        }

        [TestMethod]
        public void test_resize_1()
        {
            ndarray a = np.array(new Int32[,] { { 0, 1 }, { 2, 3 } });
            print(a);

            ndarray b = np.resize(a, new shape(2, 3));
            print(b);

            var ExpectedDataB = new Int32[,]
            {
                { 0,1,2 },
                { 3,0,1 },
            };
            AssertArray(b, ExpectedDataB);


            ndarray c = np.resize(a, new shape(1, 4));
            print(c);
            var ExpectedDataC = new Int32[,]
            {
                { 0,1,2,3 },
            };
            AssertArray(c, ExpectedDataC);

            ndarray d = np.resize(a, new shape(2, 4));
            print(d);
            var ExpectedDataD = new Int32[,]
            {
                { 0,1,2,3 },
                { 0,1,2,3 },
            };
            AssertArray(d, ExpectedDataD);

        }

        [TestMethod]
        public void test_squeeze_1()
        {
            ndarray x = np.array(new Int32[,,] { { { 0 }, { 1 }, { 2 } } });
            print(x);
            AssertArray(x, new Int32[1, 3, 1] { { { 0 }, { 1 }, { 2 } } });

            ndarray a = np.squeeze(x);
            print(a);
            AssertArray(a, new Int32[] {0,1,2});

            ndarray b = np.squeeze(x, axis: 0);
            print(b);
            AssertArray(b, new Int32[3,1] { { 0 }, { 1 }, { 2 } });

            bool CaughtException = false;
              
            try
            {
                ndarray c = np.squeeze(x, axis: 1);
                print(c);
            }
            catch (Exception ex)
            {
                if (ex.Message.Contains("cannot select an axis to squeeze out which has size not equal to one"))
                    CaughtException = true;
            }
            Assert.IsTrue(CaughtException);

            ndarray d = np.squeeze(x, axis: 2);
            print(d);
            AssertArray(d, new Int32[,] { { 0, 1, 2 } });
        }

        [TestMethod]
        public void test_squeeze_2()
        {
            ndarray x = np.arange(0,32,1,dtype:np.Float32).reshape(new shape(-1,1,8,1));
            print(x);

            var ExpectedDataX = new float[,,,]
                {{{{0.0f},{1.0f},{2.0f},{3.0f},{4.0f},{5.0f},{6.0f},{7.0f}}},
                 {{{8.0f},{ 9.0f},{10.0f},{11.0f},{12.0f},{13.0f},{14.0f},{15.0f}}},
                 {{{16.0f},{17.0f},{18.0f},{19.0f},{20.0f},{21.0f},{22.0f},{23.0f}}},
                 {{{24.0f},{25.0f},{26.0f},{27.0f},{28.0f},{29.0f},{30.0f},{31.0f}}}};

            AssertArray(x, ExpectedDataX);

            ndarray a = np.squeeze(x);
            print(a);

            var ExpectedDataA = new float[,]
                {{0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f},
                 {8.0f,  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f},
                 {16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f},
                 {24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f}};
            AssertArray(a, ExpectedDataA);

            ndarray b = np.squeeze(x, axis: 1);
            print(b);

            var ExpectedDataB = new float[,,]
               {{{0.0f},{1.0f},{2.0f},{3.0f},{4.0f},{5.0f},{6.0f},{7.0f}},
                {{8.0f},{ 9.0f},{10.0f},{11.0f},{12.0f},{13.0f},{14.0f},{15.0f}},
                {{16.0f},{17.0f},{18.0f},{19.0f},{20.0f},{21.0f},{22.0f},{23.0f}},
                {{24.0f},{25.0f},{26.0f},{27.0f},{28.0f},{29.0f},{30.0f},{31.0f}}};

            AssertArray(b, ExpectedDataB);

            bool CaughtException = false;

            try
            {
                ndarray c = np.squeeze(x, axis: 0);
                print(c);
            }
            catch (Exception ex)
            {
                if (ex.Message.Contains("cannot select an axis to squeeze out which has size not equal to one"))
                    CaughtException = true;
            }
            Assert.IsTrue(CaughtException);

            CaughtException = false;
            try
            {
                ndarray d = np.squeeze(x, axis: 2);
                print(d);
            }
            catch (Exception ex)
            {
                if (ex.Message.Contains("cannot select an axis to squeeze out which has size not equal to one"))
                    CaughtException = true;
            }
            Assert.IsTrue(CaughtException);


            ndarray e = np.squeeze(x, axis: 3);
            print(e);

            var ExpectedDataE = new float[4, 1, 8]
                {{{0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f}},
                 {{8.0f,  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,}},
                 {{16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,}},
                 {{24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,}}};

            AssertArray(e, ExpectedDataE);
        }

        [TestMethod]
        public void test_diagonal_1()
        {
            ndarray a = np.arange(4).reshape(new shape(2, 2));
            print(a);
            print("*****");

            ndarray b = a.diagonal();
            print(b);
            AssertArray(b, new Int32[] { 0, 3 });
            print("*****");

            ndarray c = a.diagonal(1);
            print(c);
            AssertArray(c, new Int32[] { 1 });
            print("*****");

            a = np.arange(8).reshape(new shape(2, 2, 2));
            print(a);
            print("*****");
            b = a.diagonal(0, // Main diagonals of two arrays created by skipping
                           0, // across the outer(left)-most axis last and
                           1); //the "middle" (row) axis first.

            print(b);
            AssertArray(b, new Int32[,] { {0,6}, {1,7} });
            print("*****");

            ndarray d = a.A(":", ":", 0);
            print(d);
            AssertArray(d, new Int32[,] { { 0, 2 }, { 4, 6 } });
            print("*****");

            ndarray e = a.A(":", ":", 1);
            print(e);
            AssertArray(e, new Int32[,] { { 1, 3 }, { 5, 7 } });
            print("*****");
        }

        [TestMethod]
        public void test_trace_1()
        {
            ndarray a = np.trace(np.eye(3));
            print(a);
            Assert.AreEqual(a.GetItem(0), 3.0);
            print("*****");

            a = np.arange(8).reshape(new shape(2, 2, 2));
            ndarray b = np.trace(a);
            print(b);
            AssertArray(b, new Int32[] { 6, 8 });
            print("*****");

            a = np.arange(24).reshape(new shape(2, 2, 2, 3));
            var c = np.trace(a);
            print(c);
            AssertArray(c, new Int32[,] { { 18, 20, 22 }, { 24, 26, 28 } });

        }

        [TestMethod]
        public void test_nonzero_1()
        {
            ndarray x = np.array(new Int32[,] { { 1, 0, 0 }, { 0, 2, 0 }, { 1, 1, 0 } });
            print(x);
            print("*****");

            ndarray[] y = np.nonzero(x);
            print(y);
            AssertArray(y[0], new Int64[] { 0, 1, 2, 2 });
            AssertArray(y[1], new Int64[] { 0, 1, 0, 1 });
            print("*****");

            ndarray z = x.A(np.nonzero(x));
            print(z);
            AssertArray(z, new Int32[] { 1,2,1,1 });
            print("*****");

            //ndarray q = np.transpose(np.nonzero(x));
            //print(q);

        }


        [TestMethod]
        public void test_compress_1()
        {
            ndarray a = np.array(new Int32[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            print(a);
            print("*****");

            ndarray b = np.compress(new int[] { 0, 1 }, a, axis: 0);
            print(b);
            AssertArray(b, new int[,] {{3,4}});
            print("*****");

            ndarray c = np.compress(new bool[] { false, true, true }, a, axis: 0);
            print(c);
            AssertArray(c, new int[,] { { 3, 4 }, { 5, 6 } });
            print("*****");

            ndarray d = np.compress(new bool[] { false, true }, a, axis: 1);
            print(d);
            AssertArray(d, new int[,] { { 2 },{ 4 },{ 6 } });
            print("*****");

            ndarray e = np.compress(new bool[] { false, true }, a);
            AssertArray(e, new int[] {2});
            print(e);

        }

        [TestMethod]
        public void test_clip_1()
        {
            ndarray a = np.arange(10);
            print(a);
            print("*****");

            ndarray b = np.clip(a, 1, 8);
            print(b);
            print("*****");
            AssertArray(b, new Int32[] { 1, 1, 2, 3, 4, 5, 6, 7, 8, 8 });

            ndarray c = np.clip(a, 3, 6, @out: a);
            print(c);
            AssertArray(c, new Int32[] { 3, 3, 3, 3, 4, 5, 6, 6, 6, 6, });
            print(a);
            AssertArray(a, new Int32[] { 3, 3, 3, 3, 4, 5, 6, 6, 6, 6, });
            print("*****");


            a = np.arange(10);
            print(a);
            b = np.clip(a, np.array(new Int32[] { 3, 4, 1, 1, 1, 4, 4, 4, 4, 4 }), 8);
            print(b);
            AssertArray(a, new Int32[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            AssertArray(b, new Int32[] { 3, 4, 2, 3, 4, 5, 6, 7, 8, 8 });
            print("*****");
        }

        [TestMethod]
        public void test_clip_2()
        {
            ndarray a = np.arange(16).reshape(new shape(4,4));
            print(a);
            print("*****");

            ndarray b = np.clip(a, 1, 8);
            print(b);
            print("*****");
            AssertArray(b, new Int32[,] { { 1, 1, 2, 3 }, { 4, 5, 6, 7 }, { 8, 8, 8, 8 }, { 8, 8, 8, 8 } });

            ndarray c = np.clip(a, 3, 6, @out: a);
            print(c);
            AssertArray(c, new Int32[,] { { 3, 3, 3, 3 }, { 4, 5, 6, 6 }, { 6, 6, 6, 6 }, { 6, 6, 6, 6 } });
            print(a);
            AssertArray(a, new Int32[,] { { 3, 3, 3, 3 }, { 4, 5, 6, 6 }, { 6, 6, 6, 6 }, { 6, 6, 6, 6 } });
            print("*****");


            // TODO: UFUNC - uncomment when fix the multi-array UFUNC problem.  This might work then.
            //a = np.arange(16).reshape(new shape(4, 4));
            //print(a);
            //b = np.clip(a, np.array(new Int32[] { 3, 4, 1, 1 }), 8);
            //print(b);
            //AssertArray(a, new Int32[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            //AssertArray(b, new Int32[] { 3, 4, 2, 3, 4, 5, 6, 7, 8, 8 });
            //print("*****");
        }

        [TestMethod]
        public void test_sum_1()
        {
            UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));
            x = x * 3;
            var y = np.sum(x);

            print(x);
            print(y);

            Assert.AreEqual(y.GetItem(0), (UInt32)1578);

        }

        [TestMethod]
        public void test_sum_2()
        {
            UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));
            x = x * 3;

            var y = np.sum(x, axis:0);
            print(y);
            AssertArray(y, new UInt32[,] { {339,450}, {339,450} });

            print("*****");

            y = np.sum(x, axis: 1);
            print(y);
            AssertArray(y, new UInt32[,] { { 105, 180 }, { 264, 315 }, { 309, 405 } });

            print("*****");

            y = np.sum(x, axis: 2);
            print(y);
            AssertArray(y, new UInt32[,] { { 75, 210 }, { 504, 75 }, { 210, 504 } });

            print("*****");

        }

        [TestMethod]
        public void test_sum_3()
        {
            double[] TestData = new double[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.Float64).reshape(new shape(3, 2, -1));
            x = x * 3.456;

            var y = np.sum(x, axis: 0);
            print(y);
            AssertArray(y, new double[,] { { 390.528, 518.4 }, { 390.528, 518.4 } });
            print("*****");

            y = np.sum(x, axis: 1);
            print(y);
            AssertArray(y, new double[,] { { 120.96, 207.36 }, { 304.128, 362.88 }, { 355.968, 466.56 } });
            print("*****");

            y = np.sum(x, axis: 2);
            print(y);
            AssertArray(y, new double[,] { { 86.4, 241.92 }, { 580.608, 86.4 }, { 241.92, 580.608 } });

            print("*****");

        }

        [TestMethod]
        public void test_any_1()
        {
            float[] TestData = new float[] { 2.5f, -1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f };
            var x = np.array(TestData);
            var y = np.any(x);

            print(x);
            print(y);
            Assert.AreEqual(true, y.GetItem(0));

            TestData = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };
            x = np.array(TestData);
            y = np.any(x);

            print(x);
            print(y);
            Assert.AreEqual(false, y.GetItem(0));

        }

        [TestMethod]
        public void test_any_2()
        {
            ndarray data = np.array(new bool[,] { { true, false }, { true, true } });
            ndarray a = np.any(data);
            print(a);
            Assert.AreEqual(true, a.GetItem(0));
            print("*****");

            data = np.array(new bool[,] { { true, false }, { false, false } });
            ndarray b = np.any(data, axis: 0);
            print(b);
            AssertArray(b, new bool[] { true, false });
            print("*****");

            data = np.array(new Int32[] { -1, 0, 5 });
            ndarray c = np.any(data);
            print(c);
            Assert.AreEqual(true, c.GetItem(0));
            print("*****");

            ndarray d = np.any(np.NaN);
            print(d);
            Assert.AreEqual(true, d.GetItem(0));
            print("*****");

            // o=np.array([False])
            // z=np.any([-1, 4, 5], out=o)
            // print(z, o)

        }

        [TestMethod]
        public void test_any_3()
        {
            float[] TestData = new float[] { 2.5f, -1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f, 0f };
            var x = np.array(TestData).reshape(new shape(3,3));
            print(x);

            var y = np.any(x, axis:0);
            print(y);
            AssertArray(y, new bool[] { true, true, true });

            y = np.any(x, axis: 1);
            print(y);
            AssertArray(y, new bool[] { true, true, true });


            TestData = new float[] { 2.5f, -1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 0, 0, 0 };
            x = np.array(TestData).reshape(new shape(3, 3));
            print(x);

            y = np.any(x, axis: 0);
            print(y);
            AssertArray(y, new bool[] { true, true, true });

            y = np.any(x, axis: 1);
            print(y);
            AssertArray(y, new bool[] { true, true, false });


        }

        [TestMethod]
        public void test_all_1()
        {
            float[] TestData = new float[] { 2.5f, -1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f };
            var x = np.array(TestData);
            var y = np.all(x);

            print(x);
            print(y);
            Assert.AreEqual(true, y.GetItem(0));

            TestData = new float[] { 1.0f, 1.0f, 0.0f, 1.0f };
            x = np.array(TestData);
            y = np.all(x);

            print(x);
            print(y);
            Assert.AreEqual(false, y.GetItem(0));

        }

        [TestMethod]
        public void test_all_2()
        {
            ndarray data = np.array(new bool[,] { { true, false }, { true, true } });
            ndarray a = np.all(data);
            print(a);
            Assert.AreEqual(false, a.GetItem(0));

            data = np.array(new bool[,] { { true, false }, { true, true } });
            ndarray b = np.all(data, axis: 0);
            print(b);
            AssertArray(b, new bool[] { true, false });

            data = np.array(new bool[,] { { true, false }, { true, true } });
            b = np.all(data, axis: 1);
            print(b);
            AssertArray(b, new bool[] { false, true });

            data = np.array(new int[] { -1, 4, 5 });
            ndarray c = np.all(data);
            print(c);
            Assert.AreEqual(true, c.GetItem(0));

            data = np.array(new float[] { 1.0f, np.NaN });
            ndarray d = np.all(data);
            print(d);
            Assert.AreEqual(true, d.GetItem(0));

        }


        [TestMethod]
        public void test_cumsum_1()
        {
            UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));
            x = x * 3;
            var y = np.cumsum(x);

            print(x);
            print(y);

            AssertArray(y, new UInt32[] { 30, 75, 150, 285, 519, 789, 819, 864, 939, 1074, 1308, 1578 });

        }

        [TestMethod]
        public void test_cumsum_2()
        {
            ndarray a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            print(a);
            print("*****");

            ndarray b = np.cumsum(a);
            print(b);
            AssertArray(b, new Int32[] { 1, 3, 6, 10, 15, 21 });
            print("*****");

            ndarray c = np.cumsum(a, dtype: np.Float32);     // specifies type of output value(s)
            print(c);
            AssertArray(c, new float[] { 1f, 3f, 6f, 10f, 15f, 21f });

            print("*****");

            ndarray d = np.cumsum(a, axis: 0);     // sum over rows for each of the 3 columns
            print(d);
            AssertArray(d, new Int32[,] { { 1, 2, 3 }, { 5, 7, 9 } });
            print("*****");


            ndarray e = np.cumsum(a, axis: 1);    // sum over columns for each of the 2 rows
            print(e);
            AssertArray(e, new Int32[,] { { 1, 3, 6 }, { 4, 9, 15 } });
        }

        [TestMethod]
        public void test_cumsum_3()
        {
            ndarray a = np.array(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }).reshape(new shape(2, 3, -1));
            print(a);
            print("*****");

            ndarray b = np.cumsum(a);
            print(b);
            AssertArray(b, new Int32[] { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78 });
            print("*****");

            ndarray c = np.cumsum(a, dtype: np.Float32);     // specifies type of output value(s)
            print(c);
            AssertArray(c, new float[] { 1f, 3f, 6f, 10f, 15f, 21f, 28f, 36f, 45f, 55f, 66f, 78f });
            print("*****");

            ndarray d = np.cumsum(a, axis: 0);     // sum over rows for each of the 3 columns
            print(d);

            var ExpectedDataD = new int[,,]
            {{{1,  2},
              {3,  4},
              {5,  6}},

             {{ 8, 10},
              {12, 14},
              {16, 18}}};

            AssertArray(d, ExpectedDataD);
            print("*****");



            ndarray e = np.cumsum(a, axis: 1);    // sum over columns for each of the 2 rows
            print(e);

            var ExpectedDataE = new int[,,]
            {{{1,  2},
              {4,  6},
              {9,  12}},

             {{ 7, 8},
              {16, 18},
              {27, 30}}};

            AssertArray(e, ExpectedDataE);
            print("*****");

            ndarray f = np.cumsum(a, axis: 2);    // sum over columns for each of the 2 rows
            print(f);

            var ExpectedDataF = new int[,,]
            {{{1,  3},
              {3,  7},
              {5,  11}},

             {{7, 15},
              {9, 19},
              {11, 23}}};

            AssertArray(f, ExpectedDataF);
            print("*****");

        }

        [TestMethod]
        public void test_ptp_4()
        {
            ndarray a = np.arange(4).reshape(new shape(2, 2));
            print(a);
            print("*****");

            ndarray b = np.ptp(a, axis: 0);
            print(b);
            AssertArray(b, new Int32[] { 2, 2 });
            print("*****");

            ndarray c = np.ptp(a, axis: 1);
            print(c);
            AssertArray(c, new Int32[] { 1, 1 });

            ndarray d = np.ptp(a);
            print(d);
            Assert.AreEqual(3, d.GetItem(0));
        }

        [TestMethod]
        public void test_amax_1()
        {
            ndarray a = np.arange(4).reshape(new shape(2, 2));
            print(a);
            print("*****");

            ndarray b = np.amax(a);          // Maximum of the flattened array
            print(b);
            Assert.AreEqual(3, b.GetItem(0));
            print("*****");

            ndarray c = np.amax(a, axis: 0);  // Maxima along the first axis
            print(c);
            AssertArray(c, new Int32[] { 2, 3 });
            print("*****");

            ndarray d = np.amax(a, axis: 1);   // Maxima along the second axis
            print(d);
            AssertArray(d, new Int32[] { 1, 3 });
            print("*****");

            ndarray e = np.arange(5, dtype: np.Float32);
            e[2] = np.NaN;
            ndarray f = np.amax(e);
            print(f);
            Assert.AreEqual(np.NaN, f.GetItem(0));
            print("*****");

            //ndarray g = np.nanmax(b);
            //print(g);
        }

        [TestMethod]
        public void test_amax_2()
        {
            ndarray a = np.arange(30.25, 46.25).reshape(new shape(4, 4));
            print(a);
            print("*****");

            ndarray b = np.amax(a);          // Maximum of the flattened array
            print(b);
            Assert.AreEqual(45.25f, b.GetItem(0));
            print("*****");

            ndarray c = np.amax(a, axis: 0);  // Maxima along the first axis
            print(c);
            AssertArray(c, new float[] { 42.25f, 43.25f, 44.25f, 45.25f });
            print("*****");

            ndarray d = np.amax(a, axis: 1);   // Maxima along the second axis
            print(d);
            AssertArray(d, new float[] { 33.25f, 37.25f, 41.25f, 45.25f });
            print("*****");

            ndarray e = np.arange(5, dtype: np.Float32);
            e[2] = np.NaN;
            ndarray f = np.amax(e);
            print(f);
            Assert.AreEqual(np.NaN, f.GetItem(0));
            print("*****");

            //ndarray g = np.nanmax(b);
            //print(g);
        }


        [TestMethod]
        public void test_amin_1()
        {
            ndarray a = np.arange(1, 5).reshape(new shape(2, 2));
            print(a);
            print("*****");

            ndarray b = np.amin(a);          // Minimum of the flattened array
            print(b);
            Assert.AreEqual(1, b.GetItem(0));
            print("*****");

            ndarray c = np.amin(a, axis: 0);  // Minima along the first axis
            print(c);
            AssertArray(c, new Int32[] { 1,2 });
            print("*****");

            ndarray d = np.amin(a, axis: 1);   // Minima along the second axis
            print(d);
            AssertArray(d, new Int32[] { 1, 3 });
            print("*****");

            ndarray e = np.arange(5, dtype: np.Float32);
            e[2] = np.NaN;
            ndarray f = np.amin(e);
            print(f);
            Assert.AreEqual(np.NaN, f.GetItem(0));
            print("*****");

            //ndarray g = np.nanmin(b);
            //print(g);

        }

        [TestMethod]
        public void test_amin_2()
        {
            ndarray a = np.arange(30.25, 46.25).reshape(new shape(4, 4));
            print(a);
            print("*****");

            ndarray b = np.amin(a);          // Minimum of the flattened array
            print(b);
            Assert.AreEqual(30.25f, b.GetItem(0));
            print("*****");

            ndarray c = np.amin(a, axis: 0);  // Minima along the first axis
            print(c);
            AssertArray(c, new float[] { 30.25f, 31.25f, 32.25f, 33.25f });
            print("*****");

            ndarray d = np.amin(a, axis: 1);   // Minima along the second axis
            print(d);
            AssertArray(d, new float[] { 30.25f, 34.25f, 38.25f, 42.25f });
            print("*****");

            ndarray e = np.arange(5, dtype: np.Float32);
            e[2] = np.NaN;
            ndarray f = np.amin(e);
            print(f);
            Assert.AreEqual(np.NaN, f.GetItem(0));
            print("*****");

            //ndarray g = np.nanmin(b);
            //print(g);

        }

        [TestMethod]
        public void test_prod_1()
        {
            //UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            //var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));

            UInt64[] TestData = new UInt64[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt64);

            var y = np.prod(x);

            print(x);
            print(y);
            Assert.AreEqual((UInt64)1403336390624999936, y.GetItem(0));

        }


        [TestMethod]
        public void test_prod_2()
        {
            ndarray a = np.prod(np.array(new double[] { 1.0, 2.0 }));
            print(a);
            Assert.AreEqual((double)2, a.GetItem(0));
            print("*****");

            ndarray b = np.prod(np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }));
            print(b);
            Assert.AreEqual((double)24, b.GetItem(0));
            print("*****");

            ndarray c = np.prod(np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }), axis: 1);
            print(c);
            AssertArray(c, new double[] { 2, 12 });
            print("*****");

            ndarray d = np.array(new byte[] { 1, 2, 3 }, dtype: np.UInt8);
            bool e = np.prod(d).Dtype.TypeNum == NPY_TYPES.NPY_UINT64;  // note: different typenum from python
            print(e);
            Assert.AreEqual(true, e);
            print("*****");

            ndarray f = np.array(new sbyte[] { 1, 2, 3 }, dtype: np.Int8);
            bool g = np.prod(f).Dtype.TypeNum == NPY_TYPES.NPY_UINT64; // note: different typenum from python
            print(g);
            Assert.AreEqual(true, g);

            print("*****");

        }

        [TestMethod]
        public void test_prod_3()
        {
            ndarray a = np.array(new Int32[] { 1, 2, 3 });
            ndarray b = np.prod(a);          // intermediate results 1, 1*2
                                                // total product 1*2*3 = 6
            print(b);
            Assert.AreEqual((UInt64)6, b.GetItem(0));
            print("*****");

            a = np.array(new Int32[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            ndarray c = np.prod(a, dtype: np.Float32); //specify type of output
            print(c);
            Assert.AreEqual((float)720, c.GetItem(0));
            print("*****");

            ndarray d = np.prod(a, axis: 0);
            print(d);
            AssertArray(d, new Int64[] { 4, 10, 18 });
            print("*****");


            ndarray e = np.prod(a, axis: 1);
            print(e);
            AssertArray(e, new Int64[] { 6, 120 });
            print("*****");

        }

        [TestMethod]
        public void test_cumprod_1()
        {

            bool CaughtException = false;

            try
            {
                UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
                var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));
                x = x * 3;

                var y = np.cumprod(x);
                print(y);

                AssertArray(y, new UInt32[] {30,1350,101250,13668750,3198487500,303198504,
                            506020528,1296087280,2717265488,1758620720,3495355360,3148109376 });
            }
            catch (Exception ex)
            {
                if (ex.Message.Contains("NpyExc_OverflowError"))
                    CaughtException = true;
            }
            Assert.IsTrue(CaughtException);

            try
            {
                Int32[] TestData2 = new Int32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
                var x = np.array(TestData2, dtype: np.Int32).reshape(new shape(3, 2, -1));
                x = x * 3;

                var y = np.cumprod(x);

                print(y);

                AssertArray(y, new Int32[] { 30, 1350, 101250, 13668750, -1096479796, 303198504,
                            506020528, 1296087280, -1577701808, 1758620720, -799611936 -1146857920});
            }
            catch (Exception ex)
            {
                CaughtException = true;
            }
            Assert.IsTrue(CaughtException);

   

        }

        [TestMethod]
        public void test_cumprod_1a()
        {

            bool CaughtException = false;

            try
            {
                UInt64[] TestData = new UInt64[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
                var x = np.array(TestData, dtype: np.UInt64).reshape(new shape(3, 2, -1));
                x = x * 1;

                var y = np.cumprod(x);
                print(y);

                AssertArray(y, new UInt64[] {10, 150, 3750, 168750, 13162500, 1184625000,
                                11846250000, 177693750000, 4442343750000,199905468750000,
                                15592626562500000, 1403336390625000000 });
            }
            catch (Exception ex)
            {
                CaughtException = true;
            }
            Assert.IsFalse(CaughtException);

            try
            {
                Int64[] TestData2 = new Int64[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
                var x = np.array(TestData2, dtype: np.Int64).reshape(new shape(3, 2, -1));
                x = x * 1;

                var y = np.cumprod(x);

                print(y);

                AssertArray(y, new Int64[] {10, 150, 3750, 168750, 13162500, 1184625000,
                                11846250000, 177693750000, 4442343750000,199905468750000,
                                15592626562500000, 1403336390625000000 });
            }
            catch (Exception ex)
            {
                CaughtException = true;
            }
            Assert.IsFalse(CaughtException);

        }

        [TestMethod]
        public void test_cumprod_2()
        {
            ndarray a = np.array(new Int32[] { 1, 2, 3 });
            ndarray b = np.cumprod(a);          // intermediate results 1, 1*2
                                                // total product 1*2*3 = 6
            print(b);
            AssertArray(b, new Int32[] { 1, 2, 6 });
            print("*****");

            a = np.array(new Int32[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            ndarray c = np.cumprod(a, dtype: np.Float32); //specify type of output
            print(c);
            AssertArray(c, new float[] { 1f, 2f, 6f, 24f, 120f, 720f });
            print("*****");

            ndarray d = np.cumprod(a, axis: 0);
            print(d);
            AssertArray(d, new Int32[,] { { 1, 2, 3 }, { 4, 10, 18} });
            print("*****");

            ndarray e = np.cumprod(a, axis: 1);
            print(e);
            AssertArray(e, new Int32[,] { { 1, 2, 6 }, { 4, 20, 120 } });
            print("*****");

        }

        [TestMethod]
        public void test_size_2()
        {
            ndarray a = np.array(new Int32[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            print(np.size(a));
            print(np.size(a, 1));
            print(np.size(a, 0));
        }

        [TestMethod]
        public void test_around_1()
        {
            ndarray a = np.around(np.array(new double[] { 0.37, 1.64 }));
            print(a);
            AssertArray(a, new double[] {0,2});

            ndarray b = np.around(np.array(new double[] { 0.37, 1.64 }), decimals: 1);
            print(b);
            AssertArray(b, new double[] { 0.4, 1.6 });

            ndarray c = np.around(np.array(new double[] { .5, 1.5, 2.5, 3.5, 4.5 })); // rounds to nearest even value
            print(c);
            AssertArray(c, new double[] { 0.0, 2.0, 2.0, 4.0, 4.0 });

            ndarray d = np.around(np.array(new int[] { 1, 2, 3, 11 }), decimals: 1); // ndarray of ints is returned
            print(d);
            AssertArray(d, new double[] { 1,2,3,11 });

            ndarray e = np.around(np.array(new int[] { 1, 2, 3, 11 }), decimals: -1);
            print(e);
            AssertArray(e, new double[] { 0, 0, 0, 10 });

        }

        [TestMethod]
        public void test_ndarray_mean_1()
        {
            var x = np.arange(0, 12, dtype: np.UInt8).reshape(new shape(3, -1));

            print("X");
            print(x);

            var y = (ndarray)np.mean(x);

            print("Y");
            print(y);

            Assert.AreEqual(5.5, y.GetItem(0));
        }


        [TestMethod]
        public void test_mean_1()
        {
            UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));
            x = x * 3;
            var y = np.mean(x);

            print(x);
            print(y);
            Assert.AreEqual(131.5, y.GetItem(0));

        }

        [TestMethod]
        public void test_mean_2()
        {
            ndarray a = np.zeros(new shape(2, 512 * 512), dtype: np.Float32);
            a[0, ":"] = 1.0;
            a[1, ":"] = 0.1;
            ndarray b = np.mean(a);
            print(b);
            Assert.AreEqual(0.5500000007450581, (double)b.GetItem(0), 0.0000001);

            ndarray c = np.mean(a, dtype: np.Float64);
            print(c);
            Assert.AreEqual(0.5500000007450581, c.GetItem(0));
        }


        [TestMethod]
        public void test_mean_3()
        {
            ndarray x = np.array(new int[] { 1, 2, 2, 2, 1 });
            var mean = x.Mean<int>();

            ndarray y = x.A("1:3");
            mean = y.Mean<int>();


        }


        [TestMethod]
        public void test_std_1()
        {
            ndarray a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            ndarray b = np.std(a);
            print(b);
            Assert.AreEqual(1.11803398874989, (double)b.GetItem(0), 0.0000001);

            ndarray c = np.std(a, axis: 0);
            print(c);
            AssertArray(c, new double[] { 1.0, 1.0 });

            ndarray d = np.std(a, axis: 1);
            print(d);
            AssertArray(d, new double[] { 1.11803398874989, 1.80277563773199 });

            // In single precision, std() can be inaccurate:
            a = np.zeros(new shape(2, 512 * 512), dtype: np.Float32);
            a[0, ":"] = 1.0;
            a[1, ":"] = 0.1;
            b = np.std(a);
            print(b);
            Assert.AreEqual(0.450010849825055, (double)b.GetItem(0), 0.0000001);
            // Computing the standard deviation in float64 is more accurate:
            c = np.std(a, dtype: np.Float64);
            print(c);
            Assert.AreEqual(0.449999999255527, (double)c.GetItem(0), 0.0000001);

        }

        [Ignore] // not implemented yet
        [TestMethod]
        public void xxx_test_var_1()
        {
            ndarray a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            ndarray b = np.var(a);
            print(b);

            ndarray c = np.var(a, axis: 0);
            print(c);

            ndarray d = np.var(a, axis: 1);
            print(d);

            // In single precision, std() can be inaccurate:
            a = np.zeros(new shape(2, 512 * 512), dtype: np.Float32);
            a[0, ":"] = 1.0;
            a[1, ":"] = 0.1;
            b = np.var(a);
            print(b);

            // Computing the standard deviation in float64 is more accurate:
            c = np.var(a, dtype: np.Float64);
            print(c);

        }

        [Ignore] // not implemented yet
        [TestMethod]
        public void xxx_test_place_1()
        {

        }

        [Ignore] // not implemented yet
        [TestMethod]
        public void test_lexsort_1()
        {

        }

        [Ignore] // not implemented yet
        [TestMethod]
        public void test_msort_1()
        {

        }

        [Ignore] // not implemented yet
        [TestMethod]
        public void test_sort_complex_1()
        {

        }

        [Ignore] // not implemented yet
        [TestMethod]
        public void test_extract_1()
        {

        }

    }
}
