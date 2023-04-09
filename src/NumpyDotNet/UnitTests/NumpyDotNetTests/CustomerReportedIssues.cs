using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumpyLib;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNetTests
{

    [TestClass]
    public class CustomerReportedIssues : TestBaseClass
    {
        [TestMethod]
        public void test_tensordot_asiamartini_bugreport()
        {
            var alpha_0 = np.array(new double[] { 1.0, 1.0, 1.0 });
            var temp = np.array(new double[,,] { { { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0, 0.0 } },
                                                 { { -18.0, 12.0, 12.0, -10.0 }, { 12.0, 10.0, 18.0, 12.0 }, { 12.0, 18.0, -10.0, -12.0 }, { -10.0, 12.0, -12.0, 18.0 } },
                                                 { { -3.5, 21.5, 5.5, -2.5 }, { 21.5, 2.5, 5.5, 3.5 }, { 5.5, 5.5, -20.5, -5.5 }, { -2.5, 3.5, -5.5, 21.5 } }
                                               });

            var matrix = np.tensordot(alpha_0, temp, axes: (new long[] { 0 }, new long[] { 0 }));

            AssertArray(matrix, new double[,] { { -21.5, 33.5, 17.5, -12.5 }, { 33.5, 12.5, 23.5, 15.5 }, { 17.5, 23.5, -30.5, -17.5 }, { -12.5, 15.5, -17.5, 39.5 } });

            print(matrix);
        }


        [TestMethod]
        public void test_matmul_asiamartini_bugreport()
        {
            var rq = np.array(new double[] { 0.5, 0.5, 0.5, 0.5 });
            var am = np.array(new double[,] { { -21.5, 33.5, 17.5, -12.5 }, { 33.5, 12.5, 23.5, 15.5 }, { 17.5, 23.5, -30.5, -17.5 }, { -12.5, 15.5, -17.5, 39.5 } });

            var temp1 = np.matmul(rq.T, am);

            AssertArray(temp1, new double[] { 8.5, 42.5, -3.5, 12.5 });
            print(temp1);
        }

        [TestMethod]
        public void test_randint_bitstormGER_bugreport()
        {
            var random = new np.random();
            random.seed(8357);

            var r1 = random.randint(0, 4242, dtype: np.UInt32);
            AssertArray(r1, new UInt32[] { 2193 });

            random.seed(8357);
            var r2 = random.randint(0, 4242, new shape(1), dtype: np.UInt32);
            AssertArray(r2, new UInt32[] { 2193 });

        }


        [TestMethod]
        public void test_customer_goodgood_reported_issue()
        {

            var img = np.array(new float[]
                { 208.0f, 54.0f, 1.0f, 255.0f, 255.0f,240.0f, 251.0f, 255.0f,252.0f,
                  176.0f, 51.0f, 20.0f,255.0f, 255.0f, 238.0f,250.0f, 255.0f, 249.0f,
                  146.0f, 53.0f, 28.0f,255.0f, 255.0f, 242.0f,251.0f, 255.0f, 251.0f,
                  255.0f, 255.0f, 240.0f,255.0f, 255.0f, 240.0f,255.0f, 255.0f, 240.0f,
                  253.0f, 255.0f, 251.0f,251.0f, 255.0f, 249.0f,255.0f, 252.0f, 255.0f });

            img = img.reshape(5, 3, 3);
            print(img);
            var output = img.Copy();

            ndarray equalFlags = np.equal(img, np.array(new float[] { 255, 255, 240 }));
            ndarray valid = np.all(equalFlags, axis: -1);
            ndarray[] rscs = valid.NonZero();

            //output[rscs[0], rscs[1], ":"] = np.array(new float[] { 255, 255, 255 });
            output[rscs[0], rscs[1], ":"] = new float[] { 255, 255, 255 };

            var ExpectedDataB = new float[,,]
                { { { 208.0f, 54.0f, 1.0f },
                    { 255.0f, 255.0f, 255.0f },
                    { 251.0f, 255.0f, 252.0f } },
                  { { 176.0f, 51.0f, 20.0f },
                    { 255.0f, 255.0f, 238.0f },
                    { 250.0f, 255.0f, 249.0f } },
                  { { 146.0f, 53.0f, 28.0f },
                    { 255.0f, 255.0f, 242.0f },
                    { 251.0f, 255.0f, 251.0f } },
                  { { 255.0f, 255.0f, 255.0f },
                    { 255.0f, 255.0f, 255.0f },
                    { 255.0f, 255.0f, 255.0f } },
                  { { 253.0f, 255.0f, 251.0f },
                    { 251.0f, 255.0f, 249.0f },
                    { 255.0f, 252.0f, 255.0f } } };

            print(output);

            AssertArray(output, ExpectedDataB);
            return;

        }

        [TestMethod]
        public void test_customer_goodgood_reported_issue_2()
        {
            var imgnd = np.arange(1, 301).reshape(new shape(10, 10, 3));


            var r = (ndarray)imgnd[":", ":", "0"];
            var g = (ndarray)imgnd[":", ":", 1];
            var b = (ndarray)imgnd[":", ":", 2];

            r = imgnd.A(":", ":", "0");
            g = imgnd.A(":", ":", 1);
            b = imgnd.A(":", ":", 2);

            print(r.shape);
            print(g.shape);
            print(b.shape);
            print(r);
            print(g);
            print(b);

        }

        [TestMethod]
        public void test_tuple_msever_1()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            print(a);
            var b = np.array(new int[] { 2, 3, 4 });
            print(b);

            var c = np.column_stack(new object[] { a, b });
            print(c);

            var ExpectedDataC = new Int32[,]
            {
                {1,2},
                {2,3},
                {3,4},
            };

            AssertArray(c, ExpectedDataC);
        }


        [TestMethod]
        public void test_append_msever_1()
        {
            ndarray a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }, dtype: np.Int32);
            ndarray b = np.array(new int[] { 7, 8, 9 }, dtype: np.Int32);


            //b = b[np.newaxis, "..."] as ndarray;
            b = np.expand_dims(b, axis: 0);


            ndarray c = np.append(a, b, axis: 0);

            print(c);
            print(c.shape);

            var ExpectedDataC = new Int32[,]
            {
                { 1,2,3 },
                { 4,5,6 },
                { 7,8,9 },
            };

            AssertArray(c, ExpectedDataC);

        }

        [TestMethod]
        public void test_append_msever_2()
        {
            var empty_array = np.empty((0, 4), dtype: np.Int32);
            print("Empty 2D Numpy array:");
            print(empty_array);

            // Append a row to the 2D numpy array
            empty_array = np.append(empty_array, np.array(new Int32[,] { { 11, 21, 31, 41 } }), axis: 0);
            // Append 2nd rows to the 2D Numpy array
            empty_array = np.append(empty_array, np.array(new Int32[,] { { 15, 25, 35, 45 } }), axis: 0);
            print("2D Numpy array:");
            print(empty_array);

            // Append multiple rows i.e 2 rows to the 2D Numpy array
            empty_array = np.append(empty_array, np.array(new Int32[,] { { 16, 26, 36, 46 }, { 17, 27, 37, 47 } }), axis: 0);
            print("2D Numpy array:");
            print(empty_array);

            var ExpectedDataC = new Int32[,]
            {
                { 11,21,31,41 },
                { 15,25,35,45 },
                { 16,26,36,46 },
                { 17,27,37,47 },
            };

            AssertArray(empty_array, ExpectedDataC);


        }

        [TestMethod]
        public void test_slice_msever_1()
        {
            var a = np.array(new int[,] { { 1, 3, 0 }, { 0, 0, 5 } });

            var col0 = (ndarray)a[":", 0];
            var col1 = (ndarray)a[":", 1];
            var col2 = (ndarray)a[":", 2];

            print(col0);
            AssertArray(col0, new Int32[] { 1, 0 });

            print(col1);
            AssertArray(col1, new Int32[] { 3, 0 });

            print(col2);
            AssertArray(col2, new Int32[] { 0, 5 });

        }

        [TestMethod]
        public void test_hsplit_msever_1()
        {
            var a = np.array(new int[,] { { 1, 3, 0 }, { 0, 0, 5 } });

            var hsplitret = np.hsplit(np.argwhere(a), 2);

            var rowcol = hsplitret.ToArray();

            print(rowcol[0]);
            AssertArray(rowcol[0], new Int64[,] { { 0 }, { 0 }, { 1 } });

            print(rowcol[1]);
            AssertArray(rowcol[1], new Int64[,] { { 0 }, { 1 }, { 2 } });


        }


        [TestMethod]
        public void test_take_msever_1()
        {
            ndarray testVector = np.array(new System.Double[] { 1.011163, 1.01644999999999, 1.01220500000001, 1.01843699999999, 1.00985100000001, 1.018964, 1.005825, 1.016707, 8.11556899999999, 1.010744, 1.01700600000001, 1.01323099999999, 1.010389, 1.015216, 1.015418, 1.01704600000001, 1.01191, 1.01164299999999, 1.01062400000001, 1.014199, 1.012952, 1.017645, 1.01591999999999, 1.018655, 1.00942400000001, 1.012852, 1.010543, 1.02000700000001, 1.008196, 1.01396099999999 });
            ndarray testVector2 = testVector.reshape(15, 2);
            ndarray testDataMode1 = np.array(new System.Double[] { 1, 2, 2, 3, 4, 7, 9 });

            print(testVector2);
            print(testDataMode1);

            print("np.take()");
            ndarray testTake = np.take(testVector2, testDataMode1.astype(np.intp), axis: 0);
            print(testTake);

            ndarray testVector3 = np.arange(0.0, 30000.0, 0.5, dtype: np.Float64);
            ndarray testVector4 = testVector3.reshape(30000, 2);
            ndarray testIndex = np.arange(0, 30000, 100, dtype: np.intp);

            print("test BIG np.take()");

            for (int i = 0; i < 99; i++)
            {
                ndarray testBigTake = np.take(testVector4, testIndex, axis: 0);
                VerifyIndexByOffset(testIndex, 100);
                VerifyTakeByOffset(testBigTake, 100);
            }

        }

        private void VerifyTakeByOffset(ndarray testBigTake, int offset)
        {
            Int64 arrayLength = testBigTake.Size / 2;

            double LastRowCol0 = (double)testBigTake[0, 0];
            double LastRowCol1 = (double)testBigTake[0, 1];

            for (Int64 i = 1; i < arrayLength; i++)
            {
                double CurrentRowCol0 = (double)testBigTake[i, 0];
                double CurrentRowCol1 = (double)testBigTake[i, 1];

                if ((CurrentRowCol0 - LastRowCol0) != offset)
                {
                    throw new Exception("bad take");
                }
                if ((CurrentRowCol1 - LastRowCol1) != offset)
                {
                    throw new Exception("bad take");
                }

                LastRowCol0 = CurrentRowCol0;
                LastRowCol1 = CurrentRowCol1;
            }

        }

        private void VerifyIndexByOffset(ndarray testIndex, int offset)
        {
            Int64 arrayLength = testIndex.Size;

            Int64 LastIndex = (Int64)testIndex[0];
            for (Int64 i = 1; i < arrayLength; i++)
            {
                Int64 CurrentIndex = (Int64)testIndex[i];

                if ((CurrentIndex - LastIndex) != offset)
                {
                    throw new Exception("bad index");
                }

                LastIndex = CurrentIndex;
            }
        }


        [TestMethod]
        public void test_HadrianTang_1()
        {
            Assert.IsTrue(np.array_equal(new bool[0], new bool[0]));
            Assert.IsTrue(np.array_equal(new sbyte[0], new sbyte[0]));
            Assert.IsTrue(np.array_equal(new Int32[0], new Int32[0]));
            Assert.IsTrue(np.array_equal(new float[0], new float[0]));
            Assert.IsTrue(np.array_equal(new System.Numerics.Complex[0], new System.Numerics.Complex[0]));

            Assert.IsTrue(np.array_equiv(new bool[0], new bool[0]));
            Assert.IsTrue(np.array_equiv(new sbyte[0], new sbyte[0]));
            Assert.IsTrue(np.array_equiv(new Int32[0], new Int32[0]));
            Assert.IsTrue(np.array_equiv(new float[0], new float[0]));
            Assert.IsTrue(np.array_equiv(new System.Numerics.Complex[0], new System.Numerics.Complex[0]));


            return;
        }

        [TestMethod]
        public void test_HadrianTang_2()
        {
            Assert.IsTrue((bool)np.all(new bool[0]));
            Assert.IsTrue((bool)np.all(new sbyte[0]));
            Assert.IsTrue((bool)np.all(new Int32[0]));
            Assert.IsTrue((bool)np.all(new float[0]));
            Assert.IsTrue((bool)np.all(new System.Numerics.Complex[0]));

            Assert.IsFalse((bool)np.any(new bool[0]));
            Assert.IsFalse((bool)np.any(new sbyte[0]));
            Assert.IsFalse((bool)np.any(new Int32[0]));
            Assert.IsFalse((bool)np.any(new float[0]));
            Assert.IsFalse((bool)np.any(new System.Numerics.Complex[0]));


            return;
        }

        [TestMethod]
        public void test_HadrianTang_3()
        {

            var x = np.sum(np.array(new Int32[0]));

            var y = np.prod(np.array(new Int32[0]));


            return;
        }

        [TestMethod]
        public void test_HadrianTang_4()
        {

            var x = np.logical_and(new Int32[0], new Int32[0]);
            Assert.IsTrue(x.Size == 0);
            print(x);

            var y = np.logical_or(new Int32[0], new Int32[0]);
            Assert.IsTrue(y.Size == 0);
            print(x);

            return;
        }

        [TestMethod]
        public void test_HadrianTang_5()
        {
            var x = np.not_equal(new bool[0], new bool[0]);
            Assert.IsTrue(x.Size == 0);
            print(x);

            x = np.not_equal(new Int32[0], new Int32[0]);
            Assert.IsTrue(x.Size == 0);
            print(x);

            x = np.not_equal(new double[0], new double[0]);
            Assert.IsTrue(x.Size == 0);
            print(x);


            x = np.not_equal(new System.Numerics.BigInteger[0], new System.Numerics.BigInteger[0]);
            Assert.IsTrue(x.Size == 0);
            print(x);

            return;
        }

        [TestMethod]
        public void test_HadrianTang_6()
        {

            var x = np.concatenate((new Int32[0], new Int32[0]));
            Assert.IsTrue(x.Size == 0);
            print(x);

            x = np.stack(new[] { new Int32[0], new Int32[0] });
            Assert.IsTrue(x.Size == 0);
            print(x);


            return;
        }

        [TestMethod]
        public void test_HadrianTang_7()
        {

            var x = np.array(new Int32[0]);
            print(x);

            x = np.multiply(x, 5);
            print(x);

            x = x + 5;
            print(x);


            return;
        }

        [TestMethod]
        public void test_HadrianTang_8()
        {

            var x = np.array(new[,]{
                    {"0", "1"},
                    {"1", "0"},
                }).astype(np.Float64);


            AssertArray(x, new double[,] { { 0, 1 }, { 1, 0 } });
            print(x);

            return;
        }

        [TestMethod]
        public void test_HadrianTang_9()
        {

            var x = np.delete(np.array(new[,] { { "0", "1", "@" }, { "1", "0", "@" }, }), 1, 1);

            //AssertArray(x, new string[,] { { "0", "@" }, { "1", "@" } });

            print(x);

            return;
        }

        [TestMethod]
        public void test_HadrianTang_10()
        {

            var x = np.logical_and(np.array(new[] { 0, 1, 2 }), np.array(new[] { 1, 0, 2 }));
            AssertArray(x, new bool[] { false, false, true });
            print(x);

            x = np.logical_or(np.array(new[] { 1, 0, 2 }), np.array(new[] { 1, 0, 0 }));
            AssertArray(x, new bool[] { true, false, true });
            print(x);

            return;
        }

        [TestMethod]
        public void test_HadrianTang_11()
        {
            var a = (Int64)np.argmax(np.arange(5));
            Assert.AreEqual(a, 4);
            print(a);
            var b = np.arange(7)[a];
            print(b);
            Assert.AreEqual(b, 4);

            return;
        }

        [TestMethod]
        public void test_HadrianTang_12()
        {
            var a = np.array(4).shape;
            //AssertShape(a, 0);

            var index = np.array(4);
            var x = (Int32)np.arange(7)[index];
            Assert.AreEqual(x, 4);
            print(x);

            return;
        }

        [TestMethod]
        public void test_HadrianTang_13()
        {
            var objecta = 9.234;
            var objectb = 33;

            var a = np.min(objecta);
            var b = np.max(objectb);
            // var c = ndarray.this[object];

            return;
        }

        [TestMethod]
        public void test_HadrianTang_14()
        {
            ndarray a = np.array(2);
            ndarray b = np.array(2, np.Int32);

            print(a.ndim);
            print(b.ndim);

            return;
        }

        [TestMethod]
        public void test_HadrianTang_15()
        {
            var A = (ndarray)np.stack(new[] { np.arange(1, 1100) }, 1)[":", np.array(new[] { 0 })];
            AssertArray(A[1001] as ndarray, new int[1] { 1002 });

            return;
        }

        [TestMethod]
        public void test_HadrianTang_16()
        {
            AssertArray(np.delete(np.arange(16).reshape(4, 4), 1, 0), new int[,] {
                { 0, 1, 2, 3 },
                { 8, 9, 10, 11 },
                { 12, 13, 14, 15 }
            });
            AssertArray(np.delete(np.arange(16).reshape(4, 4), new Slice(1, 3), 0), new int[,] {
                { 0, 1, 2, 3 },
                { 12, 13, 14, 15 }
            });
            AssertArray(np.delete(np.arange(16).reshape(1, 16), new Slice(0, 11, 2), 1), new int[,] {
                { 1, 3, 5, 7, 9, 11, 12, 13, 14, 15 }
            });
            AssertArray(np.delete(np.arange(16).reshape(2, 2, 2, 2), 0, 1), new int[,,,] {
                { { { 4, 5 },
                    { 6, 7 } } },
                { { { 12, 13 },
                    { 14, 15 } } }
            });
            AssertArray(np.delete(np.arange(16).reshape(2, 2, 2, 2), new Slice(0, 1), 1), new int[,,,] {
                { { { 4, 5 },
                    { 6, 7 } } },
                { { { 12, 13 },
                    { 14, 15 } } }
            });
        }

        [TestMethod]
        public void test_HadrianTang_17()
        {
            var A1 = np.array(new string[,] { { null, null }, { null, null } });
            AssertArray(A1, new string[,] { { null, null }, { null, null } });

            var A2 = np.array(new object[,] { { null, null }, { null, null } });
            AssertArray(A2, new object[,] { { null, null }, { null, null } });

            var A3 = np.array(new string[,] { { null, null }, { null, null } }, dtype: np.Strings);
            AssertArray(A3, new string[,] { { null, null }, { null, null } });

            var A4 = np.array(new object[,] { { null, null }, { null, null } }, dtype: np.Object);
            AssertArray(A4, new object[,] { { null, null }, { null, null } });

        }
        [TestMethod]
        public void test_HadrianTang_18()
        {
            foreach (var x in np.array(new System.Exception[0]).Flat)
                throw new System.Exception("Failed.");
            foreach (var x in np.array(new string[0]).Flat)
                throw new System.Exception("Failed.");
            foreach (var x in np.array(new System.ConsoleKey[0]).Flat)
                throw new System.Exception("Failed.");
            foreach (var x in np.array(new int[0]).Flat)
                throw new System.Exception("Failed.");
        }

        [TestMethod]
        public void test_ChengYenTang_1()
        {
            var a = np.array(new int[] { 1, 2, 3 });

            var b = np.less(double.NegativeInfinity, a);
            AssertArray(b, new bool[] { true, true, true });
            print(b);

            b = double.NegativeInfinity < a;
            AssertArray(b, new bool[] { true, true, true });
            print(b);

            var c = a > double.NegativeInfinity;
            AssertArray(c, new bool[] { true, true, true });
            print(c);
        }

        [TestMethod]
        public void test_ChengYenTang_2()
        {
            var low = np.array(new double[2, 3] { { 30, 8, 7 }, { 2, double.NegativeInfinity, 3 } });
            var high = np.array(new double[2, 3] { { 30, 22, 10 }, { double.PositiveInfinity, 5, 3 } });

            var a = low < high;
            AssertArray(a, new bool[,] { { false, true, true }, { true, true, false } });
            print(a);

            var b = low > high;
            AssertArray(b, new bool[,] { { false, false, false }, { false, false, false } });
            print(b);

            var c = low <= high;
            AssertArray(c, new bool[,] { { true, true, true }, { true, true, true } });
            print(c);

            var d = low >= high;
            AssertArray(d, new bool[,] { { true, false, false }, { false, false, true } });
            print(d);


        }


        [TestMethod]
        public void test_ChengYenTang_3()
        {
            var a = np.arange(0, 32);
            var b = np.reshape(a, new shape(2) + new shape(16));
            print(b.shape);

            var c = np.reshape(a, new shape(2, 2) + new shape(8));
            print(c.shape);

            var d = np.reshape(a, new shape(2, 2) + new shape(2, 4));
            print(d.shape);

            var e = np.reshape(a, new shape(new int[] { 2, 2 }, new int[] { 2, 4 }));
            print(e.shape);

            var f = np.reshape(a, new shape(new long[] { 2, 2 }, new long[] { 2, 4 }));
            print(f.shape);

            var g = np.reshape(a, new shape(new int[] { 2, 2 }, new int[] { 2, 4 }, new int[] { 1, 1 }));
            print(g.shape);

            var h = np.reshape(a, new shape(new long[] { 2, 2 }, new long[] { 2, 4 }, new long[] { 1, 1 }));
            print(h.shape);


        }


        [TestMethod]
        public void test_ChengYenTang_4()
        {
            ndarray low = np.array(new double[2, 3] { { 9, 8, 7 }, { 2, double.NegativeInfinity, 1 } });
            print(low);

            ndarray stack_low = np.repeat(low, 3, axis: 0);
            print(stack_low);

            ndarray observation = np.array(new double[2, 2, 3] { { { 9, 8, 7 }, { 2, double.NegativeInfinity, 1 } }, { { 30, 22, 10 }, { double.PositiveInfinity, 5, 3 } } });
            print(observation);

            ndarray stackedobs = np.zeros(new shape(2) + stack_low.shape);
            print(stackedobs);


            stackedobs[":", $"-{observation.shape[1]}:", "..."] = observation;
            print(stackedobs);

        }

        [TestMethod]
        public void test_ChengYenTang_5()
        {
            var a = np.array(new int[,] { { 0 }, { 0 }, { 0 } });
            print(a.shape);
            print(a.shape[-1]);

            Assert.AreEqual(1, a.shape[-1]);
            Assert.AreEqual(3, a.shape[-2]);
        }

        [TestMethod]
        public void test_ChengYenTang_6()
        {
            var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            print(a.shape);

            var b = (ndarray)a[0, "..."];
            AssertArray(b, new int[] { 1, 2 });
            print(b);

            var c = (ndarray)a[0, "...", ":-1"];
            AssertArray(c, new int[] { 1 });
            print(c);

            var d = (ndarray)a[0, "...", new Slice(0, -1, 1)];
            AssertArray(d, new int[] { 1 });
            print(d);

        }

        [TestMethod]
        public void test_ChengYenTang_7()
        {
            var stackedobs = np.arange(0, 3 * 2 * 2 * 4).reshape(3, 2, 2, 4);

            var ExpectedData = new int[,,,] { { { { 2, 3 }, { 6, 7 } },
                                            { { 10, 11 }, { 14, 15 } } },
                                            { { { 18, 19 }, { 22, 23 } },
                                            { { 26, 27 }, { 30, 31 } } },
                                          { { { 34, 35 }, { 38, 39 } },
                                            { { 42, 43 }, { 46, 47 } } } };


            var A = (ndarray)stackedobs["...", "-2:"];
            AssertArray(A, ExpectedData);
            print("A");
            print(A);

            var A1 = (ndarray)stackedobs["...", new Slice(-2, null, null)];
            AssertArray(A1, ExpectedData);
            print("A1");
            print(A1);

            var B = (ndarray)stackedobs["...", 1, "-2:"];
            AssertArray(B, new int[,,] { { { 6, 7 }, { 14, 15 } }, { { 22, 23 }, { 30, 31 } }, { { 38, 39 }, { 46, 47 } } });
            print("B");
            print(B);

            var C = (ndarray)stackedobs[":", ":", ":", "-2:"];
            AssertArray(C, ExpectedData);
            print("C");
            print(C);


            bool GotException = false;
            try
            {
                var expectException = stackedobs["...", "...", "...", "-3:"];
                GotException = false;
            }
            catch (Exception ex)
            {
                GotException = true;
            }
            Assert.IsTrue(GotException);

        }

        [TestMethod]
        public void test_ChengYenTang_8()
        {
            var A = np.arange(0, 3 * 2 * 2 * 4).reshape(3, 2, 2, 4);
            var i = A.shape["1:"];
            AssertArray(np.array(i.ToArray()), new npy_intp[] { 2, 2, 4 });

            var j = A.shape["1:2"];
            AssertArray(np.array(j.ToArray()), new npy_intp[] { 2 });

            var k = A.shape["0:2"];
            AssertArray(np.array(k.ToArray()), new npy_intp[] { 3, 2 });

            var l = A.shape["1::2"];
            AssertArray(np.array(l.ToArray()), new npy_intp[] { 2, 4 });

            var m = A.shape[":"];
            AssertArray(np.array(m.ToArray()), new npy_intp[] { 3, 2, 2, 4 });

            var n = A.shape["::"];
            AssertArray(np.array(n.ToArray()), new npy_intp[] { 3, 2, 2, 4 });


        }

        [TestMethod]
        public void test_ChengYenTang_8A()
        {
            var A = np.arange(0, 3 * 2 * 2 * 4).reshape(3, 2, 2, 4);
            var i = A.shape[new Slice(1)];
            AssertArray(np.array(i.ToArray()), new npy_intp[] { 2, 2, 4 });

            var j = A.shape[new Slice(1, 2)];
            AssertArray(np.array(j.ToArray()), new npy_intp[] { 2 });

            var k = A.shape[new Slice(0, 2)];
            AssertArray(np.array(k.ToArray()), new npy_intp[] { 3, 2 });

            var l = A.shape[new Slice(1, null, 2)];
            AssertArray(np.array(l.ToArray()), new npy_intp[] { 2, 4 });

            var m = A.shape[new Slice(null, null)];
            AssertArray(np.array(m.ToArray()), new npy_intp[] { 3, 2, 2, 4 });

            var n = A.shape[new Slice(null, null)];
            AssertArray(np.array(n.ToArray()), new npy_intp[] { 3, 2, 2, 4 });


        }

        [TestMethod]
        public void test_SimonCraenen_1()
        {
            ndarray a = np.arange(0, 4, dtype: np.Float32);
            var abytes = a.tobytes(NumpyLib.NPY_ORDER.NPY_CORDER);

            ndarray b = np.frombuffer(abytes, dtype: np.Float32);
            AssertArray(b, new float[] { 0, 1, 2, 3 });

            /////////////
            a = np.arange(0, 4, dtype: np.Float64);
            abytes = a.tobytes(NumpyLib.NPY_ORDER.NPY_CORDER);

            b = np.frombuffer(abytes, dtype: np.Float64);
            AssertArray(b, new double[] { 0, 1, 2, 3 });

            /////////////
            a = np.arange(0, 4, dtype: np.Int8);
            abytes = a.tobytes(NumpyLib.NPY_ORDER.NPY_CORDER);

            b = np.frombuffer(abytes, dtype: np.Int8);
            AssertArray(b, new sbyte[] { 0, 1, 2, 3 });

            /////////////
            a = np.arange(0, 8, dtype: np.UInt16).reshape(4, 2);
            abytes = a.tobytes(NumpyLib.NPY_ORDER.NPY_CORDER);

            b = np.frombuffer(abytes, dtype: np.UInt16);
            AssertArray(b, new UInt16[] { 0, 1, 2, 3, 4, 5, 6, 7 });

            /////////////
            a = np.arange(0, 8, dtype: np.UInt32).reshape(4, 2);
            abytes = a.tobytes(NumpyLib.NPY_ORDER.NPY_CORDER);

            b = np.frombuffer(abytes, dtype: np.UInt32);
            AssertArray(b, new UInt32[] { 0, 1, 2, 3, 4, 5, 6, 7 });

            /////////////
            a = np.arange(0, 8, dtype: np.Int64).reshape(4, 2);
            abytes = a.tobytes(NumpyLib.NPY_ORDER.NPY_CORDER);

            b = np.frombuffer(abytes, dtype: np.Int64);
            AssertArray(b, new Int64[] { 0, 1, 2, 3, 4, 5, 6, 7 });

            b = np.frombuffer(abytes, dtype: np.Int64, count: 32, offset: 8);
            AssertArray(b, new Int64[] { 1, 2, 3, 4 });

            /////////////
            a = np.arange(0, 8, dtype: np.Float64).reshape(4, 2);
            abytes = a.tobytes(NumpyLib.NPY_ORDER.NPY_CORDER);

            b = np.frombuffer(abytes, dtype: np.Float64);
            AssertArray(b, new double[] { 0, 1, 2, 3, 4, 5, 6, 7 });

            b = np.frombuffer(abytes, dtype: null, count: 32, offset: 8);
            AssertArray(b, new double[] { 1, 2, 3, 4 });

            /////////////
            a = np.arange(0, 8, dtype: np.Decimal).reshape(4, 2);
            abytes = a.tobytes(NumpyLib.NPY_ORDER.NPY_CORDER);

            bool GotException = false;
            try
            {
                b = np.frombuffer(abytes, dtype: np.Decimal);
            }
            catch (Exception ex)
            {
                GotException = true;
            }
            Assert.IsTrue(GotException);


            return;
        }

        [TestMethod]
        public void test_Rainyl_1a()
        {
            var arr = np.array(new Int32[] { -9, 7, 5, 3, 1, -1, -3, -5, -11, 13, 17, 21 }).reshape(4, 3);
            var idx2 = np.argsort(arr);
            print(arr);
            print(idx2);

            var x = arr.T[0, ":"] as ndarray;
            var y = arr.T[1, ":"] as ndarray;

            var idx = np.argsort(x);
            AssertArray(idx, new npy_intp[] { 0, 2, 1, 3 });
            print(x);
            print(idx);

            idx = np.argsort(y);
            AssertArray(idx, new npy_intp[] { 2, 1, 0, 3 });
            print(y);
            print(idx);
        }

        [TestMethod]
        public void test_Rainyl_1b()
        {
            var arr = np.array(new Int16[] { -9, 7, 5, 3, 1, -1, -3, -5, -11, 13, 17, 21 }).reshape(4, 3);
            var idx2 = np.argsort(arr);
            print(arr);
            print(idx2);

            var x = arr.T[0, ":"] as ndarray;
            var y = arr.T[1, ":"] as ndarray;

            var idx = np.argsort(x);
            AssertArray(idx, new npy_intp[] { 0, 2, 1, 3 });
            print(x);
            print(idx);

            idx = np.argsort(y);
            AssertArray(idx, new npy_intp[] { 2, 1, 0, 3 });
            print(y);
            print(idx);
        }


        [TestMethod]
        public void test_Rainyl_1c()
        {
            var arr = np.array(new Int64[] { -9, 7, 5, 3, 1, -1, -3, -5, -11, 13, 17, 21 }).reshape(4, 3);
            var idx2 = np.argsort(arr);
            print(arr);
            print(idx2);

            var x = arr.T[0, ":"] as ndarray;
            var y = arr.T[1, ":"] as ndarray;

            var idx = np.argsort(x);
            AssertArray(idx, new npy_intp[] { 0, 2, 1, 3 });
            print(x);
            print(idx);

            idx = np.argsort(y);
            AssertArray(idx, new npy_intp[] { 2, 1, 0, 3 });
            print(y);
            print(idx);
        }

        [TestMethod]
        public void test_Rainyl_1d()
        {
            var arr = np.array(new BigInteger[] { -9, 7, 5, 3, 1, -1, -3, -5, -11, 13, 17, 21 }).reshape(4, 3);
            var idx2 = np.argsort(arr);
            print(arr);
            print(idx2);

            var x = arr.T[0, ":"] as ndarray;
            var y = arr.T[1, ":"] as ndarray;

            var idx = np.argsort(x);
            AssertArray(idx, new npy_intp[] { 0, 2, 1, 3 });
            print(x);
            print(idx);

            idx = np.argsort(y);
            AssertArray(idx, new npy_intp[] { 2, 1, 0, 3 });
            print(y);
            print(idx);
        }

        [TestMethod]
        public void test_Rainyl_2a()
        {
            var arr = np.array(new Int32[] { 7, -9, -5, -3, 1, -1, 33, 5, -11, 13, 17, -21 }).reshape(4, 3);
            var idx2 = np.argsort(arr);
            print(arr);
            print(idx2);

            var x = arr.T[":", 0] as ndarray;
            var y = arr.T[":", 1] as ndarray;

            var idx = np.argsort(x);
            AssertArray(idx, new npy_intp[] { 1, 2, 0 });
            print(x);
            print(idx);

            idx = np.argsort(y);
            AssertArray(idx, new npy_intp[] { 0, 2, 1 });
            print(y);
            print(idx);
        }

        [TestMethod]
        public void test_Rainyl_2b()
        {
            var arr = np.array(new Int16[] { 7, -9, -5, -3, 1, -1, 33, 5, -11, 13, 17, -21 }).reshape(4, 3);
            var idx2 = np.argsort(arr);
            print(arr);
            print(idx2);

            var x = arr.T[":", 0] as ndarray;
            var y = arr.T[":", 1] as ndarray;

            var idx = np.argsort(x);
            AssertArray(idx, new npy_intp[] { 1, 2, 0 });
            print(x);
            print(idx);

            idx = np.argsort(y);
            AssertArray(idx, new npy_intp[] { 0, 2, 1 });
            print(y);
            print(idx);
        }

        [TestMethod]
        public void test_Rainyl_2c()
        {
            var arr = np.array(new Int64[] { 7, -9, -5, -3, 1, -1, 33, 5, -11, 13, 17, -21 }).reshape(4, 3);
            var idx2 = np.argsort(arr);
            print(arr);
            print(idx2);

            var x = arr.T[":", 0] as ndarray;
            var y = arr.T[":", 1] as ndarray;

            var idx = np.argsort(x);
            AssertArray(idx, new npy_intp[] { 1, 2, 0 });
            print(x);
            print(idx);

            idx = np.argsort(y);
            AssertArray(idx, new npy_intp[] { 0, 2, 1 });
            print(y);
            print(idx);
        }

        [TestMethod]
        public void test_Rainyl_2d()
        {
            var arr = np.array(new Complex[] { 7, -9, -5, -3, 1, -1, 33, 5, -11, 13, 17, -21 }).reshape(4, 3);
            var idx2 = np.argsort(arr);
            print(arr);
            print(idx2);

            var x = arr.T[":", 0] as ndarray;
            var y = arr.T[":", 1] as ndarray;

            var idx = np.argsort(x);
            AssertArray(idx, new npy_intp[] { 1, 2, 0 });
            print(x);
            print(idx);

            idx = np.argsort(y);
            AssertArray(idx, new npy_intp[] { 0, 2, 1 });
            print(y);
            print(idx);
        }

        [TestMethod]
        public void test_Rainyl_3()
        {
            var a = np.asarray(new int[,] { { 1, 2, 3, 4, 5 }, { 10, 11, 12, 13, 14 } });
            //a = a.flatten();
            var b = a[":", "::-1"] as ndarray;
     
            string Line1Output = string.Format($"output: {b[0, 0]}, {b[0, 1]}, {b[0, 2]}, {b[0, 3]}, {b[0, 4]}");
            Assert.AreEqual(0, string.Compare("output: 5, 4, 3, 2, 1", Line1Output));

            string Line2Output = string.Format($"output: {b[1, 0]}, {b[1, 1]}, {b[1, 2]}, {b[1, 3]}, {b[1, 4]}");
            Assert.AreEqual(0, string.Compare("output: 14, 13, 12, 11, 10", Line2Output));

        }

        [TestMethod]
        public void test_Rainyl_3a()
        {
            var a = np.asarray(new int[,] { { 1, 2, 3, 4, 5 }, { 10, 11, 12, 13, 14 } });
            //a = a.flatten();
            var b = a["::-1", "::-1"] as ndarray;



            string Line1Output = string.Format($"output: {b[0, 0]}, {b[0, 1]}, {b[0, 2]}, {b[0, 3]}, {b[0, 4]}");
            Assert.AreEqual(0, string.Compare("output: 14, 13, 12, 11, 10", Line1Output));

            string Line2Output = string.Format($"output: {b[1, 0]}, {b[1, 1]}, {b[1, 2]}, {b[1, 3]}, {b[1, 4]}");
            Assert.AreEqual(0, string.Compare("output: 5, 4, 3, 2, 1", Line2Output));

        }

        [TestMethod]
        public void test_Rainyl_3b()
        {
            var a = np.asarray(new int[,] { { 1, 2, 3, 4, 5 }, { 10, 11, 12, 13, 14 } });
            //a = a.flatten();
            var b = a["::-2", "::-2"] as ndarray;
            print(b);


            string Line1Output = string.Format($"output: {b[0, 0]}, {b[0, 1]}, {b[0, 2]}");
            print(Line1Output);
            Assert.AreEqual(0, string.Compare("output: 14, 12, 10", Line1Output));

            b[0, 0] = 88;
            b[0, 1] = 77;
            b[0, 2] = 66;

            string Line2Output = string.Format($"output: {b[0, 0]}, {b[0, 1]}, {b[0, 2]}");
            print(Line2Output);
            Assert.AreEqual(0, string.Compare("output: 88, 77, 66", Line2Output));

            AssertArray(a, new int[,] { { 1, 2, 3, 4, 5 }, { 66, 11, 77, 13, 88 } });

            return;

        }

    }
}
