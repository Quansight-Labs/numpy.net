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

    }
}
