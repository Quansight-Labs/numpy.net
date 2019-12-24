using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumpyLib;

namespace NumpyDotNetTests 
{
    [TestClass]
    public class StatisticsTests : TestBaseClass
    {
        #region amin/amax
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
            AssertArray(c, new Int32[] { 1, 2 });
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

            ndarray c = np.amin(a, axis: 0);  // Minimum along the first axis
            print(c);
            AssertArray(c, new float[] { 30.25f, 31.25f, 32.25f, 33.25f });
            print("*****");

            ndarray d = np.amin(a, axis: 1);   // Minimum along the second axis
            print(d);
            AssertArray(d, new float[] { 30.25f, 34.25f, 38.25f, 42.25f });
            print("*****");

            ndarray e = np.arange(5, dtype: np.Float32);
            e[2] = np.NaN;
            ndarray f = np.amin(e);
            print(f);
            Assert.AreEqual(np.NaN, f.GetItem(0));
            print("*****");

        }

        [TestMethod]
        public void test_amin_2_DECIMAL()
        {
            ndarray a = np.arange(30.25m, 46.25m).reshape(new shape(4, 4));
            print(a);
            print("*****");

            ndarray b = np.amin(a);          // Minimum of the flattened array
            print(b);
            Assert.AreEqual(30.25m, b.GetItem(0));
            print("*****");

            ndarray c = np.amin(a, axis: 0);  // Minimum along the first axis
            print(c);
            AssertArray(c, new decimal[] { 30.25m, 31.25m, 32.25m, 33.25m });
            print("*****");

            ndarray d = np.amin(a, axis: 1);   // Minimum along the second axis
            print(d);
            AssertArray(d, new decimal[] { 30.25m, 34.25m, 38.25m, 42.25m });
            print("*****");


            // decimals don't support NAN
            //ndarray e = np.arange(5, dtype: np.Decimal);
            //e[2] = np.NaN;
            //ndarray f = np.amin(e);
            //print(f);
            //Assert.AreEqual(np.NaN, f.GetItem(0));
            //print("*****");

     

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
        public void test_amax_2_DECIMAL()
        {
            ndarray a = np.arange(30.25m, 46.25m).reshape(new shape(4, 4));
            print(a);
            print("*****");

            ndarray b = np.amax(a);          // Maximum of the flattened array
            print(b);
            Assert.AreEqual(45.25m, b.GetItem(0));
            print("*****");

            ndarray c = np.amax(a, axis: 0);  // Maxima along the first axis
            print(c);
            AssertArray(c, new decimal[] { 42.25m, 43.25m, 44.25m, 45.25m });
            print("*****");

            ndarray d = np.amax(a, axis: 1);   // Maxima along the second axis
            print(d);
            AssertArray(d, new decimal[] { 33.25m, 37.25m, 41.25m, 45.25m });
            print("*****");

            // decimals don't support NAN
            //ndarray e = np.arange(5, dtype: np.Float32);
            //e[2] = np.NaN;
            //ndarray f = np.amax(e);
            //print(f);
            //Assert.AreEqual(np.NaN, f.GetItem(0));
            //print("*****");

            //ndarray g = np.nanmax(b);
            //print(g);
        }

        #endregion

        #region nanmin/nanmax
        // see NANFunctionsTests
        #endregion

        #region ptp
        [TestMethod]
        public void test_ptp_1()
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
        public void test_ptp_1_DECIMAL()
        {
            ndarray a = np.arange(4, dtype: np.Decimal).reshape(new shape(2, 2));
            print(a);
            print("*****");

            ndarray b = np.ptp(a, axis: 0);
            print(b);
            AssertArray(b, new decimal[] { 2, 2 });
            print("*****");

            ndarray c = np.ptp(a, axis: 1);
            print(c);
            AssertArray(c, new decimal[] { 1, 1 });

            ndarray d = np.ptp(a);
            print(d);
            Assert.AreEqual(3m, d.GetItem(0));
        }

        #endregion

        #region percentile/quantile

        [TestMethod]
        public void test_percentile_1()
        {
            var a = np.array(new double[,] { { 10, 7, 4 }, { 3, 2, 1 } });

            var b = np.percentile(a, 50);
            Assert.AreEqual((double)3.5, b.GetItem(0));
            print(b);

            var c = np.percentile(a, 50, axis: 0);
            AssertArray(c, new double[] { 6.5, 4.5, 2.5 });
            print(c);

            var d = np.percentile(a, 50, axis : 1);
            AssertArray(d, new double[] {7.0, 2.0});
            print(d);

            var e = np.percentile(a, 50, axis : 1, keepdims : true);
            AssertArray(e, new double[,] { { 7.0 }, { 2.0 } });
            print(e);

            // note: we dont support the out parameter
            //var m = np.percentile(a, 50, axis : 0);
            //var n = np.zeros_like(m);
            //var o = np.percentile(a, 50, axis : 0);
            //print(o);
            //print(n);
            // note: we don't support the overwrite_input flag
            //b = a.Copy();
            //c = np.percentile(b, 50, axis: 1, overwrite_input: true);
            //print(c);

            //Assert.IsFalse((bool)np.all(a.Equals(b)).GetItem(0));

            return;
        }

        [TestMethod]
        public void test_percentile_2()
        {
            var a = np.array(new int[,] { { 10, 7, 4 }, { 3, 2, 1 } });

            var b = np.percentile(a, new int[] { 50, 75 });
            AssertArray(b, new double[] { 3.5, 6.25  });
            print(b);

            var c = np.percentile(a, new int[] { 50, 75 }, axis: 0);
            AssertArray(c, new double[,] { { 6.5, 4.5, 2.5 }, { 8.25, 5.75, 3.25 } });
            print(c);

            var d = np.percentile(a, new int[] { 50, 75 }, axis: 1);
            AssertArray(d, new double[,] { { 7.0, 2.0 }, { 8.5, 2.5 } });
            print(d);

            var e = np.percentile(a, new int[] { 50, 75 }, axis: 1, keepdims: true);
            AssertArray(e, new double[,,] { { { 7.0 }, { 2.0 } }, { { 8.5 }, { 2.5 } } });
            print(e);

            // note: we dont support the out parameter
            //var m = np.percentile(a, 50, axis : 0);
            //var n = np.zeros_like(m);
            //var o = np.percentile(a, 50, axis : 0);
            //print(o);
            //print(n);
            // note: we don't support the overwrite_input flag
            //b = a.Copy();
            //c = np.percentile(b, 50, axis: 1, overwrite_input: true);
            //print(c);

            //Assert.IsFalse((bool)np.all(a.Equals(b)).GetItem(0));

            return;
        }

        [TestMethod]
        public void test_percentile_2_DECIMAL()
        {
            var a = np.array(new decimal[,] { { 10, 7, 4 }, { 3, 2, 1 } });

            var b = np.percentile(a, new decimal[] { 50, 75 });
            AssertArray(b, new decimal[] { 3.5m, 6.25m });
            print(b);

            var c = np.percentile(a, new decimal[] { 50, 75 }, axis: 0);
            AssertArray(c, new decimal[,] { { 6.5m, 4.5m, 2.5m }, { 8.25m, 5.75m, 3.25m } });
            print(c);

            var d = np.percentile(a, new decimal[] { 50, 75 }, axis: 1);
            AssertArray(d, new decimal[,] { { 7.0m, 2.0m }, { 8.5m, 2.5m } });
            print(d);

            var e = np.percentile(a, new decimal[] { 50, 75 }, axis: 1, keepdims: true);
            AssertArray(e, new decimal[,,] { { { 7.0m }, { 2.0m } }, { { 8.5m }, { 2.5m } } });
            print(e);

            // note: we dont support the out parameter
            //var m = np.percentile(a, 50, axis : 0);
            //var n = np.zeros_like(m);
            //var o = np.percentile(a, 50, axis : 0);
            //print(o);
            //print(n);
            // note: we don't support the overwrite_input flag
            //b = a.Copy();
            //c = np.percentile(b, 50, axis: 1, overwrite_input: true);
            //print(c);

            //Assert.IsFalse((bool)np.all(a.Equals(b)).GetItem(0));

            return;
        }

        [TestMethod]
        public void test_quantile_1()
        {
            var a = np.array(new int[,] { { 10, 7, 4 }, { 3, 2, 1 } });

            var b = np.quantile(a, 0.5);
            Assert.AreEqual((double)3.5, b.GetItem(0));
            print(b);

            var c = np.quantile(a, 0.5, axis: 0);
            AssertArray(c, new double[] { 6.5, 4.5, 2.5 });
            print(c);

            var d = np.quantile(a, 0.5, axis: 1);
            AssertArray(d, new double[] { 7.0, 2.0 });
            print(d);

            var e = np.quantile(a, 0.5, axis: 1, keepdims: true);
            AssertArray(e, new double[,] { { 7.0 }, { 2.0 } });
            print(e);

            // note: we dont support the out parameter

            //var m = np.quantile(a, 0.5, axis: 0);
            //var n = np.zeros_like(m);
            //var o = np.quantile(a, 0.5, axis: 0);
            //print(o);
            //print(n);
            // note: we don't support the overwrite_input flag
            //b = a.Copy();
            //c = np.quantile(b, 0.5, axis: 1, overwrite_input: true);
            //print(c);

            //Assert.IsFalse((bool)np.all(a.Equals(b)).GetItem(0));

            return;
        }

        [TestMethod]
        public void test_quantile_2()
        {
            var a = np.array(new int[,] { { 10, 7, 4 }, { 3, 2, 1 } });

            var b = np.quantile(a, new double[] { 0.5, 0.75 });
            AssertArray(b, new double[] { 3.5, 6.25 });
            print(b);

            var c = np.quantile(a, new double[] { 0.5, 0.75 }, axis: 0);
            AssertArray(c, new double[,] { { 6.5, 4.5, 2.5 }, { 8.25, 5.75, 3.25 } });
            print(c);

            var d = np.quantile(a, new double[] { 0.5, 0.75 }, axis: 1);
            AssertArray(d, new double[,] { { 7.0, 2.0 }, { 8.5, 2.5 } });
            print(d);

            var e = np.quantile(a, new double[] { 0.5, 0.75 }, axis: 1, keepdims: true);
            AssertArray(e, new double[,,] { { { 7.0 }, { 2.0 } }, { { 8.5 }, { 2.5 } } });
            print(e);

            // note: we dont support the out parameter

            //var m = np.quantile(a, 0.5, axis: 0);
            //var n = np.zeros_like(m);
            //var o = np.quantile(a, 0.5, axis: 0);
            //print(o);
            //print(n);
            // note: we don't support the overwrite_input flag
            //b = a.Copy();
            //c = np.quantile(b, 0.5, axis: 1, overwrite_input: true);
            //print(c);

            //Assert.IsFalse((bool)np.all(a.Equals(b)).GetItem(0));

            return;
        }

        [TestMethod]
        public void test_quantile_2_DECIMAL()
        {
            var a = np.array(new decimal[,] { { 10, 7, 4 }, { 3, 2, 1 } });

            var b = np.quantile(a, new decimal[] { 0.5m, 0.75m });
            AssertArray(b, new decimal[] { 3.5m, 6.25m });
            print(b);

            var c = np.quantile(a, new decimal[] { 0.5m, 0.75m }, axis: 0);
            AssertArray(c, new decimal[,] { { 6.5m, 4.5m, 2.5m }, { 8.25m, 5.75m, 3.25m } });
            print(c);

            var d = np.quantile(a, new decimal[] { 0.5m, 0.75m }, axis: 1);
            AssertArray(d, new decimal[,] { { 7.0m, 2.0m }, { 8.5m, 2.5m } });
            print(d);

            var e = np.quantile(a, new decimal[] { 0.5m, 0.75m }, axis: 1, keepdims: true);
            AssertArray(e, new decimal[,,] { { { 7.0m }, { 2.0m } }, { { 8.5m }, { 2.5m } } });
            print(e);

            // note: we dont support the out parameter

            //var m = np.quantile(a, 0.5, axis: 0);
            //var n = np.zeros_like(m);
            //var o = np.quantile(a, 0.5, axis: 0);
            //print(o);
            //print(n);
            // note: we don't support the overwrite_input flag
            //b = a.Copy();
            //c = np.quantile(b, 0.5, axis: 1, overwrite_input: true);
            //print(c);

            //Assert.IsFalse((bool)np.all(a.Equals(b)).GetItem(0));

            return;
        }

        #endregion

        #region nanpercentile/nanquantile
        // see NANFunctionsTests
        #endregion

        #region median/average/mean


        [TestMethod]
        public void test_median_1()
        {
            var a = np.array(new int[,] { { 10, 7, 4 }, { 3, 2, 1 } });

            var b = np.median(a);
            Assert.AreEqual((double)3.5, b.GetItem(0));
            print(b);

            var c = np.median(a, axis: 0);
            AssertArray(c, new double[] { 6.5, 4.5, 2.5 });
            print(c);

            var d = np.median(a, axis: 1);
            AssertArray(d, new double[] { 7.0, 2.0 });
            print(d);

            var e = np.median(a, axis: 1, keepdims: true);
            AssertArray(e, new double[,] { { 7.0 }, { 2.0 } });
            print(e);

            // note: we dont support the out parameter

            //var m = np.median(a, 0.5, axis: 0);
            //var n = np.zeros_like(m);
            //var o = np.median(a, 0.5, axis: 0);
            //print(o);
            //print(n);
            // note: we don't support the overwrite_input flag
            //b = a.Copy();
            //c = np.median(b, 0.5, axis: 1, overwrite_input: true);
            //print(c);

            //Assert.IsFalse((bool)np.all(a.Equals(b)).GetItem(0));

            return;
        }

        [TestMethod]
        public void test_median_2()
        {
            var a = np.arange(0, 64, 1).reshape((4, 4, 4));

            var b = np.median(a, axis : new int[] { 0, 2 }, keepdims: true);
            AssertArray(b, new double[,,] { { { 25.5 }, { 29.5 }, { 33.5 }, { 37.5 } } });
            print(b);

            var c = np.median(a, new int[] { 0, 1 }, keepdims: true);
            AssertArray(c, new double[,,] { { { 30, 31, 32, 33} } });
            print(c);

            var d = np.median(a, new int[] { 1, 2 }, keepdims: true);
            AssertArray(d, new double[,,] { { { 7.5 } }, { { 23.5 } }, { { 39.5 } }, { { 55.5 } } });
            print(d);

            return;
        }

        [TestMethod]
        public void test_median_2_DECIMAL()
        {
            var a = np.arange(0, 64, 1, np.Decimal).reshape((4, 4, 4));

            var b = np.median(a, axis: new int[] { 0, 2 }, keepdims: true);
            AssertArray(b, new decimal[,,] { { { 25.5m }, { 29.5m }, { 33.5m }, { 37.5m } } });
            print(b);

            var c = np.median(a, new int[] { 0, 1 }, keepdims: true);
            AssertArray(c, new decimal[,,] { { { 30, 31, 32, 33 } } });
            print(c);

            var d = np.median(a, new int[] { 1, 2 }, keepdims: true);
            AssertArray(d, new decimal[,,] { { { 7.5m } }, { { 23.5m } }, { { 39.5m } }, { { 55.5m } } });
            print(d);

            return;
        }

        [TestMethod]
        public void test_median_3()
        {
            var a = np.array(new double[,] { { 10.0, 7.2, 4.2 }, { 3.2, 2.2, 1.2 } });

            var b = np.median(a);
            Assert.AreEqual((double)3.7, b.GetItem(0));
            print(b);

            var c = np.median(a, axis: 0);
            AssertArray(c, new double[] { 6.6, 4.7, 2.7 });
            print(c);

            var d = np.median(a, axis: 1);
            AssertArray(d, new double[] { 7.2, 2.2 });
            print(d);

            var e = np.median(a, axis: 1, keepdims: true);
            AssertArray(e, new double[,] { { 7.2 }, { 2.2 } });
            print(e);

            // note: we dont support the out parameter

            //var m = np.median(a, 0.5, axis: 0);
            //var n = np.zeros_like(m);
            //var o = np.median(a, 0.5, axis: 0);
            //print(o);
            //print(n);
            // note: we don't support the overwrite_input flag
            //b = a.Copy();
            //c = np.median(b, 0.5, axis: 1, overwrite_input: true);
            //print(c);

            //Assert.IsFalse((bool)np.all(a.Equals(b)).GetItem(0));

            return;
        }


        [TestMethod]
        public void test_average_1()
        {
            UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));
            x = x * 3;
            var y = np.average(x);

            print(x);
            print(y);

            Assert.AreEqual(131.5, y.GetItem(0));

        }

        [TestMethod]
        public void test_average_2()
        {

            var a = np.array(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            var w = new int[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

            var x = np.average(a);
            Assert.AreEqual(5.5, x.GetItem(0));
            print(x);
            print("********");

            x = np.average(a, weights : w);
            Assert.AreEqual(4.0, x.GetItem(0));
            print(x);
            print("********");

            x = np.average(a.reshape((2,-1)), weights: np.array(w).reshape((2, -1)));
            Assert.AreEqual(4.0, x.GetItem(0));
            print(x);
            print("********");

            x = np.average(a.reshape((2, -1)), axis:0, weights: np.array(w).reshape((2, -1)));
            AssertArray(x, new double[] { 2.66666666666667, 3.53846153846154, 4.36363636363636, 5.11111111111111, 5.71428571428571 });
            print(x);
            print("********");

            x = np.average(a.reshape((2, -1)), axis: 1, weights: np.array(w).reshape((2, -1)));
            AssertArray(x, new double[] { 2.75, 7.33333333333333 });
            print(x);
            print("********");

            x = np.average(a.reshape((1, 2, -1, 1)), axis: 1, weights: np.array(w).reshape((1, 2, -1, 1)));
            AssertArray(x, new double[,,] { { { 2.66666666666667 }, { 3.53846153846154 }, { 4.36363636363636 }, { 5.11111111111111 }, {5.71428571428571} } });
            print(x);
            print("********");

            x = np.average(a.reshape((1, -1, 2, 1)), axis: 1, weights: np.array(w).reshape((1, -1, 2, 1)));
            AssertArray(x, new double[,,] { { { 3.66666666666667 }, { 4.4 } } });
            print(x);
            print("********");

            x = np.average(a.reshape((2, -1, 1, 1)), axis: 1, weights: np.array(w).reshape((2, -1, 1, 1)));
            AssertArray(x, new double[,,] { { { 2.75 } }, { { 7.33333333333333 } } });
            print(x);

        }

        [TestMethod]
        public void test_average_3()
        {

            var a = np.array(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            var w = new int[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

            var x1 = np.average(a, axis:null, weights: null, returned: true);
            Assert.AreEqual(5.5, x1.retval.GetItem(0));
            Assert.AreEqual((double)10.0, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a, axis:null, weights: w, returned:true);
            Assert.AreEqual(4.0, x1.retval.GetItem(0));
            Assert.AreEqual((double)55.0, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis:null, weights: np.array(w).reshape((2, -1)), returned: true);
            Assert.AreEqual(4.0, x1.retval.GetItem(0));
            Assert.AreEqual((double)55.0, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: 0, weights: np.array(w).reshape((2, -1)), returned: true);
            AssertArray(x1.retval, new double[] { 2.66666666666667, 3.53846153846154, 4.36363636363636, 5.11111111111111, 5.71428571428571 });
            AssertArray(x1.sum_of_weights, new double[] { 15.0, 13.0, 11.0, 9.0, 7.0 });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: 1, weights: np.array(w).reshape((2, -1)), returned: true);
            AssertArray(x1.retval, new double[] { 2.75, 7.33333333333333 });
            AssertArray(x1.sum_of_weights, new double[] { 40.0, 15.0 });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((1, 2, -1, 1)), axis: 1, weights: np.array(w).reshape((1, 2, -1, 1)), returned: true);
            AssertArray(x1.retval, new double[,,] { { { 2.66666666666667 }, { 3.53846153846154 }, { 4.36363636363636 }, { 5.11111111111111 }, { 5.71428571428571 } } });
            AssertArray(x1.sum_of_weights, new double[,,] { { { 15.0 }, { 13.0 }, { 11.0 }, { 9.0 }, { 7.0 } } });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((1, -1, 2, 1)), axis: 1, weights: np.array(w).reshape((1, -1, 2, 1)), returned : true);
            AssertArray(x1.retval, new double[,,] { { { 3.66666666666667 }, { 4.4 } } });
            AssertArray(x1.sum_of_weights, new double[,,] { { { 30.0 }, { 25.0 } } });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1, 1, 1)), axis: 1, weights: np.array(w).reshape((2, -1, 1, 1)), returned: false);
            AssertArray(x1.retval, new double[,,] { { { 2.75 } }, { { 7.33333333333333 } } });
            Assert.AreEqual(null, x1.sum_of_weights);
            print(x1);

        }

        [TestMethod]
        public void test_average_3_DECIMAL()
        {

            var a = np.array(new decimal[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            var w = new int[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

            var x1 = np.average(a, axis: null, weights: null, returned: true);
            Assert.AreEqual(5.5m, x1.retval.GetItem(0));
            Assert.AreEqual(10.0m, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a, axis: null, weights: w, returned: true);
            Assert.AreEqual(4.0m, x1.retval.GetItem(0));
            Assert.AreEqual(55.0m, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: null, weights: np.array(w).reshape((2, -1)), returned: true);
            Assert.AreEqual(4.0m, x1.retval.GetItem(0));
            Assert.AreEqual(55.0m, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: 0, weights: np.array(w).reshape((2, -1)), returned: true);
            AssertArray(x1.retval, new decimal[] { 2.6666666666666666666666666667m, 3.5384615384615384615384615385m, 4.3636363636363636363636363636m,
                                                   5.1111111111111111111111111111m, 5.7142857142857142857142857143m });
            AssertArray(x1.sum_of_weights, new decimal[] { 15.0m, 13.0m, 11.0m, 9.0m, 7.0m });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: 1, weights: np.array(w).reshape((2, -1)), returned: true);
            AssertArray(x1.retval, new decimal[] { 2.75m, 7.3333333333333333333333333333m });
            AssertArray(x1.sum_of_weights, new decimal[] { 40.0m, 15.0m });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((1, 2, -1, 1)), axis: 1, weights: np.array(w).reshape((1, 2, -1, 1)), returned: true);
            AssertArray(x1.retval, new decimal[,,] { { { 2.6666666666666666666666666667m }, { 3.5384615384615384615384615385m }, { 4.3636363636363636363636363636m }, 
                                                      { 5.1111111111111111111111111111m }, { 5.7142857142857142857142857143m } } });
            AssertArray(x1.sum_of_weights, new decimal[,,] { { { 15.0m }, { 13.0m }, { 11.0m }, { 9.0m }, { 7.0m } } });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((1, -1, 2, 1)), axis: 1, weights: np.array(w).reshape((1, -1, 2, 1)), returned: true);
            AssertArray(x1.retval, new decimal[,,] { { { 3.6666666666666666666666666667m }, { 4.4m } } });
            AssertArray(x1.sum_of_weights, new decimal[,,] { { { 30.0m }, { 25.0m } } });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1, 1, 1)), axis: 1, weights: np.array(w).reshape((2, -1, 1, 1)), returned: false);
            AssertArray(x1.retval, new decimal[,,] { { { 2.75m } }, { { 7.3333333333333333333333333333m } } });
            Assert.AreEqual(null, x1.sum_of_weights);
            print(x1);

        }



        [TestMethod]
        public void test_mean_1()
        {
            UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));
            x = x * 3;
            print(x);

            var y = np.mean(x);
            print(y);
            Assert.AreEqual(131.5, y.GetItem(0));

            y = np.mean(x, axis: 0);
            print(y);
            AssertArray(y, new double[,] { { 113, 150 }, { 113, 150 } });

            y = np.mean(x, axis: 1);
            print(y);
            AssertArray(y, new double[,] { { 52.5, 90 }, { 132, 157.5 }, { 154.5, 202.5 } });

            y = np.mean(x, axis: 2);
            print(y);
            AssertArray(y, new double[,] { { 37.5, 105 }, { 252, 37.5 }, { 105, 252 } });

        }

        [TestMethod]
        public void test_mean_1_DECIMAL()
        {
            decimal[] TestData = new decimal[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.Decimal).reshape(new shape(3, 2, -1));
            x = x * 3;
            print(x);

            var y = np.mean(x);
            print(y);
            Assert.AreEqual(131.5m, y.GetItem(0));

            y = np.mean(x, axis: 0);
            print(y);
            AssertArray(y, new decimal[,] { { 113, 150 }, { 113, 150 } });

            y = np.mean(x, axis: 1);
            print(y);
            AssertArray(y, new decimal[,] { { 52.5m, 90 }, { 132, 157.5m }, { 154.5m, 202.5m } });

            y = np.mean(x, axis: 2);
            print(y);
            AssertArray(y, new decimal[,] { { 37.5m, 105 }, { 252, 37.5m }, { 105, 252 } });

        }

        [TestMethod]
        public void test_mean_2()
        {
            ndarray a = np.zeros(new shape(2, 512 * 512), dtype: np.Float32);
            a[0, ":"] = 1.0;
            a[1, ":"] = 0.1;
            ndarray b = np.mean(a);
            print(b);
            Assert.AreEqual(0.546875f, (double)b.GetItem(0), 0.0000001);

            ndarray c = np.mean(a, dtype: np.Float64);
            print(c);
            Assert.AreEqual(0.5500000007450581, c.GetItem(0));
        }

        [TestMethod]
        public void test_mean_2_DECIMAL()
        {
            ndarray a = np.zeros(new shape(2, 512 * 512), dtype: np.Decimal);
            a[0, ":"] = 1.0;
            a[1, ":"] = 0.1;
            ndarray b = np.mean(a);
            print(b);
            Assert.AreEqual(0.55m, (decimal)b.GetItem(0));

            ndarray c = np.mean(a, dtype: np.Decimal);
            print(c);
            Assert.AreEqual(0.55m, c.GetItem(0));
        }


        [TestMethod]
        public void test_mean_3()
        {
            ndarray x = np.array(new int[] { 1, 2, 2, 2, 1 });
            var mean = x.Mean<int>();

            ndarray y = x.A("1:3");
            mean = y.Mean<int>();


        }



        #endregion

        #region std/var

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
            AssertArray(d, new double[] { 1.1180339887498949, 1.1180339887498949 }); // NOTES: TODO: slightly different than python. keepdims issue

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

        [TestMethod]
        public void test_std_1_DECIMAL()
        {
            ndarray a = np.array(new decimal[,] { { 1, 2 }, { 3, 4 } });
            ndarray b = np.std(a);
            print(b);
            Assert.AreEqual(1.1180339887498948482045868344m, (decimal)b.GetItem(0));

            ndarray c = np.std(a, axis: 0);
            print(c);
            AssertArray(c, new decimal[] { 1.0m, 1.0m });

            ndarray d = np.std(a, axis: 1);
            print(d);
            AssertArray(d, new decimal[] { 1.1180339887498948482045868344m, 1.1180339887498948482045868344m }); // NOTES: TODO: slightly different than python. keepdims issue

            // In single precision, std() can be inaccurate:
            a = np.zeros(new shape(2, 512 * 512), dtype: np.Decimal);
            a[0, ":"] = 1.0;
            a[1, ":"] = 0.1;
            b = np.std(a);
            print(b);
            Assert.AreEqual(0.45m, b.GetItem(0));
            // Computing the standard deviation in float64 is more accurate:
            c = np.std(a, dtype: np.Decimal);
            print(c);
            Assert.AreEqual(0.45m, c.GetItem(0));

        }

        [TestMethod]
        public void test_var_1()
        {
            ndarray a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            ndarray b = np.var(a);
            Assert.AreEqual(1.25, b.GetItem(0));
            print(b);

            ndarray c = np.var(a, axis: 0);
            AssertArray(c, new double[] { 1.0, 1.0 });
            print(c);

            ndarray d = np.var(a, axis: 1);
            AssertArray(d, new double[] { 1.25, 1.25 }); // NOTES: TODO: slightly different than python. keepdims issue
            print(d);

            // In single precision, std() can be inaccurate:
            a = np.zeros(new shape(2, 512 * 512), dtype: np.Float32);
            a[0, ":"] = 1.0;
            a[1, ":"] = 0.1;
            b = np.var(a);
            Assert.AreEqual((double)0.202509764960269, Convert.ToDouble(b.GetItem(0)), 0.00000001);
            print(b);

            // Computing the standard deviation in float64 is more accurate:
            c = np.var(a, dtype: np.Float64);
            Assert.AreEqual((double)0.202499999329974, Convert.ToDouble(c.GetItem(0)), 0.00000001);
            print(c);

        }

        [TestMethod]
        public void test_var_1_DECIMAL()
        {
            ndarray a = np.array(new decimal[,] { { 1, 2 }, { 3, 4 } });
            ndarray b = np.var(a);
            Assert.AreEqual(1.25m, b.GetItem(0));
            print(b);

            ndarray c = np.var(a, axis: 0);
            AssertArray(c, new decimal[] { 1.0m, 1.0m });
            print(c);

            ndarray d = np.var(a, axis: 1);
            AssertArray(d, new decimal[] { 1.25m, 1.25m }); // NOTES: TODO: slightly different than python. keepdims issue
            print(d);

            // In single precision, std() can be inaccurate:
            a = np.zeros(new shape(2, 512 * 512), dtype: np.Decimal);
            a[0, ":"] = 1.0m;
            a[1, ":"] = 0.1m;
            b = np.var(a);
            Assert.AreEqual(0.2025m, b.GetItem(0));
            print(b);

            // Computing the standard deviation in float64 is more accurate:
            c = np.var(a, dtype: np.Decimal);
            Assert.AreEqual(0.2025m, c.GetItem(0));
            print(c);

        }

        #endregion

        #region nanmedian/nanmean
        // see NANFunctionsTests
        #endregion

        #region nanstd/nanvar
        // see NANFunctionsTests
        #endregion

        #region Correlating
        [TestMethod]
        public void test_corrcoef_1()
        {
            var x1 = np.array(new int[,] { { 0, 2 }, { 1, 1 }, { 2, 0 } }).T;
            print(x1);

            // Note how  increases while  decreases. The covariance matrix shows this clearly:

            var a = np.corrcoef(x1);
            AssertArray(a, new double[,] { { 1, -1 }, { -1, 1 } });
            print(a);

            var x = new double[] { -2.1, -1, 4.3 };
            var y = new double[] { 3, 1.1, 0.12 };
            var X = np.stack(new object[] { x, y }, axis: 0);
            a = np.corrcoef(X);
            AssertArray(a, new double[,] { { 1.0, -0.855357809522795 }, { -0.855357809522795, 1.0 } });
            print(a);


            var b = np.corrcoef(x, y);
            AssertArray(b, new double[,] { { 1.0, -0.855357809522795 }, { -0.855357809522795, 1.0 } });
            print(b);

            var c = np.corrcoef(x, y, rowvar:false);
            AssertArray(a, new double[,] { { 1.0, -0.855357809522795 }, { -0.855357809522795, 1.0 } });
            print(c);
    

            return;
        }

        [TestMethod]
        public void test_corrcoef_1_DECIMAL()
        {
            var x1 = np.array(new decimal[,] { { 0, 2 }, { 1, 1 }, { 2, 0 } }).T;
            print(x1);

            // Note how  increases while  decreases. The covariance matrix shows this clearly:

            var a = np.corrcoef(x1);
            AssertArray(a, new decimal[,] { { 1, -1 }, { -1, 1 } });
            print(a);

            var x = new decimal[] { -2.1m, -1, 4.3m };
            var y = new decimal[] { 3, 1.1m, 0.12m };
            var X = np.stack(new object[] { x, y }, axis: 0);
            a = np.corrcoef(X);
            AssertArray(a, new decimal[,] { { 1.0m, -0.8553578095227944904571128856m }, { -0.8553578095227944904571128856m, 1.0m } });
            print(a);


            var b = np.corrcoef(x, y);
            AssertArray(b, new decimal[,] { { 1.0m, -0.8553578095227944904571128856m }, { -0.8553578095227944904571128856m, 1.0m } });
            print(b);

            var c = np.corrcoef(x, y, rowvar: false);
            AssertArray(a, new decimal[,] { { 1.0m, -0.8553578095227944904571128856m }, { -0.8553578095227944904571128856m, 1.0m } });
            print(c);


            return;
        }



        [TestMethod]
        public void test_correlate_1()
        {
            var a = np.correlate(new int[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f });
            AssertArray(a, new double[] { 3.5 });
            print(a);

            var b = np.correlate(new int[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_SAME);
            AssertArray(b, new double[] { 2.0, 3.5, 3.0 });
            print(b);

            var c = np.correlate(new int[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_FULL);
            AssertArray(c, new double[] { 0.5, 2.0, 3.5, 3.0, 0.0 });
            print(c);

            return;
        }

        [TestMethod]
        public void test_correlate_1_DECIMAL()
        {
            var a = np.correlate(new decimal[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f });
            AssertArray(a, new decimal[] { 3.5m });
            print(a);

            var b = np.correlate(new decimal[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_SAME);
            AssertArray(b, new decimal[] { 2.0m, 3.5m, 3.0m });
            print(b);

            var c = np.correlate(new decimal[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_FULL);
            AssertArray(c, new decimal[] { 0.5m, 2.0m, 3.5m, 3.0m, 0.0m });
            print(c);

            return;
        }


        [TestMethod]
        public void test_cov_1()
        {
            var x1 = np.array(new int[,] { { 0, 2 }, { 1, 1 }, { 2, 0 } }).T;
            print(x1);

            // Note how  increases while  decreases. The covariance matrix shows this clearly:

            var a = np.cov(x1);
            AssertArray(a, new double[,] { {1,-1 }, {-1, 1 } });
            print(a);

            var x = new double[] { -2.1, -1, 4.3 };
            var y = new double[] { 3, 1.1, 0.12 };
            var X = np.stack(new object[] { x, y }, axis: 0);
            a = np.cov(X);
            AssertArray(a, new double[,] { { 11.71, -4.286 }, { -4.286, 2.14413333333333 } });
            print(a);


            var b = np.cov(x, y);
            AssertArray(b, new double[,] { { 11.71, -4.286 }, { -4.286, 2.14413333333333 } });
            print(b);

            var c = np.cov(x);
            Assert.AreEqual((double)11.709999999999999, c.GetItem(0));
            print(c);

            var d = np.cov(X, rowvar: false);
            AssertArray(d, new double[,] { { 13.005, 5.355, -10.659 }, { 5.355, 2.205, -4.389 }, { -10.659, -4.389, 8.7362 } });
            print(d);

            var e = np.cov(X, rowvar: false, bias: true);
            AssertArray(e, new double[,] { { 6.5025, 2.6775, -5.3295 }, { 2.6775, 1.1025, -2.1945 }, { -5.3295, -2.1945, 4.3681 } });
            print(e);

            var f = np.cov(X, rowvar: false, bias: true, fweights: new int[] { 1, 2 });
            AssertArray(f, new double[,] { { 5.78, 2.38, -4.73733333333333 }, { 2.38, 0.98, -1.95066666666667 }, { -4.73733333333333, -1.95066666666667, 3.88275555555555 } });
            print(f);

            var g = np.cov(X, rowvar: false, bias: true, fweights: new int[] { 1, 2 }, aweights: new int[] { 1, 2 });
            AssertArray(g, new double[,] { { 4.1616, 1.7136, -3.41088 }, { 1.7136, 0.7056, -1.40448 }, { -3.41088, -1.40448, 2.795584 } });
            print(g);

            return;
        }


        [TestMethod]
        public void test_cov_1_DECIMAL()
        {
            var x1 = np.array(new decimal[,] { { 0, 2 }, { 1, 1 }, { 2, 0 } }).T;
            print(x1);

            // Note how  increases while  decreases. The covariance matrix shows this clearly:

            var a = np.cov(x1);
            AssertArray(a, new decimal[,] { { 1, -1 }, { -1, 1 } });
            print(a);

            var x = new decimal[] { -2.1m, -1, 4.3m };
            var y = new decimal[] { 3, 1.1m, 0.12m };
            var X = np.stack(new object[] { x, y }, axis: 0);
            a = np.cov(X);
            AssertArray(a, new decimal[,] { { 11.710m, -4.2860000000000000000000000000m }, { -4.2860000000000000000000000000m, 2.1441333333333333333333333334m } });
            print(a);


            var b = np.cov(x, y);
            AssertArray(b, new decimal[,] { { 11.710m, -4.2860000000000000000000000000m }, { -4.2860000000000000000000000000m, 2.1441333333333333333333333334m } });
            print(b);

            var c = np.cov(x);
            Assert.AreEqual(11.710m, c.GetItem(0));
            print(c);

            var d = np.cov(X, rowvar: false);
            AssertArray(d, new decimal[,] { { 13.00500m, 5.35500m, -10.65900m }, { 5.35500m, 2.20500m, -4.38900m }, { -10.65900m, -4.38900m, 8.73620m } });
            print(d);

            var e = np.cov(X, rowvar: false, bias: true);
            AssertArray(e, new decimal[,] { { 6.50250m, 2.67750m, -5.32950m }, { 2.67750m, 1.10250m, -2.19450m }, { -5.32950m, -2.19450m, 4.36810m } });
            print(e);

            var f = np.cov(X, rowvar: false, bias: true, fweights: new int[] { 1, 2 });
            AssertArray(f, new decimal[,] { { 5.7799999999999999999999999994m, 2.3799999999999999999999999998m, -4.7373333333333333333333333329m }, 
                                            { 2.3799999999999999999999999998m, 0.9799999999999999999999999999m, -1.9506666666666666666666666665m }, 
                                            { -4.7373333333333333333333333329m, -1.9506666666666666666666666665m, 3.8827555555555555555555555553m } });
            print(f);

            var g = np.cov(X, rowvar: false, bias: true, fweights: new int[] { 1, 2 }, aweights: new int[] { 1, 2 });
            AssertArray(g, new decimal[,] { { 4.16160m, 1.71360m, -3.410880m }, { 1.71360m, 0.70560m, -1.404480m }, { -3.410880m, -1.404480m, 2.7955840m } });
            print(g);

            return;
        }


        #endregion

        #region Histograms

        // see HistogramTests.cs

        #endregion

    }
}
