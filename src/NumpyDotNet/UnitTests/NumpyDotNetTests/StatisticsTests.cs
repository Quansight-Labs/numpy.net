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

        #endregion

        #region percentile/quantile

        [Ignore]
        [TestMethod]
        public void test_percentile_1()
        {

        }

        [Ignore]
        [TestMethod]
        public void test_quantile_1()
        {

        }

        #endregion

        #region nanpercentile/nanquantile
        // see NANFunctionsTests
        #endregion

        #region median/average/mean


        [Ignore] // need to implement Nanfunctions
        [TestMethod]
        public void test_median_placeholder()
        {
            // see the NANFunctionsTest version
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

        #endregion

        #region nanmedian/nanmean
        // see NANFunctionsTests
        #endregion

        #region nanstd/nanvar
        // see NANFunctionsTests
        #endregion

        #region Correlating
        [Ignore]
        [TestMethod]
        public void test_corrcoef_1()
        {

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


        [Ignore]
        [TestMethod]
        public void test_cov_1()
        {

        }

        #endregion

        #region Histograms

        [Ignore]
        [TestMethod]
        public void test_histogram_1()
        {

        }

        [Ignore]
        [TestMethod]
        public void test_histogram2d_1()
        {

        }

        [Ignore]
        [TestMethod]
        public void test_histogramdd_1()
        {

        }

        [Ignore]
        [TestMethod]
        public void test_bincount_1()
        {

        }

        [Ignore]
        [TestMethod]
        public void histogram_bin_edges_1()
        {

        }

        [Ignore]
        [TestMethod]
        public void digitize_1()
        {

        }

        #endregion

    }
}
