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
    public class HistogramTests : TestBaseClass
    {
        #region bincount
        [TestMethod]
        public void test_bincount_1()
        {
            var x = np.arange(5);
            var a = np.bincount(x);
            AssertArray(a, new npy_intp[] {1,1,1,1,1 });
            print(a);

            x = np.array(new int[] { 0, 1, 1, 3, 2, 1, 7 });
            a = np.bincount(x);
            AssertArray(a, new npy_intp[] { 1, 3, 1, 1, 0, 0, 0, 1 });
            print(a);

            x = np.array(new int[] { 0, 1, 1, 3, 2, 1, 7, 23 });
            a = np.bincount(x);
            AssertArray(a, new npy_intp[] { 1, 3, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 });
            print(a);
            Assert.IsTrue(a.size == (int)np.amax(x) + 1);

        }

        [TestMethod]
        public void test_bincount_2()
        {
            var x = np.arange(5, dtype: np.Int64);
            var a = np.bincount(x);
            AssertArray(a, new npy_intp[] { 1, 1, 1, 1, 1 });
            print(a);

            x = np.array(new Int16[] { 0, 1, 1, 3, 2, 1, 7 }, dtype: np.Int16);
            a = np.bincount(x);
            AssertArray(a, new npy_intp[] { 1, 3, 1, 1, 0, 0, 0, 1 });
            print(a);

            x = np.array(new sbyte[] { 0, 1, 1, 3, 2, 1, 7, 23 }, dtype: np.Int8);
            a = np.bincount(x);
            AssertArray(a, new npy_intp[] { 1, 3, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 });
            print(a);
            Assert.IsTrue(a.size == (sbyte)np.amax(x) + 1);

        }

        [TestMethod]
        public void test_bincount_3()
        {

            var w = np.array(new double[] { 0.3, 0.5, 0.2, 0.7, 1.0, -0.6 });  // weights

            var x = np.arange(6, dtype: np.Int64);
            var a = np.bincount(x, weights: w);
            AssertArray(a, new double[] { 0.3, 0.5, 0.2, 0.7, 1.0, -0.6 });
            print(a);

            x = np.array(new Int16[] { 0, 1, 3, 2, 1, 7 }, dtype: np.Int16);
            a = np.bincount(x, weights: w);
            AssertArray(a, new double[] { 0.3, 1.5, 0.7, 0.2, 0.0, 0.0, 0.0, -0.6 });
            print(a);

            x = np.array(new sbyte[] { 0, 1, 3, 2, 1, 7 }, dtype: np.Int8);
            a = np.bincount(x, weights: w);
            AssertArray(a, new double[] { 0.3, 1.5, 0.7, 0.2, 0.0, 0.0, 0.0, -0.6 });
            print(a);

        }

        [TestMethod]
        public void test_bincount_4()
        {
            var x = np.arange(5, dtype: np.Int64);
            var a = np.bincount(x, minlength: 8);
            AssertArray(a, new npy_intp[] { 1, 1, 1, 1, 1, 0, 0, 0 });
            print(a);

            x = np.array(new Int16[] { 0, 1, 1, 3, 2, 1, 7 }, dtype: np.Int16);
            a = np.bincount(x, minlength: 10);
            AssertArray(a, new npy_intp[] { 1, 3, 1, 1, 0, 0, 0, 1, 0, 0 });
            print(a);

            x = np.array(new sbyte[] { 0, 1, 1, 3, 2, 1, 7, 23 }, dtype: np.Int8);
            a = np.bincount(x, minlength: 32);
            AssertArray(a, new npy_intp[] { 1, 3, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 });
            print(a);
            print(a.size == (sbyte)np.amax(x) + 1);

        }

        [TestMethod]
        public void test_bincount_slice()
        {

            var w = np.array(new double[] { 0.3, 0.5, 0.2, 0.7, 1.0, -0.6, .19, -0.8, 0.3, 0.5 });  // weights

            var x = np.arange(10, dtype: np.Int64);
            var a = np.bincount(x["::2"], weights: w["::2"]);
            AssertArray(a, new double[] { 0.3, 0.0, 0.2, 0.0, 1.0, 0.0, 0.19, 0.0, 0.3 });
            print(a);

       
        }

        // python does not support unsigned integers.  That seems weird.
        // I see no reason to not support them so I will.  Easy to change if necessary.
        [TestMethod]
        public void test_bincount_uint64()
        {

            
            var x = np.arange(5, dtype: np.UInt64);
            var a = np.bincount(x);
            AssertArray(a, new npy_intp[] { 1, 1, 1, 1, 1 });
            print(a);

            x = np.array(new UInt32[] { 0, 1, 1, 3, 2, 1, 7 });
            a = np.bincount(x);
            AssertArray(a, new npy_intp[] { 1, 3, 1, 1, 0, 0, 0, 1 });
            print(a);

            x = np.array(new byte[] { 0, 1, 1, 3, 2, 1, 7, 23 });
            a = np.bincount(x);
            AssertArray(a, new npy_intp[] { 1, 3, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 });
            print(a);
            Assert.IsTrue(a.size == (byte)np.amax(x) + 1);

        }

        [TestMethod]
        public void test_bincount_double()
        {
            try
            {
                var x = np.arange(5, dtype: np.Float64);
                var a = np.bincount(x);
                print(a);
                Assert.Fail("This should have thrown an exception");
            }
            catch (Exception ex)
            {
                print(ex.Message);
            }

        }
        [TestMethod]
        public void test_bincount_not1d()
        {
            try
            {
                var x = np.arange(100, dtype: np.Int64).reshape(10,10);
                var a = np.bincount(x);
                print(a);
                Assert.Fail("This should have thrown an exception");
            }
            catch (Exception ex)
            {
                print(ex.Message);
            }

        }
        #endregion

        #region digitize
        [TestMethod]
        public void test_digitize_1()
        {
            var x = np.array(new double[] { 0.2, 6.4, 3.0, 1.6 });
            var bins = np.array(new double[] { 0.0, 1.0, 2.5, 4.0, 10.0 });
            var inds = np.digitize(x, bins);
            AssertArray(inds, new npy_intp[] { 1, 4, 3, 2 });
            print(inds);

        }

        [TestMethod]
        public void test_digitize_2()
        {
            var x = np.array(new double[] { 1.2, 10.0, 12.4, 15.5, 20.0});
            var bins = np.array(new Int32[] { 0, 5, 10, 15, 20 });

            var inds = np.digitize(x, bins, right: true);
            AssertArray(inds, new npy_intp[] { 1, 2, 3, 4, 4 });
            print(inds);

            inds = np.digitize(x, bins, right: false);
            AssertArray(inds, new npy_intp[] { 1, 3, 3, 4, 5 });
            print(inds);

        }

        [TestMethod]
        public void test_digitize_3()
        {
            var x = np.array(new double[] { 1.2, 10.0, 12.4, 15.5, 20.0 });
            var bins = np.array(new Int32[] { 20, 15, 10, 5, 0 });

            var inds = np.digitize(x, bins, right: true);
            AssertArray(inds, new npy_intp[] { 4, 3, 2, 1, 1 });
            print(inds);

            inds = np.digitize(x, bins, right: false);
            AssertArray(inds, new npy_intp[] { 4, 2, 2, 1, 0 });
            print(inds);

        }

        #endregion

        #region Histogram

        [TestMethod]
        public void test_histogram_1()
        {
            var x = np.histogram(new int[] { 1, 2, 1 }, bins: new int[] { 0, 1, 2, 3 });
            AssertArray(x.hist, new npy_intp[] { 0, 2, 1 });
            AssertArray(x.bin_edges, new Int32[] { 0, 1, 2, 3 });
            print(x);

            x = np.histogram(np.arange(4), bins: np.arange(5), density: true);
            AssertArray(x.hist, new double[] { 0.25, 0.25, 0.25, 0.25 });
            AssertArray(x.bin_edges, new Int32[] { 0, 1, 2, 3, 4 });
            print(x);

            x = np.histogram(np.arange(4), bins: np.arange(5), density: false);
            AssertArray(x.hist, new npy_intp[] { 1, 1, 1, 1 });
            AssertArray(x.bin_edges, new Int32[] { 0, 1, 2, 3, 4 });
            print(x);

            x = np.histogram(new int[,] { { 1, 2, 1 }, { 1, 0, 1 } }, bins: new int[] { 0, 1, 2, 3 });
            AssertArray(x.hist, new npy_intp[] { 1, 4, 1 });
            AssertArray(x.bin_edges, new Int32[] { 0, 1, 2, 3 });
            print(x);

        }

        [TestMethod]
        public void test_histogram_1a()
        {
            var x = np.histogram(new float[] { 1, 2, 1 }, bins: new double[] { 0, 1, 2, 3 });
            AssertArray(x.hist, new npy_intp[] { 0, 2, 1 });
            AssertArray(x.bin_edges, new double[] { 0, 1, 2, 3 });
            print(x);

            x = np.histogram(np.arange(4, dtype: np.Float32), bins: np.arange(5, dtype: np.Float64), density: true);
            AssertArray(x.hist, new double[] { 0.25, 0.25, 0.25, 0.25 });
            AssertArray(x.bin_edges, new double[] { 0, 1, 2, 3, 4 });
            print(x);

            x = np.histogram(np.arange(4, dtype: np.Int16), bins: np.arange(5, dtype: np.Float64), density: false);
            AssertArray(x.hist, new npy_intp[] { 1, 1, 1, 1 });
            AssertArray(x.bin_edges, new double[] { 0, 1, 2, 3, 4 });
            print(x);

            x = np.histogram(new int[,] { { 1, 2, 1 }, { 1, 0, 1 } }, bins: new int[] { 0, 1, 2, 3 });
            AssertArray(x.hist, new npy_intp[] { 1, 4, 1 });
            AssertArray(x.bin_edges, new Int32[] { 0, 1, 2, 3 });
            print(x);

        }

        [TestMethod]
        public void test_histogram_2()
        {
            var x = np.histogram(new int[] { 1, 2, 1 }, bins: 4);
            AssertArray(x.hist, new npy_intp[] { 2, 0, 0, 1 });
            AssertArray(x.bin_edges, new double[] { 1.0, 1.25, 1.5, 1.75, 2.0 });
            print(x);

            x = np.histogram(np.arange(4), bins: 5, density: true);
            AssertArray(x.hist, new double[] { 0.416666650109822, 0.416666650109822, 0.0, 0.416666650109822, 0.416666650109822 });
            AssertArray(x.bin_edges, new double[] { 0.0, 0.6, 1.2, 1.8, 2.4, 3.0 });
            print(x);

            x = np.histogram(np.arange(4), bins: 5, density: false);
            AssertArray(x.hist, new npy_intp[] { 1, 1, 0, 1, 1 });
            AssertArray(x.bin_edges, new double[] { 0.0, 0.6, 1.2, 1.8, 2.4, 3.0 });
            print(x);

            x = np.histogram(new int[,] { { 1, 2, 1 }, { 1, 0, 1 } }, bins: 8);
            AssertArray(x.hist, new npy_intp[] { 1, 0, 0, 0, 4, 0, 0, 1 });
            AssertArray(x.bin_edges, new double[] { 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0 });
            print(x);

        }

        [TestMethod]
        public void test_histogram_3()
        {
            var x = np.histogram(new int[] { 1, 2, 1 }, bins: Histogram_BinSelector.auto);
            AssertArray(x.hist, new npy_intp[] { 2, 0, 1 });
            AssertArray(x.bin_edges, new double[] { 1.0, 1.33333333333333, 1.66666666666667, 2.0 });
            print(x);

            x = np.histogram(np.arange(4), bins: Histogram_BinSelector.doane, density: true);
            AssertArray(x.hist, new double[] { 0.25, 0.25, 0.5 });
            AssertArray(x.bin_edges, new double[] { 0.0, 1.0, 2.0, 3.0 });
            print(x);

            x = np.histogram(np.arange(4), bins: Histogram_BinSelector.fd, density: false);
            AssertArray(x.hist, new npy_intp[] { 2,2 });
            AssertArray(x.bin_edges, new double[] { 0.0, 1.5, 3.0 });
            print(x);

            x = np.histogram(new int[,] { { 1, 2, 1 }, { 1, 0, 1 } }, bins: Histogram_BinSelector.rice);
            AssertArray(x.hist, new npy_intp[] { 1, 0, 4, 1 });
            AssertArray(x.bin_edges, new double[] { 0.0, 0.5, 1.0, 1.5, 2.0 });
            print(x);

            x = np.histogram(np.arange(4), bins: Histogram_BinSelector.scott, density: true);
            AssertArray(x.hist, new double[] { 0.333333333333333, 0.333333333333333});
            AssertArray(x.bin_edges, new double[] { 0.0, 1.5, 3.0 });
            print(x);

            x = np.histogram(np.arange(4), bins: Histogram_BinSelector.sqrt, density: false);
            AssertArray(x.hist, new npy_intp[] { 2, 2 });
            AssertArray(x.bin_edges, new double[] { 0.0, 1.5, 3.0 });
            print(x);

            x = np.histogram(new int[,] { { 1, 2, 1 }, { 1, 0, 1 } }, bins: Histogram_BinSelector.sturges);
            AssertArray(x.hist, new npy_intp[] { 1, 0, 4, 1 });
            AssertArray(x.bin_edges, new double[] {  0.0, 0.5, 1.0, 1.5, 2.0 });
            print(x);

        }


        [TestMethod]
        public void test_histogram_4()
        {

            var x = np.histogram(np.arange(40), bins: np.arange(5), range: (15.0f, 30.0f), density: true);
            AssertArray(x.hist, new double[] { 0.2, 0.2, 0.2, 0.4 });
            AssertArray(x.bin_edges, new Int32[] { 0, 1, 2, 3, 4 });
            print(x);

            x = np.histogram(np.arange(40), bins: np.arange(4), range: (15.0f, 30.0f), density: false);
            AssertArray(x.hist, new npy_intp[] { 1, 1, 2 });
            AssertArray(x.bin_edges, new Int32[] { 0, 1, 2, 3 });
            print(x);

            x = np.histogram(np.arange(40), bins: 6, range: (15.0f, 30.0f), density: true);
            AssertArray(x.hist, new double[] { 0.075, 0.05, 0.075, 0.05, 0.075, 0.075 });
            AssertArray(x.bin_edges, new double[] { 15, 17.5, 20, 22.5, 25, 27.5, 30 });
            print(x);

            x = np.histogram(np.arange(40), bins: 4, range: (15.0f, 30.0f), density: false);
            AssertArray(x.hist, new npy_intp[] { 4, 4, 4, 4 });
            AssertArray(x.bin_edges, new double[] { 15, 18.75, 22.5, 26.25, 30 });
            print(x);

        }

        [TestMethod]
        public void test_histogram_5()
        {
            var weights = np.arange(40, dtype : np.Float64);
            weights.fill(0.5);

            var x = np.histogram(np.arange(40), bins: np.arange(5), range: (15.0f, 30.0f), weights: weights, density: true);
            AssertArray(x.hist, new double[] { 0.2, 0.2, 0.2, 0.4 });
            AssertArray(x.bin_edges, new Int32[] { 0, 1, 2, 3, 4 });
            print(x);

            x = np.histogram(np.arange(40), bins: np.arange(4), range: (15.0f, 30.0f), weights: weights, density: false);
            AssertArray(x.hist, new double[] { 0.5, 0.5, 1.0});
            AssertArray(x.bin_edges, new Int32[] { 0, 1, 2, 3 });
            print(x);

            x = np.histogram(np.arange(40), bins: 6, range: (15.0f, 30.0f), weights: weights, density: true);
            AssertArray(x.hist, new double[] { 0.075, 0.05, 0.075, 0.05, 0.075, 0.075 });
            AssertArray(x.bin_edges, new double[] { 15, 17.5, 20, 22.5, 25, 27.5, 30 });
            print(x);

            x = np.histogram(np.arange(40), bins: 4, range: (15.0f, 30.0f), weights: weights, density: false);
            AssertArray(x.hist, new double[] { 2, 2, 2, 2 });
            AssertArray(x.bin_edges, new double[] { 15, 18.75, 22.5, 26.25, 30 });
            print(x);

        }


        #endregion

        #region histogramdd

        //[Ignore]
        [TestMethod]
        public void test_histogramdd_1()
        {
            var random = new np.random();
            random.seed(8765);

            var r = random.randint(10, 30, new shape(3000));


            var x = np.histogramdd(r.reshape(-1, 4), bins : new int[] { 2, 2, 2, 2 });
            var ExpectedHist = new double[,,,]
            {{{{48.0, 48.0},{38.0, 58.0}},{{50.0, 45.0},{46.0, 49.0}}},
             {{{44.0, 40.0},{44.0, 50.0}},{{43.0, 36.0},{60.0, 51.0}}}};
            AssertArray(x.hist, ExpectedHist);
            AssertArray(x.bin_edges[0], new double[] { 10.0, 19.5, 29 });
            AssertArray(x.bin_edges[1], new double[] { 10.0, 19.5, 29 });
            AssertArray(x.bin_edges[2], new double[] { 10.0, 19.5, 29 });
            AssertArray(x.bin_edges[3], new double[] { 10.0, 19.5, 29 });
            print(x.hist);
            print(x.bin_edges);

            /////
            x = np.histogramdd(r.reshape(-1, 4), bins: new double[] { 2, 2, 2, 2 });
            ExpectedHist = new double[,,,]
            {{{{48.0, 48.0},{38.0, 58.0}},{{50.0, 45.0},{46.0, 49.0}}},
             {{{44.0, 40.0},{44.0, 50.0}},{{43.0, 36.0},{60.0, 51.0}}}};
            AssertArray(x.hist, ExpectedHist);
            AssertArray(x.bin_edges[0], new double[] { 10.0, 19.5, 29 });
            AssertArray(x.bin_edges[1], new double[] { 10.0, 19.5, 29 });
            AssertArray(x.bin_edges[2], new double[] { 10.0, 19.5, 29 });
            AssertArray(x.bin_edges[3], new double[] { 10.0, 19.5, 29 });
            print(x.hist);
            print(x.bin_edges);

            /////
            x = np.histogramdd(r.reshape(-1, 2), bins: new Int32[] { 3,3 }, density:true);
            var ExpectedHist2 = new double[,]
            { { 0.00307479224376731, 0.00259279778393352, 0.00282548476454294 },
              { 0.00245983379501385, 0.00237673130193906, 0.00280886426592798 },
              { 0.00292520775623269, 0.00250969529085873, 0.0033573407202216 } };
            AssertArray(x.hist, ExpectedHist2);
            AssertArray(x.bin_edges[0], new double[] { 10.0, 16.3333333333333, 22.6666666666667, 29.0 });
            AssertArray(x.bin_edges[1], new double[] { 10.0, 16.3333333333333, 22.6666666666667, 29.0 });
            print(x.hist);
            print(x.bin_edges);

            x = np.histogramdd(r.reshape(-1, 3), bins: new Int32[] { 4, 4, 4 }, density: false);
            var ExpectedHist3 = new double[,,]
            {
                {{ 19.0, 21.0, 19.0, 15.0 },{ 22.0, 11.0, 11.0, 10.0 },{ 17.0, 13.0, 7.0, 12.0 }, { 12.0, 11.0, 7.0, 23.0 }},
                {{ 12.0, 13.0, 16.0, 14.0 },{ 12.0, 14.0, 15.0, 20.0 },{ 17.0, 16.0, 19.0, 12.0 },{ 13.0, 13.0, 17.0, 16.0 }},
                {{ 23.0, 22.0, 9.0, 13.0 }, { 12.0, 23.0, 20.0, 21.0 },{ 23.0, 14.0, 14.0, 25.0 },{ 19.0, 19.0, 13.0, 21.0 }},
                {{ 18.0, 18.0, 13.0, 11.0 },{ 11.0, 11.0, 15.0, 20.0 },{ 14.0, 15.0, 15.0, 24.0 },{ 13.0, 15.0, 11.0, 16.0 }}};
            AssertArray(x.hist, ExpectedHist3);
            AssertArray(x.bin_edges[0], new double[] { 10.0, 14.75, 19.5, 24.25, 29.0 });
            AssertArray(x.bin_edges[1], new double[] { 10.0, 14.75, 19.5, 24.25, 29.0 });
            AssertArray(x.bin_edges[2], new double[] { 10.0, 14.75, 19.5, 24.25, 29.0 });
            print(x.hist);
            print(x.bin_edges);

        }

        #endregion

        #region histogram_bin_edges

        [TestMethod]
        public void test_histogram_bin_edges_1()
        {
            var arr = np.arange(40, dtype : np.Float64);

            var x = np.histogram_bin_edges(arr, bins: Histogram_BinSelector.auto, range: (10, 30));
            AssertArray(x, new double[] { 10.0, 13.3333333333333, 16.6666666666667, 20.0, 23.3333333333333, 26.6666666666667, 30.0 });
            print(x);

            x = np.histogram_bin_edges(arr, bins : 4, range : (10, 30));
            AssertArray(x, new double[] { 10.0, 15.0, 20.0, 25.0, 30.0 });
            print(x);

            x = np.histogram_bin_edges(arr, bins : np.arange(5), range : (10, 30));
            AssertArray(x, new int[] { 0, 1, 2, 3, 4 });
            print(x);
        }

        #endregion
    }
}
