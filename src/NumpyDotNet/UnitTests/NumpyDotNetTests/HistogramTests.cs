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
    }
}
