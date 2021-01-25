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
    public class HistogramTests : TestBaseClass
    {
        [TestMethod]
        public void test_bincount_1()
        {
            var x = np.arange(5);
            var a = np.bincount(x);
            print(a);

            x = np.array(new int[] { 0, 1, 1, 3, 2, 1, 7 });
            a = np.bincount(x);
            print(a);

            x = np.array(new int[] { 0, 1, 1, 3, 2, 1, 7, 23 });
            a = np.bincount(x);
            print(a);
            print(a.size == (int)np.amax(x) + 1);

        }

        [TestMethod]
        public void test_bincount_2()
        {
            var x = np.arange(5, dtype: np.Int64);
            var a = np.bincount(x);
            print(a);

            x = np.array(new int[] { 0, 1, 1, 3, 2, 1, 7 }, dtype: np.Int16);
            a = np.bincount(x);
            print(a);

            x = np.array(new int[] { 0, 1, 1, 3, 2, 1, 7, 23 }, dtype: np.Int8);
            a = np.bincount(x);
            print(a);
            print(a.size == (int)np.amax(x) + 1);

        }

        [TestMethod]
        public void test_bincount_3()
        {

            var w = np.array(new float[] { 0.3f, 0.5f, 0.2f, 0.7f, 1.0f, -0.6f });  // weights

            var x = np.arange(5, dtype: np.Int64);
            var a = np.bincount(x, weights: w);
            print(a);

            x = np.array(new int[] { 0, 1, 3, 2, 1, 7 }, dtype: np.Int16);
            a = np.bincount(x, weights: w);
            print(a);

            x = np.array(new int[] { 0, 1, 3, 2, 1, 7 }, dtype: np.Int8);
            a = np.bincount(x, weights: w);
            print(a);
            print(a.size == (int)np.amax(x) + 1);

        }

        [TestMethod]
        public void test_bincount_4()
        {
            var x = np.arange(5, dtype: np.Int64);
            var a = np.bincount(x, minlength: 8);
            print(a);

            x = np.array(new int[] { 0, 1, 1, 3, 2, 1, 7 }, dtype: np.Int16);
            a = np.bincount(x, minlength: 10);
            print(a);

            x = np.array(new int[] { 0, 1, 1, 3, 2, 1, 7, 23 }, dtype: np.Int8);
            a = np.bincount(x, minlength: 32);
            print(a);
            print(a.size == (int)np.amax(x) + 1);

        }

        [TestMethod]
        public void test_bincount_uint64()
        {
            try
            {
                var x = np.arange(5, dtype: np.UInt64);
                var a = np.bincount(x);
                print(a);
            }
            catch
            {

            }

        }

        [TestMethod]
        public void test_bincount_double()
        {
            try
            {
                var x = np.arange(5, dtype: np.Float64);
                var a = np.bincount(x);
                print(a);
            }
            catch
            {

            }

        }


    }
}
