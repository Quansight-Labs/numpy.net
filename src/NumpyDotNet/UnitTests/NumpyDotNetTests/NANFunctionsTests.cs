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
    public class NANFunctionsTests : TestBaseClass
    {
        [Ignore] // need to implement Nanfunctions
        [TestMethod]
        public void test_nanprod_1()
        {

        }
        [Ignore] // need to implement Nanfunctions
        [TestMethod]
        public void test_nansum_1()
        {

        }

        [Ignore] // need to implement Nanfunctions
        [TestMethod]
        public void test_nancumproduct_placeholder()
        {
            // see the NANFunctionsTest version
        }

        [Ignore] // need to implement Nanfunctions
        [TestMethod]
        public void test_nancumsum_placeholder()
        {
            // see the NANFunctionsTest version
        }

        [Ignore] // need to implement Nanfunctions
        [TestMethod]
        public void test_nanpercentile_placeholder()
        {
            // see the NANFunctionsTest version
        }

        [Ignore] // need to implement Nanfunctions
        [TestMethod]
        public void test_nanquantile_placeholder()
        {
            // see the NANFunctionsTest version
        }


        [Ignore] // need to implement Nanfunctions
        [TestMethod]
        public void test_nanmedian_placeholder()
        {
            // see the NANFunctionsTest version
        }

        [Ignore] // need to implement Nanfunctions
        [TestMethod]
        public void test_nanmean_placeholder()
        {
            // see the NANFunctionsTest version
        }

        [Ignore] // need to implement Nanfunctions
        [TestMethod]
        public void test_nanstd_placeholder()
        {
            // see the NANFunctionsTest version
        }

        [Ignore] // need to implement Nanfunctions
        [TestMethod]
        public void test_nanvar_placeholder()
        {
            // see the NANFunctionsTest version
        }

        [TestMethod]
        public void test_nanmin_1()
        {
            var a = np.array(new float[,] { { 1, 2 }, { 3, float.NaN } });
            var b = np.nanmin(a);
            Assert.AreEqual(1.0f, b.GetItem(0));
            print(b);


            var c = np.nanmin(a, axis: 0);
            AssertArray(c, new float[] { 1.0f, 2.0f });
            print(c);

            var d = np.nanmin(a, axis: 1);
            AssertArray(d, new float[] { 1.0f, 3.0f });
            print(d);

            // When positive infinity and negative infinity are present:

            var e = np.nanmin(new float[] { 1, 2, float.NaN, float.PositiveInfinity });
            Assert.AreEqual(1.0f, e.GetItem(0));
            print(e);

            var f = np.nanmin(new float[] { 1, 2, float.NaN, float.NegativeInfinity });
            Assert.AreEqual(float.NegativeInfinity, f.GetItem(0));
            print(f);

            var g = np.amin(new float[] { 1, 2, -3, float.NegativeInfinity });
            Assert.AreEqual(float.NegativeInfinity, g.GetItem(0));
            print(g);
        }

        [TestMethod]
        public void test_nanmin_2()
        {
            var a = np.array(new double[,] { { 1, 2 }, { 3, double.NaN } });
            var b = np.nanmin(a);
            Assert.AreEqual(1.0, b.GetItem(0));
            print(b);


            var c = np.nanmin(a, axis: 0);
            AssertArray(c, new double[] { 1.0, 2.0 });
            print(c);

            var d = np.nanmin(a, axis: 1);
            AssertArray(d, new double[] { 1.0, 3.0 });
            print(d);

            // When positive infinity and negative infinity are present:

            var e = np.nanmin(new double[] { 1, 2, double.NaN, double.PositiveInfinity });
            Assert.AreEqual(1.0, e.GetItem(0));
            print(e);

            var f = np.nanmin(new double[] { 1, 2, double.NaN, double.NegativeInfinity });
            Assert.AreEqual(double.NegativeInfinity, f.GetItem(0));
            print(f);

            var g = np.amin(new double[] { 1, 2, -3, double.NegativeInfinity });
            Assert.AreEqual(double.NegativeInfinity, g.GetItem(0));
            print(g);
        }

        [TestMethod]
        public void test_nanmin_3()
        {
            var a = np.array(new long[,] { { 1, 2 }, { 3,-4 } });
            var b = np.nanmin(a);
            Assert.AreEqual((long)-4, b.GetItem(0));
            print(b);


            var c = np.nanmin(a, axis: 0);
            AssertArray(c, new long[] { 1, -4 });
            print(c);

            var d = np.nanmin(a, axis: 1);
            AssertArray(d, new long[] { 1, -4 });
            print(d);

        }

        [Ignore] // need to implement Nanfunctions
        [TestMethod]
        public void test_nanmax_placeholder()
        {
            // see the NANFunctionsTest version
        }

        [Ignore]
        [TestMethod]
        public void xxx_Test_NANFunctions_Placeholder()
        {
        }
    }
}
