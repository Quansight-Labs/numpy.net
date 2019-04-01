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
            print(b);


            var c = np.nanmin(a, axis: 0);
            print(c);

            var d = np.nanmin(a, axis: 1);
            print(d);

            // When positive infinity and negative infinity are present:

            var e = np.nanmin(new float[] { 1, 2, float.NaN, float.PositiveInfinity });
            print(e);

            var f = np.nanmin(new float[] { 1, 2, float.NaN, float.NegativeInfinity });
            print(f);

            var g = np.amin(new float[] { 1, 2, -3, float.NegativeInfinity });
            print(g);
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
