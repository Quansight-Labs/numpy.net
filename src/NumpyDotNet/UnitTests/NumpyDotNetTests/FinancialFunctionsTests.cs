using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumpyDotNetTests
{
    [TestClass]
    public class FinancialFunctionsTests : TestBaseClass
    {
        #region npf.fv tests
        [TestMethod]
        public void test_fv_int()
        {
            var x = npf.fv(75, 20, -2000, 0, 0);
            AssertArray(x, new Int32[] { -2147483648 });
            Assert.AreEqual(-2147483648, (Int32)x);
            print(x);
        }

        [TestMethod]
        public void test_fv_float()
        {
            var x = npf.fv(0.075, 20, -2000, 0, 0);
            AssertArray(x, new double[] { 86609.362673042924 });
            Assert.AreEqual(86609.362673042924, (double)x);
            print(x);
        }

        [TestMethod]
        public void test_fv_float_FLOAT()
        {
            var x = npf.fv(0.075f, 20, -2000, 0, 0);
            AssertArray(x, new float[] { 86609.37f });
            Assert.AreEqual(86609.37f, (float)x);
            print(x);
        }

        [TestMethod]
        public void test_fv_decimal()
        {
            var x = npf.fv(0.075m, 20m, -2000m, 0, 0);
            AssertArray(x, new decimal[] { 86609.36267304293333333333333m });
            Assert.AreEqual(86609.36267304293333333333333m, (decimal)x);
            print(x);
        }

        [TestMethod]
        public void test_fv_when_is_begin_float()
        {
            var x = npf.fv(0.075, 20, -2000, 0, "begin");
            AssertArray(x, new double[] { 93105.064873521143 });
            Assert.AreEqual(93105.064873521143, (double)x);
            print(x);
        }

        [TestMethod]
        public void test_fv_when_is_begin_decimal()
        {
            var x = npf.fv(0.075m, 20m, -2000m, 0, "begin");
            AssertArray(x, new decimal[] { 93105.06487352115333333333333m });
            Assert.AreEqual(93105.06487352115333333333333m, (decimal)x);
            print(x);
        }

        [TestMethod]
        public void test_fv_when_is_end_float()
        {
            var x = npf.fv(0.075, 20, -2000, 0, "end");
            AssertArray(x, new double[] { 86609.362673042924 });
            Assert.AreEqual(86609.362673042924, (double)x);
            print(x);
        }

        [TestMethod]
        public void test_fv_when_is_end_decimal()
        {
            var x = npf.fv(0.075m, 20m, -2000m, 0, "end");
            AssertArray(x, new decimal[] { 86609.36267304293333333333333m });
            Assert.AreEqual(86609.36267304293333333333333m, (decimal)x);
            print(x);
        }

        [TestMethod]
        public void test_fv_broadcast()
        {
            var result = npf.fv(new double[,] { { 0.1 }, { 0.2 } }, 5, 100, 0, new int[] { 0, 1 });
            AssertArray(result, new double[,] { { -610.510000000001, -671.561000000001 }, { -744.16, -892.992 } });
            print(result);
        }

        [TestMethod]
        public void test_fv_broadcast_FLOAT()
        {
            var result = npf.fv(new float[,] { { 0.1f }, { 0.2f } }, 5, 100, 0, new int[] { 0, 1 });
            AssertArray(result, new float[,] { { -610.51f, -671.561f }, { -744.160034f, -892.992f } });
            print(result);
        }

        [TestMethod]
        public void test_fv_some_rates_zero()
        {
            var result = npf.fv(new double[,] { { 0.0 }, { 0.1 } }, 5, 100, 0, 0);
            AssertArray(result, new double[,] { { -500.0 }, { -610.510000000001 } });
            print(result);
        }

        [TestMethod]
        public void test_fv_float_array_1()
        {
            var x = npf.fv(new double[] { -0.075, 1.075, -1.075 }, new double[] { 20 }, new int[] { -2100, 2000, -2500 }, 0, new object[] { "begin", "end", "begin" });
            AssertArray(x, new double[] { 20453.28791585521, -4073646637.206109, -174.4186046511627 });
            print(x);
        }

        [TestMethod]
        public void test_fv_float_array_1A()
        {
            var x = npf.fv(new double[] { -0.075, 1.075, -1.075 }, new double[] { 20 }, new int[] { -2100, 2000, -2500 }, 0, new object[] { 1, 0, 1 });
            AssertArray(x, new double[] { 20453.28791585521, -4073646637.206109, -174.4186046511627 });
            print(x);
        }

        [TestMethod]
        public void test_fv_float_array_1B()
        {
            var x = npf.fv(new double[] { -0.075, 1.075, -1.075 }, new double[] { 20 }, new int[] { -2100, 2000, -2500 }, 0, new object[] { 1, "end", "begin" });
            AssertArray(x, new double[] { 20453.28791585521, -4073646637.206109, -174.4186046511627 });
            print(x);
        }



        [TestMethod]
        public void test_fv_float_array_2()
        {
            try
            {
                var x = npf.fv(new double[] { -0.075, 1.075, -1.075 }, new double[] { 20 }, new int[] { -2100, 2000, -2500 }, 0, new object[] { "begin", "end", "xxx" });
                print(x);
                Assert.Fail("Expected exception was not caught");
            }
            catch
            {

            }
        }

        [TestMethod]
        public void test_fv_float_array_3()
        {
            try
            {
                var x = npf.fv(new double[] { -0.075, 1.075, -1.075 }, new double[] { 20 }, new int[] { -2100, 2000, -2500 }, 0, new object[] { "begin", "end" });
                print(x);
                Assert.Fail("Expected exception was not caught");
            }
            catch
            {

            }
        }
        #endregion

    }
}
