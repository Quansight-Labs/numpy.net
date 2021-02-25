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
        public void test_fv_complex()
        {
            var x = npf.fv((Complex)0.075, 20, -2000, 0, 0);
            AssertArray(x, new Complex[] { 86609.362673042924 });
            Assert.AreEqual(86609.362673042924, (Complex)x);
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

        #region npf.pmt tests

        [TestMethod]
        public void test_pmt_1_DOUBLE()
        {
            var res = npf.pmt(0.08 / 12, 5 * 12, 15000);
            AssertArray(res, new double[] { -304.14591432620773 });
            Assert.AreEqual(-304.14591432620773, (double)res);
            print(res);

            res = npf.pmt(0.0, 5 * 12, 15000);
            AssertArray(res, new double[] { -250.0 });
            Assert.AreEqual(-250.0, (double)res);
            print(res);

            res = npf.pmt(new double[,] { { 0.0, 0.8 }, { 0.3, 0.8 } }, new int[] { 12, 3 }, new int[] { 2000, 20000 });
            AssertArray(res, new double[,] { { -166.666666666667, -19311.2582781457 }, { -626.908140170076, -19311.2582781457 } });
            print(res);
        }

        [TestMethod]
        public void test_pmt_1_FLOAT()
        {
            var res = npf.pmt(0.08f / 12f, 5 * 12, 15000);
            AssertArray(res, new double[] { -304.14591170539364 });
            Assert.AreEqual(-304.14591170539364, (double)res);
            print(res);

            res = npf.pmt(0.0f, 5 * 12, 15000);
            AssertArray(res, new double[] { -250.0 });
            Assert.AreEqual(-250.0, (double)res);
            print(res);

            res = npf.pmt(new float[,] { { 0.0f, 0.8f }, { 0.3f, 0.8f } }, new int[] { 12, 3 }, new int[] { 2000, 20000 });
            AssertArray(res, new double[,] { { -166.666666666667, -19311.2584865018 }, { -626.908161987423, -19311.2584865018 } });
            print(res);
        }

        [TestMethod]
        public void test_pmt_1_DECIMAL()
        {
            var res = npf.pmt(0.08m / 12m, 5m * 12m, 15000m);
            AssertArray(res, new decimal[] { -304.14591432608479334671439365m });
            Assert.AreEqual(-304.14591432608479334671439365m, (decimal)res);
            print(res);

            res = npf.pmt(0.0m, 5m * 12m, 15000m);
            AssertArray(res, new decimal[] { -250.0m });
            Assert.AreEqual(-250.0m, (decimal)res);
            print(res);

            res = npf.pmt(new decimal[,] { { 0.0m, 0.8m }, { 0.3m, 0.8m } }, new decimal[] { 12m, 3m }, new decimal[] { 2000m, 20000m });
            AssertArray(res, new decimal[,] { { -166.66666666666666666666666667m, -19311.258278145695364238410596m }, { -626.90814017007577484025865996m, -19311.258278145695364238410596m } });
            print(res);
        }

        [TestMethod]
        public void test_pmt_1_COMPLEX()
        {
            var res = npf.pmt((Complex)0.08 / (Complex)12, 5 * 12, 15000);
            AssertArray(res, new Complex[] { -304.145914326208 });
            //Assert.AreEqual((Complex)(-304.145914326208), (Complex)res);
            print(res);

            res = npf.pmt((Complex)0.0, 5 * 12, 15000);
            AssertArray(res, new Complex[] { -250.0 });
            Assert.AreEqual(-250.0, (Complex)res);
            print(res);

            res = npf.pmt(new Complex[,] { { 0.0, 0.8 }, { 0.3, 0.8 } }, new int[] { 12, 3 }, new int[] { 2000, 20000 });
            AssertArray(res, new Complex[,] { { -166.666666666667, -19311.2582781457 }, { -626.908140170076, -19311.2582781457 } });
            print(res);
        }

        [TestMethod]
        public void test_pmt_when_DOUBLE()
        {
            var res = npf.pmt(0.08 / 12, 5 * 12, 15000, 0, 0);
            AssertArray(res, new double[] { -304.14591432620773 });
            Assert.AreEqual(-304.14591432620773, (double)res);
            print(res);

            res = npf.pmt(0.08 / 12, 5 * 12, 15000, 0, "end");
            AssertArray(res, new double[] { -304.14591432620773 });
            Assert.AreEqual(-304.14591432620773, (double)res);
            print(res);

            res = npf.pmt(0.08 / 12, 5 * 12, 15000, 0, 1);
            AssertArray(res, new double[] { -302.131702973054 });
            Assert.AreEqual(-302.131702973054, (double)res);
            print(res);

            res = npf.pmt(0.08 / 12, 5 * 12, 15000, 0, "begin");
            AssertArray(res, new double[] { -302.131702973054 });
            Assert.AreEqual(-302.131702973054, (double)res);
            print(res);

        }

        [TestMethod]
        public void test_pmt_when_DECIMAL()
        {
            var res = npf.pmt(0.08m / 12m, 5m * 12m, 15000m, 0, 0);
            AssertArray(res, new decimal[] { -304.14591432608479334671439365m });
            Assert.AreEqual(-304.14591432608479334671439365m, (decimal)res);
            print(res);

            res = npf.pmt(0.08m / 12m, 5m * 12m, 15000m, 0, "end");
            AssertArray(res, new decimal[] { -304.14591432608479334671439365m });
            Assert.AreEqual(-304.14591432608479334671439365m, (decimal)res);
            print(res);

            res = npf.pmt(0.08m / 12m, 5m * 12m, 15000m, 0, 1);
            AssertArray(res, new decimal[] { -302.13170297293091348447465287m });
            Assert.AreEqual(-302.13170297293091348447465287m, (decimal)res);
            print(res);

            res = npf.pmt(0.08m / 12m, 5m * 12m, 15000m, 0, "begin");
            AssertArray(res, new decimal[] { -302.13170297293091348447465287m });
            Assert.AreEqual(-302.13170297293091348447465287m, (decimal)res);
            print(res);

        }

        #endregion

        #region npf.nper tests

        [TestMethod]
        public void test_nper_broadcast_DOUBLE()
        {
            var res = npf.nper(0.075, -2000, 0, 100000.0, new int[] { 0, 1 });
            AssertArray(res, new double[] { 21.5449441973233, 20.7615644051895 });
            print(res);
        }

        [TestMethod]
        public void test_nper_broadcast_DECIMAL()
        {
            var res = npf.nper(0.075m, -2000m, 0m, 100000.0m, new int[] { 0, 1 });
            AssertArray(res, new decimal[] { 21.5449441973233m, 20.7615644051895m });
            print(res);
        }

        [TestMethod]
        public void test_nper_basic_values_DOUBLE()
        {
            var res = npf.nper(new double[] { 0, 0.075 }, -2000, 0, 100000);
            AssertArray(res, new double[] { 50.0, 21.5449441973233 });
            print(res);
        }

        [TestMethod]
        public void test_nper_basic_values_DECIMAL()
        {
            var res = npf.nper(new decimal[] { 0, 0.075m }, -2000, 0, 100000);
            AssertArray(res, new decimal[] { 50.0m, 21.5449441973234m });
            print(res);
        }

        [TestMethod]
        public void test_nper_gh_18_DOUBLE()
        {
            var res = npf.nper(0.1, 0, -500, 1500);
            AssertArray(res, new double[] { 11.5267046072476 });
            print(res);
        }

        [TestMethod]
        public void test_nper_gh_18_DECIMAL()
        {
            var res = npf.nper(0.1m, 0, -500, 1500);
            AssertArray(res, new decimal[] { 11.5267046072476m });
            print(res);
        }

        [TestMethod]
        public void test_nper_infinite_payments_DOUBLE()
        {
            var res = npf.nper(0.0, -0.0, 1000.0);
            AssertArray(res, new double[] { double.PositiveInfinity });
            print(res);
        }

        [TestMethod]
        public void test_nper_infinite_payments_DECIMAL()
        {
            var res = npf.nper(0m, -0.0, 1000);
            AssertArray(res, new decimal[] { 0m });
            print(res);
        }

        [TestMethod]
        public void test_nper_no_interest_DOUBLE()
        {
            var res = npf.nper(0, -100, 1000);
            AssertArray(res, new double[] { 10.0 });
            print(res);
        }


        [TestMethod]
        public void test_nper_no_interest_DECIMAL()
        {
            var res = npf.nper(0m, -100, 1000);
            AssertArray(res, new decimal[] { 10.0m });
            print(res);
        }

        #endregion

        #region npf.ipmt tests

        [TestMethod]
        public void test_ipmt_DOUBLE()
        {
            var res = npf.ipmt(0.1 / 12, 1, 24, 2000);
            AssertArray(res, new double[] { -16.6666666666667 });
            print(res);
        }

        [TestMethod]
        public void test_ipmt_DECIMAL()
        {
            var res = npf.ipmt(0.1m / 12m, 1, 24, 2000);
            AssertArray(res, new decimal[] { -16.666666666666666666666666600m });
            print(res);
        }

        [TestMethod]
        public void test_ipmt_when_is_begin_DOUBLE()
        {
            var res = npf.ipmt(0.1 / 12, 1, 24, 2000, 0, "begin");
            AssertArray(res, new double[] { 0.0 });
            print(res);

            res = npf.ipmt(0.1 / 12, 1, 24, 2000, 0, 1);
            AssertArray(res, new double[] { 0.0 });
            print(res);
        }

        [TestMethod]
        public void test_ipmt_when_is_begin_DECIMAL()
        {
            var res = npf.ipmt(0.1m / 12m, 1, 24, 2000, 0, "begin");
            AssertArray(res, new decimal[] { 0.0m });
            print(res);

            res = npf.ipmt(0.1m / 12m, 1, 24, 2000, 0, 1);
            AssertArray(res, new decimal[] { 0.0m });
            print(res);
        }
        [TestMethod]
        public void test_ipmt_when_is_end_DOUBLE()
        {
            var res = npf.ipmt(0.1 / 12, 1, 24, 2000, 0, "end");
            AssertArray(res, new double[] { -16.6666666666667 });
            print(res);

            res = npf.ipmt(0.1 / 12, 1, 24, 2000, 0, 0);
            AssertArray(res, new double[] { -16.6666666666667 });
            print(res);
        }
        [TestMethod]
        public void test_ipmt_when_is_end_DECIMAL()
        {
            var res = npf.ipmt(0.1m / 12m, 1, 24, 2000, 0, "end");
            AssertArray(res, new decimal[] { -16.666666666666666666666666600m });
            print(res);

            res = npf.ipmt(0.1m / 12m, 1, 24, 2000, 0, 0);
            AssertArray(res, new decimal[] { -16.666666666666666666666666600m });
            print(res);
        }
        [TestMethod]
        public void test_ipmt_gh_17_DOUBLE()
        {
            var rate = 0.001988079518355057;

            var res = npf.ipmt(rate, 0, 360, 300000, fv:0, when :"begin");
            AssertArray(res, new double[] { double.NaN });
            print(res);

            res = npf.ipmt(rate, 1, 360, 300000, fv: 0, when: "begin");
            AssertArray(res, new double[] { 0.0 });
            print(res);

            res = npf.ipmt(rate, 2, 360, 300000, fv: 0, when: "begin");
            AssertArray(res, new double[] { -594.107157704708 });
            print(res);

            res = npf.ipmt(rate, 3, 360, 300000, fv: 0, when : "begin");
            AssertArray(res, new double[] { -592.971592174841 });
            print(res);

        }
        [TestMethod]
        public void test_ipmt_gh_17_DECIMAL()
        {
            var rate = 0.001988079518355057m;

            var res = npf.ipmt(rate, 0, 360, 300000, fv: 0, when: "begin");
            AssertArray(res, new decimal[] { 0.0m });
            print(res);

            res = npf.ipmt(rate, 1, 360, 300000, fv: 0, when: "begin");
            AssertArray(res, new decimal[] { 0.0m });
            print(res);

            res = npf.ipmt(rate, 2, 360, 300000, fv: 0, when: "begin");
            AssertArray(res, new decimal[] {-594.1071577047065727486948811m });
            print(res);

            res = npf.ipmt(rate, 3, 360, 300000, fv: 0, when: "begin");
            AssertArray(res, new decimal[] { -592.97159217484060101674297453m });
            print(res);
        }

        [TestMethod]
        public void test_ipmt_broadcasting_DOUBLE()
        {
            var res = npf.ipmt(0.1 / 12, np.arange(5), 24, 2000);
            AssertArray(res, new double[] { double.NaN, -16.6666666666667, -16.0364734499303, -15.4010286230544, -14.7602884226213 });
            print(res);
        }
        [TestMethod]
        public void test_ipmt_broadcasting_DECIMAL()
        {
            var res = npf.ipmt(0.1m / 12m, np.arange(5), 24, 2000);
            AssertArray(res, new decimal[] { 0m, -16.666666666666666666666666600m, -16.036473449930246209851847828m, -15.401028623054689584968442671m, -14.760288422620935478577742361m });
            print(res);
        }

        #endregion
    }
}
