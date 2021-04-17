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
            AssertArray(x, new float[] { 86609.45f });
            Assert.AreEqual(86609.45f, (float)x);
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
            AssertArray(result, new float[,] { { -610.510254f, -671.5613f }, { -744.1603f, -892.9924f } });
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
            AssertArray(res, new double[] { -304.14613557314783 });
            Assert.AreEqual(-304.14613557314783, (double)res);
            print(res);

            res = npf.pmt(0.0f, 5 * 12, 15000);
            AssertArray(res, new double[] { -250.0 });
            Assert.AreEqual(-250.0, (double)res);
            print(res);

            res = npf.pmt(new float[,] { { 0.0f, 0.8f }, { 0.3f, 0.8f } }, new int[] { 12, 3 }, new int[] { 2000, 20000 });
            AssertArray(res, new double[,] { { -166.66666666666666, -19311.258768925054 }, { -626.90816408223657, -19311.258768925054 } });
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

        #region npf.ppmt tests

        [TestMethod]
        public void test_ppmt_DOUBLE()
        {
            var res = npf.ppmt(0.1 / 12, 1, 60, 55000);
            AssertArray(res, new double[] { -710.254125786425 });
            print(res);
        }
        [TestMethod]
        public void test_ppmt_DECIMAL()
        {
            var res = npf.ppmt(0.1m / 12m, 1, 60, 55000);
            AssertArray(res, new decimal[] { -710.25412578678316794930885490m });
            print(res);
        }
        [TestMethod]
        public void test_ppmt_begin_DOUBLE()
        {
            var res = npf.ppmt(0.1 / 12, 1, 60, 55000, 0, 1);
            AssertArray(res, new double[] { -1158.92971152373 });
            print(res);

            res = npf.ppmt(0.1 / 12, 1, 60, 55000, 0, "begin");
            AssertArray(res, new double[] { -1158.92971152373 });
            print(res);
        }
        [TestMethod]
        public void test_ppmt_begin_DECIMAL()
        {
            var res = npf.ppmt(0.1m / 12m, 1, 60, 55000, 0, 1);
            AssertArray(res, new decimal[] { -1158.9297115240863117834848926m });
            print(res);

            res = npf.ppmt(0.1m / 12m, 1, 60, 55000, 0, "begin");
            AssertArray(res, new decimal[] { -1158.9297115240863117834848926m });
            print(res);
        }
        [TestMethod]
        public void test_ppmt_end_DOUBLE()
        {
            var res = npf.ppmt(0.1 / 12, 1, 60, 55000, 0, 0);
            AssertArray(res, new double[] { -710.254125786425 });
            print(res);

            res = npf.ppmt(0.1 / 12, 1, 60, 55000, 0, "end");
            AssertArray(res, new double[] { -710.254125786425 });
            print(res);
        }
        [TestMethod]
        public void test_ppmt_end_DECIMAL()
        {
            var res = npf.ppmt(0.1m / 12m, 1, 60, 55000, 0, 0);
            AssertArray(res, new decimal[] { -710.25412578678316794930885490m });
            print(res);

            res = npf.ppmt(0.1m / 12m, 1, 60, 55000, 0, "end");
            AssertArray(res, new decimal[] { -710.25412578678316794930885490m });
            print(res);
        }
        [TestMethod]
        public void test_ppmt_invalid_per_DOUBLE()
        {
            var res = npf.ppmt(0.1 / 12, 0, 60, 15000);
            AssertArray(res, new double[] { double.NaN });
            print(res);
        }
        [TestMethod]
        public void test_ppmt_invalid_per_DECIMAL()
        {
            var res = npf.ppmt(0.1m / 12m, 0, 60, 15000);
            AssertArray(res, new decimal[] { -318.70567066912268216799332357m });
            print(res);
        }
        [TestMethod]
        public void test_ppmt_broadcast_DOUBLE()
        {
            var res = npf.ppmt(0.1 / 12, np.arange(1, 5), 24, 2000, 0);
            AssertArray(res, new double[] { -75.6231860083666, -76.253379225103, -76.8888240519789, -77.529564252412 });
            print(res);

            res = npf.ppmt(0.1 / 12, np.arange(1, 5), 24, 2000, 0, "end");
            AssertArray(res, new double[] { -75.6231860083666, -76.253379225103, -76.8888240519789, -77.529564252412 });
            print(res);

            res = npf.ppmt(0.1 / 12, np.arange(1, 5), 24, 2000, 0, "begin");
            AssertArray(res, new double[] { -91.5271266198677, -75.6231860083666, -76.253379225103, -76.8888240519789 });
            print(res);
        }

        [TestMethod]
        public void test_ppmt_broadcast_DECIMAL()
        {
            var res = npf.ppmt(0.1m / 12m, np.arange(1, 5), 24, 2000, 0);
            AssertArray(res, new decimal[] { -75.623186008400704092181612840m, -76.253379225137124548996431612m, -76.888824052012681173879836769m, -77.529564252446435280270537079m });
            print(res);

            res = npf.ppmt(0.1m / 12m, np.arange(1, 5), 24, 2000, 0, "end");
            AssertArray(res, new decimal[] { -75.623186008400704092181612840m, -76.253379225137124548996431612m, -76.888824052012681173879836769m, -77.529564252446435280270537079m });
            print(res);

            res = npf.ppmt(0.1m / 12m, np.arange(1, 5), 24, 2000, 0, "begin");
            AssertArray(res, new decimal[] { -91.52712661990182728853257433m, -75.623186008400704092181612844m, -76.253379225136795390225375764m, -76.888824052013247844314754175m });
            print(res);

        }


        #endregion

        #region npf.pv tests

        [TestMethod]
        public void test_pv_DOUBLE()
        {
            var res = npf.pv(0.07, 20, 12000);
            AssertArray(res, new double[] { -127128.170946194 });
            print(res);

            res = npf.pv(0.07, 20, 12000, 0);
            AssertArray(res, new double[] { -127128.170946194 });
            print(res);

            res = npf.pv(0.07, 20, 12000, 222220);
            AssertArray(res, new double[] { -184554.041751492 });
            print(res);
        }
        [TestMethod]
        public void test_pv_DECIMAL()
        {
            var res = npf.pv(0.07m, 20, 12000);
            AssertArray(res, new decimal[] { -127128.17094619401705869097106m });
            print(res);

            res = npf.pv(0.07m, 20, 12000, 0);
            AssertArray(res, new decimal[] { -127128.17094619401705869097106m });
            print(res);

            res = npf.pv(0.07m, 20, 12000, 222220);
            AssertArray(res, new decimal[] { -184554.04175149191168891218562m });
            print(res);
        }
        [TestMethod]
        public void test_pv_begin_DOUBLE()
        {
            var res = npf.pv(0.07, 20, 12000, 0, 1);
            AssertArray(res, new double[] { -136027.142912428 });
            print(res);

            res = npf.pv(0.07, 20, 12000, 0, "begin");
            AssertArray(res, new double[] { -136027.142912428 });
            print(res);
        }
        [TestMethod]
        public void test_pv_begin_DECIMAL()
        {
            var res = npf.pv(0.07m, 20, 12000, 0, 1);
            AssertArray(res, new decimal[] { -136027.14291242755173737883254m });
            print(res);

            res = npf.pv(0.07m, 20, 12000, 0, "begin");
            AssertArray(res, new decimal[] { -136027.14291242755173737883254m });
            print(res);
        }
        [TestMethod]
        public void test_pv_end_DOUBLE()
        {
            var res = npf.pv(0.07, 20, 12000, 0, 0);
            AssertArray(res, new double[] { -127128.170946194 });
            print(res);

            res = npf.pv(0.07, 20, 12000, 0, "end");
            AssertArray(res, new double[] { -127128.170946194 });
            print(res);
        }
        [TestMethod]
        public void test_pv_end_DECIMAL()
        {
            var res = npf.pv(0.07m, 20, 12000, 0, 0);
            AssertArray(res, new decimal[] { -127128.17094619401705869097106m });
            print(res);

            res = npf.pv(0.07m, 20, 12000, 0, "end");
            AssertArray(res, new decimal[] { -127128.17094619401705869097106m });
            print(res);
        }
        #endregion

        #region npf.rate tests

        [TestMethod]
        public void test_rate_DOUBLE()
        {
            var res = npf.rate(10, 0, -3500, 10000);
            AssertArray(res, new double[] { 0.11069085371426901 });
            print(res);
        }

        [TestMethod]
        public void test_rate_DECIMAL()
        {
            var res = npf.rate(10m, 0m, -3500, 10000);
            AssertArray(res, new decimal[] { 0.1106908537142690373121121834m });
            print(res);
        }

        [TestMethod]
        public void test_rate_begin_DOUBLE()
        {
            var res = npf.rate(10, 0, -3500, 10000, 1);
            AssertArray(res, new double[] { 0.11069085371426901 });
            print(res);

            res = npf.rate(10, 0, -3500, 10000, "begin");
            AssertArray(res, new double[] { 0.11069085371426901 });
            print(res);
        }
        [TestMethod]
        public void test_rate_begin_DECIMAL()
        {
            var res = npf.rate(10m, 0m, -3500, 10000, 1);
            AssertArray(res, new decimal[] { 0.1106908537142690373121121834m });
            print(res);

            res = npf.rate(10m, 0m, -3500, 10000, "begin");
            AssertArray(res, new decimal[] { 0.1106908537142690373121121834m });
            print(res);
        }

        [TestMethod]
        public void test_rate_end_DOUBLE()
        {
            var res = npf.rate(10, 0, -3500, 10000, 0);
            AssertArray(res, new double[] { 0.11069085371426901 });
            print(res);

            res = npf.rate(10, 0, -3500, 10000, "end");
            AssertArray(res, new double[] { 0.11069085371426901 });
            print(res);
        }
        [TestMethod]
        public void test_rate_end_DECIMAL()
        {
            var res = npf.rate(10m, 0m, -3500, 10000, 0);
            AssertArray(res, new decimal[] { 0.1106908537142690373121121834m });
            print(res);

            res = npf.rate(10m, 0m, -3500, 10000, "end");
            AssertArray(res, new decimal[] { 0.1106908537142690373121121834m });
            print(res);

        }

        [TestMethod]
        public void test_rate_infeasable_solution_DOUBLE()
        {
            var res = npf.rate(12.0, 400.0, 10000.0, 5000.0, when: 0);
            AssertArray(res, new double[] { double.NaN });
            print(res);

            res = npf.rate(12.0, 400.0, 10000.0, 5000.0, when: 1);
            AssertArray(res, new double[] { double.NaN });
            print(res);

            res = npf.rate(12.0, 400.0, 10000.0, 5000.0, when: "end");
            AssertArray(res, new double[] { double.NaN });
            print(res);

            res = npf.rate(12.0, 400.0, 10000.0, 5000.0, when: "begin");
            AssertArray(res, new double[] { double.NaN });
            print(res);

   
        }
        [TestMethod]
        public void test_rate_infeasable_solution_DECIMAL()
        {
            try
            {
                var res = npf.rate(12.0m, 400.0m, 10000.0, 5000.0, when: 0);
                print(res);
                Assert.Fail("This should have thrown exception");
            }
            catch
            {

            }

            try
            {
                var res = npf.rate(12.0m, 400.0m, 10000.0, 5000.0, when: 1);
                print(res);
                Assert.Fail("This should have thrown exception");
            }
            catch
            {

            }

            try
            {
                var res = npf.rate(12.0m, 400.0m, 10000.0, 5000.0, when: "end");
                print(res);
                Assert.Fail("This should have thrown exception");
            }
            catch
            {

            }

            try
            {
                var res = npf.rate(12.0m, 400.0m, 10000.0, 5000.0, when: "begin");
                print(res);
                Assert.Fail("This should have thrown exception");
            }
            catch
            {

            }

        }


        #endregion

        #region npf.irr tests

        [TestMethod]
        public void test_irr_basic_values_DOUBLE()
        {
            var cashflows = np.array(new double[] { -150000, 15000, 25000, 35000, 45000, 60000 });
            var res = npf.irr(cashflows);
            Assert.AreEqual(0.052432888859684834, (double)res);
            print(res);

            cashflows = np.array(new double[] { -100, 0, 0, 74 });
            res = npf.irr(cashflows);
            Assert.AreEqual(-0.0954958303502164, (double)res);
            print(res);

            cashflows = np.array(new double[] { -100, 39, 59, 55, 20 });
            res = npf.irr(cashflows);
            Assert.AreEqual(0.28094842115988539, (double)res);
            print(res);

            cashflows = np.array(new double[] { -100, 100, 0, -7 });
            res = npf.irr(cashflows);
            Assert.AreEqual(-0.083299666184932869, (double)res);
            print(res);

            cashflows = np.array(new double[] { -100, 100, 0, 7});
            res = npf.irr(cashflows);
            Assert.AreEqual(0.06205848562992939, (double)res);
            print(res);

            cashflows = np.array(new double[] { -5, 10.5, 1, -8, 1 });
            res = npf.irr(cashflows);
            Assert.AreEqual(0.088598338524376138, (double)res);
            print(res);

        }

        [TestMethod]
        public void test_irr_basic_values_DECIMAL()
        {
            var cashflows = np.array(new decimal[] { -150000, 15000, 25000, 35000, 45000, 60000 });
            var res = npf.irr(cashflows);
            Assert.AreEqual(0.0524328888596848960853398738m, (decimal)res);
            print(res);

            cashflows = np.array(new decimal[] { -100, 0, 0, 74 });
            res = npf.irr(cashflows);
            Assert.AreEqual(-0.0954958303502164492046778255m, (decimal)res);
            print(res);

            cashflows = np.array(new decimal[] { -100, 39, 59, 55, 20 });
            res = npf.irr(cashflows);
            Assert.AreEqual(0.2809484211598854548230841175m, (decimal)res);
            print(res);

            cashflows = np.array(new decimal[] { -100, 100, 0, -7 });
            res = npf.irr(cashflows);
            Assert.AreEqual(-0.0832996661849326850317816561m, (decimal)res);
            print(res);

            cashflows = np.array(new decimal[] { -100, 100, 0, 7 });
            res = npf.irr(cashflows);
            Assert.AreEqual(0.0620584856299294607997074641m, (decimal)res);
            print(res);

            cashflows = np.array(new decimal[] { -5, 10.5m, 1, -8, 1 });
            res = npf.irr(cashflows);
            Assert.AreEqual(0.0885983385243761645016418267m, (decimal)res);
            print(res);

        }

        [TestMethod]
        public void test_irr_trailing_zeros_DOUBLE()
        {
            var cashflows = np.array(new double[] { -5, 10.5, 1, -8, 1, 0, 0, 0 });
            var res = npf.irr(cashflows);
            Assert.AreEqual(0.088598338524376138, (double)res);
            print(res);
        }

        [TestMethod]
        public void test_irr_trailing_zeros_DECIMAL()
        {
            var cashflows = np.array(new decimal[] { -5, 10.5m, 1, -8, 1, 0, 0, 0 });
            var res = npf.irr(cashflows);
            Assert.AreEqual(0.0885983385243761645016418267m, (decimal)res);
            print(res);
        }

        [TestMethod]
        public void test_irr_2dim()
        {
            var cashflows = np.array(new double[] { -150000, 15000, 25000, 35000, 45000, 60000, -150000, 15000, 25000, 35000, 45000, 60000 }).reshape(2,-1);

            try
            {
                var res = npf.irr(cashflows);
                Assert.Fail("This should have thrown rank-1 exception");
                print(res);
            }
            catch
            {

            }
 
        }

        [TestMethod]
        public void test_irr_gh_6744()
        {
            var cashflows = np.array(new double[] { -1, -2, -3 });
            var res = npf.irr(cashflows);
            Assert.AreEqual(double.NaN, (double)res);
            print(res);
        }

        [TestMethod]
        public void test_irr_gh_6744_DECIMAL()
        {
            var cashflows = np.array(new decimal[] { -1, -2, -3 });

            try
            {
                var res = npf.irr(cashflows);
                Assert.Fail("This should have thrown an exception");
                print(res);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

   
        }



        [TestMethod]
        public void test_irr_gh_15()
        {
            // note: this gets different result than python.  I don't care.

            try
            {

                double[] v = {
                     -3000.0,
                     2.3926932267015667e-07,
                     4.1672087103345505e-16,
                     5.3965110036378706e-25,
                     5.1962551071806174e-34,
                     3.7202955645436402e-43,
                     1.9804961711632469e-52,
                     7.8393517651814181e-62,
                     2.3072565113911438e-71,
                     5.0491839233308912e-81,
                     8.2159177668499263e-91,
                     9.9403244366963527e-101,
                     8.942410813633967e-111,
                     5.9816122646481191e-121,
                     2.9750309031844241e-131,
                     1.1002067043497954e-141,
                     3.0252876563518021e-152,
                     6.1854121948207909e-163,
                     9.4032980015353301e-174,
                     1.0629218520017728e-184,
                     8.9337141847171845e-196,
                     5.5830607698467935e-207,
                     2.5943122036622652e-218,
                     8.9635842466507006e-230,
                     2.3027710094332358e-241,
                     4.3987510596745562e-253,
                     6.2476630372575209e-265,
                     6.598046841695288e-277,
                     5.1811095266842017e-289,
                     3.0250999925830644e-301,
                     1.3133070599585015e-313,

                };


                var res = npf.irr(v);
                Assert.Fail("Should have thrown exception here");
                print(res);
            }
            catch
            {

            }


        }

        #endregion

        #region  npf.npv tests

        [TestMethod]
        public void test_npv_DOUBLE()
        {
            var res = npf.npv(0.05, new Int32[] { -15000, 1500, 2500, 3500, 4500, 6000 });
            AssertArray(res, new double[] { 122.89485495093959 });
            print(res);
        }

        [TestMethod]
        public void test_npv_DECIMAL()
        {
            var res = npf.npv(0.05m, new Int32[] { -15000, 1500, 2500, 3500, 4500, 6000 });
            AssertArray(res, new decimal[] {  122.89485495095m });
            print(res);
        }

        [TestMethod]
        public void test_npv_irr_congruence_DOUBLE()
        {

            var cashflows = np.array(new Int32[] { -40000, 5000, 8000, 12000, 30000 });
            var res = npf.npv(cashflows, cashflows);
            AssertArray(res, new double[] { -39999.0000749843 });
            print(res);
        }

        [TestMethod]
        public void test_npv_irr_congruence_DECIMAL()
        {

            var cashflows = np.array(new decimal[] { -40000, 5000, 8000, 12000, 30000 });
            var res = npf.npv(cashflows, cashflows);
            AssertArray(res, new decimal[] { -39999.000074984309394346985692m });
            print(res);
        }

        #endregion

        #region npf.mirr tests

        [TestMethod]
        public void test_mirr_DOUBLE()
        {
            var val = new Int32[] { -4500, -800, 800, 800, 600, 600, 800, 800, 700, 3000 };
            var res = npf.mirr(val, 0.08, 0.055);
            AssertArray(res, new double[] { 0.0665971750315535 });
            print(res);

            val = new Int32[] { -120000, 39000, 30000, 21000, 37000, 46000 };
            res = npf.mirr(val, 0.10, 0.12);
            AssertArray(res, new double[] { 0.126094130365905 });
            print(res);

            val = new Int32[] { 100, 200, -50, 300, -200 };
            res = npf.mirr(val, 0.05, 0.06);
            AssertArray(res, new double[] { 0.342823387842177 });
            print(res);

            val = new Int32[] { 39000, 30000, 21000, 37000, 46000 };
            res = npf.mirr(val, 0.10, 0.12);
            AssertArray(res, new double[] { double.NaN });
            print(res);

        }

        [TestMethod]
        public void test_mirr_DECIMAL()
        {
            var val = new Int32[] { -4500, -800, 800, 800, 600, 600, 800, 800, 700, 3000 };
            var res = npf.mirr(val, 0.08m, 0.055m);
            AssertArray(res, new decimal[] { 0.0665971750315565m });
            print(res);

            val = new Int32[] { -120000, 39000, 30000, 21000, 37000, 46000 };
            res = npf.mirr(val, 0.10m, 0.12m);
            AssertArray(res, new decimal[] { 0.126094130365904m });
            print(res);

            val = new Int32[] { 100, 200, -50, 300, -200 };
            res = npf.mirr(val, 0.05m, 0.06m);
            AssertArray(res, new decimal[] { 0.3428233878421744m });
            print(res);

            val = new Int32[] { 39000, 30000, 21000, 37000, 46000 };
            res = npf.mirr(val, 0.10m, 0.12m);
            AssertArray(res, new double[] { double.NaN });  // NOTE: return double.NaN cuz decimals don't support NaN values.
            print(res);

        }

        #endregion
    }
}
