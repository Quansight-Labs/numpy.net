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
        [Ignore]
        [TestMethod]
        public void test_fv_int()
        {
            var x = npf.fv(75, 20, -2000, 0, 0);
            print(x);
        }

        [Ignore]
        [TestMethod]
        public void test_fv_float()
        {
            var x = npf.fv(0.075, 20, -2000, 0, 0);
            print(x);
        }

        [Ignore]
        [TestMethod]
        public void test_fv_decimal()
        {
            var x = npf.fv(0.075m, 20m, -2000m, 0, 0);
            print(x);
        }

        [Ignore]
        [TestMethod]
        public void test_fv_when_is_begin_float()
        {
            var x = npf.fv(0.075, 20, -2000, 0, "begin");
            print(x);
        }

        [Ignore]
        [TestMethod]
        public void test_fv_when_is_begin_decimal()
        {
            var x = npf.fv(0.075m, 20m, -2000m, 0, "begin");
            print(x);
        }

        [Ignore]
        [TestMethod]
        public void test_fv_when_is_end_float()
        {
            var x = npf.fv(0.075, 20, -2000, 0, "end");
            print(x);
        }

        [Ignore]
        [TestMethod]
        public void test_fv_when_is_end_decimal()
        {
            var x = npf.fv(0.075m, 20m, -2000m, 0, "begin");
            print(x);
        }

        [Ignore]
        [TestMethod]
        public void test_fv_broadcast()
        {
            var result = npf.fv(new double[,] { { 0.1 }, { 0.2 } }, 5, 100, 0, new int[] { 0, 1 });
            print(result);
        }

        [Ignore]
        [TestMethod]
        public void test_fv_some_rates_zero()
        {
            var result = npf.fv(new double[,] { { 0.0 }, { 0.1 } }, 5, 100, 0, 0);
            print(result);
        }
    }
}
