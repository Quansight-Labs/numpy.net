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
    public class UFUNCTests : TestBaseClass
    {
        [TestMethod]
        public void test_UFUNC_AddReduce_1()
        {
            var x = np.arange(8);

            var a = np.ufunc.add.reduce(x);
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.add.reduce(x);
            print(b);

            var c = np.ufunc.add.reduce(x, 0);
            print(c);

            var d = np.ufunc.add.reduce(x, 1);
            print(d);

            var e = np.ufunc.add.reduce(x, 2);
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_AddReduceAt_1()
        {
            var a = np.ufunc.add.reduceat(np.arange(8),new long[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"];
            print(a);

            double retstep = 0; 
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.add.reduceat(x, new long[] { 0, 3, 1, 2, 0 });
            print(b);

            var c = np.ufunc.multiply.reduceat(x, new long[] { 0, 3 }, axis : 1);
            print(c);


        }

        [TestMethod]
        public void test_UFUNC_AddOuter_1()
        {
            var x = np.arange(8);

            var a = np.ufunc.add.outer(x,x);
            print(a);

            //x = np.arange(8).reshape((2, 2, 2));
            //var b = np.ufunc.add.outer(x,x);
            //print(b);

        }

        //[TestMethod]
        //public void test_UFUNC_AddReduce_2()
        //{
        //    var x = np.arange(8).reshape((2, 2, 2));
        //    var b = np.ufunc.add.reduce(x);
        //    print(b);

        //    var c = np.ufunc.add.reduce(x, (0, 1));
        //    print(c);

        //    var d = np.ufunc.add.reduce(x, (1, 2));
        //    print(d);

        //    var e = np.ufunc.add.reduce(x, (2, 1));
        //    print(e);

        //}
    }
}
