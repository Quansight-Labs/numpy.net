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
