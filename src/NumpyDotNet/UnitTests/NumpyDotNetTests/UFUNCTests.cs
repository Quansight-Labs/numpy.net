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
            Assert.AreEqual(28, a.GetItem(0));
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.add.reduce(x);
            AssertArray(b, new int[,] { {4,6}, {8,10} });
            print(b);

            var c = np.ufunc.add.reduce(x, 0);
            AssertArray(c, new int[,] { { 4, 6 }, { 8, 10 } });
            print(c);

            var d = np.ufunc.add.reduce(x, 1);
            AssertArray(d, new int[,] { { 2, 4 }, { 10, 12 } });
            print(d);

            var e = np.ufunc.add.reduce(x, 2);
            AssertArray(e, new int[,] { { 1, 5 }, { 9, 13 } });
            print(e);

        }

        #if NOT_PLANNING_TODO
        [Ignore]  // don't currently support reduce on multiple axis
        [TestMethod]
        public void test_UFUNC_AddReduce_2()
        {
            var x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.add.reduce(x);
            print(b);

            var c = np.ufunc.add.reduce(x, (0, 1));
            print(c);

            var d = np.ufunc.add.reduce(x, (1, 2));
            print(d);

            var e = np.ufunc.add.reduce(x, (2, 1));
            print(e);

        }
        #endif

        [TestMethod]
        public void test_UFUNC_AddReduceAt_1()
        {
            var a = np.ufunc.add.reduceat(np.arange(8),new long[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new int[] { 6,10,14,18});
            print(a);

            double retstep = 0; 
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.add.reduceat(x, new long[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{12.0, 15.0, 18.0, 21.0},{12.0, 13.0, 14.0, 15.0}, {4.0, 5.0, 6.0, 7.0}, 
                                          {8.0, 9.0, 10.0, 11.0}, {24.0, 28.0, 32.0, 36.0}});
            print(b);

            var c = np.ufunc.multiply.reduceat(x, new long[] { 0, 3 }, axis : 1);
            AssertArray(c, new double[,] { { 0.0, 3.0 }, { 120.0, 7.0 }, { 720.0, 11.0 }, { 2184.0, 15.0 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_AddOuter_1()
        {
            var x = np.arange(4);

            var a = np.ufunc.add.outer(x,x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new int[,] { { 0, 1, 2, 3 }, { 1, 2, 3, 4 }, { 2, 3, 4, 5 }, { 3, 4, 5, 6 } });
            print(a);

            x = np.arange(6).reshape((3, 2));
            var y = np.arange(6).reshape((2, 3));
            var b = np.ufunc.add.outer(x, y);
            AssertShape(b, 3, 2, 2, 3);
            print(b.shape);

            var ExpectedDataB = new int[,,,]

                {{{{0,  1,  2}, {3,  4,  5}}, {{1,  2,  3}, { 4,  5,  6}}},
                 {{{2,  3,  4}, {5,  6,  7}}, {{3,  4,  5}, { 6,  7,  8}}},
                 {{{4,  5,  6}, {7,  8,  9}}, {{5,  6,  7}, { 8,  9, 10}}}};

            AssertArray(b, ExpectedDataB);

            print(b);
        }

    
    }
}
