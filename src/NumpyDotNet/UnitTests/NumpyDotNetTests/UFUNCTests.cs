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
        #region UFUNC ADD tests
        [TestMethod]
        public void test_UFUNC_AddAccumulate_1()
        {
            var x = np.arange(8);

            var a = np.ufunc.accumulate(NpyArray_Ops.npy_op_add, x);
            AssertArray(a, new int[] { 0, 1, 3, 6, 10, 15, 21, 28 });
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(NpyArray_Ops.npy_op_add, x);
            AssertArray(b, new int[,,] { { { 0, 1 }, { 2, 3 } }, {{ 4, 6 }, { 8, 10 } } });
            print(b);

            var c = np.ufunc.accumulate(NpyArray_Ops.npy_op_add, x, 0);
            AssertArray(c, new int[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 6 }, { 8, 10 } } });
            print(c);

            var d = np.ufunc.accumulate(NpyArray_Ops.npy_op_add, x, 1);
            AssertArray(d, new int[,,] { { { 0, 1 }, { 2, 4 } }, { { 4, 5 }, { 10, 12 } } });
            print(d);

            var e = np.ufunc.accumulate(NpyArray_Ops.npy_op_add, x, 2);
            AssertArray(e, new int[,,] { { { 0, 1 }, { 2, 5 } }, { { 4, 9 }, { 6, 13 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_AddReduce_1()
        {
            var x = np.arange(8);

            var a = np.ufunc.reduce(NpyArray_Ops.npy_op_add, x);
            Assert.AreEqual(28, a.GetItem(0));
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.reduce(NpyArray_Ops.npy_op_add, x);
            AssertArray(b, new int[,] { { 4, 6 }, { 8, 10 } });
            print(b);

            var c = np.ufunc.reduce(NpyArray_Ops.npy_op_add, x, 0);
            AssertArray(c, new int[,] { { 4, 6 }, { 8, 10 } });
            print(c);

            var d = np.ufunc.reduce(NpyArray_Ops.npy_op_add, x, 1);
            AssertArray(d, new int[,] { { 2, 4 }, { 10, 12 } });
            print(d);

            var e = np.ufunc.reduce(NpyArray_Ops.npy_op_add, x, 2);
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
            var a = np.ufunc.reduceat(NpyArray_Ops.npy_op_add, np.arange(8),new long[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new int[] { 6,10,14,18});
            print(a);

            double retstep = 0; 
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(NpyArray_Ops.npy_op_add, x, new long[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{12.0, 15.0, 18.0, 21.0},{12.0, 13.0, 14.0, 15.0}, {4.0, 5.0, 6.0, 7.0}, 
                                          {8.0, 9.0, 10.0, 11.0}, {24.0, 28.0, 32.0, 36.0}});
            print(b);

            var c = np.ufunc.reduceat(NpyArray_Ops.npy_op_multiply, x, new long[] { 0, 3 }, axis : 1);
            AssertArray(c, new double[,] { { 0.0, 3.0 }, { 120.0, 7.0 }, { 720.0, 11.0 }, { 2184.0, 15.0 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_AddOuter_1()
        {
            var x = np.arange(4);

            var a = np.ufunc.outer(NpyArray_Ops.npy_op_add, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new int[,] { { 0, 1, 2, 3 }, { 1, 2, 3, 4 }, { 2, 3, 4, 5 }, { 3, 4, 5, 6 } });
            print(a);

            x = np.arange(6).reshape((3, 2));
            var y = np.arange(6).reshape((2, 3));
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_add, null, x, y);
            AssertShape(b, 3, 2, 2, 3);
            print(b.shape);

            var ExpectedDataB = new int[,,,]

                {{{{0,  1,  2}, {3,  4,  5}}, {{1,  2,  3}, { 4,  5,  6}}},
                 {{{2,  3,  4}, {5,  6,  7}}, {{3,  4,  5}, { 6,  7,  8}}},
                 {{{4,  5,  6}, {7,  8,  9}}, {{5,  6,  7}, { 8,  9, 10}}}};

            AssertArray(b, ExpectedDataB);

            print(b);
        }

        #endregion


        #region UFUNC SUBTRACT tests
        [TestMethod]
        public void test_UFUNC_SubtractAccumulate_1()
        {
            var x = np.arange(8);

            var a = np.ufunc.accumulate(NpyArray_Ops.npy_op_subtract, x);
            AssertArray(a, new int[] { 0, -1, -3, -6, -10, -15, -21, -28 });
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(NpyArray_Ops.npy_op_subtract, x);
            AssertArray(b, new int[,,] { { { 0, 1 }, { 2, 3 } }, { { -4, -4 }, { -4, -4 } } });
            print(b);

            var c = np.ufunc.accumulate(NpyArray_Ops.npy_op_subtract, x, 0);
            AssertArray(c, new int[,,] { { { 0, 1 }, { 2, 3 } }, { { -4, -4 }, { -4, -4 } } });
            print(c);

            var d = np.ufunc.accumulate(NpyArray_Ops.npy_op_subtract, x, 1);
            AssertArray(d, new int[,,] { { { 0, 1 }, { -2, -2 } }, { { 4, 5 }, { -2, -2 } } });
            print(d);

            var e = np.ufunc.accumulate(NpyArray_Ops.npy_op_subtract, x, 2);
            AssertArray(e, new int[,,] { { { 0, -1 }, { 2, -1 } }, { { 4, -1 }, { 6, -1 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_SubtractReduce_1()
        {
            var x = np.arange(8);

            var a = np.ufunc.reduce(NpyArray_Ops.npy_op_subtract, x);
            Assert.AreEqual(-28, a.GetItem(0));
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.reduce(NpyArray_Ops.npy_op_subtract, x);
            AssertArray(b, new int[,] { { -4, -4 }, { -4, -4 } });
            print(b);

            var c = np.ufunc.reduce(NpyArray_Ops.npy_op_subtract, x, 0);
            AssertArray(c, new int[,] { { -4, -4 }, { -4, -4 } });
            print(c);

            var d = np.ufunc.reduce(NpyArray_Ops.npy_op_subtract, x, 1);
            AssertArray(d, new int[,] { { -2, -2 }, { -2, -2 } });
            print(d);

            var e = np.ufunc.reduce(NpyArray_Ops.npy_op_subtract, x, 2);
            AssertArray(e, new int[,] { { -1, -1 }, { -1, -1 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_SubtractReduceAt_1()
        {
            var a = np.ufunc.reduceat(NpyArray_Ops.npy_op_subtract, np.arange(8), new long[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new int[] { -6, -8, -10, -12 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(NpyArray_Ops.npy_op_subtract, x, new long[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{-12.0, -13.0, -14.0, -15.0},{12.0, 13.0, 14.0, 15.0}, {4.0, 5.0, 6.0, 7.0},
                                          {8.0, 9.0, 10.0, 11.0}, {-24.0, -26.0, -28.0, -30.0}});
            print(b);

            var c = np.ufunc.reduceat(NpyArray_Ops.npy_op_multiply, x, new long[] { 0, 3 }, axis: 1);
            AssertArray(c, new double[,] { { 0.0, 3.0 }, { 120.0, 7.0 }, { 720.0, 11.0 }, { 2184.0, 15.0 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_SubtractOuter_1()
        {
            var x = np.arange(4);

            var a = np.ufunc.outer(NpyArray_Ops.npy_op_subtract, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new int[,] { { 0, -1, -2, -3 }, { 1, 0, -1, -2 }, { 2, 1, 0, -1 }, { 3, 2, 1, 0 } });
            print(a);

            x = np.arange(6).reshape((3, 2));
            var y = np.arange(6).reshape((2, 3));
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_subtract, null, x, y);
            AssertShape(b, 3, 2, 2, 3);
            print(b.shape);

            var ExpectedDataB = new int[,,,]

                {{{{0,  -1,  -2}, {-3,  -4,  -5}}, {{1,  0,  -1}, {-2, -3, -4}}},
                 {{{2,  1,  0}, {-1,  -2,  -3}}, {{3,  2,  1}, { 0,  -1,  -2}}},
                 {{{4,  3,  2}, {1,  0,  -1}}, {{5,  4,  3}, { 2,  1, 0}}}};

            AssertArray(b, ExpectedDataB);

            print(b);
        }

        #endregion

        #region UFUNC MULTIPLY tests
        [TestMethod]
        public void test_UFUNC_MultiplyAccumulate_1()
        {
            var x = np.arange(8);

            var a = np.ufunc.accumulate(NpyArray_Ops.npy_op_multiply, x);
            AssertArray(a, new int[] { 0, 0, 0, 0, 0, 0, 0, 0 });
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(NpyArray_Ops.npy_op_multiply, x);
            AssertArray(b, new int[,,] { { { 0, 1 }, { 2, 3 } }, { { 0, 5 }, { 12, 21 } } });
            print(b);

            var c = np.ufunc.accumulate(NpyArray_Ops.npy_op_multiply, x, 0);
            AssertArray(c, new int[,,] { { { 0, 1 }, { 2, 3 } }, { { 0, 5 }, { 12, 21 } } });
            print(c);

            var d = np.ufunc.accumulate(NpyArray_Ops.npy_op_multiply, x, 1);
            AssertArray(d, new int[,,] { { { 0, 1 }, { 0, 3 } }, { { 4, 5 }, { 24, 35 } } });
            print(d);

            var e = np.ufunc.accumulate(NpyArray_Ops.npy_op_multiply, x, 2);
            AssertArray(e, new int[,,] { { { 0, 0 }, { 2, 6 } }, { { 4, 20 }, { 6, 42 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_MultiplyReduce_1()
        {
            var x = np.arange(8);

            var a = np.ufunc.reduce(NpyArray_Ops.npy_op_multiply, x);
            Assert.AreEqual(0, a.GetItem(0));
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.reduce(NpyArray_Ops.npy_op_multiply, x);
            AssertArray(b, new int[,] { { 0, 5 }, { 12, 21 } });
            print(b);

            var c = np.ufunc.reduce(NpyArray_Ops.npy_op_multiply, x, 0);
            AssertArray(c, new int[,] { { 0, 5 }, { 12, 21 } });
            print(c);

            var d = np.ufunc.reduce(NpyArray_Ops.npy_op_multiply, x, 1);
            AssertArray(d, new int[,] { { 0, 3 }, { 24, 35 } });
            print(d);

            var e = np.ufunc.reduce(NpyArray_Ops.npy_op_multiply, x, 2);
            AssertArray(e, new int[,] { { 0, 6 }, { 20, 42 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_MultiplyReduceAt_1()
        {
            var a = np.ufunc.reduceat(NpyArray_Ops.npy_op_multiply, np.arange(8), new long[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new int[] { 0, 24, 120, 360 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(NpyArray_Ops.npy_op_multiply, x, new long[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{0.0, 45.0, 120.0, 231.0},{12.0, 13.0, 14.0, 15.0}, {4.0, 5.0, 6.0, 7.0},
                                          {8.0, 9.0, 10.0, 11.0}, {0.0, 585.0, 1680.0, 3465.0}});
            print(b);

            var c = np.ufunc.reduceat(NpyArray_Ops.npy_op_multiply, x, new long[] { 0, 3 }, axis: 1);
            AssertArray(c, new double[,] { { 0.0, 3.0 }, { 120.0, 7.0 }, { 720.0, 11.0 }, { 2184.0, 15.0 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_MultiplyOuter_1()
        {
            var x = np.arange(4);

            var a = np.ufunc.outer(NpyArray_Ops.npy_op_multiply, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new int[,] { { 0, 0, 0, 0 }, { 0, 1, 2, 3 }, { 0, 2, 4, 6 }, { 0, 3, 6, 9 } });
            print(a);

            x = np.arange(6).reshape((3, 2));
            var y = np.arange(6).reshape((2, 3));
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_multiply, null, x, y);
            AssertShape(b, 3, 2, 2, 3);
            print(b.shape);

            var ExpectedDataB = new int[,,,]

                {{{{0,  0,  0}, {0,  0,  0}}, {{0,  1,  2}, {3, 4, 5}}},
                 {{{0,  2,  4}, {6,  8,  10}}, {{0,  3,  6}, { 9,  12,  15}}},
                 {{{0,  4,  8}, {12, 16, 20}}, {{0,  5,  10}, { 15, 20, 25}}}};

            AssertArray(b, ExpectedDataB);

            print(b);
        }

        #endregion

    }
}
