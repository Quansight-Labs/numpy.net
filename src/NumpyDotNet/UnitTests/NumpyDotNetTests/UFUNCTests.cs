using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumpyLib;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNetTests
{
    //internal class MyInt32Handlers : IArrayHandlers
    //{
    //    public MyInt32Handlers()
    //    {
    //        AddOperation = My_AddOperation;
    //        SubtractOperation = My_SubtractOperation;
    //        MultiplyOperation = My_MultiplyOperation;
    //        DivideOperation = My_DivideOperation;
    //        RemainderOperation = My_RemainderOperation;
    //        FModOperation = My_FModOperation;
    //        PowerOperation = My_PowerOperation;
    //        SquareOperation = My_SquareOperation;
    //        ReciprocalOperation = My_ReciprocalOperation;
    //    }

    //    public NumericOperation AddOperation { get; set; }
    //    public NumericOperation SubtractOperation { get; set; }
    //    public NumericOperation MultiplyOperation { get; set; }
    //    public NumericOperation DivideOperation { get; set; }
    //    public NumericOperation RemainderOperation { get; set; }
    //    public NumericOperation FModOperation { get; set; }
    //    public NumericOperation PowerOperation { get; set; }
    //    public NumericOperation SquareOperation { get; set; }
    //    public NumericOperation ReciprocalOperation { get; set; }

    //    private object My_AddOperation(object bValue, object operand)
    //    {
    //        Int32 dValue = (Int32)bValue;
    //        return dValue + (double)operand;
    //    }
    //    private static object My_SubtractOperation(object bValue, object operand)
    //    {
    //        Int32 dValue = (Int32)bValue;
    //        return dValue - (double)operand;
    //    }
    //    private static object My_MultiplyOperation(object bValue, object operand)
    //    {
    //        Int32 dValue = (Int32)bValue;
    //        return dValue * (double)operand;
    //    }
    //    private static object My_DivideOperation(object bValue, object operand)
    //    {
    //        Int32 dValue = (Int32)bValue;
    //        double doperand = (double)operand;
    //        if (doperand == 0)
    //        {
    //            dValue = 0;
    //            return dValue;
    //        }
    //        return dValue / doperand;
    //    }
    //    private static object My_RemainderOperation(object bValue, object operand)
    //    {
    //        Int32 dValue = (Int32)bValue;
    //        double doperand = (double)operand;
    //        if (doperand == 0)
    //        {
    //            dValue = 0;
    //            return dValue;
    //        }
    //        var rem = dValue % doperand;
    //        if ((dValue > 0) == (doperand > 0) || rem == 0)
    //        {
    //            return rem;
    //        }
    //        else
    //        {
    //            return rem + doperand;
    //        }
    //    }
    //    private static object My_FModOperation(object bValue, object operand)
    //    {
    //        Int32 dValue = (Int32)bValue;
    //        double doperand = (double)operand;
    //        if (doperand == 0)
    //        {
    //            dValue = 0;
    //            return dValue;
    //        }
    //        return dValue % doperand;
    //    }
    //    private static object My_PowerOperation(object bValue, object operand)
    //    {
    //        Int32 dValue = (Int32)bValue;
    //        return Math.Pow(dValue, (double)operand);
    //    }
    //    private static object My_SquareOperation(object bValue, object operand)
    //    {
    //        Int32 dValue = (Int32)bValue;
    //        return dValue * dValue;
    //    }
    //    private static object My_ReciprocalOperation(object bValue, object operand)
    //    {
    //        Int32 dValue = (Int32)bValue;
    //        return 1 / dValue;
    //    }
    //}

    [TestClass]
    public class UFUNCTests : TestBaseClass
    {
        private object MYTESTINT32_AddOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue + (double)operand;
        }

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
            var a = np.ufunc.reduceat(NpyArray_Ops.npy_op_add, np.arange(8),new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new int[] { 6,10,14,18});
            print(a);

            double retstep = 0; 
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(NpyArray_Ops.npy_op_add, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{12.0, 15.0, 18.0, 21.0},{12.0, 13.0, 14.0, 15.0}, {4.0, 5.0, 6.0, 7.0}, 
                                          {8.0, 9.0, 10.0, 11.0}, {24.0, 28.0, 32.0, 36.0}});
            print(b);

            var c = np.ufunc.reduceat(NpyArray_Ops.npy_op_multiply, x, new npy_intp[] { 0, 3 }, axis : 1);
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
            var a = np.ufunc.reduceat(NpyArray_Ops.npy_op_subtract, np.arange(8), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new int[] { -6, -8, -10, -12 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(NpyArray_Ops.npy_op_subtract, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{-12.0, -13.0, -14.0, -15.0},{12.0, 13.0, 14.0, 15.0}, {4.0, 5.0, 6.0, 7.0},
                                          {8.0, 9.0, 10.0, 11.0}, {-24.0, -26.0, -28.0, -30.0}});
            print(b);

            var c = np.ufunc.reduceat(NpyArray_Ops.npy_op_multiply, x, new npy_intp[] { 0, 3 }, axis: 1);
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
            var a = np.ufunc.reduceat(NpyArray_Ops.npy_op_multiply, np.arange(8), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new int[] { 0, 24, 120, 360 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(NpyArray_Ops.npy_op_multiply, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{0.0, 45.0, 120.0, 231.0},{12.0, 13.0, 14.0, 15.0}, {4.0, 5.0, 6.0, 7.0},
                                          {8.0, 9.0, 10.0, 11.0}, {0.0, 585.0, 1680.0, 3465.0}});
            print(b);

            var c = np.ufunc.reduceat(NpyArray_Ops.npy_op_multiply, x, new npy_intp[] { 0, 3 }, axis: 1);
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

        #region UFUNC DIVIDE tests
        [TestMethod]
        public void test_UFUNC_DivideAccumulate_1()
        {
            var x = np.arange(8, 16, dtype : np.Float64);

            var a = np.ufunc.accumulate(NpyArray_Ops.npy_op_divide, x);
            AssertArray(a, new double[] { 8.00000000e+00, 8.88888889e-01, 8.88888889e-02, 8.08080808e-03, 6.73400673e-04, 5.18000518e-05, 3.70000370e-06, 2.46666913e-07 });
            print(a);

            x = np.arange(8, 16, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(NpyArray_Ops.npy_op_divide, x);
            AssertArray(b, new double[,,] { { { 8, 9 }, { 10, 11 } }, { { 0.66666667, 0.69230769 }, { 0.71428571, 0.73333333 } } });
            print(b);

            var c = np.ufunc.accumulate(NpyArray_Ops.npy_op_divide, x, 0);
            AssertArray(c, new double[,,] { { { 8, 9 }, { 10, 11 } }, { { 0.66666667, 0.69230769 }, { 0.71428571, 0.73333333 } } });
            print(c);

            var d = np.ufunc.accumulate(NpyArray_Ops.npy_op_divide, x, 1);
            AssertArray(d, new double[,,] { { { 8, 9 }, { 0.8, 0.81818182 } }, { { 12, 13 }, { 0.85714286, 0.86666667 } } });
            print(d);

            var e = np.ufunc.accumulate(NpyArray_Ops.npy_op_divide, x, 2);
            AssertArray(e, new double[,,] { { { 8.0, 0.88888889 }, { 10.0, 0.90909091 } }, { { 12.0, 0.92307692 }, { 14.0, 0.93333333 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_DivideReduce_1()
        {
            var x = np.arange(8, 16, dtype: np.Float64);

            var a = np.ufunc.reduce(NpyArray_Ops.npy_op_divide, x);
            Assert.AreEqual(2.4666691333357994e-07, a.GetItem(0));
            print(a);

            x = np.arange(8, 16, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.reduce(NpyArray_Ops.npy_op_divide, x);
            AssertArray(b, new double[,] { { 0.66666667, 0.69230769 }, { 0.71428571, 0.73333333 } });
            print(b);

            var c = np.ufunc.reduce(NpyArray_Ops.npy_op_divide, x, 0);
            AssertArray(c, new double[,] { { 0.66666667, 0.69230769 }, { 0.71428571, 0.73333333 } });
            print(c);

            var d = np.ufunc.reduce(NpyArray_Ops.npy_op_divide, x, 1);
            AssertArray(d, new double[,] { { 0.8, 0.81818182 }, { 0.85714286, 0.86666667 } });
            print(d);

            var e = np.ufunc.reduce(NpyArray_Ops.npy_op_divide, x, 2);
            AssertArray(e, new double[,] { { 0.88888889, 0.90909091 }, { 0.92307692, 0.93333333 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_DivideReduceAt_1()
        {
            var a = np.ufunc.reduceat(NpyArray_Ops.npy_op_divide, np.arange(8, 16, dtype: np.Float64), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new double[] { 0.00808081, 0.00681818, 0.00582751, 0.00503663 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(NpyArray_Ops.npy_op_divide, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{0.00000000e+00, 2.22222222e-02, 3.33333333e-02, 3.89610390e-02},
                                          {1.20000000e+01, 1.30000000e+01, 1.40000000e+01, 1.50000000e+01}, 
                                          {4.00000000e+00, 5.00000000e+00, 6.00000000e+00, 7.00000000e+00},
                                          {8.00000000e+00, 9.00000000e+00, 1.00000000e+01, 1.10000000e+01}, 
                                          {0.00000000e+00, 1.70940171e-03, 2.38095238e-03, 2.59740260e-03}});
            print(b);

            var c = np.ufunc.reduceat(NpyArray_Ops.npy_op_divide, x, new npy_intp[] { 0, 3 }, axis: 1);
            AssertArray(c, new double[,] { { 0.0, 3.0 }, { 0.13333333,  7.0  }, { 0.08888889, 11.0  }, { 0.06593407, 15.0  } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_DivideOuter_1()
        {
            var x = np.arange(4, 8, dtype: np.Float64);

            var a = np.ufunc.outer(NpyArray_Ops.npy_op_divide, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new double[,] { { 1.0, 0.8, 0.66666667, 0.57142857 },
                                        {1.25, 1.0, 0.83333333, 0.71428571 },
                                        { 1.5, 1.2, 1.0, 0.85714286 },
                                        { 1.75,1.4, 1.16666667, 1.0 } });
            print(a);

            x = np.arange(8,14, dtype:np.Float64).reshape((3, 2));
            var y = np.arange(8, 14, dtype: np.Float64).reshape((2, 3));
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_divide, null, x, y);
            AssertShape(b, 3, 2, 2, 3);
            print(b.shape);

            var ExpectedDataB = new double[,,,]

                {{{{1.0, 0.88888889, 0.8 }, {0.72727273, 0.66666667, 0.61538462}}, {{1.125, 1.0, 0.9},        {0.81818182, 0.75, 0.69230769}}},
                 {{{1.25, 1.11111111, 1.0}, {0.90909091, 0.83333333, 0.76923077}}, {{1.375, 1.22222222, 1.1}, {1.0, 0.91666667, 0.84615385}}},
                 {{{1.5, 1.33333333, 1.2 }, {1.09090909, 1.0,  0.92307692}},       {{1.625, 1.44444444, 1.3}, {1.18181818, 1.08333333, 1.0}}}};

            //AssertArray(b, ExpectedDataB);

            print(b);
        }

        #endregion

        #region UFUNC REMAINDER tests
        [TestMethod]
        public void test_UFUNC_RemainderAccumulate_1()
        {
            var x = np.arange(16, 8, -1, dtype: np.Float64);

            var a = np.ufunc.accumulate(NpyArray_Ops.npy_op_remainder, x);
            AssertArray(a, new double[] { 16,  1,  1,  1,  1,  1,  1,  1 });
            print(a);

            x = np.arange(16, 8, -1, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(NpyArray_Ops.npy_op_remainder, x);
            AssertArray(b, new double[,,] { { { 16, 15 }, { 14, 13 } }, { { 4,4 }, { 4,4 } } });
            print(b);

            var c = np.ufunc.accumulate(NpyArray_Ops.npy_op_remainder, x, 0);
            AssertArray(c, new double[,,] { { { 16, 15 }, { 14, 13 } }, { { 4, 4 }, { 4, 4 } } });
            print(c);

            var d = np.ufunc.accumulate(NpyArray_Ops.npy_op_remainder, x, 1);
            AssertArray(d, new double[,,] { { { 16, 15 }, { 2, 2 } }, { { 12, 11 }, { 2, 2 } } });
            print(d);

            var e = np.ufunc.accumulate(NpyArray_Ops.npy_op_remainder, x, 2);
            AssertArray(e, new double[,,] { { { 16, 1 }, { 14, 1 } }, { { 12, 1 }, { 10, 1 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_RemainderReduce_1()
        {
            var x = np.arange(16,8,-1, dtype: np.Float64);

            var a = np.ufunc.reduce(NpyArray_Ops.npy_op_remainder, x);
            Assert.AreEqual(1.0, a.GetItem(0));
            print(a);

            x = np.arange(16,8,-1, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.reduce(NpyArray_Ops.npy_op_remainder, x);
            AssertArray(b, new double[,] { { 4, 4 }, { 4, 4 } });
            print(b);

            var c = np.ufunc.reduce(NpyArray_Ops.npy_op_remainder, x, 0);
            AssertArray(c, new double[,] { { 4, 4 }, { 4, 4 } });
            print(c);

            var d = np.ufunc.reduce(NpyArray_Ops.npy_op_remainder, x, 1);
            AssertArray(d, new double[,] { { 2, 2 }, { 2, 2 } });
            print(d);

            var e = np.ufunc.reduce(NpyArray_Ops.npy_op_remainder, x, 2);
            AssertArray(e, new double[,] { { 1, 1 }, { 1, 1 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_RemainderReduceAt_1()
        {
            var a = np.ufunc.reduceat(NpyArray_Ops.npy_op_remainder, np.arange(16, 8, -1, dtype: np.Float64), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new double[] { 1,1,1,1 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(NpyArray_Ops.npy_op_remainder, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{0,1,2,3},
                                          {12,13,14,15},
                                          {4,5,6,7},
                                          {8,9,10,11},
                                          {0,1,2,3}});
            print(b);

            var c = np.ufunc.reduceat(NpyArray_Ops.npy_op_remainder, x, new npy_intp[] { 0, 3 }, axis: 1);
            AssertArray(c, new double[,] { { 0, 3 }, { 4, 7 }, { 8, 11 }, { 12, 15 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_RemainderOuter_1()
        {
            var x = np.arange(4, 8, dtype: np.Float64);

            var a = np.ufunc.outer(NpyArray_Ops.npy_op_remainder, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new double[,] { { 0,4,4,4 },
                                           { 1,0,5,5 },
                                           { 2,1,0,6 },
                                           { 3,2,1,0 } });
            print(a);

            x = np.arange(14, 8, -1, dtype: np.Float64).reshape((3, 2));
            var y = np.arange(14, 8, -1, dtype: np.Float64).reshape((2, 3));
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_remainder, null, x, y);
            AssertShape(b, 3, 2, 2, 3);
            print(b.shape);

            var ExpectedDataB = new double[,,,]

                {{{{0,1,2 },    {3,4,5}},   {{13,0,1},   {2,3,4}}},
                 {{{12,12,0},   {1,2,3}},   {{11,11,11}, {0,1,2}}},
                 {{{10,10,10 }, {10,0,1}},  {{9,9,9},    {9,9,0}}}};

            AssertArray(b, ExpectedDataB);

            print(b);
        }

        #endregion

        #region UFUNC FMOD tests
        [TestMethod]
        public void test_UFUNC_FModAccumulate_1()
        {
            var x = np.arange(16, 8, -1, dtype: np.Float64);

            var a = np.ufunc.accumulate(NpyArray_Ops.npy_op_fmod, x);
            AssertArray(a, new double[] { 16, 1, 1, 1, 1, 1, 1, 1 });
            print(a);

            x = np.arange(16, 8, -1, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(NpyArray_Ops.npy_op_fmod, x);
            AssertArray(b, new double[,,] { { { 16, 15 }, { 14, 13 } }, { { 4, 4 }, { 4, 4 } } });
            print(b);

            var c = np.ufunc.accumulate(NpyArray_Ops.npy_op_fmod, x, 0);
            AssertArray(c, new double[,,] { { { 16, 15 }, { 14, 13 } }, { { 4, 4 }, { 4, 4 } } });
            print(c);

            var d = np.ufunc.accumulate(NpyArray_Ops.npy_op_fmod, x, 1);
            AssertArray(d, new double[,,] { { { 16, 15 }, { 2, 2 } }, { { 12, 11 }, { 2, 2 } } });
            print(d);

            var e = np.ufunc.accumulate(NpyArray_Ops.npy_op_fmod, x, 2);
            AssertArray(e, new double[,,] { { { 16, 1 }, { 14, 1 } }, { { 12, 1 }, { 10, 1 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_FModReduce_1()
        {
            var x = np.arange(16, 8, -1, dtype: np.Float64);

            var a = np.ufunc.reduce(NpyArray_Ops.npy_op_fmod, x);
            Assert.AreEqual(1.0, a.GetItem(0));
            print(a);

            x = np.arange(16, 8, -1, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.reduce(NpyArray_Ops.npy_op_fmod, x);
            AssertArray(b, new double[,] { { 4, 4 }, { 4, 4 } });
            print(b);

            var c = np.ufunc.reduce(NpyArray_Ops.npy_op_fmod, x, 0);
            AssertArray(c, new double[,] { { 4, 4 }, { 4, 4 } });
            print(c);

            var d = np.ufunc.reduce(NpyArray_Ops.npy_op_fmod, x, 1);
            AssertArray(d, new double[,] { { 2, 2 }, { 2, 2 } });
            print(d);

            var e = np.ufunc.reduce(NpyArray_Ops.npy_op_fmod, x, 2);
            AssertArray(e, new double[,] { { 1, 1 }, { 1, 1 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_FModReduceAt_1()
        {
            var a = np.ufunc.reduceat(NpyArray_Ops.npy_op_fmod, np.arange(16, 8, -1, dtype: np.Float64), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new double[] { 1, 1, 1, 1 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(NpyArray_Ops.npy_op_fmod, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{0,1,2,3},
                                          {12,13,14,15},
                                          {4,5,6,7},
                                          {8,9,10,11},
                                          {0,1,2,3}});
            print(b);

            var c = np.ufunc.reduceat(NpyArray_Ops.npy_op_fmod, x, new npy_intp[] { 0, 3 }, axis: 1);
            AssertArray(c, new double[,] { { 0, 3 }, { 4, 7 }, { 8, 11 }, { 12, 15 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_FModOuter_1()
        {
            var x = np.arange(4, 8, dtype: np.Float64);

            var a = np.ufunc.outer(NpyArray_Ops.npy_op_fmod, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new double[,] { { 0,4,4,4 },
                                           { 1,0,5,5 },
                                           { 2,1,0,6 },
                                           { 3,2,1,0 } });
            print(a);

            x = np.arange(14, 8,-1, dtype: np.Float64).reshape((3, 2));
            var y = np.arange(14, 8, -1, dtype: np.Float64).reshape((2, 3));
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_fmod, null, x, y);
            AssertShape(b, 3, 2, 2, 3);
            print(b.shape);

            var ExpectedDataB = new double[,,,]

                {{{{0,1,2 },    {3,4,5}},   {{13,0,1},   {2,3,4}}},
                 {{{12,12,0},   {1,2,3}},   {{11,11,11}, {0,1,2}}},
                 {{{10,10,10 }, {10,0,1}},  {{9,9,9},    {9,9,0}}}};

            AssertArray(b, ExpectedDataB);

            print(b);
        }

        #endregion

        #region UFUNC POWER tests
        [TestMethod]
        public void test_UFUNC_PowerAccumulate_1()
        {
            var x = np.arange(16, 8, -1, dtype: np.Float64);

            var a = np.ufunc.accumulate(NpyArray_Ops.npy_op_power, x);
            AssertArray(a, new double[] {1.6000000e+001, 1.152921504606847E+18, 7.33155940312959E+252, double.PositiveInfinity,
                                        double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity });
            print(a);

            x = np.arange(16, 8, -1, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(NpyArray_Ops.npy_op_power, x);
            AssertArray(b, new double[,,] { { { 1.60000000e+01, 1.50000000e+01 }, { 1.40000000e+01, 1.30000000e+01 } }, 
                                            { { 281474976710656, 8649755859375 }, { 289254654976, 10604499373 } } });

            


            print(b);

            var c = np.ufunc.accumulate(NpyArray_Ops.npy_op_power, x, 0);
            AssertArray(c, new double[,,] { { { 1.60000000e+01, 1.50000000e+01 }, { 1.40000000e+01, 1.30000000e+01 } },
                                            { { 281474976710656, 8649755859375 }, { 289254654976, 10604499373 } } });
            print(c);

            var d = np.ufunc.accumulate(NpyArray_Ops.npy_op_power, x, 1);
            AssertArray(d, new double[,,] { { { 1.60000000e+01, 1.50000000e+01 }, { 72057594037927936, 1946195068359375 } }, 
                                            { { 1.20000000e+01, 1.10000000e+01 }, { 61917364224, 2357947691 } } });
            print(d);

            var e = np.ufunc.accumulate(NpyArray_Ops.npy_op_power, x, 2);
            AssertArray(e, new double[,,] { { { 1.60000000e+01, 1.152921504606847E+18 }, { 1.40000000e+01, 793714773254144 } }, 
                                            { { 1.20000000e+01, 743008370688 }, { 1.00000000e+01, 1.00000000e+09} } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_PowerReduce_1()
        {
            var x = np.arange(16, 8, -1, dtype: np.Float64);

            var a = np.ufunc.reduce(NpyArray_Ops.npy_op_power, x);
            Assert.AreEqual(double.PositiveInfinity, a.GetItem(0));
            print(a);

            x = np.arange(16, 8, -1, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.reduce(NpyArray_Ops.npy_op_power, x);
            AssertArray(b, new double[,] { { 281474976710656, 8649755859375 }, { 289254654976, 10604499373 } });
            print(b);

            var c = np.ufunc.reduce(NpyArray_Ops.npy_op_power, x, 0);
            AssertArray(c, new double[,] { { 281474976710656, 8649755859375 }, { 289254654976, 10604499373 } });
            print(c);

            var d = np.ufunc.reduce(NpyArray_Ops.npy_op_power, x, 1);
            AssertArray(d, new double[,] { { 72057594037927936, 1946195068359375 }, { 61917364224, 2357947691 } });
            print(d);

            var e = np.ufunc.reduce(NpyArray_Ops.npy_op_power, x, 2);
            AssertArray(e, new double[,] { { 1.152921504606847E+18, 793714773254144 }, { 743008370688, 1.00000000e+09 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_PowerReduceAt_1()
        {
            var a = np.ufunc.reduceat(NpyArray_Ops.npy_op_power, np.arange(16, 8, -1, dtype: np.Float64), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new double[] { double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(NpyArray_Ops.npy_op_power, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{0.00000000e+000, 1.00000000e+000, 1.152921504606847E+18, 5.4744010894202194E+36},
                                          {1.20000000e+001, 1.30000000e+001, 1.40000000e+001, 1.50000000e+001},
                                          {4.00000000e+000, 5.00000000e+000, 6.00000000e+000, 7.00000000e+000},
                                          {8.00000000e+000, 9.00000000e+000, 1.00000000e+001, 1.10000000e+001},
                                          {0.00000000e+000, 1.00000000e+000, 7.331559403129590e+252, double.PositiveInfinity}});
            print(b);

            var c = np.ufunc.reduceat(NpyArray_Ops.npy_op_power, x, new npy_intp[] { 0, 3 }, axis: 1);
            AssertArray(c, new double[,] { { 0.00000000e+000, 3.00000000e+000 }, { 1.152921504606847E+18, 7.00000000e+000 }, 
                                           { 1.8971375900641885E+81, 1.10000000e+001 }, { 2.5762427384904039E+196, 1.50000000e+001 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_PowerOuter_1()
        {
            var x = np.arange(4, 8, dtype: np.Float64);

            var a = np.ufunc.outer(NpyArray_Ops.npy_op_power, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new double[,] { { 2.56000e+02, 1.02400e+03, 4.09600e+03, 1.63840e+04 },
                                           { 6.25000e+02, 3.12500e+03, 1.56250e+04, 7.81250e+04 },
                                           { 1.29600e+03, 7.77600e+03, 4.66560e+04, 2.79936e+05 },
                                           { 2.40100e+03, 1.68070e+04, 1.17649e+05, 8.23543e+05} });
            print(a);

            x = np.arange(14, 8, -1, dtype: np.Float64).reshape((3, 2));
            var y = np.arange(14, 8, -1, dtype: np.Float64).reshape((2, 3));
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_power, null, x, y);
            AssertShape(b, 3, 2, 2, 3);
            print(b.shape);

            var ExpectedDataB = new double[,,,]

                {{{{1.1112006825558016e+16, 7.9371477325414400e+14, 5.6693912375296000e+13}, {4.0495651696640000e+12, 2.8925465497600000e+11, 2.0661046784000000e+10}},   
                  {{3.9373763856992890e+15, 3.0287510659225300e+14, 2.3298085122481000e+13}, {1.7921603940370000e+12, 1.3785849184900000e+11, 1.0604499373000000e+10}}},
                 {{{1.2839184645488640e+15, 1.0699320537907200e+14, 8.9161004482560000e+12}, {7.4300837068800000e+11, 6.1917364224000000e+10, 5.1597803520000000e+09}},   
                  {{3.7974983358324100e+14, 3.4522712143931000e+13, 3.1384283767210000e+12}, {2.8531167061100000e+11, 2.5937424601000000e+10, 2.3579476910000000e+09}}},
                 {{{1.0000000000000000e+14, 1.0000000000000000e+13, 1.0000000000000000e+12}, {1.0000000000000000e+11, 1.0000000000000000e+10, 1.0000000000000000e+09}},  
                  {{2.2876792454961000e+13, 2.5418658283290000e+12, 2.8242953648100000e+11}, {3.1381059609000000e+10, 3.4867844010000000e+09, 3.8742048900000000e+08}}}};

            AssertArray(b, ExpectedDataB);

            print(b);
        }

        #endregion

        #region UFUNC DOUBLE Tests

        [TestMethod]
        public void test_AddOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_add, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                    {{3,  4,  5,  6,  7},
                     {4,  5,  6,  7,  8},
                     {5,  6,  7,  8,  9},
                     {6,  7,  8,  9, 10},
                     {7,  8,  9, 10, 11}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_subtract, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                    {{-3, -4, -5, -6, -7},
                     {-2, -3, -4, -5, -6},
                     {-1, -2, -3, -4, -5},
                     {0, -1, -2, -3, -4},
                     {1,  0, -1, -2, -3}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_multiply, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                    {{0,  0,  0,  0,  0,},
                     {3,  4,  5,  6,  7,},
                     {6,  8, 10, 12, 14},
                     {9, 12, 15, 18, 21,},
                     {12, 16, 20, 24, 28,}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_divide, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                    {{0,  0,  0,  0,  0,},
                     {0.33333333, 0.25,       0.2,        0.16666667, 0.14285714},
                     {0.66666667, 0.5,        0.4,        0.33333333, 0.28571429},
                     {1,         0.75,       0.6,        0.5,        0.42857143},
                     {1.33333333, 1,         0.8,        0.66666667, 0.57142857}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_remainder, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                    {{0, 0, 0, 0, 0,},
                     {1, 1, 1, 1, 1,},
                     {2, 2, 2, 2, 2},
                     {0, 3, 3, 3, 3,},
                     {1, 0, 4, 4, 4,}};

            AssertArray(b, ExpectedData);


        }

        [TestMethod]
        public void test_FModOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_fmod, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                    {{0, 0, 0, 0, 0,},
                     {1, 1, 1, 1, 1,},
                     {2, 2, 2, 2, 2},
                     {0, 3, 3, 3, 3,},
                     {1, 0, 4, 4, 4,}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_square, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { 0.0, 0.0, 0.0, 0.0, 0.0 },
                  { 1.0, 1.0, 1.0, 1.0, 1.0 },
                  { 4.0, 4.0, 4.0, 4.0, 4.0 },
                  { 9.0, 9.0, 9.0, 9.0, 9.0 },
                  { 16.0, 16.0, 16.0, 16.0, 16.0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_reciprocal, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { Double.PositiveInfinity, Double.PositiveInfinity, Double.PositiveInfinity, Double.PositiveInfinity,Double.PositiveInfinity },
                  { 1.0, 1.0, 1.0, 1.0, 1.0 },
                  { 0.5, 0.5, 0.5, 0.5, 0.5 },
                  { 0.333333333333333, 0.333333333333333, 0.333333333333333, 0.333333333333333, 0.333333333333333 },
                  { 0.25, 0.25, 0.25, 0.25, 0.25 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_ones_like, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { 1.0, 1.0, 1.0, 1.0, 1.0 },
                  { 1.0, 1.0, 1.0, 1.0, 1.0 },
                  { 1.0, 1.0, 1.0, 1.0, 1.0 },
                  { 1.0, 1.0, 1.0, 1.0, 1.0 },
                  { 1.0, 1.0, 1.0, 1.0, 1.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_sqrt, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { 0.0, 0.0, 0.0, 0.0, 0.0 },
                  { 1.0, 1.0, 1.0, 1.0, 1.0 },
                  { 1.4142135623731, 1.4142135623731, 1.4142135623731, 1.4142135623731, 1.4142135623731  },
                  { 1.73205080756888, 1.73205080756888, 1.73205080756888, 1.73205080756888, 1.73205080756888 },
                  { 2.0, 2.0, 2.0, 2.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_negative, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { 0.0, 0.0, 0.0, 0.0, 0.0 },
                  { -1.0, -1.0, -1.0, -1.0, -1.0 },
                  { -2.0, -2.0, -2.0, -2.0, -2.0 },
                  { -3.0, -3.0, -3.0, -3.0, -3.0 },
                  {-4.0, -4.0, -4.0, -4.0, -4.0  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_absolute, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { 0.0, 0.0, 0.0, 0.0, 0.0 },
                  { 1.0, 1.0, 1.0, 1.0, 1.0 },
                  { 2.0, 2.0, 2.0, 2.0, 2.0 },
                  { 3.0, 3.0, 3.0, 3.0, 3.0 },
                  { 4.0, 4.0, 4.0, 4.0, 4.0  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_invert, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { 0.0, 0.0, 0.0, 0.0, 0.0 },
                  { 1.0, 1.0, 1.0, 1.0, 1.0 },
                  { 2.0, 2.0, 2.0, 2.0, 2.0 },
                  { 3.0, 3.0, 3.0, 3.0, 3.0 },
                  { 4.0, 4.0, 4.0, 4.0, 4.0  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_left_shift, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                  { { 0.0, 0.0, 0.0, 0.0, 0.0 },
                  { 8.0, 16.0, 32.0, 64.0, 128.0 },
                  { 16.0, 32.0, 64.0, 128.0, 256.0 },
                  { 24.0, 48.0, 96.0, 192.0, 384.0 },
                  { 32.0, 64.0, 128.0, 256.0, 512.0  } };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64) * 1024 * 4;
            var a2 = np.arange(3, 8, dtype: np.Float64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_right_shift, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { 0.0, 0.0, 0.0, 0.0, 0.0  },
                  { 512.0, 256.0, 128.0, 64.0, 32.0 },
                  { 1024.0, 512.0, 256.0, 128.0, 64.0  },
                  { 1536.0, 768.0, 384.0, 192.0, 96.0  },
                  { 2048.0, 1024.0, 512.0, 256.0, 128.0  } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_bitwise_and, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { {  0.0, 0.0, 0.0, 0.0, 0.0  },
                  { 1.0, 0.0, 1.0, 0.0, 1.0  },
                  { 2.0, 0.0, 0.0, 2.0, 2.0  },
                  { 3.0, 0.0, 1.0, 2.0, 3.0  },
                  { 0.0, 4.0, 4.0, 4.0, 4.0  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_bitwise_or, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { 3.0, 4.0, 5.0, 6.0, 7.0 },
                  { 3.0, 5.0, 5.0, 7.0, 7.0 },
                  { 3.0, 6.0, 7.0, 6.0, 7.0 },
                  { 3.0, 7.0, 7.0, 7.0, 7.0 },
                  { 7.0, 4.0, 5.0, 6.0, 7.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_bitwise_xor, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { 3.0, 4.0, 5.0, 6.0, 7.0 },
                  { 2.0, 5.0, 4.0, 7.0, 6.0 },
                  { 1.0, 6.0, 7.0, 4.0, 5.0 },
                  { 0.0, 7.0, 6.0, 5.0, 4.0 },
                  { 7.0, 0.0, 1.0, 2.0, 3.0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype : np.Float64);
            var a2 = np.arange(3, 8, dtype : np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_less, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true }, 
                                         { true, true, true, true, true }, 
                                         { true, true, true, true, true },
                                         { false, true, true, true, true }, 
                                         { false, false, true, true, true } });

        }


        [TestMethod]
        public void test_LessEqualOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_less_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true } });

        }

        [TestMethod]
        public void test_EqualOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false},
                                        {false, true, false, false, false}});

        }


        [TestMethod]
        public void test_NotEqualOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_not_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true },
                                         { true, false, true, true, true } });

        }

        [TestMethod]
        public void test_GreaterOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_greater, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false}});

        }

        [TestMethod]
        public void test_GreaterEqualOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_greater_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false},
                                        {true, true, false, false, false}});

        }

        [TestMethod]
        public void test_FloorDivideOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_floor_divide, null, a1, a2);
            print(b);


            var ExpectedData = new double[,]
                { { 0.0, 0.0, 0.0, 0.0, 0.0 },
                  { 0.0, 0.0, 0.0, 0.0, 0.0 },
                  { 0.0, 0.0, 0.0, 0.0, 0.0 },
                  { 1.0, 0.0, 0.0, 0.0, 0.0 },
                  { 1.0, 1.0, 0.0, 0.0, 0.0} };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_TrueDivideOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_true_divide, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                      {{0,  0,  0,  0,  0,},
                     {0.33333333, 0.25,       0.2,        0.16666667, 0.14285714},
                     {0.66666667, 0.5,        0.4,        0.33333333, 0.28571429},
                     {1,         0.75,       0.6,        0.5,        0.42857143},
                     {1.33333333, 1,         0.8,        0.66666667, 0.57142857}};

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_logical_and, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true}});

        }

        [TestMethod]
        public void test_LogicalOrOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(NpyArray_Ops.npy_op_logical_or, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true}});

        }



        #endregion
    }
}
