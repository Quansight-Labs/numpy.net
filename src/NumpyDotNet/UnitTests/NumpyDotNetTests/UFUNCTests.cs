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

            var a = np.ufunc.accumulate(UFuncOperation.add, x);
            AssertArray(a, new int[] { 0, 1, 3, 6, 10, 15, 21, 28 });
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(UFuncOperation.add, x);
            AssertArray(b, new int[,,] { { { 0, 1 }, { 2, 3 } }, {{ 4, 6 }, { 8, 10 } } });
            print(b);

            var c = np.ufunc.accumulate(UFuncOperation.add, x, 0);
            AssertArray(c, new int[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 6 }, { 8, 10 } } });
            print(c);

            var d = np.ufunc.accumulate(UFuncOperation.add, x, 1);
            AssertArray(d, new int[,,] { { { 0, 1 }, { 2, 4 } }, { { 4, 5 }, { 10, 12 } } });
            print(d);

            var e = np.ufunc.accumulate(UFuncOperation.add, x, 2);
            AssertArray(e, new int[,,] { { { 0, 1 }, { 2, 5 } }, { { 4, 9 }, { 6, 13 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_AddReduce_1()
        {
            var x = np.arange(8);

            var a = np.ufunc.reduce(UFuncOperation.add, x);
            Assert.AreEqual(28, a.GetItem(0));
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.reduce(UFuncOperation.add, x);
            AssertArray(b, new int[,] { { 4, 6 }, { 8, 10 } });
            print(b);

            var c = np.ufunc.reduce(UFuncOperation.add, x, 0);
            AssertArray(c, new int[,] { { 4, 6 }, { 8, 10 } });
            print(c);

            var d = np.ufunc.reduce(UFuncOperation.add, x, 1);
            AssertArray(d, new int[,] { { 2, 4 }, { 10, 12 } });
            print(d);

            var e = np.ufunc.reduce(UFuncOperation.add, x, 2);
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
            var a = np.ufunc.reduceat(UFuncOperation.add, np.arange(8),new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new int[] { 6,10,14,18});
            print(a);

            double retstep = 0; 
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(UFuncOperation.add, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{12.0, 15.0, 18.0, 21.0},{12.0, 13.0, 14.0, 15.0}, {4.0, 5.0, 6.0, 7.0}, 
                                          {8.0, 9.0, 10.0, 11.0}, {24.0, 28.0, 32.0, 36.0}});
            print(b);

            var c = np.ufunc.reduceat(UFuncOperation.multiply, x, new npy_intp[] { 0, 3 }, axis : 1);
            AssertArray(c, new double[,] { { 0.0, 3.0 }, { 120.0, 7.0 }, { 720.0, 11.0 }, { 2184.0, 15.0 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_AddOuter_1()
        {
            var x = np.arange(4);

            var a = np.ufunc.outer(UFuncOperation.add, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new int[,] { { 0, 1, 2, 3 }, { 1, 2, 3, 4 }, { 2, 3, 4, 5 }, { 3, 4, 5, 6 } });
            print(a);

            x = np.arange(6).reshape((3, 2));
            var y = np.arange(6).reshape((2, 3));
            var b = np.ufunc.outer(UFuncOperation.add, null, x, y);
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

            var a = np.ufunc.accumulate(UFuncOperation.subtract, x);
            AssertArray(a, new int[] { 0, -1, -3, -6, -10, -15, -21, -28 });
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(UFuncOperation.subtract, x);
            AssertArray(b, new int[,,] { { { 0, 1 }, { 2, 3 } }, { { -4, -4 }, { -4, -4 } } });
            print(b);

            var c = np.ufunc.accumulate(UFuncOperation.subtract, x, 0);
            AssertArray(c, new int[,,] { { { 0, 1 }, { 2, 3 } }, { { -4, -4 }, { -4, -4 } } });
            print(c);

            var d = np.ufunc.accumulate(UFuncOperation.subtract, x, 1);
            AssertArray(d, new int[,,] { { { 0, 1 }, { -2, -2 } }, { { 4, 5 }, { -2, -2 } } });
            print(d);

            var e = np.ufunc.accumulate(UFuncOperation.subtract, x, 2);
            AssertArray(e, new int[,,] { { { 0, -1 }, { 2, -1 } }, { { 4, -1 }, { 6, -1 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_SubtractReduce_1()
        {
            var x = np.arange(8);

            var a = np.ufunc.reduce(UFuncOperation.subtract, x);
            Assert.AreEqual(-28, a.GetItem(0));
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.reduce(UFuncOperation.subtract, x);
            AssertArray(b, new int[,] { { -4, -4 }, { -4, -4 } });
            print(b);

            var c = np.ufunc.reduce(UFuncOperation.subtract, x, 0);
            AssertArray(c, new int[,] { { -4, -4 }, { -4, -4 } });
            print(c);

            var d = np.ufunc.reduce(UFuncOperation.subtract, x, 1);
            AssertArray(d, new int[,] { { -2, -2 }, { -2, -2 } });
            print(d);

            var e = np.ufunc.reduce(UFuncOperation.subtract, x, 2);
            AssertArray(e, new int[,] { { -1, -1 }, { -1, -1 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_SubtractReduceAt_1()
        {
            var a = np.ufunc.reduceat(UFuncOperation.subtract, np.arange(8), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new int[] { -6, -8, -10, -12 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(UFuncOperation.subtract, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{-12.0, -13.0, -14.0, -15.0},{12.0, 13.0, 14.0, 15.0}, {4.0, 5.0, 6.0, 7.0},
                                          {8.0, 9.0, 10.0, 11.0}, {-24.0, -26.0, -28.0, -30.0}});
            print(b);

            var c = np.ufunc.reduceat(UFuncOperation.multiply, x, new npy_intp[] { 0, 3 }, axis: 1);
            AssertArray(c, new double[,] { { 0.0, 3.0 }, { 120.0, 7.0 }, { 720.0, 11.0 }, { 2184.0, 15.0 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_SubtractOuter_1()
        {
            var x = np.arange(4);

            var a = np.ufunc.outer(UFuncOperation.subtract, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new int[,] { { 0, -1, -2, -3 }, { 1, 0, -1, -2 }, { 2, 1, 0, -1 }, { 3, 2, 1, 0 } });
            print(a);

            x = np.arange(6).reshape((3, 2));
            var y = np.arange(6).reshape((2, 3));
            var b = np.ufunc.outer(UFuncOperation.subtract, null, x, y);
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

            var a = np.ufunc.accumulate(UFuncOperation.multiply, x);
            AssertArray(a, new int[] { 0, 0, 0, 0, 0, 0, 0, 0 });
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(UFuncOperation.multiply, x);
            AssertArray(b, new int[,,] { { { 0, 1 }, { 2, 3 } }, { { 0, 5 }, { 12, 21 } } });
            print(b);

            var c = np.ufunc.accumulate(UFuncOperation.multiply, x, 0);
            AssertArray(c, new int[,,] { { { 0, 1 }, { 2, 3 } }, { { 0, 5 }, { 12, 21 } } });
            print(c);

            var d = np.ufunc.accumulate(UFuncOperation.multiply, x, 1);
            AssertArray(d, new int[,,] { { { 0, 1 }, { 0, 3 } }, { { 4, 5 }, { 24, 35 } } });
            print(d);

            var e = np.ufunc.accumulate(UFuncOperation.multiply, x, 2);
            AssertArray(e, new int[,,] { { { 0, 0 }, { 2, 6 } }, { { 4, 20 }, { 6, 42 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_MultiplyReduce_1()
        {
            var x = np.arange(8);

            var a = np.ufunc.reduce(UFuncOperation.multiply, x);
            Assert.AreEqual(0, a.GetItem(0));
            print(a);

            x = np.arange(8).reshape((2, 2, 2));
            var b = np.ufunc.reduce(UFuncOperation.multiply, x);
            AssertArray(b, new int[,] { { 0, 5 }, { 12, 21 } });
            print(b);

            var c = np.ufunc.reduce(UFuncOperation.multiply, x, 0);
            AssertArray(c, new int[,] { { 0, 5 }, { 12, 21 } });
            print(c);

            var d = np.ufunc.reduce(UFuncOperation.multiply, x, 1);
            AssertArray(d, new int[,] { { 0, 3 }, { 24, 35 } });
            print(d);

            var e = np.ufunc.reduce(UFuncOperation.multiply, x, 2);
            AssertArray(e, new int[,] { { 0, 6 }, { 20, 42 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_MultiplyReduceAt_1()
        {
            var a = np.ufunc.reduceat(UFuncOperation.multiply, np.arange(8), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new int[] { 0, 24, 120, 360 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(UFuncOperation.multiply, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{0.0, 45.0, 120.0, 231.0},{12.0, 13.0, 14.0, 15.0}, {4.0, 5.0, 6.0, 7.0},
                                          {8.0, 9.0, 10.0, 11.0}, {0.0, 585.0, 1680.0, 3465.0}});
            print(b);

            var c = np.ufunc.reduceat(UFuncOperation.multiply, x, new npy_intp[] { 0, 3 }, axis: 1);
            AssertArray(c, new double[,] { { 0.0, 3.0 }, { 120.0, 7.0 }, { 720.0, 11.0 }, { 2184.0, 15.0 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_MultiplyOuter_1()
        {
            var x = np.arange(4);

            var a = np.ufunc.outer(UFuncOperation.multiply, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new int[,] { { 0, 0, 0, 0 }, { 0, 1, 2, 3 }, { 0, 2, 4, 6 }, { 0, 3, 6, 9 } });
            print(a);

            x = np.arange(6).reshape((3, 2));
            var y = np.arange(6).reshape((2, 3));
            var b = np.ufunc.outer(UFuncOperation.multiply, null, x, y);
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

            var a = np.ufunc.accumulate(UFuncOperation.divide, x);
            AssertArray(a, new double[] { 8.00000000e+00, 8.88888889e-01, 8.88888889e-02, 8.08080808e-03, 6.73400673e-04, 5.18000518e-05, 3.70000370e-06, 2.46666913e-07 });
            print(a);

            x = np.arange(8, 16, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(UFuncOperation.divide, x);
            AssertArray(b, new double[,,] { { { 8, 9 }, { 10, 11 } }, { { 0.66666667, 0.69230769 }, { 0.71428571, 0.73333333 } } });
            print(b);

            var c = np.ufunc.accumulate(UFuncOperation.divide, x, 0);
            AssertArray(c, new double[,,] { { { 8, 9 }, { 10, 11 } }, { { 0.66666667, 0.69230769 }, { 0.71428571, 0.73333333 } } });
            print(c);

            var d = np.ufunc.accumulate(UFuncOperation.divide, x, 1);
            AssertArray(d, new double[,,] { { { 8, 9 }, { 0.8, 0.81818182 } }, { { 12, 13 }, { 0.85714286, 0.86666667 } } });
            print(d);

            var e = np.ufunc.accumulate(UFuncOperation.divide, x, 2);
            AssertArray(e, new double[,,] { { { 8.0, 0.88888889 }, { 10.0, 0.90909091 } }, { { 12.0, 0.92307692 }, { 14.0, 0.93333333 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_DivideReduce_1()
        {
            var x = np.arange(8, 16, dtype: np.Float64);

            var a = np.ufunc.reduce(UFuncOperation.divide, x);
            Assert.AreEqual(2.4666691333357994e-07, a.GetItem(0));
            print(a);

            x = np.arange(8, 16, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.reduce(UFuncOperation.divide, x);
            AssertArray(b, new double[,] { { 0.66666667, 0.69230769 }, { 0.71428571, 0.73333333 } });
            print(b);

            var c = np.ufunc.reduce(UFuncOperation.divide, x, 0);
            AssertArray(c, new double[,] { { 0.66666667, 0.69230769 }, { 0.71428571, 0.73333333 } });
            print(c);

            var d = np.ufunc.reduce(UFuncOperation.divide, x, 1);
            AssertArray(d, new double[,] { { 0.8, 0.81818182 }, { 0.85714286, 0.86666667 } });
            print(d);

            var e = np.ufunc.reduce(UFuncOperation.divide, x, 2);
            AssertArray(e, new double[,] { { 0.88888889, 0.90909091 }, { 0.92307692, 0.93333333 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_DivideReduceAt_1()
        {
            var a = np.ufunc.reduceat(UFuncOperation.divide, np.arange(8, 16, dtype: np.Float64), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new double[] { 0.00808081, 0.00681818, 0.00582751, 0.00503663 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(UFuncOperation.divide, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{0.00000000e+00, 2.22222222e-02, 3.33333333e-02, 3.89610390e-02},
                                          {1.20000000e+01, 1.30000000e+01, 1.40000000e+01, 1.50000000e+01}, 
                                          {4.00000000e+00, 5.00000000e+00, 6.00000000e+00, 7.00000000e+00},
                                          {8.00000000e+00, 9.00000000e+00, 1.00000000e+01, 1.10000000e+01}, 
                                          {0.00000000e+00, 1.70940171e-03, 2.38095238e-03, 2.59740260e-03}});
            print(b);

            var c = np.ufunc.reduceat(UFuncOperation.divide, x, new npy_intp[] { 0, 3 }, axis: 1);
            AssertArray(c, new double[,] { { 0.0, 3.0 }, { 0.13333333,  7.0  }, { 0.08888889, 11.0  }, { 0.06593407, 15.0  } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_DivideOuter_1()
        {
            var x = np.arange(4, 8, dtype: np.Float64);

            var a = np.ufunc.outer(UFuncOperation.divide, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new double[,] { { 1.0, 0.8, 0.66666667, 0.57142857 },
                                        {1.25, 1.0, 0.83333333, 0.71428571 },
                                        { 1.5, 1.2, 1.0, 0.85714286 },
                                        { 1.75,1.4, 1.16666667, 1.0 } });
            print(a);

            x = np.arange(8,14, dtype:np.Float64).reshape((3, 2));
            var y = np.arange(8, 14, dtype: np.Float64).reshape((2, 3));
            var b = np.ufunc.outer(UFuncOperation.divide, null, x, y);
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

            var a = np.ufunc.accumulate(UFuncOperation.remainder, x);
            AssertArray(a, new double[] { 16,  1,  1,  1,  1,  1,  1,  1 });
            print(a);

            x = np.arange(16, 8, -1, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(UFuncOperation.remainder, x);
            AssertArray(b, new double[,,] { { { 16, 15 }, { 14, 13 } }, { { 4,4 }, { 4,4 } } });
            print(b);

            var c = np.ufunc.accumulate(UFuncOperation.remainder, x, 0);
            AssertArray(c, new double[,,] { { { 16, 15 }, { 14, 13 } }, { { 4, 4 }, { 4, 4 } } });
            print(c);

            var d = np.ufunc.accumulate(UFuncOperation.remainder, x, 1);
            AssertArray(d, new double[,,] { { { 16, 15 }, { 2, 2 } }, { { 12, 11 }, { 2, 2 } } });
            print(d);

            var e = np.ufunc.accumulate(UFuncOperation.remainder, x, 2);
            AssertArray(e, new double[,,] { { { 16, 1 }, { 14, 1 } }, { { 12, 1 }, { 10, 1 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_RemainderReduce_1()
        {
            var x = np.arange(16,8,-1, dtype: np.Float64);

            var a = np.ufunc.reduce(UFuncOperation.remainder, x);
            Assert.AreEqual(1.0, a.GetItem(0));
            print(a);

            x = np.arange(16,8,-1, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.reduce(UFuncOperation.remainder, x);
            AssertArray(b, new double[,] { { 4, 4 }, { 4, 4 } });
            print(b);

            var c = np.ufunc.reduce(UFuncOperation.remainder, x, 0);
            AssertArray(c, new double[,] { { 4, 4 }, { 4, 4 } });
            print(c);

            var d = np.ufunc.reduce(UFuncOperation.remainder, x, 1);
            AssertArray(d, new double[,] { { 2, 2 }, { 2, 2 } });
            print(d);

            var e = np.ufunc.reduce(UFuncOperation.remainder, x, 2);
            AssertArray(e, new double[,] { { 1, 1 }, { 1, 1 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_RemainderReduceAt_1()
        {
            var a = np.ufunc.reduceat(UFuncOperation.remainder, np.arange(16, 8, -1, dtype: np.Float64), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new double[] { 1,1,1,1 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(UFuncOperation.remainder, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{0,1,2,3},
                                          {12,13,14,15},
                                          {4,5,6,7},
                                          {8,9,10,11},
                                          {0,1,2,3}});
            print(b);

            var c = np.ufunc.reduceat(UFuncOperation.remainder, x, new npy_intp[] { 0, 3 }, axis: 1);
            AssertArray(c, new double[,] { { 0, 3 }, { 4, 7 }, { 8, 11 }, { 12, 15 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_RemainderOuter_1()
        {
            var x = np.arange(4, 8, dtype: np.Float64);

            var a = np.ufunc.outer(UFuncOperation.remainder, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new double[,] { { 0,4,4,4 },
                                           { 1,0,5,5 },
                                           { 2,1,0,6 },
                                           { 3,2,1,0 } });
            print(a);

            x = np.arange(14, 8, -1, dtype: np.Float64).reshape((3, 2));
            var y = np.arange(14, 8, -1, dtype: np.Float64).reshape((2, 3));
            var b = np.ufunc.outer(UFuncOperation.remainder, null, x, y);
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

            var a = np.ufunc.accumulate(UFuncOperation.fmod, x);
            AssertArray(a, new double[] { 16, 1, 1, 1, 1, 1, 1, 1 });
            print(a);

            x = np.arange(16, 8, -1, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(UFuncOperation.fmod, x);
            AssertArray(b, new double[,,] { { { 16, 15 }, { 14, 13 } }, { { 4, 4 }, { 4, 4 } } });
            print(b);

            var c = np.ufunc.accumulate(UFuncOperation.fmod, x, 0);
            AssertArray(c, new double[,,] { { { 16, 15 }, { 14, 13 } }, { { 4, 4 }, { 4, 4 } } });
            print(c);

            var d = np.ufunc.accumulate(UFuncOperation.fmod, x, 1);
            AssertArray(d, new double[,,] { { { 16, 15 }, { 2, 2 } }, { { 12, 11 }, { 2, 2 } } });
            print(d);

            var e = np.ufunc.accumulate(UFuncOperation.fmod, x, 2);
            AssertArray(e, new double[,,] { { { 16, 1 }, { 14, 1 } }, { { 12, 1 }, { 10, 1 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_FModReduce_1()
        {
            var x = np.arange(16, 8, -1, dtype: np.Float64);

            var a = np.ufunc.reduce(UFuncOperation.fmod, x);
            Assert.AreEqual(1.0, a.GetItem(0));
            print(a);

            x = np.arange(16, 8, -1, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.reduce(UFuncOperation.fmod, x);
            AssertArray(b, new double[,] { { 4, 4 }, { 4, 4 } });
            print(b);

            var c = np.ufunc.reduce(UFuncOperation.fmod, x, 0);
            AssertArray(c, new double[,] { { 4, 4 }, { 4, 4 } });
            print(c);

            var d = np.ufunc.reduce(UFuncOperation.fmod, x, 1);
            AssertArray(d, new double[,] { { 2, 2 }, { 2, 2 } });
            print(d);

            var e = np.ufunc.reduce(UFuncOperation.fmod, x, 2);
            AssertArray(e, new double[,] { { 1, 1 }, { 1, 1 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_FModReduceAt_1()
        {
            var a = np.ufunc.reduceat(UFuncOperation.fmod, np.arange(16, 8, -1, dtype: np.Float64), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new double[] { 1, 1, 1, 1 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(UFuncOperation.fmod, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{0,1,2,3},
                                          {12,13,14,15},
                                          {4,5,6,7},
                                          {8,9,10,11},
                                          {0,1,2,3}});
            print(b);

            var c = np.ufunc.reduceat(UFuncOperation.fmod, x, new npy_intp[] { 0, 3 }, axis: 1);
            AssertArray(c, new double[,] { { 0, 3 }, { 4, 7 }, { 8, 11 }, { 12, 15 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_FModOuter_1()
        {
            var x = np.arange(4, 8, dtype: np.Float64);

            var a = np.ufunc.outer(UFuncOperation.fmod, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new double[,] { { 0,4,4,4 },
                                           { 1,0,5,5 },
                                           { 2,1,0,6 },
                                           { 3,2,1,0 } });
            print(a);

            x = np.arange(14, 8,-1, dtype: np.Float64).reshape((3, 2));
            var y = np.arange(14, 8, -1, dtype: np.Float64).reshape((2, 3));
            var b = np.ufunc.outer(UFuncOperation.fmod, null, x, y);
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

            var a = np.ufunc.accumulate(UFuncOperation.power, x);
            AssertArray(a, new double[] {1.6000000e+001, 1.152921504606847E+18, 7.33155940312959E+252, double.PositiveInfinity,
                                        double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity });
            print(a);

            x = np.arange(16, 8, -1, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(UFuncOperation.power, x);
            AssertArray(b, new double[,,] { { { 1.60000000e+01, 1.50000000e+01 }, { 1.40000000e+01, 1.30000000e+01 } }, 
                                            { { 281474976710656, 8649755859375 }, { 289254654976, 10604499373 } } });

            


            print(b);

            var c = np.ufunc.accumulate(UFuncOperation.power, x, 0);
            AssertArray(c, new double[,,] { { { 1.60000000e+01, 1.50000000e+01 }, { 1.40000000e+01, 1.30000000e+01 } },
                                            { { 281474976710656, 8649755859375 }, { 289254654976, 10604499373 } } });
            print(c);

            var d = np.ufunc.accumulate(UFuncOperation.power, x, 1);
            AssertArray(d, new double[,,] { { { 1.60000000e+01, 1.50000000e+01 }, { 72057594037927936, 1946195068359375 } }, 
                                            { { 1.20000000e+01, 1.10000000e+01 }, { 61917364224, 2357947691 } } });
            print(d);

            var e = np.ufunc.accumulate(UFuncOperation.power, x, 2);
            AssertArray(e, new double[,,] { { { 1.60000000e+01, 1.152921504606847E+18 }, { 1.40000000e+01, 793714773254144 } }, 
                                            { { 1.20000000e+01, 743008370688 }, { 1.00000000e+01, 1.00000000e+09} } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_PowerReduce_1()
        {
            var x = np.arange(16, 8, -1, dtype: np.Float64);

            var a = np.ufunc.reduce(UFuncOperation.power, x);
            Assert.AreEqual(double.PositiveInfinity, a.GetItem(0));
            print(a);

            x = np.arange(16, 8, -1, dtype: np.Float64).reshape((2, 2, 2));
            var b = np.ufunc.reduce(UFuncOperation.power, x);
            AssertArray(b, new double[,] { { 281474976710656, 8649755859375 }, { 289254654976, 10604499373 } });
            print(b);

            var c = np.ufunc.reduce(UFuncOperation.power, x, 0);
            AssertArray(c, new double[,] { { 281474976710656, 8649755859375 }, { 289254654976, 10604499373 } });
            print(c);

            var d = np.ufunc.reduce(UFuncOperation.power, x, 1);
            AssertArray(d, new double[,] { { 72057594037927936, 1946195068359375 }, { 61917364224, 2357947691 } });
            print(d);

            var e = np.ufunc.reduce(UFuncOperation.power, x, 2);
            AssertArray(e, new double[,] { { 1.152921504606847E+18, 793714773254144 }, { 743008370688, 1.00000000e+09 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_PowerReduceAt_1()
        {
            var a = np.ufunc.reduceat(UFuncOperation.power, np.arange(16, 8, -1, dtype: np.Float64), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new double[] { double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16).reshape((4, 4));
            var b = np.ufunc.reduceat(UFuncOperation.power, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new double[,] {{0.00000000e+000, 1.00000000e+000, 1.152921504606847E+18, 5.4744010894202194E+36},
                                          {1.20000000e+001, 1.30000000e+001, 1.40000000e+001, 1.50000000e+001},
                                          {4.00000000e+000, 5.00000000e+000, 6.00000000e+000, 7.00000000e+000},
                                          {8.00000000e+000, 9.00000000e+000, 1.00000000e+001, 1.10000000e+001},
                                          {0.00000000e+000, 1.00000000e+000, 7.331559403129590e+252, double.PositiveInfinity}});
            print(b);

            var c = np.ufunc.reduceat(UFuncOperation.power, x, new npy_intp[] { 0, 3 }, axis: 1);
            AssertArray(c, new double[,] { { 0.00000000e+000, 3.00000000e+000 }, { 1.152921504606847E+18, 7.00000000e+000 }, 
                                           { 1.8971375900641885E+81, 1.10000000e+001 }, { 2.5762427384904039E+196, 1.50000000e+001 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_PowerOuter_1()
        {
            var x = np.arange(4, 8, dtype: np.Float64);

            var a = np.ufunc.outer(UFuncOperation.power, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new double[,] { { 2.56000e+02, 1.02400e+03, 4.09600e+03, 1.63840e+04 },
                                           { 6.25000e+02, 3.12500e+03, 1.56250e+04, 7.81250e+04 },
                                           { 1.29600e+03, 7.77600e+03, 4.66560e+04, 2.79936e+05 },
                                           { 2.40100e+03, 1.68070e+04, 1.17649e+05, 8.23543e+05} });
            print(a);

            x = np.arange(14, 8, -1, dtype: np.Float64).reshape((3, 2));
            var y = np.arange(14, 8, -1, dtype: np.Float64).reshape((2, 3));
            var b = np.ufunc.outer(UFuncOperation.power, null, x, y);
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

        #region OUTER tests
        [TestMethod]
        public void test_AddOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(UFuncOperation.add, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.subtract, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.multiply, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.divide, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.remainder, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.fmod, null, a1, a2);
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
            var b = np.ufunc.outer(UFuncOperation.square, null, a1, a2);
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
            var b = np.ufunc.outer(UFuncOperation.reciprocal, null, a1, a2);
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
            var b = np.ufunc.outer(UFuncOperation.ones_like, null, a1, a2);
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
            var b = np.ufunc.outer(UFuncOperation.sqrt, null, a1, a2);
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
            var b = np.ufunc.outer(UFuncOperation.negative, null, a1, a2);
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
            var b = np.ufunc.outer(UFuncOperation.absolute, null, a1, a2);
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
            var b = np.ufunc.outer(UFuncOperation.invert, null, a1, a2);
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
            var b = np.ufunc.outer(UFuncOperation.left_shift, null, a1, a2);
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
            var b = np.ufunc.outer(UFuncOperation.right_shift, null, a1, a2);
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
            var b = np.ufunc.outer(UFuncOperation.bitwise_and, null, a1, a2);
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
            var b = np.ufunc.outer(UFuncOperation.bitwise_or, null, a1, a2);
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
            var b = np.ufunc.outer(UFuncOperation.bitwise_xor, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.less, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.less_equal, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.equal, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.not_equal, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.greater, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.greater_equal, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.floor_divide, null, a1, a2);
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
        public void test_trueDivideOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(UFuncOperation.true_divide, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.logical_and, null, a1, a2);
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

            var b = np.ufunc.outer(UFuncOperation.logical_or, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true}});

        }


        [TestMethod]
        public void test_FloorOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(UFuncOperation.floor, null, a1, a2);
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
        public void test_CeilOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(UFuncOperation.ceil, null, a1, a2);
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
        public void test_MaximumOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(UFuncOperation.maximum, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { 3.0, 4.0, 5.0, 6.0, 7.0 },
                  { 3.0, 4.0, 5.0, 6.0, 7.0 },
                  { 3.0, 4.0, 5.0, 6.0, 7.0 },
                  { 3.0, 4.0, 5.0, 6.0, 7.0 },
                  { 4.0, 4.0, 5.0, 6.0, 7.0  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(UFuncOperation.minimum, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { 0.0, 0.0, 0.0, 0.0, 0.0 },
                  { 1.0, 1.0, 1.0, 1.0, 1.0 },
                  { 2.0, 2.0, 2.0, 2.0, 2.0 },
                  { 3.0, 3.0, 3.0, 3.0, 3.0 },
                  { 3.0, 4.0, 4.0, 4.0, 4.0  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(UFuncOperation.rint, null, a1, a2);
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
        public void test_ConjugateOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(UFuncOperation.conjugate, null, a1, a2);
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
        public void test_IsNANOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(UFuncOperation.isnan, null, a1, a2);
            print(b);


            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false}});

        }


        [TestMethod]
        public void test_FMaxOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(UFuncOperation.fmax, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { 3.0, 4.0, 5.0, 6.0, 7.0 },
                  { 3.0, 4.0, 5.0, 6.0, 7.0 },
                  { 3.0, 4.0, 5.0, 6.0, 7.0 },
                  { 3.0, 4.0, 5.0, 6.0, 7.0 },
                  { 4.0, 4.0, 5.0, 6.0, 7.0  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(UFuncOperation.fmin, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
                { { 0.0, 0.0, 0.0, 0.0, 0.0 },
                  { 1.0, 1.0, 1.0, 1.0, 1.0 },
                  { 2.0, 2.0, 2.0, 2.0, 2.0 },
                  { 3.0, 3.0, 3.0, 3.0, 3.0 },
                  { 3.0, 4.0, 4.0, 4.0, 4.0  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideOuter_DOUBLE()
        {
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

            var b = np.ufunc.outer(UFuncOperation.heaviside, null, a1, a2);
            print(b);

            var ExpectedData = new double[,]
               { { 3.0, 4.0, 5.0, 6.0, 7.0 },
                 { 1.0, 1.0, 1.0, 1.0, 1.0 },
                 { 1.0, 1.0, 1.0, 1.0, 1.0 },
                 { 1.0, 1.0, 1.0, 1.0, 1.0 },
                 { 1.0, 1.0, 1.0, 1.0, 1.0 } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #region REDUCE tests
        [TestMethod]
        public void test_AddReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype : np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.add, a1);
            print(b);

            var ExpectedData = new double[] { 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.subtract, a1);
            print(b);

            var ExpectedData = new double[] { -450, -458, -466, -474, -482, -490, -498, -506, -514, -522 };
 
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.multiply, a1);
            print(b);

            var ExpectedData = new double[] 
                { 0.0, 478015854767451.0, 1242688846823424,
                  2394832584543399, 4060162871525376, 6393838623046875,
                  9585618768101376, 1.38656961199054E+16,19511273389031424,
                  26853950884211452 };
            

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.divide, a1);
            print(b);

            var ExpectedData = new double[] 
            
            { 0.0, 2.09198082035686E-15, 3.21882666785402E-15, 3.75809150839492E-15,
              3.9407286126896E-15, 3.91001422993167E-15, 3.75562609685661E-15,
              3.53390118867932E-15, 3.2801549506235E-15, 3.0163159361263E-15 };
            
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.remainder, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    
            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.fmod, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
  
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.square, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 1.3407807929942597E+154, 1.9323349832288915E+244, 
                        double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity,
                        double.PositiveInfinity,double.PositiveInfinity, double.PositiveInfinity };
             
            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.reciprocal, a1);
            print(b);

            var ExpectedData = new double[] { double.PositiveInfinity, 1.0, 0.5, 0.333333333333333, 0.25, 0.2, 0.166666666666667, 0.142857142857143, 0.125, 0.111111111111111 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.ones_like, a1);
            print(b);

            var ExpectedData = new double[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
   
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.sqrt, a1);
            print(b);

            var ExpectedData = new double[] 
            { 0.0, 1.0, 1.00135471989211, 1.00214803084618,
                1.0027112750502, 1.00314837919044, 1.0035056607184,
                1.00380783722035, 1.00406966796055, 1.00430067572887 };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.negative, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.absolute, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
  
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.invert, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
 
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.left_shift, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.right_shift, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_and, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 };
 
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_or, a1);
            print(b);

            var ExpectedData = new double[] { 126.0, 127.0, 126.0, 127.0, 126.0, 127.0, 126.0, 127.0, 126.0, 127.0 };
      
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_xor, a1);
            print(b);

            var ExpectedData = new double[] { 106.0, 106.0, 94.0, 94.0, 42.0, 42.0, 6.0, 6.0, 26.0, 26.0 };
      
            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.less, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });
    
        }


        [TestMethod]
        public void test_LessEqualReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.less_equal, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });
 
        }

        [TestMethod]
        public void test_EqualReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.equal, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
    
        }


        [TestMethod]
        public void test_NotEqualReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.not_equal,a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });
        }

        [TestMethod]
        public void test_GreaterReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.greater, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_GreaterEqualReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.greater_equal, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_FloorDivideReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.floor_divide, a1);
            print(b);


            var ExpectedData = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
      
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.true_divide, a1);
            print(b);

            var ExpectedData = new double[] 
            
            { 0.0, 2.09198082035686E-15, 3.21882666785402E-15, 3.75809150839492E-15,
              3.9407286126896E-15, 3.91001422993167E-15, 3.75562609685661E-15,
                3.53390118867932E-15, 3.2801549506235E-15, 3.0163159361263E-15 };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.logical_and, a1);
            print(b);

            AssertArray(b, new bool[] { false, true, true, true, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_LogicalOrReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.logical_or, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }


        [TestMethod]
        public void test_FloorReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.floor, a1);
            print(b);


            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.ceil, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.maximum, a1);
            print(b);

            var ExpectedData = new double[] { 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0 };
   
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.minimum, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
 
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.rint, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.conjugate, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.isnan, a1);
            print(b);


            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }


        [TestMethod]
        public void test_FMaxReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.fmax, a1);
            print(b);

            var ExpectedData = new double[] { 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.fmin, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10,10));

            var b = np.ufunc.reduce(UFuncOperation.heaviside, a1);
            print(b);

            var ExpectedData = new double[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #region ACCUMULATE tests
        [TestMethod]
        public void test_AddAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.add, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { 3.0, 5.0, 7.0 },  { 9.0, 12.0, 15.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.subtract, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { -3.0, -3.0, -3.0 },  { -9.0, -10.0, -11.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.multiply, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 4.0, 10.0 }, { 0.0, 28.0, 80.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.divide, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 0.25, 0.4 }, { 0.0, 0.0357142857142857, 0.05 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.remainder, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmod, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.square, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { 0.0, 1.0, 4.0 },  { 0.0, 1.0, 16.0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.reciprocal, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { double.PositiveInfinity, 1.0, 0.5 },  { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.ones_like, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { 1.0, 1.0, 1.0 },  { 1.0, 1.0, 1.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.sqrt, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 1.4142135623731 },  { 0.0, 1.0, 1.18920711500272 } };
            
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.negative, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { 0.0, -1.0, -2.0 },  { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.absolute, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { 0.0, 1.0, 2.0 },  { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.invert, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.left_shift, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { 0.0, 16.0, 64.0 },  { 0.0, 2048.0, 16384.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.right_shift, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { 0.0, 0.0, 0.0 },  { 0.0, 0.0, 0.0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_and, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { 0.0, 0.0, 0.0 },  { 0.0, 0.0, 0.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_or, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { 3.0, 5.0, 7.0 }, { 7.0, 7.0, 15.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_xor, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { 3.0, 5.0, 7.0 },  { 5.0, 2.0, 15.0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.less, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }


        [TestMethod]
        public void test_LessEqualAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.less_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true },  { true, true, true },  { true, true, true } });

        }

        [TestMethod]
        public void test_EqualAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true },  { false, false, false },  { false, false, false } });

        }


        [TestMethod]
        public void test_NotEqualAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.not_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });
        }

        [TestMethod]
        public void test_GreaterAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.greater, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true },  { false, false, false },  { false, false, false } });
        }

        [TestMethod]
        public void test_GreaterEqualAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.greater_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true },  { false, false, false },  { false, false, false } });
        }

        [TestMethod]
        public void test_FloorDivideAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.floor_divide, a1);
            print(b);


            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.true_divide, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 0.25, 0.4 }, { 0.0, 0.0357142857142857, 0.05 } };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.logical_and, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true },  { false, true, true },  { false, true, true } });

        }

        [TestMethod]
        public void test_LogicalOrAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.logical_or, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true },  { true, true, true },  { true, true, true } });

        }


        [TestMethod]
        public void test_FloorAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.floor, a1);
            print(b);


            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.ceil, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.maximum, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 3.0, 4.0, 5.0 }, { 6.0, 7.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.minimum, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.rint, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.conjugate, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 },  { 0.0, 1.0, 2.0 },  { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.isnan, a1);
            print(b);


            AssertArray(b, new bool[,] { { false, true, true },  { false, false, false },  { false, false, false } });
        }


        [TestMethod]
        public void test_FMaxAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmax, a1);
            print(b);

            var ExpectedData = new double[,]{ { 0.0, 1.0, 2.0 },  { 3.0, 4.0, 5.0 },  { 6.0, 7.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmin, a1);
            print(b);

            var ExpectedData = new double[,] {{ 0.0, 1.0, 2.0 },  { 0.0, 1.0, 2.0 },  { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.heaviside, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 3.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #region REDUCEAT tests
        [TestMethod]
        public void test_AddReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.add, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 1.0, 2.0 }, { 7.0, 5.0 }, { 13.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.subtract, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { -1.0, 2.0 }, { -1.0, 5.0 }, { -1.0, 8.0 } };
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.multiply, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 12.0, 5.0 }, { 42.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 },{ 0.75, 5.0 },{ 0.857142857142857, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.remainder, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 3.0, 5.0 }, { 6.0, 8.0 } };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmod, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 3.0, 5.0 }, { 6.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.square, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 9.0, 5.0 }, { 36.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.reciprocal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { double.PositiveInfinity, 2.0 }, { 0.333333333333333, 5.0 }, { 0.166666666666667, 8.0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.ones_like, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 1.0, 2.0 }, { 1.0, 5.0 }, { 1.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.sqrt, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 1.73205080756888, 5.0 }, { 2.44948974278318, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.negative, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 },{ -3.0, 5.0 },{ -6.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.absolute, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 3.0, 5.0 }, { 6.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.invert, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 3.0, 5.0 }, { 6.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.left_shift, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 48.0, 5.0 }, { 768.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.right_shift, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 0.0, 5.0 }, { 0.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_and, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 },{ 0.0, 5.0 }, { 6.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_or, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 1.0, 2.0 }, { 7.0, 5.0 }, { 7.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_xor, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 1.0, 2.0 }, { 7.0, 5.0 }, { 1.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.less, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }


        [TestMethod]
        public void test_LessEqualReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.less_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }

        [TestMethod]
        public void test_EqualReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });

        }


        [TestMethod]
        public void test_NotEqualReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.not_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });
        }

        [TestMethod]
        public void test_GreaterReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.greater, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }

        [TestMethod]
        public void test_GreaterEqualReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.greater_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }

        [TestMethod]
        public void test_FloorDivideReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.floor_divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.true_divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 0.25, 0.4 }, { 0.0, 0.0357142857142857, 0.05 } };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.logical_and, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, true, true }, { false, true, true } });

        }

        [TestMethod]
        public void test_LogicalOrReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.logical_or, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }


        [TestMethod]
        public void test_FloorReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.floor, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.ceil, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.maximum, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 3.0, 4.0, 5.0 }, { 6.0, 7.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.minimum, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.rint, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.conjugate, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.isnan, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }


        [TestMethod]
        public void test_FMaxReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmax, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 3.0, 4.0, 5.0 }, { 6.0, 7.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmin, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.heaviside, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 3.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #endregion
    }
}
