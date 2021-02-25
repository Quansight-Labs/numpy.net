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
    [TestClass]
    public class UFUNC_DOUBLE_Tests : TestBaseClass
    {
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
                { { double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity },
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
            var a1 = np.arange(0, 5, dtype: np.Float64);
            var a2 = np.arange(3, 8, dtype: np.Float64);

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
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.add, a1);
            print(b);

            var ExpectedData = new double[] { 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.subtract, a1);
            print(b);

            var ExpectedData = new double[] { -450, -458, -466, -474, -482, -490, -498, -506, -514, -522 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

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
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

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
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.remainder, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmod, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

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
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.reciprocal, a1);
            print(b);

            var ExpectedData = new double[] { 0, 1.0, 0.5, 0.333333333333333, 0.25, 0.2, 0.166666666666667, 0.142857142857143, 0.125, 0.111111111111111 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.ones_like, a1);
            print(b);

            var ExpectedData = new double[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

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
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.negative, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.absolute, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.invert, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.left_shift, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.right_shift, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_and, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_or, a1);
            print(b);

            var ExpectedData = new double[] { 126.0, 127.0, 126.0, 127.0, 126.0, 127.0, 126.0, 127.0, 126.0, 127.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_xor, a1);
            print(b);

            var ExpectedData = new double[] { 106.0, 106.0, 94.0, 94.0, 42.0, 42.0, 6.0, 6.0, 26.0, 26.0 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.less, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }


        [TestMethod]
        public void test_LessEqualReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.less_equal, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_EqualReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.equal, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });

        }


        [TestMethod]
        public void test_NotEqualReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.not_equal, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });
        }

        [TestMethod]
        public void test_GreaterReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.greater, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_GreaterEqualReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.greater_equal, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_FloorDivideReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.floor_divide, a1);
            print(b);


            var ExpectedData = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

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
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.logical_and, a1);
            print(b);

            AssertArray(b, new bool[] { false, true, true, true, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_LogicalOrReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.logical_or, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }


        [TestMethod]
        public void test_FloorReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.floor, a1);
            print(b);


            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

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
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.minimum, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.rint, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.conjugate, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.isnan, a1);
            print(b);


            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }


        [TestMethod]
        public void test_FMaxReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmax, a1);
            print(b);

            var ExpectedData = new double[] { 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmin, a1);
            print(b);

            var ExpectedData = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideReduce_DOUBLE()
        {
            var a1 = np.arange(0, 100, dtype: np.Float64).reshape((10, 10));

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

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 3.0, 5.0, 7.0 }, { 9.0, 12.0, 15.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.subtract, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { -3.0, -3.0, -3.0 }, { -9.0, -10.0, -11.0 } };

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

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 4.0 }, { 0.0, 1.0, 16.0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.reciprocal, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { double.PositiveInfinity, 1.0, 0.5 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.ones_like, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.sqrt, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 1.4142135623731 }, { 0.0, 1.0, 1.18920711500272 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.negative, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, -1.0, -2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.absolute, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

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

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 16.0, 64.0 }, { 0.0, 2048.0, 16384.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.right_shift, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_and, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_or, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 3.0, 5.0, 7.0 }, { 7.0, 7.0, 15.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_xor, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 3.0, 5.0, 7.0 }, { 5.0, 2.0, 15.0 } };

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

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }

        [TestMethod]
        public void test_EqualAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });

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

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }

        [TestMethod]
        public void test_GreaterEqualAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.greater_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }

        [TestMethod]
        public void test_FloorDivideAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.floor_divide, a1);
            print(b);


            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } };

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

            AssertArray(b, new bool[,] { { false, true, true }, { false, true, true }, { false, true, true } });

        }

        [TestMethod]
        public void test_LogicalOrAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.logical_or, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

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

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

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

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.isnan, a1);
            print(b);


            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }


        [TestMethod]
        public void test_FMaxAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmax, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 3.0, 4.0, 5.0 }, { 6.0, 7.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinAccumulate_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmin, a1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 2.0 } };

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

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 0.75, 5.0 }, { 0.857142857142857, 8.0 } };

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

            var ExpectedData = new double[,] { { 0, 2.0 }, { 0.333333333333333, 5.0 }, { 0.166666666666667, 8.0 } };

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

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { -3.0, 5.0 }, { -6.0, 8.0 } };

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

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 0.0, 5.0 }, { 6.0, 8.0 } };

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

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }


        [TestMethod]
        public void test_LessEqualReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.less_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }

        [TestMethod]
        public void test_EqualReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });

        }


        [TestMethod]
        public void test_NotEqualReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.not_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });
        }

        [TestMethod]
        public void test_GreaterReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.greater, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }

        [TestMethod]
        public void test_GreaterEqualReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.greater_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }

        [TestMethod]
        public void test_FloorDivideReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.floor_divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 0.0, 5.0 }, { 0.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.true_divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 0.75, 5.0 }, { 0.857142857142857, 8.0 } };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.logical_and, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { true, true }, { true, true } });

        }

        [TestMethod]
        public void test_LogicalOrReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.logical_or, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }


        [TestMethod]
        public void test_FloorReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.floor, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 3.0, 5.0 }, { 6.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.ceil, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 3.0, 5.0 }, { 6.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.maximum, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 1.0, 2.0 }, { 4.0, 5.0 }, { 7.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.minimum, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 3.0, 5.0 }, { 6.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.rint, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 3.0, 5.0 }, { 6.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.conjugate, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 3.0, 5.0 }, { 6.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.isnan, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }


        [TestMethod]
        public void test_FMaxReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmax, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 1.0, 2.0 }, { 4.0, 5.0 }, { 7.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmin, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 0.0, 2.0 }, { 3.0, 5.0 }, { 6.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideReduceAt_DOUBLE()
        {
            var a1 = np.arange(0, 9, dtype: np.Float64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.heaviside, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new double[,] { { 1.0, 2.0 }, { 1.0, 5.0 }, { 1.0, 8.0 } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #endregion
    }
}
