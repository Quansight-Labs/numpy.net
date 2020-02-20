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
    public class UFUNC_FLOAT_Tests : TestBaseClass
    {
        #region UFUNC FLOAT Tests

        #region OUTER tests
        [TestMethod]
        public void test_AddOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.add, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                    {{3,  4,  5,  6,  7},
                     {4,  5,  6,  7,  8},
                     {5,  6,  7,  8,  9},
                     {6,  7,  8,  9, 10},
                     {7,  8,  9, 10, 11}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.subtract, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                    {{-3, -4, -5, -6, -7},
                     {-2, -3, -4, -5, -6},
                     {-1, -2, -3, -4, -5},
                     {0, -1, -2, -3, -4},
                     {1,  0, -1, -2, -3}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.multiply, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                    {{0,  0,  0,  0,  0,},
                     {3,  4,  5,  6,  7,},
                     {6,  8, 10, 12, 14},
                     {9, 12, 15, 18, 21,},
                     {12, 16, 20, 24, 28,}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.divide, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                    {{0,  0,  0,  0,  0,},
                     {0.33333333f, 0.25f,       0.2f,        0.16666667f, 0.142857149f},
                     {0.66666667f, 0.5f,        0.4f,        0.33333333f, 0.2857143f},
                     {1,         0.75f,       0.6f,        0.5f,        0.428571433f},
                     {1.33333333f, 1,         0.8f,        0.66666667f, 0.5714286f}};

            AssertArray(b, ExpectedData);
        }


[TestMethod]
        public void test_RemainderOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.remainder, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                    {{0, 0, 0, 0, 0,},
                     {1, 1, 1, 1, 1,},
                     {2, 2, 2, 2, 2},
                     {0, 3, 3, 3, 3,},
                     {1, 0, 4, 4, 4,}};

            AssertArray(b, ExpectedData);


        }

        [TestMethod]
        public void test_FModOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.fmod, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                    {{0, 0, 0, 0, 0,},
                     {1, 1, 1, 1, 1,},
                     {2, 2, 2, 2, 2},
                     {0, 3, 3, 3, 3,},
                     {1, 0, 4, 4, 4,}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.square, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 4.0f, 4.0f, 4.0f, 4.0f, 4.0f },
                  { 9.0f, 9.0f, 9.0f, 9.0f, 9.0f },
                  { 16.0f, 16.0f, 16.0f, 16.0f, 16.0f } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.reciprocal, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity,float.PositiveInfinity },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f },
                  { 0.333333333333333f, 0.333333333333333f, 0.333333333333333f, 0.333333333333333f, 0.333333333333333f },
                  { 0.25f, 0.25f, 0.25f, 0.25f, 0.25f} };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.ones_like, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.sqrt, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 1.4142135623731f, 1.4142135623731f, 1.4142135623731f, 1.4142135623731f, 1.4142135623731f  },
                  { 1.73205080756888f, 1.73205080756888f, 1.73205080756888f, 1.73205080756888f, 1.73205080756888f },
                  { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.negative, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { -1.0f, -1.0f, -1.0f, -1.0f, -1.0f },
                  { -2.0f, -2.0f, -2.0f, -2.0f, -2.0f },
                  { -3.0f, -3.0f, -3.0f, -3.0f, -3.0f },
                  {-4.0f, -4.0f, -4.0f, -4.0f, -4.0f  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.absolute, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f },
                  { 3.0f, 3.0f, 3.0f, 3.0f, 3.0f },
                  { 4.0f, 4.0f, 4.0f, 4.0f, 4.0f  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.invert, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f },
                  { 3.0f, 3.0f, 3.0f, 3.0f, 3.0f },
                  { 4.0f, 4.0f, 4.0f, 4.0f, 4.0f  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.left_shift, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                  { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 8.0f, 16.0f, 32.0f, 64.0f, 128.0f },
                  { 16.0f, 32.0f, 64.0f, 128.0f, 256.0f },
                  { 24.0f, 48.0f, 96.0f, 192.0f, 384.0f },
                  { 32.0f, 64.0f, 128.0f, 256.0f, 512.0f  } };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32) * 1024 * 4;
            var a2 = np.arange(3, 8, dtype: np.Float32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.right_shift, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f  },
                  { 512.0f, 256.0f, 128.0f, 64.0f, 32.0f },
                  { 1024.0f, 512.0f, 256.0f, 128.0f, 64.0f  },
                  { 1536.0f, 768.0f, 384.0f, 192.0f, 96.0f  },
                  { 2048.0f, 1024.0f, 512.0f, 256.0f, 128.0f  } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.bitwise_and, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { {  0.0f, 0.0f, 0.0f, 0.0f, 0.0f  },
                  { 1.0f, 0.0f, 1.0f, 0.0f, 1.0f  },
                  { 2.0f, 0.0f, 0.0f, 2.0f, 2.0f  },
                  { 3.0f, 0.0f, 1.0f, 2.0f, 3.0f  },
                  { 0.0f, 4.0f, 4.0f, 4.0f, 4.0f  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.bitwise_or, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f },
                  { 3.0f, 5.0f, 5.0f, 7.0f, 7.0f },
                  { 3.0f, 6.0f, 7.0f, 6.0f, 7.0f },
                  { 3.0f, 7.0f, 7.0f, 7.0f, 7.0f },
                  { 7.0f, 4.0f, 5.0f, 6.0f, 7.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.bitwise_xor, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f },
                  { 2.0f, 5.0f, 4.0f, 7.0f, 6.0f },
                  { 1.0f, 6.0f, 7.0f, 4.0f, 5.0f },
                  { 0.0f, 7.0f, 6.0f, 5.0f, 4.0f },
                  { 7.0f, 0.0f, 1.0f, 2.0f, 3.0f } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.less, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true },
                                         { false, false, true, true, true } });

        }


        [TestMethod]
        public void test_LessEqualOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.less_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true } });


        }

        [TestMethod]
        public void test_EqualOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false},
                                        {false, true, false, false, false}});

        }


        [TestMethod]
        public void test_NotEqualOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.not_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true },
                                         { true, false, true, true, true } });

        }

        [TestMethod]
        public void test_GreaterOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.greater, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false}});

        }

        [TestMethod]
        public void test_GreaterEqualOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.greater_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false},
                                        {true, true, false, false, false}});

        }

        [TestMethod]
        public void test_FloorDivideOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.floor_divide, null, a1, a2);
            print(b);


            var ExpectedData = new float[,]
                { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 1.0f, 1.0f, 0.0f, 0.0f, 0.0f} };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.true_divide, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                      {{0,  0,  0,  0,  0,},
                     {0.33333333f, 0.25f,       0.2f,        0.16666667f, 0.142857149f},
                     {0.66666667f, 0.5f,        0.4f,        0.33333333f, 0.28571429f},
                     {1f,         0.75f,       0.6f,        0.5f,        0.42857143f},
                     {1.33333333f, 1,         0.8f,        0.66666667f, 0.57142857f}};

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.logical_and, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true}});

        }

        [TestMethod]
        public void test_LogicalOrOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.logical_or, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true}});

        }


        [TestMethod]
        public void test_FloorOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.floor, null, a1, a2);
            print(b);


            var ExpectedData = new float[,]
                { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f },
                  { 3.0f, 3.0f, 3.0f, 3.0f, 3.0f },
                  { 4.0f, 4.0f, 4.0f, 4.0f, 4.0f  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.ceil, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                 { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f },
                  { 3.0f, 3.0f, 3.0f, 3.0f, 3.0f },
                  { 4.0f, 4.0f, 4.0f, 4.0f, 4.0f  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.maximum, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f },
                  { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f },
                  { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f },
                  { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f },
                  { 4.0f, 4.0f, 5.0f, 6.0f, 7.0f  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.minimum, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f },
                  { 3.0f, 3.0f, 3.0f, 3.0f, 3.0f },
                  { 3.0f, 4.0f, 4.0f, 4.0f, 4.0f  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.rint, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f },
                  { 3.0f, 3.0f, 3.0f, 3.0f, 3.0f },
                  { 4.0f, 4.0f, 4.0f, 4.0f, 4.0f  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.conjugate, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f },
                  { 3.0f, 3.0f, 3.0f, 3.0f, 3.0f },
                  { 4.0f, 4.0f, 4.0f, 4.0f, 4.0f  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.isnan, null, a1, a2);
            print(b);


            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false}});

        }


        [TestMethod]
        public void test_FMaxOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.fmax, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f },
                  { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f },
                  { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f },
                  { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f },
                  { 4.0f, 4.0f, 5.0f, 6.0f, 7.0f  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.fmin, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
                { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                  { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f },
                  { 3.0f, 3.0f, 3.0f, 3.0f, 3.0f },
                  { 3.0f, 4.0f, 4.0f, 4.0f, 4.0f  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideOuter_FLOAT()
        {
            var a1 = np.arange(0, 5, dtype: np.Float32);
            var a2 = np.arange(3, 8, dtype: np.Float32);

            var b = np.ufunc.outer(UFuncOperation.heaviside, null, a1, a2);
            print(b);

            var ExpectedData = new float[,]
               { { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f },
                 { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                 { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                 { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                 { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #region REDUCE tests
        [TestMethod]
        public void test_AddReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.add, a1);
            print(b);

            var ExpectedData = new float[] { 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.subtract, a1);
            print(b);

            var ExpectedData = new float[] { -450, -458, -466, -474, -482, -490, -498, -506, -514, -522 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.multiply, a1);
            print(b);

            var ExpectedData = new float[]
                { 0.0f, 478015854767451.0f, 1.24268893E+15f,
                  2394832584543399, 4.060163E+15f, 6393838623046875,
                  9585618768101376, 1.38656974E+16f,19511273389031424,
                  26853950884211452 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.divide, a1);
            print(b);

            var ExpectedData = new float[]

            { 0.0f, 2.09198082035686E-15f, 3.21882666785402E-15f, 3.75809150839492E-15f,
              3.9407286126896E-15f, 3.91001422993167E-15f, 3.75562609685661E-15f,
              3.53390118867932E-15f, 3.2801549506235E-15f, 3.0163159361263E-15f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.remainder, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmod, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.square, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, 1.0f, float.PositiveInfinity, float.PositiveInfinity,
                        float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity,
                        float.PositiveInfinity,float.PositiveInfinity, float.PositiveInfinity };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.reciprocal, a1);
            print(b);

            var ExpectedData = new float[] { float.PositiveInfinity, 1.0f, 0.5f, 0.333333333333333f, 0.25f, 0.2f, 0.166666666666667f, 0.142857142857143f, 0.125f, 0.111111111111111f };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.ones_like, a1);
            print(b);

            var ExpectedData = new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.sqrt, a1);
            print(b);

            var ExpectedData = new float[]
            { 0.0f, 1.0f, 1.00135471989211f, 1.00214803084618f,
                1.0027112750502f, 1.00314832f, 1.0035056607184f,
                1.0038079f, 1.00406966796055f, 1.00430067572887f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.negative, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -9.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.absolute, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.invert, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.left_shift, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.right_shift, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_and, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_or, a1);
            print(b);

            var ExpectedData = new float[] { 126.0f, 127.0f, 126.0f, 127.0f, 126.0f, 127.0f, 126.0f, 127.0f, 126.0f, 127.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_xor, a1);
            print(b);

            var ExpectedData = new float[] { 106.0f, 106.0f, 94.0f, 94.0f, 42.0f, 42.0f, 6.0f, 6.0f, 26.0f, 26.0f };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.less, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }


        [TestMethod]
        public void test_LessEqualReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.less_equal, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_EqualReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.equal, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });

        }


        [TestMethod]
        public void test_NotEqualReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.not_equal, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });
        }

        [TestMethod]
        public void test_GreaterReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.greater, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_GreaterEqualReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.greater_equal, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_FloorDivideReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.floor_divide, a1);
            print(b);


            var ExpectedData = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.true_divide, a1);
            print(b);

            var ExpectedData = new float[]

            { 0.0f, 2.09198082035686E-15f, 3.21882666785402E-15f, 3.75809150839492E-15f,
              3.9407286126896E-15f, 3.91001422993167E-15f, 3.75562609685661E-15f,
                3.53390118867932E-15f, 3.2801549506235E-15f, 3.0163159361263E-15f };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.logical_and, a1);
            print(b);

            AssertArray(b, new bool[] { false, true, true, true, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_LogicalOrReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.logical_or, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }


        [TestMethod]
        public void test_FloorReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.floor, a1);
            print(b);


            var ExpectedData = new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.ceil, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.maximum, a1);
            print(b);

            var ExpectedData = new float[] { 90.0f, 91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f, 97.0f, 98.0f, 99.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.minimum, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.rint, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.conjugate, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.isnan, a1);
            print(b);


            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }


        [TestMethod]
        public void test_FMaxReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmax, a1);
            print(b);

            var ExpectedData = new float[] { 90.0f, 91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f, 97.0f, 98.0f, 99.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmin, a1);
            print(b);

            var ExpectedData = new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideReduce_FLOAT()
        {
            var a1 = np.arange(0, 100, dtype: np.Float32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.heaviside, a1);
            print(b);

            var ExpectedData = new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #region ACCUMULATE tests
        [TestMethod]
        public void test_AddAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.add, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 3.0f, 5.0f, 7.0f }, { 9.0f, 12.0f, 15.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.subtract, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { -3.0f, -3.0f, -3.0f }, { -9.0f, -10.0f, -11.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.multiply, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 4.0f, 10.0f }, { 0.0f, 28.0f, 80.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.divide, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 0.25f, 0.4f }, { 0.0f, 0.0357142857142857f, 0.05f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.remainder, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f } };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmod, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.square, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 4.0f }, { 0.0f, 1.0f, 16.0f } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.reciprocal, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { float.PositiveInfinity, 1.0f, 0.5f }, { 0.0f, 1.0f, 2.0f } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.ones_like, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.sqrt, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 1.4142135623731f }, { 0.0f, 1.0f, 1.18920711500272f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.negative, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, -1.0f, -2.0f }, { 0.0f, 1.0f, 2.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.absolute, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.invert, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.left_shift, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 16.0f, 64.0f }, { 0.0f, 2048.0f, 16384.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.right_shift, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_and, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_or, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 3.0f, 5.0f, 7.0f }, { 7.0f, 7.0f, 15.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_xor, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 3.0f, 5.0f, 7.0f }, { 5.0f, 2.0f, 15.0f } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.less, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }


        [TestMethod]
        public void test_LessEqualAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.less_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }

        [TestMethod]
        public void test_EqualAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });

        }


        [TestMethod]
        public void test_NotEqualAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.not_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });
        }

        [TestMethod]
        public void test_GreaterAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.greater, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }

        [TestMethod]
        public void test_GreaterEqualAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.greater_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }

        [TestMethod]
        public void test_FloorDivideAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.floor_divide, a1);
            print(b);


            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.true_divide, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 0.25f, 0.4f }, { 0.0f, 0.0357142857142857f, 0.05f } };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.logical_and, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, true, true }, { false, true, true } });

        }

        [TestMethod]
        public void test_LogicalOrAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.logical_or, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }


        [TestMethod]
        public void test_FloorAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.floor, a1);
            print(b);


            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.ceil, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.maximum, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 3.0f, 4.0f, 5.0f }, { 6.0f, 7.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.minimum, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.rint, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.conjugate, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.isnan, a1);
            print(b);


            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }


        [TestMethod]
        public void test_FMaxAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmax, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 3.0f, 4.0f, 5.0f }, { 6.0f, 7.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmin, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f }, { 0.0f, 1.0f, 2.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideAccumulate_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.heaviside, a1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 1.0f, 2.0f }, { 3.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #region REDUCEAT tests
        [TestMethod]
        public void test_AddReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.add, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 1.0f, 2.0f }, { 7.0f, 5.0f }, { 13.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.subtract, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { -1.0f, 2.0f }, { -1.0f, 5.0f }, { -1.0f, 8.0f } };
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.multiply, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 12.0f, 5.0f }, { 42.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 0.75f, 5.0f }, { 0.857142857142857f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.remainder, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 3.0f, 5.0f }, { 6.0f, 8.0f } };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmod, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 3.0f, 5.0f }, { 6.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.square, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 9.0f, 5.0f }, { 36.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.reciprocal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { float.PositiveInfinity, 2.0f }, { 0.333333333333333f, 5.0f }, { 0.166666666666667f, 8.0f } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.ones_like, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 1.0f, 2.0f }, { 1.0f, 5.0f }, { 1.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.sqrt, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 1.73205080756888f, 5.0f }, { 2.44948974278318f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.negative, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { -3.0f, 5.0f }, { -6.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.absolute, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 3.0f, 5.0f }, { 6.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.invert, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 3.0f, 5.0f }, { 6.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.left_shift, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 48.0f, 5.0f }, { 768.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.right_shift, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 0.0f, 5.0f }, { 0.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_and, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 0.0f, 5.0f }, { 6.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_or, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 1.0f, 2.0f }, { 7.0f, 5.0f }, { 7.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_xor, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 1.0f, 2.0f }, { 7.0f, 5.0f }, { 1.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.less, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }


        [TestMethod]
        public void test_LessEqualReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.less_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }

        [TestMethod]
        public void test_EqualReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });

        }


        [TestMethod]
        public void test_NotEqualReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.not_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });
        }

        [TestMethod]
        public void test_GreaterReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.greater, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }

        [TestMethod]
        public void test_GreaterEqualReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.greater_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }

        [TestMethod]
        public void test_FloorDivideReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.floor_divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 0.0f, 5.0f }, { 0.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.true_divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 0.75f, 5.0f }, { 0.857142857142857f, 8.0f } };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.logical_and, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { true, true }, { true, true } });

        }

        [TestMethod]
        public void test_LogicalOrReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.logical_or, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }


        [TestMethod]
        public void test_FloorReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.floor, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 3.0f, 5.0f }, { 6.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.ceil, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 3.0f, 5.0f }, { 6.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.maximum, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 1.0f, 2.0f }, { 4.0f, 5.0f }, { 7.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.minimum, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 3.0f, 5.0f }, { 6.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.rint, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 3.0f, 5.0f }, { 6.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.conjugate, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 3.0f, 5.0f }, { 6.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.isnan, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }


        [TestMethod]
        public void test_FMaxReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmax, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 1.0f, 2.0f }, { 4.0f, 5.0f }, { 7.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmin, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 0.0f, 2.0f }, { 3.0f, 5.0f }, { 6.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideReduceAt_FLOAT()
        {
            var a1 = np.arange(0, 9, dtype: np.Float32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.heaviside, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new float[,] { { 1.0f, 2.0f }, { 1.0f, 5.0f }, { 1.0f, 8.0f } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #endregion
    }
}
