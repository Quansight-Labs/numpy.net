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
    public class UFUNC_INT32_Tests : TestBaseClass
    {
        #region UFUNC INT32 Tests

        #region OUTER tests
        [TestMethod]
        public void test_AddOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.add, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                    {{3,  4,  5,  6,  7},
                     {4,  5,  6,  7,  8},
                     {5,  6,  7,  8,  9},
                     {6,  7,  8,  9, 10},
                     {7,  8,  9, 10, 11}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.subtract, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                    {{-3, -4, -5, -6, -7},
                     {-2, -3, -4, -5, -6},
                     {-1, -2, -3, -4, -5},
                     {0, -1, -2, -3, -4},
                     {1,  0, -1, -2, -3}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.multiply, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                    {{0,  0,  0,  0,  0,},
                     {3,  4,  5,  6,  7,},
                     {6,  8, 10, 12, 14},
                     {9, 12, 15, 18, 21,},
                     {12, 16, 20, 24, 28,}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.divide, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                    {{0,  0,  0,  0,  0},
                     {0,  0,  0,  0,  0},
                     {0,  0,  0,  0,  0},
                     {1,  0,  0,  0,  0},
                     {1,  1,  0,  0,  0}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.remainder, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                    {{0, 0, 0, 0, 0,},
                     {1, 1, 1, 1, 1,},
                     {2, 2, 2, 2, 2},
                     {0, 3, 3, 3, 3,},
                     {1, 0, 4, 4, 4,}};

            AssertArray(b, ExpectedData);


        }

        [TestMethod]
        public void test_FModOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.fmod, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                    {{0, 0, 0, 0, 0,},
                     {1, 1, 1, 1, 1,},
                     {2, 2, 2, 2, 2},
                     {0, 3, 3, 3, 3,},
                     {1, 0, 4, 4, 4,}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.square, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 4, 4, 4, 4, 4 },
                  { 9, 9, 9, 9, 9 },
                  { 16, 16, 16, 16, 16 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.reciprocal, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                    { { 0, 0, 0, 0, 0 },
                      { 1, 1, 1, 1, 1 },
                      { 0, 0, 0, 0, 0 },
                      { 0, 0, 0, 0, 0 },
                      { 0, 0, 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.ones_like, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 1, 1, 1, 1, 1 },
                  { 1, 1, 1, 1, 1 },
                  { 1, 1, 1, 1, 1 },
                  { 1, 1, 1, 1, 1 },
                  { 1, 1, 1, 1, 1 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.sqrt, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
             { { 0, 0, 0, 0, 0 },
               { 1, 1, 1, 1, 1 },
               { 1, 1, 1, 1, 1 },
               { 2, 2, 2, 2, 2 },
               { 2, 2, 2, 2, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.negative, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 0, 0, 0, 0, 0 },
                  { -1, -1, -1, -1, -1 },
                  { -2, -2, -2, -2, -2 },
                  { -3, -3, -3, -3, -3 },
                  {-4, -4, -4, -4, -4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.absolute, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.invert, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
               { { -1, -1, -1, -1, -1 },
                 { -2, -2, -2, -2, -2 },
                 { -3, -3, -3, -3, -3 },
                 { -4, -4, -4, -4, -4 },
                 { -5, -5, -5, -5, -5 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.left_shift, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                  { { 0, 0, 0, 0, 0 },
                  { 8, 16, 32, 64, 128 },
                  { 16, 32, 64, 128, 256 },
                  { 24, 48, 96, 192, 384 },
                  { 32, 64, 128, 256, 512  } };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32) * 1024 * 4;
            var a2 = np.arange(3, 8, dtype: np.Int32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.right_shift, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 0, 0, 0, 0, 0  },
                  { 512, 256, 128, 64, 32 },
                  { 1024, 512, 256, 128, 64  },
                  { 1536, 768, 384, 192, 96  },
                  { 2048, 1024, 512, 256, 128  } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.bitwise_and, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { {  0, 0, 0, 0, 0  },
                  { 1, 0, 1, 0, 1  },
                  { 2, 0, 0, 2, 2  },
                  { 3, 0, 1, 2, 3  },
                  { 0, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.bitwise_or, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 3, 4, 5, 6, 7 },
                  { 3, 5, 5, 7, 7 },
                  { 3, 6, 7, 6, 7 },
                  { 3, 7, 7, 7, 7 },
                  { 7, 4, 5, 6, 7 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.bitwise_xor, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 3, 4, 5, 6, 7 },
                  { 2, 5, 4, 7, 6 },
                  { 1, 6, 7, 4, 5 },
                  { 0, 7, 6, 5, 4 },
                  { 7, 0, 1, 2, 3 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.less, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true },
                                         { false, false, true, true, true } });

        }


        [TestMethod]
        public void test_LessEqualOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.less_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true } });

        }

        [TestMethod]
        public void test_EqualOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false},
                                        {false, true, false, false, false}});

        }


        [TestMethod]
        public void test_NotEqualOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.not_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true },
                                         { true, false, true, true, true } });

        }

        [TestMethod]
        public void test_GreaterOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.greater, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false}});

        }

        [TestMethod]
        public void test_GreaterEqualOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.greater_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false},
                                        {true, true, false, false, false}});

        }

        [TestMethod]
        public void test_FloorDivideOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.floor_divide, null, a1, a2);
            print(b);


            var ExpectedData = new Int32[,]
                { { 0, 0, 0, 0, 0 },
                  { 0, 0, 0, 0, 0 },
                  { 0, 0, 0, 0, 0 },
                  { 1, 0, 0, 0, 0 },
                  { 1, 1, 0, 0, 0} };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.true_divide, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 0, 0, 0, 0, 0 },
                  { 0, 0, 0, 0, 0 },
                  { 0, 0, 0, 0, 0 },
                  { 1, 0, 0, 0, 0 },
                  { 1, 1, 0, 0, 0} };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.logical_and, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true}});

        }

        [TestMethod]
        public void test_LogicalOrOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.logical_or, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true}});

        }


        [TestMethod]
        public void test_FloorOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.floor, null, a1, a2);
            print(b);


            var ExpectedData = new Int32[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.ceil, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                 { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.maximum, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 4, 4, 5, 6, 7  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.minimum, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 3, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.rint, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.conjugate, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.isnan, null, a1, a2);
            print(b);


            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false}});

        }


        [TestMethod]
        public void test_FMaxOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.fmax, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 4, 4, 5, 6, 7  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.fmin, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 3, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideOuter_INT32()
        {
            var a1 = np.arange(0, 5, dtype: np.Int32);
            var a2 = np.arange(3, 8, dtype: np.Int32);

            var b = np.ufunc.outer(UFuncOperation.heaviside, null, a1, a2);
            print(b);

            var ExpectedData = new Int32[,]
               { { 3, 4, 5, 6, 7 },
                 { 1, 1, 1, 1, 1 },
                 { 1, 1, 1, 1, 1 },
                 { 1, 1, 1, 1, 1 },
                 { 1, 1, 1, 1, 1 } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #region REDUCE tests
        [TestMethod]
        public void test_AddReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.add, a1);
            print(b);

            var ExpectedData = new Int32[] { 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.subtract, a1);
            print(b);

            var ExpectedData = new Int32[] { -450, -458, -466, -474, -482, -490, -498, -506, -514, -522 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.multiply, a1);
            print(b);

            var ExpectedData = new Int32[] 
            
                { 0, -1120375461, 189267968, 1769966759, 1437597696, -1881096997, -912261120, -204841865, 1467482112, -1421546757 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.divide, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
 
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.remainder, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmod, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.square, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 1, 0, 1995565057, 0, -59967487, 0, 684437505, 0, -1876701183 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.reciprocal, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.ones_like, a1);
            print(b);

            var ExpectedData = new Int32[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.sqrt, a1);
            print(b);

            var ExpectedData = new Int32[] {  0, 1, 1, 1, 1, 1, 1, 1, 1, 1  };



            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.negative, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, -1, -2, -3, -4, -5, -6, -7, -8, -9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.absolute, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.invert, a1);
            print(b);

            var ExpectedData = new Int32[] { -1, -2, -3, -4, -5, -6, -7, -8, -9, -10 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.left_shift, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.right_shift, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_and, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_or, a1);
            print(b);

            var ExpectedData = new Int32[] { 126, 127, 126, 127, 126, 127, 126, 127, 126, 127 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_xor, a1);
            print(b);

            var ExpectedData = new Int32[] { 106, 106, 94, 94, 42, 42, 6, 6, 26, 26 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.less, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }


        [TestMethod]
        public void test_LessEqualReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.less_equal, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_EqualReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.equal, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });

        }


        [TestMethod]
        public void test_NotEqualReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.not_equal, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });
        }

        [TestMethod]
        public void test_GreaterReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.greater, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_GreaterEqualReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.greater_equal, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_FloorDivideReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.floor_divide, a1);
            print(b);


            var ExpectedData = new Int32[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.true_divide, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };


            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.logical_and, a1);
            print(b);

            AssertArray(b, new bool[] { false, true, true, true, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_LogicalOrReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.logical_or, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }


        [TestMethod]
        public void test_FloorReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.floor, a1);
            print(b);


            var ExpectedData = new Int32[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.ceil, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.maximum, a1);
            print(b);

            var ExpectedData = new Int32[] { 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.minimum, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.rint, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.conjugate, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.isnan, a1);
            print(b);


            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }


        [TestMethod]
        public void test_FMaxReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmax, a1);
            print(b);

            var ExpectedData = new Int32[] { 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmin, a1);
            print(b);

            var ExpectedData = new Int32[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideReduce_INT32()
        {
            var a1 = np.arange(0, 100, dtype: np.Int32).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.heaviside, a1);
            print(b);

            var ExpectedData = new Int32[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #region ACCUMULATE tests
        [TestMethod]
        public void test_AddAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.add, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 3, 5, 7 }, { 9, 12, 15 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.subtract, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { -3, -3, -3 }, { -9, -10, -11 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.multiply, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 4, 10 }, { 0, 28, 80 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.divide, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.remainder, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmod, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.square, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 1, 4 }, { 0, 1, 16 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.reciprocal, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 1, 0 }, { 0, 1, 0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.ones_like, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 1, 1, 1 }, { 1, 1, 1 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.sqrt, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 1, 1 }, { 0, 1, 1 } };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.negative, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, -1, -2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.absolute, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.invert, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { -1, -2, -3 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.left_shift, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 16, 64 }, { 0, 2048, 16384 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.right_shift, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_and, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_or, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 3, 5, 7 }, { 7, 7, 15 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_xor, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 3, 5, 7 }, { 5, 2, 15 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.less, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }


        [TestMethod]
        public void test_LessEqualAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.less_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }

        [TestMethod]
        public void test_EqualAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });

        }


        [TestMethod]
        public void test_NotEqualAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.not_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });
        }

        [TestMethod]
        public void test_GreaterAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.greater, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }

        [TestMethod]
        public void test_GreaterEqualAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.greater_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }

        [TestMethod]
        public void test_FloorDivideAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.floor_divide, a1);
            print(b);


            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.true_divide, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.logical_and, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, true, true }, { false, true, true } });

        }

        [TestMethod]
        public void test_LogicalOrAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.logical_or, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }


        [TestMethod]
        public void test_FloorAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.floor, a1);
            print(b);


            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.ceil, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.maximum, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.minimum, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.rint, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.conjugate, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.isnan, a1);
            print(b);


            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }


        [TestMethod]
        public void test_FMaxAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmax, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmin, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideAccumulate_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.heaviside, a1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 1, 2 }, { 3, 1, 1 }, { 1, 1, 1 } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #region REDUCEAT tests
        [TestMethod]
        public void test_AddReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.add, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 1, 2 }, { 7, 5 }, { 13, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.subtract, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { -1, 2 }, { -1, 5 }, { -1, 8 } };
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.multiply, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 12, 5 }, { 42, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.remainder, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmod, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.square, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 9, 5 }, { 36, 8 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.reciprocal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.ones_like, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 1, 2 }, { 1, 5 }, { 1, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.sqrt, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 2, 5 }, { 2, 8 } };
             

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.negative, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { -3, 5 }, { -6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.absolute, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.invert, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { -1, 2 }, { -4, 5 }, { -7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.left_shift, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 48, 5 }, { 768, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.right_shift, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_and, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 0, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_or, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 1, 2 }, { 7, 5 }, { 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_xor, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 1, 2 }, { 7, 5 }, { 1, 8 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.less, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }


        [TestMethod]
        public void test_LessEqualReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.less_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }

        [TestMethod]
        public void test_EqualReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });

        }


        [TestMethod]
        public void test_NotEqualReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.not_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });
        }

        [TestMethod]
        public void test_GreaterReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.greater, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }

        [TestMethod]
        public void test_GreaterEqualReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.greater_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }

        [TestMethod]
        public void test_FloorDivideReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.floor_divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            var ExpectedData = new Int32[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.true_divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.logical_and, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { true, true }, { true, true } });

        }

        [TestMethod]
        public void test_LogicalOrReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.logical_or, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }


        [TestMethod]
        public void test_FloorReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.floor, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            var ExpectedData = new Int32[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.ceil, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.maximum, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 1, 2 }, { 4, 5 }, { 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.minimum, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.rint, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.conjugate, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.isnan, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }


        [TestMethod]
        public void test_FMaxReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmax, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 1, 2 }, { 4, 5 }, { 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmin, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideReduceAt_INT32()
        {
            var a1 = np.arange(0, 9, dtype: np.Int32).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.heaviside, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new Int32[,] { { 1, 2 }, { 1, 5 }, { 1, 8 } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #endregion
    }
}
