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
    public class UFUNC_UINT64_Tests : TestBaseClass
    {
        #region UFUNC UINT64 Tests

        #region OUTER tests
        [TestMethod]
        public void test_AddOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.add, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                    {{3,  4,  5,  6,  7},
                     {4,  5,  6,  7,  8},
                     {5,  6,  7,  8,  9},
                     {6,  7,  8,  9, 10},
                     {7,  8,  9, 10, 11}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.subtract, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
          { { 18446744073709551613, 18446744073709551612, 18446744073709551611, 18446744073709551610, 18446744073709551609 },
            { 18446744073709551614, 18446744073709551613, 18446744073709551612, 18446744073709551611, 18446744073709551610 },
            { 18446744073709551615, 18446744073709551614, 18446744073709551613, 18446744073709551612, 18446744073709551611 },
            { 0, 18446744073709551615, 18446744073709551614, 18446744073709551613, 18446744073709551612 },
            { 1, 0, 18446744073709551615, 18446744073709551614, 18446744073709551613 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.multiply, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                    {{0,  0,  0,  0,  0,},
                     {3,  4,  5,  6,  7,},
                     {6,  8, 10, 12, 14},
                     {9, 12, 15, 18, 21,},
                     {12, 16, 20, 24, 28,}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.divide, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                    {{0,  0,  0,  0,  0},
                     {0,  0,  0,  0,  0},
                     {0,  0,  0,  0,  0},
                     {1,  0,  0,  0,  0},
                     {1,  1,  0,  0,  0}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.remainder, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                    {{0, 0, 0, 0, 0,},
                     {1, 1, 1, 1, 1,},
                     {2, 2, 2, 2, 2},
                     {0, 3, 3, 3, 3,},
                     {1, 0, 4, 4, 4,}};

            AssertArray(b, ExpectedData);


        }

        [TestMethod]
        public void test_FModOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.fmod, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                    {{0, 0, 0, 0, 0,},
                     {1, 1, 1, 1, 1,},
                     {2, 2, 2, 2, 2},
                     {0, 3, 3, 3, 3,},
                     {1, 0, 4, 4, 4,}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.square, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 4, 4, 4, 4, 4 },
                  { 9, 9, 9, 9, 9 },
                  { 16, 16, 16, 16, 16 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.reciprocal, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                    { { 0, 0, 0, 0, 0 },
                      { 1, 1, 1, 1, 1 },
                      { 0, 0, 0, 0, 0 },
                      { 0, 0, 0, 0, 0 },
                      { 0, 0, 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.ones_like, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 1, 1, 1, 1, 1 },
                  { 1, 1, 1, 1, 1 },
                  { 1, 1, 1, 1, 1 },
                  { 1, 1, 1, 1, 1 },
                  { 1, 1, 1, 1, 1 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.sqrt, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
             { { 0, 0, 0, 0, 0 },
               { 1, 1, 1, 1, 1 },
               { 1, 1, 1, 1, 1 },
               { 2, 2, 2, 2, 2 },
               { 2, 2, 2, 2, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.negative, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                   { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.absolute, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.invert, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.left_shift, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                  { { 0, 0, 0, 0, 0 },
                  { 8, 16, 32, 64, 128 },
                  { 16, 32, 64, 128, 256 },
                  { 24, 48, 96, 192, 384 },
                  { 32, 64, 128, 256, 512  } };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64) * 1024 * 4;
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.right_shift, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 0, 0, 0, 0, 0  },
                  { 512, 256, 128, 64, 32 },
                  { 1024, 512, 256, 128, 64  },
                  { 1536, 768, 384, 192, 96  },
                  { 2048, 1024, 512, 256, 128  } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.bitwise_and, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { {  0, 0, 0, 0, 0  },
                  { 1, 0, 1, 0, 1  },
                  { 2, 0, 0, 2, 2  },
                  { 3, 0, 1, 2, 3  },
                  { 0, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.bitwise_or, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 3, 4, 5, 6, 7 },
                  { 3, 5, 5, 7, 7 },
                  { 3, 6, 7, 6, 7 },
                  { 3, 7, 7, 7, 7 },
                  { 7, 4, 5, 6, 7 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.bitwise_xor, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 3, 4, 5, 6, 7 },
                  { 2, 5, 4, 7, 6 },
                  { 1, 6, 7, 4, 5 },
                  { 0, 7, 6, 5, 4 },
                  { 7, 0, 1, 2, 3 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.less, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true },
                                         { false, false, true, true, true } });

        }


        [TestMethod]
        public void test_LessEqualOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.less_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true } });

        }

        [TestMethod]
        public void test_EqualOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false},
                                        {false, true, false, false, false}});

        }


        [TestMethod]
        public void test_NotEqualOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.not_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true },
                                         { true, false, true, true, true } });

        }

        [TestMethod]
        public void test_GreaterOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.greater, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false}});

        }

        [TestMethod]
        public void test_GreaterEqualOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.greater_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false},
                                        {true, true, false, false, false}});

        }

        [TestMethod]
        public void test_FloorDivideOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.floor_divide, null, a1, a2);
            print(b);


            var ExpectedData = new UInt64[,]
                { { 0, 0, 0, 0, 0 },
                  { 0, 0, 0, 0, 0 },
                  { 0, 0, 0, 0, 0 },
                  { 1, 0, 0, 0, 0 },
                  { 1, 1, 0, 0, 0} };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.true_divide, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 0, 0, 0, 0, 0 },
                  { 0, 0, 0, 0, 0 },
                  { 0, 0, 0, 0, 0 },
                  { 1, 0, 0, 0, 0 },
                  { 1, 1, 0, 0, 0} };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.logical_and, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true}});

        }

        [TestMethod]
        public void test_LogicalOrOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.logical_or, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true}});

        }


        [TestMethod]
        public void test_FloorOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.floor, null, a1, a2);
            print(b);


            var ExpectedData = new UInt64[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.ceil, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                 { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.maximum, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 4, 4, 5, 6, 7  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.minimum, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 3, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.rint, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.conjugate, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.isnan, null, a1, a2);
            print(b);


            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false}});

        }


        [TestMethod]
        public void test_FMaxOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.fmax, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 4, 4, 5, 6, 7  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.fmin, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 3, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideOuter_UINT64()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt64);
            var a2 = np.arange(3, 8, dtype: np.UInt64);

            var b = np.ufunc.outer(UFuncOperation.heaviside, null, a1, a2);
            print(b);

            var ExpectedData = new UInt64[,]
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
        public void test_AddReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.add, a1);
            print(b);

            var ExpectedData = new UInt64[] { 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.subtract, a1);
            print(b);

            var ExpectedData = new UInt64[] 
                { 18446744073709551166, 18446744073709551158, 18446744073709551150,
                  18446744073709551142, 18446744073709551134, 18446744073709551126,
                  18446744073709551118, 18446744073709551110, 18446744073709551102, 18446744073709551094 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.multiply, a1);
            print(b);

            var ExpectedData = new UInt64[] 
            
                { 0, 478015854767451, 1242688846823424, 2394832584543399,
                  4060162871525376, 6393838623046875, 9585618768101376,
                  13865696119905399, 19511273389031424, 26853950884211451 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.divide, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
 
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.remainder, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmod, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.square, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 1, 0, 2118395047680534529, 0, 8643096425819600897,
                                              0, 17388776497041092609, 0, 11359380750133874689 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.reciprocal, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.ones_like, a1);
            print(b);

            var ExpectedData = new UInt64[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.sqrt, a1);
            print(b);

            var ExpectedData = new UInt64[] {  0, 1, 1, 1, 1, 1, 1, 1, 1, 1  };



            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.negative, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.absolute, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.invert, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.left_shift, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.right_shift, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_and, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_or, a1);
            print(b);

            var ExpectedData = new UInt64[] { 126, 127, 126, 127, 126, 127, 126, 127, 126, 127 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_xor, a1);
            print(b);

            var ExpectedData = new UInt64[] { 106, 106, 94, 94, 42, 42, 6, 6, 26, 26 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.less, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }


        [TestMethod]
        public void test_LessEqualReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.less_equal, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_EqualReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.equal, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });

        }


        [TestMethod]
        public void test_NotEqualReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.not_equal, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });
        }

        [TestMethod]
        public void test_GreaterReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.greater, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_GreaterEqualReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.greater_equal, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_FloorDivideReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.floor_divide, a1);
            print(b);


            var ExpectedData = new UInt64[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.true_divide, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };


            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.logical_and, a1);
            print(b);

            AssertArray(b, new bool[] { false, true, true, true, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_LogicalOrReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.logical_or, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }


        [TestMethod]
        public void test_FloorReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.floor, a1);
            print(b);


            var ExpectedData = new UInt64[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.ceil, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.maximum, a1);
            print(b);

            var ExpectedData = new UInt64[] { 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.minimum, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.rint, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.conjugate, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.isnan, a1);
            print(b);


            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }


        [TestMethod]
        public void test_FMaxReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmax, a1);
            print(b);

            var ExpectedData = new UInt64[] { 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmin, a1);
            print(b);

            var ExpectedData = new UInt64[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideReduce_UINT64()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt64).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.heaviside, a1);
            print(b);

            var ExpectedData = new UInt64[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #region ACCUMULATE tests
        [TestMethod]
        public void test_AddAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.add, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 3, 5, 7 }, { 9, 12, 15 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.subtract, a1);
            print(b);

            var ExpectedData = new UInt64[,] 
                    { { 0, 1, 2 },
                      { 18446744073709551613, 18446744073709551613, 18446744073709551613 },
                      { 18446744073709551607, 18446744073709551606, 18446744073709551605 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.multiply, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 4, 10 }, { 0, 28, 80 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.divide, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.remainder, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmod, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.square, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 4 }, { 0, 1, 16 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.reciprocal, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 0 }, { 0, 1, 0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.ones_like, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 1, 1, 1 }, { 1, 1, 1 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.sqrt, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 1 }, { 0, 1, 1 } };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.negative, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.absolute, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.invert, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.left_shift, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 16, 64 }, { 0, 2048, 16384 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.right_shift, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_and, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_or, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 3, 5, 7 }, { 7, 7, 15 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_xor, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 3, 5, 7 }, { 5, 2, 15 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.less, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }


        [TestMethod]
        public void test_LessEqualAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.less_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }

        [TestMethod]
        public void test_EqualAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });

        }


        [TestMethod]
        public void test_NotEqualAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.not_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });
        }

        [TestMethod]
        public void test_GreaterAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.greater, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }

        [TestMethod]
        public void test_GreaterEqualAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.greater_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }

        [TestMethod]
        public void test_FloorDivideAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.floor_divide, a1);
            print(b);


            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.true_divide, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.logical_and, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, true, true }, { false, true, true } });

        }

        [TestMethod]
        public void test_LogicalOrAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.logical_or, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }


        [TestMethod]
        public void test_FloorAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.floor, a1);
            print(b);


            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.ceil, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.maximum, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.minimum, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.rint, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.conjugate, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.isnan, a1);
            print(b);


            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }


        [TestMethod]
        public void test_FMaxAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmax, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmin, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideAccumulate_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.heaviside, a1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 1, 2 }, { 3, 1, 1 }, { 1, 1, 1 } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #region REDUCEAT tests
        [TestMethod]
        public void test_AddReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.add, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 1, 2 }, { 7, 5 }, { 13, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.subtract, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] 
                { { 18446744073709551615, 2 },
                  { 18446744073709551615, 5 },
                  { 18446744073709551615, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.multiply, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 12, 5 }, { 42, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.remainder, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmod, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.square, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 9, 5 }, { 36, 8 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.reciprocal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.ones_like, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 1, 2 }, { 1, 5 }, { 1, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.sqrt, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 2, 5 }, { 2, 8 } };
             

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.negative, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.absolute, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.invert, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.left_shift, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 48, 5 }, { 768, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.right_shift, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_and, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 0, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_or, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 1, 2 }, { 7, 5 }, { 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_xor, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 1, 2 }, { 7, 5 }, { 1, 8 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.less, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }


        [TestMethod]
        public void test_LessEqualReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.less_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }

        [TestMethod]
        public void test_EqualReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });

        }


        [TestMethod]
        public void test_NotEqualReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.not_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });
        }

        [TestMethod]
        public void test_GreaterReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.greater, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }

        [TestMethod]
        public void test_GreaterEqualReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.greater_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }

        [TestMethod]
        public void test_FloorDivideReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.floor_divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            var ExpectedData = new UInt64[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.true_divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.logical_and, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { true, true }, { true, true } });

        }

        [TestMethod]
        public void test_LogicalOrReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.logical_or, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }


        [TestMethod]
        public void test_FloorReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.floor, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            var ExpectedData = new UInt64[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.ceil, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.maximum, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 1, 2 }, { 4, 5 }, { 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.minimum, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.rint, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.conjugate, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.isnan, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }


        [TestMethod]
        public void test_FMaxReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmax, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 1, 2 }, { 4, 5 }, { 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmin, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideReduceAt_UINT64()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt64).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.heaviside, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt64[,] { { 1, 2 }, { 1, 5 }, { 1, 8 } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #endregion
    }
}
