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
    public class UFUNC_UINT16_Tests : TestBaseClass
    {
        #region UFUNC UINT16 Tests

        #region OUTER tests
        [TestMethod]
        public void test_AddOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.add, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                    {{3,  4,  5,  6,  7},
                     {4,  5,  6,  7,  8},
                     {5,  6,  7,  8,  9},
                     {6,  7,  8,  9, 10},
                     {7,  8,  9, 10, 11}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.subtract, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 65533, 65532, 65531, 65530, 65529 },
                  { 65534, 65533, 65532, 65531, 65530 },
                  { 65535, 65534, 65533, 65532, 65531 },
                  { 0, 65535, 65534, 65533, 65532 },
                  { 1, 0, 65535, 65534, 65533 }
                };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.multiply, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                    {{0,  0,  0,  0,  0,},
                     {3,  4,  5,  6,  7,},
                     {6,  8, 10, 12, 14},
                     {9, 12, 15, 18, 21,},
                     {12, 16, 20, 24, 28,}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.divide, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                    {{0,  0,  0,  0,  0},
                     {0,  0,  0,  0,  0},
                     {0,  0,  0,  0,  0},
                     {1,  0,  0,  0,  0},
                     {1,  1,  0,  0,  0}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.remainder, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                    {{0, 0, 0, 0, 0,},
                     {1, 1, 1, 1, 1,},
                     {2, 2, 2, 2, 2},
                     {0, 3, 3, 3, 3,},
                     {1, 0, 4, 4, 4,}};

            AssertArray(b, ExpectedData);


        }

        [TestMethod]
        public void test_FModOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.fmod, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                    {{0, 0, 0, 0, 0,},
                     {1, 1, 1, 1, 1,},
                     {2, 2, 2, 2, 2},
                     {0, 3, 3, 3, 3,},
                     {1, 0, 4, 4, 4,}};

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.square, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 4, 4, 4, 4, 4 },
                  { 9, 9, 9, 9, 9 },
                  { 16, 16, 16, 16, 16 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.reciprocal, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                    { { 0, 0, 0, 0, 0 },
                      { 1, 1, 1, 1, 1 },
                      { 0, 0, 0, 0, 0 },
                      { 0, 0, 0, 0, 0 },
                      { 0, 0, 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.ones_like, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 1, 1, 1, 1, 1 },
                  { 1, 1, 1, 1, 1 },
                  { 1, 1, 1, 1, 1 },
                  { 1, 1, 1, 1, 1 },
                  { 1, 1, 1, 1, 1 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.sqrt, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
             { { 0, 0, 0, 0, 0 },
               { 1, 1, 1, 1, 1 },
               { 1, 1, 1, 1, 1 },
               { 2, 2, 2, 2, 2 },
               { 2, 2, 2, 2, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.negative, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                   { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.absolute, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.invert, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
            {
                { 65535, 65535, 65535, 65535, 65535 },
                { 65534, 65534, 65534, 65534, 65534 },
                { 65533, 65533, 65533, 65533, 65533 },
                { 65532, 65532, 65532, 65532, 65532 },
                { 65531, 65531, 65531, 65531, 65531 },
            };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.left_shift, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                  { { 0, 0, 0, 0, 0 },
                  { 8, 16, 32, 64, 128 },
                  { 16, 32, 64, 128, 256 },
                  { 24, 48, 96, 192, 384 },
                  { 32, 64, 128, 256, 512  } };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16) * 1024 * 4;
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.right_shift, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 0, 0, 0, 0, 0  },
                  { 512, 256, 128, 64, 32 },
                  { 1024, 512, 256, 128, 64  },
                  { 1536, 768, 384, 192, 96  },
                  { 2048, 1024, 512, 256, 128  } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.bitwise_and, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { {  0, 0, 0, 0, 0  },
                  { 1, 0, 1, 0, 1  },
                  { 2, 0, 0, 2, 2  },
                  { 3, 0, 1, 2, 3  },
                  { 0, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.bitwise_or, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 3, 4, 5, 6, 7 },
                  { 3, 5, 5, 7, 7 },
                  { 3, 6, 7, 6, 7 },
                  { 3, 7, 7, 7, 7 },
                  { 7, 4, 5, 6, 7 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.outer(UFuncOperation.bitwise_xor, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 3, 4, 5, 6, 7 },
                  { 2, 5, 4, 7, 6 },
                  { 1, 6, 7, 4, 5 },
                  { 0, 7, 6, 5, 4 },
                  { 7, 0, 1, 2, 3 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.less, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true },
                                         { false, false, true, true, true } });

        }


        [TestMethod]
        public void test_LessEqualOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.less_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true } });

        }

        [TestMethod]
        public void test_EqualOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false},
                                        {false, true, false, false, false}});

        }


        [TestMethod]
        public void test_NotEqualOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.not_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] { { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { true, true, true, true, true },
                                         { false, true, true, true, true },
                                         { true, false, true, true, true } });

        }

        [TestMethod]
        public void test_GreaterOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.greater, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false}});

        }

        [TestMethod]
        public void test_GreaterEqualOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.greater_equal, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {true, false, false, false, false},
                                        {true, true, false, false, false}});

        }

        [TestMethod]
        public void test_FloorDivideOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.floor_divide, null, a1, a2);
            print(b);


            var ExpectedData = new UInt16[,]
                { { 0, 0, 0, 0, 0 },
                  { 0, 0, 0, 0, 0 },
                  { 0, 0, 0, 0, 0 },
                  { 1, 0, 0, 0, 0 },
                  { 1, 1, 0, 0, 0} };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.true_divide, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 0, 0, 0, 0, 0 },
                  { 0, 0, 0, 0, 0 },
                  { 0, 0, 0, 0, 0 },
                  { 1, 0, 0, 0, 0 },
                  { 1, 1, 0, 0, 0} };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.logical_and, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true}});

        }

        [TestMethod]
        public void test_LogicalOrOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.logical_or, null, a1, a2);
            print(b);

            AssertArray(b, new bool[,] {{true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true},
                                        {true, true, true, true, true}});

        }


        [TestMethod]
        public void test_FloorOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.floor, null, a1, a2);
            print(b);


            var ExpectedData = new UInt16[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.ceil, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                 { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.maximum, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 4, 4, 5, 6, 7  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.minimum, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 3, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.rint, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.conjugate, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.isnan, null, a1, a2);
            print(b);


            AssertArray(b, new bool[,] {{false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false},
                                        {false, false, false, false, false}});

        }


        [TestMethod]
        public void test_FMaxOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.fmax, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 3, 4, 5, 6, 7 },
                  { 4, 4, 5, 6, 7  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.fmin, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 0, 0, 0, 0, 0 },
                  { 1, 1, 1, 1, 1 },
                  { 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3 },
                  { 3, 4, 4, 4, 4  } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideOuter_UINT16()
        {
            var a1 = np.arange(0, 5, dtype: np.UInt16);
            var a2 = np.arange(3, 8, dtype: np.UInt16);

            var b = np.ufunc.outer(UFuncOperation.heaviside, null, a1, a2);
            print(b);

            var ExpectedData = new UInt16[,]
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
        public void test_AddReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.add, a1);
            print(b);

            var ExpectedData = new UInt16[] { 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.subtract, a1);
            print(b);

            var ExpectedData = new UInt16[] { 65086, 65078, 65070, 65062, 65054, 65046, 65038, 65030, 65022, 65014 };
             

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.multiply, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 27995, 0, 36007, 0, 48347, 0, 23671, 0, 60155 };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.divide, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
 
            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.remainder, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmod, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.square, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 1, 0, 59393, 0, 63489, 0, 45057, 0, 53249 };


            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.reciprocal, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.ones_like, a1);
            print(b);

            var ExpectedData = new UInt16[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.sqrt, a1);
            print(b);

            var ExpectedData = new UInt16[] {  0, 1, 1, 1, 1, 1, 1, 1, 1, 1  };



            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.negative, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.absolute, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.invert, a1);
            print(b);

            var ExpectedData = new UInt16[] { 65535, 65534, 65533, 65532, 65531, 65530, 65529, 65528, 65527, 65526 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.left_shift, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.right_shift, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_and, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_or, a1);
            print(b);

            var ExpectedData = new UInt16[] { 126, 127, 126, 127, 126, 127, 126, 127, 126, 127 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduce(UFuncOperation.bitwise_xor, a1);
            print(b);

            var ExpectedData = new UInt16[] { 106, 106, 94, 94, 42, 42, 6, 6, 26, 26 };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.less, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }


        [TestMethod]
        public void test_LessEqualReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.less_equal, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_EqualReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.equal, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });

        }


        [TestMethod]
        public void test_NotEqualReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.not_equal, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });
        }

        [TestMethod]
        public void test_GreaterReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.greater, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_GreaterEqualReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.greater_equal, a1);
            print(b);

            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_FloorDivideReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.floor_divide, a1);
            print(b);


            var ExpectedData = new UInt16[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.true_divide, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };


            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.logical_and, a1);
            print(b);

            AssertArray(b, new bool[] { false, true, true, true, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_LogicalOrReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.logical_or, a1);
            print(b);

            AssertArray(b, new bool[] { true, true, true, true, true, true, true, true, true, true });

        }


        [TestMethod]
        public void test_FloorReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.floor, a1);
            print(b);


            var ExpectedData = new UInt16[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.ceil, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.maximum, a1);
            print(b);

            var ExpectedData = new UInt16[] { 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.minimum, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.rint, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.conjugate, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.isnan, a1);
            print(b);


            AssertArray(b, new bool[] { false, false, false, false, false, false, false, false, false, false });
        }


        [TestMethod]
        public void test_FMaxReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmax, a1);
            print(b);

            var ExpectedData = new UInt16[] { 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.fmin, a1);
            print(b);

            var ExpectedData = new UInt16[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideReduce_UINT16()
        {
            var a1 = np.arange(0, 100, dtype: np.UInt16).reshape((10, 10));

            var b = np.ufunc.reduce(UFuncOperation.heaviside, a1);
            print(b);

            var ExpectedData = new UInt16[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #region ACCUMULATE tests
        [TestMethod]
        public void test_AddAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.add, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 3, 5, 7 }, { 9, 12, 15 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.subtract, a1);
            print(b);

            var ExpectedData = new UInt16[,] 
                { { 0, 1, 2 },
                  { 65533, 65533, 65533  },
                  { 65527, 65526, 65525 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.multiply, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 4, 10 }, { 0, 28, 80 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.divide, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.remainder, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmod, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.square, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 1, 4 }, { 0, 1, 16 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.reciprocal, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 1, 0 }, { 0, 1, 0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.ones_like, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 1, 1, 1 }, { 1, 1, 1 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.sqrt, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 1, 1 }, { 0, 1, 1 } };


            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.negative, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.absolute, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.invert, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 65535, 65534, 65533 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.left_shift, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 16, 64 }, { 0, 2048, 16384 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.right_shift, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_and, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_or, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 3, 5, 7 }, { 7, 7, 15 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.accumulate(UFuncOperation.bitwise_xor, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 3, 5, 7 }, { 5, 2, 15 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.less, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }


        [TestMethod]
        public void test_LessEqualAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.less_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }

        [TestMethod]
        public void test_EqualAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });

        }


        [TestMethod]
        public void test_NotEqualAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.not_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });
        }

        [TestMethod]
        public void test_GreaterAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.greater, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }

        [TestMethod]
        public void test_GreaterEqualAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.greater_equal, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }

        [TestMethod]
        public void test_FloorDivideAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.floor_divide, a1);
            print(b);


            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.true_divide, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 0, 0 }, { 0, 0, 0 } };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.logical_and, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { false, true, true }, { false, true, true } });

        }

        [TestMethod]
        public void test_LogicalOrAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.logical_or, a1);
            print(b);

            AssertArray(b, new bool[,] { { false, true, true }, { true, true, true }, { true, true, true } });

        }


        [TestMethod]
        public void test_FloorAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.floor, a1);
            print(b);


            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.ceil, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.maximum, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.minimum, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.rint, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.conjugate, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.isnan, a1);
            print(b);


            AssertArray(b, new bool[,] { { false, true, true }, { false, false, false }, { false, false, false } });
        }


        [TestMethod]
        public void test_FMaxAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmax, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.fmin, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 0, 1, 2 }, { 0, 1, 2 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideAccumulate_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.accumulate(UFuncOperation.heaviside, a1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 1, 2 }, { 3, 1, 1 }, { 1, 1, 1 } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #region REDUCEAT tests
        [TestMethod]
        public void test_AddReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.add, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 1, 2 }, { 7, 5 }, { 13, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SubtractReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.subtract, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,]
                { { 65535, 2 },
                  { 65535, 5 },
                  { 65535, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MultiplyReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.multiply, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 12, 5 }, { 42, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_DivideReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RemainderReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.remainder, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);
        }

        [TestMethod]
        public void test_FModReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmod, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SquareReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.square, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 9, 5 }, { 36, 8 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_ReciprocalReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.reciprocal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_OnesLikeReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.ones_like, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 1, 2 }, { 1, 5 }, { 1, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_SqrtReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.sqrt, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 2, 5 }, { 2, 8 } };
             

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_NegativeReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.negative, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_AbsoluteReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.absolute, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_InvertReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.invert, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 65535, 2 }, { 65532, 5 }, { 65529, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_LeftShiftReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.left_shift, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 48, 5 }, { 768, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RightShiftReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.right_shift, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_BitwiseAndReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_and, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 0, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseOrReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_or, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 1, 2 }, { 7, 5 }, { 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_BitwiseXorReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            // note: python throws an exception here because squares don't have two arguments.  I don't care.
            var b = np.ufunc.reduceat(UFuncOperation.bitwise_xor, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 1, 2 }, { 7, 5 }, { 1, 8 } };

            AssertArray(b, ExpectedData);

        }


        [TestMethod]
        public void test_LessReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.less, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }


        [TestMethod]
        public void test_LessEqualReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.less_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }

        [TestMethod]
        public void test_EqualReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });

        }


        [TestMethod]
        public void test_NotEqualReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.not_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });
        }

        [TestMethod]
        public void test_GreaterReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.greater, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }

        [TestMethod]
        public void test_GreaterEqualReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.greater_equal, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }

        [TestMethod]
        public void test_FloorDivideReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.floor_divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            var ExpectedData = new UInt16[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_trueDivideReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.true_divide, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 0, 5 }, { 0, 8 } };

            AssertArray(b, ExpectedData);
        }


        [TestMethod]
        public void test_LogicalAndReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.logical_and, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { false, true }, { true, true }, { true, true } });

        }

        [TestMethod]
        public void test_LogicalOrReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.logical_or, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            AssertArray(b, new bool[,] { { true, true }, { true, true }, { true, true } });

        }


        [TestMethod]
        public void test_FloorReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.floor, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            var ExpectedData = new UInt16[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_CeilReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.ceil, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MaximumReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.maximum, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 1, 2 }, { 4, 5 }, { 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_MinimumReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.minimum, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_RintReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.rint, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_ConjugateReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.conjugate, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_IsNANReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.isnan, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);


            AssertArray(b, new bool[,] { { false, true }, { false, true }, { false, true } });
        }


        [TestMethod]
        public void test_FMaxReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmax, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 1, 2 }, { 4, 5 }, { 7, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_FMinReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.fmin, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 0, 2 }, { 3, 5 }, { 6, 8 } };

            AssertArray(b, ExpectedData);

        }

        [TestMethod]
        public void test_HeavisideReduceAt_UINT16()
        {
            var a1 = np.arange(0, 9, dtype: np.UInt16).reshape((3, 3));

            var b = np.ufunc.reduceat(UFuncOperation.heaviside, a1, new npy_intp[] { 0, 2 }, axis: 1);
            print(b);

            var ExpectedData = new UInt16[,] { { 1, 2 }, { 1, 5 }, { 1, 8 } };

            AssertArray(b, ExpectedData);

        }
        #endregion

        #endregion
    }
}
