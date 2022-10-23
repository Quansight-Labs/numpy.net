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
    public class NumericTests : TestBaseClass
    {
        [TestMethod]
        public void test_zeros_1()
        {
            var x = np.zeros(new shape(10));
            print(x);
            print("Update sixth value to 11");
            x[6] = 11;
            print(x);
            print(x.shape);
            print(x.strides);

            AssertArray(x, new double[] { 0, 0, 0, 0, 0, 0, 11, 0, 0, 0 });
            AssertShape(x, 10);
            AssertStrides(x, sizeof(double));
        }

        [TestMethod]
        public void test_zeros_like_1()
        {
            var a = new Int32[] { 1, 2, 3, 4, 5, 6 };
            var b = np.zeros_like(a, dtype: null);
            b[2] = 99;

            AssertArray(b, new Int32[] { 0, 0, 99, 0, 0, 0 });

            return;

        }

        [TestMethod]
        public void test_zeros_like_2()
        {
            var a = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            var b = np.zeros_like(a);
            b[1, 2] = 99;

            AssertArray(b, new double[,] { { 0, 0, 0 }, { 0, 0, 99 } });

            return;
        }

        [TestMethod]
        public void test_zeros_like_3()
        {
            var a = new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } };
            var b = np.zeros_like(a);
            b[0, 0, 2] = 99;
            b[0, 1, 1] = 88;

            AssertArray(b, new double[,,] { { { 0, 0, 99 }, { 0, 88, 0 } } });

            return;
        }

        [TestMethod]
        public void test_ones_1()
        {
            var x = np.ones(new shape(10));
            print(x);
            print("Update sixth value to 11");
            x[6] = 11;
            print(x);
            print(x.shape);
            print(x.strides);

            AssertArray(x, new double[] { 1, 1, 1, 1, 1, 1, 11, 1, 1, 1 });
            AssertShape(x, 10);
            AssertStrides(x, sizeof(double));
        }

        [TestMethod]
        public void test_ones_like_1()
        {
            var a = new Int32[] { 1, 2, 3, 4, 5, 6 };
            var b = np.ones_like(a, dtype: null);
            b[2] = 99;

            AssertArray(b, new Int32[] { 1, 1, 99, 1, 1, 1 });

            return;

        }

        [TestMethod]
        public void test_ones_like_2()
        {
            var a = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            var b = np.ones_like(a);
            b[1, 2] = 99;

            AssertArray(b, new double[,] { { 1, 1, 1 }, { 1, 1, 99 } });

            return;
        }

        [TestMethod]
        public void test_ones_like_3()
        {
            var a = new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } };
            var b = np.ones_like(a);
            b[0, 0, 2] = 99;
            b[0, 1, 1] = 88;

            AssertArray(b, new double[,,] { { { 1, 1, 99 }, { 1, 88, 1 } } });

            return;
        }

        [TestMethod]
        public void test_empty()
        {
            var a = np.empty((2, 3));
            AssertShape(a, 2, 3);
            Assert.AreEqual(a.Dtype.TypeNum, NPY_TYPES.NPY_DOUBLE);

            var b = np.empty((2, 4), np.Int32);
            AssertShape(b, 2, 4);
            Assert.AreEqual(b.Dtype.TypeNum, NPY_TYPES.NPY_INT32);
        }

        [TestMethod]
        public void test_empty_like_1()
        {
            var a = new Int32[] { 1, 2, 3, 4, 5, 6 };
            var b = np.empty_like(a, dtype: null);
            b[2] = 99;

            AssertArray(b, new Int32[] { 0, 0, 99, 0, 0, 0 });

            return;

        }

        [TestMethod]
        public void test_empty_like_2()
        {
            var a = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            var b = np.empty_like(a);
            b[1, 2] = 99;

            AssertArray(b, new double[,] { { 0, 0, 0 }, { 0, 0, 99 } });

            return;
        }

        [TestMethod]
        public void test_empty_like_3()
        {
            var a = new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } };
            var b = np.empty_like(a);
            b[0, 0, 2] = 99;
            b[0, 1, 1] = 88;

            AssertArray(b, new double[,,] { { { 0, 0, 99 }, { 0, 88, 0 } } });

            return;
        }

        [TestMethod]
        public void test_full_1()
        {
            var x = np.full((10), 99);
            print(x);
            print("Update sixth value to 11");
            x[6] = 11;
            print(x);
            print(x.shape);
            print(x.strides);

            AssertArray(x, new double[] { 99, 99, 99, 99, 99, 99, 11, 99, 99, 99 });
            AssertShape(x, 10);
            AssertStrides(x, sizeof(double));

        }

        [TestMethod]
        public void test_full_2()
        {
            var x = np.full((100), 99).reshape(new shape(10, 10));
            print(x);
            print("Update sixth value to 11");
            x[6] = 55;
            print(x);
            print(x.shape);
            print(x.strides);

            //AssertArray(y, new float[] { 60, 61, 62, 63, 64, 65, 66, 67, 68, 69 });
            //AssertShape(y, 10);
            //AssertStrides(y, sizeof(float));

            //x[5, 5] = 12;
            //print(x);
            //print(x.shape);
            //print(x.strides);
        }

        [TestMethod]
        public void test_full_3()
        {
            var x = np.full((100), 1, np.Float32);
            print(x);
            print("Update sixth value to 11");

            var y = x[62];
            print(y);
        }

        [TestMethod]
        public void test_full_4()
        {
            var x = np.full((100), 1, np.Float32).reshape(new shape(10, 10));
            print(x);
            print("Update sixth value to 11");
            x[6] = 55;
            print(x);
            print(x.shape);
            print(x.strides);

            //ndarray y = x[5] as ndarray;
            //AssertArray(y, new float[] { 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 });
            //AssertShape(y, 10);
            //AssertStrides(y, sizeof(float));

            //y = x[6] as ndarray;
            //AssertArray(y, new float[] { 55, 55, 55, 55, 55, 55, 55, 55, 55, 55 });
            //AssertShape(y, 10);
            //AssertStrides(y, sizeof(float));

            //y = x[7] as ndarray;
            //AssertArray(y, new float[] { 70, 71, 72, 73, 74, 75, 76, 77, 78, 79 });
            //AssertShape(y, 10);
            //AssertStrides(y, sizeof(float));

        }

        [TestMethod]
        public void test_full_like_1()
        {
            var a = new Int32[] { 1, 2, 3, 4, 5, 6 };
            var b = np.full_like(a, 66, dtype: null);
            b[2] = 99;

            AssertArray(b, new Int32[] { 66, 66, 99, 66, 66, 66 });

            return;

        }

        [TestMethod]
        public void test_full_like_2()
        {
            var a = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            var b = np.full_like(a, 55);
            b[1, 2] = 99;

            AssertArray(b, new double[,] { { 55, 55, 55 }, { 55, 55, 99 } });

            return;
        }

        [TestMethod]
        public void test_full_like_3()
        {
            var a = new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } };
            var b = np.full_like(a, 33);
            b[0, 0, 2] = 99;
            b[0, 1, 1] = 88;

            AssertArray(b, new double[,,] { { { 33, 33, 99 }, { 33, 88, 33 } } });

            return;
        }

        [TestMethod]
        public void test_count_nonzero_1()
        {
            var a = np.count_nonzero(np.eye(4));
            Assert.AreEqual(4, (int)a);
            print(a);

            var b = np.count_nonzero(new int[,] { { 0, 1, 7, 0, 0 }, { 3, 0, 0, 2, 19 } });
            Assert.AreEqual(5, (int)b);
            print(b);

            var c = np.count_nonzero(new int[,] { { 0, 1, 7, 0, 0 }, { 3, 0, 0, 2, 19 } }, axis:0);
            AssertArray(c, new int[] { 1,1,1,1,1});
            print(c);

            var d = np.count_nonzero(new int[,] { { 0, 1, 7, 0, 0 }, { 3, 0, 0, 2, 19 } }, axis:1);
            AssertArray(d, new int[] { 2,3 });
            print(d);

            return;
        }

        [TestMethod]
        public void test_asarray_1()
        {
            var a = new int[] { 1, 2 };
            var b = np.asarray(a);

            AssertArray(b, new int[] { 1, 2 });
            print(b);

            var c = np.array(new float[] { 1.0f, 2.0f }, dtype: np.Float32);
            var d = np.asarray(c, dtype: np.Float32);

            c[0] = 3.0f;
            AssertArray(d, new float[] { 3.0f, 2.0f });
            print(d);

            var e = np.asarray(a, dtype: np.Float64);
            AssertArray(e, new double[] { 1.0, 2.0 });

            print(e);

            return;
        }

        [TestMethod]
        public void test_asanyarray_1()
        {
            var a = new int[] { 1, 2 };
            var b = np.asanyarray(a);

            AssertArray(b, new int[] { 1, 2 });
            print(b);

            var c = np.array(new float[] { 1.0f, 2.0f }, dtype: np.Float32);
            var d = np.asanyarray(c, dtype: np.Float32);

            c[0] = 3.0f;
            AssertArray(d, new float[] { 3.0f, 2.0f });
            print(d);

            var e = np.asanyarray(a, dtype: np.Float64);
            AssertArray(e, new double[] { 1.0, 2.0 });

            print(e);

            return;
        }

        [TestMethod]
        public void test_asanyarray_2()
        {
            var a = new int[] { 1, 2,3,4 };
            var b = new int[] { 10, 20, 30, 40 };

            try
            {
                // NOTE:  This is different behavior than python.  Necessary to support NPY_TYPES.NPY_OBJECT.
                var c = np.asanyarray(new object[] { a, b });
                print(c);
                Assert.Fail("This should have thrown an exception");
            }
            catch
            {
                
            }

            var aa = np.array(a);
            var ba = np.array(b);
            var ca = np.asanyarray(new ndarray[] { aa, ba, aa });

            AssertArray(ca, new Int32[,] { { 1, 2, 3, 4 }, { 10, 20, 30, 40 }, { 1, 2, 3, 4 } });
            print(ca);


            return;
        }

        [TestMethod]
        public void test_asanyarray_3()
        {
            var a = new int[] { 1, 2, 3, 4 };
            var b = new int[] { 10, 20, 30, 40 };

            try
            {
                // NOTE:  This is different behavior than python.  Necessary to support NPY_TYPES.NPY_OBJECT.
                var c = np.asanyarray(new object[] { a, b });
                print(c);
                Assert.Fail("This should have thrown an exception");
            }
            catch
            {

            }

            var aa = np.array(a).reshape(2,2);
            var ba = np.array(b).reshape(2,2);
            var ca = np.asanyarray(new ndarray[] { aa, ba });

            //AssertArray(ca, new Int32[,,] { { 1, 2 }, {3, 4 }, { 10, 20 }, {30, 40 }, { 1, 2 }, {3, 4 } });
            print(ca);


            return;
        }


        [TestMethod]
        public void test_ascontiguousarray_1()
        {
            var x = np.arange(6).reshape((2, 3));
            var y = np.ascontiguousarray(x, dtype: np.Float32);

            AssertArray(y, new float[,] { { 0f, 1f, 2f }, { 3f, 4f, 5 } });
            print(y);

            Assert.AreEqual(x.flags.c_contiguous, true);
            Assert.AreEqual(y.flags.c_contiguous, true);

            return;
        }

        [TestMethod]
        public void test_asfortranarray_1()
        {
            var x = np.arange(6).reshape((2, 3));
            var y = np.asfortranarray(x, dtype: np.Float32);

            AssertArray(y, new float[,] { { 0f, 1f, 2f }, { 3f, 4f, 5 } });
            print(y);

            Assert.AreEqual(x.flags.f_contiguous, false);
            Assert.AreEqual(y.flags.f_contiguous, true);

            return;
        }

#if NOT_PLANNING_TODO
        [Ignore] // need to fully implement np.require.
        [TestMethod]
        public void xxx_test_require_1()
        {
            var x = np.arange(6).reshape((2, 3));
            Assert.AreEqual(x.flags.c_contiguous, true);
            Assert.AreEqual(x.flags.f_contiguous, false);
            Assert.AreEqual(x.flags.owndata, false);
            Assert.AreEqual(x.flags.writeable, true);
            Assert.AreEqual(x.flags.aligned, true);
            //Assert.AreEqual(x.flags.writebackifcopy, false);
            Assert.AreEqual(x.flags.updateifcopy, false);

            var y = np.require(x, np.Float32, new char[] { 'A', 'O', 'W', 'F' });

            Assert.AreEqual(y.flags.c_contiguous, false);
            Assert.AreEqual(y.flags.f_contiguous, true);
            Assert.AreEqual(y.flags.owndata, true);
            Assert.AreEqual(y.flags.writeable, true);
            Assert.AreEqual(y.flags.aligned, true);
            //Assert.AreEqual(y.flags.writebackifcopy, false);
            Assert.AreEqual(y.flags.updateifcopy, false);

            return;
        }
#endif

        [TestMethod]
        public void test_isfortran_1()
        {

            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }, order: NPY_ORDER.NPY_CORDER);
            var a1 = np.isfortran(a);
            Assert.AreEqual(false, a1);
            print(a1);

            var b = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }, order: NPY_ORDER.NPY_FORTRANORDER);
            var b1 = np.isfortran(b);
            Assert.AreEqual(true, b1);
            print(b1);

            var c = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }, order: NPY_ORDER.NPY_CORDER);
            var c1 = np.isfortran(c);
            Assert.AreEqual(false, c1);
            print(c1);

            var d = a.T;
            var d1 = np.isfortran(d);
            Assert.AreEqual(true, d1);
            print(d1);

            // C-ordered arrays evaluate as False even if they are also FORTRAN-ordered.

            var e1 = np.isfortran(np.array(new int[] { 1, 2 }, order: NPY_ORDER.NPY_FORTRANORDER));
            Assert.AreEqual(false, e1);
            print(e1);

            return;

        }

        [TestMethod]
        public void test_argwhere_1()
        {
            var x = np.arange(6).reshape((2, 3));
            var y = np.argwhere(x > 1);

            var ExpectedY = new npy_intp[,] {{0, 2}, {1, 0}, {1, 1}, {1, 2}};
            AssertArray(y, ExpectedY);
            print(y);

            var a = np.arange(12).reshape((2, 3, 2));
            var b = np.argwhere(a > 1);

            var ExpectedB = new npy_intp[,]
                {{0, 1, 0}, {0, 1, 1}, {0, 2, 0}, {0, 2, 1}, {1, 0, 0},
                 {1, 0, 1}, {1, 1, 0}, {1, 1, 1}, {1, 2, 0}, {1, 2, 1}};

            AssertArray(b, ExpectedB);

            print(b);

            return;
        }

        [TestMethod]
        public void test_flatnonzero_1()
        {
            var x = np.arange(-2, 3);

            var y = np.flatnonzero(x);
            AssertArray(y, new npy_intp[] {0,1,3,4});
            print(y);

            // Use the indices of the non-zero elements as an index array to extract these elements:

            var z = x.ravel()[np.flatnonzero(x)] as ndarray;
            AssertArray(z, new Int32[] { -2,-1,1,2 });
            print(z);

            return;
        }

        [TestMethod]  
        public void test_outer_1()
        {
            var a = np.arange(2, 10).reshape((2, 4));
            var b = np.arange(12, 20).reshape((2, 4));
            var c = np.outer(a, b);

            var ExpectedDataC = new int[,]
                {{24,  26,  28,  30,  32,  34,  36,  38},
                 {36,  39,  42,  45,  48,  51,  54,  57},
                 {48,  52,  56,  60,  64,  68,  72,  76},
                 {60,  65,  70,  75,  80,  85,  90,  95},
                 {72,  78,  84,  90,  96, 102, 108, 114},
                 {84,  91,  98, 105, 112, 119, 126, 133},
                 {96, 104, 112, 120, 128, 136, 144, 152},
                 {108, 117, 126, 135, 144, 153, 162, 171}};

            AssertArray(c, ExpectedDataC);

            print(c);

            return;
        }

        [TestMethod]
        public void test_inner_1()
        {
            var a = np.arange(1, 5, dtype:np.Int16).reshape((2, 2));
            var b = np.arange(11, 15, dtype: np.Int32).reshape((2, 2));
            var c = np.inner(a, b);
            AssertArray(c, new Int32[,] { { 35, 41 }, { 81, 95 } });
            print(c);


            a = np.arange(2, 10).reshape((2, 4));
            b = np.arange(12, 20).reshape((2, 4));
            c = np.inner(a, b);
            print(c);
            AssertArray(c, new Int32[,] { { 194, 250 }, { 410, 530 } });
            print(c.shape);

            return;
        }

        [TestMethod]
        public void test_inner_2()
        {
            var a = np.array(new bool[] { true, false, false, true}).reshape((2, 2));
            var b = np.array(new bool[] { true, false, true, true }).reshape((2, 2));
            var c = np.inner(a, b);
            AssertArray(c, new bool[,] { { true, true }, { false, true } });


            b = np.arange(11, 15, dtype: np.Int16).reshape((2, 2));
            c = np.inner(a, b);
            AssertArray(c, new Int16[,] { { 11, 13 }, { 12, 14 } });
            print(c);
            c = np.inner(b, a);
            AssertArray(c, new Int16[,] { { 11, 12 }, { 13, 14 } });
            print(c);

            a = np.arange(0, 80, dtype: np.Int32).reshape((-1, 4, 5, 2));
            b = np.arange(100, 180, dtype: np.Float32).reshape((-1, 4, 5, 2));
            c = np.inner(a, b);
            //print(c);
            Assert.AreEqual(c.Dtype.TypeNum, NPY_TYPES.NPY_DOUBLE); // note: this is a different type than python produces
            AssertShape(c.shape, 2, 4, 5, 2, 4, 5);
            print(c.shape);

            Assert.AreEqual((double)17633600, c.Sum().GetItem(0));

            print(c.Sum(axis: 1));

            return;
        }

        [TestMethod]
        public void test_tensordot_1()
        {
            var a = np.arange(60.0, dtype: np.Float64).reshape((3, 4, 5));
            var b = np.arange(24.0, dtype: np.Float64).reshape((4, 3, 2));
            var c = np.tensordot(a, b, axes: (new npy_intp[] { 1, 0 },new npy_intp[] { 0, 1 }));
            AssertShape(c, 5, 2);
            print(c.shape);
            AssertArray(c, new double[,] { { 4400.0, 4730.0 }, { 4532.0, 4874.0 }, { 4664.0, 5018.0 }, { 4796.0, 5162.0 }, { 4928.0, 5306.0 } });
            print(c);
        }

        [TestMethod]
        public void test_tensordot_2()
        {
            var a = np.arange(12.0, dtype: np.Float64).reshape((3, 4));
            var b = np.arange(24.0, dtype: np.Float64).reshape((4, 3, 2));
            var c = np.tensordot(a, b, axis: 1);
            AssertShape(c, 3, 3, 2);
            print(c.shape);
            AssertArray(c, new double[,,] {{{84,90},{96,102},{108,114}},{{228,250},{272,294},{316,338}},{{372,410},{448,486},{524,562}}});


            c = np.tensordot(a, b, axis: 0);
            AssertShape(c, 3, 4,4,3, 2);
            print(c.shape);

            print(c);
        }

        [TestMethod]
        public void test_dot_1()
        {
            var a = new int[,] { { 1, 0 }, { 0, 1 } };
            var b = new int[,] { { 4, 1 }, { 2, 2 } };
            var c = np.dot(a, b);
            AssertArray(c, new int[,] { {4,1}, {2,2} });
            print(c);

            var d = np.dot(3, 4);
            Assert.AreEqual(12, d.GetItem(0));
            print(d);

            var e = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6));
            var f = np.arange(3 * 4 * 5 * 6).A("::-1").reshape((5, 4, 6, 3));
            var g = np.dot(e, f);
            AssertShape(g.shape, 3, 4, 5, 5, 4, 3);
            Assert.AreEqual(695768400, g.Sum().GetItem(0));

            // TODO: NOTE: this crazy indexing is not currently working
            //g = g.A(2, 3, 2, 1, 2, 2);
            //Assert.AreEqual(499128, g.GetItem(0));
            //print(g);

        }


        [TestMethod]
        public void test_roll_forward()
        {
            var a = np.arange(10, dtype: np.UInt16);

            print("A");
            print(a);
            print(a.shape);
            print(a.strides);

            var b = np.roll(a, 2);
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);
            AssertArray(b, new UInt16[] { 8, 9, 0, 1, 2, 3, 4, 5, 6, 7 });
            AssertShape(b, 10);

            var c = np.roll(b, 2);
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);
            AssertArray(c, new UInt16[] { 6, 7, 8, 9, 0, 1, 2, 3, 4, 5 });
            AssertShape(c, 10);

        }

        [TestMethod]
        public void test_roll_backward()
        {
            var a = np.arange(10, dtype: np.UInt16);

            print("A");
            print(a);
            print(a.shape);
            print(a.strides);

            var b = np.roll(a, -2);
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);
            AssertArray(b, new UInt16[] { 2, 3, 4, 5, 6, 7, 8, 9, 0, 1 });
            AssertShape(b, 10);

            var c = np.roll(b, -6);
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);
            AssertArray(c, new UInt16[] { 8, 9, 0, 1, 2, 3, 4, 5, 6, 7 });
            AssertShape(c, 10);
        }

        [TestMethod]
        public void test_roll_backward_1()
        {
            var a = np.arange(16, dtype: np.UInt16).reshape((4, 4));

            print("A");
            print(a);

            var b = np.roll(a, -2, axis: 0);
            print("B");
            print(b);
            AssertArray(b, new UInt16[,] { { 8, 9, 10, 11 }, { 12, 13, 14, 15 }, { 0, 1, 2, 3 }, { 4, 5, 6, 7 } });

            var c = np.roll(b, -2, axis: 1);
            print("C");
            print(c);
            AssertArray(c, new UInt16[,] { { 10, 11, 8, 9 }, { 14, 15, 12, 13 }, { 2, 3, 0, 1 }, { 6, 7, 4, 5 } });
        }

        [TestMethod]
        public void test_roll_with_axis()
        {
            var x = np.arange(10);
            var A = np.roll(x, 2);
            AssertArray(A, new int[] { 8, 9, 0, 1, 2, 3, 4, 5, 6, 7 });
            print("A");
            print(A);

            var x2 = np.reshape(x, new shape(2, 5));
            var B = np.roll(x2, 1);
            AssertArray(B, new int[,] { { 9, 0, 1, 2, 3 }, { 4, 5, 6, 7, 8 } });
            print(B);

            var C = np.roll(x2, 1, axis: 0);
            AssertArray(C, new int[,] { { 5, 6, 7, 8, 9 },  { 0, 1, 2, 3, 4 } });
            print(C);

            var D = np.roll(x2, 1, axis: 1);
            AssertArray(D, new int[,] { { 4, 0, 1, 2, 3 }, { 9, 5, 6, 7, 8 } });
            print(D);

        }

        [TestMethod]
        public void test_roll_with_axis_2()
        {
            var x = np.arange(12);
            var A = np.roll(x, 2);
            AssertArray(A, new int[] { 10,11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            print("A");
            print(A);

            var x2 = np.reshape(x, new shape(2,2,3));
            var B = np.roll(x2, 1);
            AssertArray(B, new int[,,] { { { 11, 0, 1 }, { 2, 3, 4 } },{ { 5, 6, 7 }, { 8, 9, 10 } } });
            print(B);

            var C = np.roll(x2, 1, axis: 0);
            AssertArray(C, new int[,,] { { { 6, 7, 8 }, { 9, 10, 11 } },{ { 0, 1, 2 }, { 3, 4, 5 } } });
            print(C);

            var D = np.roll(x2, 1, axis: 1);
            AssertArray(D, new int[,,] { { { 3, 4, 5 }, { 0, 1, 2 } }, { { 9, 10, 11 }, { 6, 7, 8 } } });
            print(D);

            var E = np.roll(x2, 1, axis: 2);
            AssertArray(E, new int[,,] { { { 2, 0, 1 }, { 5, 3, 4 } }, { { 8, 6, 7 }, { 11, 9, 10 } } });
            print(E);

        }

        [TestMethod]
        public void test_roll_with_axis_3()
        {
            var x = np.arange(16);
            var A = np.roll(x, 2);
            AssertArray(A, new int[] { 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 });
            print("A");
            print(A);

            var x2 = np.reshape(x, new shape(2, 2, 2, 2));
            var B = np.roll(x2, 1);
            AssertArray(B, new int[,,,] { { { { 15, 0 },{ 1, 2 }},{ { 3, 4 },{ 5, 6 } } },{ { { 7, 8 },{ 9, 10 } },{ { 11, 12 }, { 13, 14 } } } });
            print(B);

            var C = np.roll(x2, 1, axis: 0);
            AssertArray(C, new int[,,,] { { { { 8, 9 },{ 10, 11 } },{ { 12, 13 },{ 14, 15 } } },{ { { 0, 1 },{ 2, 3 } },{ { 4, 5 },{ 6, 7 } } } });
            print(C);

            var D = np.roll(x2, 1, axis: 1);
            AssertArray(D, new int[,,,] { { { { 4, 5 },{ 6, 7 } },{ { 0, 1 },{ 2, 3 } } },{ { { 12, 13 },{ 14, 15 } },{ { 8, 9 },{ 10, 11 } } } });
            print(D);

            var E = np.roll(x2, 1, axis: 2);
            AssertArray(E, new int[,,,] { { { { 2, 3 }, { 0, 1 } }, { { 6, 7 },{ 4, 5 } } },{ { { 10, 11 },{ 8, 9 } },{ { 14, 15 }, { 12, 13 } } } });
            print(E);

            var F = np.roll(x2, 1, axis: 3);
            AssertArray(F, new int[,,,] { { { { 1, 0 }, { 3, 2 } }, { { 5, 4 }, { 7, 6 } } }, { { { 9, 8 }, { 11, 10 } }, { { 13, 12 }, { 15, 14 } } } });
            print(F);


        }

        [TestMethod]
        public void test_ndarray_rollaxis()
        {
            var a = np.ones((3, 4, 5, 6));
            var b = np.rollaxis(a, 3, 1).shape;
            AssertShape(b, 3, 6, 4, 5);
            print(b);

            var c = np.rollaxis(a, 2).shape;
            AssertShape(c, 5, 3, 4, 6);
            print(c);

            var d = np.rollaxis(a, 1, 4).shape;
            AssertShape(d, 3, 5, 6, 4);
            print(d);
        }

        [TestMethod]
        public void test_ndarray_moveaxis()
        {
            var x = np.zeros((3, 4, 5));
            var b = np.moveaxis(x, 0, -1).shape;
            AssertShape(b, 4, 5, 3);
            print(b);

            var c = np.moveaxis(x, -1, 0).shape;
            AssertShape(c, 5, 3, 4);
            print(c);

            // These all achieve the same result:
            var d = np.transpose(x).shape;
            AssertShape(d, 5, 4, 3);
            print(d);

            var e = np.swapaxes(x, 0, -1).shape;
            AssertShape(e, 5, 4, 3);
            print(e);

            var f = np.moveaxis(x, new int[] { 0, 1 }, new int[] { -1, -2 }).shape;
            AssertShape(f, 5, 4, 3);
            print(f);

            var g = np.moveaxis(x, new int[] { 0, 1, 2 }, new int[] { -1, -2, -3 }).shape;
            AssertShape(g, 5, 4, 3);
            print(g);
        }

        [TestMethod]
        public void test_indices_1()
        {
            var grid = np.indices((2, 3));
            AssertShape(grid, 2, 2, 3);
            print(grid.shape);
            AssertArray(grid[0] as ndarray, new Int32[,] { {0,0,0}, {1,1,1} });
            print(grid[0]);
            AssertArray(grid[1] as ndarray, new Int32[,] { { 0, 1, 2 }, { 0, 1, 2 } });
            print(grid[1]);

            var x = np.arange(20).reshape((5, 4));
            var y = x[grid[0], grid[1]];
            AssertArray(y as ndarray, new Int32[,] { { 0, 1, 2 }, { 4, 5, 6 } });
            print(y);

            return;
        }
 
#if NOT_PLANNING_TODO
        [Ignore] // not implemented yet
        [TestMethod]
        public void xxx_test_fromfunction_1()
        {

        }
#endif

        [TestMethod]
        public void test_isscalar_1()
        {

            bool a = np.isscalar(3.1);
            Assert.AreEqual(true, a);
            print(a);

            bool b = np.isscalar(np.array(3.1));
            Assert.AreEqual(false, b);
            print(b);

            bool c = np.isscalar(new double[] { 3.1 });
            Assert.AreEqual(false, c);
            print(c);

            bool d = np.isscalar(false);
            Assert.AreEqual(true, d);
            print(d);

            bool e = np.isscalar("numpy");
            Assert.AreEqual(false, e);
            print(e);

            return;
        }

#if NOT_PLANNING_TODO
        [Ignore] // not implemented yet
        [TestMethod]
        public void xxx_test_binary_repr()
        {

        }

        [Ignore] // not implemented yet
        [TestMethod]
        public void xxx_test_base_repr()
        {

        }
#endif

        [TestMethod]
        public void test_identity_1()
        {
            ndarray a = np.identity(2, dtype: np.Float64);

            print(a);
            print(a.shape);
            print(a.strides);

            var ExpectedDataA = new double[2, 2]
            {
                { 1,0 },
                { 0,1 },
            };
            AssertArray(a, ExpectedDataA);
            AssertShape(a, 2, 2);
            AssertStrides(a, sizeof(double) * 2, sizeof(double) * 1);

            ndarray b = np.identity(5, dtype: np.Int8);

            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new sbyte[5, 5]
            {
                { 1, 0, 0, 0, 0 },
                { 0, 1, 0, 0, 0 },
                { 0, 0, 1, 0, 0 },
                { 0, 0, 0, 1, 0 },
                { 0, 0, 0, 0, 1 },
            };
            AssertArray(b, ExpectedDataB);
            AssertShape(b, 5, 5);
            AssertStrides(b, sizeof(byte) * 5, sizeof(byte) * 1);
        }

        [TestMethod]
        public void test_allclose_1()
        {
            bool a = np.allclose(new double[] { 1e10, 1e-7 }, new double[] { 1.00001e10, 1e-8 });
            Assert.AreEqual(false, a);
            print(a);

            bool b = np.allclose(new double[] { 1e10, 1e-8 }, new double[] { 1.00001e10, 1e-9 });
            Assert.AreEqual(true, b);
            print(b);

            bool c = np.allclose(new double[] { 1e10, 1e-8 }, new double[] { 1.0001e10, 1e-9 });
            Assert.AreEqual(false, c);
            print(c);

            bool d = np.allclose(new double[] { 1.0, np.NaN }, new double[] { 1.0, np.NaN });
            Assert.AreEqual(false, d);
            print(d);

            bool e = np.allclose(new double[] { 1.0, np.NaN }, new double[] { 1.0, np.NaN }, equal_nan: true);
            Assert.AreEqual(true, e);
            print(e);

            return;
        }

        [TestMethod]
        public void test_isclose_1()
        {
            var a = np.isclose(new double[] { 1e10, 1e-7 }, new double[] { 1.00001e10, 1e-8 });
            AssertArray(a, new bool[] { true, false });
            print(a);

            var b = np.isclose(new double[] { 1e10, 1e-8 }, new double[] { 1.00001e10, 1e-9 });
            AssertArray(b, new bool[] { true, true });
            print(b);

            var c = np.isclose(new double[] { 1e10, 1e-8 }, new double[] { 1.0001e10, 1e-9 });
            AssertArray(c, new bool[] { false, true });
            print(c);

            var d = np.isclose(new double[] { 1.0, np.NaN }, new double[] { 1.0, np.NaN });
            AssertArray(d, new bool[] { true, false });
            print(d);

            var e = np.isclose(new double[] { 1.0, np.NaN }, new double[] { 1.0, np.NaN }, equal_nan: true);
            AssertArray(e, new bool[] { true, true });
            print(e);

            var f = np.isclose(new double[] { 1e-8, 1e-7 }, new double[] { 0.0, 0.0 });
            AssertArray(f, new bool[] { true, false });
            print(f);

            var g = np.isclose(new double[] { 1e-100, 1e-7 }, new double[] { 0.0, 0.0 }, atol:0.0);
            AssertArray(g, new bool[] { false, false });
            print(g);

            var h = np.isclose(new double[] { 1e-10, 1e-10 }, new double[] { 1e-20, 0.0 });
            AssertArray(h, new bool[] { true, true });
            print(h);

            var i = np.isclose(new double[] { 1e-10, 1e-10 }, new double[] { 1e-20, 0.999999e-10 }, atol: 0.0);
            AssertArray(i, new bool[] { false, true });
            print(i);
        }

        [TestMethod]
        public void test_array_equal_1()
        {
            var a = np.array_equal(new int[] { 1, 2 }, new int[] { 1, 2 });
            Assert.AreEqual(true, a);
            print(a);

            var b = np.array_equal(np.array(new int[] { 1, 2 }), np.array(new int[] { 1, 2 }));
            Assert.AreEqual(true, b);
            print(b);

            var c = np.array_equal(new int[] { 1, 2 }, new int[] { 1, 2, 3 });
            Assert.AreEqual(false, c);
            print(c);

            var d = np.array_equal(new int[] { 1, 2 }, new int[] { 1, 4 });
            Assert.AreEqual(false, d);
            print(d);
        }

        [TestMethod]
        public void test_array_equiv_1()
        {
            var a = np.array_equiv(new int[] { 1, 2 }, new int[] { 1, 2 });
            Assert.AreEqual(true, a);
            print(a);

            var b = np.array_equiv(new int[] { 1, 2 }, new int[] { 1, 3 });
            Assert.AreEqual(false, b);
            print(b);

            var c = np.array_equiv(new int[] { 1, 2 }, new int[,] { { 1, 2 }, { 1, 2 } });
            Assert.AreEqual(true, c);
            print(c);

            var d = np.array_equiv(new int[] { 1, 2 }, new int[,] { { 1, 2, 1, 2 }, { 1, 2, 1, 2 } });
            Assert.AreEqual(false, d);
            print(d);

            var e = np.array_equiv(new int[] { 1, 2 }, new int[,] { { 1, 2 }, { 1, 3 } });
            Assert.AreEqual(false, e);
            print(e);
        }

        [TestMethod]
        public void test_array_dtype_checks()
        {
            var a = np.array(new int[] { 1, 2 });
            Assert.AreEqual(NPY_TYPES.NPY_INT32, a.Dtype.TypeNum);
            Assert.AreEqual(true,a.IsInteger);
            Assert.AreEqual(true, a.IsNumber);
            Assert.AreEqual(true, a.IsSignedInteger);
            Assert.AreEqual(false, a.IsUnsignedInteger);
            Assert.AreEqual(false, a.IsComplex);
            Assert.AreEqual(false, a.IsFloatingPoint);
            Assert.AreEqual(false, a.IsInexact);
            Assert.AreEqual(false, a.IsDecimal);

            a = np.array(new double[] { 1, 2 });
            Assert.AreEqual(NPY_TYPES.NPY_DOUBLE, a.Dtype.TypeNum);
            Assert.AreEqual(false, a.IsInteger);
            Assert.AreEqual(true, a.IsNumber);
            Assert.AreEqual(false, a.IsSignedInteger);
            Assert.AreEqual(false, a.IsUnsignedInteger);
            Assert.AreEqual(false, a.IsComplex);
            Assert.AreEqual(true, a.IsFloatingPoint);
            Assert.AreEqual(true, a.IsInexact);
            Assert.AreEqual(false, a.IsDecimal);

            a = np.array(new Complex[] { 1, 2 });
            Assert.AreEqual(NPY_TYPES.NPY_COMPLEX, a.Dtype.TypeNum);
            Assert.AreEqual(false, a.IsInteger);
            Assert.AreEqual(true, a.IsNumber);
            Assert.AreEqual(false, a.IsSignedInteger);
            Assert.AreEqual(false, a.IsUnsignedInteger);
            Assert.AreEqual(true, a.IsComplex);
            Assert.AreEqual(true, a.IsFloatingPoint);
            Assert.AreEqual(true, a.IsInexact);
            Assert.AreEqual(false, a.IsDecimal);

            a = np.array(new Decimal[] { 1, 2 });
            Assert.AreEqual(NPY_TYPES.NPY_DECIMAL, a.Dtype.TypeNum);
            Assert.AreEqual(false, a.IsInteger);
            Assert.AreEqual(true, a.IsNumber);
            Assert.AreEqual(false, a.IsSignedInteger);
            Assert.AreEqual(false, a.IsUnsignedInteger);
            Assert.AreEqual(false, a.IsComplex);
            Assert.AreEqual(true, a.IsFloatingPoint);
            Assert.AreEqual(true, a.IsInexact);
            Assert.AreEqual(true, a.IsDecimal);

            a = np.array(new BigInteger[] { -1, 2 });
            Assert.AreEqual(NPY_TYPES.NPY_BIGINT, a.Dtype.TypeNum);
            Assert.AreEqual(true, a.IsInteger);
            Assert.AreEqual(true, a.IsNumber);
            Assert.AreEqual(true, a.IsSignedInteger);
            Assert.AreEqual(false, a.IsUnsignedInteger);
            Assert.AreEqual(false, a.IsComplex);
            Assert.AreEqual(false, a.IsFloatingPoint);
            Assert.AreEqual(false, a.IsInexact);
            Assert.AreEqual(false, a.IsDecimal);


            a = np.array(new bool[] { true, false });
            Assert.AreEqual(NPY_TYPES.NPY_BOOL, a.Dtype.TypeNum);
            Assert.AreEqual(true, a.IsBool);
            Assert.AreEqual(false, a.IsInteger);
            Assert.AreEqual(false, a.IsNumber);
        }


#if NOT_PLANNING_TODO

        [Ignore]
        [TestMethod]
        public void test_result_type_placeholder()
        {

        }

        [Ignore]
        [TestMethod]
        public void test_promote_type_placeholder()
        {

        }

        [Ignore]
        [TestMethod]
        public void test_min_scalar_type_placeholder()
        {

        }


        [Ignore]
        [TestMethod]
        public void test_can_cast_placeholder()
        {

        }

#endif
    }
}
