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
    public class ComplexNumbersTests : TestBaseClass
    {
        private static int SizeofComplex = sizeof(double) * 2;

        #region from ArrayCreationTests
        [TestMethod]
        public void test_asfarray_COMPLEX()
        {
            int CaughtExceptions = 0;

            try
            {
                var a = np.asfarray(new Complex[] { 2, 3 });
                AssertArray(a, new double[] { 2, 3 });
                print(a);

            }
            catch
            {
                CaughtExceptions++;
            }

            try
            {
                var b = np.asfarray(new Complex[] { 2, 3 }, dtype: np.Float32);
                AssertArray(b, new float[] { 2, 3 });
                print(b);
            }
            catch
            {
                CaughtExceptions++;
            }

            try
            {
                var c = np.asfarray(new Complex[] { 2, 3 }, dtype: np.Int8);
                AssertArray(c, new double[] { 2, 3 });
                print(c);
            }
            catch
            {
                CaughtExceptions++;
            }

            Assert.AreEqual(0, CaughtExceptions);

            return;
        }

        [TestMethod]
        public void test_copy_1_COMPLEX()
        {
            var x = np.array(new Complex[] { 1, 2, 3 });
            var y = x;

            var z = np.copy(x);

            // Note that, when we modify x, y changes, but not z:

            x[0] = 10;

            Assert.AreEqual(new Complex(10, 0), y[0]);

            Assert.AreEqual(new Complex(1, 0), z[0]);

            return;
        }

        [TestMethod]
        public void test_linspace_1_COMPLEX()
        {
            Complex retstep = 0;

            var a = np.linspace(new Complex(2.0, 1.3), new Complex(3.0, 5.6), ref retstep, num: 5);
            AssertArray(a, new Complex[] { new Complex(2.0, 1.3), new Complex(2.25, 2.375), new Complex(2.5, 3.45), new Complex(2.75, 4.525) , new Complex(3.0, 5.6) });
            print(a);

            var b = np.linspace(new Complex(2.0, 0), new Complex(3.0, 0), ref retstep, num: 5, endpoint: false);
            AssertArray(b, new Complex[] { 2.0, 2.2, 2.4, 2.6, 2.8 });
            print(b);

            var c = np.linspace(new Complex(2.0, 2.0), new Complex(3.0, 3.0), ref retstep, num: 5);
            AssertArray(c, new Complex[] { new Complex(2.0, 2.0), new Complex(2.25, 2.25), new Complex(2.5, 2.5), new Complex(2.75, 2.75), new Complex(3.0, 3.0) });
            print(c);
        }

        [TestMethod]
        public void test_logspace_1_COMPLEX()
        {
            var a = np.logspace(new Complex(2.0, 0), new Complex(3.0, 0), num: 4);
            AssertArray(a, new Complex[] { new Complex(100, 0), new Complex(215.443469003188, 0), new Complex(464.158883361278, 0), new Complex(1000, 0) });
            print(a);

            var b = np.logspace(new Complex(2.0, 0), new Complex(3.0, 0), num: 4, endpoint: false);
            AssertArray(b, new Complex[] { 100, 177.827941003892, 316.227766016838, 562.341325190349 });
            print(b);

            var c = np.logspace(new Complex(2.0, 0), new Complex(3.0, 0), num: 4, _base: 2.0);
            AssertArray(c, new Complex[] { 4, 05.03968419957949, 06.3496042078728, 8 });
            print(c);
        }
 
        [TestMethod]
        public void test_geomspace_1_COMPLEX()
        {
            var a = np.geomspace(new Complex(1, 0), new Complex(1000, 0), num: 4);
            AssertArray(a, new Complex[] { new Complex(1, 0), new Complex(9.9999999999999, 0), new Complex(99.999999999998, 0), new Complex(999.99999999997, 0) });
            print(a);

            var b = np.geomspace(new Complex(1, 0), new Complex(1000, 0), num: 3, endpoint: false);
            AssertArray(b, new Complex[] { new Complex(1, 0), new Complex(9.9999999999999, 0), new Complex(99.999999999998, 0) });
            print(b);

            var c = np.geomspace(new Complex(1, 0), new Complex(1000, 0), num: 4, endpoint: false);
            AssertArray(c, new Complex[] { new Complex(1, 0), new Complex(5.62341325190345, 0), new Complex(31.6227766016833, 0), new Complex(177.827941003888, 0) });
            print(c);

            var d = np.geomspace(new Complex(1, 0), new Complex(256, 0), num: 9);
            AssertArray(d, new Complex[] { new Complex(1, 0), new Complex(1.99999999999999, 0), new Complex(3.99999999999998, 0),
                           new Complex(7.99999999999993, 0),  new Complex(15.9999999999998, 0), new Complex(31.9999999999995, 0),
                           new Complex(63.9999999999989, 0),  new Complex(127.999999999997, 0), new Complex(255.999999999994, 0) });
            print(d);
        }

        [TestMethod]
        public void test_meshgrid_1_COMPLEX()
        {
            int nx = 3;
            int ny = 2;

            Complex ret = 0;

            var x = np.linspace(new Complex(0,0), new Complex(1, 0), ref ret, nx);
            var y = np.linspace(new Complex(0, 0), new Complex(1, 0), ref ret, ny);

            ndarray[] xv = np.meshgrid(new ndarray[] { x });
            AssertArray(xv[0], new Complex[] { 0.0, 0.5, 1.0 });
            print(xv[0]);

            print("************");

            ndarray[] xyv = np.meshgrid(new ndarray[] { x, y });
            AssertArray(xyv[0], new Complex[,] { { 0.0, 0.5, 1.0 }, { 0.0, 0.5, 1.0 } });
            AssertArray(xyv[1], new Complex[,] { { 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 } });

            print(xyv[0]);
            print(xyv[1]);

            print("************");

            xyv = np.meshgrid(new ndarray[] { x, y }, sparse: true);
            AssertArray(xyv[0], new Complex[,] { { 0.0, 0.5, 1.0 } });
            AssertArray(xyv[1], new Complex[,] { { 0.0 }, { 1.0 } });

            print(xyv[0]);
            print(xyv[1]);

            print("************");
        }

        [TestMethod]
        public void test_OneDimensionalArray_COMPLEX()
        {
            Complex[] l = new Complex[] { 12.23, 13.32, 100, 36.32 };
            print("Original List:", l);
            var a = np.array(l);
            print("One-dimensional numpy array: ", a);
            print(a.shape);
            print(a.strides);

            AssertArray(a, l);
            AssertShape(a, 4);
            AssertStrides(a, SizeofComplex);
        }

        [TestMethod]
        public void test_reverse_array_COMPLEX()
        {
            var x = np.arange(0, 40, dtype: np.Complex);
            print("Original array:");
            print(x);
            print("Reverse array:");
            x = (ndarray)x[new Slice(null, null, -1)];
            //x = (ndarray)x.A("::-1");
            print(x);

            AssertArray(x, new Complex[] { 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 });
            AssertShape(x, 40);
            AssertStrides(x, -SizeofComplex);

            var y = x + 100;
            print(y);

            var z = x.reshape((5, -1));
            print(z);
        }

        [TestMethod]
        public void test_checkerboard_1_COMPLEX()
        {
            var x = np.ones((3, 3), dtype: np.Complex);
            print("Checkerboard pattern:");
            x = np.zeros((8, 8), dtype: np.Complex);
            x["1::2", "::2"] = 1;
            x["::2", "1::2"] = 1;
            print(x);

            var ExpectedData = new Complex[8, 8]
            {
                 { 0, 1, 0, 1, 0, 1, 0, 1 },
                 { 1, 0, 1, 0, 1, 0, 1, 0 },
                 { 0, 1, 0, 1, 0, 1, 0, 1 },
                 { 1, 0, 1, 0, 1, 0, 1, 0 },
                 { 0, 1, 0, 1, 0, 1, 0, 1, },
                 { 1, 0, 1, 0, 1, 0, 1, 0, },
                 { 0, 1, 0, 1, 0, 1, 0, 1, },
                 { 1, 0, 1, 0, 1, 0, 1, 0, },
            };

            AssertArray(x, ExpectedData);
            AssertShape(x, 8, 8);
            AssertStrides(x, SizeofComplex * 8, SizeofComplex);

        }

        [TestMethod]
        public void test_F2C_1_COMPLEX()
        {
            Complex[] fvalues = new Complex[] { new Complex(0,-1), new Complex(12,-1.1), new Complex(45.21, 45.3456), new Complex(34, 87), new Complex(99.91, 789) };
            ndarray F = (ndarray)np.array(fvalues);
            print("Values in Fahrenheit degrees:");
            print(F);
            print("Values in  Centigrade degrees:");

            ndarray C = (Complex)5 * F / 9 - 5 * 32 / 9;
            print(C);

            AssertArray(C, new Complex[] { new Complex(-17, -0.555555555555556), new Complex(-10.3333333333333, -0.611111111111111),
                            new Complex(8.11666666666667, 25.192), new Complex(1.88888888888889, 48.3333333333333), new Complex(38.5055555555556, 438.333333333333) });
        }

        [TestMethod]
        public void test_ArrayStats_1_COMPLEX()
        {
            var x = np.array(new Complex[] { 1, 2, 3 }, dtype: np.Complex);
            print("Size of the array: ", x.size);
            print("Length of one array element in bytes: ", x.ItemSize);
            print("Total bytes consumed by the elements of the array: ", x.nbytes);

            Assert.AreEqual(3, x.size);
            Assert.AreEqual(SizeofComplex, x.ItemSize);
            Assert.AreEqual(SizeofComplex * 3, x.nbytes);

        }

        [TestMethod]
        public void test_ndarray_flatten_COMPLEX()
        {
            var x = np.arange(new Complex(0.73, 0), new Complex(25.73, 0), dtype: np.Complex).reshape(new shape(5, 5));
            var y = x.flatten();
            print(x);
            print(y);

            AssertArray(y, new Complex[] { 0.73, 1.73, 2.73, 3.73, 4.73, 5.73, 6.73, 7.73, 8.73, 9.73,
                                         10.73, 11.73, 12.73, 13.73, 14.73, 15.73, 16.73, 17.73, 18.73,
                                         19.73, 20.73, 21.73, 22.73, 23.73, 24.73 });

            y = x.flatten(order: NPY_ORDER.NPY_FORTRANORDER);
            print(y);

            AssertArray(y, new Complex[] { 0.73, 5.73, 10.73, 15.73, 20.73,  1.73, 6.73, 11.73, 16.73,
                                         21.73, 2.73,  7.73, 12.73, 17.73, 22.73, 3.73, 8.73, 13.73, 18.73,
                                         23.73, 4.73,  9.73, 14.73, 19.73, 24.73 });

            y = x.flatten(order: NPY_ORDER.NPY_KORDER);
            print(y);

            AssertArray(y, new Complex[] { 0.73, 1.73, 2.73, 3.73, 4.73, 5.73, 6.73, 7.73, 8.73, 9.73,
                                         10.73, 11.73, 12.73, 13.73, 14.73, 15.73, 16.73, 17.73, 18.73,
                                         19.73, 20.73, 21.73, 22.73, 23.73, 24.73 });
        }

        [TestMethod]
        public void test_ndarray_byteswap_COMPLEX()
        {
            var x = np.arange(32, 64, dtype: np.Complex);
            print(x);
            var y = x.byteswap(true);
            print(y);

            // complex can't be swapped.  Data should be unchanged
            AssertArray(y, x.AsComplexArray());

            y = x.byteswap(false);
            print(y);

            // complex can't be swapped.  Data should be unchanged
            AssertArray(y, x.AsComplexArray());

        }

        [TestMethod]
        public void test_ndarray_view_COMPLEX()
        {
            var x = np.arange(256 + 32, 256 + 64, dtype: np.Complex);
            print(x);
            print(x.shape);
            print(x.Dtype);

            AssertArray(x, new Complex[] { 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
                                         299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
                                         310, 311, 312, 313, 314, 315, 316, 317, 318, 319});

            // Complex can't be mapped by something besides another Complex
            var y = x.view(np.UInt64);

            try
            {
                Assert.AreEqual((UInt64)0, (UInt64)y.Sum().GetItem(0));
                Assert.Fail("This should have thrown an exception");
            }
            catch
            {

            }

            y = x.view(np.Complex);
            AssertArray(y, y.AsComplexArray());

            y[5] = 1000;

            AssertArray(x, new Complex[] { 288, 289, 290, 291, 292, 1000, 294, 295, 296, 297, 298,
                                         299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
                                         310, 311, 312, 313, 314, 315, 316, 317, 318, 319});

        }

        [TestMethod]
        public void test_ndarray_delete1_COMPLEX()
        {
            var x = np.arange(0, 32, dtype: np.Complex).reshape(new shape(8, 4));
            print("X");
            print(x);
            print(x.shape);

            var ExpectedDataX = new Complex[8, 4]
            {
                    { 0, 1, 2, 3},
                    { 4, 5, 6, 7},
                    { 8, 9, 10, 11 },
                    { 12, 13, 14, 15},
                    { 16, 17, 18, 19},
                    { 20, 21, 22, 23},
                    { 24, 25, 26, 27},
                    { 28, 29, 30, 31},
            };

            AssertArray(x, ExpectedDataX);
            AssertShape(x, 8, 4);

            var y = np.delete(x, 0, axis: 1);
            y[1] = 99;
            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Complex[8, 3]
            {
                    { 1, 2, 3},
                    { 99, 99, 99},
                    { 9, 10, 11 },
                    { 13, 14, 15},
                    { 17, 18, 19},
                    { 21, 22, 23},
                    { 25, 26, 27},
                    { 29, 30, 31},
            };

            AssertArray(y, ExpectedDataY);
            AssertShape(y, 8, 3);

            print("X");
            print(x);


            AssertArray(x, ExpectedDataX);
            AssertShape(x, 8, 4);
        }

        [TestMethod]
        public void test_ndarray_delete2_COMPLEX()
        {
            var x = np.arange(0, 32, dtype: np.Complex);
            print("X");
            print(x);
            print(x.shape);

            var ExpectedDataX = new Complex[] {0,  1,  2,  3,  4,  5,  6,  7,
                                             8,  9,  10, 11, 12, 13, 14, 15,
                                             16, 17, 18, 19, 20, 21, 22, 23,
                                             24, 25, 26, 27, 28, 29, 30, 31 };
            AssertArray(x, ExpectedDataX);
            AssertShape(x, 32);

            var y = np.delete(x, 1, 0);
            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Complex[] {0,  2,  3,  4,  5,  6,  7,
                                             8,  9,  10, 11, 12, 13, 14, 15,
                                             16, 17, 18, 19, 20, 21, 22, 23,
                                             24, 25, 26, 27, 28, 29, 30, 31 };
            AssertArray(y, ExpectedDataY);
            AssertShape(y, 31);

            print("X");
            print(x);

            AssertArray(x, ExpectedDataX);
        }

        [TestMethod]
        public void test_ndarray_delete3_COMPLEX()
        {
            var x = np.arange(0, 32, dtype: np.Complex).reshape(new shape(8, 4));
            print("X");
            print(x);
            print(x.shape);

            var ExpectedDataX = new Complex[8, 4]
            {
                { 0, 1, 2, 3},
                { 4, 5, 6, 7},
                { 8, 9, 10, 11 },
                { 12, 13, 14, 15},
                { 16, 17, 18, 19},
                { 20, 21, 22, 23},
                { 24, 25, 26, 27},
                { 28, 29, 30, 31},
            };

            AssertArray(x, ExpectedDataX);
            AssertShape(x, 8, 4);

            var mask = np.ones_like(x, dtype: np.Bool);
            mask[new Slice(null), 0] = false;
            print(mask);

            var ExpectedDataMask = new bool[8, 4]
            {
                { false, true, true, true },
                { false, true, true, true },
                { false, true, true, true },
                { false, true, true, true },
                { false, true, true, true },
                { false, true, true, true },
                { false, true, true, true },
                { false, true, true, true },
            };

            AssertArray(mask, ExpectedDataMask);
            AssertShape(mask, 8, 4);

            var y = ((ndarray)(x[mask])).reshape(new shape(8, 3));

            print("Y");
            print(y);

            var ExpectedDataY = new Complex[8, 3]
            {
                { 1, 2, 3},
                { 5, 6, 7},
                { 9, 10, 11 },
                { 13, 14, 15},
                { 17, 18, 19},
                { 21, 22, 23},
                { 25, 26, 27},
                { 29, 30, 31},
            };

            AssertArray(y, ExpectedDataY);
            AssertShape(y, 8, 3);


            print("X");
            print(x);

            AssertArray(x, ExpectedDataX);
            AssertShape(x, 8, 4);
        }

        [TestMethod]
        public void test_ndarray_unique_1_COMPLEX()
        {
            var x = np.array(new Complex[] { 1, 2, 3, 1, 3, 4, 5, 4, 4 });

            print("X");
            print(x);

            var result = np.unique(x, return_counts: true, return_index: true, return_inverse: true);
            var uvalues = result.data;
            var indexes = result.indices;
            var inverse = result.inverse;
            var counts = result.counts;

            print("uvalues");
            print(uvalues);
            AssertArray(uvalues, new Complex[] { 1, 2, 3, 4, 5 });

            print("indexes");
            print(indexes);
            AssertArray(indexes, new npy_intp[] { 0, 1, 2, 5, 6 });

            print("inverse");
            print(inverse);
            AssertArray(inverse, new npy_intp[] { 0, 1, 2, 0, 2, 3, 4, 3, 3 });

            print("counts");
            print(counts);
            AssertArray(counts, new npy_intp[] { 2, 1, 2, 3, 1 });
        }

        [TestMethod]
        public void test_ndarray_where_1_COMPLEX()
        {
            var x = np.array(new Complex[] { 1, 2, 3, 1, 3, 4, 5, 4, 4 }).reshape(new shape(3, 3));

            print("X");
            print(x);

            ndarray[] y = (ndarray[])np.where(x == 3);
            print("Y");
            print(y);


        }

        [TestMethod]
        public void test_ndarray_where_2_COMPLEX()
        {
            var x = np.array(new Complex[] { 1, 2, 3, 1, 3, 4, 5, 4, 4 }).reshape(new shape(3, 3));

            print("X");
            print(x);

            ndarray[] y = (ndarray[])np.where(x == 3);
            print("Y");
            print(y);

            Assert.AreEqual(2, y.Length);
            AssertArray(y[0], new npy_intp[] { 0, 1 });
            AssertArray(y[1], new npy_intp[] { 2, 1 });

            var z = x.SliceMe(y) as ndarray;
            print("Z");
            print(z);
            AssertArray(z, new Complex[] { 3, 3 });
        }

        [TestMethod]
        public void test_ndarray_where_3_COMPLEX()
        {
            var x = np.arange(0, 1000, dtype: np.Complex).reshape(new shape(-1, 10));

            //print("X");
            //print(x);

            ndarray[] y = (ndarray[])np.where(x % 10 == 0);
            print("Y");
            print(y);

            var z = x[y] as ndarray;
            print("Z");
            print(z);

            var ExpectedDataZ = new Complex[]
            {
                0,  10,  20,  30,  40,  50,  60,  70,  80,
                90, 100, 110, 120, 130, 140, 150, 160, 170,
                180, 190, 200, 210, 220, 230, 240, 250, 260,
                270, 280, 290, 300, 310, 320, 330, 340, 350,
                360, 370, 380, 390, 400, 410, 420, 430, 440,
                450, 460, 470, 480, 490, 500, 510, 520, 530,
                540, 550, 560, 570, 580, 590, 600, 610, 620,
                630, 640, 650, 660, 670, 680, 690, 700, 710,
                720, 730, 740, 750, 760, 770, 780, 790, 800,
                810, 820, 830, 840, 850, 860, 870, 880, 890,
                900, 910, 920, 930, 940, 950, 960, 970, 980, 990
            };

            AssertArray(z, ExpectedDataZ);

        }

        [TestMethod]
        public void test_ndarray_where_4_COMPLEX()
        {
            var x = np.arange(0, 3000000, dtype: np.Complex);

            var y = np.where(x % 7 == 0);
            //print("Y");
            //print(y);

            var z = x[y] as ndarray;
            var m = np.mean(z);
            print("M");
            Assert.AreEqual(new Complex(1499998.5, 0), m.GetItem(0));
            print(m);

            return;
        }

        [TestMethod]
        public void test_ndarray_where_5_COMPLEX()
        {
            var a = np.arange(10, dtype: np.Complex);

            var b = np.where(a < 5, a, 10 * a) as ndarray;
            AssertArray(b, new Complex[] { 0, 1, 2, 3, 4, 50, 60, 70, 80, 90 });
            print(b);

            a = np.array(new Complex[,] { { 0, 1, 2 }, { 0, 2, 4 }, { 0, 3, 6 } });
            b = np.where(a < 4, a, -1) as ndarray;  // -1 is broadcast
            AssertArray(b, new Complex[,] { { 0, 1, 2 }, { 0, 2, -1 }, { 0, 3, -1 } });
            print(b);

            var c = np.where(new bool[,] { { true, false }, { true, true } },
                                    new Complex[,] { { 1, 2 }, { 3, 4 } },
                                    new Complex[,] { { 9, 8 }, { 7, 6 } }) as ndarray;

            AssertArray(c, new Complex[,] { { 1, 8 }, { 3, 4 } });

            print(c);

            return;
        }

        [TestMethod]
        public void test_arange_slice_1_COMPLEX()
        {
            var a = np.arange(0, 1024, dtype: np.Complex).reshape(new shape(2, 4, -1));

            print("A");
            // print(a);
            print(a.shape);
            print(a.strides);

            AssertShape(a, 2, 4, 128);
            AssertStrides(a, SizeofComplex * 512, SizeofComplex * 128, SizeofComplex);

            var b = (ndarray)a[":", ":", 122];
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new Complex[2, 4]
            {
                { 122, 250, 378, 506},
                { 634, 762, 890, 1018 },
            };

            AssertArray(b, ExpectedDataB);
            AssertShape(b, 2, 4);
            AssertStrides(b, SizeofComplex * 512, SizeofComplex * 128);

            var c = (ndarray)a.A(":", ":", new npy_intp[] { 122 });
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            var ExpectedDataC = new Complex[2, 4, 1]
            {
                {
                    { 122 },
                    { 250 },
                    { 378 },
                    { 506 },
                },
                {
                    { 634 },
                    { 762 },
                    { 890 },
                    { 1018 },
                },

            };

            AssertArray(c, ExpectedDataC);
            AssertShape(c, 2, 4, 1);
            AssertStrides(c, SizeofComplex*4, SizeofComplex, SizeofComplex*8);

            var d = (ndarray)a.A(":", ":", new npy_intp[] { 122, 123 });
            print("D");
            print(d);
            print(d.shape);
            print(d.strides);

            var ExpectedDataD = new Complex[2, 4, 2]
            {
                {
                    { 122, 123 },
                    { 250, 251 },
                    { 378, 379 },
                    { 506, 507 },
                },
                {
                    { 634, 635 },
                    { 762, 763 },
                    { 890, 891 },
                    { 1018, 1019 },
                },

            };

            AssertArray(d, ExpectedDataD);
            AssertShape(d, 2, 4, 2);
            AssertStrides(d, SizeofComplex*4, SizeofComplex, SizeofComplex*8);

        }

        [TestMethod]
        public void test_arange_slice_2A_COMPLEX()
        {
            var a = np.arange(0, 32, dtype: np.Complex).reshape(new shape(2, 4, -1));

            print("A");
            // print(a);
            print(a.shape);
            print(a.strides);

            var b = (ndarray)a[":", ":", np.where(a > 20)];
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new Complex[,,,]
                {{{{1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
                   {1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3},
                   {1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3}},

                  {{5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5},
                   {5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7},
                   {5,  6,  7,  4,  5,  6,  7,  4,  5,  6,  7}},

                  {{9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9},
                   {9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11},
                   {9, 10, 11,  8,  9, 10, 11,  8,  9, 10, 11}},

                  {{13, 13 ,13, 13, 13, 13, 13, 13, 13, 13, 13},
                   {13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15},
                   {13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15}}},

                 {{{17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17},
                   {17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19},
                   {17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19}},

                  {{21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21},
                   {21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23},
                   {21, 22, 23, 20, 21, 22, 23, 20, 21, 22, 23}},

                  {{25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25},
                   {25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27},
                   {25, 26, 27, 24, 25, 26, 27, 24, 25, 26, 27}},

                  {{29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29},
                   {29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31},
                   {29, 30, 31, 28, 29, 30, 31, 28, 29, 30, 31}}}};

            AssertArray(b, ExpectedDataB);
            AssertStrides(b, SizeofComplex*4, SizeofComplex, SizeofComplex*88, SizeofComplex*8);
        }

        [TestMethod]
        public void test_insert_1_COMPLEX()
        {
            Complex[,] TestData = new Complex[,] { { new Complex(1, .5), 1 }, { 2, 2 }, { 3, 3 } };
            ndarray a = np.array(TestData, dtype: np.Complex);
            ndarray b = np.insert(a, 1, 5);
            ndarray c = np.insert(a, 0, new Complex[] { 999, 100, 101 });

            print(a);
            print(a.shape);

            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new Complex[] { new Complex(1, .5), 5, 1, 2, 2, 3, 3 });
            AssertShape(b, 7);
            AssertStrides(b, SizeofComplex);

            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            AssertArray(c, new Complex[] { 999, 100, 101, new Complex(1, .5), 1, 2, 2, 3, 3 });
            AssertShape(c, 9);
            AssertStrides(c, SizeofComplex);
        }

        [TestMethod]
        public void test_insert_2_COMPLEX()
        {
            Complex[] TestData1 = new Complex[] { 1, 1, 2, 2, 3, 3 };
            Complex[] TestData2 = new Complex[] { 90, 91, 92, 92, 93, 93 };

            ndarray a = np.array(TestData1, dtype: np.Complex);
            ndarray b = np.array(TestData2, dtype: np.Complex);
            ndarray c = np.insert(a, new Slice(null), b);

            print(a);
            print(a.shape);

            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new Complex[] { 90, 91, 92, 92, 93, 93 });
            AssertShape(b, 6);
            AssertStrides(b, SizeofComplex);

            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            AssertArray(c, new Complex[] { 90, 1, 91, 1, 92, 2, 92, 2, 93, 3, 93, 3 });
            AssertShape(c, 12);
            AssertStrides(c, SizeofComplex);

        }

        [TestMethod]
        public void test_append_1_COMPLEX()
        {
            Complex[] TestData = new Complex[] { 1, 1, 2, 2, 3, 3 };
            ndarray a = np.array(TestData, dtype: np.Complex);
            ndarray b = np.append(a, (Complex)1);

            print(a);
            print(a.shape);

            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new Complex[] { 1, 1, 2, 2, 3, 3, 1 });
            AssertShape(b, 7);
            AssertStrides(b, SizeofComplex);
        }

        [TestMethod]
        public void test_append_3_COMPLEX()
        {
            Complex[] TestData1 = new Complex[] { 1, 1, 2, 2, 3, 3 };
            Complex[] TestData2 = new Complex[] { 4, 4, 5, 5, 6, 6 };
            ndarray a = np.array(TestData1, dtype: np.Complex);
            ndarray b = np.array(TestData2, dtype: np.Complex);

            ndarray c = np.append(a, b);

            print(a);
            print(a.shape);

            print(b);
            print(b.shape);

            print(c);
            print(c.shape);
            print(c.strides);

            AssertArray(c, new Complex[] { 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6 });
            AssertShape(c, 12);
            AssertStrides(c, SizeofComplex);
        }

        [TestMethod]
        public void test_append_4_COMPLEX()
        {
            Complex[] TestData1 = new Complex[] { 1, 1, 2, 2, 3, 3 };
            Complex[] TestData2 = new Complex[] { 4, 4, 5, 5, 6, 6 };
            ndarray a = np.array(TestData1, dtype: np.Complex).reshape((2, -1));
            ndarray b = np.array(TestData2, dtype: np.Complex).reshape((2, -1));

            ndarray c = np.append(a, b, axis: 1);

            print(a);
            print(a.shape);
            print("");

            print(b);
            print(b.shape);
            print("");

            print(c);
            print(c.shape);
            print(c.strides);
            print("");

            var ExpectedDataC = new Complex[,]
            {
                { 1, 1, 2, 4, 4, 5 },
                { 2, 3, 3, 5, 6, 6 },
            };

            AssertArray(c, ExpectedDataC);
            AssertShape(c, 2, 6);
            AssertStrides(c, SizeofComplex, SizeofComplex*2); 

        }

        [TestMethod]
        public void test_flat_2_COMPLEX()
        {
            var x = np.arange(1, 7, dtype: np.Complex).reshape((2, 3));
            print(x);

            Assert.AreEqual(new Complex(4,0), x.Flat[3]);
            print(x.Flat[3]);

            print(x.T);
            Assert.AreEqual(new Complex(5, 0), x.T.Flat[3]);
            print(x.T.Flat[3]);

            x.flat = 3;
            AssertArray(x, new Complex[,] { { 3, 3, 3 }, { 3, 3, 3 } });
            print(x);

            x.Flat[new int[] { 1, 4 }] = 1;
            AssertArray(x, new Complex[,] { { 3, 1, 3 }, { 3, 1, 3 } });
            print(x);
        }

        [TestMethod]
        public void test_intersect1d_1_COMPLEX()
        {
            ndarray a = np.array(new Complex[] { 1, 3, 4, 3 });
            ndarray b = np.array(new Complex[] { 3, 1, 2, 1 });

            ndarray c = np.intersect1d(a, b);
            print(c);

            AssertArray(c, new Complex[] { 1, 3 });
            AssertShape(c, 2);
            AssertStrides(c, SizeofComplex);

        }

        [TestMethod]
        public void test_setxor1d_1_COMPLEX()
        {
            ndarray a = np.array(new Complex[] { 1, 2, 3, 2, 4 });
            ndarray b = np.array(new Complex[] { 2, 3, 5, 7, 5 });

            ndarray c = np.setxor1d(a, b);
            print(c);

            AssertArray(c, new Complex[] { 1, 4, 5, 7 });
            AssertShape(c, 4);
            AssertStrides(c, SizeofComplex);
        }

        [TestMethod]
        public void test_in1d_1_COMPLEX()
        {
            ndarray test = np.array(new Complex[] { 0, 1, 2, 5, 0 });
            ndarray states = np.array(new Complex[] { 0, 2 });

            ndarray mask = np.in1d(test, states);
            print(mask);
            print(test[mask]);

            AssertArray(mask, new bool[] { true, false, true, false, true });
            AssertShape(mask, 5);
            AssertStrides(mask, 1);

            ndarray a = test[mask] as ndarray;
            AssertArray(a, new Complex[] { 0, 2, 0 });
            AssertShape(a, 3);
            AssertStrides(a, SizeofComplex);

            mask = np.in1d(test, states, invert: true);
            print(mask);
            print(test[mask]);

            AssertArray(mask, new bool[] { false, true, false, true, false });
            AssertShape(mask, 5);
            AssertStrides(mask, 1);

            ndarray b = test[mask] as ndarray;
            AssertArray(b, new Complex[] { 1, 5 });
            AssertShape(b, 2);
            AssertStrides(b, SizeofComplex);

        }

        [TestMethod]
        public void test_isin_1_COMPLEX()
        {
            ndarray element = 2 * np.arange(4, dtype: np.Complex).reshape(new shape(2, 2));
            print(element);

            ndarray test_elements = np.array(new Complex[] { 1, 2, 4, 8 });
            ndarray mask = np.isin(element, test_elements);
            print(mask);
            print(element[mask]);

            ndarray a = element[mask] as ndarray;

            var ExpectedDataMask = new bool[2, 2]
            {
                { false, true },
                { true, false },
            };

            AssertArray(mask, ExpectedDataMask);
            AssertShape(mask, 2, 2);
            AssertStrides(mask, 2, 1);

            AssertArray(a, new Complex[] { 2, 4 });
            AssertShape(a, 2);
            AssertStrides(a, SizeofComplex);

            print("***********");

            mask = np.isin(element, test_elements, invert: true);
            print(mask);
            print(element[mask]);

            a = element[mask] as ndarray;


            ExpectedDataMask = new bool[2, 2]
            {
                { true, false },
                { false, true },
            };


            AssertArray(mask, ExpectedDataMask);
            AssertShape(mask, 2, 2);
            AssertStrides(mask, 2, 1);

            AssertArray(a, new Complex[] { 0, 6 });
            AssertShape(a, 2);
            AssertStrides(a, SizeofComplex);
        }

        [TestMethod]
        public void test_union1d_1_COMPLEX()
        {
            ndarray a1 = np.array(new Complex[] { -1, 0, 1 });
            ndarray a2 = np.array(new Complex[] { -2, 0, 2 });

            ndarray a = np.union1d(a1, a2);
            print(a);

            AssertArray(a, new Complex[] { -2, -1, 0, 1, 2 });
            AssertShape(a, 5);
            AssertStrides(a, SizeofComplex);
        }

        [TestMethod]
        public void test_Ellipsis_indexing_1_COMPLEX()
        {
            var a = np.array(new Complex[] { 10.0, 7, 4, 3, 2, 1 });

            var b = a.A("...", -1);
            Assert.AreEqual((Complex)1.0, b.GetItem(0));
            print(b);
            print("********");


            a = np.array(new Complex[,] { { 10.0, 7, 4 }, { 3, 2, 1 } });
            var c = a.A("...", -1);
            AssertArray(c, new Complex[] { 4.0, 1.0 });
            print(c);
            print("********");

            var TestData = new Complex[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            a = np.array(TestData, dtype: np.Complex).reshape((1, 3, 2, -1, 1));
            var d = a["...", -1] as ndarray;
            AssertArray(d, new Complex[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } } });
            print(d);
            print("********");

            var e = a[0, "...", -1] as ndarray;
            AssertArray(e, new Complex[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
            print(e);
            print("********");

            var f = a[0, ":", ":", ":", -1] as ndarray;
            AssertArray(f, new Complex[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
            print(f);
            print("********");

            var g = a.A(0, 1, "...", -1);
            AssertArray(g, new Complex[,] { { 5, 6 }, { 7, 8 } });
            print(g);
            print("********");

            var h = a.A(0, 2, 1, "...", -1);
            AssertArray(h, new Complex[] { 11, 12 });
            print(h);
            print("********");

            var i = a[":", 2, 1, 1, "..."] as ndarray;
            AssertArray(i, new Complex[,] { { 12 } });
            print(i);
        }

        [TestMethod]
        public void test_concatenate_1_COMPLEX()
        {

            var a = np.array(new Complex[,] { { 1, 2 }, { 3, 4 } });
            var b = np.array(new Complex[,] { { 5, 6 } });
            var c = np.concatenate((a, b), axis: 0);
            AssertArray(c, new Complex[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            print(c);

            var d = np.concatenate((a, b.T), axis: 1);
            AssertArray(d, new Complex[,] { { 1, 2, 5 }, { 3, 4, 6 } });
            print(d);

            var e = np.concatenate((a, b), axis: null);
            AssertArray(e, new Complex[] { 1, 2, 3, 4, 5, 6 });
            print(e);

            var f = np.concatenate((np.eye(2, dtype: np.Complex), np.ones((2, 2), dtype: np.Complex)), axis: 0);
            AssertArray(f, new Complex[,] { { 1, 0 }, { 0, 1 }, { 1, 1 }, { 1, 1 }, });
            print(f);

            var g = np.concatenate((np.eye(2, dtype: np.Complex), np.ones((2, 2), dtype: np.Complex)), axis: 1);
            AssertArray(g, new Complex[,] { { 1, 0, 1, 1 }, { 0, 1, 1, 1 } });
            print(g);
        }

        [TestMethod]
        public void test_concatenate_3_COMPLEX()
        {

            var a = np.array(new Complex[,,,] { { { { 1, 2 }, { 3, 4 }, { 5, 6 } } } });
            var c = np.concatenate(a, axis: -1);
            AssertArray(c, new Complex[,,] { { { 1, 2 }, { 3, 4 }, { 5, 6 } } });
            print(c);

            var d = np.concatenate(a, axis: -2);
            AssertArray(d, new Complex[,,] { { { 1, 2 }, { 3, 4 }, { 5, 6 } } });
            print(d);

            c = np.concatenate((a, a, a), axis: -1);
            AssertArray(c, new Complex[,,,] { { { { 1, 2, 1, 2, 1, 2 }, { 3, 4, 3, 4, 3, 4 }, { 5, 6, 5, 6, 5, 6 } } } });
            print(c);

            d = np.concatenate((a, a, a), axis: -2);
            AssertArray(d, new Complex[,,,] { { { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 1, 2 }, { 3, 4 }, { 5, 6 }, { 1, 2 }, { 3, 4 }, { 5, 6 } } } });
            print(d);


        }

        [TestMethod]
        public void test_multi_index_selection_COMPLEX()
        {
            var x = np.arange(10).astype(np.Complex);

            var y = x.reshape(new shape(2, 5));
            print(y);
            Assert.AreEqual((Complex)3, y[0, 3]);
            Assert.AreEqual((Complex)8, y[1, 3]);

            x = np.arange(20, dtype: np.Complex);
            y = x.reshape(new shape(2, 2, 5));
            print(y);
            Assert.AreEqual((Complex)3, y[0, 0, 3]);
            Assert.AreEqual((Complex)8, y[0, 1, 3]);

            Assert.AreEqual((Complex)13, y[1, 0, 3]);
            Assert.AreEqual((Complex)18, y[1, 1, 3]);

        }

        [TestMethod]
        public void test_multi_index_setting_COMPLEX()
        {
            var x = np.arange(10, dtype: np.Int32).astype(np.Complex);

            var y = x.reshape(new shape(2, 5));

            y[0, 3] = new Complex(55,0);
            y[1, 3] = new Complex(66,0);

            Assert.AreEqual((Complex)55, (Complex)y[0, 3]);
            Assert.AreEqual((Complex)66, (Complex)y[1, 3]);

            x = np.arange(20, dtype: np.Int32).astype(np.Complex);
            y = x.reshape(new shape(2, 2, 5));

            y[1, 0, 3] = new Complex(55,0);
            y[1, 1, 3] = new Complex(66,0);

            Assert.AreEqual((Complex)55, (Complex)y[1, 0, 3]);
            Assert.AreEqual((Complex)66, (Complex)y[1, 1, 3]);

        }

        #endregion

        #region from NumericalOperationsTests

        [TestMethod]
        public void test_add_operations_COMPLEX()
        {
            var a = np.arange(0, 20, 1, dtype: np.Complex);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            var b = a + 8;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new Complex[,]
            {{8,  9, 10, 11},
             {12, 13, 14, 15},
             {16, 17, 18, 19},
             {20, 21, 22, 23},
             {24, 25, 26, 27}
            };
            AssertArray(b, ExpectedDataB);

            a = np.arange(0, 20, 1, dtype: np.Complex);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            b = a + new Complex(2400, 0);
            print(b);
            print(b.shape);
            print(b.strides);

            ExpectedDataB = new Complex[,]
            {{2400, 2401, 2402, 2403},
             {2404, 2405, 2406, 2407},
             {2408, 2409, 2410, 2411},
             {2412, 2413, 2414, 2415},
             {2416, 2417, 2418, 2419}
            };
            AssertArray(b, ExpectedDataB);

        }

        [TestMethod]
        public void test_add_operations_COMPLEX_2()
        {
            var a = np.arange(0, 20, 1, dtype: np.Complex);
            a = a.reshape(new shape(5, -1));
            print(a);

            var ExpectedDataA = new Complex[,]
                {{0,  1,  2,  3},
                 {4,  5,  6,  7},
                 {8,  9, 10, 11},
                 {12, 13, 14, 15},
                 {16, 17, 18, 19}};
            AssertArray(a, ExpectedDataA);

            var b = np.array(new Complex[] { 2 });
            var c = a + b;
            print(c);

            var ExpectedDataC = new Complex[,]
                {{2,  3,  4,  5},
                 {6,  7,  8,  9},
                 {10, 11, 12, 13},
                 {14, 15, 16, 17},
                 {18, 19, 20, 21}};
            AssertArray(c, ExpectedDataC);


            b = np.array(new Complex[] { 10, 20, 30, 40 });
            var d = a + b;
            print(d);

            var ExpectedDataD = new Complex[,]
                {{10, 21, 32, 43},
                 {14, 25, 36, 47},
                 {18, 29, 40, 51},
                 {22, 33, 44, 55},
                 {26, 37, 48, 59}};
            AssertArray(d, ExpectedDataD);
        }

        [TestMethod]
        public void test_subtract_operations_COMPLEX()
        {
            var a = np.arange(0, 20, 1, dtype: np.Complex);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            var b = a - new Complex(8, 0);
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new Complex[,]
            {{-8, -7, -6, -5},
             {-4, -3, -2, -1},
             {0,  1,  2,  3},
             {4,  5,  6,  7},
             {8,  9, 10, 11}
            };
            AssertArray(b, ExpectedDataB);

            a = np.arange(0, 20, 1, dtype: np.Complex);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            b = a - 2400;
            print(b);
            print(b.shape);
            print(b.strides);

            ExpectedDataB = new Complex[,]
            {{-2400, -2399, -2398, -2397},
             {-2396, -2395, -2394, -2393},
             {-2392, -2391, -2390, -2389},
             {-2388, -2387, -2386, -2385},
             {-2384, -2383, -2382, -2381}
            };

            AssertArray(b, ExpectedDataB);

        }

        [TestMethod]
        public void test_subtract_operations_COMPLEX_2()
        {
            var a = np.arange(100, 102, 1, dtype: np.Complex);
            var b = np.array(new Complex[] { 1, 63 });
            var c = a - b;
            print(a);
            print("****");
            print(b);
            print("****");
            print(c);
            print("****");
            AssertArray(c, new Complex[] { 99, 38 });


            a = np.arange(0, 4, 1, dtype: np.Complex).reshape(new shape(2, 2));
            b = np.array(new Complex[] { 65, 78 }).reshape(new shape(1, 2));
            c = a - b;
            print(a);
            print("****");
            print(b);
            print("****");
            print(c);
            print("****");
            AssertArray(c, new Complex[,] { { -65, -77 }, { -63, -75 } });

        }


        [TestMethod]
        public void test_multiply_1x_COMPLEX()
        {
            Complex[] fvalues = new Complex[] { new Complex(0, -1), new Complex(12, -1.1), new Complex(45.21, 45.3456), new Complex(34, 87), new Complex(99.91, 789) };
            ndarray F = (ndarray)np.array(fvalues);

            Complex[] evalues = new Complex[] { new Complex(5, -1.567), new Complex(-12.56, -2.1), new Complex(145.21, 415.3456), new Complex(-34, 87), new Complex(99.91, 7189) };
            ndarray E = (ndarray)np.array(evalues);

            ndarray G = F * E;

            print(G);
            AssertArray(G, new Complex[] { new Complex(-1.567, -5), new Complex(-153.03, -11.384), new Complex(-12269.15133936, 25362.409152),
                                        new Complex(-8725, 0), new Complex(-5662138.9919, 797081.98) });

        }

        [TestMethod]
        public void test_multiply_operations_COMPLEX()
        {
            var a = np.arange(0, 20, 1, dtype: np.Complex);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            double multiplierB1 = 9023.67;
            var b = a * multiplierB1;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB1 = new Complex[,]
            {
                {0*multiplierB1,  1*multiplierB1,  2*multiplierB1,  3*multiplierB1},
                {4*multiplierB1,  5*multiplierB1,  6*multiplierB1,  7*multiplierB1},
                {8*multiplierB1,  9*multiplierB1,  10*multiplierB1, 11*multiplierB1},
                {12*multiplierB1, 13*multiplierB1, 14*multiplierB1, 15*multiplierB1},
                {16*multiplierB1, 17*multiplierB1, 18*multiplierB1, 19*multiplierB1}
            };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(0, 20, 1, dtype: np.Complex);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            Complex multiplierB2 = 990425023.67864101;
            b = a * multiplierB2;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB2 = new Complex[,]
            {
                {0*multiplierB2,  1*multiplierB2,  2*multiplierB2,  3*multiplierB2},
                {4*multiplierB2,  5*multiplierB2,  6*multiplierB2,  7*multiplierB2},
                {8*multiplierB2,  9*multiplierB2,  10*multiplierB2, 11*multiplierB2},
                {12*multiplierB2, 13*multiplierB2, 14*multiplierB2, 15*multiplierB2},
                {16*multiplierB2, 17*multiplierB2, 18*multiplierB2, 19*multiplierB2}
            };
            AssertArray(b, ExpectedDataB2);
        }

        [TestMethod]
        public void test_division_operations_COMPLEX()
        {
            var a = np.arange(20000, 20020, 1, dtype: np.Complex);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            double divisorB1 = 611;
            var b = a / divisorB1;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB1 = new Complex[,]
            {
                {20000/divisorB1, 20001/divisorB1, 20002/divisorB1, 20003/divisorB1},
                {20004/divisorB1, 20005/divisorB1, 20006/divisorB1, 20007/divisorB1},
                {20008/divisorB1, 20009/divisorB1, 20010/divisorB1, 20011/divisorB1},
                {20012/divisorB1, 20013/divisorB1, 20014/divisorB1, 20015/divisorB1},
                {20016/divisorB1, 20017/divisorB1, 20018/divisorB1, 20019/divisorB1}
            };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(2000000, 2000020, 1, dtype: np.Complex);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            double divisorB2 = 2411;
            b = a / divisorB2;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB2 = new Complex[,]
            {
                {2000000/divisorB2, 2000001/divisorB2, 2000002/divisorB2, 2000003/divisorB2},
                {2000004/divisorB2, 2000005/divisorB2, 2000006/divisorB2, 2000007/divisorB2},
                {2000008/divisorB2, 2000009/divisorB2, 2000010/divisorB2, 2000011/divisorB2},
                {2000012/divisorB2, 2000013/divisorB2, 2000014/divisorB2, 2000015/divisorB2},
                {2000016/divisorB2, 2000017/divisorB2, 2000018/divisorB2, 2000019/divisorB2},
            };
            AssertArray(b, ExpectedDataB2);
        }

        [TestMethod]
        public void test_leftshift_operations_COMPLEX()
        {
            var a = np.arange(0, 20, 1, dtype: np.Complex);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            var b = a << 8;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB1 = new Complex[,]
            {
                {0,  256,  512,  768},
                {1024, 1280, 1536, 1792},
                {2048, 2304, 2560, 2816},
                {3072, 3328, 3584, 3840},
                {4096, 4352, 4608, 4864}
            };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(0, 20, 1, dtype: np.Complex);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            b = a << 24;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB2 = new Complex[,]
            {
                {0,  16777216,  33554432,  50331648},
                {67108864,  83886080, 100663296, 117440512},
                {134217728, 150994944, 167772160, 184549376},
                {201326592, 218103808, 234881024, 251658240},
                {268435456, 285212672, 301989888, 318767104}
            };
            AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_rightshift_operations_COMPLEX()
        {
            var a = np.arange(20000, 20020, 1, dtype: np.Complex);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            var b = a >> 8;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB1 = new Complex[,]
            {
                {78, 78, 78, 78},
                {78, 78, 78, 78},
                {78, 78, 78, 78},
                {78, 78, 78, 78},
                {78, 78, 78, 78}
            };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(2123450, 2123470, 1, dtype: np.Complex);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            b = a >> 8;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB2 = new Complex[,]
            {
                {8294 , 8294 , 8294 , 8294 },
                {8294 , 8294 , 8294 , 8294 },
                {8294 , 8294 , 8294 , 8294 },
                {8294 , 8294 , 8294 , 8294 },
                {8294 , 8294 , 8294 , 8294 }
            };
            AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_bitwiseand_operations_COMPLEX()
        {
            var a = np.arange(0.499, 32.499, 1, dtype: np.Complex);
            print(a);

            var b = a & 0x0f;
            print(b);

            var ExpectedDataB1 = new Complex[]
            { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
              0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Complex);
            print(a);

            b = a & 0xFF;
            print(b);

            var ExpectedDataB2 = new Complex[]
            { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };
            AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_bitwiseor_operations_COMPLEX()
        {
            var a = np.arange(0.499, 32.499, 1, dtype: np.Complex);
            print(a);

            var b = a | 0x100;
            print(b);

            var ExpectedDataB1 = new Complex[]
            { 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
              272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287 };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Complex);
            print(a);

            b = a | 0x1000;
            print(b);

            var ExpectedDataB2 = new Complex[]
            { 6144, 6145, 6146, 6147, 6148, 6149, 6150, 6151, 6152, 6153, 6154, 6155, 6156, 6157,
              6158, 6159, 6160, 6161, 6162, 6163, 6164, 6165, 6166, 6167, 6168, 6169, 6170, 6171,
              6172, 6173, 6174, 6175 };
            AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_bitwisexor_operations_COMPLEX()
        {
            var a = np.arange(0.499, 32.499, 1, dtype: np.Complex);
            print(a);

            var b = a ^ 0xAAA;
            print(b);

            var ExpectedDataB1 = new Complex[]
            { 2730, 2731, 2728, 2729, 2734, 2735, 2732, 2733, 2722, 2723, 2720, 2721, 2726, 2727, 2724,
              2725, 2746, 2747, 2744, 2745, 2750, 2751, 2748, 2749, 2738, 2739, 2736, 2737, 2742, 2743, 2740, 2741 };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Complex);
            print(a);

            b = a ^ 0xAAAA;
            print(b);

            var ExpectedDataB2 = new Complex[]
            { 41642, 41643, 41640, 41641, 41646, 41647, 41644, 41645, 41634, 41635, 41632, 41633,
              41638, 41639, 41636, 41637, 41658, 41659, 41656, 41657, 41662, 41663, 41660, 41661,
              41650, 41651, 41648, 41649, 41654, 41655, 41652, 41653};
            AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_remainder_operations_COMPLEX()
        {
            var a = np.arange(0, 32, 1, dtype: np.Complex);
            print(a);

            var b = a % 6;
            print(b);

            AssertArray(b, new Complex[] { 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
                                         4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1 });

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Complex);
            print(a);

            b = a % 6;
            print(b);

            AssertArray(b, new Complex[] { 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                                         0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3 });

        }

        [TestMethod]
        public void test_sqrt_operations_COMPLEX()
        {
            var a = np.arange(0, 32, 1, dtype: np.Complex);
            print(a);

            var b = np.sqrt(a);
            print(b);

            var ExpectedDataB1 = new Complex[]
            {
                new Complex(0, 0), new Complex(1, 0), new Complex(1.4142135623731, 0), new Complex(1.73205080756888, 0),
                new Complex(2, 0), new Complex(2.23606797749979, 0), new Complex(2.44948974278318, 0), new Complex(2.64575131106459, 0),
                new Complex(2.82842712474619, 0), new Complex(3, 0), new Complex(3.16227766016838, 0), new Complex(3.3166247903554, 0),
                new Complex(3.46410161513775, 0), new Complex(3.60555127546399, 0), new Complex(3.74165738677394, 0),
                new Complex(3.87298334620742, 0), new Complex(4, 0), new Complex(4.12310562561766, 0), new Complex(4.24264068711928, 0),
                new Complex(4.35889894354067, 0), new Complex(4.47213595499958, 0), new Complex(4.58257569495584, 0),
                new Complex(4.69041575982343, 0), new Complex(4.79583152331272, 0), new Complex(4.89897948556636, 0),
                new Complex(5, 0), new Complex(5.09901951359278, 0), new Complex(5.19615242270663, 0), new Complex(5.29150262212918, 0),
                new Complex(5.3851648071345, 0), new Complex(5.47722557505166, 0), new Complex(5.56776436283002, 0) };


            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Complex);
            print(a);

            b = np.sqrt(a);
            print(b);

            //var ExpectedDataB2 = new Complex[]
            //{
            //    45.25483399593904156165403917500m, 45.26588119102510082052098860500m, 45.27692569068708313286904083500m, 45.28796749689700967470411927000m,
            //    45.29900661162449818342010884000m, 45.31004303683676705673235173000m, 45.32107677449863944262772470000m, 45.33210782657254732035535008500m,
            //    45.34313619501853557248191637000m, 45.35416188179426604803550841500m, 45.36518488885502161676177205000m, 45.37620521815371021451616237500m,
            //    45.38722287164086887981595016500m, 45.39823785126466778157558615000m, 45.40925015897091423804894846500m, 45.42025979670305672700192442500m,
            //    45.43126676640218888713870387000m, 45.44227107000705351080508783000m, 45.45327270945404652799204274000m, 45.46427168667722098166265753000m,
            //    45.47526800360829099442558795000m, 45.48626166217663572657800009000m, 45.49725266430930332554095262000m, 45.50824101193101486671008537000m,
            //    45.51922670696416828574441006500m, 45.53020975132884230231592748500m, 45.54119014694280033534272413000m, 45.55216789572149440972813052500m,
            //    45.56314299957806905462845237000m, 45.57411546042336519327171558000m, 45.58508528016592402434979572500m, 45.59605246071199089500623262000m
            //};
            //AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_cbrt_operations_COMPLEX()
        {
            var a = np.arange(0, 32, 1, dtype: np.Complex);
            print(a);

            var b = np.cbrt(a);
            print(b);

            var ExpectedDataB1 = new Complex[]
            {
                0, 1, 01.25992104989487, 01.44224957030741, 01.5874010519682, 01.7099759466767, 01.81712059283214, 01.91293118277239,
                2, 2.0800838230519, 02.15443469003188, 02.22398009056931,
                02.28942848510666, 02.35133468772076, 02.41014226417523,
                02.46621207433047, 02.51984209978974, 02.57128159065823,
                02.62074139420889, 02.66840164872194, 02.7144176165949,
                02.75892417638112, 02.80203933065538, 02.84386697985156,
                02.88449914061481, 02.92401773821286, 02.96249606840737, 3,
                03.03658897187566, 03.07231682568584, 03.10723250595386,
                03.14138065239139
            };

            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Complex);
            print(a);

            b = np.cbrt(a);
            print(b);

            //var ExpectedDataB2 = new Complex[]
            //{
            //    12.69920841574560000000000000000m, 12.70127500787570000000000000000m, 12.70334092772480000000000000000m,
            //    12.70540617583920000000000000000m, 12.70747075276460000000000000000m, 12.70953465904630000000000000000m,
            //    12.71159789522850000000000000000m, 12.71366046185490000000000000000m, 12.71572235946840000000000000000m,
            //    12.71778358861120000000000000000m, 12.71984414982490000000000000000m, 12.72190404365040000000000000000m,
            //    12.72396327062770000000000000000m, 12.72602183129630000000000000000m, 12.72807972619490000000000000000m,
            //    12.73013695586160000000000000000m, 12.73219352083380000000000000000m, 12.73424942164790000000000000000m,
            //    12.73630465884010000000000000000m, 12.73835923294560000000000000000m, 12.74041314449890000000000000000m,
            //    12.74246639403400000000000000000m, 12.74451898208400000000000000000m, 12.74657090918140000000000000000m,
            //    12.74862217585820000000000000000m, 12.75067278264530000000000000000m, 12.75272273007330000000000000000m,
            //    12.75477201867200000000000000000m, 12.75682064897040000000000000000m, 12.75886862149700000000000000000m,
            //    12.76091593677950000000000000000m, 12.76296259534500000000000000000m
            //};
            //AssertArray(b, ExpectedDataB2);
        }

        [TestMethod]
        public void test_negative_operations_COMPLEX()
        {
            var a = np.arange(0.333, 32.333, 1, dtype: np.Complex);
            print(a);

            var b = -a;
            print(b);

            var ExpectedDataB2 = new Complex[]
            {
                -00.33300000000000000000000000000, -01.33300000000000000000000000000, -02.33300000000000000000000000000,
                -03.33300000000000000000000000000, -04.33300000000000000000000000000, -05.33300000000000000000000000000,
                -06.33300000000000000000000000000, -07.33300000000000000000000000000, -08.33300000000000000000000000000,
                -09.33300000000000000000000000000, -10.33300000000000000000000000000, -11.33300000000000000000000000000,
                -12.33300000000000000000000000000, -13.33300000000000000000000000000, -14.33300000000000000000000000000,
                -15.33300000000000000000000000000, -16.33300000000000000000000000000, -17.33300000000000000000000000000,
                -18.33300000000000000000000000000, -19.33300000000000000000000000000, -20.33300000000000000000000000000,
                -21.33300000000000000000000000000, -22.33300000000000000000000000000, -23.33300000000000000000000000000,
                -24.33300000000000000000000000000, -25.33300000000000000000000000000, -26.33300000000000000000000000000,
                -27.33300000000000000000000000000, -28.33300000000000000000000000000, -29.33300000000000000000000000000,
                -30.33300000000000000000000000000, -31.33300000000000000000000000000
            };
            AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_invert_operations_COMPLEX()
        {
            var a = np.arange(-32, 32, 1, dtype: np.Complex);
            print(a);

            var b = ~a;
            print(b);

            // this should not be changed at all.  complex can't be inverted.
            AssertArray(b, a.AsComplexArray());

        }

        [TestMethod]
        public void test_LESS_operations_COMPLEX()
        {
            var a = np.arange(-5, 5, 1, dtype: np.Complex);
            print(a);

            var b = a < -2;
            print(b);

            AssertArray(b, new Boolean[] { true, true, true, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_LESSEQUAL_operations_COMPLEX()
        {
            var a = np.arange(-5, 5, 1, dtype: np.Complex);
            print(a);

            var b = a <= -2;
            print(b);

            AssertArray(b, new Boolean[] { true, true, true, true, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_EQUAL_operations_COMPLEX()
        {
            var a = np.arange(-5, 5, 1, dtype: np.Complex);
            print(a);

            var b = a == -2;
            print(b);

            AssertArray(b, new Boolean[] { false, false, false, true, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_NOTEQUAL_operations_COMPLEX()
        {
            var a = np.arange(-5, 5, 1, dtype: np.Complex);
            print(a);

            var b = a != -2;
            print(b);

            AssertArray(b, new Boolean[] { true, true, true, false, true, true, true, true, true, true });
        }

        [TestMethod]
        public void test_GREATER_operations_COMPLEX()
        {
            var a = np.arange(-5, 5, 1, dtype: np.Complex);
            print(a);

            var b = a > -2;
            print(b);

            AssertArray(b, new Boolean[] { false, false, false, false, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_GREATEREQUAL_operations_COMPLEX()
        {
            var a = np.arange(-5, 5, 1, dtype: np.Complex);
            print(a);

            var b = a >= -2;
            print(b);

            AssertArray(b, new Boolean[] { false, false, false, true, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_arrayarray_or_COMPLEX()
        {
            var a = np.arange(0, 32, 1, dtype: np.Complex);
            var b = np.arange(33, 33 + 32, 1, dtype: np.Complex);
            var c = a | b;
            print(a);
            print(b);
            print(c);

            AssertArray(c, new Complex[] {33, 35, 35, 39, 37, 39, 39, 47, 41, 43, 43, 47,
                                        45, 47, 47, 63, 49, 51, 51, 55, 53, 55, 55, 63,
                                        57, 59, 59, 63, 61, 63, 63, 95 });
        }

        [TestMethod]
        public void test_bitwise_and_COMPLEX()
        {
            var x = np.arange(1023, 1039, 1, dtype: np.Complex).reshape(new shape(2, -1));
            var y = np.bitwise_and(x, 0x3FF);
            var z = x & 0x3FF;

            print(x);
            print(y);
            print(z);

            var ExpectedData = new Complex[,]
            {
                { 1023, 0, 1,  2,  3,  4,  5,  6 },
                {  7, 8, 9, 10, 11, 12, 13, 14 }
            };

            AssertArray(y, ExpectedData);
            AssertArray(z, ExpectedData);
        }

        [TestMethod]
        public void test_bitwise_or_COMPLEX()
        {
            var x = np.arange(1023, 1039, 1, dtype: np.Complex).reshape(new shape(2, -1));
            var y = np.bitwise_or(x, 0x10);
            var z = x | 0x10;

            print(x);
            print(y);
            print(z);

            var ExpectedData = new Complex[,]
            {
                { 1023, 1040, 1041, 1042, 1043, 1044, 1045, 1046 },
                { 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054 }
            };

            AssertArray(y, ExpectedData);
            AssertArray(z, ExpectedData);
        }

        [TestMethod]
        public void test_bitwise_xor_COMPLEX()
        {
            var a = np.bitwise_xor(13, 17);
            Assert.AreEqual(28, a.GetItem(0));
            print(a);

            var b = np.bitwise_xor(31, 5);
            Assert.AreEqual(26, b.GetItem(0));
            print(b);

            var c = np.bitwise_xor(new Complex[] { 31, 3 }, 5);
            AssertArray(c, new Complex[] { 26, 6 });
            print(c);

            var d = np.bitwise_xor(new Complex[] { 31, 3 }, new Complex[] { 5, 6 });
            AssertArray(d, new Complex[] { 26, 5 });
            print(d);

            var e = np.bitwise_xor(new bool[] { true, true }, new bool[] { false, true });
            AssertArray(e, new bool[] { true, false });
            print(e);

            return;
        }

        [TestMethod]
        public void test_bitwise_not_COMPLEX()
        {
            var a = np.bitwise_not(13);
            Assert.AreEqual(-14, a.GetItem(0));
            print(a);

            var b = np.bitwise_not(31);
            Assert.AreEqual(-32, b.GetItem(0));
            print(b);

            // can't inverse a complex
            var c = np.bitwise_not(new Complex[] { 31, 3 });
            AssertArray(c, new Complex[] { 31, 3 });
            print(c);

            // can't inverse a complex
            var d = np.bitwise_not(new Complex[] { 31, 3 });
            AssertArray(d, new Complex[] { 31, 3 });
            print(d);

            var e = np.bitwise_not(new bool[] { true, false });
            AssertArray(e, new bool[] { false, true });
            print(e);

            return;
        }

        [TestMethod]
        public void test_invert_COMPLEX()
        {
            var a = np.invert(13);
            Assert.AreEqual(-14, a.GetItem(0));
            print(a);

            var b = np.invert(31);
            Assert.AreEqual(-32, b.GetItem(0));
            print(b);

            // can't inverse a complex
            var c = np.invert(new Complex[] { 31, 3 });
            AssertArray(c, new Complex[] { 31, 3 });
            print(c);

            // can't inverse a complex
            var d = np.invert(new Complex[] { 31, 3 });
            AssertArray(d, new Complex[] { 31, 3 });
            print(d);

            var e = np.invert(new bool[] { true, false });
            AssertArray(e, new bool[] { false, true });
            print(e);

            return;
        }

        [TestMethod]
        public void test_right_shift_COMPLEX()
        {
            var x = np.arange(1023, 1039, 1, dtype: np.Complex).reshape(new shape(2, -1));
            var y = np.right_shift(x, 2);
            var z = x >> 2;

            print(x);
            print(y);
            print(z);

            var ExpectedData = new Complex[,]
            {
                { 255, 256, 256, 256, 256, 257, 257, 257 },
                { 257, 258, 258, 258, 258, 259, 259, 259 }
            };

            AssertArray(y, ExpectedData);
            AssertArray(z, ExpectedData);
        }

        [TestMethod]
        public void test_left_shift_COMPLEX()
        {
            var x = np.arange(1023, 1039, 1, dtype: np.Complex).reshape(new shape(2, -1));
            var y = np.left_shift(x, 2);
            var z = x << 2;

            print(x);
            print(y);
            print(z);

            var ExpectedData = new Complex[,]
            {
                { 4092, 4096, 4100, 4104, 4108, 4112, 4116, 4120 },
                { 4124, 4128, 4132, 4136, 4140, 4144, 4148, 4152 }
            };

            AssertArray(y, ExpectedData);
            AssertArray(z, ExpectedData);
        }

        [TestMethod]
        public void test_min_COMPLEX()
        {
            Complex[] TestData = new Complex[] { 2.5, -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0 };
            var x = np.array(TestData);
            Complex y = (Complex)np.min(x);

            print(x);
            print(y);

            Assert.AreEqual(new Complex(-1.7, 0), y);
        }

        [TestMethod]
        public void test_max_COMPLEX()
        {
            Complex[] TestData = new Complex[] { 2.5, -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0 };
            var x = np.array(TestData);
            Complex y = (Complex)np.max(x);

            print(x);
            print(y);

            Assert.AreEqual(new Complex(2.5, 0), y);
        }

        [TestMethod]
        public void test_isnan_COMPLEX()
        {
            Complex[] TestData = new Complex[] { 2.5, -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0 };
            var x = np.array(TestData);
            var y = np.isnan(x);

            print(x);
            print(y);

            // complex don't support NAN so must be false
            AssertArray(y, new bool[] { false, false, false, false, false, false, false, false });

        }

        [TestMethod]
        public void test_setdiff1d_COMPLEX()
        {
            Complex[] TestDataA = new Complex[] { 1, 2, 3, 2, 4, };
            Complex[] TestDataB = new Complex[] { 3, 4, 5, 6 };

            var a = np.array(TestDataA);
            var b = np.array(TestDataB);
            ndarray c = np.setdiff1d(a, b);

            print(a);
            print(b);
            print(c);

            AssertArray(c, new Complex[] { 1, 2 });

        }

        [TestMethod] 
        public void test_setdiff1d_2_COMPLEX()
        {
            Complex[] TestDataB = new Complex[] { 3, 4, 5, 6 };

            var a = np.arange(1, 39, dtype: np.Complex).reshape(new shape(2, -1));
            var b = np.array(TestDataB);
            ndarray c = np.setdiff1d(a, b);

            print(a);
            print(b);
            print(c);

            AssertArray(c, new Complex[] {1,  2,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                                         29, 30, 31, 32, 33, 34, 35, 36, 37, 38 });

        }

        [TestMethod]
        public void test_rot90_1_COMPLEX()
        {
            ndarray m = np.array(new Complex[,] { { 1, 2 }, { 3, 4 } }, np.Complex);
            print(m);
            print("************");

            ndarray n = np.rot90(m);
            print(n);
            AssertArray(n, new Complex[,] { { 2, 4 }, { 1, 3 }, });
            print("************");

            n = np.rot90(m, 2);
            print(n);
            AssertArray(n, new Complex[,] { { 4, 3 }, { 2, 1 }, });
            print("************");

            m = np.arange(8, dtype: np.Complex).reshape(new shape(2, 2, 2));
            n = np.rot90(m, 1, new int[] { 1, 2 });
            print(n);
            AssertArray(n, new Complex[,,] { { { 1, 3 }, { 0, 2 } }, { { 5, 7 }, { 4, 6 } } });

        }

        [TestMethod]
        public void test_flip_1_COMPLEX()
        {
            ndarray A = np.arange(8, dtype: np.Complex).reshape(new shape(2, 2, 2));
            ndarray B = np.flip(A, 0);
            print(A);
            print("************");
            print(B);
            AssertArray(B, new Complex[,,] { { { 4, 5 }, { 6, 7 } }, { { 0, 1 }, { 2, 3 } } });

            print("************");
            ndarray C = np.flip(A, 1);
            print(C);
            AssertArray(C, new Complex[,,] { { { 2, 3 }, { 0, 1 } }, { { 6, 7 }, { 4, 5 } } });
            print("************");

        }

        [TestMethod]
        public void test_trim_zeros_1_COMPLEX()
        {
            ndarray a = np.array(new Complex[] { 0, 0, 0, 1, 2, 3, 0, 2, 1, 0 });

            var b = np.trim_zeros(a);
            print(b);
            AssertArray(b, new Complex[] { 1, 2, 3, 0, 2, 1 });

            var c = np.trim_zeros(a, "b");
            print(c);
            AssertArray(c, new Complex[] { 0, 0, 0, 1, 2, 3, 0, 2, 1 });
        }

        [TestMethod]
        public void test_logical_and_1_COMPLEX()
        {

            var x = np.arange(5, dtype: np.Complex);
            var c = np.logical_and(x > 1, x < 4);
            AssertArray(c, new bool[] { false, false, true, true, false });
            print(c);

            var y = np.arange(6, dtype: np.Complex).reshape((2, 3));
            var d = np.logical_and(y > 1, y < 4);
            AssertArray(d, new bool[,] { { false, false, true }, { true, false, false } });
            print(d);
        }

        [TestMethod]
        public void test_logical_or_1_COMPLEX()
        {

            var x = np.arange(5, dtype: np.Complex);
            var c = np.logical_or(x < 1, x > 3);
            AssertArray(c, new bool[] { true, false, false, false, true });
            print(c);

            var y = np.arange(6, dtype: np.Complex).reshape((2, 3));
            var d = np.logical_or(y < 1, y > 3);
            AssertArray(d, new bool[,] { { true, false, false }, { false, true, true } });
            print(d);
        }

        [TestMethod]
        public void test_logical_xor_1_COMPLEX()
        {

            var x = np.arange(5, dtype: np.Complex);
            var c = np.logical_xor(x < 1, x > 3);
            AssertArray(c, new bool[] { true, false, false, false, true });
            print(c);

            var y = np.arange(6, dtype: np.Complex).reshape((2, 3));
            var d = np.logical_xor(y < 1, y > 3);
            AssertArray(d, new bool[,] { { true, false, false }, { false, true, true } });
            print(d);

            var e = np.logical_xor(0, np.eye(2, dtype: np.Complex));
            AssertArray(e, new bool[,] { { true, false }, { false, true } });
        }

        [TestMethod]
        public void test_logical_not_1_COMPLEX()
        {
            var x = np.arange(5, dtype: np.Complex);
            var c = np.logical_not(x < 3);
            AssertArray(c, new bool[] { false, false, false, true, true });
            print(c);
        }

        [TestMethod]
        public void test_greater_1_COMPLEX()
        {
            var a = np.greater(new Complex[] { 4, 2, 1 }, new Complex[] { 2, 2, 2 });
            AssertArray(a, new bool[] { true, false, false });
            print(a);

            var b = np.greater(new Complex[] { 4, 2, 1 }, 1);
            AssertArray(b, new bool[] { true, true, false });
            print(b);

            var c = np.greater(2, new Complex[] { 4, 2, 1 });
            AssertArray(c, new bool[] { false, false, true });
            print(c);

        }

        [TestMethod]
        public void test_greater_equal_1_COMPLEX()
        {
            var a = np.greater_equal(new Complex[] { 4, 2, 1 }, new Complex[] { 2, 2, 2 });
            AssertArray(a, new bool[] { true, true, false });
            print(a);

            var b = np.greater_equal(new Complex[] { 4, 2, 1 }, 1);
            AssertArray(b, new bool[] { true, true, true });
            print(b);

            var c = np.greater_equal(2, new Complex[] { 4, 2, 1 });
            AssertArray(c, new bool[] { false, true, true });
            print(c);
        }

        [TestMethod]
        public void test_less_1_COMPLEX()
        {
            var a = np.less(new Complex[] { 4, 2, 1 }, new Complex[] { 2, 2, 2 });
            AssertArray(a, new bool[] { false, false, true });
            print(a);

            var b = np.less(new Complex[] { 4, 2, 1 }, 1);
            AssertArray(b, new bool[] { false, false, false });
            print(b);

            var c = np.less(2, new Complex[] { 4, 2, 1 });
            AssertArray(c, new bool[] { true, false, false });
            print(c);
        }

        [TestMethod]
        public void test_less_equal_1_COMPLEX()
        {
            var a = np.less_equal(new Complex[] { 4, 2, 1 }, new Complex[] { 2, 2, 2 });
            AssertArray(a, new bool[] { false, true, true });
            print(a);

            var b = np.less_equal(new Complex[] { 4, 2, 1 }, 1);
            AssertArray(b, new bool[] { false, false, true });
            print(b);

            var c = np.less_equal(2, new Complex[] { 4, 2, 1 });
            AssertArray(c, new bool[] { true, true, false });
            print(c);
        }

        [TestMethod]
        public void test_equal_1_COMPLEX()
        {
            var a = np.equal(new Complex[] { 4, 2, 1 }, new Complex[] { 2, 2, 2 });
            AssertArray(a, new bool[] { false, true, false });
            print(a);

            var b = np.equal(new Complex[] { 4, 2, 1 }, 1);
            AssertArray(b, new bool[] { false, false, true });
            print(b);

            var c = np.equal(2, new Complex[] { 4, 2, 1 });
            AssertArray(c, new bool[] { false, true, false });
            print(c);
        }

        [TestMethod]
        public void test_not_equal_1_COMPLEX()
        {
            var a = np.not_equal(new Complex[] { 4, 2, 1 }, new Complex[] { 2, 2, 2 });
            AssertArray(a, new bool[] { true, false, true });
            print(a);

            var b = np.not_equal(new Complex[] { 4, 2, 1 }, 1);
            AssertArray(b, new bool[] { true, true, false });
            print(b);

            var c = np.not_equal(2, new Complex[] { 4, 2, 1 });
            AssertArray(c, new bool[] { true, false, true });
            print(c);
        }

        [TestMethod]
        public void test_copyto_1_COMPLEX()
        {
            var a = np.zeros((10, 5), dtype: np.Complex);
            var b = new int[] { 11, 22, 33, 44, 55 };
            np.copyto(a, b);

            AssertShape(a, 10, 5);
            Assert.AreEqual(new Complex(1650, 0), a.Sum().GetItem(0));
            print(a);

            a = np.zeros((10, 5), dtype: np.Complex);
            np.copyto(a, 99);
            AssertShape(a, 10, 5);
            Assert.AreEqual(new Complex(4950, 0), a.Sum().GetItem(0));
            print(a);

            a = np.zeros((10, 5), dtype: np.Complex);
            var c = np.arange(11, 60, 11);

            try
            {
                np.copyto(c, a);
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains("not broadcastable"));
                return;
            }

            Assert.IsTrue(false);

        }

        [TestMethod]
        public void test_copyto_2_COMPLEX()
        {
            var a = np.zeros((1, 2, 2, 1, 2), dtype: np.Complex);
            var b = new int[] { 1, 2 };
            np.copyto(a, b);

            AssertArray(a, new Complex[,,,,] { { { { { 1.0, 2.0 } }, { { 1.0, 2.0 } } }, { { { 1.0, 2.0 } }, { { 1.0, 2.0, } } } } });

        }

        #endregion

        #region from MathematicalFunctionsTests

        [TestMethod]
        public void test_sin_1_COMPLEX()
        {
            var ExpectedResult = new Complex[] { 0, 0.909297426825682, -0.756802495307928, -0.279415498198926, 0.989358246623382 };

            var a = np.arange(0, 10, dtype: np.Complex);
            a = a["::2"] as ndarray;
            var b = np.sin(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.arange(0, 10, dtype: np.Complex).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.sin(a);

            var ExpectedDataB = new Complex[,,]
                {{{ 0,                  0.841470984807897, 0.909297426825682, 0.141120008059867, -0.756802495307928},
                  {-0.958924274663138, -0.279415498198926, 0.656986598718789, 0.989358246623382,  0.412118485241757}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.array(new Complex[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.sin(a, where: a > 2);
            //AssertArray(b, new Complex[,] { { NaN, np.NaN, np.NaN, 0.141120008059867, -0.756802495307928 } });
            print(b);

            a = np.array(new Complex[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.sin(a, where: new bool[,] { { false, false, false, true, true } });
            //AssertArray(b, new Complex[,] { { np.NaN, np.NaN, np.NaN, 0.141120008059867, -0.756802495307928 } });
            print(b);

        }

        [TestMethod]
        public void test_cos_1_COMPLEX()
        {
            var ExpectedResult = new Complex[] { 1.0, -0.416146836547142, -0.653643620863612, 0.960170286650366, -0.145500033808614 };

            var a = np.arange(0, 10, dtype: np.Complex);
            a = a["::2"] as ndarray;
            var b = np.cos(a);
            AssertArray(b, ExpectedResult);
            print(b);

            print("********");

            a = np.arange(0, 10, dtype: np.Complex).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.cos(a);

            var ExpectedDataB = new Complex[,,]
                {{{ 1.0,               0.54030230586814, -0.416146836547142, -0.989992496600445, -0.653643620863612},
                  { 0.283662185463226, 0.960170286650366, 0.753902254343305, -0.145500033808614, -0.911130261884677}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.array(new Complex[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.cos(a, where: a > 2);
            //AssertArray(b, new Complex[,] { { new Complex(double.NaN, 0), new Complex(double.NaN, 0), new Complex(double.NaN, 0), -0.989992496600445, -0.65364362086361 } });
            print(b);

            a = np.array(new Complex[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.cos(a, where: new bool[,] { { false, false, false, true, true } });
            //AssertArray(b, new Complex[,] { { np.NaN, np.NaN, np.NaN, -0.989992496600445, -0.65364362086361 } });
            print(b);

        }

        [TestMethod]
        public void test_tan_1_COMPLEX()
        {
            var ExpectedResult = new Complex[] { 0.0, -2.18503986326152, 1.15782128234958, -0.291006191384749, -6.79971145522038 };

            var a = np.arange(0, 10, dtype: np.Complex);
            a = a["::2"] as ndarray;
            var b = np.tan(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.arange(0, 10, dtype: np.Complex).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.tan(a);

            var ExpectedDataB = new Complex[,,]
                {{{ 0.0, 1.5574077246549, -2.18503986326152, -0.142546543074278, 1.15782128234958},
                  { -3.38051500624659, -0.291006191384749, 0.871447982724319, -6.79971145522038, -0.45231565944181}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.array(new Complex[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.tan(a, where: a > 2);
            //AssertArray(b, new Complex[,] { { np.NaN, np.NaN, np.NaN, -0.142546543074278, 1.15782128234958 } });
            print(b);

            a = np.array(new Complex[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.tan(a, where: new bool[,] { { false, false, false, true, true } });
            //AssertArray(b, new Complex[,] { { np.NaN, np.NaN, np.NaN, -0.142546543074278, 1.15782128234958 } });
            print(b);

        }

        [TestMethod]
        public void test_arcsin_1_COMPLEX()
        {
            var ExpectedResult = new Complex[] { -1.5707963267949, -0.958241588455558, -0.6897750007855, -0.471861837279642,
                                                -0.276226630763592, -0.091034778037415, 0.091034778037415, 0.276226630763592,
                                                 0.471861837279642, 0.6897750007855, 0.958241588455558, 1.5707963267949 };

            Complex ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Complex);
            var b = np.arcsin(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Complex).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arcsin(a);

            var ExpectedDataB = new Complex[,,]
                {{{ -1.5707963267949, -0.958241588455558, -0.6897750007855},
                  { -0.471861837279642, -0.276226630763592, -0.091034778037415}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Complex);
            a = a.A("::2");
            b = np.arcsin(a, where: a > -0.5);
            //AssertArray(b, new Complex[] { np.NaN, np.NaN, -0.276226630763592, 0.091034778037415, 0.471861837279642, 0.958241588455558 });
            print(b);

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Complex);
            a = a.A("::2");
            b = np.arcsin(a, where: new bool[] { false, false, true, true, true, true });
            //AssertArray(b, new Complex[] { np.NaN, np.NaN, -0.276226630763592, 0.091034778037415, 0.471861837279642, 0.958241588455558 });
            print(b);

        }

        [TestMethod]
        public void test_arccos_1_COMPLEX()
        {
            var ExpectedResult = new Complex[] { 3.14159265358979, 2.52903791525045, 2.2605713275804, 2.04265816407454,
                                                1.84702295755849, 1.66183110483231, 1.47976154875748, 1.29456969603131,
                                                1.09893448951525, 0.881021326009397, 0.612554738339339, 0.0 };

            Complex ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Complex);
            var b = np.arccos(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Complex).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arccos(a);

            var ExpectedDataB = new Complex[,,]
                {{{3.14159265358979, 2.52903791525045, 2.2605713275804},
                  {2.04265816407454, 1.84702295755849, 1.66183110483231}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Complex);
            a = a.A("::2");
            b = np.arccos(a, where: a > -0.5);
            //AssertArray(b, new Complex[] { np.NaN, np.NaN, 1.84702295755849, 1.47976154875748, 1.09893448951525, 0.612554738339339 });
            print(b);

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Complex);
            a = a.A("::2");
            b = np.arccos(a, where: new bool[] { false, false, true, true, true, true });
            //AssertArray(b, new Complex[] { np.NaN, np.NaN, 1.84702295755849, 1.47976154875748, 1.09893448951525, 0.612554738339339 });
            print(b);

        }

        [TestMethod]
        public void test_arctan_1_COMPLEX()
        {
            var ExpectedResult = new Complex[] { -0.785398163397448, -0.685729510906286, -0.566729217523506, -0.426627493126876,
                                                -0.266252049150925, -0.090659887200745, 0.090659887200745,   0.266252049150925,
                                                 0.426627493126876, 0.566729217523506, 0.685729510906286, 0.785398163397448 };

            Complex ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Complex);
            var b = np.arctan(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Complex).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arctan(a);

            var ExpectedDataB = new Complex[,,]
                {{{-0.785398163397448, -0.685729510906286, -0.566729217523506},
                  {-0.426627493126876, -0.266252049150925, -0.090659887200745}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Complex);
            a = a.A("::2");
            b = np.arctan(a, where: a > -0.5);
            //AssertArray(b, new Complex[] { np.NaN, np.NaN, -0.266252049150925, 0.090659887200745, 0.426627493126876, 0.685729510906286 });
            print(b);

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Complex);
            a = a.A("::2");
            b = np.arctan(a, where: new bool[] { false, false, true, true, true, true });
            //AssertArray(b, new Complex[] { np.NaN, np.NaN, -0.266252049150925, 0.090659887200745, 0.426627493126876, 0.685729510906286 });
            print(b);

        }

        [TestMethod]
        public void test_hypot_1_COMPLEX()
        {

            var a = np.hypot(np.ones((3, 3), dtype: np.Complex) * 3, np.ones((3, 3), dtype: np.Complex) * 4);
            print(a);
            AssertArray(a, new Complex[,] { { 5, 5, 5 }, { 5, 5, 5 }, { 5, 5, 5 } });

            var b = np.hypot(np.ones((3, 3), dtype: np.Complex) * 3, new Complex[] { 4 });
            print(b);
            AssertArray(b, new Complex[,] { { 5, 5, 5 }, { 5, 5, 5 }, { 5, 5, 5 } });

        }

        [TestMethod]
        public void test_arctan2_1_COMPLEX()
        {
            var x = np.array(new Complex[] { -1, +1, +1, -1 });
            var y = np.array(new Complex[] { -1, -1, +1, +1 });
            var z = np.arctan2(y, x) * 180 / Math.PI;
            AssertArray(z, new Complex[] { -135.0, -45.0, 45.0, 135.0 });
            print(z);

            var a = np.arctan2(new Complex[] { 1.0, -1.0 }, new Complex[] { 0.0, 0.0 });
            AssertArray(a, new Complex[] { 1.5707963267949, -1.5707963267949 });
            print(a);

        }

        #region Hyperbolic functions

        [TestMethod]
        public void test_sinh_1_COMPLEX()
        {
            var ExpectedResult = new Complex[] { 0.0, 3.62686040784702, 27.2899171971278, 201.713157370279, 1490.47882578955 };

            var a = np.arange(0, 10, dtype: np.Complex);
            a = a["::2"] as ndarray;
            var b = np.sinh(a);
            AssertArray(b, ExpectedResult);
            print(b);
        }

        [TestMethod]
        public void test_cosh_1_COMPLEX()
        {
            var ExpectedResult = new Complex[] { 1.0, 3.76219569108363, 27.3082328360165, 201.715636122456, 1490.47916125218 };

            var a = np.arange(0, 10, dtype: np.Complex);
            a = a["::2"] as ndarray;
            var b = np.cosh(a);
            AssertArray(b, ExpectedResult);
            print(b);

 
            print("********");

            a = np.arange(0, 10, dtype: np.Complex).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.cosh(a);

            var ExpectedDataB = new Complex[,,]
                {{{ 1.0,               1.54308063481524, 3.76219569108363, 10.0676619957778, 27.3082328360165},
                  { 74.2099485247878, 201.715636122456, 548.317035155212, 1490.47916125218, 4051.54202549259}}};

            AssertArray(b, ExpectedDataB);
            print(b);


        }

        [TestMethod]
        public void test_tanh_1_COMPLEX()
        {
            var ExpectedResult = new Complex[] { 0.0, 0.964027580075817, 0.999329299739067, 0.999987711650796, 0.999999774929676 };

            var a = np.arange(0, 10, dtype: np.Complex);
            a = a["::2"] as ndarray;
            var b = np.tanh(a);
            AssertArray(b, ExpectedResult);
            print(b);

            print("********");

            a = np.arange(0, 10, dtype: np.Complex).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.tanh(a);

            var ExpectedDataB = new Complex[,,]
                {{{ 0.0, 0.761594155955765, 0.964027580075817, 0.99505475368673, 0.999329299739067},
                  { 0.999909204262595, 0.999987711650796, 0.999998336943945, 0.999999774929676, 0.999999969540041}}};

            AssertArray(b, ExpectedDataB);
            print(b);

        }

        [TestMethod]
        public void test_arcsinh_1_COMPLEX()
        {
            var ExpectedResult = new Complex[] { -0.881373587019543, -0.7468029948789, -0.599755399970846, -0.440191235352683,
                                                -0.26945474934928, -0.090784335188522, 0.0907843351885222, 0.269454749349279,
                                                 0.440191235352683, 0.599755399970846, 0.7468029948789, 0.881373587019543 };

            Complex ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            var b = np.arcsinh(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arcsinh(a);

            var ExpectedDataB = new Complex[,,]
                {{{ -0.881373587019543, -0.7468029948789, -0.599755399970846},
                  { -0.440191235352683, -0.26945474934928, -0.090784335188522}}};

            AssertArray(b, ExpectedDataB);
            print(b);
             
        }

        [TestMethod]
        public void test_arccosh_1_COMPLEX()
        {
            var ExpectedResult = new Complex[] { 0.0, 0.423235459210748, 0.594240703336901, 0.722717193587915,
                                                0.82887090230963, 0.920606859928063, 1.00201733044986, 1.07555476344184,
                                                1.1428302089675, 1.20497120816827, 1.26280443110946, 1.31695789692482 };

            Complex ref_step = 0;
            var a = np.linspace(1.0, 2.0, ref ref_step, 12);
            var b = np.arccosh(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(1.0, 2.0, ref ref_step, 12).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arccosh(a);

            var ExpectedDataB = new Complex[,,]
                {{{0.0, 0.423235459210748, 0.594240703336901},
                  {0.722717193587915, 0.82887090230963, 0.920606859928063}}};

            AssertArray(b, ExpectedDataB);
            print(b);

  
        }

        [TestMethod]
        public void test_arctanh_1_COMPLEX()
        {
            var ExpectedResult = new Complex[] { double.NegativeInfinity, -1.15129254649702, -0.752038698388137, -0.490414626505863,
                                                     -0.279807893967711, -0.0911607783969772, 0.0911607783969772, 0.279807893967711,
                                                      0.490414626505863, 0.752038698388137, 1.15129254649702, double.PositiveInfinity };

            Complex ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            var b = np.arctanh(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arctanh(a);

            var ExpectedDataB = new Complex[,,]
                {{{double.NegativeInfinity, -1.15129254649702, -0.752038698388137},
                  {-0.490414626505863, -0.279807893967711, -0.0911607783969772}}};

            AssertArray(b, ExpectedDataB);
            print(b);

        }

        #endregion

        [TestMethod]
        public void test_degrees_1_COMPLEX()
        {
            var rad = np.arange(12.0, dtype: np.Complex) * Math.PI / 6;
            var a = np.degrees(rad);
            AssertArray(a, new Complex[] { 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330 });
            print(a);

            //var _out = np.zeros((rad.shape));
            //var r = np.degrees(rad, _out);
            //print(np.all(r == _out));

        }

        [TestMethod]
        public void test_radians_1_COMPLEX()
        {
            var deg = np.arange(12.0, dtype: np.Complex) * 30.0;
            var a = np.radians(deg);
            AssertArray(a, new Complex[] { 0.0, 0.523598775598299, 1.0471975511966, 1.5707963267949, 2.0943951023932,
                                         2.61799387799149, 3.14159265358979, 3.66519142918809, 4.18879020478639,
                                        4.71238898038469, 5.23598775598299, 5.75958653158129 });
            print(a);

            //var _out = np.zeros((deg.shape));
            //var r = np.radians(deg, _out);
            //print(np.all(r == _out));

        }

 
        [TestMethod]
        public void test_around_1_COMPLEX()
        {
    
            ndarray a = np.around(np.array(new Complex[] { new Complex(0.37, .49), new Complex(1.64, .51) } ));
            print(a);
            AssertArray(a, new Complex[] { new Complex(0, 0), new Complex(2, 1) });

            ndarray b = np.around(np.array(new Complex[] { 0.37, 1.64 }), decimals: 1);
            print(b);
            AssertArray(b, new Complex[] { 0.4, 1.6 });

            ndarray c = np.around(np.array(new Complex[] { .5, 1.5, 2.5, 3.5, 4.5 })); // rounds to nearest even value
            print(c);
            AssertArray(c, new Complex[] { 0.0, 2.0, 2.0, 4.0, 4.0 });

            ndarray d = np.around(np.array(new Complex[] { 1, 2, 3, 11 }), decimals: 1); // ndarray of ints is returned
            print(d);
            AssertArray(d, new Complex[] { 1, 2, 3, 11 });

            ndarray e = np.around(np.array(new Complex[] { 1, 2, 3, 11 }), decimals: -1);
            print(e);
            AssertArray(e, new Complex[] { 0, 0, 0, 10 });
        }

        [TestMethod]
        public void test_round_1_COMPLEX()
        {
            Complex ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12).reshape((2, 2, 3));
            print(a);

            var ExpectedData1 = new Complex[,,] { { { -1.0, -0.82, -0.64 }, { -0.45, -0.27, -0.09 } }, { { 0.09, 0.27, 0.45 }, { 0.64, 0.82, 1.0 } } };

            print("********");
            var b = np.round_(a, 2);
            AssertArray(b, ExpectedData1);
            print(b);

            print("********");

            var c = np.round(a, 2);
            AssertArray(c, ExpectedData1);
            print(c);

            var ExpectedData2 = new Complex[,,] { { { -1.0, -0.8182, -0.6364 }, { -0.4545, -0.2727, -0.0909 } }, { { 0.0909, 0.2727, 0.4545 }, { 0.6364, 0.8182, 1.0 } } };

            print("********");
            b = np.round_(a, 4);
            AssertArray(b, ExpectedData2);
            print(b);

            print("********");

            c = np.round(a, 4);
            AssertArray(c, ExpectedData2);
            print(c);

        }

        [TestMethod]
        public void test_fix_1_COMPLEX()
        {
            var a = np.fix((Complex)3.14);
            Assert.AreEqual((Complex)3.0, a.GetItem(0));
            print(a);

            var b = np.fix((Complex)3);
            Assert.AreEqual((Complex)3, b.GetItem(0));
            print(b);

            var c = np.fix(new Complex[] { 2.1, 2.9, -2.1, -2.9 });
            AssertArray(c, new Complex[] { 2.0, 2.0, -2.0, -2.0 });
            print(c);
        }

        [TestMethod]
        public void test_floor_1_COMPLEX()
        {
            Complex[] TestData = new Complex[] { -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0 };
            var x = np.array(TestData);
            var y = np.floor(x);

            print(x);
            print(y);

            AssertArray(y, new Complex[] { -2.0, -2.0, -1.0, 0.0, 1.0, 1.0, 2.0 });

        }

        [TestMethod]
        public void test_ceil_1_COMPLEX()
        {
            Complex[] TestData = new Complex[] { -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0 };
            var x = np.array(TestData);
            var y = np.ceil(x);

            print(x);
            print(y);

            AssertArray(y, new Complex[] { -1.0, -1.0, -0.0, 1.0, 2.0, 2.0, 2.0 });

        }

        [TestMethod]
        public void test_trunc_1_COMPLEX()
        {
            var a = np.trunc((Complex)3.14);
            Assert.AreEqual(new Complex(3.0, 0), a.GetItem(0));
            print(a);

            var b = np.trunc(3m);
            Assert.AreEqual(3m, b.GetItem(0));
            print(b);

            var c = np.trunc(new Complex[] { new Complex(2.1, 2.1), 2.9, -2.1, -2.9 });
            AssertArray(c, new Complex[] {new Complex(2.0, 2.0), 2.0, -2.0, -2.0 });
            print(c);
        }

        [TestMethod]
        public void test_prod_2_COMPLEX()
        {
            ndarray a = np.prod(np.array(new Complex[] { 1.0, 2.0 }));
            print(a);
            Assert.AreEqual((Complex)2, a.GetItem(0));
            print("*****");

            ndarray b = np.prod(np.array(new Complex[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }));
            print(b);
            Assert.AreEqual((Complex)24, b.GetItem(0));
            print("*****");

            ndarray c = np.prod(np.array(new Complex[,] { { new Complex(1.0, 3.0), 2.0 }, { new Complex(3.0, 4.0), 4.0 } }), axis: 1);
            print(c);
            AssertArray(c, new Complex[] { new Complex(2, 6), new Complex(12, 16) });
            print("*****");

            ndarray d = np.array(new Complex[] { 1, 2, 3 }, dtype: np.Complex);
            bool e = np.prod(d).Dtype.TypeNum == NPY_TYPES.NPY_COMPLEX;
            print(e);
            Assert.AreEqual(true, e);
            print("*****");

        }

        [TestMethod]
        public void test_sum_2_COMPLEX()
        {
            Complex[] TestData = new Complex[] { new Complex(10.5, 2.5), 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.Complex).reshape(new shape(3, 2, -1));
            x = x * 3;

            var y = np.sum(x, axis: 0);
            print(y);
            AssertArray(y, new Complex[,] { { new Complex(340.5, 7.5), new Complex(450, 0)  }, { new Complex(339, 0), new Complex(450, 0) } });

            print("*****");

            y = np.sum(x, axis: 1);
            print(y);
            AssertArray(y, new Complex[,] { { new Complex(106.5, 7.5), 180 }, { 264, 315 }, { 309, 405 } });

            print("*****");

            y = np.sum(x, axis: 2);
            print(y);
            AssertArray(y, new Complex[,] { { new Complex(76.5, 7.5), 210 }, { 504, 75 }, { 210, 504 } });

            print("*****");

        }

        [TestMethod]
        public void test_cumprod_2_COMPLEX()
        {
            var k1 = new Complex(1, 4);
            var k2 = new Complex(2, 2.25);

            var k3 = System.Numerics.Complex.Multiply(k1, k2);

            ndarray a = np.array(new Complex[] { new Complex(1, 4), new Complex(2, 2.25), new Complex(3, 6.67) });
            ndarray b = np.cumprod(a);          // intermediate results 1, 1*2
                                                // total product 1*2*3 = 6
            print(b);
            AssertArray(b, new Complex[] { new Complex(1, 4), k3, new Complex(-89.3675, -15.94) });
            print("*****");

            a = np.array(new Complex[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            ndarray c = np.cumprod(a, dtype: np.Complex); //specify type of output
            print(c);
            AssertArray(c, new Complex[] { 1, 2, 6, 24, 120, 720 });
            print("*****");

            ndarray d = np.cumprod(a, axis: 0);
            print(d);
            AssertArray(d, new Complex[,] { { 1, 2, 3 }, { 4, 10, 18 } });
            print("*****");

            ndarray e = np.cumprod(a, axis: 1);
            print(e);
            AssertArray(e, new Complex[,] { { 1, 2, 6 }, { 4, 20, 120 } });
            print("*****");

        }

        [TestMethod]
        public void test_cumsum_3_COMPLEX()
        {
            ndarray a = np.array(new Complex[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, new Complex(11, 0.25), new Complex(12, 0.5) }).reshape(new shape(2, 3, -1));
            print(a);
            print("*****");

            ndarray b = np.cumsum(a);
            print(b);
            AssertArray(b, new Complex[] { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, new Complex(66, 0.25), new Complex(78, 0.75) });
            print("*****");

            ndarray c = np.cumsum(a, dtype: np.Complex);     // specifies type of output value(s)
            print(c);
            AssertArray(c, new Complex[] { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, new Complex(66, 0.25), new Complex(78, 0.75) });
            print("*****");

            ndarray d = np.cumsum(a, axis: 0);     // sum over rows for each of the 3 columns
            print(d);

            var ExpectedDataD = new Complex[,,]
            {{{1,  2},
              {3,  4},
              {5,  6}},

             {{ 8, 10},
              {12, 14},
              {new Complex(16, 0.25), new Complex(18, 0.5) }}};

            AssertArray(d, ExpectedDataD);
            print("*****");



            ndarray e = np.cumsum(a, axis: 1);    // sum over columns for each of the 2 rows
            print(e);

            var ExpectedDataE = new Complex[,,]
            {{{1,  2},
              {4,  6},
              {9,  12}},

             {{ 7, 8},
              {16, 18},
              {new Complex(27, 0.25), new Complex(30, 0.5)}}};

            AssertArray(e, ExpectedDataE);
            print("*****");

            ndarray f = np.cumsum(a, axis: 2);    // sum over columns for each of the 2 rows
            print(f);

            var ExpectedDataF = new Complex[,,]
            {{{1,  3},
              {3,  7},
              {5,  11}},

             {{7, 15},
              {9, 19},
              {new Complex(11, 0.25), new Complex(23, 0.75)}}};

            AssertArray(f, ExpectedDataF);
            print("*****");

        }

        [TestMethod]
        public void test_diff_3_COMPLEX()
        {
            Complex[] TestData = new Complex[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.Complex).reshape(new shape(3, 2, -1));
            x = x * 3;
            var y = np.diff(x, axis: 2);

            print(x);
            print(y);

            var ExpectedData = new Complex[,,]
                {
                 {{15},
                  {60}},

                 {{36},
                  {15}},

                 {{60},
                  {36}}
                };

            AssertArray(y, ExpectedData);

        }

        [TestMethod]
        public void test_ediff1d_1_COMPLEX()
        {
            ndarray x = np.array(new Complex[] { 1, 2, 4, 7, 0 });
            ndarray y = np.ediff1d(x);
            print(y);
            AssertArray(y, new Complex[] { 1, 2, 3, -7 });

            y = np.ediff1d(x, to_begin: np.array(new Complex[] { -99 }), to_end: np.array(new Complex[] { 88, 99 }));
            print(y);
            AssertArray(y, new Complex[] { -99, 1, 2, 3, -7, 88, 99 });

            x = np.array(new Complex[,] { { 1, 2, 4 }, { 1, 6, 24 } });
            y = np.ediff1d(x);
            print(y);
            AssertArray(y, new Complex[] { 1, 2, -3, 5, 18 });

        }

        [TestMethod]
        public void test_gradient_1_COMPLEX()
        {
            var f = np.array(new Complex[] { 1, 2, 4, 7, 11, 16 }, dtype: np.Complex);
            var a = np.gradient(f);
            AssertArray(a[0], new Complex[] { 1, 1.5, 2.5, 3.5, 4.5, 5 });
            print(a[0]);
            print("***********");

            var b = np.gradient(f, new object[] { 2 });
            AssertArray(b[0], new Complex[] { 0.5, 0.75, 1.25, 1.75, 2.25, 2.5 });
            print(b[0]);
            print("***********");

            // Spacing can be also specified with an array that represents the coordinates
            // of the values F along the dimensions.
            // For instance a uniform spacing:

            var x = np.arange(f.size);
            var c = np.gradient(f, new object[] { x });
            AssertArray(c[0], new Complex[] { 1.0, 1.5, 2.5, 3.5, 4.5, 5.0 });
            print(c[0]);
            print("***********");

            // Or a non uniform one:

            x = np.array(new Complex[] { 0.0, 1.0, 1.5, 3.5, 4.0, 6.0 }, dtype: np.Complex);
            var d = np.gradient(f, new object[] { x });
            AssertArray(d[0], new Complex[] { 1.0, 02.99999999999999999999999999990, 3.5, 6.7, 6.9, 2.5 });
            print(d[0]);
        }

        [TestMethod]
        public void test_cross_2_COMPLEX()
        {
            // Multiple vector cross-products. Note that the direction of the cross
            // product vector is defined by the `right-hand rule`.

            var x = np.array(new Complex[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var y = np.array(new Complex[,] { { 4, 5, 6 }, { 1, 2, 3 } });
            var a = np.cross(x, y);
            AssertArray(a, new Complex[,] { { -3, 6, -3 }, { 3, -6, 3 } });
            print(a);


            // The orientation of `c` can be changed using the `axisc` keyword.

            var b = np.cross(x, y, axisc: 0);
            AssertArray(b, new Complex[,] { { -3, 3 }, { 6, -6 }, { -3, 3 } });
            print(b);

            // Change the vector definition of `x` and `y` using `axisa` and `axisb`.

            x = np.array(new Complex[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
            y = np.array(new Complex[,] { { 7, 8, 9 }, { 4, 5, 6 }, { 1, 2, 3 } });
            a = np.cross(x, y);
            AssertArray(a, new Complex[,] { { -6, 12, -6 }, { 0, 0, 0 }, { 6, -12, 6 } });
            print(a);

            b = np.cross(x, y, axisa: 0, axisb: 0);
            AssertArray(b, new Complex[,] { { -24, 48, -24 }, { -30, 60, -30 }, { -36, 72, -36 } });
            print(b);

            return;
        }

        [TestMethod]
        public void test_trapz_1_COMPLEX()
        {
            var a = np.trapz(new Complex[] { 1, 2, 3 });
            Assert.AreEqual((Complex)4.0, a.GetItem(0));
            print(a);

            var b = np.trapz(new Complex[] { 1, 2, 3 }, x: new int[] { 4, 6, 8 });
            Assert.AreEqual((Complex)8.0, b.GetItem(0));
            print(b);

            var c = np.trapz(new Complex[] { 1, 2, 3 }, dx: 2);
            Assert.AreEqual((Complex)8.0, c.GetItem(0));
            print(c);

            a = np.arange(6, dtype: np.Complex).reshape((2, 3));
            b = np.trapz(a, axis: 0);
            AssertArray(b, new Complex[] { 1.5, 2.5, 3.5 });
            print(b);

            c = np.trapz(a, axis: 1);
            AssertArray(c, new Complex[] { 2.0, 8.0 });
            print(c);
        }

        [TestMethod]
        public void test_exp_1_COMPLEX()
        {
            var x = np.array(new Complex[] { -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, -4.2 });
            var a = np.exp(x);
            AssertArray(a, new Complex[] { 0.182683524052735, 0.22313016014843, 0.818730753077982, 1.22140275816017,
                                          4.48168907033806, 5.4739473917272, 7.38905609893065, 0.0149955768204777 });
            print(a);


            a = np.exp(x.reshape((2, -1)));
            AssertArray(a, new Complex[,] { {0.182683524052735, 0.22313016014843, 0.818730753077982, 1.22140275816017 },
                                           {4.48168907033806, 5.4739473917272, 7.38905609893065, 0.0149955768204777  } });
            print(a);

            a = np.exp(x, where: x > 0);
            //AssertArray(a, new Complex[] { double.NaN, double.NaN, double.NaN, 1.22140275816017,
            //                              4.48168907033806, 5.4739473917272, 7.38905609893065, double.NaN });
            print(a);

        }

        [TestMethod]
        public void test_exp2_1_COMPLEX()
        {
            var x = np.array(new Complex[] { -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, -4.2 });
            var a = np.exp2(x);
            AssertArray(a, new Complex[] { 0.307786103336229, 0.353553390593274, 0.870550563296124, 1.14869835499704,
                                          2.82842712474619,  3.24900958542494,  4.0,               0.0544094102060078 });
            print(a);


            a = np.exp2(x.reshape((2, -1)));
            AssertArray(a, new Complex[,] { {0.307786103336229, 0.353553390593274, 0.870550563296124, 1.14869835499704, },
                                           {2.82842712474619,  3.24900958542494,  4.0,               0.0544094102060078  } });
            print(a);

            a = np.exp2(x, where: x > 0);
            //AssertArray(a, new Complex[] { double.NaN, double.NaN, double.NaN, 1.14869835499704,
            //                              2.82842712474619,  3.24900958542494,  4.0, double.NaN });
            print(a);

        }

        [TestMethod]
        public void test_i0_1_COMPLEX()
        {
            var a = np.i0((Complex)5.0);
            //Assert.AreEqual(new Complex(27.2398718236044, 0), a.GetItem(0));
            print(a);

            a = np.i0(new Complex[] { 5.0, 6.0 });
            AssertArray(a, new Complex[] { 27.2398718236044, 67.234406976478 });
            print(a);

            a = np.i0(new Complex[,] { { 27.2398718236044, 67.234406976478 }, { 389.40628328, 427.56411572 } });
            AssertArray(a, new Complex[,] { { 51935526724.2882, 7.7171998335650329E+27 }, { 2.6475747102348978E+167, 9.4248115430920975E+183 } });
            print(a);

            return;

        }

        [TestMethod]
        public void test_sinc_1_COMPLEX()
        {
            Complex retstep = 0;
            var x = np.linspace(-4, 4, ref retstep, 10, dtype: np.Complex);
            var a = np.sinc(x);
            AssertArray(a, new Complex[] {-3.89817183e-17, -3.49934120e-02,  9.20725429e-02, -2.06748336e-01, 7.05316598e-01,
                                          7.05316598e-01, -2.06748336e-01,  9.20725429e-02, -3.49934120e-02, -3.89817183e-17 });
            print(a);

            print("********");

            var xx = np.outer(x, x);
            var b = np.sinc(xx);

            var ExpectedDataB = new Complex[,]

                {{-3.89817183e-17,  2.51898785e-02,  1.22476942e-02, -5.16870839e-02, -1.15090679e-01,
                  -1.15090679e-01, -5.16870839e-02,  1.22476942e-02,  2.51898785e-02, -3.89817183e-17},
                 { 2.51898785e-02, -2.78216241e-02,  1.23470027e-02,  3.44387931e-02, -2.14755666e-01,
                  -2.14755666e-01,  3.44387931e-02,  1.23470027e-02, -2.78216241e-02,  2.51898785e-02},
                 { 1.22476942e-02,  1.23470027e-02,  1.24217991e-02,  1.24718138e-02,  1.24968663e-02,
                   1.24968663e-02,  1.24718138e-02,  1.24217991e-02,  1.23470027e-02,  1.22476942e-02},
                 {-5.16870839e-02,  3.44387931e-02,  1.24718138e-02, -1.15090679e-01,  5.14582086e-01,
                   5.14582086e-01, -1.15090679e-01,  1.24718138e-02,  3.44387931e-02, -5.16870839e-02},
                 {-1.15090679e-01, -2.14755666e-01,  1.24968663e-02,  5.14582086e-01,  9.37041792e-01,
                   9.37041792e-01,  5.14582086e-01,  1.24968663e-02, -2.14755666e-01, -1.15090679e-01},
                 {-1.15090679e-01, -2.14755666e-01,  1.24968663e-02,  5.14582086e-01,  9.37041792e-01,
                   9.37041792e-01,  5.14582086e-01,  1.24968663e-02, -2.14755666e-01, -1.15090679e-01},
                 {-5.16870839e-02,  3.44387931e-02,  1.24718138e-02, -1.15090679e-01,  5.14582086e-01,
                   5.14582086e-01, -1.15090679e-01,  1.24718138e-02,  3.44387931e-02, -5.16870839e-02},
                 { 1.22476942e-02,  1.23470027e-02,  1.24217991e-02,  1.24718138e-02,  1.24968663e-02,
                   1.24968663e-02,  1.24718138e-02,  1.24217991e-02,  1.23470027e-02,  1.22476942e-02},
                 { 2.51898785e-02, -2.78216241e-02,  1.23470027e-02,  3.44387931e-02, -2.14755666e-01,
                  -2.14755666e-01,  3.44387931e-02,  1.23470027e-02, -2.78216241e-02,  2.51898785e-02},
                 { -3.89817183e-17,  2.51898785e-02,  1.22476942e-02, -5.16870839e-02, -1.15090679e-01,
                  -1.15090679e-01, -5.16870839e-02,  1.22476942e-02,  2.51898785e-02, -3.89817183e-17} };

            AssertArray(b, ExpectedDataB);

            print(b);

        }

        [TestMethod]
        public void test_signbit_1_COMPLEX()
        {
            var a = np.signbit((Complex)(-1.2));
            Assert.AreEqual(true, a.GetItem(0));
            print(a);

            var b = np.signbit(np.array(new Complex[] { 1, -2.3, 2.1 }));
            AssertArray(b, new bool[] { false, true, false });
            print(b);

            var c = np.signbit(np.array(new Complex[] { +0.0, -0.0 }));  // note: different result than python.  No such thing as -0.0
            AssertArray(c, new bool[] { false, false });
            print(c);

            var f = np.signbit(np.array(new Complex[] { -1, 0, 1 }));
            AssertArray(f, new bool[] { true, false, false });
            print(f);
        }

        [TestMethod]
        public void test_copysign_1_COMPLEX()
        {
            var a = np.copysign((Complex)1.3, (Complex)(-1));
            Assert.AreEqual((Complex)(-1.3), a.GetItem(0));
            print(a);

            var b = np.divide(1, np.copysign((Complex)0, (Complex)1));
            Assert.AreEqual((Complex)0, b.GetItem(0));  // note: python gets a np.inf value here
            print(b);

            var c = 1 / np.copysign((Complex)0, (Complex)(-1));
            Assert.AreEqual((Complex)0, c.GetItem(0));  // note: python gets a -np.inf value here
            print(c);


            var d = np.copysign(new Complex[] { -1, 0, 1 }, (Complex)(-1.1));
            AssertArray(d, new Complex[] { -1, 0, -1 });
            print(d);

            var e = np.copysign(new Complex[] { -1, 0, 1 }, np.arange(3, dtype: np.Complex) - 1);
            AssertArray(e, new Complex[] { -1, 0, 1 });
            print(e);
        }

        [TestMethod]
        public void test_frexp_1_COMPLEX()
        {
            var x = np.arange(9, dtype: np.Complex);
            var results = np.frexp(x);

            AssertArray(results[0], new Complex[] { 0.0, 0.5, 0.5, 0.75, 0.5, 0.625, 0.75, 0.875, 0.5 });
            AssertArray(results[1], new int[] { 0, 1, 2, 2, 3, 3, 3, 3, 4 });

            print(results[0]);
            print(results[1]);

            print("***************");


            x = np.arange(9, dtype: np.Complex).reshape((3, 3));
            results = np.frexp(x, where: x < 5);

            //AssertArray(results[0], new Complex[,] { { 0.0, 0.5, 0.5 }, { 0.75, 0.5, double.NaN }, { double.NaN, double.NaN, double.NaN } });
            AssertArray(results[1], new int[,] { { 0, 1, 2 }, { 2, 3, 0 }, { 0, 0, 0 } });

            print(results[0]);
            print(results[1]);
        }

        [TestMethod]
        public void test_ldexp_1_COMPLEX()
        {
            var a = np.ldexp((Complex)5, np.arange(4, dtype: np.Complex));
            AssertArray(a, new Complex[] { 5.0f, 10.0f, 20.0f, 40.0f });
            print(a);

            var b = np.ldexp(np.arange(4, dtype: np.Complex), (Complex)5);
            AssertArray(b, new Complex[] { 0.0, 32.0, 64.0, 96.0 });
            print(b);
        }

        [TestMethod]
        public void test_lcm_1_COMPLEX()
        {
            int success = 0;
            try
            {
                var a = np.lcm((Complex)12, (Complex)20);
                Assert.AreEqual((Complex)60, a.GetItem(0));
                print(a);
                success++;
            }
            catch
            { }

            try
            {
                var d = np.lcm(np.arange(6, dtype: np.Complex), new Complex[] { 20 });
                AssertArray(d, new Complex[] { 0, 20, 20, 60, 20, 20 });
                print(d);
                success++;
            }
            catch
            { }

            try
            {

            }
            catch
            {
                var e = np.lcm(new Complex[] { 20, 21 }, np.arange(6, dtype: np.Complex).reshape((3, 2)));
                AssertArray(e, new Complex[,] { { 0, 21 }, { 20, 21 }, { 20, 105 } });
                print(e);
                success++;
            }

            Assert.AreEqual(0, success, "Did not catch all exceptions as expected");
        }

        [TestMethod]
        public void test_gcd_1_COMPLEX()
        {
            int success = 0;

            try
            {
                var a = np.gcd((Complex)12, (Complex)20);
                Assert.AreEqual((Complex)4, a.GetItem(0));
                print(a);
                success++;
            }
            catch
            { }

            try
            {
                var d = np.gcd(np.arange(6, dtype: np.Complex), new Complex[] { 20 });
                AssertArray(d, new int[] { 20, 1, 2, 1, 4, 5 });
                print(d);
                success++;
            }
            catch
            { }

            try
            {
                var e = np.gcd(new Complex[] { 20, 20 }, np.arange(6, dtype: np.Complex).reshape((3, 2)));
                AssertArray(e, new int[,] { { 20, 1 }, { 2, 1 }, { 4, 5 } });
                print(e);
                success++;
            }
            catch
            { }

            try
            {
                var f = np.gcd(new Complex[] { 20, 20 }, np.arange(6, dtype: np.Complex).reshape((3, 2)));
                AssertArray(f, new long[,] { { 20, 1 }, { 2, 1 }, { 4, 5 } });
                print(f);
                success++;
            }
            catch
            { }

            Assert.AreEqual(0, success, "Did not catch all exceptions as expected");
        }

        [TestMethod]
        public void test_add_1_COMPLEX()
        {
            var a = np.add((Complex)1.0, (Complex)4.0);
            Assert.AreEqual((Complex)5.0, a.GetItem(0));
            print(a);

            var b = np.arange((Complex)9.0).reshape((3, 3));
            var c = np.arange((Complex)3.0);
            var d = np.add(b, c);
            AssertArray(d, new Complex[,] { { 0, 2, 4 }, { 3, 5, 7 }, { 6, 8, 10 } });
            print(d);

        }

        [TestMethod]
        public void test_reciprocal_operations_COMPLEX()
        {
            var a = np.arange(1, 32, 1, dtype: np.Complex);
            print(a);

            var b = np.reciprocal(a);
            print(b);

            var ExpectedDataB1 = new Complex[]
            {
               1.0, 0.5, 0.3333333333333333333333333333, 0.25, 0.2, 0.1666666666666666666666666667,
                0.1428571428571428571428571429, 0.125, 0.1111111111111111111111111111, 0.1, 0.0909090909090909090909090909,
                0.0833333333333333333333333333, 0.0769230769230769230769230769, 0.0714285714285714285714285714,
                0.0666666666666666666666666667, 0.0625, 0.0588235294117647058823529412, 0.0555555555555555555555555556,
                0.0526315789473684210526315789, 0.05, 0.0476190476190476190476190476, 0.0454545454545454545454545455,
                0.0434782608695652173913043478, 0.0416666666666666666666666667, 0.04, 0.0384615384615384615384615385,
                0.037037037037037037037037037, 0.0357142857142857142857142857, 0.0344827586206896551724137931,
                0.0333333333333333333333333333, 0.0322580645161290322580645161
            };

            AssertArray(b, ExpectedDataB1);


            a = np.arange(2048, 2048 + 32, 1, dtype: np.Complex);
            print(a);

            b = np.reciprocal(a);
            print(b);

            var ExpectedDataB2 = new Complex[]
            {
                0.00048828125, 0.0004880429477794045876037091, 0.000487804878048780487804878, 0.0004875670404680643588493418,
                0.0004873294346978557504873294, 0.0004870920603994154895275207, 0.0004868549172346640701071081,
                0.0004866180048661800486618005, 0.0004863813229571984435797665, 0.000486144871171609139523578,
                0.000485908649173955296404276, 0.0004856726566294317629917436, 0.0004854368932038834951456311,
                0.0004852013585638039786511402, 0.0004849660523763336566440349, 0.0004847309743092583616093068,
                0.0004844961240310077519379845, 0.0004842615012106537530266344, 0.0004840271055179090029041626,
                0.0004837929366231253023705854, 0.0004835589941972920696324952, 0.0004833252779120347994200097,
                0.0004830917874396135265700483, 0.0004828585224529212940608402, 0.0004826254826254826254826255,
                0.0004823926676314520019295707, 0.0004821600771456123432979749, 0.0004819277108433734939759036,
                0.0004816955684007707129094412, 0.0004814636494944631680308137, 0.0004812319538017324350336862,
                0.0004810004810004810004810005
            };
            AssertArray(b, ExpectedDataB2);
        }

        [TestMethod]
        public void test_positive_1_COMPLEX()
        {
            var d = np.positive(new Complex[] { -1, -0, 1 });
            AssertArray(d, new Complex[] { -1, -0, 1 });
            print(d);

            var e = np.positive(new Complex[,] { { 1, 0, -1 }, { -2, 3, -4 } });
            AssertArray(e, new Complex[,] { { 1, 0, -1 }, { -2, 3, -4 } });
            print(e);
        }

        [TestMethod]
        public void test_negative_1_COMPLEX()
        {
            var a = np.array(new Complex[] { -1, -0, 1 });
            var d = np.negative(a);
            AssertArray(d, new Complex[] { 1, 0, -1 });
            print(d);

            var e = np.negative(new Complex[,] { { 1, 0, -1 }, { -2, 3, -4 } });
            AssertArray(e, new Complex[,] { { -1, 0, 1 }, { 2, -3, 4 } });
            print(e);
        }

        [TestMethod]
        public void test_multiply_1_COMPLEX()
        {
            var a = np.multiply((Complex)2.0, (Complex)4.0);
            Assert.AreEqual((Complex)8.0, a.GetItem(0));
            print(a);

            var b = np.arange((Complex)9.0).reshape((3, 3));
            var c = np.arange((Complex)3.0);
            var d = np.multiply(b, c);
            AssertArray(d, new Complex[,] { { 0, 1, 4 }, { 0, 4, 10 }, { 0, 7, 16 } });
            print(d);
        }

        [TestMethod]
        public void test_divide_COMPLEX()
        {
            var a = np.divide((Complex)7, (Complex)3);
            Assert.AreEqual((Complex)2.3333333333333333333333333333, a.GetItem(0));
            print(a);

            var b = np.divide(new Complex[] { 1.0, 2.0, 3.0, 4.0 }, 2.5m);
            AssertArray(b, new Complex[] { 0.4, 0.8, 1.2, 1.6 });
            print(b);

            var c = np.divide(new Complex[] { 1.0, 2.0, 3.0, 4.0 }, new Complex[] { 0.5, 2.5, 2.5, 3.5 });
            AssertArray(c, new Complex[] { 2.0, 0.8, 1.2, 1.1428571428571428571428571429 });
            print(c);

            return;
        }

        [TestMethod]
        public void test_power_operations_COMPLEX()
        {
            var a = np.arange(0, 32, 1, dtype: np.Complex);
            print(a);

            var b = np.power(a, (Complex)3.23);
            print(b);

            var ExpectedDataB1 = new Complex[]
            { 0.0, 1.0, 9.38267959385503, 34.7617516700826, 88.0346763609436, 180.997724101542,
              326.15837804154, 536.619770563306, 826.001161443457, 1208.37937917249, 1698.24365246174,
              2310.45956851781, 3060.23955801521, 3963.11822364251, 5034.9313709235, 6291.79793806794,
              7750.10424197608, 9426.49010646868, 11337.836542597, 13501.2547250997, 15934.0760633466,
              18653.8432055784, 21678.3018459592, 25025.3932276144, 28713.2472532973, 32760.176129938,
              37184.6684850056, 42005.3839020428, 47241.1478304245, 52910.9468307066, 59033.9241221692,
              65629.3754035258
            };

            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Complex);
            print(a);

            b = np.power(a, 4);
            print(b);

            var ExpectedDataB2 = new Complex[]
            {
             17592186044416, 17626570956801, 17661006250000, 17695491973201,
             17730028175616, 17764614906481, 17799252215056, 17833940150625,
             17868678762496, 17903468100001, 17938308212496, 17973199149361,
             18008140960000, 18043133693841, 18078177400336, 18113272128961,
             18148417929216, 18183614850625, 18218862942736, 18254162255121,
             18289512837376, 18324914739121, 18360368010000, 18395872699681,
             18431428857856, 18467036534241, 18502695778576, 18538406640625,
             18574169170176, 18609983417041, 18645849431056, 18681767262081
            };

            AssertArray(b, ExpectedDataB2);

            b = np.power(a, (Complex)0);
            print(b);
            var ExpectedDataB3 = new Complex[]
            {
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            };
            AssertArray(b, ExpectedDataB3);


            b = np.power(a, (Complex)0.5);
            print(b);

            var ExpectedDataB4 = new Complex[]
            {
                45.254833995939, 45.2658811910251, 45.2769256906871, 45.287967496897, 45.2990066116245,
                45.3100430368368, 45.3210767744986, 45.3321078265725, 45.3431361950185, 45.3541618817943,
                45.365184888855, 45.3762052181537, 45.3872228716409, 45.3982378512647, 45.4092501589709,
                45.4202597967031, 45.4312667664022, 45.442271070007, 45.453272709454, 45.4642716866772,
                45.4752680036083, 45.4862616621766, 45.4972526643093, 45.508241011931, 45.5192267069642,
                45.5302097513288, 45.5411901469428, 45.5521678957215, 45.5631429995781, 45.5741154604234,
                45.5850852801659, 45.596052460712
            };

            AssertArray(b, ExpectedDataB4);

        }

        [TestMethod]
        public void test_subtract_1_COMPLEX()
        {
            var a = np.subtract((Complex)1.0, (Complex)4.0);
            Assert.AreEqual((Complex)(-3.0), a.GetItem(0));
            print(a);

            var b = np.arange(9.0, dtype: np.Complex).reshape((3, 3));
            var c = np.arange(3.0, dtype: np.Complex);
            var d = np.subtract(b, c);
            AssertArray(d, new Complex[,] { { 0, 0, 0 }, { 3, 3, 3 }, { 6, 6, 6 } });
            print(d);
        }

        [TestMethod]
        public void test_true_divide_COMPLEX()
        {
            var a = np.true_divide((Complex)7, (Complex)3);
            Assert.AreEqual((Complex)2.3333333333333333333333333333, a.GetItem(0));
            print(a);

            var b = np.true_divide(new Complex[] { 1.0, 2.0, 3.0, 4.0 }, (Complex)2.5);
            AssertArray(b, new Complex[] { 0.4, 0.8, 1.2, 1.6 });
            print(b);

            var c = np.true_divide(new Complex[] { 1.0, 2.0, 3.0, 4.0 }, new Complex[] { 0.5, 2.5, 2.5, 3.5 });
            AssertArray(c, new Complex[] { 2.0, 0.8, 1.2, 1.1428571428571428571428571429 });
            print(c);

            return;
        }

        [TestMethod]
        public void test_floor_divide_COMPLEX()
        {
            var a = np.floor_divide((Complex)7, 3);
            Assert.AreEqual((Complex)2.0, a.GetItem(0));
            print(a);

            var b = np.floor_divide(new Complex[] { 1.0, 2.0, 3.0, 4.0 }, (Complex)2.5);
            AssertArray(b, new Complex[] { 0, 0, 1, 1 });
            print(b);

            var c = np.floor_divide(new Complex[] { 1.0, 2.0, 3.0, 4.0 }, new Complex[] { 0.5, 2.5, 2.5, 3.5 });
            AssertArray(c, new Complex[] { 2, 0, 1, 1 });
            print(c);

            return;

        }

        [TestMethod]
        public void test_float_power_COMPLEX()
        {
            var x1 = new Complex[] { 0, 1, 2, 3, 4, 5 };

            var a = np.float_power(x1, (Complex)3);
            AssertArray(a, new Complex[] { 0.0, 1.0, 8.0, 27.0, 64.0, 125.0 });
            print(a);

            var x2 = new Complex[] { 1.0, 2.0, 3.0, 3.0, 2.0, 1.0 };
            var b = np.float_power(x1, x2);
            AssertArray(b, new Complex[] { 0.0, 1.0, 8.0, 27.0, 16.0, 5.0 });
            print(b);

            var x3 = np.array(new Complex[,] { { 1, 2, 3, 3, 2, 1 }, { 1, 2, 3, 3, 2, 1 } });
            var c = np.float_power(x1, x3);
            AssertArray(c, new Complex[,] { { 0.0, 1.0, 8.0, 27.0, 16.0, 5.0 }, { 0.0, 1.0, 8.0, 27.0, 16.0, 5.0 } });
            print(c);

            return;
        }

        [TestMethod]
        public void test_fmod_2_COMPLEX()
        {
            var x = np.fmod(new Complex[] { -4, -7 }, new Complex[] { 2, 3 });
            AssertArray(x, new Complex[] { 0, -1 });
            print(x);

            var y = np.fmod(np.arange(7, dtype: np.Complex), -5);
            AssertArray(y, new Complex[] { 0, 1, 2, 3, 4, 0, 1 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_mod_1_COMPLEX
            ()
        {
            var x = np.mod(new Complex[] { 4, 7 }, new Complex[] { 2, 3 });
            AssertArray(x, new Complex[] { 0, 1 });
            print(x);

            var y = np.mod(np.arange(7, dtype: np.Complex), 5);
            AssertArray(y, new Complex[] { 0, 1, 2, 3, 4, 0, 1 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_modf_1_COMPLEX()
        {
            var x = np.modf(new Complex[] { 0, 3.5 });
            AssertArray(x[0], new Complex[] { 0, 0.5 });
            AssertArray(x[1], new Complex[] { 0, 3.0 });
            print(x);

            var y = np.modf(np.arange(7, dtype: np.Complex));
            AssertArray(y[0], new Complex[] { 0, 0, 0, 0, 0, 0, 0 });
            AssertArray(y[1], new Complex[] { 0, 1, 2, 3, 4, 5, 6 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_remainder_2_COMPLEX()
        {
            var x = np.remainder(new Complex[] { -4, -7 }, new Complex[] { 2, 3 });
            AssertArray(x, new Complex[] { 0, 2 });
            print(x);

            var y = np.remainder(np.arange(7, dtype: np.Complex), -5);
            AssertArray(y, new Complex[] { 0, -4, -3, -2, -1, 0, -4 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_divmod_1_COMPLEX()
        {
            var a = np.divmod((Complex)7, (Complex)3);
            Assert.AreEqual((Complex)2, a[0].GetItem(0));
            Assert.AreEqual((Complex)1, a[1].GetItem(0));

            print(a);

            var b = np.divmod(new Complex[] { 1.0, 2.0, 3.0, 4.0 }, 2.5m);
            AssertArray(b[0], new Complex[] { 0, 0, 1, 1 });
            AssertArray(b[1], new Complex[] { 1, 2, 0.5, 1.5 });
            print(b);

            var c = np.divmod(new Complex[] { 1.0, 2.0, 3.0, 4.0 }, new Complex[] { 0.5, 2.5, 2.5, 3.5 });
            AssertArray(c[0], new Complex[] { 2, 0, 1, 1 });
            AssertArray(c[1], new Complex[] { 0, 2, 0.5, 0.5 });
            print(c);

            return;

        }

        [TestMethod]
        public void test_convolve_1_COMPLEX()
        {
            var a = np.convolve(new Complex[] { 1, 2, 3 }, new Complex[] { 0, 1, 0.5f });
            AssertArray(a, new Complex[] { 0.0, 1.0, 2.5, 4.0, 1.5 });
            print(a);

            var b = np.convolve(new Complex[] { 1, 2, 3 }, new Complex[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_SAME);
            AssertArray(b, new Complex[] { 1.0, 2.5, 4.0 });
            print(b);

            var c = np.convolve(new Complex[] { 1, 2, 3 }, new Complex[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_VALID);
            AssertArray(c, new Complex[] { 2.5 });
            print(c);

            return;
        }

        [TestMethod]
        public void test_clip_2_COMPLEX()
        {
            ndarray a = np.arange(16, dtype: np.Complex).reshape(new shape(4, 4));
            print(a);
            print("*****");

            ndarray b = np.clip(a, 1, 8);
            print(b);
            print("*****");
            AssertArray(b, new Complex[,] { { 1, 1, 2, 3 }, { 4, 5, 6, 7 }, { 8, 8, 8, 8 }, { 8, 8, 8, 8 } });

            ndarray c = np.clip(a, 3, 6, @out: a);
            print(c);
            AssertArray(c, new Complex[,] { { 3, 3, 3, 3 }, { 4, 5, 6, 6 }, { 6, 6, 6, 6 }, { 6, 6, 6, 6 } });
            print(a);
            AssertArray(a, new Complex[,] { { 3, 3, 3, 3 }, { 4, 5, 6, 6 }, { 6, 6, 6, 6 }, { 6, 6, 6, 6 } });
            print("*****");

            a = np.arange(16, dtype: np.Complex).reshape(new shape(4, 4));
            print(a);
            b = np.clip(a, np.array(new Complex[] { 3, 4, 1, 1 }), 8);
            print(b);
            AssertArray(b, new Complex[,] { { 3, 4, 2, 3 }, { 4, 5, 6, 7 }, { 8, 8, 8, 8 }, { 8, 8, 8, 8 } });

        }

        [TestMethod]
        public void test_square_operations_COMPLEX()
        {
            var a = np.arange(0, 32, 1, dtype: np.Complex);
            print(a);

            var b = np.square(a);
            print(b);

            var ExpectedDataB1 = new Complex[]
            {
                0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144, 169, 196, 225, 256, 289,
                324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961
            };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Complex);
            print(a);

            b = np.square(a);
            print(b);

            var ExpectedDataB2 = new Complex[]
            {
                4194304, 4198401, 4202500, 4206601, 4210704, 4214809, 4218916, 4223025, 4227136,
                4231249, 4235364, 4239481, 4243600, 4247721, 4251844, 4255969, 4260096, 4264225,
                4268356, 4272489, 4276624, 4280761, 4284900, 4289041, 4293184, 4297329, 4301476,
                4305625, 4309776, 4313929, 4318084, 4322241
            };
            AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_absolute_operations_COMPLEX()
        {
            var a = np.arange(-32, 32, 1, dtype: np.Complex);
            print(a);

            var b = np.absolute(a);
            print(b);

            var ExpectedDataB = new Complex[]
            {
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,
                2,  1,  0,  1,  2,  3,  4,  5,   6,  7,  8,  9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31
            };
        }

        [TestMethod]
        public void test_fabs_1_COMPLEX()
        {
            var a = np.arange(-32, 32, 1, dtype: np.Complex);
            print(a);

            var b = np.fabs(a);
            print(b);

            var ExpectedDataB = new Complex[]
            {
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,
                2,  1,  0,  1,  2,  3,  4,  5,   6,  7,  8,  9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31
            };
        }

        [TestMethod]
        public void test_sign_1_COMPLEX()
        {
            var a = np.sign(new Complex(-1.2, 0));
            Assert.AreEqual((Complex)(-1.0), a.GetItem(0));
            print(a);

            var b = np.sign(np.array(new Complex[] { 1, -2.3, 2.1 }));
            AssertArray(b, new Complex[] { 1, -1, 1 });
            print(b);

            var c = np.sign(np.array(new Complex[] { +0.0, -0.0 }));
            AssertArray(c, new Complex[] { 0, 0 });
            print(c);


            var f = np.sign(np.array(new Complex[] { -1, 0, 1 }));
            AssertArray(f, new Complex[] { -1, 0, 1 });
            print(f);
        }

        [TestMethod]
        public void test_heaviside_1_COMPLEX()
        {
            var a = np.heaviside(new Complex[] { -1.5, 0.0, 2.0 }, 0.5m);
            AssertArray(a, new Complex[] { 0.0, 0.5, 1.0 });
            print(a);

            var b = np.heaviside(new Complex[] { -1.5, 0, 2.0 }, 1);
            AssertArray(b, new Complex[] { 0.0, 1.0, 1.0 });
            print(b);

            var c = np.heaviside(new Complex[] { -1, 0, 2 }, 1);
            AssertArray(c, new Complex[] { 0, 1, 1 });
            print(c);

        }

        [TestMethod]
        public void test_maximum_1_COMPLEX()
        {
            var a = np.maximum(new Complex[] { 2, 3, 4 }, new Complex[] { 1, 5, 2 });
            AssertArray(a, new Complex[] { 2, 5, 4 });
            print(a);

            var b = np.maximum(np.eye(2, dtype: np.Complex), new Complex[] { 0.5, 2 }); // broadcasting
            AssertArray(b, new Complex[,] { { 1, 2 }, { 0.5, 2.0 } });
            print(b);

            //var c = np.maximum(new float[] { float.NaN, 0, float.NaN }, new float[] { 0, float.NaN, float.NaN });
            //AssertArray(c, new double[] { double.NaN, double.NaN, double.NaN });
            //print(c);

            //var d = np.maximum(double.PositiveInfinity, 1);
            //Assert.AreEqual(double.PositiveInfinity, d.GetItem(0));
            //print(d);
        }

        [TestMethod]
        public void test_minimum_1_COMPLEX()
        {
            var a = np.minimum(new Complex[] { 2, 3, 4 }, new Complex[] { 1, 5, 2 });
            AssertArray(a, new Complex[] { 1, 3, 2 });
            print(a);

            var b = np.minimum(np.eye(2, dtype: np.Complex), new Complex[] { 0.5, 2 }); // broadcasting
            AssertArray(b, new Complex[,] { { 0.5, 0.0 }, { 0.0, 1.0 } });
            print(b);

            //var c = np.minimum(new float[] { float.NaN, 0, float.NaN }, new float[] { 0, float.NaN, float.NaN });
            //AssertArray(c, new double[] { float.NaN, float.NaN, float.NaN });
            //print(c);

            //var d = np.minimum(double.PositiveInfinity, 1);
            //Assert.AreEqual((double)1, d.GetItem(0));
            //print(d);
        }

        [TestMethod]
        public void test_fmax_1_COMPLEX()
        {
            var a = np.fmax(new Complex[] { 2, 3, 4 }, new Complex[] { 1, 5, 2 });
            AssertArray(a, new Complex[] { 2, 5, 4 });
            print(a);

            var b = np.fmax(np.eye(2, dtype: np.Complex), new Complex[] { 0.5, 2 }); // broadcasting
            AssertArray(b, new Complex[,] { { 1, 2 }, { 0.5, 2.0 } });
            print(b);

            //var c = np.fmax(new float[] { float.NaN, 0, float.NaN }, new float[] { 0, float.NaN, float.NaN });
            //AssertArray(c, new double[] { 0.0, 0.0, double.NaN });
            //print(c);

            //var d = np.fmax(double.PositiveInfinity, 1);
            //Assert.AreEqual(double.PositiveInfinity, d.GetItem(0));
            //print(d);
        }

        [TestMethod]
        public void test_fmin_1_COMPLEX()
        {
            var a = np.fmin(new Complex[] { 2, 3, 4 }, new Complex[] { 1, 5, 2 });
            AssertArray(a, new Complex[] { 1, 3, 2 });
            print(a);

            var b = np.fmin(np.eye(2, dtype: np.Complex), new Complex[] { 0.5, 2 }); // broadcasting
            AssertArray(b, new Complex[,] { { 0.5, 0.0 }, { 0.0, 1.0 } });
            print(b);

            //var c = np.fmin(new float[] { float.NaN, 0, float.NaN }, new float[] { 0, float.NaN, float.NaN });
            //AssertArray(c, new double[] { 0.0, 0.0, double.NaN });
            //print(c);

            //var d = np.fmin(double.PositiveInfinity, 1);
            //Assert.AreEqual((double)1, d.GetItem(0));
            //print(d);
        }

        [TestMethod]
        public void test_nan_to_num_1_COMPLEX()
        {
            Complex a1 = (Complex)np.nan_to_num((Complex)2.0);
            Assert.AreEqual(a1, (Complex)2.0);
            print(a1);

            ndarray x = np.array(new Complex[] { 1.0, 2.0, 3.0, -128, 128 });
            ndarray d = np.nan_to_num(x);
            AssertArray(d, new Complex[] { 1.0, 2.0, 3.0, -128, 128 });
            print(d);

        }

        #endregion

        #region from FromNumericTests

        [TestMethod]
        public void test_take_1_COMPLEX()
        {
            var a = np.array(new Complex[] { 4, 3, 5, 7, 6, 8, 9, 12, 14, 16, 18, 20, 22, 24, 26, 28 });
            var indices = np.array(new Int32[] { 0, 1, 4 });
            ndarray b = np.take(a, indices);
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new Complex[] { 4, 3, 6 });
            AssertShape(b, 3);
            AssertStrides(b, SizeofComplex);


            a = np.array(new Complex[] { 4, 3, 5, 7, 6, 8, 9, 12, 14, 16, 18, 20, 22, 24, 26, 28 });
            indices = np.array(new Int32[,] { { 0, 1 }, { 2, 3 } });
            ndarray c = np.take(a, indices);
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            var ExpectedDataC = new Complex[2, 2]
            {
                { 4, 3 },
                { 5, 7 },
            };
            AssertArray(c, ExpectedDataC);
            AssertShape(c, 2, 2);
            AssertStrides(c, SizeofComplex * 2, SizeofComplex);

            ndarray d = np.take(a.reshape(new shape(4, -1)), indices, axis: 0);
            print("D");
            print(d);
            print(d.shape);
            print(d.strides);

            var ExpectedDataD = new Complex[2, 2, 4]
            {
                {
                    { 4, 3, 5, 7 },
                    { 6, 8, 9, 12 },
                },
                {
                    { 14, 16, 18, 20 },
                    { 22, 24, 26, 28 },
                },

            };
            AssertArray(d, ExpectedDataD);
            AssertShape(d, 2, 2, 4);
            AssertStrides(d, SizeofComplex * 8, SizeofComplex * 4, SizeofComplex * 1);

            ndarray e = np.take(a.reshape(new shape(4, -1)), indices, axis: 1);
            print("E");
            print(e);
            print(e.shape);
            print(e.strides);

            var ExpectedDataE = new Complex[4, 2, 2]
            {
                {
                    { 4, 3 },
                    { 5, 7 },
                },
                {
                    { 6, 8 },
                    { 9, 12 },
                },
                {
                    { 14, 16 },
                    { 18, 20 },
                },
                {
                    { 22, 24 },
                    { 26, 28 },
                },

            };

            AssertArray(e, ExpectedDataE);
            AssertShape(e, 4, 2, 2);
            AssertStrides(e, SizeofComplex * 4, SizeofComplex * 2, SizeofComplex * 1);

        }

        [TestMethod]
        public void test_ravel_1_COMPLEX()
        {
            var a = np.array(new Complex[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var b = np.ravel(a);
            AssertArray(b, new Complex[] { 1, 2, 3, 4, 5, 6 });
            print(b);

            var c = a.reshape(-1);
            AssertArray(c, new Complex[] { 1, 2, 3, 4, 5, 6 });
            print(c);

            var d = np.ravel(a, order: NPY_ORDER.NPY_FORTRANORDER);
            AssertArray(d, new Complex[] { 1, 4, 2, 5, 3, 6 });
            print(d);

            // When order is 'A', it will preserve the array's 'C' or 'F' ordering:
            var e = np.ravel(a.T);
            AssertArray(e, new Complex[] { 1, 4, 2, 5, 3, 6 });
            print(e);

            var f = np.ravel(a.T, order: NPY_ORDER.NPY_ANYORDER);
            AssertArray(f, new Complex[] { 1, 2, 3, 4, 5, 6 });
            print(f);
        }

        [TestMethod]
        public void test_choose_1_COMPLEX()
        {
            ndarray choice1 = np.array(new Complex[] { 0, 1, 2, 3 });
            ndarray choice2 = np.array(new Complex[] { 10, 11, 12, 13 });
            ndarray choice3 = np.array(new Complex[] { 20, 21, 22, 23 });
            ndarray choice4 = np.array(new Complex[] { 30, 31, 32, 33 });

            ndarray[] choices = new ndarray[] { choice1, choice2, choice3, choice4 };

            ndarray a = np.choose(np.array(new Int32[] { 2, 3, 1, 0 }), choices);

            print(a);

            AssertArray(a, new Complex[] { 20, 31, 12, 3 });
        }

        [TestMethod]
        public void test_choose_2_COMPLEX()
        {
            ndarray choice1 = np.array(new Complex[] { 0, 1, 2, 3 });
            ndarray choice2 = np.array(new Complex[] { 10, 11, 12, 13 });
            ndarray choice3 = np.array(new Complex[] { 20, 21, 22, 23 });
            ndarray choice4 = np.array(new Complex[] { 30, 31, 32, 33 });

            ndarray[] choices = new ndarray[] { choice1, choice2, choice3, choice4 };

            ndarray a = np.choose(np.array(new Int32[] { 2, 4, 1, 0 }), choices, mode: NPY_CLIPMODE.NPY_CLIP);
            print(a);
            AssertArray(a, new Complex[] { 20, 31, 12, 3 });

            a = np.choose(np.array(new Int32[] { 2, 4, 1, 0 }), choices, mode: NPY_CLIPMODE.NPY_WRAP);
            print(a);
            AssertArray(a, new Complex[] { 20, 1, 12, 3 });

            try
            {
                a = np.choose(np.array(new Int32[] { 2, 4, 1, 0 }), choices, mode: NPY_CLIPMODE.NPY_RAISE);
                print(a);
                AssertArray(a, new Complex[] { 20, 1, 12, 3 });
            }
            catch (Exception ex)
            {
                if (ex.Message.Contains("invalid entry in choice array"))
                    return;
            }
            Assert.Fail("Should have caught exception from np.choose");


        }

        [TestMethod]
        public void test_select_1_COMPLEX()
        {
            var x = np.arange(10, dtype: np.Complex);
            var condlist = new ndarray[] { x < 3, x > 5 };
            var choicelist = new ndarray[] { x, np.array(np.power(x, 2), dtype: np.Complex) };
            var y = np.select(condlist, choicelist);

            AssertArray(y, new Complex[] { 0, 1, 2, 0, 0, 0, 36, 49, 64, 81 });
            print(y);
        }

        [TestMethod]
        public void test_repeat_1_COMPLEX()
        {
            ndarray x = np.array(new Complex[] { 1, 2, 3, 4 }).reshape(new shape(2, 2));
            var y = new Int32[] { 2 };

            ndarray z = np.repeat(x, y);
            print(z);
            print("");
            AssertArray(z, new Complex[] { 1, 1, 2, 2, 3, 3, 4, 4 });

            z = np.repeat((Complex)3, 4);
            print(z);
            print("");
            AssertArray(z, new Complex[] { 3, 3, 3, 3 });

            z = np.repeat(x, 3, axis: 0);
            print(z);
            print("");

            var ExpectedData1 = new Complex[6, 2]
            {
                { 1, 2 },
                { 1, 2 },
                { 1, 2 },
                { 3, 4 },
                { 3, 4 },
                { 3, 4 },
            };

            AssertArray(z, ExpectedData1);
            AssertShape(z, 6, 2);

            z = np.repeat(x, 3, axis: 1);
            print(z);
            print("");

            var ExpectedData2 = new Complex[2, 6]
            {
                { 1, 1, 1, 2, 2, 2 },
                { 3, 3, 3, 4, 4, 4 },
            };

            AssertArray(z, ExpectedData2);
            AssertShape(z, 2, 6);



            z = np.repeat(x, new Int32[] { 1, 2 }, axis: 0);
            print(z);

            var ExpectedData3 = new Complex[3, 2]
            {
                { 1, 2 },
                { 3, 4 },
                { 3, 4 },
            };

            AssertArray(z, ExpectedData3);
            AssertShape(z, 3, 2);
        }

        [TestMethod]
        public void test_put_1_COMPLEX()
        {
            ndarray a = np.arange(5, dtype: np.Complex);
            np.put(a, new int[] { 0, 2 }, new int[] { -44, -55 });
            print(a);
            AssertArray(a, new Complex[] { -44, 1, -55, 3, 4 });

            a = np.arange(5, dtype: np.Complex);
            np.put(a, 22, -5, mode: NPY_CLIPMODE.NPY_CLIP);
            print(a);
            AssertArray(a, new Complex[] { 0, 1, 2, 3, -5 });

            a = np.arange(5, dtype: np.Complex);
            np.put(a, 22, -5, mode: NPY_CLIPMODE.NPY_WRAP);
            print(a);
            AssertArray(a, new Complex[] { 0, 1, -5, 3, 4 });

            try
            {
                a = np.arange(5, dtype: np.Complex);
                np.put(a, 22, -5, mode: NPY_CLIPMODE.NPY_RAISE);
                print(a);
            }
            catch (Exception ex)
            {
                return;
            }
            throw new Exception("this should have caught an exception");

        }

        [TestMethod]
        public void test_putmask_1_COMPLEX()
        {
            var x = np.arange(6, dtype: np.Complex).reshape((2, 3));
            np.putmask(x, x > 2, np.power(x, 2).astype(np.Int32));
            AssertArray(x, new Complex[,] { { 0, 1, 2, }, { 9, 16, 25 } });
            print(x);


            // If values is smaller than a it is repeated:

            x = np.arange(5, dtype: np.Complex);
            np.putmask(x, x > 1, new Int32[] { -33, -44 });
            AssertArray(x, new Complex[] { 0, 1, -33, -44, -33 });
            print(x);

            return;
        }

        [TestMethod]
        public void test_swapaxes_1_COMPLEX()
        {
            ndarray x = np.array(new Complex[,] { { 1, 2, 3 } });
            print(x);
            print("********");

            ndarray y = np.swapaxes(x, 0, 1);
            print(y);
            AssertArray(y, new Complex[3, 1] { { 1 }, { 2 }, { 3 } });
            print("********");

            x = np.array(new Complex[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } });
            print(x);

            var ExpectedDataX = new Complex[2, 2, 2]
            {
                {
                    { 0,1 },
                    { 2,3 },
                },
                {
                    { 4,5 },
                    { 6,7 },
                },
            };
            AssertArray(x, ExpectedDataX);

            print("********");

            y = np.swapaxes(x, 0, 2);
            print(y);

            var ExpectedDataY = new Complex[2, 2, 2]
            {
                {
                    { 0,4 },
                    { 2,6 },
                },
                {
                    { 1,5 },
                    { 3,7 },
                },
            };
            AssertArray(y, ExpectedDataY);
        }

        [TestMethod]
        public void test_ndarray_T_1_COMPLEX()
        {
            var x = np.arange(0, 32, dtype: np.Complex).reshape(new shape(8, 4));
            print("X");
            print(x);
            print(x.shape);

            var y = x.T;

            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Complex[4, 8]
            {
                { 0, 4,  8, 12, 16, 20, 24, 28 },
                { 1, 5,  9, 13, 17, 21, 25, 29 },
                { 2, 6, 10, 14, 18, 22, 26, 30 },
                { 3, 7, 11, 15, 19, 23, 27, 31 },
            };

            AssertArray(y, ExpectedDataY);

        }

        [TestMethod]
        public void test_ndarray_transpose_1_COMPLEX()
        {
            var x = np.arange(0, 64, dtype: np.Complex).reshape(new shape(2, 4, -1, 4));
            print("X");
            print(x);
            print(x.shape);

            var y = np.transpose(x, new long[] { 1, 2, 3, 0 });

            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Complex[4, 2, 4, 2]
                {{{ {0, 32},
                    {1, 33},
                    {2, 34},
                    {3, 35}},
                   {{4, 36},
                    {5, 37},
                    {6, 38},
                    {7, 39}}},
                  {{{8, 40},
                    {9, 41},
                    {10, 42},
                    {11, 43}},
                   {{12, 44},
                    {13, 45},
                    {14, 46},
                    {15, 47}}},
                  {{{16, 48},
                    {17, 49},
                    {18, 50},
                    {19, 51}},
                   {{20, 52},
                    {21, 53},
                    {22, 54},
                    {23, 55}}},
                  {{{24, 56},
                    {25, 57},
                    {26, 58},
                    {27, 59}},
                   {{28, 60},
                    {29, 61},
                    {30, 62},
                    {31, 63}}}};

            AssertArray(y, ExpectedDataY);

        }

        [TestMethod]
        public void test_partition_3_COMPLEX()
        {
            var a = np.arange(22, 10, -1, dtype: np.Complex).reshape((3, 4, 1));
            var b = np.partition(a, 1, axis: 0);
            AssertArray(b, new Complex[,,] { { { 14 }, { 13 }, { 12 }, { 11 } }, { { 18 }, { 17 }, { 16 }, { 15 } }, { { 22 }, { 21 }, { 20 }, { 19 } } });
            print(b);

            var c = np.partition(a, 2, axis: 1);
            AssertArray(c, new Complex[,,] { { { 19 }, { 20 }, { 21 }, { 22 } }, { { 15 }, { 16 }, { 17 }, { 18 } }, { { 11 }, { 12 }, { 13 }, { 14 } } });
            print(c);

            var d = np.partition(a, 0, axis: 2);
            AssertArray(d, new Complex[,,] { { { 22 }, { 21 }, { 20 }, { 19 } }, { { 18 }, { 17 }, { 16 }, { 15 } }, { { 14 }, { 13 }, { 12 }, { 11 } } });
            print(d);

            try
            {
                var e = np.partition(a, 4, axis: 1);
                print(e);
            }
            catch (Exception ex)
            {
                return;
            }

            Assert.Fail("Should have caught the exception");

        }

        [TestMethod]
        public void test_argpartition_3_COMPLEX()
        {
            var a = np.arange(22, 10, -1, np.Complex).reshape((3, 4, 1));
            var b = np.argpartition(a, 1, axis: 0);
            AssertArray(b, new npy_intp[,,] { { { 2 }, { 2 }, { 2 }, { 2 } }, { { 1 }, { 1 }, { 1 }, { 1 } }, { { 0 }, { 0 }, { 0 }, { 0 } } });
            print(b);

            var c = np.argpartition(a, 2, axis: 1);
            AssertArray(c, new npy_intp[,,] { { { 3 }, { 2 }, { 1 }, { 0 } }, { { 3 }, { 2 }, { 1 }, { 0 } }, { { 3 }, { 2 }, { 1 }, { 0 } } });
            print(c);

            var d = np.argpartition(a, 0, axis: 2);
            AssertArray(d, new npy_intp[,,] { { { 0 }, { 0 }, { 0 }, { 0 } }, { { 0 }, { 0 }, { 0 }, { 0 } }, { { 0 }, { 0 }, { 0 }, { 0 } } });
            print(d);

            try
            {
                var e = np.partition(a, 4, axis: 1);
                print(e);
            }
            catch (Exception ex)
            {
                return;
            }

            Assert.Fail("Should have caught the exception");

        }

        [TestMethod]
        public void test_sort_2_COMPLEX()
        {
            var InputData = new Complex[]
                {32.2, 31.2, 30.2, 29.2, 28.2, 27.2, 26.2, 25.2, 24.2, 23.2, 22.2, 21.2, 20.2, 19.2, 18.2, 17.2,
                 16.2, 15.2, 14.2, 13.2, 12.2, 11.2, 10.2, 9.2,  8.2,  7.2,  6.2,  5.2,  4.2,  3.2,  2.2,  1.2};

            var a = np.array(InputData).reshape(new shape(8, 4));
            ndarray b = np.sort(a);                 // sort along the last axis
            print(b);

            var ExpectedDataB = new Complex[8, 4]
            {
             {29.2, 30.2, 31.2, 32.2},
             {25.2, 26.2, 27.2, 28.2},
             {21.2, 22.2, 23.2, 24.2},
             {17.2, 18.2, 19.2, 20.2},
             {13.2, 14.2, 15.2, 16.2},
             {9.2, 10.2, 11.2, 12.2},
             {5.2,  6.2,  7.2,  8.2},
             {1.2,  2.2,  3.2,  4.2},
            };

            AssertArray(b, ExpectedDataB);

            ndarray c = np.sort(a, axis: null);     // sort the flattened array
            print(c);
            print("********");

            var ExpectedDataC = new Complex[]
            {1.2,  2.2,  3.2,  4.2,  5.2,  6.2,  7.2,  8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 16.2,
            17.2, 18.2, 19.2, 20.2, 21.2, 22.2, 23.2, 24.2, 25.2, 26.2, 27.2, 28.2, 29.2, 30.2, 31.2, 32.2};

            AssertArray(c, ExpectedDataC);

            ndarray d = np.sort(a, axis: 0);        // sort along the first axis
            print(d);

            var ExpectedDataD = new Complex[8, 4]
            {
                {4.2,  3.2,  2.2,  1.2},
                {8.2,  7.2,  6.2,  5.2},
                {12.2, 11.2, 10.2, 9.2},
                {16.2, 15.2, 14.2, 13.2},
                {20.2, 19.2, 18.2, 17.2},
                {24.2, 23.2, 22.2, 21.2},
                {28.2, 27.2, 26.2, 25.2},
                {32.2, 31.2, 30.2, 29.2},
            };

            AssertArray(d, ExpectedDataD);
            print("********");

        }

        [TestMethod]
        public void test_msort_1_COMPLEX()
        {
            var a = np.array(new Complex[,] { { 1, 4 }, { 3, 1 } });
            ndarray b = np.msort(a);
            print(b);
            AssertArray(b, new Complex[,] { { 1, 1 }, { 3, 4 } });

            a = np.arange(32.2, 0.2, -1.0, dtype: np.Complex);
            b = np.msort(a);

            var ExpectedDataB = new Complex[]
            {1.2,  2.2,  3.2,  4.2,  5.2,  6.2,  7.2,  8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 16.2,
            17.2, 18.2, 19.2, 20.2, 21.2, 22.2, 23.2, 24.2, 25.2, 26.2, 27.2, 28.2, 29.2, 30.2, 31.2, 32.2};
            AssertArray(b, ExpectedDataB);
            print(b);

        }

        [TestMethod]
        public void test_ndarray_argsort_2_COMPLEX()
        {
            var ar = np.array(new Complex[] { 1, 2, 3, 1, 3, 4, 5, 4, 4, 1, 9, 6, 9, 11, 23, 9, 5, 0, 11, 12 }).reshape(new shape(5, 4));

            ndarray perm1 = np.argsort(ar, kind: NPY_SORTKIND.NPY_MERGESORT);
            ndarray perm2 = np.argsort(ar, kind: NPY_SORTKIND.NPY_QUICKSORT);
            ndarray perm3 = np.argsort(ar);

            print(perm1);

            var Perm1Expected = new npy_intp[,]
            {{0, 3, 1, 2},
             {0, 1, 3, 2},
             {1, 0, 3, 2},
             {0, 3, 1, 2},
             {1, 0, 2, 3}};
            AssertArray(perm1, Perm1Expected);

            print(perm2);
            var Perm2Expected = new npy_intp[,]
            {{0, 3, 1, 2},
             {0, 1, 3, 2},
             {1, 0, 3, 2},
             {0, 3, 1, 2},
             {1, 0, 2, 3}};
            AssertArray(perm2, Perm2Expected);


            print(perm3);
            var Perm3Expected = new npy_intp[,]
            {{0, 3, 1, 2},
             {0, 1, 3, 2},
             {1, 0, 3, 2},
             {0, 3, 1, 2},
             {1, 0, 2, 3}};
            AssertArray(perm3, Perm3Expected);
        }

        [TestMethod]
        public void test_argmin_1_COMPLEX()
        {
            ndarray a = np.array(new Complex[] { 32, 33, 45, 98, 11, 02 }).reshape(new shape(2, 3));
            print(a);

            ndarray b = np.argmin(a);
            print(b);
            Assert.AreEqual(b.GetItem(0), (npy_intp)5);
            print("********");

            ndarray c = np.argmin(a, axis: 0);
            print(c);
            AssertArray(c, new npy_intp[] { 0, 1, 1 });
            print("********");

            ndarray d = np.argmin(a, axis: 1);
            print(d);
            AssertArray(d, new npy_intp[] { 0, 2 });
            print("********");

        }

        [TestMethod]
        public void test_argmax_1_COMPLEX()
        {
            ndarray a = np.array(new Complex[] { 32, 33, 45, 98, 11, 02 }).reshape(new shape(2, 3));
            print(a);
            ndarray b = np.argmax(a);
            print(b);
            Assert.AreEqual(b.GetItem(0), (npy_intp)3);
            print("********");

            ndarray c = np.argmax(a, axis: 0);
            print(c);
            AssertArray(c, new npy_intp[] { 1, 0, 0 });
            print("********");

            ndarray d = np.argmax(a, axis: 1);
            print(d);
            AssertArray(d, new npy_intp[] { 2, 0 });
            print("********");

        }

        [TestMethod]
        public void test_searchsorted_1_COMPLEX()
        {
            ndarray arr = np.array(new Complex[] { 1, 2, 3, 4, 5 });
            ndarray a = np.searchsorted(arr, 3);
            print(a);
            Assert.AreEqual(a.GetItem(0), (npy_intp)2);


            ndarray b = np.searchsorted(arr, 3, side: NPY_SEARCHSIDE.NPY_SEARCHRIGHT);
            print(b);
            Assert.AreEqual(b.GetItem(0), (npy_intp)3);


            ndarray c = np.searchsorted(arr, new Int32[] { -10, 10, 2, 3 });
            print(c);
            AssertArray(c, new npy_intp[] { 0, 5, 1, 2 });


            ndarray d = np.searchsorted(np.array(new Complex[] { 15, 14, 13, 12, 11 }), 13);
            print(d);
            Assert.AreEqual(d.GetItem(0), (npy_intp)0);
        }

        [TestMethod]
        public void test_resize_1_COMPLEX()
        {
            ndarray a = np.array(new Complex[,] { { 0, 1 }, { 2, 3 } });
            print(a);

            ndarray b = np.resize(a, new shape(2, 3));
            print(b);

            var ExpectedDataB = new Complex[,]
            {
                { 0,1,2 },
                { 3,0,1 },
            };
            AssertArray(b, ExpectedDataB);


            ndarray c = np.resize(a, new shape(1, 4));
            print(c);
            var ExpectedDataC = new Complex[,]
            {
                { 0,1,2,3 },
            };
            AssertArray(c, ExpectedDataC);

            ndarray d = np.resize(a, new shape(2, 4));
            print(d);
            var ExpectedDataD = new Complex[,]
            {
                { 0,1,2,3 },
                { 0,1,2,3 },
            };
            AssertArray(d, ExpectedDataD);

        }

        [TestMethod]
        public void test_squeeze_1_COMPLEX()
        {
            ndarray x = np.array(new Complex[,,] { { { 0 }, { 1 }, { 2 } } });
            print(x);
            AssertArray(x, new Complex[1, 3, 1] { { { 0 }, { 1 }, { 2 } } });

            ndarray a = np.squeeze(x);
            print(a);
            AssertArray(a, new Complex[] { 0, 1, 2 });

            ndarray b = np.squeeze(x, axis: 0);
            print(b);
            AssertArray(b, new Complex[3, 1] { { 0 }, { 1 }, { 2 } });

            bool CaughtException = false;

            try
            {
                ndarray c = np.squeeze(x, axis: 1);
                print(c);
            }
            catch (Exception ex)
            {
                if (ex.Message.Contains("cannot select an axis to squeeze out which has size not equal to one"))
                    CaughtException = true;
            }
            Assert.IsTrue(CaughtException);

            ndarray d = np.squeeze(x, axis: 2);
            print(d);
            AssertArray(d, new Complex[,] { { 0, 1, 2 } });
        }

        [TestMethod]
        public void test_diagonal_1_COMPLEX()
        {
            ndarray a = np.arange(4, dtype: np.Complex).reshape(new shape(2, 2));
            print(a);
            print("*****");

            ndarray b = a.diagonal();
            print(b);
            AssertArray(b, new Complex[] { 0, 3 });
            print("*****");

            ndarray c = a.diagonal(1);
            print(c);
            AssertArray(c, new Complex[] { 1 });
            print("*****");

            a = np.arange(8, dtype: np.Complex).reshape(new shape(2, 2, 2));
            print(a);
            print("*****");
            b = a.diagonal(0, // Main diagonals of two arrays created by skipping
                           0, // across the outer(left)-most axis last and
                           1); //the "middle" (row) axis first.

            print(b);
            AssertArray(b, new Complex[,] { { 0, 6 }, { 1, 7 } });
            print("*****");

            ndarray d = a.A(":", ":", 0);
            print(d);
            AssertArray(d, new Complex[,] { { 0, 2 }, { 4, 6 } });
            print("*****");

            ndarray e = a.A(":", ":", 1);
            print(e);
            AssertArray(e, new Complex[,] { { 1, 3 }, { 5, 7 } });
            print("*****");
        }

        [TestMethod]
        public void test_trace_1_COMPLEX()
        {
            ndarray a = np.trace(np.eye(3, dtype: np.Complex));
            print(a);
            Assert.AreEqual(a.GetItem(0), (Complex)3.0);
            print("*****");

            a = np.arange(8, dtype: np.Complex).reshape(new shape(2, 2, 2));
            ndarray b = np.trace(a);
            print(b);
            AssertArray(b, new Complex[] { 6, 8 });
            print("*****");

            a = np.arange(24, dtype: np.Complex).reshape(new shape(2, 2, 2, 3));
            var c = np.trace(a);
            print(c);
            AssertArray(c, new Complex[,] { { 18, 20, 22 }, { 24, 26, 28 } });

        }

        [TestMethod]
        public void test_nonzero_COMPLEX()
        {
            ndarray x = np.array(new Complex[,] { { 1, 0, 0 }, { 0, 2, 0 }, { 1, 1, 0 } });
            print(x);
            print("*****");

            ndarray[] y = np.nonzero(x);
            print(y);
            AssertArray(y[0], new npy_intp[] { 0, 1, 2, 2 });
            AssertArray(y[1], new npy_intp[] { 0, 1, 0, 1 });
            print("*****");

            ndarray z = x.A(np.nonzero(x));
            print(z);
            AssertArray(z, new Complex[] { 1, 2, 1, 1 });
            print("*****");

            //ndarray q = np.transpose(np.nonzero(x));
            //print(q);

        }

        [TestMethod]
        public void test_compress_1_COMPLEX()
        {
            ndarray a = np.array(new Complex[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            print(a);
            print("*****");

            ndarray b = np.compress(new int[] { 0, 1 }, a, axis: 0);
            print(b);
            AssertArray(b, new Complex[,] { { 3, 4 } });
            print("*****");

            ndarray c = np.compress(new bool[] { false, true, true }, a, axis: 0);
            print(c);
            AssertArray(c, new Complex[,] { { 3, 4 }, { 5, 6 } });
            print("*****");

            ndarray d = np.compress(new bool[] { false, true }, a, axis: 1);
            print(d);
            AssertArray(d, new Complex[,] { { 2 }, { 4 }, { 6 } });
            print("*****");

            ndarray e = np.compress(new bool[] { false, true }, a);
            AssertArray(e, new Complex[] { 2 });
            print(e);

        }

        [TestMethod]
        public void test_any_1_COMPLEX()
        {
            Complex[] TestData = new Complex[] { 2.5, -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0 };
            var x = np.array(TestData);
            var y = np.any(x);

            print(x);
            print(y);
            Assert.AreEqual(true, y.GetItem(0));

            TestData = new Complex[] { 0.0, 0.0, 0.0, 0.0 };
            x = np.array(TestData);
            y = np.any(x);

            print(x);
            print(y);
            Assert.AreEqual(false, y.GetItem(0));

        }

        [TestMethod]
        public void test_all_1_COMPLEX()
        {
            Complex[] TestData = new Complex[] { 2.5, -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0 };
            var x = np.array(TestData);
            var y = np.all(x);

            print(x);
            print(y);
            Assert.AreEqual(true, y.GetItem(0));

            TestData = new Complex[] { 1.0, 1.0, 0.0, 1.0 };
            x = np.array(TestData);
            y = np.all(x);

            print(x);
            print(y);
            Assert.AreEqual(false, y.GetItem(0));

        }

        [TestMethod]
        public void test_ndarray_mean_1_COMPLEX()
        {
            var x = np.arange(0, 12, dtype: np.Complex).reshape(new shape(3, -1));

            print("X");
            print(x);

            var y = (ndarray)np.mean(x);
            Assert.AreEqual((Complex)5.5, y.GetItem(0));

            print("Y");
            print(y);

            y = (ndarray)np.mean(x, axis: 0);
            AssertArray(y, new Complex[] { 4, 5, 6, 7 });

            print("Y");
            print(y);

            y = (ndarray)np.mean(x, axis: 1);
            AssertArray(y, new Complex[] { 1.5, 5.5, 9.5 });

            print("Y");
            print(y);

        }

        [TestMethod]
        public void test_place_1_COMPLEX_()
        {
            var arr = np.arange(6, dtype: np.Complex).reshape((2, 3));
            np.place(arr, arr > 2, new Int32[] { 44, 55 });
            AssertArray(arr, new Complex[,] { { 0, 1, 2 }, { 44, 55, 44 } });
            print(arr);

            arr = np.arange(16, dtype: np.Complex).reshape((2, 4, 2));
            np.place(arr, arr > 12, new Int32[] { 33 });
            AssertArray(arr, new Complex[,,] { { { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 } }, { { 8, 9 }, { 10, 11 }, { 12, 33 }, { 33, 33 } } });
            print(arr);

            arr = np.arange(6, dtype: np.Complex).reshape((2, 3));
            np.place(arr, arr > 2, new Int32[] { 44, 55, 66, 77, 88, 99, 11, 22, 33 });
            AssertArray(arr, new Complex[,] { { 0, 1, 2 }, { 44, 55, 66 } });
            print(arr);

        }

        [TestMethod]
        public void test_extract_1_COMPLEX()
        {
            var arr = np.arange(12, dtype: np.Complex).reshape((3, 4));
            var condition = np.mod(arr, 3) == 0;
            print(condition);

            var b = np.extract(condition, arr);
            AssertArray(b, new Complex[] { 0, 3, 6, 9 });
            print(b);
        }

        [TestMethod]
        public void test_viewfromaxis_1_COMPLEX()
        {
            Complex[] TestData = new Complex[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            var a = np.zeros_like(TestData).reshape(new shape(3, 2, -1));
            //print(a);


            var b = np.ViewFromAxis(a, 0);
            b[":"] = 99;
            //print(a);
            AssertArray(a, new Complex[,,] { { { 99, 0 }, { 0, 0 } }, { { 99, 0 }, { 0, 0 } }, { { 99, 0 }, { 0, 0 } } });
            //print(a);
            AssertArray(np.sum(a, axis: 0), new Complex[,] { { 297, 0 }, { 0, 0 } });

            b = np.ViewFromAxis(a, 1);
            b[":"] = 11;
            AssertArray(a, new Complex[,,] { { { 11, 0 }, { 11, 0 } }, { { 99, 0 }, { 0, 0 } }, { { 99, 0 }, { 0, 0 } } });
            //print(a);
            AssertArray(np.sum(a, axis: 1), new Complex[,] { { 22, 0 }, { 99, 0 }, { 99, 0 } });

            b = np.ViewFromAxis(a, 2);
            b[":"] = 22;
            AssertArray(a, new Complex[,,] { { { 22, 22 }, { 11, 0 } }, { { 99, 0 }, { 0, 0 } }, { { 99, 0 }, { 0, 0 } } });
            //print(a);
            AssertArray(np.sum(a, axis: 2), new Complex[,] { { 44, 11 }, { 99, 0 }, { 99, 0 } });

            Assert.AreEqual((Complex)253, np.sum(a).GetItem(0));


        }

        [TestMethod]
        public void test_unwrap_1_COMPLEX()
        {
            double retstep = 0;

            var phase = np.linspace(0, Math.PI, ref retstep, num: 5, dtype: np.Complex);
            phase["3:"] = phase.A("3:") + Math.PI;
            print(phase);

            var x = np.unwrap(phase);
            AssertArray(x, new Complex[] { 0, 00.785398163397448, 01.5707963267949, -00.78539816339746, -00.00000000000001 });
            print(x);
        }

        #endregion

        #region from NumericTests

        [TestMethod]
        public void test_zeros_1_COMPLEX()
        {
            var x = np.zeros(new shape(10), dtype: np.Complex);
            print(x);
            print("Update sixth value to 11");
            x[6] = 11;
            print(x);
            print(x.shape);
            print(x.strides);

            AssertArray(x, new Complex[] { 0, 0, 0, 0, 0, 0, 11, 0, 0, 0 });
            AssertShape(x, 10);
            AssertStrides(x, SizeofComplex);
        }

        [TestMethod]
        public void test_zeros_like_2_COMPLEX()
        {
            var a = new Complex[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            var b = np.zeros_like(a);
            b[1, 2] = 99;

            AssertArray(b, new Complex[,] { { 0, 0, 0 }, { 0, 0, 99 } });

            return;
        }

        [TestMethod]
        public void test_ones_1_COMPLEX()
        {
            var x = np.ones(new shape(10), dtype: np.Complex);
            print(x);
            print("Update sixth value to 11");
            x[6] = 11;
            print(x);
            print(x.shape);
            print(x.strides);

            AssertArray(x, new Complex[] { 1, 1, 1, 1, 1, 1, 11, 1, 1, 1 });
            AssertShape(x, 10);
            AssertStrides(x, SizeofComplex);
        }

        [TestMethod]
        public void test_ones_like_3_COMPLEX()
        {
            var a = new Complex[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } };
            var b = np.ones_like(a);
            b[0, 0, 2] = 99;
            b[0, 1, 1] = 88;

            AssertArray(b, new Complex[,,] { { { 1, 1, 99 }, { 1, 88, 1 } } });

            return;
        }

        [TestMethod]
        public void test_empty_COMPLEX()
        {
            var a = np.empty((2, 3));
            AssertShape(a, 2, 3);
            Assert.AreEqual(a.Dtype.TypeNum, NPY_TYPES.NPY_DOUBLE);

            var b = np.empty((2, 4), np.Complex);
            AssertShape(b, 2, 4);
            Assert.AreEqual(b.Dtype.TypeNum, NPY_TYPES.NPY_COMPLEX);
        }

        [TestMethod]
        public void test_empty_like_3_COMPLEX()
        {
            var a = new Complex[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } };
            var b = np.empty_like(a);
            b[0, 0, 2] = 99;
            b[0, 1, 1] = 88;

            AssertArray(b, new Complex[,,] { { { 0, 0, 99 }, { 0, 88, 0 } } });

            return;
        }

        [TestMethod]
        public void test_full_2_COMPLEX()
        {
            var x = np.full((100), 99, dtype: np.Complex).reshape(new shape(10, 10));
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
        public void test_count_nonzero_1_COMPLEX()
        {
            var a = np.count_nonzero(np.eye(4, dtype: np.Complex));
            Assert.AreEqual(4, (int)a);
            print(a);

            var b = np.count_nonzero(new Complex[,] { { 0, 1, 7, 0, 0 }, { 3, 0, 0, 2, 19 } });
            Assert.AreEqual(5, (int)b);
            print(b);

            var c = np.count_nonzero(new Complex[,] { { 0, 1, 7, 0, 0 }, { 3, 0, 0, 2, 19 } }, axis: 0);
            AssertArray(c, new int[] { 1, 1, 1, 1, 1 });
            print(c);

            var d = np.count_nonzero(new Complex[,] { { 0, 1, 7, 0, 0 }, { 3, 0, 0, 2, 19 } }, axis: 1);
            AssertArray(d, new int[] { 2, 3 });
            print(d);

            return;
        }

        [TestMethod]
        public void test_asarray_1_COMPLEX()
        {
            var a = new Complex[] { 1, 2 };
            var b = np.asarray(a);

            AssertArray(b, new Complex[] { 1, 2 });
            print(b);

            var c = np.array(new Complex[] { 1.0, 2.0 }, dtype: np.Complex);
            var d = np.asarray(c, dtype: np.Complex);

            c[0] = 3.0f;
            AssertArray(d, new Complex[] { 3.0, 2.0 });
            print(d);

            var e = np.asarray(a, dtype: np.Complex);
            AssertArray(e, new Complex[] { 1.0, 2.0 });

            print(e);

            return;
        }

        [TestMethod]
        public void test_ascontiguousarray_1_COMPLEX()
        {
            var x = np.arange(6, dtype: np.Complex).reshape((2, 3));
            var y = np.ascontiguousarray(x, dtype: np.Complex);

            AssertArray(y, new Complex[,] { { 0, 1, 2 }, { 3, 4, 5 } });
            print(y);

            Assert.AreEqual(x.flags.c_contiguous, true);
            Assert.AreEqual(y.flags.c_contiguous, true);

            return;
        }

        [TestMethod]
        public void test_asfortranarray_1_COMPLEX()
        {
            var x = np.arange(6, dtype: np.Complex).reshape((2, 3));
            var y = np.asfortranarray(x, dtype: np.Complex);

            AssertArray(y, new Complex[,] { { 0, 1, 2 }, { 3, 4, 5 } });
            print(y);

            Assert.AreEqual(x.flags.f_contiguous, false);
            Assert.AreEqual(y.flags.f_contiguous, true);

            return;
        }

        [TestMethod]
        public void test_isfortran_1_COMPLEX()
        {

            var a = np.array(new Complex[,] { { 1, 2, 3 }, { 4, 5, 6 } }, order: NPY_ORDER.NPY_CORDER);
            var a1 = np.isfortran(a);
            Assert.AreEqual(false, a1);
            print(a1);

            var b = np.array(new Complex[,] { { 1, 2, 3 }, { 4, 5, 6 } }, order: NPY_ORDER.NPY_FORTRANORDER);
            var b1 = np.isfortran(b);
            Assert.AreEqual(true, b1);
            print(b1);

            var c = np.array(new Complex[,] { { 1, 2, 3 }, { 4, 5, 6 } }, order: NPY_ORDER.NPY_CORDER);
            var c1 = np.isfortran(c);
            Assert.AreEqual(false, c1);
            print(c1);

            var d = a.T;
            var d1 = np.isfortran(d);
            Assert.AreEqual(true, d1);
            print(d1);

            // C-ordered arrays evaluate as False even if they are also FORTRAN-ordered.

            var e1 = np.isfortran(np.array(new Complex[] { 1, 2 }, order: NPY_ORDER.NPY_FORTRANORDER));
            Assert.AreEqual(false, e1);
            print(e1);

            return;

        }

        [TestMethod]
        public void test_argwhere_1_COMPLEX()
        {
            var x = np.arange(6, dtype: np.Complex).reshape((2, 3));
            var y = np.argwhere(x > 1);

            var ExpectedY = new npy_intp[,] { { 0, 2 }, { 1, 0 }, { 1, 1 }, { 1, 2 } };
            AssertArray(y, ExpectedY);
            print(y);

            var a = np.arange(12, dtype: np.Complex).reshape((2, 3, 2));
            var b = np.argwhere(a > 1);

            var ExpectedB = new npy_intp[,]
                {{0, 1, 0}, {0, 1, 1}, {0, 2, 0}, {0, 2, 1}, {1, 0, 0},
                 {1, 0, 1}, {1, 1, 0}, {1, 1, 1}, {1, 2, 0}, {1, 2, 1}};

            AssertArray(b, ExpectedB);

            print(b);

            return;
        }

        [TestMethod]
        public void test_flatnonzero_1_COMPLEX()
        {
            var x = np.arange(-2, 3, dtype: np.Complex);

            var y = np.flatnonzero(x);
            AssertArray(y, new npy_intp[] { 0, 1, 3, 4 });
            print(y);

            // Use the indices of the non-zero elements as an index array to extract these elements:

            var z = x.ravel()[np.flatnonzero(x)] as ndarray;
            AssertArray(z, new Complex[] { -2, -1, 1, 2 });
            print(z);

            return;
        }

        [TestMethod]
        public void test_outer_1_COMPLEX()
        {
            var a = np.arange(2, 10, dtype: np.Complex).reshape((2, 4));
            var b = np.arange(12, 20, dtype: np.Complex).reshape((2, 4));
            var c = np.outer(a, b);

            var ExpectedDataC = new Complex[,]
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

            //a = np.arange(2000, 10000, dtype: np.Complex).reshape((-1, 4000));
            //b = np.arange(12000, 20000, dtype: np.Complex).reshape((-1, 4000));

            //System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            //sw.Start();
            //c = np.outer(a, b);
            //sw.Stop();
            //Console.WriteLine(sw.ElapsedMilliseconds);


            return;
        }

        [TestMethod]
        public void test_inner_1_COMPLEX()
        {
            var a = np.arange(1, 5, dtype: np.Complex).reshape((2, 2));
            var b = np.arange(11, 15, dtype: np.Complex).reshape((2, 2));
            var c = np.inner(a, b);
            AssertArray(c, new Complex[,] { { 35, 41 }, { 81, 95 } });
            print(c);


            a = np.arange(2, 10, dtype: np.Complex).reshape((2, 4));
            b = np.arange(12, 20, dtype: np.Complex).reshape((2, 4));
            c = np.inner(a, b);
            print(c);
            AssertArray(c, new Complex[,] { { 194, 250 }, { 410, 530 } });
            print(c.shape);

            return;
        }

        [TestMethod]
        public void test_tensordot_2_COMPLEX()
        {
            var a = np.arange(12.0, dtype: np.Complex).reshape((3, 4));
            var b = np.arange(24.0, dtype: np.Complex).reshape((4, 3, 2));
            var c = np.tensordot(a, b, axis: 1);
            AssertShape(c, 3, 3, 2);
            print(c.shape);
            AssertArray(c, new Complex[,,] { { { 84, 90 }, { 96, 102 }, { 108, 114 } }, { { 228, 250 }, { 272, 294 }, { 316, 338 } }, { { 372, 410 }, { 448, 486 }, { 524, 562 } } });


            c = np.tensordot(a, b, axis: 0);
            AssertShape(c, 3, 4, 4, 3, 2);
            print(c.shape);

            print(c);
        }

        [TestMethod]
        public void test_dot_1_COMPLEX()
        {
            var a = new Complex[,] { { 1, 0 }, { 0, 1 } };
            var b = new Complex[,] { { 4, 1 }, { 2, 2 } };
            var c = np.dot(a, b);
            AssertArray(c, new Complex[,] { { 4, 1 }, { 2, 2 } });
            print(c);

            var d = np.dot(3m, 4m);
            Assert.AreEqual(12m, d.GetItem(0));
            print(d);

            var e = np.arange(3 * 4 * 5 * 6, dtype: np.Complex).reshape((3, 4, 5, 6));
            var f = np.arange(3 * 4 * 5 * 6, dtype: np.Complex).A("::-1").reshape((5, 4, 6, 3));
            var g = np.dot(e, f);
            AssertShape(g.shape, 3, 4, 5, 5, 4, 3);
            Assert.AreEqual((Complex)695768400, g.Sum().GetItem(0));

            // TODO: NOTE: this crazy indexing is not currently working
            //g = g.A(2, 3, 2, 1, 2, 2);
            //Assert.AreEqual(499128, g.GetItem(0));
            //print(g);

        }

        [TestMethod]
        public void test_roll_forward_COMPLEX()
        {
            var a = np.arange(10, dtype: np.Complex);

            print("A");
            print(a);
            print(a.shape);
            print(a.strides);

            var b = np.roll(a, 2);
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);
            AssertArray(b, new Complex[] { 8, 9, 0, 1, 2, 3, 4, 5, 6, 7 });
            AssertShape(b, 10);

            var c = np.roll(b, 2);
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);
            AssertArray(c, new Complex[] { 6, 7, 8, 9, 0, 1, 2, 3, 4, 5 });
            AssertShape(c, 10);

        }

        [TestMethod]
        public void test_roll_backward_COMPLEX()
        {
            var a = np.arange(10, dtype: np.Complex);

            print("A");
            print(a);
            print(a.shape);
            print(a.strides);

            var b = np.roll(a, -2);
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);
            AssertArray(b, new Complex[] { 2, 3, 4, 5, 6, 7, 8, 9, 0, 1 });
            AssertShape(b, 10);

            var c = np.roll(b, -6);
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);
            AssertArray(c, new Complex[] { 8, 9, 0, 1, 2, 3, 4, 5, 6, 7 });
            AssertShape(c, 10);
        }

        [TestMethod]
        public void test_ndarray_rollaxis_COMPLEX()
        {
            var a = np.ones((3, 4, 5, 6), dtype: np.Complex);
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
        public void test_ndarray_moveaxis_COMPLEX()
        {
            var x = np.zeros((3, 4, 5), np.Complex);
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
        public void test_indices_1_COMPLEX()
        {
            var grid = np.indices((2, 3), dtype: np.Complex);
            AssertShape(grid, 2, 2, 3);
            print(grid.shape);
            AssertArray(grid[0] as ndarray, new Complex[,] { { 0, 0, 0 }, { 1, 1, 1 } });
            print(grid[0]);
            AssertArray(grid[1] as ndarray, new Complex[,] { { 0, 1, 2 }, { 0, 1, 2 } });
            print(grid[1]);

            var x = np.arange(20, dtype: np.Complex).reshape((5, 4));

            bool CaughtException = false;
            try
            {
                var y = x[grid[0], grid[1]];
                AssertArray(y as ndarray, new Complex[,] { { 0, 1, 2 }, { 4, 5, 6 } });
                print(y);
            }
            catch
            {
                CaughtException = true;
            }

            Assert.IsTrue(CaughtException, "indexing with decimal should have thrown an exception");

            return;
        }

        [TestMethod]
        public void test_isscalar_1_COMPLEX()
        {

            bool a = np.isscalar((Complex)3.1);
            Assert.AreEqual(true, a);
            print(a);

            bool b = np.isscalar(np.array((Complex)3.1));
            Assert.AreEqual(false, b);
            print(b);

            bool c = np.isscalar(new Complex[] { 3.1 });
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

        [TestMethod]
        public void test_identity_1_COMPLEX()
        {
            ndarray a = np.identity(2, dtype: np.Complex);

            print(a);
            print(a.shape);
            print(a.strides);

            var ExpectedDataA = new Complex[2, 2]
            {
                { 1,0 },
                { 0,1 },
            };
            AssertArray(a, ExpectedDataA);
            AssertShape(a, 2, 2);
            AssertStrides(a, SizeofComplex * 2, SizeofComplex * 1);

            ndarray b = np.identity(5, dtype: np.Complex);

            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new Complex[5, 5]
            {
                { 1, 0, 0, 0, 0 },
                { 0, 1, 0, 0, 0 },
                { 0, 0, 1, 0, 0 },
                { 0, 0, 0, 1, 0 },
                { 0, 0, 0, 0, 1 },
            };
            AssertArray(b, ExpectedDataB);
            AssertShape(b, 5, 5);
            AssertStrides(b, SizeofComplex * 5, SizeofComplex * 1);
        }

        [TestMethod]
        public void test_allclose_1_COMPLEX()
        {
            bool a = np.allclose(new Complex[] { 1e10, 1e-7 }, new Complex[] { 1.00001e10, 1e-8 });
            Assert.AreEqual(false, a);
            print(a);

            bool b = np.allclose(new Complex[] { 1e10, 1e-8 }, new Complex[] { 1.00001e10, 1e-9 });
            Assert.AreEqual(true, b);
            print(b);

            bool c = np.allclose(new Complex[] { 1e10, 1e-8 }, new Complex[] { 1.0001e10, 1e-9 });
            Assert.AreEqual(false, c);
            print(c);

            //bool d = np.allclose(new Complex[] { 1.0m, np.NaN }, new Complex[] { 1.0m, np.NaN });
            //Assert.AreEqual(false, d);
            //print(d);

            //bool e = np.allclose(new Complex[] { 1.0m, np.NaN }, new Complex[] { 1.0m, np.NaN }, equal_nan: true);
            //Assert.AreEqual(true, e);
            //print(e);

            return;
        }

        [TestMethod]
        public void test_isclose_1_COMPLEX()
        {
            var a = np.isclose(new Complex[] { 1e10, 1e-7 }, new Complex[] { 1.00001e10, 1e-8 });
            AssertArray(a, new bool[] { true, false });
            print(a);

            var b = np.isclose(new Complex[] { 1e10, 1e-8 }, new Complex[] { 1.00001e10, 1e-9 });
            AssertArray(b, new bool[] { true, true });
            print(b);

            var c = np.isclose(new Complex[] { 1e10, 1e-8 }, new Complex[] { 1.0001e10, 1e-9 });
            AssertArray(c, new bool[] { false, true });
            print(c);

            //var d = np.isclose(new double[] { 1.0, np.NaN }, new double[] { 1.0, np.NaN });
            //AssertArray(d, new bool[] { true, false });
            //print(d);

            //var e = np.isclose(new double[] { 1.0, np.NaN }, new double[] { 1.0, np.NaN }, equal_nan: true);
            //AssertArray(e, new bool[] { true, true });
            //print(e);

            var f = np.isclose(new Complex[] { 1e-8, 1e-7 }, new Complex[] { 0.0, 0.0 });
            AssertArray(f, new bool[] { true, false });
            print(f);

            var g = np.isclose(new Complex[] { 1e-100, 1e-7 }, new Complex[] { 0.0, 0.0 }, atol: 0.0);
            //AssertArray(g, new bool[] { false, false });
            print(g);

            var h = np.isclose(new Complex[] { 1e-10, 1e-10 }, new Complex[] { 1e-20, 0.0 });
            AssertArray(h, new bool[] { true, true });
            print(h);

            var i = np.isclose(new Complex[] { 1e-10, 1e-10 }, new Complex[] { 1e-20, 0.999999e-10 }, atol: 0.0);
            AssertArray(i, new bool[] { false, true });
            print(i);
        }

        [TestMethod]
        public void test_array_equal_1_COMPLEX()
        {
            var a = np.array_equal(new Complex[] { 1, 2 }, new Complex[] { 1, 2 });
            Assert.AreEqual(true, a);
            print(a);

            var b = np.array_equal(np.array(new Complex[] { 1, 2 }), np.array(new Complex[] { 1, 2 }));
            Assert.AreEqual(true, b);
            print(b);

            var c = np.array_equal(new Complex[] { 1, 2 }, new Complex[] { 1, 2, 3 });
            Assert.AreEqual(false, c);
            print(c);

            var d = np.array_equal(new Complex[] { 1, 2 }, new Complex[] { 1, 4 });
            Assert.AreEqual(false, d);
            print(d);
        }

        [TestMethod]
        public void test_array_equiv_1_COMPLEX()
        {
            var a = np.array_equiv(new Complex[] { 1, 2 }, new Complex[] { 1, 2 });
            Assert.AreEqual(true, a);
            print(a);

            var b = np.array_equiv(new Complex[] { 1, 2 }, new Complex[] { 1, 3 });
            Assert.AreEqual(false, b);
            print(b);

            var c = np.array_equiv(new Complex[] { 1, 2 }, new Complex[,] { { 1, 2 }, { 1, 2 } });
            Assert.AreEqual(true, c);
            print(c);

            var d = np.array_equiv(new Complex[] { 1, 2 }, new Complex[,] { { 1, 2, 1, 2 }, { 1, 2, 1, 2 } });
            Assert.AreEqual(false, d);
            print(d);

            var e = np.array_equiv(new Complex[] { 1, 2 }, new Complex[,] { { 1, 2 }, { 1, 3 } });
            Assert.AreEqual(false, e);
            print(e);
        }

        #endregion

        #region from NANFunctionsTests

        [TestMethod]
        public void test_nanprod_1_COMPLEX()
        {

            var x = np.nanprod(1m);
            Assert.AreEqual(1m, x.GetItem(0));
            print(x);

            var y = np.nanprod(new Complex[] { 1 });
            Assert.AreEqual((Complex)1, y.GetItem(0));
            print(y);



            var a = np.array(new Complex[,] { { 1, 2 }, { 3, 4 } });
            var b = np.nanprod(a);
            Assert.AreEqual((Complex)24.0, b.GetItem(0));
            print(b);

            var c = np.nanprod(a, axis: 0);
            AssertArray(c, new Complex[] { 3, 8 });
            print(c);

            var d = np.nanprod(a, axis: 1);
            AssertArray(d, new Complex[] { 2, 12 });
            print(d);

            return;
        }

        #endregion

        #region from StatisticsTests

        [TestMethod]
        public void test_amin_2_COMPLEX()
        {
            ndarray a = np.arange(30.25, 46.25, dtype: np.Complex).reshape(new shape(4, 4));
            print(a);
            print("*****");

            ndarray b = np.amin(a);          // Minimum of the flattened array
            print(b);
            Assert.AreEqual((Complex)30.25, b.GetItem(0));
            print("*****");

            ndarray c = np.amin(a, axis: 0);  // Minimum along the first axis
            print(c);
            AssertArray(c, new Complex[] { 30.25, 31.25, 32.25, 33.25 });
            print("*****");

            ndarray d = np.amin(a, axis: 1);   // Minimum along the second axis
            print(d);
            AssertArray(d, new Complex[] { 30.25, 34.25, 38.25, 42.25 });
            print("*****");

            // Complex don't support NAN
            //ndarray e = np.arange(5, dtype: np.Complex);
            //e[2] = np.NaN;
            //ndarray f = np.amin(e);
            //print(f);
            //Assert.AreEqual(np.NaN, f.GetItem(0));
            //print("*****");

        }

        [TestMethod]
        public void test_amax_2_COMPLEX()
        {
            ndarray a = np.arange(30.25, 46.25, dtype: np.Complex).reshape(new shape(4, 4));
            print(a);
            print("*****");

            ndarray b = np.amax(a);          // Maximum of the flattened array
            print(b);
            Assert.AreEqual((Complex)45.25, b.GetItem(0));
            print("*****");

            ndarray c = np.amax(a, axis: 0);  // Maxima along the first axis
            print(c);
            AssertArray(c, new Complex[] { 42.25, 43.25, 44.25, 45.25 });
            print("*****");

            ndarray d = np.amax(a, axis: 1);   // Maxima along the second axis
            print(d);
            AssertArray(d, new Complex[] { 33.25, 37.25, 41.25, 45.25 });
            print("*****");

            // Complex don't support NAN
            //ndarray e = np.arange(5, dtype: np.Complex);
            //e[2] = np.NaN;
            //ndarray f = np.amax(e);
            //print(f);
            //Assert.AreEqual(np.NaN, f.GetItem(0));
            //print("*****");

            //ndarray g = np.nanmax(b);
            //print(g);
        }

        [TestMethod]
        public void test_ptp_1_COMPLEX()
        {
            ndarray a = np.arange(4, dtype: np.Complex).reshape(new shape(2, 2));
            print(a);
            print("*****");

            ndarray b = np.ptp(a, axis: 0);
            print(b);
            AssertArray(b, new Complex[] { 2, 2 });
            print("*****");

            ndarray c = np.ptp(a, axis: 1);
            print(c);
            AssertArray(c, new Complex[] { 1, 1 });

            ndarray d = np.ptp(a);
            print(d);
            Assert.AreEqual((Complex)3, d.GetItem(0));
        }

        [TestMethod]
        public void test_percentile_2_COMPLEX()
        {
            var a = np.array(new Complex[,] { { 10, 7, 4 }, { 3, 2, 1 } });

            var b = np.percentile(a, new Complex[] { 50, 75 });
            AssertArray(b, new Complex[] { 3.5, 6.25 });
            print(b);

            var c = np.percentile(a, new Complex[] { 50, 75 }, axis: 0);
            AssertArray(c, new Complex[,] { { 6.5, 4.5, 2.5 }, { 8.25, 5.75, 3.25 } });
            print(c);

            var d = np.percentile(a, new Complex[] { 50, 75 }, axis: 1);
            AssertArray(d, new Complex[,] { { 7.0, 2.0 }, { 8.5, 2.5 } });
            print(d);

            var e = np.percentile(a, new Complex[] { 50, 75 }, axis: 1, keepdims: true);
            AssertArray(e, new Complex[,,] { { { 7.0 }, { 2.0 } }, { { 8.5 }, { 2.5 } } });
            print(e);

            // note: we dont support the out parameter
            //var m = np.percentile(a, 50, axis : 0);
            //var n = np.zeros_like(m);
            //var o = np.percentile(a, 50, axis : 0);
            //print(o);
            //print(n);
            // note: we don't support the overwrite_input flag
            //b = a.Copy();
            //c = np.percentile(b, 50, axis: 1, overwrite_input: true);
            //print(c);

            //Assert.IsFalse((bool)np.all(a.Equals(b)).GetItem(0));

            return;
        }

        [TestMethod]
        public void test_quantile_2_COMPLEX()
        {
            var a = np.array(new Complex[,] { { 10, 7, 4 }, { 3, 2, 1 } });

            var b = np.quantile(a, new Complex[] { 0.5, 0.75 });
            AssertArray(b, new Complex[] { 3.5, 6.25 });
            print(b);

            var c = np.quantile(a, new Complex[] { 0.5, 0.75 }, axis: 0);
            AssertArray(c, new Complex[,] { { 6.5, 4.5, 2.5 }, { 8.25, 5.75, 3.25 } });
            print(c);

            var d = np.quantile(a, new Complex[] { 0.5, 0.75 }, axis: 1);
            AssertArray(d, new Complex[,] { { 7.0, 2.0 }, { 8.5, 2.5 } });
            print(d);

            var e = np.quantile(a, new Complex[] { 0.5, 0.75 }, axis: 1, keepdims: true);
            AssertArray(e, new Complex[,,] { { { 7.0 }, { 2.0 } }, { { 8.5 }, { 2.5 } } });
            print(e);

            // note: we dont support the out parameter

            //var m = np.quantile(a, 0.5, axis: 0);
            //var n = np.zeros_like(m);
            //var o = np.quantile(a, 0.5, axis: 0);
            //print(o);
            //print(n);
            // note: we don't support the overwrite_input flag
            //b = a.Copy();
            //c = np.quantile(b, 0.5, axis: 1, overwrite_input: true);
            //print(c);

            //Assert.IsFalse((bool)np.all(a.Equals(b)).GetItem(0));

            return;
        }

        [TestMethod]
        public void test_median_2_COMPLEX()
        {
            var a = np.arange(0, 64, 1, np.Complex).reshape((4, 4, 4));

            var b = np.median(a, axis: new int[] { 0, 2 }, keepdims: true);
            AssertArray(b, new Complex[,,] { { { 25.5 }, { 29.5 }, { 33.5 }, { 37.5 } } });
            print(b);

            var c = np.median(a, new int[] { 0, 1 }, keepdims: true);
            AssertArray(c, new Complex[,,] { { { 30, 31, 32, 33 } } });
            print(c);

            var d = np.median(a, new int[] { 1, 2 }, keepdims: true);
            AssertArray(d, new Complex[,,] { { { 7.5 } }, { { 23.5 } }, { { 39.5 } }, { { 55.5 } } });
            print(d);

            return;
        }

        [TestMethod]
        public void test_average_3_COMPLEX()
        {

            var a = np.array(new Complex[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            var w = new int[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

            var x1 = np.average(a, axis: null, weights: null, returned: true);
            Assert.AreEqual((Complex)5.5, x1.retval.GetItem(0));
            Assert.AreEqual((Complex)10.0, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a, axis: null, weights: w, returned: true);
            Assert.AreEqual((Complex)4.0, x1.retval.GetItem(0));
            Assert.AreEqual((Complex)55.0, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: null, weights: np.array(w).reshape((2, -1)), returned: true);
            Assert.AreEqual((Complex)4.0, x1.retval.GetItem(0));
            Assert.AreEqual((Complex)55.0, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: 0, weights: np.array(w).reshape((2, -1)), returned: true);
            AssertArray(x1.retval, new Complex[] { 2.6666666666666666666666666667, 3.5384615384615384615384615385, 4.3636363636363636363636363636,
                                                   5.1111111111111111111111111111, 5.7142857142857142857142857143 });
            AssertArray(x1.sum_of_weights, new Complex[] { 15.0, 13.0, 11.0, 9.0, 7.0 });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: 1, weights: np.array(w).reshape((2, -1)), returned: true);
            AssertArray(x1.retval, new Complex[] { 2.75, 7.3333333333333333333333333333 });
            AssertArray(x1.sum_of_weights, new Complex[] { 40.0, 15.0 });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((1, 2, -1, 1)), axis: 1, weights: np.array(w).reshape((1, 2, -1, 1)), returned: true);
            AssertArray(x1.retval, new Complex[,,] { { { 2.6666666666666666666666666667 }, { 3.5384615384615384615384615385 }, { 4.3636363636363636363636363636 },
                                                      { 5.1111111111111111111111111111 }, { 5.7142857142857142857142857143 } } });
            AssertArray(x1.sum_of_weights, new Complex[,,] { { { 15.0 }, { 13.0 }, { 11.0 }, { 9.0 }, { 7.0 } } });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((1, -1, 2, 1)), axis: 1, weights: np.array(w).reshape((1, -1, 2, 1)), returned: true);
            AssertArray(x1.retval, new Complex[,,] { { { 3.6666666666666666666666666667 }, { 4.4 } } });
            AssertArray(x1.sum_of_weights, new Complex[,,] { { { 30.0 }, { 25.0 } } });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1, 1, 1)), axis: 1, weights: np.array(w).reshape((2, -1, 1, 1)), returned: false);
            AssertArray(x1.retval, new Complex[,,] { { { 2.75 } }, { { 7.3333333333333333333333333333 } } });
            Assert.AreEqual(null, x1.sum_of_weights);
            print(x1);

        }

        [TestMethod]
        public void test_mean_1_COMPLEX()
        {
            Complex[] TestData = new Complex[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.Complex).reshape(new shape(3, 2, -1));
            x = x * 3;
            print(x);

            var y = np.mean(x);
            print(y);
            Assert.AreEqual((Complex)131.5, y.GetItem(0));

            y = np.mean(x, axis: 0);
            print(y);
            AssertArray(y, new Complex[,] { { 113, 150 }, { 113, 150 } });

            y = np.mean(x, axis: 1);
            print(y);
            AssertArray(y, new Complex[,] { { 52.5, 90 }, { 132, 157.5 }, { 154.5, 202.5 } });

            y = np.mean(x, axis: 2);
            print(y);
            AssertArray(y, new Complex[,] { { 37.5, 105 }, { 252, 37.5 }, { 105, 252 } });

        }

        [TestMethod]
        public void test_mean_2_COMPLEX()
        {
            ndarray a = np.zeros(new shape(2, 512 * 512), dtype: np.Complex);
            a[0, ":"] = 1.0;
            a[1, ":"] = 0.1;
            ndarray b = np.mean(a);
            print(b);
            Assert.AreEqual((Complex)0.54999999998835847, (Complex)b.GetItem(0));

            ndarray c = np.mean(a, dtype: np.Complex);
            print(c);
            Assert.AreEqual((Complex)0.54999999998835847, c.GetItem(0));
        }

        [TestMethod]
        public void test_std_1_COMPLEX()
        {
            ndarray a = np.array(new Complex[,] { { 1, 2 }, { 3, 4 } });
            ndarray b = np.std(a);
            print(b);
            Assert.AreEqual(1.1180339887498948482045868344, (Complex)b.GetItem(0));

            ndarray c = np.std(a, axis: 0);
            print(c);
            AssertArray(c, new Complex[] { 1.0, 1.0 });

            ndarray d = np.std(a, axis: 1);
            print(d);
            AssertArray(d, new Complex[] { 0.5, 0.5}); 

            // In single precision, std() can be inaccurate:
            a = np.zeros(new shape(2, 512 * 512), dtype: np.Complex);
            a[0, ":"] = 1.0;
            a[1, ":"] = 0.1;
            b = np.std(a);
            print(b);
            Assert.AreEqual((Complex)0.44999999999905771, b.GetItem(0));
            // Computing the standard deviation in float64 is more accurate:
            c = np.std(a, dtype: np.Complex);
            print(c);
            Assert.AreEqual((Complex)0.44999999999905771, c.GetItem(0));

        }

        [TestMethod]
        public void test_var_1_COMPLEX()
        {
            ndarray a = np.array(new Complex[,] { { 1, 2 }, { 3, 4 } });
            ndarray b = np.var(a);
            Assert.AreEqual((Complex)1.25, b.GetItem(0));
            print(b);

            ndarray c = np.var(a, axis: 0);
            AssertArray(c, new Complex[] { 1.0, 1.0 });
            print(c);

            ndarray d = np.var(a, axis: 1);
            AssertArray(d, new Complex[] { 0.25, 0.25 });
            print(d);

            // In single precision, std() can be inaccurate:
            a = np.zeros(new shape(2, 512 * 512), dtype: np.Complex);
            a[0, ":"] = 1.0;
            a[1, ":"] = 0.1;
            b = np.var(a);
            Assert.AreEqual((Complex)0.20249999999915194, b.GetItem(0));
            print(b);

            // Computing the standard deviation in float64 is more accurate:
            c = np.var(a, dtype: np.Complex);
            Assert.AreEqual((Complex)0.20249999999915194, c.GetItem(0));
            print(c);

        }

        [TestMethod]
        public void test_corrcoef_1_COMPLEX()
        {
            var x1 = np.array(new Complex[,] { { 0, 2 }, { 1, 1 }, { 2, 0 } }).T;
            print(x1);

            // Note how  increases while  decreases. The covariance matrix shows this clearly:

            var a = np.corrcoef(x1);
            AssertArray(a, new Complex[,] { { 1, -1 }, { -1, 1 } });
            print(a);

            var x = new Complex[] { -2.1, -1, 4.3 };
            var y = new Complex[] { 3, 1.1, 0.12 };
            var X = np.stack(new object[] { x, y }, axis: 0);
            a = np.corrcoef(X);
            AssertArray(a, new Complex[,] { { 1.0, -0.8553578095227944904571128856, }, { -0.8553578095227944904571128856, 1.0 } });
            print(a);


            var b = np.corrcoef(x, y);
            AssertArray(b, new Complex[,] { { 1.0, -0.8553578095227944904571128856, }, { -0.8553578095227944904571128856, 1.0 } });
            print(b);

            var c = np.corrcoef(x, y, rowvar: false);
            AssertArray(c, new Complex[,] { { 1.0, -0.8553578095227944904571128856, }, { -0.8553578095227944904571128856, 1.0 } });
            print(c);


            return;
        }

        [TestMethod]
        public void test_correlate_1_COMPLEX()
        {
            var a = np.correlate(new Complex[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f });
            AssertArray(a, new Complex[] { 3.5 });
            print(a);

            var b = np.correlate(new Complex[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_SAME);
            AssertArray(b, new Complex[] { 2.0, 3.5, 3.0 });
            print(b);

            var c = np.correlate(new Complex[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_FULL);
            AssertArray(c, new Complex[] { 0.5, 2.0, 3.5, 3.0, 0.0 });
            print(c);

            return;
        }

        [TestMethod]
        public void test_cov_1_COMPLEX()
        {
            var x1 = np.array(new Complex[,] { { 0, 2 }, { 1, 1 }, { 2, 0 } }).T;
            print(x1);

            // Note how  increases while  decreases. The covariance matrix shows this clearly:

            var a = np.cov(x1);
            AssertArray(a, new Complex[,] { { 1, -1 }, { -1, 1 } });
            print(a);

            var x = new Complex[] { -2.1, -1, 4.3 };
            var y = new Complex[] { 3, 1.1, 0.12 };
            var X = np.stack(new object[] { x, y }, axis: 0);
            a = np.cov(X);
            AssertArray(a, new Complex[,] { { 11.710, -4.286 }, { -4.286, 2.1441333333333333333333333334 } });
            print(a);


            var b = np.cov(x, y);
            AssertArray(b, new Complex[,] { { 11.710, -4.286 }, { -4.286, 2.1441333333333333333333333334 } });
            print(b);

            var c = np.cov(x);
            Assert.AreEqual((Complex)11.709999999999999, c.GetItem(0));
            print(c);

            var d = np.cov(X, rowvar: false);
            AssertArray(d, new Complex[,] { { 13.00500, 5.35500, -10.65900 }, { 5.35500, 2.20500, -4.38900 }, { -10.65900, -4.38900, 8.73620 } });
            print(d);

            var e = np.cov(X, rowvar: false, bias: true);
            AssertArray(e, new Complex[,] { { 6.50250, 2.67750, -5.32950 }, { 2.67750, 1.10250, -2.19450 }, { -5.32950, -2.19450, 4.36810 } });
            print(e);

            var f = np.cov(X, rowvar: false, bias: true, fweights: new int[] { 1, 2 });
            AssertArray(f, new Complex[,] { { 5.7799999999999999999999999994, 2.3799999999999999999999999998, -4.7373333333333333333333333329 },
                                            { 2.3799999999999999999999999998, 0.9799999999999999999999999999, -1.9506666666666666666666666665 },
                                            { -4.7373333333333333333333333329, -1.9506666666666666666666666665, 3.8827555555555555555555555553 } });

            print(f);

            var g = np.cov(X, rowvar: false, bias: true, fweights: new int[] { 1, 2 }, aweights: new int[] { 1, 2 });
            AssertArray(g, new Complex[,] { { 4.16160, 1.71360, -3.410880 }, { 1.71360, 0.70560, -1.404480 }, { -3.410880, -1.404480, 2.7955840 } });
            print(g);

            return;
        }

        #endregion

        #region from TwoDimBaseTests

        [TestMethod]
        public void test_diag_1_COMPLEX()
        {
            ndarray m = np.arange(9, dtype: np.Complex);
            var n = np.diag(m);

            print(m);
            print(n);

            var ExpectedDataN = new Complex[,]
                {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                 {0, 1, 0, 0, 0, 0, 0, 0, 0},
                 {0, 0, 2, 0, 0, 0, 0, 0, 0},
                 {0, 0, 0, 3, 0, 0, 0, 0, 0},
                 {0, 0, 0, 0, 4, 0, 0, 0, 0},
                 {0, 0, 0, 0, 0, 5, 0, 0, 0},
                 {0, 0, 0, 0, 0, 0, 6, 0, 0},
                 {0, 0, 0, 0, 0, 0, 0, 7, 0},
                 {0, 0, 0, 0, 0, 0, 0, 0, 8}};

            AssertArray(n, ExpectedDataN);

            m = np.arange(9, dtype: np.Complex).reshape(new shape(3, 3));
            n = np.diag(m);

            print(m);
            print(n);
            AssertArray(n, new Complex[] { 0, 4, 8 });
        }

        [TestMethod]
        public void test_diagflat_1_COMPLEX()
        {
            ndarray m = np.arange(1, 5, dtype: np.Complex).reshape(new shape(2, 2));
            var n = np.diagflat(m);

            print(m);
            print(n);

            var ExpectedDataN = new Complex[,]
            {
             {1, 0, 0, 0},
             {0, 2, 0, 0},
             {0, 0, 3, 0},
             {0, 0, 0, 4}
            };
            AssertArray(n, ExpectedDataN);

            m = np.arange(1, 3, dtype: np.Complex);
            n = np.diagflat(m, 1);

            print(m);
            print(n);

            ExpectedDataN = new Complex[,]
            {
             {0, 1, 0},
             {0, 0, 2},
             {0, 0, 0},
            };

            AssertArray(n, ExpectedDataN);

            m = np.arange(1, 3, dtype: np.Complex);
            n = np.diagflat(m, -1);

            print(m);
            print(n);

            ExpectedDataN = new Complex[,]
            {
             {0, 0, 0},
             {1, 0, 0},
             {0, 2, 0},
            };

            AssertArray(n, ExpectedDataN);

        }

        [TestMethod]
        public void test_fliplr_1_COMPLEX()
        {
            ndarray m = np.arange(8, dtype: np.Complex).reshape(new shape(2, 2, 2));
            var n = np.fliplr(m);

            print(m);
            print(n);

            AssertArray(n, new Complex[,,] { { { 2, 3 }, { 0, 1 } }, { { 6, 7 }, { 4, 5 } } });
        }

        [TestMethod]
        public void test_flipud_1_COMPLEX()
        {
            ndarray m = np.arange(8, dtype: np.Complex).reshape(new shape(2, 2, 2));
            var n = np.flipud(m);

            print(m);
            print(n);

            AssertArray(n, new Complex[,,] { { { 4, 5 }, { 6, 7 } }, { { 0, 1 }, { 2, 3 } } });
        }

        [TestMethod]
        public void test_tri_1_COMPLEX()
        {
            ndarray a = np.tri(3, 5, 2, dtype: np.Complex);
            print(a);

            var ExpectedDataA = new Complex[,]
            {
             {1, 1, 1, 0, 0},
             {1, 1, 1, 1, 0},
             {1, 1, 1, 1, 1}
            };
            AssertArray(a, ExpectedDataA);

            print("***********");
            ndarray b = np.tri(3, 5, -1, dtype: np.Complex);
            print(b);

            var ExpectedDataB = new Complex[,]
            {
             {0.0, 0.0, 0.0, 0.0, 0.0},
             {1.0, 0.0, 0.0, 0.0, 0.0},
             {1.0, 1.0, 0.0, 0.0, 0.0}
            };
            AssertArray(b, ExpectedDataB);
        }

        [TestMethod]
        public void test_tril_1_COMPLEX()
        {
            ndarray a = np.array(new Complex[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } });
            ndarray b = np.tril(a, -1);
            print(a);
            print("***********");
            print(b);

            var ExpectedDataB = new Complex[,]
            {
             {0, 0, 0},
             {4, 0, 0},
             {7, 8, 0},
             {10, 11, 12},
            };
            AssertArray(b, ExpectedDataB);

        }

        [TestMethod]
        public void test_triu_1_COMPLEX()
        {
            ndarray a = np.array(new Complex[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } });
            ndarray b = np.triu(a, -1);
            print(a);
            print("***********");
            print(b);

            var ExpectedDataB = new Complex[,]
            {
             {1, 2, 3},
             {4, 5, 6},
             {0, 8, 9},
             {0, 0, 12},
            };
            AssertArray(b, ExpectedDataB);

        }

        [TestMethod]
        public void test_vander_1_COMPLEX()
        {
            var x = np.array(new Complex[] { 1, 2, 3, 5 });
            int N = 3;
            var y = np.vander(x, N);
            AssertArray(y, new Complex[,] { { 1, 1, 1 }, { 4, 2, 1 }, { 9, 3, 1 }, { 25, 5, 1 } });
            print(y);

            y = np.vander(x);
            AssertArray(y, new Complex[,] { { 1, 1, 1, 1 }, { 8, 4, 2, 1 }, { 27, 9, 3, 1 }, { 125, 25, 5, 1 } });
            print(y);

            y = np.vander(x, increasing: true);
            AssertArray(y, new Complex[,] { { 1, 1, 1, 1 }, { 1, 2, 4, 8 }, { 1, 3, 9, 27 }, { 1, 5, 25, 125 } });
            print(y);

            return;
        }

        [TestMethod]
        public void test_mask_indices_COMPLEX()
        {
            var iu = np.mask_indices(3, np.triu);
            AssertArray(iu[0], new npy_intp[] { 0, 0, 0, 1, 1, 2 });
            AssertArray(iu[1], new npy_intp[] { 0, 1, 2, 1, 2, 2 });
            print(iu);

            var a = np.arange(9, dtype: np.Complex).reshape((3, 3));
            var b = a[iu] as ndarray;
            AssertArray(b, new Complex[] { 0, 1, 2, 4, 5, 8 });
            print(b);

            var iu1 = np.mask_indices(3, np.triu, 1);

            var c = a[iu1] as ndarray;
            AssertArray(c, new Complex[] { 1, 2, 5 });
            print(c);

            return;
        }

        [TestMethod]
        public void test_tril_indices_COMPLEX()
        {
            var il1 = np.tril_indices(4);
            var il2 = np.tril_indices(4, 2);

            var a = np.arange(16, dtype: np.Complex).reshape((4, 4));
            var b = a[il1] as ndarray;
            AssertArray(b, new Complex[] { 0, 4, 5, 8, 9, 10, 12, 13, 14, 15 });
            print(b);

            a[il1] = -1;

            var ExpectedDataA1 = new Complex[,]
                {{-1,  1, 2,  3}, {-1, -1,  6,  7},
                 {-1, -1,-1, 11}, {-1, -1, -1, -1}};
            AssertArray(a, ExpectedDataA1);
            print(a);

            a[il2] = -10;

            var ExpectedDataA2 = new Complex[,]
                {{-10, -10, -10,  3}, {-10, -10, -10, -10},
                 {-10, -10,-10, -10}, {-10, -10, -10, -10}};
            AssertArray(a, ExpectedDataA2);
            print(a);

            return;
        }

        [TestMethod]
        public void test_tril_indices_from_COMPLEX()
        {
            var a = np.arange(16, dtype: np.Complex).reshape((4, 4));
            var il1 = np.tril_indices_from(a, 0);

            AssertArray(il1[0], new npy_intp[] { 0, 1, 1, 2, 2, 2, 3, 3, 3, 3 });
            AssertArray(il1[1], new npy_intp[] { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3 });
            print(il1);

            var il2 = np.tril_indices_from(a, 2);
            AssertArray(il2[0], new npy_intp[] { 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 });
            AssertArray(il2[1], new npy_intp[] { 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 });

            print(il2);

            return;
        }

        [TestMethod]
        public void test_triu_indices_COMPLEX()
        {
            var il1 = np.triu_indices(4);
            var il2 = np.triu_indices(4, 2);

            var a = np.arange(16, dtype: np.Complex).reshape((4, 4));
            var b = a[il1] as ndarray;
            AssertArray(b, new Complex[] { 0, 1, 2, 3, 5, 6, 7, 10, 11, 15 });
            print(b);

            a[il1] = -1;

            var ExpectedDataA1 = new Complex[,]
                {{-1, -1, -1, -1}, { 4, -1, -1, -1},
                 { 8,  9, -1, -1}, {12, 13, 14, -1}};
            AssertArray(a, ExpectedDataA1);
            print(a);

            a[il2] = -10;

            var ExpectedDataA2 = new Complex[,]
                {{-1, -1, -10, -10}, {4,  -1, -1, -10},
                 { 8,  9, -1,  -1},  {12, 13, 14, -1}};
            AssertArray(a, ExpectedDataA2);
            print(a);

            return;
        }

        [TestMethod]
        public void test_triu_indices_from_COMPLEX()
        {
            var a = np.arange(16, dtype: np.Complex).reshape((4, 4));
            var il1 = np.triu_indices_from(a, 0);

            AssertArray(il1[0], new npy_intp[] { 0, 0, 0, 0, 1, 1, 1, 2, 2, 3 });
            AssertArray(il1[1], new npy_intp[] { 0, 1, 2, 3, 1, 2, 3, 2, 3, 3 });
            print(il1);

            var il2 = np.triu_indices_from(a, 2);
            AssertArray(il2[0], new npy_intp[] { 0, 0, 1 });
            AssertArray(il2[1], new npy_intp[] { 2, 3, 3 });

            print(il2);

            return;
        }

        #endregion

        #region from ShapeBaseTests

        [TestMethod]
        public void test_atleast_1d_COMPLEX()
        {
            var a = np.atleast_1d((Complex)1.0);
            print(a);
            AssertArray(a.ElementAt(0), new Complex[] { 1.0 });

            print("**************");
            var x = np.arange(9.0, dtype: np.Complex).reshape(new shape(3, 3));
            var b = np.atleast_1d(x);
            print(b);

            var ExpectedB = new Complex[,]
                {{0.0, 1.0, 2.0},
                 {3.0, 4.0, 5.0},
                 {6.0, 7.0, 8.0}};
            AssertArray(b.ElementAt(0), ExpectedB);

            print("**************");

            var c = np.atleast_1d(new object[] { (Complex)1, new Complex[] { 3, 4 } });

            AssertArray(c.ElementAt(0), new Complex[] { 1 });
            AssertArray(c.ElementAt(1), new Complex[] { 3, 4 });
            print(c);

        }

        [TestMethod]
        public void test_atleast_2d_COMPLEX()
        {
            var a = np.atleast_2d((Complex)1.0);
            print(a);
            AssertArray(a.ElementAt(0), new Complex[,] { { 1.0 } });

            print("**************");
            var x = np.arange((Complex)9.0, dtype: np.Complex).reshape(new shape(3, 3));
            var b = np.atleast_2d(x);
            print(b);

            var ExpectedB = new Complex[,]
                {{0.0, 1.0, 2.0},
                 {3.0, 4.0, 5.0},
                 {6.0, 7.0, 8.0}};
            AssertArray(b.ElementAt(0), ExpectedB);

            print("**************");

            var c = np.atleast_2d(new object[] { (Complex)1, new Complex[] { 3, 4 }, new Complex[] { 5, 6 } });

            AssertArray(c.ElementAt(0), new Complex[,] { { 1 } });
            AssertArray(c.ElementAt(1), new Complex[,] { { 3, 4 } });
            AssertArray(c.ElementAt(2), new Complex[,] { { 5, 6 } });
            print(c);

        }

        [TestMethod]
        public void test_atleast_3d_COMPLEX()
        {
            var a = np.atleast_3d((Complex)1.0);
            print(a);
            AssertArray(a.ElementAt(0), new Complex[,,] { { { 1.0 } } });

            print("**************");
            var x = np.arange(9.0, dtype: np.Complex).reshape(new shape(3, 3));
            var b = np.atleast_3d(x);
            print(b);

            var ExpectedB = new Complex[,,]
             {{{0.0},
               {1.0},
               {2.0}},
              {{3.0},
               {4.0},
               {5.0}},
              {{6.0},
               {7.0},
               {8.0}}};

            AssertArray(b.ElementAt(0), ExpectedB);

            print("**************");

            var c = np.atleast_3d(new object[] { new Complex[] { 1, 2 }, new Complex[] { 3, 4 }, new Complex[] { 5, 6 } });

            AssertArray(c.ElementAt(0), new Complex[,,] { { { 1 }, { 2 } } });
            AssertArray(c.ElementAt(1), new Complex[,,] { { { 3 }, { 4 } } });
            AssertArray(c.ElementAt(2), new Complex[,,] { { { 5 }, { 6 } } });
            print(c);


        }

        [TestMethod]
        public void test_vstack_2_COMPLEX()
        {
            var a = np.array(new Complex[,] { { 1 }, { 2 }, { 3 } });
            var b = np.array(new Complex[,] { { 2 }, { 3 }, { 4 } });
            var c = np.vstack(new object[] { a, b });

            AssertArray(c, new Complex[,] { { 1 }, { 2 }, { 3 }, { 2 }, { 3 }, { 4 } });

            print(c);
        }

        [TestMethod]
        public void test_hstack_2_COMPLEX()
        {
            var a = np.array(new Complex[,] { { 1 }, { 2 }, { 3 } });
            var b = np.array(new Complex[,] { { 2 }, { 3 }, { 4 } });
            var c = np.hstack(new object[] { a, b });

            AssertArray(c, new Complex[,] { { 1, 2 }, { 2, 3 }, { 3, 4 } });

            print(c);
        }

        [TestMethod]
        public void test_stack_1_COMPLEX()
        {
            var a = np.array(new Complex[,] { { 1 }, { 2 }, { 3 } });
            var b = np.array(new Complex[,] { { 2 }, { 3 }, { 4 } });

            var c = np.stack(new object[] { a, b }, axis: 0);
            AssertArray(c, new Complex[,,] { { { 1 }, { 2 }, { 3 } }, { { 2 }, { 3 }, { 4 } } });
            print(c);
            print("**************");

            var d = np.stack(new object[] { a, b }, axis: 1);
            AssertArray(d, new Complex[,,] { { { 1 }, { 2 } }, { { 2 }, { 3 } }, { { 3 }, { 4 } } });
            print(d);
            print("**************");

            var e = np.stack(new object[] { a, b }, axis: 2);
            AssertArray(e, new Complex[,,] { { { 1, 2 } }, { { 2, 3 } }, { { 3, 4 } } });
            print(e);

        }

        [TestMethod]
        public void test_block_2_COMPLEX()
        {
            var a = np.array(new Complex[] { 1, 2, 3 });
            var b = np.array(new Complex[] { 2, 3, 4 });
            var c = np.block(new object[] { a, b, 10 });    // hstack([a, b, 10])

            AssertArray(c, new Complex[] { 1, 2, 3, 2, 3, 4, 10 });
            print(c);
            print("**************");

            a = np.array(new Complex[] { 1, 2, 3 });
            b = np.array(new Complex[] { 2, 3, 4 });
            c = np.block(new object[] { new object[] { a }, new object[] { b } });    // vstack([a, b])

            AssertArray(c, new Complex[,] { { 1, 2, 3 }, { 2, 3, 4 } });
            print(c);

        }

        [TestMethod]
        public void test_expand_dims_1_COMPLEX()
        {
            var a = np.array(new Complex[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }).reshape(new shape(2, -1, 2));
            var b = np.expand_dims(a, axis: 0);

            var ExpectedDataB = new Complex[,,,]
            {{{{1,  2}, {3,  4}, {5,  6}},
              {{7,  8}, {9, 10}, {11, 12}}}};

            AssertArray(b, ExpectedDataB);
            print(b);
            print("**************");

            var c = np.expand_dims(a, axis: 1);
            var ExpectedDataC = new Complex[,,,]
                {{{{1,  2}, {3,  4}, {5,  6}}},
                {{{ 7,  8},{ 9, 10}, {11, 12}}}};
            AssertArray(c, ExpectedDataC);
            print(c);
            print("**************");

            var d = np.expand_dims(a, axis: 2);
            var ExpectedDataD = new Complex[,,,]
            {{{{1,  2}},{{3,  4}},{{5,  6}}},
             {{{7,  8}},{{9, 10}},{{11, 12}}}};

            AssertArray(d, ExpectedDataD);
            print(d);

        }

        [TestMethod]
        public void test_column_stack_1_COMPLEX()
        {
            var a = np.array(new Complex[] { 1, 2, 3 });
            var b = np.array(new Complex[] { 2, 3, 4 });
            var c = np.column_stack(new object[] { a, b });

            AssertArray(c, new Complex[,] { { 1, 2 }, { 2, 3 }, { 3, 4 } });
            print(c);
        }

        [TestMethod]
        public void test_row_stack_1_COMPLEX()
        {
            var a = np.array(new Complex[] { 1, 2, 3 });
            var b = np.array(new Complex[] { 2, 3, 4 });
            var c = np.row_stack(new object[] { a, b });

            AssertArray(c, new Complex[,] { { 1, 2, 3 }, { 2, 3, 4 } });

            print(c);
        }

        [TestMethod]
        public void test_dstack_1_COMPLEX()
        {
            var a = np.array(new Complex[] { 1, 2, 3 });
            var b = np.array(new Complex[] { 2, 3, 4 });
            var c = np.dstack(new object[] { a, b });

            AssertArray(c, new Complex[,,] { { { 1, 2 }, { 2, 3 }, { 3, 4 } } });
            print(c);

            a = np.array(new Complex[,] { { 1 }, { 2 }, { 3 } });
            b = np.array(new Complex[,] { { 2 }, { 3 }, { 4 } });
            c = np.dstack(new object[] { a, b });

            AssertArray(c, new Complex[,,] { { { 1, 2 } }, { { 2, 3 } }, { { 3, 4 } } });

            print(c);
        }

        [TestMethod]
        public void test_array_split_2_COMPLEX()
        {
            var x = np.arange(16.0, dtype: np.Complex).reshape(new shape(2, 8, 1));
            var y = np.array_split(x, 3, axis: 0);


            AssertArray(y.ElementAt(0), new Complex[,,] { { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 } } });
            AssertArray(y.ElementAt(1), new Complex[,,] { { { 8 }, { 9 }, { 10 }, { 11 }, { 12 }, { 13 }, { 14 }, { 15 } } });
            AssertShape(y.ElementAt(2), 0, 8, 1);

            print(y);

            print("**************");

            x = np.arange(16.0, dtype: np.Complex).reshape(new shape(2, 8, 1));
            y = np.array_split(x, 3, axis: 1);

            AssertArray(y.ElementAt(0), new Complex[,,] { { { 0 }, { 1 }, { 2 } }, { { 8 }, { 9 }, { 10 } } });
            AssertArray(y.ElementAt(1), new Complex[,,] { { { 3 }, { 4 }, { 5 } }, { { 11 }, { 12 }, { 13 } } });
            AssertArray(y.ElementAt(2), new Complex[,,] { { { 6 }, { 7 } }, { { 14 }, { 15 } } });


            print(y);

            print("**************");

            x = np.arange(16.0, dtype: np.Complex).reshape(new shape(2, 8, 1));
            y = np.array_split(x, 3, axis: 2);

            AssertArray(y.ElementAt(0), new Complex[,,] { { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 } }, { { 8 }, { 9 }, { 10 }, { 11 }, { 12 }, { 13 }, { 14 }, { 15 } } });
            AssertShape(y.ElementAt(1), 2, 8, 0);
            AssertShape(y.ElementAt(2), 2, 8, 0);
            print(y);
        }

        [TestMethod]
        public void test_split_2_COMPLEX()
        {
            var x = np.arange(16.0, dtype: np.Complex).reshape(new shape(8, 2, 1));
            var y = np.split(x, new Int32[] { 2, 3 }, axis: 0);

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new Complex[,,] { { { 0 }, { 1 } }, { { 2 }, { 3 } } });
            AssertArray(y.ElementAt(1), new Complex[,,] { { { 4 }, { 5 } } });
            AssertArray(y.ElementAt(2), new Complex[,,] { { { 6 }, { 7 } }, { { 8 }, { 9 } }, { { 10 }, { 11 } }, { { 12 }, { 13 } }, { { 14 }, { 15 } } });


            print(y);

            print("**************");

            x = np.arange(16.0, dtype: np.Complex).reshape(new shape(8, 2, 1));
            y = np.split(x, new int[] { 2, 3 }, axis: 1);

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new Complex[,,] {{{0},{1}},{{2}, {3}}, {{4}, {5}}, {{6}, { 7}},
                                                        {{8},{9}},{{10},{11}}, {{12}, {13}}, {{14}, {15}}});
            AssertShape(y.ElementAt(1), 8, 0, 1);
            AssertShape(y.ElementAt(2), 8, 0, 1);

            print(y);

            print("**************");

            x = np.arange(16.0, dtype: np.Complex).reshape(new shape(8, 2, 1));
            y = np.split(x, new int[] { 2, 3 }, axis: 2);

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new Complex[,,] {{{ 0},{ 1}},{{ 2}, { 3}}, {{ 4}, { 5}}, {{ 6}, { 7}},
                                                        {{ 8},{ 9}},{{10}, {11}}, {{12}, {13}}, {{14}, {15}}});
            AssertShape(y.ElementAt(1), 8, 2, 0);
            AssertShape(y.ElementAt(2), 8, 2, 0);

            print(y);
        }

        [TestMethod]
        public void test_hsplit_2_COMPLEX()
        {
            var x = np.arange(8, dtype: np.Complex).reshape(new shape(2, 2, 2));
            var y = np.hsplit(x, 2);

            Assert.AreEqual(2, y.Count);
            AssertArray(y.ElementAt(0), new Complex[,,] { { { 0, 1 } }, { { 4, 5 } } });
            AssertArray(y.ElementAt(1), new Complex[,,] { { { 2, 3 } }, { { 6, 7 } } });
            print(y);

            print("**************");

            x = np.arange(8, dtype: np.Complex).reshape(new shape(2, 2, 2));
            y = np.hsplit(x, new Int32[] { 3, 6 });

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new Complex[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } });
            AssertShape(y.ElementAt(1), 2, 0, 2);
            AssertShape(y.ElementAt(2), 2, 0, 2);

            print(y);
        }

        [TestMethod]
        public void test_vsplit_2_COMPLEX()
        {
            var x = np.arange(8, dtype: np.Complex).reshape(new shape(2, 2, 2));
            var y = np.vsplit(x, 2);

            Assert.AreEqual(2, y.Count);
            AssertArray(y.ElementAt(0), new Complex[,,] { { { 0, 1 }, { 2, 3 } } });
            AssertArray(y.ElementAt(1), new Complex[,,] { { { 4, 5 }, { 6, 7 } } });
            print(y);

            print("**************");

            x = np.arange(8, dtype: np.Complex).reshape(new shape(2, 2, 2));
            y = np.vsplit(x, new int[] { 3, 6 });

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new Complex[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } });
            AssertShape(y.ElementAt(1), 0, 2, 2);
            AssertShape(y.ElementAt(2), 0, 2, 2);

            print(y);
        }

        [TestMethod]
        public void test_dsplit_1_COMPLEX()
        {
            var x = np.arange(16, dtype: np.Complex).reshape(new shape(2, 2, 4));
            var y = np.dsplit(x, 2);

            Assert.AreEqual(2, y.Count);
            AssertArray(y.ElementAt(0), new Complex[,,] { { { 0, 1 }, { 4, 5 } }, { { 8, 9 }, { 12, 13 } } });
            AssertArray(y.ElementAt(1), new Complex[,,] { { { 2, 3 }, { 6, 7 } }, { { 10, 11 }, { 14, 15 } } });
            print(y);


            print("**************");

            x = np.arange(16, dtype: np.Complex).reshape(new shape(2, 2, 4));
            y = np.dsplit(x, new int[] { 3, 6 });

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new Complex[,,] { { { 0, 1, 2 }, { 4, 5, 6 } }, { { 8, 9, 10 }, { 12, 13, 14 } } });
            AssertArray(y.ElementAt(1), new Complex[,,] { { { 3 }, { 7 } }, { { 11 }, { 15 } } });
            AssertShape(y.ElementAt(2), 2, 2, 0);

            print(y);
        }

        [TestMethod]
        public void test_kron_1_COMPLEX()
        {

            var a = np.kron(new Complex[] { 1, 10, 100 }, new Complex[] { 5, 6, 7 });
            AssertArray(a, new Complex[] { 5, 6, 7, 50, 60, 70, 500, 600, 700 });
            print(a);

            var b = np.kron(new Complex[] { 5, 6, 7 }, new Complex[] { 1, 10, 100 });
            AssertArray(b, new Complex[] { 5, 50, 500, 6, 60, 600, 7, 70, 700 });
            print(b);

            var x = np.array(new Complex[,] { { 2, 3 }, { 4, 5 } });
            var y = np.array(new Complex[,] { { 5, 6 }, { 7, 8 } });

            var c = np.kron(x, y);
            AssertArray(c, new Complex[,] { { 10, 12, 15, 18 }, { 14, 16, 21, 24 }, { 20, 24, 25, 30 }, { 28, 32, 35, 40 } });
            print(c);
            print(c.shape);

            c = np.kron(np.eye(2, dtype: np.Complex), np.ones(new shape(2, 2), dtype: np.Complex));
            AssertArray(c, new Complex[,] { { 1, 1, 0, 0 }, { 1, 1, 0, 0 }, { 0, 0, 1, 1 }, { 0, 0, 1, 1 } });


            x = np.array(new Complex[,,] { { { 2, 3, 3 }, { 4, 5, 3 } } });
            y = np.array(new Complex[,,] { { { 5, 6, 6, 6 }, { 7, 8, 6, 6 } } });

            c = np.kron(x, y);
            AssertArray(c, new Complex[,,] { { { 10, 12, 12, 12, 15, 18, 18, 18, 15, 18, 18, 18 },
                                           { 14, 16, 12, 12, 21, 24, 18, 18, 21, 24, 18, 18 },
                                           { 20, 24, 24, 24, 25, 30, 30, 30, 15, 18, 18, 18 },
                                           { 28, 32, 24, 24, 35, 40, 30, 30, 21, 24, 18, 18 } } });
            print(c);
            print(c.shape);


            var d = np.kron(np.ones((5, 7, 9, 11), dtype: np.Int32), np.ones((3, 4, 6, 8), dtype: np.Int32));
            AssertShape(d, 15, 28, 54, 88);
            print(d.shape);

        }

        [TestMethod]
        public void test_tile_2_COMPLEX()
        {
            var a = np.array(new Complex[,] { { 1, 2 }, { 3, 4 } });
            var b = np.tile(a, 2);
            AssertArray(b, new Complex[,] { { 1, 2, 1, 2 }, { 3, 4, 3, 4 } });
            print(b);
            print("**************");

            var c = np.tile(a, new Int16[] { 2, 1 });
            AssertArray(c, new Complex[,] { { 1, 2 }, { 3, 4 }, { 1, 2 }, { 3, 4 } });
            print(c);
            print("**************");

            var d = np.array(new Complex[] { 1, 2, 3, 4 });
            var e = np.tile(d, new float[] { 4, 1 });

            AssertArray(e, new Complex[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 } });
            print(e);

            try
            {
                var f = np.array(new Complex[] { 1, 2, 3, 4 });
                var g = np.tile(d, new Complex[] { 4, 1 });

                AssertArray(g, new Complex[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 } });
                print(e);
                Assert.Fail("This should have thrown an exception");
            }
            catch (Exception ex)
            {

            }
     


        }

        #endregion

        #region from UFUNCTests

        [TestMethod]
        public void test_UFUNC_AddReduce_1_COMPLEX()
        {
            var x = np.arange(8, dtype: np.Complex);

            var a = np.ufunc.reduce(UFuncOperation.add, x);
            Assert.AreEqual((Complex)28m, a.GetItem(0));
            print(a);

            x = np.arange(8, dtype: np.Complex).reshape((2, 2, 2));
            var b = np.ufunc.reduce(UFuncOperation.add, x);
            AssertArray(b, new Complex[,] { { 4, 6 }, { 8, 10 } });
            print(b);

            var c = np.ufunc.reduce(UFuncOperation.add, x, 0);
            AssertArray(c, new Complex[,] { { 4, 6 }, { 8, 10 } });
            print(c);

            var d = np.ufunc.reduce(UFuncOperation.add, x, 1);
            AssertArray(d, new Complex[,] { { 2, 4 }, { 10, 12 } });
            print(d);

            var e = np.ufunc.reduce(UFuncOperation.add, x, 2);
            AssertArray(e, new Complex[,] { { 1, 5 }, { 9, 13 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_AddAccumulate_1_COMPLEX()
        {
            var x = np.arange(8, dtype: np.Complex);

            var a = np.ufunc.accumulate(UFuncOperation.add, x);
            AssertArray(a, new Complex[] { 0, 1, 3, 6, 10, 15, 21, 28 });
            print(a);

            x = np.arange(8, dtype: np.Complex).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(UFuncOperation.add, x);
            AssertArray(b, new Complex[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 6 }, { 8, 10 } } });
            print(b);

            var c = np.ufunc.accumulate(UFuncOperation.add, x, 0);
            AssertArray(c, new Complex[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 6 }, { 8, 10 } } });
            print(c);

            var d = np.ufunc.accumulate(UFuncOperation.add, x, 1);
            AssertArray(d, new Complex[,,] { { { 0, 1 }, { 2, 4 } }, { { 4, 5 }, { 10, 12 } } });
            print(d);

            var e = np.ufunc.accumulate(UFuncOperation.add, x, 2);
            AssertArray(e, new Complex[,,] { { { 0, 1 }, { 2, 5 } }, { { 4, 9 }, { 6, 13 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_AddReduceAt_1_COMPLEX()
        {
            var a = np.ufunc.reduceat(UFuncOperation.add, np.arange(8, dtype: np.Complex), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new Complex[] { 6, 10, 14, 18 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16, dtype: np.Complex).reshape((4, 4));
            var b = np.ufunc.reduceat(UFuncOperation.add, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new Complex[,] {{12.0, 15.0, 18.0, 21.0},{12.0, 13.0, 14.0, 15.0}, {4.0, 5.0, 6.0, 7.0},
                                          {8.0, 9.0, 10.0, 11.0}, {24.0, 28.0, 32.0, 36.0}});
            print(b);

            var c = np.ufunc.reduceat(UFuncOperation.multiply, x, new npy_intp[] { 0, 3 }, axis: 1);
            AssertArray(c, new Complex[,] { { 0.0, 3.0 }, { 120.0, 7.0 }, { 720.0, 11.0 }, { 2184.0, 15.0 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_AddOuter_1_COMPLEX()
        {
            var x = np.arange(4, dtype: np.Complex);

            var a = np.ufunc.outer(UFuncOperation.add, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new Complex[,] { { 0, 1, 2, 3 }, { 1, 2, 3, 4 }, { 2, 3, 4, 5 }, { 3, 4, 5, 6 } });
            print(a);

            x = np.arange(6, dtype: np.Complex).reshape((3, 2));
            var y = np.arange(6, dtype: np.Complex).reshape((2, 3));
            var b = np.ufunc.outer(UFuncOperation.add, null, x, y);
            AssertShape(b, 3, 2, 2, 3);
            print(b.shape);

            var ExpectedDataB = new Complex[,,,]

                {{{{0,  1,  2}, {3,  4,  5}}, {{1,  2,  3}, { 4,  5,  6}}},
                 {{{2,  3,  4}, {5,  6,  7}}, {{3,  4,  5}, { 6,  7,  8}}},
                 {{{4,  5,  6}, {7,  8,  9}}, {{5,  6,  7}, { 8,  9, 10}}}};

            AssertArray(b, ExpectedDataB);

            print(b);
        }

        #endregion

        #region from IndexTricksTests

        [TestMethod]
        public void test_mgrid_1_COMPLEX()
        {
            var a = (ndarray)np.mgrid(new Slice[] { new Slice((Complex)0, (Complex)5) });
            print(a);
            AssertArray(a, new Complex[] { 0, 1, 2, 3, 4 });
            print("************");

            var b = (ndarray)np.mgrid(new Slice[] { new Slice((Complex)0.0, (Complex)5.5) });
            print(b);
            AssertArray(b, new Complex[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
            print("************");

            var c = (ndarray)np.mgrid(new Slice[] { new Slice((Complex)0, (Complex)5), new Slice((Complex)0, (Complex)5) });
            print(c);

            var ExpectedCArray = new Complex[,,]
                {{{0, 0, 0, 0, 0},  {1, 1, 1, 1, 1},  {2, 2, 2, 2, 2},  {3, 3, 3, 3, 3},  {4, 4, 4, 4, 4}},
                 {{0, 1, 2, 3, 4},  {0, 1, 2, 3, 4},  {0, 1, 2, 3, 4},  {0, 1, 2, 3, 4},  {0, 1, 2, 3, 4}}};
            AssertArray(c, ExpectedCArray);


            print("************");

            var d = (ndarray)np.mgrid(new Slice[] { new Slice((Complex)0, (Complex)5.5), new Slice((Complex)0, (Complex)5.5) });
            print(d);
            var ExpectedDArray = new Complex[,,]
                {{{0, 0, 0, 0, 0, 0},  {1, 1, 1, 1, 1, 1},  {2, 2, 2, 2, 2, 2},  {3, 3, 3, 3, 3, 3},  {4, 4, 4, 4, 4, 4}, {5, 5, 5, 5, 5, 5}},
                 {{0, 1, 2, 3, 4, 5},  {0, 1, 2, 3, 4, 5},  {0, 1, 2, 3, 4, 5},  {0, 1, 2, 3, 4, 5},  {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}}};
            AssertArray(d, ExpectedDArray);

            print("************");

            var e = (ndarray)np.mgrid(new Slice[] { new Slice((Complex)3, (Complex)5), new Slice((Complex)4, (Complex)6), new Slice((Complex)2, (Complex)4.2) });
            print(e);
            var ExpectedEArray = new Complex[,,,]
                {
                    {{{3, 3, 3}, {3, 3, 3}}, {{4, 4, 4}, {4, 4, 4}}},
                    {{{4, 4, 4}, {5, 5, 5}}, {{4, 4, 4}, {5, 5, 5}}},
                    {{{2, 3, 4}, {2, 3, 4}}, {{2, 3, 4}, {2, 3, 4}}},
                };
            AssertArray(e, ExpectedEArray);

        }

        [TestMethod]
        public void test_ogrid_1_COMPLEX_TODO()
        {
            var a = (ndarray)np.ogrid(new Slice[] { new Slice(0m, 5m) });
            print(a);
            AssertArray(a, new decimal[] { 0, 1, 2, 3, 4 });
            print("************");

            var b = (ndarray)np.ogrid(new Slice[] { new Slice(0.0m, 5.5m) });
            print(b);
            AssertArray(b, new decimal[] { 0.0m, 1.0m, 2.0m, 3.0m, 4.0m, 5.0m });
            print("************");

            var c = (ndarray[])np.ogrid(new Slice[] { new Slice(0m, 5m), new Slice(0m, 5m) });
            print(c);
            AssertArray(c[0], new decimal[,] { { 0 }, { 1 }, { 2 }, { 3 }, { 4 } });
            AssertArray(c[1], new decimal[,] { { 0, 1, 2, 3, 4 } });


            print("************");

            var d = (ndarray[])np.ogrid(new Slice[] { new Slice(0m, 5.5m), new Slice(0m, 5.5m) });
            print(d);
            AssertArray(d[0], new decimal[,] { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
            AssertArray(d[1], new decimal[,] { { 0, 1, 2, 3, 4, 5 } });

            print("************");

            var e = (ndarray[])np.ogrid(new Slice[] { new Slice(3m, 5m), new Slice(4m, 6m), new Slice(2m, 4.2m) });
            print(e);
            AssertArray(e[0], new decimal[,,] { { { 3 } }, { { 4 } } });
            AssertArray(e[1], new decimal[,,] { { { 4 }, { 5 } } });
            AssertArray(e[2], new decimal[,,] { { { 2, 3, 4 } } });

        }

        [TestMethod]
        public void test_fill_diagonal_1_COMPLEX()
        {
            var a = np.zeros((3, 3), np.Complex);
            np.fill_diagonal(a, 5);
            AssertArray(a, new Complex[,] { { 5, 0, 0 }, { 0, 5, 0 }, { 0, 0, 5 } });
            print(a);

            a = np.zeros((3, 3, 3, 3), np.Complex);
            np.fill_diagonal(a, 4);
            AssertArray(a[0, 0] as ndarray, new Complex[,] { { 4, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } });
            print(a[0, 0]);
            AssertArray(a[1, 1] as ndarray, new Complex[,] { { 0, 0, 0 }, { 0, 4, 0 }, { 0, 0, 0 } });
            print(a[1, 1]);
            AssertArray(a[2, 2] as ndarray, new Complex[,] { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 4 } });
            print(a[2, 2]);

            // tall matrices no wrap
            a = np.zeros((5, 3), np.Complex);
            np.fill_diagonal(a, 4);
            AssertArray(a, new Complex[,] { { 4, 0, 0 }, { 0, 4, 0 }, { 0, 0, 4 }, { 0, 0, 0 }, { 0, 0, 0 } });
            print(a);

            // tall matrices wrap
            a = np.zeros((5, 3), np.Complex);
            np.fill_diagonal(a, 4, wrap: true);
            AssertArray(a, new Complex[,] { { 4, 0, 0 }, { 0, 4, 0 }, { 0, 0, 4 }, { 0, 0, 0 }, { 4, 0, 0 } });
            print(a);

            // wide matrices wrap
            a = np.zeros((3, 5), np.Complex);
            np.fill_diagonal(a, 4, wrap: true);
            AssertArray(a, new Complex[,] { { 4, 0, 0, 0, 0 }, { 0, 4, 0, 0, 0 }, { 0, 0, 4, 0, 0 } });
            print(a);


        }

        [TestMethod]
        public void test_diag_indices_1_COMPLEX()
        {
            var di = np.diag_indices(4);
            AssertArray(di[0], new Int32[] { 0, 1, 2, 3 });
            AssertArray(di[1], new Int32[] { 0, 1, 2, 3 });
            print(di);

            var a = np.arange(16, dtype: np.Complex).reshape((4, 4));
            a[di] = 100;

            AssertArray(a, new Complex[,] { { 100, 1, 2, 3 }, { 4, 100, 6, 7 }, { 8, 9, 100, 11 }, { 12, 13, 14, 100 } });
            print(a);

            return;

        }

        [TestMethod]
        public void test_diag_indices_from_1_COMPLEX()
        {
            var a = np.arange(16, dtype: np.Complex).reshape((4, 4));
            var di = np.diag_indices_from(a);
            AssertArray(di[0], new Int32[] { 0, 1, 2, 3 });
            AssertArray(di[1], new Int32[] { 0, 1, 2, 3 });
            print(di);
        }

        #endregion

        #region from StrideTricksTests

        [TestMethod]
        public void test_broadcast_1_COMPLEX()
        {
            var x = np.array(new Complex[,] { { 11 }, { 2 }, { 3 } });
            var y = np.array(new Complex[] { 4, 5, 6 });
            var b = np.broadcast(x, y);
            Assert.AreEqual(b.shape.iDims.Length, 2);
            Assert.AreEqual(b.shape.iDims[0], 3);
            Assert.AreEqual(b.shape.iDims[1], 3);
            print(b.shape);

            Assert.AreEqual(b.index, 0);
            print(b.index);

            foreach (var uv in b)
            {
                print(uv);
            }
            Assert.AreEqual(b.index, 9);
            print(b.index);

        }

        [TestMethod]
        public void test_broadcast_to_1_COMPLEX()
        {
            var a = np.broadcast_to((Complex)5, (4, 4));
            AssertArray(a, new Complex[,] { { 5, 5, 5, 5 }, { 5, 5, 5, 5 }, { 5, 5, 5, 5 }, { 5, 5, 5, 5 } });
            AssertStrides(a, 0, 0);
            print(a);
            print(a.shape);
            print(a.strides);
            print("*************");


            var b = np.broadcast_to(new Complex[] { 1, 2, 3 }, (3, 3));
            AssertArray(b, new Complex[,] { { 1, 2, 3 }, { 1, 2, 3 }, { 1, 2, 3 } });
            AssertStrides(b, 0, 16);
            print(b);
            print(b.shape);
            print(b.strides);
            print("*************");


        }

        [TestMethod]
        public void test_broadcast_arrays_1_COMPLEX()
        {
            var x = np.array(new Complex[,] { { 1, 2, 3 } });
            var y = np.array(new Complex[,] { { 4 }, { 5 } });
            var z = np.broadcast_arrays(false, new ndarray[] { x, y });

            print(z);

        }

        [TestMethod]
        public void test_as_strided_1_COMPLEX()
        {
            var y = np.zeros((10, 10), np.Complex);
            AssertStrides(y, 160, 16);
            print(y.strides);

            var n = 1000;
            var a = np.arange(n, dtype: np.Complex);

            var b = np.as_strided(a, (n, n), (0, 8));

            //print(b);

            Assert.AreEqual(1000000, b.size);
            print(b.size);
            AssertShape(b, 1000, 1000);
            print(b.shape);
            AssertStrides(b, 0, 8);
            print(b.strides);
            Assert.AreEqual(16000000, b.nbytes);
            print(b.nbytes);

        }

        #endregion

        #region from IteratorTests

        [TestMethod]
        public void test_nditer_1_COMPLEX()
        {
            var a = np.arange(0.1, 6.1, dtype: np.Complex).reshape((2, 3));
            var b = np.array(new Complex[] { 7, 8, 9 });

            foreach (var aa in new nditer(a))
            {
                print(aa);
            }

            foreach (var aa in new nditer((a, b)))
            {
                print(aa);
            }

            foreach (var aa in new nditer((a, b, a, b)))
            {
                print(aa);
            }

        }

        [TestMethod]
        public void test_ndindex_1_COMPLEX()
        {
            var a = np.arange(0.1, 6.1, dtype: np.Complex).reshape((2, 3)); 

            foreach (var aa in new ndindex((2, 3)))
            {
                print(aa);
            }


            foreach (var aa in new ndindex((2, 3, 2)))
            {
                print(aa);
            }

            foreach (var aa in new ndindex((3)))
            {
                print(aa);
            }

        }

        [TestMethod]
        public void test_ndenumerate_1_COMPLEX()
        {
            var a = np.arange(0.1, 6.1, dtype: np.Complex).reshape((2, 3));

            foreach (ValueTuple<npy_intp[], object> aa in new ndenumerate(a))
            {
                print(aa.Item1);
                print(aa.Item2);
            }
        }

        #endregion

        #region COMPLEX number specific tests
        [TestMethod]
        public void test_angle_1_COMPLEX()
        {
            var a = np.angle(new Complex[] { new Complex(1.0, 0), new Complex(0, 1), new Complex(1, 1) });
            print(a);
            AssertArray(a, new double[] { 0, 1.57079633, 0.78539816 });

            var b = np.angle(new Complex[] { new Complex(1.0, 1) }, deg: true);
            print(b);
            Assert.AreEqual((double)45, b.GetItem(0));

            var c = np.angle(new int[] { -1, 2, -3 });
            print(c);
            AssertArray(c, new double[] { 3.14159265, 0.0, 3.14159265 });
        }

        [TestMethod]
        public void test_real_1_COMPLEX()
        {
            var x1 = np.array(new Complex[,] { { 0, 2 }, { 1, 1 }, { 2, 0 } }).T;

            var Real = x1.Real;
            print(Real);
            AssertArray(Real, new double[,] { { 0, 1, 2 }, { 2, 1, 0} });

        }

        [TestMethod]
        public void test_image_1_COMPLEX()
        {
            var x1 = np.array(new Complex[,] { { new Complex(0, 123), new Complex(2, 234) }, { new Complex(1,789), new Complex(1,678) }, { new Complex(2, 456), new Complex(0, 222) } }).T;

            var Imag = x1.Imag;

            print(Imag);

            AssertArray(Imag, new double[,] { { 123, 789, 456 }, { 234, 678, 222 } });
        }

        [TestMethod]
        public void test_conj_1_COMPLEX()
        {
            var a = np.arange(new Complex(0, -10), new Complex(10, 0), dtype: np.Complex);
            print(a);
            AssertArray(a, new Complex[] { new Complex(0, -10), new Complex(1, -10), new Complex(2, -10), new Complex(3, -10), new Complex(4, -10),
                                           new Complex(5, -10), new Complex(6, -10), new Complex(7, -10), new Complex(8, -10), new Complex(9, -10), });

            var b = np.conj(a);
            print(b);
            AssertArray(b, new Complex[] { new Complex(0, 10), new Complex(1, 10), new Complex(2, 10), new Complex(3, 10), new Complex(4, 10),
                                           new Complex(5, 10), new Complex(6, 10), new Complex(7, 10), new Complex(8, 10), new Complex(9, 10), });


            var c = np.conjugate(a);
            print(c);
            AssertArray(c, new Complex[] { new Complex(0, 10), new Complex(1, 10), new Complex(2, 10), new Complex(3, 10), new Complex(4, 10),
                                           new Complex(5, 10), new Complex(6, 10), new Complex(7, 10), new Complex(8, 10), new Complex(9, 10), });


        }

        [TestMethod]
        public void test_sort_complex_1()
        {
            var IntTestData = new Int32[] { 5, 3, 6, 2, 1 };
            var ComplexTextData = new Complex[] { new Complex(3, -2), new Complex(1, 2), new Complex(2,-1), new Complex(3,-3),new Complex(3,5) };
            var ComplexSortedData = new Complex[] { new Complex(1, 2), new Complex(2, -1), new Complex(3, -3), new Complex(3, -2),  new Complex(3, 5) };

            var a = np.sort(IntTestData);
            print(a);
            AssertArray(a, new Int32[] { 1,2,3,5,6});

            var b = np.sort_complex(IntTestData);
            print(b);
            AssertArray(b, new Complex[] { 1, 2, 3, 5, 6 });

            var c = np.sort(ComplexTextData);
            print(c);
            AssertArray(c, ComplexSortedData);

            var d = np.sort_complex(ComplexTextData);
            print(d);
            AssertArray(d, ComplexSortedData);

        }

#if NOT_PLANNING_TODO
        [Ignore]
        [TestMethod]
        public void test_real_if_close_1_COMPLEX()
        {

        }
#endif
        #endregion

    }
}
