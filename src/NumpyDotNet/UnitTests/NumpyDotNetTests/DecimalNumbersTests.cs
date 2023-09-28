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
    public class DecimalNumbersTests : TestBaseClass
    {
        #region from ArrayCreationTests
        [TestMethod]
        public void test_asfarray_DECIMAL()
        {
            var a = np.asfarray(new decimal[] { 2, 3 });
            AssertArray(a, new double[] { 2, 3 });
            print(a);

            var b = np.asfarray(new decimal[] { 2, 3 }, dtype: np.Float32);
            AssertArray(b, new float[] { 2, 3 });
            print(b);

            var c = np.asfarray(new decimal[] { 2, 3 }, dtype: np.Int8);
            AssertArray(c, new double[] { 2, 3 });
            print(c);


            return;
        }

        [TestMethod]
        public void test_copy_1_DECIMAL()
        {
            var x = np.array(new decimal[] { 1, 2, 3 });
            var y = x;

            var z = np.copy(x);

            // Note that, when we modify x, y changes, but not z:

            x[0] = 10;

            Assert.AreEqual(10m, y[0]);

            Assert.AreEqual(1m, z[0]);

            return;
        }

        [TestMethod]
        public void test_linspace_1_DECIMAL()
        {
            decimal retstep = 0;

            var a = np.linspace(2.0m, 3.0m, ref retstep, num: 5);
            AssertArray(a, new decimal[] { 2.0m, 2.25m, 2.5m, 2.75m, 3.0m });
            print(a);

            var b = np.linspace(2.0m, 3.0m, ref retstep, num: 5, endpoint: false);
            AssertArray(b, new decimal[] { 2.0m, 2.2m, 2.4m, 2.6m, 2.8m });
            print(b);

            var c = np.linspace(2.0m, 3.0m, ref retstep, num: 5);
            AssertArray(c, new decimal[] { 2.0m, 2.25m, 2.5m, 2.75m, 3.0m });
            print(c);
        }

        [TestMethod]
        public void test_logspace_1_DECIMAL()
        {
            var a = np.logspace(2.0m, 3.0m, num: 4);
            AssertArray(a, new decimal[] { 100, 215.44346900318800000000000000000m, 464.15888336127800000000000000000m, 1000 });
            print(a);

            var b = np.logspace(2.0m, 3.0m, num: 4, endpoint: false);
            AssertArray(b, new decimal[] { 100, 177.82794100389200000000000000000m, 316.22776601683800000000000000000m, 562.34132519034900000000000000000m });
            print(b);

            var c = np.logspace(2.0m, 3.0m, num: 4, _base: 2.0);
            AssertArray(c, new decimal[] { 4, 05.03968419957949000000000000000m, 06.34960420787280000000000000000m, 8 });
            print(c);
        }

        [TestMethod]
        public void test_geomspace_1_DECIMAL()
        {
            var a = np.geomspace(1m, 1000m, num: 4);
            AssertArray(a, new decimal[] { 1.0m, 10.0m, 100.0m, 1000.0m });
            print(a);

            var b = np.geomspace(1m, 1000m, num: 3, endpoint: false);
            AssertArray(b, new decimal[] { 1.0m, 10.0m, 100.0m });
            print(b);

            var c = np.geomspace(1m, 1000m, num: 4, endpoint: false);
            AssertArray(c, new decimal[] { 1, 05.62341325190349000000000000000m, 31.62277660168380000000000000000m, 177.82794100389200000000000000000m });
            print(c);

            var d = np.geomspace(1m, 256m, num: 9);
            AssertArray(d, new decimal[] { 1.0m, 2.0m, 4.0m, 8.0m, 16.0m, 32.0m, 64.0m, 128.0m, 256.0m });
            print(d);
        }

        [TestMethod]
        public void test_meshgrid_1_DECIMAL()
        {
            int nx = 3;
            int ny = 2;

            decimal ret = 0;

            var x = np.linspace(0m, 1m, ref ret, nx);
            var y = np.linspace(0m, 1m, ref ret, ny);

            ndarray[] xv = np.meshgrid(new ndarray[] { x });
            AssertArray(xv[0], new decimal[] { 0.0m, 0.5m, 1.0m });
            print(xv[0]);

            print("************");

            ndarray[] xyv = np.meshgrid(new ndarray[] { x, y });
            AssertArray(xyv[0], new decimal[,] { { 0.0m, 0.5m, 1.0m }, { 0.0m, 0.5m, 1.0m } });
            AssertArray(xyv[1], new decimal[,] { { 0.0m, 0.0m, 0.0m }, { 1.0m, 1.0m, 1.0m } });

            print(xyv[0]);
            print(xyv[1]);

            print("************");

            xyv = np.meshgrid(new ndarray[] { x, y }, sparse: true);
            AssertArray(xyv[0], new decimal[,] { { 0.0m, 0.5m, 1.0m } });
            AssertArray(xyv[1], new decimal[,] { { 0.0m }, { 1.0m } });

            print(xyv[0]);
            print(xyv[1]);

            print("************");


        }

        [TestMethod]
        public void test_OneDimensionalArray_DECIMAL()
        {
            decimal[] l = new decimal[] { 12.23m, 13.32m, 100m, 36.32m };
            print("Original List:", l);
            var a = np.array(l);
            print("One-dimensional numpy array: ", a);
            print(a.shape);
            print(a.strides);

            AssertArray(a, l);
            AssertShape(a, 4);
            AssertStrides(a, sizeof(decimal));
        }

        [TestMethod]
        public void test_reverse_array_DECIMAL()
        {
            var x = np.arange(0, 40, dtype: np.Decimal);
            print("Original array:");
            print(x);
            print("Reverse array:");
            //x = (ndarray)x[new Slice(null, null, -1)];
            x = (ndarray)x["::-1"];
            print(x);

            AssertArray(x, new decimal[] { 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 });
            AssertShape(x, 40);
            AssertStrides(x, -sizeof(decimal));

            var y = x + 100;
            print(y);

            var z = x.reshape((5, -1));
            print(z);
        }

        [TestMethod]
        public void test_checkerboard_1_DECIMAL()
        {
            var x = np.ones((3, 3), dtype: np.Decimal);
            print("Checkerboard pattern:");
            x = np.zeros((8, 8), dtype: np.Decimal);
            x["1::2", "::2"] = 1;
            x["::2", "1::2"] = 1;
            print(x);

            var ExpectedData = new decimal[8, 8]
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
            AssertStrides(x, sizeof(decimal) * 8, sizeof(decimal));

        }

        [TestMethod]
        public void test_F2C_1_DECIMAL()
        {
            decimal[] fvalues = new decimal[] { 0, 12, 45.21m, 34, 99.91m };
            ndarray F = (ndarray)np.array(fvalues);
            print("Values in Fahrenheit degrees:");
            print(F);
            print("Values in  Centigrade degrees:");

            ndarray C = (decimal)5 * F / 9 - 5 * 32 / 9;
            print(C);

            AssertArray(C, new decimal[] { -17, -10.33333333333333333333333333300m, 08.11666666666666666666666666700m,
                                            01.88888888888888888888888888900m, 38.50555555555555555555555555600m });

        }

        [TestMethod]
        public void test_ArrayStats_1_DECIMAL()
        {
            var x = np.array(new decimal[] { 1, 2, 3 }, dtype: np.Decimal);
            print("Size of the array: ", x.size);
            print("Length of one array element in bytes: ", x.ItemSize);
            print("Total bytes consumed by the elements of the array: ", x.nbytes);

            Assert.AreEqual(3, x.size);
            Assert.AreEqual(16, x.ItemSize);
            Assert.AreEqual(48, x.nbytes);

        }

        [TestMethod]
        public void test_ndarray_flatten_DECIMAL()
        {
            var x = np.arange(0.73m, 25.73m, dtype: np.Decimal).reshape(new shape(5, 5));
            var y = x.flatten();
            print(x);
            print(y);

            AssertArray(y, new decimal[] { 0.73m, 1.73m, 2.73m, 3.73m, 4.73m, 5.73m, 6.73m, 7.73m, 8.73m, 9.73m,
                                         10.73m, 11.73m, 12.73m, 13.73m, 14.73m, 15.73m, 16.73m, 17.73m, 18.73m,
                                         19.73m, 20.73m, 21.73m, 22.73m, 23.73m, 24.73m });

            y = x.flatten(order: NPY_ORDER.NPY_FORTRANORDER);
            print(y);

            AssertArray(y, new decimal[] { 0.73m, 5.73m, 10.73m, 15.73m, 20.73m,  1.73m, 6.73m, 11.73m, 16.73m,
                                         21.73m, 2.73m,  7.73m, 12.73m, 17.73m, 22.73m, 3.73m, 8.73m, 13.73m, 18.73m,
                                         23.73m, 4.73m,  9.73m, 14.73m, 19.73m, 24.73m });

            y = x.flatten(order: NPY_ORDER.NPY_KORDER);
            print(y);

            AssertArray(y, new decimal[] { 0.73m, 1.73m, 2.73m, 3.73m, 4.73m, 5.73m, 6.73m, 7.73m, 8.73m, 9.73m,
                                         10.73m, 11.73m, 12.73m, 13.73m, 14.73m, 15.73m, 16.73m, 17.73m, 18.73m,
                                         19.73m, 20.73m, 21.73m, 22.73m, 23.73m, 24.73m });
        }

        [TestMethod]
        public void test_ndarray_byteswap_DECIMAL()
        {
            var x = np.arange(32, 64, dtype: np.Decimal);
            print(x);
            var y = x.byteswap(true);
            print(y);

            // decimals can't be swapped.  Data should be unchanged
            AssertArray(y, x.AsDecimalArray());

            y = x.byteswap(false);
            print(y);

            // decimals can't be swapped.  Data should be unchanged
            AssertArray(y, x.AsDecimalArray());

        }

        [TestMethod]
        public void test_ndarray_view_DECIMAL()
        {
            var x = np.arange(256 + 32, 256 + 64, dtype: np.Decimal);
            print(x);
            print(x.shape);
            print(x.Dtype);

            AssertArray(x, new decimal[] { 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
                                         299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
                                         310, 311, 312, 313, 314, 315, 316, 317, 318, 319});

            // decimals can't be mapped by something besides another decimal
            var y = x.view(np.UInt64);

            try
            {
                Assert.AreEqual((UInt64)0, (UInt64)y.Sum().GetItem(0));
                Assert.Fail("This should have thrown an exception");
            }
            catch
            {

            }


            y = x.view(np.Decimal);
            AssertArray(y, y.AsDecimalArray());

            y[5] = 1000;

            AssertArray(x, new decimal[] { 288, 289, 290, 291, 292, 1000, 294, 295, 296, 297, 298,
                                         299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
                                         310, 311, 312, 313, 314, 315, 316, 317, 318, 319});

        }

        [TestMethod]
        public void test_ndarray_delete1_DECIMAL()
        {
            var x = np.arange(0, 32, dtype: np.Decimal).reshape(new shape(8, 4));
            print("X");
            print(x);
            print(x.shape);

            var ExpectedDataX = new decimal[8, 4]
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

            var ExpectedDataY = new decimal[8, 3]
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
        public void test_ndarray_delete2_DECIMAL()
        {
            var x = np.arange(0, 32, dtype: np.Decimal);
            print("X");
            print(x);
            print(x.shape);

            var ExpectedDataX = new decimal[] {0,  1,  2,  3,  4,  5,  6,  7,
                                             8,  9,  10, 11, 12, 13, 14, 15,
                                             16, 17, 18, 19, 20, 21, 22, 23,
                                             24, 25, 26, 27, 28, 29, 30, 31 };
            AssertArray(x, ExpectedDataX);
            AssertShape(x, 32);

            var y = np.delete(x, 1, 0);
            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new decimal[] {0,  2,  3,  4,  5,  6,  7,
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
        public void test_ndarray_delete3_DECIMAL()
        {
            var x = np.arange(0, 32, dtype: np.Decimal).reshape(new shape(8, 4));
            print("X");
            print(x);
            print(x.shape);

            var ExpectedDataX = new decimal[8, 4]
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

            var ExpectedDataY = new decimal[8, 3]
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
        public void test_ndarray_unique_1_DECIMAL()
        {
            var x = np.array(new decimal[] { 1, 2, 3, 1, 3, 4, 5, 4, 4 });

            print("X");
            print(x);

            var result = np.unique(x, return_counts: true, return_index: true, return_inverse: true);
            var uvalues = result.data;
            var indexes = result.indices;
            var inverse = result.inverse;
            var counts = result.counts;

            print("uvalues");
            print(uvalues);
            AssertArray(uvalues, new decimal[] { 1, 2, 3, 4, 5 });

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
        public void test_ndarray_where_1_DECIMAL()
        {
            var x = np.array(new decimal[] { 1, 2, 3, 1, 3, 4, 5, 4, 4 }).reshape(new shape(3, 3));

            print("X");
            print(x);

            ndarray[] y = (ndarray[])np.where(x == 3);
            print("Y");
            print(y);


        }

        [TestMethod]
        public void test_ndarray_where_2_DECIMAL()
        {
            var x = np.array(new decimal[] { 1, 2, 3, 1, 3, 4, 5, 4, 4 }).reshape(new shape(3, 3));

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
            AssertArray(z, new decimal[] { 3, 3 });
        }

        [TestMethod]
        public void test_ndarray_where_3_DECIMAL()
        {
            var x = np.arange(0, 1000, dtype: np.Decimal).reshape(new shape(-1, 10));

            //print("X");
            //print(x);

            ndarray[] y = (ndarray[])np.where(x % 10 == 0);
            print("Y");
            print(y);

            var z = x[y] as ndarray;
            print("Z");
            print(z);

            var ExpectedDataZ = new decimal[]
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
        public void test_ndarray_where_4_DECIMAL()
        {
            var x = np.arange(0, 3000000, dtype: np.Decimal);

            var y = np.where(x % 7 == 0);
            //print("Y");
            //print(y);

            var z = x[y] as ndarray;
            var m = np.mean(z);
            print("M");
            Assert.AreEqual(1499998.5m, m.GetItem(0));
            print(m);

            return;
        }

        [TestMethod]
        public void test_ndarray_where_5_DECIMAL()
        {
            var a = np.arange(10, dtype: np.Decimal);

            var b = np.where(a < 5, a, 10 * a) as ndarray;
            AssertArray(b, new decimal[] { 0, 1, 2, 3, 4, 50, 60, 70, 80, 90 });
            print(b);

            a = np.array(new decimal[,] { { 0, 1, 2 }, { 0, 2, 4 }, { 0, 3, 6 } });
            b = np.where(a < 4, a, -1) as ndarray;  // -1 is broadcast
            AssertArray(b, new decimal[,] { { 0, 1, 2 }, { 0, 2, -1 }, { 0, 3, -1 } });
            print(b);

            var c = np.where(new bool[,] { { true, false }, { true, true } },
                                    new decimal[,] { { 1, 2 }, { 3, 4 } },
                                    new decimal[,] { { 9, 8 }, { 7, 6 } }) as ndarray;

            AssertArray(c, new decimal[,] { { 1, 8 }, { 3, 4 } });

            print(c);

            return;
        }

        [TestMethod]
        public void test_arange_slice_1_DECIMAL()
        {
            var a = np.arange(0, 1024, dtype: np.Decimal).reshape(new shape(2, 4, -1));

            print("A");
            // print(a);
            print(a.shape);
            print(a.strides);

            AssertShape(a, 2, 4, 128);
            AssertStrides(a, 8192, 2048, 16);

            var b = (ndarray)a[":", ":", 122];
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new decimal[2, 4]
            {
                { 122, 250, 378, 506},
                { 634, 762, 890, 1018 },
            };

            AssertArray(b, ExpectedDataB);
            AssertShape(b, 2, 4);
            AssertStrides(b, 8192, 2048);

            var c = (ndarray)a.A(":", ":", new Int64[] { 122 });
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            var ExpectedDataC = new decimal[2, 4, 1]
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
            AssertStrides(c, 64, 16, 128);

            var d = (ndarray)a.A(":", ":", new Int64[] { 122, 123 });
            print("D");
            print(d);
            print(d.shape);
            print(d.strides);

            var ExpectedDataD = new decimal[2, 4, 2]
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
            AssertStrides(d, 64, 16, 128);

        }

        [TestMethod]
        public void test_arange_slice_2A_DECIMAL()
        {
            var a = np.arange(0, 32, dtype: np.Decimal).reshape(new shape(2, 4, -1));

            print("A");
            // print(a);
            print(a.shape);
            print(a.strides);

            var b = (ndarray)a[":", ":", np.where(a > 20)];
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new decimal[,,,]
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
            AssertStrides(b, 64, 16, 1408, 128);
        }

        [TestMethod]
        public void test_insert_1_DECIMAL()
        {
            decimal[,] TestData = new decimal[,] { { 1, 1 }, { 2, 2 }, { 3, 3 } };
            ndarray a = np.array(TestData, dtype: np.Decimal);
            ndarray b = np.insert(a, 1, 5);
            ndarray c = np.insert(a, 0, new decimal[] { 999, 100, 101 });

            print(a);
            print(a.shape);

            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new decimal[] { 1, 5, 1, 2, 2, 3, 3 });
            AssertShape(b, 7);
            AssertStrides(b, 16);

            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            AssertArray(c, new decimal[] { 999, 100, 101, 1, 1, 2, 2, 3, 3 });
            AssertShape(c, 9);
            AssertStrides(c, 16);
        }

        [TestMethod]
        public void test_insert_2_DECIMAL()
        {
            decimal[] TestData1 = new decimal[] { 1, 1, 2, 2, 3, 3 };
            decimal[] TestData2 = new decimal[] { 90, 91, 92, 92, 93, 93 };

            ndarray a = np.array(TestData1, dtype: np.Decimal);
            ndarray b = np.array(TestData2, dtype: np.Decimal);
            ndarray c = np.insert(a, new Slice(null), b);

            print(a);
            print(a.shape);

            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new decimal[] { 90, 91, 92, 92, 93, 93 });
            AssertShape(b, 6);
            //AssertStrides(b, 4);

            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            AssertArray(c, new decimal[] { 90, 1, 91, 1, 92, 2, 92, 2, 93, 3, 93, 3 });
            AssertShape(c, 12);
            //AssertStrides(c, 4);

        }

        [TestMethod]
        public void test_append_1_DECIMAL()
        {
            decimal[] TestData = new decimal[] { 1, 1, 2, 2, 3, 3 };
            ndarray a = np.array(TestData, dtype: np.Decimal);
            ndarray b = np.append(a, (decimal)1);

            print(a);
            print(a.shape);

            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new decimal[] { 1, 1, 2, 2, 3, 3, 1 });
            AssertShape(b, 7);
            AssertStrides(b, 16);
        }

        [TestMethod]
        public void test_append_3_DECIMAL()
        {
            decimal[] TestData1 = new decimal[] { 1, 1, 2, 2, 3, 3 };
            decimal[] TestData2 = new decimal[] { 4, 4, 5, 5, 6, 6 };
            ndarray a = np.array(TestData1, dtype: np.Decimal);
            ndarray b = np.array(TestData2, dtype: np.Decimal);

            ndarray c = np.append(a, b);

            print(a);
            print(a.shape);

            print(b);
            print(b.shape);

            print(c);
            print(c.shape);
            print(c.strides);

            AssertArray(c, new decimal[] { 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6 });
            AssertShape(c, 12);
            AssertStrides(c, 16);
        }

        [TestMethod]
        public void test_append_4_DECIMAL()
        {
            decimal[] TestData1 = new decimal[] { 1, 1, 2, 2, 3, 3 };
            decimal[] TestData2 = new decimal[] { 4, 4, 5, 5, 6, 6 };
            ndarray a = np.array(TestData1, dtype: np.Decimal).reshape((2, -1));
            ndarray b = np.array(TestData2, dtype: np.Decimal).reshape((2, -1));

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

            var ExpectedDataC = new decimal[,]
            {
                { 1, 1, 2, 4, 4, 5 },
                { 2, 3, 3, 5, 6, 6 },
            };

            AssertArray(c, ExpectedDataC);
            AssertShape(c, 2, 6);
            //AssertStrides(c, 24, 4); 

        }

        [TestMethod]
        public void test_flat_2_DECIMAL()
        {
            var x = np.arange(1, 7, dtype: np.Decimal).reshape((2, 3));
            print(x);

            Assert.AreEqual(4m, x.Flat[3]);
            print(x.Flat[3]);

            print(x.T);
            Assert.AreEqual(5m, x.T.Flat[3]);
            print(x.T.Flat[3]);

            x.flat = 3;
            AssertArray(x, new decimal[,] { { 3, 3, 3 }, { 3, 3, 3 } });
            print(x);

            x.Flat[new int[] { 1, 4 }] = 1;
            AssertArray(x, new decimal[,] { { 3, 1, 3 }, { 3, 1, 3 } });
            print(x);
        }

        [TestMethod]
        public void test_intersect1d_1_DECIMAL()
        {
            ndarray a = np.array(new decimal[] { 1, 3, 4, 3 });
            ndarray b = np.array(new decimal[] { 3, 1, 2, 1 });

            ndarray c = np.intersect1d(a, b);
            print(c);

            AssertArray(c, new decimal[] { 1, 3 });
            AssertShape(c, 2);
            AssertStrides(c, 16);

        }

        [TestMethod]
        public void test_setxor1d_1_DECIMAL()
        {
            ndarray a = np.array(new decimal[] { 1, 2, 3, 2, 4 });
            ndarray b = np.array(new decimal[] { 2, 3, 5, 7, 5 });

            ndarray c = np.setxor1d(a, b);
            print(c);

            AssertArray(c, new decimal[] { 1, 4, 5, 7 });
            AssertShape(c, 4);
            AssertStrides(c, 16);
        }

        [TestMethod]
        public void test_in1d_1_DECIMAL()
        {
            ndarray test = np.array(new decimal[] { 0, 1, 2, 5, 0 });
            ndarray states = np.array(new decimal[] { 0, 2 });

            ndarray mask = np.in1d(test, states);
            print(mask);
            print(test[mask]);

            AssertArray(mask, new bool[] { true, false, true, false, true });
            AssertShape(mask, 5);
            AssertStrides(mask, 1);

            ndarray a = test[mask] as ndarray;
            AssertArray(a, new decimal[] { 0, 2, 0 });
            AssertShape(a, 3);
            AssertStrides(a, 16);

            mask = np.in1d(test, states, invert: true);
            print(mask);
            print(test[mask]);

            AssertArray(mask, new bool[] { false, true, false, true, false });
            AssertShape(mask, 5);
            AssertStrides(mask, 1);

            ndarray b = test[mask] as ndarray;
            AssertArray(b, new decimal[] { 1, 5 });
            AssertShape(b, 2);
            AssertStrides(b, 16);

        }

        [TestMethod]
        public void test_isin_1_DECIMAL()
        {
            ndarray element = 2 * np.arange(4, dtype: np.Decimal).reshape(new shape(2, 2));
            print(element);

            ndarray test_elements = np.array(new decimal[] { 1, 2, 4, 8 });
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

            AssertArray(a, new decimal[] { 2, 4 });
            AssertShape(a, 2);
            AssertStrides(a, 16);

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

            AssertArray(a, new decimal[] { 0, 6 });
            AssertShape(a, 2);
            AssertStrides(a, 16);
        }

        [TestMethod]
        public void test_union1d_1_DECIMAL()
        {
            ndarray a1 = np.array(new decimal[] { -1, 0, 1 });
            ndarray a2 = np.array(new decimal[] { -2, 0, 2 });

            ndarray a = np.union1d(a1, a2);
            print(a);

            AssertArray(a, new decimal[] { -2, -1, 0, 1, 2 });
            AssertShape(a, 5);
            AssertStrides(a, 16);
        }

        [TestMethod]
        public void test_Ellipsis_indexing_1_DECIMAL()
        {
            var a = np.array(new decimal[] { 10.0m, 7, 4, 3, 2, 1 });

            var b = a.A("...", -1);
            Assert.AreEqual((decimal)1.0, b.GetItem(0));
            print(b);
            print("********");


            a = np.array(new decimal[,] { { 10.0m, 7, 4 }, { 3, 2, 1 } });
            var c = a.A("...", -1);
            AssertArray(c, new decimal[] { 4.0m, 1.0m });
            print(c);
            print("********");

            var TestData = new decimal[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            a = np.array(TestData, dtype: np.Decimal).reshape((1, 3, 2, -1, 1));
            var d = a["...", -1] as ndarray;
            AssertArray(d, new decimal[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } } });
            print(d);
            print("********");

            var e = a[0, "...", -1] as ndarray;
            AssertArray(e, new decimal[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
            print(e);
            print("********");

            var f = a[0, ":", ":", ":", -1] as ndarray;
            AssertArray(f, new decimal[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } });
            print(f);
            print("********");

            var g = a.A(0, 1, "...", -1);
            AssertArray(g, new decimal[,] { { 5, 6 }, { 7, 8 } });
            print(g);
            print("********");

            var h = a.A(0, 2, 1, "...", -1);
            AssertArray(h, new decimal[] { 11, 12 });
            print(h);
            print("********");

            var i = a[":", 2, 1, 1, "..."] as ndarray;
            AssertArray(i, new decimal[,] { { 12 } });
            print(i);
        }

        [TestMethod]
        public void test_concatenate_1_DECIMAL()
        {

            var a = np.array(new decimal[,] { { 1, 2 }, { 3, 4 } });
            var b = np.array(new decimal[,] { { 5, 6 } });
            var c = np.concatenate((a, b), axis: 0);
            AssertArray(c, new decimal[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            print(c);

            var d = np.concatenate((a, b.T), axis: 1);
            AssertArray(d, new decimal[,] { { 1, 2, 5 }, { 3, 4, 6 } });
            print(d);

            var e = np.concatenate((a, b), axis: null);
            AssertArray(e, new decimal[] { 1, 2, 3, 4, 5, 6 });
            print(e);

            var f = np.concatenate((np.eye(2, dtype: np.Decimal), np.ones((2, 2), dtype: np.Decimal)), axis: 0);
            AssertArray(f, new decimal[,] { { 1, 0 }, { 0, 1 }, { 1, 1 }, { 1, 1 }, });
            print(f);

            var g = np.concatenate((np.eye(2, dtype: np.Decimal), np.ones((2, 2), dtype: np.Decimal)), axis: 1);
            AssertArray(g, new decimal[,] { { 1, 0, 1, 1 }, { 0, 1, 1, 1 } });
            print(g);
        }

        [TestMethod]
        public void test_concatenate_3_DECIMAL()
        {

            var a = np.array(new decimal[,,,] { { { { 1, 2 }, { 3, 4 }, { 5, 6 } } } });
            var c = np.concatenate(a, axis: -1);
            AssertArray(c, new decimal[,,] { { { 1, 2 }, { 3, 4 }, { 5, 6 } } });
            print(c);

            var d = np.concatenate(a, axis: -2);
            AssertArray(d, new decimal[,,] { { { 1, 2 }, { 3, 4 }, { 5, 6 } } });
            print(d);

            c = np.concatenate((a, a, a), axis: -1);
            AssertArray(c, new decimal[,,,] { { { { 1, 2, 1, 2, 1, 2 }, { 3, 4, 3, 4, 3, 4 }, { 5, 6, 5, 6, 5, 6 } } } });
            print(c);

            d = np.concatenate((a, a, a), axis: -2);
            AssertArray(d, new decimal[,,,] { { { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 1, 2 }, { 3, 4 }, { 5, 6 }, { 1, 2 }, { 3, 4 }, { 5, 6 } } } });
            print(d);


        }

        [TestMethod]
        public void test_multi_index_selection_DECIMAL()
        {
            var x = np.arange(10).astype(np.Decimal);

            var y = x.reshape(new shape(2, 5));
            print(y);
            Assert.AreEqual((Decimal)3, y[0, 3]);
            Assert.AreEqual((Decimal)8, y[1, 3]);

            x = np.arange(20).astype(np.Decimal);
            y = x.reshape(new shape(2, 2, 5));
            print(y);
            Assert.AreEqual((Decimal)3, y[0, 0, 3]);
            Assert.AreEqual((Decimal)8, y[0, 1, 3]);

            Assert.AreEqual((Decimal)13, y[1, 0, 3]);
            Assert.AreEqual((Decimal)18, y[1, 1, 3]);

        }

        [TestMethod]
        public void test_multi_index_setting_DECIMAL()
        {
            var x = np.arange(10, dtype: np.Int32).astype(np.Decimal);

            var y = x.reshape(new shape(2, 5));

            y[0, 3] = new Decimal(55);
            y[1, 3] = new Decimal(66);

            Assert.AreEqual((Decimal)55, (Decimal)y[0, 3]);
            Assert.AreEqual((Decimal)66, (Decimal)y[1, 3]);

            x = np.arange(20, dtype: np.Int32).astype(np.Decimal);
            y = x.reshape(new shape(2, 2, 5));

            y[1, 0, 3] = new Decimal(55);
            y[1, 1, 3] = new Decimal(66);

            Assert.AreEqual((Decimal)55, (Decimal)y[1, 0, 3]);
            Assert.AreEqual((Decimal)66, (Decimal)y[1, 1, 3]);

        }

        #endregion

        #region from NumericalOperationsTests

        [TestMethod]
        public void test_add_operations_DECIMAL()
        {
            var a = np.arange(0m, 20m, 1, dtype: np.Decimal);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            var b = a + 8m;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new decimal[,]
            {{8,  9, 10, 11},
             {12, 13, 14, 15},
             {16, 17, 18, 19},
             {20, 21, 22, 23},
             {24, 25, 26, 27}
            };
            AssertArray(b, ExpectedDataB);

            a = np.arange(0m, 20m, 1, dtype: np.Decimal);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            b = a + 2400;
            print(b);
            print(b.shape);
            print(b.strides);

            ExpectedDataB = new decimal[,]
            {{2400, 2401, 2402, 2403},
             {2404, 2405, 2406, 2407},
             {2408, 2409, 2410, 2411},
             {2412, 2413, 2414, 2415},
             {2416, 2417, 2418, 2419}
            };
            AssertArray(b, ExpectedDataB);

        }

        [TestMethod]
        public void test_add_operations_DECIMAL_2()
        {
            var a = np.arange(0, 20, 1, dtype: np.Decimal);
            a = a.reshape(new shape(5, -1));
            print(a);

            var ExpectedDataA = new Decimal[,]
                {{0,  1,  2,  3},
                 {4,  5,  6,  7},
                 {8,  9, 10, 11},
                 {12, 13, 14, 15},
                 {16, 17, 18, 19}};
            AssertArray(a, ExpectedDataA);

            var b = np.array(new Decimal[] { 2 });
            var c = a + b;
            print(c);

            var ExpectedDataC = new Decimal[,]
                {{2,  3,  4,  5},
                 {6,  7,  8,  9},
                 {10, 11, 12, 13},
                 {14, 15, 16, 17},
                 {18, 19, 20, 21}};
            AssertArray(c, ExpectedDataC);


            b = np.array(new Decimal[] { 10, 20, 30, 40 });
            var d = a + b;
            print(d);

            var ExpectedDataD = new Decimal[,]
                {{10, 21, 32, 43},
                 {14, 25, 36, 47},
                 {18, 29, 40, 51},
                 {22, 33, 44, 55},
                 {26, 37, 48, 59}};
            AssertArray(d, ExpectedDataD);
        }

        [TestMethod]
        public void test_subtract_operations_DECIMAL()
        {
            var a = np.arange(0m, 20m, 1m, dtype: np.Decimal);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            var b = a - 8;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new decimal[,]
            {{-8, -7, -6, -5},
             {-4, -3, -2, -1},
             {0,  1,  2,  3},
             {4,  5,  6,  7},
             {8,  9, 10, 11}
            };
            AssertArray(b, ExpectedDataB);

            a = np.arange(0m, 20m, 1m, dtype: np.Decimal);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            b = a - 2400;
            print(b);
            print(b.shape);
            print(b.strides);

            ExpectedDataB = new decimal[,]
            {{-2400, -2399, -2398, -2397},
             {-2396, -2395, -2394, -2393},
             {-2392, -2391, -2390, -2389},
             {-2388, -2387, -2386, -2385},
             {-2384, -2383, -2382, -2381}
            };

            AssertArray(b, ExpectedDataB);

        }

        [TestMethod]
        public void test_subtract_operations_DECIMAL_2()
        {
            var a = np.arange(100, 102, 1, dtype: np.Decimal);
            var b = np.array(new Decimal[] { 1, 63 });
            var c = a - b;
            print(a);
            print("****");
            print(b);
            print("****");
            print(c);
            print("****");
            AssertArray(c, new decimal[] { 99, 38 });


            a = np.arange(0, 4, 1, dtype: np.Decimal).reshape(new shape(2, 2));
            b = np.array(new Decimal[] { 65, 78 }).reshape(new shape(1, 2));
            c = a - b;
            print(a);
            print("****");
            print(b);
            print("****");
            print(c);
            print("****");
            AssertArray(c, new Decimal[,] { { -65, -77 }, { -63, -75 } });

        }

        [TestMethod]
        public void test_multiply_operations_DECIMAL()
        {
            var a = np.arange(0, 20, 1, dtype: np.Decimal);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            decimal multiplierB1 = 9023.67m;
            var b = a * multiplierB1;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB1 = new decimal[,]
            {
                {0m*multiplierB1,  1m*multiplierB1,  2m*multiplierB1,  3m*multiplierB1},
                {4m*multiplierB1,  5m*multiplierB1,  6m*multiplierB1,  7m*multiplierB1},
                {8m*multiplierB1,  9m*multiplierB1,  10m*multiplierB1, 11m*multiplierB1},
                {12m*multiplierB1, 13m*multiplierB1, 14m*multiplierB1, 15m*multiplierB1},
                {16m*multiplierB1, 17m*multiplierB1, 18m*multiplierB1, 19m*multiplierB1}
            };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(0, 20, 1, dtype: np.Decimal);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            decimal multiplierB2 = 990425023.67864101m;
            b = a * multiplierB2;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB2 = new decimal[,]
            {
                {0m*multiplierB2,  1m*multiplierB2,  2m*multiplierB2,  3m*multiplierB2},
                {4m*multiplierB2,  5m*multiplierB2,  6m*multiplierB2,  7m*multiplierB2},
                {8m*multiplierB2,  9m*multiplierB2,  10m*multiplierB2, 11m*multiplierB2},
                {12m*multiplierB2, 13m*multiplierB2, 14m*multiplierB2, 15m*multiplierB2},
                {16m*multiplierB2, 17m*multiplierB2, 18m*multiplierB2, 19m*multiplierB2}
            };
            AssertArray(b, ExpectedDataB2);
        }

        [TestMethod]
        public void test_division_operations_DECIMAL()
        {
            var a = np.arange(20000, 20020, 1, dtype: np.Decimal);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            decimal divisorB1 = 611m;
            var b = a / divisorB1;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB1 = new decimal[,]
            {
                {20000m/divisorB1, 20001m/divisorB1, 20002m/divisorB1, 20003m/divisorB1},
                {20004m/divisorB1, 20005m/divisorB1, 20006m/divisorB1, 20007m/divisorB1},
                {20008m/divisorB1, 20009m/divisorB1, 20010m/divisorB1, 20011m/divisorB1},
                {20012m/divisorB1, 20013m/divisorB1, 20014m/divisorB1, 20015m/divisorB1},
                {20016m/divisorB1, 20017m/divisorB1, 20018m/divisorB1, 20019m/divisorB1}
            };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(2000000, 2000020, 1, dtype: np.Decimal);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            decimal divisorB2 = 2411m;
            b = a / divisorB2;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB2 = new decimal[,]
            {
                {2000000m/divisorB2, 2000001m/divisorB2, 2000002m/divisorB2, 2000003m/divisorB2},
                {2000004m/divisorB2, 2000005m/divisorB2, 2000006m/divisorB2, 2000007m/divisorB2},
                {2000008m/divisorB2, 2000009m/divisorB2, 2000010m/divisorB2, 2000011m/divisorB2},
                {2000012m/divisorB2, 2000013m/divisorB2, 2000014m/divisorB2, 2000015m/divisorB2},
                {2000016m/divisorB2, 2000017m/divisorB2, 2000018m/divisorB2, 2000019m/divisorB2},
            };
            AssertArray(b, ExpectedDataB2);
        }

        [TestMethod]
        public void test_leftshift_operations_DECIMAL()
        {
            var a = np.arange(0, 20, 1, dtype: np.Decimal);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            var b = a << 8;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB1 = new decimal[,]
            {
                {0,  256,  512,  768},
                {1024, 1280, 1536, 1792},
                {2048, 2304, 2560, 2816},
                {3072, 3328, 3584, 3840},
                {4096, 4352, 4608, 4864}
            };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(0, 20, 1, dtype: np.Decimal);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            b = a << 24;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB2 = new decimal[,]
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
        public void test_rightshift_operations_DECIMAL()
        {
            var a = np.arange(20000, 20020, 1, dtype: np.Decimal);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            var b = a >> 8;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB1 = new decimal[,]
            {
                {78, 78, 78, 78},
                {78, 78, 78, 78},
                {78, 78, 78, 78},
                {78, 78, 78, 78},
                {78, 78, 78, 78}
            };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(2123450, 2123470, 1, dtype: np.Decimal);
            a = a.reshape(new shape(5, -1));
            print(a);
            print(a.shape);
            print(a.strides);

            b = a >> 8;
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB2 = new decimal[,]
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
        public void test_bitwiseand_operations_DECIMAL()
        {
            var a = np.arange(0.499, 32.499, 1, dtype: np.Decimal);
            print(a);

            var b = a & 0x0f;
            print(b);

            var ExpectedDataB1 = new decimal[]
            { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
              0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Decimal);
            print(a);

            b = a & 0xFF;
            print(b);

            var ExpectedDataB2 = new decimal[]
            { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };
            AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_bitwiseor_operations_DECIMAL()
        {
            var a = np.arange(0.499, 32.499, 1, dtype: np.Decimal);
            print(a);

            var b = a | 0x100;
            print(b);

            var ExpectedDataB1 = new decimal[]
            { 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
              272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287 };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Decimal);
            print(a);

            b = a | 0x1000;
            print(b);

            var ExpectedDataB2 = new decimal[]
            { 6144, 6145, 6146, 6147, 6148, 6149, 6150, 6151, 6152, 6153, 6154, 6155, 6156, 6157,
              6158, 6159, 6160, 6161, 6162, 6163, 6164, 6165, 6166, 6167, 6168, 6169, 6170, 6171,
              6172, 6173, 6174, 6175 };
            AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_bitwisexor_operations_DECIMAL()
        {
            var a = np.arange(0.499, 32.499, 1, dtype: np.Decimal);
            print(a);

            var b = a ^ 0xAAA;
            print(b);

            var ExpectedDataB1 = new decimal[]
            { 2730, 2731, 2728, 2729, 2734, 2735, 2732, 2733, 2722, 2723, 2720, 2721, 2726, 2727, 2724,
              2725, 2746, 2747, 2744, 2745, 2750, 2751, 2748, 2749, 2738, 2739, 2736, 2737, 2742, 2743, 2740, 2741 };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Decimal);
            print(a);

            b = a ^ 0xAAAA;
            print(b);

            var ExpectedDataB2 = new decimal[]
            { 41642, 41643, 41640, 41641, 41646, 41647, 41644, 41645, 41634, 41635, 41632, 41633,
              41638, 41639, 41636, 41637, 41658, 41659, 41656, 41657, 41662, 41663, 41660, 41661,
              41650, 41651, 41648, 41649, 41654, 41655, 41652, 41653};
            AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_remainder_operations_DECIMAL()
        {
            var a = np.arange(0, 32, 1, dtype: np.Decimal);
            print(a);

            var b = a % 6;
            print(b);

            AssertArray(b, new decimal[] { 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
                                         4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1 });

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Decimal);
            print(a);

            b = a % 6;
            print(b);

            AssertArray(b, new decimal[] { 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                                         0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3 });

        }

        [TestMethod]
        public void test_sqrt_operations_DECIMAL()
        {
            var a = np.arange(0, 32, 1, dtype: np.Decimal);
            print(a);

            var b = np.sqrt(a);
            print(b);

            var ExpectedDataB1 = new decimal[]
            {
               0m, 1m, 01.41421356237309504880168872420m, 01.73205080756887729352744634150m, 2m,
                02.23606797749978969640917366880m, 02.44948974278317809819728407470m, 02.64575131106459059050161575360m, 02.82842712474619009760337744840m,
                3m, 03.16227766016837933199889354440m, 03.31662479035539984911493273660m, 03.46410161513775458705489268300m,
                03.60555127546398929311922126740m, 03.74165738677394138558374873230m, 03.87298334620741688517926539980m, 4m,
                04.12310562561766054982140985600m, 04.24264068711928514640506617250m, 04.35889894354067355223698198400m, 04.47213595499957939281834733750m,
                04.58257569495584000658804719400m, 04.69041575982342955456563011350m, 04.79583152331271954159743806400m, 04.89897948556635619639456814950m,
                5m, 05.09901951359278483002822410900m, 05.19615242270663188058233902450m, 05.29150262212918118100323150750m,
                05.38516480713450403125071049150m, 05.47722557505166113456969782800m, 05.56776436283002192211947129900m

            };

            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Decimal);
            print(a);

            b = np.sqrt(a);
            print(b);

            var ExpectedDataB2 = new decimal[]
            {
                45.25483399593904156165403917500m, 45.26588119102510082052098860500m, 45.27692569068708313286904083500m, 45.28796749689700967470411927000m,
                45.29900661162449818342010884000m, 45.31004303683676705673235173000m, 45.32107677449863944262772470000m, 45.33210782657254732035535008500m,
                45.34313619501853557248191637000m, 45.35416188179426604803550841500m, 45.36518488885502161676177205000m, 45.37620521815371021451616237500m,
                45.38722287164086887981595016500m, 45.39823785126466778157558615000m, 45.40925015897091423804894846500m, 45.42025979670305672700192442500m,
                45.43126676640218888713870387000m, 45.44227107000705351080508783000m, 45.45327270945404652799204274000m, 45.46427168667722098166265753000m,
                45.47526800360829099442558795000m, 45.48626166217663572657800009000m, 45.49725266430930332554095262000m, 45.50824101193101486671008537000m,
                45.51922670696416828574441006500m, 45.53020975132884230231592748500m, 45.54119014694280033534272413000m, 45.55216789572149440972813052500m,
                45.56314299957806905462845237000m, 45.57411546042336519327171558000m, 45.58508528016592402434979572500m, 45.59605246071199089500623262000m
            };
            AssertArray(b, ExpectedDataB2);

        }
        
        [TestMethod]
        public void test_cbrt_operations_DECIMAL()
        {
            var a = np.arange(0, 32, 1, dtype: np.Decimal);
            print(a);

            var b = np.cbrt(a);
            print(b);

            var ExpectedDataB1 = new decimal[]
            {
                0, 1, 01.25992104989487000000000000000m, 01.44224957030741000000000000000m, 01.58740105196820000000000000000m,
                01.70997594667670000000000000000m, 01.81712059283214000000000000000m, 01.91293118277239000000000000000m,
                2, 02.08008382305190000000000000000m, 02.15443469003188000000000000000m, 02.22398009056931000000000000000m,
                02.28942848510666000000000000000m, 02.35133468772076000000000000000m, 02.41014226417523000000000000000m,
                02.46621207433047000000000000000m, 02.51984209978974000000000000000m, 02.57128159065823000000000000000m,
                02.62074139420889000000000000000m, 02.66840164872194000000000000000m, 02.71441761659490000000000000000m,
                02.75892417638112000000000000000m, 02.80203933065538000000000000000m, 02.84386697985156000000000000000m,
                02.88449914061481000000000000000m, 02.92401773821286000000000000000m, 02.96249606840737000000000000000m, 3,
                03.03658897187566000000000000000m, 03.07231682568584000000000000000m, 03.10723250595386000000000000000m,
                03.14138065239139000000000000000m
            };

            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Decimal);
            print(a);

            b = np.cbrt(a);
            print(b);

            var ExpectedDataB2 = new decimal[]
            {
                12.69920841574560000000000000000m, 12.70127500787570000000000000000m, 12.70334092772480000000000000000m,
                12.70540617583920000000000000000m, 12.70747075276460000000000000000m, 12.70953465904630000000000000000m,
                12.71159789522850000000000000000m, 12.71366046185490000000000000000m, 12.71572235946840000000000000000m,
                12.71778358861120000000000000000m, 12.71984414982490000000000000000m, 12.72190404365040000000000000000m,
                12.72396327062770000000000000000m, 12.72602183129630000000000000000m, 12.72807972619490000000000000000m,
                12.73013695586160000000000000000m, 12.73219352083380000000000000000m, 12.73424942164790000000000000000m,
                12.73630465884010000000000000000m, 12.73835923294560000000000000000m, 12.74041314449890000000000000000m,
                12.74246639403400000000000000000m, 12.74451898208400000000000000000m, 12.74657090918140000000000000000m,
                12.74862217585820000000000000000m, 12.75067278264530000000000000000m, 12.75272273007330000000000000000m,
                12.75477201867200000000000000000m, 12.75682064897040000000000000000m, 12.75886862149700000000000000000m,
                12.76091593677950000000000000000m, 12.76296259534500000000000000000m
            };
            AssertArray(b, ExpectedDataB2);
        }

        [TestMethod]
        public void test_negative_operations_DECIMAL()
        {
            var a = np.arange(0.333, 32.333, 1, dtype: np.Decimal);
            print(a);

            var b = -a;
            print(b);

            var ExpectedDataB2 = new decimal[]
            {
                -00.33300000000000000000000000000m, -01.33300000000000000000000000000m, -02.33300000000000000000000000000m,
                -03.33300000000000000000000000000m, -04.33300000000000000000000000000m, -05.33300000000000000000000000000m,
                -06.33300000000000000000000000000m, -07.33300000000000000000000000000m, -08.33300000000000000000000000000m,
                -09.33300000000000000000000000000m, -10.33300000000000000000000000000m, -11.33300000000000000000000000000m,
                -12.33300000000000000000000000000m, -13.33300000000000000000000000000m, -14.33300000000000000000000000000m,
                -15.33300000000000000000000000000m, -16.33300000000000000000000000000m, -17.33300000000000000000000000000m,
                -18.33300000000000000000000000000m, -19.33300000000000000000000000000m, -20.33300000000000000000000000000m,
                -21.33300000000000000000000000000m, -22.33300000000000000000000000000m, -23.33300000000000000000000000000m,
                -24.33300000000000000000000000000m, -25.33300000000000000000000000000m, -26.33300000000000000000000000000m,
                -27.33300000000000000000000000000m, -28.33300000000000000000000000000m, -29.33300000000000000000000000000m,
                -30.33300000000000000000000000000m, -31.33300000000000000000000000000m
            };
            AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_invert_operations_DECIMAL()
        {
            var a = np.arange(-32, 32, 1, dtype: np.Decimal);
            print(a);

            var b = ~a;
            print(b);

            // this should not be changed at all.  Decimals can't be inverted.
            AssertArray(b, a.AsDecimalArray());

        }

        [TestMethod]
        public void test_LESS_operations_DECIMAL()
        {
            var a = np.arange(-5, 5, 1, dtype: np.Decimal);
            print(a);

            var b = a < -2;
            print(b);

            AssertArray(b, new Boolean[] { true, true, true, false, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_LESSEQUAL_operations_DECIMAL()
        {
            var a = np.arange(-5, 5, 1, dtype: np.Decimal);
            print(a);

            var b = a <= -2;
            print(b);

            AssertArray(b, new Boolean[] { true, true, true, true, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_EQUAL_operations_DECIMAL()
        {
            var a = np.arange(-5, 5, 1, dtype: np.Decimal);
            print(a);

            var b = a == -2;
            print(b);

            AssertArray(b, new Boolean[] { false, false, false, true, false, false, false, false, false, false });
        }

        [TestMethod]
        public void test_NOTEQUAL_operations_DECIMAL()
        {
            var a = np.arange(-5, 5, 1, dtype: np.Decimal);
            print(a);

            var b = a != -2;
            print(b);

            AssertArray(b, new Boolean[] { true, true, true, false, true, true, true, true, true, true });
        }

        [TestMethod]
        public void test_GREATER_operations_DECIMAL()
        {
            var a = np.arange(-5, 5, 1, dtype: np.Decimal);
            print(a);

            var b = a > -2;
            print(b);

            AssertArray(b, new Boolean[] { false, false, false, false, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_GREATEREQUAL_operations_DECIMAL()
        {
            var a = np.arange(-5, 5, 1, dtype: np.Decimal);
            print(a);

            var b = a >= -2;
            print(b);

            AssertArray(b, new Boolean[] { false, false, false, true, true, true, true, true, true, true });

        }

        [TestMethod]
        public void test_arrayarray_or_DECIMAL()
        {
            var a = np.arange(0, 32, 1, dtype: np.Decimal);
            var b = np.arange(33, 33 + 32, 1, dtype: np.Decimal);
            var c = a | b;
            print(a);
            print(b);
            print(c);

            AssertArray(c, new decimal[] {33, 35, 35, 39, 37, 39, 39, 47, 41, 43, 43, 47,
                                        45, 47, 47, 63, 49, 51, 51, 55, 53, 55, 55, 63,
                                        57, 59, 59, 63, 61, 63, 63, 95 });
        }

        [TestMethod]
        public void test_bitwise_and_DECIMAL()
        {
            var x = np.arange(1023, 1039, 1, dtype: np.Decimal).reshape(new shape(2, -1));
            var y = np.bitwise_and(x, 0x3FF);
            var z = x & 0x3FF;

            print(x);
            print(y);
            print(z);

            var ExpectedData = new decimal[,]
            {
                { 1023, 0, 1,  2,  3,  4,  5,  6 },
                {  7, 8, 9, 10, 11, 12, 13, 14 }
            };

            AssertArray(y, ExpectedData);
            AssertArray(z, ExpectedData);
        }

        [TestMethod]
        public void test_bitwise_or_DECIMAL()
        {
            var x = np.arange(1023, 1039, 1, dtype: np.Decimal).reshape(new shape(2, -1));
            var y = np.bitwise_or(x, 0x10);
            var z = x | 0x10;

            print(x);
            print(y);
            print(z);

            var ExpectedData = new decimal[,]
            {
                { 1023, 1040, 1041, 1042, 1043, 1044, 1045, 1046 },
                { 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054 }
            };

            AssertArray(y, ExpectedData);
            AssertArray(z, ExpectedData);
        }

        [TestMethod]
        public void test_bitwise_xor_DECIMAL()
        {
            var a = np.bitwise_xor(13, 17);
            Assert.AreEqual(28, a.GetItem(0));
            print(a);

            var b = np.bitwise_xor(31, 5);
            Assert.AreEqual(26, b.GetItem(0));
            print(b);

            var c = np.bitwise_xor(new decimal[] { 31, 3 }, 5);
            AssertArray(c, new decimal[] { 26, 6 });
            print(c);

            var d = np.bitwise_xor(new decimal[] { 31, 3 }, new decimal[] { 5, 6 });
            AssertArray(d, new decimal[] { 26, 5 });
            print(d);

            var e = np.bitwise_xor(new bool[] { true, true }, new bool[] { false, true });
            AssertArray(e, new bool[] { true, false });
            print(e);

            return;
        }

        [TestMethod]
        public void test_bitwise_not_DECIMAL()
        {
            var a = np.bitwise_not(13);
            Assert.AreEqual(-14, a.GetItem(0));
            print(a);

            var b = np.bitwise_not(31);
            Assert.AreEqual(-32, b.GetItem(0));
            print(b);

            // can't inverse a decimal
            var c = np.bitwise_not(new decimal[] { 31, 3 });
            AssertArray(c, new decimal[] { 31, 3 });
            print(c);

            // can't inverse a decimal
            var d = np.bitwise_not(new decimal[] { 31, 3 });
            AssertArray(d, new decimal[] { 31, 3 });
            print(d);

            var e = np.bitwise_not(new bool[] { true, false });
            AssertArray(e, new bool[] { false, true });
            print(e);

            return;
        }

        [TestMethod]
        public void test_invert_DECIMAL()
        {
            var a = np.invert(13);
            Assert.AreEqual(-14, a.GetItem(0));
            print(a);

            var b = np.invert(31);
            Assert.AreEqual(-32, b.GetItem(0));
            print(b);

            // can't inverse a decimal
            var c = np.invert(new decimal[] { 31, 3 });
            AssertArray(c, new decimal[] { 31, 3 });
            print(c);

            // can't inverse a decimal
            var d = np.invert(new decimal[] { 31, 3 });
            AssertArray(d, new decimal[] { 31, 3 });
            print(d);

            var e = np.invert(new bool[] { true, false });
            AssertArray(e, new bool[] { false, true });
            print(e);

            return;
        }

        [TestMethod]
        public void test_right_shift_DECIMAL()
        {
            var x = np.arange(1023, 1039, 1, dtype: np.Decimal).reshape(new shape(2, -1));
            var y = np.right_shift(x, 2);
            var z = x >> 2;

            print(x);
            print(y);
            print(z);

            var ExpectedData = new decimal[,]
            {
                { 255, 256, 256, 256, 256, 257, 257, 257 },
                { 257, 258, 258, 258, 258, 259, 259, 259 }
            };

            AssertArray(y, ExpectedData);
            AssertArray(z, ExpectedData);
        }

        [TestMethod]
        public void test_left_shift_DECIMAL()
        {
            var x = np.arange(1023, 1039, 1, dtype: np.Decimal).reshape(new shape(2, -1));
            var y = np.left_shift(x, 2);
            var z = x << 2;

            print(x);
            print(y);
            print(z);

            var ExpectedData = new decimal[,]
            {
                { 4092, 4096, 4100, 4104, 4108, 4112, 4116, 4120 },
                { 4124, 4128, 4132, 4136, 4140, 4144, 4148, 4152 }
            };

            AssertArray(y, ExpectedData);
            AssertArray(z, ExpectedData);
        }

        [TestMethod]
        public void test_min_DECIMAL()
        {
            decimal[] TestData = new decimal[] { 2.5m, -1.7m, -1.5m, -0.2m, 0.2m, 1.5m, 1.7m, 2.0m };
            var x = np.array(TestData);
            decimal y = (decimal)np.min(x);

            print(x);
            print(y);

            Assert.AreEqual(-1.7m, y);
        }

        [TestMethod]
        public void test_max_DECIMAL()
        {
            decimal[] TestData = new decimal[] { 2.5m, -1.7m, -1.5m, -0.2m, 0.2m, 1.5m, 1.7m, 2.0m };
            var x = np.array(TestData);
            decimal y = (decimal)np.max(x);

            print(x);
            print(y);

            Assert.AreEqual(2.5m, y);
        }

        [TestMethod]
        public void test_isnan_DECIMAL()
        {
            decimal[] TestData = new decimal[] { -1.7m, 0, 0, 0.2m, 1.5m, 0, 2.0m };
            var x = np.array(TestData);
            var y = np.isnan(x);

            print(x);
            print(y);

            // decimals don't support NAN so must be false
            AssertArray(y, new bool[] { false, false, false, false, false, false, false });

        }

        [TestMethod]
        public void test_setdiff1d_DECIMAL()
        {
            decimal[] TestDataA = new decimal[] { 1, 2, 3, 2, 4, };
            decimal[] TestDataB = new decimal[] { 3, 4, 5, 6 };

            var a = np.array(TestDataA);
            var b = np.array(TestDataB);
            ndarray c = np.setdiff1d(a, b);

            print(a);
            print(b);
            print(c);

            AssertArray(c, new decimal[] { 1, 2 });

        }

        [TestMethod]
        public void test_setdiff1d_2_DECIMAL()
        {
            decimal[] TestDataB = new decimal[] { 3, 4, 5, 6 };

            var a = np.arange(1, 39, dtype: np.Decimal).reshape(new shape(2, -1));
            var b = np.array(TestDataB);
            ndarray c = np.setdiff1d(a, b);

            print(a);
            print(b);
            print(c);

            AssertArray(c, new decimal[] {1,  2,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                                         29, 30, 31, 32, 33, 34, 35, 36, 37, 38 });

        }

        [TestMethod]
        public void test_rot90_1_DECIMAL()
        {
            ndarray m = np.array(new Int32[,] { { 1, 2 }, { 3, 4 } }, np.Decimal);
            print(m);
            print("************");

            ndarray n = np.rot90(m);
            print(n);
            AssertArray(n, new decimal[,] { { 2, 4 }, { 1, 3 }, });
            print("************");

            n = np.rot90(m, 2);
            print(n);
            AssertArray(n, new decimal[,] { { 4, 3 }, { 2, 1 }, });
            print("************");

            m = np.arange(8, dtype: np.Decimal).reshape(new shape(2, 2, 2));
            n = np.rot90(m, 1, new int[] { 1, 2 });
            print(n);
            AssertArray(n, new decimal[,,] { { { 1, 3 }, { 0, 2 } }, { { 5, 7 }, { 4, 6 } } });

        }

        [TestMethod]
        public void test_flip_1_DECIMAL()
        {
            ndarray A = np.arange(8, dtype: np.Decimal).reshape(new shape(2, 2, 2));
            ndarray B = np.flip(A, 0);
            print(A);
            print("************");
            print(B);
            AssertArray(B, new decimal[,,] { { { 4, 5 }, { 6, 7 } }, { { 0, 1 }, { 2, 3 } } });

            print("************");
            ndarray C = np.flip(A, 1);
            print(C);
            AssertArray(C, new decimal[,,] { { { 2, 3 }, { 0, 1 } }, { { 6, 7 }, { 4, 5 } } });
            print("************");

        }

        [TestMethod]
        public void test_trim_zeros_1_DECIMAL()
        {
            ndarray a = np.array(new decimal[] { 0, 0, 0, 1, 2, 3, 0, 2, 1, 0 });

            var b = np.trim_zeros(a);
            print(b);
            AssertArray(b, new decimal[] { 1, 2, 3, 0, 2, 1 });

            var c = np.trim_zeros(a, "b");
            print(c);
            AssertArray(c, new decimal[] { 0, 0, 0, 1, 2, 3, 0, 2, 1 });
        }

        [TestMethod]
        public void test_logical_and_1_DECIMAL()
        {

            var x = np.arange(5, dtype: np.Decimal);
            var c = np.logical_and(x > 1, x < 4);
            AssertArray(c, new bool[] { false, false, true, true, false });
            print(c);

            var y = np.arange(6, dtype: np.Decimal).reshape((2, 3));
            var d = np.logical_and(y > 1, y < 4);
            AssertArray(d, new bool[,] { { false, false, true }, { true, false, false } });
            print(d);
        }

        [TestMethod]
        public void test_logical_or_1_DECIMAL()
        {

            var x = np.arange(5, dtype: np.Decimal);
            var c = np.logical_or(x < 1, x > 3);
            AssertArray(c, new bool[] { true, false, false, false, true });
            print(c);

            var y = np.arange(6, dtype: np.Decimal).reshape((2, 3));
            var d = np.logical_or(y < 1, y > 3);
            AssertArray(d, new bool[,] { { true, false, false }, { false, true, true } });
            print(d);
        }

        [TestMethod]
        public void test_logical_xor_1_DECIMAL()
        {

            var x = np.arange(5, dtype: np.Decimal);
            var c = np.logical_xor(x < 1, x > 3);
            AssertArray(c, new bool[] { true, false, false, false, true });
            print(c);

            var y = np.arange(6, dtype: np.Decimal).reshape((2, 3));
            var d = np.logical_xor(y < 1, y > 3);
            AssertArray(d, new bool[,] { { true, false, false }, { false, true, true } });
            print(d);

            var e = np.logical_xor(0, np.eye(2, dtype: np.Decimal));
            AssertArray(e, new bool[,] { { true, false }, { false, true } });
        }

        [TestMethod]
        public void test_logical_not_1_DECIMAL()
        {
            var x = np.arange(5, dtype: np.Decimal);
            var c = np.logical_not(x < 3);
            AssertArray(c, new bool[] { false, false, false, true, true });
            print(c);
        }

        [TestMethod]
        public void test_greater_1_DECIMAL()
        {
            var a = np.greater(new decimal[] { 4, 2, 1 }, new decimal[] { 2, 2, 2 });
            AssertArray(a, new bool[] { true, false, false });
            print(a);

            var b = np.greater(new decimal[] { 4, 2, 1 }, 1);
            AssertArray(b, new bool[] { true, true, false });
            print(b);

            var c = np.greater(2, new decimal[] { 4, 2, 1 });
            AssertArray(c, new bool[] { false, false, true });
            print(c);

        }

        [TestMethod]
        public void test_greater_equal_1_DECIMAL()
        {
            var a = np.greater_equal(new decimal[] { 4, 2, 1 }, new decimal[] { 2, 2, 2 });
            AssertArray(a, new bool[] { true, true, false });
            print(a);

            var b = np.greater_equal(new decimal[] { 4, 2, 1 }, 1);
            AssertArray(b, new bool[] { true, true, true });
            print(b);

            var c = np.greater_equal(2, new decimal[] { 4, 2, 1 });
            AssertArray(c, new bool[] { false, true, true });
            print(c);
        }

        [TestMethod]
        public void test_less_1_DECIMAL()
        {
            var a = np.less(new decimal[] { 4, 2, 1 }, new decimal[] { 2, 2, 2 });
            AssertArray(a, new bool[] { false, false, true });
            print(a);

            var b = np.less(new decimal[] { 4, 2, 1 }, 1);
            AssertArray(b, new bool[] { false, false, false });
            print(b);

            var c = np.less(2, new decimal[] { 4, 2, 1 });
            AssertArray(c, new bool[] { true, false, false });
            print(c);
        }

        [TestMethod]
        public void test_less_equal_1_DECIMAL()
        {
            var a = np.less_equal(new decimal[] { 4, 2, 1 }, new decimal[] { 2, 2, 2 });
            AssertArray(a, new bool[] { false, true, true });
            print(a);

            var b = np.less_equal(new decimal[] { 4, 2, 1 }, 1);
            AssertArray(b, new bool[] { false, false, true });
            print(b);

            var c = np.less_equal(2, new decimal[] { 4, 2, 1 });
            AssertArray(c, new bool[] { true, true, false });
            print(c);
        }

        [TestMethod]
        public void test_equal_1_DECIMAL()
        {
            var a = np.equal(new decimal[] { 4, 2, 1 }, new decimal[] { 2, 2, 2 });
            AssertArray(a, new bool[] { false, true, false });
            print(a);

            var b = np.equal(new decimal[] { 4, 2, 1 }, 1);
            AssertArray(b, new bool[] { false, false, true });
            print(b);

            var c = np.equal(2, new decimal[] { 4, 2, 1 });
            AssertArray(c, new bool[] { false, true, false });
            print(c);
        }

        [TestMethod]
        public void test_not_equal_1_DECIMAL()
        {
            var a = np.not_equal(new decimal[] { 4, 2, 1 }, new decimal[] { 2, 2, 2 });
            AssertArray(a, new bool[] { true, false, true });
            print(a);

            var b = np.not_equal(new decimal[] { 4, 2, 1 }, 1);
            AssertArray(b, new bool[] { true, true, false });
            print(b);

            var c = np.not_equal(2, new decimal[] { 4, 2, 1 });
            AssertArray(c, new bool[] { true, false, true });
            print(c);
        }

        [TestMethod]
        public void test_copyto_1_DECIMAL()
        {
            var a = np.zeros((10, 5), dtype: np.Decimal);
            var b = new int[] { 11, 22, 33, 44, 55 };
            np.copyto(a, b);

            AssertShape(a, 10, 5);
            Assert.AreEqual(1650m, a.Sum().GetItem(0));
            print(a);

            a = np.zeros((10, 5), dtype: np.Decimal);
            np.copyto(a, 99);
            AssertShape(a, 10, 5);
            Assert.AreEqual(4950m, a.Sum().GetItem(0));
            print(a);

            a = np.zeros((10, 5), dtype: np.Decimal);
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
        public void test_copyto_2_DECIMAL()
        {
            var a = np.zeros((1, 2, 2, 1, 2), dtype: np.Decimal);
            var b = new int[] { 1, 2 };
            np.copyto(a, b);

            AssertArray(a, new decimal[,,,,] { { { { { 1.0m, 2.0m } }, { { 1.0m, 2.0m } } }, { { { 1.0m, 2.0m } }, { { 1.0m, 2.0m, } } } } });

        }

        #endregion

        #region from MathematicalFunctionsTests

        [TestMethod]
        public void test_sin_1_DECIMAL()
        {
            var ExpectedResult = new double[] { 0, 0.909297426825682, -0.756802495307928, -0.279415498198926, 0.989358246623382 };

            var a = np.arange(0, 10, dtype: np.Decimal);
            a = a["::2"] as ndarray;
            var b = np.sin(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.arange(0, 10, dtype: np.Decimal).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.sin(a);

            var ExpectedDataB = new double[,,]
                {{{ 0,                  0.841470984807897, 0.909297426825682, 0.141120008059867, -0.756802495307928},
                  {-0.958924274663138, -0.279415498198926, 0.656986598718789, 0.989358246623382,  0.412118485241757}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.array(new decimal[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.sin(a, where: a > 2);
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, 0.141120008059867, -0.756802495307928 } });
            print(b);

            a = np.array(new decimal[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.sin(a, where: new bool[,] { { false, false, false, true, true } });
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, 0.141120008059867, -0.756802495307928 } });
            print(b);

        }

        [TestMethod]
        public void test_cos_1_DECIMAL()
        {
            var ExpectedResult = new double[] { 1.0, -0.416146836547142, -0.653643620863612, 0.960170286650366, -0.145500033808614 };

            var a = np.arange(0, 10, dtype: np.Decimal);
            a = a["::2"] as ndarray;
            var b = np.cos(a);
            AssertArray(b, ExpectedResult);
            print(b);

            print("********");

            a = np.arange(0, 10, dtype: np.Decimal).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.cos(a);

            var ExpectedDataB = new double[,,]
                {{{ 1.0,               0.54030230586814, -0.416146836547142, -0.989992496600445, -0.653643620863612},
                  { 0.283662185463226, 0.960170286650366, 0.753902254343305, -0.145500033808614, -0.911130261884677}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.array(new decimal[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.cos(a, where: a > 2);
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, -0.989992496600445, -0.65364362086361 } });
            print(b);

            a = np.array(new decimal[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.cos(a, where: new bool[,] { { false, false, false, true, true } });
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, -0.989992496600445, -0.65364362086361 } });
            print(b);

        }

        [TestMethod]
        public void test_tan_1_DECIMAL()
        {
            var ExpectedResult = new double[] { 0.0, -2.18503986326152, 1.15782128234958, -0.291006191384749, -6.79971145522038 };

            var a = np.arange(0, 10, dtype: np.Decimal);
            a = a["::2"] as ndarray;
            var b = np.tan(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.arange(0, 10, dtype: np.Decimal).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.tan(a);

            var ExpectedDataB = new double[,,]
                {{{ 0.0, 1.5574077246549, -2.18503986326152, -0.142546543074278, 1.15782128234958},
                  { -3.38051500624659, -0.291006191384749, 0.871447982724319, -6.79971145522038, -0.45231565944181}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.array(new decimal[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.tan(a, where: a > 2);
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, -0.142546543074278, 1.15782128234958 } });
            print(b);

            a = np.array(new decimal[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.tan(a, where: new bool[,] { { false, false, false, true, true } });
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, -0.142546543074278, 1.15782128234958 } });
            print(b);

        }

        [TestMethod]
        public void test_arcsin_1_DECIMAL()
        {
            var ExpectedResult = new double[] { -1.5707963267949, -0.958241588455558, -0.6897750007855, -0.471861837279642,
                                                -0.276226630763592, -0.091034778037415, 0.091034778037415, 0.276226630763592,
                                                 0.471861837279642, 0.6897750007855, 0.958241588455558, 1.5707963267949 };

            double ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Decimal);
            var b = np.arcsin(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Decimal).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arcsin(a);

            var ExpectedDataB = new double[,,]
                {{{ -1.5707963267949, -0.958241588455558, -0.6897750007855},
                  { -0.471861837279642, -0.276226630763592, -0.091034778037415}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Decimal);
            a = a.A("::2");
            b = np.arcsin(a, where: a > -0.5);
            AssertArray(b, new double[] { np.NaN, np.NaN, -0.276226630763592, 0.091034778037415, 0.471861837279642, 0.958241588455558 });
            print(b);

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Decimal);
            a = a.A("::2");
            b = np.arcsin(a, where: new bool[] { false, false, true, true, true, true });
            AssertArray(b, new double[] { np.NaN, np.NaN, -0.276226630763592, 0.091034778037415, 0.471861837279642, 0.958241588455558 });
            print(b);

        }

        [TestMethod]
        public void test_arccos_1_DECIMAL()
        {
            var ExpectedResult = new double[] { 3.14159265358979, 2.52903791525045, 2.2605713275804, 2.04265816407454,
                                                1.84702295755849, 1.66183110483231, 1.47976154875748, 1.29456969603131,
                                                1.09893448951525, 0.881021326009397, 0.612554738339339, 0.0 };

            double ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Decimal);
            var b = np.arccos(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Decimal).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arccos(a);

            var ExpectedDataB = new double[,,]
                {{{3.14159265358979, 2.52903791525045, 2.2605713275804},
                  {2.04265816407454, 1.84702295755849, 1.66183110483231}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Decimal);
            a = a.A("::2");
            b = np.arccos(a, where: a > -0.5);
            AssertArray(b, new double[] { np.NaN, np.NaN, 1.84702295755849, 1.47976154875748, 1.09893448951525, 0.612554738339339 });
            print(b);

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Decimal);
            a = a.A("::2");
            b = np.arccos(a, where: new bool[] { false, false, true, true, true, true });
            AssertArray(b, new double[] { np.NaN, np.NaN, 1.84702295755849, 1.47976154875748, 1.09893448951525, 0.612554738339339 });
            print(b);

        }

        [TestMethod]
        public void test_arctan_1_DECIMAL()
        {
            var ExpectedResult = new double[] { -0.785398163397448, -0.685729510906286, -0.566729217523506, -0.426627493126876,
                                                -0.266252049150925, -0.090659887200745, 0.090659887200745,   0.266252049150925,
                                                 0.426627493126876, 0.566729217523506, 0.685729510906286, 0.785398163397448 };

            double ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Decimal);
            var b = np.arctan(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Decimal).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arctan(a);

            var ExpectedDataB = new double[,,]
                {{{-0.785398163397448, -0.685729510906286, -0.566729217523506},
                  {-0.426627493126876, -0.266252049150925, -0.090659887200745}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Decimal);
            a = a.A("::2");
            b = np.arctan(a, where: a > -0.5);
            AssertArray(b, new double[] { np.NaN, np.NaN, -0.266252049150925, 0.090659887200745, 0.426627493126876, 0.685729510906286 });
            print(b);

            a = np.linspace(-1.0, 1.0, ref ref_step, 12, dtype: np.Decimal);
            a = a.A("::2");
            b = np.arctan(a, where: new bool[] { false, false, true, true, true, true });
            AssertArray(b, new double[] { np.NaN, np.NaN, -0.266252049150925, 0.090659887200745, 0.426627493126876, 0.685729510906286 });
            print(b);

        }

        [TestMethod]
        public void test_hypot_1_DECIMAL()
        {

            var a = np.hypot(np.ones((3, 3), dtype: np.Decimal) * 3, np.ones((3, 3), dtype: np.Decimal) * 4);
            print(a);
            AssertArray(a, new decimal[,] { { 5, 5, 5 }, { 5, 5, 5 }, { 5, 5, 5 } });

            var b = np.hypot(np.ones((3, 3), dtype: np.Decimal) * 3, new decimal[] { 4 });
            print(b);
            AssertArray(b, new decimal[,] { { 5, 5, 5 }, { 5, 5, 5 }, { 5, 5, 5 } });

        }

        [TestMethod]
        public void test_arctan2_1_DECIMAL()
        {
            var x = np.array(new decimal[] { -1, +1, +1, -1 });
            var y = np.array(new decimal[] { -1, -1, +1, +1 });
            var z = np.arctan2(y, x) * 180 / Math.PI;
            AssertArray(z, new double[] { -135.0, -45.0, 45.0, 135.0 });
            print(z);

            var a = np.arctan2(new decimal[] { 1.0m, -1.0m }, new decimal[] { 0.0m, 0.0m });
            AssertArray(a, new double[] { 1.5707963267949, -1.5707963267949 });
            print(a);

        }

        [TestMethod]
        public void test_degrees_1_DECIMAL()
        {
            var rad = np.arange(12.0, dtype: np.Decimal) * Math.PI / 6;
            var a = np.degrees(rad);
            AssertArray(a, new double[] { 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330 });
            print(a);

            //var _out = np.zeros((rad.shape));
            //var r = np.degrees(rad, _out);
            //print(np.all(r == _out));

        }

        [TestMethod]
        public void test_radians_1_DECIMAL()
        {
            var deg = np.arange(12.0, dtype: np.Decimal) * 30.0;
            var a = np.radians(deg);
            AssertArray(a, new double[] { 0.0, 0.523598775598299, 1.0471975511966, 1.5707963267949, 2.0943951023932,
                                         2.61799387799149, 3.14159265358979, 3.66519142918809, 4.18879020478639,
                                        4.71238898038469, 5.23598775598299, 5.75958653158129 });
            print(a);

            //var _out = np.zeros((deg.shape));
            //var r = np.radians(deg, _out);
            //print(np.all(r == _out));

        }

        [TestMethod]
        public void test_around_1_DECIMAL()
        {
            ndarray a = np.around(np.array(new decimal[] { 0.37m, 1.64m }));
            print(a);
            AssertArray(a, new decimal[] { 0, 2 });

            ndarray b = np.around(np.array(new decimal[] { 0.37m, 1.64m }), decimals: 1);
            print(b);
            AssertArray(b, new decimal[] { 0.4m, 1.6m });

            ndarray c = np.around(np.array(new decimal[] { .5m, 1.5m, 2.5m, 3.5m, 4.5m })); // rounds to nearest even value
            print(c);
            AssertArray(c, new decimal[] { 0.0m, 2.0m, 2.0m, 4.0m, 4.0m });

            ndarray d = np.around(np.array(new decimal[] { 1, 2, 3, 11 }), decimals: 1); // ndarray of ints is returned
            print(d);
            AssertArray(d, new decimal[] { 1, 2, 3, 11 });

            ndarray e = np.around(np.array(new decimal[] { 1, 2, 3, 11 }), decimals: -1);
            print(e);
            AssertArray(e, new decimal[] { 0, 0, 0, 10 });
        }

        [TestMethod]
        public void test_round_1_DECIMAL()
        {
            decimal ref_step = 0;
            var a = np.linspace(-1.0m, 1.0m, ref ref_step, 12).reshape((2, 2, 3));
            print(a);

            var ExpectedData1 = new decimal[,,] { { { -1.0m, -0.82m, -0.64m }, { -0.45m, -0.27m, -0.09m } }, { { 0.09m, 0.27m, 0.45m }, { 0.64m, 0.82m, 1.0m } } };

            print("********");
            var b = np.round_(a, 2);
            AssertArray(b, ExpectedData1);
            print(b);

            print("********");

            var c = np.round(a, 2);
            AssertArray(c, ExpectedData1);
            print(c);

            var ExpectedData2 = new decimal[,,] { { { -1.0m, -0.8182m, -0.6364m }, { -0.4545m, -0.2727m, -0.0909m } }, { { 0.0909m, 0.2727m, 0.4545m }, { 0.6364m, 0.8182m, 1.0m } } };

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
        public void test_fix_1_DECIMAL()
        {
            var a = np.fix(3.14m);
            Assert.AreEqual(3.0m, a.GetItem(0));
            print(a);

            var b = np.fix(3m);
            Assert.AreEqual(3m, b.GetItem(0));
            print(b);

            var c = np.fix(new decimal[] { 2.1m, 2.9m, -2.1m, -2.9m });
            AssertArray(c, new decimal[] { 2.0m, 2.0m, -2.0m, -2.0m });
            print(c);
        }

        [TestMethod]
        public void test_floor_1_DECIMAL()
        {
            decimal[] TestData = new decimal[] { -1.7m, -1.5m, -0.2m, 0.2m, 1.5m, 1.7m, 2.0m };
            var x = np.array(TestData);
            var y = np.floor(x);

            print(x);
            print(y);

            AssertArray(y, new decimal[] { -2.0m, -2.0m, -1.0m, 0.0m, 1.0m, 1.0m, 2.0m });

        }

        [TestMethod]
        public void test_ceil_1_DECIMAL()
        {
            decimal[] TestData = new decimal[] { -1.7m, -1.5m, -0.2m, 0.2m, 1.5m, 1.7m, 2.0m };
            var x = np.array(TestData);
            var y = np.ceil(x);

            print(x);
            print(y);

            AssertArray(y, new decimal[] { -1.0m, -1.0m, -0.0m, 1.0m, 2.0m, 2.0m, 2.0m });

        }

        [TestMethod]
        public void test_trunc_1_DECIMAL()
        {
            var a = np.trunc(3.14m);
            Assert.AreEqual(3.0m, a.GetItem(0));
            print(a);

            var b = np.trunc(3m);
            Assert.AreEqual(3m, b.GetItem(0));
            print(b);

            var c = np.trunc(new decimal[] { 2.1m, 2.9m, -2.1m, -2.9m });
            AssertArray(c, new decimal[] { 2.0m, 2.0m, -2.0m, -2.0m });
            print(c);
        }

        [TestMethod]
        public void test_prod_2_DECIMAL()
        {
            ndarray a = np.prod(np.array(new decimal[] { 1.0m, 2.0m }));
            print(a);
            Assert.AreEqual(2m, a.GetItem(0));
            print("*****");

            ndarray b = np.prod(np.array(new decimal[,] { { 1.0m, 2.0m }, { 3.0m, 4.0m } }));
            print(b);
            Assert.AreEqual(24m, b.GetItem(0));
            print("*****");

            ndarray c = np.prod(np.array(new decimal[,] { { 1.0m, 2.0m }, { 3.0m, 4.0m } }), axis: 1);
            print(c);
            AssertArray(c, new decimal[] { 2, 12 });
            print("*****");

            ndarray d = np.array(new decimal[] { 1, 2, 3 }, dtype: np.Decimal);
            bool e = np.prod(d).Dtype.TypeNum == NPY_TYPES.NPY_DECIMAL;
            print(e);
            Assert.AreEqual(true, e);
            print("*****");

        }

        [TestMethod]
        public void test_sum_2_DECIMAL()
        {
            decimal[] TestData = new decimal[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.Decimal).reshape(new shape(3, 2, -1));
            x = x * 3;

            var y = np.sum(x, axis: 0);
            print(y);
            AssertArray(y, new decimal[,] { { 339, 450 }, { 339, 450 } });

            print("*****");

            y = np.sum(x, axis: 1);
            print(y);
            AssertArray(y, new decimal[,] { { 105, 180 }, { 264, 315 }, { 309, 405 } });

            print("*****");

            y = np.sum(x, axis: 2);
            print(y);
            AssertArray(y, new decimal[,] { { 75, 210 }, { 504, 75 }, { 210, 504 } });

            print("*****");

        }

        [TestMethod]
        public void test_cumprod_2_DECIMAL()
        {
            ndarray a = np.array(new decimal[] { 1, 2, 3 });
            ndarray b = np.cumprod(a);          // intermediate results 1, 1*2
                                                // total product 1*2*3 = 6
            print(b);
            AssertArray(b, new decimal[] { 1, 2, 6 });
            print("*****");

            a = np.array(new decimal[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            ndarray c = np.cumprod(a, dtype: np.Decimal); //specify type of output
            print(c);
            AssertArray(c, new decimal[] { 1m, 2m, 6m, 24m, 120m, 720m });
            print("*****");

            ndarray d = np.cumprod(a, axis: 0);
            print(d);
            AssertArray(d, new decimal[,] { { 1, 2, 3 }, { 4, 10, 18 } });
            print("*****");

            ndarray e = np.cumprod(a, axis: 1);
            print(e);
            AssertArray(e, new decimal[,] { { 1, 2, 6 }, { 4, 20, 120 } });
            print("*****");

        }

        [TestMethod]
        public void test_cumsum_3_DECIMAL()
        {
            ndarray a = np.array(new decimal[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }).reshape(new shape(2, 3, -1));
            print(a);
            print("*****");

            ndarray b = np.cumsum(a);
            print(b);
            AssertArray(b, new decimal[] { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78 });
            print("*****");

            ndarray c = np.cumsum(a, dtype: np.Decimal);     // specifies type of output value(s)
            print(c);
            AssertArray(c, new decimal[] { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78 });
            print("*****");

            ndarray d = np.cumsum(a, axis: 0);     // sum over rows for each of the 3 columns
            print(d);

            var ExpectedDataD = new decimal[,,]
            {{{1,  2},
              {3,  4},
              {5,  6}},

             {{ 8, 10},
              {12, 14},
              {16, 18}}};

            AssertArray(d, ExpectedDataD);
            print("*****");



            ndarray e = np.cumsum(a, axis: 1);    // sum over columns for each of the 2 rows
            print(e);

            var ExpectedDataE = new decimal[,,]
            {{{1,  2},
              {4,  6},
              {9,  12}},

             {{ 7, 8},
              {16, 18},
              {27, 30}}};

            AssertArray(e, ExpectedDataE);
            print("*****");

            ndarray f = np.cumsum(a, axis: 2);    // sum over columns for each of the 2 rows
            print(f);

            var ExpectedDataF = new decimal[,,]
            {{{1,  3},
              {3,  7},
              {5,  11}},

             {{7, 15},
              {9, 19},
              {11, 23}}};

            AssertArray(f, ExpectedDataF);
            print("*****");

        }

        [TestMethod]
        public void test_diff_3_DECIMAL()
        {
            decimal[] TestData = new decimal[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.Decimal).reshape(new shape(3, 2, -1));
            x = x * 3;
            var y = np.diff(x, axis: 2);

            print(x);
            print(y);

            var ExpectedData = new decimal[,,]
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
        public void test_ediff1d_1_DECIMAL()
        {
            ndarray x = np.array(new decimal[] { 1, 2, 4, 7, 0 });
            ndarray y = np.ediff1d(x);
            print(y);
            AssertArray(y, new decimal[] { 1, 2, 3, -7 });

            y = np.ediff1d(x, to_begin: np.array(new decimal[] { -99 }), to_end: np.array(new decimal[] { 88, 99 }));
            print(y);
            AssertArray(y, new decimal[] { -99, 1, 2, 3, -7, 88, 99 });

            x = np.array(new decimal[,] { { 1, 2, 4 }, { 1, 6, 24 } });
            y = np.ediff1d(x);
            print(y);
            AssertArray(y, new decimal[] { 1, 2, -3, 5, 18 });

        }

        [TestMethod]
        public void test_gradient_1_DECIMAL()
        {
            var f = np.array(new decimal[] { 1, 2, 4, 7, 11, 16 }, dtype: np.Decimal);
            var a = np.gradient(f);
            AssertArray(a[0], new decimal[] { 1m, 1.5m, 2.5m, 3.5m, 4.5m, 5m });
            print(a[0]);
            print("***********");

            var b = np.gradient(f, new object[] { 2 });
            AssertArray(b[0], new decimal[] { 0.5m, 0.75m, 1.25m, 1.75m, 2.25m, 2.5m });
            print(b[0]);
            print("***********");

            // Spacing can be also specified with an array that represents the coordinates
            // of the values F along the dimensions.
            // For instance a uniform spacing:

            var x = np.arange(f.size);
            var c = np.gradient(f, new object[] { x });
            AssertArray(c[0], new decimal[] { 1.0m, 1.5m, 2.5m, 3.5m, 4.5m, 5.0m });
            print(c[0]);
            print("***********");

            // Or a non uniform one:

            x = np.array(new decimal[] { 0.0m, 1.0m, 1.5m, 3.5m, 4.0m, 6.0m }, dtype: np.Decimal);
            var d = np.gradient(f, new object[] { x });
            AssertArray(d[0], new decimal[] { 1.0m, 02.99999999999999999999999999990m, 3.5m, 6.7m, 6.9m, 2.5m });
            print(d[0]);
        }

        [TestMethod]
        public void test_cross_2_DECIMAL()
        {
            // Multiple vector cross-products. Note that the direction of the cross
            // product vector is defined by the `right-hand rule`.

            var x = np.array(new decimal[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var y = np.array(new decimal[,] { { 4, 5, 6 }, { 1, 2, 3 } });
            var a = np.cross(x, y);
            AssertArray(a, new decimal[,] { { -3, 6, -3 }, { 3, -6, 3 } });
            print(a);


            // The orientation of `c` can be changed using the `axisc` keyword.

            var b = np.cross(x, y, axisc: 0);
            AssertArray(b, new decimal[,] { { -3, 3 }, { 6, -6 }, { -3, 3 } });
            print(b);

            // Change the vector definition of `x` and `y` using `axisa` and `axisb`.

            x = np.array(new decimal[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
            y = np.array(new decimal[,] { { 7, 8, 9 }, { 4, 5, 6 }, { 1, 2, 3 } });
            a = np.cross(x, y);
            AssertArray(a, new decimal[,] { { -6, 12, -6 }, { 0, 0, 0 }, { 6, -12, 6 } });
            print(a);

            b = np.cross(x, y, axisa: 0, axisb: 0);
            AssertArray(b, new decimal[,] { { -24, 48, -24 }, { -30, 60, -30 }, { -36, 72, -36 } });
            print(b);

            return;
        }

        [TestMethod]
        public void test_trapz_1_DECIMAL()
        {
            var a = np.trapz(new decimal[] { 1, 2, 3 });
            Assert.AreEqual((double)4.0, a.GetItem(0));
            print(a);

            var b = np.trapz(new decimal[] { 1, 2, 3 }, x: new int[] { 4, 6, 8 });
            Assert.AreEqual((double)8.0, b.GetItem(0));
            print(b);

            var c = np.trapz(new decimal[] { 1, 2, 3 }, dx: 2);
            Assert.AreEqual((double)8.0, c.GetItem(0));
            print(c);

            a = np.arange(6, dtype: np.Decimal).reshape((2, 3));
            b = np.trapz(a, axis: 0);
            AssertArray(b, new double[] { 1.5, 2.5, 3.5 });
            print(b);

            c = np.trapz(a, axis: 1);
            AssertArray(c, new double[] { 2.0, 8.0 });
            print(c);
        }

        [TestMethod]
        public void test_exp_1_DECIMAL()
        {
            var x = np.array(new decimal[] { -1.7m, -1.5m, -0.2m, 0.2m, 1.5m, 1.7m, 2.0m, -4.2m });
            var a = np.exp(x);
            AssertArray(a, new double[] { 0.182683524052735, 0.22313016014843, 0.818730753077982, 1.22140275816017,
                                          4.48168907033806, 5.4739473917272, 7.38905609893065, 0.0149955768204777 });
            print(a);


            a = np.exp(x.reshape((2, -1)));
            AssertArray(a, new double[,] { {0.182683524052735, 0.22313016014843, 0.818730753077982, 1.22140275816017 },
                                           {4.48168907033806, 5.4739473917272, 7.38905609893065, 0.0149955768204777  } });
            print(a);

            a = np.exp(x, where: x > 0);
            AssertArray(a, new double[] { double.NaN, double.NaN, double.NaN, 1.22140275816017,
                                          4.48168907033806, 5.4739473917272, 7.38905609893065, double.NaN });
            print(a);

        }

        [TestMethod]
        public void test_exp2_1_DECIMAL()
        {
            var x = np.array(new decimal[] { -1.7m, -1.5m, -0.2m, 0.2m, 1.5m, 1.7m, 2.0m, -4.2m });
            var a = np.exp2(x);
            AssertArray(a, new double[] { 0.307786103336229, 0.353553390593274, 0.870550563296124, 1.14869835499704,
                                          2.82842712474619,  3.24900958542494,  4.0,               0.0544094102060078 });
            print(a);


            a = np.exp2(x.reshape((2, -1)));
            AssertArray(a, new double[,] { {0.307786103336229, 0.353553390593274, 0.870550563296124, 1.14869835499704, },
                                           {2.82842712474619,  3.24900958542494,  4.0,               0.0544094102060078  } });
            print(a);

            a = np.exp2(x, where: x > 0);
            AssertArray(a, new double[] { double.NaN, double.NaN, double.NaN, 1.14869835499704,
                                          2.82842712474619,  3.24900958542494,  4.0, double.NaN });
            print(a);

        }

        [TestMethod]
        public void test_i0_1_DECIMAL()
        {
            var a = np.i0(5.0m);
            Assert.AreEqual(27.2398718236044m, a.GetItem(0));
            print(a);

            a = np.i0(new decimal[] { 5.0m, 6.0m });
            AssertArray(a, new decimal[] { 27.2398718236044m, 67.234406976478m });
            print(a);

            a = np.i0(new decimal[,] { { 27.2398718236044m, 67.234406976478m }, { 389.40628328m, 427.56411572m } });
            AssertArray(a, new decimal[,] { { 51935526724.2882m, 7717199833565030000000000000.0m }, { 0.0m, 0.0m } });
            print(a);

            return;

        }

        [TestMethod]
        public void test_sinc_1_DECIMAL()
        {
            double retstep = 0;
            var x = np.linspace(-4, 4, ref retstep, 10, dtype: np.Decimal);
            var a = np.sinc(x);
            AssertArray(a, new double[] {-3.89817183e-17, -3.49934120e-02,  9.20725429e-02, -2.06748336e-01, 7.05316598e-01,
                                          7.05316598e-01, -2.06748336e-01,  9.20725429e-02, -3.49934120e-02, -3.89817183e-17 });
            print(a);

            print("********");

            var xx = np.outer(x, x);
            var b = np.sinc(xx);

            var ExpectedDataB = new double[,]

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
        public void test_signbit_1_DECIMAL()
        {
            var a = np.signbit(-1.2m);
            Assert.AreEqual(true, a.GetItem(0));
            print(a);

            var b = np.signbit(np.array(new decimal[] { 1, -2.3m, 2.1m }));
            AssertArray(b, new bool[] { false, true, false });
            print(b);

            var c = np.signbit(np.array(new decimal[] { +0.0m, -0.0m }));  // note: different result than python.  No such thing as -0.0
            AssertArray(c, new bool[] { false, false });
            print(c);

            var f = np.signbit(np.array(new decimal[] { -1, 0, 1 }));
            AssertArray(f, new bool[] { true, false, false });
            print(f);
        }

        [TestMethod]
        public void test_copysign_1_DECIMAL()
        {
            var a = np.copysign(1.3m, -1m);
            Assert.AreEqual(-1.3, a.GetItem(0));
            print(a);

            var b = np.divide(1, np.copysign(0m, 1m));
            Assert.AreEqual((double)0, b.GetItem(0));  // note: python gets a np.inf value here
            print(b);

            var c = 1 / np.copysign(0m, -1m);
            Assert.AreEqual((double)0, c.GetItem(0));  // note: python gets a -np.inf value here
            print(c);


            var d = np.copysign(new decimal[] { -1, 0, 1 }, -1.1m);
            AssertArray(d, new double[] { -1, 0, -1 });
            print(d);

            var e = np.copysign(new decimal[] { -1, 0, 1 }, np.arange(3) - 1);
            AssertArray(e, new double[] { -1, 0, 1 });
            print(e);
        }

        [TestMethod]
        public void test_frexp_1_DECIMAL()
        {
            var x = np.arange(9, dtype: np.Decimal);
            var results = np.frexp(x);

            AssertArray(results[0], new double[] { 0.0, 0.5, 0.5, 0.75, 0.5, 0.625, 0.75, 0.875, 0.5 });
            AssertArray(results[1], new int[] { 0, 1, 2, 2, 3, 3, 3, 3, 4 });

            print(results[0]);
            print(results[1]);

            print("***************");


            x = np.arange(9, dtype: np.Decimal).reshape((3, 3));
            results = np.frexp(x, where: x < 5);

            AssertArray(results[0], new double[,] { { 0.0, 0.5, 0.5 }, { 0.75, 0.5, double.NaN }, { double.NaN, double.NaN, double.NaN } });
            AssertArray(results[1], new int[,] { { 0, 1, 2 }, { 2, 3, 0 }, { 0, 0, 0 } });

            print(results[0]);
            print(results[1]);
        }

        [TestMethod]
        public void test_ldexp_1_DECIMAL()
        {
            var a = np.ldexp(5m, np.arange(4, dtype: np.Decimal));
            AssertArray(a, new double[] { 5.0f, 10.0f, 20.0f, 40.0f });
            print(a);

            var b = np.ldexp(np.arange(4, dtype: np.Decimal), 5m);
            AssertArray(b, new double[] { 0.0, 32.0, 64.0, 96.0 });
            print(b);
        }

        [TestMethod]
        public void test_lcm_1_DECIMAL()
        {
            int success = 0;
            try
            {
                var a = np.lcm(12m, 20m);
                Assert.AreEqual(60m, a.GetItem(0));
                print(a);
                success++;
            }
            catch
            { }

            try
            {
                var d = np.lcm(np.arange(6, dtype: np.Decimal), new decimal[] { 20 });
                AssertArray(d, new decimal[] { 0, 20, 20, 60, 20, 20 });
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
                var e = np.lcm(new decimal[] { 20, 21 }, np.arange(6, dtype: np.Decimal).reshape((3, 2)));
                AssertArray(e, new decimal[,] { { 0, 21 }, { 20, 21 }, { 20, 105 } });
                print(e);
                success++;
            }

            Assert.AreEqual(0, success, "Did not catch all exceptions as expected");
        }

        [TestMethod]
        public void test_gcd_1_DECIMAL()
        {
            int success = 0;

            try
            {
                var a = np.gcd(12m, 20m);
                Assert.AreEqual(4, a.GetItem(0));
                print(a);
                success++;
            }
            catch
            { }

            try
            {
                var d = np.gcd(np.arange(6, dtype: np.Decimal), new decimal[] { 20 });
                AssertArray(d, new int[] { 20, 1, 2, 1, 4, 5 });
                print(d);
                success++;
            }
            catch
            { }

            try
            {
                var e = np.gcd(new decimal[] { 20, 20 }, np.arange(6, dtype: np.Decimal).reshape((3, 2)));
                AssertArray(e, new int[,] { { 20, 1 }, { 2, 1 }, { 4, 5 } });
                print(e);
                success++;
            }
            catch
            { }

            try
            {
                var f = np.gcd(new decimal[] { 20, 20 }, np.arange(6, dtype: np.Decimal).reshape((3, 2)));
                AssertArray(f, new long[,] { { 20, 1 }, { 2, 1 }, { 4, 5 } });
                print(f);
                success++;
            }
            catch
            { }

            Assert.AreEqual(0, success, "Did not catch all exceptions as expected");
        }

        [TestMethod]
        public void test_add_1_DECIMAL()
        {
            var a = np.add(1.0m, 4.0m);
            Assert.AreEqual(5.0m, a.GetItem(0));
            print(a);

            var b = np.arange(9.0m).reshape((3, 3));
            var c = np.arange(3.0m);
            var d = np.add(b, c);
            AssertArray(d, new decimal[,] { { 0, 2, 4 }, { 3, 5, 7 }, { 6, 8, 10 } });
            print(d);

        }

        [TestMethod]
        public void test_reciprocal_operations_DECIMAL()
        {
            var a = np.arange(1, 32, 1, dtype: np.Decimal);
            print(a);

            var b = np.reciprocal(a);
            print(b);

            var ExpectedDataB1 = new decimal[]
            {
               1.0m, 0.5m, 0.3333333333333333333333333333m, 0.25m, 0.2m, 0.1666666666666666666666666667m,
                0.1428571428571428571428571429m, 0.125m, 0.1111111111111111111111111111m, 0.1m, 0.0909090909090909090909090909m,
                0.0833333333333333333333333333m, 0.0769230769230769230769230769m, 0.0714285714285714285714285714m,
                0.0666666666666666666666666667m, 0.0625m, 0.0588235294117647058823529412m, 0.0555555555555555555555555556m,
                0.0526315789473684210526315789m, 0.05m, 0.0476190476190476190476190476m, 0.0454545454545454545454545455m,
                0.0434782608695652173913043478m, 0.0416666666666666666666666667m, 0.04m, 0.0384615384615384615384615385m,
                0.037037037037037037037037037m, 0.0357142857142857142857142857m, 0.0344827586206896551724137931m,
                0.0333333333333333333333333333m, 0.0322580645161290322580645161m
            };

            AssertArray(b, ExpectedDataB1);


            a = np.arange(2048, 2048 + 32, 1, dtype: np.Decimal);
            print(a);

            b = np.reciprocal(a);
            print(b);

            var ExpectedDataB2 = new decimal[]
            {
                0.00048828125m, 0.0004880429477794045876037091m, 0.000487804878048780487804878m, 0.0004875670404680643588493418m,
                0.0004873294346978557504873294m, 0.0004870920603994154895275207m, 0.0004868549172346640701071081m,
                0.0004866180048661800486618005m, 0.0004863813229571984435797665m, 0.000486144871171609139523578m,
                0.000485908649173955296404276m, 0.0004856726566294317629917436m, 0.0004854368932038834951456311m,
                0.0004852013585638039786511402m, 0.0004849660523763336566440349m, 0.0004847309743092583616093068m,
                0.0004844961240310077519379845m, 0.0004842615012106537530266344m, 0.0004840271055179090029041626m,
                0.0004837929366231253023705854m, 0.0004835589941972920696324952m, 0.0004833252779120347994200097m,
                0.0004830917874396135265700483m, 0.0004828585224529212940608402m, 0.0004826254826254826254826255m,
                0.0004823926676314520019295707m, 0.0004821600771456123432979749m, 0.0004819277108433734939759036m,
                0.0004816955684007707129094412m, 0.0004814636494944631680308137m, 0.0004812319538017324350336862m,
                0.0004810004810004810004810005m
            };
            AssertArray(b, ExpectedDataB2);
        }

        [TestMethod]
        public void test_positive_1_DECIMAL()
        {
            var d = np.positive(new decimal[] { -1, -0, 1 });
            AssertArray(d, new decimal[] { -1, -0, 1 });
            print(d);

            var e = np.positive(new decimal[,] { { 1, 0, -1 }, { -2, 3, -4 } });
            AssertArray(e, new decimal[,] { { 1, 0, -1 }, { -2, 3, -4 } });
            print(e);
        }

        [TestMethod]
        public void test_negative_1_DECIMAL()
        {
            var d = np.negative(new decimal[] { -1, -0, 1 });
            AssertArray(d, new decimal[] { 1, 0, -1 });
            print(d);

            var e = np.negative(new decimal[,] { { 1, 0, -1 }, { -2, 3, -4 } });
            AssertArray(e, new decimal[,] { { -1, 0, 1 }, { 2, -3, 4 } });
            print(e);
        }

        [TestMethod]
        public void test_multiply_1_DECIMAL()
        {
            var a = np.multiply(2.0m, 4.0m);
            Assert.AreEqual(8.0m, a.GetItem(0));
            print(a);

            var b = np.arange(9.0m).reshape((3, 3));
            var c = np.arange(3.0m);
            var d = np.multiply(b, c);
            AssertArray(d, new decimal[,] { { 0, 1, 4 }, { 0, 4, 10 }, { 0, 7, 16 } });
            print(d);
        }

        [TestMethod]
        public void test_divide_DECIMAL()
        {
            var a = np.divide(7m, 3m);
            Assert.AreEqual(2.3333333333333333333333333333m, a.GetItem(0));
            print(a);

            var b = np.divide(new decimal[] { 1.0m, 2.0m, 3.0m, 4.0m }, 2.5m);
            AssertArray(b, new decimal[] { 0.4m, 0.8m, 1.2m, 1.6m });
            print(b);

            var c = np.divide(new decimal[] { 1.0m, 2.0m, 3.0m, 4.0m }, new decimal[] { 0.5m, 2.5m, 2.5m, 3.5m });
            AssertArray(c, new decimal[] { 2.0m, 0.8m, 1.2m, 1.1428571428571428571428571429m });
            print(c);

            return;
        }

        [TestMethod]
        public void test_power_operations_DECIMAL()
        {
            var a = np.arange(0, 32, 1, dtype: np.Decimal);
            print(a);

            var b = np.power(a, 3.23m);
            print(b);

            var ExpectedDataB1 = new decimal[]
            { 0.0m, 1.0m, 9.38267959385503m, 34.7617516700826m, 88.0346763609436m, 180.997724101542m,
              326.15837804154m, 536.619770563306m, 826.001161443457m, 1208.37937917249m, 1698.24365246174m,
              2310.45956851781m, 3060.23955801521m, 3963.11822364251m, 5034.9313709235m, 6291.79793806794m,
              7750.10424197608m, 9426.49010646868m, 11337.836542597m, 13501.2547250997m, 15934.0760633466m,
              18653.8432055784m, 21678.3018459592m, 25025.3932276144m, 28713.2472532973m, 32760.176129938m,
              37184.6684850056m, 42005.3839020428m, 47241.1478304245m, 52910.9468307066m, 59033.9241221692m,
              65629.3754035258m
            };

            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Decimal);
            print(a);

            b = np.power(a, 4m);
            print(b);

            var ExpectedDataB2 = new decimal[]
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

            b = np.power(a, 0m);
            print(b);
            var ExpectedDataB3 = new decimal[]
            {
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            };
            AssertArray(b, ExpectedDataB3);


            b = np.power(a, 0.5m);
            print(b);

            var ExpectedDataB4 = new decimal[]
            {
                45.254833995939m, 45.2658811910251m, 45.2769256906871m, 45.287967496897m, 45.2990066116245m,
                45.3100430368368m, 45.3210767744986m, 45.3321078265725m, 45.3431361950185m, 45.3541618817943m,
                45.365184888855m, 45.3762052181537m, 45.3872228716409m, 45.3982378512647m, 45.4092501589709m,
                45.4202597967031m, 45.4312667664022m, 45.442271070007m, 45.453272709454m, 45.4642716866772m,
                45.4752680036083m, 45.4862616621766m, 45.4972526643093m, 45.508241011931m, 45.5192267069642m,
                45.5302097513288m, 45.5411901469428m, 45.5521678957215m, 45.5631429995781m, 45.5741154604234m,
                45.5850852801659m, 45.596052460712m
            };

            AssertArray(b, ExpectedDataB4);

        }

        [TestMethod]
        public void test_subtract_1_DECIMAL()
        {
            var a = np.subtract(1.0m, 4.0m);
            Assert.AreEqual(-3.0m, a.GetItem(0));
            print(a);

            var b = np.arange(9.0, dtype: np.Decimal).reshape((3, 3));
            var c = np.arange(3.0, dtype: np.Decimal);
            var d = np.subtract(b, c);
            AssertArray(d, new decimal[,] { { 0, 0, 0 }, { 3, 3, 3 }, { 6, 6, 6 } });
            print(d);
        }

        [TestMethod]
        public void test_true_divide_DECIMAL()
        {
            var a = np.true_divide(7m, 3m);
            Assert.AreEqual(2.3333333333333333333333333333m, a.GetItem(0));
            print(a);

            var b = np.true_divide(new decimal[] { 1.0m, 2.0m, 3.0m, 4.0m }, 2.5m);
            AssertArray(b, new decimal[] { 0.4m, 0.8m, 1.2m, 1.6m });
            print(b);

            var c = np.true_divide(new decimal[] { 1.0m, 2.0m, 3.0m, 4.0m }, new decimal[] { 0.5m, 2.5m, 2.5m, 3.5m });
            AssertArray(c, new decimal[] { 2.0m, 0.8m, 1.2m, 1.1428571428571428571428571429m });
            print(c);

            return;
        }

        [TestMethod]
        public void test_floor_divide_DECIMAL()
        {
            var a = np.floor_divide(7m, 3m);
            Assert.AreEqual(2.0m, a.GetItem(0));
            print(a);

            var b = np.floor_divide(new decimal[] { 1.0m, 2.0m, 3.0m, 4.0m }, 2.5m);
            AssertArray(b, new decimal[] { 0, 0, 1, 1 });
            print(b);

            var c = np.floor_divide(new decimal[] { 1.0m, 2.0m, 3.0m, 4.0m }, new decimal[] { 0.5m, 2.5m, 2.5m, 3.5m });
            AssertArray(c, new decimal[] { 2, 0, 1, 1 });
            print(c);

            return;

        }

        [TestMethod]
        public void test_float_power_DECIMAL()
        {
            var x1 = new decimal[] { 0, 1, 2, 3, 4, 5 };

            var a = np.float_power(x1, 3m);
            AssertArray(a, new double[] { 0.0, 1.0, 8.0, 27.0, 64.0, 125.0 });
            print(a);

            var x2 = new decimal[] { 1.0m, 2.0m, 3.0m, 3.0m, 2.0m, 1.0m };
            var b = np.float_power(x1, x2);
            AssertArray(b, new double[] { 0.0, 1.0, 8.0, 27.0, 16.0, 5.0 });
            print(b);

            var x3 = np.array(new decimal[,] { { 1, 2, 3, 3, 2, 1 }, { 1, 2, 3, 3, 2, 1 } });
            var c = np.float_power(x1, x3);
            AssertArray(c, new double[,] { { 0.0, 1.0, 8.0, 27.0, 16.0, 5.0 }, { 0.0, 1.0, 8.0, 27.0, 16.0, 5.0 } });
            print(c);

            return;
        }

        [TestMethod]
        public void test_fmod_2_DECIMAL()
        {
            var x = np.fmod(new decimal[] { -4, -7 }, new decimal[] { 2, 3 });
            AssertArray(x, new decimal[] { 0, -1 });
            print(x);

            var y = np.fmod(np.arange(7, dtype: np.Decimal), -5m);
            AssertArray(y, new decimal[] { 0, 1, 2, 3, 4, 0, 1 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_mod_1_DECIMAL()
        {
            var x = np.mod(new decimal[] { 4, 7 }, new decimal[] { 2, 3 });
            AssertArray(x, new decimal[] { 0, 1 });
            print(x);

            var y = np.mod(np.arange(7, dtype: np.Decimal), 5m);
            AssertArray(y, new decimal[] { 0, 1, 2, 3, 4, 0, 1 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_modf_1_DECIMAL()
        {
            var x = np.modf(new decimal[] { 0, 3.5m });
            AssertArray(x[0], new decimal[] { 0m, 0.5m });
            AssertArray(x[1], new decimal[] { 0m, 3.0m });
            print(x);

            var y = np.modf(np.arange(7, dtype: np.Decimal));
            AssertArray(y[0], new decimal[] { 0, 0, 0, 0, 0, 0, 0 });
            AssertArray(y[1], new decimal[] { 0, 1, 2, 3, 4, 5, 6 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_remainder_2_DECIMAL()
        {
            var x = np.remainder(new decimal[] { -4, -7 }, new decimal[] { 2, 3 });
            AssertArray(x, new decimal[] { 0, 2 });
            print(x);

            var y = np.remainder(np.arange(7, dtype: np.Decimal), -5m);
            AssertArray(y, new decimal[] { 0, -4, -3, -2, -1, 0, -4 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_divmod_1_DECIMAL()
        {
            var a = np.divmod(7m, 3m);
            Assert.AreEqual(2m, a[0].GetItem(0));
            Assert.AreEqual(1m, a[1].GetItem(0));

            print(a);

            var b = np.divmod(new decimal[] { 1.0m, 2.0m, 3.0m, 4.0m }, 2.5m);
            AssertArray(b[0], new decimal[] { 0, 0, 1, 1 });
            AssertArray(b[1], new decimal[] { 1, 2, 0.5m, 1.5m });
            print(b);

            var c = np.divmod(new decimal[] { 1.0m, 2.0m, 3.0m, 4.0m }, new decimal[] { 0.5m, 2.5m, 2.5m, 3.5m });
            AssertArray(c[0], new decimal[] { 2, 0, 1, 1 });
            AssertArray(c[1], new decimal[] { 0, 2, 0.5m, 0.5m });
            print(c);

            return;

        }

        [TestMethod]
        public void test_convolve_1_DECIMAL()
        {
            var a = np.convolve(new decimal[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f });
            AssertArray(a, new decimal[] { 0.0m, 1.0m, 2.5m, 4.0m, 1.5m });
            print(a);

            var b = np.convolve(new decimal[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_SAME);
            AssertArray(b, new decimal[] { 1.0m, 2.5m, 4.0m });
            print(b);

            var c = np.convolve(new decimal[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_VALID);
            AssertArray(c, new decimal[] { 2.5m });
            print(c);

            return;
        }

        [TestMethod]
        public void test_clip_2_DECIMAL()
        {
            ndarray a = np.arange(16, dtype: np.Decimal).reshape(new shape(4, 4));
            print(a);
            print("*****");

            ndarray b = np.clip(a, 1, 8);
            print(b);
            print("*****");
            AssertArray(b, new decimal[,] { { 1, 1, 2, 3 }, { 4, 5, 6, 7 }, { 8, 8, 8, 8 }, { 8, 8, 8, 8 } });

            ndarray c = np.clip(a, 3, 6, @out: a);
            print(c);
            AssertArray(c, new decimal[,] { { 3, 3, 3, 3 }, { 4, 5, 6, 6 }, { 6, 6, 6, 6 }, { 6, 6, 6, 6 } });
            print(a);
            AssertArray(a, new decimal[,] { { 3, 3, 3, 3 }, { 4, 5, 6, 6 }, { 6, 6, 6, 6 }, { 6, 6, 6, 6 } });
            print("*****");

            a = np.arange(16, dtype: np.Decimal).reshape(new shape(4, 4));
            print(a);
            b = np.clip(a, np.array(new decimal[] { 3, 4, 1, 1 }), 8);
            print(b);
            AssertArray(b, new decimal[,] { { 3, 4, 2, 3 }, { 4, 5, 6, 7 }, { 8, 8, 8, 8 }, { 8, 8, 8, 8 } });

        }

        [TestMethod]
        public void test_square_operations_DECIMAL()
        {
            var a = np.arange(0, 32, 1, dtype: np.Decimal);
            print(a);

            var b = np.square(a);
            print(b);

            var ExpectedDataB1 = new decimal[]
            {
                0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144, 169, 196, 225, 256, 289,
                324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961
            };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Decimal);
            print(a);

            b = np.square(a);
            print(b);

            var ExpectedDataB2 = new decimal[]
            {
                4194304, 4198401, 4202500, 4206601, 4210704, 4214809, 4218916, 4223025, 4227136,
                4231249, 4235364, 4239481, 4243600, 4247721, 4251844, 4255969, 4260096, 4264225,
                4268356, 4272489, 4276624, 4280761, 4284900, 4289041, 4293184, 4297329, 4301476,
                4305625, 4309776, 4313929, 4318084, 4322241
            };
            AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_absolute_operations_DECIMAL()
        {
            var a = np.arange(-32, 32, 1, dtype: np.Decimal);
            print(a);

            var b = np.absolute(a);
            print(b);

            var ExpectedDataB = new decimal[]
            {
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,
                2,  1,  0,  1,  2,  3,  4,  5,   6,  7,  8,  9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31
            };
        }

        [TestMethod]
        public void test_fabs_1_DECIMAL()
        {
            var a = np.arange(-32, 32, 1, dtype: np.Decimal);
            print(a);

            var b = np.fabs(a);
            print(b);

            var ExpectedDataB = new decimal[]
            {
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,
                2,  1,  0,  1,  2,  3,  4,  5,   6,  7,  8,  9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31
            };
        }

        [TestMethod]
        public void test_sign_1_DECIMAL()
        {
            var a = np.sign(-1.2m);
            Assert.AreEqual(-1.0, a.GetItem(0));
            print(a);

            var b = np.sign(np.array(new decimal[] { 1, -2.3m, 2.1m }));
            AssertArray(b, new double[] { 1, -1, 1 });
            print(b);

            var c = np.sign(np.array(new decimal[] { +0.0m, -0.0m }));
            AssertArray(c, new double[] { 0, 0 });
            print(c);


            var f = np.sign(np.array(new decimal[] { -1, 0, 1 }));
            AssertArray(f, new double[] { -1, 0, 1 });
            print(f);
        }

        [TestMethod]
        public void test_heaviside_1_DECIMAL()
        {
            var a = np.heaviside(new decimal[] { -1.5m, 0.0m, 2.0m }, 0.5m);
            AssertArray(a, new decimal[] { 0.0m, 0.5m, 1.0m });
            print(a);

            var b = np.heaviside(new decimal[] { -1.5m, 0m, 2.0m }, 1);
            AssertArray(b, new decimal[] { 0.0m, 1.0m, 1.0m });
            print(b);

            var c = np.heaviside(new decimal[] { -1, 0, 2 }, 1);
            AssertArray(c, new decimal[] { 0, 1, 1 });
            print(c);

        }

        [TestMethod]
        public void test_maximum_1_DECIMAL()
        {
            var a = np.maximum(new decimal[] { 2, 3, 4 }, new decimal[] { 1, 5, 2 });
            AssertArray(a, new decimal[] { 2, 5, 4 });
            print(a);

            var b = np.maximum(np.eye(2, dtype: np.Decimal), new decimal[] { 0.5m, 2m }); // broadcasting
            AssertArray(b, new decimal[,] { { 1, 2 }, { 0.5m, 2.0m } });
            print(b);

            //var c = np.maximum(new float[] { float.NaN, 0, float.NaN }, new float[] { 0, float.NaN, float.NaN });
            //AssertArray(c, new double[] { double.NaN, double.NaN, double.NaN });
            //print(c);

            //var d = np.maximum(double.PositiveInfinity, 1);
            //Assert.AreEqual(double.PositiveInfinity, d.GetItem(0));
            //print(d);
        }

        [TestMethod]
        public void test_minimum_1_DECIMAL()
        {
            var a = np.minimum(new decimal[] { 2, 3, 4 }, new decimal[] { 1, 5, 2 });
            AssertArray(a, new decimal[] { 1, 3, 2 });
            print(a);

            var b = np.minimum(np.eye(2, dtype: np.Decimal), new decimal[] { 0.5m, 2 }); // broadcasting
            AssertArray(b, new decimal[,] { { 0.5m, 0.0m }, { 0.0m, 1.0m } });
            print(b);

            //var c = np.minimum(new float[] { float.NaN, 0, float.NaN }, new float[] { 0, float.NaN, float.NaN });
            //AssertArray(c, new double[] { float.NaN, float.NaN, float.NaN });
            //print(c);

            //var d = np.minimum(double.PositiveInfinity, 1);
            //Assert.AreEqual((double)1, d.GetItem(0));
            //print(d);
        }

        [TestMethod]
        public void test_fmax_1_DECIMAL()
        {
            var a = np.fmax(new decimal[] { 2, 3, 4 }, new decimal[] { 1, 5, 2 });
            AssertArray(a, new decimal[] { 2, 5, 4 });
            print(a);

            var b = np.fmax(np.eye(2, dtype: np.Decimal), new decimal[] { 0.5m, 2m }); // broadcasting
            AssertArray(b, new decimal[,] { { 1, 2 }, { 0.5m, 2.0m } });
            print(b);

            //var c = np.fmax(new float[] { float.NaN, 0, float.NaN }, new float[] { 0, float.NaN, float.NaN });
            //AssertArray(c, new double[] { 0.0, 0.0, double.NaN });
            //print(c);

            //var d = np.fmax(double.PositiveInfinity, 1);
            //Assert.AreEqual(double.PositiveInfinity, d.GetItem(0));
            //print(d);
        }

        [TestMethod]
        public void test_fmin_1_DECIMAL()
        {
            var a = np.fmin(new decimal[] { 2, 3, 4 }, new decimal[] { 1, 5, 2 });
            AssertArray(a, new decimal[] { 1, 3, 2 });
            print(a);

            var b = np.fmin(np.eye(2, dtype: np.Decimal), new decimal[] { 0.5m, 2 }); // broadcasting
            AssertArray(b, new decimal[,] { { 0.5m, 0.0m }, { 0.0m, 1.0m } });
            print(b);

            //var c = np.fmin(new float[] { float.NaN, 0, float.NaN }, new float[] { 0, float.NaN, float.NaN });
            //AssertArray(c, new double[] { 0.0, 0.0, double.NaN });
            //print(c);

            //var d = np.fmin(double.PositiveInfinity, 1);
            //Assert.AreEqual((double)1, d.GetItem(0));
            //print(d);
        }

        [TestMethod]
        public void test_nan_to_num_1_DECIMAL()
        {
            decimal a1 = (decimal)np.nan_to_num(2.0m);
            Assert.AreEqual(a1, 2.0m);
            print(a1);

            ndarray x = np.array(new decimal[] { 1.0m, 2.0m, 3.0m, -128, 128 });
            ndarray d = np.nan_to_num(x);
            AssertArray(d, new decimal[] { 1.0m, 2.0m, 3.0m, -128, 128 });
            print(d);

        }

        #endregion

        #region from FromNumericTests

        [TestMethod]
        public void test_take_1_DECIMAL()
        {
            var a = np.array(new decimal[] { 4, 3, 5, 7, 6, 8, 9, 12, 14, 16, 18, 20, 22, 24, 26, 28 });
            var indices = np.array(new Int32[] { 0, 1, 4 });
            ndarray b = np.take(a, indices);
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new decimal[] { 4, 3, 6 });
            AssertShape(b, 3);
            AssertStrides(b, sizeof(decimal));


            a = np.array(new decimal[] { 4, 3, 5, 7, 6, 8, 9, 12, 14, 16, 18, 20, 22, 24, 26, 28 });
            indices = np.array(new Int32[,] { { 0, 1 }, { 2, 3 } });
            ndarray c = np.take(a, indices);
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            var ExpectedDataC = new decimal[2, 2]
            {
                { 4, 3 },
                { 5, 7 },
            };
            AssertArray(c, ExpectedDataC);
            AssertShape(c, 2, 2);
            AssertStrides(c, sizeof(decimal) * 2, sizeof(decimal));

            ndarray d = np.take(a.reshape(new shape(4, -1)), indices, axis: 0);
            print("D");
            print(d);
            print(d.shape);
            print(d.strides);

            var ExpectedDataD = new decimal[2, 2, 4]
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
            AssertStrides(d, sizeof(decimal) * 8, sizeof(decimal) * 4, sizeof(decimal) * 1);

            ndarray e = np.take(a.reshape(new shape(4, -1)), indices, axis: 1);
            print("E");
            print(e);
            print(e.shape);
            print(e.strides);

            var ExpectedDataE = new decimal[4, 2, 2]
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
            AssertStrides(e, sizeof(decimal) * 4, sizeof(decimal) * 2, sizeof(decimal) * 1);

        }

        [TestMethod]
        public void test_ravel_1_DECIMAL()
        {
            var a = np.array(new decimal[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var b = np.ravel(a);
            AssertArray(b, new decimal[] { 1, 2, 3, 4, 5, 6 });
            print(b);

            var c = a.reshape(-1);
            AssertArray(c, new decimal[] { 1, 2, 3, 4, 5, 6 });
            print(c);

            var d = np.ravel(a, order: NPY_ORDER.NPY_FORTRANORDER);
            AssertArray(d, new decimal[] { 1, 4, 2, 5, 3, 6 });
            print(d);

            // When order is 'A', it will preserve the array's 'C' or 'F' ordering:
            var e = np.ravel(a.T);
            AssertArray(e, new decimal[] { 1, 4, 2, 5, 3, 6 });
            print(e);

            var f = np.ravel(a.T, order: NPY_ORDER.NPY_ANYORDER);
            AssertArray(f, new decimal[] { 1, 2, 3, 4, 5, 6 });
            print(f);
        }

        [TestMethod]
        public void test_choose_1_DECIMAL()
        {
            ndarray choice1 = np.array(new decimal[] { 0, 1, 2, 3 });
            ndarray choice2 = np.array(new decimal[] { 10, 11, 12, 13 });
            ndarray choice3 = np.array(new decimal[] { 20, 21, 22, 23 });
            ndarray choice4 = np.array(new decimal[] { 30, 31, 32, 33 });

            ndarray[] choices = new ndarray[] { choice1, choice2, choice3, choice4 };

            ndarray a = np.choose(np.array(new Int32[] { 2, 3, 1, 0 }), choices);

            print(a);

            AssertArray(a, new decimal[] { 20, 31, 12, 3 });
        }

        [TestMethod]
        public void test_choose_2_DECIMAL()
        {
            ndarray choice1 = np.array(new decimal[] { 0, 1, 2, 3 });
            ndarray choice2 = np.array(new decimal[] { 10, 11, 12, 13 });
            ndarray choice3 = np.array(new decimal[] { 20, 21, 22, 23 });
            ndarray choice4 = np.array(new decimal[] { 30, 31, 32, 33 });

            ndarray[] choices = new ndarray[] { choice1, choice2, choice3, choice4 };

            ndarray a = np.choose(np.array(new Int32[] { 2, 4, 1, 0 }), choices, mode: NPY_CLIPMODE.NPY_CLIP);
            print(a);
            AssertArray(a, new decimal[] { 20, 31, 12, 3 });

            a = np.choose(np.array(new Int32[] { 2, 4, 1, 0 }), choices, mode: NPY_CLIPMODE.NPY_WRAP);
            print(a);
            AssertArray(a, new decimal[] { 20, 1, 12, 3 });

            try
            {
                a = np.choose(np.array(new Int32[] { 2, 4, 1, 0 }), choices, mode: NPY_CLIPMODE.NPY_RAISE);
                print(a);
                AssertArray(a, new decimal[] { 20, 1, 12, 3 });
            }
            catch (Exception ex)
            {
                if (ex.Message.Contains("invalid entry in choice array"))
                    return;
            }
            Assert.Fail("Should have caught exception from np.choose");


        }

        [TestMethod]
        public void test_select_1_DECIMAL()
        {
            var x = np.arange(10, dtype: np.Decimal);
            var condlist = new ndarray[] { x < 3, x > 5 };
            var choicelist = new ndarray[] { x, np.array(np.power(x, 2), dtype: np.Decimal) };
            var y = np.select(condlist, choicelist);

            AssertArray(y, new decimal[] { 0, 1, 2, 0, 0, 0, 36, 49, 64, 81 });
            print(y);
        }

        [TestMethod]
        public void test_repeat_1_DECIMAL()
        {
            ndarray x = np.array(new decimal[] { 1, 2, 3, 4 }).reshape(new shape(2, 2));
            var y = new Int32[] { 2 };

            ndarray z = np.repeat(x, y);
            print(z);
            print("");
            AssertArray(z, new decimal[] { 1, 1, 2, 2, 3, 3, 4, 4 });

            z = np.repeat(3m, 4);
            print(z);
            print("");
            AssertArray(z, new decimal[] { 3, 3, 3, 3 });

            z = np.repeat(x, 3, axis: 0);
            print(z);
            print("");

            var ExpectedData1 = new decimal[6, 2]
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

            var ExpectedData2 = new decimal[2, 6]
            {
                { 1, 1, 1, 2, 2, 2 },
                { 3, 3, 3, 4, 4, 4 },
            };

            AssertArray(z, ExpectedData2);
            AssertShape(z, 2, 6);



            z = np.repeat(x, new Int32[] { 1, 2 }, axis: 0);
            print(z);

            var ExpectedData3 = new decimal[3, 2]
            {
                { 1, 2 },
                { 3, 4 },
                { 3, 4 },
            };

            AssertArray(z, ExpectedData3);
            AssertShape(z, 3, 2);
        }

        [TestMethod]
        public void test_put_1_DECIMAL()
        {
            ndarray a = np.arange(5, dtype: np.Decimal);
            np.put(a, new int[] { 0, 2 }, new int[] { -44, -55 });
            print(a);
            AssertArray(a, new decimal[] { -44, 1, -55, 3, 4 });

            a = np.arange(5, dtype: np.Decimal);
            np.put(a, 22, -5, mode: NPY_CLIPMODE.NPY_CLIP);
            print(a);
            AssertArray(a, new decimal[] { 0, 1, 2, 3, -5 });

            a = np.arange(5, dtype: np.Decimal);
            np.put(a, 22, -5, mode: NPY_CLIPMODE.NPY_WRAP);
            print(a);
            AssertArray(a, new decimal[] { 0, 1, -5, 3, 4 });

            try
            {
                a = np.arange(5, dtype: np.Decimal);
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
        public void test_putmask_1_DECIMAL()
        {
            var x = np.arange(6, dtype: np.Decimal).reshape((2, 3));
            np.putmask(x, x > 2, np.power(x, 2).astype(np.Int32));
            AssertArray(x, new decimal[,] { { 0, 1, 2, }, { 9, 16, 25 } });
            print(x);


            // If values is smaller than a it is repeated:

            x = np.arange(5, dtype: np.Decimal);
            np.putmask(x, x > 1, new Int32[] { -33, -44 });
            AssertArray(x, new decimal[] { 0, 1, -33, -44, -33 });
            print(x);

            return;
        }

        [TestMethod]
        public void test_swapaxes_1_DECIMAL()
        {
            ndarray x = np.array(new decimal[,] { { 1, 2, 3 } });
            print(x);
            print("********");

            ndarray y = np.swapaxes(x, 0, 1);
            print(y);
            AssertArray(y, new decimal[3, 1] { { 1 }, { 2 }, { 3 } });
            print("********");

            x = np.array(new decimal[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } });
            print(x);

            var ExpectedDataX = new decimal[2, 2, 2]
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

            var ExpectedDataY = new decimal[2, 2, 2]
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
        public void test_ndarray_T_1_DECIMAL()
        {
            var x = np.arange(0, 32, dtype: np.Decimal).reshape(new shape(8, 4));
            print("X");
            print(x);
            print(x.shape);

            var y = x.T;

            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new decimal[4, 8]
            {
                { 0, 4,  8, 12, 16, 20, 24, 28 },
                { 1, 5,  9, 13, 17, 21, 25, 29 },
                { 2, 6, 10, 14, 18, 22, 26, 30 },
                { 3, 7, 11, 15, 19, 23, 27, 31 },
            };

            AssertArray(y, ExpectedDataY);

        }

        [TestMethod]
        public void test_ndarray_transpose_1_DECIMAL()
        {
            var x = np.arange(0, 64, dtype: np.Decimal).reshape(new shape(2, 4, -1, 4));
            print("X");
            print(x);
            print(x.shape);

            var y = np.transpose(x, new long[] { 1, 2, 3, 0 });

            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = new Decimal[4, 2, 4, 2]
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
        public void test_partition_3_DECIMAL()
        {
            var a = np.arange(22, 10, -1, dtype: np.Decimal).reshape((3, 4, 1));
            var b = np.partition(a, 1, axis: 0);
            AssertArray(b, new decimal[,,] { { { 14 }, { 13 }, { 12 }, { 11 } }, { { 18 }, { 17 }, { 16 }, { 15 } }, { { 22 }, { 21 }, { 20 }, { 19 } } });
            print(b);

            var c = np.partition(a, 2, axis: 1);
            AssertArray(c, new decimal[,,] { { { 19 }, { 20 }, { 21 }, { 22 } }, { { 15 }, { 16 }, { 17 }, { 18 } }, { { 11 }, { 12 }, { 13 }, { 14 } } });
            print(c);

            var d = np.partition(a, 0, axis: 2);
            AssertArray(d, new decimal[,,] { { { 22 }, { 21 }, { 20 }, { 19 } }, { { 18 }, { 17 }, { 16 }, { 15 } }, { { 14 }, { 13 }, { 12 }, { 11 } } });
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
        public void test_argpartition_3_DECIMAL()
        {
            var a = np.arange(22, 10, -1, np.Decimal).reshape((3, 4, 1));
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
        public void test_sort_2_DECIMAL()
        {
            var InputData = new decimal[]
                {32.2m, 31.2m, 30.2m, 29.2m, 28.2m, 27.2m, 26.2m, 25.2m, 24.2m, 23.2m, 22.2m, 21.2m, 20.2m, 19.2m, 18.2m, 17.2m,
                 16.2m, 15.2m, 14.2m, 13.2m, 12.2m, 11.2m, 10.2m, 9.2m,  8.2m,  7.2m,  6.2m,  5.2m,  4.2m,  3.2m,  2.2m,  1.2m};

            var a = np.array(InputData).reshape(new shape(8, 4));
            ndarray b = np.sort(a);                 // sort along the last axis
            print(b);

            var ExpectedDataB = new decimal[8, 4]
            {
             {29.2m, 30.2m, 31.2m, 32.2m},
             {25.2m, 26.2m, 27.2m, 28.2m},
             {21.2m, 22.2m, 23.2m, 24.2m},
             {17.2m, 18.2m, 19.2m, 20.2m},
             {13.2m, 14.2m, 15.2m, 16.2m},
             {9.2m, 10.2m, 11.2m, 12.2m},
             {5.2m,  6.2m,  7.2m,  8.2m},
             {1.2m,  2.2m,  3.2m,  4.2m},
            };

            AssertArray(b, ExpectedDataB);

            ndarray c = np.sort(a, axis: null);     // sort the flattened array
            print(c);
            print("********");

            var ExpectedDataC = new decimal[]
            {1.2m,  2.2m,  3.2m,  4.2m,  5.2m,  6.2m,  7.2m,  8.2m, 9.2m, 10.2m, 11.2m, 12.2m, 13.2m, 14.2m, 15.2m, 16.2m,
            17.2m, 18.2m, 19.2m, 20.2m, 21.2m, 22.2m, 23.2m, 24.2m, 25.2m, 26.2m, 27.2m, 28.2m, 29.2m, 30.2m, 31.2m, 32.2m};

            AssertArray(c, ExpectedDataC);

            ndarray d = np.sort(a, axis: 0);        // sort along the first axis
            print(d);

            var ExpectedDataD = new decimal[8, 4]
            {
                {4.2m,  3.2m,  2.2m,  1.2m},
                {8.2m,  7.2m,  6.2m,  5.2m},
                {12.2m, 11.2m, 10.2m, 9.2m},
                {16.2m, 15.2m, 14.2m, 13.2m},
                {20.2m, 19.2m, 18.2m, 17.2m},
                {24.2m, 23.2m, 22.2m, 21.2m},
                {28.2m, 27.2m, 26.2m, 25.2m},
                {32.2m, 31.2m, 30.2m, 29.2m},
            };

            AssertArray(d, ExpectedDataD);
            print("********");

        }

        [TestMethod]
        public void test_msort_1_DECIMAL()
        {
            var a = np.array(new decimal[,] { { 1, 4 }, { 3, 1 } });
            ndarray b = np.msort(a);
            print(b);
            AssertArray(b, new decimal[,] { { 1, 1 }, { 3, 4 } });

            a = np.arange(32.2, 0.2, -1.0, dtype: np.Decimal);
            b = np.msort(a);

            var ExpectedDataB = new decimal[]
            {1.2m,  2.2m,  3.2m,  4.2m,  5.2m,  6.2m,  7.2m,  8.2m, 9.2m, 10.2m, 11.2m, 12.2m, 13.2m, 14.2m, 15.2m, 16.2m,
            17.2m, 18.2m, 19.2m, 20.2m, 21.2m, 22.2m, 23.2m, 24.2m, 25.2m, 26.2m, 27.2m, 28.2m, 29.2m, 30.2m, 31.2m, 32.2m};
            AssertArray(b, ExpectedDataB);
            print(b);

        }

        [TestMethod]
        public void test_ndarray_argsort_2_DECIMAL()
        {
            var ar = np.array(new decimal[] { 1, 2, 3, 1, 3, 4, 5, 4, 4, 1, 9, 6, 9, 11, 23, 9, 5, 0, 11, 12 }).reshape(new shape(5, 4));

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
        public void test_argmin_1_DECIMAL()
        {
            ndarray a = np.array(new decimal[] { 32, 33, 45, 98, 11, 02 }).reshape(new shape(2, 3));
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
        public void test_argmax_1_DECIMAL()
        {
            ndarray a = np.array(new decimal[] { 32, 33, 45, 98, 11, 02 }).reshape(new shape(2, 3));
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
        public void test_searchsorted_1_DECIMAL()
        {
            ndarray arr = np.array(new decimal[] { 1, 2, 3, 4, 5 });
            ndarray a = np.searchsorted(arr, 3);
            print(a);
            Assert.AreEqual(a.GetItem(0), (npy_intp)2);


            ndarray b = np.searchsorted(arr, 3, side: NPY_SEARCHSIDE.NPY_SEARCHRIGHT);
            print(b);
            Assert.AreEqual(b.GetItem(0), (npy_intp)3);


            ndarray c = np.searchsorted(arr, new Int32[] { -10, 10, 2, 3 });
            print(c);
            AssertArray(c, new npy_intp[] { 0, 5, 1, 2 });


            ndarray d = np.searchsorted(np.array(new decimal[] { 15, 14, 13, 12, 11 }), 13);
            print(d);
            Assert.AreEqual(d.GetItem(0), (npy_intp)0);
        }

        [TestMethod]
        public void test_resize_1_DECIMAL()
        {
            ndarray a = np.array(new decimal[,] { { 0, 1 }, { 2, 3 } });
            print(a);

            ndarray b = np.resize(a, new shape(2, 3));
            print(b);

            var ExpectedDataB = new decimal[,]
            {
                { 0,1,2 },
                { 3,0,1 },
            };
            AssertArray(b, ExpectedDataB);


            ndarray c = np.resize(a, new shape(1, 4));
            print(c);
            var ExpectedDataC = new decimal[,]
            {
                { 0,1,2,3 },
            };
            AssertArray(c, ExpectedDataC);

            ndarray d = np.resize(a, new shape(2, 4));
            print(d);
            var ExpectedDataD = new decimal[,]
            {
                { 0,1,2,3 },
                { 0,1,2,3 },
            };
            AssertArray(d, ExpectedDataD);

        }

        [TestMethod]
        public void test_squeeze_1_DECIMAL()
        {
            ndarray x = np.array(new decimal[,,] { { { 0 }, { 1 }, { 2 } } });
            print(x);
            AssertArray(x, new decimal[1, 3, 1] { { { 0 }, { 1 }, { 2 } } });

            ndarray a = np.squeeze(x);
            print(a);
            AssertArray(a, new decimal[] { 0, 1, 2 });

            ndarray b = np.squeeze(x, axis: 0);
            print(b);
            AssertArray(b, new decimal[3, 1] { { 0 }, { 1 }, { 2 } });

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
            AssertArray(d, new decimal[,] { { 0, 1, 2 } });
        }

        [TestMethod]
        public void test_diagonal_1_DECIMAL()
        {
            ndarray a = np.arange(4, dtype: np.Decimal).reshape(new shape(2, 2));
            print(a);
            print("*****");

            ndarray b = a.diagonal();
            print(b);
            AssertArray(b, new decimal[] { 0, 3 });
            print("*****");

            ndarray c = a.diagonal(1);
            print(c);
            AssertArray(c, new decimal[] { 1 });
            print("*****");

            a = np.arange(8, dtype: np.Decimal).reshape(new shape(2, 2, 2));
            print(a);
            print("*****");
            b = a.diagonal(0, // Main diagonals of two arrays created by skipping
                           0, // across the outer(left)-most axis last and
                           1); //the "middle" (row) axis first.

            print(b);
            AssertArray(b, new decimal[,] { { 0, 6 }, { 1, 7 } });
            print("*****");

            ndarray d = a.A(":", ":", 0);
            print(d);
            AssertArray(d, new decimal[,] { { 0, 2 }, { 4, 6 } });
            print("*****");

            ndarray e = a.A(":", ":", 1);
            print(e);
            AssertArray(e, new decimal[,] { { 1, 3 }, { 5, 7 } });
            print("*****");
        }

        [TestMethod]
        public void test_trace_1_DECIMAL()
        {
            ndarray a = np.trace(np.eye(3, dtype: np.Decimal));
            print(a);
            Assert.AreEqual(a.GetItem(0), 3.0m);
            print("*****");

            a = np.arange(8, dtype: np.Decimal).reshape(new shape(2, 2, 2));
            ndarray b = np.trace(a);
            print(b);
            AssertArray(b, new decimal[] { 6, 8 });
            print("*****");

            a = np.arange(24, dtype: np.Decimal).reshape(new shape(2, 2, 2, 3));
            var c = np.trace(a);
            print(c);
            AssertArray(c, new decimal[,] { { 18, 20, 22 }, { 24, 26, 28 } });

        }

        [TestMethod]
        public void test_nonzero_DECIMAL()
        {
            ndarray x = np.array(new decimal[,] { { 1, 0, 0 }, { 0, 2, 0 }, { 1, 1, 0 } });
            print(x);
            print("*****");

            ndarray[] y = np.nonzero(x);
            print(y);
            AssertArray(y[0], new npy_intp[] { 0, 1, 2, 2 });
            AssertArray(y[1], new npy_intp[] { 0, 1, 0, 1 });
            print("*****");

            ndarray z = x.A(np.nonzero(x));
            print(z);
            AssertArray(z, new decimal[] { 1, 2, 1, 1 });
            print("*****");

            //ndarray q = np.transpose(np.nonzero(x));
            //print(q);

        }

        [TestMethod]
        public void test_compress_1_DECIMAL()
        {
            ndarray a = np.array(new decimal[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            print(a);
            print("*****");

            ndarray b = np.compress(new int[] { 0, 1 }, a, axis: 0);
            print(b);
            AssertArray(b, new decimal[,] { { 3, 4 } });
            print("*****");

            ndarray c = np.compress(new bool[] { false, true, true }, a, axis: 0);
            print(c);
            AssertArray(c, new decimal[,] { { 3, 4 }, { 5, 6 } });
            print("*****");

            ndarray d = np.compress(new bool[] { false, true }, a, axis: 1);
            print(d);
            AssertArray(d, new decimal[,] { { 2 }, { 4 }, { 6 } });
            print("*****");

            ndarray e = np.compress(new bool[] { false, true }, a);
            AssertArray(e, new decimal[] { 2 });
            print(e);

        }

        [TestMethod]
        public void test_any_1_DECIMAL()
        {
            decimal[] TestData = new decimal[] { 2.5m, -1.7m, -1.5m, -0.2m, 0.2m, 1.5m, 1.7m, 2.0m };
            var x = np.array(TestData);
            var y = np.any(x);

            print(x);
            print(y);
            Assert.AreEqual(true, y.GetItem(0));

            TestData = new decimal[] { 0.0m, 0.0m, 0.0m, 0.0m };
            x = np.array(TestData);
            y = np.any(x);

            print(x);
            print(y);
            Assert.AreEqual(false, y.GetItem(0));

        }

        [TestMethod]
        public void test_all_1_DECIMAL()
        {
            decimal[] TestData = new decimal[] { 2.5m, -1.7m, -1.5m, -0.2m, 0.2m, 1.5m, 1.7m, 2.0m };
            var x = np.array(TestData);
            var y = np.all(x);

            print(x);
            print(y);
            Assert.AreEqual(true, y.GetItem(0));

            TestData = new decimal[] { 1.0m, 1.0m, 0.0m, 1.0m };
            x = np.array(TestData);
            y = np.all(x);

            print(x);
            print(y);
            Assert.AreEqual(false, y.GetItem(0));

        }

        [TestMethod]
        public void test_ndarray_mean_1_DECIMAL()
        {
            var x = np.arange(0, 12, dtype: np.Decimal).reshape(new shape(3, -1));

            print("X");
            print(x);

            var y = (ndarray)np.mean(x);
            Assert.AreEqual(5.5m, y.GetItem(0));

            print("Y");
            print(y);

            y = (ndarray)np.mean(x, axis: 0);
            AssertArray(y, new decimal[] { 4, 5, 6, 7 });

            print("Y");
            print(y);

            y = (ndarray)np.mean(x, axis: 1);
            AssertArray(y, new decimal[] { 1.5m, 5.5m, 9.5m });

            print("Y");
            print(y);

        }

        [TestMethod]
        public void test_place_1_DECIMAL()
        {
            var arr = np.arange(6, dtype: np.Decimal).reshape((2, 3));
            np.place(arr, arr > 2, new Int32[] { 44, 55 });
            AssertArray(arr, new decimal[,] { { 0, 1, 2 }, { 44, 55, 44 } });
            print(arr);

            arr = np.arange(16, dtype: np.Decimal).reshape((2, 4, 2));
            np.place(arr, arr > 12, new Int32[] { 33 });
            AssertArray(arr, new decimal[,,] { { { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 } }, { { 8, 9 }, { 10, 11 }, { 12, 33 }, { 33, 33 } } });
            print(arr);

            arr = np.arange(6, dtype: np.Decimal).reshape((2, 3));
            np.place(arr, arr > 2, new Int32[] { 44, 55, 66, 77, 88, 99, 11, 22, 33 });
            AssertArray(arr, new decimal[,] { { 0, 1, 2 }, { 44, 55, 66 } });
            print(arr);

        }

        [TestMethod]
        public void test_extract_1_DECIMAL()
        {
            var arr = np.arange(12, dtype: np.Decimal).reshape((3, 4));
            var condition = np.mod(arr, 3) == 0;
            print(condition);

            var b = np.extract(condition, arr);
            AssertArray(b, new decimal[] { 0, 3, 6, 9 });
            print(b);
        }

        [TestMethod]
        public void test_viewfromaxis_1_DECIMAL()
        {
            decimal[] TestData = new decimal[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            var a = np.zeros_like(TestData).reshape(new shape(3, 2, -1));
            //print(a);


            var b = np.ViewFromAxis(a, 0);
            b[":"] = 99;
            //print(a);
            AssertArray(a, new decimal[,,] { { { 99, 0 }, { 0, 0 } }, { { 99, 0 }, { 0, 0 } }, { { 99, 0 }, { 0, 0 } } });
            //print(a);
            AssertArray(np.sum(a, axis: 0), new decimal[,] { { 297, 0 }, { 0, 0 } });

            b = np.ViewFromAxis(a, 1);
            b[":"] = 11;
            AssertArray(a, new decimal[,,] { { { 11, 0 }, { 11, 0 } }, { { 99, 0 }, { 0, 0 } }, { { 99, 0 }, { 0, 0 } } });
            //print(a);
            AssertArray(np.sum(a, axis: 1), new decimal[,] { { 22, 0 }, { 99, 0 }, { 99, 0 } });

            b = np.ViewFromAxis(a, 2);
            b[":"] = 22;
            AssertArray(a, new decimal[,,] { { { 22, 22 }, { 11, 0 } }, { { 99, 0 }, { 0, 0 } }, { { 99, 0 }, { 0, 0 } } });
            //print(a);
            AssertArray(np.sum(a, axis: 2), new decimal[,] { { 44, 11 }, { 99, 0 }, { 99, 0 } });

            Assert.AreEqual((decimal)253, np.sum(a).GetItem(0));


        }

        [TestMethod]
        public void test_unwrap_1_DECIMAL()
        {
            double retstep = 0;

            var phase = np.linspace(0, Math.PI, ref retstep, num: 5, dtype: np.Decimal);
            phase["3:"] = phase.A("3:") + Math.PI;
            print(phase);

            var x = np.unwrap(phase);
            AssertArray(x, new decimal[] { 0, 00.785398163397448m, 01.5707963267949m, -00.78539816339746m, -00.00000000000001m });
            print(x);
        }

        #endregion

        #region from NumericTests

        [TestMethod]
        public void test_zeros_1_DECIMAL()
        {
            var x = np.zeros(new shape(10), dtype: np.Decimal);
            print(x);
            print("Update sixth value to 11");
            x[6] = 11;
            print(x);
            print(x.shape);
            print(x.strides);

            AssertArray(x, new decimal[] { 0, 0, 0, 0, 0, 0, 11, 0, 0, 0 });
            AssertShape(x, 10);
            AssertStrides(x, sizeof(decimal));
        }

        [TestMethod]
        public void test_zeros_like_2_DECIMAL()
        {
            var a = new decimal[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            var b = np.zeros_like(a);
            b[1, 2] = 99;

            AssertArray(b, new decimal[,] { { 0, 0, 0 }, { 0, 0, 99 } });

            return;
        }

        [TestMethod]
        public void test_ones_1_DECIMAL()
        {
            var x = np.ones(new shape(10), dtype: np.Decimal);
            print(x);
            print("Update sixth value to 11");
            x[6] = 11;
            print(x);
            print(x.shape);
            print(x.strides);

            AssertArray(x, new decimal[] { 1, 1, 1, 1, 1, 1, 11, 1, 1, 1 });
            AssertShape(x, 10);
            AssertStrides(x, sizeof(decimal));
        }

        [TestMethod]
        public void test_ones_like_3_DECIMAL()
        {
            var a = new decimal[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } };
            var b = np.ones_like(a);
            b[0, 0, 2] = 99;
            b[0, 1, 1] = 88;

            AssertArray(b, new decimal[,,] { { { 1, 1, 99 }, { 1, 88, 1 } } });

            return;
        }

        [TestMethod]
        public void test_empty_DECIMAL()
        {
            var a = np.empty((2, 3));
            AssertShape(a, 2, 3);
            Assert.AreEqual(a.Dtype.TypeNum, NPY_TYPES.NPY_DOUBLE);

            var b = np.empty((2, 4), np.Decimal);
            AssertShape(b, 2, 4);
            Assert.AreEqual(b.Dtype.TypeNum, NPY_TYPES.NPY_DECIMAL);
        }

        [TestMethod]
        public void test_empty_like_3_DECIMAL()
        {
            var a = new decimal[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } };
            var b = np.empty_like(a);
            b[0, 0, 2] = 99;
            b[0, 1, 1] = 88;

            AssertArray(b, new decimal[,,] { { { 0, 0, 99 }, { 0, 88, 0 } } });

            return;
        }

        [TestMethod]
        public void test_full_2_DECIMAL()
        {
            var x = np.full((100), 99, dtype: np.Decimal).reshape(new shape(10, 10));
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
        public void test_count_nonzero_1_DECIMAL()
        {
            var a = np.count_nonzero(np.eye(4, dtype: np.Decimal));
            Assert.AreEqual(4, (int)a);
            print(a);

            var b = np.count_nonzero(new decimal[,] { { 0, 1, 7, 0, 0 }, { 3, 0, 0, 2, 19 } });
            Assert.AreEqual(5, (int)b);
            print(b);

            var c = np.count_nonzero(new decimal[,] { { 0, 1, 7, 0, 0 }, { 3, 0, 0, 2, 19 } }, axis: 0);
            AssertArray(c, new int[] { 1, 1, 1, 1, 1 });
            print(c);

            var d = np.count_nonzero(new decimal[,] { { 0, 1, 7, 0, 0 }, { 3, 0, 0, 2, 19 } }, axis: 1);
            AssertArray(d, new int[] { 2, 3 });
            print(d);

            return;
        }

        [TestMethod]
        public void test_asarray_1_DECIMAL()
        {
            var a = new decimal[] { 1, 2 };
            var b = np.asarray(a);

            AssertArray(b, new decimal[] { 1, 2 });
            print(b);

            var c = np.array(new decimal[] { 1.0m, 2.0m }, dtype: np.Decimal);
            var d = np.asarray(c, dtype: np.Decimal);

            c[0] = 3.0f;
            AssertArray(d, new decimal[] { 3.0m, 2.0m });
            print(d);

            var e = np.asarray(a, dtype: np.Decimal);
            AssertArray(e, new decimal[] { 1.0m, 2.0m });

            print(e);

            return;
        }

        [TestMethod]
        public void test_ascontiguousarray_1_DECIMAL()
        {
            var x = np.arange(6, dtype: np.Decimal).reshape((2, 3));
            var y = np.ascontiguousarray(x, dtype: np.Decimal);

            AssertArray(y, new decimal[,] { { 0m, 1m, 2m }, { 3m, 4m, 5m } });
            print(y);

            Assert.AreEqual(x.flags.c_contiguous, true);
            Assert.AreEqual(y.flags.c_contiguous, true);

            return;
        }

        [TestMethod]
        public void test_asfortranarray_1_DECIMAL()
        {
            var x = np.arange(6, dtype: np.Decimal).reshape((2, 3));
            var y = np.asfortranarray(x, dtype: np.Decimal);

            AssertArray(y, new decimal[,] { { 0m, 1m, 2m }, { 3m, 4m, 5m } });
            print(y);

            Assert.AreEqual(x.flags.f_contiguous, false);
            Assert.AreEqual(y.flags.f_contiguous, true);

            return;
        }

        [TestMethod]
        public void test_isfortran_1_DECIMAL()
        {

            var a = np.array(new decimal[,] { { 1, 2, 3 }, { 4, 5, 6 } }, order: NPY_ORDER.NPY_CORDER);
            var a1 = np.isfortran(a);
            Assert.AreEqual(false, a1);
            print(a1);

            var b = np.array(new decimal[,] { { 1, 2, 3 }, { 4, 5, 6 } }, order: NPY_ORDER.NPY_FORTRANORDER);
            var b1 = np.isfortran(b);
            Assert.AreEqual(true, b1);
            print(b1);

            var c = np.array(new decimal[,] { { 1, 2, 3 }, { 4, 5, 6 } }, order: NPY_ORDER.NPY_CORDER);
            var c1 = np.isfortran(c);
            Assert.AreEqual(false, c1);
            print(c1);

            var d = a.T;
            var d1 = np.isfortran(d);
            Assert.AreEqual(true, d1);
            print(d1);

            // C-ordered arrays evaluate as False even if they are also FORTRAN-ordered.

            var e1 = np.isfortran(np.array(new decimal[] { 1, 2 }, order: NPY_ORDER.NPY_FORTRANORDER));
            Assert.AreEqual(false, e1);
            print(e1);

            return;

        }

        [TestMethod]
        public void test_argwhere_1_DECIMAL()
        {
            var x = np.arange(6, dtype: np.Decimal).reshape((2, 3));
            var y = np.argwhere(x > 1);

            var ExpectedY = new npy_intp[,] { { 0, 2 }, { 1, 0 }, { 1, 1 }, { 1, 2 } };
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
        public void test_flatnonzero_1_DECIMAL()
        {
            var x = np.arange(-2, 3, dtype: np.Decimal);

            var y = np.flatnonzero(x);
            AssertArray(y, new npy_intp[] { 0, 1, 3, 4 });
            print(y);

            // Use the indices of the non-zero elements as an index array to extract these elements:

            var z = x.ravel()[np.flatnonzero(x)] as ndarray;
            AssertArray(z, new decimal[] { -2, -1, 1, 2 });
            print(z);

            return;
        }

        [TestMethod]
        public void test_outer_1_DECIMAL()
        {
            var a = np.arange(2, 10, dtype: np.Decimal).reshape((2, 4));
            var b = np.arange(12, 20, dtype: np.Decimal).reshape((2, 4));
            var c = np.outer(a, b);

            var ExpectedDataC = new decimal[,]
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

            //a = np.arange(2000, 10000, dtype: np.Decimal).reshape((-1, 4000));
            //b = np.arange(12000, 20000, dtype: np.Decimal).reshape((-1, 4000));

            //System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            //sw.Start();
            //c = np.outer(a, b);
            //sw.Stop();
            //Console.WriteLine(sw.ElapsedMilliseconds);


            return;
        }

        [TestMethod]
        public void test_inner_1_DECIMAL()
        {
            var a = np.arange(1, 5, dtype: np.Decimal).reshape((2, 2));
            var b = np.arange(11, 15, dtype: np.Decimal).reshape((2, 2));
            var c = np.inner(a, b);
            AssertArray(c, new decimal[,] { { 35, 41 }, { 81, 95 } });
            print(c);


            a = np.arange(2, 10, dtype: np.Decimal).reshape((2, 4));
            b = np.arange(12, 20, dtype: np.Decimal).reshape((2, 4));
            c = np.inner(a, b);
            print(c);
            AssertArray(c, new decimal[,] { { 194, 250 }, { 410, 530 } });
            print(c.shape);

            return;
        }

        [TestMethod]
        public void test_tensordot_2_DECIMAL()
        {
            var a = np.arange(12.0, dtype: np.Decimal).reshape((3, 4));
            var b = np.arange(24.0, dtype: np.Decimal).reshape((4, 3, 2));
            var c = np.tensordot(a, b, axis: 1);
            AssertShape(c, 3, 3, 2);
            print(c.shape);
            AssertArray(c, new decimal[,,] { { { 84, 90 }, { 96, 102 }, { 108, 114 } }, { { 228, 250 }, { 272, 294 }, { 316, 338 } }, { { 372, 410 }, { 448, 486 }, { 524, 562 } } });


            c = np.tensordot(a, b, axis: 0);
            AssertShape(c, 3, 4, 4, 3, 2);
            print(c.shape);

            print(c);
        }

        [TestMethod]
        public void test_dot_1_DECIMAL()
        {
            var a = new decimal[,] { { 1, 0 }, { 0, 1 } };
            var b = new decimal[,] { { 4, 1 }, { 2, 2 } };
            var c = np.dot(a, b);
            AssertArray(c, new decimal[,] { { 4, 1 }, { 2, 2 } });
            print(c);

            var d = np.dot(3m, 4m);
            Assert.AreEqual(12m, d.GetItem(0));
            print(d);

            var e = np.arange(3 * 4 * 5 * 6, dtype: np.Decimal).reshape((3, 4, 5, 6));
            var f = np.arange(3 * 4 * 5 * 6, dtype: np.Decimal).A("::-1").reshape((5, 4, 6, 3));
            var g = np.dot(e, f);
            AssertShape(g.shape, 3, 4, 5, 5, 4, 3);
            Assert.AreEqual(695768400m, g.Sum().GetItem(0));

            // TODO: NOTE: this crazy indexing is not currently working
            //g = g.A(2, 3, 2, 1, 2, 2);
            //Assert.AreEqual(499128, g.GetItem(0));
            //print(g);

        }

        [TestMethod]
        public void test_roll_forward_DECIMAL()
        {
            var a = np.arange(10, dtype: np.Decimal);

            print("A");
            print(a);
            print(a.shape);
            print(a.strides);

            var b = np.roll(a, 2);
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);
            AssertArray(b, new decimal[] { 8, 9, 0, 1, 2, 3, 4, 5, 6, 7 });
            AssertShape(b, 10);

            var c = np.roll(b, 2);
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);
            AssertArray(c, new decimal[] { 6, 7, 8, 9, 0, 1, 2, 3, 4, 5 });
            AssertShape(c, 10);

        }

        [TestMethod]
        public void test_roll_backward_DECIMAL()
        {
            var a = np.arange(10, dtype: np.Decimal);

            print("A");
            print(a);
            print(a.shape);
            print(a.strides);

            var b = np.roll(a, -2);
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);
            AssertArray(b, new decimal[] { 2, 3, 4, 5, 6, 7, 8, 9, 0, 1 });
            AssertShape(b, 10);

            var c = np.roll(b, -6);
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);
            AssertArray(c, new decimal[] { 8, 9, 0, 1, 2, 3, 4, 5, 6, 7 });
            AssertShape(c, 10);
        }

        [TestMethod]
        public void test_ndarray_rollaxis_DECIMAL()
        {
            var a = np.ones((3, 4, 5, 6), dtype: np.Decimal);
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
        public void test_ndarray_moveaxis_DECIMAL()
        {
            var x = np.zeros((3, 4, 5), np.Decimal);
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
        public void test_indices_1_DECIMAL()
        {
            var grid = np.indices((2, 3), dtype: np.Decimal);
            AssertShape(grid, 2, 2, 3);
            print(grid.shape);
            AssertArray(grid[0] as ndarray, new decimal[,] { { 0, 0, 0 }, { 1, 1, 1 } });
            print(grid[0]);
            AssertArray(grid[1] as ndarray, new decimal[,] { { 0, 1, 2 }, { 0, 1, 2 } });
            print(grid[1]);

            var x = np.arange(20, dtype: np.Decimal).reshape((5, 4));

            bool CaughtException = false;
            try
            {
                var y = x[grid[0], grid[1]];
                AssertArray(y as ndarray, new decimal[,] { { 0, 1, 2 }, { 4, 5, 6 } });
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
        public void test_isscalar_1_DECIMAL()
        {

            bool a = np.isscalar(3.1m);
            Assert.AreEqual(true, a);
            print(a);

            bool b = np.isscalar(np.array(3.1m));
            Assert.AreEqual(false, b);
            print(b);

            bool c = np.isscalar(new decimal[] { 3.1m });
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
        public void test_identity_1_DECIMAL()
        {
            ndarray a = np.identity(2, dtype: np.Decimal);

            print(a);
            print(a.shape);
            print(a.strides);

            var ExpectedDataA = new decimal[2, 2]
            {
                { 1,0 },
                { 0,1 },
            };
            AssertArray(a, ExpectedDataA);
            AssertShape(a, 2, 2);
            AssertStrides(a, sizeof(decimal) * 2, sizeof(decimal) * 1);

            ndarray b = np.identity(5, dtype: np.Decimal);

            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new decimal[5, 5]
            {
                { 1, 0, 0, 0, 0 },
                { 0, 1, 0, 0, 0 },
                { 0, 0, 1, 0, 0 },
                { 0, 0, 0, 1, 0 },
                { 0, 0, 0, 0, 1 },
            };
            AssertArray(b, ExpectedDataB);
            AssertShape(b, 5, 5);
            AssertStrides(b, sizeof(decimal) * 5, sizeof(decimal) * 1);
        }

        [TestMethod]
        public void test_allclose_1_DECIMAL()
        {
            bool a = np.allclose(new decimal[] { 1e10m, 1e-7m }, new decimal[] { 1.00001e10m, 1e-8m });
            Assert.AreEqual(false, a);
            print(a);

            bool b = np.allclose(new decimal[] { 1e10m, 1e-8m }, new decimal[] { 1.00001e10m, 1e-9m });
            Assert.AreEqual(true, b);
            print(b);

            bool c = np.allclose(new decimal[] { 1e10m, 1e-8m }, new decimal[] { 1.0001e10m, 1e-9m });
            Assert.AreEqual(false, c);
            print(c);

            //bool d = np.allclose(new decimal[] { 1.0m, np.NaN }, new decimal[] { 1.0m, np.NaN });
            //Assert.AreEqual(false, d);
            //print(d);

            //bool e = np.allclose(new decimal[] { 1.0m, np.NaN }, new decimal[] { 1.0m, np.NaN }, equal_nan: true);
            //Assert.AreEqual(true, e);
            //print(e);

            return;
        }

        [TestMethod]
        public void test_isclose_1_DECIMAL()
        {
            var a = np.isclose(new decimal[] { 1e10m, 1e-7m }, new decimal[] { 1.00001e10m, 1e-8m });
            AssertArray(a, new bool[] { true, false });
            print(a);

            var b = np.isclose(new decimal[] { 1e10m, 1e-8m }, new decimal[] { 1.00001e10m, 1e-9m });
            AssertArray(b, new bool[] { true, true });
            print(b);

            var c = np.isclose(new decimal[] { 1e10m, 1e-8m }, new decimal[] { 1.0001e10m, 1e-9m });
            AssertArray(c, new bool[] { false, true });
            print(c);

            //var d = np.isclose(new double[] { 1.0, np.NaN }, new double[] { 1.0, np.NaN });
            //AssertArray(d, new bool[] { true, false });
            //print(d);

            //var e = np.isclose(new double[] { 1.0, np.NaN }, new double[] { 1.0, np.NaN }, equal_nan: true);
            //AssertArray(e, new bool[] { true, true });
            //print(e);

            var f = np.isclose(new decimal[] { 1e-8m, 1e-7m }, new decimal[] { 0.0m, 0.0m });
            AssertArray(f, new bool[] { true, false });
            print(f);

            var g = np.isclose(new decimal[] { 1e-100m, 1e-7m }, new decimal[] { 0.0m, 0.0m }, atol: 0.0);
            //AssertArray(g, new bool[] { false, false });
            print(g);

            var h = np.isclose(new decimal[] { 1e-10m, 1e-10m }, new decimal[] { 1e-20m, 0.0m });
            AssertArray(h, new bool[] { true, true });
            print(h);

            var i = np.isclose(new decimal[] { 1e-10m, 1e-10m }, new decimal[] { 1e-20m, 0.999999e-10m }, atol: 0.0);
            AssertArray(i, new bool[] { false, true });
            print(i);
        }

        [TestMethod]
        public void test_array_equal_1_DECIMAL()
        {
            var a = np.array_equal(new decimal[] { 1, 2 }, new decimal[] { 1, 2 });
            Assert.AreEqual(true, a);
            print(a);

            var b = np.array_equal(np.array(new decimal[] { 1, 2 }), np.array(new decimal[] { 1, 2 }));
            Assert.AreEqual(true, b);
            print(b);

            var c = np.array_equal(new decimal[] { 1, 2 }, new decimal[] { 1, 2, 3 });
            Assert.AreEqual(false, c);
            print(c);

            var d = np.array_equal(new decimal[] { 1, 2 }, new decimal[] { 1, 4 });
            Assert.AreEqual(false, d);
            print(d);
        }

        [TestMethod]
        public void test_array_equiv_1_DECIMAL()
        {
            var a = np.array_equiv(new decimal[] { 1, 2 }, new decimal[] { 1, 2 });
            Assert.AreEqual(true, a);
            print(a);

            var b = np.array_equiv(new decimal[] { 1, 2 }, new decimal[] { 1, 3 });
            Assert.AreEqual(false, b);
            print(b);

            var c = np.array_equiv(new decimal[] { 1, 2 }, new decimal[,] { { 1, 2 }, { 1, 2 } });
            Assert.AreEqual(true, c);
            print(c);

            var d = np.array_equiv(new decimal[] { 1, 2 }, new decimal[,] { { 1, 2, 1, 2 }, { 1, 2, 1, 2 } });
            Assert.AreEqual(false, d);
            print(d);

            var e = np.array_equiv(new decimal[] { 1, 2 }, new decimal[,] { { 1, 2 }, { 1, 3 } });
            Assert.AreEqual(false, e);
            print(e);
        }

        #endregion

        #region from NANFunctionsTests

        [TestMethod]
        public void test_nanprod_1_DECIMAL()
        {

            var x = np.nanprod(1m);
            Assert.AreEqual(1m, x.GetItem(0));
            print(x);

            var y = np.nanprod(new decimal[] { 1 });
            Assert.AreEqual(1m, y.GetItem(0));
            print(y);



            var a = np.array(new decimal[,] { { 1, 2 }, { 3, 4 } });
            var b = np.nanprod(a);
            Assert.AreEqual(24.0m, b.GetItem(0));
            print(b);

            var c = np.nanprod(a, axis: 0);
            AssertArray(c, new decimal[] { 3, 8 });
            print(c);

            var d = np.nanprod(a, axis: 1);
            AssertArray(d, new decimal[] { 2, 12 });
            print(d);

            return;
        }

        #endregion

        #region from StatisticsTests

        [TestMethod]
        public void test_amin_2_DECIMAL()
        {
            ndarray a = np.arange(30.25m, 46.25m).reshape(new shape(4, 4));
            print(a);
            print("*****");

            ndarray b = np.amin(a);          // Minimum of the flattened array
            print(b);
            Assert.AreEqual(30.25m, b.GetItem(0));
            print("*****");

            ndarray c = np.amin(a, axis: 0);  // Minimum along the first axis
            print(c);
            AssertArray(c, new decimal[] { 30.25m, 31.25m, 32.25m, 33.25m });
            print("*****");

            ndarray d = np.amin(a, axis: 1);   // Minimum along the second axis
            print(d);
            AssertArray(d, new decimal[] { 30.25m, 34.25m, 38.25m, 42.25m });
            print("*****");

            // decimals don't support NAN
            //ndarray e = np.arange(5, dtype: np.Decimal);
            //e[2] = np.NaN;
            //ndarray f = np.amin(e);
            //print(f);
            //Assert.AreEqual(np.NaN, f.GetItem(0));
            //print("*****");

        }

        [TestMethod]
        public void test_amax_2_DECIMAL()
        {
            ndarray a = np.arange(30.25m, 46.25m).reshape(new shape(4, 4));
            print(a);
            print("*****");

            ndarray b = np.amax(a);          // Maximum of the flattened array
            print(b);
            Assert.AreEqual(45.25m, b.GetItem(0));
            print("*****");

            ndarray c = np.amax(a, axis: 0);  // Maxima along the first axis
            print(c);
            AssertArray(c, new decimal[] { 42.25m, 43.25m, 44.25m, 45.25m });
            print("*****");

            ndarray d = np.amax(a, axis: 1);   // Maxima along the second axis
            print(d);
            AssertArray(d, new decimal[] { 33.25m, 37.25m, 41.25m, 45.25m });
            print("*****");

            // decimals don't support NAN
            //ndarray e = np.arange(5, dtype: np.Float32);
            //e[2] = np.NaN;
            //ndarray f = np.amax(e);
            //print(f);
            //Assert.AreEqual(np.NaN, f.GetItem(0));
            //print("*****");

            //ndarray g = np.nanmax(b);
            //print(g);
        }

        [TestMethod]
        public void test_ptp_1_DECIMAL()
        {
            ndarray a = np.arange(4, dtype: np.Decimal).reshape(new shape(2, 2));
            print(a);
            print("*****");

            ndarray b = np.ptp(a, axis: 0);
            print(b);
            AssertArray(b, new decimal[] { 2, 2 });
            print("*****");

            ndarray c = np.ptp(a, axis: 1);
            print(c);
            AssertArray(c, new decimal[] { 1, 1 });

            ndarray d = np.ptp(a);
            print(d);
            Assert.AreEqual(3m, d.GetItem(0));
        }

        [TestMethod]
        public void test_percentile_2_DECIMAL()
        {
            var a = np.array(new decimal[,] { { 10, 7, 4 }, { 3, 2, 1 } });

            var b = np.percentile(a, new decimal[] { 50, 75 });
            AssertArray(b, new decimal[] { 3.5m, 6.25m });
            print(b);

            var c = np.percentile(a, new decimal[] { 50, 75 }, axis: 0);
            AssertArray(c, new decimal[,] { { 6.5m, 4.5m, 2.5m }, { 8.25m, 5.75m, 3.25m } });
            print(c);

            var d = np.percentile(a, new decimal[] { 50, 75 }, axis: 1);
            AssertArray(d, new decimal[,] { { 7.0m, 2.0m }, { 8.5m, 2.5m } });
            print(d);

            var e = np.percentile(a, new decimal[] { 50, 75 }, axis: 1, keepdims: true);
            AssertArray(e, new decimal[,,] { { { 7.0m }, { 2.0m } }, { { 8.5m }, { 2.5m } } });
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
        public void test_quantile_2_DECIMAL()
        {
            var a = np.array(new decimal[,] { { 10, 7, 4 }, { 3, 2, 1 } });

            var b = np.quantile(a, new decimal[] { 0.5m, 0.75m });
            AssertArray(b, new decimal[] { 3.5m, 6.25m });
            print(b);

            var c = np.quantile(a, new decimal[] { 0.5m, 0.75m }, axis: 0);
            AssertArray(c, new decimal[,] { { 6.5m, 4.5m, 2.5m }, { 8.25m, 5.75m, 3.25m } });
            print(c);

            var d = np.quantile(a, new decimal[] { 0.5m, 0.75m }, axis: 1);
            AssertArray(d, new decimal[,] { { 7.0m, 2.0m }, { 8.5m, 2.5m } });
            print(d);

            var e = np.quantile(a, new decimal[] { 0.5m, 0.75m }, axis: 1, keepdims: true);
            AssertArray(e, new decimal[,,] { { { 7.0m }, { 2.0m } }, { { 8.5m }, { 2.5m } } });
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
        public void test_median_2_DECIMAL()
        {
            var a = np.arange(0, 64, 1, np.Decimal).reshape((4, 4, 4));

            var b = np.median(a, axis: new int[] { 0, 2 }, keepdims: true);
            AssertArray(b, new decimal[,,] { { { 25.5m }, { 29.5m }, { 33.5m }, { 37.5m } } });
            print(b);

            var c = np.median(a, new int[] { 0, 1 }, keepdims: true);
            AssertArray(c, new decimal[,,] { { { 30, 31, 32, 33 } } });
            print(c);

            var d = np.median(a, new int[] { 1, 2 }, keepdims: true);
            AssertArray(d, new decimal[,,] { { { 7.5m } }, { { 23.5m } }, { { 39.5m } }, { { 55.5m } } });
            print(d);

            return;
        }

        [TestMethod]
        public void test_average_3_DECIMAL()
        {

            var a = np.array(new decimal[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            var w = new int[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

            var x1 = np.average(a, axis: null, weights: null, returned: true);
            Assert.AreEqual(5.5m, x1.retval.GetItem(0));
            Assert.AreEqual(10.0m, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a, axis: null, weights: w, returned: true);
            Assert.AreEqual(4.0m, x1.retval.GetItem(0));
            Assert.AreEqual(55.0m, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: null, weights: np.array(w).reshape((2, -1)), returned: true);
            Assert.AreEqual(4.0m, x1.retval.GetItem(0));
            Assert.AreEqual(55.0m, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: 0, weights: np.array(w).reshape((2, -1)), returned: true);
            AssertArray(x1.retval, new decimal[] { 2.6666666666666666666666666667m, 3.5384615384615384615384615385m, 4.3636363636363636363636363636m,
                                                   5.1111111111111111111111111111m, 5.7142857142857142857142857143m });
            AssertArray(x1.sum_of_weights, new decimal[] { 15.0m, 13.0m, 11.0m, 9.0m, 7.0m });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: 1, weights: np.array(w).reshape((2, -1)), returned: true);
            AssertArray(x1.retval, new decimal[] { 2.75m, 7.3333333333333333333333333333m });
            AssertArray(x1.sum_of_weights, new decimal[] { 40.0m, 15.0m });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((1, 2, -1, 1)), axis: 1, weights: np.array(w).reshape((1, 2, -1, 1)), returned: true);
            AssertArray(x1.retval, new decimal[,,] { { { 2.6666666666666666666666666667m }, { 3.5384615384615384615384615385m }, { 4.3636363636363636363636363636m },
                                                      { 5.1111111111111111111111111111m }, { 5.7142857142857142857142857143m } } });
            AssertArray(x1.sum_of_weights, new decimal[,,] { { { 15.0m }, { 13.0m }, { 11.0m }, { 9.0m }, { 7.0m } } });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((1, -1, 2, 1)), axis: 1, weights: np.array(w).reshape((1, -1, 2, 1)), returned: true);
            AssertArray(x1.retval, new decimal[,,] { { { 3.6666666666666666666666666667m }, { 4.4m } } });
            AssertArray(x1.sum_of_weights, new decimal[,,] { { { 30.0m }, { 25.0m } } });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1, 1, 1)), axis: 1, weights: np.array(w).reshape((2, -1, 1, 1)), returned: false);
            AssertArray(x1.retval, new decimal[,,] { { { 2.75m } }, { { 7.3333333333333333333333333333m } } });
            Assert.AreEqual(null, x1.sum_of_weights);
            print(x1);

        }

        [TestMethod]
        public void test_mean_1_DECIMAL()
        {
            decimal[] TestData = new decimal[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.Decimal).reshape(new shape(3, 2, -1));
            x = x * 3;
            print(x);

            var y = np.mean(x);
            print(y);
            Assert.AreEqual(131.5m, y.GetItem(0));

            y = np.mean(x, axis: 0);
            print(y);
            AssertArray(y, new decimal[,] { { 113, 150 }, { 113, 150 } });

            y = np.mean(x, axis: 1);
            print(y);
            AssertArray(y, new decimal[,] { { 52.5m, 90 }, { 132, 157.5m }, { 154.5m, 202.5m } });

            y = np.mean(x, axis: 2);
            print(y);
            AssertArray(y, new decimal[,] { { 37.5m, 105 }, { 252, 37.5m }, { 105, 252 } });

        }

        [TestMethod]
        public void test_mean_2_DECIMAL()
        {
            ndarray a = np.zeros(new shape(2, 512 * 512), dtype: np.Decimal);
            a[0, ":"] = 1.0;
            a[1, ":"] = 0.1;
            ndarray b = np.mean(a);
            print(b);
            Assert.AreEqual(0.55m, (decimal)b.GetItem(0));

            ndarray c = np.mean(a, dtype: np.Decimal);
            print(c);
            Assert.AreEqual(0.55m, c.GetItem(0));
        }

        [TestMethod]
        public void test_std_1_DECIMAL()
        {
            ndarray a = np.array(new decimal[,] { { 1, 2 }, { 3, 4 } });
            ndarray b = np.std(a);
            print(b);
            Assert.AreEqual(1.1180339887498948482045868344m, (decimal)b.GetItem(0));

            ndarray c = np.std(a, axis: 0);
            print(c);
            AssertArray(c, new decimal[] { 1.0m, 1.0m });

            ndarray d = np.std(a, axis: 1);
            print(d);
            AssertArray(d, new decimal[] { 0.5m, 0.5m }); 

            // In single precision, std() can be inaccurate:
            a = np.zeros(new shape(2, 512 * 512), dtype: np.Decimal);
            a[0, ":"] = 1.0;
            a[1, ":"] = 0.1;
            b = np.std(a);
            print(b);
            Assert.AreEqual(0.45m, b.GetItem(0));
            // Computing the standard deviation in float64 is more accurate:
            c = np.std(a, dtype: np.Decimal);
            print(c);
            Assert.AreEqual(0.45m, c.GetItem(0));

        }

        [TestMethod]
        public void test_var_1_DECIMAL()
        {
            ndarray a = np.array(new decimal[,] { { 1, 2 }, { 3, 4 } });
            ndarray b = np.var(a);
            Assert.AreEqual(1.25m, b.GetItem(0));
            print(b);

            ndarray c = np.var(a, axis: 0);
            AssertArray(c, new decimal[] { 1.0m, 1.0m });
            print(c);

            ndarray d = np.var(a, axis: 1);
            AssertArray(d, new decimal[] { 0.25m, 0.25m });
            print(d);

            // In single precision, std() can be inaccurate:
            a = np.zeros(new shape(2, 512 * 512), dtype: np.Decimal);
            a[0, ":"] = 1.0m;
            a[1, ":"] = 0.1m;
            b = np.var(a);
            Assert.AreEqual(0.2025m, b.GetItem(0));
            print(b);

            // Computing the standard deviation in float64 is more accurate:
            c = np.var(a, dtype: np.Decimal);
            Assert.AreEqual(0.2025m, c.GetItem(0));
            print(c);

        }

        [TestMethod]
        public void test_corrcoef_1_DECIMAL()
        {
            var x1 = np.array(new decimal[,] { { 0, 2 }, { 1, 1 }, { 2, 0 } }).T;
            print(x1);

            // Note how  increases while  decreases. The covariance matrix shows this clearly:

            var a = np.corrcoef(x1);
            AssertArray(a, new decimal[,] { { 1, -1 }, { -1, 1 } });
            print(a);

            var x = new decimal[] { -2.1m, -1, 4.3m };
            var y = new decimal[] { 3, 1.1m, 0.12m };
            var X = np.stack(new object[] { x, y }, axis: 0);
            a = np.corrcoef(X);
            AssertArray(a, new decimal[,] { { 1.0m, -0.8553578095227944904571128856m }, { -0.8553578095227944904571128856m, 1.0m } });
            print(a);


            var b = np.corrcoef(x, y);
            AssertArray(b, new decimal[,] { { 1.0m, -0.8553578095227944904571128856m }, { -0.8553578095227944904571128856m, 1.0m } });
            print(b);

            var c = np.corrcoef(x, y, rowvar: false);
            AssertArray(a, new decimal[,] { { 1.0m, -0.8553578095227944904571128856m }, { -0.8553578095227944904571128856m, 1.0m } });
            print(c);


            return;
        }

        [TestMethod]
        public void test_correlate_1_DECIMAL()
        {
            var a = np.correlate(new decimal[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f });
            AssertArray(a, new decimal[] { 3.5m });
            print(a);

            var b = np.correlate(new decimal[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_SAME);
            AssertArray(b, new decimal[] { 2.0m, 3.5m, 3.0m });
            print(b);

            var c = np.correlate(new decimal[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_FULL);
            AssertArray(c, new decimal[] { 0.5m, 2.0m, 3.5m, 3.0m, 0.0m });
            print(c);

            return;
        }

        [TestMethod]
        public void test_cov_1_DECIMAL()
        {
            var x1 = np.array(new decimal[,] { { 0, 2 }, { 1, 1 }, { 2, 0 } }).T;
            print(x1);

            // Note how  increases while  decreases. The covariance matrix shows this clearly:

            var a = np.cov(x1);
            AssertArray(a, new decimal[,] { { 1, -1 }, { -1, 1 } });
            print(a);

            var x = new decimal[] { -2.1m, -1, 4.3m };
            var y = new decimal[] { 3, 1.1m, 0.12m };
            var X = np.stack(new object[] { x, y }, axis: 0);
            a = np.cov(X);
            AssertArray(a, new decimal[,] { { 11.710m, -4.2860000000000000000000000000m }, { -4.2860000000000000000000000000m, 2.1441333333333333333333333334m } });
            print(a);


            var b = np.cov(x, y);
            AssertArray(b, new decimal[,] { { 11.710m, -4.2860000000000000000000000000m }, { -4.2860000000000000000000000000m, 2.1441333333333333333333333334m } });
            print(b);

            var c = np.cov(x);
            Assert.AreEqual(11.710m, c.GetItem(0));
            print(c);

            var d = np.cov(X, rowvar: false);
            AssertArray(d, new decimal[,] { { 13.00500m, 5.35500m, -10.65900m }, { 5.35500m, 2.20500m, -4.38900m }, { -10.65900m, -4.38900m, 8.73620m } });
            print(d);

            var e = np.cov(X, rowvar: false, bias: true);
            AssertArray(e, new decimal[,] { { 6.50250m, 2.67750m, -5.32950m }, { 2.67750m, 1.10250m, -2.19450m }, { -5.32950m, -2.19450m, 4.36810m } });
            print(e);

            var f = np.cov(X, rowvar: false, bias: true, fweights: new int[] { 1, 2 });
            AssertArray(f, new decimal[,] { { 5.7799999999999999999999999994m, 2.3799999999999999999999999998m, -4.7373333333333389999999999995m },
                                            { 2.3799999999999999999999999998m, 0.9799999999999999999999999999m, -1.9506666666666689999999999998m },
                                            { -4.7373333333333333333333333329m, -1.9506666666666666666666666665m, 3.8827555555555601999999999996m }});
            print(f);

            var g = np.cov(X, rowvar: false, bias: true, fweights: new int[] { 1, 2 }, aweights: new int[] { 1, 2 });
            AssertArray(g, new decimal[,] { { 4.16160m, 1.71360m, -3.410880m }, { 1.71360m, 0.70560m, -1.404480m }, { -3.410880m, -1.404480m, 2.7955840m } });
            print(g);

            return;
        }

        #endregion

        #region from TwoDimBaseTests

        [TestMethod]
        public void test_diag_1_DECIMAL()
        {
            ndarray m = np.arange(9, dtype: np.Decimal);
            var n = np.diag(m);

            print(m);
            print(n);

            var ExpectedDataN = new decimal[,]
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

            m = np.arange(9, dtype: np.Decimal).reshape(new shape(3, 3));
            n = np.diag(m);

            print(m);
            print(n);
            AssertArray(n, new decimal[] { 0, 4, 8 });
        }

        [TestMethod]
        public void test_diagflat_1_DECIMAL()
        {
            ndarray m = np.arange(1, 5, dtype: np.Decimal).reshape(new shape(2, 2));
            var n = np.diagflat(m);

            print(m);
            print(n);

            var ExpectedDataN = new decimal[,]
            {
             {1, 0, 0, 0},
             {0, 2, 0, 0},
             {0, 0, 3, 0},
             {0, 0, 0, 4}
            };
            AssertArray(n, ExpectedDataN);

            m = np.arange(1, 3, dtype: np.Decimal);
            n = np.diagflat(m, 1);

            print(m);
            print(n);

            ExpectedDataN = new decimal[,]
            {
             {0, 1, 0},
             {0, 0, 2},
             {0, 0, 0},
            };

            AssertArray(n, ExpectedDataN);

            m = np.arange(1, 3, dtype: np.Decimal);
            n = np.diagflat(m, -1);

            print(m);
            print(n);

            ExpectedDataN = new decimal[,]
            {
             {0, 0, 0},
             {1, 0, 0},
             {0, 2, 0},
            };

            AssertArray(n, ExpectedDataN);

        }

        [TestMethod]
        public void test_fliplr_1_DECIMAL()
        {
            ndarray m = np.arange(8, dtype: np.Decimal).reshape(new shape(2, 2, 2));
            var n = np.fliplr(m);

            print(m);
            print(n);

            AssertArray(n, new decimal[,,] { { { 2, 3 }, { 0, 1 } }, { { 6, 7 }, { 4, 5 } } });
        }

        [TestMethod]
        public void test_flipud_1_DECIMAL()
        {
            ndarray m = np.arange(8, dtype: np.Decimal).reshape(new shape(2, 2, 2));
            var n = np.flipud(m);

            print(m);
            print(n);

            AssertArray(n, new decimal[,,] { { { 4, 5 }, { 6, 7 } }, { { 0, 1 }, { 2, 3 } } });
        }

        [TestMethod]
        public void test_tri_1_DECIMAL()
        {
            ndarray a = np.tri(3, 5, 2, dtype: np.Decimal);
            print(a);

            var ExpectedDataA = new decimal[,]
            {
             {1, 1, 1, 0, 0},
             {1, 1, 1, 1, 0},
             {1, 1, 1, 1, 1}
            };
            AssertArray(a, ExpectedDataA);

            print("***********");
            ndarray b = np.tri(3, 5, -1, dtype: np.Decimal);
            print(b);

            var ExpectedDataB = new decimal[,]
            {
             {0.0m, 0.0m, 0.0m, 0.0m, 0.0m},
             {1.0m, 0.0m, 0.0m, 0.0m, 0.0m},
             {1.0m, 1.0m, 0.0m, 0.0m, 0.0m}
            };
            AssertArray(b, ExpectedDataB);
        }

        [TestMethod]
        public void test_tril_1_DECIMAL()
        {
            ndarray a = np.array(new decimal[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } });
            ndarray b = np.tril(a, -1);
            print(a);
            print("***********");
            print(b);

            var ExpectedDataB = new decimal[,]
            {
             {0, 0, 0},
             {4, 0, 0},
             {7, 8, 0},
             {10, 11, 12},
            };
            AssertArray(b, ExpectedDataB);

        }

        [TestMethod]
        public void test_triu_1_DECIMAL()
        {
            ndarray a = np.array(new decimal[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } });
            ndarray b = np.triu(a, -1);
            print(a);
            print("***********");
            print(b);

            var ExpectedDataB = new decimal[,]
            {
             {1, 2, 3},
             {4, 5, 6},
             {0, 8, 9},
             {0, 0, 12},
            };
            AssertArray(b, ExpectedDataB);

        }

        [TestMethod]
        public void test_vander_1_DECIMAL()
        {
            var x = np.array(new decimal[] { 1, 2, 3, 5 });
            int N = 3;
            var y = np.vander(x, N);
            AssertArray(y, new decimal[,] { { 1, 1, 1 }, { 4, 2, 1 }, { 9, 3, 1 }, { 25, 5, 1 } });
            print(y);

            y = np.vander(x);
            AssertArray(y, new decimal[,] { { 1, 1, 1, 1 }, { 8, 4, 2, 1 }, { 27, 9, 3, 1 }, { 125, 25, 5, 1 } });
            print(y);

            y = np.vander(x, increasing: true);
            AssertArray(y, new decimal[,] { { 1, 1, 1, 1 }, { 1, 2, 4, 8 }, { 1, 3, 9, 27 }, { 1, 5, 25, 125 } });
            print(y);

            return;
        }

        [TestMethod]
        public void test_mask_indices_DECIMAL()
        {
            var iu = np.mask_indices(3, np.triu);
            AssertArray(iu[0], new npy_intp[] { 0, 0, 0, 1, 1, 2 });
            AssertArray(iu[1], new npy_intp[] { 0, 1, 2, 1, 2, 2 });
            print(iu);

            var a = np.arange(9, dtype: np.Decimal).reshape((3, 3));
            var b = a[iu] as ndarray;
            AssertArray(b, new decimal[] { 0, 1, 2, 4, 5, 8 });
            print(b);

            var iu1 = np.mask_indices(3, np.triu, 1);

            var c = a[iu1] as ndarray;
            AssertArray(c, new decimal[] { 1, 2, 5 });
            print(c);

            return;
        }

        [TestMethod]
        public void test_tril_indices_DECIMAL()
        {
            var il1 = np.tril_indices(4);
            var il2 = np.tril_indices(4, 2);

            var a = np.arange(16, dtype: np.Decimal).reshape((4, 4));
            var b = a[il1] as ndarray;
            AssertArray(b, new decimal[] { 0, 4, 5, 8, 9, 10, 12, 13, 14, 15 });
            print(b);

            a[il1] = -1;

            var ExpectedDataA1 = new decimal[,]
                {{-1,  1, 2,  3}, {-1, -1,  6,  7},
                 {-1, -1,-1, 11}, {-1, -1, -1, -1}};
            AssertArray(a, ExpectedDataA1);
            print(a);

            a[il2] = -10;

            var ExpectedDataA2 = new decimal[,]
                {{-10, -10, -10,  3}, {-10, -10, -10, -10},
                 {-10, -10,-10, -10}, {-10, -10, -10, -10}};
            AssertArray(a, ExpectedDataA2);
            print(a);

            return;
        }

        [TestMethod]
        public void test_tril_indices_from_DECIMAL()
        {
            var a = np.arange(16, dtype: np.Decimal).reshape((4, 4));
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
        public void test_triu_indices_DECIMAL()
        {
            var il1 = np.triu_indices(4);
            var il2 = np.triu_indices(4, 2);

            var a = np.arange(16, dtype: np.Decimal).reshape((4, 4));
            var b = a[il1] as ndarray;
            AssertArray(b, new decimal[] { 0, 1, 2, 3, 5, 6, 7, 10, 11, 15 });
            print(b);

            a[il1] = -1;

            var ExpectedDataA1 = new decimal[,]
                {{-1, -1, -1, -1}, { 4, -1, -1, -1},
                 { 8,  9, -1, -1}, {12, 13, 14, -1}};
            AssertArray(a, ExpectedDataA1);
            print(a);

            a[il2] = -10;

            var ExpectedDataA2 = new decimal[,]
                {{-1, -1, -10, -10}, {4,  -1, -1, -10},
                 { 8,  9, -1,  -1},  {12, 13, 14, -1}};
            AssertArray(a, ExpectedDataA2);
            print(a);

            return;
        }

        [TestMethod]
        public void test_triu_indices_from_DECIMAL()
        {
            var a = np.arange(16, dtype: np.Decimal).reshape((4, 4));
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
        public void test_atleast_1d_DECIMAL()
        {
            var a = np.atleast_1d(1.0m);
            print(a);
            AssertArray(a.ElementAt(0), new decimal[] { 1.0m });

            print("**************");
            var x = np.arange(9.0, dtype: np.Decimal).reshape(new shape(3, 3));
            var b = np.atleast_1d(x);
            print(b);

            var ExpectedB = new decimal[,]
                {{0.0m, 1.0m, 2.0m},
                 {3.0m, 4.0m, 5.0m},
                 {6.0m, 7.0m, 8.0m}};
            AssertArray(b.ElementAt(0), ExpectedB);

            print("**************");

            var c = np.atleast_1d(new object[] { 1m, new decimal[] { 3, 4 } });

            AssertArray(c.ElementAt(0), new decimal[] { 1 });
            AssertArray(c.ElementAt(1), new decimal[] { 3, 4 });
            print(c);

        }

        [TestMethod]
        public void test_atleast_2d_DECIMAL()
        {
            var a = np.atleast_2d(1.0m);
            print(a);
            AssertArray(a.ElementAt(0), new decimal[,] { { 1.0m } });

            print("**************");
            var x = np.arange(9.0, dtype: np.Decimal).reshape(new shape(3, 3));
            var b = np.atleast_2d(x);
            print(b);

            var ExpectedB = new decimal[,]
                {{0.0m, 1.0m, 2.0m},
                 {3.0m, 4.0m, 5.0m},
                 {6.0m, 7.0m, 8.0m}};
            AssertArray(b.ElementAt(0), ExpectedB);

            print("**************");

            var c = np.atleast_2d(new object[] { 1m, new decimal[] { 3, 4 }, new decimal[] { 5, 6 } });

            AssertArray(c.ElementAt(0), new decimal[,] { { 1 } });
            AssertArray(c.ElementAt(1), new decimal[,] { { 3, 4 } });
            AssertArray(c.ElementAt(2), new decimal[,] { { 5, 6 } });
            print(c);

        }

        [TestMethod]
        public void test_atleast_3d_DECIMAL()
        {
            var a = np.atleast_3d(1.0m);
            print(a);
            AssertArray(a.ElementAt(0), new decimal[,,] { { { 1.0m } } });

            print("**************");
            var x = np.arange(9.0, dtype: np.Decimal).reshape(new shape(3, 3));
            var b = np.atleast_3d(x);
            print(b);

            var ExpectedB = new decimal[,,]
             {{{0.0m},
               {1.0m},
               {2.0m}},
              {{3.0m},
               {4.0m},
               {5.0m}},
              {{6.0m},
               {7.0m},
               {8.0m}}};

            AssertArray(b.ElementAt(0), ExpectedB);

            print("**************");

            var c = np.atleast_3d(new object[] { new decimal[] { 1, 2 }, new decimal[] { 3, 4 }, new decimal[] { 5, 6 } });

            AssertArray(c.ElementAt(0), new decimal[,,] { { { 1 }, { 2 } } });
            AssertArray(c.ElementAt(1), new decimal[,,] { { { 3 }, { 4 } } });
            AssertArray(c.ElementAt(2), new decimal[,,] { { { 5 }, { 6 } } });
            print(c);


        }

        [TestMethod]
        public void test_vstack_2_DECIMAL()
        {
            var a = np.array(new decimal[,] { { 1 }, { 2 }, { 3 } });
            var b = np.array(new decimal[,] { { 2 }, { 3 }, { 4 } });
            var c = np.vstack(new object[] { a, b });

            AssertArray(c, new decimal[,] { { 1 }, { 2 }, { 3 }, { 2 }, { 3 }, { 4 } });

            print(c);
        }

        [TestMethod]
        public void test_hstack_2_DECIMAL()
        {
            var a = np.array(new decimal[,] { { 1 }, { 2 }, { 3 } });
            var b = np.array(new decimal[,] { { 2 }, { 3 }, { 4 } });
            var c = np.hstack(new object[] { a, b });

            AssertArray(c, new decimal[,] { { 1, 2 }, { 2, 3 }, { 3, 4 } });

            print(c);
        }

        [TestMethod]
        public void test_stack_1_DECIMAL()
        {
            var a = np.array(new decimal[,] { { 1 }, { 2 }, { 3 } });
            var b = np.array(new decimal[,] { { 2 }, { 3 }, { 4 } });

            var c = np.stack(new object[] { a, b }, axis: 0);
            AssertArray(c, new decimal[,,] { { { 1 }, { 2 }, { 3 } }, { { 2 }, { 3 }, { 4 } } });
            print(c);
            print("**************");

            var d = np.stack(new object[] { a, b }, axis: 1);
            AssertArray(d, new decimal[,,] { { { 1 }, { 2 } }, { { 2 }, { 3 } }, { { 3 }, { 4 } } });
            print(d);
            print("**************");

            var e = np.stack(new object[] { a, b }, axis: 2);
            AssertArray(e, new decimal[,,] { { { 1, 2 } }, { { 2, 3 } }, { { 3, 4 } } });
            print(e);

        }

        [TestMethod]
        public void test_block_2_DECIMAL()
        {
            var a = np.array(new decimal[] { 1, 2, 3 });
            var b = np.array(new decimal[] { 2, 3, 4 });
            var c = np.block(new object[] { a, b, 10 });    // hstack([a, b, 10])

            AssertArray(c, new decimal[] { 1, 2, 3, 2, 3, 4, 10 });
            print(c);
            print("**************");

            a = np.array(new decimal[] { 1, 2, 3 });
            b = np.array(new decimal[] { 2, 3, 4 });
            c = np.block(new object[] { new object[] { a }, new object[] { b } });    // vstack([a, b])

            AssertArray(c, new decimal[,] { { 1, 2, 3 }, { 2, 3, 4 } });
            print(c);

        }

        [TestMethod]
        public void test_expand_dims_1_DECIMAL()
        {
            var a = np.array(new decimal[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }).reshape(new shape(2, -1, 2));
            var b = np.expand_dims(a, axis: 0);

            var ExpectedDataB = new decimal[,,,]
            {{{{1,  2}, {3,  4}, {5,  6}},
              {{7,  8}, {9, 10}, {11, 12}}}};

            AssertArray(b, ExpectedDataB);
            print(b);
            print("**************");

            var c = np.expand_dims(a, axis: 1);
            var ExpectedDataC = new decimal[,,,]
                {{{{1,  2}, {3,  4}, {5,  6}}},
                {{{ 7,  8},{ 9, 10}, {11, 12}}}};
            AssertArray(c, ExpectedDataC);
            print(c);
            print("**************");

            var d = np.expand_dims(a, axis: 2);
            var ExpectedDataD = new decimal[,,,]
            {{{{1,  2}},{{3,  4}},{{5,  6}}},
             {{{7,  8}},{{9, 10}},{{11, 12}}}};

            AssertArray(d, ExpectedDataD);
            print(d);

        }

        [TestMethod]
        public void test_column_stack_1_DECIMAL()
        {
            var a = np.array(new decimal[] { 1, 2, 3 });
            var b = np.array(new decimal[] { 2, 3, 4 });
            var c = np.column_stack(new object[] { a, b });

            AssertArray(c, new decimal[,] { { 1, 2 }, { 2, 3 }, { 3, 4 } });
            print(c);
        }

        [TestMethod]
        public void test_row_stack_1_DECIMAL()
        {
            var a = np.array(new decimal[] { 1, 2, 3 });
            var b = np.array(new decimal[] { 2, 3, 4 });
            var c = np.row_stack(new object[] { a, b });

            AssertArray(c, new decimal[,] { { 1, 2, 3 }, { 2, 3, 4 } });

            print(c);
        }

        [TestMethod]
        public void test_dstack_1_DECIMAL()
        {
            var a = np.array(new decimal[] { 1, 2, 3 });
            var b = np.array(new decimal[] { 2, 3, 4 });
            var c = np.dstack(new object[] { a, b });

            AssertArray(c, new decimal[,,] { { { 1, 2 }, { 2, 3 }, { 3, 4 } } });
            print(c);

            a = np.array(new decimal[,] { { 1 }, { 2 }, { 3 } });
            b = np.array(new decimal[,] { { 2 }, { 3 }, { 4 } });
            c = np.dstack(new object[] { a, b });

            AssertArray(c, new decimal[,,] { { { 1, 2 } }, { { 2, 3 } }, { { 3, 4 } } });

            print(c);
        }

        [TestMethod]
        public void test_array_split_2_DECIMAL()
        {
            var x = np.arange(16.0, dtype: np.Decimal).reshape(new shape(2, 8, 1));
            var y = np.array_split(x, 3, axis: 0);


            AssertArray(y.ElementAt(0), new decimal[,,] { { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 } } });
            AssertArray(y.ElementAt(1), new decimal[,,] { { { 8 }, { 9 }, { 10 }, { 11 }, { 12 }, { 13 }, { 14 }, { 15 } } });
            AssertShape(y.ElementAt(2), 0, 8, 1);

            print(y);

            print("**************");

            x = np.arange(16.0, dtype: np.Decimal).reshape(new shape(2, 8, 1));
            y = np.array_split(x, 3, axis: 1);

            AssertArray(y.ElementAt(0), new decimal[,,] { { { 0 }, { 1 }, { 2 } }, { { 8 }, { 9 }, { 10 } } });
            AssertArray(y.ElementAt(1), new decimal[,,] { { { 3 }, { 4 }, { 5 } }, { { 11 }, { 12 }, { 13 } } });
            AssertArray(y.ElementAt(2), new decimal[,,] { { { 6 }, { 7 } }, { { 14 }, { 15 } } });


            print(y);

            print("**************");

            x = np.arange(16.0, dtype: np.Decimal).reshape(new shape(2, 8, 1));
            y = np.array_split(x, 3, axis: 2);

            AssertArray(y.ElementAt(0), new decimal[,,] { { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 } }, { { 8 }, { 9 }, { 10 }, { 11 }, { 12 }, { 13 }, { 14 }, { 15 } } });
            AssertShape(y.ElementAt(1), 2, 8, 0);
            AssertShape(y.ElementAt(2), 2, 8, 0);
            print(y);
        }

        [TestMethod]
        public void test_split_2_DECIMAL()
        {
            var x = np.arange(16.0, dtype: np.Decimal).reshape(new shape(8, 2, 1));
            var y = np.split(x, new Int32[] { 2, 3 }, axis: 0);

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new decimal[,,] { { { 0 }, { 1 } }, { { 2 }, { 3 } } });
            AssertArray(y.ElementAt(1), new decimal[,,] { { { 4 }, { 5 } } });
            AssertArray(y.ElementAt(2), new decimal[,,] { { { 6 }, { 7 } }, { { 8 }, { 9 } }, { { 10 }, { 11 } }, { { 12 }, { 13 } }, { { 14 }, { 15 } } });


            print(y);

            print("**************");

            x = np.arange(16.0, dtype: np.Decimal).reshape(new shape(8, 2, 1));
            y = np.split(x, new int[] { 2, 3 }, axis: 1);

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new decimal[,,] {{{0},{1}},{{2}, {3}}, {{4}, {5}}, {{6}, { 7}},
                                                        {{8},{9}},{{10},{11}}, {{12}, {13}}, {{14}, {15}}});
            AssertShape(y.ElementAt(1), 8, 0, 1);
            AssertShape(y.ElementAt(2), 8, 0, 1);

            print(y);

            print("**************");

            x = np.arange(16.0, dtype: np.Decimal).reshape(new shape(8, 2, 1));
            y = np.split(x, new int[] { 2, 3 }, axis: 2);

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new decimal[,,] {{{ 0},{ 1}},{{ 2}, { 3}}, {{ 4}, { 5}}, {{ 6}, { 7}},
                                                        {{ 8},{ 9}},{{10}, {11}}, {{12}, {13}}, {{14}, {15}}});
            AssertShape(y.ElementAt(1), 8, 2, 0);
            AssertShape(y.ElementAt(2), 8, 2, 0);

            print(y);
        }

        [TestMethod]
        public void test_hsplit_2_DECIMAL()
        {
            var x = np.arange(8, dtype: np.Decimal).reshape(new shape(2, 2, 2));
            var y = np.hsplit(x, 2);

            Assert.AreEqual(2, y.Count);
            AssertArray(y.ElementAt(0), new decimal[,,] { { { 0, 1 } }, { { 4, 5 } } });
            AssertArray(y.ElementAt(1), new decimal[,,] { { { 2, 3 } }, { { 6, 7 } } });
            print(y);

            print("**************");

            x = np.arange(8, dtype: np.Decimal).reshape(new shape(2, 2, 2));
            y = np.hsplit(x, new Int32[] { 3, 6 });

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new decimal[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } });
            AssertShape(y.ElementAt(1), 2, 0, 2);
            AssertShape(y.ElementAt(2), 2, 0, 2);

            print(y);
        }

        [TestMethod]
        public void test_vsplit_2_DECIMAL()
        {
            var x = np.arange(8, dtype: np.Decimal).reshape(new shape(2, 2, 2));
            var y = np.vsplit(x, 2);

            Assert.AreEqual(2, y.Count);
            AssertArray(y.ElementAt(0), new decimal[,,] { { { 0, 1 }, { 2, 3 } } });
            AssertArray(y.ElementAt(1), new decimal[,,] { { { 4, 5 }, { 6, 7 } } });
            print(y);

            print("**************");

            x = np.arange(8, dtype: np.Decimal).reshape(new shape(2, 2, 2));
            y = np.vsplit(x, new int[] { 3, 6 });

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new decimal[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } });
            AssertShape(y.ElementAt(1), 0, 2, 2);
            AssertShape(y.ElementAt(2), 0, 2, 2);

            print(y);
        }

        [TestMethod]
        public void test_dsplit_1_DECIMAL()
        {
            var x = np.arange(16, dtype: np.Decimal).reshape(new shape(2, 2, 4));
            var y = np.dsplit(x, 2);

            Assert.AreEqual(2, y.Count);
            AssertArray(y.ElementAt(0), new decimal[,,] { { { 0, 1 }, { 4, 5 } }, { { 8, 9 }, { 12, 13 } } });
            AssertArray(y.ElementAt(1), new decimal[,,] { { { 2, 3 }, { 6, 7 } }, { { 10, 11 }, { 14, 15 } } });
            print(y);


            print("**************");

            x = np.arange(16, dtype: np.Decimal).reshape(new shape(2, 2, 4));
            y = np.dsplit(x, new int[] { 3, 6 });

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new decimal[,,] { { { 0, 1, 2 }, { 4, 5, 6 } }, { { 8, 9, 10 }, { 12, 13, 14 } } });
            AssertArray(y.ElementAt(1), new decimal[,,] { { { 3 }, { 7 } }, { { 11 }, { 15 } } });
            AssertShape(y.ElementAt(2), 2, 2, 0);

            print(y);
        }

        [TestMethod]
        public void test_kron_1_DECIMAL()
        {

            var a = np.kron(new decimal[] { 1, 10, 100 }, new decimal[] { 5, 6, 7 });
            AssertArray(a, new decimal[] { 5, 6, 7, 50, 60, 70, 500, 600, 700 });
            print(a);

            var b = np.kron(new decimal[] { 5, 6, 7 }, new decimal[] { 1, 10, 100 });
            AssertArray(b, new decimal[] { 5, 50, 500, 6, 60, 600, 7, 70, 700 });
            print(b);

            var x = np.array(new decimal[,] { { 2, 3 }, { 4, 5 } });
            var y = np.array(new decimal[,] { { 5, 6 }, { 7, 8 } });

            var c = np.kron(x, y);
            AssertArray(c, new decimal[,] { { 10, 12, 15, 18 }, { 14, 16, 21, 24 }, { 20, 24, 25, 30 }, { 28, 32, 35, 40 } });
            print(c);
            print(c.shape);

            c = np.kron(np.eye(2, dtype: np.Decimal), np.ones(new shape(2, 2), dtype: np.Decimal));
            AssertArray(c, new decimal[,] { { 1, 1, 0, 0 }, { 1, 1, 0, 0 }, { 0, 0, 1, 1 }, { 0, 0, 1, 1 } });


            x = np.array(new decimal[,,] { { { 2, 3, 3 }, { 4, 5, 3 } } });
            y = np.array(new decimal[,,] { { { 5, 6, 6, 6 }, { 7, 8, 6, 6 } } });

            c = np.kron(x, y);
            AssertArray(c, new decimal[,,] { { { 10, 12, 12, 12, 15, 18, 18, 18, 15, 18, 18, 18 },
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
        public void test_tile_2_DECIMAL()
        {
            var a = np.array(new decimal[,] { { 1, 2 }, { 3, 4 } });
            var b = np.tile(a, 2);
            AssertArray(b, new decimal[,] { { 1, 2, 1, 2 }, { 3, 4, 3, 4 } });
            print(b);
            print("**************");

            var c = np.tile(a, new decimal[] { 2, 1 });
            AssertArray(c, new decimal[,] { { 1, 2 }, { 3, 4 }, { 1, 2 }, { 3, 4 } });
            print(c);
            print("**************");

            var d = np.array(new decimal[] { 1, 2, 3, 4 });
            var e = np.tile(d, new decimal[] { 4, 1 });

            AssertArray(e, new decimal[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 } });
            print(e);
        }

        #endregion

        #region from UFUNCTests

        [TestMethod]
        public void test_UFUNC_AddReduce_1_DECIMAL()
        {
            var x = np.arange(8, dtype: np.Decimal);

            var a = np.ufunc.reduce(UFuncOperation.add, x);
            Assert.AreEqual(28m, a.GetItem(0));
            print(a);

            x = np.arange(8, dtype: np.Decimal).reshape((2, 2, 2));
            var b = np.ufunc.reduce(UFuncOperation.add, x);
            AssertArray(b, new decimal[,] { { 4, 6 }, { 8, 10 } });
            print(b);

            var c = np.ufunc.reduce(UFuncOperation.add, x, 0);
            AssertArray(c, new decimal[,] { { 4, 6 }, { 8, 10 } });
            print(c);

            var d = np.ufunc.reduce(UFuncOperation.add, x, 1);
            AssertArray(d, new decimal[,] { { 2, 4 }, { 10, 12 } });
            print(d);

            var e = np.ufunc.reduce(UFuncOperation.add, x, 2);
            AssertArray(e, new decimal[,] { { 1, 5 }, { 9, 13 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_AddAccumulate_1_DECIMAL()
        {
            var x = np.arange(8, dtype: np.Decimal);

            var a = np.ufunc.accumulate(UFuncOperation.add, x);
            AssertArray(a, new decimal[] { 0, 1, 3, 6, 10, 15, 21, 28 });
            print(a);

            x = np.arange(8, dtype: np.Decimal).reshape((2, 2, 2));
            var b = np.ufunc.accumulate(UFuncOperation.add, x);
            AssertArray(b, new decimal[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 6 }, { 8, 10 } } });
            print(b);

            var c = np.ufunc.accumulate(UFuncOperation.add, x, 0);
            AssertArray(c, new decimal[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 6 }, { 8, 10 } } });
            print(c);

            var d = np.ufunc.accumulate(UFuncOperation.add, x, 1);
            AssertArray(d, new decimal[,,] { { { 0, 1 }, { 2, 4 } }, { { 4, 5 }, { 10, 12 } } });
            print(d);

            var e = np.ufunc.accumulate(UFuncOperation.add, x, 2);
            AssertArray(e, new decimal[,,] { { { 0, 1 }, { 2, 5 } }, { { 4, 9 }, { 6, 13 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_AddReduceAt_1_DECIMAL()
        {
            var a = np.ufunc.reduceat(UFuncOperation.add, np.arange(8, dtype: np.Decimal), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 })["::2"] as ndarray;
            AssertArray(a, new decimal[] { 6, 10, 14, 18 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16, dtype: np.Decimal).reshape((4, 4));
            var b = np.ufunc.reduceat(UFuncOperation.add, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new decimal[,] {{12.0m, 15.0m, 18.0m, 21.0m},{12.0m, 13.0m, 14.0m, 15.0m}, {4.0m, 5.0m, 6.0m, 7.0m},
                                          {8.0m, 9.0m, 10.0m, 11.0m}, {24.0m, 28.0m, 32.0m, 36.0m}});
            print(b);

            var c = np.ufunc.reduceat(UFuncOperation.multiply, x, new npy_intp[] { 0, 3 }, axis: 1);
            AssertArray(c, new decimal[,] { { 0.0m, 3.0m }, { 120.0m, 7.0m }, { 720.0m, 11.0m }, { 2184.0m, 15.0m } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_AddOuter_1_DECIMAL()
        {
            var x = np.arange(4, dtype: np.Decimal);

            var a = np.ufunc.outer(UFuncOperation.add, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new decimal[,] { { 0, 1, 2, 3 }, { 1, 2, 3, 4 }, { 2, 3, 4, 5 }, { 3, 4, 5, 6 } });
            print(a);

            x = np.arange(6, dtype: np.Decimal).reshape((3, 2));
            var y = np.arange(6, dtype: np.Decimal).reshape((2, 3));
            var b = np.ufunc.outer(UFuncOperation.add, null, x, y);
            AssertShape(b, 3, 2, 2, 3);
            print(b.shape);

            var ExpectedDataB = new decimal[,,,]

                {{{{0,  1,  2}, {3,  4,  5}}, {{1,  2,  3}, { 4,  5,  6}}},
                 {{{2,  3,  4}, {5,  6,  7}}, {{3,  4,  5}, { 6,  7,  8}}},
                 {{{4,  5,  6}, {7,  8,  9}}, {{5,  6,  7}, { 8,  9, 10}}}};

            AssertArray(b, ExpectedDataB);

            print(b);
        }

        #endregion

        #region from IndexTricksTests

        [TestMethod]
        public void test_mgrid_1_DECIMAL()
        {
            var a = (ndarray)np.mgrid(new Slice[] { new Slice(0m, 5m) });
            print(a);
            AssertArray(a, new decimal[] { 0, 1, 2, 3, 4 });
            print("************");

            var b = (ndarray)np.mgrid(new Slice[] { new Slice(0.0m, 5.5m) });
            print(b);
            AssertArray(b, new decimal[] { 0.0m, 1.0m, 2.0m, 3.0m, 4.0m, 5.0m });
            print("************");

            var c = (ndarray)np.mgrid(new Slice[] { new Slice(0m, 5m), new Slice(0m, 5m) });
            print(c);

            var ExpectedCArray = new decimal[,,]
                {{{0, 0, 0, 0, 0},  {1, 1, 1, 1, 1},  {2, 2, 2, 2, 2},  {3, 3, 3, 3, 3},  {4, 4, 4, 4, 4}},
                 {{0, 1, 2, 3, 4},  {0, 1, 2, 3, 4},  {0, 1, 2, 3, 4},  {0, 1, 2, 3, 4},  {0, 1, 2, 3, 4}}};
            AssertArray(c, ExpectedCArray);


            print("************");

            var d = (ndarray)np.mgrid(new Slice[] { new Slice(0m, 5.5m), new Slice(0m, 5.5m) });
            print(d);
            var ExpectedDArray = new decimal[,,]
                {{{0, 0, 0, 0, 0, 0},  {1, 1, 1, 1, 1, 1},  {2, 2, 2, 2, 2, 2},  {3, 3, 3, 3, 3, 3},  {4, 4, 4, 4, 4, 4}, {5, 5, 5, 5, 5, 5}},
                 {{0, 1, 2, 3, 4, 5},  {0, 1, 2, 3, 4, 5},  {0, 1, 2, 3, 4, 5},  {0, 1, 2, 3, 4, 5},  {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}}};
            AssertArray(d, ExpectedDArray);

            print("************");

            var e = (ndarray)np.mgrid(new Slice[] { new Slice(3m, 5m), new Slice(4m, 6m), new Slice(2m, 4.2m) });
            print(e);
            var ExpectedEArray = new decimal[,,,]
                {
                    {{{3, 3, 3}, {3, 3, 3}}, {{4, 4, 4}, {4, 4, 4}}},
                    {{{4, 4, 4}, {5, 5, 5}}, {{4, 4, 4}, {5, 5, 5}}},
                    {{{2, 3, 4}, {2, 3, 4}}, {{2, 3, 4}, {2, 3, 4}}},
                };
            AssertArray(e, ExpectedEArray);

        }

        [TestMethod]
        public void test_ogrid_1_DECIMAL()
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
        public void test_fill_diagonal_1_DECIMAL()
        {
            var a = np.zeros((3, 3), np.Decimal);
            np.fill_diagonal(a, 5);
            AssertArray(a, new decimal[,] { { 5, 0, 0 }, { 0, 5, 0 }, { 0, 0, 5 } });
            print(a);

            a = np.zeros((3, 3, 3, 3), np.Decimal);
            np.fill_diagonal(a, 4);
            AssertArray(a[0, 0] as ndarray, new decimal[,] { { 4, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } });
            print(a[0, 0]);
            AssertArray(a[1, 1] as ndarray, new decimal[,] { { 0, 0, 0 }, { 0, 4, 0 }, { 0, 0, 0 } });
            print(a[1, 1]);
            AssertArray(a[2, 2] as ndarray, new decimal[,] { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 4 } });
            print(a[2, 2]);

            // tall matrices no wrap
            a = np.zeros((5, 3), np.Decimal);
            np.fill_diagonal(a, 4);
            AssertArray(a, new decimal[,] { { 4, 0, 0 }, { 0, 4, 0 }, { 0, 0, 4 }, { 0, 0, 0 }, { 0, 0, 0 } });
            print(a);

            // tall matrices wrap
            a = np.zeros((5, 3), np.Decimal);
            np.fill_diagonal(a, 4, wrap: true);
            AssertArray(a, new decimal[,] { { 4, 0, 0 }, { 0, 4, 0 }, { 0, 0, 4 }, { 0, 0, 0 }, { 4, 0, 0 } });
            print(a);

            // wide matrices wrap
            a = np.zeros((3, 5), np.Decimal);
            np.fill_diagonal(a, 4, wrap: true);
            AssertArray(a, new decimal[,] { { 4, 0, 0, 0, 0 }, { 0, 4, 0, 0, 0 }, { 0, 0, 4, 0, 0 } });
            print(a);


        }

        [TestMethod]
        public void test_diag_indices_1_DECIMAL()
        {
            var di = np.diag_indices(4);
            AssertArray(di[0], new Int32[] { 0, 1, 2, 3 });
            AssertArray(di[1], new Int32[] { 0, 1, 2, 3 });
            print(di);

            var a = np.arange(16, dtype: np.Decimal).reshape((4, 4));
            a[di] = 100;

            AssertArray(a, new decimal[,] { { 100, 1, 2, 3 }, { 4, 100, 6, 7 }, { 8, 9, 100, 11 }, { 12, 13, 14, 100 } });
            print(a);

            return;

        }

        [TestMethod]
        public void test_diag_indices_from_1_DECIMAL()
        {
            var a = np.arange(16, dtype: np.Decimal).reshape((4, 4));
            var di = np.diag_indices_from(a);
            AssertArray(di[0], new Int32[] { 0, 1, 2, 3 });
            AssertArray(di[1], new Int32[] { 0, 1, 2, 3 });
            print(di);
        }

        #endregion

        #region from StrideTricksTests

        [TestMethod]
        public void test_broadcast_1_DECIMAL()
        {
            var x = np.array(new decimal[,] { { 11 }, { 2 }, { 3 } });
            var y = np.array(new decimal[] { 4, 5, 6 });
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
        public void test_broadcast_to_1_DECIMAL()
        {
            var a = np.broadcast_to(5m, (4, 4));
            AssertArray(a, new decimal[,] { { 5, 5, 5, 5 }, { 5, 5, 5, 5 }, { 5, 5, 5, 5 }, { 5, 5, 5, 5 } });
            AssertStrides(a, 0, 0);
            print(a);
            print(a.shape);
            print(a.strides);
            print("*************");


            var b = np.broadcast_to(new decimal[] { 1, 2, 3 }, (3, 3));
            AssertArray(b, new decimal[,] { { 1, 2, 3 }, { 1, 2, 3 }, { 1, 2, 3 } });
            AssertStrides(b, 0, 16);
            print(b);
            print(b.shape);
            print(b.strides);
            print("*************");


        }

        [TestMethod]
        public void test_broadcast_arrays_1_DECIMAL()
        {
            var x = np.array(new decimal[,] { { 1, 2, 3 } });
            var y = np.array(new decimal[,] { { 4 }, { 5 } });
            var z = np.broadcast_arrays(false, new ndarray[] { x, y });

            print(z);

        }

        [TestMethod]
        public void test_as_strided_1_DECIMAL()
        {
            var y = np.zeros((10, 10), np.Decimal);
            AssertStrides(y, 160, 16);
            print(y.strides);

            var n = 1000;
            var a = np.arange(n, dtype: np.Decimal);

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
        public void test_nditer_1_DECIMAL()
        {
            var a = np.arange(0.1, 6.1, dtype: np.Decimal).reshape((2, 3));
            var b = np.array(new decimal[] { 7, 8, 9 });

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
        public void test_ndindex_1_DECIMAL()
        {
            var a = np.arange(0.1, 6.1, dtype: np.Decimal).reshape((2, 3));  // force numpy to be initialized

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
        public void test_ndenumerate_1_DECIMAL()
        {
            var a = np.arange(0.1, 6.1, dtype: np.Decimal).reshape((2, 3));

            foreach (ValueTuple<npy_intp[], object> aa in new ndenumerate(a))
            {
                print(aa.Item1);
                print(aa.Item2);
            }
        }

        #endregion

    }
}
