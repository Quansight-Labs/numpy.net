using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumpyLib;

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

            var c = np.logspace(2.0m, 3.0m, num: 4, _base: 2.0m);
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

            ndarray C = 5 * F / 9 - 5 * 32 / 9;
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
            Assert.AreEqual((UInt64)0, (UInt64)y.Sum().GetItem(0));

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

            var y = np.delete(x, new Slice(null), 0).reshape(new shape(8, 3));
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
            AssertArray(indexes, new Int64[] { 0, 1, 2, 5, 6 });

            print("inverse");
            print(inverse);
            AssertArray(inverse, new Int64[] { 0, 1, 2, 0, 2, 3, 4, 3, 3 });

            print("counts");
            print(counts);
            AssertArray(counts, new Int64[] { 2, 1, 2, 3, 1 });
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
            AssertArray(y[0], new Int64[] { 0, 1 });
            AssertArray(y[1], new Int64[] { 2, 1 });

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


        #endregion

        #region from MathematicalFunctionsTests

        #endregion
    }
}
