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


        #endregion

        #region from FromNumericTests


        #endregion

        #region from NumericTests


        #endregion

        #region from NANFunctionsTests


        #endregion

        #region from StatisticsTests


        #endregion

        #region from TwoDimBaseTests


        #endregion

        #region from ShapeBaseTests


        #endregion

        #region from UFUNCTests


        #endregion

        #region from IndexTricksTests


        #endregion

        #region from StrideTricksTests


        #endregion

        #region from IteratorTests


        #endregion

    }
}
