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
    public class StringOperationsTests : TestBaseClass
    {
        /// <summary>
        /// functions that test numpy support for strings
        /// </summary>
        private int SizeOfString = IntPtr.Size;

        #region from ArrayCreationTests
        [TestMethod]
        public void test_asfarray_STRING()
        {
            var a = np.asfarray(new string[] { "2", "3" });
            AssertArray(a, new double[] { 0, 0 });
            print(a);

            try
            {
                var b = np.asfarray(new string[] { "2", "3" }, dtype: np.Strings);
                AssertArray(a, new double[] { 2, 3 });
                print(a);
                Assert.Fail("This function should have thrown exception");
            }
            catch (Exception ex)
            {

            }


            return;
        }

    

        [TestMethod]
        public void test_copy_1_STRING()
        {
            var x = np.array(new String[] { "1", "2", "3" });
            var y = x;

            var z = np.copy(x);

            // Note that, when we modify x, y changes, but not z:

            x[0] = "10";

            Assert.AreEqual("10", y[0]);

            Assert.AreEqual("1", z[0]);

            return;
        }


        [TestMethod]
        public void test_meshgrid_1_STRING()
        {

            var x = np.array(new string[] { "0", "50", "100" });
            var y = np.array(new string[] { "0", "100" });

            ndarray[] xv = np.meshgrid(new ndarray[] { x });
            AssertArray(xv[0], new string[] { "0", "50", "100" });
            print(xv[0]);

            print("************");

            ndarray[] xyv = np.meshgrid(new ndarray[] { x, y });
            AssertArray(xyv[0], new string[,] { { "0", "50", "100" }, { "0", "50", "100" } });
            AssertArray(xyv[1], new string[,] { { "0", "0", "0" }, { "100", "100", "100" } });

            print(xyv[0]);
            print(xyv[1]);

            print("************");

            xyv = np.meshgrid(new ndarray[] { x, y }, sparse: true);
            AssertArray(xyv[0], new string[,] { { "0", "50", "100" } });
            AssertArray(xyv[1], new string[,] { { "0" }, { "100" } });

            print(xyv[0]);
            print(xyv[1]);

            print("************");

            x = np.array(new string[] { "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4" });
            y = np.array(new string[] { "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4" });
            xyv = np.meshgrid(new ndarray[] { x, y }, sparse: true);

            AssertArray(xyv[0], new string[,] { { "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4" } });
            AssertArray(xyv[1], new string[,] { { "-5" }, { "-4" }, { "-3" }, { "-2" }, { "-1" }, { "0" }, { "1" }, { "2" }, { "3" }, { "4" } });

            print(xyv[0]);
            print(xyv[1]);

            print("************");


        }

        [TestMethod]
        public void test_OneDimensionalArray_STRING()
        {
            string[] l = new string[] { "12", "13", "100", "36" };
            print("Original List:", l);
            var a = np.array(l);
            print("One-dimensional numpy array: ", a);
            print(a.shape);
            print(a.strides);

            AssertArray(a, l);
            AssertShape(a, 4);
            AssertStrides(a, SizeOfString);
        }

        [TestMethod]
        public void test_reverse_array_STRING()
        {
            var x = np.array(new string[] { "-5A", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4H" });
            print("Original array:");
            print(x);
            print("Reverse array:");
            //x = (ndarray)x[new Slice(null, null, -1)];
            x = (ndarray)x["::-1"];
            print(x);

            AssertArray(x, new string[] { "4H", "3", "2", "1", "0", "-1", "-2", "-3", "-4", "-5A" });
            AssertShape(x, 10);
            AssertStrides(x, -SizeOfString);

            var y = x + 100;
            print(y);

            var z = x.reshape((5, -1));
            print(z);
        }

        [TestMethod]
        public void test_checkerboard_1_STRING()
        {
            print("Checkerboard pattern:");
            var x = np.full((8, 8), "X", dtype: np.Strings);
            x["1::2", "::2"] = "Y";
            x["::2", "1::2"] = "Y";
            print(x);

            var ExpectedData = new string[8, 8]
            {
                 { "X", "Y", "X", "Y", "X", "Y", "X", "Y" },
                 { "Y", "X", "Y", "X", "Y", "X", "Y", "X" },
                 { "X", "Y", "X", "Y", "X", "Y", "X", "Y" },
                 { "Y", "X", "Y", "X", "Y", "X", "Y", "X" },
                 { "X", "Y", "X", "Y", "X", "Y", "X", "Y" },
                 { "Y", "X", "Y", "X", "Y", "X", "Y", "X" },
                 { "X", "Y", "X", "Y", "X", "Y", "X", "Y" },
                 { "Y", "X", "Y", "X", "Y", "X", "Y", "X" },
            };

            AssertArray(x, ExpectedData);
            AssertShape(x, 8, 8);
            AssertStrides(x, SizeOfString * 8, SizeOfString);

        }


        [TestMethod]
        public void test_ArrayStats_1_STRING()
        {
            var x = np.array(new string[] { "1", "2", "3" }, dtype: np.Strings);
            print("Size of the array: ", x.size);
            print("Length of one array element in bytes: ", x.ItemSize);
            print("Total bytes consumed by the elements of the array: ", x.nbytes);

            Assert.AreEqual(3, x.size);
            Assert.AreEqual(SizeOfString, x.ItemSize);
            Assert.AreEqual(SizeOfString * 3, x.nbytes);

        }

        [TestMethod]
        public void test_ndarray_flatten_STRING()
        {
            var x = np.array(new string[] { "A", "B", "C", "D" }, dtype: np.Strings).reshape(new shape(2, 2));
            var y = x.flatten();
            print(x);
            print(y);
            AssertArray(y, new string[] { "A", "B", "C", "D" });

            y = x.flatten(order: NPY_ORDER.NPY_FORTRANORDER);
            print(y);
            AssertArray(y, new string[] { "A", "C", "B", "D" });

            y = x.flatten(order: NPY_ORDER.NPY_KORDER);
            print(y);
            AssertArray(y, new string[] { "A", "B", "C", "D" });

        }

        [TestMethod]
        public void test_ndarray_byteswap_STRING()
        {
            var x = np.array(new string[] { "A", "B", "C", "D" }, dtype: np.Strings);
            print(x);
            var y = x.byteswap(true);
            print(y);

            // strings can't be swapped.  Data should be unchanged
            AssertArray(y, x.AsStringArray());

            y = x.byteswap(false);
            print(y);

            // strings can't be swapped.  Data should be unchanged
            AssertArray(y, x.AsStringArray());

        }

        [TestMethod]
        public void test_ndarray_create_expected_exception_STRING()
        {
            try
            {
                var x = np.arange(256 + 32, 256 + 64, dtype: np.Strings);
                Assert.Fail("This should have thrown an exception");
            }
            catch (Exception ex)
            {

            }

            try
            {
                var x = np.geomspace(256 + 32, 256 + 64, dtype: np.Strings);
                Assert.Fail("This should have thrown an exception");
            }
            catch (Exception ex)
            {

            }

            try
            {
                var x = np.logspace(256 + 32, 256 + 64, dtype: np.Strings);
                Assert.Fail("This should have thrown an exception");
            }
            catch (Exception ex)
            {

            }
        }

        [TestMethod]
        public void test_ndarray_view_STRING()
        {
            var x = np.arange(256 + 32, 256 + 64, dtype: np.Int32).astype(np.Strings);
            print(x);
            print(x.shape);
            print(x.Dtype);

            AssertArray(x, asstring( new Int32[] { 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
                                         299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
                                         310, 311, 312, 313, 314, 315, 316, 317, 318, 319}));

            // strings can't be mapped by something besides another strings
            var y = x.view(np.UInt64);
            Assert.AreEqual((UInt64)0, (UInt64)y.Sum().GetItem(0));

            y = x.view(np.Strings);
            AssertArray(y, y.AsStringArray());

            y[5] = "1000";

            AssertArray(x, asstring( new Int32[] { 288, 289, 290, 291, 292, 1000, 294, 295, 296, 297, 298,
                                         299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
                                         310, 311, 312, 313, 314, 315, 316, 317, 318, 319}));

        }


        [TestMethod]
        public void test_ndarray_delete1_STRING()
        {
            var x = np.arange(0, 32, dtype: np.Int32).reshape(new shape(8, 4)).astype(np.Strings);

            print("X");
            print(x);
            print(x.shape);

            var ExpectedDataX = asstring( new Int32[8, 4]
            {
                    { 0, 1, 2, 3},
                    { 4, 5, 6, 7},
                    { 8, 9, 10, 11 },
                    { 12, 13, 14, 15},
                    { 16, 17, 18, 19},
                    { 20, 21, 22, 23},
                    { 24, 25, 26, 27},
                    { 28, 29, 30, 31},
            });

            AssertArray(x, ExpectedDataX);
            AssertShape(x, 8, 4);

            var y = np.delete(x, new Slice(null), 0).reshape(new shape(8, 3));
            y[1] = "99";
            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = asstring(new Int32[8, 3]
            {
                    { 1, 2, 3},
                    { 99, 99, 99},
                    { 9, 10, 11 },
                    { 13, 14, 15},
                    { 17, 18, 19},
                    { 21, 22, 23},
                    { 25, 26, 27},
                    { 29, 30, 31},
            });

            AssertArray(y, ExpectedDataY);
            AssertShape(y, 8, 3);

            print("X");
            print(x);


            AssertArray(x, ExpectedDataX);
            AssertShape(x, 8, 4);
        }

        [TestMethod]
        public void test_ndarray_delete2_STRING()
        {
            var x = np.arange(0, 32, dtype: np.Int32).astype(np.Strings);

            print("X");
            print(x);
            print(x.shape);

            var ExpectedDataX = asstring( new Int32[] {0,  1,  2,  3,  4,  5,  6,  7,
                                             8,  9,  10, 11, 12, 13, 14, 15,
                                             16, 17, 18, 19, 20, 21, 22, 23,
                                             24, 25, 26, 27, 28, 29, 30, 31 });
            AssertArray(x, ExpectedDataX);
            AssertShape(x, 32);

            var y = np.delete(x, 1, 0);
            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = asstring( new Int32[] {0,  2,  3,  4,  5,  6,  7,
                                             8,  9,  10, 11, 12, 13, 14, 15,
                                             16, 17, 18, 19, 20, 21, 22, 23,
                                             24, 25, 26, 27, 28, 29, 30, 31 });
            AssertArray(y, ExpectedDataY);
            AssertShape(y, 31);

            print("X");
            print(x);

            AssertArray(x, ExpectedDataX);
        }

        [TestMethod]
        public void test_ndarray_delete3_STRING()
        {
            var x = np.arange(0, 32, dtype: np.Int32).reshape(new shape(8, 4)).astype(np.Strings);

            print("X");
            print(x);
            print(x.shape);

            var ExpectedDataX = asstring(new Int32[8, 4]
            {
                { 0, 1, 2, 3},
                { 4, 5, 6, 7},
                { 8, 9, 10, 11 },
                { 12, 13, 14, 15},
                { 16, 17, 18, 19},
                { 20, 21, 22, 23},
                { 24, 25, 26, 27},
                { 28, 29, 30, 31},
            });

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

            var ExpectedDataY = asstring(new Int32[8, 3]
            {
                { 1, 2, 3},
                { 5, 6, 7},
                { 9, 10, 11 },
                { 13, 14, 15},
                { 17, 18, 19},
                { 21, 22, 23},
                { 25, 26, 27},
                { 29, 30, 31},
            });

            AssertArray(y, ExpectedDataY);
            AssertShape(y, 8, 3);


            print("X");
            print(x);

            AssertArray(x, ExpectedDataX);
            AssertShape(x, 8, 4);
        }

        [TestMethod]
        public void test_ndarray_unique_1_STRING()
        {
            var x = np.array(new Int32[] { 1, 2, 3, 1, 3, 4, 5, 4, 4 }).astype(np.Strings);

            print("X");
            print(x);

            var result = np.unique(x, return_counts: true, return_index: true, return_inverse: true);
            var uvalues = result.data;
            var indexes = result.indices;
            var inverse = result.inverse;
            var counts = result.counts;

            print("uvalues");
            print(uvalues);
            AssertArray(uvalues, asstring(new Int32[] { 1, 2, 3, 4, 5 }));

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
        public void test_ndarray_where_1_STRING()
        {
            var x = np.array(new Int32[] { 1, 2, 3, 1, 3, 4, 5, 4, 4 }).reshape(new shape(3, 3)).astype(np.Strings);

            print("X");
            print(x);

            ndarray[] y = (ndarray[])np.where(x.Equals("3"));
            print("Y");
            print(y);
        }

        [TestMethod]
        public void test_ndarray_where_2_STRING()
        {
            var x = np.array(new Int32[] { 1, 2, 3, 1, 3, 4, 5, 4, 4 }).reshape(new shape(3, 3)).astype(np.Strings);

            print("X");
            print(x);

            ndarray[] y = (ndarray[])np.where(x.Equals("3"));

            Assert.AreEqual(2, y.Length);
            AssertArray(y[0], new npy_intp[] { 0, 1 });
            AssertArray(y[1], new npy_intp[] { 2, 1 });

            var z = x.SliceMe(y) as ndarray;
            print("Z");
            print(z);
            AssertArray(z, asstring(new int[] { 3, 3 }));

            ///////////////////////

            y = (ndarray[])np.where(x.NotEquals("3"));

            Assert.AreEqual(2, y.Length);
            AssertArray(y[0], new npy_intp[] { 0,0,1,1,2,2,2 });
            AssertArray(y[1], new npy_intp[] { 0,1,0,2, 0,1,2 });

            z = x.SliceMe(y) as ndarray;
            print("Z");
            print(z);
            AssertArray(z, asstring(new int[] { 1,2,1,4,5,4,4 }));
        }

        [TestMethod]
        public void test_ndarray_where_3_STRING()
        {
            var x = np.arange(0, 100, dtype: np.Int32).reshape(new shape(-1, 10)).astype(np.Strings);

            ndarray[] y = (ndarray[])np.where(x < "13");
            var z = x[y] as ndarray;
            var ExpectedDataZ = asstring(new int[]
            {
                0,  1,  10,  11,  12
            });

            AssertArray(z, ExpectedDataZ);

            ////////////////////

            y = (ndarray[])np.where(x <= "13");
            z = x[y] as ndarray;
            ExpectedDataZ = asstring(new int[]
            {
                0,  1,  10,  11,  12, 13
            });

            AssertArray(z, ExpectedDataZ);

            ///////////////////

            y = (ndarray[])np.where(x > "93");
            z = x[y] as ndarray;
            ExpectedDataZ = asstring(new int[]
            {
                94,  95,  96,  97, 98, 99
            });

            AssertArray(z, ExpectedDataZ);

            ///////////////////

            y = (ndarray[])np.where(x >= "93");
            z = x[y] as ndarray;
            ExpectedDataZ = asstring(new int[]
            {
                93,  94,  95,  96,  97, 98, 99
            });

            AssertArray(z, ExpectedDataZ);

        }

        [TestMethod]
        public void test_ndarray_where_4_STRING()
        {
            var x = np.arange(0, 30, dtype: np.Int32).astype(np.Strings);

            var y = np.where(x % 7 == 0);
            print("Y");
            print(y);

            var z = x[y] as ndarray;
            print(z);
            //var m = np.mean(z);
            //print("M");
            //Assert.AreEqual(1499998.5, m.GetItem(0));
            //print(m);

            return;
        }

   
        [TestMethod]
        public void test_ndarray_where_5_STRING()
        {
            var a1 = np.arange(10, dtype: np.Int32).astype(np.Strings);
            var a2 = np.arange(10, 100, 10, dtype: np.Int32).astype(np.Strings);

            var b = np.where(a1 < 5, a1, a2) as ndarray;
            AssertArray(b, asstring(new int[] { 0, 1, 2, 3, 4, 60, 70, 80, 90, 10 }));
            print(b);

            a1 = np.array(new Int32[,] { { 0, 1, 2 }, { 0, 2, 4 }, { 0, 3, 6 } }).astype(np.Strings);
            b = np.where(a1 < 4, a1, -1) as ndarray;  // -1 is broadcast
            AssertArray(b, asstring(new Int32[,] { { 0, 1, 2 }, { 0, 2, -1 }, { 0, 3, -1 } }));
            print(b);

            var c = np.where(new bool[,] { { true, false }, { true, true } },
                                    asstring(new Int32[,] { { 1, 2 }, { 3, 4 } }),
                                    asstring(new Int32[,] { { 9, 8 }, { 7, 6 } })) as ndarray;

            AssertArray(c, asstring(new Int32[,] { { 1, 8 }, { 3, 4 } }));

            print(c);

            return;
        }

        [TestMethod]
        public void test_arange_slice_1_STRING()
        {
            var a = np.arange(0, 1024, dtype: np.Int32).reshape(new shape(2, 4, -1)).astype(np.Strings);

            print("A");
            // print(a);
            print(a.shape);
            print(a.strides);

            AssertShape(a, 2, 4, 128);

            var b = (ndarray)a[":", ":", 122];
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = asstring(new Int32[2, 4]
            {
                { 122, 250, 378, 506},
                { 634, 762, 890, 1018 },
            });

            AssertArray(b, ExpectedDataB);
            AssertShape(b, 2, 4);
            //AssertStrides(b, 8192, 2048);

            var c = (ndarray)a.A(":", ":", new Int64[] { 122 });
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            var ExpectedDataC = asstring(new Int32[2, 4, 1]
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

            });

            AssertArray(c, ExpectedDataC);
            AssertShape(c, 2, 4, 1);
            //AssertStrides(c, 64, 16, 128);

            var d = (ndarray)a.A(":", ":", new Int64[] { 122, 123 });
            print("D");
            print(d);
            print(d.shape);
            print(d.strides);

            var ExpectedDataD = asstring( new Int32[2, 4, 2]
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

            });

            AssertArray(d, ExpectedDataD);
            AssertShape(d, 2, 4, 2);
            //AssertStrides(d, 64, 16, 128);

        }

        [TestMethod]
        public void test_arange_slice_2A_STRING()
        {
            var a = np.arange(0, 32, dtype: np.Int32).reshape(new shape(2, 4, -1)).astype(np.Strings);

            print("A");
            // print(a);
            print(a.shape);
            print(a.strides);

            var b = (ndarray)a[":", ":", np.where(a > "4")];
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = asstring(new Int32[,,,]
                { { { { 0, 0, 0, 0, 0 },
                      { 1, 1, 1, 2, 2 },
                      { 1, 2, 3, 0, 1 } },
                    { { 4, 4, 4, 4, 4 },
                      { 5, 5, 5, 6, 6 },
                      { 5, 6, 7, 4, 5 } },
                    { { 8, 8, 8, 8, 8 },
                      { 9, 9, 9, 10, 10 },
                      { 9, 10, 11, 8, 9 } },
                    { { 12, 12, 12, 12, 12 },
                      { 13, 13, 13, 14, 14 },
                      { 13, 14, 15, 12, 13 } } },
                  { { { 16, 16, 16, 16, 16 },
                      { 17, 17, 17, 18, 18 },
                      { 17, 18, 19, 16, 17 } },
                    { { 20, 20, 20, 20, 20 },
                      { 21, 21, 21, 22, 22 },
                      { 21, 22, 23, 20, 21 } },
                    { { 24, 24, 24, 24, 24 },
                      { 25, 25, 25, 26, 26 },
                      { 25, 26, 27, 24, 25 } },
                    { { 28, 28, 28, 28, 28 },
                      { 29, 29, 29, 30, 30 },
                      { 29, 30, 31, 28, 29 } } } });

            AssertArray(b, ExpectedDataB);
            //AssertStrides(b, 64, 16, 1408, 128);
        }

        [TestMethod]
        public void test_insert_1_STRING()
        {
            Int32[,] TestData = new Int32[,] { { 1, 1 }, { 2, 2 }, { 3, 3 } };
            ndarray a = np.array(TestData, dtype: np.Int32).astype(np.Strings);

            ndarray b = np.insert(a, 1, 5);
            ndarray c = np.insert(a, 0, asstring(new Int32[] { 999, 100, 101 }));

            print(a);
            print(a.shape);

            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, asstring(new int[] { 1, 5, 1, 2, 2, 3, 3 }));
            AssertShape(b, 7);
            //AssertStrides(b, 16);

            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            AssertArray(c, asstring(new int[] { 999, 100, 101, 1, 1, 2, 2, 3, 3 }));
            AssertShape(c, 9);
            //AssertStrides(c, 16);
        }

        [TestMethod]
        public void test_insert_2_STRING()
        {
            int[] TestData1 = new int[] { 1, 1, 2, 2, 3, 3 };
            int[] TestData2 = new int[] { 90, 91, 92, 92, 93, 93 };

            ndarray a = np.array(TestData1, dtype: np.Int32).astype(np.Strings);
            ndarray b = np.array(TestData2, dtype: np.Int32).astype(np.Strings);

            ndarray c = np.insert(a, new Slice(null), b);

            print(a);
            print(a.shape);

            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, asstring(new Int32[] { 90, 91, 92, 92, 93, 93 }));
            AssertShape(b, 6);
            //AssertStrides(b, 4);

            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            AssertArray(c, asstring(new Int32[] { 90, 1, 91, 1, 92, 2, 92, 2, 93, 3, 93, 3 }));
            AssertShape(c, 12);
            //AssertStrides(c, 4);

        }

        [TestMethod]
        public void test_append_1_STRING()
        {
            int[] TestData = new int[] { 1, 1, 2, 2, 3, 3 };
            ndarray a = np.array(TestData, dtype: np.Int32).astype(np.Strings);
            ndarray b = np.append(a, 1);

            print(a);
            print(a.shape);

            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, asstring(new Int32[] { 1, 1, 2, 2, 3, 3, 1 }));
            AssertShape(b, 7);
            //AssertStrides(b, 16);
        }

        [TestMethod]
        public void test_append_3_STRING()
        {
            string[] TestData1 = new string[] { "1", "1", "2", "2", "3", "3" };
            string[] TestData2 = new string[] { "4", "4", "5", "5", "6", "6" };
            ndarray a = np.array(TestData1, dtype: np.Strings);
            ndarray b = np.array(TestData2, dtype: np.Strings);


            ndarray c = np.append(a, b);

            print(a);
            print(a.shape);

            print(b);
            print(b.shape);

            print(c);
            print(c.shape);
            print(c.strides);

            AssertArray(c, new string[] { "1", "1", "2", "2", "3", "3", "4", "4", "5", "5", "6", "6" });
            AssertShape(c, 12);
            //AssertStrides(c, 16);
        }

        [TestMethod]
        public void test_append_4_STRING()
        {
            string[] TestData1 = new string[] { "1", "1", "2", "2", "3", "3" };
            string[] TestData2 = new string[] { "4", "4", "5", "5", "6", "6" };
            ndarray a = np.array(TestData1, dtype: np.Strings).reshape((2, -1));
            ndarray b = np.array(TestData2, dtype: np.Strings).reshape((2, -1));

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

            var ExpectedDataC = new string[,]
            {
                { "1", "1", "2", "4", "4", "5" },
                { "2", "3", "3", "5", "6", "6" },
            };

            AssertArray(c, ExpectedDataC);
            AssertShape(c, 2, 6);
            //AssertStrides(c, 24, 4); 

        }

        [TestMethod]
        public void test_flat_2_STRING()
        {
            var x = np.arange(1, 7, dtype: np.Int32).reshape((2, 3)).astype(np.Strings);

            print(x);

            Assert.AreEqual("4", x.Flat[3]);
            print(x.Flat[3]);

            print(x.T);
            Assert.AreEqual("5", x.T.Flat[3]);
            print(x.T.Flat[3]);

            x.flat = 3;
            AssertArray(x, asstring(new Int32[,] { { 3, 3, 3 }, { 3, 3, 3 } }));
            print(x);

            x.Flat[new int[] { 1, 4 }] = "1";
            AssertArray(x, asstring(new Int32[,] { { 3, 1, 3 }, { 3, 1, 3 } }));
            print(x);
        }

        [TestMethod]
        public void test_intersect1d_1_STRING()
        {
            ndarray a = np.array(new Int32[] { 1, 3, 4, 3 }).astype(np.Strings);
            ndarray b = np.array(new Int32[] { 3, 1, 2, 1 }).astype(np.Strings);

            ndarray c = np.intersect1d(a, b);
            print(c);

            AssertArray(c, new string[] { "1", "3" });
            AssertShape(c, 2);
            //AssertStrides(c, 16);

        }

        [TestMethod]
        public void test_setxor1d_1_STRING()
        {
            ndarray a = np.array(new Int32[] { 1, 2, 3, 2, 4 }).astype(np.Strings);
            ndarray b = np.array(new Int32[] { 2, 3, 5, 7, 5 }).astype(np.Strings);

            ndarray c = np.setxor1d(a, b);
            print(c);

            AssertArray(c, new string[] { "1", "4", "5", "7" });
            AssertShape(c, 4);
            //AssertStrides(c, 16);
        }

        [TestMethod]
        public void test_in1d_1_STRING()
        {
            ndarray test = np.array(new Int32[] { 0, 1, 2, 5, 0 }).astype(np.Strings);
            ndarray states = np.array(new Int32[] { 0, 2 }).astype(np.Strings);

            ndarray mask = np.in1d(test, states);
            print(mask);
            print(test[mask]);

            AssertArray(mask, new bool[] { true, false, true, false, true });
            AssertShape(mask, 5);
            //AssertStrides(mask, 1);

            ndarray a = test[mask] as ndarray;
            AssertArray(a, new string[] { "0", "2", "0" });
            AssertShape(a, 3);
            //AssertStrides(a, 16);

            mask = np.in1d(test, states, invert: true);
            print(mask);
            print(test[mask]);

            AssertArray(mask, new bool[] { false, true, false, true, false });
            AssertShape(mask, 5);
            //AssertStrides(mask, 1);

            ndarray b = test[mask] as ndarray;
            AssertArray(b, new string[] { "1", "5" });
            AssertShape(b, 2);
            //AssertStrides(b, 16);

        }

        [TestMethod]
        public void test_isin_1_STRING()
        {
            ndarray element = np.arange(0, 8, 2, dtype: np.Int32).reshape(new shape(2, 2)).astype(np.Strings);
            print(element);

            ndarray test_elements = np.array(new string[] { "1", "2", "4", "8" });
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
            //AssertStrides(mask, 2, 1);

            AssertArray(a, new string[] { "2", "4" });
            AssertShape(a, 2);
            //AssertStrides(a, 16);

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
            //AssertStrides(mask, 2, 1);

            AssertArray(a, new string[] { "0", "6" });
            AssertShape(a, 2);
            //AssertStrides(a, 16);
        }

        [TestMethod]
        public void test_union1d_1_STRING()
        {
            ndarray a1 = np.array(new string[] { "-1", "0", "1" });
            ndarray a2 = np.array(new string[] { "-2", "0", "2" });

            ndarray a = np.union1d(a1, a2);
            print(a);

            AssertArray(a, new string[] { "0", "1", "-1", "2", "-2" });
            AssertShape(a, 5);
            //AssertStrides(a, 16);
        }

        [TestMethod]
        public void test_Ellipsis_indexing_1_STRING()
        {
            var a = np.array(new Int32[] { 10, 7, 4, 3, 2, 1 }).astype(np.Strings);

            var b = a.A("...", -1);
            Assert.AreEqual("1", b.GetItem(0));
            print(b);
            print("********");


            a = np.array(new Int32[,] { { 10, 7, 4 }, { 3, 2, 1 } }).astype(np.Strings);
            var c = a.A("...", -1);
            AssertArray(c, new string[] { "4", "1" });
            print(c);
            print("********");

            var TestData = new Int32[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
            a = np.array(TestData, dtype: np.Int32).reshape((1, 3, 2, -1, 1)).astype(np.Strings);
            var d = a["...", -1] as ndarray;
            AssertArray(d, asstring(new Int32[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } } }));
            print(d);
            print("********");

            var e = a[0, "...", -1] as ndarray;
            AssertArray(e, asstring(new Int32[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } }));
            print(e);
            print("********");

            var f = a[0, ":", ":", ":", -1] as ndarray;
            AssertArray(f, asstring(new Int32[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } }, { { 9, 10 }, { 11, 12 } } }));
            print(f);
            print("********");

            var g = a.A(0, 1, "...", -1);
            AssertArray(g, asstring(new Int32[,] { { 5, 6 }, { 7, 8 } }));
            print(g);
            print("********");

            var h = a.A(0, 2, 1, "...", -1);
            AssertArray(h, asstring(new Int32[] { 11, 12 }));
            print(h);
            print("********");

            var i = a[":", 2, 1, 1, "..."] as ndarray;
            AssertArray(i, asstring(new Int32[,] { { 12 } }));
            print(i);
        }

        [TestMethod]
        public void test_concatenate_1_STRING()
        {

            var a = np.array(new Int32[,] { { 1, 2 }, { 3, 4 } }).astype(np.Strings);
            var b = np.array(new Int32[,] { { 5, 6 } }).astype(np.Strings);
            var c = np.concatenate((a, b), axis: 0);
            AssertArray(c, asstring(new Int32[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }));
            print(c);

            var d = np.concatenate((a, b.T), axis: 1);
            AssertArray(d, asstring(new Int32[,] { { 1, 2, 5 }, { 3, 4, 6 } }));
            print(d);

            var e = np.concatenate((a, b), axis: null);
            AssertArray(e, asstring(new Int32[] { 1, 2, 3, 4, 5, 6 }));
            print(e);

            var f = np.concatenate((np.eye(2, dtype: np.Int32).astype(np.Strings), np.ones((2, 2), dtype: np.Int32).astype(np.Strings)), axis: 0);
            AssertArray(f, asstring(new Int32[,] { { 1, 0 }, { 0, 1 }, { 1, 1 }, { 1, 1 }, }));
            print(f);

            var g = np.concatenate((np.eye(2, dtype: np.Int32).astype(np.Strings), np.ones((2, 2), dtype: np.Int32).astype(np.Strings)), axis: 1);
            AssertArray(g, asstring(new Int32[,] { { 1, 0, 1, 1 }, { 0, 1, 1, 1 } }));
            print(g);
        }

        [TestMethod]
        public void test_concatenate_3_STRING()
        {

            var a = np.array(new Int32[,,,] { { { { 1, 2 }, { 3, 4 }, { 5, 6 } } } }).astype(np.Strings);
            var c = np.concatenate(a, axis: -1);
            AssertArray(c, asstring(new Int32[,,] { { { 1, 2 }, { 3, 4 }, { 5, 6 } } }));
            print(c);

            var d = np.concatenate(a, axis: -2);
            AssertArray(d, asstring(new Int32[,,] { { { 1, 2 }, { 3, 4 }, { 5, 6 } } }));
            print(d);

            c = np.concatenate((a, a, a), axis: -1);
            AssertArray(c, asstring(new Int32[,,,] { { { { 1, 2, 1, 2, 1, 2 }, { 3, 4, 3, 4, 3, 4 }, { 5, 6, 5, 6, 5, 6 } } } }));
            print(c);

            d = np.concatenate((a, a, a), axis: -2);
            AssertArray(d, asstring(new Int32[,,,] { { { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 1, 2 }, { 3, 4 }, { 5, 6 }, { 1, 2 }, { 3, 4 }, { 5, 6 } } } }));
            print(d);


        }


        [TestMethod]
        public void test_multi_index_selection_STRING()
        {
            var x = np.arange(10).astype(np.Strings);

            var y = x.reshape(new shape(2, 5));
            print(y);
            Assert.AreEqual("3", y[0, 3]);
            Assert.AreEqual("8", y[1, 3]);

            x = np.arange(20).astype(np.Strings);
            y = x.reshape(new shape(2, 2, 5));
            print(y);
            Assert.AreEqual("3", y[0, 0, 3]);
            Assert.AreEqual("8", y[0, 1, 3]);

            Assert.AreEqual("13", y[1, 0, 3]);
            Assert.AreEqual("18", y[1, 1, 3]);

        }

        [TestMethod]
        public void test_multi_index_setting_STRING()
        {
            var x = np.arange(10, dtype: np.Int32).astype(np.Strings);

            var y = x.reshape(new shape(2, 5));

            y[0, 3] = 55;
            y[1, 3] = 66;

            Assert.AreEqual("55", (string)y[0, 3]);
            Assert.AreEqual("66", (string)y[1, 3]);

            x = np.arange(20, dtype: np.Int32).astype(np.Strings);
            y = x.reshape(new shape(2, 2, 5));

            y[1, 0, 3] = 55;
            y[1, 1, 3] = 66;

            Assert.AreEqual("55", (string)y[1, 0, 3]);
            Assert.AreEqual("66", (string)y[1, 1, 3]);

        }


        #endregion

        #region from NumericalOperationsTests


        [TestMethod]
        public void test_min_STRING()
        {
            var TestData = new string[] { "AA", "BB", "CC", "aa", "bb", "cc" };
            var x = np.array(TestData);
            string y = (string)np.min(x);

            print(x);
            print(y);

            Assert.AreEqual("aa", y);
        }

        [TestMethod]
        public void test_max_STRING()
        {
            var TestData = new string[] { "AA", "BB", "CC", "aa", "bb", "cc" };
            var x = np.array(TestData);
            string y = (string)np.max(x);

            print(x);
            print(y);

            Assert.AreEqual("CC", y);
        }


        [TestMethod]
        public void test_setdiff1d_STRING()
        {
            var TestDataA = new string[] { "1", "2", "3", "2", "4", };
            var TestDataB = new string[] { "3", "4", "5", "6" };

            var a = np.array(TestDataA);
            var b = np.array(TestDataB);
            ndarray c = np.setdiff1d(a, b);

            print(a);
            print(b);
            print(c);

            AssertArray(c, new string[] { "1", "2" });

        }

        [TestMethod]
        public void test_setdiff1d_2_STRING()
        {
            var TestDataB = new string[] { "3", "4", "5", "6" };

            var a = np.arange(1, 39).reshape(new shape(2, -1)).astype(np.Strings);
            var b = np.array(TestDataB);
            ndarray c = np.setdiff1d(a, b);

            print(a);
            print(b);
            print(c);

            AssertArray(c, asstring(new Int32[] { 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 38, 7, 8, 9 }));

        }

        [TestMethod]
        public void test_rot90_1_STRING()
        {
            ndarray m = np.array(new Int32[,] { { 1, 2 }, { 3, 4 } }).astype(np.Strings);
            print(m);
            print("************");

            ndarray n = np.rot90(m);
            print(n);
            AssertArray(n, asstring(new Int32[,] { { 2, 4 }, { 1, 3 }, }));
            print("************");

            n = np.rot90(m, 2);
            print(n);
            AssertArray(n, asstring(new Int32[,] { { 4, 3 }, { 2, 1 }, }));
            print("************");

            m = np.arange(8, dtype: np.Int32).reshape(new shape(2, 2, 2)).astype(np.Strings);
            n = np.rot90(m, 1, new int[] { 1, 2 });
            print(n);
            AssertArray(n, asstring(new Int32[,,] { { { 1, 3 }, { 0, 2 } }, { { 5, 7 }, { 4, 6 } } }));

        }

        [TestMethod]
        public void test_flip_1_STRING()
        {
            ndarray A = np.arange(8, dtype: np.Int32).reshape(new shape(2, 2, 2)).astype(np.Strings);
            ndarray B = np.flip(A, 0);
            print(A);
            print("************");
            print(B);
            AssertArray(B, asstring(new Int32[,,] { { { 4, 5 }, { 6, 7 } }, { { 0, 1 }, { 2, 3 } } }));

            print("************");
            ndarray C = np.flip(A, 1);
            print(C);
            AssertArray(C, asstring(new Int32[,,] { { { 2, 3 }, { 0, 1 } }, { { 6, 7 }, { 4, 5 } } }));
            print("************");

        }

        [TestMethod]
        public void test_trim_zeros_1_STRING()
        {
            ndarray a = np.array(new string[] { null, null, null, "1", "2", "3", "0", "2", "1", null }, np.Strings);

            var b = np.trim_zeros(a);
            print(b);
            AssertArray(b, new string[] { "1", "2", "3", "0", "2", "1" });

            var c = np.trim_zeros(a, "b");
            print(c);
            AssertArray(c, new string[] { null, null, null, "1", "2", "3", "0", "2", "1" });
        }

        [TestMethod]
        public void test_logical_and_1_STRING()
        {
            var x = np.arange(5).astype(np.Strings);
            var c = np.logical_and(x > 1, x < 4);
            AssertArray(c, new bool[] { false, false, true, true, false });
            print(c);

            var y = np.arange(6).reshape((2, 3)).astype(np.Strings);
            var d = np.logical_and(y > 1, y < 4);
            AssertArray(d, new bool[,] { { false, false, true }, { true, false, false } });
            print(d);
        }

        [TestMethod]
        public void test_logical_or_1_STRING()
        {
            var x = np.arange(5).astype(np.Strings);
            var c = np.logical_or(x < 1, x > 3);
            AssertArray(c, new bool[] { true, false, false, false, true });
            print(c);

            var y = np.arange(6).reshape((2, 3)).astype(np.Strings);
            var d = np.logical_or(y < 1, y > 3);
            AssertArray(d, new bool[,] { { true, false, false }, { false, true, true } });
            print(d);
        }

        [TestMethod]
        public void test_logical_xor_1_STRING()
        {
            var x = np.arange(5).astype(np.Strings);
            var c = np.logical_xor(x < 1, x > 3);
            AssertArray(c, new bool[] { true, false, false, false, true });
            print(c);

            var y = np.arange(6).reshape((2, 3)).astype(np.Strings);
            var d = np.logical_xor(y < 1, y > 3);
            AssertArray(d, new bool[,] { { true, false, false }, { false, true, true } });
            print(d);

            var e = np.logical_xor(0, np.eye(2).astype(np.Strings));
            AssertArray(e, new bool[,] { { true, false }, { false, true } });
        }

        [TestMethod]
        public void test_logical_not_1_STRING()
        {
            var x = np.arange(5).astype(np.Strings);
            var c = np.logical_not(x < 3);
            AssertArray(c, new bool[] { false, false, false, true, true });
            print(c);

            ///////////////

            x = np.array(new string[] { "AA", "BB", "CC", "DD", "EE" });
            c = np.logical_not(x < "CC");
            AssertArray(c, new bool[] { false, false, true, true, true });
            print(c);

            ///////////////

            x = np.array(new string[] { "aa", "bb", "cc", "dd", "ee" });
            c = np.logical_not(x < "cc");
            AssertArray(c, new bool[] { false, false, true, true, true });
            print(c);

        }


        [TestMethod]
        public void test_copyto_1_STRING()
        {
            var a = np.zeros((2, 5), dtype: np.Strings);
            var b = new int[] { 11, 22, 33, 44, 55 };
            np.copyto(a, b);

            AssertShape(a, 2, 5);
            AssertArray(a, asstring(new Int32[,] { { 11, 22, 33, 44, 55 }, { 11, 22, 33, 44, 55 } }));
            print(a);

            a = np.zeros((4, 5), dtype: np.Strings);
            np.copyto(a, 99);
            AssertArray(a, asstring(new Int32[,] { { 99, 99, 99, 99, 99 }, { 99, 99, 99, 99, 99 }, { 99, 99, 99, 99, 99 }, { 99, 99, 99, 99, 99 } }));
            print(a);

            a = np.zeros((10, 5), dtype: np.Strings);
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
        public void test_copyto_2_STRING()
        {
            var a = np.zeros((1, 2, 2, 1, 2), dtype: np.Strings);
            var b = new int[] { 1, 2 };
            np.copyto(a, b);

            AssertArray(a, asstring(new Int32[,,,,] { { { { { 1, 2 } }, { { 1, 2 } } }, { { { 1, 2 } }, { { 1, 2, } } } } }));

        }

        #endregion

        #region from MathematicalFunctionsTests

        private string MathFunctionExceptionPrefix = "Arrays of type";

        [TestMethod]
        public void test_sin_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.sin(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_cos_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.cos(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_tan_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.tan(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_arcsin_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.arcsin(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_arccos_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.arccos(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_arctan_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.arctan(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_hypot_1_STRING()
        {
            try
            {
                var b = np.hypot(np.ones((3, 3), dtype: np.Strings) * 3, np.ones((3, 3), dtype: np.Strings) * 4);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_arctan2_1_STRING()
        {
            try
            {
                var x = np.array(new string[] { "-1", "+1", "+1", "-1" });
                var y = np.array(new string[] { "-1", "-1", "+1", "+1" });
                var z = np.arctan2(y, x);
                print(z);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        #region Hyperbolic functions

        [TestMethod]
        public void test_sinh_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.sinh(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }
        }

        [TestMethod]
        public void test_cosh_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.cosh(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }


        }

        [TestMethod]
        public void test_tanh_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.tanh(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_arcsinh_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.arcsinh(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_arccosh_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.arccosh(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_arctanh_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.arctanh(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        #endregion

        [TestMethod]
        public void test_degrees_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.degrees(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_radians_1_STRING()
        {
            var a = np.arange(0, 10, dtype: np.Int32).astype(np.Strings);
            a = a["::2"] as ndarray;

            try
            {
                var b = np.radians(a);
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_around_1_STRING()
        {
            try
            {
                ndarray a = np.around(np.array(new string[] { "37", "164.2" }));
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_round_1_STRING()
        {
            try
            {
                var a = np.round(np.array(new string[] { "37" }));
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }


        }

        [TestMethod]
        public void test_fix_1_STRING()
        {
            try
            {
                var a = np.fix("3.14");
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }

        }

        [TestMethod]
        public void test_floor_1_STRING()
        {
            string[] TestData = new string[] { "b", "a", "c", "d" };
            var x = np.array(TestData);
            var y = np.floor(x);

            print(x);
            print(y);

            AssertArray(y, TestData);

        }

        [TestMethod]
        public void test_ceil_1_STRING()
        {
            string[] TestData = new string[] { "b", "a", "c", "d" };
            var x = np.array(TestData);
            var y = np.ceil(x);

            print(x);
            print(y);

            AssertArray(y, TestData);

        }

        [TestMethod]
        public void test_trunc_1_STRING()
        {
            var a = np.trunc("3.14");
            Assert.AreEqual("3.14", a.GetItem(0));
            print(a);

            var c = np.trunc(np.array(new string[] { "21", "29f", "-21", "-29" }));
            AssertArray(c, new string[] { "21", "29f", "-21", "-29" });
            print(c);
        }

        [TestMethod]
        public void test_prod_2_STRING()
        {
            ndarray a = np.prod(np.array(new string[] { "XYZ", "ABC" }));
            print(a);
            Assert.AreEqual("XYZ", a.GetItem(0));
            print("*****");

            ndarray b = np.prod(np.array(new string[,] { { "AA", "BB" }, { "CC", "DD" } }));
            print(b);
            Assert.AreEqual("AA", b.GetItem(0));
            print("*****");

            ndarray c = np.prod(np.array(new string[,] { { "AA", "BB" }, { "CC", "DD" } }), axis: 1);
            print(c);
            AssertArray(c, new string[] { "AA", "CC" });
            print("*****");

            ndarray d = np.array(new string[] { "1", "2", "3" }, dtype: np.Strings);
            bool e = np.prod(d).Dtype.TypeNum == NPY_TYPES.NPY_STRING;
            print(e);
            Assert.AreEqual(true, e);
            print("*****");


            try
            {
                a = np.array(new string[,] { { "1", "2" }, { "3", "4" } }, dtype: np.Strings);
                a[1, 1] = "X";

                b = np.prod(a);
            }
            catch
            {
                Assert.Fail("Should not have throw exception");
            }



        }

        [TestMethod]
        public void test_sum_2_STRING()
        {
            string[] TestData = new string[] { "A", "B", "C", "D", "E", "F", "G", "H" };
            var x = np.array(TestData, dtype: np.Strings).reshape(new shape(2, 2, 2));
            x = x * 3;

            var y = np.sum(x, axis: 0);
            print(y);
            AssertArray(y, new string[,] {{ "AE", "BF" }, { "CG", "DH" }});

            print("*****");

            y = np.sum(x, axis: 1);
            print(y);
            AssertArray(y, new string[,] { { "AC", "BD" }, { "EG", "FH" } });

            print("*****");

            y = np.sum(x, axis: 2);
            print(y);
            AssertArray(y, new string[,] { { "AB", "CD" }, { "EF", "GH" } });

            print("*****");

            try
            {
                x[0, 1, 1] = 99;
                y = np.sum(x, axis: 2);
            }
            catch (Exception ex)
            {
                Assert.Fail("This should not have thrown an exception");
            }


        }

        [TestMethod]
        public void test_cumprod_2_STRING()
        {
            ndarray a = np.array(new string[] { "1", "2", "3" });
            ndarray b = np.cumprod(a);          // intermediate results 1, 1*2
                                                // total product 1*2*3 = 6
            print(b);
            AssertArray(b, new string[] { "1", "1", "1" });
            print("*****");

            a = np.array(new String[,] { { "a", "b", "c" }, { "d", "e", "f" } }, dtype: np.Strings);
            ndarray c = np.cumprod(a, dtype: np.Strings); //specify type of output
            print(c);
            AssertArray(c, new string[] { "a", "a", "a", "a", "a", "a" });
            print("*****");

            ndarray d = np.cumprod(a, axis: 0);
            print(d);
            AssertArray(d, new string[,] { { "a", "b", "c" }, { "a", "b", "c" } });
            print("*****");

            ndarray e = np.cumprod(a, axis: 1);
            print(e);
            AssertArray(e, new string[,] { { "a", "a", "a" }, { "d", "d", "d" } });
            print("*****");


            try
            {
                a[0] = 99;
                ndarray f = np.cumprod(a, axis: 1);
            }
            catch (Exception ex)
            {
                Assert.Fail("Should not have thrown an exception");
            }

        }

        [TestMethod]
        public void test_cumsum_3_STRING()
        {
            ndarray a = np.array(new string[] { "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L" }).reshape(new shape(2, 3, -1));
            print(a);
            print("*****");

            ndarray b = np.cumsum(a);
            print(b);
            AssertArray(b, new string[] { "A", "AB", "ABC", "ABCD", "ABCDE", "ABCDEF", "ABCDEFG", "ABCDEFGH",
                                          "ABCDEFGHI", "ABCDEFGHIJ", "ABCDEFGHIJK", "ABCDEFGHIJKL" });
            print("*****");

            ndarray c = np.cumsum(a, dtype: np.Strings);     // specifies type of output value(s)
            print(c);
            AssertArray(c, new string[] { "A", "AB", "ABC", "ABCD", "ABCDE", "ABCDEF", "ABCDEFG", "ABCDEFGH",
                                          "ABCDEFGHI", "ABCDEFGHIJ", "ABCDEFGHIJK", "ABCDEFGHIJKL" });
            print("*****");

            ndarray d = np.cumsum(a, axis: 0);     // sum over rows for each of the 3 columns
            print(d);

            var ExpectedDataD = new string[,,]
            {{{"A", "B"},
              {"C", "D"},
              {"E", "F"}},
             {{"AG", "BH"},
              {"CI", "DJ"},
              {"EK", "FL"}}};


            AssertArray(d, ExpectedDataD);
            print("*****");



            ndarray e = np.cumsum(a, axis: 1);    // sum over columns for each of the 2 rows
            print(e);

            var ExpectedDataE = new string[,,]
            {{{"A", "B"},
              {"AC", "BD"},
              {"ACE", "BDF"}},
             {{"G", "H"},
              {"GI", "HJ"},
              {"GIK", "HJL"}}};

            AssertArray(e, ExpectedDataE);
            print("*****");

            try
            {
                a[1, 1, 1] = 99;
                ndarray f = np.cumsum(a, axis: 2);    // sum over columns for each of the 2 rows
            }
            catch (Exception ex)
            {
                Assert.Fail("this should not have thrown exception");
            }


        }

        [TestMethod]
        public void test_diff_3_STRING()
        {
            var TestData = new string[] { "A", "JJJJJJABABABABADDDDDEEEEFFFFZZ", "B", "C", "AA" , "D", "E", "F", "G", "H", "I", "J" };
            var x = np.array(TestData, dtype: np.Strings).reshape(new shape(3, 2, -1));
            x = x * 3;
            var y = np.diff(x, axis: 2);

            print(x);
            print(y);

            var ExpectedData = new string[,,]
                {
                 {{"JJJJJJBBBBDDDDDEEEEFFFFZZ"},
                  {"C"}},

                 {{"D"},
                  {"F"}},

                 {{"H"},
                  {"J"}}
                };

            AssertArray(y, ExpectedData);

            try
            {
                x[0] = 99;
                y = np.diff(x, axis: 2);
            }
            catch (Exception ex)
            {
                Assert.Fail("This should not have thrown an exception");
            }

        }

        [TestMethod]
        public void test_ediff1d_1_STRING()
        {
            ndarray x = np.array(new string[] { "2", "2C", "CCC", "DD", "DD77DD" });
            ndarray y = np.ediff1d(x);
            print(y);
            AssertArray(y, new string[] { "C", "CCC", "DD", "77" });

     
        }

        [TestMethod]
        public void test_gradient_1_STRING()
        {
            // note: This function requires that all values be cast to doubles.
            // since strings get cast to 0 in this case, effectively this function does not work with strings
            var f = np.array(new string[] { "1", "2", "4", "7", "11", "16" }, dtype: np.Strings);
            var a = np.gradient(f);
            AssertArray(a[0], new double[] { 0, 0, 0, 0, 0, 0 });
            print(a[0]);
            print("***********");
        }

        [TestMethod]
        public void test_cross_2_STRING()
        {
            // Multiple vector cross-products. Note that the direction of the cross
            // product vector is defined by the `right-hand rule`.

            var x = np.array(new string[,] { { "A", "B", "C" }, { "D", "E", "F" } });
            var y = np.array(new string[,] { { "G", "H", "I" }, { "J", "K", "L" } });
            var a = np.cross(x, y);
            AssertArray(a, new string[,] { { "B", "C", "A" }, { "E", "F", "D" } });
            print(a);


            // The orientation of `c` can be changed using the `axisc` keyword.

            var b = np.cross(x, y, axisc: 0);
            AssertArray(b, new string[,] { { "B", "E" }, { "C", "F" }, { "A", "D" } });
            print(b);

            // Change the vector definition of `x` and `y` using `axisa` and `axisb`.

            x = np.array(new string[,] { { "A", "B", "C" }, { "D", "E", "F" }, { "G", "H", "I" } });
            y = np.array(new string[,] { { "K", "L", "M" }, { "N", "O", "P" }, { "Q", "R", "S" } });
            a = np.cross(x, y);
            AssertArray(a, new string[,] { { "B", "C", "A" }, { "E", "F", "D" }, { "H", "I", "G" } });
            print(a);

            b = np.cross(x, y, axisa: 0, axisb: 0);
            AssertArray(b, new string[,] { { "D", "G", "A" }, { "E", "H", "B" }, { "F", "I", "C" } });
            print(b);

            return;
        }

        [TestMethod]
        public void test_trapz_1_STRING()
        {

            try
            {
                var a = np.trapz(new string[] { "1", "2", "3" });
                Assert.Fail("This should have caused an exception");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains(MathFunctionExceptionPrefix));
            }


        }

        [TestMethod]
        public void test_exp_1_STRING()
        {

            var x = np.array(new string[] { "A", "B", "C" });

            try
            {
                var a = np.exp(x);
                Assert.Fail("This should throw an exception");
            }
            catch
            {

            }


        }

        [TestMethod]
        public void test_exp2_1_STRING()
        {
            var x = np.array(new string[] { "A", "B", "C" });
            try
            {
                var a = np.exp2(x);
                Assert.Fail("This should throw an exception");
            }
            catch
            {

            }

        }

        [TestMethod]
        public void test_i0_1_STRING()
        {
            var x = np.array(new string[] { "A", "B", "C" });
            try
            {
                var a = np.i0("5");
                Assert.Fail("This should throw an exception");
            }
            catch
            {

            }

            return;

        }

        [TestMethod]
        public void test_sinc_1_STRING()
        {
            try
            {
                double retstep = 0;
                var x = np.linspace(-4, 4, ref retstep, 10, dtype: np.Int64).astype(np.Strings);
                var a = np.sinc(x);
                Assert.Fail("This should throw an exception");
            }
            catch
            {

            }

        }

        [TestMethod]
        public void test_signbit_1_STRING()
        {
  
            try
            {
                var b = np.signbit(np.array(new string[] { "A", "B", "C" }));
                Assert.Fail("This should have thrown an exception");
            }
            catch
            {

            }

        }

        [TestMethod]
        public void test_copysign_1_STRING()
        {
 
            try
            {
                var d = np.copysign(np.array(new string[] { "A", "B", "C" }, dtype: np.Strings), -1.1);
                Assert.Fail("This should have thrown an exception");
            }
            catch
            {

            }

        }

        [TestMethod]
        public void test_frexp_1_STRING()
        {
            var x = np.arange(9, dtype: np.Int32).astype(np.Strings);

            try
            {
                var results = np.frexp(x);
                Assert.Fail("This should have thrown an exception");
            }
            catch
            {

            }


        }

        [TestMethod]
        public void test_ldexp_1_STRING()
        {
            try
            {
                var a = np.ldexp("5", np.arange(4, dtype: np.Int32)).astype(np.Strings);
                Assert.Fail("This should have thrown an exception");
            }
            catch
            {

            }

        }

        [TestMethod]
        public void test_lcm_1_STRING()
        {
            try
            {
                var d = np.lcm(np.arange(6, dtype: np.Int32).astype(np.Strings), new Int32[] { 20 });
                Assert.Fail("This should have thrown an exception");
            }
            catch
            {
            }

        }

        [TestMethod]
        public void test_gcd_1_STRING()
        {
            try
            {
                var d = np.gcd(np.arange(6, dtype: np.Int32).astype(np.Strings), new Int32[] { 20 });
                Assert.Fail("This should have thrown an exception");
            }
            catch
            {
            }

        }

        [TestMethod]
        public void test_add_1_STRING()
        {
            var a = np.add("1", "4");
            Assert.AreEqual("14", a.GetItem(0));
            print(a);

            var b = np.arange(9, dtype: np.Int32).reshape((3, 3)).astype(np.Strings);
            var c = np.arange(3, dtype: np.Int32).astype(np.Strings);
            var d = np.add(b, c);
            AssertArray(d, new String[,] { { "00", "11", "22" }, { "30", "41", "52" }, { "60", "71", "82" } });
            print(d);

            try
            {
                b[0] = new StringBuilder();
                d = np.add(b, c);
                Assert.Fail("This should have thrown an exception");
            }
            catch
            {

            }

        }

        [TestMethod]
        public void test_reciprocal_operations_STRING()
        {
            var a = np.arange(1, 32, 1, dtype: np.Float32).astype(np.Strings);
            print(a);

            var b = np.reciprocal(a);
            print(b);

            var ExpectedDataB1 = asstring(new Int32[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 });


            AssertArray(b, ExpectedDataB1);


            try
            {
                a[4] = "X";
                b = np.reciprocal(a);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }

        }

        [TestMethod]
        public void test_positive_1_STRING()
        {
            var d = np.positive(np.array(new string[] { "ABC", "JKL", "xyz" }));
            AssertArray(d, new string[] {"ABC", "JKL", "xyz" });
            print(d);

            var e = np.positive(np.array(new string[,] { { "ABC", "JKL", "xyz" }, { "abc", "jkl", "XYZ" } }));
            AssertArray(e, new string[,] { {"ABC", "JKL", "xyz" }, { "abc", "jkl", "XYZ" } });
            print(e);

            try
            {
                d = np.array(new Int32[] { -1, -0, 1 }).astype(np.Strings);
                d[1] = "X";
                d = np.positive(d);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }
        }

        [TestMethod]
        public void test_negative_1_STRING()
        {

            // this should reverse the strings
            var d = np.negative(np.array(new string[] { "ABC", "JKL", "xyz" }));
            AssertArray(d, new string[] { "CBA", "LKJ", "zyx" });
            print(d);

            var e = np.negative(np.array(new string[,] { { "ABC", "JKL", "xyz" }, { "abc", "jkl", "XYZ" } }));
            AssertArray(e, new string[,] { { "CBA", "LKJ", "zyx" }, { "cba", "lkj", "ZYX" } });
            print(e);

            try
            {
                d = np.array(new Int32[] { -1, -0, 1 }).astype(np.Strings);
                d[1] = "X";
                d = np.negative(d);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }

        }

        [TestMethod]
        public void test_multiply_1_STRING()
        {
            var a = np.multiply("2","4");
            Assert.AreEqual("2", a.GetItem(0));
            print(a);

            var b = np.arange(9).reshape((3, 3)).astype(np.Strings);
            var c = np.arange(3).astype(np.Strings);
            var d = np.multiply(b, c);
            AssertArray(d, new string[,] { { "0", "1", "2" }, { "3", "4", "5" }, { "6", "7", "8" } });
            print(d);

            try
            {
                c[1] = 99;
                d = np.multiply(b, c);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }
        }

        [TestMethod]
        public void test_divide_STRING()
        {
            var a = np.divide("7", "3");
            Assert.AreEqual("7", a.GetItem(0));
            print(a);

            var b = np.divide(np.array(new Int32[] { 1, 2, 3, 4 }).astype(np.Strings), 2);
            AssertArray(b, new string[] { "1", "2", "3", "4" });
            print(b);

            var c = np.divide(np.array(new Int32[] { 2, 4, 6, 8 }).astype(np.Strings), np.array(new Int32[] { 1, 2, 3, 4 }).astype(np.Strings));
            AssertArray(c, new string[] { "2", "4", "6", "8" });
            print(c);


            try
            {
                c[1] = 99;
                var d = np.divide(b, c);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }
            return;


        }

        [TestMethod]
        public void test_power_operations_STRING()
        {
            var a = np.arange(0, 4, 1, dtype: np.Int32).astype(np.Strings);
            print(a);

            var b = np.power(a, 3);
            print(b);

            var ExpectedDataB1 = asstring( new Int32[]
            {
                0, 1, 2, 3
            });

            AssertArray(b, ExpectedDataB1);

    
            b = np.power(a, 0);
            print(b);
    
            AssertArray(b, ExpectedDataB1);

            try
            {
                a[1] = 99;
                var d = np.power(a, 1);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }
            return;


        }

        [TestMethod]
        public void test_subtract_1_STRING()
        {
            var a = np.subtract("AABBCCFFB", "BB");
            Assert.AreEqual("AACCFFB", a.GetItem(0));
            print(a);

            var b = np.array(new string[] {"A1","B1","C1","D1","E1","F1", "G1", "H1", "I1" }).reshape((3, 3));
            var c = np.array(new string[] { "1", "1", "1" });
            var d = np.subtract(b, c);
            AssertArray(d, new string[,] { { "A", "B", "C" }, { "D", "E", "F" }, { "G", "H", "I" } });
            print(d);

            try
            {
                c[1] = 99;
                d = np.subtract(b, c);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }
            return;
        }

        [TestMethod]
        public void test_true_divide_STRING()
        {
            var a = np.true_divide("7", "3");
            Assert.AreEqual("7", a.GetItem(0));
            print(a);

            var b = np.true_divide(np.array(new string[] { "AA", "BB", "CC", "DD" }), 2.5);
            AssertArray(b, new string[] { "AA", "BB", "CC", "DD" });
            print(b);

            var c = np.true_divide(np.array(new string[] { "AA", "BB", "CC", "DD" }), new double[] { 0.5, 2.5, 2.5, 3.5 });
            AssertArray(c, new string[] { "AA", "BB", "CC", "DD" });
            print(c);

            return;
        }

        [TestMethod]
        public void test_floor_divide_STRING()
        {
            var a = np.floor_divide("7", "3");
            Assert.AreEqual("7", a.GetItem(0));
            print(a);

            var b = np.floor_divide(np.array(new string[] { "AA", "BB", "CC", "DD" }), 2.5);
            AssertArray(b, new string[] { "AA", "BB", "CC", "DD" });
            print(b);

            var c = np.floor_divide(np.array(new string[] { "AA", "BB", "CC", "DD" }), new double[] { 0.5, 2.5, 2.5, 3.5 });
            AssertArray(c, new string[] { "AA", "BB", "CC", "DD" });
            print(c);

            return;

        }

        [TestMethod]
        public void test_float_power_STRING()
        {
            var x1 = new int[] { 0, 1, 2, 3, 4, 5 };

            try
            {
                var a = np.float_power(np.array(x1).astype(np.Strings), 3);
                Assert.Fail("This should have thrown an exception");
            }
            catch
            {

            }


            return;
        }

        [TestMethod]
        public void test_fmod_2_STRING()
        {
            var x = np.fmod(new string[] { "AA", "BB", "CC", "DD" }, new string[] { "EE", "FF", "GG", "HH" });
            AssertArray(x, new string[] { "AA", "BB", "CC", "DD" });
            print(x);

            try
            {
                x = np.array(new string[] { "AA", "BB", "CC", "DD" });
                x[2] = 99;
                var a = np.fmod(x, 3);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }

            return;
        }

        [TestMethod]
        public void test_mod_1_STRING()
        {
            var x = np.mod(new string[] { "AA", "BB", "CC", "DD" }, new string[] { "EE", "FF", "GG", "HH" });
            AssertArray(x, new string[] { "AA", "BB", "CC", "DD" });
            print(x);

            try
            {
                x = np.array(new string[] { "AA", "BB", "CC", "DD" });
                x[2] = 99;
                var a = np.mod(x, 3);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }

            return;
        }

        [TestMethod]
        public void test_modf_1_STRING()
        {
            var x = np.modf(new string[] { "AA", "BB", "CC", "DD" });
            AssertArray(x[0], new string[] { "AA", "BB", "CC", "DD" });
            AssertArray(x[1], new string[] { "AA", "BB", "CC", "DD" });
            print(x);

            try
            {
                var x1 = np.array(new string[] { "AA", "BB", "CC", "DD" });
                x1[2] = 99;
                var a = np.mod(x, 3);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }



            return;
        }

        [TestMethod]
        public void test_remainder_2_STRING()
        {
            var x = np.remainder(new string[] { "AA", "BB", "CC", "DD" }, new string[] { "EE", "FF", "GG", "HH" });
            AssertArray(x, new string[] { "AA", "BB", "CC", "DD" });
            print(x);


            /////////////////////////////

            try
            {
                var x1 = np.array(new string[] { "AA", "BB", "CC", "DD" });
                x1[2] = 99;
                var a = np.remainder(x, 3);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }

            return;
        }

        [TestMethod]
        public void test_divmod_1_STRING()
        {
            var a = np.divmod("7", "3");
            Assert.AreEqual("7", a[0].GetItem(0));
            Assert.AreEqual("7", a[1].GetItem(0));
            print(a);

            var b = np.divmod(np.array(new string[] { "AA", "BB", "CC", "DD" }), 2.5);
            AssertArray(b[0], new string[] { "AA", "BB", "CC", "DD" });
            AssertArray(b[1], new string[] { "AA", "BB", "CC", "DD" });
            print(b);

            var c = np.divmod(np.array(new string[] { "AA", "BB", "CC", "DD" }), new double[] { 0.5, 2.5, 2.5, 3.5 });
            AssertArray(c[0], new string[] { "AA", "BB", "CC", "DD" });
            AssertArray(c[1], new string[] { "AA", "BB", "CC", "DD" });
            print(c);

     
            print(a);


            /////////////////////////////

            var x1 = np.arange(7).astype(np.Strings);
            x1[2] = "X";

            try
            {
                a = np.divmod(x1, 3);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }

            return;

        }

        [TestMethod]
        public void test_convolve_1_STRING()
        {
            var a = np.convolve(np.array(new string[] { "AA", "BB", "CC" }), np.array(new string[] { "DD", "EE", "FF" }));
            AssertArray(a, new string[] {  "AADD", "BBDD", "CCDD", "CCEE", "CCFF" });
            print(a);

            var b = np.convolve(new string[] { "AA", "BB", "CC" }, new string[] { "DD", "EE", "FF" }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_SAME);
            AssertArray(b, new string[] { "BBDD", "CCDD", "CCEE" });
            print(b);

            var c = np.convolve(new string[] { "AA", "BB", "CC" }, new string[] { "DD", "EE", "FF" }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_VALID);
            AssertArray(c, new string[] { "CCDD" });
            print(c);

            return;
        }

        [TestMethod]
        public void test_clip_2_STRING()
        {
            ndarray a = np.arange(16).reshape(new shape(4, 4)).astype(np.Strings);
            print(a);
            print("*****");

            ndarray b = np.clip(a, "1", "8");
            print(b);
            print("*****");
            AssertArray(b, asstring( new Int32[,] { { 1, 1, 2, 3 }, { 4, 5, 6, 7 }, { 8, 8, 10, 11 }, { 12, 13, 14, 15 } }));

            ndarray c = np.clip(a, "3", "6", @out: a);
            print(c);
            AssertArray(c, asstring(new Int32[,] { { 3, 3, 3, 3 }, { 4, 5, 6, 6 }, { 6, 6, 3, 3 }, { 3, 3, 3, 3 } }));
            print(a);
            AssertArray(a, asstring(new Int32[,] { { 3, 3, 3, 3 }, { 4, 5, 6, 6 }, { 6, 6, 3, 3 }, { 3, 3, 3, 3 } }));
            print("*****");

            a = np.arange(16).reshape(new shape(4, 4)).astype(np.Strings);
            print(a);
            b = np.clip(a, np.array(new Int32[] { 3, 4, 1, 1 }).astype(np.Strings), 8);
            print(b);
            AssertArray(b, asstring(new Int32[,] { { 3, 4, 2, 3 }, { 4, 5, 6, 7 }, { 8, 8, 10, 11 }, { 3, 4, 14, 15 } }));

            a["...", "..."] = "X";
            print(a);
            try
            {
                b = np.clip(a, np.array(new Int32[] { 3, 4, 1, 1 }).astype(np.Strings), 8);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }


        }

        [TestMethod]
        public void test_square_operations_STRING()
        {
            var a = np.array(new string[] { "AA", "BB", "CC" });
            print(a);

            var b = np.square(a);
            print(b);

            var ExpectedDataB1 = new string[] { "AA", "BB", "CC" };
      
            AssertArray(b, ExpectedDataB1);

  
            //////////////////////

            a["..."] = "X";
            print(a);
            try
            {
                b = np.square(a);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }

        }

        [TestMethod]
        public void test_absolute_operations_STRING()
        {
            var a = np.array(new string[] { "AA", "BB", "CC" });
            print(a);

            var b = np.absolute(a);
            print(b);

            var ExpectedDataB = new string[] { "AA", "BB", "CC" };
            AssertArray(b, ExpectedDataB);

            //////////////////////

            a["..."] = "X";
            print(a);
            try
            {
                b = np.absolute(a);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }

        }

        [TestMethod]
        public void test_fabs_1_STRING()
        {
            var a = np.array(new string[] { "AA", "BB", "CC" });
            print(a);

            var b = np.fabs(a);
            print(b);

            var ExpectedDataB = new string[] { "AA", "BB", "CC" };
            AssertArray(b, ExpectedDataB);

            //////////////////////

            a["..."] = "X";
            print(a);
            try
            {
                b = np.fabs(a);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }

        }

        [TestMethod]
        public void test_sign_1_STRING()
        {
 
            try
            {
                var a = np.sign("-199");

                var b = np.sign(np.array(new Int32[] { 1, -23, 21 }).astype(np.Strings));
                Assert.Fail("This should have thrown an exception");
            }
            catch
            {

            }

        }

        [TestMethod]
        public void test_heaviside_1_STRING()
        {
            var a = np.heaviside(np.array(new string[] { "AA", "BB", "CC" }), 5);
            AssertArray(a, new string[] { "AA", "BB", "CC" });
            print(a);

            var b = np.heaviside(np.array(new string[] { "AA", "BB", "CC" }), 1);
            AssertArray(b, new string[] { "AA", "BB", "CC" });
            print(b);

            var c = np.heaviside(np.array(new string[] { "AA", "BB", "CC" }), 1);
            AssertArray(c, new string[] { "AA", "BB", "CC" });
            print(c);

            //////////////////////

            a = np.array(new string[] { "A", "B", "C" });
            print(a);
            try
            {
                b = np.heaviside(a, 1);
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }


        }

        [TestMethod]
        public void test_maximum_1_STRING()
        {
            var a = np.maximum(np.array(new string[] { "AA", "EE", "CC" }), np.array(new string[] { "DD", "BB", "FF" }));
            AssertArray(a, new string[] { "DD", "EE", "FF" });
            print(a);

            var b = np.maximum(np.eye(2, dtype: np.Strings), new string[] { "5", "2" }); // broadcasting
            AssertArray(b, new string[,] { { "5", "2" }, { "5", "2" } });
            print(b);

            //////////////////////

            a = np.array(new string[] { "A", "B", "C" });
            print(a);
            try
            {
                b = np.maximum(a, np.array(new Int32[] { 1, 5, 2 }).astype(np.Strings));
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }



        }


        [TestMethod]
        public void test_minimum_1_STRING()
        {
            var a = np.minimum(np.array(new string[] { "AA", "EE", "CC" }), np.array(new string[] { "DD", "BB", "FF" }));
            AssertArray(a, new string[] { "AA", "BB", "CC" });
            print(a);

            var b = np.minimum(np.eye(2, dtype: np.Strings), new string[] { "5", "2" }); // broadcasting
            AssertArray(b, new string[,] { { "1", "0" }, { "0", "1" } });
            print(b);

            //////////////////////

            a = np.array(new string[] { "A", "B", "C" });
            print(a);
            try
            {
                b = np.minimum(a, np.array(new Int32[] { 1, 5, 2 }).astype(np.Strings));
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }

        }

        [TestMethod]
        public void test_fmax_1_STRING()
        {
            var a = np.fmax(np.array(new string[] { "AA", "EE", "CC" }), np.array(new string[] { "DD", "BB", "FF" }));
            AssertArray(a, new string[] { "DD", "EE", "FF" });
            print(a);

            var b = np.fmax(np.eye(2, dtype: np.Strings), new string[] { "5", "2" }); // broadcasting
            AssertArray(b, new string[,] { { "5", "2" }, { "5", "2" } });
            print(b);

            //////////////////////

            a = np.array(new string[] { "A", "B", "C" });
            print(a);
            try
            {
                b = np.fmax(a, np.array(new Int32[] { 1, 5, 2 }).astype(np.Strings));
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }

        }

        [TestMethod]
        public void test_fmin_1_STRING()
        {
            var a = np.fmin(np.array(new string[] { "AA", "EE", "CC" }), np.array(new string[] { "DD", "BB", "FF" }));
            AssertArray(a, new string[] { "AA", "BB", "CC" });
            print(a);

            var b = np.fmin(np.eye(2, dtype: np.Strings), new string[] { "5", "2" }); // broadcasting
            AssertArray(b, new string[,] { { "1", "0" }, { "0", "1" } });
            print(b);

            //////////////////////

            a = np.array(new string[] { "A", "B", "C" });
            print(a);
            try
            {
                b = np.fmin(a, np.array(new Int32[] { 1, 5, 2 }).astype(np.Strings));
            }
            catch
            {
                Assert.Fail("This should NOT have thrown an exception");
            }


        }


        #endregion

        #region from FromNumericTests

        [TestMethod]
        public void test_take_1_STRING()
        {
            var a = np.array(asstring(new Int32[] { 4, 3, 5, 7, 6, 8, 9, 12, 14, 16, 18, 20, 22, 24, 26, 28 }));
            var indices = np.array(new Int32[] { 0, 1, 4 });
            ndarray b = np.take(a, indices);
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);

            AssertArray(b, new string[] { "4", "3", "6" });
            AssertShape(b, 3);
            AssertStrides(b, SizeOfString);


            a = np.array(asstring(new Int32[] { 4, 3, 5, 7, 6, 8, 9, 12, 14, 16, 18, 20, 22, 24, 26, 28 }));
            indices = np.array(new Int32[,] { { 0, 1 }, { 2, 3 } });
            ndarray c = np.take(a, indices);
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);

            var ExpectedDataC = new string[2, 2]
            {
                { "4", "3" },
                { "5", "7" },
            };
            AssertArray(c, ExpectedDataC);
            AssertShape(c, 2, 2);
            AssertStrides(c, SizeOfString * 2, SizeOfString);

            ndarray d = np.take(a.reshape(new shape(4, -1)), indices, axis: 0);
            print("D");
            print(d);
            print(d.shape);
            print(d.strides);

            var ExpectedDataD = new Int32[2, 2, 4]
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
            AssertArray(d, asstring(ExpectedDataD));
            AssertShape(d, 2, 2, 4);
            AssertStrides(d, SizeOfString * 8, SizeOfString * 4, SizeOfString * 1);

            ndarray e = np.take(a.reshape(new shape(4, -1)), indices, axis: 1);
            print("E");
            print(e);
            print(e.shape);
            print(e.strides);

            var ExpectedDataE = new Int32[4, 2, 2]
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

            AssertArray(e, asstring(ExpectedDataE));
            AssertShape(e, 4, 2, 2);
            AssertStrides(e, SizeOfString * 4, SizeOfString * 2, SizeOfString * 1);

        }

        [TestMethod]
        public void test_ravel_1_STRING()
        {
            var a = np.array(new string[,] { { "1", "2", "3" }, { "4", "5", "6" } });
            var b = np.ravel(a);
            AssertArray(b, new string[] { "1", "2", "3", "4", "5", "6" });
            print(b);

            var c = a.reshape(-1);
            AssertArray(c, new string[] { "1", "2", "3", "4", "5", "6" });
            print(c);

            var d = np.ravel(a, order: NPY_ORDER.NPY_FORTRANORDER);
            AssertArray(d, new string[] { "1", "4", "2", "5", "3", "6" });
            print(d);

            // When order is 'A', it will preserve the array's 'C' or 'F' ordering:
            var e = np.ravel(a.T);
            AssertArray(e, new string[] { "1", "4", "2", "5", "3", "6" });
            print(e);

            var f = np.ravel(a.T, order: NPY_ORDER.NPY_ANYORDER);
            AssertArray(f, new string[] { "1", "2", "3", "4", "5", "6" });
            print(f);
        }

        [TestMethod]
        public void test_choose_1_STRING()
        {
            ndarray choice1 = np.array(new string[] { "0", "1", "2", "3" });
            ndarray choice2 = np.array(new string[] { "10", "11", "12", "13" });
            ndarray choice3 = np.array(new string[] { "20", "21", "22", "23" });
            ndarray choice4 = np.array(new string[] { "30", "31", "32", "33" });

            ndarray[] choices = new ndarray[] { choice1, choice2, choice3, choice4 };

            ndarray a = np.choose(np.array(new Int32[] { 2, 3, 1, 0 }), choices);

            print(a);

            AssertArray(a, new string[] { "20", "31", "12", "3" });
        }

        [TestMethod]
        public void test_choose_2_STRING()
        {
            ndarray choice1 = np.array(new string[] { "0", "1", "2", "3" });
            ndarray choice2 = np.array(new string[] { "10", "11", "12", "13" });
            ndarray choice3 = np.array(new string[] { "20", "21", "22", "23" });
            ndarray choice4 = np.array(new string[] { "30", "31", "32", "33" });

            ndarray[] choices = new ndarray[] { choice1, choice2, choice3, choice4 };

            ndarray a = np.choose(np.array(new Int32[] { 2, 4, 1, 0 }), choices, mode: NPY_CLIPMODE.NPY_CLIP);
            print(a);
            AssertArray(a, new string[] { "20", "31", "12", "3" });

            a = np.choose(np.array(new Int32[] { 2, 4, 1, 0 }), choices, mode: NPY_CLIPMODE.NPY_WRAP);
            print(a);
            AssertArray(a, new string[] { "20", "1", "12", "3" });

            try
            {
                a = np.choose(np.array(new Int32[] { 2, 4, 1, 0 }), choices, mode: NPY_CLIPMODE.NPY_RAISE);
                print(a);
                AssertArray(a, new string[] { "20", "1", "12", "3" });
            }
            catch (Exception ex)
            {
                if (ex.Message.Contains("invalid entry in choice array"))
                    return;
            }
            Assert.Fail("Should have caught exception from np.choose");


        }

        [TestMethod]
        public void test_select_1_STRING()
        {
            var x = np.arange(10, dtype: np.Int32);
            var condlist = new ndarray[] { x < 3, x > 5 };
            var choicelist = new ndarray[] { x.astype(np.Strings), np.array(np.power(x, 2), dtype: np.Strings) };
            var y = np.select(condlist, choicelist);

            AssertArray(y, new string[] { "0", "1", "2", "0", "0", "0", "36", "49", "64", "81" });
            print(y);
        }

        [TestMethod]
        public void test_repeat_1_STRING()
        {
            ndarray x = np.array(new Int32[] { 1, 2, 3, 4 }).reshape(new shape(2, 2)).astype(np.Strings);
            var y = new Int32[] { 2 };

            ndarray z = np.repeat(x, y);
            print(z);
            print("");
            AssertArray(z, asstring(new Int32[] { 1, 1, 2, 2, 3, 3, 4, 4 }));

            z = np.repeat("3", 4);
            print(z);
            print("");
            AssertArray(z, asstring(new Int32[] { 3, 3, 3, 3 }));

            z = np.repeat(x, 3, axis: 0);
            print(z);
            print("");

            var ExpectedData1 = asstring(new Int32[6, 2]
            {
                { 1, 2 },
                { 1, 2 },
                { 1, 2 },
                { 3, 4 },
                { 3, 4 },
                { 3, 4 },
            });

            AssertArray(z, ExpectedData1);
            AssertShape(z, 6, 2);

            z = np.repeat(x, 3, axis: 1);
            print(z);
            print("");

            var ExpectedData2 = asstring(new Int32[2, 6]
            {
                { 1, 1, 1, 2, 2, 2 },
                { 3, 3, 3, 4, 4, 4 },
            });

            AssertArray(z, ExpectedData2);
            AssertShape(z, 2, 6);



            z = np.repeat(x, new Int32[] { 1, 2 }, axis: 0);
            print(z);

            var ExpectedData3 = asstring(new Int32[3, 2]
            {
                { 1, 2 },
                { 3, 4 },
                { 3, 4 },
            });

            AssertArray(z, ExpectedData3);
            AssertShape(z, 3, 2);
        }

        [TestMethod]
        public void test_put_1_STRING()
        {
            ndarray a = np.arange(5, dtype: np.Int32).astype(np.Strings);
            np.put(a, new int[] { 0, 2 }, new int[] { -44, -55 });
            print(a);
            AssertArray(a, asstring(new Int32[] { -44, 1, -55, 3, 4 }));

            a = np.arange(5, dtype: np.Int32).astype(np.Strings);
            np.put(a, 22, -5, mode: NPY_CLIPMODE.NPY_CLIP);
            print(a);
            AssertArray(a, asstring(new Int32[] { 0, 1, 2, 3, -5 }));

            a = np.arange(5, dtype: np.Int32).astype(np.Strings);
            np.put(a, 22, -5, mode: NPY_CLIPMODE.NPY_WRAP);
            print(a);
            AssertArray(a, asstring(new Int32[] { 0, 1, -5, 3, 4 }));

            try
            {
                a = np.arange(5, dtype: np.Int32).astype(np.Strings);
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
        public void test_putmask_1_STRING()
        {
            var x = np.arange(6, dtype: np.Int32).reshape((2, 3)).astype(np.Strings);
            np.putmask(x, x > 2, x);
            AssertArray(x, asstring(new Int32[,] { { 0, 1, 2, }, { 3, 4, 5 } }));
            print(x);


            // If values is smaller than a it is repeated:

            x = np.arange(5, dtype: np.Int32).astype(np.Strings);
            np.putmask(x, x > 1, new Int32[] { -33, -44 });
            AssertArray(x, asstring(new Int32[] { 0, 1, -33, -44, -33 }));
            print(x);

            return;
        }

        [TestMethod]
        public void test_swapaxes_1_STRING()
        {
            ndarray x = np.array(new Int32[,] { { 1, 2, 3 } }).astype(np.Strings);
            print(x);
            print("********");

            ndarray y = np.swapaxes(x, 0, 1);
            print(y);
            AssertArray(y, asstring(new Int32[3, 1] { { 1 }, { 2 }, { 3 } }));
            print("********");

            x = np.array(new Int32[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } }).astype(np.Strings);
            print(x);

            var ExpectedDataX = asstring(new Int32[2, 2, 2]
            {
                {
                    { 0,1 },
                    { 2,3 },
                },
                {
                    { 4,5 },
                    { 6,7 },
                },
            });
            AssertArray(x, ExpectedDataX);

            print("********");

            y = np.swapaxes(x, 0, 2);
            print(y);

            var ExpectedDataY = asstring(new Int32[2, 2, 2]
            {
                {
                    { 0,4 },
                    { 2,6 },
                },
                {
                    { 1,5 },
                    { 3,7 },
                },
            });
            AssertArray(y, ExpectedDataY);
        }

        [TestMethod]
        public void test_ndarray_T_1_STRING()
        {
            var x = np.arange(0, 32, dtype: np.Int32).reshape(new shape(8, 4)).astype(np.Strings);
            print("X");
            print(x);
            print(x.shape);

            var y = x.T;

            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = asstring(new Int32[4, 8]
            {
                { 0, 4,  8, 12, 16, 20, 24, 28 },
                { 1, 5,  9, 13, 17, 21, 25, 29 },
                { 2, 6, 10, 14, 18, 22, 26, 30 },
                { 3, 7, 11, 15, 19, 23, 27, 31 },
            });

            AssertArray(y, ExpectedDataY);

        }

        [TestMethod]
        public void test_ndarray_transpose_1_STRING()
        {
            var x = np.arange(0, 64, dtype: np.Int32).reshape(new shape(2, 4, -1, 4)).astype(np.Strings);
            print("X");
            print(x);
            print(x.shape);

            var y = np.transpose(x, new long[] { 1, 2, 3, 0 });

            print("Y");
            print(y);
            print(y.shape);

            var ExpectedDataY = asstring(new Int32[4, 2, 4, 2]
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
                    {31, 63}}}});

            AssertArray(y, ExpectedDataY);

        }

        [TestMethod]
        public void test_partition_3_STRING()
        {
            var a = np.arange(22, 10, -1, dtype: np.Int32).reshape((3, 4, 1)).astype(np.Strings);
            var b = np.partition(a, 1, axis: 0);
            AssertArray(b, asstring(new Int32[,,] { { { 14 }, { 13 }, { 12 }, { 11 } }, { { 18 }, { 17 }, { 16 }, { 15 } }, { { 22 }, { 21 }, { 20 }, { 19 } } }));
            print(b);

            var c = np.partition(a, 2, axis: 1);
            AssertArray(c, asstring(new Int32[,,] { { { 19 }, { 20 }, { 21 }, { 22 } }, { { 15 }, { 16 }, { 17 }, { 18 } }, { { 11 }, { 12 }, { 13 }, { 14 } } }));
            print(c);

            var d = np.partition(a, 0, axis: 2);
            AssertArray(d, asstring(new Int32[,,] { { { 22 }, { 21 }, { 20 }, { 19 } }, { { 18 }, { 17 }, { 16 }, { 15 } }, { { 14 }, { 13 }, { 12 }, { 11 } } }));
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
        public void test_argpartition_3_STRING()
        {
            var a = np.arange(22, 10, -1, np.Int32).reshape((3, 4, 1)).astype(np.Strings);
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
                var e = np.argpartition(a, 4, axis: 1);
                print(e);
            }
            catch (Exception ex)
            {
                return;
            }

            Assert.Fail("Should have caught the exception");

        }

        [TestMethod]
        public void test_sort_2_STRING()
        {
            var InputData = asstring(new Int32[]
                {32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                 16, 15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1});

            var a = np.array(InputData).reshape(new shape(8, 4));
            ndarray b = np.sort(a);                 // sort along the last axis
            print(b);

            var ExpectedDataB = asstring(new Int32[8, 4]
            {
             {29, 30, 31, 32},
             {25, 26, 27, 28},
             {21, 22, 23, 24},
             {17, 18, 19, 20},
             {13, 14, 15, 16},
             {10, 11, 12, 9},
             {5,  6,  7,  8},
             {1,  2,  3,  4},
            });

            AssertArray(b, ExpectedDataB);

            ndarray c = np.sort(a, axis: null);     // sort the flattened array
            print(c);
            print("********");

            var ExpectedDataC = asstring(new Int32[]
                  { 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 4, 5, 6, 7, 8, 9 });

            AssertArray(c, ExpectedDataC);

            ndarray d = np.sort(a, axis: 0);        // sort along the first axis
            print(d);

            var ExpectedDataD = asstring(new Int32[8, 4]
                { { 12, 11, 10, 1 },
                  { 16, 15, 14, 13 },
                  { 20, 19, 18, 17 },
                  { 24, 23, 2, 21 },
                  { 28, 27, 22, 25 },
                  { 32, 3, 26, 29 },
                  { 4, 31, 30, 5 },
                  { 8, 7, 6, 9 } });

            AssertArray(d, ExpectedDataD);
            print("********");

        }

        [TestMethod]
        public void test_msort_1_STRING()
        {
            var a = np.array(new Int32[,] { { 1, 4 }, { 3, 1 } }).astype(np.Strings);
            ndarray b = np.msort(a);
            print(b);
            AssertArray(b, asstring(new Int32[,] { { 1, 1 }, { 3, 4 } }));

            a = np.arange(32, 0, -1.0, dtype: np.Int32).astype(np.Strings);
            b = np.msort(a);

            var ExpectedDataB = asstring( new Int32[]
            {1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23,
             24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 4, 5, 6, 7, 8, 9 });
            AssertArray(b, ExpectedDataB);
            print(b);

        }

        [TestMethod]
        public void test_ndarray_argsort_2_STRING()
        {
            var ar = np.array(new Int32[] { 1, 2, 3, 1, 3, 4, 5, 4, 4, 1, 9, 6, 9, 11, 23, 9, 5, 0, 11, 12 }).reshape(new shape(5, 4)).astype(np.Strings);

            ndarray perm1 = np.argsort(ar, kind: NPY_SORTKIND.NPY_MERGESORT);
            ndarray perm2 = np.argsort(ar, kind: NPY_SORTKIND.NPY_QUICKSORT);
            ndarray perm3 = np.argsort(ar);

            print(perm1);

            var Perm1Expected = new npy_intp[,]
            {{0, 3, 1, 2},
             {0, 1, 3, 2},
             {1, 0, 3, 2},
             {1, 2, 0, 3},
             {1, 2, 3, 0}};
            AssertArray(perm1, Perm1Expected);

            print(perm2);
            var Perm2Expected = new npy_intp[,]
            {{0, 3, 1, 2},
             {0, 1, 3, 2},
             {1, 0, 3, 2},
             {1, 2, 0, 3},
             {1, 2, 3, 0}};
            AssertArray(perm2, Perm2Expected);


            print(perm3);
            var Perm3Expected = new npy_intp[,]
            {{0, 3, 1, 2},
             {0, 1, 3, 2},
             {1, 0, 3, 2},
             {1, 2, 0, 3},
             {1, 2, 3, 0}};
            AssertArray(perm3, Perm3Expected);
        }

        [TestMethod]
        public void test_argmin_1_STRING()
        {
            ndarray a = np.array(new string[] { "32", "33", "45", "98", "11", "02" }).reshape(new shape(2, 3));
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
        public void test_argmax_1_STRING()
        { 
            ndarray a = np.array(new string[] { "32", "33", "45", "98", "11", "02" }).reshape(new shape(2, 3));
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
        public void test_searchsorted_1_STRING()
        {
            ndarray arr = np.array(new string[] { "1", "2", "3", "4", "5" });
            ndarray a = np.searchsorted(arr, 3);
            print(a);
            Assert.AreEqual(a.GetItem(0), (npy_intp)2);


            ndarray b = np.searchsorted(arr, 3, side: NPY_SEARCHSIDE.NPY_SEARCHRIGHT);
            print(b);
            Assert.AreEqual(b.GetItem(0), (npy_intp)3);


            ndarray c = np.searchsorted(arr, new Int32[] { -10, 10, 2, 3 });
            print(c);
            AssertArray(c, new npy_intp[] { 1, 1, 1, 2 });


            ndarray d = np.searchsorted(np.array(new string[] { "15", "14", "13", "12", "11" }), 13);
            print(d);
            Assert.AreEqual(d.GetItem(0), (npy_intp)0);
        }

        [TestMethod]
        public void test_resize_1_STRING()
        {
            ndarray a = np.array(new string[,] { { "0", "1" }, { "2", "3" } });
            print(a);

            ndarray b = np.resize(a, new shape(2, 3));
            print(b);

            var ExpectedDataB = asstring(new Int32[,]
            {
                { 0,1,2 },
                { 3,0,1 },
            });
            AssertArray(b, ExpectedDataB);


            ndarray c = np.resize(a, new shape(1, 4));
            print(c);
            var ExpectedDataC = asstring(new Int32[,]
            {
                { 0,1,2,3 },
            })
            ;
            AssertArray(c, ExpectedDataC);

            ndarray d = np.resize(a, new shape(2, 4));
            print(d);
            var ExpectedDataD = asstring(new Int32[,]
            {
                { 0,1,2,3 },
                { 0,1,2,3 },
            });
            AssertArray(d, ExpectedDataD);

        }

        [TestMethod]
        public void test_squeeze_1_STRING()
        {
            ndarray x = np.array(new string[,,] { { { "0" }, { "1" }, { "2" } } });
            print(x);
            AssertArray(x, new string[1, 3, 1] { { { "0" }, { "1" }, { "2" } } });

            ndarray a = np.squeeze(x);
            print(a);
            AssertArray(a, new string[] { "0", "1", "2" });

            ndarray b = np.squeeze(x, axis: 0);
            print(b);
            AssertArray(b, new string[3, 1] { { "0" }, { "1" }, { "2" } });

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
            AssertArray(d, new string[,] { { "0", "1", "2" } });
        }

        [TestMethod]
        public void test_diagonal_1_STRING()
        {
            ndarray a = np.arange(4, dtype: np.Int32).reshape(new shape(2, 2)).astype(np.Strings);
            print(a);
            print("*****");

            ndarray b = a.diagonal();
            print(b);
            AssertArray(b, new String[] { "0", "3" });
            print("*****");

            ndarray c = a.diagonal(1);
            print(c);
            AssertArray(c, new String[] { "1" });
            print("*****");

            a = np.arange(8, dtype: np.Int32).reshape(new shape(2, 2, 2)).astype(np.Strings);
            print(a);
            print("*****");
            b = a.diagonal(0, // Main diagonals of two arrays created by skipping
                           0, // across the outer(left)-most axis last and
                           1); //the "middle" (row) axis first.

            print(b);
            AssertArray(b, new string[,] { { "0", "6" }, { "1", "7" } });
            print("*****");

            ndarray d = a.A(":", ":", 0);
            print(d);
            AssertArray(d, new string[,] { { "0", "2" }, { "4", "6" } });
            print("*****");

            ndarray e = a.A(":", ":", 1);
            print(e);
            AssertArray(e, new string[,] { { "1", "3" }, { "5", "7" } });
            print("*****");
        }

        [TestMethod]
        public void test_trace_1_STRING()
        {
            ndarray a = np.trace(np.eye(3).astype(np.Strings));
            print(a);
            Assert.AreEqual(a.GetItem(0), "111");
            print("*****");

            a = np.arange(8, dtype: np.Int32).reshape(new shape(2, 2, 2)).astype(np.Strings);
            ndarray b = np.trace(a);
            print(b);
            AssertArray(b, new string[] { "06", "17" });
            print("*****");

            a = np.arange(24, dtype: np.Int32).reshape(new shape(2, 2, 2, 3)).astype(np.Strings);
            var c = np.trace(a);
            print(c);
            AssertArray(c, new String[,] { { "018", "119", "220" }, { "321", "422", "523" } });

        }

        [TestMethod]
        public void test_nonzero_1_STRING()
        {
            ndarray x = np.array(new string[,] { { "1", null, null }, { null, "2", null }, { "1", "1", null } });
            print(x);
            print("*****");

            ndarray[] y = np.nonzero(x);
            print(y);
            AssertArray(y[0], new npy_intp[] { 0, 1, 2, 2 });
            AssertArray(y[1], new npy_intp[] { 0, 1, 0, 1 });
            print("*****");

            ndarray z = x.A(np.nonzero(x));
            print(z);
            AssertArray(z, new string[] { "1", "2", "1", "1" });
            print("*****");

            //ndarray q = np.transpose(np.nonzero(x));
            //print(q);

        }

        [TestMethod]
        public void test_compress_1_STRING()
        {
            ndarray a = np.array(new string[,] { { "1", "2" }, { "3", "4" }, { "5", "6" } });
            print(a);
            print("*****");

            ndarray b = np.compress(new int[] { 0, 1 }, a, axis: 0);
            print(b);
            AssertArray(b, new string[,] { { "3", "4" } });
            print("*****");

            ndarray c = np.compress(new bool[] { false, true, true }, a, axis: 0);
            print(c);
            AssertArray(c, new string[,] { { "3", "4" }, { "5", "6" } });
            print("*****");

            ndarray d = np.compress(new bool[] { false, true }, a, axis: 1);
            print(d);
            AssertArray(d, new string[,] { { "2" }, { "4" }, { "6" } });
            print("*****");

            ndarray e = np.compress(new bool[] { false, true }, a);
            AssertArray(e, new string[] { "2" });
            print(e);

        }

        [TestMethod]
        public void test_any_1_STRING()
        {
            var TestData = new string[] { "AA", "BB", "CC", null, "DD", "EE", "FF", "GG" };
            var x = np.array(TestData);
            var y = np.any(x);

            print(x);
            print(y);
            Assert.AreEqual(true, y.GetItem(0));

            TestData = new string[] { null, null, null, null };
            x = np.array(TestData, dtype: np.Strings);
            y = np.any(x);

            print(x);
            print(y);
            Assert.AreEqual(false, y.GetItem(0));

        }

        [TestMethod]
        public void test_all_1_STRING()
        {
            var TestData = new string[] { "AA", "BB", "CC", null, "DD", "EE", "FF", "GG" };
            var x = np.array(TestData);
            var y = np.all(x);

            print(x);
            print(y);
            Assert.AreEqual(false, y.GetItem(0));

            TestData = new string[] { null, null, null, null };
            x = np.array(TestData, dtype: np.Strings);
            y = np.all(x);

            print(x);
            print(y);
            Assert.AreEqual(false, y.GetItem(0));

        }

        [TestMethod]
        public void test_ndarray_mean_1_STRING()
        {
            var x = np.arange(0, 12, dtype: np.Int32).reshape(new shape(3, -1)).astype(np.Strings);

            print("X");
            print(x);

            var y = (ndarray)np.mean(x, dtype: np.Strings);
            //Assert.AreEqual((double)5, y.GetItem(0));

            print("Y");
            print(y);

            y = (ndarray)np.mean(x, axis: 0, dtype: np.Strings);
            //AssertArray(y, new double[] { 4, 5, 6, 7 });

            print("Y");
            print(y);

            y = (ndarray)np.mean(x, axis: 1, dtype: np.Strings);
            //AssertArray(y, new double[] { 1, 5, 9 });

            print("Y");
            print(y);

        }

        [TestMethod]
        public void test_place_1_STRING()
        {
            var arr = np.arange(6, dtype: np.Int32).reshape((2, 3)).astype(np.Strings);
            np.place(arr, arr > 2, new Int32[] { 44, 55 });
            AssertArray(arr, new String[,] { { "0", "1", "2" }, { "44", "55", "44" } });
            print(arr);

            arr = np.arange(16, dtype: np.Int32).reshape((2, 4, 2)).astype(np.Strings);
            np.place(arr, arr > 12, new Int32[] { 33 });
            AssertArray(arr, new String[,,] { { { "0", "1" }, { "33", "33" }, { "33", "33" }, { "33", "33" } }, { { "33", "33" }, 
                                                { "10", "11" }, { "12", "33" }, { "33", "33" } } });
            print(arr);

            arr = np.arange(6, dtype: np.Int32).reshape((2, 3)).astype(np.Strings);
            np.place(arr, arr > 2, new Int32[] { 44, 55, 66, 77, 88, 99, 11, 22, 33 });
            AssertArray(arr, new String[,] { { "0", "1", "2" }, { "44", "55", "66" } });
            print(arr);

        }

        [TestMethod]
        public void test_extract_1_STRING()
        {
            var arr = np.arange(12, dtype: np.Int32).reshape((3, 4)).astype(np.Strings);
            var condition = np.mod(arr, 3) == 0;
            print(condition);

            var b = np.extract(condition, arr);
            AssertArray(b, new string[] { "0", "3", "6", "9" });
            print(b);
        }

        [TestMethod]
        public void test_viewfromaxis_1_STRING()
        {
            string[] TestData = new string[] { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12" };
            var a = np.zeros_like(TestData).reshape(new shape(3, 2, -1));
            //print(a);


            var b = np.ViewFromAxis(a, 0);
            b[":"] = 99;
            //print(a);
            AssertArray(a, new string[,,] { { { "99", "0" }, { "0", "0" } }, { { "99", "0" }, { "0", "0" } }, { { "99", "0" }, { "0", "0" } } });
            //print(a);
            //AssertArray(np.sum(a, axis: 0), new string[,] { { "999999", "0" }, { "0", "0" } });

            b = np.ViewFromAxis(a, 1);
            b[":"] = 11;
            //AssertArray(a, new string[,,] { { { 11, 0 }, { 11, 0 } }, { { 99, 0 }, { 0, 0 } }, { { 99, 0 }, { 0, 0 } } });
            //print(a);
            //AssertArray(np.sum(a, axis: 1), new string[,] { { 22, 0 }, { 99, 0 }, { 99, 0 } });

            b = np.ViewFromAxis(a, 2);
            b[":"] = 22;
            //AssertArray(a, new string[,,] { { { 22, 22 }, { 11, 0 } }, { { 99, 0 }, { 0, 0 } }, { { 99, 0 }, { 0, 0 } } });
            //print(a);
            //AssertArray(np.sum(a, axis: 2), new string[,] { { 44, 11 }, { 99, 0 }, { 99, 0 } });

            //Assert.AreEqual("253", np.sum(a).GetItem(0));


        }



        #endregion

        #region from NumericTests

        [TestMethod]
        public void test_zeros_1_STRING()
        {
            var x = np.zeros(new shape(10), dtype: np.Strings);
            print(x);
            print("Update sixth value to 11");
            x[6] = 11;
            print(x);
            print(x.shape);
            print(x.strides);

            AssertArray(x, new string[] { "0", "0", "0", "0", "0", "0", "11", "0", "0", "0" });
            AssertShape(x, 10);
            AssertStrides(x, SizeOfString);
        }

        [TestMethod]
        public void test_zeros_like_2_STRING()
        {
            var a = new string[,] { { "1", "2", "3" }, { "4", "5", "6" } };
            var b = np.zeros_like(a);
            b[1, 2] = 99;

            AssertArray(b, new string[,] { { "0", "0", "0" }, { "0", "0", "99" } });

            return;
        }

        [TestMethod]
        public void test_ones_1_STRING()
        {
            var x = np.ones(new shape(10), dtype: np.Strings);
            print(x);
            print("Update sixth value to 11");
            x[6] = 11;
            print(x);
            print(x.shape);
            print(x.strides);

            AssertArray(x, new string[] { "1", "1", "1", "1", "1", "1", "11", "1", "1", "1" });
            AssertShape(x, 10);
            AssertStrides(x, SizeOfString);
        }

        [TestMethod]
        public void test_ones_like_3_STRING()
        {
            var a = new string[,,] { { { "1", "2", "3" }, { "4", "5", "6" } } };
            var b = np.ones_like(a);
            b[0, 0, 2] = 99;
            b[0, 1, 1] = 88;

            AssertArray(b, new string[,,] { { { "1", "1", "99" }, { "1", "88", "1" } } });

            return;
        }

        [TestMethod]
        public void test_empty_STRING()
        {
            var a = np.empty((2, 3));
            AssertShape(a, 2, 3);
            Assert.AreEqual(a.Dtype.TypeNum, NPY_TYPES.NPY_DOUBLE);

            var b = np.empty((2, 4), np.Strings);
            AssertShape(b, 2, 4);
            Assert.AreEqual(b.Dtype.TypeNum, NPY_TYPES.NPY_STRING);
        }

        [TestMethod]
        public void test_empty_like_3_STRING()
        {
            var a = new string[,,] { { { "1", "2", "3" }, { "4", "5", "6" } } };
            var b = np.empty_like(a);
            b[0, 0, 2] = 99;
            b[0, 1, 1] = 88;

            AssertArray(b, new string[,,] { { { "0", "0", "99" }, { "0", "88", "0" } } });

            return;
        }

        [TestMethod]
        public void test_full_2_STRING()
        {
            var x = np.full((100), 99, dtype: np.Strings).reshape(new shape(10, 10));
            print(x);
            print("Update sixth value to 55");
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
        public void test_count_nonzero_1_STRING()
        {
            var a = np.count_nonzero_i(np.eye(4, dtype: np.Strings));
            Assert.AreEqual(16, a);
            print(a);

            var b = np.count_nonzero_i(new string[,] { { null, "1", "7", null, null }, { "3", null, null, "2", "19" } });
            Assert.AreEqual(5, b);
            print(b);

            var c = np.count_nonzero(new string[,] { { null, "1", "7", null, null }, { "3", null, null, "2", "19" } }, axis: 0);
            AssertArray(c, new int[] { 1, 1, 1, 1, 1 });
            print(c);

            var d = np.count_nonzero(new string[,] { { null, "1", "7", null, null }, { "3", null, null, "2", "19" } }, axis: 1);
            AssertArray(d, new int[] { 2, 3 });
            print(d);

            return;
        }

        [TestMethod]
        public void test_asarray_1_STRING()
        {
            var a = new string[] { "1", "2" };
            var b = np.asarray(a);

            AssertArray(b, new string[] { "1", "2" });
            print(b);

            var c = np.array(new string[] { "1", "2" }, dtype: np.Strings);
            var d = np.asarray(c, dtype: np.Strings);

            c[0] = 3;
            AssertArray(d, new string[] { "3", "2" });
            print(d);

            var e = np.asarray(a, dtype: np.Strings);
            AssertArray(e, new string[] { "1", "2" });

            print(e);

            return;
        }

        [TestMethod]
        public void test_ascontiguousarray_1_STRING()
        {
            var x = np.arange(6, dtype: np.Int32).reshape((2, 3)).astype(np.Strings);
            var y = np.ascontiguousarray(x, dtype: np.Strings);

            AssertArray(y, new String[,] { { "0", "1", "2" }, { "3", "4", "5" } });
            print(y);

            Assert.AreEqual(x.flags.c_contiguous, true);
            Assert.AreEqual(y.flags.c_contiguous, true);

            return;
        }

        [TestMethod]
        public void test_asfortranarray_1_STRING()
        {
            var x = np.arange(6, dtype: np.Int32).reshape((2, 3)).astype(np.Strings);
            var y = np.asfortranarray(x, dtype: np.Strings);

            AssertArray(y, new String[,] { { "0", "1", "2" }, { "3", "4", "5" } });
            print(y);

            Assert.AreEqual(x.flags.f_contiguous, false);
            Assert.AreEqual(y.flags.f_contiguous, true);

            return;
        }

        [TestMethod]
        public void test_isfortran_1_STRING()
        {

            var a = np.array(new String[,] { { "1", "2", "3" }, { "4", "5", "6" } }, order: NPY_ORDER.NPY_CORDER);
            var a1 = np.isfortran(a);
            Assert.AreEqual(false, a1);
            print(a1);

            var b = np.array(new String[,] { { "1", "2", "3" }, { "4", "5", "6" } }, order: NPY_ORDER.NPY_FORTRANORDER);
            var b1 = np.isfortran(b);
            Assert.AreEqual(true, b1);
            print(b1);

            var c = np.array(new String[,] { { "1", "2", "3" }, { "4", "5", "6" } }, order: NPY_ORDER.NPY_CORDER);
            var c1 = np.isfortran(c);
            Assert.AreEqual(false, c1);
            print(c1);

            var d = a.T;
            var d1 = np.isfortran(d);
            Assert.AreEqual(true, d1);
            print(d1);

            // C-ordered arrays evaluate as False even if they are also FORTRAN-ordered.

            var e1 = np.isfortran(np.array(new String[] { "1", "2" }, order: NPY_ORDER.NPY_FORTRANORDER));
            Assert.AreEqual(false, e1);
            print(e1);

            return;

        }

        [TestMethod]
        public void test_argwhere_1_STRING()
        {
            var x = np.arange(6, dtype: np.Int32).reshape((2, 3)).astype(np.Strings);
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
        public void test_flatnonzero_1_STRING()
        {
            var x = np.array(new string[] { "-2", "-1", null, "1", "2" });

            var y = np.flatnonzero(x);
            AssertArray(y, new npy_intp[] { 0, 1, 3, 4 });
            print(y);

            // Use the indices of the non-zero elements as an index array to extract these elements:

            var z = x.ravel()[np.flatnonzero(x)] as ndarray;
            AssertArray(z, new String[] { "-2", "-1", "1", "2" });
            print(z);

            return;
        }

        [TestMethod]
        public void test_outer_1_STRING()
        {
            var a = np.arange(2, 10, dtype: np.Int32).reshape((2, 4)).astype(np.Strings);
            var b = np.arange(12, 20, dtype: np.Int32).reshape((2, 4)).astype(np.Strings);
            var c = np.outer(a, b);

            var ExpectedDataC =asstring( new Int32[,]
                { { 2, 2, 2, 2, 2, 2, 2, 2 },
                  { 3, 3, 3, 3, 3, 3, 3, 3 },
                  { 4, 4, 4, 4, 4, 4, 4, 4 },
                  { 5, 5, 5, 5, 5, 5, 5, 5 },
                  { 6, 6, 6, 6, 6, 6, 6, 6 },
                  { 7, 7, 7, 7, 7, 7, 7, 7 },
                  { 8, 8, 8, 8, 8, 8, 8, 8 },
                  { 9, 9, 9, 9, 9, 9, 9, 9 } });

            AssertArray(c, ExpectedDataC);

            print(c);

            return;
        }

        [TestMethod]
        public void test_inner_1_STRING()
        {
            var a = np.arange(1, 5, dtype: np.Int32).reshape((2, 2)).astype(np.Strings);
            var b = np.arange(11, 15, dtype: np.Int32).reshape((2, 2)).astype(np.Strings);
            var c = np.inner(a, b);
            AssertArray(c, new string[,] { { "212", "214" }, { "412", "414" } });
            print(c);


            a = np.arange(2, 10, dtype: np.Int32).reshape((2, 4)).astype(np.Strings);
            b = np.arange(12, 20, dtype: np.Int32).reshape((2, 4)).astype(np.Strings);
            c = np.inner(a, b);
            print(c);
            AssertArray(c, new string[,] { {  "515", "519" }, { "915", "919" } });
            print(c.shape);

            return;
        }

        [TestMethod]
        public void test_tensordot_2_STRING()
        {
            var a = np.arange(12.0, dtype: np.Int32).reshape((3, 4)).astype(np.Strings);
            var b = np.arange(24.0, dtype: np.Int32).reshape((4, 3, 2)).astype(np.Strings);
            var c = np.tensordot(a, b, axis: 1);
            AssertShape(c, 3, 3, 2);
            print(c.shape);
            print(c);
            AssertArray(c,asstring( new Int32[,,] {{{ 318, 319 },  { 320, 321 },  { 322, 323 }}, {{ 718, 719 },  { 720, 721 },  { 722, 723 }}, {{ 1118, 1119 },{ 1120, 1121 },{ 1122, 1123 }}}));


            c = np.tensordot(a, b, axis: 0);
            AssertShape(c, 3, 4, 4, 3, 2);
            print(c.shape);

            print(c);
        }

        [TestMethod]
        public void test_dot_1_STRING()
        {
            var a = new string[,] { { "1", "0" }, { "0", "1" } };
            var b = new string[,] { { "4", "1" }, { "2", "2" } };
            var c = np.dot(a, b);
            AssertArray(c, new string[,] { { "02", "02" }, { "12", "12" } });
            print(c);

            var d = np.dot("3", "4");
            Assert.AreEqual("34", d.GetItem(0));
            print(d);

   
        }

        [TestMethod]
        public void test_roll_forward_STRING()
        {
            var a = np.arange(10, dtype: np.Int32).astype(np.Strings);

            print("A");
            print(a);
            print(a.shape);
            print(a.strides);

            var b = np.roll(a, 2);
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);
            AssertArray(b, new string[] { "8", "9", "0", "1", "2", "3", "4", "5", "6", "7" });
            AssertShape(b, 10);

            var c = np.roll(b, 2);
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);
            AssertArray(c, new string[] { "6", "7", "8", "9", "0", "1", "2", "3", "4", "5" });
            AssertShape(c, 10);

        }

        [TestMethod]
        public void test_roll_backward_STRING()
        {
            var a = np.arange(10, dtype: np.Int32).astype(np.Strings);

            print("A");
            print(a);
            print(a.shape);
            print(a.strides);

            var b = np.roll(a, -2);
            print("B");
            print(b);
            print(b.shape);
            print(b.strides);
            AssertArray(b, new string[] { "2", "3", "4", "5", "6", "7", "8", "9", "0", "1" });
            AssertShape(b, 10);

            var c = np.roll(b, -6);
            print("C");
            print(c);
            print(c.shape);
            print(c.strides);
            AssertArray(c, new string[] { "8", "9", "0", "1", "2", "3", "4", "5", "6", "7" });
            AssertShape(c, 10);
        }

        [TestMethod]
        public void test_ndarray_rollaxis_STRING_TODO()
        {
            var a = np.ones((3, 4, 5, 6), dtype: np.Object);
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
        public void test_ndarray_moveaxis_STRING_TODO()
        {
            var x = np.zeros((3, 4, 5), np.Object);
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
        public void test_indices_1_STRING_TODO()
        {
            var grid = np.indices((2, 3), dtype: np.Int32);
            AssertShape(grid, 2, 2, 3);
            print(grid.shape);
            AssertArray(grid[0] as ndarray, new Object[,] { { 0, 0, 0 }, { 1, 1, 1 } });
            print(grid[0]);
            AssertArray(grid[1] as ndarray, new Object[,] { { 0, 1, 2 }, { 0, 1, 2 } });
            print(grid[1]);

            var x = np.arange(20, dtype: np.Int32).reshape((5, 4)).astype(np.Object);

            var y = x[grid[0], grid[1]];
            AssertArray(y as ndarray, new Object[,] { { 0, 1, 2 }, { 4, 5, 6 } });
            print(y);

            return;
        }

        [TestMethod]
        public void test_isscalar_1_STRING_TODO()
        {

            bool a = np.isscalar((Object)3);
            Assert.AreEqual(true, a);
            print(a);

            bool b = np.isscalar(np.array((Object)3));
            Assert.AreEqual(false, b);
            print(b);

            bool c = np.isscalar(new Object[] { 3 });
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
        public void test_identity_1_STRING_TODO()
        {
            ndarray a = np.identity(2, dtype: np.Int32).astype(np.Object);

            print(a);
            print(a.shape);
            print(a.strides);

            var ExpectedDataA = new Object[2, 2]
            {
                { 1,0 },
                { 0,1 },
            };
            AssertArray(a, ExpectedDataA);
            AssertShape(a, 2, 2);
            AssertStrides(a, SizeOfString * 2, SizeOfString * 1);

            ndarray b = np.identity(5, dtype: np.Object);

            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new Int32[5, 5]
            {
                { 1, 0, 0, 0, 0 },
                { 0, 1, 0, 0, 0 },
                { 0, 0, 1, 0, 0 },
                { 0, 0, 0, 1, 0 },
                { 0, 0, 0, 0, 1 },
            };
            //AssertArray(b, ExpectedDataB);
            AssertShape(b, 5, 5);
            AssertStrides(b, SizeOfString * 5, SizeOfString * 1);
        }


        [TestMethod]
        public void test_array_equal_1_STRING_TODO()
        {
            var a = np.array_equal(new Object[] { 1, 2 }, new Object[] { 1, 2 });
            Assert.AreEqual(true, a);
            print(a);

            var b = np.array_equal(np.array(new Object[] { 1, 2 }), np.array(new Object[] { 1, 2 }));
            Assert.AreEqual(true, b);
            print(b);

            var c = np.array_equal(new Object[] { 1, 2 }, new Object[] { 1, 2, 3 });
            Assert.AreEqual(false, c);
            print(c);

            var d = np.array_equal(new Object[] { 1, 2 }, new Object[] { 1, 4 });
            Assert.AreEqual(false, d);
            print(d);
        }

        [TestMethod]
        public void test_array_equiv_1_STRING_TODO()
        {
            var a = np.array_equiv(new Object[] { 1, 2 }, new Object[] { 1, 2 });
            Assert.AreEqual(true, a);
            print(a);

            var b = np.array_equiv(new Object[] { 1, 2 }, new Object[] { 1, 3 });
            Assert.AreEqual(false, b);
            print(b);

            var c = np.array_equiv(new Object[] { 1, 2 }, new Object[,] { { 1, 2 }, { 1, 2 } });
            Assert.AreEqual(true, c);
            print(c);

            var d = np.array_equiv(new Object[] { 1, 2 }, new Object[,] { { 1, 2, 1, 2 }, { 1, 2, 1, 2 } });
            Assert.AreEqual(false, d);
            print(d);

            var e = np.array_equiv(new Object[] { 1, 2 }, new Object[,] { { 1, 2 }, { 1, 3 } });
            Assert.AreEqual(false, e);
            print(e);
        }

        #endregion

        #region from NANFunctionsTests

        [TestMethod]
        public void test_nanprod_1_STRING_TODO()
        {

            var x = np.nanprod((Object)1);
            Assert.AreEqual((Int64)1, x.GetItem(0));
            print(x);

            var y = np.nanprod(new Object[] { 1 });
            Assert.AreEqual((Int64)1, y.GetItem(0));
            print(y);



            var a = np.array(new Object[,] { { 1, 2 }, { 3, 4 } });
            var b = np.nanprod(a);
            Assert.AreEqual((Int64)24.0, b.GetItem(0));
            print(b);

            var c = np.nanprod(a, axis: 0);
            AssertArray(c, new Int64[] { 3, 8 });
            print(c);

            var d = np.nanprod(a, axis: 1);
            AssertArray(d, new Int64[] { 2, 12 });
            print(d);

            return;
        }

        #endregion

        #region from StatisticsTests

        [TestMethod]
        public void test_amin_2_STRING_TODO()
        {
            ndarray a = np.arange(30, 46).reshape(new shape(4, 4)).astype(np.Object);
            print(a);
            print("*****");

            ndarray b = np.amin(a);          // Minimum of the flattened array
            print(b);
            Assert.AreEqual(30, b.GetItem(0));
            print("*****");

            ndarray c = np.amin(a, axis: 0);  // Minimum along the first axis
            print(c);
            AssertArray(c, new Int32[] { 30, 31, 32, 33 });
            print("*****");

            ndarray d = np.amin(a, axis: 1);   // Minimum along the second axis
            print(d);
            AssertArray(d, new Int32[] { 30, 34, 38, 42 });
            print("*****");

            // Object don't support NAN
            //ndarray e = np.arange(5, dtype: np.Decimal);
            //e[2] = np.NaN;
            //ndarray f = np.amin(e);
            //print(f);
            //Assert.AreEqual(np.NaN, f.GetItem(0));
            //print("*****");

        }

        [TestMethod]
        public void test_amax_2_STRING_TODO()
        {
            ndarray a = np.arange(30, 46).reshape(new shape(4, 4)).astype(np.Object);
            print(a);
            print("*****");

            ndarray b = np.amax(a);          // Maximum of the flattened array
            print(b);
            Assert.AreEqual(45, b.GetItem(0));
            print("*****");

            ndarray c = np.amax(a, axis: 0);  // Maxima along the first axis
            print(c);
            AssertArray(c, new Int32[] { 42, 43, 44, 45 });
            print("*****");

            ndarray d = np.amax(a, axis: 1);   // Maxima along the second axis
            print(d);
            AssertArray(d, new Int32[] { 33, 37, 41, 45 });
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
        public void test_ptp_1_STRING_TODO()
        {
            ndarray a = np.arange(4, dtype: np.Int32).reshape(new shape(2, 2)).astype(np.Object);
            print(a);
            print("*****");

            ndarray b = np.ptp(a, axis: 0);
            print(b);
            AssertArray(b, new int[] { 2, 2 });
            print("*****");

            ndarray c = np.ptp(a, axis: 1);
            print(c);
            AssertArray(c, new int[] { 1, 1 });

            ndarray d = np.ptp(a);
            print(d);
            Assert.AreEqual((int)3, d.GetItem(0));
        }

        [TestMethod]
        public void test_percentile_2_STRING_TODO()
        {
            var a = np.array(new Object[,] { { 10, 7, 4 }, { 3, 2, 1 } });

            var b = np.percentile(a, new Object[] { 50, 75 });
            AssertArray(b, new double[] { 3.5, 6.25 });
            print(b);

            var c = np.percentile(a, new Object[] { 50, 75 }, axis: 0);
            AssertArray(c, new double[,] { { 6.5, 4.5, 2.5 }, { 8.25, 5.75, 3.25 } });
            print(c);

            var d = np.percentile(a, new Object[] { 50, 75 }, axis: 1);
            AssertArray(d, new double[,] { { 7.0, 2.0 }, { 8.5, 2.5 } });
            print(d);

            var e = np.percentile(a, new Object[] { 50, 75 }, axis: 1, keepdims: true);
            AssertArray(e, new double[,,] { { { 7.0 }, { 2.0 } }, { { 8.5 }, { 2.5 } } });
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
        public void test_quantile_2_STRING_TODO()
        {
            var a = np.array(new Object[,] { { 10, 7, 4 }, { 3, 2, 1 } });

            var b = np.quantile(a, new double[] { 0.5, 0.75 });
            AssertArray(b, new double[] { 3.5, 6.25 });
            print(b);

            var c = np.quantile(a, new double[] { 0.5, 0.75 }, axis: 0);
            AssertArray(c, new double[,] { { 6.5, 4.5, 2.5 }, { 8.25, 5.75, 3.25 } });
            print(c);

            var d = np.quantile(a, new double[] { 0.5, 0.75 }, axis: 1);
            AssertArray(d, new double[,] { { 7.0, 2.0 }, { 8.5, 2.5 } });
            print(d);

            var e = np.quantile(a, new double[] { 0.5, 0.75 }, axis: 1, keepdims: true);
            AssertArray(e, new double[,,] { { { 7.0 }, { 2.0 } }, { { 8.5 }, { 2.5 } } });
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
        public void test_median_2_STRING_TODO()
        {
            var a = np.arange(0, 64, 1, np.Int32).reshape((4, 4, 4)).astype(np.Object);

            var b = np.median(a, axis: new int[] { 0, 2 }, keepdims: true);
            AssertArray(b, new double[,,] { { { 25 }, { 29 }, { 33 }, { 37 } } });
            print(b);

            var c = np.median(a, new int[] { 0, 1 }, keepdims: true);
            AssertArray(c, new double[,,] { { { 30, 31, 32, 33 } } });
            print(c);

            var d = np.median(a, new int[] { 1, 2 }, keepdims: true);
            AssertArray(d, new double[,,] { { { 7 } }, { { 23 } }, { { 39 } }, { { 55 } } });
            print(d);

            return;
        }

        [TestMethod]
        public void test_average_3_STRING_TODO()
        {

            var a = np.array(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }).astype(np.Object);
            var w = new int[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

            var x1 = np.average(a, axis: null, weights: null, returned: true);
            Assert.AreEqual(5.5, x1.retval.GetItem(0));
            Assert.AreEqual((npy_intp)10.0, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a, axis: null, weights: w, returned: true);
            Assert.AreEqual(4.0, x1.retval.GetItem(0));
            Assert.AreEqual(55, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: null, weights: np.array(w).reshape((2, -1)), returned: true);
            Assert.AreEqual(4.0, x1.retval.GetItem(0));
            Assert.AreEqual(55, x1.sum_of_weights.GetItem(0));
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: 0, weights: np.array(w).reshape((2, -1)), returned: true);
            AssertArray(x1.retval, new double[] { 2.66666666666667, 3.53846153846154, 4.36363636363636, 5.11111111111111, 5.71428571428571 });
            AssertArray(x1.sum_of_weights, new Int32[] { 15, 13, 11, 9, 7 });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1)), axis: 1, weights: np.array(w).reshape((2, -1)), returned: true);
            AssertArray(x1.retval, new double[] { 2.75, 7.33333333333333 });
            AssertArray(x1.sum_of_weights, new Int32[] { 40, 15 });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((1, 2, -1, 1)), axis: 1, weights: np.array(w).reshape((1, 2, -1, 1)), returned: true);
            AssertArray(x1.retval, new double[,,] { { { 2.66666666666667 }, { 3.53846153846154 }, { 4.36363636363636 }, { 5.11111111111111 }, { 5.71428571428571 } } });
            AssertArray(x1.sum_of_weights, new Int32[,,] { { { 15 }, { 13 }, { 11 }, { 9 }, { 7 } } });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((1, -1, 2, 1)), axis: 1, weights: np.array(w).reshape((1, -1, 2, 1)), returned: true);
            AssertArray(x1.retval, new double[,,] { { { 3.66666666666667 }, { 4.4 } } });
            AssertArray(x1.sum_of_weights, new Int32[,,] { { { 30 }, { 25 } } });
            print(x1);
            print("********");

            x1 = np.average(a.reshape((2, -1, 1, 1)), axis: 1, weights: np.array(w).reshape((2, -1, 1, 1)), returned: false);
            AssertArray(x1.retval, new double[,,] { { { 2.75 } }, { { 7.33333333333333 } } });
            Assert.AreEqual(null, x1.sum_of_weights);
            print(x1);
        }

        [TestMethod]
        public void test_mean_1_STRING_TODO()
        {
            Int32[] TestData = new Int32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.Int32).reshape(new shape(3, 2, -1)).astype(np.Object);
            x = x * 3;
            print(x);

            var y = np.mean(x);
            print(y);
            Assert.AreEqual((double)131, y.GetItem(0));

            y = np.mean(x, axis: 0);
            print(y);
            AssertArray(y, new double[,] { { 113, 150 }, { 113, 150 } });

            y = np.mean(x, axis: 1);
            print(y);
            AssertArray(y, new double[,] { { 52, 90 }, { 132, 157 }, { 154, 202 } });

            y = np.mean(x, axis: 2);
            print(y);
            AssertArray(y, new double[,] { { 37, 105 }, { 252, 37 }, { 105, 252 } });

        }

        [TestMethod]
        public void test_mean_2_STRING_TODO()
        {
            ndarray a = np.zeros(new shape(2, 512 * 512), dtype: np.Int32).astype(np.Object);
            a[0, ":"] = 1.0;
            a[1, ":"] = 0.1;
            ndarray b = np.mean(a);
            print(b);
            Assert.AreEqual(0.54999999998835847, (double)b.GetItem(0));

            ndarray c = np.mean(a, dtype: np.Object);
            print(c);
            Assert.AreEqual(0.54999999998835847, c.GetItem(0));
        }

        [TestMethod]
        public void test_std_1_STRING_TODO()
        {
            ndarray a = np.array(new Int32[,] { { 1, 2 }, { 3, 4 } }).astype(np.Object);
            ndarray b = np.std(a);
            print(b);
            Assert.AreEqual(1.2247448713915889, (double)b.GetItem(0));

            ndarray c = np.std(a, axis: 0);
            print(c);
            AssertArray(c, new double[] { 1.0, 1.0 });

            ndarray d = np.std(a, axis: 1);
            print(d);
            AssertArray(d, new double[] { 0.70710678118654757, 1.5811388300841898 });

            // In single precision, std() can be inaccurate:
            a = np.zeros(new shape(2, 512 * 512), dtype: np.Int32).astype(np.Object);
            a[0, ":"] = 1;
            a[1, ":"] = 0;
            b = np.std(a);
            print(b);
            Assert.AreEqual(0.70710678118654757, b.GetItem(0));
            // Computing the standard deviation in float64 is more accurate:
            c = np.std(a);
            print(c);
            Assert.AreEqual(0.70710678118654757, c.GetItem(0));

        }

        [TestMethod]
        public void test_var_1_STRING_TODO()
        {
            ndarray a = np.array(new Object[,] { { 1, 2 }, { 3, 4 } });
            ndarray b = np.var(a);
            Assert.AreEqual(1.25, b.GetItem(0));
            print(b);

            ndarray c = np.var(a, axis: 0);
            AssertArray(c, new double[] { 1.0, 1.0 });
            print(c);

            ndarray d = np.var(a, axis: 1);
            AssertArray(d, new double[] { 1.25, 1.25 });
            print(d);

            // In single precision, std() can be inaccurate:
            a = np.zeros(new shape(2, 512 * 512), dtype: np.Object);
            a[0, ":"] = 1;
            a[1, ":"] = 0;
            b = np.var(a);
            Assert.AreEqual(0.5, b.GetItem(0));
            print(b);


        }

        [TestMethod]
        public void test_corrcoef_1_STRING_TODO()
        {
            var x1 = np.array(new Object[,] { { 0, 2 }, { 1, 1 }, { 2, 0 } }).T;
            print(x1);

            // Note how  increases while  decreases. The covariance matrix shows this clearly:

            var a = np.corrcoef(x1);
            AssertArray(a, new double[,] { { 1, -1 }, { -1, 1 } });
            print(a);

            var x = new Object[] { -21, -1, 43 };
            var y = new Object[] { 3, 11, 12 };
            var X = np.stack(new object[] { x, y }, axis: 0);
            a = np.corrcoef(X);
            AssertArray(a, new double[,] { { 1.0, 0.804905985486053 }, { 0.804905985486053, 1.0 } });
            print(a);


            var b = np.corrcoef(x, y);
            AssertArray(b, new double[,] { { 1.0, 0.804905985486053 }, { 0.804905985486053, 1.0 } });
            print(b);

            var c = np.corrcoef(x, y, rowvar: false);
            AssertArray(c, new double[,] { { 1.0, 0.804905985486053 }, { 0.804905985486053, 1.0 } });
            print(c);


            return;
        }

        [TestMethod]
        public void test_correlate_1_STRING_TODO()
        {
            var a = np.correlate(new Object[] { 1, 2, 3 }, new Object[] { 0, 1, 5 });
            AssertArray(a, new Object[] { 17 });
            print(a);

            var b = np.correlate(new Object[] { 1, 2, 3 }, new Object[] { 0, 1, 5 }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_SAME);
            AssertArray(b, new Object[] { 11, 17, 3 });
            print(b);

            var c = np.correlate(new Object[] { 1, 2, 3 }, new Object[] { 0, 1, 5 }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_FULL);
            AssertArray(c, new Object[] { 5, 11, 17, 3, 0 });
            print(c);

            return;
        }


        [TestMethod]
        public void test_cov_1_STRING_TODO()
        {
            var x1 = np.array(new Object[,] { { 0, 2 }, { 1, 1 }, { 2, 0 } }).T;
            print(x1);

            // Note how  increases while  decreases. The covariance matrix shows this clearly:

            var a = np.cov(x1);
            AssertArray(a, new double[,] { { 1, -1 }, { -1, 1 } });
            print(a);

            var x = new Object[] { -21, -1, 43 };
            var y = new Object[] { 3, 11, 12 };
            var X = np.stack(new object[] { x, y }, axis: 0);
            a = np.cov(X);
            AssertArray(a, new double[,] { { 1072.0, 130.0 }, { 130.0, 24.3333333333333 } });
            print(a);


            var b = np.cov(x, y);
            AssertArray(b, new double[,] { { 1072.0, 130.0 }, { 130.0, 24.3333333333333 } });
            print(b);

            var c = np.cov(x);
            Assert.AreEqual(1072.0, c.GetItem(0));
            print(c);

            var d = np.cov(X, rowvar: false);
            AssertArray(d, new double[,] { { 288.0, 144.0, -372.0 }, { 144.0, 72.0, -186.0 }, { -372.0, -186.0, 480.5 } });
            print(d);

            var e = np.cov(X, rowvar: false, bias: true);
            AssertArray(e, new double[,] { { 144.0, 72.0, -186.0 }, { 72.0, 36.0, -93.0 }, { -186.0, -93.0, 240.25 } });
            print(e);

            var f = np.cov(X, rowvar: false, bias: true, fweights: new int[] { 1, 2 });
            AssertArray(f, new double[,] { { 128.0, 64.0, -165.333333333333 },
                                            { 64.0, 32.0, -82.6666666666667 },
                                            { -165.333333333333, -82.6666666666667, 213.555555555556 }});
            print(f);

            var g = np.cov(X, rowvar: false, bias: true, fweights: new int[] { 1, 2 }, aweights: new int[] { 1, 2 });
            AssertArray(g, new double[,] { { 92.16, 46.08, -119.04 }, { 46.08, 23.04, -59.52 }, { -119.04, -59.52, 153.76 } });
            print(g);

            return;
        }

        #endregion

        #region from TwoDimBaseTests

        [TestMethod]
        public void test_diag_1_STRING_TODO()
        {
            ndarray m = np.arange(9, dtype: np.Float64).astype(np.Object);
            var n = np.diag(m);

            print(m);
            print(n);

            var ExpectedDataN = new double[,]
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

            m = np.arange(9, dtype: np.Float64).reshape(new shape(3, 3)).astype(np.Object);
            n = np.diag(m);

            print(m);
            print(n);
            AssertArray(n, new double[] { 0, 4, 8 });
        }

        [TestMethod]
        public void test_diagflat_1_STRING_TODO()
        {
            ndarray m = np.arange(1, 5, dtype: np.Float64).reshape(new shape(2, 2)).astype(np.Object);
            var n = np.diagflat(m);

            print(m);
            print(n);

            var ExpectedDataN = new double[,]
            {
             {1, 0, 0, 0},
             {0, 2, 0, 0},
             {0, 0, 3, 0},
             {0, 0, 0, 4}
            };
            AssertArray(n, ExpectedDataN);

            m = np.arange(1, 3, dtype: np.Float64).astype(np.Object);
            n = np.diagflat(m, 1);

            print(m);
            print(n);

            ExpectedDataN = new double[,]
            {
             {0, 1, 0},
             {0, 0, 2},
             {0, 0, 0},
            };

            AssertArray(n, ExpectedDataN);

            m = np.arange(1, 3, dtype: np.Float64).astype(np.Object);
            n = np.diagflat(m, -1);

            print(m);
            print(n);

            ExpectedDataN = new double[,]
            {
             {0, 0, 0},
             {1, 0, 0},
             {0, 2, 0},
            };

            AssertArray(n, ExpectedDataN);

        }

        [TestMethod]
        public void test_fliplr_1_STRING_TODO()
        {
            ndarray m = np.arange(8, dtype: np.Int32).reshape(new shape(2, 2, 2)).astype(np.Object);
            var n = np.fliplr(m);

            print(m);
            print(n);

            AssertArray(n, new Object[,,] { { { 2, 3 }, { 0, 1 } }, { { 6, 7 }, { 4, 5 } } });
        }

        [TestMethod]
        public void test_flipud_1_STRING_TODO()
        {
            ndarray m = np.arange(8, dtype: np.Int32).reshape(new shape(2, 2, 2)).astype(np.Object);
            var n = np.flipud(m);

            print(m);
            print(n);

            AssertArray(n, new Object[,,] { { { 4, 5 }, { 6, 7 } }, { { 0, 1 }, { 2, 3 } } });
        }

        [TestMethod]
        public void test_tri_1_STRING_TODO()
        {
            ndarray a = np.tri(3, 5, 2, dtype: np.Object);
            print(a);

            var ExpectedDataA = new Object[,]
            {
             {1, 1, 1, 0, 0},
             {1, 1, 1, 1, 0},
             {1, 1, 1, 1, 1}
            };
            AssertArray(a, ExpectedDataA);

            print("***********");
            ndarray b = np.tri(3, 5, -1, dtype: np.Object);
            print(b);

            var ExpectedDataB = new Object[,]
            {
             {0, 0, 0, 0, 0},
             {1, 0, 0, 0, 0},
             {1, 1, 0, 0, 0}
            };
            AssertArray(b, ExpectedDataB);
        }

        [TestMethod]
        public void test_tril_1_STRING_TODO()
        {
            ndarray a = np.array(new Object[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } });
            ndarray b = np.tril(a, -1);
            print(a);
            print("***********");
            print(b);

            var ExpectedDataB = new Object[,]
            {
             {0, 0, 0},
             {4, 0, 0},
             {7, 8, 0},
             {10, 11, 12},
            };
            AssertArray(b, ExpectedDataB);

        }

        [TestMethod]
        public void test_triu_1_STRING_TODO()
        {
            ndarray a = np.array(new Object[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } });
            ndarray b = np.triu(a, -1);
            print(a);
            print("***********");
            print(b);

            var ExpectedDataB = new Object[,]
            {
             {1, 2, 3},
             {4, 5, 6},
             {0, 8, 9},
             {0, 0, 12},
            };
            AssertArray(b, ExpectedDataB);

        }

        [TestMethod]
        public void test_vander_1_STRING_TODO()
        {
            var x = np.array(new Object[] { 1, 2, 3, 5 });
            int N = 3;
            var y = np.vander(x, N);
            AssertArray(y, new Object[,] { { 1, 1, 1 }, { 4, 2, 1 }, { 9, 3, 1 }, { 25, 5, 1 } });
            print(y);

            y = np.vander(x);
            AssertArray(y, new Object[,] { { 1, 1, 1, 1 }, { 8, 4, 2, 1 }, { 27, 9, 3, 1 }, { 125, 25, 5, 1 } });
            print(y);

            y = np.vander(x, increasing: true);
            AssertArray(y, new Object[,] { { 1, 1, 1, 1 }, { 1, 2, 4, 8 }, { 1, 3, 9, 27 }, { 1, 5, 25, 125 } });
            print(y);

            return;
        }

        [TestMethod]
        public void test_mask_indices_STRING_TODO()
        {
            var iu = np.mask_indices(3, np.triu);
            AssertArray(iu[0], new npy_intp[] { 0, 0, 0, 1, 1, 2 });
            AssertArray(iu[1], new npy_intp[] { 0, 1, 2, 1, 2, 2 });
            print(iu);

            var a = np.arange(9, dtype: np.Int32).reshape((3, 3)).astype(np.Object);
            var b = a[iu] as ndarray;
            AssertArray(b, new Object[] { 0, 1, 2, 4, 5, 8 });
            print(b);

            var iu1 = np.mask_indices(3, np.triu, 1);

            var c = a[iu1] as ndarray;
            AssertArray(c, new Object[] { 1, 2, 5 });
            print(c);

            return;
        }

        [TestMethod]
        public void test_tril_indices_STRING_TODO()
        {
            var il1 = np.tril_indices(4);
            var il2 = np.tril_indices(4, 2);

            var a = np.arange(16, dtype: np.Int32).reshape((4, 4)).astype(np.Object);
            var b = a[il1] as ndarray;
            AssertArray(b, new Object[] { 0, 4, 5, 8, 9, 10, 12, 13, 14, 15 });
            print(b);

            a[il1] = -1;

            var ExpectedDataA1 = new Object[,]
                {{-1,  1, 2,  3}, {-1, -1,  6,  7},
                 {-1, -1,-1, 11}, {-1, -1, -1, -1}};
            AssertArray(a, ExpectedDataA1);
            print(a);

            a[il2] = -10;

            var ExpectedDataA2 = new Object[,]
                {{-10, -10, -10,  3}, {-10, -10, -10, -10},
                 {-10, -10,-10, -10}, {-10, -10, -10, -10}};
            AssertArray(a, ExpectedDataA2);
            print(a);

            return;
        }

        [TestMethod]
        public void test_tril_indices_from_STRING_TODO()
        {
            var a = np.arange(16, dtype: np.Int32).reshape((4, 4)).astype(np.Object);
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
        public void test_triu_indices_STRING_TODO()
        {
            var il1 = np.triu_indices(4);
            var il2 = np.triu_indices(4, 2);

            var a = np.arange(16, dtype: np.Int32).reshape((4, 4)).astype(np.Object);
            var b = a[il1] as ndarray;
            AssertArray(b, new Object[] { 0, 1, 2, 3, 5, 6, 7, 10, 11, 15 });
            print(b);

            a[il1] = -1;

            var ExpectedDataA1 = new Object[,]
                {{-1, -1, -1, -1}, { 4, -1, -1, -1},
                 { 8,  9, -1, -1}, {12, 13, 14, -1}};
            AssertArray(a, ExpectedDataA1);
            print(a);

            a[il2] = -10;

            var ExpectedDataA2 = new Object[,]
                {{-1, -1, -10, -10}, {4,  -1, -1, -10},
                 { 8,  9, -1,  -1},  {12, 13, 14, -1}};
            AssertArray(a, ExpectedDataA2);
            print(a);

            return;
        }

        [TestMethod]
        public void test_triu_indices_from_STRING_TODO()
        {
            var a = np.arange(16, dtype: np.Int32).reshape((4, 4)).astype(np.Object);
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
        public void test_atleast_1d_STRING_TODO()
        {
            var a = np.atleast_1d((Object)1);
            print(a);
            AssertArray(a.ElementAt(0), new Object[] { 1 });

            print("**************");
            var x = np.arange(9.0, dtype: np.Int32).reshape(new shape(3, 3)).astype(np.Object);
            var b = np.atleast_1d(x);
            print(b);

            var ExpectedB = new Object[,]
                {{0, 1, 2},
                 {3, 4, 5},
                 {6, 7, 8}};
            AssertArray(b.ElementAt(0), ExpectedB);

            print("**************");

            var c = np.atleast_1d(new object[] { (Object)1, new Object[] { 3, 4 } });

            AssertArray(c.ElementAt(0), new Object[] { 1 });
            AssertArray(c.ElementAt(1), new Object[] { 3, 4 });
            print(c);

        }

        [TestMethod]
        public void test_atleast_2d_STRING_TODO()
        {
            var a = np.atleast_2d((Object)1);
            print(a);
            AssertArray(a.ElementAt(0), new Object[,] { { 1 } });

            print("**************");
            var x = np.arange(9.0, dtype: np.Int32).reshape(new shape(3, 3)).astype(np.Object);
            var b = np.atleast_2d(x);
            print(b);

            var ExpectedB = new Object[,]
                {{0, 1, 2},
                 {3, 4, 5},
                 {6, 7, 8}};
            AssertArray(b.ElementAt(0), ExpectedB);

            print("**************");

            var c = np.atleast_2d(new object[] { (Object)1, new Object[] { 3, 4 }, new Object[] { 5, 6 } });

            AssertArray(c.ElementAt(0), new Object[,] { { 1 } });
            AssertArray(c.ElementAt(1), new Object[,] { { 3, 4 } });
            AssertArray(c.ElementAt(2), new Object[,] { { 5, 6 } });
            print(c);

        }

        [TestMethod]
        public void test_atleast_3d_STRING_TODO()
        {
            var a = np.atleast_3d((Object)1);
            print(a);
            AssertArray(a.ElementAt(0), new Object[,,] { { { 1 } } });

            print("**************");
            var x = np.arange(9.0, dtype: np.Int32).reshape(new shape(3, 3)).astype(np.Object);
            var b = np.atleast_3d(x);
            print(b);

            var ExpectedB = new Object[,,]
             {{{0},
               {1},
               {2}},
              {{3},
               {4},
               {5}},
              {{6},
               {7},
               {8}}};

            AssertArray(b.ElementAt(0), ExpectedB);

            print("**************");

            var c = np.atleast_3d(new object[] { new Object[] { 1, 2 }, new Object[] { 3, 4 }, new Object[] { 5, 6 } });

            AssertArray(c.ElementAt(0), new Object[,,] { { { 1 }, { 2 } } });
            AssertArray(c.ElementAt(1), new Object[,,] { { { 3 }, { 4 } } });
            AssertArray(c.ElementAt(2), new Object[,,] { { { 5 }, { 6 } } });
            print(c);


        }

        [TestMethod]
        public void test_vstack_2_STRING_TODO()
        {
            var a = np.array(new Object[,] { { 1 }, { 2 }, { 3 } });
            var b = np.array(new Object[,] { { 2 }, { 3 }, { 4 } });
            var c = np.vstack(new object[] { a, b });

            AssertArray(c, new Object[,] { { 1 }, { 2 }, { 3 }, { 2 }, { 3 }, { 4 } });

            print(c);
        }

        [TestMethod]
        public void test_hstack_2_STRING_TODO()
        {
            var a = np.array(new Object[,] { { 1 }, { 2 }, { 3 } });
            var b = np.array(new Object[,] { { 2 }, { 3 }, { 4 } });
            var c = np.hstack(new object[] { a, b });

            AssertArray(c, new Object[,] { { 1, 2 }, { 2, 3 }, { 3, 4 } });

            print(c);
        }

        [TestMethod]
        public void test_stack_1_STRING_TODO()
        {
            var a = np.array(new Object[,] { { 1 }, { 2 }, { 3 } });
            var b = np.array(new Object[,] { { 2 }, { 3 }, { 4 } });

            var c = np.stack(new object[] { a, b }, axis: 0);
            AssertArray(c, new Object[,,] { { { 1 }, { 2 }, { 3 } }, { { 2 }, { 3 }, { 4 } } });
            print(c);
            print("**************");

            var d = np.stack(new object[] { a, b }, axis: 1);
            AssertArray(d, new Object[,,] { { { 1 }, { 2 } }, { { 2 }, { 3 } }, { { 3 }, { 4 } } });
            print(d);
            print("**************");

            var e = np.stack(new object[] { a, b }, axis: 2);
            AssertArray(e, new Object[,,] { { { 1, 2 } }, { { 2, 3 } }, { { 3, 4 } } });
            print(e);

        }

        [TestMethod]
        public void test_block_2_STRING_TODO()
        {
            var a = np.array(new Object[] { 1, 2, 3 });
            var b = np.array(new Object[] { 2, 3, 4 });
            var c = np.block(new object[] { a, b, 10 });    // hstack([a, b, 10])

            AssertArray(c, new Object[] { 1, 2, 3, 2, 3, 4, 10 });
            print(c);
            print("**************");

            a = np.array(new Object[] { 1, 2, 3 });
            b = np.array(new Object[] { 2, 3, 4 });
            c = np.block(new object[] { new object[] { a }, new object[] { b } });    // vstack([a, b])

            AssertArray(c, new Object[,] { { 1, 2, 3 }, { 2, 3, 4 } });
            print(c);

        }

        [TestMethod]
        public void test_expand_dims_1_STRING_TODO()
        {
            var a = np.array(new Object[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }).reshape(new shape(2, -1, 2));
            var b = np.expand_dims(a, axis: 0);

            var ExpectedDataB = new Object[,,,]
            {{{{1,  2}, {3,  4}, {5,  6}},
              {{7,  8}, {9, 10}, {11, 12}}}};

            AssertArray(b, ExpectedDataB);
            print(b);
            print("**************");

            var c = np.expand_dims(a, axis: 1);
            var ExpectedDataC = new Object[,,,]
                {{{{1,  2}, {3,  4}, {5,  6}}},
                {{{ 7,  8},{ 9, 10}, {11, 12}}}};
            AssertArray(c, ExpectedDataC);
            print(c);
            print("**************");

            var d = np.expand_dims(a, axis: 2);
            var ExpectedDataD = new Object[,,,]
            {{{{1,  2}},{{3,  4}},{{5,  6}}},
             {{{7,  8}},{{9, 10}},{{11, 12}}}};

            AssertArray(d, ExpectedDataD);
            print(d);

        }

        [TestMethod]
        public void test_column_stack_1_STRING_TODO()
        {
            var a = np.array(new Object[] { 1, 2, 3 });
            var b = np.array(new Object[] { 2, 3, 4 });
            var c = np.column_stack(new object[] { a, b });

            AssertArray(c, new Object[,] { { 1, 2 }, { 2, 3 }, { 3, 4 } });
            print(c);
        }

        [TestMethod]
        public void test_row_stack_1_STRING_TODO()
        {
            var a = np.array(new Object[] { 1, 2, 3 });
            var b = np.array(new Object[] { 2, 3, 4 });
            var c = np.row_stack(new object[] { a, b });

            AssertArray(c, new Object[,] { { 1, 2, 3 }, { 2, 3, 4 } });

            print(c);
        }

        [TestMethod]
        public void test_dstack_1_STRING_TODO()
        {
            var a = np.array(new Object[] { 1, 2, 3 });
            var b = np.array(new Object[] { 2, 3, 4 });
            var c = np.dstack(new object[] { a, b });

            AssertArray(c, new Object[,,] { { { 1, 2 }, { 2, 3 }, { 3, 4 } } });
            print(c);

            a = np.array(new Object[,] { { 1 }, { 2 }, { 3 } });
            b = np.array(new Object[,] { { 2 }, { 3 }, { 4 } });
            c = np.dstack(new object[] { a, b });

            AssertArray(c, new Object[,,] { { { 1, 2 } }, { { 2, 3 } }, { { 3, 4 } } });

            print(c);
        }

        [TestMethod]
        public void test_array_split_2_STRING_TODO()
        {
            var x = np.arange(16.0, dtype: np.Int32).reshape(new shape(2, 8, 1)).astype(np.Object);
            var y = np.array_split(x, 3, axis: 0);


            AssertArray(y.ElementAt(0), new Object[,,] { { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 } } });
            AssertArray(y.ElementAt(1), new Object[,,] { { { 8 }, { 9 }, { 10 }, { 11 }, { 12 }, { 13 }, { 14 }, { 15 } } });
            AssertShape(y.ElementAt(2), 0, 8, 1);

            print(y);

            print("**************");

            x = np.arange(16.0, dtype: np.Int32).reshape(new shape(2, 8, 1)).astype(np.Object);
            y = np.array_split(x, 3, axis: 1);

            AssertArray(y.ElementAt(0), new Object[,,] { { { 0 }, { 1 }, { 2 } }, { { 8 }, { 9 }, { 10 } } });
            AssertArray(y.ElementAt(1), new Object[,,] { { { 3 }, { 4 }, { 5 } }, { { 11 }, { 12 }, { 13 } } });
            AssertArray(y.ElementAt(2), new Object[,,] { { { 6 }, { 7 } }, { { 14 }, { 15 } } });


            print(y);

            print("**************");

            x = np.arange(16.0, dtype: np.Int32).reshape(new shape(2, 8, 1)).astype(np.Object);
            y = np.array_split(x, 3, axis: 2);

            AssertArray(y.ElementAt(0), new Object[,,] { { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 } }, { { 8 }, { 9 }, { 10 }, { 11 }, { 12 }, { 13 }, { 14 }, { 15 } } });
            AssertShape(y.ElementAt(1), 2, 8, 0);
            AssertShape(y.ElementAt(2), 2, 8, 0);
            print(y);
        }

        [TestMethod]
        public void test_split_2_STRING_TODO()
        {
            var x = np.arange(16.0, dtype: np.Int32).reshape(new shape(8, 2, 1)).astype(np.Object);
            var y = np.split(x, new Int32[] { 2, 3 }, axis: 0);

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new Object[,,] { { { 0 }, { 1 } }, { { 2 }, { 3 } } });
            AssertArray(y.ElementAt(1), new Object[,,] { { { 4 }, { 5 } } });
            AssertArray(y.ElementAt(2), new Object[,,] { { { 6 }, { 7 } }, { { 8 }, { 9 } }, { { 10 }, { 11 } }, { { 12 }, { 13 } }, { { 14 }, { 15 } } });


            print(y);

            print("**************");

            x = np.arange(16.0, dtype: np.Int32).reshape(new shape(8, 2, 1)).astype(np.Object);
            y = np.split(x, new int[] { 2, 3 }, axis: 1);

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new Object[,,] {{{0},{1}},{{2}, {3}}, {{4}, {5}}, {{6}, { 7}},
                                                        {{8},{9}},{{10},{11}}, {{12}, {13}}, {{14}, {15}}});
            AssertShape(y.ElementAt(1), 8, 0, 1);
            AssertShape(y.ElementAt(2), 8, 0, 1);

            print(y);

            print("**************");

            x = np.arange(16.0, dtype: np.Int32).reshape(new shape(8, 2, 1)).astype(np.Object);
            y = np.split(x, new int[] { 2, 3 }, axis: 2);

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new Object[,,] {{{ 0},{ 1}},{{ 2}, { 3}}, {{ 4}, { 5}}, {{ 6}, { 7}},
                                                        {{ 8},{ 9}},{{10}, {11}}, {{12}, {13}}, {{14}, {15}}});
            AssertShape(y.ElementAt(1), 8, 2, 0);
            AssertShape(y.ElementAt(2), 8, 2, 0);

            print(y);
        }

        [TestMethod]
        public void test_hsplit_2_STRING_TODO()
        {
            var x = np.arange(8, dtype: np.Int32).reshape(new shape(2, 2, 2)).astype(np.Object);
            var y = np.hsplit(x, 2);

            Assert.AreEqual(2, y.Count);
            AssertArray(y.ElementAt(0), new Object[,,] { { { 0, 1 } }, { { 4, 5 } } });
            AssertArray(y.ElementAt(1), new Object[,,] { { { 2, 3 } }, { { 6, 7 } } });
            print(y);

            print("**************");

            x = np.arange(8, dtype: np.Int32).reshape(new shape(2, 2, 2)).astype(np.Object);
            y = np.hsplit(x, new Int32[] { 3, 6 });

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new Object[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } });
            AssertShape(y.ElementAt(1), 2, 0, 2);
            AssertShape(y.ElementAt(2), 2, 0, 2);

            print(y);
        }

        [TestMethod]
        public void test_vsplit_2_STRING_TODO()
        {
            var x = np.arange(8, dtype: np.Int32).reshape(new shape(2, 2, 2)).astype(np.Object);
            var y = np.vsplit(x, 2);

            Assert.AreEqual(2, y.Count);
            AssertArray(y.ElementAt(0), new Object[,,] { { { 0, 1 }, { 2, 3 } } });
            AssertArray(y.ElementAt(1), new Object[,,] { { { 4, 5 }, { 6, 7 } } });
            print(y);

            print("**************");

            x = np.arange(8, dtype: np.Int32).reshape(new shape(2, 2, 2)).astype(np.Object);
            y = np.vsplit(x, new int[] { 3, 6 });

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new Object[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } });
            AssertShape(y.ElementAt(1), 0, 2, 2);
            AssertShape(y.ElementAt(2), 0, 2, 2);

            print(y);
        }

        [TestMethod]
        public void test_dsplit_1_STRING_TODO()
        {
            var x = np.arange(16, dtype: np.Int32).reshape(new shape(2, 2, 4)).astype(np.Object);
            var y = np.dsplit(x, 2);

            Assert.AreEqual(2, y.Count);
            AssertArray(y.ElementAt(0), new Object[,,] { { { 0, 1 }, { 4, 5 } }, { { 8, 9 }, { 12, 13 } } });
            AssertArray(y.ElementAt(1), new Object[,,] { { { 2, 3 }, { 6, 7 } }, { { 10, 11 }, { 14, 15 } } });
            print(y);


            print("**************");

            x = np.arange(16, dtype: np.Int32).reshape(new shape(2, 2, 4)).astype(np.Object);
            y = np.dsplit(x, new int[] { 3, 6 });

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new Object[,,] { { { 0, 1, 2 }, { 4, 5, 6 } }, { { 8, 9, 10 }, { 12, 13, 14 } } });
            AssertArray(y.ElementAt(1), new Object[,,] { { { 3 }, { 7 } }, { { 11 }, { 15 } } });
            AssertShape(y.ElementAt(2), 2, 2, 0);

            print(y);
        }

        [TestMethod]
        public void test_kron_1_STRING_TODO()
        {

            var a = np.kron(new Object[] { 1, 10, 100 }, new Object[] { 5, 6, 7 });
            AssertArray(a, new Object[] { 5, 6, 7, 50, 60, 70, 500, 600, 700 });
            print(a);

            var b = np.kron(new Object[] { 5, 6, 7 }, new Object[] { 1, 10, 100 });
            AssertArray(b, new Object[] { 5, 50, 500, 6, 60, 600, 7, 70, 700 });
            print(b);

            var x = np.array(new Object[,] { { 2, 3 }, { 4, 5 } });
            var y = np.array(new Object[,] { { 5, 6 }, { 7, 8 } });

            var c = np.kron(x, y);
            AssertArray(c, new Object[,] { { 10, 12, 15, 18 }, { 14, 16, 21, 24 }, { 20, 24, 25, 30 }, { 28, 32, 35, 40 } });
            print(c);
            print(c.shape);

            c = np.kron(np.eye(2, dtype: np.Object), np.ones(new shape(2, 2), dtype: np.Object));
            AssertArray(c, new double[,] { { 1, 1, 0, 0 }, { 1, 1, 0, 0 }, { 0, 0, 1, 1 }, { 0, 0, 1, 1 } });


            x = np.array(new Object[,,] { { { 2, 3, 3 }, { 4, 5, 3 } } });
            y = np.array(new Object[,,] { { { 5, 6, 6, 6 }, { 7, 8, 6, 6 } } });

            c = np.kron(x, y);
            AssertArray(c, new Int32[,,] { { { 10, 12, 12, 12, 15, 18, 18, 18, 15, 18, 18, 18 },
                                           { 14, 16, 12, 12, 21, 24, 18, 18, 21, 24, 18, 18 },
                                           { 20, 24, 24, 24, 25, 30, 30, 30, 15, 18, 18, 18 },
                                           { 28, 32, 24, 24, 35, 40, 30, 30, 21, 24, 18, 18 } } });
            print(c);
            print(c.shape);


            var d = np.kron(np.ones((5, 7, 9, 11), dtype: np.Object), np.ones((3, 4, 6, 8), dtype: np.Object));
            AssertShape(d, 15, 28, 54, 88);
            print(d.shape);

        }

        [TestMethod]
        public void test_tile_2_STRING_TODO()
        {
            var a = np.array(new Object[,] { { 1, 2 }, { 3, 4 } });
            var b = np.tile(a, 2);
            AssertArray(b, new Object[,] { { 1, 2, 1, 2 }, { 3, 4, 3, 4 } });
            print(b);
            print("**************");

            var c = np.tile(a, new Int32[] { 2, 1 });
            AssertArray(c, new Object[,] { { 1, 2 }, { 3, 4 }, { 1, 2 }, { 3, 4 } });
            print(c);
            print("**************");

            var d = np.array(new Object[] { 1, 2, 3, 4 });
            var e = np.tile(d, new Int32[] { 4, 1 });

            AssertArray(e, new Object[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 } });
            print(e);
        }

        #endregion

        #region from UFUNCTests

        [TestMethod]
        public void test_UFUNC_AddReduce_1_STRING_TODO()
        {
            var x = np.arange(8, dtype: np.Int32).astype(np.Object);

            var a = np.ufunc.reduce(NpyArray_Ops.npy_op_add, x);
            Assert.AreEqual((Object)28, a.GetItem(0));
            print(a);

            x = np.arange(8, dtype: np.Int32).reshape((2, 2, 2)).astype(np.Object);
            var b = np.ufunc.reduce(NpyArray_Ops.npy_op_add, x);
            AssertArray(b, new Object[,] { { 4, 6 }, { 8, 10 } });
            print(b);

            var c = np.ufunc.reduce(NpyArray_Ops.npy_op_add, x, 0);
            AssertArray(c, new Object[,] { { 4, 6 }, { 8, 10 } });
            print(c);

            var d = np.ufunc.reduce(NpyArray_Ops.npy_op_add, x, 1);
            AssertArray(d, new Object[,] { { 2, 4 }, { 10, 12 } });
            print(d);

            var e = np.ufunc.reduce(NpyArray_Ops.npy_op_add, x, 2);
            AssertArray(e, new Object[,] { { 1, 5 }, { 9, 13 } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_AddAccumulate_1_STRING_TODO()
        {
            var x = np.arange(8, dtype: np.Int32).astype(np.Object);

            var a = np.ufunc.accumulate(NpyArray_Ops.npy_op_add, x);
            AssertArray(a, new Object[] { 0, 1, 3, 6, 10, 15, 21, 28 });
            print(a);

            x = np.arange(8, dtype: np.Int32).reshape((2, 2, 2)).astype(np.Object);
            var b = np.ufunc.accumulate(NpyArray_Ops.npy_op_add, x);
            AssertArray(b, new Object[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 6 }, { 8, 10 } } });
            print(b);

            var c = np.ufunc.accumulate(NpyArray_Ops.npy_op_add, x, 0);
            AssertArray(c, new Object[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 6 }, { 8, 10 } } });
            print(c);

            var d = np.ufunc.accumulate(NpyArray_Ops.npy_op_add, x, 1);
            AssertArray(d, new Object[,,] { { { 0, 1 }, { 2, 4 } }, { { 4, 5 }, { 10, 12 } } });
            print(d);

            var e = np.ufunc.accumulate(NpyArray_Ops.npy_op_add, x, 2);
            AssertArray(e, new Object[,,] { { { 0, 1 }, { 2, 5 } }, { { 4, 9 }, { 6, 13 } } });
            print(e);

        }

        [TestMethod]
        public void test_UFUNC_AddReduceAt_1_STRING_TODO()
        {
            var a = np.ufunc.reduceat(NpyArray_Ops.npy_op_add, np.arange(8, dtype: np.Int32), new npy_intp[] { 0, 4, 1, 5, 2, 6, 3, 7 }).astype(np.Object)["::2"] as ndarray;
            AssertArray(a, new Object[] { 6, 10, 14, 18 });
            print(a);

            double retstep = 0;
            var x = np.linspace(0, 15, ref retstep, 16, dtype: np.Int32).reshape((4, 4)).astype(np.Object);
            var b = np.ufunc.reduceat(NpyArray_Ops.npy_op_add, x, new npy_intp[] { 0, 3, 1, 2, 0 });
            AssertArray(b, new Object[,] {{12, 15, 18, 21},{12, 13, 14, 15}, {4, 5, 6, 7},
                                          {8, 9, 10, 11}, {24, 28, 32, 36}});
            print(b);

            var c = np.ufunc.reduceat(NpyArray_Ops.npy_op_multiply, x, new npy_intp[] { 0, 3 }, axis: 1);
            AssertArray(c, new Object[,] { { 0, 3 }, { 120, 7 }, { 720, 11 }, { 2184, 15 } });
            print(c);
        }

        [TestMethod]
        public void test_UFUNC_AddOuter_1_STRING_TODO()
        {
            var x = np.arange(4, dtype: np.Int32);

            var a = np.ufunc.outer(NpyArray_Ops.npy_op_add, null, x, x);
            AssertShape(a, 4, 4);
            print(a.shape);
            AssertArray(a, new Object[,] { { 0, 1, 2, 3 }, { 1, 2, 3, 4 }, { 2, 3, 4, 5 }, { 3, 4, 5, 6 } });
            print(a);

            x = np.arange(6, dtype: np.Int32).reshape((3, 2)).astype(np.Object);
            var y = np.arange(6, dtype: np.Int32).reshape((2, 3)).astype(np.Object);
            var b = np.ufunc.outer(NpyArray_Ops.npy_op_add, null, x, y);
            AssertShape(b, 3, 2, 2, 3);
            print(b.shape);

            var ExpectedDataB = new Object[,,,]

                {{{{0,  1,  2}, {3,  4,  5}}, {{1,  2,  3}, { 4,  5,  6}}},
                 {{{2,  3,  4}, {5,  6,  7}}, {{3,  4,  5}, { 6,  7,  8}}},
                 {{{4,  5,  6}, {7,  8,  9}}, {{5,  6,  7}, { 8,  9, 10}}}};

            AssertArray(b, ExpectedDataB);

            print(b);
        }

        #endregion

        #region from IndexTricksTests

        [TestMethod]
        public void test_mgrid_1_STRING_TODO()
        {
            var a = (ndarray)np.mgrid(new Slice[] { new Slice((object)0, (object)5) });
            print(a);
            AssertArray(a, new Object[] { 0, 1, 2, 3, 4 });
            print("************");

            var b = (ndarray)np.mgrid(new Slice[] { new Slice((Object)0, (Object)6) });
            print(b);
            AssertArray(b, new Object[] { 0, 1, 2, 3, 4, 5 });
            print("************");

            var c = (ndarray)np.mgrid(new Slice[] { new Slice((Object)0, (Object)5), new Slice((Object)0, (Object)5) });
            print(c);

            var ExpectedCArray = new Object[,,]
                {{{0, 0, 0, 0, 0},  {1, 1, 1, 1, 1},  {2, 2, 2, 2, 2},  {3, 3, 3, 3, 3},  {4, 4, 4, 4, 4}},
                 {{0, 1, 2, 3, 4},  {0, 1, 2, 3, 4},  {0, 1, 2, 3, 4},  {0, 1, 2, 3, 4},  {0, 1, 2, 3, 4}}};
            AssertArray(c, ExpectedCArray);


            print("************");

            var d = (ndarray)np.mgrid(new Slice[] { new Slice((Object)0, (Object)6), new Slice((Object)0, (Object)6) });
            print(d);
            var ExpectedDArray = new Object[,,]
                {{{0, 0, 0, 0, 0, 0},  {1, 1, 1, 1, 1, 1},  {2, 2, 2, 2, 2, 2},  {3, 3, 3, 3, 3, 3},  {4, 4, 4, 4, 4, 4}, {5, 5, 5, 5, 5, 5}},
                 {{0, 1, 2, 3, 4, 5},  {0, 1, 2, 3, 4, 5},  {0, 1, 2, 3, 4, 5},  {0, 1, 2, 3, 4, 5},  {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}}};
            AssertArray(d, ExpectedDArray);

            print("************");

            var e = (ndarray)np.mgrid(new Slice[] { new Slice((Object)3, (Object)5), new Slice((Object)4, (Object)6), new Slice((Object)2, (Object)5) });
            print(e);
            var ExpectedEArray = new Object[,,,]
                {
                    {{{3, 3, 3}, {3, 3, 3}}, {{4, 4, 4}, {4, 4, 4}}},
                    {{{4, 4, 4}, {5, 5, 5}}, {{4, 4, 4}, {5, 5, 5}}},
                    {{{2, 3, 4}, {2, 3, 4}}, {{2, 3, 4}, {2, 3, 4}}},
                };
            AssertArray(e, ExpectedEArray);

        }

        [TestMethod]
        public void test_ogrid_1_STRING_TODO()
        {
            var a = (ndarray)np.ogrid(new Slice[] { new Slice((Object)0, (Object)5) });
            print(a);
            AssertArray(a, new Object[] { 0, 1, 2, 3, 4 });
            print("************");

            var b = (ndarray)np.ogrid(new Slice[] { new Slice((Object)0, (Object)6) });
            print(b);
            AssertArray(b, new Object[] { 0, 1, 2, 3, 4, 5 });
            print("************");

            var c = (ndarray[])np.ogrid(new Slice[] { new Slice((Object)0, (Object)5), new Slice((Object)0, (Object)5) });
            print(c);
            AssertArray(c[0], new Object[,] { { 0 }, { 1 }, { 2 }, { 3 }, { 4 } });
            AssertArray(c[1], new Object[,] { { 0, 1, 2, 3, 4 } });


            print("************");

            var d = (ndarray[])np.ogrid(new Slice[] { new Slice((Object)0, (Object)6), new Slice((Object)0, (Object)6) });
            print(d);
            AssertArray(d[0], new Object[,] { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
            AssertArray(d[1], new Object[,] { { 0, 1, 2, 3, 4, 5 } });

            print("************");

            var e = (ndarray[])np.ogrid(new Slice[] { new Slice((Object)3, (Object)5), new Slice((Object)4, (Object)6), new Slice((Object)2, (Object)5) });
            print(e);
            AssertArray(e[0], new Object[,,] { { { 3 } }, { { 4 } } });
            AssertArray(e[1], new Object[,,] { { { 4 }, { 5 } } });
            AssertArray(e[2], new Object[,,] { { { 2, 3, 4 } } });

        }

        [TestMethod]
        public void test_fill_diagonal_1_STRING_TODO()
        {
            var a = np.zeros((3, 3), np.Int32).astype(np.Object);
            np.fill_diagonal(a, 5);
            AssertArray(a, new Object[,] { { 5, 0, 0 }, { 0, 5, 0 }, { 0, 0, 5 } });
            print(a);

            a = np.zeros((3, 3, 3, 3), np.Int32).astype(np.Object);
            np.fill_diagonal(a, 4);
            AssertArray(a[0, 0] as ndarray, new Object[,] { { 4, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } });
            print(a[0, 0]);
            AssertArray(a[1, 1] as ndarray, new Object[,] { { 0, 0, 0 }, { 0, 4, 0 }, { 0, 0, 0 } });
            print(a[1, 1]);
            AssertArray(a[2, 2] as ndarray, new Object[,] { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 4 } });
            print(a[2, 2]);

            // tall matrices no wrap
            a = np.zeros((5, 3), np.Int32).astype(np.Object);
            np.fill_diagonal(a, 4);
            AssertArray(a, new Object[,] { { 4, 0, 0 }, { 0, 4, 0 }, { 0, 0, 4 }, { 0, 0, 0 }, { 0, 0, 0 } });
            print(a);

            // tall matrices wrap
            a = np.zeros((5, 3), np.Int32).astype(np.Object);
            np.fill_diagonal(a, 4, wrap: true);
            AssertArray(a, new Object[,] { { 4, 0, 0 }, { 0, 4, 0 }, { 0, 0, 4 }, { 0, 0, 0 }, { 4, 0, 0 } });
            print(a);

            // wide matrices wrap
            a = np.zeros((3, 5), np.Int32).astype(np.Object);
            np.fill_diagonal(a, 4, wrap: true);
            AssertArray(a, new Object[,] { { 4, 0, 0, 0, 0 }, { 0, 4, 0, 0, 0 }, { 0, 0, 4, 0, 0 } });
            print(a);


        }

        [TestMethod]
        public void test_diag_indices_1_STRING_TODO()
        {
            var di = np.diag_indices(4);
            AssertArray(di[0], new Int32[] { 0, 1, 2, 3 });
            AssertArray(di[1], new Int32[] { 0, 1, 2, 3 });
            print(di);

            var a = np.arange(16, dtype: np.Int32).reshape((4, 4)).astype(np.Object);
            a[di] = 100;

            AssertArray(a, new Object[,] { { 100, 1, 2, 3 }, { 4, 100, 6, 7 }, { 8, 9, 100, 11 }, { 12, 13, 14, 100 } });
            print(a);

            return;

        }

        [TestMethod]
        public void test_diag_indices_from_1_STRING_TODO()
        {
            var a = np.arange(16, dtype: np.Int32).reshape((4, 4)).astype(np.Object);
            var di = np.diag_indices_from(a);
            AssertArray(di[0], new Int32[] { 0, 1, 2, 3 });
            AssertArray(di[1], new Int32[] { 0, 1, 2, 3 });
            print(di);
        }

        #endregion

        #region from StrideTricksTests

        [TestMethod]
        public void test_broadcast_1_STRING_TODO()
        {
            var x = np.array(new Object[,] { { 11 }, { 2 }, { 3 } });
            var y = np.array(new Object[] { 4, 5, 6 });
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
        public void test_broadcast_to_1_STRING_TODO()
        {
            var a = np.broadcast_to((Object)5, (4, 4));
            AssertArray(a, new Object[,] { { 5, 5, 5, 5 }, { 5, 5, 5, 5 }, { 5, 5, 5, 5 }, { 5, 5, 5, 5 } });
            AssertStrides(a, 0, 0);
            print(a);
            print(a.shape);
            print(a.strides);
            print("*************");


            var b = np.broadcast_to(new Object[] { 1, 2, 3 }, (3, 3));
            AssertArray(b, new Object[,] { { 1, 2, 3 }, { 1, 2, 3 }, { 1, 2, 3 } });
            AssertStrides(b, 0, SizeOfString);
            print(b);
            print(b.shape);
            print(b.strides);
            print("*************");


        }

        [TestMethod]
        public void test_broadcast_arrays_1_STRING_TODO()
        {
            var x = np.array(new Object[,] { { 1, 2, 3 } });
            var y = np.array(new Object[,] { { 4 }, { 5 } });
            var z = np.broadcast_arrays(false, new ndarray[] { x, y });

            print(z);

        }

        [TestMethod]
        public void test_as_strided_1_STRING_TODO()
        {
            var y = np.zeros((10, 10), np.Int32).astype(np.Object);
            AssertStrides(y, SizeOfString * 10, SizeOfString * 1);
            print(y.strides);

            var n = 1000;
            var a = np.arange(n, dtype: np.Float64).astype(np.Object);

            var b = np.as_strided(a, (n, n), (0, 8));

            //print(b);

            Assert.AreEqual(1000000, b.size);
            print(b.size);
            AssertShape(b, 1000, 1000);
            print(b.shape);
            AssertStrides(b, 0, 8);
            print(b.strides);
            Assert.AreEqual(4000000, b.nbytes);
            print(b.nbytes);

        }

        #endregion

        #region from IteratorTests

        [TestMethod]
        public void test_nditer_1_STRING_TODO()
        {
            var a = np.arange(0.1, 6.1, dtype: np.Int32).reshape((2, 3)).astype(np.Object);
            var b = np.array(new Object[] { 7, 8, 9 });

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
        public void test_ndindex_1_STRING_TODO()
        {
            var a = np.arange(0.1, 6.1, dtype: np.Int32).reshape((2, 3)).astype(np.Object);

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
        public void test_ndenumerate_1_STRING_TODO()
        {
            var a = np.arange(0.1, 6.1, dtype: np.Int32).reshape((2, 3)).astype(np.Object);

            foreach (ValueTuple<npy_intp[], object> aa in new ndenumerate(a))
            {
                print(aa.Item1);
                print(aa.Item2);
            }
        }

        #endregion

        #region very cool object unique tests

        public class ObjectDemoData
        {
            public System.Int64 iInt64;
            public System.Numerics.Complex iComplex;
            public System.Numerics.BigInteger iBigInt;
            public System.Double iDouble;

            public ObjectDemoData(Int64 iValue)
            {
                this.iInt64 = iValue;
                this.iDouble = iValue;
                this.iComplex = new Complex((double)iValue, 0);
                this.iBigInt = iValue;
            }

            private static ObjectDemoData Copy(ObjectDemoData a)
            {
                ObjectDemoData b = new ObjectDemoData(0);
                b.iInt64 = a.iInt64;
                b.iDouble = a.iDouble;
                b.iComplex = a.iComplex;
                b.iBigInt = a.iBigInt;
                return b;
            }

            public override string ToString()
            {
                return string.Format("{0}:{1}:{2}:{3}",
                    iInt64.ToString(), iDouble.ToString(), iComplex.ToString(), iBigInt.ToString());
            }
            #region ADD operations
            public static ObjectDemoData operator +(ObjectDemoData a, Int64 iValue)
            {
                var b = Copy(a);

                b.iInt64 += iValue;
                b.iDouble += iValue;
                b.iComplex += new Complex((double)iValue, 0);
                b.iBigInt += iValue;

                return b;
            }

            public static ObjectDemoData operator +(ObjectDemoData a, double dValue)
            {
                var b = Copy(a);

                b.iInt64 += (Int64)dValue;
                b.iDouble += dValue;
                b.iComplex += new Complex((double)dValue, 0);
                b.iBigInt += (Int64)dValue;

                return b;
            }

            public static ObjectDemoData operator +(ObjectDemoData a, Complex iValue)
            {
                var b = Copy(a);

                b.iInt64 += (Int64)iValue.Real;
                b.iDouble += iValue.Real;
                b.iComplex += iValue;
                b.iBigInt += (Int64)iValue.Real;

                return b;
            }

            public static ObjectDemoData operator +(ObjectDemoData a, BigInteger iValue)
            {
                var b = Copy(a);

                b.iInt64 += (Int64)iValue;
                b.iDouble += (Int64)iValue;
                b.iComplex += new Complex((double)iValue, 0);
                b.iBigInt += iValue;

                return b;
            }

            public static ObjectDemoData operator +(ObjectDemoData a, string sValue)
            {
                var b = Copy(a);

                Int64 iValue = 0;
                Int64.TryParse(sValue, out iValue);

                b.iInt64 += iValue;
                b.iDouble += iValue;
                b.iComplex += new Complex((double)iValue, 0);
                b.iBigInt += iValue;

                return b;
            }

            public static ObjectDemoData operator +(ObjectDemoData a, ObjectDemoData iValue)
            {
                var b = Copy(a);

                b.iInt64 += iValue.iInt64;
                b.iDouble += iValue.iDouble;
                b.iComplex += iValue.iComplex;
                b.iBigInt += iValue.iBigInt;

                return b;
            }
            #endregion

            #region SUBTRACT operations
            public static ObjectDemoData operator -(ObjectDemoData a, Int64 iValue)
            {
                var b = Copy(a);

                b.iInt64 -= iValue;
                b.iDouble -= iValue;
                b.iComplex -= new Complex((double)iValue, 0);
                b.iBigInt -= iValue;

                return b;
            }

            public static ObjectDemoData operator -(ObjectDemoData a, double dValue)
            {
                var b = Copy(a);

                b.iInt64 -= (Int64)dValue;
                b.iDouble -= dValue;
                b.iComplex -= new Complex((double)dValue, 0);
                b.iBigInt -= (Int64)dValue;

                return b;
            }

            public static ObjectDemoData operator -(ObjectDemoData a, Complex iValue)
            {
                var b = Copy(a);

                b.iInt64 -= (Int64)iValue.Real;
                b.iDouble -= iValue.Real;
                b.iComplex -= iValue;
                b.iBigInt -= (Int64)iValue.Real;

                return b;
            }

            public static ObjectDemoData operator -(ObjectDemoData a, BigInteger iValue)
            {
                var b = Copy(a);

                b.iInt64 -= (Int64)iValue;
                b.iDouble -= (Int64)iValue;
                b.iComplex -= new Complex((double)iValue, 0);
                b.iBigInt -= iValue;

                return b;
            }

            public static ObjectDemoData operator -(ObjectDemoData a, string sValue)
            {
                var b = Copy(a);

                Int64 iValue = 0;
                Int64.TryParse(sValue, out iValue);

                b.iInt64 -= iValue;
                b.iDouble -= iValue;
                b.iComplex -= new Complex((double)iValue, 0);
                b.iBigInt -= iValue;

                return b;
            }

            public static ObjectDemoData operator -(ObjectDemoData a, ObjectDemoData iValue)
            {
                var b = Copy(a);

                b.iInt64 -= iValue.iInt64;
                b.iDouble -= iValue.iDouble;
                b.iComplex -= iValue.iComplex;
                b.iBigInt -= iValue.iBigInt;

                return b;
            }

            #endregion

            #region MULTIPLY operations
            public static ObjectDemoData operator *(ObjectDemoData a, Int64 iValue)
            {
                var b = Copy(a);

                b.iInt64 *= iValue;
                b.iDouble *= iValue;
                b.iComplex *= new Complex((double)iValue, 0);
                b.iBigInt *= iValue;

                return b;
            }

            public static ObjectDemoData operator *(ObjectDemoData a, double dValue)
            {
                var b = Copy(a);

                b.iInt64 *= (Int64)dValue;
                b.iDouble *= dValue;
                b.iComplex *= new Complex((double)dValue, 0);
                b.iBigInt *= (Int64)dValue;

                return b;
            }

            public static ObjectDemoData operator *(ObjectDemoData a, Complex iValue)
            {
                var b = Copy(a);

                b.iInt64 *= (Int64)iValue.Real;
                b.iDouble *= iValue.Real;
                b.iComplex *= iValue;
                b.iBigInt *= (Int64)iValue.Real;

                return b;
            }

            public static ObjectDemoData operator *(ObjectDemoData a, BigInteger iValue)
            {
                var b = Copy(a);

                b.iInt64 *= (Int64)iValue;
                b.iDouble *= (Int64)iValue;
                b.iComplex *= new Complex((double)iValue, 0);
                b.iBigInt *= iValue;

                return b;
            }

            public static ObjectDemoData operator *(ObjectDemoData a, string sValue)
            {
                var b = Copy(a);

                Int64 iValue = 0;
                Int64.TryParse(sValue, out iValue);

                b.iInt64 *= iValue;
                b.iDouble *= iValue;
                b.iComplex *= new Complex((double)iValue, 0);
                b.iBigInt *= iValue;

                return b;
            }
            public static ObjectDemoData operator *(ObjectDemoData a, ObjectDemoData iValue)
            {
                var b = Copy(a);

                b.iInt64 *= iValue.iInt64;
                b.iDouble *= iValue.iDouble;
                b.iComplex *= iValue.iComplex;
                b.iBigInt *= iValue.iBigInt;

                return b;
            }
            #endregion

            #region DIVIDE operations
            public static ObjectDemoData operator /(ObjectDemoData a, Int64 iValue)
            {
                var b = Copy(a);

                b.iInt64 /= iValue;
                b.iDouble /= iValue;
                b.iComplex /= new Complex((double)iValue, 0);
                b.iBigInt /= iValue;

                return b;
            }

            public static ObjectDemoData operator /(ObjectDemoData a, double dValue)
            {
                var b = Copy(a);

                b.iInt64 /= (Int64)dValue;
                b.iDouble /= dValue;
                b.iComplex /= new Complex((double)dValue, 0);
                b.iBigInt /= (Int64)dValue;

                return b;
            }

            public static ObjectDemoData operator /(ObjectDemoData a, Complex iValue)
            {
                var b = Copy(a);

                b.iInt64 /= (Int64)iValue.Real;
                b.iDouble /= iValue.Real;
                b.iComplex /= iValue;
                b.iBigInt /= (Int64)iValue.Real;

                return b;
            }

            public static ObjectDemoData operator /(ObjectDemoData a, BigInteger iValue)
            {
                var b = Copy(a);

                b.iInt64 /= (Int64)iValue;
                b.iDouble /= (Int64)iValue;
                b.iComplex /= new Complex((double)iValue, 0);
                b.iBigInt /= iValue;

                return b;
            }

            public static ObjectDemoData operator /(ObjectDemoData a, string sValue)
            {
                var b = Copy(a);

                Int64 iValue = 0;
                Int64.TryParse(sValue, out iValue);

                b.iInt64 /= iValue;
                b.iDouble /= iValue;
                b.iComplex /= new Complex((double)iValue, 0);
                b.iBigInt /= iValue;

                return b;
            }
            public static ObjectDemoData operator /(ObjectDemoData a, ObjectDemoData iValue)
            {
                var b = Copy(a);

                b.iInt64 /= iValue.iInt64;
                b.iDouble /= iValue.iDouble;
                b.iComplex /= iValue.iComplex;
                b.iBigInt /= iValue.iBigInt;

                return b;
            }
            #endregion

            #region LEFTSHIFT operations
            public static ObjectDemoData operator <<(ObjectDemoData a, int iValue)
            {
                var b = Copy(a);

                b.iInt64 <<= iValue;
                //b.iDouble <<= iValue;
                //b.iComplex <<= new Complex((double)iValue, 0);
                b.iBigInt <<= iValue;

                return b;
            }

            #endregion

            #region RIGHTSHIFT operations
            public static ObjectDemoData operator >>(ObjectDemoData a, int iValue)
            {
                var b = Copy(a);

                b.iInt64 >>= iValue;
                //b.iDouble >>= iValue;
                //b.iComplex >>= new Complex((double)iValue, 0);
                b.iBigInt >>= iValue;

                return b;
            }

            #endregion

            #region BITWISEAND operations
            public static ObjectDemoData operator &(ObjectDemoData a, Int32 iValue)
            {
                var b = Copy(a);

                b.iInt64 &= iValue;
                //b.iDouble &= iValue;
                //b.iComplex &= new Complex((double)iValue, 0);
                b.iBigInt &= iValue;

                return b;
            }
            public static ObjectDemoData operator &(ObjectDemoData a, Int64 iValue)
            {
                var b = Copy(a);

                b.iInt64 &= iValue;
                //b.iDouble &= iValue;
                //b.iComplex &= new Complex((double)iValue, 0);
                b.iBigInt &= iValue;

                return b;
            }
            #endregion

            #region BITWISEOR operations
            public static ObjectDemoData operator |(ObjectDemoData a, Int32 iValue)
            {
                var b = Copy(a);

                b.iInt64 |= (UInt32)iValue;
                //b.iDouble &= iValue;
                //b.iComplex &= new Complex((double)iValue, 0);
                b.iBigInt |= iValue;

                return b;
            }
            public static ObjectDemoData operator |(ObjectDemoData a, Int64 iValue)
            {
                var b = Copy(a);

                b.iInt64 |= iValue;
                //b.iDouble &= iValue;
                //b.iComplex &= new Complex((double)iValue, 0);
                b.iBigInt |= iValue;

                return b;
            }
            #endregion

            #region BITWISEXOR operations
            public static ObjectDemoData operator ^(ObjectDemoData a, Int32 iValue)
            {
                var b = Copy(a);

                b.iInt64 ^= iValue;
                //b.iDouble &= iValue;
                //b.iComplex &= new Complex((double)iValue, 0);
                b.iBigInt ^= iValue;

                return b;
            }
            public static ObjectDemoData operator ^(ObjectDemoData a, Int64 iValue)
            {
                var b = Copy(a);

                b.iInt64 ^= iValue;
                //b.iDouble &= iValue;
                //b.iComplex &= new Complex((double)iValue, 0);
                b.iBigInt ^= iValue;

                return b;
            }
            #endregion

            #region REMAINDER operations
            public static ObjectDemoData operator %(ObjectDemoData a, double iValue)
            {
                var b = Copy(a);

                b.iInt64 %= (Int64)iValue;
                b.iDouble %= iValue;
                //b.iComplex %= iValue; 
                b.iBigInt %= (Int64)iValue;

                return b;
            }
            public static ObjectDemoData operator %(ObjectDemoData a, Int64 iValue)
            {
                var b = Copy(a);

                b.iInt64 %= iValue;
                b.iDouble %= iValue;
                //b.iComplex %= new Complex((double)iValue, 0);
                b.iBigInt %= iValue;

                return b;
            }
            public static ObjectDemoData operator %(ObjectDemoData a, ObjectDemoData iValue)
            {
                var b = Copy(a);

                b.iInt64 %= iValue.iInt64;
                b.iDouble %= iValue.iDouble;
                //b.iComplex %= iValue.iComplex;
                b.iBigInt %= iValue.iBigInt;

                return b;
            }
            #endregion

            #region NEGATIVE operations
            public static ObjectDemoData operator -(ObjectDemoData a)
            {
                var b = Copy(a);

                b.iInt64 = -b.iInt64;
                b.iDouble = -b.iDouble;
                b.iComplex = -b.iComplex;
                b.iBigInt = -b.iBigInt;

                return b;
            }
            #endregion

            #region INVERT operations
            public static ObjectDemoData operator ~(ObjectDemoData a)
            {
                var b = Copy(a);

                b.iInt64 = ~b.iInt64;
                //b.iDouble = ~b.iDouble;
                //b.iComplex = ~b.iComplex;
                b.iBigInt = ~b.iBigInt;

                return b;
            }
            #endregion

            #region LESS operations
            public static bool operator <(ObjectDemoData a, Int64 iValue)
            {
                return a.iInt64 < iValue;
            }

            public static bool operator <(ObjectDemoData a, double dValue)
            {
                return a.iDouble < dValue;
            }

            public static bool operator <(ObjectDemoData a, Complex iValue)
            {
                return a.iComplex.Real < iValue.Real;
            }

            public static bool operator <(ObjectDemoData a, BigInteger iValue)
            {
                return a.iBigInt < iValue;
            }

            public static bool operator <(ObjectDemoData a, ObjectDemoData iValue)
            {
                return a.iInt64 < iValue.iInt64;
            }
            #endregion

            #region LESSEQUAL operations
            public static bool operator <=(ObjectDemoData a, Int64 iValue)
            {
                return a.iInt64 <= iValue;
            }

            public static bool operator <=(ObjectDemoData a, double dValue)
            {
                return a.iDouble <= dValue;
            }

            public static bool operator <=(ObjectDemoData a, Complex iValue)
            {
                return a.iComplex.Real <= iValue.Real;
            }

            public static bool operator <=(ObjectDemoData a, BigInteger iValue)
            {
                return a.iBigInt <= iValue;
            }

            public static bool operator <=(ObjectDemoData a, ObjectDemoData iValue)
            {
                return a.iInt64 <= iValue.iInt64;
            }
            #endregion

            #region EQUAL operations
            public override bool Equals(object obj)
            {
                return base.Equals(obj);
            }
            public override int GetHashCode()
            {
                return base.GetHashCode();
            }

            public static bool operator ==(ObjectDemoData a, Int64 iValue)
            {
                return a.iInt64 == iValue;
            }

            public static bool operator ==(ObjectDemoData a, double dValue)
            {
                return a.iDouble == dValue;
            }

            public static bool operator ==(ObjectDemoData a, Complex iValue)
            {
                return a.iComplex.Real == iValue.Real;
            }

            public static bool operator ==(ObjectDemoData a, BigInteger iValue)
            {
                return a.iBigInt == iValue;
            }

            public static bool operator ==(ObjectDemoData a, ObjectDemoData iValue)
            {
                return a.iInt64 == iValue.iInt64;
            }
            #endregion

            #region NOTEQUAL operations


            public static bool operator !=(ObjectDemoData a, Int64 iValue)
            {
                return a.iInt64 != iValue;
            }

            public static bool operator !=(ObjectDemoData a, double dValue)
            {
                return a.iDouble != dValue;
            }

            public static bool operator !=(ObjectDemoData a, Complex iValue)
            {
                return a.iComplex.Real != iValue.Real;
            }

            public static bool operator !=(ObjectDemoData a, BigInteger iValue)
            {
                return a.iBigInt != iValue;
            }

            public static bool operator !=(ObjectDemoData a, ObjectDemoData iValue)
            {
                return a.iInt64 != iValue.iInt64;
            }
            #endregion

            #region GREATER operations
            public static bool operator >(ObjectDemoData a, Int64 iValue)
            {
                return a.iInt64 > iValue;
            }

            public static bool operator >(ObjectDemoData a, double dValue)
            {
                return a.iDouble > dValue;
            }

            public static bool operator >(ObjectDemoData a, Complex iValue)
            {
                return a.iComplex.Real > iValue.Real;
            }

            public static bool operator >(ObjectDemoData a, BigInteger iValue)
            {
                return a.iBigInt > iValue;
            }

            public static bool operator >(ObjectDemoData a, ObjectDemoData iValue)
            {
                return a.iInt64 > iValue.iInt64;
            }
            #endregion

            #region GREATEREQUAL operations
            public static bool operator >=(ObjectDemoData a, Int64 iValue)
            {
                return a.iInt64 >= iValue;
            }

            public static bool operator >=(ObjectDemoData a, double dValue)
            {
                return a.iDouble >= dValue;
            }

            public static bool operator >=(ObjectDemoData a, Complex iValue)
            {
                return a.iComplex.Real >= iValue.Real;
            }

            public static bool operator >=(ObjectDemoData a, BigInteger iValue)
            {
                return a.iBigInt >= iValue;
            }

            public static bool operator >=(ObjectDemoData a, ObjectDemoData iValue)
            {
                return a.iInt64 >= iValue.iInt64;
            }
            #endregion
        }


        // https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/operator-overloading

        // build a class that supports these overloads and see how the default object class dynamic keyword
        // handles them.  It may not be necessary to have user defined types.


        [TestMethod]
        public void test_add_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(0), new ObjectDemoData(1),
                new ObjectDemoData(2), new ObjectDemoData(3)
            };

            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoData1";

            aa = aa.reshape(new shape(2, 2));
            print(aa);

            var bb = aa + 8;
            bb.Name += " (BB)";
            print(bb);
            AssertShape(bb, 2, 2);

            var bb0 = bb.item(0) as ObjectDemoData;
            var bb1 = bb.item(1) as ObjectDemoData;
            var bb2 = bb.item(2) as ObjectDemoData;
            var bb3 = bb.item(3) as ObjectDemoData;

            Assert.AreEqual(bb0.iInt64, 8);
            Assert.AreEqual(bb1.iInt64, 9);
            Assert.AreEqual(bb2.iInt64, 10);
            Assert.AreEqual(bb3.iInt64, 11);

            var cc = aa + (double)5.5;
            cc.Name += " (CC)";
            print(cc);
            AssertShape(cc, 2, 2);

            var cc0 = cc.item(0) as ObjectDemoData;
            var cc1 = cc.item(1) as ObjectDemoData;
            var cc2 = cc.item(2) as ObjectDemoData;
            var cc3 = cc.item(3) as ObjectDemoData;

            Assert.AreEqual(cc0.iInt64, 5);
            Assert.AreEqual(cc1.iDouble, 6.5);
            Assert.AreEqual(cc2.iComplex, new Complex(7.5, 0));
            Assert.AreEqual(cc3.iBigInt, new BigInteger(8));

            var dd = aa + new System.Numerics.Complex(7.5, 1.5);
            dd.Name += " (DD)";
            print(dd);
            AssertShape(dd, 2, 2);

            var dd0 = dd.item(0) as ObjectDemoData;
            var dd1 = dd.item(1) as ObjectDemoData;
            var dd2 = dd.item(2) as ObjectDemoData;
            var dd3 = dd.item(3) as ObjectDemoData;

            Assert.AreEqual(dd0.iInt64, 7);
            Assert.AreEqual(dd1.iDouble, 8.5);
            Assert.AreEqual(dd2.iComplex, new Complex(9.5, 1.5));
            Assert.AreEqual(dd3.iBigInt, new BigInteger(10));

            var ee = aa + new System.Numerics.BigInteger(11);
            ee.Name += " (EE)";
            print(ee);
            AssertShape(ee, 2, 2);

            var ee0 = ee.item(0) as ObjectDemoData;
            var ee1 = ee.item(1) as ObjectDemoData;
            var ee2 = ee.item(2) as ObjectDemoData;
            var ee3 = ee.item(3) as ObjectDemoData;

            Assert.AreEqual(ee0.iInt64, 11);
            Assert.AreEqual(ee1.iDouble, 12);
            Assert.AreEqual(ee2.iComplex, new Complex(13, 0));
            Assert.AreEqual(ee3.iBigInt, new BigInteger(14));

            var ff = np.add(aa, "23");
            ff.Name += " (FF)";
            print(ff);
            AssertShape(ff, 2, 2);

            var ff0 = ff.item(0) as ObjectDemoData;
            var ff1 = ff.item(1) as ObjectDemoData;
            var ff2 = ff.item(2) as ObjectDemoData;
            var ff3 = ff.item(3) as ObjectDemoData;

            Assert.AreEqual(ff0.iInt64, 23);
            Assert.AreEqual(ff1.iDouble, 24);
            Assert.AreEqual(ff2.iComplex, new Complex(25, 0));
            Assert.AreEqual(ff3.iBigInt, new BigInteger(26));

            var aa0 = aa.item(0) as ObjectDemoData;
            var aa1 = aa.item(1) as ObjectDemoData;
            var aa2 = aa.item(2) as ObjectDemoData;
            var aa3 = aa.item(3) as ObjectDemoData;

            Assert.AreEqual(aa0.iInt64, 0);
            Assert.AreEqual(aa1.iInt64, 1);
            Assert.AreEqual(aa2.iInt64, 2);
            Assert.AreEqual(aa3.iInt64, 3);

            var gg = np.add(aa, ff2);
            gg.Name += " (GG)";
            print(gg);
            AssertShape(gg, 2, 2);

            var gg0 = gg.item(0) as ObjectDemoData;
            var gg1 = gg.item(1) as ObjectDemoData;
            var gg2 = gg.item(2) as ObjectDemoData;
            var gg3 = gg.item(3) as ObjectDemoData;

            Assert.AreEqual(gg0.iInt64, 25);
            Assert.AreEqual(gg1.iDouble, 26);
            Assert.AreEqual(gg2.iComplex, new Complex(27, 0));
            Assert.AreEqual(gg3.iBigInt, new BigInteger(28));

            var hh = aa + ff;
            hh.Name += " (hh)";
            print(hh);
            AssertShape(hh, 2, 2);

            var hh0 = hh.item(0) as ObjectDemoData;
            var hh1 = hh.item(1) as ObjectDemoData;
            var hh2 = hh.item(2) as ObjectDemoData;
            var hh3 = hh.item(3) as ObjectDemoData;

            Assert.AreEqual(hh0.iInt64, 23);
            Assert.AreEqual(hh1.iDouble, 25);
            Assert.AreEqual(hh2.iComplex, new Complex(27, 0));
            Assert.AreEqual(hh3.iBigInt, new BigInteger(29));



        }

        [TestMethod]
        public void test_subtract_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(0), new ObjectDemoData(1),
                new ObjectDemoData(2), new ObjectDemoData(3)
            };

            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoData2";

            aa = aa.reshape(new shape(2, 2));
            print(aa);

            var bb = aa - 8;
            bb.Name += " (BB)";
            print(bb);
            AssertShape(bb, 2, 2);

            var bb0 = bb.item(0) as ObjectDemoData;
            var bb1 = bb.item(1) as ObjectDemoData;
            var bb2 = bb.item(2) as ObjectDemoData;
            var bb3 = bb.item(3) as ObjectDemoData;

            Assert.AreEqual(bb0.iInt64, -8);
            Assert.AreEqual(bb1.iInt64, -7);
            Assert.AreEqual(bb2.iInt64, -6);
            Assert.AreEqual(bb3.iInt64, -5);

            var cc = aa - (double)5.5;
            cc.Name += " (CC)";
            print(cc);
            AssertShape(cc, 2, 2);

            var cc0 = cc.item(0) as ObjectDemoData;
            var cc1 = cc.item(1) as ObjectDemoData;
            var cc2 = cc.item(2) as ObjectDemoData;
            var cc3 = cc.item(3) as ObjectDemoData;

            Assert.AreEqual(cc0.iInt64, -5);
            Assert.AreEqual(cc1.iDouble, -4.5);
            Assert.AreEqual(cc2.iComplex, new Complex(-3.5, 0));
            Assert.AreEqual(cc3.iBigInt, new BigInteger(-2));

            var dd = aa - new System.Numerics.Complex(7.5, 1.5);
            dd.Name += " (DD)";
            print(dd);
            AssertShape(dd, 2, 2);

            var dd0 = dd.item(0) as ObjectDemoData;
            var dd1 = dd.item(1) as ObjectDemoData;
            var dd2 = dd.item(2) as ObjectDemoData;
            var dd3 = dd.item(3) as ObjectDemoData;

            Assert.AreEqual(dd0.iInt64, -7);
            Assert.AreEqual(dd1.iDouble, -6.5);
            Assert.AreEqual(dd2.iComplex, new Complex(-5.5, -1.5));
            Assert.AreEqual(dd3.iBigInt, new BigInteger(-4));

            var ee = aa - new System.Numerics.BigInteger(11);
            ee.Name += " (EE)";
            print(ee);
            AssertShape(ee, 2, 2);

            var ee0 = ee.item(0) as ObjectDemoData;
            var ee1 = ee.item(1) as ObjectDemoData;
            var ee2 = ee.item(2) as ObjectDemoData;
            var ee3 = ee.item(3) as ObjectDemoData;

            Assert.AreEqual(ee0.iInt64, -11);
            Assert.AreEqual(ee1.iDouble, -10);
            Assert.AreEqual(ee2.iComplex, new Complex(-9, 0));
            Assert.AreEqual(ee3.iBigInt, new BigInteger(-8));

            var ff = np.subtract(aa, "23");
            ff.Name += " (FF)";
            print(ff);
            AssertShape(ff, 2, 2);

            var ff0 = ff.item(0) as ObjectDemoData;
            var ff1 = ff.item(1) as ObjectDemoData;
            var ff2 = ff.item(2) as ObjectDemoData;
            var ff3 = ff.item(3) as ObjectDemoData;

            Assert.AreEqual(ff0.iInt64, -23);
            Assert.AreEqual(ff1.iDouble, -22);
            Assert.AreEqual(ff2.iComplex, new Complex(-21, 0));
            Assert.AreEqual(ff3.iBigInt, new BigInteger(-20));

            var aa0 = aa.item(0) as ObjectDemoData;
            var aa1 = aa.item(1) as ObjectDemoData;
            var aa2 = aa.item(2) as ObjectDemoData;
            var aa3 = aa.item(3) as ObjectDemoData;

            Assert.AreEqual(aa0.iInt64, 0);
            Assert.AreEqual(aa1.iInt64, 1);
            Assert.AreEqual(aa2.iInt64, 2);
            Assert.AreEqual(aa3.iInt64, 3);

            var gg = np.subtract(aa, ff2);
            gg.Name += " (GG)";
            print(gg);
            AssertShape(gg, 2, 2);

            var gg0 = gg.item(0) as ObjectDemoData;
            var gg1 = gg.item(1) as ObjectDemoData;
            var gg2 = gg.item(2) as ObjectDemoData;
            var gg3 = gg.item(3) as ObjectDemoData;

            Assert.AreEqual(gg0.iInt64, 21);
            Assert.AreEqual(gg1.iDouble, 22);
            Assert.AreEqual(gg2.iComplex, new Complex(23, 0));
            Assert.AreEqual(gg3.iBigInt, new BigInteger(24));

            var hh = aa - ff;
            hh.Name += " (hh)";
            print(hh);
            AssertShape(hh, 2, 2);

            var hh0 = hh.item(0) as ObjectDemoData;
            var hh1 = hh.item(1) as ObjectDemoData;
            var hh2 = hh.item(2) as ObjectDemoData;
            var hh3 = hh.item(3) as ObjectDemoData;

            Assert.AreEqual(hh0.iInt64, 23);
            Assert.AreEqual(hh1.iDouble, 23);
            Assert.AreEqual(hh2.iComplex, new Complex(23, 0));
            Assert.AreEqual(hh3.iBigInt, new BigInteger(23));
        }

        [TestMethod]
        public void test_multiply_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(0), new ObjectDemoData(1),
                new ObjectDemoData(2), new ObjectDemoData(3)
            };

            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoData2";

            aa = aa.reshape(new shape(2, 2));
            print(aa);

            var bb = np.multiply(aa, 8);
            bb.Name += " (BB)";
            print(bb);
            AssertShape(bb, 2, 2);

            var bb0 = bb.item(0) as ObjectDemoData;
            var bb1 = bb.item(1) as ObjectDemoData;
            var bb2 = bb.item(2) as ObjectDemoData;
            var bb3 = bb.item(3) as ObjectDemoData;

            Assert.AreEqual(bb0.iInt64, 0);
            Assert.AreEqual(bb1.iInt64, 8);
            Assert.AreEqual(bb2.iInt64, 16);
            Assert.AreEqual(bb3.iInt64, 24);

            var cc = aa * 5.5;
            cc.Name += " (CC)";
            print(cc);
            AssertShape(cc, 2, 2);

            var cc0 = cc.item(0) as ObjectDemoData;
            var cc1 = cc.item(1) as ObjectDemoData;
            var cc2 = cc.item(2) as ObjectDemoData;
            var cc3 = cc.item(3) as ObjectDemoData;

            Assert.AreEqual(cc0.iInt64, 0);
            Assert.AreEqual(cc1.iDouble, 5.5);
            Assert.AreEqual(cc2.iComplex, new Complex(11, 0));
            Assert.AreEqual(cc3.iBigInt, new BigInteger(15));

            var dd = aa * new System.Numerics.Complex(7.5, 1.5);
            dd.Name += " (DD)";
            print(dd);
            AssertShape(dd, 2, 2);

            var dd0 = dd.item(0) as ObjectDemoData;
            var dd1 = dd.item(1) as ObjectDemoData;
            var dd2 = dd.item(2) as ObjectDemoData;
            var dd3 = dd.item(3) as ObjectDemoData;

            Assert.AreEqual(dd0.iInt64, 0);
            Assert.AreEqual(dd1.iDouble, 7.5);
            Assert.AreEqual(dd2.iComplex, new Complex(15, 3));
            Assert.AreEqual(dd3.iBigInt, new BigInteger(21));

            var ee = aa * new System.Numerics.BigInteger(11);
            ee.Name += " (EE)";
            print(ee);
            AssertShape(ee, 2, 2);

            var ee0 = ee.item(0) as ObjectDemoData;
            var ee1 = ee.item(1) as ObjectDemoData;
            var ee2 = ee.item(2) as ObjectDemoData;
            var ee3 = ee.item(3) as ObjectDemoData;

            Assert.AreEqual(ee0.iInt64, 0);
            Assert.AreEqual(ee1.iDouble, 11);
            Assert.AreEqual(ee2.iComplex, new Complex(22, 0));
            Assert.AreEqual(ee3.iBigInt, new BigInteger(33));

            var ff = np.multiply(aa, "23");
            ff.Name += " (FF)";
            print(ff);
            AssertShape(ff, 2, 2);

            var ff0 = ff.item(0) as ObjectDemoData;
            var ff1 = ff.item(1) as ObjectDemoData;
            var ff2 = ff.item(2) as ObjectDemoData;
            var ff3 = ff.item(3) as ObjectDemoData;

            Assert.AreEqual(ff0.iInt64, 0);
            Assert.AreEqual(ff1.iDouble, 23);
            Assert.AreEqual(ff2.iComplex, new Complex(46, 0));
            Assert.AreEqual(ff3.iBigInt, new BigInteger(69));

            var aa0 = aa.item(0) as ObjectDemoData;
            var aa1 = aa.item(1) as ObjectDemoData;
            var aa2 = aa.item(2) as ObjectDemoData;
            var aa3 = aa.item(3) as ObjectDemoData;

            Assert.AreEqual(aa0.iInt64, 0);
            Assert.AreEqual(aa1.iInt64, 1);
            Assert.AreEqual(aa2.iInt64, 2);
            Assert.AreEqual(aa3.iInt64, 3);

            var gg = np.multiply(aa, ff2);
            gg.Name += " (GG)";
            print(gg);
            AssertShape(gg, 2, 2);

            var gg0 = gg.item(0) as ObjectDemoData;
            var gg1 = gg.item(1) as ObjectDemoData;
            var gg2 = gg.item(2) as ObjectDemoData;
            var gg3 = gg.item(3) as ObjectDemoData;

            Assert.AreEqual(gg0.iInt64, 0);
            Assert.AreEqual(gg1.iDouble, 46);
            Assert.AreEqual(gg2.iComplex, new Complex(92, 0));
            Assert.AreEqual(gg3.iBigInt, new BigInteger(138));

            var hh = aa * ff;
            hh.Name += " (hh)";
            print(hh);
            AssertShape(hh, 2, 2);

            var hh0 = hh.item(0) as ObjectDemoData;
            var hh1 = hh.item(1) as ObjectDemoData;
            var hh2 = hh.item(2) as ObjectDemoData;
            var hh3 = hh.item(3) as ObjectDemoData;

            Assert.AreEqual(hh0.iInt64, 0);
            Assert.AreEqual(hh1.iDouble, 23);
            Assert.AreEqual(hh2.iComplex, new Complex(92, 0));
            Assert.AreEqual(hh3.iBigInt, new BigInteger(207));
        }

        [TestMethod]
        public void test_divide_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(200), new ObjectDemoData(208),
                new ObjectDemoData(216), new ObjectDemoData(224)
            };

            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoData2";

            aa = aa.reshape(new shape(2, 2));
            print(aa);

            var bb = np.divide(aa, 8);
            bb.Name += " (BB)";
            print(bb);
            AssertShape(bb, 2, 2);

            var bb0 = bb.item(0) as ObjectDemoData;
            var bb1 = bb.item(1) as ObjectDemoData;
            var bb2 = bb.item(2) as ObjectDemoData;
            var bb3 = bb.item(3) as ObjectDemoData;

            Assert.AreEqual(bb0.iInt64, 25);
            Assert.AreEqual(bb1.iInt64, 26);
            Assert.AreEqual(bb2.iInt64, 27);
            Assert.AreEqual(bb3.iInt64, 28);

            var cc = aa / 5.5;
            cc.Name += " (CC)";
            print(cc);
            AssertShape(cc, 2, 2);

            var cc0 = cc.item(0) as ObjectDemoData;
            var cc1 = cc.item(1) as ObjectDemoData;
            var cc2 = cc.item(2) as ObjectDemoData;
            var cc3 = cc.item(3) as ObjectDemoData;

            Assert.AreEqual(cc0.iInt64, 40);
            Assert.AreEqual(cc1.iDouble, 37.81818181818182);
            Assert.AreEqual(cc2.iComplex, new Complex(39.272727272727273, 0));
            Assert.AreEqual(cc3.iBigInt, new BigInteger(44));

            var dd = aa / new System.Numerics.Complex(7.5, 1.5);
            dd.Name += " (DD)";
            print(dd);
            AssertShape(dd, 2, 2);

            var dd0 = dd.item(0) as ObjectDemoData;
            var dd1 = dd.item(1) as ObjectDemoData;
            var dd2 = dd.item(2) as ObjectDemoData;
            var dd3 = dd.item(3) as ObjectDemoData;

            Assert.AreEqual(dd0.iInt64, 28);
            Assert.AreEqual(dd1.iDouble, 27.733333333333334);
            Assert.AreEqual(dd2.iComplex, new Complex(27.692307692307693, -5.5384615384615392));
            Assert.AreEqual(dd3.iBigInt, new BigInteger(32));

            var ee = aa / new System.Numerics.BigInteger(11);
            ee.Name += " (EE)";
            print(ee);
            AssertShape(ee, 2, 2);

            var ee0 = ee.item(0) as ObjectDemoData;
            var ee1 = ee.item(1) as ObjectDemoData;
            var ee2 = ee.item(2) as ObjectDemoData;
            var ee3 = ee.item(3) as ObjectDemoData;

            Assert.AreEqual(ee0.iInt64, 18);
            Assert.AreEqual(ee1.iDouble, 18.90909090909091);
            Assert.AreEqual(ee2.iComplex, new Complex(19.636363636363637, 0));
            Assert.AreEqual(ee3.iBigInt, new BigInteger(20));

            var ff = np.divide(aa, "23");
            ff.Name += " (FF)";
            print(ff);
            AssertShape(ff, 2, 2);

            var ff0 = ff.item(0) as ObjectDemoData;
            var ff1 = ff.item(1) as ObjectDemoData;
            var ff2 = ff.item(2) as ObjectDemoData;
            var ff3 = ff.item(3) as ObjectDemoData;

            Assert.AreEqual(ff0.iInt64, 8);
            Assert.AreEqual(ff1.iDouble, 9.0434782608695645);
            Assert.AreEqual(ff2.iComplex, new Complex(9.3913043478260878, 0));
            Assert.AreEqual(ff3.iBigInt, new BigInteger(9));

            var aa0 = aa.item(0) as ObjectDemoData;
            var aa1 = aa.item(1) as ObjectDemoData;
            var aa2 = aa.item(2) as ObjectDemoData;
            var aa3 = aa.item(3) as ObjectDemoData;

            Assert.AreEqual(aa0.iInt64, 200);
            Assert.AreEqual(aa1.iInt64, 208);
            Assert.AreEqual(aa2.iInt64, 216);
            Assert.AreEqual(aa3.iInt64, 224);

            var gg = np.divide(aa, ff2);
            gg.Name += " (GG)";
            print(gg);
            AssertShape(gg, 2, 2);

            var gg0 = gg.item(0) as ObjectDemoData;
            var gg1 = gg.item(1) as ObjectDemoData;
            var gg2 = gg.item(2) as ObjectDemoData;
            var gg3 = gg.item(3) as ObjectDemoData;

            Assert.AreEqual(gg0.iInt64, 22);
            Assert.AreEqual(gg1.iDouble, 22.148148148148145);
            Assert.AreEqual(gg2.iComplex, new Complex(22.999999999999996, 0));
            Assert.AreEqual(gg3.iBigInt, new BigInteger(24));

            var hh = (aa / ff) as ndarray;
            hh.Name += " (hh)";
            print(hh);
            AssertShape(hh, 2, 2);

            var hh0 = hh.item(0) as ObjectDemoData;
            var hh1 = hh.item(1) as ObjectDemoData;
            var hh2 = hh.item(2) as ObjectDemoData;
            var hh3 = hh.item(3) as ObjectDemoData;

            Assert.AreEqual(hh0.iInt64, 25);
            Assert.AreEqual(hh1.iDouble, 23);
            Assert.AreEqual(hh2.iComplex, new Complex(22.999999999999996, 0));
            Assert.AreEqual(hh3.iBigInt, new BigInteger(24));
        }

        [TestMethod]
        public void test_leftshift_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(200), new ObjectDemoData(208),
                new ObjectDemoData(216), new ObjectDemoData(224)
            };

            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoData2";

            aa = aa.reshape(new shape(2, 2));
            print(aa);

            var bb = np.left_shift(aa, 8);
            bb.Name += " (BB)";
            print(bb);
            AssertShape(bb, 2, 2);

            var bb0 = bb.item(0) as ObjectDemoData;
            var bb1 = bb.item(1) as ObjectDemoData;
            var bb2 = bb.item(2) as ObjectDemoData;
            var bb3 = bb.item(3) as ObjectDemoData;

            Assert.AreEqual(bb0.iInt64, 51200);
            Assert.AreEqual(bb1.iInt64, 53248);
            Assert.AreEqual(bb2.iInt64, 55296);
            Assert.AreEqual(bb3.iInt64, 57344);

            var cc = aa << 5;
            cc.Name += " (CC)";
            print(cc);
            AssertShape(cc, 2, 2);

            var cc0 = cc.item(0) as ObjectDemoData;
            var cc1 = cc.item(1) as ObjectDemoData;
            var cc2 = cc.item(2) as ObjectDemoData;
            var cc3 = cc.item(3) as ObjectDemoData;

            Assert.AreEqual(cc0.iInt64, 6400);
            Assert.AreEqual(cc1.iDouble, 208);
            Assert.AreEqual(cc2.iComplex, new Complex(216, 0));
            Assert.AreEqual(cc3.iBigInt, new BigInteger(7168));
        }

        [TestMethod]
        public void test_rightshift_operations_CUSTOMOBJECT()
        {

            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(51200), new ObjectDemoData(53248),
                new ObjectDemoData(55296), new ObjectDemoData(57344)
            };

            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoData2";

            aa = aa.reshape(new shape(2, 2));
            print(aa);

            var bb = np.right_shift(aa, 8);
            bb.Name += " (BB)";
            print(bb);
            AssertShape(bb, 2, 2);

            var bb0 = bb.item(0) as ObjectDemoData;
            var bb1 = bb.item(1) as ObjectDemoData;
            var bb2 = bb.item(2) as ObjectDemoData;
            var bb3 = bb.item(3) as ObjectDemoData;

            Assert.AreEqual(bb0.iInt64, 200);
            Assert.AreEqual(bb1.iInt64, 208);
            Assert.AreEqual(bb2.iInt64, 216);
            Assert.AreEqual(bb3.iInt64, 224);

            var cc = aa >> 5;
            cc.Name += " (CC)";
            print(cc);
            AssertShape(cc, 2, 2);

            var cc0 = cc.item(0) as ObjectDemoData;
            var cc1 = cc.item(1) as ObjectDemoData;
            var cc2 = cc.item(2) as ObjectDemoData;
            var cc3 = cc.item(3) as ObjectDemoData;

            Assert.AreEqual(cc0.iInt64, 1600);
            Assert.AreEqual(cc1.iDouble, 53248);
            Assert.AreEqual(cc2.iComplex, new Complex(55296, 0));
            Assert.AreEqual(cc3.iBigInt, new BigInteger(1792));
        }

        [TestMethod]
        public void test_bitwiseand_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(1008), new ObjectDemoData(1009),
                new ObjectDemoData(1010), new ObjectDemoData(1011)
            };
            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoDataBitwiseAnd";

            aa = aa.reshape(new shape(2, 2));
            print(aa);

            var bb = np.bitwise_and(aa, 0x0f);
            bb.Name += " (BB)";
            print(bb);
            AssertShape(bb, 2, 2);

            var bb0 = bb.item(0) as ObjectDemoData;
            var bb1 = bb.item(1) as ObjectDemoData;
            var bb2 = bb.item(2) as ObjectDemoData;
            var bb3 = bb.item(3) as ObjectDemoData;

            Assert.AreEqual(bb0.iInt64, 0);
            Assert.AreEqual(bb1.iInt64, 1);
            Assert.AreEqual(bb2.iInt64, 2);
            Assert.AreEqual(bb3.iInt64, 3);

            var cc = aa & (Int64)0xff;
            cc.Name += " (CC)";
            print(cc);
            AssertShape(cc, 2, 2);

            var cc0 = cc.item(0) as ObjectDemoData;
            var cc1 = cc.item(1) as ObjectDemoData;
            var cc2 = cc.item(2) as ObjectDemoData;
            var cc3 = cc.item(3) as ObjectDemoData;

            Assert.AreEqual(cc0.iInt64, 240);
            Assert.AreEqual(cc1.iDouble, 1009);
            Assert.AreEqual(cc2.iComplex, new Complex(1010, 0));
            Assert.AreEqual(cc3.iBigInt, new BigInteger(243));
        }

        [TestMethod]
        public void test_bitwiseor_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(1008), new ObjectDemoData(1009),
                new ObjectDemoData(1010), new ObjectDemoData(1011)
            };
            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoDataBitwiseOr";

            aa = aa.reshape(new shape(2, 2));
            print(aa);

            var bb = np.bitwise_or(aa, 0x01);
            bb.Name += " (BB)";
            print(bb);
            AssertShape(bb, 2, 2);

            var bb0 = bb.item(0) as ObjectDemoData;
            var bb1 = bb.item(1) as ObjectDemoData;
            var bb2 = bb.item(2) as ObjectDemoData;
            var bb3 = bb.item(3) as ObjectDemoData;

            Assert.AreEqual(bb0.iInt64, 1009);
            Assert.AreEqual(bb1.iInt64, 1009);
            Assert.AreEqual(bb2.iInt64, 1011);
            Assert.AreEqual(bb3.iInt64, 1011);

            var cc = aa | (Int64)0x1000;
            cc.Name += " (CC)";
            print(cc);
            AssertShape(cc, 2, 2);

            var cc0 = cc.item(0) as ObjectDemoData;
            var cc1 = cc.item(1) as ObjectDemoData;
            var cc2 = cc.item(2) as ObjectDemoData;
            var cc3 = cc.item(3) as ObjectDemoData;

            Assert.AreEqual(cc0.iInt64, 5104);
            Assert.AreEqual(cc1.iDouble, 1009);
            Assert.AreEqual(cc2.iComplex, new Complex(1010, 0));
            Assert.AreEqual(cc3.iBigInt, new BigInteger(5107));
        }

        [TestMethod]
        public void test_bitwisexor_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(1008), new ObjectDemoData(1009),
                new ObjectDemoData(1010), new ObjectDemoData(1011)
            };
            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoDataBitwiseXor";

            aa = aa.reshape(new shape(2, 2));
            print(aa);

            var bb = np.bitwise_xor(aa, 0xAAA);
            bb.Name += " (BB)";
            print(bb);
            AssertShape(bb, 2, 2);

            var bb0 = bb.item(0) as ObjectDemoData;
            var bb1 = bb.item(1) as ObjectDemoData;
            var bb2 = bb.item(2) as ObjectDemoData;
            var bb3 = bb.item(3) as ObjectDemoData;

            Assert.AreEqual(bb0.iInt64, 2394);
            Assert.AreEqual(bb1.iInt64, 2395);
            Assert.AreEqual(bb2.iInt64, 2392);
            Assert.AreEqual(bb3.iInt64, 2393);

            var cc = aa ^ (Int64)0xAAAA;
            cc.Name += " (CC)";
            print(cc);
            AssertShape(cc, 2, 2);

            var cc0 = cc.item(0) as ObjectDemoData;
            var cc1 = cc.item(1) as ObjectDemoData;
            var cc2 = cc.item(2) as ObjectDemoData;
            var cc3 = cc.item(3) as ObjectDemoData;

            Assert.AreEqual(cc0.iInt64, 43354);
            Assert.AreEqual(cc1.iDouble, 1009);
            Assert.AreEqual(cc2.iComplex, new Complex(1010, 0));
            Assert.AreEqual(cc3.iBigInt, new BigInteger(43353));

            var dd = cc ^ (Int64)0xAAAA;
            dd.Name += " (dd)";
            print(dd);
            AssertShape(dd, 2, 2);

            var dd0 = dd.item(0) as ObjectDemoData;
            var dd1 = dd.item(1) as ObjectDemoData;
            var dd2 = dd.item(2) as ObjectDemoData;
            var dd3 = dd.item(3) as ObjectDemoData;

            Assert.AreEqual(dd0.iInt64, 1008);
            Assert.AreEqual(dd1.iDouble, 1009);
            Assert.AreEqual(dd2.iComplex, new Complex(1010, 0));
            Assert.AreEqual(dd3.iBigInt, new BigInteger(1011));

        }

        [TestMethod]
        public void test_remainder_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(1008), new ObjectDemoData(1009),
                new ObjectDemoData(1010), new ObjectDemoData(1011)
            };
            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoDataRemainder";

            aa = aa.reshape(new shape(2, 2));
            print(aa);

            var bb = np.remainder(aa, 6);
            bb.Name += " (BB)";
            print(bb);
            AssertShape(bb, 2, 2);

            var bb0 = bb.item(0) as ObjectDemoData;
            var bb1 = bb.item(1) as ObjectDemoData;
            var bb2 = bb.item(2) as ObjectDemoData;
            var bb3 = bb.item(3) as ObjectDemoData;

            Assert.AreEqual(bb0.iInt64, 0);
            Assert.AreEqual(bb1.iInt64, 1);
            Assert.AreEqual(bb2.iInt64, 2);
            Assert.AreEqual(bb3.iInt64, 3);

            var cc = aa % 6;
            cc.Name += " (CC)";
            print(cc);
            AssertShape(cc, 2, 2);

            var cc0 = cc.item(0) as ObjectDemoData;
            var cc1 = cc.item(1) as ObjectDemoData;
            var cc2 = cc.item(2) as ObjectDemoData;
            var cc3 = cc.item(3) as ObjectDemoData;

            Assert.AreEqual(cc0.iInt64, 0);
            Assert.AreEqual(cc1.iDouble, 1);
            Assert.AreEqual(cc2.iComplex, new Complex(1010, 0));
            Assert.AreEqual(cc3.iBigInt, new BigInteger(3));

            var dd = aa % new ObjectDemoData(33);
            dd.Name += " (dd)";
            print(dd);
            AssertShape(dd, 2, 2);

            var dd0 = dd.item(0) as ObjectDemoData;
            var dd1 = dd.item(1) as ObjectDemoData;
            var dd2 = dd.item(2) as ObjectDemoData;
            var dd3 = dd.item(3) as ObjectDemoData;

            Assert.AreEqual(dd0.iInt64, 18);
            Assert.AreEqual(dd1.iDouble, 19);
            Assert.AreEqual(dd2.iComplex, new Complex(1010, 0));
            Assert.AreEqual(dd3.iBigInt, new BigInteger(21));


        }

        [TestMethod]
        public void test_negative_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(1008), new ObjectDemoData(1009),
                new ObjectDemoData(1010), new ObjectDemoData(1011)
            };
            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoDataNegative";

            aa = aa.reshape(new shape(2, 2));
            print(aa);

            var bb = np.negative(aa);
            bb.Name += " (BB)";
            print(bb);
            AssertShape(bb, 2, 2);

            var bb0 = bb.item(0) as ObjectDemoData;
            var bb1 = bb.item(1) as ObjectDemoData;
            var bb2 = bb.item(2) as ObjectDemoData;
            var bb3 = bb.item(3) as ObjectDemoData;

            Assert.AreEqual(bb0.iInt64, -1008);
            Assert.AreEqual(bb1.iInt64, -1009);
            Assert.AreEqual(bb2.iInt64, -1010);
            Assert.AreEqual(bb3.iInt64, -1011);

            var cc = -aa;
            cc.Name += " (CC)";
            print(cc);
            AssertShape(cc, 2, 2);

            var cc0 = cc.item(0) as ObjectDemoData;
            var cc1 = cc.item(1) as ObjectDemoData;
            var cc2 = cc.item(2) as ObjectDemoData;
            var cc3 = cc.item(3) as ObjectDemoData;

            Assert.AreEqual(cc0.iInt64, -1008);
            Assert.AreEqual(cc1.iDouble, -1009);
            Assert.AreEqual(cc2.iComplex, new Complex(-1010, 0));
            Assert.AreEqual(cc3.iBigInt, new BigInteger(-1011));

        }

        [TestMethod]
        public void test_invert_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(1008), new ObjectDemoData(1009),
                new ObjectDemoData(1010), new ObjectDemoData(1011)
            };
            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoDataInvert";

            aa = aa.reshape(new shape(2, 2));
            print(aa);

            var bb = np.invert(aa);
            bb.Name += " (BB)";
            print(bb);
            AssertShape(bb, 2, 2);

            var bb0 = bb.item(0) as ObjectDemoData;
            var bb1 = bb.item(1) as ObjectDemoData;
            var bb2 = bb.item(2) as ObjectDemoData;
            var bb3 = bb.item(3) as ObjectDemoData;

            Assert.AreEqual(bb0.iInt64, -1009);
            Assert.AreEqual(bb1.iInt64, -1010);
            Assert.AreEqual(bb2.iInt64, -1011);
            Assert.AreEqual(bb3.iInt64, -1012);

            var cc = ~aa;
            cc.Name += " (CC)";
            print(cc);
            AssertShape(cc, 2, 2);

            var cc0 = cc.item(0) as ObjectDemoData;
            var cc1 = cc.item(1) as ObjectDemoData;
            var cc2 = cc.item(2) as ObjectDemoData;
            var cc3 = cc.item(3) as ObjectDemoData;

            Assert.AreEqual(cc0.iInt64, -1009);
            Assert.AreEqual(cc1.iDouble, 1009);
            Assert.AreEqual(cc2.iComplex, new Complex(1010, 0));
            Assert.AreEqual(cc3.iBigInt, new BigInteger(-1012));

        }

        [TestMethod]
        public void test_less_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(-2), new ObjectDemoData(-1),
                new ObjectDemoData(1), new ObjectDemoData(2)
            };
            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoDataLess";

            var bb = aa < 0;
            AssertArray(bb, new Boolean[] { true, true, false, false });

            bb = np.less(aa, 0);
            AssertArray(bb, new Boolean[] { true, true, false, false });

            bb = aa < new ObjectDemoData(-1);
            AssertArray(bb, new Boolean[] { true, false, false, false });

        }

        [TestMethod]
        public void test_lessequal_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(-2), new ObjectDemoData(-1),
                new ObjectDemoData(1), new ObjectDemoData(2)
            };
            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoDataLess";

            var bb = aa <= -1;
            AssertArray(bb, new Boolean[] { true, true, false, false });

            bb = np.less_equal(aa, -1);
            AssertArray(bb, new Boolean[] { true, true, false, false });

            bb = aa <= new ObjectDemoData(-1);
            AssertArray(bb, new Boolean[] { true, true, false, false });

        }

        [TestMethod]
        public void test_equal_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(-2), new ObjectDemoData(-1),
                new ObjectDemoData(1), new ObjectDemoData(2)
            };
            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoDataLess";

            var bb = aa == -1;
            AssertArray(bb, new Boolean[] { false, true, false, false });

            bb = np.equal(aa, -1);
            AssertArray(bb, new Boolean[] { false, true, false, false });

            bb = np.equal(aa, new ObjectDemoData(-1));
            AssertArray(bb, new Boolean[] { false, true, false, false });

        }

        [TestMethod]
        public void test_notequal_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(-2), new ObjectDemoData(-1),
                new ObjectDemoData(1), new ObjectDemoData(2)
            };
            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoDataLess";

            var bb = aa != -1;
            AssertArray(bb, new Boolean[] { true, false, true, true });

            bb = np.not_equal(aa, -1);
            AssertArray(bb, new Boolean[] { true, false, true, true });

            bb = np.not_equal(aa, new ObjectDemoData(-1));
            AssertArray(bb, new Boolean[] { true, false, true, true });

        }

        [TestMethod]
        public void test_greater_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(-2), new ObjectDemoData(-1),
                new ObjectDemoData(1), new ObjectDemoData(2)
            };
            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoDataLess";

            var bb = aa > -1;
            AssertArray(bb, new Boolean[] { false, false, true, true });

            bb = np.greater(aa, -1);
            AssertArray(bb, new Boolean[] { false, false, true, true });

            bb = np.greater(aa, new ObjectDemoData(-1));
            AssertArray(bb, new Boolean[] { false, false, true, true });

        }

        [TestMethod]
        public void test_greaterequal_operations_CUSTOMOBJECT()
        {
            ObjectDemoData[] DemoData = new ObjectDemoData[]
            {
                new ObjectDemoData(-2), new ObjectDemoData(-1),
                new ObjectDemoData(1), new ObjectDemoData(2)
            };
            ndarray aa = np.array(DemoData, dtype: np.Object);
            aa.Name = "ObjectDemoDataLess";

            var bb = aa >= -1;
            AssertArray(bb, new Boolean[] { false, true, true, true });

            bb = np.greater_equal(aa, -1);
            AssertArray(bb, new Boolean[] { false, true, true, true });

            bb = np.greater_equal(aa, new ObjectDemoData(-1));
            AssertArray(bb, new Boolean[] { false, true, true, true });

        }

        [TestMethod]
        public void TODO_Cleanup_DTYPE_HASNAMES()
        {
            Assert.Fail("Cleanup the DTYPE.HASNAMES code");
        }

        #endregion




        [TestMethod]
        public void xxxtest_lexsort_1()
        {
            Assert.Fail("need to implement this now");
        }





        private string[] asstring(int[] array)
        {
            string[] output = new string[array.Length];

            for (int i = 0; i < array.Length; i++)
            {
                output[i] = array[i].ToString();
            }
            return output;
        }

        private string[,] asstring(int[,] array)
        {
            string[,] output = new string[array.GetLength(0), array.GetLength(1)];

            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    output[i, j] = array[i, j].ToString();
                }
            }
            return output;
        }
        private string[,,] asstring(int[,,] array)
        {
            string[,,] output = new string[array.GetLength(0), array.GetLength(1), array.GetLength(2)];

            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    for (int k = 0; k < array.GetLength(2); k++)
                    {
                        output[i, j, k] = array[i, j, k].ToString();
                    }
                }
            }
            return output;
        }
        private string[,,,] asstring(int[,,,] array)
        {
            string[,,,] output = new string[array.GetLength(0), array.GetLength(1), array.GetLength(2), array.GetLength(3)];

            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    for (int k = 0; k < array.GetLength(2); k++)
                    {
                        for (int l = 0; l < array.GetLength(3); l++)
                        {
                            output[i, j, k, l] = array[i, j, k, l].ToString();
                        }
                    }
                }
            }
            return output;
        }
        private string[,,,,] asstring(int[,,,,] array)
        {
            string[,,,,] output = new string[array.GetLength(0), array.GetLength(1), array.GetLength(2), array.GetLength(3), array.GetLength(4)];

            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    for (int k = 0; k < array.GetLength(2); k++)
                    {
                        for (int l = 0; l < array.GetLength(3); l++)
                        {
                            for (int m = 0; m < array.GetLength(m); m++)
                            {
                                output[i, j, k, l, m] = array[i, j, k, l, m].ToString();
                            }
                        }
                    }
                }
            }
            return output;
        }
    }
}
