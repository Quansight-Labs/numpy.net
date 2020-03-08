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
    public class RandomSamplingTests : TestBaseClass
    {
        #region Simple Random Data
        [TestMethod]
        public void test_rand_1()
        {
            ndarray arr = np.random.rand(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 2, 3, 4);

            np.random.seed(8765);
            float f = np.random.rand();
            print(f);
            Assert.AreEqual(0.03278543f, f);

            arr = np.random.rand(5000000);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(0.99999999331246381, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(7.223258213784334e-08, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(0.49987999522694609, avg.GetItem(0));
        }

        [TestMethod]
        public void test_randn_1()
        {
            float fr = np.random.randn();
            ndarray arr = np.random.randn(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 2, 3, 4);

            np.random.seed(1234);

            fr = np.random.randn();
            print(fr);
            Assert.AreEqual(0.471435163732f, fr);

            arr = np.random.randn(5000000);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(5.19094054905629, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(-5.387212036872835, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(-0.00028345468151361842, avg.GetItem(0));
        }

        [TestMethod]
        public void test_randbool_1()
        {
            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.Bool);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_BOOL);
            AssertShape(arr, 4);
            print(arr);
            //AssertArray(arr, new bool[] { false, false, false, false });

            arr = np.random.randint(20, null, new shape(4, 5), dtype: np.Bool);
            AssertShape(arr, 4, 5);
            print(arr);

            arr = np.random.randint(20, 21, new shape(2, 3), dtype: np.Bool);
            AssertShape(arr, 2, 3);
            print(arr);
            //AssertArray(arr, new SByte[,] { { 20, 20, 20 }, { 20, 20, 20 } });

            arr = np.random.randint(-2, 3, new shape(5000000), dtype: np.Bool);
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));
        }

        [TestMethod]
        public void test_randint8_1()
        {
            np.random.seed(9292);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.Int8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_BYTE);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new SByte[] { 2, 2, 2, 2 });

            arr = np.random.randint(2, 8, new shape(5000000), dtype: np.Int8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_BYTE);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((sbyte)7, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((sbyte)2, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(4.4994078, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new sbyte[] { 3, 6, 7, 5, 5, 5, 7, 7, 4, 2 });


            arr = np.random.randint(-2, 3, new shape(5000000), dtype: np.Int8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_BYTE);

            amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((sbyte)2, amax.GetItem(0));

            amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((sbyte)(-2), amin.GetItem(0));

            avg = np.average(arr);
            print(avg);
            Assert.AreEqual(-0.0003146, avg.GetItem(0));

            first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new sbyte[] { 0, 1, -1, -2, -2, -1, 2, -1, -2, -2 });

        }



        [TestMethod]
        public void test_randuint8_1()
        {
            np.random.seed(1313);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.UInt8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UBYTE);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new Byte[] { 2, 2, 2, 2 });

            arr = np.random.randint(2, 128, new shape(5000000), dtype: np.UInt8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UBYTE);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((byte)127, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((byte)2, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(64.5080776, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new byte[] { 114, 37, 22, 53, 90, 27, 34, 98, 123, 114 });

        }

        [TestMethod]
        public void test_randint16_1()
        {
            np.random.seed(8381);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.Int16);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT16);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new Int16[] { 2, 2, 2, 2 });

            arr = np.random.randint(2, 2478, new shape(5000000), dtype: np.Int16);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT16);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((Int16)2477, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((Int16)2, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(1239.6188784, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new Int16[] { 1854, 1641, 1219, 1190, 1714, 644, 441, 484, 579, 393 });


            arr = np.random.randint(-2067, 3000, new shape(5000000), dtype: np.Int16);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT16);

            amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((Int16)2999, amax.GetItem(0));

            amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((Int16)(-2067), amin.GetItem(0));

            avg = np.average(arr);
            print(avg);
            Assert.AreEqual(466.3735356, avg.GetItem(0));

            first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new Int16[] {-1761, -1165, 2645, -1210, 1741, -1692, -1042, -354, 2637, -706 });
        }


        [TestMethod]
        public void test_randuint16_1()
        {
            np.random.seed(5555);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.UInt16);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UINT16);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new UInt16[] { 2, 2, 2, 2 });

            arr = np.random.randint(23, 12801, new shape(5000000), dtype: np.UInt16);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UINT16);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((UInt16)12800, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((UInt16)23, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(6411.5873838, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new UInt16[] { 3781, 11097, 4916, 2759, 12505, 827, 12017, 3982, 7669, 11128 });
        }

        [TestMethod]
        public void test_randint_1()
        {
            np.random.seed(701);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.Int32);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT32);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new Int32[] { 2, 2, 2, 2 });

            arr = np.random.randint(9, 128000, new shape(5000000), dtype: np.Int32);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT32);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((Int32)127999, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((Int32)9, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(64017.8102496, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new Int32[] { 116207, 47965, 90333, 86510, 49239, 115311, 100417, 99653, 14878, 92386 });


            arr = np.random.randint(-20000, 300000, new shape(5000000), dtype: np.Int32);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT32);

            amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((Int32)299999, amax.GetItem(0));

            amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((Int32)(-20000), amin.GetItem(0));

            avg = np.average(arr);
            print(avg);
            Assert.AreEqual(139952.5096864, avg.GetItem(0));

            first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new Int32[] { 211742, 98578, 96618, 50714, 285136, 270233, 186899, 278442, 215280, 268431 });
        }


        [TestMethod]
        public void test_randuint_1()
        {
            np.random.seed(8357);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.UInt32);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UINT32);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new UInt32[] { 2, 2, 2, 2 });

            arr = np.random.randint(29, 13000, new shape(5000000), dtype: np.UInt32);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UINT32);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((UInt32)12999, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((UInt32)29, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(6512.7336774, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new UInt32[] { 6511, 10414, 12909, 5962, 353, 8634, 8330, 1452, 5009, 5444 });
        }

        [TestMethod]
        public void test_randint64_1()
        {
            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.Int64);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT64);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new Int64[] { 2, 2, 2, 2 });

            arr = np.random.randint(20, null, new shape(4, 5), dtype: np.Int64);
            AssertShape(arr, 4, 5);
            print(arr);

            arr = np.random.randint(20, 21, new shape(2, 3), dtype: np.Int64);
            AssertShape(arr, 2, 3);
            print(arr);
            AssertArray(arr, new Int64[,] { { 20, 20, 20 }, { 20, 20, 20 } });

            arr = np.random.randint(-2, 3, new shape(5000000), dtype: np.Int64);
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));

        }


        [TestMethod]
        public void test_randuint64_1()
        {
            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.UInt64);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UINT64);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new UInt64[] { 2, 2, 2, 2 });

            arr = np.random.randint(20, null, new shape(4, 5), dtype: np.UInt64);
            AssertShape(arr, 4, 5);
            print(arr);

            arr = np.random.randint(20, 21, new shape(2, 3), dtype: np.UInt64);
            AssertShape(arr, 2, 3);
            print(arr);
            AssertArray(arr, new UInt64[,] { { 20, 20, 20 }, { 20, 20, 20 } });

            arr = np.random.randint(2, 5, new shape(5000000), dtype: np.UInt64);
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));
        }


        [TestMethod]
        public void test_rand_bytes_1()
        {

            byte br = np.random.getbyte();
            var bytes = np.random.bytes(24);
            Assert.AreEqual(bytes.Length, 24);

            np.random.seed(6432);
            br = np.random.getbyte();
            var arr = np.array(np.random.bytes(24)).reshape(2,3,4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UBYTE);
            AssertShape(arr, 2, 3, 4);

            print(br);
            Assert.AreEqual(236, br);

            print(arr);

            var ExpectedData = new byte[,,]
            { { { 134, 74, 135, 220 },
                { 88, 52, 7, 2 },
                { 46, 199, 221, 108 } },
              { { 253, 137, 142, 205 },
                { 213, 153, 46, 81 },
                { 79, 95, 51, 55 } } };

            AssertArray(arr, ExpectedData);
        }
        #endregion

        #region shuffle/permutation

        [TestMethod]
        public void test_rand_shuffle_1()
        {
            var arr = np.arange(10);
            np.random.shuffle(arr);
            print(arr);

            arr = np.arange(9).reshape((3,3));
            print(arr);

            np.random.shuffle(arr);
            print(arr);

        }

        [TestMethod]
        public void test_rand_permutation_1()
        {
            var arr = np.random.permutation(10);
            print(arr);
            AssertShape(arr, 10);

            arr = np.random.permutation(np.arange(5));
            print(arr);
            AssertShape(arr, 5);

        }

        #endregion

        [TestMethod]
        public void test_rand_beta_1()
        {
            var a = np.arange(1,11, dtype: np.Float64);
            var b = np.arange(1,11, dtype: np.Float64);

            ndarray arr = np.random.beta(b, b, new shape(10));
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 10);
            print(arr);
        }

        [TestMethod]
        public void test_rand_binomial_1()
        {
            np.random.seed(123);

            ndarray arr = np.random.binomial(9, 0.1, new shape(20));
            AssertArray(arr, new Int64[] { 1, 0, 0, 1, 1, 1, 3, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1 });
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT64);

            var s = np.sum(arr == 0);
            Assert.AreEqual(6, s.GetItem(0));


            arr = np.random.binomial(9, 0.1, new shape(20000));
            s = np.sum(arr == 0);
            Assert.AreEqual(7711, s.GetItem(0));
            print(s);
        }


        [TestMethod]
        public void test_uniform_1()
        {
            ndarray arr = np.random.uniform(-1, 1, 40);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 40);
            print(arr);
        }


        [TestMethod]
        public void test_standard_normal_1()
        {
            np.random.seed(1234);
            ndarray arr = np.random.standard_normal(5000000);
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));

        }

    }
}
