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
            float fr = np.random.rand();
            ndarray arr = np.random.rand(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 2, 3, 4);


            np.random.seed(8765);
            fr = np.random.rand();
            arr = np.random.rand(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 2, 3, 4);

            print(fr);
            Assert.AreEqual(0.03278543f, fr);

            print(arr);

            np.random.seed(null);
            arr = np.random.rand(500000);
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));

        }

        [TestMethod]
        public void test_randn_1()
        {

            float fr = np.random.randn();
            ndarray arr = np.random.randn(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 2, 3, 4);


            np.random.seed(6432);
            fr = np.random.randn();
            arr = np.random.randn(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 2, 3, 4);

            print(fr);
            Assert.AreEqual(0.749749541f, fr);

            print(arr);

            np.random.seed(null);
            arr = np.random.randn(500000);
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));


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
            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.Int8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_BYTE);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new SByte[] { 2, 2, 2, 2 });

            arr = np.random.randint(20, null, new shape(4, 5), dtype: np.Int8);
            AssertShape(arr, 4, 5);
            print(arr);

            arr = np.random.randint(20, 21, new shape(2, 3), dtype: np.Int8);
            AssertShape(arr, 2, 3);
            print(arr);
            AssertArray(arr, new SByte[,] { { 20, 20, 20 }, { 20, 20, 20 } });

            arr = np.random.randint(-2, 3, new shape(5000000), dtype: np.Int8);
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));
        }



        [TestMethod]
        public void test_randuint8_1()
        {
            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.UInt8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UBYTE);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new Byte[] { 2, 2, 2, 2 });

            arr = np.random.randint(20, null, new shape(4, 5), dtype: np.UInt8);
            AssertShape(arr, 4, 5);
            print(arr);

            arr = np.random.randint(20, 21, new shape(2, 3), dtype: np.UInt8);
            AssertShape(arr, 2, 3);
            print(arr);
            AssertArray(arr, new Byte[,] { { 20, 20, 20 }, { 20, 20, 20 } });

            arr = np.random.randint(2, 5, new shape(5000000), dtype: np.UInt8);
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));
        }

        [TestMethod]
        public void test_randint16_1()
        {
            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.Int16);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT16);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new Int16[] { 2, 2, 2, 2 });

            arr = np.random.randint(20, null, new shape(4, 5), dtype: np.Int16);
            AssertShape(arr, 4, 5);
            print(arr);

            arr = np.random.randint(20, 21, new shape(2, 3), dtype: np.Int16);
            AssertShape(arr, 2, 3);
            print(arr);
            AssertArray(arr, new Int16[,] { { 20, 20, 20 }, { 20, 20, 20 } });

            arr = np.random.randint(-2, 3, new shape(5000000), dtype: np.Int16);
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));
        }


        [TestMethod]
        public void test_randuint16_1()
        {
            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.UInt16);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UINT16);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new UInt16[] { 2, 2, 2, 2 });

            arr = np.random.randint(20, null, new shape(4, 5), dtype: np.UInt16);
            AssertShape(arr, 4, 5);
            print(arr);

            arr = np.random.randint(20, 21, new shape(2, 3), dtype: np.UInt16);
            AssertShape(arr, 2, 3);
            print(arr);
            AssertArray(arr, new UInt16[,] { { 20, 20, 20 }, { 20, 20, 20 } });

            arr = np.random.randint(2, 5, new shape(5000000), dtype: np.UInt16);
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));
        }

        [TestMethod]
        public void test_randint_1()
        {
            ndarray arr = np.random.randint(2, 3, new shape(4));
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT32);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new Int32[] { 2, 2, 2, 2 });

            arr = np.random.randint(20, null, new shape(4,5));
            AssertShape(arr, 4, 5);
            print(arr);

            arr = np.random.randint(20, 21, new shape(2, 3));
            AssertShape(arr, 2, 3);
            print(arr);
            AssertArray(arr, new Int32[,] { { 20, 20, 20 }, { 20, 20, 20 } });

            arr = np.random.randint(-2, 3, new shape(5000000));
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));
        }


        [TestMethod]
        public void test_randuint_1()
        {
            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.UInt32);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UINT32);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new UInt32[] { 2, 2, 2, 2 });

            arr = np.random.randint(20, null, new shape(4, 5), dtype: np.UInt32);
            AssertShape(arr, 4, 5);
            print(arr);

            arr = np.random.randint(20, 21, new shape(2, 3), dtype: np.UInt32);
            AssertShape(arr, 2, 3);
            print(arr);
            AssertArray(arr, new UInt32[,] { { 20, 20, 20 }, { 20, 20, 20 } });

            arr = np.random.randint(2, 5, new shape(5000000), dtype: np.UInt32);
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));
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
