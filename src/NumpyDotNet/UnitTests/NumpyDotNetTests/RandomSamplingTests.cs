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
        public void test_randd_1()
        {

            double dr = np.random.randd();
            ndarray arr = np.random.randd(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 2, 3, 4);


            np.random.seed(6432);
            dr = np.random.randd();
            arr = np.random.randd(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 2, 3, 4);

            print(dr);
            Assert.AreEqual(178117288.340587.ToString(), dr.ToString());

            print(arr);

            var ExpectedData = new double[,,]
         { { { 598552268.405803, 163676321.111939, 478583886.092547, 125461049.513929 },
             { 45803165.3247941, 512923082.843819, 409922266.507972, 1092012782.24225 },
             { 1391872905.49059, 807030483.924994, 1747050.3741545, 74836869.4357915 } },
           { { 12175507.5988969, 49211069.4265853, 1087741937.25169, 395491778.646227 },
             { 616597143.760458, 1632960594.76456, 119944335.963269, 474481135.729486 },
             { 51604064.7015829, 13524202.6515921, 303072113.281086, 368565244.525343 } } };

            //AssertArray(arr, ExpectedData);
        }

        [TestMethod]
        public void test_bytes_1()
        {

            byte br = np.random.bytes();
            ndarray arr = np.random.bytes(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UBYTE);
            AssertShape(arr, 2, 3, 4);


            np.random.seed(6432);
            br = np.random.bytes();
            arr = np.random.bytes(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UBYTE);
            AssertShape(arr, 2, 3, 4);

            print(br);
            Assert.AreEqual(154, br);

            print(arr);

            var ExpectedData = new byte[,,]
                { { { 204, 182, 44, 21 },
                    { 138, 6, 211, 169 },
                    { 171, 7, 115, 155 } },
                    { { 139, 82, 248, 91 },
                    { 16, 98, 21, 14 },
                    { 98, 197, 39, 195 } } };

            AssertArray(arr, ExpectedData);
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
