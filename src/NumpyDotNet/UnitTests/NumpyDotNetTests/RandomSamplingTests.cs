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
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_FLOAT);
            AssertShape(arr, 2, 3, 4);


            np.random.seed(8765);
            fr = np.random.rand();
            arr = np.random.rand(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_FLOAT);
            AssertShape(arr, 2, 3, 4);

            print(fr);
            Assert.AreEqual(0.7841221f, fr);

            print(arr);

            var ExpectedData = new float[,,]
                { { { 0.6315085f, 0.3977592f, 0.3854478f, 0.4785422f },
                    { 0.2731604f, 0.1229814f, 0.7059686f, 0.3544289f },
                    { 0.3770753f, 0.7349837f, 0.4294267f, 0.1411899f } },
                  { { 0.5069469f, 0.2355543f, 0.956023f, 0.30962f },
                    { 0.2894487f, 0.9491175f, 0.1653861f, 0.4271754f },
                    { 0.2242058f, 0.3208179f, 0.04925749f, 0.7200206f } } };

            //AssertArray(arr, ExpectedData);

            
        }

        [TestMethod]
        public void test_randn_1()
        {

            float fr = np.random.randn();
            ndarray arr = np.random.randn(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_FLOAT);
            AssertShape(arr, 2, 3, 4);


            np.random.seed(6432);
            fr = np.random.randn();
            arr = np.random.randn(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_FLOAT);
            AssertShape(arr, 2, 3, 4);

            print(fr);
            Assert.AreEqual(1.781173E+08f, fr);

            print(arr);

            var ExpectedData = new float[,,]
                { { { 5.985523E+08f, 1.636763E+08f, 4.785839E+08f, 1.25461E+08f },
                    { 4.580316E+07f, 5.129231E+08f, 4.099223E+08f, 1.092013E+09f },
                    { 1.391873E+09f, 8.070305E+08f, 1747050.0f, 7.483687E+07f } },
                  { { 1.217551E+07f, 4.921107E+07f, 1.087742E+09f, 3.954918E+08f },
                    { 6.165971E+08f, 1.632961E+09f, 1.199443E+08f, 4.744811E+08f },
                    { 5.160406E+07f, 1.35242E+07f, 3.030721E+08f, 3.685652E+08f } } };

            //AssertArray(arr, ExpectedData);
        }

        [TestMethod]
        public void test_randint_1()
        {
            ndarray arr = np.random.randint(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT32);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new Int32[] { 2, 2, 2, 2 });

            arr = np.random.randint(20, null, 4,5);
            AssertShape(arr, 4, 5);
            print(arr);

            arr = np.random.randint(20, 21, 2, 3);
            AssertShape(arr, 2, 3);
            print(arr);
            AssertArray(arr, new Int32[,] { { 20, 20, 20 }, { 20, 20, 20 } }); 

        }

        [TestMethod]
        public void test_randint64_1()
        {
            ndarray arr = np.random.randint64(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT64);
            AssertShape(arr, 2,3, 4);
            print(arr);

            arr = np.random.randint64(4, 5);
            AssertShape(arr, 4, 5);
            print(arr);

            arr = np.random.randint64(3);
            AssertShape(arr, 3);
            print(arr);

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
            ndarray arr = np.random.standard_normal(new int[] { 5, 10 });
            print(arr);
        }

    }
}
