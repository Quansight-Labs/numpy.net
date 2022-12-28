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
    public class UserDefinedRandomGenerator : NumpyDotNet.RandomAPI.IRandomGenerator
    {
        private System.Random rnd = null;

 

        public double getNextDouble(rk_state state)
        {
            return rnd.NextDouble();
        }

        public ulong getNextUInt64(rk_state state)
        {
            ulong randomValue = (ulong)rnd.Next() << 32 + rnd.Next();
            return randomValue;
        }

        public void Seed(ulong? seedValue, rk_state state)
        {
            if (seedValue.HasValue)
                rnd = new Random((int)seedValue.Value);
            else
                rnd = new Random();
            return;
        }

        public string ToSerialization()
        {
            throw new NotImplementedException("This random number generator does not support serialization");
        }

        public void FromSerialization(string SerializedFormat)
        {
            throw new NotImplementedException("This random number generator does not support serialization");
        }
    }


    [TestClass]
    public class RandomUserDefinedTests : TestBaseClass
    {
        [TestMethod]
        public void test_rand_UD_1()
        {
            var random = new np.random(new UserDefinedRandomGenerator());
            ndarray arr = random.rand(new shape(2, 3, 4));
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 2, 3, 4);

            random.seed(8765);
            double f = random.rand();
            print(f);
            Assert.AreEqual(0.78412213771795958, f);

            arr = random.rand(new shape(5000000));

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(0.99999997671693563, (double)amax);

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(5.1688402915228346E-08, (double)amin);

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(0.50011801294498859, (double)avg);
        }

        [TestMethod]
        public void test_randn_UC_1()
        {
            var random = new np.random(new UserDefinedRandomGenerator());

            double fr = random.randn();
            ndarray arr = random.randn(new shape(2, 3, 4));
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 2, 3, 4);

            random.seed(1234);

            fr = random.randn();
            print(fr);
            //Assert.AreEqual(-0.29042058667075177, fr);

            arr = random.randn(new shape(5000000));
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(5.070295032710753, (double)amax);

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(-5.04633622266897, (double)amin);

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(-0.00044428007674556522, (double)avg);
        }

        [TestMethod]
        public void test_randbool_UC_1()
        {
            var random = new np.random(new UserDefinedRandomGenerator());

            ndarray arr = random.randint(2, 3, new shape(4), dtype: np.Bool);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_BOOL);
            AssertShape(arr, 4);
            print(arr);
            //AssertArray(arr, new bool[] { false, false, false, false });

            arr = random.randint(20, null, new shape(4, 5), dtype: np.Bool);
            AssertShape(arr, 4, 5);
            print(arr);

            arr = random.randint(20, 21, new shape(2, 3), dtype: np.Bool);
            AssertShape(arr, 2, 3);
            print(arr);
            //AssertArray(arr, new SByte[,] { { 20, 20, 20 }, { 20, 20, 20 } });

            arr = random.randint(-2, 3, new shape(5000000), dtype: np.Bool);
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));
        }

        [TestMethod]
        public void test_randint8_UD_1()
        {
            var random = new np.random(new UserDefinedRandomGenerator());

            random.seed(9292);

            ndarray arr = random.randint(2, 3, new shape(4), dtype: np.Int8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_BYTE);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new SByte[] { 2, 2, 2, 2 });

            arr = random.randint(2, 8, new shape(5000000), dtype: np.Int8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_BYTE);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((sbyte)7, (sbyte)amax);

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((sbyte)2, (sbyte)amin);

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(2.4605506, (double)avg);

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new sbyte[] { 2,2,2,2,2,2,2,2,2,2 });


            arr = random.randint(-2, 3, new shape(5000000), dtype: np.Int8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_BYTE);

            amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((sbyte)2, (sbyte)amax);

            amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((sbyte)(-2), (sbyte)amin);

            avg = np.average(arr);
            print(avg);
            Assert.AreEqual(-1.6648296, (double)avg);

            first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new sbyte[] { -2, -2, 0, -1, -2, -2, -2, -2, -2, -2 });
         }
    }
}
