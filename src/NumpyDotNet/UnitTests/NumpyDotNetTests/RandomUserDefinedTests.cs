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
        public double getNextDouble(rk_state state)
        {
            return 0;
        }

        public ulong getNextUInt64(rk_state state)
        {
            return 1;
        }

        public void Seed(ulong? seedValue, rk_state state)
        {
            return;
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
            Assert.AreEqual(0.032785430047761466, f);

            arr = random.rand(new shape(5000000));

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
    }
}
