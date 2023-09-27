using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumpyLib;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNetTests
{
    [TestClass]
    public class TuningTests : TestBaseClass
    {
        [TestMethod]
        public void test_EnableTryCatch_1()
        {
            np.tuning.EnableTryCatchOnCalculations = true;
            Assert.AreEqual(true, np.tuning.EnableTryCatchOnCalculations);

            np.tuning.EnableTryCatchOnCalculations = false;
            Assert.AreEqual(false, np.tuning.EnableTryCatchOnCalculations);

        }

        [TestMethod]
        public void test_EnableTryCatch_2()
        {
           // np.tuning.EnableTryCatchOnCalculations = true;

            bool ?TaskTryCatchOnCalculationsStart = null;
            bool ?TaskTryCatchOnCalculationsSetting = null;

            var t = Task.Run(() =>
            {
                TaskTryCatchOnCalculationsStart = np.tuning.EnableTryCatchOnCalculations;
                np.tuning.EnableTryCatchOnCalculations = false;
                TaskTryCatchOnCalculationsSetting = np.tuning.EnableTryCatchOnCalculations;
            });
            t.Wait();



            Assert.AreEqual(true, np.tuning.EnableTryCatchOnCalculations);
            Assert.AreEqual(false, TaskTryCatchOnCalculationsSetting.Value);
            Assert.AreEqual(true, TaskTryCatchOnCalculationsStart.Value);


        }

        [TestMethod]
        public void test_largearray_matmul_INT64_1_tuning()
        {

            var largeTests = new LargeArrayTests();
            largeTests.test_largearray_matmul_INT64_1();

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();


            np.tuning.EnableTryCatchOnCalculations = true;
            sw.Restart();
            largeTests.test_largearray_matmul_INT64_1();
            sw.Stop();

            var EnabledTryCatchMS = sw.ElapsedMilliseconds;

            np.tuning.EnableTryCatchOnCalculations = false;
            sw.Restart();
            largeTests.test_largearray_matmul_INT64_1();
            sw.Stop();

            np.tuning.EnableTryCatchOnCalculations = true;

            var DisabledTryCatchMS = sw.ElapsedMilliseconds;

            Console.WriteLine(string.Format("Enabled tryCatch calculations took {0} milliseconds\n", EnabledTryCatchMS));
            Console.WriteLine(string.Format("Disabled tryCatch calculations took {0} milliseconds\n", DisabledTryCatchMS));
        }

        [TestMethod]
        public void test_largearray_matmul_INT64_2_tuning()
        {

            var largeTests = new LargeArrayTests();
            largeTests.test_largearray_matmul_INT64_2();

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();


            np.tuning.EnableTryCatchOnCalculations = true;
            sw.Restart();
            largeTests.test_largearray_matmul_INT64_2();
            sw.Stop();

            var EnabledTryCatchMS = sw.ElapsedMilliseconds;

            np.tuning.EnableTryCatchOnCalculations = false;
            sw.Restart();
            largeTests.test_largearray_matmul_INT64_2();
            sw.Stop();

            np.tuning.EnableTryCatchOnCalculations = true;

            var DisabledTryCatchMS = sw.ElapsedMilliseconds;

            Console.WriteLine(string.Format("Enabled tryCatch calculations took {0} milliseconds\n", EnabledTryCatchMS));
            Console.WriteLine(string.Format("Disabled tryCatch calculations took {0} milliseconds\n", DisabledTryCatchMS));
        }

        [TestMethod]
        public void test_LargeArrayAdd_1()
        {

            var largeTests = new LargeArrayTests();
            largeTests.test_largearray_add_INT64_1();

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();


            np.tuning.EnableTryCatchOnCalculations = true;
            sw.Restart();
            largeTests.test_largearray_add_INT64_1();
            sw.Stop();

            var EnabledTryCatchMS = sw.ElapsedMilliseconds;

            np.tuning.EnableTryCatchOnCalculations = false;
            sw.Restart();
            largeTests.test_largearray_add_INT64_1();
            sw.Stop();

            np.tuning.EnableTryCatchOnCalculations = true;

            var DisabledTryCatchMS = sw.ElapsedMilliseconds;

            Console.WriteLine(string.Format("Enabled tryCatch calculations took {0} milliseconds\n", EnabledTryCatchMS));
            Console.WriteLine(string.Format("Disabled tryCatch calculations took {0} milliseconds\n", DisabledTryCatchMS));
        }





    }
}