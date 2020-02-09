using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumpyLib;
using System.Threading.Tasks;

namespace NumpyDotNetTests
{
    [TestClass]
    public class PerformanceTests : TestBaseClass
    {
        //[Ignore]
        [TestMethod]
        public void test_ScalarOperationPerformance()
        {
            int LoopCount = 200;

            var matrix = np.arange(1600000, dtype: np.Int64).reshape((40, -1));

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            //matrix = matrix["1:40:2", "1:-2:3"] as ndarray;

            for (int i = 0; i < LoopCount; i++)
            {
                matrix = matrix / 3;
                matrix = matrix + i;
            }

            var output = matrix[new Slice(15, 25, 2), new Slice(15, 25, 2)];

            sw.Stop();

            Console.WriteLine(string.Format("Int64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

        [Ignore]
        [TestMethod]
        public void test_ScalarOperationPerformance_NotContiguous()
        {
            int LoopCount = 200;

            var kk = new bool[Int32.MaxValue / 2];

            var matrix = np.arange(16000000, dtype: np.Int64).reshape((40, -1));

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            matrix = matrix["1:40:2", "1:-2:3"] as ndarray;

            for (int i = 0; i < LoopCount; i++)
            {
                matrix = matrix / 3;
                matrix = matrix + i;
            }

            var output = matrix[new Slice(15, 25, 2), new Slice(15, 25, 2)];

            sw.Stop();

            Console.WriteLine(string.Format("Int64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

        [Ignore]
        [TestMethod]
        public void test_ScalarOperationPerformance_InLine()
        {
            int LoopCount = 200;

            // get the numeric ops background threads started
            var x = np.arange(1);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

            sw.Start();

            var matrix = np.arange(1600000).astype(np.Int64).reshape((40, -1));

            var src = matrix.Array.data.datap as Int64[];

            for (int i = 0; i < LoopCount; i++)
            {
                Parallel.For(0, src.Length, index => src[index] = src[index] / 3);
                Parallel.For(0, src.Length, index => src[index] = src[index] + i);
            }

            var output = matrix[new Slice(15, 25, 2), new Slice(15, 25, 2)];

            sw.Stop();

            Console.WriteLine(string.Format("Int64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

        [Ignore]
        [TestMethod]
        public void test_ScalarOperationPerformance_InLine2()
        {
            int LoopCount = 200;

            // get the numeric ops background threads started
            var x = np.arange(1);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

            sw.Start();

            var matrix = np.arange(1600000).astype(np.Int64).reshape((40, -1));

            var src = matrix.Array.data.datap as Int64[];
            int[] divisor = new int[] { 3 };

            for (int i = 0; i < LoopCount; i++)
            {
                Parallel.For(0, src.Length, index => src[index] = (long)(double)Divider(src[index], divisor[0]));

                int[] adder = new int[] { i };
                Parallel.For(0, src.Length, index => src[index] = src[index] + adder[0]);
            }

            var output = matrix[new Slice(15, 25, 2), new Slice(15, 25, 2)];

            sw.Stop();

            Console.WriteLine(string.Format("Int64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

        object Divider(dynamic src, dynamic divisor)
        {
            return (Int64)src / (double)divisor;
        }

        [Ignore]
        [TestMethod]
        public void test_MathOperation_Sin()
        {

            var a = np.arange(0, 10000000, dtype: np.Float64);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            var b = np.sin(a);

            sw.Stop();

            Console.WriteLine(string.Format("np.sin calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [TestMethod]
        public void xxx_test_KEVIN()
        {
            double sigma = 0.4;
            ndarray im;
            int x0, y0;
            int size = (int)(8 * sigma + 1);
            if (size % 2 == 0) size += 1;
            var x = np.arange(0, size, 1, np.Float32);
            var y = x.A(":", np.newaxis) * 4;

            x0 = y0 = size / 2;

            var gaus = np.exp(-4 * np.log(2) * (np.power((x - x0), 2) + np.power((y - y0), 2)) / np.power(sigma, 2));
            print(gaus);

            Assert.Fail("Take this out of final release");

        }

        [TestMethod]
        public void xxx_test_KEVIN2()
        {
            Assert.Fail("Take this out of final release");

            make_gaussian(null, 0, null, null);
        }

        [TestMethod]
        public void xxx_test_PARALLEL_MemCpy()
        {
            Assert.Fail("Try to parallel all of the for loops in MemCpy");

            make_gaussian(null, 0, null, null);
        }

        public ndarray make_gaussian(ndarray im, double sigma, ndarray xc, ndarray yc)
        {

            int size = (int)(8 * sigma + 1);

            if (size % 2 == 0)
                size += 1;

            var x = np.arange(0, size, 1, np.Float32);
            var y = x.A(":", np.newaxis) * 4;


            int x0 = size / 2;
            int y0 = size / 2;

            var gaus = np.exp(-4 * np.log(2) * (np.power((x - x0),2) + np.power((y - y0),2) / np.power(sigma,2)));

            im[xc - (size / 2) + ":" + (xc + size / 2 + 1), (yc - size / 2) + ":" + (yc + size / 2 + 1)] = gaus;

            return im;

        }


    }
}
