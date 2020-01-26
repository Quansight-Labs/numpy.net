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
        [TestMethod]
        public void test_ScalarOperationPerformance()
        {
            int LoopCount = 200;

            var kk = new bool[Int32.MaxValue / 2];

            // get the numeric ops background threads started
            var x = np.arange(1);
            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

            sw.Start();


            var matrix = np.arange(1600000).astype(np.Int64).reshape((40, -1));

            // matrix = (ndarray)(matrix / np.arange(40).astype(np.Int32));

            //matrix = matrix["1:3:2", "1:-2:3"] as ndarray;

            for (int i = 0; i < LoopCount; i++)
            {
                matrix = matrix / 3;
                matrix = matrix / 1;
            }

            var output = matrix[new Slice(15, 25, 2), new Slice(15, 25, 2)];

            sw.Stop();

            Console.WriteLine(string.Format("Int64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

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


    }
}
