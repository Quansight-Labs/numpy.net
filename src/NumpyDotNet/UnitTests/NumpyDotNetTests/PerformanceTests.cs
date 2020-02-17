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
        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation()
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
        public void Performance_ScalarOperation_NotContiguous()
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
        public void Performance_ScalarOperation_InLine()
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
        public void Performance_ScalarOperation_InLine2()
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
        public void Performance_MathOperation_Sin()
        {

            var a = np.arange(0, 10000000, dtype: np.Float64);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            var b = np.sin(a);

            sw.Stop();

            Console.WriteLine(string.Format("np.sin calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Float64);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.reduce(UFuncOperation.add, a);
                //Assert.AreEqual(49999995000000.0, b.item(0));
               // print(b);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }



        [Ignore]
        [TestMethod]
        public void Performance_AddAccumulate()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Float64);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.accumulate(UFuncOperation.add, a);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce_IDEAL()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Float64);
            var aa = a.AsDoubleArray();

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                double total = 0;

                long O2_CalculatedStep = 1;
                long O2_CalculatedOffset = 0;
                try
                {
                    for (int j = 0; j < aa.Length; j++)
                    {

                        long O2_Index = ((j * O2_CalculatedStep) + O2_CalculatedOffset);
                        double O2Value = aa[O2_Index];                                            // get operand 2

                        total = total + O2Value;
                    }
                    Assert.AreEqual(49999995000000.0, total);
                }
                catch (Exception ex)
                {

                }

     
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduceAt()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Float64).reshape((40, -1));

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.reduceat(UFuncOperation.add, a, new long[] { 10, 20, 30, 39 });
                print(b.shape);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }
        [Ignore]
        [TestMethod]
        public void Performance_AddOuter()
        {

            int LoopCount = 200;
            var c = np.arange(0, 2000, dtype: np.Float64);
            var a = c[new Slice(0, 2000, 2)];

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
               var b = np.ufunc.outer(UFuncOperation.add, np.Float64, a, a);
               // print(b.shape);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        //[Ignore]
        [TestMethod]
        public void Performance_AddOuter_NotSameType()
        {

            int LoopCount = 200;
            var a1 = np.arange(0, 1000, dtype: np.Float64);
            var a2 = np.arange(0, 1000, dtype: np.Int16);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.outer(UFuncOperation.add, np.Float64, a2, a1);
                // print(b.shape);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }




    }
}
