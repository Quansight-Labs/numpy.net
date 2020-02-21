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
        #region DOUBLE tests
        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_DOUBLE()
        {
            int LoopCount = 200;

            var matrix = np.arange(1600000, dtype: np.Float64).reshape((40, -1));

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

            Console.WriteLine(string.Format("DOUBLE calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }
   
        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_NotContiguous_DOUBLE()
        {
            int LoopCount = 200;

            var kk = new bool[Int32.MaxValue / 2];

            var matrix = np.arange(16000000, dtype: np.Float64).reshape((40, -1));

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

            Console.WriteLine(string.Format("DOUBLE calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_InLine_DOUBLE()
        {
            int LoopCount = 200;

            // get the numeric ops background threads started
            var x = np.arange(1);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

            sw.Start();

            var matrix = np.arange(1600000).astype(np.Float64).reshape((40, -1));

            var src = matrix.Array.data.datap as double[];

            for (int i = 0; i < LoopCount; i++)
            {
                Parallel.For(0, src.Length, index => src[index] = src[index] / 3);
                Parallel.For(0, src.Length, index => src[index] = src[index] + i);
            }

            var output = matrix[new Slice(15, 25, 2), new Slice(15, 25, 2)];

            sw.Stop();

            Console.WriteLine(string.Format("DOUBLE calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }
   

        [Ignore]
        [TestMethod]
        public void Performance_MathOperation_Sin_DOUBLE()
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
        public void Performance_AddReduce_DOUBLE()
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
        public void Performance_AddReduce2_DOUBLE()
        {

            int LoopCount = 200;
            var a = np.arange(0, 4000 * 10 * 4000, dtype: np.Float64).reshape(-1, 4000);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.reduce(UFuncOperation.add, a);
                var c = np.sum(b);
                Assert.AreEqual(1.279999992E+16, c.item(0));
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }



        [Ignore]
        [TestMethod]
        public void Performance_AddAccumulate_DOUBLE()
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

            Console.WriteLine(string.Format("AddAccumulate calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddAccumulate2_DOUBLE()
        {

            int LoopCount = 200;
            var a = np.arange(0, 4000 * 4000, dtype: np.Float64).reshape(4000,4000);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.accumulate(UFuncOperation.add, a);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddAccumulate calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce_IDEAL_DOUBLE()
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
        public void Performance_AddReduceAt_DOUBLE()
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
        public void Performance_AddOuter_DOUBLE()
        {

            int LoopCount = 200;
            var a = np.arange(0, 1000, dtype: np.Float64);

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

        [Ignore]
        [TestMethod]
        public void Performance_AddOuter_NotSameType_DOUBLE()
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
        #endregion

        #region FLOAT tests
        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_FLOAT()
        {
            int LoopCount = 200;

            var matrix = np.arange(1600000, dtype: np.Float32).reshape((40, -1));

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

            Console.WriteLine(string.Format("FLOAT calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_NotContiguous_FLOAT()
        {
            int LoopCount = 200;

            var kk = new bool[Int32.MaxValue / 2];

            var matrix = np.arange(16000000, dtype: np.Float32).reshape((40, -1));

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

            Console.WriteLine(string.Format("FLOAT calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_InLine_FLOAT()
        {
            int LoopCount = 200;

            // get the numeric ops background threads started
            var x = np.arange(1);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

            sw.Start();

            var matrix = np.arange(1600000).astype(np.Float32).reshape((40, -1));

            var src = matrix.Array.data.datap as double[];

            for (int i = 0; i < LoopCount; i++)
            {
                Parallel.For(0, src.Length, index => src[index] = src[index] / 3);
                Parallel.For(0, src.Length, index => src[index] = src[index] + i);
            }

            var output = matrix[new Slice(15, 25, 2), new Slice(15, 25, 2)];

            sw.Stop();

            Console.WriteLine(string.Format("FLOAT calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }


        [Ignore]
        [TestMethod]
        public void Performance_MathOperation_Sin_FLOAT()
        {

            var a = np.arange(0, 10000000, dtype: np.Float32);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            var b = np.sin(a);

            sw.Stop();

            Console.WriteLine(string.Format("np.sin calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce_FLOAT()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Float32);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.reduce(UFuncOperation.add, a);
                Assert.AreEqual(4.871488E+13f, b.item(0));
                // print(b);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce2_FLOAT()
        {

            int LoopCount = 200;
            var a = np.arange(0, 4000 * 10 * 4000, dtype: np.Float32).reshape(-1, 4000);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.reduce(UFuncOperation.add, a);
                var c = np.sum(b);

                Assert.AreEqual(1.27994739E+16f, c.item(0));
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }



        [Ignore]
        [TestMethod]
        public void Performance_AddAccumulate_FLOAT()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Float32);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.accumulate(UFuncOperation.add, a);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddAccumulate calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddAccumulate2_FLOAT()
        {

            int LoopCount = 200;
            var a = np.arange(0, 4000 * 4000, dtype: np.Float32).reshape(4000, 4000);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.accumulate(UFuncOperation.add, a);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddAccumulate calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce_IDEAL_FLOAT()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Float32);
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
        public void Performance_AddReduceAt_FLOAT()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Float32).reshape((40, -1));

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
        public void Performance_AddOuter_FLOAT()
        {

            int LoopCount = 200;
            var a = np.arange(0, 1000, dtype: np.Float32);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.outer(UFuncOperation.add, np.Float32, a, a);
                // print(b.shape);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddOuter_NotSameType_FLOAT()
        {

            int LoopCount = 200;
            var a1 = np.arange(0, 1000, dtype: np.Float32);
            var a2 = np.arange(0, 1000, dtype: np.Int16);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.outer(UFuncOperation.add, np.Float32, a2, a1);
                // print(b.shape);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }
        #endregion

        #region INT64 tests
        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_INT64()
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

            Console.WriteLine(string.Format("INT64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_NotContiguous_INT64()
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

            Console.WriteLine(string.Format("INT64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_InLine_INT64()
        {
            int LoopCount = 200;

            // get the numeric ops background threads started
            var x = np.arange(1);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

            sw.Start();

            var matrix = np.arange(1600000).astype(np.Int64).reshape((40, -1));

            var src = matrix.Array.data.datap as double[];

            for (int i = 0; i < LoopCount; i++)
            {
                Parallel.For(0, src.Length, index => src[index] = src[index] / 3);
                Parallel.For(0, src.Length, index => src[index] = src[index] + i);
            }

            var output = matrix[new Slice(15, 25, 2), new Slice(15, 25, 2)];

            sw.Stop();

            Console.WriteLine(string.Format("INT64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }


        [Ignore]
        [TestMethod]
        public void Performance_MathOperation_Sin_INT64()
        {

            var a = np.arange(0, 10000000, dtype: np.Int64);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            var b = np.sin(a);

            sw.Stop();

            Console.WriteLine(string.Format("np.sin calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce_INT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Int64);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.reduce(UFuncOperation.add, a);
                Assert.AreEqual(49999995000000, b.item(0));
                // print(b);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce2_INT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 4000 * 10 * 4000, dtype: np.Int64).reshape(-1, 4000);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.reduce(UFuncOperation.add, a);
                var c = np.sum(b);

                Assert.AreEqual(12799999920000000, c.item(0));
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }



        [Ignore]
        [TestMethod]
        public void Performance_AddAccumulate_INT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Int64);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.accumulate(UFuncOperation.add, a);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddAccumulate calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddAccumulate2_INT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 4000 * 4000, dtype: np.Int64).reshape(4000, 4000);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.accumulate(UFuncOperation.add, a);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddAccumulate calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce_IDEAL_INT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Int64);
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
        public void Performance_AddReduceAt_INT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Int64).reshape((40, -1));

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
        public void Performance_AddOuter_INT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 1000, dtype: np.Int64);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.outer(UFuncOperation.add, np.Int64, a, a);
                // print(b.shape);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddOuter_NotSameType_INT64()
        {

            int LoopCount = 200;
            var a1 = np.arange(0, 1000, dtype: np.Int64);
            var a2 = np.arange(0, 1000, dtype: np.Int16);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.outer(UFuncOperation.add, np.Int64, a2, a1);
                // print(b.shape);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }
        #endregion

        #region UINT64 tests
        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_UINT64()
        {
            int LoopCount = 200;

            var matrix = np.arange(1600000, dtype: np.UInt64).reshape((40, -1));

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

            Console.WriteLine(string.Format("UINT64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_NotContiguous_UINT64()
        {
            int LoopCount = 200;

            var kk = new bool[Int32.MaxValue / 2];

            var matrix = np.arange(16000000, dtype: np.UInt64).reshape((40, -1));

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

            Console.WriteLine(string.Format("UINT64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_InLine_UINT64()
        {
            int LoopCount = 200;

            // get the numeric ops background threads started
            var x = np.arange(1);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

            sw.Start();

            var matrix = np.arange(1600000).astype(np.UInt64).reshape((40, -1));

            var src = matrix.Array.data.datap as double[];

            for (int i = 0; i < LoopCount; i++)
            {
                Parallel.For(0, src.Length, index => src[index] = src[index] / 3);
                Parallel.For(0, src.Length, index => src[index] = src[index] + i);
            }

            var output = matrix[new Slice(15, 25, 2), new Slice(15, 25, 2)];

            sw.Stop();

            Console.WriteLine(string.Format("UINT64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }


        [Ignore]
        [TestMethod]
        public void Performance_MathOperation_Sin_UINT64()
        {

            var a = np.arange(0, 10000000, dtype: np.UInt64);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            var b = np.sin(a);

            sw.Stop();

            Console.WriteLine(string.Format("np.sin calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce_UINT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.UInt64);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.reduce(UFuncOperation.add, a);
                Assert.AreEqual((UInt64)49999995000000, b.item(0));
                // print(b);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce2_UINT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 4000 * 10 * 4000, dtype: np.UInt64).reshape(-1, 4000);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.reduce(UFuncOperation.add, a);
                var c = np.sum(b);

                Assert.AreEqual((UInt64)12799999920000000, c.item(0));
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }



        [Ignore]
        [TestMethod]
        public void Performance_AddAccumulate_UINT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.UInt64);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.accumulate(UFuncOperation.add, a);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddAccumulate calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddAccumulate2_UINT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 4000 * 4000, dtype: np.UInt64).reshape(4000, 4000);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.accumulate(UFuncOperation.add, a);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddAccumulate calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce_IDEAL_UINT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.UInt64);
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
        public void Performance_AddReduceAt_UINT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.UInt64).reshape((40, -1));

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
        public void Performance_AddOuter_UINT64()
        {

            int LoopCount = 200;
            var a = np.arange(0, 1000, dtype: np.UInt64);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.outer(UFuncOperation.add, np.UInt64, a, a);
                // print(b.shape);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddOuter_NotSameType_UINT64()
        {

            int LoopCount = 200;
            var a1 = np.arange(0, 1000, dtype: np.UInt64);
            var a2 = np.arange(0, 1000, dtype: np.Int16);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.outer(UFuncOperation.add, np.UInt64, a2, a1);
                // print(b.shape);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }
        #endregion

        #region DECIMAL tests
        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_DECIMAL()
        {
            int LoopCount = 200;

            var matrix = np.arange(1600000, dtype: np.Decimal).reshape((40, -1));

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

            Console.WriteLine(string.Format("UINT64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_NotContiguous_DECIMAL()
        {
            int LoopCount = 200;

            var kk = new bool[Int32.MaxValue / 2];

            var matrix = np.arange(16000000, dtype: np.Decimal).reshape((40, -1));

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

            Console.WriteLine(string.Format("UINT64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }

        [Ignore]
        [TestMethod]
        public void Performance_ScalarOperation_InLine_DECIMAL()
        {
            int LoopCount = 200;

            // get the numeric ops background threads started
            var x = np.arange(1);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

            sw.Start();

            var matrix = np.arange(1600000).astype(np.Decimal).reshape((40, -1));

            var src = matrix.Array.data.datap as double[];

            for (int i = 0; i < LoopCount; i++)
            {
                Parallel.For(0, src.Length, index => src[index] = src[index] / 3);
                Parallel.For(0, src.Length, index => src[index] = src[index] + i);
            }

            var output = matrix[new Slice(15, 25, 2), new Slice(15, 25, 2)];

            sw.Stop();

            Console.WriteLine(string.Format("UINT64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");
        }


        [Ignore]
        [TestMethod]
        public void Performance_MathOperation_Sin_DECIMAL()
        {

            var a = np.arange(0, 10000000, dtype: np.Decimal);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            var b = np.sin(a);

            sw.Stop();

            Console.WriteLine(string.Format("np.sin calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce_DECIMAL()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Decimal);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.reduce(UFuncOperation.add, a);
                Assert.AreEqual((UInt64)49999995000000, b.item(0));
                // print(b);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce2_DECIMAL()
        {

            int LoopCount = 200;
            var a = np.arange(0, 4000 * 10 * 4000, dtype: np.Decimal).reshape(-1, 4000);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.reduce(UFuncOperation.add, a);
                var c = np.sum(b);

                Assert.AreEqual((UInt64)12799999920000000, c.item(0));
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }



        [Ignore]
        [TestMethod]
        public void Performance_AddAccumulate_DECIMAL()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Decimal);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.accumulate(UFuncOperation.add, a);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddAccumulate calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddAccumulate2_DECIMAL()
        {

            int LoopCount = 200;
            var a = np.arange(0, 4000 * 4000, dtype: np.Decimal).reshape(4000, 4000);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.accumulate(UFuncOperation.add, a);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddAccumulate calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddReduce_IDEAL_DECIMAL()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Decimal);
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
        public void Performance_AddReduceAt_DECIMAL()
        {

            int LoopCount = 200;
            var a = np.arange(0, 10000000, dtype: np.Decimal).reshape((40, -1));

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
        public void Performance_AddOuter_DECIMAL()
        {

            int LoopCount = 200;
            var a = np.arange(0, 1000, dtype: np.Decimal);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.outer(UFuncOperation.add, np.Decimal, a, a);
                // print(b.shape);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }

        [Ignore]
        [TestMethod]
        public void Performance_AddOuter_NotSameType_DECIMAL()
        {

            int LoopCount = 200;
            var a1 = np.arange(0, 1000, dtype: np.Decimal);
            var a2 = np.arange(0, 1000, dtype: np.Int16);

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.outer(UFuncOperation.add, np.Decimal, a2, a1);
                // print(b.shape);
            }

            sw.Stop();

            Console.WriteLine(string.Format("AddReduce calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine("************\n");

        }
        #endregion

    }
}
