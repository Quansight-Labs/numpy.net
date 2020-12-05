using NumpyDotNet;
using NumpyLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {

            SIMDTestInt();
            SIMDTestInt64();
            SIMDTestShort();
            SIMDTestFloat();
            SIMDTestDouble();

            Console.ReadLine();

        }

        #region Int tests
        private static void SIMDTestInt()
        {
            var lhs = new int[33177600];
            var rhs = new int[33177600];

            for (int i = 0; i < lhs.Length; i++)
            {
                lhs[i] = i+100;
                rhs[i] = i + 1;
            }

            bool isSimd = Vector.IsHardwareAccelerated;

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            var naiveTimesMs = new List<long>();
            var fixedTimesMs = new List<long>();
            var hwTimesMs = new List<long>();

            for (int i = 0; i < 10; i++)
            {
                sw.Restart();

                //var result1 = ArrayAddition(lhs, rhs);
                //var result1 = ArrayDivide(lhs, rhs);
                ArrayCopy(lhs, rhs);

                var ts1 = sw.ElapsedMilliseconds;
                naiveTimesMs.Add(ts1);

                /////
                sw.Restart();

                //var resultFixed = ArrayAdditionFixed(lhs, rhs);

                var tsFixed = sw.ElapsedMilliseconds;
                fixedTimesMs.Add(tsFixed);

                /////
                sw.Restart();

                //var result2 = SIMDArrayAddition(lhs, rhs);
                //var result2 = SIMDArrayDivide(lhs, rhs);
                SIMDArrayCopy(lhs, rhs);

                var ts2 = sw.ElapsedMilliseconds;
                hwTimesMs.Add(ts2);


                Console.WriteLine("{0} : {1} : {2}", ts1, ts2, tsFixed);
            }

            Console.WriteLine("Int32 array addition:");
            Console.WriteLine($"Naive method average time:          {naiveTimesMs.Average():.##}");
            Console.WriteLine($"Fixed method average time:          {fixedTimesMs.Average():.##}");
            Console.WriteLine($"HW accelerated method average time: {hwTimesMs.Average():.##}");
            Console.WriteLine($"Hardware speedup:                   {naiveTimesMs.Average() / hwTimesMs.Average():P}%");
        }

        private static int[] ArrayAddition(int[] lhs, int[] rhs)
        {
            var result = new int[lhs.Length];
    
            for (int i = 0; i < lhs.Length; ++i)
            {
                result[i] = lhs[i] + rhs[i];
            }

            return result;
        }

        private static int[] ArrayDivide(int[] lhs, int[] rhs)
        {
            var result = new int[lhs.Length];

            for (int i = 0; i < lhs.Length; ++i)
            {
                result[i] = lhs[i] / rhs[i];
            }

            return result;
        }

        private static unsafe int[] ArrayAdditionFixed(int[] lhs, int[] rhs)
        {
            var result = new int[lhs.Length];

            fixed (int* plhs = lhs  )
            {
                fixed (int* prhs = rhs)
                {
                    fixed (int* presult = result)
                    {
                        for (int i = 0; i < lhs.Length; i += 1)
                        {
                            presult[i] = plhs[i] + prhs[i];
                            //presult[i+1] = plhs[i+1] + prhs[i+1];
                            //presult[i + 2] = plhs[i + 2] + prhs[i + 2];
                            //presult[i + 3] = plhs[i + 3] + prhs[i + 3];
                        }
                    }
                }
            }
     

            return result;
        }
   

        private static int[] SIMDArrayAddition(int[] lhs, int[] rhs)
        {
            var simdLength = Vector<int>.Count;
            var result = new int[lhs.Length];
            var i = 0;
            for (i = 0; i <= lhs.Length - simdLength; i += simdLength)
            {
                var va = new Vector<int>(lhs, i);
                var vb = new Vector<int>(rhs, i);
                (va + vb).CopyTo(result, i);
            }

            for (; i < lhs.Length; ++i)
            {
                result[i] = lhs[i] + rhs[i];
            }

            return result;
        }

        private static int[] SIMDArrayDivide(int[] lhs, int[] rhs)
        {
            var simdLength = Vector<int>.Count;
            var result = new int[lhs.Length];
            var i = 0;
            for (i = 0; i <= lhs.Length - simdLength; i += simdLength)
            {
                var va = new Vector<int>(lhs, i);
                var vb = new Vector<int>(rhs, i);
                (va / vb).CopyTo(result, i);
            }

            for (; i < lhs.Length; ++i)
            {
                result[i] = lhs[i] / rhs[i];
            }

            return result;
        }

        private static void ArrayCopy(int[] lhs, int[] rhs)
        {
            Array.Copy(lhs, rhs, lhs.Length);
        }

        private static void SIMDArrayCopy(int[] lhs, int[] rhs)
        {
            var simdLength = Vector<int>.Count;
            var i = 0;
            for (i = 0; i <= lhs.Length - simdLength; i += simdLength)
            {
                var va = new Vector<int>(lhs, i);
                va.CopyTo(rhs, i);
            }

            for (; i < lhs.Length; ++i)
            {
                rhs[i] = lhs[i];
            }

        }

        #endregion

        #region Int64 tests
        private static void SIMDTestInt64()
        {
            var lhs = new Int64[33177600];
            var rhs = new Int64[33177600];

            for (Int64 i = 0; i < lhs.Length; i++)
            {
                lhs[i] = i;
                rhs[i] = i + 1;
            }

            bool isSimd = Vector.IsHardwareAccelerated;

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            var naiveTimesMs = new List<long>();
            var hwTimesMs = new List<long>();

            for (int i = 0; i < 10; i++)
            {
                sw.Restart();

                var result1 = ArrayAddition(lhs, rhs);

                var ts1 = sw.ElapsedMilliseconds;
                naiveTimesMs.Add(ts1);

                sw.Restart();

                var result2 = SIMDArrayAddition(lhs, rhs);

                var ts2 = sw.ElapsedMilliseconds;
                hwTimesMs.Add(ts2);


                Console.WriteLine("{0} : {1} : {2}", ts1, ts2, isSimd);
            }

            Console.WriteLine("Int64 array addition:");
            Console.WriteLine($"Naive method average time:          {naiveTimesMs.Average():.##}");
            Console.WriteLine($"HW accelerated method average time: {hwTimesMs.Average():.##}");
            Console.WriteLine($"Hardware speedup:                   {naiveTimesMs.Average() / hwTimesMs.Average():P}%");
        }

        private static Int64[] ArrayAddition(Int64[] lhs, Int64[] rhs)
        {
            var result = new Int64[lhs.Length];

            for (int i = 0; i < lhs.Length; ++i)
            {
                result[i] = lhs[i] + rhs[i];
            }

            return result;
        }

        private static Int64[] SIMDArrayAddition(Int64[] lhs, Int64[] rhs)
        {
            var simdLength = Vector<Int64>.Count;
            var result = new Int64[lhs.Length];
            var i = 0;
            for (i = 0; i <= lhs.Length - simdLength; i += simdLength)
            {
                var va = new Vector<Int64>(lhs, i);
                var vb = new Vector<Int64>(rhs, i);
                (va + vb).CopyTo(result, i);
            }

            for (; i < lhs.Length; ++i)
            {
                result[i] = lhs[i] + rhs[i];
            }

            return result;
        }
        #endregion

        #region short tests
        private static void SIMDTestShort()
        {
            var lhs = new short[33177600];
            var rhs = new short[33177600];

            for (int i = 0; i < lhs.Length; i++)
            {
                lhs[i] = (short)i;
                rhs[i] = (short)(i + 1);
            }

            bool isSimd = Vector.IsHardwareAccelerated;

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            var naiveTimesMs = new List<long>();
            var hwTimesMs = new List<long>();

            for (int i = 0; i < 10; i++)
            {
                sw.Restart();

                //var result1 = ArrayAddition(lhs, rhs);
                ArrayCopy(lhs, rhs);

                var ts1 = sw.ElapsedMilliseconds;
                naiveTimesMs.Add(ts1);

                sw.Restart();

                //var result2 = SIMDArrayAddition(lhs, rhs);
                SIMDArrayCopy(lhs, rhs);

                var ts2 = sw.ElapsedMilliseconds;
                hwTimesMs.Add(ts2);


                Console.WriteLine("{0} : {1} : {2}", ts1, ts2, isSimd);
            }

            Console.WriteLine("short array addition:");
            Console.WriteLine($"Naive method average time:          {naiveTimesMs.Average():.##}");
            Console.WriteLine($"HW accelerated method average time: {hwTimesMs.Average():.##}");
            Console.WriteLine($"Hardware speedup:                   {naiveTimesMs.Average() / hwTimesMs.Average():P}%");
        }

        private static short[] ArrayAddition(short[] lhs, short[] rhs)
        {
            var result = new short[lhs.Length];

            for (int i = 0; i < lhs.Length; ++i)
            {
                result[i] = (short)(lhs[i] + rhs[i]);
            }

            return result;
        }

        private static short[] SIMDArrayAddition(short[] lhs, short[] rhs)
        {
            var simdLength = Vector<short>.Count;
            var result = new short[lhs.Length];

            var i = 0;
            for (i = 0; i <= lhs.Length - simdLength; i += simdLength)
            {
                var va = new Vector<short>(lhs, i);
                var vb = new Vector<short>(rhs, i);
                (va + vb).CopyTo(result, i);
            }

            for (; i < lhs.Length; ++i)
            {
                result[i] = (short)(lhs[i] + rhs[i]);
            }

            return result;
        }

        private static void ArrayCopy(short[] lhs, short[] rhs)
        {
            //for (int i = 0; i < rhs.Length; i++)
            //{
            //    rhs[i] = lhs[i];
            //}

            Array.Copy(lhs, rhs, lhs.Length);
        }

        private static void SIMDArrayCopy(short[] lhs, short[] rhs)
        {
            var simdLength = Vector<short>.Count;
            var i = 0;
            for (i = 0; i <= lhs.Length - simdLength; i += simdLength)
            {
                var va = new Vector<short>(lhs, i);
                va.CopyTo(rhs, i);
            }

            for (; i < lhs.Length; ++i)
            {
                rhs[i] = lhs[i];
            }

        }

        #endregion

        #region float tests
        private static void SIMDTestFloat()
        {
            var lhs = new float[33177600];
            var rhs = new float[33177600];

            for (int i = 0; i < lhs.Length; i++)
            {
                lhs[i] = (float)i;
                rhs[i] = (float)(i + 1);
            }

            bool isSimd = Vector.IsHardwareAccelerated;

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            var naiveTimesMs = new List<long>();
            var hwTimesMs = new List<long>();

            for (int i = 0; i < 10; i++)
            {
                sw.Restart();

                var result1 = ArrayAddition(lhs, rhs);

                var ts1 = sw.ElapsedMilliseconds;
                naiveTimesMs.Add(ts1);

                sw.Restart();

                var result2 = SIMDArrayAddition(lhs, rhs);

                var ts2 = sw.ElapsedMilliseconds;
                hwTimesMs.Add(ts2);


                Console.WriteLine("{0} : {1} : {2}", ts1, ts2, isSimd);
            }

            Console.WriteLine("float array addition:");
            Console.WriteLine($"Naive method average time:          {naiveTimesMs.Average():.##}");
            Console.WriteLine($"HW accelerated method average time: {hwTimesMs.Average():.##}");
            Console.WriteLine($"Hardware speedup:                   {naiveTimesMs.Average() / hwTimesMs.Average():P}%");
        }

        private static float[] ArrayAddition(float[] lhs, float[] rhs)
        {
            var result = new float[lhs.Length];

            for (int i = 0; i < lhs.Length; ++i)
            {
                result[i] = (float)(lhs[i] + rhs[i]);
            }

            return result;
        }

        private static float[] SIMDArrayAddition(float[] lhs, float[] rhs)
        {
            var simdLength = Vector<float>.Count;
            var result = new float[lhs.Length];
            var i = 0;
            for (i = 0; i <= lhs.Length - simdLength; i += simdLength)
            {
                var va = new Vector<float>(lhs, i);
                var vb = new Vector<float>(rhs, i);
                (va + vb).CopyTo(result, i);
            }

            for (; i < lhs.Length; ++i)
            {
                result[i] = (float)(lhs[i] + rhs[i]);
            }

            return result;
        }
        #endregion

        #region double tests
        private static void SIMDTestDouble()
        {
            var lhs = new double[33177600];
            var rhs = new double[33177600];

            for (int i = 0; i < lhs.Length; i++)
            {
                lhs[i] = (double)i+100;
                rhs[i] = (double)(i + 1);
            }

            bool isSimd = Vector.IsHardwareAccelerated;

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            var naiveTimesMs = new List<long>();
            var hwTimesMs = new List<long>();

            for (int i = 0; i < 10; i++)
            {
                sw.Restart();

                //var result1 = ArrayAddition(lhs, rhs);
                var result1 = ArrayDivide(lhs, rhs);

                var ts1 = sw.ElapsedMilliseconds;
                naiveTimesMs.Add(ts1);

                sw.Restart();

                //var result2 = SIMDArrayAddition(lhs, rhs);
                var result2 = SIMDArrayDivide(lhs, rhs);

                var ts2 = sw.ElapsedMilliseconds;
                hwTimesMs.Add(ts2);


                Console.WriteLine("{0} : {1} : {2}", ts1, ts2, isSimd);
            }

            Console.WriteLine("double array addition:");
            Console.WriteLine($"Naive method average time:          {naiveTimesMs.Average():.##}");
            Console.WriteLine($"HW accelerated method average time: {hwTimesMs.Average():.##}");
            Console.WriteLine($"Hardware speedup:                   {naiveTimesMs.Average() / hwTimesMs.Average():P}%");
        }

        private static double[] ArrayAddition(double[] lhs, double[] rhs)
        {
            var result = new double[lhs.Length];

            for (int i = 0; i < lhs.Length; ++i)
            {
                result[i] = lhs[i] + rhs[i];
            }

            return result;
        }

        private static double[] ArrayDivide(double[] lhs, double[] rhs)
        {
            var result = new double[lhs.Length];

            for (int i = 0; i < lhs.Length; ++i)
            {
                result[i] = lhs[i] / rhs[i];
            }

            return result;
        }

        private static double[] SIMDArrayAddition(double[] lhs, double[] rhs)
        {
            var simdLength = Vector<double>.Count;
            var result = new double[lhs.Length];
            var i = 0;
            for (i = 0; i <= lhs.Length - simdLength; i += simdLength)
            {
                var va = new Vector<double>(lhs, i);
                var vb = new Vector<double>(rhs, i);
                (va + vb).CopyTo(result, i);
            }

            for (; i < lhs.Length; ++i)
            {
                result[i] = (float)(lhs[i] + rhs[i]);
            }

            return result;
        }

        private static double[] SIMDArrayDivide(double[] lhs, double[] rhs)
        {
            var simdLength = Vector<double>.Count;
            var result = new double[lhs.Length];
            var i = 0;
            for (i = 0; i <= lhs.Length - simdLength; i += simdLength)
            {
                var va = new Vector<double>(lhs, i);
                var vb = new Vector<double>(rhs, i);
                (va / vb).CopyTo(result, i);
            }

            for (; i < lhs.Length; ++i)
            {
                result[i] = lhs[i] / rhs[i];
            }

            return result;
        }

        #endregion

    }
}
