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

            int[] lhs = new int[33177600];
            int[] rhs = new int[33177600];

            for (int i = 0; i < lhs.Length; i++)
            {
                lhs[i] = i;
                rhs[i] = i + 1;
            }

            bool isSimd = Vector.IsHardwareAccelerated;

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            var naiveTimesMs = new List<long>();
            var hwTimesMs = new List<long>();
            var hwTimesMs2 = new List<long>();

            for (int i = 0; i < 10; i++)
            {
                sw.Restart();

                int[] result1 = ArrayAddition(lhs, rhs);

                var ts1 = sw.ElapsedMilliseconds;
                naiveTimesMs.Add(ts1);

                sw.Restart();

                int[] result2 = SIMDArrayAddition(lhs, rhs);

                var ts2 = sw.ElapsedMilliseconds;
                hwTimesMs.Add(ts2);

      
                Console.WriteLine("{0} : {1} : {2}", ts1, ts2, isSimd);
            }

            Console.WriteLine("Int array addition:");
            Console.WriteLine($"Naive method average time:          {naiveTimesMs.Average():.##}");
            Console.WriteLine($"HW accelerated method average time: {hwTimesMs.Average():.##}");
            Console.WriteLine($"HW2 accelerated method average time: {hwTimesMs2.Average():.##}");
            Console.WriteLine($"Hardware speedup:                   {naiveTimesMs.Average() / hwTimesMs.Average():P}%");

            Console.ReadLine();


        }

        public static int[] ArrayAddition(int[] lhs, int[] rhs)
        {
            var result = new int[lhs.Length];
    
            for (int i = 0; i < lhs.Length; ++i)
            {
                result[i] = lhs[i] + rhs[i];
            }

            return result;
        }

        public static int[] SIMDArrayAddition(int[] lhs, int[] rhs)
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



    }
}
