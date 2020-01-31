using NumpyDotNet;
using NumpyLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Int64VsBigInteger
{
    class Program
    {
        static void Main(string[] args)
        {
            int LoopCount = 10000;

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

            sw.Start();
            var matrix = np.arange(0, 1600, dtype: np.Int64).reshape(new shape(40, -1));
            matrix.Name = "SampleApp_Int64";

            for (int i = 0; i < LoopCount; i++)
            {
                matrix = matrix / 3;
                matrix = matrix * 3;
            }

            var output = matrix.A(new Slice(15, 25, 2), new Slice(15, 25, 2));
            output.Name = "SampleApp_Int64(View)";

            sw.Stop();

            Console.WriteLine(string.Format("Int64 calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output);
            Console.WriteLine("************\n");

            sw.Reset();
            sw.Start();

            matrix = np.arange(0, 1600, dtype: np.BigInt).reshape(new shape(40, -1));
            matrix.Name = "SampleApp_BitInteger";

            for (int i = 0; i < LoopCount; i++)
            {
                matrix = matrix / 3;
                matrix = matrix * 3;
            }

            output = matrix.A(new Slice(15, 25, 2), new Slice(15, 25, 2));
            output.Name = "SampleApp_BitInteger(View)";

            sw.Stop();

            Console.WriteLine(string.Format("BigInteger calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));

            Console.WriteLine(output);



            Console.ReadLine();
        }
    }
}
