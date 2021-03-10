using NumpyDotNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DoubleVsDecimal
{
    class Program
    {
        static void Main(string[] args)
        {
            int LoopCount = 10000;

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

            sw.Start();
            var matrix = np.arange(0, 1600, dtype: np.Float64).reshape(new shape(40, -1));
            matrix.Name = "SampleApp_Doubles";

            for (int i = 0; i < LoopCount; i++)
            {
                matrix = matrix / 3;
                matrix = matrix * 3;
            }

            var output = matrix.A(new Slice(15, 25, 2), new Slice(15, 25, 2));
            output.Name = "SampleApp_Doubles(View)";

            sw.Stop();

            Console.WriteLine(string.Format("DOUBLE calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output);
            Console.WriteLine("************\n");

            sw.Reset();
            sw.Start();

            matrix = np.arange(0, 1600, dtype: np.Decimal).reshape(new shape(40, -1));
            matrix.Name = "SampleApp_Decimal";

            for (int i = 0; i < LoopCount; i++)
            {
                matrix = matrix / 3m;
                matrix = matrix * 3m;
            }

            output = matrix.A(new Slice(15, 25, 2), new Slice(15, 25, 2));
            output.Name = "SampleApp_Decimal(View)";

            sw.Stop();

            Console.WriteLine(string.Format("DECIMAL calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));

            Console.WriteLine(output);



            Console.ReadLine();

        }
    }
}
