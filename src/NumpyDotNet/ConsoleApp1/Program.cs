using NumpyDotNet;
using NumpyLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            int LoopCount = 100;

            ndarray matrixOrig = np.arange(16000000, dtype: np.Float64).reshape((40, -1));
            ndarray matrix = matrixOrig;

           // System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            //sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                var b = np.ufunc.reduce(UFuncOperation.add, matrixOrig);
            }
 

            //var output = matrix[new Slice(15, 25, 2), new Slice(15, 25, 2)];

            //sw.Stop();

            //Console.WriteLine(string.Format("DOUBLE calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            //Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");

            //Console.ReadLine();

        }
    }
}
