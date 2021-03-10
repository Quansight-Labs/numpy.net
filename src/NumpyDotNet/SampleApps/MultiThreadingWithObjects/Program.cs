using NumpyDotNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MultiThreadingWithObjects
{
    class Program
    {
        private static bool StressTestThreadsRunning = false;

        static void Main(string[] args)
        {
            int iNumThreads = 10;


            StressTestThreadsRunning = true;

            for (int i = 0; i < iNumThreads; i++)
            {
                int remainder = i % 2;
                if (remainder == 0)
                {
                    Task.Run(() => TaskOne(10));
                }
                else if (remainder == 1)
                {
                    Task.Run(() => TaskTwo(12));
                }
            }


            Console.ReadLine();
        }

        static void TaskOne(int LoopCount)
        {
            var ObjectData = new object[] { new ObjectDemoData(1), new ObjectDemoData(2), new ObjectDemoData(3), new ObjectDemoData(4),
                                 new ObjectDemoData(5),new ObjectDemoData(6),new ObjectDemoData(7),new ObjectDemoData(8),
                                 new ObjectDemoData(9),new ObjectDemoData(10),new ObjectDemoData(11),new ObjectDemoData(12),
                                 new ObjectDemoData(13),new ObjectDemoData(14),new ObjectDemoData(15),new ObjectDemoData(16)};


            var matrix = np.array(ObjectData, dtype: np.Object).reshape((16, 1));


            for (int i = 0; i < LoopCount; i++)
            {
                matrix = matrix + 1;
                matrix = matrix + "x";
                matrix = matrix + new ObjectDemoData(3);

                matrix.Name = string.Format("SampleApp_ObjectData_TaskOne ({0})", i);
                Console.WriteLine(matrix);
                Console.WriteLine("************\n");
            }

        }

        static void TaskTwo(int LoopCount)
        {
            var ObjectData = new object[] { new ObjectDemoData(1), new ObjectDemoData(2), new ObjectDemoData(3), new ObjectDemoData(4),
                                 new ObjectDemoData(5),new ObjectDemoData(6),new ObjectDemoData(7),new ObjectDemoData(8),
                                 new ObjectDemoData(9),new ObjectDemoData(10),new ObjectDemoData(11),new ObjectDemoData(12),
                                 new ObjectDemoData(13),new ObjectDemoData(14),new ObjectDemoData(15),new ObjectDemoData(16)};


            var matrix = np.array(ObjectData, dtype: np.Object).reshape((16, 1));


            for (int i = 0; i < LoopCount; i++)
            {
                matrix = matrix + -1;
                matrix = matrix + "A";
                matrix = matrix + new ObjectDemoData(-3);

                matrix.Name = string.Format("SampleApp_ObjectData_TaskTwo ({0})", i);
                Console.WriteLine(matrix);
                Console.WriteLine("************\n");
            }
 
        }

    }
}
