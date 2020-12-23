using NumpyDotNet;
using NumpyLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using npy_intp = System.Int64;


namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {

            //for (int i = 0; i < 10; i++)
            //{
            //    Test2();
            //}

            Test4();



        }

        static void Test5()
        {
            const int TestDataSize = 10000000;
            const int TestLoops = 1000;
            Int64[] Results = new Int64[TestDataSize];
            Int64[] Results2 = new Int64[TestDataSize];

            /////////// 
            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Restart();

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; i++)
                {
                    Results[i] = GetItem1(-i, 1000);
                }
            }

            var ts1 = sw.ElapsedMilliseconds;

            //////////////

            sw.Restart();

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; i++)
                {
                    Results2[i] = GetItem2(-i, 1000);
                }
            }

            var ts2 = sw.ElapsedMilliseconds;

            for (int i = 0; i < TestDataSize; i++)
            {
                if (Results[i] != Results2[i])
                {
                    Console.WriteLine("Not same result");
                }
            }

            Console.WriteLine("{0} : {1}", ts1.ToString(), ts2.ToString());
            Console.ReadLine();



        }
        public static Int64 GetItem1(Int64 AdjustedIndex, Int32 Length)
        {

            if (AdjustedIndex < 0)
            {
                AdjustedIndex = Length - Math.Abs(AdjustedIndex);
            }

            return AdjustedIndex;
        }

        public static Int64 GetItem2(Int64 AdjustedIndex, Int32 Length)
        {

            if (AdjustedIndex < 0)
            {
                AdjustedIndex = Length - -AdjustedIndex;
            }

            return AdjustedIndex;
        }


        static void Test4()
        {
            const int TestDataSize = 10000000;
            const int TestLoops = 1000;

            Int64[] TestData1 = new Int64[TestDataSize];
            Int64[] TestData2 = new Int64[TestDataSize];
            Int64[] Results = new Int64[TestDataSize];
            Int64[] Results2 = new Int64[TestDataSize];
            Int64[] Results3 = new Int64[TestDataSize];

            for (int i = 0; i < TestDataSize; i++)
            {
                TestData1[i] = i + 1000;
                TestData2[i] = i + 1000 + 1;
            }

            int ItemSize = 8;
            int ItemSizeRightShift = 3;

            /////////// 
            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Restart();

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; i++)
                {
                    Results[i] = TestData1[i] / ItemSize;
                }
            }
            
            var ts1 = sw.ElapsedMilliseconds;

            //////////////

            sw.Restart();

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; i++)
                {
                    Results2[i] = TestData1[i] >> ItemSizeRightShift;
                }
            }

            var ts2 = sw.ElapsedMilliseconds;

            //////////////

            sw.Restart();

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; i++)
                {
                    Results3[i] = TestData1[i];
                }
            }

            var ts3 = sw.ElapsedMilliseconds;


            for (int i = 0; i < TestDataSize; i++)
            {
                if (Results[i] != Results2[i])
                {
                    Console.WriteLine("Not same result");
                }
            }

            Console.WriteLine("{0} : {1} : {2}", ts1.ToString(), ts2.ToString(), ts3.ToString());
            Console.ReadLine();

        }

        static void Test3()
        {
            int LoopCount = 500;

            ndarray matrixOrig = np.arange(16000000, dtype: np.Float64).reshape((40, -1));
            ndarray matrix = null;

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            for (int i = 0; i < LoopCount; i++)
            {
                matrix = matrixOrig["1:40:2", "1:-2:3"] as ndarray;

                matrix = matrix / 3;
                //matrix = matrix + i;
            }

            //Assert.AreEqual(7290200441084.1943, np.sum(matrix).GetItem(0));

            var output = matrix[new Slice(15, 25, 2), new Slice(15, 25, 2)];

            sw.Stop();

            Console.WriteLine(string.Format("DOUBLE calculations took {0} milliseconds\n", sw.ElapsedMilliseconds));
            Console.WriteLine(output.ToString());
            Console.WriteLine("************\n");

            Console.ReadLine();
        }

        static void Test1()
        {
            const int TestDataSize = 10000000;
            const int TestLoops = 100;

            double[] TestData1 = new double[TestDataSize];
            double[] TestData2 = new double[TestDataSize];
            double[] Results = new double[TestDataSize];

            for (int i = 0; i < TestDataSize; i++)
            {
                TestData1[i] = i;
                TestData2[i] = i + 1;
            }

            /////////// 
            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Restart();

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; i++)
                {
                    Results[i] = TestData1[i] * TestData2[i];
                }
            }
            var ts1 = sw.ElapsedMilliseconds;

            /////////// 
            sw.Restart();

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize;)
                {
                    Results[i] = TestData1[i] * TestData2[i]; i++;
                    Results[i] = TestData1[i] * TestData2[i]; i++;
                    Results[i] = TestData1[i] * TestData2[i]; i++;
                    Results[i] = TestData1[i] * TestData2[i]; i++;
                    Results[i] = TestData1[i] * TestData2[i]; i++;
                    Results[i] = TestData1[i] * TestData2[i]; i++;
                    Results[i] = TestData1[i] * TestData2[i]; i++;
                    Results[i] = TestData1[i] * TestData2[i]; i++;
                    Results[i] = TestData1[i] * TestData2[i]; i++;
                    Results[i] = TestData1[i] * TestData2[i]; i++;
                }
            }

            var ts2 = sw.ElapsedMilliseconds;
            /////////// 
            
            sw.Restart();

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; i+= 10)
                {
                    Results[i + 0] = TestData1[i + 0] * TestData2[i + 0];
                    Results[i + 1] = TestData1[i + 1] * TestData2[i + 1];
                    Results[i + 2] = TestData1[i + 2] * TestData2[i + 2];
                    Results[i + 3] = TestData1[i + 3] * TestData2[i + 3];
                    Results[i + 4] = TestData1[i + 4] * TestData2[i + 4];
                    Results[i + 5] = TestData1[i + 5] * TestData2[i + 5];
                    Results[i + 6] = TestData1[i + 6] * TestData2[i + 6];
                    Results[i + 7] = TestData1[i + 7] * TestData2[i + 7];
                    Results[i + 8] = TestData1[i + 8] * TestData2[i + 8];
                    Results[i + 9] = TestData1[i + 9] * TestData2[i + 9];
                }
            }

            var ts3 = sw.ElapsedMilliseconds;
            /////////// 

            sw.Restart();

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; i++)
                {
                    Results[i] = AddDouble(TestData1[i],TestData2[i]);
                }
            }
            var ts4 = sw.ElapsedMilliseconds;

            /////////// 

            sw.Restart();

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; )
                {
                    Results[i] = AddDouble(TestData1[i], TestData2[i]); i++;
                    Results[i] = AddDouble(TestData1[i], TestData2[i]); i++;
                    Results[i] = AddDouble(TestData1[i], TestData2[i]); i++;
                    Results[i] = AddDouble(TestData1[i], TestData2[i]); i++;
                    Results[i] = AddDouble(TestData1[i], TestData2[i]); i++;
                    Results[i] = AddDouble(TestData1[i], TestData2[i]); i++;
                    Results[i] = AddDouble(TestData1[i], TestData2[i]); i++;
                    Results[i] = AddDouble(TestData1[i], TestData2[i]); i++;
                    Results[i] = AddDouble(TestData1[i], TestData2[i]); i++;
                    Results[i] = AddDouble(TestData1[i], TestData2[i]); i++;
                }
            }
            var ts5 = sw.ElapsedMilliseconds;

            /////////// 

            sw.Restart();

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; i+= 10)
                {
                    AddDouble(TestData1, TestData2, Results, i);
                }
            }
            var ts6 = sw.ElapsedMilliseconds;

            /////////// 

            sw.Restart();

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; i += 100)
                {
                    AddDouble2(TestData1, TestData2, Results, i);
                }
            }
            var ts7 = sw.ElapsedMilliseconds;

            Console.WriteLine("{0} : {1} : {2} : {3} : {4}: {5} : {6}", ts1.ToString(), ts2.ToString(), ts3.ToString(), ts4.ToString(), ts5.ToString(), ts6.ToString(), ts7.ToString());
        }



        static void Test2()
        {
            const int TestDataSize = 10000000;
            const int TestLoops = 100;

            double[] TestData1 = new double[TestDataSize];
            double[] TestData2 = new double[TestDataSize];
            double[] Results = new double[TestDataSize];

            for (int i = 0; i < TestDataSize; i++)
            {
                TestData1[i] = i;
                TestData2[i] = i + 1;
            }

            long x, y, z, zz, xx;

            /////////// 
            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Restart();

            //{
                NpyArrayIterO IterT1 = new NpyArrayIterO();
                IterT1.nd_m1 = 0;
                NpyArrayIterO IterT2 = new NpyArrayIterO();
                IterT2.nd_m1 = 1;
                IterT2.contiguous = true;
                NpyArrayIterO IterT3 = new NpyArrayIterO();
                IterT3.nd_m1 = 1;
                IterT3.contiguous = false;

                for (int j = 0; j < TestLoops; j++)
                {
                    for (int i = 0; i < TestDataSize; i++)
                    {
                        Results[i] = TestData1[i] * TestData2[i];
                        NpyArray_ITER_NEXT(IterT1);
                        NpyArray_ITER_NEXT(IterT2);
                        NpyArray_ITER_NEXT(IterT3);
                    }
                }

                x = IterT1.data_offset + IterT2.data_offset + IterT3.data_offset;

          //  }

            var ts1 = sw.ElapsedMilliseconds;



            /////////// 

            sw.Restart();

        //    {
                NpyArrayIterS IterS1 = new NpyArrayIterS();
                IterS1.nd_m1 = 0;
                NpyArrayIterS IterS2 = new NpyArrayIterS();
                IterS2.nd_m1 = 1;
                IterS2.contiguous = true;
                NpyArrayIterS IterS3 = new NpyArrayIterS();
                IterS3.nd_m1 = 1;
                IterS3.contiguous = false;

                for (int j = 0; j < TestLoops; j++)
                {
                    for (int i = 0; i < TestDataSize; i++)
                    {
                        Results[i] = TestData1[i] * TestData2[i];
                        NpyArray_ITER_NEXT(ref IterS1);
                        NpyArray_ITER_NEXT(ref IterS2);
                        NpyArray_ITER_NEXT(ref IterS3);
                    }
                }
                y = IterS1.data_offset + IterS2.data_offset + IterS3.data_offset;

        //    }


            var ts2 = sw.ElapsedMilliseconds;



            /////////// 

            sw.Restart();

        //    {
                NpyArrayIterO IterA1 = new NpyArrayIterO();
                IterA1.nd_m1 = 0;
                NpyArrayIterO IterA2 = new NpyArrayIterO();
                IterA2.nd_m1 = 1;
                IterA2.contiguous = true;
                NpyArrayIterO IterA3 = new NpyArrayIterO();
                IterA3.nd_m1 = 1;
                IterA3.contiguous = false;

                NpyArray_ITER_NEXTCB IterCBA1 = NpyArray_ITER_SELECT(IterA1);
                NpyArray_ITER_NEXTCB IterCBA2 = NpyArray_ITER_SELECT(IterA2);
                NpyArray_ITER_NEXTCB IterCBA3 = NpyArray_ITER_SELECT(IterA3);

                for (int j = 0; j < TestLoops; j++)
                {
                    for (int i = 0; i < TestDataSize; i++)
                    {
                        Results[i] = TestData1[i] * TestData2[i];
                        IterCBA1(IterA1);
                        IterCBA2(IterA2);
                        IterCBA3(IterA3);
                    }
                }
                z = IterA1.data_offset + IterA2.data_offset + IterA3.data_offset;

    //        }

            var ts3 = sw.ElapsedMilliseconds;



            /////////// 

            sw.Restart();

      //      {
                NpyArrayIterS IterS1A = new NpyArrayIterS();
                IterS1A.nd_m1 = 0;
                NpyArrayIterS IterS2A = new NpyArrayIterS();
                IterS2A.nd_m1 = 1;
                IterS2A.contiguous = true;
                NpyArrayIterS IterS3A = new NpyArrayIterS();
                IterS3A.nd_m1 = 1;
                IterS3A.contiguous = false;

                NpyArray_ITER_NEXTCB2 IterCBS1A = NpyArray_ITER_SELECT(ref IterS1A);
                NpyArray_ITER_NEXTCB2 IterCBS2A = NpyArray_ITER_SELECT(ref IterS2A);
                NpyArray_ITER_NEXTCB2 IterCBS3A = NpyArray_ITER_SELECT(ref IterS3A);

                for (int j = 0; j < TestLoops; j++)
                {
                    for (int i = 0; i < TestDataSize; i++)
                    {
                        Results[i] = TestData1[i] * TestData2[i];
                        IterCBS1A(ref IterS1A);
                        IterCBS2A(ref IterS2A);
                        IterCBS3A(ref IterS3A);
                    }
                }

                zz = IterS1A.data_offset + IterS2A.data_offset + IterS3A.data_offset;

    //        }



            var ts4 = sw.ElapsedMilliseconds;

            ///////////////
            sw.Restart();

            //{
            NpyArrayIterO IterC1 = new NpyArrayIterO();
            IterC1.nd_m1 = 0;
            NpyArrayIterO IterC2 = new NpyArrayIterO();
            IterC2.nd_m1 = 1;
            IterC2.contiguous = true;
            NpyArrayIterO IterC3 = new NpyArrayIterO();
            IterC3.nd_m1 = 1;
            IterC3.contiguous = false;

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; i++)
                {
                    Results[i] = TestData1[i] * TestData2[i];
                    IterC1.index++;
                    IterC1.data_offset += 2;

                    IterC2.index++;
                    IterC2.data_offset += 2;

                    IterC3.index++;
                    IterC3.data_offset += 2;
                }
            }

            xx = IterC1.data_offset + IterC2.data_offset + IterC3.data_offset;

            //  }

            var ts5 = sw.ElapsedMilliseconds;

            ///////////////
            sw.Restart();

            //{
            NpyArrayIterO IterD1 = new NpyArrayIterO();
            IterD1.nd_m1 = 0;
            NpyArrayIterO IterD2 = new NpyArrayIterO();
            IterD2.nd_m1 = 1;
            IterD2.contiguous = true;
            NpyArrayIterO IterD3 = new NpyArrayIterO();
            IterD3.nd_m1 = 1;
            IterD3.contiguous = false;


            bool IsContiguous1 = true;
            bool IsContiguous2 = true;
            bool IsContiguous3 = true;

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; i++)
                {
                    Results[i] = TestData1[i] * TestData2[i];

                    if (IsContiguous1)
                    {
                        NpyArray_ITER_NEXT1(IterD1);
                    }
                    else
                    {
                        NpyArray_ITER_NEXT(IterD1);
                    }

                    if (IsContiguous2)
                    {
                        NpyArray_ITER_NEXT1(IterD2);
                    }
                    else
                    {
                        NpyArray_ITER_NEXT(IterD2);
                    }

                    if (IsContiguous3)
                    {
                        NpyArray_ITER_NEXT1(IterD3);
                    }
                    else
                    {
                        NpyArray_ITER_NEXT(IterD3);
                    }
        
                }
            }

//            xx = IterD1.data_offset + IterD2.data_offset + IterD3.data_offset;

            //  }

            var ts6 = sw.ElapsedMilliseconds;

            ///////////////
            sw.Restart();

            //{
            NpyArrayIterO IterE1 = new NpyArrayIterO();
            IterE1.nd_m1 = 0;
            NpyArrayIterO IterE2 = new NpyArrayIterO();
            IterE2.nd_m1 = 1;
            IterE2.contiguous = true;
            NpyArrayIterO IterE3 = new NpyArrayIterO();
            IterE3.nd_m1 = 1;
            IterE3.contiguous = false;


            IsContiguous1 = true;
            IsContiguous2 = true;
            IsContiguous3 = true;

            for (int j = 0; j < TestLoops; j++)
            {
                for (int i = 0; i < TestDataSize; i++)
                {
                    Results[i] = TestData1[i] * TestData2[i];

                    if (IsContiguous1)
                    {
                        IterE1.index++;
                        IterE1.data_offset += 1;
                    }
                    else
                    {
                        NpyArray_ITER_NEXT(IterE1);
                    }

                    if (IsContiguous2)
                    {
                        IterE2.index++;
                        IterE2.data_offset += 1;
                    }
                    else
                    {
                        NpyArray_ITER_NEXT(IterE2);
                    }

                    if (IsContiguous3)
                    {
                        IterE3.index++;
                        IterE3.data_offset += 1;
                    }
                    else
                    {
                        NpyArray_ITER_NEXT(IterE3);
                    }

                }
            }

            //            xx = IterD1.data_offset + IterD2.data_offset + IterD3.data_offset;

            //  }

            var ts7 = sw.ElapsedMilliseconds;


            Console.WriteLine("{0} : {1} : {2} : {3} : {4}", x.ToString(), y.ToString(), z.ToString(), zz.ToString(), xx.ToString());

            Console.WriteLine("{0} : {1} : {2} : {3} : {4} : {5} : {6}", ts1.ToString(), ts2.ToString(), ts3.ToString(), ts4.ToString(), ts5.ToString(), ts6.ToString(), ts7.ToString());
        }

        static double AddDouble(double a, double b)
        {
            return a + b;
        }

        static void AddDouble(double[] TestData1, double[] TestData2, double []Results, int i)
        {
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
        }

        static void AddDouble2(double[] TestData1, double[] TestData2, double[] Results, int i)
        {
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;

            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;

            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;

            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;

            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;

            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;

            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;

            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;

            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
            Results[i] = TestData1[i] * TestData2[i]; i++;
        }

        internal class NpyArrayIterO
        {
            public int index = 0;
            public int nd_m1 = 0;
            public bool contiguous = false;
            public npy_intp data_offset = 0;
        }

        internal struct NpyArrayIterS
        {
            public int index;
            public int nd_m1;
            public bool contiguous;
            public npy_intp data_offset;
        }

        internal delegate void NpyArray_ITER_NEXTCB(NpyArrayIterO it);
        internal delegate void NpyArray_ITER_NEXTCB2(ref NpyArrayIterS it);

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_NEXT(NpyArrayIterO it)
        {
            //Debug.Assert(Validate(it));

            it.index++;
            if (it.nd_m1 == 0)
            {
                it.data_offset += 1;
            }
            else if (it.contiguous)
            {
                it.data_offset += 2;
            }
            else if (it.nd_m1 == 1)
            {
                it.data_offset += 3;
            }
            else
            {
                it.data_offset += 4;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NpyArray_ITER_NEXTCB NpyArray_ITER_SELECT(NpyArrayIterO it)
        {
            //Debug.Assert(Validate(it));

            if (it.nd_m1 == 0)
            {
                return NpyArray_ITER_NEXT1;
            }
            else if (it.contiguous)
            {
                return NpyArray_ITER_NEXT2;
            }
            else if (it.nd_m1 == 1)
            {
                return NpyArray_ITER_NEXT3;
            }
            else
            {
                return NpyArray_ITER_NEXT4;
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_NEXT1(NpyArrayIterO it)
        {
            //Debug.Assert(Validate(it));

            it.index++;
            it.data_offset += 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_NEXT2(NpyArrayIterO it)
        {
            //Debug.Assert(Validate(it));

            it.index++;
            it.data_offset += 2;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_NEXT3(NpyArrayIterO it)
        {
            //Debug.Assert(Validate(it));

            it.index++;
            it.data_offset += 3;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_NEXT4(NpyArrayIterO it)
        {
            //Debug.Assert(Validate(it));

            it.index++;
            it.data_offset += 4;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_NEXT(ref NpyArrayIterS it)
        {
            //Debug.Assert(Validate(it));

            it.index++;
            if (it.nd_m1 == 0)
            {
                it.data_offset += 1;
            }
            else if (it.contiguous)
            {
                it.data_offset += 2;
            }
            else if (it.nd_m1 == 1)
            {
                it.data_offset += 3;
            }
            else
            {
                it.data_offset += 4;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NpyArray_ITER_NEXTCB2 NpyArray_ITER_SELECT(ref NpyArrayIterS it)
        {
            //Debug.Assert(Validate(it));

            if (it.nd_m1 == 0)
            {
                return NpyArray_ITER_NEXT1;
            }
            else if (it.contiguous)
            {
                return NpyArray_ITER_NEXT2;
            }
            else if (it.nd_m1 == 1)
            {
                return NpyArray_ITER_NEXT3;
            }
            else
            {
                return NpyArray_ITER_NEXT4;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_NEXT1(ref NpyArrayIterS it)
        {
            //Debug.Assert(Validate(it));

            it.index++;
            it.data_offset += 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_NEXT2(ref NpyArrayIterS it)
        {
            //Debug.Assert(Validate(it));

            it.index++;
            it.data_offset += 2;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_NEXT3(ref NpyArrayIterS it)
        {
            //Debug.Assert(Validate(it));

            it.index++;
            it.data_offset += 3;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void NpyArray_ITER_NEXT4(ref NpyArrayIterS it)
        {
            //Debug.Assert(Validate(it));

            it.index++;
            it.data_offset += 4;
        }

    }
}
