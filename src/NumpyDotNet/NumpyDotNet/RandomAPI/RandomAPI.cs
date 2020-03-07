/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018-2019
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

using NumpyDotNet.RandomAPI;
using NumpyLib;
using System;
using System.Collections.Generic;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif



namespace NumpyDotNet
{
    public static partial class np
    {
        public static class random
        {
            static Random r = new Random();
            static rk_state internal_state = new rk_state(new RandomGeneratorPython());
            static object rk_lock = new object();
            static bool IsInitialized = seed(null);

            #region seed
            public static bool seed(UInt64? seed)
            {
                _seed(seed);
                return true;
            }
            public static bool seed(Int32? seed)
            {
                if (seed.HasValue)
                    return _seed(Convert.ToUInt64(seed.Value));
                else
                    return _seed(null);

            }
            private static bool _seed(UInt64? seed)
            {
                internal_state.rndGenerator.Seed(seed, internal_state);
                return true;
            }
            #endregion

            #region rand
            public static float rand()
            {
                return random_sample();
            }

            public static ndarray rand(params Int32[] newshape)
            {
                return random_sample(newshape);
            }

            public static ndarray rand(params Int64[] newshape)
            {
                return random_sample(newshape);
            }
   
            #endregion

            #region randn
            public static float randn()
            {
                return Convert.ToSingle(standard_normal());
            }

            public static ndarray randn(params Int32[] newshape)
            {
                return standard_normal(newshape);
            }

            public static ndarray randn(params Int64[] newshape)
            {
                return standard_normal(newshape);
            }

            #endregion

            #region randint

            public static ndarray randint(int low, int? high = null, params Int32[] newshape)
            {
                return _randint(low, high, ConvertToShape(newshape));
            }

            public static ndarray randint(int low, int? high = null, params Int64[] newshape)
            {
                return _randint(low, high, ConvertToShape(newshape));
            }

            private static ndarray _randint(int low, int? high, npy_intp[] newdims)
            {
                Int32[] randomData = new Int32[CountTotalElements(newdims)];
                FillWithRandnInt32(low, high, randomData);

                return np.array(randomData, dtype: np.Int32).reshape(newdims);
            }

            private static void FillWithRandnInt32(int low, int? high, Int32[] randomData)
            {
                int _low = low;
                int _high = low;
                if (!high.HasValue)
                {
                    _high = Math.Max(0, low - 1);
                    _low = 0;
                }
                else
                {
                    _high = high.Value - 1;
                }

                lock (r)
                {
                    for (int i = 0; i < randomData.Length; i++)
                    {
                        randomData[i] = r.Next(_low, _high);
                    }
                }

            }
            #endregion

            #region randint64

            public static ndarray randint64(params Int32[] newshape)
            {
                return _randint64(ConvertToShape(newshape));
            }

            public static ndarray randint64(params Int64[] newshape)
            {
                return _randint64(ConvertToShape(newshape));
            }

            private static ndarray _randint64(npy_intp[] newdims)
            {
                Int64[] randomData = new Int64[CountTotalElements(newdims)];
                FillWithRandnInt64(randomData);

                return np.array(randomData, dtype: np.Int64).reshape(newdims);
            }

            private static void FillWithRandnInt64(Int64[] randomData)
            {
    
                lock (r)
                {
                    for (int i = 0; i < randomData.Length; i++)
                    {
                        Int64 HighWord = r.Next();
                        Int64 LowWord = r.Next();
                        randomData[i] = HighWord << 32 | LowWord;
                    }
                }

            }
            #endregion

            #region randd
            public static double randd()
            {
                lock (r)
                {
                    return r.NextDouble() * r.Next();
                }
            }

            public static ndarray randd(params Int32[] newshape)
            {
                return _randd(ConvertToShape(newshape));
            }

            public static ndarray randd(params Int64[] newshape)
            {
                return _randd(ConvertToShape(newshape));
            }

            private static ndarray _randd(npy_intp[] newdims)
            {
                double[] randomData = new double[CountTotalElements(newdims)];
                FillWithRandd(randomData);

                return np.array(randomData, dtype: np.Float64).reshape(newdims);
            }

            private static void FillWithRandd(double[] randomData)
            {
                lock (r)
                {
                    for (int i = 0; i < randomData.Length; i++)
                    {
                        randomData[i] = r.NextDouble() * r.Next();
                    }
                }

            }
            #endregion

            #region bytes
            public static byte bytes()
            {
                lock (r)
                {
                    byte[] b = new byte[1];
                    r.NextBytes(b);
                    return b[0];
                }
            }

            public static ndarray bytes(params Int32[] newshape)
            {
                return _bytes(ConvertToShape(newshape));
            }

            public static ndarray bytes(params Int64[] newshape)
            {
                return _bytes(ConvertToShape(newshape));
            }

            private static ndarray _bytes(npy_intp[] newdims)
            {
                byte[] randomData = new byte[CountTotalElements(newdims)];
                FillWithRandbytes(randomData);

                return np.array(randomData, dtype: np.UInt8).reshape(newdims);
            }

            private static void FillWithRandbytes(byte[] randomData)
            {
                lock (r)
                {
                    r.NextBytes(randomData);
                }

            }
            #endregion

            #region uniform

            public static ndarray uniform(int low = 0, int high = 1, params Int32[] newshape)
            {
                return _uniform(low, high, ConvertToShape(newshape));
            }

            public static ndarray uniform(int low = 0, int high = 1, params Int64[] newshape)
            {
                return _uniform(low, high, ConvertToShape(newshape));
            }

            private static ndarray _uniform(int low, int high, npy_intp[] newdims)
            {
                double[] randomData = new double[CountTotalElements(newdims)];
                FillWithUniform(low, high, randomData);

                return np.array(randomData, dtype: np.Float64).reshape(newdims);
            }


            private static void FillWithUniform(int low, int high, double[] randomData)
            {
                int PRECISION = 100000000;

                low *= PRECISION;
                high *= PRECISION;

                lock (r)
                {
                    for (int i = 0; i < randomData.Length; i++)
                    {
                        double random_value = r.Next(low, high + 1);

                        randomData[i] = random_value / PRECISION;
                    }
                }

            }

            #endregion

            #region standard_normal

            public static float standard_normal()
            {
                ndarray rndArray = cont0_array(internal_state, RandomDistributions.rk_gauss, 0);
                return Convert.ToSingle(rndArray.GetItem(0));
            }
            public static ndarray standard_normal(params Int32[] newshape)
            {
                return _standard_normal(ConvertToShape(newshape));
            }
            public static ndarray standard_normal(params Int64[] newshape)
            {
                return _standard_normal(ConvertToShape(newshape));
            }
            private static ndarray _standard_normal(params npy_intp[] newshape)
            {
                npy_intp size = CountTotalElements(ConvertToShape(newshape));
                ndarray rndArray = cont0_array(internal_state, RandomDistributions.rk_gauss, size);
                return rndArray.reshape(ConvertToShape(newshape));
            }

            #endregion

            #region random_sample

            public static float random_sample()
            {
                ndarray rndArray = cont0_array(internal_state, RandomDistributions.rk_double, 0);
                return Convert.ToSingle(rndArray.GetItem(0));
            }

            public static ndarray random_sample(params Int32[] newshape)
            {
                return _random_sample(ConvertToShape(newshape));
            }

            public static ndarray random_sample(params Int64[] newshape)
            {
                return _random_sample(ConvertToShape(newshape));
            }
            private static ndarray _random_sample(params npy_intp[] newshape)
            {
                npy_intp size = CountTotalElements(ConvertToShape(newshape));
                ndarray rndArray = cont0_array(internal_state, RandomDistributions.rk_double, size);
                return rndArray.reshape(ConvertToShape(newshape));
            }

            #endregion

            private static npy_intp[] ConvertToShape(Int32[] newshape)
            {
                npy_intp[] newdims = new npy_intp[newshape.Length];
                for (int i = 0; i < newshape.Length; i++)
                {
                    newdims[i] = newshape[i];
                }

                return newdims;
            }

            private static npy_intp[] ConvertToShape(Int64[] newshape)
            {
                npy_intp[] newdims = new npy_intp[newshape.Length];
                for (int i = 0; i < newshape.Length; i++)
                {
                    newdims[i] = newshape[i];
                }

                return newdims;
            }

            private static npy_intp CountTotalElements(npy_intp[] dims)
            {
                npy_intp TotalElements = 1;
                for (int i = 0; i < dims.Length; i++)
                {
                    TotalElements *= dims[i];
                }

                return TotalElements;
            }

            #region Python Version

            internal delegate double rk_cont0(rk_state state);

            private static ndarray cont0_array(rk_state state, rk_cont0 func, npy_intp size)
            {
                double[] array_data;
                ndarray array;
                npy_intp length;
                npy_intp i;

                if (size == 0)
                {
                    lock (rk_lock)
                    {
                        double rv = func(state);
                        return np.array(rv);
                    }
                }
                else
                {
                    array_data = new double[size];
                    lock (rk_lock)
                    {
                        for (i = 0; i < size; i++)
                        {
                            array_data[i] = func(state);
                        }
                    }
                    array = np.array(array_data);
                    return array;

                }

            }

            #endregion
        }
    }
}
