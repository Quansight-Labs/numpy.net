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

            #region Simple random data

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

            #region randbool

            private static ndarray _randbool(Int64 low, UInt64? high, shape newdims)
            {
                bool[] randomData = new bool[CountTotalElements(newdims)];

     
                RandomDistributions.rk_random_bool(false, true, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.Bool).reshape(newdims);
            }

            #endregion

            #region randint8

            private static ndarray _randint8(Int64 low, UInt64? high, shape newdims)
            {
                if (low < System.SByte.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for Int8"));
                if (high != null && high.Value > (UInt64)System.SByte.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for Int8"));

                SByte[] randomData = new SByte[CountTotalElements(newdims)];

                SByte _low = Convert.ToSByte(low);
                SByte? _high = null;

                if (!high.HasValue)
                {
                    _high = (SByte)Math.Max(0, _low - 1);
                    _low = 0;
                }
                else
                {
                    _high = Convert.ToSByte(high.Value);
                }
                var rng = _high.Value - _low;
                var off = _low;

                RandomDistributions.rk_random_int8(off, (SByte)rng, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.Int8).reshape(newdims);
            }

            #endregion

            #region randuint8

            private static ndarray _randuint8(Int64 low, UInt64? high, shape newdims)
            {
                if (low < System.Byte.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for UInt8"));
                if (high != null && high.Value > (UInt64)System.Byte.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for UInt8"));

                Byte[] randomData = new Byte[CountTotalElements(newdims)];

                Byte _low = Convert.ToByte(low);
                Byte? _high = null;

                if (!high.HasValue)
                {
                    _high = (Byte)Math.Max(0, _low - 1);
                    _low = 0;
                }
                else
                {
                    _high = Convert.ToByte(high.Value);
                }
                var rng = _high.Value - _low;
                var off = _low;

                RandomDistributions.rk_random_uint8(off, (Byte)rng, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.UInt8).reshape(newdims);
            }

            #endregion

            #region randint16

            private static ndarray _randint16(Int64 low, UInt64? high, shape newdims)
            {
                if (low < System.Int16.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for Int16"));
                if (high != null && high.Value > (UInt64)System.Int16.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for Int16"));

                Int16[] randomData = new Int16[CountTotalElements(newdims)];

                Int16 _low = Convert.ToInt16(low);
                Int16? _high = null;

                if (!high.HasValue)
                {
                    _high = (Int16)Math.Max(0, _low - 1);
                    _low = 0;
                }
                else
                {
                    _high = Convert.ToInt16(high.Value);
                }
                var rng = _high.Value - _low;
                var off = _low;

                RandomDistributions.rk_random_int16(off, (Int16)rng, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.Int16).reshape(newdims);
            }

            #endregion

            #region randuint16

            private static ndarray _randuint16(Int64 low, UInt64? high, shape newdims)
            {
                if (low < System.UInt16.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for UInt16"));
                if (high != null && high.Value > System.UInt16.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for UInt16"));

                UInt16[] randomData = new UInt16[CountTotalElements(newdims)];

                UInt16 _low = Convert.ToUInt16(low);
                UInt16? _high = null;

                if (!high.HasValue)
                {
                    _high = (UInt16)Math.Max(0, _low - 1);
                    _low = 0;
                }
                else
                {
                    _high = Convert.ToUInt16(high.Value);
                }
                var rng = _high.Value - _low;
                var off = _low;

                RandomDistributions.rk_random_uint16(off, (UInt16)rng, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.UInt16).reshape(newdims);
            }

            #endregion

            #region randint

            public static ndarray randint(Int64 low, UInt64? high = null, shape newshape = null, dtype dtype = null)
            {
                if (dtype == null)
                    dtype = np.Int32;

                switch (dtype.TypeNum)
                {
                    case NPY_TYPES.NPY_BOOL:
                        return _randbool(low, high, newshape);
                    case NPY_TYPES.NPY_BYTE:
                        return _randint8(low, high, newshape);
                    case NPY_TYPES.NPY_UBYTE:
                        return _randuint8(low, high, newshape);
                    case NPY_TYPES.NPY_INT16:
                        return _randint16(low, high, newshape);
                    case NPY_TYPES.NPY_UINT16:
                        return _randuint16(low, high, newshape);
                    case NPY_TYPES.NPY_INT32:
                        return _randint32(low, high, newshape, dtype);
                    case NPY_TYPES.NPY_UINT32:
                        return _randuint32(low, high, newshape);
                    case NPY_TYPES.NPY_INT64:
                        return _randint64(low, high, newshape);
                    case NPY_TYPES.NPY_UINT64:
                        return _randuint64(low, high, newshape);
                    default:
                        throw new TypeError(string.Format("Unsupported dtype {0} for randint", dtype.TypeNum.ToString()));
                }

            }

            private static ndarray _randint32(Int64 low, UInt64? high, shape newdims, dtype dtype = null)
            {
                if (low < System.Int32.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for Int32"));
                if (high > System.Int32.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for Int32"));

                Int32[] randomData = new Int32[CountTotalElements(newdims)];

                Int32 _low = Convert.ToInt32(low);
                Int32? _high = null;

                if (!high.HasValue)
                {
                    _high = Math.Max(0, _low - 1);
                    _low = 0;
                }
                else
                {
                    _high = Convert.ToInt32(high.Value);
                }
                var rng = _high.Value - _low;
                var off = _low;
                RandomDistributions.rk_random_int32(off, rng, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.Int32).reshape(newdims);
            }

            #endregion

            #region randuint
  
            private static ndarray _randuint32(Int64 low, UInt64? high, shape newdims)
            {
                if (low < System.UInt32.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for UInt32"));
                if (high > System.UInt32.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for UInt32"));

                UInt32[] randomData = new UInt32[CountTotalElements(newdims)];

                UInt32 _low = Convert.ToUInt32(low);
                UInt32? _high = null;

                if (!high.HasValue)
                {
                    _high = Math.Max(0, _low - 1);
                    _low = 0;
                }
                else
                {
                    _high = Convert.ToUInt32(high.Value);
                }
                var rng = _high.Value - _low;
                var off = _low;

                RandomDistributions.rk_random_uint32(off, rng, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.UInt32).reshape(newdims);
            }

            #endregion

            #region randint64

            private static ndarray _randint64(Int64 low, UInt64? high, shape newdims)
            {
                if (low < System.Int64.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for Int64"));
                if (high > System.Int64.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for Int64"));

                Int64[] randomData = new Int64[CountTotalElements(newdims)];

                Int64 _low = Convert.ToInt64(low);
                Int64? _high = null;

                if (!high.HasValue)
                {
                    _high = Math.Max(0, _low - 1);
                    _low = 0;
                }
                else
                {
                    _high = Convert.ToInt64(high.Value);
                }
                var rng = _high.Value - _low;
                var off = _low;

                RandomDistributions.rk_random_int64(off, rng, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.Int64).reshape(newdims);
            }


            #endregion

            #region randuint64
   
            private static ndarray _randuint64(Int64 low, UInt64? high, shape newdims)
            {
                if (low < 0)
                    throw new ValueError(string.Format("low is out of bounds for UInt64"));
                if (high > System.UInt64.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for UInt64"));


                UInt64[] randomData = new UInt64[CountTotalElements(newdims)];

                UInt64 _low = Convert.ToUInt64(low);
                UInt64? _high = null;

                if (!high.HasValue)
                {
                    _high = Math.Max(0, _low - 1);
                    _low = 0;
                }
                else
                {
                    _high = Convert.ToUInt64(high.Value);
                }
                var rng = _high.Value - _low;
                var off = _low;

                RandomDistributions.rk_random_uint64(off, rng, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.UInt64).reshape(newdims);
            }


            #endregion

            #region random_integers
            public static ndarray random_integers(Int64 low, UInt64? high = null, shape newshape = null, dtype dtype = null)
            {
                return randint(low, high + 1, newshape, dtype: np.Int32);
            }

            #endregion

            #region random_sample
            public static ndarray random_sample(Int32 size)
            {
                ndarray rndArray = cont0_array(internal_state, RandomDistributions.rk_double, size);
                return rndArray;
            }
            public static ndarray random_(Int32 size)
            {
                return random_sample(size);
            }
            public static ndarray ranf(Int32 size)
            {
                return random_sample(size);
            }
            public static ndarray sample(Int32 size)
            {
                return random_sample(size);
            }
            #endregion

            #region choice

            public static ndarray choice(object a, Int32 size, bool replace = true, object p = null)
            {
                throw new NotImplementedException("This function is not implemented in NumpyDotNet");
            }

            #endregion

            #region bytes
            public static byte getbyte()
            {
                var b = bytes(1);
                return b[0];
            }


            public static byte[] bytes(Int32 size)
            {
                byte[] b = new byte[size];

                RandomDistributions.rk_fill(b, size, internal_state);
                return b;
            }


            #endregion
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

            #region shuffle/permutation

            // todo:

            #endregion

            #region beta

            public static ndarray beta(ndarray a, ndarray b, shape newshape)
            {
                throw new NotImplementedException("this function is not ready yet");

                ndarray ba = np.any(np.less_equal(a, 0));
                if ((bool)ba.GetItem(0))
                    throw new ValueError("a <= 0");
                ndarray bb = np.any(np.less_equal(b, 0));
                if ((bool)bb.GetItem(0))
                    throw new ValueError("b <= 0");

                return cont2_array(internal_state, RandomDistributions.rk_beta, CountTotalElements(newshape), a, b);

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


            private static npy_intp CountTotalElements(shape _shape)
            {
                npy_intp TotalElements = 1;
                for (int i = 0; i < _shape.iDims.Length; i++)
                {
                    TotalElements *= _shape.iDims[i];
                }

                return TotalElements;
            }

            #region Python Version

            internal delegate double rk_cont0(rk_state state);
            internal delegate double rk_cont2(rk_state state, double a, double b);

            private static ndarray cont0_array(rk_state state, rk_cont0 func, npy_intp size)
            {
                double[] array_data;
                ndarray array;
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

            private static ndarray cont2_array(rk_state state, rk_cont2 func, npy_intp size, ndarray oa, ndarray ob)
            {
                broadcast multi;
                ndarray array;
                double[] array_data;

                if (size == 0)
                {
                    multi = np.broadcast(oa, ob);
                    array = np.empty(multi.shape, dtype: np.Float64);
                }
                else
                {
                    array = np.empty(size, dtype: np.Float64);
                    multi = np.broadcast(oa, ob, array);
                    if (multi.shape != array.shape)
                    {

                    }
                }

                array_data = array.AsDoubleArray();

                double[] oa_data = multi._core.core.iters[0].ao.data.datap as double[];
                double[] ob_data = multi._core.core.iters[1].ao.data.datap as double[];

                int index = 0;
                foreach (var x in multi)
                {
                    VoidPtr vpoa = multi._core.core.iters[0].ao.data;
                    VoidPtr vpob = multi._core.core.iters[1].ao.data;

                    array_data[index++] = func(state, oa_data[vpoa.data_offset/sizeof(double)], ob_data[vpob.data_offset / sizeof(double)]);

                }

                return np.array(array_data);

            }

            #endregion
        }
    }
}
