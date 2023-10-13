/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018-2021
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
        public class random_serializable
        {
            public string randomGeneratorSerializationData;

            public int pos;
            public bool has_gauss; /* !=0: gauss contains a gaussian deviate */
            public double gauss;

            ///* The rk_state structure has been extended to store the following
            // * information for the binomial generator. If the input values of n or p
            // * are different than nsave and psave, then the other parameters will be
            // * recomputed. RTK 2005-09-02 */

            public bool has_binomial; /* !=0: following parameters initialized for binomial */
            public double psave;
            public long nsave;
            public double r;
            public double q;
            public double fm;
            public long m;
            public double p1;
            public double xm;
            public double xl;
            public double xr;
            public double c;
            public double laml;
            public double lamr;
            public double p2;
            public double p3;
            public double p4;
        }

        public class random
        {
            IRandomGenerator _rndGenerator;
            private rk_state internal_state = null;

            private object rk_lock = new object();

            private int double_divsize = GetDivSize(sizeof(double));
            private int long_divsize = GetDivSize(sizeof(long));

            private static int GetDivSize(int elsize)
            {
                switch (elsize)
                {
                    //case 1: return 0;
                    //case 2: return 1;
                    //case 4: return 2;
                    case 8: return 3;
                    case 16: return 4;
                    //case 32: return 5;
                    //case 64: return 6;
                    //case 128: return 7;
                    //case 256: return 8;
                    //case 512: return 9;
                    //case 1024: return 10;
                    //case 2048: return 11;
                    //case 4096: return 12;
                }

                throw new Exception("Unexpected elsize in GetDivSize");
            }


            #region seed


            public bool seed(Int32? seed)
            {
                if (seed.HasValue)
                    return this.seed(Convert.ToUInt64(seed.Value));
                else
                    return this.seed((ulong?)null);

            }

            public bool seed(UInt64? seed)
            {
                internal_state.rndGenerator.Seed(seed, internal_state);
                return true;
            }

            public random(IRandomGenerator rndGenerator = null)
            {
                if (rndGenerator == null)
                {
                    rndGenerator = new RandomState();
                }
                internal_state = new rk_state(rndGenerator);
                seed(null);
                _rndGenerator = rndGenerator;
            }

            public random_serializable ToSerialization()
            {
                random_serializable serializationData = new random_serializable();
                serializationData.randomGeneratorSerializationData = _rndGenerator.ToSerialization();

                serializationData.pos = internal_state.pos;
                serializationData.has_gauss = internal_state.has_gauss;
                serializationData.gauss = internal_state.gauss;

                serializationData.has_binomial = internal_state.has_binomial;
                serializationData.psave = internal_state.psave;
                serializationData.nsave = internal_state.nsave;
                serializationData.r = internal_state.r;
                serializationData.q = internal_state.q;
                serializationData.fm = internal_state.fm;
                serializationData.m = internal_state.m;
                serializationData.p1 = internal_state.p1;
                serializationData.xm = internal_state.xm;
                serializationData.xl = internal_state.xl;
                serializationData.xr = internal_state.xr;
                serializationData.c = internal_state.c;
                serializationData.laml = internal_state.laml;
                serializationData.lamr = internal_state.lamr;
                serializationData.p2 = internal_state.p2;
                serializationData.p3 = internal_state.p3;
                serializationData.p4 = internal_state.p4;

                return serializationData;
            }

            public void FromSerialization(random_serializable serializationData)
            {
                internal_state.pos = serializationData.pos;
                internal_state.has_gauss = serializationData.has_gauss;
                internal_state.gauss = serializationData.gauss;

                internal_state.has_binomial = serializationData.has_binomial;
                internal_state.psave = serializationData.psave;
                internal_state.nsave = serializationData.nsave;
                internal_state.r = serializationData.r;
                internal_state.q = serializationData.q;
                internal_state.fm = serializationData.fm;
                internal_state.m = serializationData.m;
                internal_state.p1 = serializationData.p1;
                internal_state.xm = serializationData.xm;
                internal_state.xl = serializationData.xl;
                internal_state.xr = serializationData.xr;
                internal_state.c = serializationData.c;
                internal_state.laml = serializationData.laml;
                internal_state.lamr = serializationData.lamr;
                internal_state.p2 = serializationData.p2;
                internal_state.p3 = serializationData.p3;
                internal_state.p4 = serializationData.p4;

                _rndGenerator.FromSerialization(serializationData.randomGeneratorSerializationData);
            }


            #endregion

            #region Simple random data

            #region rand
            public double rand()
            {
                return random_sample();
            }

            public ndarray rand(shape newshape)
            {
                return random_sample(newshape);
            }

     
            #endregion

            #region randn
            public double randn()
            {
                return standard_normal();
            }

            public ndarray randn(shape newshape)
            {
                return standard_normal(newshape);
            }

            #endregion

            #region randbool

            private ndarray _randbool(Int64 low, UInt64? high, shape newdims)
            {
                newdims = ConvertToSingleElementIfNull(newdims);
                bool[] randomData = new bool[CountTotalElements(newdims)];

     
                RandomDistributions.rk_random_bool(false, true, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.Bool).reshape(newdims);
            }

            #endregion

            #region randint8

            private ndarray _randint8(Int64 low, UInt64? high, shape newdims)
            {
                if (low < System.SByte.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for Int8"));
                if (high != null && high.Value > (UInt64)System.SByte.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for Int8"));

                newdims = ConvertToSingleElementIfNull(newdims);
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

                RandomDistributions.rk_random_int8(off, (SByte)(rng-1), randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.Int8).reshape(newdims);
            }

            #endregion

            #region randuint8

            private ndarray _randuint8(Int64 low, UInt64? high, shape newdims)
            {
                if (low < System.Byte.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for UInt8"));
                if (high != null && high.Value > (UInt64)System.Byte.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for UInt8"));

                newdims = ConvertToSingleElementIfNull(newdims);
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

                RandomDistributions.rk_random_uint8(off, (Byte)(rng-1), randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.UInt8).reshape(newdims);
            }

            #endregion

            #region randint16

            private ndarray _randint16(Int64 low, UInt64? high, shape newdims)
            {
                if (low < System.Int16.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for Int16"));
                if (high != null && high.Value > (UInt64)System.Int16.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for Int16"));

                newdims = ConvertToSingleElementIfNull(newdims);
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

                RandomDistributions.rk_random_int16(off, (Int16)(rng-1), randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.Int16).reshape(newdims);
            }

            #endregion

            #region randuint16

            private ndarray _randuint16(Int64 low, UInt64? high, shape newdims)
            {
                if (low < System.UInt16.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for UInt16"));
                if (high != null && high.Value > System.UInt16.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for UInt16"));

                newdims = ConvertToSingleElementIfNull(newdims);
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

                RandomDistributions.rk_random_uint16(off, (UInt16)(rng-1), randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.UInt16).reshape(newdims);
            }

            #endregion

            #region randint

            public ndarray randint(Int64 low, UInt64? high = null, shape newshape = null, dtype dtype = null)
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

            private ndarray _randint32(Int64 low, UInt64? high, shape newdims, dtype dtype = null)
            {
                if (low < System.Int32.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for Int32"));
                if (high > System.Int32.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for Int32"));

                newdims = ConvertToSingleElementIfNull(newdims);
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
                RandomDistributions.rk_random_int32(off, rng-1, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.Int32).reshape(newdims);
            }

            #endregion

            #region randuint
  
            private ndarray _randuint32(Int64 low, UInt64? high, shape newdims)
            {
                if (low < System.UInt32.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for UInt32"));
                if (high > System.UInt32.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for UInt32"));

                newdims = ConvertToSingleElementIfNull(newdims);
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

                RandomDistributions.rk_random_uint32(off, rng-1, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.UInt32).reshape(newdims);
            }

            #endregion

            #region randint64

            private ndarray _randint64(Int64 low, UInt64? high, shape newdims)
            {
                if (low < System.Int64.MinValue)
                    throw new ValueError(string.Format("low is out of bounds for Int64"));
                if (high > System.Int64.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for Int64"));

                newdims = ConvertToSingleElementIfNull(newdims);
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

                RandomDistributions.rk_random_int64(off, rng-1, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.Int64).reshape(newdims);
            }


            #endregion

            #region randuint64
   
            private ndarray _randuint64(Int64 low, UInt64? high, shape newdims)
            {
                if (low < 0)
                    throw new ValueError(string.Format("low is out of bounds for UInt64"));
                if (high > System.UInt64.MaxValue)
                    throw new ValueError(string.Format("high is out of bounds for UInt64"));


                newdims = ConvertToSingleElementIfNull(newdims);
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

                RandomDistributions.rk_random_uint64(off, rng-1, randomData.Length, randomData, internal_state);

                return np.array(randomData, dtype: np.UInt64).reshape(newdims);
            }


            #endregion

            #region random_integers
            public ndarray random_integers(Int64 low, UInt64? high = null, shape newshape = null, dtype dtype = null)
            {
                return randint(low, high + 1, newshape, dtype: np.Int32);
            }

            #endregion

            #region random_sample

            public double random_sample()
            {
                ndarray rndArray = cont0_array(internal_state, RandomDistributions.rk_double, null);
                return Convert.ToDouble(rndArray.GetItem(0));
            }

            public ndarray random_sample(shape newdims)
            {
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                ndarray rndArray = cont0_array(internal_state, RandomDistributions.rk_double, size);
                return rndArray;
            }

            #endregion

            #region choice

            public ndarray choice(Int64 a, shape size = null, bool replace = true, double [] p = null)
            {
                return choice((object)a, size, replace, p);
            }

            public ndarray choice(object a, shape size = null, bool replace = true, double [] p = null)
            {
                Int64 pop_size = 0;

                // Format and Verify input
                ndarray aa = np.array(a, copy : false);
                if (aa.ndim == 0)
                {
                    if (aa.TypeNum != NPY_TYPES.NPY_INT64)
                    {
                        throw new ValueError("a must be a 1-dimensional or an integer");
                    }

                    pop_size = (Int64)aa;
                    if (pop_size <= 0)
                    {
                        throw new ValueError("a must be greater than 0");
                    }
                }
                else if (aa.ndim != 1)
                {
                    throw new ValueError("a must be 1 dimensional");
                }
                else
                {
                    pop_size = aa.shape[0];
                    if (pop_size == 0)
                    {
                        throw new ValueError("a must be non-empty");
                    }
                }

                ndarray _p = null;
                if (p != null)
                {

                    //double atol = np.sqrt(np.finfo(np.float64).eps)
                    double atol = 0.0001;

                    _p = np.array(p);

                    if (_p.ndim != 1)
                        throw new ValueError("p must be 1 dimensional");
                    if (_p.size != pop_size)
                        throw new ValueError("a and p must have same size");
                    if (np.anyb(_p < 0))
                        throw new ValueError("probabilities are not non-negative");
                    if (Math.Abs(kahan_sum(p) - 1.0) > atol)
                        throw new ValueError("probabilities do not sum to 1");

                }

                Int64 _size = 0;
                shape shape = size;
                if (size != null)
                {
                    _size = (Int64)np.prod(np.asanyarray(size.iDims));
                }
                else
                {
                    _size = 1;
                }

                // Actual sampling
                ndarray idx = null;
                if (replace)
                {
                    if (_p != null)
                    {
                        var cdf = _p.cumsum();
                        cdf /= cdf[-1];
                        var uniform_samples = random_sample(shape);
                        idx = cdf.searchsorted(uniform_samples, side: NPY_SEARCHSIDE.NPY_SEARCHRIGHT);
                        idx = np.array(idx, copy: false);
                    }
                    else
                    {
                        idx = randint(0, (ulong)pop_size, newshape: shape);
                    }
                }
                else
                {
                    if (_size > pop_size)
                    {
                        throw new ValueError("Cannot take a larger sample than population when 'replace=False'");
                    }

                    if (_p != null)
                    {
                        if (Convert.ToInt64(np.count_nonzero(_p > 0).GetItem(0)) < _size)
                        {
                            throw new ValueError("Fewer non-zero entries in p than size");
                        }

                        npy_intp n_uniq = 0;
                        _p = _p.Copy();
                        var found = np.zeros(shape, dtype: np.Int32);
                        var flat_found = found.ravel();

                        while (n_uniq < _size)
                        {
                            var x = rand(new shape(_size - n_uniq));
                            if (n_uniq > 0)
                            {
                                _p[flat_found["0:" + n_uniq.ToString()]] = 0;
                            }
                            var cdf = np.cumsum(_p);
                            cdf /= cdf[-1];

                            var _new = cdf.searchsorted(x, side: NPY_SEARCHSIDE.NPY_SEARCHRIGHT);
                            var unique_retval = np.unique(_new, return_index : true);
                            unique_retval.indices.Sort();
                            _new = np.take(_new, unique_retval.indices);
                            flat_found[n_uniq.ToString() +  ":" + (n_uniq + _new.size).ToString()] = _new;
                            n_uniq += _new.size;
                        }
                        idx = found;
                    }
                    else
                    {
                        ndarray t1 = permutation(pop_size);
                        idx = t1[":" + size.ToString()] as ndarray;
                        if (shape != null)
                        {
                            NpyCoreApi.SetArrayDimsOrStrides(idx, shape.iDims, shape.iDims.Length, true);
                        }
                    }
                }

       
                // Use samples as indices for a if a is array-like
                if (aa.ndim == 0)
                {
                    // In most cases a scalar will have been made an array
                    if (shape == null)
                    {
                        Int32 _idx = (Int32)idx;
                        return np.array(_idx);
                    }
                    else
                    {
                        return idx;
                    }
                }

                if (shape != null && idx.ndim == 0)
                {
                    throw new Exception("don't currently handle this specific value case");

                    //# If size == () then the user requested a 0-d array as opposed to
                    //# a scalar object when size is None. However a[idx] is always a
                    //# scalar and not an array. So this makes sure the result is an
                    //# array, taking into account that np.array(item) may not work
                    //# for object arrays.
                    //res = np.empty((), dtype = a.dtype)
                    //res[()] = a[idx]
                    //return res
                }

                return np.array(aa[idx]);

            }

            private double kahan_sum(double[] darr)
            {
                double c, y, t, sum;
                npy_intp i;

                sum = darr[0];
                c = 0.0;

                int n = darr.Length;
                for (i = 1; i < n; i++)
                {
                    y = darr[i] - c;
                    t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }

                return sum;
            }

            #endregion

            #region bytes
            public byte getbyte()
            {
                var b = bytes(1);
                return b[0];
            }


            public byte[] bytes(Int32 size)
            {
                byte[] b = new byte[size];

                RandomDistributions.rk_fill(b, size, internal_state);
                return b;
            }


            #endregion
            #endregion

            #region shuffle/permutation

            public void shuffle(ndarray x)
            {
                int n = len(x);

                if (x.ndim == 1 && x.size > 0)
                {
                    ndarray buf = np.empty(new shape(1), dtype: x.Dtype);

                    for (ulong i = (ulong)n - 1; i >= 1; i--)
                    {
                        ulong j = RandomDistributions.rk_interval(i, internal_state);
                        buf[0] = x[j];
                        x[j] = x[i];
                        x[i] = buf[0];
                    }
                    return;
                }

                if (x.ndim > 1 && x.size > 0)
                {
                    ndarray buf = np.empty_like(x[0], dtype: x.Dtype);
                    for (ulong i = (ulong)n - 1; i >= 1; i--)
                    {
                        ulong j = RandomDistributions.rk_interval(i, internal_state);
                        buf["..."] = x[j];
                        x[j] = x[i];
                        x[i] = buf;
                    }


                }

                return;
            }

            public ndarray permutation(object x)
            {
                ndarray arr;

                if (x is ndarray)
                {
                    arr = x as ndarray;
                }
                else
                {
                    arr = asanyarray(x);
                }

                if (arr.IsAScalar && arr.IsInteger)
                {
                    arr = np.arange(Convert.ToInt32(arr.GetItem(0)), dtype: arr.Dtype);
                }

                shuffle(arr);
                return arr;
            }

            #endregion

            #region beta

            public ndarray beta(ndarray a, ndarray b, shape newdims = null)
            {
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                ndarray ba = np.any(np.less_equal(a, 0));
                if ((bool)ba.GetItem(0))
                    throw new ValueError("a <= 0");
                ndarray bb = np.any(np.less_equal(b, 0));
                if ((bool)bb.GetItem(0))
                    throw new ValueError("b <= 0");

                return cont2_array(internal_state, RandomDistributions.rk_beta, size, a, b);

            }

            #endregion

            #region binomial

            public ndarray binomial(object n, object p, shape newdims = null)
            {
                ndarray on, op;
                long ln;
                double fp;

                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                on = asanyarray(n).astype(np.Int64);
                op = asanyarray(p).astype(np.Float64);

                if (on.size == 1 && op.size == 1)
                {
                    ln = (long)on.GetItem(0);
                    fp = (double)op.GetItem(0);
                    if (ln < 0)
                        throw new ValueError("n < 0");
                    if (fp < 0)
                        throw new ValueError("p < 0");
                    else if (fp > 1)
                        throw new ValueError("p > 1");
                    else if ((bool)np.isnan(op).GetItem(0))
                        throw new ValueError("p is nan");

                    return discnp_array_sc(internal_state, RandomDistributions.rk_binomial, size, ln, fp);
                }

                if ((bool)np.any(np.less(n, 0).GetItem(0)))
                    throw new ValueError("n < 0");

                if ((bool)np.any(np.less(p, 0)))
                    throw new ValueError("p < 0");

                if ((bool)np.any(np.greater(p, 1)))
                    throw new ValueError("p > 1");

                return discnp_array(internal_state, RandomDistributions.rk_binomial, size, on, op);

            }

            public ndarray negative_binomial(object n, object p, shape newdims = null)
            {
                ndarray on, op;
                long ln;
                double fp;

                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                on = asanyarray(n).astype(np.Int64);
                op = asanyarray(p).astype(np.Float64);

                if (on.size == 1 && op.size == 1)
                {
                    ln = (long)on.GetItem(0);
                    fp = (double)op.GetItem(0);
                    if (ln < 0)
                        throw new ValueError("n < 0");
                    if (fp < 0)
                        throw new ValueError("p < 0");
                    else if (fp > 1)
                        throw new ValueError("p > 1");
                    else if ((bool)np.isnan(op).GetItem(0))
                        throw new ValueError("p is nan");

                    return discdd_array_sc(internal_state, RandomDistributions.rk_negative_binomial, size, ln, fp);
                }

                if ((bool)np.any(np.less(n, 0).GetItem(0)))
                    throw new ValueError("n < 0");

                if ((bool)np.any(np.less(p, 0)))
                    throw new ValueError("p < 0");

                if ((bool)np.any(np.greater(p, 1)))
                    throw new ValueError("p > 1");

                return discdd_array(internal_state, RandomDistributions.rk_negative_binomial, size, on, op);

            }


            #endregion

            #region chisquare

            public ndarray chisquare(object df, shape newdims = null)
            {
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                ndarray odf;
                double fdf;

                odf = asanyarray(df).astype(np.Float64);

                if (odf.size == 1)
                {
                    fdf = (double)odf.GetItem(0);
                    if (fdf <= 0)
                        throw new ValueError("df <= 0");

                    return cont1_array_sc(internal_state, RandomDistributions.rk_chisquare, size, fdf);
                }


                if (np.anyb(np.less_equal(odf, 0.0)))
                {
                    throw new ValueError("df <= 0");
                }
                return cont1_array(internal_state, RandomDistributions.rk_chisquare, size, odf);
            }

            public ndarray noncentral_chisquare(object df, object nonc, shape newdims = null)
            {
                ndarray odf, ononc;
                double fdf, fnonc;
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                odf = asanyarray(df).astype(np.Float64);
                ononc = asanyarray(nonc).astype(np.Float64);

                if (odf.size == 1 && ononc.size == 1)
                {
                    fdf = (double)odf.GetItem(0);
                    fnonc = (double)ononc.GetItem(0);

                    if (fdf <= 0)
                        throw new ValueError("df <= 0");
                    if (fnonc <= 0)
                        throw new ValueError("nonc <= 0");

                    return cont2_array_sc(internal_state, RandomDistributions.rk_noncentral_chisquare, size, fdf, fnonc);
                }

                if (np.anyb(np.less_equal(odf, 0.0)))
                {
                    throw new ValueError("df <= 0");
                }
                if (np.anyb(np.less_equal(ononc, 0.0)))
                {
                    throw new ValueError("nonc <= 0");
                }
                return cont2_array(internal_state, RandomDistributions.rk_noncentral_chisquare, size, odf, ononc);
            }
            #endregion

            #region dirichlet
            public ndarray dirichlet(Int32 []alpha, Int32 size)
            {
                npy_intp k;
                npy_intp totsize;
                ndarray alpha_arr, val_arr;
                double[] alpha_data;
                double[] val_data;
                npy_intp i;
                double acc, invacc;

                k = len(alpha);
                alpha_arr = np.array(alpha).astype(np.Float64);
                if (np.anyb(np.less_equal(alpha_arr, 0)))
                {
                    throw new ValueError("alpha <= 0");
                }
                alpha_data = alpha_arr.Array.data.datap as double[];

                shape shape = new shape(size, k);

                ndarray diric = np.zeros(shape, np.Float64);
                val_arr = diric;
                val_data = val_arr.Array.data.datap as double[];

                i = 0;
                totsize = val_arr.size;

                while (i < totsize)
                {
                    acc = 0.0;
                    for (int j = 0; j < k; j++)
                    {
                        val_data[i + j] = RandomDistributions.rk_standard_gamma(internal_state, alpha_data[j]);
                        acc = acc + val_data[i + j];
                    }
                    invacc = 1 / acc;
                    for (int j = 0; j < k; j++)
                    {
                        val_data[i + j] = val_data[i + j] * invacc;
                    }
                    i = i + k;
                }

                return diric;

            }

            #endregion

            #region exponential

            public ndarray exponential(double scale = 1.0, shape newdims = null)
            {
                return exponential((object)scale, newdims);
            }

            public ndarray exponential(object scale, shape newdims = null)
            {
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                ndarray oscale = asanyarray(scale).astype(np.Float64);

                if (oscale.size == 1)
                {
                    double fscale = (double)oscale.GetItem(0);
                    if ((bool)np.signbit(fscale).GetItem(0))
                    {
                        throw new ValueError("scale < 0");
                    }

                    return cont1_array_sc(internal_state, RandomDistributions.rk_exponential, size, fscale);
                }

                if (np.anyb(np.signbit(oscale)))
                {
                    throw new ValueError("scale < 0");
                }
                return cont1_array(internal_state, RandomDistributions.rk_exponential, size, oscale);

            }
            #endregion

            #region f distribution

            public ndarray f(object dfnum, object dfden, shape newdims = null)
            {
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                ndarray odfnum, odfden;
                double fdfnum, fdfden;

                odfnum = asanyarray(dfnum).astype(np.Float64);
                odfden = asanyarray(dfden).astype(np.Float64);

                if (odfnum.size == 1 && odfden.size == 1)
                {
                    fdfnum = (double)odfnum.GetItem(0);
                    fdfden = (double)odfden.GetItem(0);

                    if (fdfnum <= 0)
                        throw new ValueError("dfnum <= 0");
                    if (fdfden <= 0)
                        throw new ValueError("dfden <= 0");

                    return cont2_array_sc(internal_state, RandomDistributions.rk_f, size, fdfnum, fdfden);
                }

                if (np.anyb(np.less_equal(odfnum, 0.0)))
                {
                    throw new ValueError("dfnum <= 0");
                }
                if (np.anyb(np.less_equal(odfden, 0.0)))
                {
                    throw new ValueError("dfden <= 0");
                }
                return cont2_array(internal_state, RandomDistributions.rk_f, size, odfnum, odfden);
            }

            #endregion

            #region gamma

            public ndarray gamma(object shape, object scale, shape newdims = null)
            {
                ndarray oshape, oscale;
                double fshape, fscale;
                npy_intp []size = null;

                if (newdims != null)
                    size = newdims.iDims;
         

                oshape = asanyarray(shape).astype(np.Float64);
                oscale = asanyarray(scale).astype(np.Float64);

                if (oshape.size == 1 && oscale.size == 1)
                {
                    fshape = (double)oshape.GetItem(0);
                    fscale = (double)oscale.GetItem(0);
                    if ((bool)np.signbit(fshape).GetItem(0))
                    {
                        throw new ValueError("shape < 0");
                    }
                    if ((bool)np.signbit(fscale).GetItem(0))
                    {
                        throw new ValueError("scale < 0");
                    }

                    return cont2_array_sc(internal_state, RandomDistributions.rk_gamma, size, fshape, fscale);
                }
  

                if (np.anyb(np.signbit(oshape)))
                {
                    throw new ValueError("shape < 0");
                }
                if (np.anyb(np.signbit(oscale)))
                {
                    throw new ValueError("scale < 0");
                }

                return cont2_array(internal_state, RandomDistributions.rk_gamma, size, oshape, oscale);
            }

            #endregion

            #region geometric

            public ndarray geometric(object p, shape newdims = null)
            {
                ndarray op;
                double fp;
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                op = asanyarray(p).astype(np.Float64);

                if (op.size == 1)
                {
                    fp = (double)op.GetItem(0);

                    if (fp < 0.0)
                    {
                        throw new ValueError("p < 0.0");
                    }
                    if (fp > 1.0)
                    {
                        throw new ValueError("p > 1.0");
                    }
                    return discd_array_sc(internal_state, RandomDistributions.rk_geometric, size, fp);
                }


                if (np.anyb(np.less(op, 0.0)))
                {
                    throw new ValueError("p < 0.0");
                }

                if (np.anyb(np.greater(op, 1.0)))
                {
                    throw new ValueError("p > 1.0");
                }

                return discd_array(internal_state, RandomDistributions.rk_geometric, size, op);
            }



            #endregion

            #region gumbel

            public ndarray gumbel(object loc, object scale = null, shape newdims = null)
            {
                ndarray oloc, oscale;
                double floc, fscale;
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                if (scale == null)
                    scale = 1.0;

                oloc = asanyarray(loc).astype(np.Float64);
                oscale = asanyarray(scale).astype(np.Float64);

                if (oloc.size == 1 && oscale.size == 1)
                {
                    floc = (double)oloc.GetItem(0);
                    fscale = (double)oscale.GetItem(0);

                     if ((bool)np.signbit(fscale).GetItem(0))
                        throw new ValueError("scale < 0");

                    return cont2_array_sc(internal_state, RandomDistributions.rk_gumbel, size, floc, fscale);
                }

                if (np.anyb(np.signbit(oscale)))
                {
                    throw new ValueError("scale < 0");
                }
                return cont2_array(internal_state, RandomDistributions.rk_gumbel, size, oloc, oscale);
            }

            #endregion

            #region hypergeometric

            public ndarray hypergeometric(object ngood, object nbad, object nsample, shape newdims = null)
            {
                ndarray ongood, onbad, onsample;
                long lngood, lnbad, lnsample;
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                ongood = asanyarray(ngood).astype(np.Int64);
                onbad = asanyarray(nbad).astype(np.Int64);
                onsample = asanyarray(nsample).astype(Int64);

                if (ongood.size ==1 && onbad.size == 1 && onsample.size == 1)
                {
                    lngood = (long)ongood.GetItem(0);
                    lnbad = (long)onbad.GetItem(0);
                    lnsample = (long)onsample.GetItem(0);

                    if (lngood < 0)
                        throw new ValueError("ngood < 0");
                    if (lnbad < 0)
                        throw new ValueError("nbad < 0");
                    if (lnsample < 1)
                        throw new ValueError("nsample < 1");
                    if (lngood + lnbad < lnsample)
                        throw new ValueError("ngood + nbad < nsample");
                    return discnmN_array_sc(internal_state, RandomDistributions.rk_hypergeometric, size, lngood, lnbad, lnsample);
                }


                if (np.anyb(np.less(ongood, 0)))
                    throw new ValueError("ngood < 0");
                if (np.anyb(np.less(onbad, 0)))
                    throw new ValueError("nbad < 0");
                if (np.anyb(np.less(onsample, 1)))
                    throw new ValueError("nsample < 1");
                if (np.anyb(np.less(np.add(ongood, onbad), onsample)))
                    throw new ValueError("ngood + nbad < nsample");
                return discnmN_array(internal_state, RandomDistributions.rk_hypergeometric, size, ongood, onbad, onsample);
            }

            #endregion

            #region laplace

            public ndarray laplace(object loc, object scale = null, shape newdims = null)
            {
                ndarray oloc, oscale;
                double floc, fscale;
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                if (scale == null)
                    scale = 1.0;

                oloc = asanyarray(loc).astype(np.Float64);
                oscale = asanyarray(scale).astype(np.Float64);

                if (oloc.size == 1 && oscale.size == 1)
                {
                    floc = (double)oloc.GetItem(0);
                    fscale = (double)oscale.GetItem(0);

                    if ((bool)np.signbit(fscale).GetItem(0))
                        throw new ValueError("scale < 0");

                    return cont2_array_sc(internal_state, RandomDistributions.rk_laplace, size, floc, fscale);
                }

                if (np.anyb(np.signbit(oscale)))
                {
                    throw new ValueError("scale < 0");
                }
                return cont2_array(internal_state, RandomDistributions.rk_laplace, size, oloc, oscale);
            }

            #endregion

            #region logistic

            public ndarray logistic(object loc, object scale = null, shape newdims = null)
            {
                ndarray oloc, oscale;
                double floc, fscale;
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                if (scale == null)
                    scale = 1.0;

                oloc = asanyarray(loc).astype(np.Float64);
                oscale = asanyarray(scale).astype(np.Float64);

                if (oloc.size == 1 && oscale.size == 1)
                {
                    floc = (double)oloc.GetItem(0);
                    fscale = (double)oscale.GetItem(0);

                    if ((bool)np.signbit(fscale).GetItem(0))
                        throw new ValueError("scale < 0");

                    return cont2_array_sc(internal_state, RandomDistributions.rk_logistic, size, floc, fscale);
                }

                if (np.anyb(np.signbit(oscale)))
                {
                    throw new ValueError("scale < 0");
                }
                return cont2_array(internal_state, RandomDistributions.rk_logistic, size, oloc, oscale);
            }

            #endregion

            #region lognormal

            public ndarray lognormal(object mean, object sigma, shape newdims = null)
            {
                ndarray omean, osigma;
                double fmean, fsigma;
                npy_intp[] size = null;

                if (newdims != null)
                    size = newdims.iDims;


                omean = asanyarray(mean).astype(np.Float64);
                osigma = asanyarray(sigma).astype(np.Float64);

                if (omean.size == 1 && osigma.size == 1)
                {
                    fmean = (double)omean.GetItem(0);
                    fsigma = (double)osigma.GetItem(0);
  
                    if ((bool)np.signbit(fsigma).GetItem(0))
                    {
                        throw new ValueError("sigma < 0");
                    }

                    return cont2_array_sc(internal_state, RandomDistributions.rk_lognormal, size, fmean, fsigma);
                }


                if (np.anyb(np.signbit(osigma)))
                {
                    throw new ValueError("sigma < 0");
                }

                return cont2_array(internal_state, RandomDistributions.rk_lognormal, size, omean, osigma);
            }

            #endregion

            #region logseries

            public ndarray logseries(object p, shape newdims = null)
            {
                ndarray op;
                double fp;
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                op = asanyarray(p).astype(np.Float64);

                if (op.size == 1)
                {
                    fp = (double)op.GetItem(0);

                    if (fp <= 0.0)
                        throw new ValueError("p <= 0.0");
                    if (fp >= 1.0)
                        throw new ValueError("p >= 1.0");

                    return discd_array_sc(internal_state, RandomDistributions.rk_logseries, size, fp);
                }


                if (np.anyb(np.less_equal(op, 0.0)))
                    throw new ValueError("p <= 0.0");
                if (np.anyb(np.greater_equal(op, 1.0)))
                    throw new ValueError("p >= 1.0");

                return discd_array(internal_state, RandomDistributions.rk_logseries, size, op);
            }

            #endregion

            #region multinomial 

            public ndarray multinomial(int n, object pvals, shape size = null)
            {
                ndarray parr = asanyarray(pvals).astype(np.Float64);
                double[] pix = parr.Array.data.datap as double[];
                npy_intp d = pix.Length;

                if (kahan_sum(pix, d - 1) > (1.0 + 1e-12))
                {
                    throw new ValueError("sum(pvals[:-1]) > 1.0");
                }

                shape newshape = _shape_from_size(size, d);

                var multin = np.zeros(newshape, np.Int64);
                ndarray mnarr = multin;
                long[] mnix = mnarr.Array.data.datap as long[];
                npy_intp sz = mnarr.size;

                npy_intp i = 0;
                while (i < sz)
                {
                    double Sum = 1.0;
                    long dn = n;
                    for (npy_intp j = 0; j < d - 1; j++)
                    {
                        mnix[i + j] = RandomDistributions.rk_binomial(internal_state, dn, pix[j] / Sum);
                        dn = dn - mnix[i + j];
                        if (dn <= 0)
                            break;
                        Sum = Sum - pix[j];
                    }

                    if (dn > 0)
                        mnix[i + d - 1] = dn;

                    i = i + d;

                }

                return multin;
            }

            #endregion

            #region noncentral_f
            public ndarray noncentral_f(object dfnum, object dfden, object nonc, shape newdims = null)
            {
                ndarray odfnum, odfden, ononc;
                double fdfnum, fdfden, fnonc;
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                odfnum = asanyarray(dfnum).astype(np.Float64);
                odfden = asanyarray(dfden).astype(np.Float64);
                ononc = asanyarray(nonc).astype(np.Float64);

                if (odfnum.size == 1 && odfden.size == 1 && ononc.size == 1)
                {
                    fdfnum = (double)odfnum.GetItem(0);
                    fdfden = (double)odfden.GetItem(0);
                    fnonc = (double)ononc.GetItem(0);

                    if (fdfnum <= 0)
                        throw new ValueError("dfnum <= 0");
                    if (fdfden <= 0)
                        throw new ValueError("dfden <= 0");
                    if (fnonc <= 0)
                        throw new ValueError("nonc <= 0");

                    return cont3_array_sc(internal_state, RandomDistributions.rk_noncentral_f, size, fdfnum, fdfden, fnonc);
                }

                if (np.anyb(np.less_equal(odfnum, 0.0)))
                {
                    throw new ValueError("dfnum <= 0");
                }
                if (np.anyb(np.less_equal(odfden, 0.0)))
                {
                    throw new ValueError("dfden <= 0");
                }
                if (np.anyb(np.less_equal(ononc, 0.0)))
                {
                    throw new ValueError("nonc <= 0");
                }
                return cont3_array(internal_state, RandomDistributions.rk_noncentral_f, size, odfnum, odfden, ononc);
            }


            #endregion

            #region normal

            public ndarray normal(object loc, object scale, shape newdims = null)
            {
                ndarray oloc, oscale;
                double floc, fscale;
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                oloc = asanyarray(loc).astype(np.Float64);
                oscale = asanyarray(scale).astype(np.Float64);

                if (oloc.size == 1 && oscale.size == 1)
                {
                    floc = (double)oloc.GetItem(0);
                    fscale = (double)oscale.GetItem(0);

                    if ((bool)np.signbit(fscale).GetItem(0))
                        throw new ValueError("scale <= 0");

                    return cont2_array_sc(internal_state, RandomDistributions.rk_normal, size, floc, fscale);
                }

                if (np.anyb(np.signbit(oscale)))
                {
                    throw new ValueError("scale <= 0");
                }
                return cont2_array(internal_state, RandomDistributions.rk_normal, size, oloc, oscale);
            }

            #endregion

            #region pareto

            public ndarray pareto(object a, shape newdims = null)
            {
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                ndarray oa;
                double fa;

                oa = asanyarray(a).astype(np.Float64);

                if (oa.size == 1)
                {
                    fa = (double)oa.GetItem(0);
                    if (fa <= 0)
                        throw new ValueError("a <= 0");

                    return cont1_array_sc(internal_state, RandomDistributions.rk_pareto, size, fa);
                }


                if (np.anyb(np.less_equal(oa, 0.0)))
                {
                    throw new ValueError("a <= 0");
                }
                return cont1_array(internal_state, RandomDistributions.rk_pareto, size, oa);
            }
            #endregion

            #region Poisson

            public ndarray poisson(object lam, shape shape = null)
            {
                ndarray olam;
                double flam;
                npy_intp[] size = null;
                if (shape != null)
                    size = shape.iDims;

                const double poisson_lam_max = 9.223372006484771e+18;

                olam = asanyarray(lam).astype(np.Float64);

                if (olam.size == 1)
                {
                    flam = (double)olam.GetItem(0);

                    if (flam < 0.0)
                    {
                        throw new ValueError("lam < 0.0");
                    }
                    if (flam > poisson_lam_max)
                    {
                        throw new ValueError("lam value too large.");
                    }
                    return discd_array_sc(internal_state, RandomDistributions.rk_poisson, size, flam);
                }


                if (np.anyb(np.less(olam, 0.0)))
                {
                    throw new ValueError("lam < 0.0");
                }

                if (np.anyb(np.greater(olam, poisson_lam_max)))
                {
                    throw new ValueError("lam value too large.");
                }

                return discd_array(internal_state, RandomDistributions.rk_poisson, size, olam);
            }


            #endregion

            #region Power
            public ndarray power(object a, shape newdims = null)
            {
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                ndarray oa;
                double fa;

                oa = asanyarray(a).astype(np.Float64);

                if (oa.size == 1)
                {
                    fa = (double)oa.GetItem(0);
                    if ((bool)np.signbit(fa).GetItem(0))
                        throw new ValueError("a < 0");

                    return cont1_array_sc(internal_state, RandomDistributions.rk_power, size, fa);
                }


                if (np.anyb(np.signbit(oa)))
                {
                    throw new ValueError("a < 0");
                }
                return cont1_array(internal_state, RandomDistributions.rk_power, size, oa);
            }
            #endregion

            #region rayleigh

            public ndarray rayleigh(object scale, shape newdims = null)
            {
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                ndarray oscale;
                double fscale;

                oscale = asanyarray(scale).astype(np.Float64);

                if (oscale.size == 1)
                {
                    fscale = (double)oscale.GetItem(0);
                    if ((bool)np.signbit(fscale).GetItem(0))
                        throw new ValueError("scale < 0");

                    return cont1_array_sc(internal_state, RandomDistributions.rk_rayleigh, size, fscale);
                }


                if (np.anyb(np.signbit(oscale)))
                {
                    throw new ValueError("scale < 0");
                }
                return cont1_array(internal_state, RandomDistributions.rk_rayleigh, size, oscale);
            }
            #endregion

            #region standard_cauchy

            public double standard_cauchy()
            {
                ndarray rndArray = cont0_array(internal_state, RandomDistributions.rk_standard_cauchy, null);
                return Convert.ToDouble(rndArray.GetItem(0));
            }


            public ndarray standard_cauchy(shape newdims)
            {
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                ndarray rndArray = cont0_array(internal_state, RandomDistributions.rk_standard_cauchy, size);
                return rndArray;
            }

            #endregion

            #region standard_exponential

            public double standard_exponential()
            {
                ndarray rndArray = cont0_array(internal_state, RandomDistributions.rk_standard_exponential, null);
                return Convert.ToDouble(rndArray.GetItem(0));
            }


            public ndarray standard_exponential(shape newdims)
            {
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                ndarray rndArray = cont0_array(internal_state, RandomDistributions.rk_standard_exponential, size);
                return rndArray;
            }

            #endregion

            #region standard_gamma

            public ndarray standard_gamma(object shape, shape newdims = null)
            {
                ndarray oshape;
                double fshape;

                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                oshape = asanyarray(shape).astype(np.Float64);

                if (oshape.size == 1)
                {
                    fshape = (double)oshape.GetItem(0);
                    if ((bool)np.signbit(fshape).GetItem(0))
                        throw new ValueError("shape < 0");

                    return cont1_array_sc(internal_state, RandomDistributions.rk_standard_gamma, size, fshape);
                }


                if (np.anyb(np.signbit(oshape)))
                {
                    throw new ValueError("shape < 0");
                }
                return cont1_array(internal_state, RandomDistributions.rk_standard_gamma, size, oshape);
            }

            #endregion

            #region standard_normal

            public double standard_normal()
            {
                ndarray rndArray = cont0_array(internal_state, RandomDistributions.rk_gauss, null);
                return Convert.ToDouble(rndArray.GetItem(0));
            }

            public ndarray standard_normal(shape newdims)
            {
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                ndarray rndArray = cont0_array(internal_state, RandomDistributions.rk_gauss, size);
                return rndArray;
            }

            #endregion

            #region standard_t

            public ndarray standard_t(object df, shape newdims = null)
            {
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                ndarray odf;
                double fdf;

                odf = asanyarray(df).astype(np.Float64);

                if (odf.size == 1)
                {
                    fdf = (double)odf.GetItem(0);
                    if (fdf <= 0)
                        throw new ValueError("df <= 0");

                    return cont1_array_sc(internal_state, RandomDistributions.rk_standard_t, size, fdf);
                }


                if (np.anyb(np.less_equal(odf, 0.0)))
                {
                    throw new ValueError("df <= 0");
                }
                return cont1_array(internal_state, RandomDistributions.rk_standard_t, size, odf);
            }

            #endregion

            #region triangular

            public ndarray triangular(object left, object mode, object right, shape newdims = null)
            {
                ndarray oleft, omode, oright;
                double fleft, fmode, fright;
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                oleft = asanyarray(left).astype(np.Float64);
                omode = asanyarray(mode).astype(np.Float64);
                oright = asanyarray(right).astype(np.Float64);

                if (oleft.size == 1 && omode.size == 1 && oright.size == 1)
                {
                    fleft = (double)oleft.GetItem(0);
                    fmode = (double)omode.GetItem(0);
                    fright = (double)oright.GetItem(0);

                    if (fleft > fmode)
                        throw new ValueError("left > mode");
                    if (fmode > fright)
                        throw new ValueError("mode > right");
                    if (fleft == fright)
                        throw new ValueError("left == right");

                    return cont3_array_sc(internal_state, RandomDistributions.rk_triangular, size, fleft, fmode, fright);
                }

                if (np.anyb(np.greater(oleft, omode)))
                {
                    throw new ValueError("left > mode");
                }
                if (np.anyb(np.greater(omode, oright)))
                {
                    throw new ValueError("mode > right");
                }
                if (np.anyb(np.equal(oleft, oright)))
                {
                    throw new ValueError("left == right");
                }
                return cont3_array(internal_state, RandomDistributions.rk_triangular, size, oleft, omode, oright);
            }
            #endregion

            #region uniform
            public ndarray uniform(double low = 0.0, double high = 1.0, shape newdims = null)
            {
                ndarray lowarr = np.array(new double[] { low });
                ndarray higharr = np.array(new double[] { high });

                return uniform(lowarr, higharr, newdims);
            }

            public ndarray uniform(object low, object high, shape newdims = null)
            {
                ndarray olow = np.asanyarray(low).astype(np.Float64);
                ndarray ohigh = np.asanyarray(high).astype(np.Float64);
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                if (olow.size == 1 && ohigh.size == 1)
                {
                    double flow = (double)olow;
                    double fhigh = (double)ohigh;
                    double fscale = fhigh - flow;

                    if (double.IsInfinity(fscale))
                    {
                        throw new Exception("Range exceeds valid bounds");
                    }


                    return cont2_array_sc(internal_state, RandomDistributions.rk_uniform, size, flow, fscale);
                }

                ndarray odiff = np.subtract(ohigh, olow);
                if (!np.allb(np.isfinite(odiff)))
                    throw new Exception("Range exceeds valid bounds");


                return cont2_array(internal_state, RandomDistributions.rk_uniform, size, olow, odiff);
            }

            #endregion

            #region vonmises

            public ndarray vonmises(object mu, object kappa, shape newdims = null)
            {
                ndarray omu, okappa;
                double fmu, fkappa;
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                omu = asanyarray(mu).astype(np.Float64);
                okappa = asanyarray(kappa).astype(np.Float64);

                if (omu.size == 1 && okappa.size == 1)
                {
                    fmu = (double)omu.GetItem(0);
                    fkappa = (double)okappa.GetItem(0);

                    if (fkappa < 0)
                        throw new ValueError("kappa < 0");

                    return cont2_array_sc(internal_state, RandomDistributions.rk_vonmises, size, fmu, fkappa);
                }

                if (np.anyb(np.less(okappa, 0.0)))
                {
                    throw new ValueError("kappa < 0");
                }
                return cont2_array(internal_state, RandomDistributions.rk_vonmises, size, omu, okappa);
            }

            #endregion

            #region wald

            public ndarray wald(object mean, object scale, shape newdims = null)
            {
                ndarray omean = np.asanyarray(mean).astype(np.Float64);
                ndarray oscale = np.asanyarray(scale).astype(np.Float64);
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                if (omean.size == 1 && oscale.size == 1)
                {
                    double fmean = Convert.ToDouble(mean);
                    double fscale = Convert.ToDouble(scale);

                    if (fmean <= 0)
                        throw new ValueError("mean <= 0");
                    if (fscale <= 0)
                        throw new ValueError("scale <= 0");

                    return cont2_array_sc(internal_state, RandomDistributions.rk_wald, size, fmean, fscale);
                }

                if (np.anyb(np.less_equal(omean, 0.0)))
                    throw new ValueError("mean <= 0.0");
                if (np.anyb(np.less_equal(oscale, 0.0)))
                    throw new ValueError("scale <= 0.0");

                return cont2_array(internal_state, RandomDistributions.rk_wald, size, omean, oscale);
            }

            #endregion

            #region weibull

            public ndarray weibull(object a, shape newdims = null)
            {
                ndarray oa;
                double fa;
                npy_intp[] size = null;
                if (newdims != null)
                    size = newdims.iDims;

                oa = asanyarray(a).astype(np.Float64);

                if (oa.size == 1)
                {
                    fa = (double)oa.GetItem(0);
                    if ((bool)np.signbit(fa).GetItem(0))
                        throw new ValueError("a < 0");

                    return cont1_array_sc(internal_state, RandomDistributions.rk_weibull, size, fa);
                }


                if (np.anyb(np.signbit(oa)))
                {
                    throw new ValueError("a < 0");
                }
                return cont1_array(internal_state, RandomDistributions.rk_weibull, size, oa);
            }


            #endregion

            #region zipf

            public ndarray zipf(object a, shape shape = null)
            {
                ndarray oa;
                double fa;
                npy_intp[] size = null;
                if (shape != null)
                    size = shape.iDims;

                oa = asanyarray(a).astype(np.Float64);

                if (oa.size == 1)
                {
                    fa = (double)oa.GetItem(0);

                    // use logic that ensures NaN is rejected.
                    if (!(fa > 1.0))
                    {
                        throw new ValueError("'a' must be a valid float > 1.0");
                    }
                    return discd_array_sc(internal_state, RandomDistributions.rk_zipf, size, fa);
                }

                // use logic that ensures NaN is rejected.
                if (!np.allb(np.greater(oa, 1.0)))
                    throw new ValueError("'a' must contain valid floats > 1.0");

                return discd_array(internal_state, RandomDistributions.rk_zipf, size, oa);
            }



            #endregion


            #region helper functions
    
            private npy_intp CountTotalElements(npy_intp[] dims)
            {
                npy_intp TotalElements = 1;
                for (int i = 0; i < dims.Length; i++)
                {
                    TotalElements *= dims[i];
                }

                return TotalElements;
            }

            private shape ConvertToSingleElementIfNull(shape _shape)
            {
                if (_shape == null)
                {
                    return new shape(1);
                }
                return _shape;
            }
            private npy_intp CountTotalElements(shape _shape)
            {
                npy_intp TotalElements = 1;
                for (int i = 0; i < _shape.iDims.Length; i++)
                {
                    TotalElements *= _shape.iDims[i];
                }

                return TotalElements;
            }
            #endregion

            #region Python Version

            private ndarray cont0_array(rk_state state, Func<rk_state, double> func, npy_intp []size)
            {
                double[] array_data;
                ndarray array;
                npy_intp i;

                if (size == null)
                {
                    lock (rk_lock)
                    {
                        double rv = func(state);
                        return np.array(rv);
                    }
                }
                else
                {
                    array_data = new double[CountTotalElements(size)];
                    lock (rk_lock)
                    {
                        for (i = 0; i < array_data.Length; i++)
                        {
                            array_data[i] = func(state);
                        }
                    }
                    array = np.array(array_data);
                    return array.reshape(size);

                }

            }

            private ndarray cont1_array(rk_state state, Func<rk_state, double, double> func, npy_intp [] size, ndarray oa)
            {
                double[] array_data;
                double[] oa_data;

                if (size == null)
                {
                    npy_intp length = CountTotalElements(oa.dims);
                    array_data = new double[length];

                    oa_data = oa.AsDoubleArray();
                    for (npy_intp i = 0; i < length; i++)
                    {
                        array_data[i] = func(state, oa_data[i]);
                    }
                }

                else
                {
                    array_data = new double[CountTotalElements(size)];

                    var iter = NpyCoreApi.IterNew(oa);

                    oa_data = iter.Iter.ao.data.datap as double[];
                    if (iter.Iter.size != array_data.Length)
                    {
                        throw new ValueError("size is not compatible with inputs");
                    }

                    int index = 0;

                    foreach (var dd in iter)
                    {
                        array_data[index++] = func(state, oa_data[iter.Iter.dataptr.data_offset >> double_divsize]);
                    }
                }

                return np.array(array_data).reshape(size);
            }

   


            private ndarray cont1_array_sc(rk_state state, Func<rk_state, double, double> func, npy_intp[] size, double a)
            {
                if (size == null)
                {
                    double rv = func(state, a);
                    return np.asanyarray(rv);
                }
                else
                {
                    var array_data = new double[CountTotalElements(size)];
                    lock (rk_lock)
                    {
                        for (int i = 0; i < array_data.Length; i++)
                        {
                            array_data[i] = func(state, a);
                        }
                    }
                    return np.array(array_data).reshape(size);
                }
            }


            private ndarray cont2_array_sc(rk_state state, Func<rk_state, double, double, double> func, npy_intp []size, double a, double b)
            {
                if (size == null)
                {
                    double rv = func(state, a, b);
                    return np.asanyarray(rv);
                }
                else
                {
                    var array_data = new double[CountTotalElements(size)];
                    lock (rk_lock)
                    {
                        for (int i = 0; i < array_data.Length; i++)
                        {
                            array_data[i] = func(state, a, b);
                        }
                    }
                    return np.array(array_data).reshape(size);
                }

            }


            private ndarray cont2_array(rk_state state, Func<rk_state, double, double, double> func, npy_intp []size, ndarray oa, ndarray ob)
            {
                broadcast multi;
                ndarray array;
                double[] array_data;

                if (size == null)
                {
                    multi = np.broadcast(oa, ob);
                    array = np.empty(multi.shape, dtype: np.Float64);
                }
                else
                {
                    array = np.empty(new shape(size), dtype: np.Float64);
                    multi = np.broadcast(oa, ob, array);
                    if (multi.shape != array.shape)
                    {
                        throw new ValueError("size is not compatible with inputs");
                    }
                }

                array_data = array.Array.data.datap as double[];

                VoidPtr vpoa = multi.IterData(0);
                VoidPtr vpob = multi.IterData(1);


                double[] oa_data = multi.IterData(0).datap as double[];
                double[] ob_data = multi.IterData(1).datap as double[];

                for (int i = 0; i < multi.size; i++)
                {
                    vpoa=  multi.IterData(0);
                    vpob = multi.IterData(1);
                    array_data[i] = func(state, oa_data[vpoa.data_offset >> double_divsize], ob_data[vpob.data_offset >> double_divsize]);
                    multi.IterNext();
                }

                return np.array(array_data);

            }

            private ndarray cont3_array_sc(rk_state state, Func<rk_state, double, double, double, double> func, npy_intp[] size, double a, double b, double c)
            {
                if (size == null)
                {
                    double rv = func(state, a, b, c);
                    return np.asanyarray(rv);
                }
                else
                {
                    var array_data = new double[CountTotalElements(size)];
                    lock (rk_lock)
                    {
                        for (int i = 0; i < array_data.Length; i++)
                        {
                            array_data[i] = func(state, a, b, c);
                        }
                    }
                    return np.array(array_data).reshape(size);
                }
            }

            private ndarray cont3_array(rk_state state, Func<rk_state, double, double, double, double> func, npy_intp[] size, ndarray oa, ndarray ob, ndarray oc)
            {
                broadcast multi;
                ndarray array;
                double[] array_data;

                if (size == null)
                {
                    multi = np.broadcast(oa, ob, oc);
                    array = np.empty(multi.shape, dtype: np.Float64);
                }
                else
                {
                    array = np.empty(new shape(size), dtype: np.Float64);
                    multi = np.broadcast(oa, ob, oc, array);
                    if (multi.shape != array.shape)
                    {
                        throw new ValueError("size is not compatible with inputs");
                    }
                }

                array_data = array.Array.data.datap as double[];

                VoidPtr vpoa = multi.IterData(0);
                VoidPtr vpob = multi.IterData(1);
                VoidPtr vpoc = multi.IterData(2);


                double[] oa_data = multi.IterData(0).datap as double[];
                double[] ob_data = multi.IterData(1).datap as double[];
                double[] oc_data = multi.IterData(2).datap as double[];

                for (int i = 0; i < multi.size; i++)
                {
                    vpoa = multi.IterData(0);
                    vpob = multi.IterData(1);
                    vpoc = multi.IterData(2);

                    array_data[i] = func(state, oa_data[vpoa.data_offset >> double_divsize], 
                                                ob_data[vpob.data_offset >> double_divsize],
                                                oc_data[vpoc.data_offset >> double_divsize]);
                    multi.IterNext();
                }

                return np.array(array_data);
            }

  
            private ndarray discd_array(rk_state state, Func<rk_state, double, long> func, npy_intp[] size, ndarray oa)
            {
                long[] array_data;
                ndarray array;
                npy_intp length;
                npy_intp i;
                broadcast multi;
                flatiter itera;

                if (size == null)
                {
                    array_data = new long[CountTotalElements(oa.dims)];
                    length = array_data.Length;
                    double[] oa_data = oa.Array.data.datap as double[];

                    itera = NpyCoreApi.IterNew(oa);

                    foreach (var dd in itera)
                    {
                        array_data[itera.Iter.index] = func(state, oa_data[itera.CurrentPtr.data_offset >> double_divsize]);
                    }

                    return np.array(array_data);
                }
                else
                {
                    array = np.empty(new shape(size), np.Int64);
                    array_data = array.Array.data.datap as long[];

                    multi = np.broadcast(array, oa);
                    if (multi.size != array.size)
                    {
                        throw new ValueError("size is not compatible with inputs");
                    }

                    double[] oa_data = multi.IterData(1).datap as double[];
                    for (i = 0; i < multi.size; i++)
                    {
                        var vpoa = multi.IterData(1);
                        array_data[i] = func(state, oa_data[vpoa.data_offset >> double_divsize]);
                        multi.IterNext();
                    }

                    return array.reshape(size);
                }

            }

            private ndarray discd_array_sc(rk_state state, Func<rk_state, double, long> func, npy_intp[] size, double a)
            {
                if (size == null)
                {
                    var rv = func(state, a);
                    return asanyarray(rv);
                }

                var array_data = new long[CountTotalElements(size)];
                var length = array_data.Length;
                for (int i = 0; i < length; i++)
                {
                    array_data[i] = func(state, a);
                }

                return asanyarray(array_data).reshape(size);
            }

            private ndarray discnp_array(rk_state state, Func<rk_state, long, double, long> func, npy_intp[] size, ndarray on, ndarray op)
            {
                broadcast multi;
                ndarray array;
                long[] array_data;

                if (size == null)
                {
                    multi = np.broadcast(on, op);
                    array = np.empty(multi.shape, dtype: np.Int64);
                }
                else
                {
                    array = np.empty(new shape(size), dtype: np.Int64);
                    multi = np.broadcast(on, op, array);
                    if (multi.shape != array.shape)
                    {
                        throw new ValueError("size is not compatible with inputs");
                    }
                }

                array_data = array.AsInt64Array();

                VoidPtr vpon = multi.IterData(0);
                VoidPtr vpop = multi.IterData(1);


                long[] on_data = multi.IterData(0).datap as long[];
                double[] op_data = multi.IterData(1).datap as double[];

                for (int i = 0; i < multi.size; i++)
                {
                    vpon = multi.IterData(0);
                    vpop = multi.IterData(1);
                    array_data[i] = func(state, on_data[vpon.data_offset >> long_divsize], op_data[vpop.data_offset >> double_divsize]);
                    multi.IterNext();
                }

                return np.array(array_data);
            }

            private ndarray discnp_array_sc(rk_state state, Func<rk_state, long, double, long> func, npy_intp[] size, long n, double p)
            {
                if (size == null)
                {
                    long rv = func(state, n, p);
                    return asanyarray(rv);
                }

                long[] array_data = new long[CountTotalElements(size)];

                for (int i = 0; i < array_data.Length; i++)
                {
                    array_data[i] = func(state, n, p);
                }

                return asanyarray(array_data).reshape(size);
            }

            private ndarray discdd_array(rk_state state, Func<rk_state, double, double, long> func, npy_intp[] size, ndarray on, ndarray op)
            {
                broadcast multi;
                ndarray array;
                long[] array_data;

                if (size == null)
                {
                    multi = np.broadcast(on, op);
                    array = np.empty(multi.shape, dtype: np.Int64);
                }
                else
                {
                    array = np.empty(new shape(size), dtype: np.Int64);
                    multi = np.broadcast(on, op, array);
                    if (multi.shape != array.shape)
                    {
                        throw new ValueError("size is not compatible with inputs");
                    }
                }

                array_data = array.AsInt64Array();

                VoidPtr vpon = multi.IterData(0);
                VoidPtr vpop = multi.IterData(1);


                long[] on_data = multi.IterData(0).datap as long[];
                double[] op_data = multi.IterData(1).datap as double[];

                for (int i = 0; i < multi.size; i++)
                {
                    vpon = multi.IterData(0);
                    vpop = multi.IterData(1);
                    array_data[i] = func(state, on_data[vpon.data_offset >> long_divsize], op_data[vpop.data_offset >> double_divsize]);
                    multi.IterNext();
                }

                return np.array(array_data);
            }

            private ndarray discdd_array_sc(rk_state state, Func<rk_state, double, double, long> func, npy_intp[] size, long n, double p)
            {
                if (size == null)
                {
                    long rv = func(state, n, p);
                    return asanyarray(rv);
                }

                long[] array_data = new long[CountTotalElements(size)];

                for (int i = 0; i < array_data.Length; i++)
                {
                    array_data[i] = func(state, n, p);
                }

                return asanyarray(array_data).reshape(size);
            }




            private ndarray discnmN_array(rk_state state, Func<rk_state, long, long, long, long> func, npy_intp[] size, ndarray on, ndarray om, ndarray oN)
            {
                broadcast multi;
                ndarray array;
                long[] array_data;

                if (size == null)
                {
                    multi = np.broadcast(on, om, oN);
                    array = np.empty(multi.shape, dtype: np.Int32);
                }
                else
                {
                    array = np.empty(new shape(size), dtype: np.Int32);
                    multi = np.broadcast(on, om, oN, array);
                    if (multi.shape != array.shape)
                    {
                        throw new ValueError("size is not compatible with inputs");
                    }
                }

                array_data = array.AsInt64Array();

                VoidPtr vpon = multi.IterData(0);
                VoidPtr vpom = multi.IterData(1);
                VoidPtr vpoN = multi.IterData(2);


                long[] on_data = multi.IterData(0).datap as long[];
                long[] om_data = multi.IterData(1).datap as long[];
                long[] oN_data = multi.IterData(2).datap as long[];

                for (int i = 0; i < multi.size; i++)
                {
                    vpon = multi.IterData(0);
                    vpom = multi.IterData(1);
                    vpoN = multi.IterData(2);

                    array_data[i] = func(state, on_data[vpon.data_offset >> long_divsize], 
                                                om_data[vpom.data_offset >> long_divsize],
                                                oN_data[vpoN.data_offset >> long_divsize]);
                    multi.IterNext();
                }

                return np.array(array_data);
            }

            private ndarray discnmN_array_sc(rk_state state, Func<rk_state, long, long, long, long> func, npy_intp[] size, long n, long m, long N)
            {
                if (size == null)
                {
                    long rv = func(state, n, m, N);
                    return asanyarray(rv);
                }

                long[] array_data = new long[CountTotalElements(size)];

                for (int i = 0; i < array_data.Length; i++)
                {
                    array_data[i] = func(state, n, m, N);
                }

                return asanyarray(array_data).reshape(size);
            }

            private double kahan_sum(double[] darr, npy_intp n)
            {
                double c, y, t, sum;
                npy_intp i;
                sum = darr[0];
                c = 0.0;
                for (i = 0; i < n; i++)
                {
                    y = darr[i] - c;
                    t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }

                return sum;
            }

            private shape _shape_from_size(shape size, npy_intp d)
            {
                if (size == null)
                {
                    return new shape(d);
                }

                npy_intp[] newdims = new npy_intp[size.iDims.Length + 1];
                Array.Copy(size.iDims, 0, newdims, 0, size.iDims.Length);
                newdims[size.iDims.Length] = d;

                return new shape(newdims);

            }



            #endregion
        }
    }
}
