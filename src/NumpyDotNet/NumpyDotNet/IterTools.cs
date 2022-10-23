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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using NumpyLib;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{
    public class IterTools
    {
        public static IEnumerable<object[]> products(object arg, int repeat = 1)
        {
            return _IterTools.products(arg, repeat);
        }
    }

    internal class _IterTools
    {
        #region products
        public static IEnumerable<object[]> products(object arg, int repeat = 1)
        {
            // product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
            // product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111

            List<object> pools = new List<object>();

            for (int i = 0; i < repeat; i++)
            {
                if (arg is IEnumerable<object>)
                {
                    bool saveToPools = false;
                    List<object> s = new List<object>();
                    IEnumerable<object> earg = arg as IEnumerable<object>;
                    foreach (var e in earg)
                    {
                        if (e is string)
                        {
                            s = new List<object>();
                            string se = e as string;
                            foreach (var _se in se)
                            {
                                s.Add(_se);
                            }
                            pools.Add(s);
                            saveToPools = false;
                        }
                        else
                        {
                            s.Add(e);
                            saveToPools = true;
                        }

                    }

                    if (saveToPools)
                    {
                        pools.Add(s);
                    }
                }
                else
                {
                    throw new Exception(string.Format("Unrecognized arguments to IterTools.product: {0}", arg.ToString()));
                }

            }

            List<List<object>> result = new List<List<object>>();

            int totalResults = ProductCountExpectedResults(pools);
            for (int i = 0; i < totalResults; i++)
            {
                List<object> newpool = new List<object>();
                result.Add(newpool);
            }

            int poolAdjustmentValue = 1;
            foreach (var p in pools)
            {
                IEnumerable<object> ep = p as IEnumerable<object>;

                poolAdjustmentValue *= ep.Count();

                int poolDividerValue = totalResults / poolAdjustmentValue;

                for (int indexIntoResults = 0; indexIntoResults < totalResults; indexIntoResults++)
                {
                    int indexIntoPool = ProductCalculateIndexIntoPool(indexIntoResults, ep.Count(), poolDividerValue);

                    List<object> newpool = result[indexIntoResults];
                    newpool.Add(ep.ElementAt(indexIntoPool));
                }
            }

            foreach (var prod1 in result)
            {
                yield return prod1.ToArray();
            }
        }

        private static int ProductCountExpectedResults(List<object> pools)
        {
            int totalPools = 1;

            foreach (var pool in pools)
            {
                IEnumerable<object> epool = pool as IEnumerable<object>;
                if (epool != null)
                {
                    totalPools *= epool.Count();
                }
                else
                {
                    totalPools *= 1;
                }
            }

            return totalPools;
        }

        private static int ProductCalculateIndexIntoPool(int indexIntoResults, int poolSize, int poolDividerValue)
        {
            int indexGroup = indexIntoResults / poolDividerValue;
            int index = indexGroup % poolSize;

            return index;
        }
        #endregion


    }
}
