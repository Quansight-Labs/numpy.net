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

using NumpyLib;
using System;
using System.Collections.Generic;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet.RandomAPI
{
    public class RandomGeneratorPython : IRandomGenerator
    {
        private const int RK_STATE_LEN = 624;
        private UInt64[] key = new ulong[RK_STATE_LEN];
 
  

        public void Seed(ulong? inseed, rk_state state)
        {
            uint pos;

            if (!inseed.HasValue)
            {
                inseed = (ulong)DateTime.Now.Ticks;
            }

            ulong seed = inseed.Value;
            seed &= 0xffffffffUL;

            /* Knuth's PRNG as used in the Mersenne Twister reference implementation */
            for (pos = 0; pos < RK_STATE_LEN; pos++)
            {
                key[pos] = seed;
                seed = (1812433253UL * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffUL;
            }
            state.pos = RK_STATE_LEN;
            state.gauss = 0;
            state.has_gauss = false;
            state.has_binomial = false;
        }

        public ulong getNextUInt64(rk_state state)
        {
            /* Magic Mersenne Twister constants */
            const int N = 624;
            const int M = 397;
            const long MATRIX_A = 0x9908b0df;
            const long UPPER_MASK = 0x80000000;
            const long LOWER_MASK = 0x7fffffff;

            ulong y;

            if (state.pos == RK_STATE_LEN)
            {
                int i;

                for (i = 0; i < N - M; i++)
                {
                    y = (key[i] & UPPER_MASK) | (key[i + 1] & LOWER_MASK);
                    key[i] = key[i + M] ^ (y >> 1) ^ (ulong)(-(long)((y & 1) & MATRIX_A));
                }
                for (; i < N - 1; i++)
                {
                    y = (key[i] & UPPER_MASK) | (key[i + 1] & LOWER_MASK);
                    key[i] = key[i + (M - N)] ^ (y >> 1) ^ (ulong)(-(long)((y & 1) & MATRIX_A));
                }
                y = (key[N - 1] & UPPER_MASK) | (key[0] & LOWER_MASK);
                key[N - 1] = key[M - 1] ^ (y >> 1) ^ (ulong)(-(long)((y & 1) & MATRIX_A));

                state.pos = 0;
            }
            y = key[state.pos++];

            /* Tempering */
            y ^= (y >> 11);
            y ^= (y << 7) & 0x9d2c5680;
            y ^= (y << 15) & 0xefc60000;
            y ^= (y >> 18);

            return (ulong)y;
        }

        public double getNextDouble(rk_state state)
        {
            /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
            long a = (long)getNextUInt64(state) >> 5;
            long b = (long)getNextUInt64(state) >> 6;
            return (a * 67108864.0 + b) / 9007199254740992.0;
        }
    }
}
