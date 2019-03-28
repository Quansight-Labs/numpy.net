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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
using npy_uintp = System.UInt64;
#else
using npy_intp = System.Int32;
using npy_uintp = System.UInt632;
#endif

namespace NumpyLib
{
    internal partial class numpyinternal
    {
        /*
        *****************************************************************************
        **                            NUMERIC SORTS                                **
        *****************************************************************************
        */


        private static void store_pivot(npy_intp pivot, npy_intp kth, npy_intp[] pivots, ref npy_intp? npiv)
        {
            if (pivots == null)
            {
                return;
            }

            /*
             * If pivot is the requested kth store it, overwritting other pivots if
             * required. This must be done so iterative partition can work without
             * manually shifting lower data offset by kth each time
             */
            if (pivot == kth && npiv == npy_defs.NPY_MAX_PIVOT_STACK)
            {
                pivots[npiv.Value - 1] = pivot;
            }
            /*
             * we only need pivots larger than current kth, larger pivots are not
             * useful as partitions on smaller kth would reorder the stored pivots
             */
            else if (pivot >= kth && npiv < npy_defs.NPY_MAX_PIVOT_STACK)
            {
                pivots[npiv.Value] = pivot;
                npiv += 1;
            }
        }


        private static NpyArray_PartitionFunc get_partition_func(NPY_TYPES NpyType, NPY_SELECTKIND which)
        {
            return Common_PartitionFunc;
        }

        private static NpyArray_ArgPartitionFunc get_argpartition_func(NPY_TYPES NpyType, NPY_SELECTKIND which)
        {
            return null;
        }

        private static int Common_PartitionFunc(VoidPtr v, npy_intp num, npy_intp kth, npy_intp[] pivots, ref npy_intp? npiv, object not_used)
        {
            switch (v.type_num)
            {
                case NPY_TYPES.NPY_INT32:
                    return partition_introselect<Int32>(v.datap as Int32[], num, kth, pivots, ref npiv, not_used);
            }
            return 0;
        }



        private static int partition_introselect<T>(T[] v,
                         npy_intp num, npy_intp kth,
                         npy_intp[] pivots,
                         ref npy_intp? npiv,
                         object NOT_USED)
        {
            npy_intp low = 0;
            npy_intp high = num - 1;
            int depth_limit;

            if (npiv == null)
                pivots = null;

            while (pivots != null && npiv > 0)
            {
                if (pivots[npiv.Value - 1] > kth)
                {
                    /* pivot larger than kth set it as upper bound */
                    high = pivots[npiv.Value - 1] - 1;
                    break;
                }
                else if (pivots[npiv.Value - 1] == kth)
                {
                    /* kth was already found in a previous iteration -> done */
                    return 0;
                }

                low = pivots[npiv.Value - 1] + 1;

                /* pop from stack */
                npiv -= 1;
            }

            /*
             * use a faster O(n*kth) algorithm for very small kth
             * e.g. for interpolating percentile
             */
            if (kth - low < 3)
            {
                DUMBSELECT(v, low, high - low + 1, kth - low);
                store_pivot(kth, kth, pivots, ref npiv);
                return 0;
            }
            else if (true  && kth == num - 1)
            {
                /* useful to check if NaN present via partition(d, (x, -1)) */
                npy_intp k;
                npy_intp maxidx = low;
                T maxval = v[low];
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(v[k], maxval))
                    {
                        maxidx = k;
                        maxval = v[k];
                    }
                }
                SWAP(v, kth, maxidx);
                return 0;
            }

            depth_limit = npy_get_msb(num) * 2;

            /* guarantee three elements */
            for (; low + 1 < high;)
            {
                npy_intp ll = low + 1;
                npy_intp hh = high;

                /*
                 * if we aren't making sufficient progress with median of 3
                 * fall back to median-of-median5 pivot for linear worst case
                 * med3 for small sizes is required to do unguarded partition
                 */
                if (depth_limit > 0 || hh - ll < 5)
                {
                    npy_intp mid = low + (high - low) / 2;
                    /* median of 3 pivot strategy,
                     * swapping for efficient partition */
                    MEDIAN3_SWAP(v, low, mid, high);
                }
                else
                {
                    npy_intp mid;
                    mid = ll + median_of_median5(v, ll, hh - ll, null, null);
                    SWAP(v, mid,low);
                    /* adapt for the larger partition than med3 pivot */
                    ll--;
                    hh++;
                }

                depth_limit--;

                /*
                 * find place to put pivot (in low):
                 * previous swapping removes need for bound checks
                 * pivot 3-lowest [x x x] 3-highest
                 */
                UNGUARDED_PARTITION(v, v[low], ref ll, ref hh);

                /* move pivot into position */
                SWAP(v, low, hh);

                /* kth pivot stored later */
                if (hh != kth)
                {
                    store_pivot(hh, kth, pivots, ref npiv);
                }

                if (hh >= kth)
                    high = hh - 1;
                if (hh <= kth)
                    low = ll;
            }

            /* two elements */
            if (high == low + 1)
            {
                if (LT(v[high], v[low]))
                {
                    SWAP(v, high, low);
                }
            }
            store_pivot(kth, kth, pivots, ref npiv);

            return 0;
        }



        /*
         * N^2 selection, fast only for very small kth
         * useful for close multiple partitions
         * (e.g. even element median, interpolating percentile)
         */
        static int DUMBSELECT<T>(T[] v, npy_intp left,  npy_intp num, npy_intp kth)
        {
            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                T minval = v[i];
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(v[k], minval))
                    {
                        minidx = k;
                        minval = v[k];
                    }
                }
                SWAP(v, i, minidx);
            }

            return 0;
        }


        static bool LT(dynamic a, dynamic b)
        {
            return a < b;
        }
 

        static void SWAP<T>(T[] v, npy_intp a, npy_intp b)
        {
            T tmp = v[b];
            v[b]= v[a];
            v[a] = tmp;
        }

        /*
         * median of 3 pivot strategy
         * gets min and median and moves median to low and min to low + 1
         * for efficient partitioning, see unguarded_partition
         */
        static void MEDIAN3_SWAP<T>(T[] v, npy_intp low, npy_intp mid, npy_intp high)
        {
            if (LT(v[high], v[mid]))
                SWAP(v, high, mid);
            if (LT(v[high], v[low]))
                SWAP(v, high, low);
            /* move pivot to low */
            if (LT(v[low], v[mid]))
                SWAP(v, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP(v, mid, low + 1);
        }



        /* select index of median of five elements */
        static npy_intp MEDIAN5<T>(T[] v, npy_intp voffset)
        {
            /* could be optimized as we only need the index (no swaps) */
            if (LT(v[voffset+1], v[voffset + 0]))
            {
                SWAP(v, voffset+1, voffset + 0);
            }
            if (LT(v[voffset + 4], v[voffset + 3]))
            {
                SWAP(v, voffset + 4, voffset + 3);
            }
            if (LT(v[voffset + 3], v[voffset + 0]))
            {
                SWAP(v, voffset + 3, voffset + 0);
            }
            if (LT(v[voffset + 4], v[voffset + 1]))
            {
                SWAP(v, voffset + 4, voffset + 1);
            }
            if (LT(v[voffset + 2], v[voffset + 1]))
            {
                SWAP(v, voffset + 2, voffset + 1);
            }
            if (LT(v[voffset + 3], v[voffset + 2]))
            {
                if (LT(v[voffset + 3], v[voffset + 1]))
                {
                    return 1;
                }
                else
                {
                    return 3;
                }
            }
            else
            {
                /* v[1] and v[2] swapped into order above */
                return 2;
            }
        }

        /*
         * select median of median of blocks of 5
         * if used as partition pivot it splits the range into at least 30%/70%
         * allowing linear time worstcase quickselect
         */
        static npy_intp median_of_median5<T>(T[] v, npy_intp voffset, npy_intp num, npy_intp[] pivots, npy_intp? npiv)
        {
            npy_intp i, subleft;
            npy_intp right = num - 1;
            npy_intp nmed = (right + 1) / 5;

            for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                npy_intp m = MEDIAN5(v, subleft);
                SWAP(v, voffset + subleft + m, voffset + i);
            }

            if (nmed > 2)
                partition_introselect(v, nmed, nmed / 2, pivots, ref npiv, null);

            return nmed / 2;
        }


        /*
         * partition and return the index were the pivot belongs
         * the data must have following property to avoid bound checks:
         *                  ll ... hh
         * lower-than-pivot [x x x x] larger-than-pivot
         */
        static void UNGUARDED_PARTITION<T>(T[] v, T pivot, ref npy_intp ll, ref npy_intp hh)
        {
            for (; ; )
            {
                do ll++; while (LT(v[ll], pivot));
                do hh--; while (LT(pivot, v[hh]));

                if (hh < ll)
                    break;

                SWAP(v, ll, hh);
            }
        }

        static int npy_get_msb(npy_intp unum)
        {
            return npy_get_msb(Convert.ToUInt64(unum));
        }
        static int npy_get_msb(npy_uintp unum)
        {
            int depth_limit = 0;
            while ((unum >>= 1) != 0)
            {
                depth_limit++;
            }
            return depth_limit;
        }

    }
}
