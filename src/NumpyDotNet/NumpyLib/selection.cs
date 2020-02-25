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
using npy_uintp = System.UInt32;
#endif

namespace NumpyLib
{
    internal class partition_Double : partition<double>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(double);
        }

        public override bool LT(NPY_TYPES num_type, double a, double b)
        {
            return a < b;
        }

    }

    internal abstract class partition<T>
    {
        public abstract int GetTypeSize(VoidPtr v);
        public abstract bool LT(NPY_TYPES num_type, T a, T b);

        public int partition_introselect(VoidPtr v, npy_intp num, npy_intp kth, npy_intp[] pivots, ref npy_intp? npiv, bool inexact)
        {
            npy_intp low = 0;
            npy_intp high = num - 1;
            int depth_limit;

            v = new VoidPtr(v);
            v.data_offset /= GetTypeSize(v);

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
            else if (inexact && kth == num - 1)
            {
                /* useful to check if NaN present via partition(d, (x, -1)) */
                npy_intp k;
                npy_intp maxidx = low;
                T maxval = GetItem(v, low);
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(v.type_num, GetItem(v, k), maxval))
                    {
                        maxidx = k;
                        maxval = GetItem(v, k);
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
                    mid = ll + median_of_median5(v, ll, hh - ll, null, null, inexact);
                    SWAP(v, mid, low);
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
                UNGUARDED_PARTITION(v, GetItem(v, low), ref ll, ref hh);

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
                if (LT(v.type_num, GetItem(v, high), GetItem(v, low)))
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
        private int DUMBSELECT(VoidPtr v, npy_intp left, npy_intp num, npy_intp kth)
        {
            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                T minval = GetItem(v, i + left);
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(v.type_num, GetItem(v, k + left), minval))
                    {
                        minidx = k;
                        minval = GetItem(v, k + left);
                    }
                }
                SWAP(v, i + left, minidx + left);
            }

            return 0;
        }

        public void SWAP(VoidPtr v, npy_intp aindex, npy_intp bindex)
        {
            var vv = v.datap as T[];

            T tmp = vv[v.data_offset + bindex];
            vv[v.data_offset + bindex] = vv[v.data_offset + aindex];
            vv[v.data_offset + aindex] = tmp;
        }

        /*
        * median of 3 pivot strategy
        * gets min and median and moves median to low and min to low + 1
        * for efficient partitioning, see unguarded_partition
        */
        void MEDIAN3_SWAP(VoidPtr v, npy_intp low, npy_intp mid, npy_intp high)
        {
            if (LT(v.type_num, GetItem(v, high), GetItem(v, mid)))
                SWAP(v, high, mid);
            if (LT(v.type_num, GetItem(v, high), GetItem(v, low)))
                SWAP(v, high, low);
            /* move pivot to low */
            if (LT(v.type_num, GetItem(v, low), GetItem(v, mid)))
                SWAP(v, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP(v, mid, low + 1);
        }

        void MEDIAN3_SWAP<T>(VoidPtr v, VoidPtr tosort, npy_intp low, npy_intp mid, npy_intp high)
        {
            if (LT(v.type_num, GetItem(v, IDX(tosort, high)), GetItem(v, IDX(tosort, mid))))
                SWAP_IDX(tosort, high, mid);
            if (LT(v.type_num, GetItem(v, IDX(tosort, high)), GetItem(v, IDX(tosort, low))))
                SWAP_IDX(tosort, high, low);
            /* move pivot to low */
            if (LT(v.type_num, GetItem(v, IDX(tosort, low)), GetItem(v, IDX(tosort, mid))))
                SWAP_IDX(tosort, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP_IDX(tosort, mid, low + 1);
        }

        static void SWAP_IDX(VoidPtr v, npy_intp a, npy_intp b)
        {
            npy_intp tmp = GetIDX(v, b);
            SetIDX(v, b, GetIDX(v, a));
            SetIDX(v, a, tmp);
        }

        /* select index of median of five elements */
        npy_intp MEDIAN5(VoidPtr v, npy_intp voffset)
        {
            /* could be optimized as we only need the index (no swaps) */
            if (LT(v.type_num, GetItem(v, voffset + 1), GetItem(v, voffset + 0)))
            {
                SWAP(v, voffset + 1, voffset + 0);
            }
            if (LT(v.type_num, GetItem(v, voffset + 4), GetItem(v, voffset + 3)))
            {
                SWAP(v, voffset + 4, voffset + 3);
            }
            if (LT(v.type_num, GetItem(v, voffset + 3), GetItem(v, voffset + 0)))
            {
                SWAP(v, voffset + 3, voffset + 0);
            }
            if (LT(v.type_num, GetItem(v, voffset + 4), GetItem(v, voffset + 1)))
            {
                SWAP(v, voffset + 4, voffset + 1);
            }
            if (LT(v.type_num, GetItem(v, voffset + 2), GetItem(v, voffset + 1)))
            {
                SWAP(v, voffset + 2, voffset + 1);
            }
            if (LT(v.type_num, GetItem(v, voffset + 3), GetItem(v, voffset + 2)))
            {
                if (LT(v.type_num, GetItem(v, voffset + 3), GetItem(v, voffset + 1)))
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

        /* select index of median of five elements */
        npy_intp MEDIAN5(VoidPtr v, VoidPtr tosort, npy_intp voffset)
        {
            /* could be optimized as we only need the index (no swaps) */
            if (LT(v.type_num, GetItem(v, IDX(tosort, voffset + 1)), GetItem(v, IDX(tosort, voffset + 0))))
            {
                SWAP_IDX(tosort, voffset + 1, voffset + 0);
            }
            if (LT(v.type_num, GetItem(v, IDX(tosort, voffset + 4)), GetItem(v, IDX(tosort, voffset + 3))))
            {
                SWAP_IDX(tosort, voffset + 4, voffset + 3);
            }
            if (LT(v.type_num, GetItem(v, IDX(tosort, voffset + 3)), GetItem(v, IDX(tosort, voffset + 0))))
            {
                SWAP_IDX(tosort, voffset + 3, voffset + 0);
            }
            if (LT(v.type_num, GetItem(v, IDX(tosort, voffset + 4)), GetItem(v, IDX(tosort, voffset + 1))))
            {
                SWAP_IDX(tosort, voffset + 4, voffset + 1);
            }
            if (LT(v.type_num, GetItem(v, IDX(tosort, voffset + 2)), GetItem(v, IDX(tosort, voffset + 1))))
            {
                SWAP_IDX(tosort, voffset + 2, voffset + 1);
            }
            if (LT(v.type_num, GetItem(v, IDX(tosort, voffset + 3)), GetItem(v, IDX(tosort, voffset + 2))))
            {
                if (LT(v.type_num, GetItem(v, IDX(tosort, voffset + 3)), GetItem(v, IDX(tosort, voffset + 1))))
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
        npy_intp median_of_median5(VoidPtr v, npy_intp voffset, npy_intp num, npy_intp[] pivots, npy_intp? npiv, bool inexact)
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
                partition_introselect(v, nmed, nmed / 2, pivots, ref npiv, inexact);

            return nmed / 2;
        }

        //npy_intp median_of_median5<T>(VoidPtr v, VoidPtr tosort, npy_intp voffset, npy_intp num, npy_intp[] pivots, npy_intp? npiv, bool inexact)
        //{
        //    npy_intp i, subleft;
        //    npy_intp right = num - 1;
        //    npy_intp nmed = (right + 1) / 5;

        //    for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
        //    {
        //        npy_intp m = MEDIAN5(v, tosort, subleft);
        //        SWAP<npy_intp>(tosort, voffset + subleft + m, voffset + i);
        //    }

        //    if (nmed > 2)
        //        argpartition_introselect(v, tosort, nmed, nmed / 2, pivots, ref npiv, inexact);

        //    return nmed / 2;
        //}

        private npy_intp IDX(VoidPtr vp, npy_intp index)
        {
            npy_intp[] vv = vp.datap as npy_intp[];
            return vv[vp.data_offset + index];
        }
        private static npy_intp GetIDX(VoidPtr v, npy_intp index)
        {
            npy_intp[] vv = v.datap as npy_intp[];
            return vv[v.data_offset + index];
        }
        private static npy_intp SetIDX(VoidPtr v, npy_intp index, npy_intp d)
        {
            npy_intp[] vv = v.datap as npy_intp[];
            return vv[v.data_offset + index] = d;
        }

        private static T GetItem(VoidPtr v, npy_intp index)
        {
            T[] vv = v.datap as T[];
            return vv[v.data_offset + index];
        }
        private static T SetItem(VoidPtr v, npy_intp index, T d)
        {
            T[] vv = v.datap as T[];
            return vv[v.data_offset + index] = d;
        }

        /*
     * partition and return the index were the pivot belongs
     * the data must have following property to avoid bound checks:
     *                  ll ... hh
     * lower-than-pivot [x x x x] larger-than-pivot
     */
        void UNGUARDED_PARTITION(VoidPtr v, T pivot, ref npy_intp ll, ref npy_intp hh)
        {
            for (; ; )
            {
                do ll++; while (LT(v.type_num, GetItem(v, ll), pivot));
                do hh--; while (LT(v.type_num, pivot, GetItem(v, hh)));

                if (hh < ll)
                    break;

                SWAP(v, ll, hh);
            }
        }

        void UNGUARDED_PARTITION(VoidPtr v, VoidPtr tosort, T pivot, ref npy_intp ll, ref npy_intp hh)
        {
            for (; ; )
            {
                do ll++; while (LT(v.type_num, GetItem(v, IDX(tosort, ll)), pivot));
                do hh--; while (LT(v.type_num, pivot, GetItem(v, IDX(tosort, hh))));

                if (hh < ll)
                    break;

                SWAP_IDX(tosort, ll, hh);
            }
        }

        private void store_pivot(npy_intp pivot, npy_intp kth, npy_intp[] pivots, ref npy_intp? npiv)
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

#if NPY_INTP_64
        static int npy_get_msb(npy_intp unum)
        {
            return npy_get_msb(Convert.ToUInt64(unum));
        }
#else
        static int npy_get_msb(npy_intp unum)
        {
            return npy_get_msb(Convert.ToUInt32(unum));
        }
#endif

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
            return Common_ArgPartitionFunc;
        }

        private static int Common_PartitionFunc(VoidPtr v, npy_intp num, npy_intp kth, npy_intp[] pivots, ref npy_intp? npiv, object not_used)
        {
            switch (v.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return partition_introselect<bool>(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_BYTE:
                    return partition_introselect<sbyte>(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UBYTE:
                    return partition_introselect<byte>(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_INT16:
                    return partition_introselect<Int16>(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UINT16:
                    return partition_introselect<UInt16>(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_INT32:
                    return partition_introselect<Int32>(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UINT32:
                    return partition_introselect<UInt32>(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_INT64:
                    return partition_introselect<Int64>(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UINT64:
                    return partition_introselect<UInt64>(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_FLOAT:
                    return partition_introselect<float>(v, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_DOUBLE:
                    var xxx = new partition_Double();
                    return xxx.partition_introselect(v, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_DECIMAL:
                    return partition_introselect<decimal>(v, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_COMPLEX:
                    return partition_introselect<System.Numerics.Complex>(v, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_BIGINT:
                    return partition_introselect<System.Numerics.BigInteger>(v, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_OBJECT:
                    return partition_introselect<Object>(v, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_STRING:
                    return partition_introselect<string>(v, num, kth, pivots, ref npiv, true);
            }
            return 0;
        }

        private static int Common_ArgPartitionFunc(VoidPtr v, VoidPtr tosort, npy_intp num, npy_intp kth, npy_intp[] pivots, ref npy_intp? npiv, object not_used)
        {
            switch (v.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return argpartition_introselect<bool>(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_BYTE:
                    return argpartition_introselect<sbyte>(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UBYTE:
                    return argpartition_introselect<byte>(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_INT16:
                    return argpartition_introselect<Int16>(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UINT16:
                    return argpartition_introselect<UInt16>(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_INT32:
                    return argpartition_introselect<Int32>(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UINT32:
                    return argpartition_introselect<UInt32>(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_INT64:
                    return argpartition_introselect<Int64>(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UINT64:
                    return argpartition_introselect<UInt64>(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_FLOAT:
                    return argpartition_introselect<float>(v, tosort, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_DOUBLE:
                    return argpartition_introselect<double>(v, tosort, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_DECIMAL:
                    return argpartition_introselect<decimal>(v, tosort, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_COMPLEX:
                    return argpartition_introselect<System.Numerics.Complex>(v, tosort, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_BIGINT:
                    return argpartition_introselect<System.Numerics.BigInteger>(v, tosort, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_OBJECT:
                    return argpartition_introselect<Object>(v, tosort, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_STRING:
                    return argpartition_introselect<string>(v, tosort, num, kth, pivots, ref npiv, true);

            }
            return 0;
        }

        private static T GetItem<T>(VoidPtr v, npy_intp index)
        {
            T[] vv = v.datap as T[];
            return vv[v.data_offset + index];
        }
        private static T SetItem<T>(VoidPtr v, npy_intp index, T d)
        {
            T[] vv = v.datap as T[];
            return vv[v.data_offset + index] = d;
        }

        private static npy_intp IDX(VoidPtr vp, npy_intp index)
        {
            return GetItem<npy_intp>(vp, index);
        }

        private static int partition_introselect<T>(VoidPtr v,
                         npy_intp num, npy_intp kth,
                         npy_intp[] pivots,
                         ref npy_intp? npiv,
                         bool inexact)
        {
            npy_intp low = 0;
            npy_intp high = num - 1;
            int depth_limit;

            v = new VoidPtr(v);
            v.data_offset /= GetTypeSize(v);

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
                DUMBSELECT<T>(v, low, high - low + 1, kth - low);
                store_pivot(kth, kth, pivots, ref npiv);
                return 0;
            }
            else if (inexact  && kth == num - 1)
            {
                /* useful to check if NaN present via partition(d, (x, -1)) */
                npy_intp k;
                npy_intp maxidx = low;
                T maxval = GetItem<T>(v,low);
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(v.type_num, GetItem<T>(v,k), maxval))
                    {
                        maxidx = k;
                        maxval = GetItem<T>(v, k);
                    }
                }
                SWAP<T>(v, kth, maxidx);
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
                    MEDIAN3_SWAP<T>(v, low, mid, high);
                }
                else
                {
                    npy_intp mid;
                    mid = ll + median_of_median5<T>(v, ll, hh - ll, null, null, inexact);
                    SWAP<T>(v, mid,low);
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
                UNGUARDED_PARTITION(v, GetItem<T>(v, low), ref ll, ref hh);

                /* move pivot into position */
                SWAP<T>(v, low, hh);

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
                if (LT(v.type_num, GetItem<T>(v, high), GetItem<T>(v, low)))
                {
                    SWAP<T>(v, high, low);
                }
            }
            store_pivot(kth, kth, pivots, ref npiv);

            return 0;
        }


        private static int argpartition_introselect<T>(VoidPtr v, VoidPtr _tosortvp,
                    npy_intp num, npy_intp kth,
                    npy_intp[] pivots,
                    ref npy_intp? npiv,
                    bool inexact)
        {
            npy_intp low = 0;
            npy_intp high = num - 1;
            int depth_limit;

            v = new VoidPtr(v);
            v.data_offset /= GetTypeSize(v);

            VoidPtr tosortvp = new VoidPtr(_tosortvp);
            tosortvp.data_offset /= GetTypeSize(tosortvp);

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
                DUMBSELECT<T>(v, tosortvp, low, high - low + 1, kth - low);
                store_pivot(kth, kth, pivots, ref npiv);
                return 0;
            }
            else if (inexact && kth == num - 1)
            {
                /* useful to check if NaN present via partition(d, (x, -1)) */
                npy_intp k;
                npy_intp maxidx = low;
                T maxval = GetItem<T>(v, IDX(tosortvp, low));
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(v.type_num, GetItem<T>(v, IDX(tosortvp, k)), maxval))
                    {
                        maxidx = k;
                        maxval = GetItem<T>(v, IDX(tosortvp, k));
                    }
                }
                SWAP<npy_intp>(tosortvp, kth, maxidx);
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
                    MEDIAN3_SWAP<T>(v, tosortvp, low, mid, high);
                }
                else
                {
                    npy_intp mid;
                    mid = ll + median_of_median5<T>(v, tosortvp, ll, hh - ll, null, null, inexact);
                    SWAP<npy_intp>(tosortvp, mid, low);
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
                UNGUARDED_PARTITION(v, tosortvp, GetItem<T>(v, IDX(tosortvp,low)), ref ll, ref hh);

                /* move pivot into position */
                SWAP<npy_intp>(tosortvp, low, hh);

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
                if (LT(v.type_num, GetItem<T>(v, IDX(tosortvp, high)), GetItem<T>(v, IDX(tosortvp, low))))
                {
                    SWAP<npy_intp>(tosortvp, high, low);
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
        static int DUMBSELECT<T>(VoidPtr v, npy_intp left,  npy_intp num, npy_intp kth)
        {
            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                T minval = GetItem<T>(v, i+left);
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(v.type_num, GetItem<T>(v, k+left), minval))
                    {
                        minidx = k;
                        minval = GetItem<T>(v, k+left);
                    }
                }
                SWAP<T>(v, i+left, minidx+left);
            }

            return 0;
        }

        static int DUMBSELECT<T>(VoidPtr v, VoidPtr tosort, npy_intp left, npy_intp num, npy_intp kth)
        {
            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                T minval = GetItem<T>(v, IDX(tosort,i + left));
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(v.type_num, GetItem<T>(v, IDX(tosort, k + left)), minval))
                    {
                        minidx = k;
                        minval = GetItem<T>(v, IDX(tosort, k + left));
                    }
                }
                SWAP<npy_intp>(tosort, i + left, minidx + left);
            }

            return 0;
        }


        static bool LT(NPY_TYPES num_type, dynamic a, dynamic b)
        {
            if (a is System.Double || a is System.Single)
            {
                if (double.IsNaN(a))
                    return false;
            }
            if (b is System.Double || b is System.Single)
            {
                if (double.IsNaN(b))
                    return true;
            }

            int CompareResult = DefaultArrayHandlers.GetArrayHandler(num_type).CompareTo(a, b);
            return CompareResult < 0;
        }
 

        static void SWAP<T>(VoidPtr v, npy_intp a, npy_intp b)
        {
            T tmp = GetItem<T>(v, b);
            SetItem<T>(v, b, GetItem<T>(v, a));
            SetItem<T>(v, a, tmp);
        }

        /*
         * median of 3 pivot strategy
         * gets min and median and moves median to low and min to low + 1
         * for efficient partitioning, see unguarded_partition
         */
        static void MEDIAN3_SWAP<T>(VoidPtr v, npy_intp low, npy_intp mid, npy_intp high)
        {
            if (LT(v.type_num, GetItem<T>(v, high), GetItem<T>(v, mid)))
                SWAP<T>(v, high, mid);
            if (LT(v.type_num, GetItem<T>(v, high), GetItem<T>(v, low)))
                SWAP<T>(v, high, low);
            /* move pivot to low */
            if (LT(v.type_num, GetItem<T>(v, low), GetItem<T>(v, mid)))
                SWAP<T>(v, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP<T>(v, mid, low + 1);
        }

        static void MEDIAN3_SWAP<T>(VoidPtr v, VoidPtr tosort, npy_intp low, npy_intp mid, npy_intp high)
        {
            if (LT(v.type_num, GetItem<T>(v, IDX(tosort, high)), GetItem<T>(v, IDX(tosort, mid))))
                SWAP<npy_intp>(tosort, high, mid);
            if (LT(v.type_num, GetItem<T>(v, IDX(tosort, high)), GetItem<T>(v, IDX(tosort, low))))
                SWAP<npy_intp>(tosort, high, low);
            /* move pivot to low */
            if (LT(v.type_num, GetItem<T>(v, IDX(tosort, low)), GetItem<T>(v, IDX(tosort, mid))))
                SWAP<npy_intp>(tosort, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP<npy_intp>(tosort, mid, low + 1);
        }

        /* select index of median of five elements */
        static npy_intp MEDIAN5<T>(VoidPtr v, npy_intp voffset)
        {
            /* could be optimized as we only need the index (no swaps) */
            if (LT(v.type_num, GetItem<T>(v, voffset+1), GetItem<T>(v, voffset + 0)))
            {
                SWAP<T>(v, voffset+1, voffset + 0);
            }
            if (LT(v.type_num, GetItem<T>(v, voffset + 4), GetItem<T>(v, voffset + 3)))
            {
                SWAP<T>(v, voffset + 4, voffset + 3);
            }
            if (LT(v.type_num, GetItem<T>(v, voffset + 3), GetItem<T>(v, voffset + 0)))
            {
                SWAP<T>(v, voffset + 3, voffset + 0);
            }
            if (LT(v.type_num, GetItem<T>(v, voffset + 4), GetItem<T>(v, voffset + 1)))
            {
                SWAP<T>(v, voffset + 4, voffset + 1);
            }
            if (LT(v.type_num, GetItem<T>(v, voffset + 2), GetItem<T>(v, voffset + 1)))
            {
                SWAP<T>(v, voffset + 2, voffset + 1);
            }
            if (LT(v.type_num, GetItem<T>(v, voffset + 3), GetItem<T>(v, voffset + 2)))
            {
                if (LT(v.type_num, GetItem<T>(v, voffset + 3), GetItem<T>(v, voffset + 1)))
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

        /* select index of median of five elements */
        static npy_intp MEDIAN5<T>(VoidPtr v, VoidPtr tosort, npy_intp voffset)
        {
            /* could be optimized as we only need the index (no swaps) */
            if (LT(v.type_num, GetItem<T>(v, IDX(tosort, voffset + 1)), GetItem<T>(v, IDX(tosort, voffset + 0))))
            {
                SWAP<npy_intp>(tosort, voffset + 1, voffset + 0);
            }
            if (LT(v.type_num, GetItem<T>(v, IDX(tosort, voffset + 4)), GetItem<T>(v, IDX(tosort, voffset + 3))))
            {
                SWAP<npy_intp>(tosort, voffset + 4, voffset + 3);
            }
            if (LT(v.type_num, GetItem<T>(v, IDX(tosort, voffset + 3)), GetItem<T>(v, IDX(tosort, voffset + 0))))
            {
                SWAP<npy_intp>(tosort, voffset + 3, voffset + 0);
            }
            if (LT(v.type_num, GetItem<T>(v, IDX(tosort, voffset + 4)), GetItem<T>(v, IDX(tosort, voffset + 1))))
            {
                SWAP<npy_intp>(tosort, voffset + 4, voffset + 1);
            }
            if (LT(v.type_num, GetItem<T>(v, IDX(tosort, voffset + 2)), GetItem<T>(v, IDX(tosort, voffset + 1))))
            {
                SWAP<npy_intp>(tosort, voffset + 2, voffset + 1);
            }
            if (LT(v.type_num, GetItem<T>(v, IDX(tosort, voffset + 3)), GetItem<T>(v, IDX(tosort, voffset + 2))))
            {
                if (LT(v.type_num, GetItem<T>(v, IDX(tosort, voffset + 3)), GetItem<T>(v, IDX(tosort, voffset + 1))))
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
        static npy_intp median_of_median5<T>(VoidPtr v, npy_intp voffset, npy_intp num, npy_intp[] pivots, npy_intp? npiv, bool inexact)
        {
            npy_intp i, subleft;
            npy_intp right = num - 1;
            npy_intp nmed = (right + 1) / 5;

            for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                npy_intp m = MEDIAN5<T>(v, subleft);
                SWAP<T>(v, voffset + subleft + m, voffset + i);
            }

            if (nmed > 2)
                partition_introselect<T>(v, nmed, nmed / 2, pivots, ref npiv, inexact);

            return nmed / 2;
        }

        static npy_intp median_of_median5<T>(VoidPtr v, VoidPtr tosort, npy_intp voffset, npy_intp num, npy_intp[] pivots, npy_intp? npiv, bool inexact)
        {
            npy_intp i, subleft;
            npy_intp right = num - 1;
            npy_intp nmed = (right + 1) / 5;

            for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                npy_intp m = MEDIAN5<T>(v, tosort, subleft);
                SWAP<npy_intp>(tosort, voffset + subleft + m, voffset + i);
            }

            if (nmed > 2)
                argpartition_introselect<T>(v, tosort, nmed, nmed / 2, pivots, ref npiv, inexact);

            return nmed / 2;
        }


        /*
         * partition and return the index were the pivot belongs
         * the data must have following property to avoid bound checks:
         *                  ll ... hh
         * lower-than-pivot [x x x x] larger-than-pivot
         */
        static void UNGUARDED_PARTITION<T>(VoidPtr v, T pivot, ref npy_intp ll, ref npy_intp hh)
        {
            for (; ; )
            {
                do ll++; while (LT(v.type_num, GetItem<T>(v, ll), pivot));
                do hh--; while (LT(v.type_num, pivot, GetItem<T>(v, hh)));

                if (hh < ll)
                    break;

                SWAP<T>(v, ll, hh);
            }
        }

        static void UNGUARDED_PARTITION<T>(VoidPtr v, VoidPtr tosort, T pivot, ref npy_intp ll, ref npy_intp hh)
        {
            for (; ; )
            {
                do ll++; while (LT(v.type_num, GetItem<T>(v, IDX(tosort, ll)), pivot));
                do hh--; while (LT(v.type_num, pivot, GetItem<T>(v, IDX(tosort, hh))));

                if (hh < ll)
                    break;

                SWAP<npy_intp>(tosort, ll, hh);
            }
        }

#if NPY_INTP_64
        static int npy_get_msb(npy_intp unum)
        {
            return npy_get_msb(Convert.ToUInt64(unum));
        }
#else
        static int npy_get_msb(npy_intp unum)
        {
            return npy_get_msb(Convert.ToUInt32(unum));
        }
#endif

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
