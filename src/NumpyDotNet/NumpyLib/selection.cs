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
using System.Runtime.CompilerServices;
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
    internal partial class numpyinternal
    {
        /*
        *****************************************************************************
        **                            NUMERIC SORTS                                **
        *****************************************************************************
        */

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
                    return new partition_bool().partition_introselect(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_BYTE:
                    return new partition_sbyte().partition_introselect(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UBYTE:
                    return new partition_ubyte().partition_introselect(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_INT16:
                    return new partition_Int16().partition_introselect(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UINT16:
                    return new partition_UInt16().partition_introselect(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_INT32:
                    return new partition_Int32Fast().partition_introselect(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UINT32:
                    return new partition_UInt32().partition_introselect(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_INT64:
                    return new partition_Int64Fast().partition_introselect(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UINT64:
                    return new partition_UInt64().partition_introselect(v, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_FLOAT:
                    return new partition_FloatFast().partition_introselect(v, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_DOUBLE:
                    return new partition_DoubleFast().partition_introselect(v, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_DECIMAL:
                    return new partition_Decimal().partition_introselect(v, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_COMPLEX:
                    return new partition_Complex().partition_introselect(v, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_BIGINT:
                    return new partition_BigInt().partition_introselect(v, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_OBJECT:
                    return new partition_Object().partition_introselect(v, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_STRING:
                    return new partition_String().partition_introselect(v, num, kth, pivots, ref npiv, true);
            }
            return 0;
        }


        private static int Common_ArgPartitionFunc(VoidPtr v, VoidPtr tosort, npy_intp num, npy_intp kth, npy_intp[] pivots, ref npy_intp? npiv, object not_used)
        {
            switch (v.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    return new partition_bool().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_BYTE:
                    return new partition_sbyte().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UBYTE:
                    return new partition_ubyte().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_INT16:
                    return new partition_Int16().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UINT16:
                    return new partition_UInt16().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_INT32:
                    return new partition_Int32Fast().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UINT32:
                    return new partition_UInt32().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_INT64:
                    return new partition_Int64Fast().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_UINT64:
                    return new partition_UInt64().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, false);
                case NPY_TYPES.NPY_FLOAT:
                    return new partition_FloatFast().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_DOUBLE:
                    return new partition_DoubleFast().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_DECIMAL:
                    return new partition_Decimal().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_COMPLEX:
                    return new partition_Complex().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_BIGINT:
                    return new partition_BigInt().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_OBJECT:
                    return new partition_Object().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, true);
                case NPY_TYPES.NPY_STRING:
                    return new partition_String().argpartition_introselect(v, tosort, num, kth, pivots, ref npiv, true);

            }
            return 0;
        }

    }

    internal class partition_bool : partition<bool>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(bool);
        }

        public override bool LT(bool a, bool b)
        {
            if (a == b)
                return false;
            if (a == false)
                return true;
            return false;
        }

    }

    internal class partition_ubyte : partition<byte>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(byte);
        }

        public override bool LT(byte a, byte b)
        {
            return a < b;
        }

    }

    internal class partition_sbyte : partition<sbyte>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(sbyte);
        }

        public override bool LT(sbyte a, sbyte b)
        {
            return a < b;
        }

    }

    internal class partition_Int16 : partition<Int16>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(Int16);
        }

        public override bool LT(Int16 a, Int16 b)
        {
            return a < b;
        }

    }

    internal class partition_UInt16 : partition<UInt16>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(UInt16);
        }

        public override bool LT(UInt16 a, UInt16 b)
        {
            return a < b;
        }

    }

    internal class partition_Int32 : partition<Int32>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(Int32);
        }

        public override bool LT(Int32 a, Int32 b)
        {
            return a < b;
        }

    }

    internal class partition_UInt32 : partition<UInt32>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(UInt32);
        }

        public override bool LT(UInt32 a, UInt32 b)
        {
            return a < b;
        }

    }

    internal class partition_Int64 : partition<Int64>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(Int64);
        }

        public override bool LT(Int64 a, Int64 b)
        {
            return a < b;
        }

    }

    internal class partition_UInt64 : partition<UInt64>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(UInt64);
        }

        public override bool LT(UInt64 a, UInt64 b)
        {
            return a < b;
        }

    }

    internal class partition_Float : partition<float>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(float);
        }

        public override bool LT(float a, float b)
        {
            if (float.IsNaN(a))
                return false;
            if (float.IsNaN(b))
                return true;
     
            return a < b;
        }

    }

    internal class partition_Double : partition<double>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(double);
        }

        public override bool LT(double a, double b)
        {
            if (double.IsNaN(a))
                return false;
            if (double.IsNaN(b))
                return true;

            return a < b;
        }

    }

    internal class partition_Decimal : partition<decimal>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(double) * 2;
        }

        public override bool LT(decimal a, decimal b)
        {
            return a < b;
        }

    }

    internal class partition_Complex : partition<System.Numerics.Complex>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(decimal);
        }

        public override bool LT(System.Numerics.Complex a, System.Numerics.Complex b)
        {
            var CompareResult = a.Real.CompareTo(b.Real);
            return CompareResult < 0;
        }

    }

    internal class partition_BigInt : partition<System.Numerics.BigInteger>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return sizeof(double) * 4;
        }

        public override bool LT(System.Numerics.BigInteger a, System.Numerics.BigInteger b)
        {
            return a < b;
        }

    }

    internal class partition_Object : partition<System.Object>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return IntPtr.Size;
        }

        public override bool LT(dynamic invalue, dynamic comparevalue)
        {
            if (invalue is IComparable && comparevalue is IComparable)
            {
                if (invalue == comparevalue)
                    return false;
                if (invalue < comparevalue)
                    return true;
                return false;
            }

            return false;
        }

    }

    internal class partition_String : partition<System.String>
    {
        public override int GetTypeSize(VoidPtr v)
        {
            return IntPtr.Size;
        }

        public override bool LT(string invalue, string comparevalue)
        {
            if (invalue == null)
            {
                if (comparevalue == null)
                {
                    return false;
                }
                return true;
            }

            var c = string.Compare(invalue.ToString(), comparevalue.ToString());
            return c < 0;
        }

    }

    internal abstract class partition<T>
    {
        public abstract int GetTypeSize(VoidPtr v);
        public abstract bool LT(T a, T b);

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

            T[] vv;
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

                vv = v.datap as T[];
                T maxval = vv[v.data_offset + low];
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(vv[v.data_offset + k], maxval))
                    {
                        maxidx = k;
                        maxval = vv[v.data_offset + k];
                    }
                }
                SWAP(vv, v.data_offset, kth, maxidx);
                return 0;
            }

            depth_limit = npy_get_msb(num) * 2;

            /* guarantee three elements */

            vv = v.datap as T[];
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
                    SWAP(vv, v.data_offset, mid, low);
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
                UNGUARDED_PARTITION(v, vv[v.data_offset + low], ref ll, ref hh);

                /* move pivot into position */
                SWAP(vv,v.data_offset, low, hh);

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
                if (LT(vv[v.data_offset + high], vv[v.data_offset + low]))
                {
                    SWAP(vv, v.data_offset, high, low);
                }
            }
            store_pivot(kth, kth, pivots, ref npiv);

            return 0;
        }


        public int argpartition_introselect(VoidPtr v, VoidPtr _tosortvp,
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
            tosortvp.data_offset /= sizeof(npy_intp);

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

            T[] vv;

            if (kth - low < 3)
            {
                DUMBSELECT(v, tosortvp, low, high - low + 1, kth - low);
                store_pivot(kth, kth, pivots, ref npiv);
                return 0;
            }
            else if (inexact && kth == num - 1)
            {
                /* useful to check if NaN present via partition(d, (x, -1)) */
                npy_intp k;
                npy_intp maxidx = low;

                vv = v.datap as T[];

                T maxval = vv[v.data_offset + IDX(tosortvp, low)];
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(vv[v.data_offset + IDX(tosortvp, k)], maxval))
                    {
                        maxidx = k;
                        maxval = vv[v.data_offset + IDX(tosortvp, k)];
                    }
                }
                SWAP_IDX(tosortvp, kth, maxidx);
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
                    MEDIAN3_SWAP(v, tosortvp, low, mid, high);
                }
                else
                {
                    npy_intp mid;
                    mid = ll + median_of_median5(v, tosortvp, ll, hh - ll, null, null, inexact);
                    SWAP_IDX(tosortvp, mid, low);
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
                vv = v.datap as T[];

                UNGUARDED_PARTITION(v, tosortvp, vv[v.data_offset + IDX(tosortvp, low)], ref ll, ref hh);

                /* move pivot into position */
                SWAP_IDX(tosortvp, low, hh);

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
                vv = v.datap as T[];

                if (LT(vv[v.data_offset + IDX(tosortvp, high)], vv[v.data_offset + IDX(tosortvp, low)]))
                {
                    SWAP_IDX(tosortvp, high, low);
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
            T[] vv = v.datap as T[];

            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                T minval = vv[v.data_offset + (i + left)];
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(vv[v.data_offset + (k + left)], minval))
                    {
                        minidx = k;
                        minval = vv[v.data_offset + (k + left)];
                    }
                }
                SWAP(vv, v.data_offset, i + left, minidx + left);
            }

            return 0;
        }
        int DUMBSELECT(VoidPtr v, VoidPtr tosort, npy_intp left, npy_intp num, npy_intp kth)
        {
            T[] vv = v.datap as T[];

            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                T minval = vv[v.data_offset + IDX(tosort, i + left)];
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(vv[v.data_offset + IDX(tosort, k + left)], minval))
                    {
                        minidx = k;
                        minval = vv[v.data_offset + IDX(tosort, k + left)];
                    }
                }
                SWAP_IDX(tosort, i + left, minidx + left);
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

        public void SWAP(T[] vv, npy_intp data_offset, npy_intp aindex, npy_intp bindex)
        {
            T tmp = vv[data_offset + bindex];
            vv[data_offset + bindex] = vv[data_offset + aindex];
            vv[data_offset + aindex] = tmp;
        }

        /*
        * median of 3 pivot strategy
        * gets min and median and moves median to low and min to low + 1
        * for efficient partitioning, see unguarded_partition
        */
        void MEDIAN3_SWAP(VoidPtr v, npy_intp low, npy_intp mid, npy_intp high)
        {
            T[] vv = v.datap as T[];

            if (LT(vv[v.data_offset + high], vv[v.data_offset + mid]))
                SWAP(vv,v.data_offset, high, mid);
            if (LT(vv[v.data_offset + high], vv[v.data_offset + low]))
                SWAP(vv, v.data_offset, high, low);
            /* move pivot to low */
            if (LT(vv[v.data_offset + low], vv[v.data_offset + mid]))
                SWAP(vv, v.data_offset, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP(vv, v.data_offset, mid, low + 1);
        }

        void MEDIAN3_SWAP(VoidPtr v, VoidPtr tosort, npy_intp low, npy_intp mid, npy_intp high)
        {
            T[] vv = v.datap as T[];

            if (LT(vv[v.data_offset + IDX(tosort, high)], vv[v.data_offset + IDX(tosort, mid)]))
                SWAP_IDX(tosort, high, mid);
            if (LT(vv[v.data_offset + IDX(tosort, high)], vv[v.data_offset + IDX(tosort, low)]))
                SWAP_IDX(tosort, high, low);
            /* move pivot to low */
            if (LT(vv[v.data_offset + IDX(tosort, low)], vv[v.data_offset + IDX(tosort, mid)]))
                SWAP_IDX(tosort, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP_IDX(tosort, mid, low + 1);
        }

        static void SWAP_IDX(VoidPtr v, npy_intp a, npy_intp b)
        {
            npy_intp[] vv = v.datap as npy_intp[];
            npy_intp tmp = vv[v.data_offset + b];

            vv[v.data_offset + b] = vv[v.data_offset + a];
            vv[v.data_offset + a] = tmp;
        }

        /* select index of median of five elements */
        npy_intp MEDIAN5(VoidPtr v, npy_intp voffset)
        {
            T[] vv = v.datap as T[];

            /* could be optimized as we only need the index (no swaps) */
            if (LT(vv[v.data_offset + voffset+1], vv[v.data_offset + voffset+0]))
            {
                SWAP(vv, v.data_offset, voffset + 1, voffset + 0);
            }
            if (LT(vv[v.data_offset + voffset + 4], vv[v.data_offset + voffset + 3]))
            {
                SWAP(vv, v.data_offset, voffset + 4, voffset + 3);
            }
            if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 0]))
            {
                SWAP(vv, v.data_offset, voffset + 3, voffset + 0);
            }
            if (LT(vv[v.data_offset + voffset + 4], vv[v.data_offset + voffset + 1]))
            {
                SWAP(vv, v.data_offset, voffset + 4, voffset + 1);
            }
            if (LT(vv[v.data_offset + voffset + 2], vv[v.data_offset + voffset + 1]))
            {
                SWAP(vv, v.data_offset, voffset + 2, voffset + 1);
            }
            if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 2]))
            {
                if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 1]))
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
            T[] vv = v.datap as T[];

            /* could be optimized as we only need the index (no swaps) */
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 1)], vv[v.data_offset + IDX(tosort, voffset + 0)]))
            {
                SWAP_IDX(tosort, voffset + 1, voffset + 0);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 4)], vv[v.data_offset + IDX(tosort, voffset + 3)]))
            {
                SWAP_IDX(tosort, voffset + 4, voffset + 3);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 0)]))
            {
                SWAP_IDX(tosort, voffset + 3, voffset + 0);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 4)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
            {
                SWAP_IDX(tosort, voffset + 4, voffset + 1);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 2)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
            {
                SWAP_IDX(tosort, voffset + 2, voffset + 1);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 2)]))
            {
                if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
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

            T[] vv = v.datap as T[];

            for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                npy_intp m = MEDIAN5(v, subleft);
                SWAP(vv, v.data_offset, voffset + subleft + m, voffset + i);
            }

            if (nmed > 2)
                partition_introselect(v, nmed, nmed / 2, pivots, ref npiv, inexact);

            return nmed / 2;
        }

        npy_intp median_of_median5(VoidPtr v, VoidPtr tosort, npy_intp voffset, npy_intp num, npy_intp[] pivots, npy_intp? npiv, bool inexact)
        {
            npy_intp i, subleft;
            npy_intp right = num - 1;
            npy_intp nmed = (right + 1) / 5;

            for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                npy_intp m = MEDIAN5(v, tosort, subleft);
                SWAP_IDX(tosort, voffset + subleft + m, voffset + i);
            }

            if (nmed > 2)
                argpartition_introselect(v, tosort, nmed, nmed / 2, pivots, ref npiv, inexact);

            return nmed / 2;
        }

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
            T[] vv = v.datap as T[];

            for (; ; )
            {
                do ll++; while (LT(vv[v.data_offset + ll], pivot));
                do hh--; while (LT(pivot, vv[v.data_offset + hh]));

                if (hh < ll)
                    break;

                SWAP(vv, v.data_offset, ll, hh);
            }
        }

        void UNGUARDED_PARTITION(VoidPtr v, VoidPtr tosort, T pivot, ref npy_intp ll, ref npy_intp hh)
        {
            T[] vv = v.datap as T[];

            for (; ; )
            {
                do ll++; while (LT(vv[v.data_offset + IDX(tosort, ll)], pivot));
                do hh--; while (LT(pivot, vv[v.data_offset + IDX(tosort, hh)]));

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

    #region data type specific duplication of the partition<T> class for much better performance
    internal class partition_DoubleFast
    {
        public int GetTypeSize(VoidPtr v)
        {
            return sizeof(double);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool LT(double a, double b)
        {
            if (double.IsNaN(a))
                return false;
            if (double.IsNaN(b))
                return true;

            return a < b;
        }

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

            double[] vv;
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

                vv = v.datap as double[];
                double maxval = vv[v.data_offset + low];
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(vv[v.data_offset + k], maxval))
                    {
                        maxidx = k;
                        maxval = vv[v.data_offset + k];
                    }
                }
                SWAP(vv, v.data_offset, kth, maxidx);
                return 0;
            }

            depth_limit = npy_get_msb(num) * 2;

            /* guarantee three elements */

            vv = v.datap as double[];
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
                    SWAP(vv, v.data_offset, mid, low);
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
                UNGUARDED_PARTITION(v, vv[v.data_offset + low], ref ll, ref hh);

                /* move pivot into position */
                SWAP(vv, v.data_offset, low, hh);

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
                if (LT(vv[v.data_offset + high], vv[v.data_offset + low]))
                {
                    SWAP(vv, v.data_offset, high, low);
                }
            }
            store_pivot(kth, kth, pivots, ref npiv);

            return 0;
        }


        public int argpartition_introselect(VoidPtr v, VoidPtr _tosortvp,
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
            tosortvp.data_offset /= sizeof(npy_intp);

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

            double[] vv;

            if (kth - low < 3)
            {
                DUMBSELECT(v, tosortvp, low, high - low + 1, kth - low);
                store_pivot(kth, kth, pivots, ref npiv);
                return 0;
            }
            else if (inexact && kth == num - 1)
            {
                /* useful to check if NaN present via partition(d, (x, -1)) */
                npy_intp k;
                npy_intp maxidx = low;

                vv = v.datap as double[];

                double maxval = vv[v.data_offset + IDX(tosortvp, low)];
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(vv[v.data_offset + IDX(tosortvp, k)], maxval))
                    {
                        maxidx = k;
                        maxval = vv[v.data_offset + IDX(tosortvp, k)];
                    }
                }
                SWAP_IDX(tosortvp, kth, maxidx);
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
                    MEDIAN3_SWAP(v, tosortvp, low, mid, high);
                }
                else
                {
                    npy_intp mid;
                    mid = ll + median_of_median5(v, tosortvp, ll, hh - ll, null, null, inexact);
                    SWAP_IDX(tosortvp, mid, low);
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
                vv = v.datap as double[];

                UNGUARDED_PARTITION(v, tosortvp, vv[v.data_offset + IDX(tosortvp, low)], ref ll, ref hh);

                /* move pivot into position */
                SWAP_IDX(tosortvp, low, hh);

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
                vv = v.datap as double[];

                if (LT(vv[v.data_offset + IDX(tosortvp, high)], vv[v.data_offset + IDX(tosortvp, low)]))
                {
                    SWAP_IDX(tosortvp, high, low);
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
            double[] vv = v.datap as double[];

            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                double minval = vv[v.data_offset + (i + left)];
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(vv[v.data_offset + (k + left)], minval))
                    {
                        minidx = k;
                        minval = vv[v.data_offset + (k + left)];
                    }
                }
                SWAP(vv, v.data_offset, i + left, minidx + left);
            }

            return 0;
        }

        int DUMBSELECT(VoidPtr v, VoidPtr tosort, npy_intp left, npy_intp num, npy_intp kth)
        {
            double[] vv = v.datap as double[];

            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                double minval = vv[v.data_offset + IDX(tosort, i + left)];
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(vv[v.data_offset + IDX(tosort, k + left)], minval))
                    {
                        minidx = k;
                        minval = vv[v.data_offset + IDX(tosort, k + left)];
                    }
                }
                SWAP_IDX(tosort, i + left, minidx + left);
            }

            return 0;
        }
        public void SWAP(VoidPtr v, npy_intp aindex, npy_intp bindex)
        {
            var vv = v.datap as double[];

            double tmp = vv[v.data_offset + bindex];
            vv[v.data_offset + bindex] = vv[v.data_offset + aindex];
            vv[v.data_offset + aindex] = tmp;
        }
        public void SWAP(double[] vv, npy_intp data_offset, npy_intp aindex, npy_intp bindex)
        {
            double tmp = vv[data_offset + bindex];
            vv[data_offset + bindex] = vv[data_offset + aindex];
            vv[data_offset + aindex] = tmp;
        }

        /*
        * median of 3 pivot strategy
        * gets min and median and moves median to low and min to low + 1
        * for efficient partitioning, see unguarded_partition
        */
        void MEDIAN3_SWAP(VoidPtr v, npy_intp low, npy_intp mid, npy_intp high)
        {
            double[] vv = v.datap as double[];

            if (LT(vv[v.data_offset + high], vv[v.data_offset + mid]))
                SWAP(vv, v.data_offset, high, mid);
            if (LT(vv[v.data_offset + high], vv[v.data_offset + low]))
                SWAP(vv, v.data_offset, high, low);
            /* move pivot to low */
            if (LT(vv[v.data_offset + low], vv[v.data_offset + mid]))
                SWAP(vv, v.data_offset, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP(vv, v.data_offset, mid, low + 1);
        }
        void MEDIAN3_SWAP(VoidPtr v, VoidPtr tosort, npy_intp low, npy_intp mid, npy_intp high)
        {
            double[] vv = v.datap as double[];

            if (LT(vv[v.data_offset + IDX(tosort, high)], vv[v.data_offset + IDX(tosort, mid)]))
                SWAP_IDX(tosort, high, mid);
            if (LT(vv[v.data_offset + IDX(tosort, high)], vv[v.data_offset + IDX(tosort, low)]))
                SWAP_IDX(tosort, high, low);
            /* move pivot to low */
            if (LT(vv[v.data_offset + IDX(tosort, low)], vv[v.data_offset + IDX(tosort, mid)]))
                SWAP_IDX(tosort, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP_IDX(tosort, mid, low + 1);
        }
        static void SWAP_IDX(VoidPtr v, npy_intp a, npy_intp b)
        {
            npy_intp[] vv = v.datap as npy_intp[];
            npy_intp tmp = vv[v.data_offset + b];

            vv[v.data_offset + b] = vv[v.data_offset + a];
            vv[v.data_offset + a] = tmp;
        }

        /* select index of median of five elements */
        npy_intp MEDIAN5(VoidPtr v, npy_intp voffset)
        {
            double[] vv = v.datap as double[];

            /* could be optimized as we only need the index (no swaps) */
            if (LT(vv[v.data_offset + voffset + 1], vv[v.data_offset + voffset + 0]))
            {
                SWAP(vv, v.data_offset, voffset + 1, voffset + 0);
            }
            if (LT(vv[v.data_offset + voffset + 4], vv[v.data_offset + voffset + 3]))
            {
                SWAP(vv, v.data_offset, voffset + 4, voffset + 3);
            }
            if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 0]))
            {
                SWAP(vv, v.data_offset, voffset + 3, voffset + 0);
            }
            if (LT(vv[v.data_offset + voffset + 4], vv[v.data_offset + voffset + 1]))
            {
                SWAP(vv, v.data_offset, voffset + 4, voffset + 1);
            }
            if (LT(vv[v.data_offset + voffset + 2], vv[v.data_offset + voffset + 1]))
            {
                SWAP(vv, v.data_offset, voffset + 2, voffset + 1);
            }
            if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 2]))
            {
                if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 1]))
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
            double[] vv = v.datap as double[];

            /* could be optimized as we only need the index (no swaps) */
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 1)], vv[v.data_offset + IDX(tosort, voffset + 0)]))
            {
                SWAP_IDX(tosort, voffset + 1, voffset + 0);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 4)], vv[v.data_offset + IDX(tosort, voffset + 3)]))
            {
                SWAP_IDX(tosort, voffset + 4, voffset + 3);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 0)]))
            {
                SWAP_IDX(tosort, voffset + 3, voffset + 0);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 4)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
            {
                SWAP_IDX(tosort, voffset + 4, voffset + 1);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 2)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
            {
                SWAP_IDX(tosort, voffset + 2, voffset + 1);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 2)]))
            {
                if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
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

            double[] vv = v.datap as double[];

            for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                npy_intp m = MEDIAN5(v, subleft);
                SWAP(vv, v.data_offset, voffset + subleft + m, voffset + i);
            }

            if (nmed > 2)
                partition_introselect(v, nmed, nmed / 2, pivots, ref npiv, inexact);

            return nmed / 2;
        }

        npy_intp median_of_median5(VoidPtr v, VoidPtr tosort, npy_intp voffset, npy_intp num, npy_intp[] pivots, npy_intp? npiv, bool inexact)
        {
            npy_intp i, subleft;
            npy_intp right = num - 1;
            npy_intp nmed = (right + 1) / 5;

            for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                npy_intp m = MEDIAN5(v, tosort, subleft);
                SWAP_IDX(tosort, voffset + subleft + m, voffset + i);
            }

            if (nmed > 2)
                argpartition_introselect(v, tosort, nmed, nmed / 2, pivots, ref npiv, inexact);

            return nmed / 2;
        }

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

        private static double GetItem(VoidPtr v, npy_intp index)
        {
            double[] vv = v.datap as double[];
            return vv[v.data_offset + index];
        }
        private static double SetItem(VoidPtr v, npy_intp index, double d)
        {
            double[] vv = v.datap as double[];
            return vv[v.data_offset + index] = d;
        }

        /*
     * partition and return the index were the pivot belongs
     * the data must have following property to avoid bound checks:
     *                  ll ... hh
     * lower-than-pivot [x x x x] larger-than-pivot
     */
        void UNGUARDED_PARTITION(VoidPtr v, double pivot, ref npy_intp ll, ref npy_intp hh)
        {
            double[] vv = v.datap as double[];

            for (; ; )
            {
                do ll++; while (LT(vv[v.data_offset + ll], pivot));
                do hh--; while (LT(pivot, vv[v.data_offset + hh]));

                if (hh < ll)
                    break;

                SWAP(vv, v.data_offset, ll, hh);
            }
        }

        void UNGUARDED_PARTITION(VoidPtr v, VoidPtr tosort, double pivot, ref npy_intp ll, ref npy_intp hh)
        {
            double[] vv = v.datap as double[];

            for (; ; )
            {
                do ll++; while (LT(vv[v.data_offset + IDX(tosort, ll)], pivot));
                do hh--; while (LT(pivot, vv[v.data_offset + IDX(tosort, hh)]));

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

    internal class partition_FloatFast
    {
        public int GetTypeSize(VoidPtr v)
        {
            return sizeof(float);
        }

        public bool LT(float a, float b)
        {
            if (float.IsNaN(a))
                return false;
            if (float.IsNaN(b))
                return true;

            return a < b;
        }

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

            float[] vv;
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

                vv = v.datap as float[];
                float maxval = vv[v.data_offset + low];
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(vv[v.data_offset + k], maxval))
                    {
                        maxidx = k;
                        maxval = vv[v.data_offset + k];
                    }
                }
                SWAP(vv, v.data_offset, kth, maxidx);
                return 0;
            }

            depth_limit = npy_get_msb(num) * 2;

            /* guarantee three elements */

            vv = v.datap as float[];
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
                    SWAP(vv, v.data_offset, mid, low);
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
                UNGUARDED_PARTITION(v, vv[v.data_offset + low], ref ll, ref hh);

                /* move pivot into position */
                SWAP(vv, v.data_offset, low, hh);

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
                if (LT(vv[v.data_offset + high], vv[v.data_offset + low]))
                {
                    SWAP(vv, v.data_offset, high, low);
                }
            }
            store_pivot(kth, kth, pivots, ref npiv);

            return 0;
        }


        public int argpartition_introselect(VoidPtr v, VoidPtr _tosortvp,
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
            tosortvp.data_offset /= sizeof(npy_intp);

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

            float[] vv;

            if (kth - low < 3)
            {
                DUMBSELECT(v, tosortvp, low, high - low + 1, kth - low);
                store_pivot(kth, kth, pivots, ref npiv);
                return 0;
            }
            else if (inexact && kth == num - 1)
            {
                /* useful to check if NaN present via partition(d, (x, -1)) */
                npy_intp k;
                npy_intp maxidx = low;

                vv = v.datap as float[];

                float maxval = vv[v.data_offset + IDX(tosortvp, low)];
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(vv[v.data_offset + IDX(tosortvp, k)], maxval))
                    {
                        maxidx = k;
                        maxval = vv[v.data_offset + IDX(tosortvp, k)];
                    }
                }
                SWAP_IDX(tosortvp, kth, maxidx);
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
                    MEDIAN3_SWAP(v, tosortvp, low, mid, high);
                }
                else
                {
                    npy_intp mid;
                    mid = ll + median_of_median5(v, tosortvp, ll, hh - ll, null, null, inexact);
                    SWAP_IDX(tosortvp, mid, low);
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
                vv = v.datap as float[];

                UNGUARDED_PARTITION(v, tosortvp, vv[v.data_offset + IDX(tosortvp, low)], ref ll, ref hh);

                /* move pivot into position */
                SWAP_IDX(tosortvp, low, hh);

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
                vv = v.datap as float[];

                if (LT(vv[v.data_offset + IDX(tosortvp, high)], vv[v.data_offset + IDX(tosortvp, low)]))
                {
                    SWAP_IDX(tosortvp, high, low);
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
            float[] vv = v.datap as float[];

            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                float minval = vv[v.data_offset + (i + left)];
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(vv[v.data_offset + (k + left)], minval))
                    {
                        minidx = k;
                        minval = vv[v.data_offset + (k + left)];
                    }
                }
                SWAP(vv, v.data_offset, i + left, minidx + left);
            }

            return 0;
        }
        int DUMBSELECT(VoidPtr v, VoidPtr tosort, npy_intp left, npy_intp num, npy_intp kth)
        {
            float[] vv = v.datap as float[];

            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                float minval = vv[v.data_offset + IDX(tosort, i + left)];
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(vv[v.data_offset + IDX(tosort, k + left)], minval))
                    {
                        minidx = k;
                        minval = vv[v.data_offset + IDX(tosort, k + left)];
                    }
                }
                SWAP_IDX(tosort, i + left, minidx + left);
            }

            return 0;
        }

        public void SWAP(VoidPtr v, npy_intp aindex, npy_intp bindex)
        {
            var vv = v.datap as float[];

            float tmp = vv[v.data_offset + bindex];
            vv[v.data_offset + bindex] = vv[v.data_offset + aindex];
            vv[v.data_offset + aindex] = tmp;
        }

        public void SWAP(float[] vv, npy_intp data_offset, npy_intp aindex, npy_intp bindex)
        {
            float tmp = vv[data_offset + bindex];
            vv[data_offset + bindex] = vv[data_offset + aindex];
            vv[data_offset + aindex] = tmp;
        }

        /*
        * median of 3 pivot strategy
        * gets min and median and moves median to low and min to low + 1
        * for efficient partitioning, see unguarded_partition
        */
        void MEDIAN3_SWAP(VoidPtr v, npy_intp low, npy_intp mid, npy_intp high)
        {
            float[] vv = v.datap as float[];

            if (LT(vv[v.data_offset + high], vv[v.data_offset + mid]))
                SWAP(vv, v.data_offset, high, mid);
            if (LT(vv[v.data_offset + high], vv[v.data_offset + low]))
                SWAP(vv, v.data_offset, high, low);
            /* move pivot to low */
            if (LT(vv[v.data_offset + low], vv[v.data_offset + mid]))
                SWAP(vv, v.data_offset, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP(vv, v.data_offset, mid, low + 1);
        }

        void MEDIAN3_SWAP(VoidPtr v, VoidPtr tosort, npy_intp low, npy_intp mid, npy_intp high)
        {
            float[] vv = v.datap as float[];

            if (LT(vv[v.data_offset + IDX(tosort, high)], vv[v.data_offset + IDX(tosort, mid)]))
                SWAP_IDX(tosort, high, mid);
            if (LT(vv[v.data_offset + IDX(tosort, high)], vv[v.data_offset + IDX(tosort, low)]))
                SWAP_IDX(tosort, high, low);
            /* move pivot to low */
            if (LT(vv[v.data_offset + IDX(tosort, low)], vv[v.data_offset + IDX(tosort, mid)]))
                SWAP_IDX(tosort, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP_IDX(tosort, mid, low + 1);
        }

        static void SWAP_IDX(VoidPtr v, npy_intp a, npy_intp b)
        {
            npy_intp[] vv = v.datap as npy_intp[];
            npy_intp tmp = vv[v.data_offset + b];

            vv[v.data_offset + b] = vv[v.data_offset + a];
            vv[v.data_offset + a] = tmp;
        }

        /* select index of median of five elements */
        npy_intp MEDIAN5(VoidPtr v, npy_intp voffset)
        {
            float[] vv = v.datap as float[];

            /* could be optimized as we only need the index (no swaps) */
            if (LT(vv[v.data_offset + voffset + 1], vv[v.data_offset + voffset + 0]))
            {
                SWAP(vv, v.data_offset, voffset + 1, voffset + 0);
            }
            if (LT(vv[v.data_offset + voffset + 4], vv[v.data_offset + voffset + 3]))
            {
                SWAP(vv, v.data_offset, voffset + 4, voffset + 3);
            }
            if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 0]))
            {
                SWAP(vv, v.data_offset, voffset + 3, voffset + 0);
            }
            if (LT(vv[v.data_offset + voffset + 4], vv[v.data_offset + voffset + 1]))
            {
                SWAP(vv, v.data_offset, voffset + 4, voffset + 1);
            }
            if (LT(vv[v.data_offset + voffset + 2], vv[v.data_offset + voffset + 1]))
            {
                SWAP(vv, v.data_offset, voffset + 2, voffset + 1);
            }
            if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 2]))
            {
                if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 1]))
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
            float[] vv = v.datap as float[];

            /* could be optimized as we only need the index (no swaps) */
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 1)], vv[v.data_offset + IDX(tosort, voffset + 0)]))
            {
                SWAP_IDX(tosort, voffset + 1, voffset + 0);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 4)], vv[v.data_offset + IDX(tosort, voffset + 3)]))
            {
                SWAP_IDX(tosort, voffset + 4, voffset + 3);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 0)]))
            {
                SWAP_IDX(tosort, voffset + 3, voffset + 0);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 4)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
            {
                SWAP_IDX(tosort, voffset + 4, voffset + 1);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 2)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
            {
                SWAP_IDX(tosort, voffset + 2, voffset + 1);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 2)]))
            {
                if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
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

            float[] vv = v.datap as float[];

            for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                npy_intp m = MEDIAN5(v, subleft);
                SWAP(vv, v.data_offset, voffset + subleft + m, voffset + i);
            }

            if (nmed > 2)
                partition_introselect(v, nmed, nmed / 2, pivots, ref npiv, inexact);

            return nmed / 2;
        }

        npy_intp median_of_median5(VoidPtr v, VoidPtr tosort, npy_intp voffset, npy_intp num, npy_intp[] pivots, npy_intp? npiv, bool inexact)
        {
            npy_intp i, subleft;
            npy_intp right = num - 1;
            npy_intp nmed = (right + 1) / 5;

            for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                npy_intp m = MEDIAN5(v, tosort, subleft);
                SWAP_IDX(tosort, voffset + subleft + m, voffset + i);
            }

            if (nmed > 2)
                argpartition_introselect(v, tosort, nmed, nmed / 2, pivots, ref npiv, inexact);

            return nmed / 2;
        }

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

        private static float GetItem(VoidPtr v, npy_intp index)
        {
            float[] vv = v.datap as float[];
            return vv[v.data_offset + index];
        }
        private static float SetItem(VoidPtr v, npy_intp index, float d)
        {
            float[] vv = v.datap as float[];
            return vv[v.data_offset + index] = d;
        }

        /*
     * partition and return the index were the pivot belongs
     * the data must have following property to avoid bound checks:
     *                  ll ... hh
     * lower-than-pivot [x x x x] larger-than-pivot
     */
        void UNGUARDED_PARTITION(VoidPtr v, float pivot, ref npy_intp ll, ref npy_intp hh)
        {
            float[] vv = v.datap as float[];

            for (; ; )
            {
                do ll++; while (LT(vv[v.data_offset + ll], pivot));
                do hh--; while (LT(pivot, vv[v.data_offset + hh]));

                if (hh < ll)
                    break;

                SWAP(vv, v.data_offset, ll, hh);
            }
        }

        void UNGUARDED_PARTITION(VoidPtr v, VoidPtr tosort, float pivot, ref npy_intp ll, ref npy_intp hh)
        {
            float[] vv = v.datap as float[];

            for (; ; )
            {
                do ll++; while (LT(vv[v.data_offset + IDX(tosort, ll)], pivot));
                do hh--; while (LT(pivot, vv[v.data_offset + IDX(tosort, hh)]));

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

    internal class partition_Int32Fast
    {
        public int GetTypeSize(VoidPtr v)
        {
            return sizeof(Int32);
        }

        public bool LT(Int32 a, Int32 b)
        {
            return a < b;
        }

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

            Int32[] vv;
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

                vv = v.datap as Int32[];
                Int32 maxval = vv[v.data_offset + low];
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(vv[v.data_offset + k], maxval))
                    {
                        maxidx = k;
                        maxval = vv[v.data_offset + k];
                    }
                }
                SWAP(vv, v.data_offset, kth, maxidx);
                return 0;
            }

            depth_limit = npy_get_msb(num) * 2;

            /* guarantee three elements */

            vv = v.datap as Int32[];
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
                    SWAP(vv, v.data_offset, mid, low);
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
                UNGUARDED_PARTITION(v, vv[v.data_offset + low], ref ll, ref hh);

                /* move pivot into position */
                SWAP(vv, v.data_offset, low, hh);

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
                if (LT(vv[v.data_offset + high], vv[v.data_offset + low]))
                {
                    SWAP(vv, v.data_offset, high, low);
                }
            }
            store_pivot(kth, kth, pivots, ref npiv);

            return 0;
        }


        public int argpartition_introselect(VoidPtr v, VoidPtr _tosortvp,
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
            tosortvp.data_offset /= sizeof(npy_intp);

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

            Int32[] vv;

            if (kth - low < 3)
            {
                DUMBSELECT(v, tosortvp, low, high - low + 1, kth - low);
                store_pivot(kth, kth, pivots, ref npiv);
                return 0;
            }
            else if (inexact && kth == num - 1)
            {
                /* useful to check if NaN present via partition(d, (x, -1)) */
                npy_intp k;
                npy_intp maxidx = low;

                vv = v.datap as Int32[];

                Int32 maxval = vv[v.data_offset + IDX(tosortvp, low)];
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(vv[v.data_offset + IDX(tosortvp, k)], maxval))
                    {
                        maxidx = k;
                        maxval = vv[v.data_offset + IDX(tosortvp, k)];
                    }
                }
                SWAP_IDX(tosortvp, kth, maxidx);
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
                    MEDIAN3_SWAP(v, tosortvp, low, mid, high);
                }
                else
                {
                    npy_intp mid;
                    mid = ll + median_of_median5(v, tosortvp, ll, hh - ll, null, null, inexact);
                    SWAP_IDX(tosortvp, mid, low);
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
                vv = v.datap as Int32[];

                UNGUARDED_PARTITION(v, tosortvp, vv[v.data_offset + IDX(tosortvp, low)], ref ll, ref hh);

                /* move pivot into position */
                SWAP_IDX(tosortvp, low, hh);

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
                vv = v.datap as Int32[];

                if (LT(vv[v.data_offset + IDX(tosortvp, high)], vv[v.data_offset + IDX(tosortvp, low)]))
                {
                    SWAP_IDX(tosortvp, high, low);
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
            Int32[] vv = v.datap as Int32[];

            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                Int32 minval = vv[v.data_offset + (i + left)];
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(vv[v.data_offset + (k + left)], minval))
                    {
                        minidx = k;
                        minval = vv[v.data_offset + (k + left)];
                    }
                }
                SWAP(vv, v.data_offset, i + left, minidx + left);
            }

            return 0;
        }
        int DUMBSELECT(VoidPtr v, VoidPtr tosort, npy_intp left, npy_intp num, npy_intp kth)
        {
            Int32[] vv = v.datap as Int32[];

            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                Int32 minval = vv[v.data_offset + IDX(tosort, i + left)];
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(vv[v.data_offset + IDX(tosort, k + left)], minval))
                    {
                        minidx = k;
                        minval = vv[v.data_offset + IDX(tosort, k + left)];
                    }
                }
                SWAP_IDX(tosort, i + left, minidx + left);
            }

            return 0;
        }

        public void SWAP(VoidPtr v, npy_intp aindex, npy_intp bindex)
        {
            var vv = v.datap as Int32[];

            Int32 tmp = vv[v.data_offset + bindex];
            vv[v.data_offset + bindex] = vv[v.data_offset + aindex];
            vv[v.data_offset + aindex] = tmp;
        }

        public void SWAP(Int32[] vv, npy_intp data_offset, npy_intp aindex, npy_intp bindex)
        {
            Int32 tmp = vv[data_offset + bindex];
            vv[data_offset + bindex] = vv[data_offset + aindex];
            vv[data_offset + aindex] = tmp;
        }

        /*
        * median of 3 pivot strategy
        * gets min and median and moves median to low and min to low + 1
        * for efficient partitioning, see unguarded_partition
        */
        void MEDIAN3_SWAP(VoidPtr v, npy_intp low, npy_intp mid, npy_intp high)
        {
            Int32[] vv = v.datap as Int32[];

            if (LT(vv[v.data_offset + high], vv[v.data_offset + mid]))
                SWAP(vv, v.data_offset, high, mid);
            if (LT(vv[v.data_offset + high], vv[v.data_offset + low]))
                SWAP(vv, v.data_offset, high, low);
            /* move pivot to low */
            if (LT(vv[v.data_offset + low], vv[v.data_offset + mid]))
                SWAP(vv, v.data_offset, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP(vv, v.data_offset, mid, low + 1);
        }

        void MEDIAN3_SWAP(VoidPtr v, VoidPtr tosort, npy_intp low, npy_intp mid, npy_intp high)
        {
            Int32[] vv = v.datap as Int32[];

            if (LT(vv[v.data_offset + IDX(tosort, high)], vv[v.data_offset + IDX(tosort, mid)]))
                SWAP_IDX(tosort, high, mid);
            if (LT(vv[v.data_offset + IDX(tosort, high)], vv[v.data_offset + IDX(tosort, low)]))
                SWAP_IDX(tosort, high, low);
            /* move pivot to low */
            if (LT(vv[v.data_offset + IDX(tosort, low)], vv[v.data_offset + IDX(tosort, mid)]))
                SWAP_IDX(tosort, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP_IDX(tosort, mid, low + 1);
        }

        static void SWAP_IDX(VoidPtr v, npy_intp a, npy_intp b)
        {
            npy_intp[] vv = v.datap as npy_intp[];
            npy_intp tmp = vv[v.data_offset + b];

            vv[v.data_offset + b] = vv[v.data_offset + a];
            vv[v.data_offset + a] = tmp;
        }

        /* select index of median of five elements */
        npy_intp MEDIAN5(VoidPtr v, npy_intp voffset)
        {
            Int32[] vv = v.datap as Int32[];

            /* could be optimized as we only need the index (no swaps) */
            if (LT(vv[v.data_offset + voffset + 1], vv[v.data_offset + voffset + 0]))
            {
                SWAP(vv, v.data_offset, voffset + 1, voffset + 0);
            }
            if (LT(vv[v.data_offset + voffset + 4], vv[v.data_offset + voffset + 3]))
            {
                SWAP(vv, v.data_offset, voffset + 4, voffset + 3);
            }
            if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 0]))
            {
                SWAP(vv, v.data_offset, voffset + 3, voffset + 0);
            }
            if (LT(vv[v.data_offset + voffset + 4], vv[v.data_offset + voffset + 1]))
            {
                SWAP(vv, v.data_offset, voffset + 4, voffset + 1);
            }
            if (LT(vv[v.data_offset + voffset + 2], vv[v.data_offset + voffset + 1]))
            {
                SWAP(vv, v.data_offset, voffset + 2, voffset + 1);
            }
            if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 2]))
            {
                if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 1]))
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
            Int32[] vv = v.datap as Int32[];

            /* could be optimized as we only need the index (no swaps) */
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 1)], vv[v.data_offset + IDX(tosort, voffset + 0)]))
            {
                SWAP_IDX(tosort, voffset + 1, voffset + 0);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 4)], vv[v.data_offset + IDX(tosort, voffset + 3)]))
            {
                SWAP_IDX(tosort, voffset + 4, voffset + 3);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 0)]))
            {
                SWAP_IDX(tosort, voffset + 3, voffset + 0);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 4)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
            {
                SWAP_IDX(tosort, voffset + 4, voffset + 1);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 2)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
            {
                SWAP_IDX(tosort, voffset + 2, voffset + 1);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 2)]))
            {
                if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
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

            Int32[] vv = v.datap as Int32[];

            for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                npy_intp m = MEDIAN5(v, subleft);
                SWAP(vv, v.data_offset, voffset + subleft + m, voffset + i);
            }

            if (nmed > 2)
                partition_introselect(v, nmed, nmed / 2, pivots, ref npiv, inexact);

            return nmed / 2;
        }

        npy_intp median_of_median5(VoidPtr v, VoidPtr tosort, npy_intp voffset, npy_intp num, npy_intp[] pivots, npy_intp? npiv, bool inexact)
        {
            npy_intp i, subleft;
            npy_intp right = num - 1;
            npy_intp nmed = (right + 1) / 5;

            for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                npy_intp m = MEDIAN5(v, tosort, subleft);
                SWAP_IDX(tosort, voffset + subleft + m, voffset + i);
            }

            if (nmed > 2)
                argpartition_introselect(v, tosort, nmed, nmed / 2, pivots, ref npiv, inexact);

            return nmed / 2;
        }

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

        private static Int32 GetItem(VoidPtr v, npy_intp index)
        {
            Int32[] vv = v.datap as Int32[];
            return vv[v.data_offset + index];
        }
        private static Int32 SetItem(VoidPtr v, npy_intp index, Int32 d)
        {
            Int32[] vv = v.datap as Int32[];
            return vv[v.data_offset + index] = d;
        }

        /*
     * partition and return the index were the pivot belongs
     * the data must have following property to avoid bound checks:
     *                  ll ... hh
     * lower-than-pivot [x x x x] larger-than-pivot
     */
        void UNGUARDED_PARTITION(VoidPtr v, Int32 pivot, ref npy_intp ll, ref npy_intp hh)
        {
            Int32[] vv = v.datap as Int32[];

            for (; ; )
            {
                do ll++; while (LT(vv[v.data_offset + ll], pivot));
                do hh--; while (LT(pivot, vv[v.data_offset + hh]));

                if (hh < ll)
                    break;

                SWAP(vv, v.data_offset, ll, hh);
            }
        }

        void UNGUARDED_PARTITION(VoidPtr v, VoidPtr tosort, Int32 pivot, ref npy_intp ll, ref npy_intp hh)
        {
            Int32[] vv = v.datap as Int32[];

            for (; ; )
            {
                do ll++; while (LT(vv[v.data_offset + IDX(tosort, ll)], pivot));
                do hh--; while (LT(pivot, vv[v.data_offset + IDX(tosort, hh)]));

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

    internal class partition_Int64Fast
    {
        public int GetTypeSize(VoidPtr v)
        {
            return sizeof(Int64);
        }

        public bool LT(Int64 a, Int64 b)
        {
            return a < b;
        }

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

            Int64[] vv;
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

                vv = v.datap as Int64[];
                Int64 maxval = vv[v.data_offset + low];
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(vv[v.data_offset + k], maxval))
                    {
                        maxidx = k;
                        maxval = vv[v.data_offset + k];
                    }
                }
                SWAP(vv, v.data_offset, kth, maxidx);
                return 0;
            }

            depth_limit = npy_get_msb(num) * 2;

            /* guarantee three elements */

            vv = v.datap as Int64[];
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
                    SWAP(vv, v.data_offset, mid, low);
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
                UNGUARDED_PARTITION(v, vv[v.data_offset + low], ref ll, ref hh);

                /* move pivot into position */
                SWAP(vv, v.data_offset, low, hh);

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
                if (LT(vv[v.data_offset + high], vv[v.data_offset + low]))
                {
                    SWAP(vv, v.data_offset, high, low);
                }
            }
            store_pivot(kth, kth, pivots, ref npiv);

            return 0;
        }


        public int argpartition_introselect(VoidPtr v, VoidPtr _tosortvp,
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
            tosortvp.data_offset /= sizeof(npy_intp);

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

            Int64[] vv;

            if (kth - low < 3)
            {
                DUMBSELECT(v, tosortvp, low, high - low + 1, kth - low);
                store_pivot(kth, kth, pivots, ref npiv);
                return 0;
            }
            else if (inexact && kth == num - 1)
            {
                /* useful to check if NaN present via partition(d, (x, -1)) */
                npy_intp k;
                npy_intp maxidx = low;

                vv = v.datap as Int64[];

                Int64 maxval = vv[v.data_offset + IDX(tosortvp, low)];
                for (k = low + 1; k < num; k++)
                {
                    if (!LT(vv[v.data_offset + IDX(tosortvp, k)], maxval))
                    {
                        maxidx = k;
                        maxval = vv[v.data_offset + IDX(tosortvp, k)];
                    }
                }
                SWAP_IDX(tosortvp, kth, maxidx);
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
                    MEDIAN3_SWAP(v, tosortvp, low, mid, high);
                }
                else
                {
                    npy_intp mid;
                    mid = ll + median_of_median5(v, tosortvp, ll, hh - ll, null, null, inexact);
                    SWAP_IDX(tosortvp, mid, low);
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
                vv = v.datap as Int64[];

                UNGUARDED_PARTITION(v, tosortvp, vv[v.data_offset + IDX(tosortvp, low)], ref ll, ref hh);

                /* move pivot into position */
                SWAP_IDX(tosortvp, low, hh);

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
                vv = v.datap as Int64[];

                if (LT(vv[v.data_offset + IDX(tosortvp, high)], vv[v.data_offset + IDX(tosortvp, low)]))
                {
                    SWAP_IDX(tosortvp, high, low);
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
            Int64[] vv = v.datap as Int64[];

            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                Int64 minval = vv[v.data_offset + (i + left)];
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(vv[v.data_offset + (k + left)], minval))
                    {
                        minidx = k;
                        minval = vv[v.data_offset + (k + left)];
                    }
                }
                SWAP(vv, v.data_offset, i + left, minidx + left);
            }

            return 0;
        }
        int DUMBSELECT(VoidPtr v, VoidPtr tosort, npy_intp left, npy_intp num, npy_intp kth)
        {
            Int64[] vv = v.datap as Int64[];

            npy_intp i;
            for (i = 0; i <= kth; i++)
            {
                npy_intp minidx = i;
                Int64 minval = vv[v.data_offset + IDX(tosort, i + left)];
                npy_intp k;
                for (k = i + 1; k < num; k++)
                {
                    if (LT(vv[v.data_offset + IDX(tosort, k + left)], minval))
                    {
                        minidx = k;
                        minval = vv[v.data_offset + IDX(tosort, k + left)];
                    }
                }
                SWAP_IDX(tosort, i + left, minidx + left);
            }

            return 0;
        }

        public void SWAP(VoidPtr v, npy_intp aindex, npy_intp bindex)
        {
            var vv = v.datap as Int64[];

            Int64 tmp = vv[v.data_offset + bindex];
            vv[v.data_offset + bindex] = vv[v.data_offset + aindex];
            vv[v.data_offset + aindex] = tmp;
        }

        public void SWAP(Int64[] vv, npy_intp data_offset, npy_intp aindex, npy_intp bindex)
        {
            Int64 tmp = vv[data_offset + bindex];
            vv[data_offset + bindex] = vv[data_offset + aindex];
            vv[data_offset + aindex] = tmp;
        }

        /*
        * median of 3 pivot strategy
        * gets min and median and moves median to low and min to low + 1
        * for efficient partitioning, see unguarded_partition
        */
        void MEDIAN3_SWAP(VoidPtr v, npy_intp low, npy_intp mid, npy_intp high)
        {
            Int64[] vv = v.datap as Int64[];

            if (LT(vv[v.data_offset + high], vv[v.data_offset + mid]))
                SWAP(vv, v.data_offset, high, mid);
            if (LT(vv[v.data_offset + high], vv[v.data_offset + low]))
                SWAP(vv, v.data_offset, high, low);
            /* move pivot to low */
            if (LT(vv[v.data_offset + low], vv[v.data_offset + mid]))
                SWAP(vv, v.data_offset, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP(vv, v.data_offset, mid, low + 1);
        }

        void MEDIAN3_SWAP(VoidPtr v, VoidPtr tosort, npy_intp low, npy_intp mid, npy_intp high)
        {
            Int64[] vv = v.datap as Int64[];

            if (LT(vv[v.data_offset + IDX(tosort, high)], vv[v.data_offset + IDX(tosort, mid)]))
                SWAP_IDX(tosort, high, mid);
            if (LT(vv[v.data_offset + IDX(tosort, high)], vv[v.data_offset + IDX(tosort, low)]))
                SWAP_IDX(tosort, high, low);
            /* move pivot to low */
            if (LT(vv[v.data_offset + IDX(tosort, low)], vv[v.data_offset + IDX(tosort, mid)]))
                SWAP_IDX(tosort, low, mid);
            /* move 3-lowest element to low + 1 */
            SWAP_IDX(tosort, mid, low + 1);
        }

        static void SWAP_IDX(VoidPtr v, npy_intp a, npy_intp b)
        {
            npy_intp[] vv = v.datap as npy_intp[];
            npy_intp tmp = vv[v.data_offset + b];

            vv[v.data_offset + b] = vv[v.data_offset + a];
            vv[v.data_offset + a] = tmp;
        }

        /* select index of median of five elements */
        npy_intp MEDIAN5(VoidPtr v, npy_intp voffset)
        {
            Int64[] vv = v.datap as Int64[];

            /* could be optimized as we only need the index (no swaps) */
            if (LT(vv[v.data_offset + voffset + 1], vv[v.data_offset + voffset + 0]))
            {
                SWAP(vv, v.data_offset, voffset + 1, voffset + 0);
            }
            if (LT(vv[v.data_offset + voffset + 4], vv[v.data_offset + voffset + 3]))
            {
                SWAP(vv, v.data_offset, voffset + 4, voffset + 3);
            }
            if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 0]))
            {
                SWAP(vv, v.data_offset, voffset + 3, voffset + 0);
            }
            if (LT(vv[v.data_offset + voffset + 4], vv[v.data_offset + voffset + 1]))
            {
                SWAP(vv, v.data_offset, voffset + 4, voffset + 1);
            }
            if (LT(vv[v.data_offset + voffset + 2], vv[v.data_offset + voffset + 1]))
            {
                SWAP(vv, v.data_offset, voffset + 2, voffset + 1);
            }
            if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 2]))
            {
                if (LT(vv[v.data_offset + voffset + 3], vv[v.data_offset + voffset + 1]))
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
            Int64[] vv = v.datap as Int64[];

            /* could be optimized as we only need the index (no swaps) */
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 1)], vv[v.data_offset + IDX(tosort, voffset + 0)]))
            {
                SWAP_IDX(tosort, voffset + 1, voffset + 0);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 4)], vv[v.data_offset + IDX(tosort, voffset + 3)]))
            {
                SWAP_IDX(tosort, voffset + 4, voffset + 3);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 0)]))
            {
                SWAP_IDX(tosort, voffset + 3, voffset + 0);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 4)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
            {
                SWAP_IDX(tosort, voffset + 4, voffset + 1);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 2)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
            {
                SWAP_IDX(tosort, voffset + 2, voffset + 1);
            }
            if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 2)]))
            {
                if (LT(vv[v.data_offset + IDX(tosort, voffset + 3)], vv[v.data_offset + IDX(tosort, voffset + 1)]))
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

            Int64[] vv = v.datap as Int64[];

            for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                npy_intp m = MEDIAN5(v, subleft);
                SWAP(vv, v.data_offset, voffset + subleft + m, voffset + i);
            }

            if (nmed > 2)
                partition_introselect(v, nmed, nmed / 2, pivots, ref npiv, inexact);

            return nmed / 2;
        }

        npy_intp median_of_median5(VoidPtr v, VoidPtr tosort, npy_intp voffset, npy_intp num, npy_intp[] pivots, npy_intp? npiv, bool inexact)
        {
            npy_intp i, subleft;
            npy_intp right = num - 1;
            npy_intp nmed = (right + 1) / 5;

            for (i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                npy_intp m = MEDIAN5(v, tosort, subleft);
                SWAP_IDX(tosort, voffset + subleft + m, voffset + i);
            }

            if (nmed > 2)
                argpartition_introselect(v, tosort, nmed, nmed / 2, pivots, ref npiv, inexact);

            return nmed / 2;
        }

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

        private static Int64 GetItem(VoidPtr v, npy_intp index)
        {
            Int64[] vv = v.datap as Int64[];
            return vv[v.data_offset + index];
        }
        private static Int64 SetItem(VoidPtr v, npy_intp index, Int64 d)
        {
            Int64[] vv = v.datap as Int64[];
            return vv[v.data_offset + index] = d;
        }

        /*
     * partition and return the index were the pivot belongs
     * the data must have following property to avoid bound checks:
     *                  ll ... hh
     * lower-than-pivot [x x x x] larger-than-pivot
     */
        void UNGUARDED_PARTITION(VoidPtr v, Int64 pivot, ref npy_intp ll, ref npy_intp hh)
        {
            Int64[] vv = v.datap as Int64[];

            for (; ; )
            {
                do ll++; while (LT(vv[v.data_offset + ll], pivot));
                do hh--; while (LT(pivot, vv[v.data_offset + hh]));

                if (hh < ll)
                    break;

                SWAP(vv, v.data_offset, ll, hh);
            }
        }

        void UNGUARDED_PARTITION(VoidPtr v, VoidPtr tosort, Int64 pivot, ref npy_intp ll, ref npy_intp hh)
        {
            Int64[] vv = v.datap as Int64[];

            for (; ; )
            {
                do ll++; while (LT(vv[v.data_offset + IDX(tosort, ll)], pivot));
                do hh--; while (LT(pivot, vv[v.data_offset + IDX(tosort, hh)]));

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
    #endregion
}
