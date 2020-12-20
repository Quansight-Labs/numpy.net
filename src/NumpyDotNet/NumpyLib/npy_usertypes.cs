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
#else
using npy_intp = System.Int32;
#endif
namespace NumpyLib
{
    internal partial class numpyinternal
    {

        static List<NpyArray_Descr> npy_userdescrs = new List<NpyArray_Descr>();

        internal static void NpyArray_InitArrFuncs(NpyArray_ArrFuncs f)
        {
            int i;

            f.cast = null;
            f.getitem = null;
            f.setitem = null;
            f.compare = null;
            f.argmax = null;
            f.scanfunc = null;
            f.fromstr = null;
            f.nonzero = null;
            f.fill = null;
            f.fillwithscalar = null;
            for (i = 0; i < npy_defs.NPY_NSORTS; i++)
            {
                f.sort[i] = null;
                f.argsort[i] = null;
            }
            f.castfuncs = null;
            f.scalarkind = null;
            f.cancastscalarkindto = null;
            f.cancastto = null;
        }

        internal static int NpyArray_GetNumusertypes()
        {
            return npy_userdescrs.Count();
        }

        internal static bool NpyArray_RegisterDataType(NpyArray_Descr descr)
        {
            NpyArray_Descr descr2;
            NPY_TYPES typenum;
            int i;
            NpyArray_ArrFuncs f;

            /* See if this type is already registered */
            for (i = 0; i < npy_userdescrs.Count(); i++)
            {
                descr2 = npy_userdescrs[i];
                if (descr2 == descr)
                {
                    return true;
                }
            }
            typenum = NPY_TYPES.NPY_USERDEF + npy_userdescrs.Count();
            descr.type_num = typenum;
            if (descr.elsize == 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "cannot register a flexible data-type");
                return false;
            }
            f = descr.f;
            if (f.nonzero == null)
            {
                f.nonzero = DefaultArrayHandlers.GetArrayHandler(typenum).NonZero;
            }
  
            if (f.getitem == null || f.setitem == null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "a required array function is missing.");
                return false;
            }
 
            npy_userdescrs.Add(descr);
            return true;
        }

        internal static int NpyArray_RegisterCastFunc(NpyArray_Descr descr, NPY_TYPES totype, NpyArray_VectorUnaryFunc castfunc)
        {
            if (totype < NPY_TYPES.NPY_NTYPES)
            {
                descr.f.cast[(int)totype] = castfunc;
                return 0;
            }
            if (!NpyTypeNum_ISUSERDEF(totype))
            {
                NpyErr_SetString(npyexc_type.NpyExc_TypeError, "invalid type number.");
                return -1;
            }
            if (descr.f.castfuncs == null)
            {
                descr.f.castfuncs = new List<NpyArray_CastFuncsItem>();
                if (descr.f.castfuncs == null)
                {
                    return -1;
                }
            }
            descr.f.castfuncs.Add(new NpyArray_CastFuncsItem() { castfunc = castfunc, totype = totype });
            return 0;
        }

        internal static int NpyArray_RegisterCanCast(NpyArray_Descr descr, NPY_TYPES totype, NPY_SCALARKIND scalar)
        {
            if (scalar == NPY_SCALARKIND.NPY_NOSCALAR)
            {
                /*
                 * register with cancastto
                 * These lists won't be freed once created
                 * -- they become part of the data-type
                 */
                if (descr.f.cancastto == null)
                {
                    descr.f.cancastto = new List<NPY_TYPES>();
                    descr.f.cancastto.Add(NPY_TYPES.NPY_NOTYPE);
                }
                descr.f.cancastto.Add(totype);
            }
            else
            {
                /* register with cancastscalarkindto */
                if (descr.f.cancastscalarkindto == null)
                {
                    descr.f.cancastscalarkindto = new Dictionary<NPY_SCALARKIND, object>();
                    for (int i = 0; i < npy_defs.NPY_NSCALARKINDS; i++)
                    {
                        descr.f.cancastscalarkindto.Add((NPY_SCALARKIND)i, null);
                    }
                }
                descr.f.cancastscalarkindto[scalar] = totype;
            }
            return 0;
        }

        internal static NpyArray_Descr NpyArray_UserDescrFromTypeNum(NPY_TYPES typenum)
        {
            return npy_userdescrs[typenum - NPY_TYPES.NPY_USERDEF];
        }

  

        static void _default_copyswap(VoidPtr dst, npy_intp dstride, VoidPtr src, npy_intp sstride, npy_intp n, bool swap, NpyArray arr)
        {

            if (src != null && dst.type_num != src.type_num)
            {
                VoidPtr _dst = new VoidPtr(dst);
                VoidPtr _src = new VoidPtr(src);

                for (npy_intp i = 0; i < n; i++)
                {
                    NpyArray_CopySwapFunc(_dst, _src, swap, arr);

                    if (swap)
                    {
                        swapvalue(_dst, arr.ItemDiv);
                    }

                    _dst.data_offset += dstride;
                    _src.data_offset += sstride;
                }
                return;
            }
            else
            {
                VoidPtr _dst = new VoidPtr(dst);
                VoidPtr _src = src != null ? new VoidPtr(src) : null;

                var helper = MemCopy.GetMemcopyHelper(_dst);
                helper.default_copyswap(_dst, dstride, _src, sstride, n, swap);

            }


        }

  

    }
}
