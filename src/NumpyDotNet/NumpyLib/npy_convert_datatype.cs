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
        /*
         * Get a cast function to cast from the input descriptor to the
         * output type_number (must be a registered data-type).
         * Returns null if un-successful.
         */
        internal static NpyArray_VectorUnaryFunc NpyArray_GetCastFunc(NpyArray_Descr descr, NPY_TYPES type_num)
        {
            NpyArray_VectorUnaryFunc castfunc = null;

            if (type_num < NPY_TYPES.NPY_NTYPES)
            {
                castfunc = descr.f.cast[(int)type_num];
            }
            else
            {
                /* Check castfuncs for casts to user defined types. */
                if (descr.f.castfuncs != null)
                {
                    NpyArray_CastFuncsItem pitem = descr.f.castfuncs.FirstOrDefault(t => t.totype == type_num);
                    if (pitem != null)
                    {
                        castfunc = pitem.castfunc;
                    }

                }
            }
            if (NpyTypeNum_ISCOMPLEX(descr.type_num) &&
                      !NpyTypeNum_ISCOMPLEX(type_num) &&
                      NpyTypeNum_ISNUMBER(type_num) &&
                      !NpyTypeNum_ISBOOL(type_num))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ComplexWarning,
                    "Casting complex values to real discards the imaginary part");
            }

            if (null == castfunc)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "No cast function available.");
                return null;
            }
            return castfunc;
        }

        /*
         * Must be broadcastable.
         * This code is very similar to NpyArray_CopyInto/NpyArray_MoveInto
         * except casting is done --- NPY_BUFSIZE is used
         * as the size of the casting buffer.
         */

        /*
         * Cast to an already created array.
         */
        internal static int NpyArray_CastTo(NpyArray dest, NpyArray src)
        {
            bool simple;
            bool same;
            NpyArray_VectorUnaryFunc castfunc = null;
            npy_intp srcSize = NpyArray_SIZE(src);
            bool iswap, oswap;

            if (srcSize == 0)
            {
                return 0;
            }
            if (!NpyArray_ISWRITEABLE(dest))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "output array is not writeable");
                return -1;
            }

            castfunc = NpyArray_GetCastFunc(src.descr, dest.descr.type_num);
            if (castfunc == null)
            {
                return -1;
            }

            same = NpyArray_SAMESHAPE(dest, src);
            simple = same && (NpyArray_ISWRITEABLE(src) && NpyArray_ISWRITEABLE(dest)) &&
                             ((NpyArray_ISCARRAY_RO(src) && NpyArray_ISCARRAY(dest)) ||
                              (NpyArray_ISFARRAY_RO(src) && NpyArray_ISFARRAY(dest)));

            if (simple)
            {
                castfunc(src.data, dest.data, srcSize, src, dest);

                if (NpyErr_Occurred())
                {
                    return -1;
                }
                return 0;
            }

            /*
             * If the input or output is OBJECT, STRING, UNICODE, or VOID
             *  then getitem and setitem are used for the cast
             *  and byteswapping is handled by those methods
             */
            if (NpyArray_ISFLEXIBLE(src) || NpyArray_ISOBJECT(src) ||
                NpyArray_ISOBJECT(dest) || NpyArray_ISFLEXIBLE(dest))
            {
                iswap = oswap = false;
            }
            else
            {
                iswap = NpyArray_ISBYTESWAPPED(src);
                oswap = NpyArray_ISBYTESWAPPED(dest);
            }

            return _broadcast_cast(dest, src, castfunc, iswap, oswap);
        }

        internal static NpyArray NpyArray_CastToType(NpyArray mp, NpyArray_Descr at, bool fortran)
        {
            NpyArray dst;
            int ret;
            NpyArray_Descr mpd;

            mpd = mp.descr;

            if (((mpd == at) ||
                 ((mpd.type_num == at.type_num) &&
                  NpyArray_EquivByteorders(mpd, at) &&
                  ((mpd.elsize == at.elsize) || (at.elsize == 0)))) &&
                NpyArray_ISBEHAVED_RO(mp))
            {
                Npy_DECREF(at);
                Npy_INCREF(mp);
                return mp;
            }

    
            dst = NpyArray_Alloc(at, mp.nd, mp.dimensions,
                         fortran, Npy_INTERFACE(mp));

            if (dst == null) {
                return null;
            }
            ret = NpyArray_CastTo(dst, mp);
            if (ret != -1)
            {
                return dst;
            }

            Npy_DECREF(dst);
            return null;
        }

        internal static int NpyArray_CastAnyTo(NpyArray dst, NpyArray mp)
        {
            bool simple;
            NpyArray_VectorUnaryFunc castfunc = null;
            npy_intp mpsize = NpyArray_SIZE(mp);

            if (mpsize == 0)
            {
                return 0;
            }
            if (!NpyArray_ISWRITEABLE(dst))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "output array is not writeable");
                return -1;
            }

            if (!(mpsize == NpyArray_SIZE(dst)))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                 "arrays must have the same number of elements for the cast.");
                return -1;
            }

            castfunc = NpyArray_GetCastFunc(mp.descr, dst.descr.type_num);
            if (castfunc == null)
            {
                return -1;
            }
            simple = (NpyArray_ISWRITEABLE(mp) && NpyArray_ISWRITEABLE(dst)) &&
                     ((NpyArray_ISCARRAY_RO(mp) && NpyArray_ISCARRAY(dst)) ||
                      (NpyArray_ISFARRAY_RO(mp) && NpyArray_ISFARRAY(dst)));
            if (simple)
            {
                castfunc(mp.data, dst.data, mpsize, mp, dst);
                return 0;
            }
            if (NpyArray_SAMESHAPE(dst, mp))
            {
                bool iswap, oswap;
                iswap = NpyArray_ISBYTESWAPPED(mp) && !NpyArray_ISFLEXIBLE(mp);
                oswap = NpyArray_ISBYTESWAPPED(dst) && !NpyArray_ISFLEXIBLE(dst);
                return _broadcast_cast(dst, mp, castfunc, iswap, oswap);
            }
            return _bufferedcast(dst, mp, castfunc);
        }

        internal static bool NpyArray_CanCastSafely(NPY_TYPES fromtype, NPY_TYPES totype)
        {
            NpyArray_Descr from, to;
            int felsize, telsize;

            if (fromtype == totype)
            {
                return true;
            }
            if (fromtype == NPY_TYPES.NPY_BOOL)
            {
                return true;
            }
            if (totype == NPY_TYPES.NPY_BOOL)
            {
                return false;
            }
            if (fromtype == NPY_TYPES.NPY_DATETIME || fromtype == NPY_TYPES.NPY_TIMEDELTA ||
                totype == NPY_TYPES.NPY_DATETIME || totype == NPY_TYPES.NPY_TIMEDELTA)
            {
                return false;
            }
            if (totype == NPY_TYPES.NPY_OBJECT || totype == NPY_TYPES.NPY_VOID)
            {
                return true;
            }
            if (fromtype == NPY_TYPES.NPY_OBJECT || fromtype == NPY_TYPES.NPY_VOID)
            {
                return false;
            }
            from = NpyArray_DescrFromType(fromtype);
            /*
             * cancastto is a NPY_NOTYPE terminated C-int-array of types that
             * the data-type can be cast to safely.
             */
            if (from.f.cancastto != null)
            {
                foreach (var cancastto in from.f.cancastto)
                {
                    if (cancastto == totype)
                        return true;
                }
            }
            if (NpyTypeNum_ISUSERDEF(totype))
            {
                return false;
            }
            to = NpyArray_DescrFromType(totype);
            telsize = to.elsize;
            felsize = from.elsize;
            Npy_DECREF(from);
            Npy_DECREF(to);

            switch (fromtype)
            {
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_SHORT:
                case NPY_TYPES.NPY_INT:
                case NPY_TYPES.NPY_LONG:
                    if (NpyTypeNum_ISINTEGER(totype))
                    {
                        if (NpyTypeNum_ISUNSIGNED(totype))
                        {
                            return false;
                        }
                        else
                        {
                            return telsize >= felsize;
                        }
                    }
                    else if (NpyTypeNum_ISFLOAT(totype))
                    {
                        if (felsize < 8)
                        {
                            return telsize > felsize;
                        }
                        else
                        {
                            return telsize >= felsize;
                        }
                    }
                    else if (NpyTypeNum_ISCOMPLEX(totype))
                    {
                        if (felsize < 8)
                        {
                            return (telsize >> 1) > felsize;
                        }
                        else
                        {
                            return (telsize >> 1) >= felsize;
                        }
                    }
                    else
                    {
                        return totype > fromtype;
                    }
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_USHORT:
                case NPY_TYPES.NPY_UINT:
                case NPY_TYPES.NPY_ULONG:
                    if (NpyTypeNum_ISINTEGER(totype))
                    {
                        if (NpyTypeNum_ISSIGNED(totype))
                        {
                            return telsize > felsize;
                        }
                        else
                        {
                            return telsize >= felsize;
                        }
                    }
                    else if (NpyTypeNum_ISFLOAT(totype))
                    {
                        if (felsize < 8)
                        {
                            return telsize > felsize;
                        }
                        else
                        {
                            return telsize >= felsize;
                        }
                    }
                    else if (NpyTypeNum_ISCOMPLEX(totype))
                    {
                        if (felsize < 8)
                        {
                            return (telsize >> 1) > felsize;
                        }
                        else
                        {
                            return (telsize >> 1) >= felsize;
                        }
                    }
                    else
                    {
                        return totype > fromtype;
                    }
                case NPY_TYPES.NPY_FLOAT:
                case NPY_TYPES.NPY_DOUBLE:
                case NPY_TYPES.NPY_DECIMAL:
                    if (NpyTypeNum_ISCOMPLEX(totype))
                    {
                        return (telsize >> 1) >= felsize;
                    }
                    else
                    {
                        return totype > fromtype;
                    }
                case NPY_TYPES.NPY_COMPLEX:
                    return totype > fromtype;
                case NPY_TYPES.NPY_STRING:
                    return totype > fromtype;
                default:
                    return false;
            }
        }

        /*NUMPY_API
        * Returns true if data of type 'from' may be cast to data of type
        * 'to' according to the rule 'casting'.
        */
        internal static bool NpyArray_CanCastTypeTo(NpyArray_Descr from, NpyArray_Descr to, NPY_CASTING casting)
        {
            return NpyArray_CanCastTo(from, to);
        }


        internal static bool NpyArray_CanCastTo(NpyArray_Descr from, NpyArray_Descr to)
        {
            NPY_TYPES fromtype = from.type_num;
            NPY_TYPES totype = to.type_num;
            bool ret;

            ret = NpyArray_CanCastSafely(fromtype, totype);
            return ret;
        }

        internal static bool NpyArray_ValidType(NPY_TYPES type)
        {
            NpyArray_Descr descr;
            bool res = true;

            descr = NpyArray_DescrFromType(type);
            if (descr == null)
            {
                res = false;
                return res;
            }
            Npy_DECREF(descr);
            return res;
        }


        internal static int _broadcast_cast(NpyArray dest, NpyArray src,  NpyArray_VectorUnaryFunc castfunc, bool iswap, bool oswap)
        {
            int delsize, selsize, maxaxis, i, N;
            NpyArrayMultiIterObject multi;
            npy_intp maxdim, ostrides, istrides;
            VoidPtr []buffers = new VoidPtr[2];
            NpyArray_CopySwapNFunc ocopyfunc, icopyfunc;

            delsize = NpyArray_ITEMSIZE(dest);
            selsize = NpyArray_ITEMSIZE(src);
            multi = NpyArray_MultiIterFromArrays(null, 0, 2, dest, src);
            if (multi == null)
            {
                return -1;
            }

            if (multi.size != NpyArray_SIZE(dest))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                 "array dimensions are not compatible for copy");
                Npy_DECREF(multi);
                return -1;
            }

            icopyfunc = src.descr.f.copyswapn;
            ocopyfunc = dest.descr.f.copyswapn;
            maxaxis = NpyArray_RemoveSmallest(multi);
            if (maxaxis < 0)
            {
                /* cast 1 0-d array to another */
                N = 1;
                maxdim = 1;
                ostrides = (npy_intp)delsize;
                istrides = (npy_intp)selsize;
            }
            else
            {
                maxdim = multi.dimensions[maxaxis];
                N = (int) Math.Min(maxdim, npy_defs.NPY_BUFSIZE);
                ostrides = multi.iters[0].strides[maxaxis];
                istrides = multi.iters[1].strides[maxaxis];

            }
            buffers[0] = NpyDataMem_NEW(dest.descr.type_num, (ulong)(N * delsize));
            buffers[1] = NpyDataMem_NEW(src.descr.type_num, (ulong)(N * selsize));
            if (buffers[0] == null || buffers[1] == null)
            {
                NpyErr_MEMORY();
                return -1;
            }
  
            if (NpyDataType_FLAGCHK(dest.descr, NpyArray_Descr_Flags.NPY_NEEDS_INIT))
            {
                //memset(buffers[0], 0, N * delsize);
            }
            if (NpyDataType_FLAGCHK(src.descr, NpyArray_Descr_Flags.NPY_NEEDS_INIT))
            {
                //memset(buffers[1], 0, N * selsize);
            }


            while (multi.index < multi.size)
            {
                _strided_buffered_cast(multi.iters[0].dataptr,
                                       ostrides,
                                       delsize, oswap, ocopyfunc,
                                       multi.iters[1].dataptr,
                                       istrides,
                                       selsize, iswap, icopyfunc,
                                       maxdim, buffers, N,
                                       castfunc, dest, src);
                NpyArray_MultiIter_NEXT(multi);
            }

            Npy_DECREF(multi);
 
            if (NpyErr_Occurred())
            {
                return -1;
            }

            return 0;
        }

        static int _bufferedcast(NpyArray dst, NpyArray src,  NpyArray_VectorUnaryFunc castfunc)
        {
            VoidPtr inbuffer;
            VoidPtr bptr;
            VoidPtr optr;
            VoidPtr outbuffer = null;
            NpyArrayIterObject it_in = null, it_out = null;
            npy_intp i, index;
            npy_intp ncopies = NpyArray_SIZE(dst) / NpyArray_SIZE(src);
            int elsize =src.descr.elsize;
            int nels = npy_defs.NPY_BUFSIZE;
            int el;
            bool inswap, outswap = false;
            bool obuf = !NpyArray_ISCARRAY(dst);
            int oelsize = dst.descr.elsize;
            NpyArray_CopySwapFunc in_csn;
            NpyArray_CopySwapFunc out_csn;
            int retval = -1;

            in_csn = src.descr.f.copyswap;
            out_csn = dst.descr.f.copyswap;

            /*
             * If the input or output is STRING, UNICODE, or VOID
             * then getitem and setitem are used for the cast
             *  and byteswapping is handled by those methods
             */

            inswap = !(NpyArray_ISFLEXIBLE(src) || NpyArray_ISNOTSWAPPED(src));

            inbuffer = NpyDataMem_NEW(dst.descr.type_num,(ulong)(npy_defs.NPY_BUFSIZE * elsize));
            if (inbuffer == null)
            {
                return -1;
            }
            if (NpyArray_ISOBJECT(src))
            {
                memset(inbuffer, 0, npy_defs.NPY_BUFSIZE * elsize);
            }
            it_in = NpyArray_IterNew(src);
            if (it_in == null)
            {
                goto exit;
            }
            if (obuf)
            {
                outswap = !(NpyArray_ISFLEXIBLE(dst) ||
                            NpyArray_ISNOTSWAPPED(dst));
                outbuffer = NpyDataMem_NEW(dst.descr.type_num, (ulong)(npy_defs.NPY_BUFSIZE * oelsize));
                if (outbuffer == null)
                {
                    goto exit;
                }
                if (NpyArray_ISOBJECT(dst))
                {
                    memset(outbuffer, 0, npy_defs.NPY_BUFSIZE * oelsize);
                }
                it_out = NpyArray_IterNew(dst);
                if (it_out == null)
                {
                    goto exit;
                }
                nels = Math.Min(nels, npy_defs.NPY_BUFSIZE);
            }

            optr = new VoidPtr((obuf) ? outbuffer : dst.data);
            bptr = new VoidPtr(inbuffer);
            el = 0;

            while (ncopies-- > 0)
            {
                index = it_in.size;
                NpyArray_ITER_RESET(it_in);
                while (index-- > 0)
                {
                    in_csn(bptr, it_in.dataptr, inswap, src);
                    bptr.data_offset += elsize;
                    NpyArray_ITER_NEXT(it_in);
                    el += 1;
                    if ((el == nels) || (index == 0))
                    {
                        /* buffer filled, do cast */
                        castfunc(inbuffer, optr, (npy_intp)el, src, dst);
                        if (obuf)
                        {
                            /* Copy from outbuffer to array */
                            for (i = 0; i < el; i++)
                            {
                                out_csn(it_out.dataptr, optr, outswap, dst);
                                optr.data_offset += oelsize;
                                NpyArray_ITER_NEXT(it_out);
                            }
                            optr = new VoidPtr(outbuffer);
                        }
                        else
                        {
                            optr.data_offset += dst.descr.elsize * nels;
                        }
                        el = 0;
                        bptr = new VoidPtr(inbuffer);
                    }
                }
            }
            retval = 0;

            exit:
            Npy_XDECREF(it_in);
            if (obuf)
            {
                Npy_XDECREF(it_out);
            }
            return retval;
        }

        static void _strided_buffered_cast(VoidPtr dptr, npy_intp dstride, int delsize, bool dswap,
                       NpyArray_CopySwapNFunc dcopyfunc,
                       VoidPtr sptr, npy_intp sstride, int selsize, bool sswap,
                       NpyArray_CopySwapNFunc scopyfunc,
                       npy_intp N, VoidPtr[] buffers, int bufsize,
                       NpyArray_VectorUnaryFunc castfunc,
                       NpyArray dest, NpyArray src)
        {
            int i;


            if (N <= bufsize)
            {
                /*
                 * 1. copy input to buffer and swap
                 * 2. cast input to output
                 * 3. swap output if necessary and copy from output buffer
                 */

                scopyfunc(buffers[1], selsize, sptr, sstride, N, sswap, src);
                castfunc(buffers[1], buffers[0], N, src, dest);
                dcopyfunc(dptr, dstride, buffers[0], delsize, N, dswap, dest);
                return;
            }

            /* otherwise we need to divide up into bufsize pieces */
            i = 0;
            while (N > 0)
            {
                int newN = (int)Math.Min(N, bufsize);

                _strided_buffered_cast(dptr + i * dstride, dstride, delsize,
                                       dswap, dcopyfunc,
                                       sptr + i * sstride, sstride, selsize,
                                       sswap, scopyfunc,
                                       (npy_intp)newN, buffers, bufsize, castfunc, dest, src);
                i += newN;
                N -= (npy_intp)bufsize;
            }
            return;
        }


        static NpyArray_Descr NpyArray_ResultType(npy_intp narrs, NpyArray[] arr, npy_intp ndtypes, NpyArray_Descr[] dtypes)
        {
            npy_intp i;
            int use_min_scalar;

            /* If there's just one type, pass it through */
            if (narrs + ndtypes == 1)
            {
                NpyArray_Descr ret = null;
                if (narrs == 1)
                {
                    ret = NpyArray_DESCR(arr[0]);
                }
                else
                {
                    ret = dtypes[0];
                }
                Npy_INCREF(ret);
                return ret;
            }

            // todo
            return arr[0].descr;
        }
    }
}
