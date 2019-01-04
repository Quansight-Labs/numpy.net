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
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif
using size_t = System.UInt64;

namespace NumpyLib
{
    internal partial class numpyinternal
    {

        /*
         * Resize (reallocate data).  Only works if nothing else is referencing this
         * array and it is contiguous.  If refcheck is 0, then the reference count is
         * not checked and assumed to be 1.  You still must own this data and have no
         * weak-references and no base object.
         */
        internal static int NpyArray_Resize(NpyArray self, NpyArray_Dims newshape, bool refcheck, NPY_ORDER fortran)
        {
            npy_intp oldsize, newsize;
            int new_nd = newshape.len, k, elsize;
            int refcnt;
            npy_intp  []new_dimensions = newshape.ptr;
            npy_intp []new_strides = new npy_intp[npy_defs.NPY_MAXDIMS];
            size_t sd;
            npy_intp[] dimptr;
            npy_intp[] strptr;
            npy_intp largest;

            if (!NpyArray_ISONESEGMENT(self))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,"resize only works on single-segment arrays");
                return -1;
            }

            if (self.descr.elsize == 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "Bad data-type size.");
                return -1;
            }
            newsize = 1;
            largest = npy_defs.NPY_MAX_INTP / self.descr.elsize;
            for (k = 0; k < new_nd; k++)
            {
                if (new_dimensions[k] == 0)
                {
                    break;
                }
                if (new_dimensions[k] < 0)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError,"negative dimensions not allowed");
                    return -1;
                }
                newsize *= new_dimensions[k];
                if (newsize <= 0 || newsize > largest)
                {
                    NpyErr_MEMORY();
                    return -1;
                }
            }
            oldsize = NpyArray_SIZE(self);

            if (oldsize != newsize)
            {
                if (!((self.flags & NPYARRAYFLAGS.NPY_OWNDATA) != 0))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "cannot resize this array: it does not own its data");
                    return -1;
                }

                /* TODO: This isn't right for usage from C.  I think we
                   need to revisit the refcounts so we don't have counts
                   of 0. */
                if (refcheck)
                {
                    refcnt = (int)self.nob_refcnt;
                }
                else
                {
                    refcnt = 0;
                }
                if ((refcnt > 0)
                    || (self.base_arr != null) || (null != self.base_obj))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "cannot resize an array references or is referenced\nby another array in this way.  Use the resize function");
                    return -1;
                }

                if (newsize == 0)
                {
                    sd = (size_t)self.descr.elsize;
                }
                else
                {
                    sd = (size_t)(newsize * self.descr.elsize);
                }
                /* Reallocate space if needed */
                VoidPtr new_data = NpyDataMem_RENEW(self.data, sd);
                if (new_data == null)
                {
                    NpyErr_MEMORY();
                    return -1;
                }
                self.data = new_data;
            }

            if ((newsize > oldsize) && NpyArray_ISWRITEABLE(self))
            {
                /* Fill new memory with zeros */
                elsize = self.descr.elsize;
                memset(self.data + oldsize * elsize, 0, (newsize - oldsize) * elsize);
            }

            if (self.nd != new_nd)
            {
                /* Different number of dimensions. */
                self.nd = new_nd;
                /* Need new dimensions and strides arrays */
                dimptr = NpyDimMem_NEW(new_nd);
                strptr = NpyDimMem_NEW(new_nd);
                if (dimptr == null || strptr == null)
                {
                    NpyErr_MEMORY();
                    return -1;
                }

                memcpy(dimptr, self.dimensions, self.nd * sizeof(npy_intp));
                memcpy(strptr, self.strides, self.nd * sizeof(npy_intp));
                self.dimensions = dimptr;
                self.strides = strptr;
            }

            /* make new_strides variable */
            sd = (size_t)self.descr.elsize;

            NPYARRAYFLAGS flags = 0;
            sd = (size_t)npy_array_fill_strides(new_strides, new_dimensions, new_nd, sd, self.flags, ref flags); self.flags = flags;
            memmove(new VoidPtr(self.dimensions), new VoidPtr(new_dimensions), new_nd * sizeof(npy_intp));
            memmove(new VoidPtr(self.strides), new VoidPtr(new_strides), new_nd * sizeof(npy_intp));
            return 0;
        }

        internal static NpyArray NpyArray_Newshape(NpyArray self, NpyArray_Dims newdims, NPY_ORDER fortran)
        {
            int i;
            npy_intp []dimensions = newdims.ptr;
            NpyArray ret;
            int n = newdims.len;
            bool same = true; 
            bool incref = true;
            npy_intp[] strides = null;
            npy_intp []newstrides = new npy_intp[npy_defs.NPY_MAXDIMS];
            NPYARRAYFLAGS flags;

            if (fortran == NPY_ORDER.NPY_ANYORDER)
            {
                fortran = NpyArray_ISFORTRAN(self) ? NPY_ORDER.NPY_FORTRANORDER : NPY_ORDER.NPY_ANYORDER;
            }
            /*  Quick check to make sure anything actually needs to be done */
            if (n == self.nd)
            {
                same = true;
                i = 0;
                while (same && i < n)
                {
                    if (NpyArray_DIM(self, i) != dimensions[i])
                    {
                        same = false;
                    }
                    i++;
                }
                if (same)
                {
                    return NpyArray_View(self, null, null);
                }
            }

            /*
             * Returns a pointer to an appropriate strides array
             * if all we are doing is inserting ones into the shape,
             * or removing ones from the shape
             * or doing a combination of the two
             * In this case we don't need to do anything but update strides and
             * dimensions.  So, we can handle non single-segment cases.
             */
            i = _check_ones(self, n, dimensions, newstrides);
            if (i == 0)
            {
                strides = newstrides;
            }
            flags = self.flags;

            if (strides == null)
            {
                /*
                 * we are really re-shaping not just adding ones to the shape somewhere
                 * fix any -1 dimensions and check new-dimensions against old size
                 */
                if (_fix_unknown_dimension(newdims, NpyArray_SIZE(self)) < 0)
                {
                    return null;
                }
                /*
                 * sometimes we have to create a new copy of the array
                 * in order to get the right orientation and
                 * because we can't just re-use the buffer with the
                 * data in the order it is in.
                 */
                if (!(NpyArray_ISONESEGMENT(self)) ||
                    (((NpyArray_CHKFLAGS(self, NPYARRAYFLAGS.NPY_CONTIGUOUS) &&
                       fortran == NPY_ORDER.NPY_FORTRANORDER) ||
                      (NpyArray_CHKFLAGS(self, NPYARRAYFLAGS.NPY_FORTRAN) &&
                          fortran == NPY_ORDER.NPY_CORDER)) && (self.nd > 1)))
                {
                    bool success = _attempt_nocopy_reshape(self, n, dimensions, newstrides, (fortran == NPY_ORDER.NPY_FORTRANORDER));
                    if (success)
                    {
                        /* no need to copy the array after all */
                        strides = newstrides;
                        flags = self.flags;
                    }
                    else
                    {
                        NpyArray  newArray;
                        newArray = NpyArray_NewCopy(self, fortran);
                        if (newArray == null)
                        {
                            return null;
                        }
                        incref = false;
                        self = newArray;
                        flags = self.flags;
                    }
                }

                /* We always have to interpret the contiguous buffer correctly */

                /* Make sure the flags argument is set. */
                if (n > 1)
                {
                    if (fortran == NPY_ORDER.NPY_FORTRANORDER)
                    {
                        flags &= ~NPYARRAYFLAGS.NPY_CONTIGUOUS;
                        flags |= NPYARRAYFLAGS.NPY_FORTRAN;
                    }
                    else
                    {
                        flags &= ~NPYARRAYFLAGS.NPY_FORTRAN;
                        flags |= NPYARRAYFLAGS.NPY_CONTIGUOUS;
                    }
                }
            }
            else if (n > 0)
            {
                /*
                 * replace any 0-valued strides with
                 * appropriate value to preserve contiguousness
                 */
                if (fortran == NPY_ORDER.NPY_FORTRANORDER)
                {
                    if (strides[0] == 0)
                    {
                        strides[0] = (npy_intp)self.descr.elsize;
                    }
                    for (i = 1; i < n; i++)
                    {
                        if (strides[i] == 0)
                        {
                            strides[i] = strides[i - 1] * dimensions[i - 1];
                        }
                    }
                }
                else
                {
                    if (strides[n - 1] == 0)
                    {
                        strides[n - 1] = (npy_intp)self.descr.elsize;
                    }
                    for (i = n - 2; i > -1; i--)
                    {
                        if (strides[i] == 0)
                        {
                            strides[i] = strides[i + 1] * dimensions[i + 1];
                        }
                    }
                }
            }

            Npy_INCREF(self.descr);
            ret = NpyArray_NewFromDescr(self.descr,
                                        n, dimensions,
                                        strides, 
                                        self.data, 
                                        flags, false, null,
                                        Npy_INTERFACE(self));

            if (ret == null)
            {
                goto fail;
            }
            if (incref)
            {
                Npy_INCREF(self);
            }
            ret.base_arr = self;
            NpyArray_UpdateFlags(ret, NPYARRAYFLAGS.NPY_CONTIGUOUS | NPYARRAYFLAGS.NPY_FORTRAN);
            Debug.Assert(null == ret.base_arr || null == ret.base_obj);
            return ret;

            fail:
            if (!incref)
            {
                Npy_DECREF(self);
            }
            return null;
        }

        /*
         * return a new view of the array object with all of its unit-length
         * dimensions squeezed out if needed, otherwise
         * return the same array.
         */
        internal static NpyArray NpyArray_Squeeze(NpyArray self)
        {
            int nd = self.nd;
            int newnd = nd;
            npy_intp[] dimensions = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp[] strides = new npy_intp[npy_defs.NPY_MAXDIMS];
            int i, j;
            NpyArray ret;

            if (nd == 0)
            {
                Npy_INCREF(self);
                return self;
            }
            for (j = 0, i = 0; i < nd; i++)
            {
                if (self.dimensions[i] == 1)
                {
                    newnd -= 1;
                }
                else
                {
                    dimensions[j] = self.dimensions[i];
                    strides[j++] = self.strides[i];
                }
            }

            Npy_INCREF(self.descr);
            ret = NpyArray_NewView(self.descr, newnd, dimensions, strides,
                                   self, 0, false);
            return ret;

        }


        internal static NpyArray NpyArray_SwapAxes(NpyArray ap, int a1, int a2)
        {
            NpyArray_Dims new_axes = new NpyArray_Dims();
            npy_intp[] dims = new npy_intp[npy_defs.NPY_MAXDIMS];
            int n, i, val;
            NpyArray ret;

            if (a1 == a2)
            {
                Npy_INCREF(ap);
                return ap;
            }

            n = ap.nd;
            if (n <= 1)
            {
                Npy_INCREF(ap);
                return ap;
            }

            if (a1 < 0)
            {
                a1 += n;
            }
            if (a2 < 0)
            {
                a2 += n;
            }
            if ((a1 < 0) || (a1 >= n))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "bad axis1 argument to swapaxes");
                return null;
            }
            if ((a2 < 0) || (a2 >= n))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,"bad axis2 argument to swapaxes");
                return null;
            }
            new_axes.ptr = dims;
            new_axes.len = n;

            for (i = 0; i < n; i++)
            {
                if (i == a1)
                {
                    val = a2;
                }
                else if (i == a2)
                {
                    val = a1;
                }
                else
                {
                    val = i;
                }
                new_axes.ptr[i] = (npy_intp)val;
            }
            ret = NpyArray_Transpose(ap, new_axes);
            return ret;
        }

        /*
         * Ravel
         * Returns a contiguous array
         */
        internal static NpyArray NpyArray_Ravel(NpyArray a, NPY_ORDER flags)
        {
            NpyArray_Dims newdim = new NpyArray_Dims() { ptr = null, len = 1 };
            npy_intp []val = new npy_intp[] { -1 };
            bool fortran = false;

            if (flags == NPY_ORDER.NPY_ANYORDER)
            {
                fortran = NpyArray_ISFORTRAN(a);
            }
            newdim.ptr = val;

            if (!fortran && NpyArray_ISCONTIGUOUS(a))
            {
                return NpyArray_Newshape(a, newdim, flags);
            }
            else if (fortran && NpyArray_ISFORTRAN(a))
            {
                return NpyArray_Newshape(a, newdim, NPY_ORDER.NPY_FORTRANORDER);
            }
            else
            {
                return NpyArray_Flatten(a, flags);
            }
        }

        internal static NpyArray NpyArray_FlatView(NpyArray a)
        {
            NpyArray r;
            npy_intp []size = new npy_intp[1];

            /* Any argument ignored */

            /* Two options:
             *  1) underlying array is contiguous
             *  -- return 1-d wrapper around it
             * 2) underlying array is not contiguous
             * -- make new 1-d contiguous array with updateifcopy flag set
             * to copy back to the old array
             */
            size[0] = NpyArray_SIZE(a);
            Npy_INCREF(a.descr);
            if (NpyArray_ISCONTIGUOUS(a))
            {
                r = NpyArray_NewView(a.descr, 1, size, null, a, 0, true);
            }
            else
            {
                r = NpyArray_Alloc(a.descr, 1, size, false, Npy_INTERFACE(a));
                if (r == null)
                {
                    return null;
                }
                if (_flat_copyinto(r, a, NPY_ORDER.NPY_CORDER) < 0)
                {
                    Npy_DECREF(r);
                    return null;
                }
                NpyArray_FLAGS_OR(r,NPYARRAYFLAGS.NPY_UPDATEIFCOPY);
                a.flags &= ~NPYARRAYFLAGS.NPY_WRITEABLE;
                Npy_INCREF(a);
                NpyArray_BASE_ARRAY_Update(r,a);
            }
            return r;
        }

        internal static NpyArray NpyArray_Transpose(NpyArray ap, NpyArray_Dims permute)
        {
            npy_intp[] axes;
            npy_intp axis;
            int i, n;
            npy_intp[] permutation = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp[] reverse_permutation = new npy_intp[npy_defs.NPY_MAXDIMS];
            NpyArray ret = null;

            if (permute == null)
            {
                n = ap.nd;
                for (i = 0; i < n; i++)
                {
                    permutation[i] = n - 1 - i;
                }
            }
            else
            {
                n = permute.len;
                axes = permute.ptr;
                if (n != ap.nd)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "axes don't match array");
                    return null;
                }
                for (i = 0; i < n; i++)
                {
                    reverse_permutation[i] = -1;
                }
                for (i = 0; i < n; i++)
                {
                    axis = axes[i];
                    if (axis < 0)
                    {
                        axis = ap.nd + axis;
                    }
                    if (axis < 0 || axis >= ap.nd)
                    {
                        NpyErr_SetString(npyexc_type.NpyExc_ValueError, "invalid axis for this array");
                        return null;
                    }
                    if (reverse_permutation[axis] != -1)
                    {
                        NpyErr_SetString(npyexc_type.NpyExc_ValueError, "repeated axis in transpose");
                        return null;
                    }
                    reverse_permutation[axis] = i;
                    permutation[i] = axis;
                }
                for (i = 0; i < n; i++)
                {
                }
            }

            /*
             * this allocates memory for dimensions and strides (but fills them
             * incorrectly), sets up descr, and points data at ap.data.
             */
            Npy_INCREF(ap.descr);
            ret = NpyArray_NewView(ap.descr, n, ap.dimensions, null,
                                   ap, 0, false);
            if (ret == null)
            {
                return null;
            }

            /* fix the dimensions and strides of the return-array */
            for (i = 0; i < n; i++)
            {
                ret.dimensions[i] = ap.dimensions[permutation[i]];
                ret.strides[i] = ap.strides[permutation[i]];
            }
            NpyArray_UpdateFlags(ret, NPYARRAYFLAGS.NPY_CONTIGUOUS | NPYARRAYFLAGS.NPY_FORTRAN);
            return ret;
        }

        internal static NpyArray NpyArray_Flatten(NpyArray a, NPY_ORDER order)
        {
            NpyArray ret;
            npy_intp []size = new npy_intp[1];

            if (order == NPY_ORDER.NPY_ANYORDER)
            {
                order = NpyArray_ISFORTRAN(a) ? NPY_ORDER.NPY_FORTRANORDER : NPY_ORDER.NPY_ANYORDER;
            }
            Npy_INCREF(a.descr);
            size[0] = NpyArray_SIZE(a);
            ret = NpyArray_Alloc(a.descr, 1, size, false, Npy_INTERFACE(a));
            if (ret == null)
            {
                return null;
            }
            if (_flat_copyinto(ret, a, order) < 0)
            {
                Npy_DECREF(ret);
                return null;
            }
            return ret;
        }

        /* inserts 0 for strides where dimension will be 1 */
        static int _check_ones(NpyArray self, int newnd, npy_intp[] newdims, npy_intp[] strides)
        {
            int nd;
            npy_intp []dims;
            bool done = false;
            int j, k;

            nd = self.nd;
            dims = self.dimensions;

            for (k = 0, j = 0; !done && (j < nd || k < newnd);)
            {
                if ((j < nd) && (k < newnd) && (newdims[k] == dims[j]))
                {
                    strides[k] = self.strides[j];
                    j++;
                    k++;
                }
                else if ((k < newnd) && (newdims[k] == 1))
                {
                    strides[k] = 0;
                    k++;
                }
                else if ((j < nd) && (dims[j] == 1))
                {
                    j++;
                }
                else
                {
                    done = true;
                }
            }
            if (done)
            {
                return -1;
            }
            return 0;
        }

        /*
         * attempt to reshape an array without copying data
         *
         * This function should correctly handle all reshapes, including
         * axes of length 1. Zero strides should work but are untested.
         *
         * If a copy is needed, returns 0
         * If no copy is needed, returns 1 and fills newstrides
         *     with appropriate strides
         *
         * The "fortran" argument describes how the array should be viewed
         * during the reshape, not how it is stored in memory (that
         * information is in self.strides).
         *
         * If some output dimensions have length 1, the strides assigned to
         * them are arbitrary. In the current implementation, they are the
         * stride of the next-fastest index.
         */
        static bool _attempt_nocopy_reshape(NpyArray self, int newnd, npy_intp[] newdims, npy_intp[] newstrides, bool fortran)
        {
            int oldnd;
            npy_intp []olddims = new npy_intp[npy_defs.NPY_MAXDIMS];
            npy_intp [] oldstrides = new npy_intp[npy_defs.NPY_MAXDIMS];
            int oi, oj, ok, ni, nj, nk;
            int np, op;

            oldnd = 0;
            for (oi = 0; oi < self.nd; oi++)
            {
                if (self.dimensions[oi] != 1)
                {
                    olddims[oldnd] = self.dimensions[oi];
                    oldstrides[oldnd] = self.strides[oi];
                    oldnd++;
                }
            }

            /*
              fprintf(stderr, "_attempt_nocopy_reshape( (");
              for (oi=0; oi<oldnd; oi++)
              fprintf(stderr, "(%d,%d), ", olddims[oi], oldstrides[oi]);
              fprintf(stderr, ") . (");
              for (ni=0; ni<newnd; ni++)
              fprintf(stderr, "(%d,*), ", newdims[ni]);
              fprintf(stderr, "), fortran=%d)\n", fortran);
            */


            np = 1;
            for (ni = 0; ni < newnd; ni++)
            {
                np = (int)(np * newdims[ni]);
            }
            op = 1;
            for (oi = 0; oi < oldnd; oi++)
            {
                op = (int)(op * olddims[oi]);
            }
            if (np != op)
            {
                /* different total sizes; no hope */
                return false;
            }
            /* the current code does not handle 0-sized arrays, so give up */
            if (np == 0)
            {
                return false;
            }

            oi = 0;
            oj = 1;
            ni = 0;
            nj = 1;
            while (ni < newnd && oi < oldnd)
            {
                np = (int)newdims[ni];
                op = (int)olddims[oi];

                while (np != op)
                {
                    if (np < op)
                    {
                        np = (int)(np * newdims[nj++]);
                    }
                    else
                    {
                        op = (int)(op *olddims[oj++]);
                    }
                }

                for (ok = oi; ok < oj - 1; ok++)
                {
                    if (fortran)
                    {
                        if (oldstrides[ok + 1] != olddims[ok] * oldstrides[ok])
                        {
                            /* not contiguous enough */
                            return false;
                        }
                    }
                    else
                    {
                        /* C order */
                        if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1])
                        {
                            /* not contiguous enough */
                            return false;
                        }
                    }
                }

                if (fortran)
                {
                    newstrides[ni] = oldstrides[oi];
                    for (nk = ni + 1; nk < nj; nk++)
                    {
                        newstrides[nk] = newstrides[nk - 1] * newdims[nk - 1];
                    }
                }
                else
                {
                    /* C order */
                    newstrides[nj - 1] = oldstrides[oj - 1];
                    for (nk = nj - 1; nk > ni; nk--)
                    {
                        newstrides[nk - 1] = newstrides[nk] * newdims[nk];
                    }
                }
                ni = nj++;
                oi = oj++;
            }

            /*
              fprintf(stderr, "success: _attempt_nocopy_reshape (");
              for (oi=0; oi<oldnd; oi++)
              fprintf(stderr, "(%d,%d), ", olddims[oi], oldstrides[oi]);
              fprintf(stderr, ") . (");
              for (ni=0; ni<newnd; ni++)
              fprintf(stderr, "(%d,%d), ", newdims[ni], newstrides[ni]);
              fprintf(stderr, ")\n");
            */

            return true;
        }


        static int _fix_unknown_dimension(NpyArray_Dims newshape, npy_intp s_original)
        {
            npy_intp[] dimensions;
            int i_unknown, s_known;
            int i, n;
            string msg= "total size of new array must be unchanged";

            dimensions = newshape.ptr;
            n = newshape.len;
            s_known = 1;
            i_unknown = -1;

            for (i = 0; i < n; i++)
            {
                if (dimensions[i] < 0)
                {
                    if (i_unknown == -1)
                    {
                        i_unknown = i;
                    }
                    else
                    {
                        NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                         "can only specify one unknown dimension");
                        return -1;
                    }
                }
                else
                {
                    s_known = (int)(s_known * dimensions[i]);
                }
            }

            if (i_unknown >= 0)
            {
                if ((s_known == 0) || (s_original % s_known != 0))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                    return -1;
                }
                dimensions[i_unknown] = (npy_intp)(s_original / s_known);
            }
            else
            {
                if (s_original != s_known)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                    return -1;
                }
            }
            return 0;
        }

    }
}
