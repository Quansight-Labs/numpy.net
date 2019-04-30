using System;
using System.Collections.Generic;
using System.Text;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyLib
{
    internal partial class numpyinternal
    {
        /* See array_assign.h for parameter documentation */
        private static int broadcast_strides(int ndim, npy_intp[] shape,
                        int strides_ndim, npy_intp[] strides_shape, npy_intp[] strides,
                        string strides_name,
                        npy_intp[] out_strides)
        {
            int idim, idim_start = ndim - strides_ndim;

            /* Can't broadcast to fewer dimensions */
            if (idim_start < 0)
            {
                goto broadcast_error;
            }

            /*
             * Process from the end to the start, so that 'strides' and 'out_strides'
             * can point to the same memory.
             */
            for (idim = ndim - 1; idim >= idim_start; --idim)
            {
                npy_intp strides_shape_value = strides_shape[idim - idim_start];
                /* If it doesn't have dimension one, it must match */
                if (strides_shape_value == 1)
                {
                    out_strides[idim] = 0;
                }
                else if (strides_shape_value != shape[idim])
                {
                    goto broadcast_error;
                }
                else
                {
                    out_strides[idim] = strides[idim - idim_start];
                }
            }

            /* New dimensions get a zero stride */
            for (idim = 0; idim < idim_start; ++idim)
            {
                out_strides[idim] = 0;
            }

            return 0;

            broadcast_error:

            {
                string errmsg = string.Format("could not broadcast %s from shape {0} {1} into shape {2}", 
                        strides_name, build_shape_string(strides_ndim, strides_shape), build_shape_string(ndim, shape));


                NpyErr_SetString(npyexc_type.NpyExc_ValueError, errmsg);

                return -1;
            }
        }

        /* See array_assign.h for parameter documentation */
        private static bool raw_array_is_aligned(int ndim, VoidPtr data, npy_intp[] strides, int alignment)
        {
            if (alignment > 1)
            {
                npy_intp align_check = (npy_intp)data;
                int idim;

                for (idim = 0; idim < ndim; ++idim)
                {
                    align_check |= strides[idim];
                }

                return npy_is_aligned((void*)align_check, alignment);
            }
            else
            {
                return true;
            }
        }


        /* Returns 1 if the arrays have overlapping data, 0 otherwise */
        private static bool arrays_overlap(NpyArray arr1, NpyArray arr2)
        {
            mem_overlap_t result;

            result = solve_may_share_memory(arr1, arr2, NPY_MAY_SHARE_BOUNDS);
            if (result == MEM_OVERLAP_NO)
            {
                return false;
            }
            else
            {
                return true;
            }
        }


    }
}
