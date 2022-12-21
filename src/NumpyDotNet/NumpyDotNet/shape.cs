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
using NumpyLib;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif


namespace NumpyDotNet
{
    public class shape
    {
        public npy_intp[] iDims = null;

        /// <summary>
        /// get shape index by position
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public npy_intp this[npy_intp index]
        {
            get
            {
                if (index < 0)
                {
                    index = iDims.Length + index;
                }

                if (index > iDims.Length - 1)
                    throw new Exception("attempting to access shape dimension outside range");
   
                return this.iDims[index];
            }

        }

        /// <summary>
        /// get shape indices by slice string
        /// </summary>
        /// <param name="slice"></param>
        /// <returns></returns>
        public IEnumerable<npy_intp> this[string slice]
        {
            get
            {
                try
                {
                    var A = np.array(iDims);
                    var B = (ndarray)A[slice];

                    var C = new List<npy_intp>();

                    foreach (var b in B)
                    {
                        C.Add((npy_intp)b);
                    }
                    return C;
                }
                catch (Exception ex)
                {
                    throw new Exception("Failure to convert shape array via the slice string");
                }
     
            }

        }
        /// <summary>
        /// get shape indices via slice
        /// </summary>
        /// <param name="slice"></param>
        /// <returns></returns>
        public IEnumerable<npy_intp> this[Slice slice]
        {
            get
            {
                try
                {
                    var A = np.array(iDims);
                    var B = (ndarray)A[slice];

                    var C = new List<npy_intp>();

                    foreach (var b in B)
                    {
                        C.Add((npy_intp)b);
                    }
                    return C;
                }
                catch (Exception ex)
                {
                    throw new Exception("Failure to convert shape array via the slice string");
                }

            }

        }

        public npy_intp lastDim
        {
            get
            {
                if (iDims == null)
                    return 0;

                return iDims[iDims.Length - 1];
            }
        }
        /// <summary>
        /// returns true of the shapes are equivalent
        /// </summary>
        /// <param name="o2"></param>
        /// <returns></returns>
        public override bool Equals(object o2)
        {
            shape s2 = o2 as shape;
            if (s2 == null)
                return false;

            if (this.iDims.Length == s2.iDims.Length)
            {
                for (int i = 0; i < this.iDims.Length; i++)
                {
                    if (this.iDims[i] != s2.iDims[i])
                        return false;
                }
                return true;
            }
            return false;
        }
        public static bool operator ==(shape s1, shape s2)
        {
            // If left hand side is null...
            if (System.Object.ReferenceEquals(s1, null) || System.Object.ReferenceEquals(s2, null))
            {
                // ...and right hand side is null...
                if (System.Object.ReferenceEquals(s1, null) && System.Object.ReferenceEquals(s2, null))
                {
                    //...both are null and are Equal.
                    return true;
                }

                // ...right hand side is not null, therefore not Equal.
                return false;
            }

            return s1.Equals(s2);
        }
        public static bool operator !=(shape s1, shape s2)
        {
            return !(s1 == s2);
        }
        /// <summary>
        /// convert an ndarray of integers into a shape
        /// </summary>
        /// <param name="arr"></param>
        public shape(ndarray arr)
        {
            if (arr.TypeNum == NPY_TYPES.NPY_INT32 || arr.TypeNum == NPY_TYPES.NPY_INT64)
            {
                if (arr.ndim != 1)
                    throw new Exception("only single dimension arrays can be used as shapes");

                iDims = new npy_intp[arr.Size];

                int i = 0;
                foreach (var d in arr)
                {
#if NPY_INTP_64
                    iDims[i] = Convert.ToInt64(d);
#else
                    iDims[i] = Convert.ToInt32(d);
#endif
                    i++;
                }
                return;
            }


            throw new Exception("Only Int32 or Int64 arrays can be converted to shape objects");

        }

        /// <summary>
        /// default constructor needed for serialization
        /// </summary>
        public shape()
        {

        }

        /// <summary>
        /// convert a collection of Int32 values into a shape
        /// </summary>
        /// <param name="dim"></param>
        public shape(IEnumerable<Int32> dim)
        {
            int nd = dim.Count();
            iDims = new npy_intp[nd];

            int i = 0;
            foreach (var d in dim)
            {
                iDims[i] = (npy_intp)d;
                i++;
            }
        }
        /// <summary>
        /// add two arrays into one and create a shape with results
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="dim2"></param>
        public shape(IEnumerable<Int32> dim, IEnumerable<Int32> dim2)
        {
            int nd = dim.Count() + dim2.Count();
            iDims = new npy_intp[nd];

            int i = 0;
            foreach (var d in dim)
            {
                iDims[i] = (npy_intp)d;
                i++;
            }
 
            foreach (var d in dim2)
            {
                iDims[i] = (npy_intp)d;
                i++;
            }
        }
        /// <summary>
        /// add three arrays into one and create a shape with results
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="dim2"></param>
        /// <param name="dim3"></param>
        public shape(IEnumerable<Int32> dim, IEnumerable<Int32> dim2, IEnumerable<Int32> dim3)
        {
            int nd = dim.Count() + dim2.Count() + dim3.Count();
            iDims = new npy_intp[nd];

            int i = 0;
            foreach (var d in dim)
            {
                iDims[i] = (npy_intp)d;
                i++;
            }

            foreach (var d in dim2)
            {
                iDims[i] = (npy_intp)d;
                i++;
            }

            foreach (var d in dim3)
            {
                iDims[i] = (npy_intp)d;
                i++;
            }
        }

        /// <summary>
        /// convert a collection of Int64 values into a shape
        /// </summary>
        /// <param name="dim"></param>
        public shape(IEnumerable<Int64> dim)
        {
            int nd = dim.Count();
            iDims = new npy_intp[nd];

            int i = 0;
            foreach (var d in dim)
            {
                iDims[i] = (npy_intp)d;
                i++;
            }
        }
        /// <summary>
        /// add two arrays into one and create a shape with results
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="dim2"></param>
        public shape(IEnumerable<Int64> dim, IEnumerable<Int64> dim2)
        {
            int nd = dim.Count() + dim2.Count();
            iDims = new npy_intp[nd];

            int i = 0;
            foreach (var d in dim)
            {
                iDims[i] = (npy_intp)d;
                i++;
            }

            foreach (var d in dim2)
            {
                iDims[i] = (npy_intp)d;
                i++;
            }
        }
        /// <summary>
        /// add three arrays into one and create a shape with results
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="dim2"></param>
        /// <param name="dim3"></param>
        public shape(IEnumerable<Int64> dim, IEnumerable<Int64> dim2, IEnumerable<Int64> dim3)
        {
            int nd = dim.Count() + dim2.Count() + dim3.Count();
            iDims = new npy_intp[nd];

            int i = 0;
            foreach (var d in dim)
            {
                iDims[i] = (npy_intp)d;
                i++;
            }

            foreach (var d in dim2)
            {
                iDims[i] = (npy_intp)d;
                i++;
            }

            foreach (var d in dim3)
            {
                iDims[i] = (npy_intp)d;
                i++;
            }
        }

        /// <summary>
        /// convert an array of npy_intp into a shape
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="nd"></param>
        public shape(npy_intp[] dim, int nd)
        {
            iDims = new npy_intp[nd];
            for (int i = 0; i < nd; i++)
            {
                iDims[i] = dim[i];
            }
        }
        /// <summary>
        /// create a new shape, copied from existing shape
        /// </summary>
        /// <param name="s"></param>
        public shape(shape s)
        {
            iDims = new npy_intp[s.iDims.Length];
            Array.Copy(s.iDims, iDims, s.iDims.Length);
        }
        /// <summary>
        /// create a 1D shape
        /// </summary>
        /// <param name="Dim1"></param>
        public shape(Int32 Dim1)
        {
            iDims = new npy_intp[1];
            iDims[0] = (npy_intp)Dim1;
        }
        /// <summary>
        /// create a 1D shape
        /// </summary>
        /// <param name="Dim1"></param>
        public shape(Int64 Dim1)
        {
            iDims = new npy_intp[1];
            iDims[0] = (npy_intp)Dim1;
        }
        /// <summary>
        /// create a 2D shape
        /// </summary>
        /// <param name="Dim1"></param>
        /// <param name="Dim2"></param>
        public shape(Int32 Dim1, Int32 Dim2)
        {
            iDims = new npy_intp[2];
            iDims[0] = (npy_intp)Dim1;
            iDims[1] = (npy_intp)Dim2;
        }
        /// <summary>
        /// create a 2D shape
        /// </summary>
        /// <param name="Dim1"></param>
        /// <param name="Dim2"></param>
        public shape(Int64 Dim1, Int64 Dim2)
        {
            iDims = new npy_intp[2];
            iDims[0] = (npy_intp)Dim1;
            iDims[1] = (npy_intp)Dim2;
        }
        /// <summary>
        /// create a 3D shape
        /// </summary>
        /// <param name="Dim1"></param>
        /// <param name="Dim2"></param>
        /// <param name="Dim3"></param>
        public shape(Int32 Dim1, Int32 Dim2, Int32 Dim3)
        {
            iDims = new npy_intp[3];
            iDims[0] = (npy_intp)Dim1;
            iDims[1] = (npy_intp)Dim2;
            iDims[2] = (npy_intp)Dim3;
        }
        /// <summary>
        /// create a 3D shape
        /// </summary>
        /// <param name="Dim1"></param>
        /// <param name="Dim2"></param>
        /// <param name="Dim3"></param>
        public shape(Int64 Dim1, Int64 Dim2, Int64 Dim3)
        {
            iDims = new npy_intp[3];
            iDims[0] = (npy_intp)Dim1;
            iDims[1] = (npy_intp)Dim2;
            iDims[2] = (npy_intp)Dim3;
        }
        /// <summary>
        /// create a 4D shape
        /// </summary>
        /// <param name="Dim1"></param>
        /// <param name="Dim2"></param>
        /// <param name="Dim3"></param>
        /// <param name="Dim4"></param>
        public shape(Int32 Dim1, Int32 Dim2, Int32 Dim3, Int32 Dim4)
        {
            iDims = new npy_intp[4];
            iDims[0] = (npy_intp)Dim1;
            iDims[1] = (npy_intp)Dim2;
            iDims[2] = (npy_intp)Dim3;
            iDims[3] = (npy_intp)Dim4;
        }
        /// <summary>
        /// create a 4D shape
        /// </summary>
        /// <param name="Dim1"></param>
        /// <param name="Dim2"></param>
        /// <param name="Dim3"></param>
        /// <param name="Dim4"></param>
        public shape(Int64 Dim1, Int64 Dim2, Int64 Dim3, Int64 Dim4)
        {
            iDims = new npy_intp[4];
            iDims[0] = (npy_intp)Dim1;
            iDims[1] = (npy_intp)Dim2;
            iDims[2] = (npy_intp)Dim3;
            iDims[3] = (npy_intp)Dim4;
        }
        /// <summary>
        /// return a string representation of this shape
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            StringBuilder buf = new StringBuilder();
            buf.Append("(");
            for (int i = 0; i < iDims.Length; i++)
            {
                if (i > 0) buf.Append(", ");
                buf.Append(iDims[i].ToString());
            }
            if (iDims.Length == 1) buf.Append(",");
            buf.Append(")");
            return buf.ToString();
        }

        /// <summary>
        /// add two shapes together
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static shape operator +(shape a, shape b)
        {
            npy_intp[] newdims = new npy_intp[a.iDims.Length + b.iDims.Length];

            Array.Copy(a.iDims, 0, newdims, 0, a.iDims.Length);
            Array.Copy(b.iDims, 0, newdims, a.iDims.Length, b.iDims.Length);

            return new shape(newdims);
        }

    }
 
}
