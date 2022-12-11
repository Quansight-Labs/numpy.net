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
using System.Numerics;
using NumpyLib;
using System.Collections;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{
    /// <summary>
    /// Efficient multi-dimensional iterator object to iterate over arrays.
    /// </summary>
    public class nditer : IEnumerable, IEnumerator, IEnumerator<object>
    {
        private NpyArrayMultiIterObject core = null;
        private NpyArrayMultiIterObject current;
        private int creationCount = 0;

        /// <summary>
        /// Efficient multi-dimensional iterator object to iterate over arrays.
        /// </summary>
        public nditer(ndarray a)
        {
            creationCount = 1;
            core = NpyCoreApi.MultiIterFromArrays(new ndarray[] { a, a });
        }
        /// <summary>
        /// Efficient multi-dimensional iterator object to iterate over arrays.
        /// </summary>
        public nditer(ValueTuple<ndarray, ndarray> arr)
        {
            core = NpyCoreApi.MultiIterFromArrays(new ndarray[] { arr.Item1, arr.Item2 });
            creationCount = core.numiter;
        }
        /// <summary>
        /// Efficient multi-dimensional iterator object to iterate over arrays.
        /// </summary>
        public nditer(ValueTuple<ndarray, ndarray, ndarray> arr)
        {
            core = NpyCoreApi.MultiIterFromArrays(new ndarray[] { arr.Item1, arr.Item2, arr.Item3 });
            creationCount = core.numiter;
        }
        /// <summary>
        /// Efficient multi-dimensional iterator object to iterate over arrays.
        /// </summary>
        public nditer(ValueTuple<ndarray, ndarray, ndarray, ndarray> arr)
        {
            core = NpyCoreApi.MultiIterFromArrays(new ndarray[] { arr.Item1, arr.Item2, arr.Item3, arr.Item4 });
            creationCount = core.numiter;
        }
        /// <summary>
        /// Efficient multi-dimensional iterator object to iterate over arrays.
        /// </summary>
        public nditer(ndarray[] arrays)
        {
            core = NpyCoreApi.MultiIterFromArrays(arrays);
            creationCount = core.numiter;
        }

        internal NpyArrayMultiIterObject iter
        {
            get { return core; }
        }

        public int ndim
        {
            get { return core.nd; }
        }
        public npy_intp index
        {
            get { return core.index; }
        }
        public int numiter
        {
            get { return core.numiter; }
        }
        public npy_intp size
        {
            get { return core.size; }
        }
        public npy_intp[] dimensions
        {
            get { return core.dimensions; }
        }

        public npy_intp[] coordinates(int index)
        {
            return core.iters[index].coordinates;
        }

        #region IEnumerator<object>

        public IEnumerator GetEnumerator()
        {
            return this;
        }

        public object Current
        {
            get
            {
                ndarray[] retArrays = new ndarray[creationCount];
                for (int i = 0; i < creationCount; i++)
                    retArrays[i] = SingleElementArray(current.iters[i]);
                
                return retArrays;
            }
        }

        private ndarray SingleElementArray(NpyArrayIterObject IterObject)
        {
            return np.array(IterObject.dataptr, 1);
        }

        public bool MoveNext()
        {
            if (current == null)
            {
                current = core;
            }
            else
            {
                NpyCoreApi.MultiIterNext(core);
            }
            return (NpyCoreApi.MultiIterDone(core));
        }

        public void IterNext()
        {
            NpyCoreApi.MultiIterNext(core);
        }

        public VoidPtr IterData(int index)
        {
            return NpyCoreApi.MultiIterData(core, index);
        }

        public void Reset()
        {
            current = null;
            NpyCoreApi.MultiIterReset(core);
        }

        #endregion

   

        public void Dispose()
        {
        }



    }

    /// <summary>
    /// An N-dimensional iterator object to index arrays.
    /// </summary>
    public class ndindex : IEnumerable, IEnumerator, IEnumerator<object>
    {
        nditer core;
        private nditer current;

        /// <summary>
        /// An N-dimensional iterator object to index arrays.
        /// </summary>
        public ndindex(object oshape)
        {

            shape newshape = NumpyExtensions.ConvertTupleToShape(oshape);

            var x = np.as_strided(np.zeros(1), shape: newshape.iDims, strides: (npy_intp[])np.zeros_like(newshape.iDims, dtype:np.intp).ToArray());

            core = new nditer(x);
        }


        #region IEnumerator<object>

        public IEnumerator GetEnumerator()
        {
            return this;
        }

        public object Current
        {
            get
            {
                npy_intp[] coords = new npy_intp[current.ndim];
                
                for (int i = 0; i < current.ndim; i++)
                {
                    coords[i] = current.coordinates(0)[i];
                }

                return coords;
            }
        }


        public bool MoveNext()
        {
            if (current == null)
            {
                current = core;
            }
 
            return (core.MoveNext());
        }

        public void Reset()
        {
            core.Reset();
        }

        #endregion



        public void Dispose()
        {
        }



    }

    /// <summary>
    /// Multidimensional index iterator.
    /// </summary>
    public class ndenumerate : IEnumerable, IEnumerator, IEnumerator<object>
    {
        flatiter core;
        private flatiter current;
        /// <summary>
        /// Multidimensional index iterator.
        /// </summary>
        /// <param name="a"></param>
        public ndenumerate(ndarray a)
        {
            core = np.asarray(a).Flat;
        }


        #region IEnumerator<object>

        public IEnumerator GetEnumerator()
        {
            return this;
        }

        public object Current
        {
            get
            {
                return (current.coords, core.Current);
            }
        }


        public bool MoveNext()
        {
            if (current == null)
            {
                current = core;
            }

            return (core.MoveNext());
        }

        public void Reset()
        {
            core.Reset();
        }

        #endregion



        public void Dispose()
        {
        }



    }

    public class broadcast : IEnumerable, IEnumerator, IEnumerator<object>
    {
        public npy_intp index
        {
            get { return core.index; }
        }
        public int ndim
        {
            get { return core.ndim; }
        }
        public int nd
        {
            get { return ndim; }
        }
        public int numiter
        {
            get { return core.numiter; }
        }

        public shape shape
        {
            get { return new shape(core.dimensions, core.ndim); }
        }

        public npy_intp size
        {
            get { return core.size; }
        }
        
        private nditer core = null;
        private nditer current;


        public broadcast(nditer iter)
        {
            core = iter;
        }


        #region IEnumerator<object>

        public IEnumerator GetEnumerator()
        {
            return this;
        }

        public object Current
        {
            get
            {
                return core.Current;
            }
        }
     
        public bool MoveNext()
        {
            if (current == null)
            {
                current = core;
            }

            return (core.MoveNext());
        }

        public VoidPtr IterData(int index)
        {
            return core.IterData(index);
        }
        public void IterNext()
        {
            core.IterNext();
        }
        public void Reset()
        {
            core.Reset();
        }

        #endregion

        public void Dispose()
        {
        }

    }

    public static partial class np
    {
        /// <summary>
        /// Produce an object that mimics broadcasting.
        /// </summary>
        /// <param name="aobjects">Input parameters.</param>
        /// <returns></returns>
        public static broadcast broadcast(params object[] aobjects)
        {
            ndarray[] arrays = new ndarray[aobjects.Length];

            for (int i = 0; i < aobjects.Length; i++)
            {
                arrays[i] = asanyarray(aobjects[i]);
            }

            var nditer = new nditer(arrays);

            broadcast bcast = new broadcast(nditer);
            return bcast;
        }
    }
}
