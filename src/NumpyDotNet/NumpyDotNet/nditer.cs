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
    public class nditer : IEnumerable, IEnumerator, IEnumerator<object>
    {
        NpyArrayMultiIterObject core = null;
        private NpyArrayMultiIterObject current;
        private int creationCount = 0;

        public nditer(ndarray a)
        {
            creationCount = 1;
            core = NpyCoreApi.MultiIterFromArrays(new ndarray[] { a, a });
        }

        public nditer(ValueTuple<ndarray, ndarray> arr)
        {
            core = NpyCoreApi.MultiIterFromArrays(new ndarray[] {arr.Item1, arr.Item2 });
            creationCount = core.numiter;
        }
        public nditer(ValueTuple<ndarray, ndarray, ndarray> arr)
        {
            core = NpyCoreApi.MultiIterFromArrays(new ndarray[] { arr.Item1, arr.Item2, arr.Item3 });
            creationCount = core.numiter;
        }
        public nditer(ValueTuple<ndarray, ndarray, ndarray, ndarray> arr)
        {
            core = NpyCoreApi.MultiIterFromArrays(new ndarray[] { arr.Item1, arr.Item2, arr.Item3, arr.Item4 });
            creationCount = core.numiter;
        }


        internal long Length
        {
            get { return core.size; }
        }

        public object index
        {
            get
            {
                return core.index;
            }
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


    public class ndindex : IEnumerable, IEnumerator, IEnumerator<object>
    {
        NpyArrayMapIterObject core = null;
        private NpyArrayMapIterObject current;
        private int creationCount = 0;

        public ndindex(object oshape)
        {

            shape newshape = NumpyExtensions.ConvertTupleToShape(oshape);
  
            creationCount = 1;
            core = NpyCoreApi.IterGetNewMap(GetIndexes(newshape.iDims), newshape.iDims.Length);

            ndarray temp = np.zeros(newshape, dtype: np.intp);

            var kk = NpyCoreApi.IterBindMap(core, temp, null);
        }

        private NpyIndex[] GetIndexes(npy_intp []dims)
        {
            object[] args = new object[1];
            args[0] = dims;
     

            NpyIndexes indexes = new NpyIndexes();
  
            NpyUtil_IndexProcessing.IndexConverter(null, args, indexes);

            return indexes.Indexes;
        }


        internal long Length
        {
            get { return core.size; }
        }

        public object index
        {
            get
            {
                return core.index;
            }
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
                NpyCoreApi.IterMapNext(core);
            }
            return (NpyCoreApi.MultiIterDone(core));
        }

        public void Reset()
        {
            current = null;
            NpyCoreApi.IterMapReset(core);
        }

        #endregion



        public void Dispose()
        {
        }



    }

}
