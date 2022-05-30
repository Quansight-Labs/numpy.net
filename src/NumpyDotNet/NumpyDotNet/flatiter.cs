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
    public class flatiter : IEnumerable, IEnumerator, IEnumerator<object>
    {
        NpyArrayIterObject core = null;

        internal flatiter(NpyArrayIterObject coreIter)
        {
            core = coreIter;
            arr = new ndarray(core.ao);
        }

        public Object this[params object[] args]
        {
            get
            {
                ndarray result;

                NpyIndexes indexes = new NpyIndexes();
                {
                    NpyUtil_IndexProcessing.IndexConverter(this.arr.ravel(), args, indexes);
                    result = NpyCoreApi.IterSubscript(this, indexes);
                }
                if (result.ndim == 0)
                {
                    return result.GetItem(0);
                }
                else
                {
                    return result;
                }
            }

            set
            {
                NpyIndexes indexes = new NpyIndexes();
                {
                    NpyUtil_IndexProcessing.IndexConverter(this.arr.ravel(), args, indexes);

                    if (indexes.NumIndexes == 1)
                    {
                        // Special cases for single assigment.
                        switch (indexes.IndexType(0))
                        {
                            case NpyIndexType.NPY_INDEX_INTP:
                                SingleAssign(indexes.GetIntP(0), value);
                                return;
                            case NpyIndexType.NPY_INDEX_BOOL:
                                if (indexes.GetBool(0))
                                {
                                    SingleAssign(0, value);
                                }
                                return;
                            default:
                                break;
                        }
                    }

                    ndarray array_val = np.FromAny(value, arr.Dtype, 0, 0, 0, null);
                    NpyCoreApi.IterSubscriptAssign(this, indexes, array_val);
                }
            }
        }

        public object this[int index]
        {
            get
            {
                return Get(index);
            }
            set
            {
                SingleAssign(index, value);
            }
        }

        public object this[BigInteger index]
        {
            get
            {
#if NPY_INTP_64
                return Get(Convert.ToInt64(index));
#else
                return Get(Convert.ToInt32(index));
#endif
            }
            set
            {
                long lIndex = (long)index;
                SingleAssign((npy_intp)lIndex, value);
            }
        }

        public object this[bool index]
        {
            set
            {
                if (index)
                {
                    SingleAssign(0, value);
                }
            }
        }



        internal long Length
        {
            get { return core.size; }
        }

        public ndarray @base
        {
            get
            {
                return arr;
            }
        }

        public object index
        {
            get
            {
                return core.index;
            }
        }

        public npy_intp[] coords
        {
            get
            {
                int nd = arr.ndim;
                npy_intp[] result = new npy_intp[nd];
                npy_intp[] coords = NpyCoreApi.IterCoords(this);
                for (int i = 0; i < nd; i++)
                {
                    result[i] = coords[i];
                }
                return result;
            }
        }

        public ndarray copy()
        {
            return arr.Flatten(NPY_ORDER.NPY_CORDER);
        }

        /// <summary>
        /// Returns a contiguous, 1-d array that can be used to update the underlying array.  If the array
        /// is contiguous this is a 1-d view of the array.  Otherwise it is a copy with UPDATEIFCOPY set so that
        /// the data will be copied back when the returned array is freed.
        /// </summary>
        /// <returns></returns>
        public ndarray FlatView(object ignored = null)
        {
            return NpyCoreApi.FlatView(arr);
        }


        public VoidPtr CurrentPtr
        {
            get { return current; }
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
                npy_intp index = (current.data_offset - arr.Array.data.data_offset) >> arr.ItemSizeDiv;
                return arr.GetItem(index);
            }
            set
            {
                npy_intp index = (current.data_offset - arr.Array.data.data_offset) >> arr.ItemSizeDiv;
                arr.SetItem(value, index);
            }
        }

        public bool MoveNext()
        {
            if (current == null)
            {
                if ((core.dataptr.datap as Array).Length > 0)
                    current = core.dataptr;
            }
            else
            {
                current = NpyCoreApi.IterNext(core);
            }
            return (current != null);
        }

        public void Reset()
        {
            current = null;
            NpyCoreApi.IterReset(core);
        }

        #endregion

        #region internal methods

        internal void SingleAssign(npy_intp index, object value)
        {
            VoidPtr pos = NpyCoreApi.IterGoto1D(this, index);
            if (pos == null)
            {
                NpyCoreApi.CheckError();
            }
            arr.SetItem(value, index);
        }

        internal object Get(npy_intp index)
        {
            VoidPtr pos = NpyCoreApi.IterGoto1D(this, index);
            if (pos == null)
            {
                NpyCoreApi.CheckError();
            }
            return arr.GetItem(index);
        }

        public void Dispose()
        {
        }

        #endregion

        internal NpyArrayIterObject Iter
        {
            get
            {
                return core;
            }
        }

        private VoidPtr current;
        private ndarray arr;
    }
}
