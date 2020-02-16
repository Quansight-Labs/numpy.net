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


namespace NumpyLib
{
    internal partial class numpyinternal
    {

        private static NpyArray NpyUFunc_PerformOuterOpArrayIter(NpyArray a, NpyArray b, NpyArray destArray, NumericOperations operations, NpyArray_Ops ops)
        {
            var destSize = NpyArray_Size(destArray);
            var aSize = NpyArray_Size(a);
            var bSize = NpyArray_Size(b);

            if (bSize == 0 || aSize == 0)
            {
                NpyArray_Resize(destArray, new NpyArray_Dims() { len = 0, ptr = new npy_intp[] { } }, false, NPY_ORDER.NPY_ANYORDER);
                return destArray;
            }

            switch (destArray.ItemType)
            {
                case NPY_TYPES.NPY_DOUBLE:
                {
                    UFUNC_Operations UFunc = new UFUNC_Double();
                    UFunc.PerformOuterOpArrayIter(a, b, destArray, operations, ops);
                    break;
                }
                default:
                {
                    var aIter = NpyArray_IterNew(a);
                    object[] aValues = new object[aSize];
                    for (long i = 0; i < aSize; i++)
                    {
                        aValues[i] = operations.srcGetItem(aIter.dataptr.data_offset - a.data.data_offset, a);
                        NpyArray_ITER_NEXT(aIter);
                    }

                    var bIter = NpyArray_IterNew(b);
                    object[] bValues = new object[bSize];
                    for (long i = 0; i < bSize; i++)
                    {
                        bValues[i] = operations.operandGetItem(bIter.dataptr.data_offset - b.data.data_offset, b);
                        NpyArray_ITER_NEXT(bIter);
                    }

                    var DestIter = NpyArray_IterNew(destArray);

                    for (long i = 0; i < aSize; i++)
                    {
                        var aValue = aValues[i];

                        for (long j = 0; j < bSize; j++)
                        {
                            var bValue = bValues[j];

                            object destValue = operations.operation(aValue, operations.ConvertOperand(aValue, bValue));

                            try
                            {
                                operations.destSetItem(DestIter.dataptr.data_offset - destArray.data.data_offset, destValue, destArray);
                            }
                            catch
                            {
                                operations.destSetItem(DestIter.dataptr.data_offset - destArray.data.data_offset, 0, destArray);
                            }
                            NpyArray_ITER_NEXT(DestIter);
                        }

                    }
                    break;
                }
            }

            if (HasBoolReturn(ops))
            {
                destArray = NpyArray_CastToType(destArray, NpyArray_DescrFromType(NPY_TYPES.NPY_BOOL), false);
            }
            return destArray;
        }

        private static bool HasBoolReturn(NpyArray_Ops ops)
        {
            switch (ops)
            {
                case NpyArray_Ops.npy_op_less:
                case NpyArray_Ops.npy_op_less_equal:
                case NpyArray_Ops.npy_op_equal:
                case NpyArray_Ops.npy_op_not_equal:
                case NpyArray_Ops.npy_op_greater:
                case NpyArray_Ops.npy_op_logical_or:
                case NpyArray_Ops.npy_op_logical_and:
                case NpyArray_Ops.npy_op_isnan:
                    return true;
                default:
                    return false;

            }
        }
    }
}
