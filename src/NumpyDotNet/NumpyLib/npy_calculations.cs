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
using size_t = System.UInt64;
using System.Threading;
using System.Collections.Concurrent;

#if NPY_INTP_64
using npy_intp = System.Int64;
using npy_ucs4 = System.Int64;
using NpyArray_UCS4 = System.UInt64;
#else
using npy_intp = System.Int32;
using npy_ucs4 = System.Int32;
using NpyArray_UCS4 = System.UInt32;
#endif

namespace NumpyLib
{

    internal partial class numpyinternal
    {
        const int NUMERICOPS_TASKSIZE = 1000;       // size of data to break into chunks
        const int NUMERICOPS_SMALL_TASKSIZE = 100;  // size of data to small to use parallel library


        internal static NumericOperation GetOperation(NpyArray srcArray, UFuncOperation operationType)
        {
            switch (operationType)
            {
                case UFuncOperation.add:
                    switch (srcArray.ItemType)
                    {
                        case NPY_TYPES.NPY_BOOL:
                            srcArray = NpyArray_CastToType(srcArray, NpyArray_DescrFromType(NPY_TYPES.NPY_INT32), false);
                            break;
                        default:
                            break;

                    }
                    break;
                default:
                    break;
            }

            return GetOperation(srcArray.data, operationType);
        }



        internal static NumericOperation GetOperation(VoidPtr vp, UFuncOperation operationType)
        {
            NPY_TYPES ItemType = vp.type_num;

            //Console.WriteLine("Getting calculation handler {0} for array type {1}", operationType, srcArray.ItemType);

            switch (operationType)
            {
                case UFuncOperation.add:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).AddOperation;
                }
                case UFuncOperation.subtract:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).SubtractOperation;
                }
                case UFuncOperation.multiply:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).MultiplyOperation;
                }
                case UFuncOperation.divide:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).DivideOperation;
                }
                case UFuncOperation.remainder:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).RemainderOperation;
                }

                case UFuncOperation.fmod:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).FModOperation;
                }

                case UFuncOperation.power:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).PowerOperation;
                }
                case UFuncOperation.square:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).SquareOperation;
                }
                case UFuncOperation.reciprocal:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).ReciprocalOperation;
                }
                case UFuncOperation.ones_like:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).OnesLikeOperation;
                }
                case UFuncOperation.sqrt:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).SqrtOperation;
                }
                case UFuncOperation.negative:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).NegativeOperation;
                }
                case UFuncOperation.absolute:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).AbsoluteOperation;
                }
                case UFuncOperation.invert:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).InvertOperation;
                }
                case UFuncOperation.left_shift:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).LeftShiftOperation;
                }
                case UFuncOperation.right_shift:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).RightShiftOperation;
                }
                case UFuncOperation.bitwise_and:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).BitWiseAndOperation;
                }
                case UFuncOperation.bitwise_xor:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).BitWiseXorOperation;
                }
                case UFuncOperation.bitwise_or:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).BitWiseOrOperation;
                }
                case UFuncOperation.less:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).LessOperation;
                }
                case UFuncOperation.less_equal:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).LessEqualOperation;
                }
                case UFuncOperation.equal:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).EqualOperation;
                }
                case UFuncOperation.not_equal:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).NotEqualOperation;
                }
                case UFuncOperation.greater:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).GreaterOperation;
                }
                case UFuncOperation.greater_equal:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).GreaterEqualOperation;
                }
                case UFuncOperation.isnan:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).IsNANOperation;
                }
                case UFuncOperation.floor_divide:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).FloorDivideOperation;
                }
                case UFuncOperation.true_divide:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).TrueDivideOperation;
                }
                case UFuncOperation.logical_or:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).LogicalOrOperation;
                }
                case UFuncOperation.logical_and:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).LogicalAndOperation;
                }
                case UFuncOperation.floor:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).FloorOperation;
                }
                case UFuncOperation.ceil:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).CeilingOperation;
                }
                case UFuncOperation.maximum:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).MaximumOperation;
                }
                case UFuncOperation.fmax:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).FMaxOperation;
                }
                case UFuncOperation.minimum:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).MinimumOperation;
                }
                case UFuncOperation.fmin:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).FMinOperation;
                }

                case UFuncOperation.heaviside:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).HeavisideOperation;
                }

                case UFuncOperation.rint:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).RintOperation;
                }
                case UFuncOperation.conjugate:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).ConjugateOperation;
                }

                default:
                    return null;
            }
        }



        internal static NpyArray NpyArray_PerformNumericOperation(UFuncOperation operationType, NpyArray x1Array, NpyArray x2Array, NpyArray outArray, NpyArray whereFilter)
        {
            if (outArray == null)
            {
                outArray = NpyArray_NumericOpArraySelection(x1Array, x2Array, operationType);
            }

            PerformNumericOpArray(x1Array, outArray, x2Array, operationType);

            return outArray;
        }

        private static NpyArray NpyArray_NumericOpArraySelection(NpyArray srcArray, NpyArray operandArray, UFuncOperation operationType)
        {
            NpyArray_Descr newtype = srcArray.descr;
            NPYARRAYFLAGS flags = srcArray.flags | NPYARRAYFLAGS.NPY_ENSURECOPY | NPYARRAYFLAGS.NPY_FORCECAST;

            var ArrayHandler = DefaultArrayHandlers.GetArrayHandler(srcArray.ItemType);

            if (!NpyArray_ISCOMPLEX(srcArray) && !NpyArray_ISOBJECT(srcArray) && !NpyArray_ISSTRING(srcArray))
            {
                if (operandArray != null)
                {
                    if (NpyArray_ISFLOAT(operandArray))
                    {
                        newtype = NpyArray_DescrFromType(DefaultArrayHandlers.GetArrayHandler(operandArray.descr.type_num).MathOpReturnType(UFuncOperation.special_operand_is_float));
                    }
                }
            }


            switch (operationType)
            {
                case UFuncOperation.add:
                {
                    break;
                }
                case UFuncOperation.subtract:
                {
                    break;
                }
                case UFuncOperation.multiply:
                {
                    break;
                }
                case UFuncOperation.divide:
                {
                    break;
                }
                case UFuncOperation.remainder:
                {
                    break;
                }
                case UFuncOperation.fmod:
                {
                    break;
                }
                case UFuncOperation.power:
                {
                    newtype = NpyArray_DescrFromType(ArrayHandler.MathOpReturnType(operationType));
                    break;
                }
                case UFuncOperation.square:
                {
                    break;
                }
                case UFuncOperation.reciprocal:
                {
                    break;
                }
                case UFuncOperation.ones_like:
                {
                    break;
                }
                case UFuncOperation.sqrt:
                {
                    newtype = NpyArray_DescrFromType(ArrayHandler.MathOpReturnType(operationType));
                    break;
                }
                case UFuncOperation.negative:
                {
                    break;
                }
                case UFuncOperation.absolute:
                {
                    break;
                }
                case UFuncOperation.invert:
                {
                    break;
                }
                case UFuncOperation.left_shift:
                {
                    break;
                }
                case UFuncOperation.right_shift:
                {
                    break;
                }
                case UFuncOperation.bitwise_and:
                {
                    break;
                }
                case UFuncOperation.bitwise_xor:
                {
                    break;
                }
                case UFuncOperation.bitwise_or:
                {
                    break;
                }
                case UFuncOperation.less:
                case UFuncOperation.less_equal:
                case UFuncOperation.equal:
                case UFuncOperation.not_equal:
                case UFuncOperation.greater:
                case UFuncOperation.greater_equal:
                case UFuncOperation.isnan:
                {
                    newtype = NpyArray_DescrFromType(NPY_TYPES.NPY_BOOL);
                    break;
                }

                case UFuncOperation.floor_divide:
                {
                    break;
                }
                case UFuncOperation.true_divide:
                {
                    newtype = NpyArray_DescrFromType(ArrayHandler.MathOpReturnType(operationType));
                    break;
                }
                case UFuncOperation.logical_or:
                {
                    break;
                }
                case UFuncOperation.logical_and:
                {
                    break;
                }
                case UFuncOperation.floor:
                {
                    break;
                }
                case UFuncOperation.ceil:
                {
                    break;
                }
                case UFuncOperation.maximum:
                {
                    break;
                }
                case UFuncOperation.minimum:
                {
                    break;
                }
                case UFuncOperation.rint:
                {
                    break;
                }
                case UFuncOperation.conjugate:
                {
                    break;
                }
            }

            var broadcastDims = GenerateBroadcastedDims(srcArray, operandArray);

            if (operandArray == null || NpyArray_Size(srcArray) >= NpyArray_Size(operandArray))
            {
                if (broadcastDims != null)
                {
                    return NpyArray_Alloc(newtype, broadcastDims.Length, broadcastDims, NpyArray_ISFORTRAN(srcArray), null);
                }
                else
                {
                    return NpyArray_Alloc(newtype, srcArray.nd, srcArray.dimensions, NpyArray_ISFORTRAN(srcArray), null);
                }

            }
            else
            {
                if (broadcastDims != null)
                {
                    return NpyArray_Alloc(newtype, broadcastDims.Length, broadcastDims, NpyArray_ISFORTRAN(operandArray), null);
                }
                else
                {
                    return NpyArray_Alloc(newtype, operandArray.nd, operandArray.dimensions, NpyArray_ISFORTRAN(operandArray), null);
                }
            }

        }

        public static npy_intp[] GenerateBroadcastedDims(NpyArray leftArray, NpyArray rightArray)
        {
            npy_intp i, nd, k, j, tmp;

            //is left a scalar
            if (leftArray.nd == 1 && leftArray.dimensions[0] == 1)
            {
                if (NpyArray_SIZE(rightArray) > 0)
                    return AdjustedDimensions(rightArray);
                return AdjustedDimensions(leftArray);
            }
            //is right a scalar
            else if (rightArray.nd == 1 && rightArray.dimensions[0] == 1)
            {
                if (NpyArray_SIZE(leftArray) > 0)
                    return AdjustedDimensions(leftArray);
                return AdjustedDimensions(rightArray);
            }
            else
            {
                tmp = 0;

                //this is the shared shape of the target broadcast
                nd = Math.Max(rightArray.nd, leftArray.nd);
                npy_intp[] newDimensions = new npy_intp[nd];

                // Discover the broadcast shape in each dimension 
                for (i = 0; i < nd; i++)
                {
                    newDimensions[i] = 1;

                    /* This prepends 1 to shapes not already equal to nd */
                    k = i + leftArray.nd - nd;
                    if (k >= 0)
                    {
                        tmp = leftArray.dimensions[k];
                        if (tmp == 1)
                        {
                            goto _continue;
                        }

                        if (newDimensions[i] == 1)
                        {
                            newDimensions[i] = tmp;
                        }
                        else if (newDimensions[i] != tmp)
                        {
                            throw new Exception("shape mismatch: objects cannot be broadcast to a single shape");
                        }
                    }

                    _continue:
                    /* This prepends 1 to shapes not already equal to nd */
                    k = i + rightArray.nd - nd;
                    if (k >= 0)
                    {
                        tmp = rightArray.dimensions[k];
                        if (tmp == 1)
                        {
                            continue;
                        }

                        if (newDimensions[i] == 1)
                        {
                            newDimensions[i] = tmp;
                        }
                        else if (newDimensions[i] != tmp)
                        {
                            throw new Exception("shape mismatch: objects cannot be broadcast to a single shape");
                        }
                    }

                }
                return newDimensions;
            }
        }

        private static npy_intp[] AdjustedDimensions(NpyArray Array)
        {
            if (Array.nd <= 0)
                return null;

            npy_intp[] Dims = new npy_intp[Array.nd];
            System.Array.Copy(Array.dimensions, Dims, Array.nd);
            return Dims;
        }

        internal static NpyArray NpyArray_NumericOpUpscaleSourceArray(NpyArray srcArray, NpyArray operandArray)
        {
            return NpyArray_NumericOpUpscaleSourceArray(srcArray, operandArray.dimensions, operandArray.nd);
        }

        internal static NpyArray NpyArray_NumericOpUpscaleSourceArray(NpyArray srcArray, npy_intp[] newdims, int nd)
        {
            if (srcArray != null && newdims != null)
            {
                npy_intp srcArraySize = NpyArray_SIZE(srcArray);
                npy_intp operandArraySize = NpyArray_MultiplyList(newdims, nd);

                if (srcArraySize < operandArraySize)
                {
                    if (operandArraySize % srcArraySize != 0)
                    {
                        throw new Exception(string.Format("Unable to broadcast array size {0} to array size {1}", srcArraySize, operandArraySize));
                    }

                    Int64 repeatNumber = operandArraySize / srcArraySize;

                    NpyArray repeatArray = NpyArray_Alloc(
                        NpyArray_DescrFromType(NPY_TYPES.NPY_INTP),
                        1, new npy_intp[] { 1 }, false, null);

                    npy_intp[] Data = repeatArray.data.datap as npy_intp[];
                    Data[0] = repeatNumber;

                    srcArray = NpyArray_Repeat(srcArray, repeatArray, -1);

                    NpyArray_Dims newDims = new NpyArray_Dims();
                    newDims.len = nd;
                    newDims.ptr = newdims;

                    srcArray = NpyArray_Newshape(srcArray, newDims, NPY_ORDER.NPY_ANYORDER);
                }
            }

            return srcArray;
        }

        #region scalar numeric functions



        #region PerformNumericOpScalarIter
        private static void PerformNumericOpScalarIter(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations, UFuncOperation op)
        {
            var destSize = NpyArray_Size(destArray);

            if (NpyArray_SIZE(operArray) == 0 || NpyArray_SIZE(srcArray) == 0)
            {
                NpyArray_Resize(destArray, new NpyArray_Dims() { len = 0, ptr = new npy_intp[] { } }, false, NPY_ORDER.NPY_ANYORDER);
                return;
            }

            bool handled = PerformNumericOpScalarAllSameType(destArray, srcArray, operArray, op);
            if (handled)
                return;

    

            var SrcIter = NpyArray_BroadcastToShape(srcArray, destArray.dimensions, destArray.nd);
            var DestIter = NpyArray_BroadcastToShape(destArray, destArray.dimensions, destArray.nd);
            var OperIter = NpyArray_BroadcastToShape(operArray, destArray.dimensions, destArray.nd);

            if (!SrcIter.requiresIteration && !DestIter.requiresIteration && !operArray.IsASlice)
            {
                PerformNumericOpScalarIterContiguousSD(srcArray, destArray, operArray, operations, SrcIter, DestIter, OperIter);
                return;
            }

            if (SrcIter.requiresIteration && !DestIter.requiresIteration && !operArray.IsASlice)
            {
                PerformNumericOpScalarIterContiguousD(srcArray, destArray, operArray, operations, SrcIter, DestIter, OperIter);
                return;
            }

            PerformNumericOpScalarSmallIter(srcArray, destArray, operArray, operations, SrcIter, DestIter, OperIter, destSize);
            return;
        }

        private static bool PerformNumericOpScalarAllSameType(NpyArray destArray, NpyArray srcArray, NpyArray operArray, UFuncOperation op)
        {
            IUFUNC_Operations UFunc = GetUFuncHandler(destArray.ItemType);
            if (UFunc != null)
            {
                if (destArray.ItemType == srcArray.ItemType && destArray.ItemType != NPY_TYPES.NPY_BOOL)
                {
                    if (operArray.ItemType != destArray.ItemType && NpyArray_SIZE(operArray) <= NUMERICOPS_TASKSIZE)
                    {
                        operArray = NpyArray_CastToType(operArray, NpyArray_DescrFromType(destArray.ItemType), NpyArray_ISFORTRAN(operArray));
                    }
                }
                if (destArray.ItemType == operArray.ItemType && destArray.ItemType != NPY_TYPES.NPY_BOOL)
                {
                    if (srcArray.ItemType != srcArray.ItemType && NpyArray_SIZE(srcArray) <= NUMERICOPS_TASKSIZE)
                    {
                        srcArray = NpyArray_CastToType(srcArray, NpyArray_DescrFromType(destArray.ItemType), NpyArray_ISFORTRAN(srcArray));
                    }
                }

                if (destArray.ItemType == srcArray.ItemType && destArray.ItemType == operArray.ItemType)
                {
                    UFunc.PerformScalarOpArrayIter(destArray, srcArray, operArray, op);
                    return true;
                }
            }

            return false;
    
        }

        private static void PerformNumericOpScalarSmallIter(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations, NpyArrayIterObject srcIter, NpyArrayIterObject destIter, NpyArrayIterObject operIter, npy_intp taskSize)
        {
            List<Exception> caughtExceptions = new List<Exception>();

            var srcParallelIters = NpyArray_ITER_ParallelSplit(srcIter);
            var destParallelIters = NpyArray_ITER_ParallelSplit(destIter);
            var operParallelIters = NpyArray_ITER_ParallelSplit(operIter);

            Parallel.For(0, destParallelIters.Count(), index =>
            //for (int index = 0; index < destParallelIters.Count(); index++) // 
            {
                var ldestIter = destParallelIters.ElementAt(index);
                var lsrcIter = srcParallelIters.ElementAt(index);
                var loperIter = operParallelIters.ElementAt(index);

                npy_intp srcDataOffset = srcArray.data.data_offset;
                npy_intp operDataOffset = operArray.data.data_offset;
                npy_intp destDataOffset = destArray.data.data_offset;

                //Parallel.For(0, taskSize, i =>
                while (ldestIter.index < ldestIter.size)
                {
                    try
                    {
                        var srcValue = operations.srcGetItem(lsrcIter.dataptr.data_offset - srcDataOffset, srcArray);
                        var operValue = operations.operandGetItem(loperIter.dataptr.data_offset - operDataOffset, operArray);

                        object destValue = null;

                        destValue = operations.operation(srcValue, operations.ConvertOperand(srcValue, operValue));

                        try
                        {
                            operations.destSetItem(ldestIter.dataptr.data_offset - destDataOffset, destValue, destArray);
                        }
                        catch
                        {
                            operations.destSetItem(ldestIter.dataptr.data_offset - destDataOffset, 0, destArray);
                        }
                    }
                    catch (Exception ex)
                    {
                        caughtExceptions.Add(ex);
                    }

                    NpyArray_ITER_PARALLEL_NEXT(ldestIter);
                    NpyArray_ITER_PARALLEL_NEXT(lsrcIter);
                    NpyArray_ITER_PARALLEL_NEXT(loperIter);
                } //);
            });


            if (caughtExceptions.Count > 0)
            {
                throw caughtExceptions[0];
            }

        }

        private static void PerformNumericOpScalarIterContiguousSD(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations, NpyArrayIterObject srcIter, NpyArrayIterObject destIter, NpyArrayIterObject operIter)
        {
            switch (srcArray.ItemType)
            {
                case NPY_TYPES.NPY_BOOL:
                    PerformNumericOpScalarIterContiguousSD_T1<bool>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    PerformNumericOpScalarIterContiguousSD_T1<sbyte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    PerformNumericOpScalarIterContiguousSD_T1<byte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT16:
                    PerformNumericOpScalarIterContiguousSD_T1<Int16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    PerformNumericOpScalarIterContiguousSD_T1<UInt16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT32:
                    PerformNumericOpScalarIterContiguousSD_T1<Int32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    PerformNumericOpScalarIterContiguousSD_T1<UInt32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT64:
                    PerformNumericOpScalarIterContiguousSD_T1<Int64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    PerformNumericOpScalarIterContiguousSD_T1<UInt64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    PerformNumericOpScalarIterContiguousSD_T1<float>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    PerformNumericOpScalarIterContiguousSD_T1<double>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    PerformNumericOpScalarIterContiguousSD_T1<decimal>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    PerformNumericOpScalarIterContiguousSD_T1<System.Numerics.Complex>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    PerformNumericOpScalarIterContiguousSD_T1<System.Numerics.BigInteger>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    PerformNumericOpScalarIterContiguousSD_T1<object>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_STRING:
                    PerformNumericOpScalarIterContiguousSD_T1<string>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
            }
        }

        private static void PerformNumericOpScalarIterContiguousD(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations, NpyArrayIterObject srcIter, NpyArrayIterObject destIter, NpyArrayIterObject operIter)
        {
            switch (srcArray.ItemType)
            {
                case NPY_TYPES.NPY_BOOL:
                    PerformNumericOpScalarIterContiguousD_T1<bool>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    PerformNumericOpScalarIterContiguousD_T1<sbyte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    PerformNumericOpScalarIterContiguousD_T1<byte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT16:
                    PerformNumericOpScalarIterContiguousD_T1<Int16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    PerformNumericOpScalarIterContiguousD_T1<UInt16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT32:
                    PerformNumericOpScalarIterContiguousD_T1<Int32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    PerformNumericOpScalarIterContiguousD_T1<UInt32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT64:
                    PerformNumericOpScalarIterContiguousD_T1<Int64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    PerformNumericOpScalarIterContiguousD_T1<UInt64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    PerformNumericOpScalarIterContiguousD_T1<float>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    PerformNumericOpScalarIterContiguousD_T1<double>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    PerformNumericOpScalarIterContiguousD_T1<decimal>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    PerformNumericOpScalarIterContiguousD_T1<System.Numerics.Complex>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    PerformNumericOpScalarIterContiguousD_T1<System.Numerics.BigInteger>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    PerformNumericOpScalarIterContiguousD_T1<object>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_STRING:
                    PerformNumericOpScalarIterContiguousD_T1<string>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;

            }
        }

        private static void PerformNumericOpScalarIterContiguousSD_T1<S>(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations, NpyArrayIterObject srcIter, NpyArrayIterObject destIter, NpyArrayIterObject operIter)
        {
            switch (destArray.ItemType)
            {
                case NPY_TYPES.NPY_BOOL:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, bool>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, sbyte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, byte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT16:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, Int16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, UInt16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT32:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, Int32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, UInt32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT64:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, Int64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, UInt64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, float>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, double>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, decimal>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, System.Numerics.Complex>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, System.Numerics.BigInteger>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, object>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_STRING:
                    PerformNumericOpScalarIterContiguous_SD_T2<S, string>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;

            }
        }

        private static void PerformNumericOpScalarIterContiguousD_T1<S>(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations, NpyArrayIterObject srcIter, NpyArrayIterObject destIter, NpyArrayIterObject operIter)
        {
            switch (destArray.ItemType)
            {
                case NPY_TYPES.NPY_BOOL:
                    PerformNumericOpScalarIterContiguous_D_T2<S, bool>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    PerformNumericOpScalarIterContiguous_D_T2<S, sbyte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    PerformNumericOpScalarIterContiguous_D_T2<S, byte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT16:
                    PerformNumericOpScalarIterContiguous_D_T2<S, Int16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    PerformNumericOpScalarIterContiguous_D_T2<S, UInt16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT32:
                    PerformNumericOpScalarIterContiguous_D_T2<S, Int32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    PerformNumericOpScalarIterContiguous_D_T2<S, UInt32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT64:
                    PerformNumericOpScalarIterContiguous_D_T2<S, Int64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    PerformNumericOpScalarIterContiguous_D_T2<S, UInt64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    PerformNumericOpScalarIterContiguous_D_T2<S, float>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    PerformNumericOpScalarIterContiguous_D_T2<S, double>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    PerformNumericOpScalarIterContiguous_D_T2<S, decimal>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    PerformNumericOpScalarIterContiguous_D_T2<S, System.Numerics.Complex>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    PerformNumericOpScalarIterContiguous_D_T2<S, System.Numerics.BigInteger>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    PerformNumericOpScalarIterContiguous_D_T2<S, object>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_STRING:
                    PerformNumericOpScalarIterContiguous_D_T2<S, string>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;

            }
        }

        private static void PerformNumericOpScalarIterContiguous_SD_T2<S, D>(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations, NpyArrayIterObject srcIter, NpyArrayIterObject destIter, NpyArrayIterObject operIter)
        {
            switch (operArray.ItemType)
            {
                case NPY_TYPES.NPY_BOOL:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, bool>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, sbyte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, byte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT16:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, Int16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, UInt16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT32:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, Int32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, UInt32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT64:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, Int64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, UInt64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, float>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, double>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, decimal>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, System.Numerics.Complex>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, System.Numerics.BigInteger>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, object>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_STRING:
                    PerformNumericOpScalarIterContiguousSD_T3<S, D, string>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;

            }
        }

        private static void PerformNumericOpScalarIterContiguous_D_T2<S, D>(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations, NpyArrayIterObject srcIter, NpyArrayIterObject destIter, NpyArrayIterObject operIter)
        {
            switch (operArray.ItemType)
            {
                case NPY_TYPES.NPY_BOOL:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, bool>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, sbyte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, byte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT16:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, Int16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, UInt16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT32:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, Int32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, UInt32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT64:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, Int64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, UInt64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, float>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, double>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, decimal>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, System.Numerics.Complex>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, System.Numerics.BigInteger>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, object>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_STRING:
                    PerformNumericOpScalarIterContiguousD_T3<S, D, string>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;

            }
        }

        private static void PerformNumericOpScalarIterContiguousSD_T3<S, D, O>(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations, NpyArrayIterObject srcIter, NpyArrayIterObject destIter, NpyArrayIterObject operIter)
        {
            S[] src = srcArray.data.datap as S[];
            D[] dest = destArray.data.datap as D[];
            O[] oper = operArray.data.datap as O[];


            int srcAdjustment = (int)srcArray.data.data_offset / srcArray.ItemSize;
            int destAdjustment = (int)destArray.data.data_offset / destArray.ItemSize;

            var exceptions = new ConcurrentQueue<Exception>();

            var loopCount = NpyArray_Size(destArray);

            if (NpyArray_Size(operArray) == 1 && !operArray.IsASlice)
            {
                object operand = operations.ConvertOperand(src[0], oper[0]);

                Parallel.For(0, loopCount, index =>
                {
                    try
                    {
                        try
                        {
                            var dValue = (D)(dynamic)operations.operation(src[index - srcAdjustment], operand);
                            dest[index - destAdjustment] = dValue;
                        }
                        catch (System.OverflowException of)
                        {
                            dest[index - destAdjustment] = default(D);
                        }
                    }
                    catch (Exception ex)
                    {
                        exceptions.Enqueue(ex);
                    }
                });
            }
            else
            {
                var IterableArraySize = CalculateIterationArraySize(operArray, destArray);
                var operOffsets = new Int32[IterableArraySize];
                NpyArray_ITER_TOARRAY(operIter, operArray, operOffsets, operOffsets.Length);

                Parallel.For(0, loopCount, index =>
                {
                    try
                    {
                        try
                        {
                            int operandIndex = (int)(index < operOffsets.Length ? index : (index % operOffsets.Length));
                            object operand = operations.ConvertOperand(src[0], operations.operandGetItem(operOffsets[operandIndex], operArray));

                            D dValue = (D)(dynamic)operations.operation(src[index - srcAdjustment], operand);

                            dest[index - destAdjustment] = dValue;
                        }
                        catch (System.OverflowException of)
                        {
                            dest[index - destAdjustment] = default(D);
                        }
                    }
                    catch (Exception ex)
                    {
                        exceptions.Enqueue(ex);
                    }
                });

            }

            if (exceptions.Count > 0)
            {
                throw exceptions.ElementAt(0);
            }

        }

        private static void PerformNumericOpScalarIterContiguousD_T3<S, D, O>(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations, NpyArrayIterObject srcIter, NpyArrayIterObject destIter, NpyArrayIterObject operIter)
        {
            S[] src = srcArray.data.datap as S[];
            D[] dest = destArray.data.datap as D[];
            O[] oper = operArray.data.datap as O[];

   
            var srcItemSize = srcArray.ItemSize;

            int srcAdjustment = (int)srcArray.data.data_offset / srcArray.ItemSize;
            int destAdjustment = (int)destArray.data.data_offset / destArray.ItemSize;

            var exceptions = new ConcurrentQueue<Exception>();


            var loopCount = NpyArray_Size(destArray);

            if (NpyArray_Size(operArray) == 1 && !operArray.IsASlice)
            {
                var srcParallelIters = NpyArray_ITER_ParallelSplit(srcIter);
                var destParallelIters = NpyArray_ITER_ParallelSplit(destIter);

                Parallel.For(0, destParallelIters.Count(), index =>
                {
                    var ldestIter = destParallelIters.ElementAt(index);
                    var lsrcIter = srcParallelIters.ElementAt(index);

                    object operand = operations.ConvertOperand(src[0], oper[0]);

                    while (ldestIter.index < ldestIter.size)
                    {

                        try
                        {
                            var srcIndex = (lsrcIter.dataptr.data_offset) / srcItemSize;
                            D dValue = (D)(dynamic)operations.operation(src[srcIndex], operand);
                            dest[ldestIter.index - destAdjustment] = dValue;
                        }
                        catch (System.OverflowException of)
                        {
                            dest[ldestIter.index - destAdjustment] = default(D);
                        }
                        catch (Exception ex)
                        {
                            exceptions.Enqueue(ex);
                        }

                        NpyArray_ITER_PARALLEL_NEXT(ldestIter);
                        NpyArray_ITER_PARALLEL_NEXT(lsrcIter);
                    }
                });
            }
            else
            {
                var srcParallelIters = NpyArray_ITER_ParallelSplit(srcIter);
                var destParallelIters = NpyArray_ITER_ParallelSplit(destIter);
                var operParallelIters = NpyArray_ITER_ParallelSplit(operIter);

                Parallel.For(0, destParallelIters.Count(), index =>
                {
                    var ldestIter = destParallelIters.ElementAt(index);
                    var lsrcIter = srcParallelIters.ElementAt(index);
                    var loperIter = operParallelIters.ElementAt(index);


                    while (ldestIter.index < ldestIter.size)
                    {

                        try
                        {
                            object operand = operations.ConvertOperand(src[0], operations.operandGetItem(loperIter.dataptr.data_offset, operArray));

                            var srcIndex = (lsrcIter.dataptr.data_offset) / srcItemSize;
                            D dValue = (D)(dynamic)operations.operation(src[srcIndex], operand);
                            dest[ldestIter.index - destAdjustment] = dValue;
                        }
                        catch (System.OverflowException of)
                        {
                            dest[ldestIter.index - destAdjustment] = default(D);
                        }
                        catch (Exception ex)
                        {
                            exceptions.Enqueue(ex);
                        }

                        NpyArray_ITER_PARALLEL_NEXT(ldestIter);
                        NpyArray_ITER_PARALLEL_NEXT(lsrcIter);
                        NpyArray_ITER_PARALLEL_NEXT(loperIter);
                    }
                });

            }

            if (exceptions.Count > 0)
            {
                throw exceptions.ElementAt(0);
            }

        }

        // calculate the smallest possible array that allows
        // Array to be correctly broadcasted into destArray
        private static long CalculateIterationArraySize(NpyArray Array, NpyArray destArray)
        {
            var OperIter = NpyArray_BroadcastToShape(Array, destArray.dimensions, destArray.nd);
            return NpyArray_ITER_COUNT(OperIter);
        }


        #endregion

   

        private static int PerformNumericOpScalar2(NpyArray srcArray, NpyArray destArray, double operand, NumericOperations operations)
        {
            npy_intp size;

            size = NpyArray_SIZE(srcArray);
            if (size == 0)
            {
                return 0;
            }
            VoidPtr srcPtr = new VoidPtr(srcArray);
            VoidPtr destPtr = new VoidPtr(destArray);

            if (NpyArray_ISONESEGMENT(srcArray))
            {

                while (size > 0)
                {
                    var aValue =  operations.srcGetItem(srcPtr.data_offset, srcArray);

                    var destValue = operations.operation(aValue, operations.ConvertOperand(aValue, operand));
                    try
                    {
                        operations.destSetItem(destPtr.data_offset, destValue, destArray);
                    }
                    catch (Exception ex)
                    {
                        operations.destSetItem(destPtr.data_offset, 0, destArray);
                    }

                    srcPtr.data_offset += srcArray.ItemSize;
                    destPtr.data_offset += destArray.ItemSize;
                    size -= 1;
                }
            }
            else
            {
                NpyArrayIterObject srcIter;
                NpyArrayIterObject destIter;

                srcIter = NpyArray_IterNew(srcArray);
                if (srcIter == null)
                {
                    return -1;
                }
                destIter = NpyArray_IterNew(destArray);
                if (destIter == null)
                {
                    return -1;
                }

                srcIter.dataptr.data_offset = 0;
                destIter.dataptr.data_offset = 0;
                while (size > 0)
                {
                    var aValue = operations.srcGetItem(srcIter.dataptr.data_offset, srcArray);

                    var destValue = operations.operation(aValue, operations.ConvertOperand(aValue, operand));
                    operations.destSetItem(destIter.dataptr.data_offset, destValue, destArray);
                    NpyArray_ITER_NEXT(srcIter);
                    NpyArray_ITER_NEXT(destIter);
                    size -= 1;
                }
                Npy_DECREF(srcIter);
                Npy_DECREF(destIter);
            }
            return 0;
        }

        #endregion

        #region array to array numeric functions
        public static void PerformNumericOpArray(NpyArray srcArray, NpyArray destArray, NpyArray operandArray, UFuncOperation operationType)
        {
            NumericOperation operation = GetOperation(srcArray, operationType);

            NumericOperations operations = NumericOperations.GetOperations(operation, srcArray, destArray, operandArray);
   
            PerformNumericOpScalarIter(srcArray, destArray, operandArray, operations, operationType);
        }

        public static NpyArray PerformOuterOpArray(NpyArray srcArray,  NpyArray operandArray, NpyArray destArray, UFuncOperation operationType)
        {
            NumericOperation operation = GetOperation(srcArray, operationType);

            NumericOperations operations = NumericOperations.GetOperations(operation, srcArray, destArray, operandArray);
  
            return NpyUFunc_PerformOuterOpArrayIter(srcArray, operandArray, destArray, operations, operationType);
        }


        internal static NpyArray PerformReduceOpArray(NpyArray srcArray, int axis, UFuncOperation ops, NPY_TYPES rtype, NpyArray outPtr, bool keepdims)
        {
            NpyArray ret = null;
            NpyArray newArray = null;

            if (null == (newArray = NpyArray_CheckAxis(srcArray, ref axis, 0)))
            {
                return null;
            }
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(ops),
                                            newArray, null, outPtr, axis,
                                            NpyArray_DescrFromType(rtype),
                                            GenericReductionOp.NPY_UFUNC_REDUCE, keepdims);
            Npy_DECREF(newArray);
            return ret;
        }

        internal static NpyArray PerformReduceAtOpArray(NpyArray srcArray, NpyArray indices, int axis, UFuncOperation ops, NPY_TYPES rtype, NpyArray outPtr)
        {
            NpyArray ret = null;
            NpyArray newArray = null;

            if (null == (newArray = NpyArray_CheckAxis(srcArray, ref axis, 0)))
            {
                return null;
            }
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(ops),
                                            newArray, indices, outPtr, axis,
                                            NpyArray_DescrFromType(rtype),
                                            GenericReductionOp.NPY_UFUNC_REDUCEAT, false);
            Npy_DECREF(newArray);
            return ret;
        }

        internal static NpyArray PerformAccumulateOpArray(NpyArray srcArray, int axis, UFuncOperation ops, NPY_TYPES rtype, NpyArray outPtr)
        {
            NpyArray ret = null;
            NpyArray newArray = null;

            if (null == (newArray = NpyArray_CheckAxis(srcArray, ref axis, 0)))
            {
                return null;
            }
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(ops),
                                            newArray, null, outPtr, axis,
                                            NpyArray_DescrFromType(rtype),
                                            GenericReductionOp.NPY_UFUNC_ACCUMULATE, false);
            Npy_DECREF(newArray);
            return ret;
        }


        #endregion
 
  
        internal static NpyArray NpyArray_ArgMax(NpyArray op, int axis, NpyArray outPtr)
        {
            NpyArray ap = null, rp = null;
            NpyArray_ArgFunc arg_func;
            VoidPtr ip;
            npy_intp[] rptr;
            npy_intp n, m;
            int elsize;
            bool copyret = false;
            int i;


            if ((ap = NpyArray_CheckAxis(op, ref axis, 0)) == null)
            {
                return null;
            }

            /*
             * We need to permute the array so that axis is placed at the end.
             * And all other dimensions are shifted left.
             */
            if (axis != ap.nd - 1)
            {
                NpyArray_Dims newaxes = new NpyArray_Dims();
                npy_intp[] dims = new npy_intp[npy_defs.NPY_MAXDIMS];

                newaxes.ptr = dims;
                newaxes.len = ap.nd;
                for (i = 0; i < axis; i++)
                    dims[i] = i;
                for (i = axis; i < ap.nd - 1; i++)
                    dims[i] = i + 1;
                dims[ap.nd - 1] = axis;
                op = NpyArray_Transpose(ap, newaxes);
                Npy_DECREF(ap);
                if (op == null)
                {
                    return null;
                }
            }
            else
            {
                op = ap;
            }

            /* Will get native-byte order contiguous copy. */
            ap = NpyArray_ContiguousFromArray(op, op.descr.type_num);
            Npy_DECREF(op);
            if (ap == null)
            {
                return null;
            }
            arg_func = ap.descr.f.argmax;
            if (arg_func == null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_TypeError, "data type not ordered");
                goto fail;
            }
            elsize = ap.descr.elsize;
            m = ap.dimensions[ap.nd - 1];
            if (m == 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "attempt to get argmax/argmin of an empty sequence");
                goto fail;
            }

            if (outPtr == null)
            {
                rp = NpyArray_New(null, ap.nd - 1,
                                  ap.dimensions, NPY_TYPES.NPY_INTP,
                                  null, null, 0, 0, Npy_INTERFACE(ap));
                if (rp == null)
                {
                    goto fail;
                }
            }
            else
            {
                if (NpyArray_SIZE(outPtr) != NpyArray_MultiplyList(ap.dimensions, ap.nd - 1))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_TypeError, "invalid shape for output array.");
                }
                rp = NpyArray_FromArray(outPtr,
                                  NpyArray_DescrFromType(NPY_TYPES.NPY_INTP),
                                  NPYARRAYFLAGS.NPY_CARRAY | NPYARRAYFLAGS.NPY_UPDATEIFCOPY);
                if (rp == null)
                {
                    goto fail;
                }
                if (rp != outPtr)
                {
                    copyret = true;
                }
            }

            n = NpyArray_SIZE(ap) / m;
            rptr = rp.data.datap as npy_intp[];

            for (ip = new VoidPtr(ap.data), i = 0; i < n; i++, ip.data_offset += elsize * m)
            {
                rptr[i] = arg_func(ip, ip.data_offset / elsize, m);
            }

            Npy_DECREF(ap);
            if (copyret)
            {
                NpyArray obj;
                obj = rp.base_arr;
                Npy_INCREF(obj);
                NpyArray_ForceUpdate(rp);
                Npy_DECREF(rp);
                rp = obj;
            }
            return rp;

            fail:
            Npy_DECREF(ap);
            Npy_XDECREF(rp);
            return null;
        }

        internal static NpyArray NpyArray_ArgMin(NpyArray op, int axis, NpyArray outPtr)
        {
            NpyArray ap = null, rp = null;
            NpyArray_ArgFunc arg_func;
            VoidPtr ip;
            npy_intp[] rptr;
            npy_intp n, m;
            int elsize;
            bool copyret = false;
            int i;


            if ((ap = NpyArray_CheckAxis(op, ref axis, 0)) == null)
            {
                return null;
            }

            /*
             * We need to permute the array so that axis is placed at the end.
             * And all other dimensions are shifted left.
             */
            if (axis != ap.nd - 1)
            {
                NpyArray_Dims newaxes = new NpyArray_Dims();
                npy_intp[] dims = new npy_intp[npy_defs.NPY_MAXDIMS];

                newaxes.ptr = dims;
                newaxes.len = ap.nd;
                for (i = 0; i < axis; i++)
                    dims[i] = i;
                for (i = axis; i < ap.nd - 1; i++)
                    dims[i] = i + 1;
                dims[ap.nd - 1] = axis;
                op = NpyArray_Transpose(ap, newaxes);
                Npy_DECREF(ap);
                if (op == null)
                {
                    return null;
                }
            }
            else
            {
                op = ap;
            }

            /* Will get native-byte order contiguous copy. */
            ap = NpyArray_ContiguousFromArray(op, op.descr.type_num);
            Npy_DECREF(op);
            if (ap == null)
            {
                return null;
            }
            arg_func = ap.descr.f.argmin;
            if (arg_func == null)
            {
                NpyErr_SetString(npyexc_type.NpyExc_TypeError, "data type not ordered");
                goto fail;
            }
            elsize = ap.descr.elsize;
            m = ap.dimensions[ap.nd - 1];
            if (m == 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "attempt to get argmax/argmin of an empty sequence");
                goto fail;
            }

            if (outPtr == null)
            {
                rp = NpyArray_New(null, ap.nd - 1,
                                  ap.dimensions, NPY_TYPES.NPY_INTP,
                                  null, null, 0, 0, Npy_INTERFACE(ap));
                if (rp == null)
                {
                    goto fail;
                }
            }
            else
            {
                if (NpyArray_SIZE(outPtr) != NpyArray_MultiplyList(ap.dimensions, ap.nd - 1))
                {
                    NpyErr_SetString(npyexc_type.NpyExc_TypeError, "invalid shape for output array.");
                }
                rp = NpyArray_FromArray(outPtr,
                                  NpyArray_DescrFromType(NPY_TYPES.NPY_INTP),
                                  NPYARRAYFLAGS.NPY_CARRAY | NPYARRAYFLAGS.NPY_UPDATEIFCOPY);
                if (rp == null)
                {
                    goto fail;
                }
                if (rp != outPtr)
                {
                    copyret = true;
                }
            }

            n = NpyArray_SIZE(ap) / m;
            rptr = rp.data.datap as npy_intp[];

            for (ip = new VoidPtr(ap.data), i = 0; i < n; i++, ip.data_offset += elsize * m)
            {
                rptr[i] = arg_func(ip, ip.data_offset / elsize, m);
            }

            Npy_DECREF(ap);
            if (copyret)
            {
                NpyArray obj;
                obj = rp.base_arr;
                Npy_INCREF(obj);
                NpyArray_ForceUpdate(rp);
                Npy_DECREF(rp);
                rp = obj;
            }
            return rp;

            fail:
            Npy_DECREF(ap);
            Npy_XDECREF(rp);
            return null;
        }


        internal static NpyArray NpyArray_Max(NpyArray srcArray, int axis, NpyArray outPtr, bool keepdims)
        {
            NpyArray ret = null;
            NpyArray newArray = null;

            if (null == (newArray = NpyArray_CheckAxis(srcArray, ref axis, 0)))
            {
                return null;
            }
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(UFuncOperation.maximum),
                                            newArray, null, outPtr, axis,
                                            newArray.descr,
                                            GenericReductionOp.NPY_UFUNC_REDUCE, keepdims);
            Npy_DECREF(newArray);
            return ret;
        }

        internal static NpyArray NpyArray_Min(NpyArray srcArray, int axis, NpyArray outPtr, bool keepdims)
        {
            NpyArray ret = null;
            NpyArray newArray = null;

            if (null == (newArray = NpyArray_CheckAxis(srcArray, ref axis, 0)))
            {
                return null;
            }
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(UFuncOperation.minimum),
                                            newArray, null, outPtr, axis,
                                            newArray.descr,
                                            GenericReductionOp.NPY_UFUNC_REDUCE, keepdims);
            Npy_DECREF(newArray);
            return ret;
        }

        internal static NpyArray NpyArray_Sum(NpyArray srcArray, int axis, NPY_TYPES rtype, NpyArray outPtr, bool keepdims)
        {
            NpyArray ret = null;
            NpyArray newArray = null;

            if (null == (newArray = NpyArray_CheckAxis(srcArray, ref axis, 0)))
            {
                return null;
            }
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(UFuncOperation.add),
                                            newArray, null, outPtr, axis,
                                            NpyArray_DescrFromType(rtype),
                                            GenericReductionOp.NPY_UFUNC_REDUCE, keepdims);
            Npy_DECREF(newArray);
            return ret;
        }


        internal static NpyArray NpyArray_Prod(NpyArray srcArray, int axis, NPY_TYPES rtype, NpyArray outPtr, bool keepdims)
        {
            NpyArray ret = null;
            NpyArray newArray = null;

            if (null == (newArray = NpyArray_CheckAxis(srcArray, ref axis, 0)))
            {
                return null;
            }
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(UFuncOperation.multiply),
                                            newArray, null, outPtr, axis,
                                            NpyArray_DescrFromType(rtype),
                                            GenericReductionOp.NPY_UFUNC_REDUCE, keepdims);
            Npy_DECREF(newArray);
            return ret;
        }

        internal static NpyArray NpyArray_CumSum(NpyArray srcArray, int axis, NPY_TYPES rtype, NpyArray outPtr)
        {
            NpyArray newArray = null;
            NpyArray ret = null;

            if (null == (newArray = NpyArray_CheckAxis(srcArray, ref axis, 0)))
            {
                return null;
            }


  
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(UFuncOperation.add),
                                            newArray, null, outPtr, axis,
                                            NpyArray_DescrFromType(rtype),
                                            GenericReductionOp.NPY_UFUNC_ACCUMULATE, false);
            Npy_DECREF(newArray);
            return ret;
        }


        internal static NpyArray NpyArray_Floor(NpyArray srcArray, NpyArray outPtr)
        {
            NumericOperation operation = GetOperation(srcArray, UFuncOperation.floor);

            if (outPtr == null)
            {
                int axis = -1;
                if (null == (outPtr = NpyArray_CheckAxis(srcArray, ref axis, NPYARRAYFLAGS.NPY_ENSURECOPY)))
                {
                    return null;
                }
            }

            NumericOperations operations = NumericOperations.GetOperations(operation, srcArray, outPtr, null);
   
            object floor = 0;
            NpyUFunc_PerformUFunc(srcArray, outPtr, ref floor, outPtr.dimensions, 0, 0, 0, operations);


            Npy_DECREF(outPtr);
            return NpyArray_Flatten(outPtr, NPY_ORDER.NPY_ANYORDER);
        }

        internal static NpyArray NpyArray_IsNaN(NpyArray srcArray)
        {
            NumericOperation operation = GetOperation(srcArray, UFuncOperation.isnan);

            NpyArray outPtr = NpyArray_FromArray(srcArray, NpyArray_DescrFromType(NPY_TYPES.NPY_BOOL), NPYARRAYFLAGS.NPY_CONTIGUOUS | NPYARRAYFLAGS.NPY_FORCECAST);

            NumericOperations operations = NumericOperations.GetOperations(operation, srcArray, outPtr, null);

            object floor = 0;
            NpyUFunc_PerformUFunc(srcArray, outPtr, ref floor, outPtr.dimensions, 0, 0, 0, operations);


            return outPtr;
        }

 
        internal static NpyArray NpyArray_CumProd(NpyArray srcArray, int axis, NPY_TYPES rtype, NpyArray outPtr)
        {
            NpyArray newArray = null;
            NpyArray ret = null;

            if (null == (newArray = NpyArray_CheckAxis(srcArray, ref axis, 0)))
            {
                return null;
            }
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(UFuncOperation.multiply),
                                            newArray, null, outPtr, axis,
                                            NpyArray_DescrFromType(rtype),
                                            GenericReductionOp.NPY_UFUNC_ACCUMULATE, false);
            Npy_DECREF(newArray);
            return ret;
        }

        internal static NpyArray NpyArray_Any(NpyArray self, int axis, NpyArray outPtr, bool keepdims)
        {
            NpyArray newArray;
            NpyArray ret;

            if (null == (newArray = NpyArray_CheckAxis(self, ref axis, 0)))
            {
                return null;
            }
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(UFuncOperation.logical_or),
                                            newArray, null, outPtr, axis,
                                            NpyArray_DescrFromType(NPY_TYPES.NPY_BOOL),
                                            GenericReductionOp.NPY_UFUNC_REDUCE, keepdims);
            Npy_DECREF(newArray);
            return ret;
        }

        internal static NpyArray NpyArray_All(NpyArray srcArray, int axis, NpyArray outPtr, bool keepdims)
        {
            NpyArray newPtr;
            NpyArray ret;

            if (null == (newPtr = NpyArray_CheckAxis(srcArray, ref axis, 0)))
            {
                return null;
            }
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(UFuncOperation.logical_and),
                                            newPtr, null, outPtr, axis,
                                            NpyArray_DescrFromType(NPY_TYPES.NPY_BOOL),
                                            GenericReductionOp.NPY_UFUNC_REDUCE, keepdims);
            Npy_DECREF(newPtr);
            return ret;
        }




    }
}
