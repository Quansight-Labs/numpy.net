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


        internal static NumericOperation GetOperation(NpyArray srcArray, NpyArray_Ops operationType)
        {
            switch (operationType)
            {
                case NpyArray_Ops.npy_op_add:
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



        internal static NumericOperation GetOperation(VoidPtr vp, NpyArray_Ops operationType)
        {
            NPY_TYPES ItemType = vp.type_num;

            //Console.WriteLine("Getting calculation handler {0} for array type {1}", operationType, srcArray.ItemType);

            switch (operationType)
            {
                case NpyArray_Ops.npy_op_add:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).AddOperation;
                }
                case NpyArray_Ops.npy_op_subtract:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).SubtractOperation;
                }
                case NpyArray_Ops.npy_op_multiply:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).MultiplyOperation;
                }
                case NpyArray_Ops.npy_op_divide:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).DivideOperation;
                }
                case NpyArray_Ops.npy_op_remainder:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).RemainderOperation;
                }

                case NpyArray_Ops.npy_op_fmod:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).FModOperation;
                }

                case NpyArray_Ops.npy_op_power:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).PowerOperation;
                }
                case NpyArray_Ops.npy_op_square:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).SquareOperation;
                }
                case NpyArray_Ops.npy_op_reciprocal:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).ReciprocalOperation;
                }
                case NpyArray_Ops.npy_op_ones_like:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).OnesLikeOperation;
                }
                case NpyArray_Ops.npy_op_sqrt:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).SqrtOperation;
                }
                case NpyArray_Ops.npy_op_negative:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).NegativeOperation;
                }
                case NpyArray_Ops.npy_op_absolute:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).AbsoluteOperation;
                }
                case NpyArray_Ops.npy_op_invert:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).InvertOperation;
                }
                case NpyArray_Ops.npy_op_left_shift:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).LeftShiftOperation;
                }
                case NpyArray_Ops.npy_op_right_shift:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).RightShiftOperation;
                }
                case NpyArray_Ops.npy_op_bitwise_and:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).BitWiseAndOperation;
                }
                case NpyArray_Ops.npy_op_bitwise_xor:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).BitWiseXorOperation;
                }
                case NpyArray_Ops.npy_op_bitwise_or:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).BitWiseOrOperation;
                }
                case NpyArray_Ops.npy_op_less:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).LessOperation;
                }
                case NpyArray_Ops.npy_op_less_equal:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).LessEqualOperation;
                }
                case NpyArray_Ops.npy_op_equal:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).EqualOperation;
                }
                case NpyArray_Ops.npy_op_not_equal:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).NotEqualOperation;
                }
                case NpyArray_Ops.npy_op_greater:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).GreaterOperation;
                }
                case NpyArray_Ops.npy_op_greater_equal:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).GreaterEqualOperation;
                }
                case NpyArray_Ops.npy_op_isnan:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).IsNANOperation;
                }
                case NpyArray_Ops.npy_op_floor_divide:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).FloorDivideOperation;
                }
                case NpyArray_Ops.npy_op_true_divide:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).TrueDivideOperation;
                }
                case NpyArray_Ops.npy_op_logical_or:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).LogicalOrOperation;
                }
                case NpyArray_Ops.npy_op_logical_and:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).LogicalAndOperation;
                }
                case NpyArray_Ops.npy_op_floor:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).FloorOperation;
                }
                case NpyArray_Ops.npy_op_ceil:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).CeilingOperation;
                }
                case NpyArray_Ops.npy_op_maximum:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).MaximumOperation;
                }
                case NpyArray_Ops.npy_op_fmax:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).FMaxOperation;
                }
                case NpyArray_Ops.npy_op_minimum:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).MinimumOperation;
                }
                case NpyArray_Ops.npy_op_fmin:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).FMinOperation;
                }

                case NpyArray_Ops.npy_op_heaviside:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).HeavisideOperation;
                }

                case NpyArray_Ops.npy_op_rint:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).RintOperation;
                }
                case NpyArray_Ops.npy_op_conjugate:
                {
                    return DefaultArrayHandlers.GetArrayHandler(ItemType).ConjugateOperation;
                }
  
                default:
                    return null;
            }
        }

  

        internal static NpyArray NpyArray_PerformUFUNCOperation(NpyArray_Ops operationType,  NpyArray x1Array,  NpyArray x2Array, NpyArray outArray, NpyArray whereFilter)
        {
            if (outArray == null)
            {
                outArray = NpyArray_NumericOpArraySelection(x1Array, x2Array, operationType);
            }

            PerformNumericOpArray(x1Array, outArray, x2Array, operationType);

            return outArray;
        }

        private static NpyArray NpyArray_NumericOpArraySelection(NpyArray srcArray, NpyArray operandArray, NpyArray_Ops operationType)
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
                         newtype = NpyArray_DescrFromType(DefaultArrayHandlers.GetArrayHandler(operandArray.descr.type_num).MathOpReturnType(NpyArray_Ops.npy_op_special_operand_is_float));
                    }
                }
            }
           

            switch (operationType)
            {
                case NpyArray_Ops.npy_op_add:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_subtract:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_multiply:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_divide:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_remainder:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_fmod:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_power:
                    {
                        newtype = NpyArray_DescrFromType(ArrayHandler.MathOpReturnType(operationType));
                        break;
                    }
                case NpyArray_Ops.npy_op_square:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_reciprocal:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_ones_like:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_sqrt:
                    {
                        newtype = NpyArray_DescrFromType(ArrayHandler.MathOpReturnType(operationType));
                        break;
                    }
                case NpyArray_Ops.npy_op_negative:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_absolute:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_invert:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_left_shift:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_right_shift:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_bitwise_and:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_bitwise_xor:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_bitwise_or:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_less:
                case NpyArray_Ops.npy_op_less_equal:
                case NpyArray_Ops.npy_op_equal:
                case NpyArray_Ops.npy_op_not_equal:
                case NpyArray_Ops.npy_op_greater:
                case NpyArray_Ops.npy_op_greater_equal:
                case NpyArray_Ops.npy_op_isnan:
                    {
                        newtype = NpyArray_DescrFromType(NPY_TYPES.NPY_BOOL);
                        break;
                    }
  
                case NpyArray_Ops.npy_op_floor_divide:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_true_divide:
                    {
                        newtype = NpyArray_DescrFromType(ArrayHandler.MathOpReturnType(operationType));
                        break;
                    }
                case NpyArray_Ops.npy_op_logical_or:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_logical_and:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_floor:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_ceil:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_maximum:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_minimum:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_rint:
                    {
                        break;
                    }
                case NpyArray_Ops.npy_op_conjugate:
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

        internal static NpyArray NpyArray_NumericOpUpscaleSourceArray(NpyArray srcArray, npy_intp [] newdims, int nd)
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
        private static void PerformNumericOpScalarIter(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations)
        {
            var destSize = NpyArray_Size(destArray);

            if (NpyArray_SIZE(operArray) == 0 || NpyArray_SIZE(srcArray) == 0)
            {
                NpyArray_Resize(destArray, new NpyArray_Dims() { len = 0, ptr = new npy_intp[] { } }, false, NPY_ORDER.NPY_ANYORDER);
                return;
            }

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

            long taskSize = NUMERICOPS_TASKSIZE;
             
            for (long i = 0; i < destSize;)
            {
                long offset_cnt = Math.Min(taskSize, destSize - i);

                PerformNumericOpScalarSmallIter(srcArray, destArray, operArray, operations, SrcIter, DestIter, OperIter, offset_cnt);

                i += offset_cnt;

                NpyArray_ITER_NEXT(SrcIter);
                NpyArray_ITER_NEXT(DestIter);
                NpyArray_ITER_NEXT(OperIter);
            }

            return;
        }

        private static void PerformNumericOpScalarSmallIter(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations, NpyArrayIterObject srcIter, NpyArrayIterObject destIter, NpyArrayIterObject operIter, npy_intp taskSize)
        {
            List<Exception> caughtExceptions = new List<Exception>();

            Int32[] destOffsets = new Int32[taskSize];
            Int32[] srcOffsets;
            Int32[] operOffsets;

            NpyArray_ITER_TOARRAY(destIter, destArray, destOffsets, taskSize);
            if (NpyArray_SAMESHAPEANDSTRIDES(destArray, srcArray))
            {
                srcOffsets = destOffsets;
            }
            else
            {
                var IterableArraySize = CalculateIterationArraySize(srcArray, destArray);
                srcOffsets = new Int32[IterableArraySize];
                NpyArray_ITER_TOARRAY(srcIter, srcArray, srcOffsets, IterableArraySize);
            }
            if (NpyArray_SAMESHAPEANDSTRIDES(destArray, operArray))
            {
                operOffsets = destOffsets;
            }
            else
            {
                var IterableArraySize = CalculateIterationArraySize(operArray, destArray);
                operOffsets = new Int32[IterableArraySize];
                NpyArray_ITER_TOARRAY(operIter, operArray, operOffsets, IterableArraySize);
            }

            if (taskSize < NUMERICOPS_SMALL_TASKSIZE)
            {
 

                try
                {
                    for (int i = 0; i < taskSize; i++)
                    {
                        int srcIndex = (int)(i < srcOffsets.Length ? i : (i % srcOffsets.Length));
                        var srcValue = operations.srcGetItem(srcOffsets[srcIndex], srcArray);

                        int operandIndex = (int)(i < operOffsets.Length ? i : (i % operOffsets.Length));
                        var operValue = operations.operandGetItem(operOffsets[operandIndex], operArray);

                        object destValue = null;

                        destValue = operations.operation(srcValue, operations.ConvertOperand(srcValue, operValue));

                        try
                        {
                            operations.destSetItem(destOffsets[i], destValue, destArray);
                        }
                        catch
                        {
                            operations.destSetItem(destOffsets[i], 0, destArray);
                        }
                    };
                }
                catch (Exception ex)
                {
                    caughtExceptions.Add(ex);
                }
            }
            else
            {
                Parallel.For(0, taskSize, i =>
                {
                    try
                    {
                        int srcIndex = (int)(i < srcOffsets.Length ? i : (i % srcOffsets.Length));
                        var srcValue = operations.srcGetItem(srcOffsets[srcIndex], srcArray);

                        int operandIndex = (int)(i < operOffsets.Length ? i : (i % operOffsets.Length));
                        var operValue = operations.operandGetItem(operOffsets[operandIndex], operArray);

                        object destValue = null;

                        destValue = operations.operation(srcValue, operations.ConvertOperand(srcValue, operValue));

                        try
                        {
                            operations.destSetItem(destOffsets[i], destValue, destArray);
                        }
                        catch
                        {
                            operations.destSetItem(destOffsets[i], 0, destArray);
                        }
                    }
                    catch (Exception ex)
                    {
                        caughtExceptions.Add(ex);
                    }
                });
            }


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
                    PerformNumericOpScalarIterContiguous_SD_T2<S,bool>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,sbyte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,byte>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT16:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,Int16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,UInt16>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT32:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,Int32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,UInt32>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_INT64:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,Int64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,UInt64>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,float>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,double>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,decimal>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,System.Numerics.Complex>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,System.Numerics.BigInteger>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    PerformNumericOpScalarIterContiguous_SD_T2<S,object>(srcArray, destArray, operArray, operations, srcIter, destIter, operIter);
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

        private static void PerformNumericOpScalarIterContiguous_SD_T2<S,D>(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations, NpyArrayIterObject srcIter, NpyArrayIterObject destIter, NpyArrayIterObject operIter)
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

            var IterableSrcArraySize = CalculateIterationArraySize(srcArray, destArray);
            var srcOffsets = new Int32[IterableSrcArraySize];
            NpyArray_ITER_TOARRAY(srcIter, operArray, srcOffsets, srcOffsets.Length);
            var srcItemSize = srcArray.ItemSize;

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
                            int srcIndex = (int)(index < srcOffsets.Length ? index : (index % srcOffsets.Length));
                            srcIndex = (srcOffsets[srcIndex] / srcItemSize);

                            D dValue = (D)(dynamic)operations.operation(src[srcIndex], operand);
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
                var IterableOperArraySize = CalculateIterationArraySize(operArray, destArray);
                var operOffsets = new Int32[IterableOperArraySize];
                NpyArray_ITER_TOARRAY(operIter, operArray, operOffsets, operOffsets.Length);

                Parallel.For(0, loopCount, index =>
                {
                    try
                    {
                        try
                        {
                            int operandIndex = (int)(index < operOffsets.Length ? index : (index % operOffsets.Length));
                            object operand = operations.ConvertOperand(src[0], operations.operandGetItem(operOffsets[operandIndex], operArray));

                            int srcIndex = (int)(index < srcOffsets.Length ? index : (index % srcOffsets.Length));
                            srcIndex = (srcOffsets[srcIndex] / srcItemSize);

                            D dValue = (D)(dynamic)operations.operation(src[srcIndex], operand);
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

        // calculate the smallest possible array that allows
        // Array to be correctly broadcasted into destArray
        private static long CalculateIterationArraySize(NpyArray Array, NpyArray destArray)
        {
            var OperIter = NpyArray_BroadcastToShape(Array, destArray.dimensions, destArray.nd);
            return NpyArray_ITER_COUNT(OperIter);
        }


        #endregion

        private static void PerformOuterOpArrayIter(NpyArray a, NpyArray b, NpyArray destArray, NumericOperations operations, NpyArray_Ops operation)
        {
            var a1 = destArray.ItemType == a.ItemType ? a : NpyArray_CastToType(a, NpyArray_DescrFromType(destArray.ItemType), false);
            var b1 = destArray.ItemType == b.ItemType ? b : NpyArray_CastToType(b, NpyArray_DescrFromType(destArray.ItemType), false);

            if (destArray.ItemType == NPY_TYPES.NPY_DOUBLE)
            {
                PerformOuterOpArrayIterDouble(a1, b1, destArray, operations, operation);
                return;
            }
  

            var destSize = NpyArray_Size(destArray);
            var aSize = NpyArray_Size(a);
            var bSize = NpyArray_Size(b);

            if (bSize == 0 || aSize == 0)
            {
                NpyArray_Resize(destArray, new NpyArray_Dims() { len = 0, ptr = new npy_intp[] { } }, false, NPY_ORDER.NPY_ANYORDER);
                return;
            }

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
        }

        private static void PerformOuterOpArrayIterDouble(NpyArray a, NpyArray b, NpyArray destArray, NumericOperations operations, NpyArray_Ops operation)
        {
            var destSize = NpyArray_Size(destArray);
            var aSize = NpyArray_Size(a);
            var bSize = NpyArray_Size(b);

            if (bSize == 0 || aSize == 0)
            {
                NpyArray_Resize(destArray, new NpyArray_Dims() { len = 0, ptr = new npy_intp[] { } }, false, NPY_ORDER.NPY_ANYORDER);
                return;
            }

            var aIter = NpyArray_IterNew(a);
            var bIter = NpyArray_IterNew(b);
            var DestIter = NpyArray_IterNew(destArray);

            double[] aValues = new double[aSize];
            for (long i = 0; i < aSize; i++)
            {
                aValues[i] = (double)operations.destGetItem(aIter.dataptr.data_offset - a.data.data_offset, a);
                NpyArray_ITER_NEXT(aIter);
            }

            double[] bValues = new double[bSize];
            for (long i = 0; i < bSize; i++)
            {
                bValues[i] = (double)operations.destGetItem(bIter.dataptr.data_offset - b.data.data_offset, b);
                NpyArray_ITER_NEXT(bIter);
            }


            double[]dp = destArray.data.datap as double[];

    
            if (DestIter.contiguous)
            {

                Parallel.For(0, aSize, i =>
                {
                    var aValue = aValues[i];

                    long destIndex = (destArray.data.data_offset / destArray.ItemSize) + i * bSize;

                    for (long j = 0; j < bSize; j++)
                    {
                        var bValue = bValues[j];

                        double destValue;
                        switch (operation)
                        {
                            case NpyArray_Ops.npy_op_add:
                                destValue = UFuncAdd(aValue, bValue);
                                break;
                            case NpyArray_Ops.npy_op_subtract:
                                destValue = UFuncSubtract(aValue, bValue);
                                break;
                            case NpyArray_Ops.npy_op_multiply:
                                destValue = UFuncMultiply(aValue, bValue);
                                break;
                            case NpyArray_Ops.npy_op_divide:
                                destValue = UFuncDivide(aValue, bValue);
                                break;
                            case NpyArray_Ops.npy_op_fmod:
                                destValue = UFuncFMod(aValue, bValue);
                                break;
                            case NpyArray_Ops.npy_op_power:
                                destValue = UFuncPower(aValue, bValue);
                                break;
                            case NpyArray_Ops.npy_op_remainder:
                                destValue = UFuncRemainder(aValue, bValue);
                                break;
                            default:
                                destValue = 0;
                                break;

                        }

                        try
                        {
                            dp[destIndex] = destValue;
                        }
                        catch
                        {
                            operations.destSetItem(destIndex, 0, destArray);
                        }
                        destIndex++;
                    }

                });
            }
            else
            {
                for (long i = 0; i < aSize; i++)
                {
                    var aValue = aValues[i];

                    for (long j = 0; j < bSize; j++)
                    {
                        var bValue = bValues[j];

                        //double destValue = op(aValue, bValue);
                        double destValue = UFuncAdd(aValue, bValue);

                        try
                        {
                            long AdjustedIndex = AdjustedIndex_SetItemFunction(DestIter.dataptr.data_offset - destArray.data.data_offset, destArray, dp.Length);
                            dp[AdjustedIndex] = destValue;
                        }
                        catch
                        {
                            long AdjustedIndex = AdjustedIndex_SetItemFunction(DestIter.dataptr.data_offset - destArray.data.data_offset, destArray, dp.Length);
                            operations.destSetItem(AdjustedIndex, 0, destArray);
                        }
                        NpyArray_ITER_NEXT(DestIter);
                    }

                }
            }

   
        }

        delegate double UFuncOperation(double a, double b);

        static double UFuncAdd(double aValue, double bValue)
        {
            return aValue + bValue;
        }

        static double UFuncSubtract(double aValue, double bValue)
        {
            return aValue - bValue;
        }
        static double UFuncMultiply(double aValue, double bValue)
        {
            return aValue * bValue;
        }

        static double UFuncDivide(double aValue, double bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue / bValue;
        }
        static double UFuncRemainder(double aValue, double bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        static double UFuncFMod(double aValue, double bValue)
        {
            if (bValue == 0)
                return 0;
            return aValue % bValue;
        }
        static double UFuncPower(double aValue, double bValue)
        {
            return Math.Pow(aValue, bValue);
        }


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
        public static void PerformNumericOpArray(NpyArray srcArray, NpyArray destArray, NpyArray operandArray, NpyArray_Ops operationType)
        {
            NumericOperation operation = GetOperation(srcArray, operationType);

            NumericOperations operations = NumericOperations.GetOperations(operation, srcArray, destArray, operandArray);
   
            PerformNumericOpScalarIter(srcArray, destArray, operandArray, operations);
        }

        public static NpyArray PerformOuterOpArray(NpyArray srcArray,  NpyArray operandArray, NpyArray destArray, NpyArray_Ops operationType)
        {
            NumericOperation operation = GetOperation(srcArray, operationType);

            NumericOperations operations = NumericOperations.GetOperations(operation, srcArray, destArray, operandArray);
  
            PerformOuterOpArrayIter(srcArray, operandArray, destArray, operations, operationType);
            return destArray;
        }


        internal static NpyArray PerformReduceOpArray(NpyArray srcArray, int axis, NpyArray_Ops ops, NPY_TYPES rtype, NpyArray outPtr, bool keepdims)
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

        internal static NpyArray PerformReduceAtOpArray(NpyArray srcArray, NpyArray indices, int axis, NpyArray_Ops ops, NPY_TYPES rtype, NpyArray outPtr)
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

        internal static NpyArray PerformAccumulateOpArray(NpyArray srcArray, int axis, NpyArray_Ops ops, NPY_TYPES rtype, NpyArray outPtr)
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
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(NpyArray_Ops.npy_op_maximum),
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
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(NpyArray_Ops.npy_op_minimum),
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
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(NpyArray_Ops.npy_op_add),
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
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(NpyArray_Ops.npy_op_multiply),
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


  
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(NpyArray_Ops.npy_op_add),
                                            newArray, null, outPtr, axis,
                                            NpyArray_DescrFromType(rtype),
                                            GenericReductionOp.NPY_UFUNC_ACCUMULATE, false);
            Npy_DECREF(newArray);
            return ret;
        }


        internal static NpyArray NpyArray_Floor(NpyArray srcArray, NpyArray outPtr)
        {
            NumericOperation operation = GetOperation(srcArray, NpyArray_Ops.npy_op_floor);

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
            PerformUFunc(srcArray, outPtr, ref floor, outPtr.dimensions, 0, 0, 0, operations);


            Npy_DECREF(outPtr);
            return NpyArray_Flatten(outPtr, NPY_ORDER.NPY_ANYORDER);
        }

        internal static NpyArray NpyArray_IsNaN(NpyArray srcArray)
        {
            NumericOperation operation = GetOperation(srcArray, NpyArray_Ops.npy_op_isnan);

            NpyArray outPtr = NpyArray_FromArray(srcArray, NpyArray_DescrFromType(NPY_TYPES.NPY_BOOL), NPYARRAYFLAGS.NPY_CONTIGUOUS | NPYARRAYFLAGS.NPY_FORCECAST);

            NumericOperations operations = NumericOperations.GetOperations(operation, srcArray, outPtr, null);

            object floor = 0;
            PerformUFunc(srcArray, outPtr, ref floor, outPtr.dimensions, 0, 0, 0, operations);


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
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(NpyArray_Ops.npy_op_multiply),
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
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(NpyArray_Ops.npy_op_logical_or),
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
            ret = NpyUFunc_GenericReduction(NpyArray_GetNumericOp(NpyArray_Ops.npy_op_logical_and),
                                            newPtr, null, outPtr, axis,
                                            NpyArray_DescrFromType(NPY_TYPES.NPY_BOOL),
                                            GenericReductionOp.NPY_UFUNC_REDUCE, keepdims);
            Npy_DECREF(newPtr);
            return ret;
        }

        private static void PerformUFunc(NpyArray srcArray, NpyArray destArray, ref object cumsum, npy_intp[] dimensions, int dimIdx, npy_intp src_offset, npy_intp dest_offset, NumericOperations operation)
        {
            if (dimIdx == destArray.nd)
            {
                var srcValue = operation.srcGetItem(src_offset, srcArray);

                cumsum = operation.operation(srcValue, operation.ConvertOperand(srcValue, cumsum));

                try
                {
                    operation.destSetItem(dest_offset, cumsum, destArray);
                }
                catch
                {
                    operation.destSetItem(dest_offset, 0, destArray);
                }
            }
            else
            {
                for (int i = 0; i < dimensions[dimIdx]; i++)
                {
                    npy_intp lsrc_offset = src_offset + srcArray.strides[dimIdx] * i;
                    npy_intp ldest_offset = dest_offset + destArray.strides[dimIdx] * i;

                    PerformUFunc(srcArray, destArray, ref cumsum, dimensions, dimIdx + 1, lsrc_offset, ldest_offset, operation);
                }
            }
        }
 




    }
}
