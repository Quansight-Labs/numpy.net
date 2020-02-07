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

        internal static NumericOperation GetOperation(ref NpyArray srcArray, NpyArray_Ops operationType)
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

            NpyArray newArray = null;
            if (operandArray == null || NpyArray_Size(srcArray) >= NpyArray_Size(operandArray))
            {
                if (operandArray != null && (srcArray.nd > 0 && operandArray.nd > 0) && (srcArray.nd < operandArray.nd))
                {
                    srcArray = NpyArray_HandleNewAxisDims(srcArray, operandArray);
                }

                newArray = NpyArray_FromArray(srcArray, newtype, flags);
            }
            else
            {
                newArray = NpyArray_FromArray(operandArray, newtype, flags);
            }
            return newArray;
        }

        internal static NpyArray NpyArray_HandleNewAxisDims(NpyArray srcArray, NpyArray operandArray)
        {

            var newdims = new npy_intp[operandArray.dimensions.Length];
            Array.Copy(operandArray.dimensions, 0, newdims, 0, operandArray.nd);

            var possibleOffsets = PossibleNewAxisOffsets(srcArray, operandArray);

            npy_intp numtoskip = possibleOffsets.Length - srcArray.nd;
            if (numtoskip < 0)
                numtoskip = 0;

            for (int i = 0; i < Math.Min(possibleOffsets.Length - numtoskip, srcArray.nd); i++)
            {
                newdims[possibleOffsets[i + numtoskip]] = srcArray.dimensions[i];
            }

     
            npy_intp srcArraySize = NpyArray_SIZE(srcArray);
            npy_intp operandArraySize = NpyArray_MultiplyList(newdims, newdims.Length);
            if (srcArraySize < operandArraySize)
            {
                srcArray = NpyArray_NumericOpUpscaleSourceArray(srcArray, newdims, newdims.Length);
            }
            else
            {
                NpyArray_Dims dims = new NpyArray_Dims()
                {
                    ptr = newdims,
                    len = newdims.Length,
                };
                srcArray = NpyArray_Newshape(srcArray, dims, NPY_ORDER.NPY_KEEPORDER);
            }

            return srcArray;
        }

        internal static npy_intp[] PossibleNewAxisOffsets(NpyArray srcArray, NpyArray operandArray)
        {
            var offsets = new List<npy_intp>();
 

            int srcIndex = 0;
            for (npy_intp i = 0; i < operandArray.nd; i++)
            {
                if (srcIndex < srcArray.nd)
                {
                    if (operandArray.dimensions[i] != srcArray.dimensions[srcIndex] && operandArray.dimensions[i] == 1)
                    {
                        if (srcIndex > 0)
                            offsets.Add(i);
                    }
                    else
                    {
                        srcIndex++;
                    }
                }
                else
                {
                    if (operandArray.dimensions[i] == 1)
                    {
                        offsets.Add(i);
                    }
                }
   
            }

            return offsets.ToArray();
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

            long taskSize = 1000;
            int taskCnt = 0;
            var srcOffsets = new Int32[taskSize];
            var destOffsets = new Int32[taskSize];
            var operOffsets = new Int32[taskSize];

            Countdown countDown = new Countdown();
            List<Exception> caughtExceptions = new List<Exception>();

            for (long i = 0; i < destSize;)
            {
                long offset_cnt = Math.Min(taskSize, destSize - i);

                NpyArray_ITER_TOARRAY(SrcIter, srcArray, srcOffsets, offset_cnt);
                NpyArray_ITER_TOARRAY(DestIter, destArray, destOffsets, offset_cnt);
                NpyArray_ITER_TOARRAY(OperIter, operArray, operOffsets, offset_cnt);

                i += offset_cnt;

                if (true) //taskCnt == taskSize || i == destSize)
                {

                    var taskData = new NumericOpTaskData()
                    {
                        operations = operations,
                        srcArray = srcArray,
                        destArray = destArray,
                        operArray = operArray,
                        srcOffsets = srcOffsets,
                        destOffsets = destOffsets,
                        operOffsets = operOffsets,
                        taskCnt = (int)offset_cnt,
                        countDown = countDown,
                        caughtExceptions = caughtExceptions,
                    };

                    lock (NumericOpTaskQueue)
                    {
                        countDown.Increment();
                        NumericOpTaskQueue.Enqueue(taskData);
                        NumericOpTaskSemaphore.Release(1);
                    }

                    //// task creation is taking huge time
                    //var newTask = new TaskFactory().StartNew(new Action<object>((_taskData) =>
                    //{
                    //    var td = _taskData as NumericOpTaskData;
                    //    NumericOpTask(td.srcArray, td.destArray, td.operArray, td.operations, td.srcOffsets, td.destOffsets, td.operOffsets, td.taskCnt);
                    //}), taskData);

                    //TaskList.Add(newTask);

                    srcOffsets = new Int32[taskSize];
                    destOffsets = new Int32[taskSize];
                    operOffsets = new Int32[taskSize];
                    taskCnt = 0;
                }

                NpyArray_ITER_NEXT(SrcIter);
                NpyArray_ITER_NEXT(DestIter);
                NpyArray_ITER_NEXT(OperIter);

            }

            countDown.Wait();

            if (caughtExceptions.Count > 0)
            {
                throw caughtExceptions[0];
            }
            return;
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

        public class Countdown : IDisposable
        {
            private readonly ManualResetEvent done;
            private long current;

            public Countdown()
            {
                current = 0;
                done = new ManualResetEvent(false);
            }

            public void Signal()
            {
                if (Interlocked.Decrement(ref current) == 0)
                {
                    done.Set();
                }
            }
            public void Increment()
            {
                Interlocked.Increment(ref current);
            }

            public void Wait()
            {
                done.WaitOne();
            }

            public void Dispose()
            {
                ((IDisposable)done).Dispose();
            }
        }

        class NumericOpTaskData
        {
            public NpyArray srcArray;
            public NpyArray destArray;
            public NpyArray operArray;
            public NumericOperations operations;
            public Int32[] srcOffsets;
            public Int32[] destOffsets;
            public Int32[] operOffsets;
            public int taskCnt;
            public Countdown countDown;
            public List<Exception> caughtExceptions;
        }

        private static bool NumericOpThreadsRunning = false;
        private static int NumericOpTaskThreadCount = 10;
        private static Queue<NumericOpTaskData> NumericOpTaskQueue = new Queue<NumericOpTaskData>();
        private static System.Threading.SemaphoreSlim NumericOpTaskSemaphore = new System.Threading.SemaphoreSlim(0, int.MaxValue);

        internal static void StartNumericOpTaskThreads()
        {
            NumericOpThreadsRunning = true;

            for (int i = 0; i < NumericOpTaskThreadCount; i++)
            {
                Task.Run(() => NumericOpTaskThread());
            }
   
        }
        internal static void StopNumericOpTaskThreads()
        {
            NumericOpThreadsRunning = false;
            NumericOpTaskSemaphore.Release(NumericOpTaskThreadCount);
        }

        private static void NumericOpTaskThread()
        {
            Thread.CurrentThread.Name = "NumericOpTaskThread";

            while (NumericOpThreadsRunning)
            {
                NumericOpTaskSemaphore.Wait();
                
                NumericOpTaskData td = null;

                lock (NumericOpTaskQueue)
                {
                    if (NumericOpTaskQueue.Count > 0)
                    {
                        td = NumericOpTaskQueue.Dequeue();
                    }
                }

                if (td != null)
                {
                    try
                    {
                        NumericOpTask(td.srcArray, td.destArray, td.operArray, td.operations, td.srcOffsets, td.destOffsets, td.operOffsets, td.taskCnt);
                        td.countDown.Signal();
                        td.srcOffsets = null;
                        td.destOffsets = null;
                        td.operOffsets = null;
                    }
                    catch (Exception ex)
                    {
                        td.countDown.Signal();
                        td.caughtExceptions.Add(ex);
                    }

                }


            }

            return;
        }

        private static void NumericOpTask(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperations operations, Int32[] srcOffsets, Int32[] destOffsets, Int32[] operOffsets, int taskCnt)
        {
            for (int i = 0; i < taskCnt; i++)
            {
                var srcValue = operations.srcGetItem(srcOffsets[i], srcArray);
                var operValue = operations.operandGetItem(operOffsets[i], operArray);

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
        #endregion

        private static void PerformOuterOpArrayIter(NpyArray a,  NpyArray b, NpyArray destArray, NumericOperations operations)
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
            var DestIter = NpyArray_IterNew(destArray);

            for (long i = 0; i < aSize; i++)
            {
                var aValue = operations.srcGetItem(aIter.dataptr.data_offset - a.data.data_offset, a);
                var bIter = NpyArray_IterNew(b);

                for (long j = 0; j < bSize; j++)
                {
                    var bValue = operations.srcGetItem(bIter.dataptr.data_offset - b.data.data_offset, b);

                    object destValue = operations.operation(aValue, operations.ConvertOperand(aValue, bValue));

                    try
                    {
                        operations.destSetItem(DestIter.dataptr.data_offset - destArray.data.data_offset, destValue, destArray);
                    }
                    catch
                    {
                        operations.destSetItem(DestIter.dataptr.data_offset - destArray.data.data_offset, 0, destArray);
                    }
                    NpyArray_ITER_NEXT(bIter);
                    NpyArray_ITER_NEXT(DestIter);
                }

                NpyArray_ITER_NEXT(aIter);
            }
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
            NumericOperation operation = GetOperation(ref srcArray, operationType);

            NumericOperations operations = NumericOperations.GetOperations(operation, srcArray, destArray, operandArray);
   
            PerformNumericOpScalarIter(srcArray, destArray, operandArray, operations);
        }

        public static NpyArray PerformOuterOpArray(NpyArray srcArray,  NpyArray operandArray, NpyArray destArray, NpyArray_Ops operationType)
        {
            NumericOperation operation = GetOperation(ref srcArray, operationType);

            NumericOperations operations = NumericOperations.GetOperations(operation, srcArray, destArray, operandArray);
  
            PerformOuterOpArrayIter(srcArray, operandArray, destArray, operations);
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
            NumericOperation operation = GetOperation(ref srcArray, NpyArray_Ops.npy_op_floor);

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
            NumericOperation operation = GetOperation(ref srcArray, NpyArray_Ops.npy_op_isnan);

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
