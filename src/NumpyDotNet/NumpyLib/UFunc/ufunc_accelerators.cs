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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NumpyLib.numpyinternal;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyLib
{
    internal partial class UFUNC_Bool : UFUNC_BASE<bool>, IUFUNC_Operations
    {
        #region Accelerator Handlers

        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
                   VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
                   VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
                   UFuncOperation ufop, npy_intp N)
        {
            bool[] retArray = Result.datap as bool[];
            bool[] Op1Array = Operand1.datap as bool[];
            bool[] Op2Array = Operand2.datap as bool[];

            switch (ufop)
            {

                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            bool[] OperandArray = Operand.datap as bool[];
            bool[] retArray = Result.datap as bool[];
            bool result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
     
                case UFuncOperation.logical_or:
                    {
                        retArray[R_Index] = LogicalOrReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_and:
                    {
                        retArray[R_Index] = LogicalAndReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
     
                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }


        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                case UFuncOperation.divide:
                    break;
                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    break;
            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            return null;
        }
        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            return null;
        }
        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            return null;
        }
        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
        #endregion

        #region Reduce accelerators
        protected bool LogicalOrReduce(bool result, bool[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result || OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }

        protected bool LogicalAndReduce(bool result, bool[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result && OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        #endregion
    }

    internal partial class UFUNC_SByte : UFUNC_BASE<sbyte>, IUFUNC_Operations
    {
        #region Accelerator Handlers

        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
                   VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
                   VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
                   UFuncOperation ufop, npy_intp N)
        {
            sbyte[] retArray = Result.datap as sbyte[];
            sbyte[] Op1Array = Operand1.datap as sbyte[];
            sbyte[] Op2Array = Operand2.datap as sbyte[];

            switch (ufop)
            {
  
                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            sbyte[] OperandArray = Operand.datap as sbyte[];
            sbyte[] retArray = Result.datap as sbyte[];
            sbyte result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                case UFuncOperation.divide:
                case UFuncOperation.logical_or:
                case UFuncOperation.logical_and:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
 
        }


        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                case UFuncOperation.divide:
                case UFuncOperation.logical_or:
                case UFuncOperation.logical_and:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    break;
            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
        #endregion
    }

    internal partial class UFUNC_UByte : UFUNC_BASE<byte>, IUFUNC_Operations
    {
        #region Accelerator Handlers

        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
                   VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
                   VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
                   UFuncOperation ufop, npy_intp N)
        {
            byte[] retArray = Result.datap as byte[];
            byte[] Op1Array = Operand1.datap as byte[];
            byte[] Op2Array = Operand2.datap as byte[];

            switch (ufop)
            {
 
                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            /* Adjust input pointer */
            byte[] OperandArray = Operand.datap as byte[];
            byte[] retArray = Result.datap as byte[];
            byte result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                case UFuncOperation.divide:
                case UFuncOperation.logical_or:
                case UFuncOperation.logical_and:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }

        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                case UFuncOperation.divide:
                case UFuncOperation.logical_or:
                case UFuncOperation.logical_and:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    break;
            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }
        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
        #endregion
    }

    internal partial class UFUNC_Int16 : UFUNC_BASE<Int16>, IUFUNC_Operations
    {
        #region Accelerator Handlers


        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
                         VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
                         VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
                         UFuncOperation ufop, npy_intp N)
        {
            Int16[] retArray = Result.datap as Int16[];
            Int16[] Op1Array = Operand1.datap as Int16[];
            Int16[] Op2Array = Operand2.datap as Int16[];

            switch (ufop)
            {
                case NumpyLib.UFuncOperation.add:
                    AddAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                                    Op2Array, O2_Index, O2_CalculatedStep,
                                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                case NumpyLib.UFuncOperation.multiply:
                    MultiplyAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                    Op2Array, O2_Index, O2_CalculatedStep,
                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }


        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            Int16[] OperandArray = Operand.datap as Int16[];
            Int16[] retArray = Result.datap as Int16[];
            Int16 result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                    {
                        retArray[R_Index] = AddReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.subtract:
                    {
                        retArray[R_Index] = SubtractReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.multiply:
                    {
                        retArray[R_Index] = MultiplyReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.divide:
                    {
                        retArray[R_Index] = DivideReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_or:
                    {
                        retArray[R_Index] = LogicalOrReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_and:
                    {
                        retArray[R_Index] = LogicalAndReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.maximum:
                    {
                        retArray[R_Index] = MaximumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.minimum:
                    {
                        retArray[R_Index] = MinimumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }


        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                    return AddReduce;

                case UFuncOperation.subtract:
                    return SubtractReduce;

                case UFuncOperation.multiply:
                    return MultiplyReduce;

                case UFuncOperation.divide:
                    return DivideReduce;

                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                    return MaximumReduce;

                case UFuncOperation.minimum:
                    return MinimumReduce;

            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                    return AddAccumulate;
                case UFuncOperation.multiply:
                    return MultiplyAccumulate;
            }

            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                    return AddSubMultScalerIterContigNoIter;

                case UFuncOperation.divide:
                case UFuncOperation.true_divide:
                case UFuncOperation.floor_divide:
                case UFuncOperation.remainder:
                    return DivisionScalerIterContigNoIter;

                case UFuncOperation.power:
                case UFuncOperation.sqrt:
                case UFuncOperation.absolute:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    return PowerSqrtScalerIterContigNoIter;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
        #endregion

        #region Reduce accelerators
        protected Int16 AddReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (Int16)(result + OperandArray[OperIndex]);
                OperIndex += OperStep;
            }
            return result;
        }
        protected Int16 SubtractReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (Int16)(result - OperandArray[OperIndex]);
                OperIndex += OperStep;
            }
            return result;
        }
        protected Int16 MultiplyReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (Int16)(result * OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int16 DivideReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = 0;
                else
                    result = (Int16)(result / bValue);

                OperIndex += OperStep;
            }

            return result;
        }
        protected Int16 LogicalOrReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = (Int16)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int16 LogicalAndReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = (Int16)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int16 MaximumReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int16 MinimumReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Min(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        #endregion

        #region Accumulate accelerators
        protected void AddAccumulate(
                Int16[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                Int16[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                Int16[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = (Int16)(Op1Array[O1_Index] + Op2Array[O2_Index]);

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }

  
        protected void MultiplyAccumulate(
                Int16[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                Int16[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                Int16[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = (Int16)(Op1Array[O1_Index] * Op2Array[O2_Index]);

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
        #endregion

        #region ScalerIterContigNoIter Accelerators Accelerators
        private void AddSubMultScalerIterContigNoIter(Int16[] src, Int16[] dest, Int16 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.add)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = (Int16)(src[srcIndex++] + operand);
                }
                return;
            }
            if (ops == UFuncOperation.subtract)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = (Int16)(src[srcIndex++] - operand);
                }
                return;
            }
            if (ops == UFuncOperation.multiply)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = (Int16)(src[srcIndex++] * operand);
                }
                return;
            }

        }

        private void DivisionScalerIterContigNoIter(Int16[] src, Int16[] dest, Int16 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (operand == 0)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = 0;
                }
                return;
            }

            if (ops == UFuncOperation.divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = (Int16)(src[srcIndex++] / operand);
                }
                return;
            }
            if (ops == UFuncOperation.true_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = (Int16)(src[srcIndex++] / operand);
                }
                return;
            }
            if (ops == UFuncOperation.floor_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToInt16(Math.Floor(Convert.ToDouble(src[srcIndex++]) / Convert.ToDouble(operand)));
                }
                return;
            }
            if (ops == UFuncOperation.remainder)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    var aValue = src[srcIndex++];
                    var rem = aValue % operand;
                    if ((aValue > 0) == (operand > 0) || rem == 0)
                    {
                        dest[destIndex++] = (Int16)rem;
                    }
                    else
                    {
                        dest[destIndex++] = (Int16)(rem + operand);
                    }
                }
                return;
            }

        }

        private void PowerSqrtScalerIterContigNoIter(Int16[] src, Int16[] dest, Int16 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.power)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToInt16(Math.Pow(src[srcIndex++], operand));
                }
                return;
            }
            if (ops == UFuncOperation.sqrt)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToInt16(Math.Sqrt(src[srcIndex++]));
                }
                return;
            }
            if (ops == UFuncOperation.absolute)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Abs(src[srcIndex++]);
                }
                return;
            }
            if (ops == UFuncOperation.maximum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Max(src[srcIndex++], operand);
                }
                return;
            }
            if (ops == UFuncOperation.minimum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Min(src[srcIndex++], operand);
                }
                return;
            }

        }
        #endregion
    }

    internal partial class UFUNC_UInt16 : UFUNC_BASE<UInt16>, IUFUNC_Operations
    {
        #region Accelerator Handlers

        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
                     VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
                     VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
                     UFuncOperation ufop, npy_intp N)
        {
            UInt16[] retArray = Result.datap as UInt16[];
            UInt16[] Op1Array = Operand1.datap as UInt16[];
            UInt16[] Op2Array = Operand2.datap as UInt16[];

            switch (ufop)
            {
                case NumpyLib.UFuncOperation.add:
                    AddAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                                    Op2Array, O2_Index, O2_CalculatedStep,
                                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                case NumpyLib.UFuncOperation.multiply:
                    MultiplyAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                    Op2Array, O2_Index, O2_CalculatedStep,
                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            UInt16[] OperandArray = Operand.datap as UInt16[];
            UInt16[] retArray = Result.datap as UInt16[];
            UInt16 result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                    {
                        retArray[R_Index] = AddReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.subtract:
                    {
                        retArray[R_Index] = SubtractReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.multiply:
                    {
                        retArray[R_Index] = MultiplyReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.divide:
                    {
                        retArray[R_Index] = DivideReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_or:
                    {
                        retArray[R_Index] = LogicalOrReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_and:
                    {
                        retArray[R_Index] = LogicalAndReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.maximum:
                    {
                        retArray[R_Index] = MaximumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.minimum:
                    {
                        retArray[R_Index] = MinimumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }



        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                    return AddReduce;

                case UFuncOperation.subtract:
                    return SubtractReduce;

                case UFuncOperation.multiply:
                    return MultiplyReduce;

                case UFuncOperation.divide:
                    return DivideReduce;

                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                    return MaximumReduce;

                case UFuncOperation.minimum:
                    return MinimumReduce;

            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                    return AddAccumulate;
                case UFuncOperation.multiply:
                    return MultiplyAccumulate;
            }

            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                    return AddSubMultScalerIterContigNoIter;

                case UFuncOperation.divide:
                case UFuncOperation.true_divide:
                case UFuncOperation.floor_divide:
                case UFuncOperation.remainder:
                    return DivisionScalerIterContigNoIter;

                case UFuncOperation.power:
                case UFuncOperation.sqrt:
                case UFuncOperation.absolute:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    return PowerSqrtScalerIterContigNoIter;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
        #endregion

        #region Reduce accelerators
        protected UInt16 AddReduce(UInt16 result, UInt16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (UInt16)(result + OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt16 SubtractReduce(UInt16 result, UInt16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (UInt16)(result - OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt16 MultiplyReduce(UInt16 result, UInt16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (UInt16)(result * OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt16 DivideReduce(UInt16 result, UInt16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = 0;
                else
                    result = (UInt16)(result / bValue);

                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt16 LogicalOrReduce(UInt16 result, UInt16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = (UInt16)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt16 LogicalAndReduce(UInt16 result, UInt16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = (UInt16)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt16 MaximumReduce(UInt16 result, UInt16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt16 MinimumReduce(UInt16 result, UInt16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Min(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        #endregion

        #region Accumulate accelerators
        protected void AddAccumulate(
                UInt16[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                UInt16[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                UInt16[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = (UInt16)(Op1Array[O1_Index] + Op2Array[O2_Index]);

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
 
        protected void MultiplyAccumulate(
                UInt16[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                UInt16[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                UInt16[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = (UInt16)(Op1Array[O1_Index] * Op2Array[O2_Index]);

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
        #endregion

        #region ScalerIterContigNoIter Accelerators
        private void AddSubMultScalerIterContigNoIter(UInt16[] src, UInt16[] dest, UInt16 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.add)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = (UInt16)(src[srcIndex++] + operand);
                }
                return;
            }
            if (ops == UFuncOperation.subtract)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = (UInt16)(src[srcIndex++] - operand);
                }
                return;
            }
            if (ops == UFuncOperation.multiply)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = (UInt16)(src[srcIndex++] * operand);
                }
                return;
            }

        }

        private void DivisionScalerIterContigNoIter(UInt16[] src, UInt16[] dest, UInt16 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (operand == 0)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = 0;
                }
                return;
            }

            if (ops == UFuncOperation.divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = (UInt16)(src[srcIndex++] / operand);
                }
                return;
            }
            if (ops == UFuncOperation.true_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = (UInt16)(src[srcIndex++] / operand);
                }
                return;
            }
            if (ops == UFuncOperation.floor_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToUInt16(Math.Floor(Convert.ToDouble(src[srcIndex++]) / Convert.ToDouble(operand)));
                }
                return;
            }
            if (ops == UFuncOperation.remainder)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    var aValue = src[srcIndex++];
                    var rem = aValue % operand;
                    if ((aValue > 0) == (operand > 0) || rem == 0)
                    {
                        dest[destIndex++] = (UInt16)rem;
                    }
                    else
                    {
                        dest[destIndex++] = (UInt16)(rem + operand);
                    }
                }
                return;
            }

        }

        private void PowerSqrtScalerIterContigNoIter(UInt16[] src, UInt16[] dest, UInt16 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.power)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToUInt16(Math.Pow(src[srcIndex++], operand));
                }
                return;
            }
            if (ops == UFuncOperation.sqrt)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToUInt16(Math.Sqrt(src[srcIndex++]));
                }
                return;
            }
            if (ops == UFuncOperation.absolute)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++];
                }
                return;
            }
            if (ops == UFuncOperation.maximum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Max(src[srcIndex++], operand);
                }
                return;
            }
            if (ops == UFuncOperation.minimum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Min(src[srcIndex++], operand);
                }
                return;
            }

        }

  
        #endregion
    }

    internal partial class UFUNC_Int32 : UFUNC_BASE<Int32>, IUFUNC_Operations
    {
        #region Accelerator Handlers

  

        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
                         VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
                         VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep, 
                         UFuncOperation ufop, npy_intp N)
        {
            Int32[] retArray = Result.datap as Int32[];
            Int32[] Op1Array = Operand1.datap as Int32[];
            Int32[] Op2Array = Operand2.datap as Int32[];

            switch (ufop)
            {
                case NumpyLib.UFuncOperation.add:
                    AddAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                                    Op2Array, O2_Index, O2_CalculatedStep,
                                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                case NumpyLib.UFuncOperation.multiply:
                    MultiplyAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                    Op2Array, O2_Index, O2_CalculatedStep,
                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }
         

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            Int32[] OperandArray = Operand.datap as Int32[];
            Int32[] retArray = Result.datap as Int32[];
            Int32 result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                    {
                        retArray[R_Index] = AddReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.subtract:
                    {
                        retArray[R_Index] = SubtractReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.multiply:
                    {
                        retArray[R_Index] = MultiplyReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.divide:
                    {
                        retArray[R_Index] = DivideReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_or:
                    {
                        retArray[R_Index] = LogicalOrReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_and:
                    {
                        retArray[R_Index] = LogicalAndReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.maximum:
                    {
                        retArray[R_Index] = MaximumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.minimum:
                    {
                        retArray[R_Index] = MinimumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }


        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                    return AddReduce;

                case UFuncOperation.subtract:
                    return SubtractReduce;

                case UFuncOperation.multiply:
                    return MultiplyReduce;

                case UFuncOperation.divide:
                    return DivideReduce;

                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                    return MaximumReduce;

                case UFuncOperation.minimum:
                    return MinimumReduce;

            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                    return AddAccumulate;
                case UFuncOperation.multiply:
                    return MultiplyAccumulate;
            }

            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                    return AddSubMultScalerIterContigNoIter;

                case UFuncOperation.divide:
                case UFuncOperation.true_divide:
                case UFuncOperation.floor_divide:
                case UFuncOperation.remainder:
                    return DivisionScalerIterContigNoIter;

                case UFuncOperation.power:
                case UFuncOperation.sqrt:
                case UFuncOperation.absolute:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    return PowerSqrtScalerIterContigNoIter;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
#endregion

#region Reduce accelerators
        protected Int32 AddReduce(Int32 result, Int32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int32 SubtractReduce(Int32 result, Int32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result - OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int32 MultiplyReduce(Int32 result, Int32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result * OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int32 DivideReduce(Int32 result, Int32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = 0;
                else
                    result = result / bValue;

                OperIndex += OperStep;
            }

            return result;
        }
        protected Int32 LogicalOrReduce(Int32 result, Int32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int32 LogicalAndReduce(Int32 result, Int32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int32 MaximumReduce(Int32 result, Int32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int32 MinimumReduce(Int32 result, Int32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Min(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
#endregion

#region Accumulate accelerators
        protected void AddAccumulate(
                Int32[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                Int32[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                Int32[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] + Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
  
        protected void MultiplyAccumulate(
                Int32[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                Int32[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                Int32[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] * Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
#endregion

#region ScalerIterContigNoIter Accelerators
        private void AddSubMultScalerIterContigNoIter(Int32[] src, Int32[] dest, Int32 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.add)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] + operand;
                }
                return;
            }
            if (ops == UFuncOperation.subtract)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] - operand;
                }
                return;
            }
            if (ops == UFuncOperation.multiply)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] * operand;
                }
                return;
            }

        }

        private void DivisionScalerIterContigNoIter(Int32[] src, Int32[] dest, Int32 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (operand == 0)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = 0;
                }
                return;
            }

            if (ops == UFuncOperation.divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.true_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.floor_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToInt32(Math.Floor(Convert.ToDouble(src[srcIndex++]) / Convert.ToDouble(operand)));
                }
                return;
            }
            if (ops == UFuncOperation.remainder)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    var aValue = src[srcIndex++];
                    var rem = aValue % operand;
                    if ((aValue > 0) == (operand > 0) || rem == 0)
                    {
                        dest[destIndex++] = rem;
                    }
                    else
                    {
                        dest[destIndex++] = rem + operand;
                    }
                }
                return;
            }

        }

        private void PowerSqrtScalerIterContigNoIter(Int32[] src, Int32[] dest, Int32 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.power)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToInt32(Math.Pow(src[srcIndex++], operand));
                }
                return;
            }
            if (ops == UFuncOperation.sqrt)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToInt32(Math.Sqrt(src[srcIndex++]));
                }
                return;
            }
            if (ops == UFuncOperation.absolute)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Abs(src[srcIndex++]);
                }
                return;
            }
            if (ops == UFuncOperation.maximum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Max(src[srcIndex++], operand);
                }
                return;
            }
            if (ops == UFuncOperation.minimum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Min(src[srcIndex++], operand);
                }
                return;
            }

        }

#endregion

    }

    internal partial class UFUNC_UInt32 : UFUNC_BASE<UInt32>, IUFUNC_Operations
    {
        #region Accelerator Handlers

        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
               VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
               VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
               UFuncOperation ufop, npy_intp N)
        {
            UInt32[] retArray = Result.datap as UInt32[];
            UInt32[] Op1Array = Operand1.datap as UInt32[];
            UInt32[] Op2Array = Operand2.datap as UInt32[];

            switch (ufop)
            {
                case NumpyLib.UFuncOperation.add:
                    AddAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                                    Op2Array, O2_Index, O2_CalculatedStep,
                                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                case NumpyLib.UFuncOperation.multiply:
                    MultiplyAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                    Op2Array, O2_Index, O2_CalculatedStep,
                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            UInt32[] OperandArray = Operand.datap as UInt32[];
            UInt32[] retArray = Result.datap as UInt32[];
            UInt32 result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                    {
                        retArray[R_Index] = AddReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.subtract:
                    {
                        retArray[R_Index] = SubtractReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.multiply:
                    {
                        retArray[R_Index] = MultiplyReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.divide:
                    {
                        retArray[R_Index] = DivideReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_or:
                    {
                        retArray[R_Index] = LogicalOrReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_and:
                    {
                        retArray[R_Index] = LogicalAndReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.maximum:
                    {
                        retArray[R_Index] = MaximumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.minimum:
                    {
                        retArray[R_Index] = MinimumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }

        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                    return AddReduce;

                case UFuncOperation.subtract:
                    return SubtractReduce;

                case UFuncOperation.multiply:
                    return MultiplyReduce;

                case UFuncOperation.divide:
                    return DivideReduce;

                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                    return MaximumReduce;

                case UFuncOperation.minimum:
                    return MinimumReduce;

            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                    return AddAccumulate;
                case UFuncOperation.multiply:
                    return MultiplyAccumulate;
            }

            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                    return AddSubMultScalerIterContigNoIter;

                case UFuncOperation.divide:
                case UFuncOperation.true_divide:
                case UFuncOperation.floor_divide:
                case UFuncOperation.remainder:
                    return DivisionScalerIterContigNoIter;

                case UFuncOperation.power:
                case UFuncOperation.sqrt:
                case UFuncOperation.absolute:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    return PowerSqrtScalerIterContigNoIter;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
#endregion

#region Reduce accelerators
        protected UInt32 AddReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt32 SubtractReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result - OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt32 MultiplyReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result * OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt32 DivideReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = 0;
                else
                    result = result / bValue;

                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt32 LogicalOrReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = (UInt32)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt32 LogicalAndReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = (UInt32)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt32 MaximumReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt32 MinimumReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Min(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
#endregion

#region Accumulate accelerators
        protected void AddAccumulate(
                UInt32[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                UInt32[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                UInt32[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] + Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
 
        protected void MultiplyAccumulate(
                UInt32[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                UInt32[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                UInt32[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] * Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
#endregion

#region ScalerIterContigNoIter Accelerators
        private void AddSubMultScalerIterContigNoIter(UInt32[] src, UInt32[] dest, UInt32 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.add)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] + operand;
                }
                return;
            }
            if (ops == UFuncOperation.subtract)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] - operand;
                }
                return;
            }
            if (ops == UFuncOperation.multiply)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] * operand;
                }
                return;
            }

        }

        private void DivisionScalerIterContigNoIter(UInt32[] src, UInt32[] dest, UInt32 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (operand == 0)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = 0;
                }
                return;
            }

            if (ops == UFuncOperation.divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.true_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.floor_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToUInt32(Math.Floor(Convert.ToDouble(src[srcIndex++]) / Convert.ToDouble(operand)));
                }
                return;
            }
            if (ops == UFuncOperation.remainder)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    var aValue = src[srcIndex++];
                    var rem = aValue % operand;
                    if ((aValue > 0) == (operand > 0) || rem == 0)
                    {
                        dest[destIndex++] = rem;
                    }
                    else
                    {
                        dest[destIndex++] = rem + operand;
                    }
                }
                return;
            }

        }

        private void PowerSqrtScalerIterContigNoIter(UInt32[] src, UInt32[] dest, UInt32 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.power)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToUInt32(Math.Pow(src[srcIndex++], operand));
                }
                return;
            }
            if (ops == UFuncOperation.sqrt)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToUInt32(Math.Sqrt(src[srcIndex++]));
                }
                return;
            }
            if (ops == UFuncOperation.absolute)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++];
                }
                return;
            }
            if (ops == UFuncOperation.maximum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Max(src[srcIndex++], operand);
                }
                return;
            }
            if (ops == UFuncOperation.minimum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Min(src[srcIndex++], operand);
                }
                return;
            }

        }

#endregion

    }

    internal partial class UFUNC_Int64 : UFUNC_BASE<Int64>, IUFUNC_Operations
    {
        #region Accelerator Handlers

        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
           VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
           VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
           UFuncOperation ufop, npy_intp N)
        {
            Int64[] retArray = Result.datap as Int64[];
            Int64[] Op1Array = Operand1.datap as Int64[];
            Int64[] Op2Array = Operand2.datap as Int64[];

            switch (ufop)
            {
                case NumpyLib.UFuncOperation.add:
                    AddAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                                    Op2Array, O2_Index, O2_CalculatedStep,
                                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                case NumpyLib.UFuncOperation.multiply:
                    MultiplyAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                    Op2Array, O2_Index, O2_CalculatedStep,
                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            Int64[] OperandArray = Operand.datap as Int64[];
            Int64[] retArray = Result.datap as Int64[];
            Int64 result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                    {
                        retArray[R_Index] = AddReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.subtract:
                    {
                        retArray[R_Index] = SubtractReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.multiply:
                    {
                        retArray[R_Index] = MultiplyReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.divide:
                    {
                        retArray[R_Index] = DivideReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_or:
                    {
                        retArray[R_Index] = LogicalOrReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_and:
                    {
                        retArray[R_Index] = LogicalAndReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.maximum:
                    {
                        retArray[R_Index] = MaximumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.minimum:
                    {
                        retArray[R_Index] = MinimumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }


        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                    return AddReduce;

                case UFuncOperation.subtract:
                    return SubtractReduce;

                case UFuncOperation.multiply:
                    return MultiplyReduce;

                case UFuncOperation.divide:
                    return DivideReduce;

                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                    return MaximumReduce;

                case UFuncOperation.minimum:
                    return MinimumReduce;

            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                    return AddAccumulate;
                case UFuncOperation.multiply:
                    return MultiplyAccumulate;
            }

            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                    return AddSubMultScalerIterContigNoIter;

                case UFuncOperation.divide:
                case UFuncOperation.true_divide:
                case UFuncOperation.floor_divide:
                case UFuncOperation.remainder:
                    return DivisionScalerIterContigNoIter;

                case UFuncOperation.power:
                case UFuncOperation.sqrt:
                case UFuncOperation.absolute:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    return PowerSqrtScalerIterContigNoIter;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
#endregion

#region Reduce accelerators
        protected Int64 AddReduce(Int64 result, Int64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int64 SubtractReduce(Int64 result, Int64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result - OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int64 MultiplyReduce(Int64 result, Int64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result * OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int64 DivideReduce(Int64 result, Int64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = 0;
                else
                    result = result / bValue;

                OperIndex += OperStep;
            }

            return result;
        }
        protected Int64 LogicalOrReduce(Int64 result, Int64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int64 LogicalAndReduce(Int64 result, Int64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int64 MaximumReduce(Int64 result, Int64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected Int64 MinimumReduce(Int64 result, Int64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Min(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
#endregion

#region Accumulate accelerators
        protected void AddAccumulate(
                Int64[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                Int64[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                Int64[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] + Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }

        protected void MultiplyAccumulate(
                Int64[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                Int64[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                Int64[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] * Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
#endregion

#region ScalerIterContigNoIter Accelerators
        private void AddSubMultScalerIterContigNoIter(Int64[] src, Int64[] dest, Int64 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.add)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] + operand;
                }
                return;
            }
            if (ops == UFuncOperation.subtract)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] - operand;
                }
                return;
            }
            if (ops == UFuncOperation.multiply)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] * operand;
                }
                return;
            }

        }

        private void DivisionScalerIterContigNoIter(Int64[] src, Int64[] dest, Int64 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (operand == 0)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = 0;
                }
                return;
            }

            if (ops == UFuncOperation.divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.true_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.floor_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToInt64(Math.Floor(Convert.ToDouble(src[srcIndex++]) / Convert.ToDouble(operand)));
                }
                return;
            }
            if (ops == UFuncOperation.remainder)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    var aValue = src[srcIndex++];
                    var rem = aValue % operand;
                    if ((aValue > 0) == (operand > 0) || rem == 0)
                    {
                        dest[destIndex++] = rem;
                    }
                    else
                    {
                        dest[destIndex++] = rem + operand;
                    }
                }
                return;
            }

        }

        private void PowerSqrtScalerIterContigNoIter(Int64[] src, Int64[] dest, Int64 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.power)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToInt64(Math.Pow(src[srcIndex++], operand));
                }
                return;
            }
            if (ops == UFuncOperation.sqrt)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToInt64(Math.Sqrt(src[srcIndex++]));
                }
                return;
            }
            if (ops == UFuncOperation.absolute)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Abs(src[srcIndex++]);
                }
                return;
            }
            if (ops == UFuncOperation.maximum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Max(src[srcIndex++], operand);
                }
                return;
            }
            if (ops == UFuncOperation.minimum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Min(src[srcIndex++], operand);
                }
                return;
            }

        }
#endregion
    }

    internal partial class UFUNC_UInt64 : UFUNC_BASE<UInt64>, IUFUNC_Operations
    {
        #region Accelerator Handlers


        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
           VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
           VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
           UFuncOperation ufop, npy_intp N)
        {
            UInt64[] retArray = Result.datap as UInt64[];
            UInt64[] Op1Array = Operand1.datap as UInt64[];
            UInt64[] Op2Array = Operand2.datap as UInt64[];

            switch (ufop)
            {
                case NumpyLib.UFuncOperation.add:
                    AddAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                                    Op2Array, O2_Index, O2_CalculatedStep,
                                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                case NumpyLib.UFuncOperation.multiply:
                    MultiplyAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                    Op2Array, O2_Index, O2_CalculatedStep,
                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            UInt64[] OperandArray = Operand.datap as UInt64[];
            UInt64[] retArray = Result.datap as UInt64[];
            UInt64 result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                    {
                        retArray[R_Index] = AddReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.subtract:
                    {
                        retArray[R_Index] = SubtractReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.multiply:
                    {
                        retArray[R_Index] = MultiplyReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.divide:
                    {
                        retArray[R_Index] = DivideReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_or:
                    {
                        retArray[R_Index] = LogicalOrReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_and:
                    {
                        retArray[R_Index] = LogicalAndReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.maximum:
                    {
                        retArray[R_Index] = MaximumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.minimum:
                    {
                        retArray[R_Index] = MinimumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }


        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                    return AddReduce;

                case UFuncOperation.subtract:
                    return SubtractReduce;

                case UFuncOperation.multiply:
                    return MultiplyReduce;

                case UFuncOperation.divide:
                    return DivideReduce;

                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                    return MaximumReduce;

                case UFuncOperation.minimum:
                    return MinimumReduce;

            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                    return AddAccumulate;
                case UFuncOperation.multiply:
                    return MultiplyAccumulate;
            }

            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                    return AddSubMultScalerIterContigNoIter;

                case UFuncOperation.divide:
                case UFuncOperation.true_divide:
                case UFuncOperation.floor_divide:
                case UFuncOperation.remainder:
                    return DivisionScalerIterContigNoIter;

                case UFuncOperation.power:
                case UFuncOperation.sqrt:
                case UFuncOperation.absolute:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    return PowerSqrtScalerIterContigNoIter;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
#endregion

#region Reduce accelerators
        protected UInt64 AddReduce(UInt64 result, UInt64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt64 SubtractReduce(UInt64 result, UInt64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result - OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt64 MultiplyReduce(UInt64 result, UInt64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result * OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt64 DivideReduce(UInt64 result, UInt64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = 0;
                else
                    result = result / bValue;

                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt64 LogicalOrReduce(UInt64 result, UInt64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = (UInt64)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt64 LogicalAndReduce(UInt64 result, UInt64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = (UInt64)(boolValue ? 1 : 0);
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt64 MaximumReduce(UInt64 result, UInt64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected UInt64 MinimumReduce(UInt64 result, UInt64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Min(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
#endregion

#region Accumulate accelerators
        protected void AddAccumulate(
          UInt64[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
          UInt64[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
          UInt64[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] + Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }

        protected void MultiplyAccumulate(
                UInt64[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                UInt64[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                UInt64[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] * Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
#endregion


#region ScalerIterContigNoIter Accelerators
        private void AddSubMultScalerIterContigNoIter(UInt64[] src, UInt64[] dest, UInt64 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.add)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] + operand;
                }
                return;
            }
            if (ops == UFuncOperation.subtract)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] - operand;
                }
                return;
            }
            if (ops == UFuncOperation.multiply)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] * operand;
                }
                return;
            }

        }

        private void DivisionScalerIterContigNoIter(UInt64[] src, UInt64[] dest, UInt64 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (operand == 0)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = 0;
                }
                return;
            }

            if (ops == UFuncOperation.divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.true_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.floor_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToUInt64(Math.Floor(Convert.ToDouble(src[srcIndex++]) / Convert.ToDouble(operand)));
                }
                return;
            }
            if (ops == UFuncOperation.remainder)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    var aValue = src[srcIndex++];
                    var rem = aValue % operand;
                    if ((aValue > 0) == (operand > 0) || rem == 0)
                    {
                        dest[destIndex++] = rem;
                    }
                    else
                    {
                        dest[destIndex++] = rem + operand;
                    }
                }
                return;
            }

        }

        private void PowerSqrtScalerIterContigNoIter(UInt64[] src, UInt64[] dest, UInt64 operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.power)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToUInt64(Math.Pow(src[srcIndex++], operand));
                }
                return;
            }
            if (ops == UFuncOperation.sqrt)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToUInt64(Math.Sqrt(src[srcIndex++]));
                }
                return;
            }
            if (ops == UFuncOperation.absolute)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++];
                }
                return;
            }
            if (ops == UFuncOperation.maximum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Max(src[srcIndex++], operand);
                }
                return;
            }
            if (ops == UFuncOperation.minimum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Min(src[srcIndex++], operand);
                }
                return;
            }

        }
#endregion

    }

    internal partial class UFUNC_Float : UFUNC_BASE<float>, IUFUNC_Operations
    {
        #region Accelerator Handlers

        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
                                VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
                                VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
                                UFuncOperation ufop, npy_intp N)
        {
            float[] retArray = Result.datap as float[];
            float[] Op1Array = Operand1.datap as float[];
            float[] Op2Array = Operand2.datap as float[];

            switch (ufop)
            {
                case NumpyLib.UFuncOperation.add:
                    AddAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                                    Op2Array, O2_Index, O2_CalculatedStep,
                                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                case NumpyLib.UFuncOperation.multiply:
                    MultiplyAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                    Op2Array, O2_Index, O2_CalculatedStep,
                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            float[] OperandArray = Operand.datap as float[];
            float[] retArray = Result.datap as float[];
            float result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                    {
                        retArray[R_Index] = AddReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.subtract:
                    {
                        retArray[R_Index] = SubtractReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.multiply:
                    {
                        retArray[R_Index] = MultiplyReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.divide:
                    {
                        retArray[R_Index] = DivideReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_or:
                    {
                        retArray[R_Index] = LogicalOrReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_and:
                    {
                        retArray[R_Index] = LogicalAndReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.maximum:
                    {
                        retArray[R_Index] = MaximumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.minimum:
                    {
                        retArray[R_Index] = MinimumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }
        
        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                    return AddReduce;

                case UFuncOperation.subtract:
                    return SubtractReduce;

                case UFuncOperation.multiply:
                    return MultiplyReduce;

                case UFuncOperation.divide:
                    return DivideReduce;

                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                    return MaximumReduce;

                case UFuncOperation.minimum:
                    return MinimumReduce;

            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                    return AddAccumulate;
                case UFuncOperation.multiply:
                    return MultiplyAccumulate;
            }

            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                    return AddSubMultScalerIterContigNoIter;

                case UFuncOperation.divide:
                case UFuncOperation.true_divide:
                case UFuncOperation.floor_divide:
                case UFuncOperation.remainder:
                    return DivisionScalerIterContigNoIter;

                case UFuncOperation.power:
                case UFuncOperation.sqrt:
                case UFuncOperation.absolute:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    return PowerSqrtScalerIterContigNoIter;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
#endregion

#region Reduce accelerators
        protected float AddReduce(float result, float[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }
            return result;
        }
        protected float SubtractReduce(float result, float[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result - OperandArray[OperIndex];
                OperIndex += OperStep;
            }
            return result;
        }
        protected float MultiplyReduce(float result, float[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result * OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected float DivideReduce(float result, float[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = float.PositiveInfinity;
                else
                    result = result / bValue;

                OperIndex += OperStep;
            }

            return result;
        }
        protected float LogicalOrReduce(float result, float[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected float LogicalAndReduce(float result, float[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected float MaximumReduce(float result, float[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected float MinimumReduce(float result, float[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Min(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
#endregion

#region Accumulate accelerators
        protected void AddAccumulate(
        float[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
        float[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
        float[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] + Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }

        protected void MultiplyAccumulate(
                float[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                float[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                float[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] * Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
#endregion


#region ScalerIterContigNoIter Accelerators
        private void AddSubMultScalerIterContigNoIter(float[] src, float[] dest, float operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.add)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] + operand;
                }
                return;
            }
            if (ops == UFuncOperation.subtract)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] - operand;
                }
                return;
            }
            if (ops == UFuncOperation.multiply)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] * operand;
                }
                return;
            }

        }

        private void DivisionScalerIterContigNoIter(float[] src, float[] dest, float operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (operand == 0)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = float.PositiveInfinity;
                }
                return;
            }

            if (ops == UFuncOperation.divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.true_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.floor_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToSingle(Math.Floor(src[srcIndex++] / operand));
                }
                return;
            }
            if (ops == UFuncOperation.remainder)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    var aValue = src[srcIndex++];
                    var rem = aValue % operand;
                    if ((aValue > 0) == (operand > 0) || rem == 0)
                    {
                        dest[destIndex++] = rem;
                    }
                    else
                    {
                        dest[destIndex++] = rem + operand;
                    }
                }
                return;
            }

        }

        private void PowerSqrtScalerIterContigNoIter(float[] src, float[] dest, float operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.power)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToSingle(Math.Pow(src[srcIndex++], operand));
                }
                return;
            }
            if (ops == UFuncOperation.sqrt)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToSingle(Math.Sqrt(src[srcIndex++]));
                }
                return;
            }
            if (ops == UFuncOperation.absolute)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Abs(src[srcIndex++]);
                }
                return;
            }
            if (ops == UFuncOperation.maximum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Max(src[srcIndex++], operand);
                }
                return;
            }
            if (ops == UFuncOperation.minimum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Min(src[srcIndex++], operand);
                }
                return;
            }

        }

#endregion
    }

    internal partial class UFUNC_Double : UFUNC_BASE<double>, IUFUNC_Operations
    {

  

        #region Accelerator Handlers

        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
                           VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
                           VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
                           UFuncOperation ufop, npy_intp N)
        {
            double[] retArray = Result.datap as double[];
            double[] Op1Array = Operand1.datap as double[];
            double[] Op2Array = Operand2.datap as double[];

            switch (ufop)
            {
                case NumpyLib.UFuncOperation.add:
                    AddAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                                    Op2Array, O2_Index, O2_CalculatedStep,
                                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                case NumpyLib.UFuncOperation.multiply:
                    MultiplyAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                    Op2Array, O2_Index, O2_CalculatedStep,
                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            double[] OperandArray = Operand.datap as double[];
            double[] retArray = Result.datap as double[];
            double result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                    {
                        retArray[R_Index] = AddReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.subtract:
                    {
                        retArray[R_Index] = SubtractReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.multiply:
                    {
                        retArray[R_Index] = MultiplyReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.divide:
                    {
                        retArray[R_Index] = DivideReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_or:
                    {
                        retArray[R_Index] = LogicalOrReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_and:
                    {
                        retArray[R_Index] = LogicalAndReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.maximum:
                    {
                        retArray[R_Index] = MaximumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.minimum:
                    {
                        retArray[R_Index] = MinimumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }


        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                    return AddReduce;

                case UFuncOperation.subtract:
                    return SubtractReduce;

                case UFuncOperation.multiply:
                    return MultiplyReduce;

                case UFuncOperation.divide:
                    return DivideReduce;

                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                    return MaximumReduce;

                case UFuncOperation.minimum:
                    return MinimumReduce;

            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                    return AddAccumulate;
                case UFuncOperation.multiply:
                    return MultiplyAccumulate;
            }

            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                //case UFuncOperation.add:
                //    return AddScalerIter;

                //case UFuncOperation.multiply:
                //    break;

                //case UFuncOperation.divide:
                //    return DivideScalerIter;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                //case UFuncOperation.add:
                //    return AddScalarOuterOpContig;

                //case UFuncOperation.multiply:
                //    return MultiplyScalarOuterOpContig;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                //case UFuncOperation.add:
                //    return AddScalarOuterOpIter;

                //case UFuncOperation.multiply:
                //    return MultiplyScalarOuterOpIter;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                    return AddSubMultScalerIterContigNoIter;

                case UFuncOperation.divide:
                case UFuncOperation.true_divide:
                case UFuncOperation.floor_divide:
                case UFuncOperation.remainder:
                    return DivisionScalerIterContigNoIter;

                case UFuncOperation.power:
                case UFuncOperation.sqrt:
                case UFuncOperation.absolute:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    return PowerSqrtScalerIterContigNoIter;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
#endregion

#region Reduce accelerators
        protected double AddReduce(double result, double[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }
            return result;
        }
        protected double SubtractReduce(double result, double[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result - OperandArray[OperIndex];
                OperIndex += OperStep;
            }
            return result;
        }
        protected double MultiplyReduce(double result, double[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result * OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected double DivideReduce(double result, double[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = double.PositiveInfinity;
                else
                    result = result / bValue;

                OperIndex += OperStep;
            }

            return result;
        }
        protected double LogicalOrReduce(double result, double[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected double LogicalAndReduce(double result, double[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected double MaximumReduce(double result, double[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected double MinimumReduce(double result, double[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Min(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
#endregion

#region Accumulate accelerators
        protected void AddAccumulate(
         double[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
         double[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
         double[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] + Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }

        protected void MultiplyAccumulate(
        double[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
        double[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
        double[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] * Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
#endregion

#region ScalerIter
        protected void AddScalerIter(
            double[] src, npy_intp[] srcOffsets,
            double[] oper, npy_intp[] operOffsets,
            double[] dest, npy_intp[] destOffsets, npy_intp offsetsLen, UFuncOperation ops)
        {
            for (npy_intp i = 0; i < offsetsLen; i++)
            {
                double srcValue, operand;
                npy_intp destIndex;

                srcValue = src[srcOffsets[i]];
                operand = oper[operOffsets[i]];
                destIndex = destOffsets[i];

                dest[destIndex] = srcValue + operand;
            }
        }
        protected void DivideScalerIter(
            double[] src, npy_intp[] srcOffsets,
            double[] oper, npy_intp[] operOffsets,
            double[] dest, npy_intp[] destOffsets, npy_intp offsetsLen, UFuncOperation ops)
        {
            for (npy_intp i = 0; i < offsetsLen; i++)
            {
                double srcValue, operand;
                npy_intp destIndex;

                srcValue = src[srcOffsets[i]];
                operand = oper[operOffsets[i]];
                destIndex = destOffsets[i];

                try
                {
                    if (operand == 0)
                        dest[destIndex] = 0;
                    else
                        dest[destIndex] = srcValue / operand;
                }
                catch
                {
                    dest[destIndex] = 0;
                }
            }
        }
#endregion

#region OuterOpContig
        protected void AddScalarOuterOpContig(NumericOperations operations, double aValue, double[] bValues, npy_intp bSize, double[] dp, npy_intp destIndex, NpyArray destArray, UFuncOperation ops)
        {
            for (npy_intp j = 0; j < bSize; j++)
            {
                dp[destIndex++] = aValue + bValues[j];
            }
        }
        protected void MultiplyScalarOuterOpContig(NumericOperations operations, double aValue, double[] bValues, npy_intp bSize, double[] dp, npy_intp destIndex, NpyArray destArray, UFuncOperation ops)
        {
            for (npy_intp j = 0; j < bSize; j++)
            {
                dp[destIndex++] = aValue * bValues[j];
            }
        }
#endregion

#region OuterOpIter
        void AddScalarOuterOpIter(NumericOperations operations, double aValue, double[] bValues, npy_intp bSize, double[] dp, NpyArrayIterObject DestIter, NpyArray destArray, UFuncOperation ops)
        {
            for (npy_intp j = 0; j < bSize; j++)
            {
                var bValue = bValues[j];

                double destValue = aValue + bValue;
                npy_intp AdjustedIndex = DestIter.dataptr.data_offset >> ItemDiv;

                try
                {
                    dp[AdjustedIndex] = destValue;
                }
                catch
                {
                    operations.destSetItem(AdjustedIndex, 0);
                }
                NpyArray_ITER_NEXT(DestIter);
            }
        }
        void MultiplyScalarOuterOpIter(NumericOperations operations, double aValue, double[] bValues, npy_intp bSize, double[] dp, NpyArrayIterObject DestIter, NpyArray destArray, UFuncOperation ops)
        {
            for (npy_intp j = 0; j < bSize; j++)
            {
                var bValue = bValues[j];

                double destValue = aValue * bValue;
                npy_intp AdjustedIndex = DestIter.dataptr.data_offset >> ItemDiv;

                try
                {
                    dp[AdjustedIndex] = destValue;
                }
                catch
                {
                    operations.destSetItem(AdjustedIndex, 0);
                }
                NpyArray_ITER_NEXT(DestIter);
            }
        }
#endregion

#region ScalerIterContigNoIter Accelerators
        private void AddSubMultScalerIterContigNoIter(double[] src, double[] dest, double operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.add)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] + operand;
                }
                return;
            }
            if (ops == UFuncOperation.subtract)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] - operand;
                }
                return;
            }
            if (ops == UFuncOperation.multiply)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] * operand;
                }
                return;
            }

        }

        private void DivisionScalerIterContigNoIter(double[] src, double[] dest, double operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (operand == 0)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = double.PositiveInfinity;
                }
                return;
            }

            if (ops == UFuncOperation.divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.true_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.floor_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Floor(src[srcIndex++] / operand);
                }
                return;
            }
            if (ops == UFuncOperation.remainder)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    var aValue = src[srcIndex++];
                    var rem = aValue % operand;
                    if ((aValue > 0) == (operand > 0) || rem == 0)
                    {
                        dest[destIndex++] = rem;
                    }
                    else
                    {
                        dest[destIndex++] = rem + operand;
                    }
                }
                return;
            }

        }

        private void PowerSqrtScalerIterContigNoIter(double[] src, double[] dest, double operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.power)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Pow(src[srcIndex++], operand);
                }
                return;
            }
            if (ops == UFuncOperation.sqrt)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Sqrt(src[srcIndex++]);
                }
                return;
            }
            if (ops == UFuncOperation.absolute)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Abs(src[srcIndex++]);
                }
                return;
            }
            if (ops == UFuncOperation.maximum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Max(src[srcIndex++], operand);
                }
                return;
            }
            if (ops == UFuncOperation.minimum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Min(src[srcIndex++], operand);
                }
                return;
            }

        }

 
#endregion
    }

    internal partial class UFUNC_Decimal : UFUNC_BASE<decimal>, IUFUNC_Operations
    {

        #region Accelerator Handlers

        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
                    VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
                    VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
                    UFuncOperation ufop, npy_intp N)
        {
            decimal[] retArray = Result.datap as decimal[];
            decimal[] Op1Array = Operand1.datap as decimal[];
            decimal[] Op2Array = Operand2.datap as decimal[];

            switch (ufop)
            {
                case NumpyLib.UFuncOperation.add:
                    AddAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                                    Op2Array, O2_Index, O2_CalculatedStep,
                                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                case NumpyLib.UFuncOperation.multiply:
                    MultiplyAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                    Op2Array, O2_Index, O2_CalculatedStep,
                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            decimal[] OperandArray = Operand.datap as decimal[];
            decimal[] retArray = Result.datap as decimal[];
            decimal result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                    {
                        retArray[R_Index] = AddReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.subtract:
                    {
                        retArray[R_Index] = SubtractReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.multiply:
                    {
                        retArray[R_Index] = MultiplyReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.divide:
                    {
                        retArray[R_Index] = DivideReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_or:
                    {
                        retArray[R_Index] = LogicalOrReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_and:
                    {
                        retArray[R_Index] = LogicalAndReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.maximum:
                    {
                        retArray[R_Index] = MaximumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.minimum:
                    {
                        retArray[R_Index] = MinimumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }

        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                    return AddReduce;

                case UFuncOperation.subtract:
                    return SubtractReduce;

                case UFuncOperation.multiply:
                    return MultiplyReduce;

                case UFuncOperation.divide:
                    return DivideReduce;

                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                    return MaximumReduce;

                case UFuncOperation.minimum:
                    return MinimumReduce;

            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                    return AddAccumulate;
                case UFuncOperation.multiply:
                    return MultiplyAccumulate;
            }

            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                    return AddSubMultScalerIterContigNoIter;

                case UFuncOperation.divide:
                case UFuncOperation.true_divide:
                case UFuncOperation.floor_divide:
                case UFuncOperation.remainder:
                    return DivisionScalerIterContigNoIter;

                case UFuncOperation.power:
                //case UFuncOperation.sqrt:
                case UFuncOperation.absolute:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    return PowerSqrtScalerIterContigNoIter;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }

#endregion

#region Reduce accelerators
        protected decimal AddReduce(decimal result, decimal[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected decimal SubtractReduce(decimal result, decimal[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result - OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected decimal MultiplyReduce(decimal result, decimal[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result * OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected decimal DivideReduce(decimal result, decimal[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = 0;
                else
                    result = result / bValue;

                OperIndex += OperStep;
            }

            return result;
        }
        protected decimal LogicalOrReduce(decimal result, decimal[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected decimal LogicalAndReduce(decimal result, decimal[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected decimal MaximumReduce(decimal result, decimal[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected decimal MinimumReduce(decimal result, decimal[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Min(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
#endregion

#region Accumulate accelerators
        protected void AddAccumulate(
                decimal[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                decimal[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                decimal[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] + Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
        protected void MultiplyAccumulate(
                decimal[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                decimal[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                decimal[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] * Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
#endregion

#region ScalerIterContigNoIter Accelerators
        private void AddSubMultScalerIterContigNoIter(decimal[] src, decimal[] dest, decimal operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.add)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] + operand;
                }
                return;
            }
            if (ops == UFuncOperation.subtract)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] - operand;
                }
                return;
            }
            if (ops == UFuncOperation.multiply)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] * operand;
                }
                return;
            }

        }

        private void DivisionScalerIterContigNoIter(decimal[] src, decimal[] dest, decimal operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (operand == 0)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = 0;
                }
                return;
            }

            if (ops == UFuncOperation.divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.true_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.floor_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Floor(src[srcIndex++] / operand);
                }
                return;
            }
            if (ops == UFuncOperation.remainder)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    var aValue = src[srcIndex++];
                    var rem = aValue % operand;
                    if ((aValue > 0) == (operand > 0) || rem == 0)
                    {
                        dest[destIndex++] = rem;
                    }
                    else
                    {
                        dest[destIndex++] = rem + operand;
                    }
                }
                return;
            }

        }

        private void PowerSqrtScalerIterContigNoIter(decimal[] src, decimal[] dest, decimal operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.power)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Convert.ToDecimal(Math.Pow(Convert.ToDouble(src[srcIndex++]), Convert.ToDouble(operand)));
                }
                return;
            }
            //if (ops == UFuncOperation.sqrt)
            //{
            //    for (npy_intp index = start; index < end; index++)
            //    {
            //        dest[destIndex++] = Math.Sqrt(src[srcIndex++]);
            //    }
            //    return;
            //}
            if (ops == UFuncOperation.absolute)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Abs(src[srcIndex++]);
                }
                return;
            }
            if (ops == UFuncOperation.maximum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Max(src[srcIndex++], operand);
                }
                return;
            }
            if (ops == UFuncOperation.minimum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Min(src[srcIndex++], operand);
                }
                return;
            }

        }

#endregion
    }

    internal partial class UFUNC_Complex : UFUNC_BASE<System.Numerics.Complex>, IUFUNC_Operations
    {
        #region Accelerator Handlers


        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
                    VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
                    VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
                    UFuncOperation ufop, npy_intp N)
        {
            System.Numerics.Complex[] retArray = Result.datap as System.Numerics.Complex[];
            System.Numerics.Complex[] Op1Array = Operand1.datap as System.Numerics.Complex[];
            System.Numerics.Complex[] Op2Array = Operand2.datap as System.Numerics.Complex[];

            switch (ufop)
            {
                case NumpyLib.UFuncOperation.add:
                    AddAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                                    Op2Array, O2_Index, O2_CalculatedStep,
                                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                case NumpyLib.UFuncOperation.multiply:
                    MultiplyAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                    Op2Array, O2_Index, O2_CalculatedStep,
                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            System.Numerics.Complex[] OperandArray = Operand.datap as System.Numerics.Complex[];
            System.Numerics.Complex[] retArray = Result.datap as System.Numerics.Complex[];
            System.Numerics.Complex result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                    {
                        retArray[R_Index] = AddReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.subtract:
                    {
                        retArray[R_Index] = SubtractReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.multiply:
                    {
                        retArray[R_Index] = MultiplyReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.divide:
                    {
                        retArray[R_Index] = DivideReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_or:
                    {
                        retArray[R_Index] = LogicalOrReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_and:
                    {
                        retArray[R_Index] = LogicalAndReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.maximum:
                    {
                        retArray[R_Index] = MaximumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.minimum:
                    {
                        retArray[R_Index] = MinimumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }


        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                    return AddReduce;

                case UFuncOperation.subtract:
                    return SubtractReduce;

                case UFuncOperation.multiply:
                    return MultiplyReduce;

                case UFuncOperation.divide:
                    return DivideReduce;

                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                    return MaximumReduce;

                case UFuncOperation.minimum:
                    return MinimumReduce;

            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                    return AddAccumulate;
                case UFuncOperation.multiply:
                    return MultiplyAccumulate;
            }

            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.subtract:
                case UFuncOperation.multiply:
                    return AddSubMultScalerIterContigNoIter;

                case UFuncOperation.divide:
                case UFuncOperation.true_divide:
                //case UFuncOperation.floor_divide:
                //case UFuncOperation.remainder:
                    return DivisionScalerIterContigNoIter;

                case UFuncOperation.power:
                case UFuncOperation.sqrt:
                case UFuncOperation.absolute:
                case UFuncOperation.maximum:
                case UFuncOperation.minimum:
                    return PowerSqrtScalerIterContigNoIter;
            }
            return null;
        }

#endregion

#region Reduce accelerators
        protected System.Numerics.Complex AddReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.Complex SubtractReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result - OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.Complex MultiplyReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result * OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.Complex DivideReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = 0;
                else
                    result = result / bValue;

                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.Complex LogicalOrReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.Complex LogicalAndReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.Complex MaximumReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Max(result.Real, OperandArray[OperIndex].Real);
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.Complex MinimumReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = Math.Min(result.Real, OperandArray[OperIndex].Real);
                OperIndex += OperStep;
            }

            return result;
        }
#endregion

#region Accumulate accelerators
        protected void AddAccumulate(
                System.Numerics.Complex[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                System.Numerics.Complex[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                System.Numerics.Complex[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] + Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
        protected void MultiplyAccumulate(
                System.Numerics.Complex[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                System.Numerics.Complex[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                System.Numerics.Complex[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] * Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
#endregion

#region ScalerIterContigNoIter Accelerators
        private void AddSubMultScalerIterContigNoIter(System.Numerics.Complex[] src, System.Numerics.Complex[] dest, System.Numerics.Complex operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.add)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] + operand;
                }
                return;
            }
            if (ops == UFuncOperation.subtract)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] - operand;
                }
                return;
            }
            if (ops == UFuncOperation.multiply)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] * operand;
                }
                return;
            }

        }

        private void DivisionScalerIterContigNoIter(System.Numerics.Complex[] src, System.Numerics.Complex[] dest, System.Numerics.Complex operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (operand == 0)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = 0;
                }
                return;
            }

            if (ops == UFuncOperation.divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            if (ops == UFuncOperation.true_divide)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = src[srcIndex++] / operand;
                }
                return;
            }
            //if (ops == UFuncOperation.floor_divide)
            //{
            //    for (npy_intp index = start; index < end; index++)
            //    {
            //        dest[destIndex++] = Math.Floor(src[srcIndex++] / operand);
            //    }
            //    return;
            //}
            //if (ops == UFuncOperation.remainder)
            //{
            //    for (npy_intp index = start; index < end; index++)
            //    {
            //        var aValue = src[srcIndex++];
            //        var rem = aValue % operand;
            //        if ((aValue > 0) == (operand > 0) || rem == 0)
            //        {
            //            dest[destIndex++] = rem;
            //        }
            //        else
            //        {
            //            dest[destIndex++] = rem + operand;
            //        }
            //    }
            //    return;
            //}

        }

        private void PowerSqrtScalerIterContigNoIter(System.Numerics.Complex[] src, System.Numerics.Complex[] dest, System.Numerics.Complex operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops)
        {
            npy_intp srcIndex = start - srcAdjustment;
            npy_intp destIndex = start - destAdjustment;

            if (ops == UFuncOperation.power)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = System.Numerics.Complex.Pow(src[srcIndex++], operand);
                }
                return;
            }
            if (ops == UFuncOperation.sqrt)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = System.Numerics.Complex.Sqrt(src[srcIndex++]);
                }
                return;
            }
            if (ops == UFuncOperation.absolute)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = System.Numerics.Complex.Abs(src[srcIndex++]);
                }
                return;
            }
            if (ops == UFuncOperation.maximum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Max(src[srcIndex++].Real, operand.Real);
                }
                return;
            }
            if (ops == UFuncOperation.minimum)
            {
                for (npy_intp index = start; index < end; index++)
                {
                    dest[destIndex++] = Math.Min(src[srcIndex++].Real, operand.Real);
                }
                return;
            }

        }
#endregion


    }

    internal partial class UFUNC_BigInt : UFUNC_BASE<System.Numerics.BigInteger>, IUFUNC_Operations
    {
        #region Accelerator Handlers

        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
               VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
               VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
               UFuncOperation ufop, npy_intp N)
        {
            System.Numerics.BigInteger[] retArray = Result.datap as System.Numerics.BigInteger[];
            System.Numerics.BigInteger[] Op1Array = Operand1.datap as System.Numerics.BigInteger[];
            System.Numerics.BigInteger[] Op2Array = Operand2.datap as System.Numerics.BigInteger[];

            switch (ufop)
            {
                case NumpyLib.UFuncOperation.add:
                    AddAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                                    Op2Array, O2_Index, O2_CalculatedStep,
                                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                case NumpyLib.UFuncOperation.multiply:
                    MultiplyAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                    Op2Array, O2_Index, O2_CalculatedStep,
                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            System.Numerics.BigInteger[] OperandArray = Operand.datap as System.Numerics.BigInteger[];
            System.Numerics.BigInteger[] retArray = Result.datap as System.Numerics.BigInteger[];
            System.Numerics.BigInteger result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                    {
                        retArray[R_Index] = AddReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.subtract:
                    {
                        retArray[R_Index] = SubtractReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.multiply:
                    {
                        retArray[R_Index] = MultiplyReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.divide:
                    {
                        retArray[R_Index] = DivideReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_or:
                    {
                        retArray[R_Index] = LogicalOrReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_and:
                    {
                        retArray[R_Index] = LogicalAndReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.maximum:
                    {
                        retArray[R_Index] = MaximumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.minimum:
                    {
                        retArray[R_Index] = MinimumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }

        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                    return AddReduce;

                case UFuncOperation.subtract:
                    return SubtractReduce;

                case UFuncOperation.multiply:
                    return MultiplyReduce;

                case UFuncOperation.divide:
                    return DivideReduce;

                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                    return MaximumReduce;

                case UFuncOperation.minimum:
                    return MinimumReduce;

            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                    return AddAccumulate;
                case UFuncOperation.multiply:
                    return MultiplyAccumulate;
            }

            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
#endregion

#region Reduce accelerators
        protected System.Numerics.BigInteger AddReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.BigInteger SubtractReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result - OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.BigInteger MultiplyReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result * OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.BigInteger DivideReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                var bValue = OperandArray[OperIndex];
                if (bValue == 0)
                    result = 0;
                else
                    result = result / bValue;

                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.BigInteger LogicalOrReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 || OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.BigInteger LogicalAndReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != 0 && OperandArray[OperIndex] != 0;
                result = boolValue ? 1 : 0;
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.BigInteger MaximumReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = System.Numerics.BigInteger.Max(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Numerics.BigInteger MinimumReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = System.Numerics.BigInteger.Min(result, OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
#endregion

#region Accumulate accelerators
        protected void AddAccumulate(
          System.Numerics.BigInteger[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
          System.Numerics.BigInteger[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
          System.Numerics.BigInteger[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] + Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
        protected void MultiplyAccumulate(
                System.Numerics.BigInteger[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                System.Numerics.BigInteger[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                System.Numerics.BigInteger[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] * Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
#endregion

    }

    internal partial class UFUNC_Object : UFUNC_BASE<System.Object>, IUFUNC_Operations
    {
        #region Accelerator Handlers

        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
                            VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
                            VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
                            UFuncOperation ufop, npy_intp N)
        {
            object[] retArray = Result.datap as object[];
            object[] Op1Array = Operand1.datap as object[];
            object[] Op2Array = Operand2.datap as object[];

            switch (ufop)
            {
                case NumpyLib.UFuncOperation.add:
                    AddAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                                    Op2Array, O2_Index, O2_CalculatedStep,
                                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                case NumpyLib.UFuncOperation.multiply:
                    MultiplyAccumulate(Op1Array, O1_Index, O1_CalculatedStep,
                    Op2Array, O2_Index, O2_CalculatedStep,
                    retArray, R_Index, R_CalculatedStep, N);
                    break;

                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            dynamic[] OperandArray = Operand.datap as dynamic[];
            dynamic[] retArray = Result.datap as dynamic[];
            dynamic result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                case UFuncOperation.add:
                    {
                        retArray[R_Index] = AddReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.subtract:
                    {
                        retArray[R_Index] = SubtractReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.multiply:
                    {
                        retArray[R_Index] = MultiplyReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.divide:
                    {
                        retArray[R_Index] = DivideReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_or:
                    {
                        retArray[R_Index] = LogicalOrReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.logical_and:
                    {
                        retArray[R_Index] = LogicalAndReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.maximum:
                    {
                        retArray[R_Index] = MaximumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }
                case UFuncOperation.minimum:
                    {
                        retArray[R_Index] = MinimumReduce(result, OperandArray, OperIndex, OperStep, N);
                        break;
                    }

                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }


        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            // these are the commonly used reduce operations.
            //
            // We can add more by implementing data type specific implementations
            // and adding them to this switch statement

            switch (ops)
            {
                case UFuncOperation.add:
                    return AddReduce;

                case UFuncOperation.subtract:
                    return SubtractReduce;

                case UFuncOperation.multiply:
                    return MultiplyReduce;

                case UFuncOperation.divide:
                    return DivideReduce;

                case UFuncOperation.logical_or:
                    return LogicalOrReduce;

                case UFuncOperation.logical_and:
                    return LogicalAndReduce;

                case UFuncOperation.maximum:
                    return MaximumReduce;

                case UFuncOperation.minimum:
                    return MinimumReduce;

            }

            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                    return AddAccumulate;
                case UFuncOperation.multiply:
                    return MultiplyAccumulate;
            }

            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            switch (ops)
            {
                case UFuncOperation.add:
                case UFuncOperation.multiply:
                    break;
            }
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            return null;
        }
        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
#endregion

#region Reduce accelerators
        protected System.Object AddReduce(dynamic result, dynamic[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Object SubtractReduce(dynamic result, dynamic[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result - OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Object MultiplyReduce(dynamic result, dynamic[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result * OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Object DivideReduce(dynamic result, dynamic[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result / OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Object LogicalOrReduce(dynamic result, dynamic[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != null || OperandArray[OperIndex] != null;
                if (boolValue)
                    result = 1;
                else
                    result = null;
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Object LogicalAndReduce(dynamic result, dynamic[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                bool boolValue = result != null && OperandArray[OperIndex] != null;
                if (boolValue)
                    result = 1;
                else
                    result = null;
                OperIndex += OperStep;
            }
            return result;
        }
        protected System.Object MaximumReduce(dynamic result, dynamic[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result > OperandArray[OperIndex] ? result : OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
        protected System.Object MinimumReduce(dynamic result, dynamic[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result <= OperandArray[OperIndex] ? result : OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
#endregion

#region Accumulate accelerators
        protected void AddAccumulate(
         dynamic[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
         dynamic[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
         dynamic[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] + Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }

        protected void MultiplyAccumulate(
                dynamic[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                dynamic[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                dynamic[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N)
        {
            while (N-- > 0)
            {
                retArray[R_Index] = Op1Array[O1_Index] * Op2Array[O2_Index];

                O1_Index += O1_Step;
                O2_Index += O2_Step;
                R_Index += R_Step;
            }
        }
#endregion
    }

    internal partial class UFUNC_String : UFUNC_BASE<System.String>, IUFUNC_Operations
    {
        #region Accelerator Handlers

        protected override void PerformAccumulateOpArrayIter_XXX(VoidPtr Operand1, npy_intp O1_Index, npy_intp O1_CalculatedStep,
                           VoidPtr Operand2, npy_intp O2_Index, npy_intp O2_CalculatedStep,
                           VoidPtr Result, npy_intp R_Index, npy_intp R_CalculatedStep,
                           UFuncOperation ufop, npy_intp N)
        {
            string[] retArray = Result.datap as string[];
            string[] Op1Array = Operand1.datap as string[];
            string[] Op2Array = Operand2.datap as string[];

            switch (ufop)
            {
   
                default:
                    var UFuncOperation = GetUFuncOperation(ufop);
                    if (UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", ufop.ToString()));
                    }

                    for (npy_intp i = 0; i < N; i++)
                    {
                        var O1Value = Op1Array[O1_Index];                                            // get operand 1
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                        retArray[R_Index] = UFuncOperation(O1Value, O2Value);

                        O1_Index += O1_CalculatedStep;
                        O2_Index += O2_CalculatedStep;
                        R_Index += R_CalculatedStep;
                    }
                    break;

            }
        }

        protected override void PerformReduceOpArrayIter_XXX(VoidPtr Operand, npy_intp N, UFuncOperation op, VoidPtr Result, npy_intp R_Index, npy_intp OperStep, npy_intp O2_CalculatedOffset)
        {
            string[] OperandArray = Operand.datap as string[];
            string[] retArray = Result.datap as string[];
            string result = retArray[R_Index];
            npy_intp OperIndex = ((0 * OperStep) + O2_CalculatedOffset);

            switch (op)
            {
                default:

                    var _UFuncOperation = GetUFuncOperation(op);
                    if (_UFuncOperation == null)
                    {
                        throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                    }

                    // note: these can't be parallelized.
                    for (npy_intp i = 0; i < N; i++)
                    {
                        var Op1Value = result;
                        var Op2Value = OperandArray[OperIndex];

                        result = _UFuncOperation(Op1Value, Op2Value);

                        OperIndex += OperStep;
                    }

                    retArray[R_Index] = result;
                    break;

            }
        }




        protected override opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops)
        {
            return null;
        }

        protected override opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops)
        {
            return null;
        }
        protected override opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops)
        {
            return null;
        }
#endregion
    }
}
