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

    }

    internal partial class UFUNC_SByte : UFUNC_BASE<sbyte>, IUFUNC_Operations
    {
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
    }

    internal partial class UFUNC_UByte : UFUNC_BASE<byte>, IUFUNC_Operations
    {
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
    }

    internal partial class UFUNC_Int16 : UFUNC_BASE<Int16>, IUFUNC_Operations
    {
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

        protected Int16 AddReduce(Int16 result, Int16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (Int16)(result + OperandArray[OperIndex]);
                OperIndex += OperStep;
            }
            return result;
        }
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
    }

    internal partial class UFUNC_UInt16 : UFUNC_BASE<UInt16>, IUFUNC_Operations
    {
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

        protected UInt16 AddReduce(UInt16 result, UInt16[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = (UInt16)(result + OperandArray[OperIndex]);
                OperIndex += OperStep;
            }

            return result;
        }
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
    }

    internal partial class UFUNC_Int32 : UFUNC_BASE<Int32>, IUFUNC_Operations
    {
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

        protected Int32 AddReduce(Int32 result, Int32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
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

    }

    internal partial class UFUNC_UInt32 : UFUNC_BASE<UInt32>, IUFUNC_Operations
    {
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

        protected UInt32 AddReduce(UInt32 result, UInt32[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
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
    }

    internal partial class UFUNC_Int64 : UFUNC_BASE<Int64>, IUFUNC_Operations
    {
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

        protected Int64 AddReduce(Int64 result, Int64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
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
    }

    internal partial class UFUNC_UInt64 : UFUNC_BASE<UInt64>, IUFUNC_Operations
    {

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

        protected UInt64 AddReduce(UInt64 result, UInt64[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
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
    }

    internal partial class UFUNC_Float : UFUNC_BASE<float>, IUFUNC_Operations
    {
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

        protected float AddReduce(float result, float[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }
            return result;
        }
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
        protected float DivideReduce(float result, float[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
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
    }

    internal partial class UFUNC_Double : UFUNC_BASE<double>, IUFUNC_Operations
    {

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
        protected double AddReduce(double result, double[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }
            return result;
        }
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

        protected void AddScalerIter(
            double[] src, npy_intp[] srcOffsets,
            double[] oper, npy_intp[] operOffsets,
            double[] dest, npy_intp[] destOffsets, npy_intp offsetsLen, UFuncOperation ops)
        {
            for (npy_intp i = 0; i < offsetsLen; i++)
            {
                double srcValue, operand;
                npy_intp destIndex;

                srcValue = src[srcOffsets[i] >> ItemDiv];
                operand = oper[operOffsets[i] >> ItemDiv];
                destIndex = destOffsets[i] >> ItemDiv;

                dest[destIndex] = srcValue + operand;
            }
        }

        protected void AddScalarOuterOpContig(NumericOperations operations, double aValue, double[] bValues, npy_intp bSize, double[] dp, npy_intp destIndex, NpyArray destArray, UFuncOperation ops)
        {
            for (npy_intp j = 0; j < bSize; j++)
            {
                dp[destIndex++] = aValue + bValues[j];
            }
        }
        void AddScalarOuterOpIter(NumericOperations operations, double aValue, double[] bValues, npy_intp bSize, double[] dp, NpyArrayIterObject DestIter, NpyArray destArray, UFuncOperation ops)
        {
            for (npy_intp j = 0; j < bSize; j++)
            {
                var bValue = bValues[j];

                double destValue = aValue + bValue;

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
        protected void MultiplyScalarOuterOpContig(NumericOperations operations, double aValue, double[] bValues, npy_intp bSize, double[] dp, npy_intp destIndex, NpyArray destArray, UFuncOperation ops)
        {
            for (npy_intp j = 0; j < bSize; j++)
            {
                dp[destIndex++] = aValue * bValues[j];
            }
        }
        void MultiplyScalarOuterOpIter(NumericOperations operations, double aValue, double[] bValues, npy_intp bSize, double[] dp, NpyArrayIterObject DestIter, NpyArray destArray, UFuncOperation ops)
        {
            for (npy_intp j = 0; j < bSize; j++)
            {
                var bValue = bValues[j];

                double destValue = aValue * bValue;

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
        protected double DivideReduce(double result, double[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
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

        protected void DivideScalerIter(
            double[] src, npy_intp[] srcOffsets,
            double[] oper, npy_intp[] operOffsets,
            double[] dest, npy_intp[] destOffsets, npy_intp offsetsLen, UFuncOperation ops)
        {
            for (npy_intp i = 0; i < offsetsLen; i++)
            {
                double srcValue, operand;
                npy_intp destIndex;

                srcValue = src[srcOffsets[i] >> ItemDiv];
                operand = oper[operOffsets[i] >> ItemDiv];
                destIndex = destOffsets[i] >> ItemDiv;

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
    }

    internal partial class UFUNC_Decimal : UFUNC_BASE<decimal>, IUFUNC_Operations
    {
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


        protected decimal AddReduce(decimal result, decimal[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
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
    }

    internal partial class UFUNC_Complex : UFUNC_BASE<System.Numerics.Complex>, IUFUNC_Operations
    {

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

        protected System.Numerics.Complex AddReduce(System.Numerics.Complex result, System.Numerics.Complex[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
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

    }

    internal partial class UFUNC_BigInt : UFUNC_BASE<System.Numerics.BigInteger>, IUFUNC_Operations
    {

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

        protected System.Numerics.BigInteger AddReduce(System.Numerics.BigInteger result, System.Numerics.BigInteger[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }

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

    }

    internal partial class UFUNC_Object : UFUNC_BASE<System.Object>, IUFUNC_Operations
    {

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

        protected System.Object AddReduce(dynamic result, dynamic[] OperandArray, npy_intp OperIndex, npy_intp OperStep, npy_intp N)
        {
            while (N-- > 0)
            {
                result = result + OperandArray[OperIndex];
                OperIndex += OperStep;
            }

            return result;
        }
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
    }

    internal partial class UFUNC_String : UFUNC_BASE<System.String>, IUFUNC_Operations
    {

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
    }
}
