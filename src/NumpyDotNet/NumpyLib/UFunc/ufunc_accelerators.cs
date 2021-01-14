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
}
