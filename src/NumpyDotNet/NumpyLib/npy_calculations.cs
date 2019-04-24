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
        internal delegate object NumericOperation(object bValue, object operand);

        internal static NumericOperation GetOperation(ref NpyArray srcArray, NpyArray_Ops operationType)
        {
            switch (operationType)
            {
                case NpyArray_Ops.npy_op_add:
                    switch (srcArray.ItemType)
                    {
                        case NPY_TYPES.NPY_BOOL:
                            srcArray = NpyArray_CastToType(srcArray, NpyArray_DescrFromType(NPY_TYPES.NPY_INT32), false);
                            return INT32_AddOperation;
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
                        #region AddOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return INT32_AddOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_AddOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_AddOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_AddOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_AddOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_AddOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_AddOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_AddOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_AddOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_AddOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_AddOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_AddOperation;
                            default:
                                return AddOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_subtract:
                    {
                        #region SubtractOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_SubtractOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_SubtractOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_SubtractOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_SubtractOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_SubtractOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_SubtractOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_SubtractOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_SubtractOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_SubtractOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_SubtractOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_SubtractOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_SubtractOperation;
                            default:
                                return SubtractOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_multiply:
                    {
                        #region MultiplyOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_MultiplyOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_MultiplyOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_MultiplyOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_MultiplyOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_MultiplyOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_MultiplyOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_MultiplyOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_MultiplyOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_MultiplyOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_MultiplyOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_MultiplyOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_MultiplyOperation;
                            default:
                                return MultiplyOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_divide:
                    {
                        #region DivideOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_DivideOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_DivideOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_DivideOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_DivideOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_DivideOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_DivideOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_DivideOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_DivideOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_DivideOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_DivideOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_DivideOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_DivideOperation;
                            default:
                                return DivideOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_remainder:
                    {
                        #region RemainderOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_RemainderOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_RemainderOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_RemainderOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_RemainderOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_RemainderOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_RemainderOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_RemainderOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_RemainderOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_RemainderOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_RemainderOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_RemainderOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_RemainderOperation;
                            default:
                                return RemainderOperation;
                        }
                        #endregion
                    }

                case NpyArray_Ops.npy_op_fmod:
                    {
                        #region FModOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_FModOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_FModOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_FModOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_FModOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_FModOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_FModOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_FModOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_FModOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_FModOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_FModOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_FModOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_FModOperation;
                            default:
                                return FModOperation;
                        }
                        #endregion
                    }

                case NpyArray_Ops.npy_op_power:
                    {
                        #region PowerOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_PowerOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_PowerOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_PowerOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_PowerOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_PowerOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_PowerOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_PowerOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_PowerOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_PowerOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_PowerOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_PowerOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_PowerOperation;
                            default:
                                return PowerOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_square:
                    {
                        #region SquareOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_SquareOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_SquareOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_SquareOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_SquareOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_SquareOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_SquareOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_SquareOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_SquareOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_SquareOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_SquareOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_SquareOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_SquareOperation;
                            default:
                                return SquareOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_reciprocal:
                    {
                        #region ReciprocalOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_ReciprocalOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_ReciprocalOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_ReciprocalOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_ReciprocalOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_ReciprocalOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_ReciprocalOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_ReciprocalOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_ReciprocalOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_ReciprocalOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_ReciprocalOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_ReciprocalOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_ReciprocalOperation;
                            default:
                                return ReciprocalOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_ones_like:
                    {
                        return OnesLikeOperation;
                    }
                case NpyArray_Ops.npy_op_sqrt:
                    {
                        #region SqrtOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_SqrtOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_SqrtOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_SqrtOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_SqrtOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_SqrtOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_SqrtOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_SqrtOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_SqrtOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_SqrtOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_SqrtOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_SqrtOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_SqrtOperation;
                            default:
                                return SqrtOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_negative:
                    {
                        #region NegativeOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_NegativeOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_NegativeOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_NegativeOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_NegativeOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_NegativeOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_NegativeOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_NegativeOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_NegativeOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_NegativeOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_NegativeOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_NegativeOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_NegativeOperation;
                            default:
                                return NegativeOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_absolute:
                    {
                        #region AbsoluteOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_AbsoluteOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_AbsoluteOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_AbsoluteOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_AbsoluteOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_AbsoluteOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_AbsoluteOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_AbsoluteOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_AbsoluteOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_AbsoluteOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_AbsoluteOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_AbsoluteOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_AbsoluteOperation;
                            default:
                                return AbsoluteOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_invert:
                    {
                        #region InvertOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_InvertOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_InvertOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_InvertOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_InvertOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_InvertOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_InvertOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_InvertOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_InvertOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_InvertOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_InvertOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_InvertOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_InvertOperation;
                            default:
                                return InvertOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_left_shift:
                    {
                        #region LeftShiftOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_LeftShiftOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_LeftShiftOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_LeftShiftOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_LeftShiftOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_LeftShiftOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_LeftShiftOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_LeftShiftOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_LeftShiftOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_LeftShiftOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_LeftShiftOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_LeftShiftOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_LeftShiftOperation;
                            default:
                                return LeftShiftOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_right_shift:
                    {
                        #region RightShiftOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_RightShiftOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_RightShiftOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_RightShiftOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_RightShiftOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_RightShiftOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_RightShiftOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_RightShiftOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_RightShiftOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_RightShiftOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_RightShiftOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_RightShiftOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_RightShiftOperation;
                            default:
                                return RightShiftOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_bitwise_and:
                    {
                        #region BitWiseAndOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_BitWiseAndOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_BitWiseAndOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_BitWiseAndOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_BitWiseAndOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_BitWiseAndOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_BitWiseAndOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_BitWiseAndOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_BitWiseAndOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_BitWiseAndOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_BitWiseAndOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_BitWiseAndOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_BitWiseAndOperation;
                            default:
                                return BitWiseAndOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_bitwise_xor:
                    {
                        #region BitWiseXorOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_BitWiseXorOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_BitWiseXorOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_BitWiseXorOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_BitWiseXorOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_BitWiseXorOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_BitWiseXorOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_BitWiseXorOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_BitWiseXorOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_BitWiseXorOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_BitWiseXorOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_BitWiseXorOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_BitWiseXorOperation;
                            default:
                                return BitWiseXorOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_bitwise_or:
                    {
                        #region BitWiseOrOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_BitWiseOrOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_BitWiseOrOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_BitWiseOrOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_BitWiseOrOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_BitWiseOrOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_BitWiseOrOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_BitWiseOrOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_BitWiseOrOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_BitWiseOrOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_BitWiseOrOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_BitWiseOrOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_BitWiseOrOperation;
                            default:
                                return BitWiseOrOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_less:
                    {
                        #region LessOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_LessOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_LessOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_LessOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_LessOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_LessOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_LessOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_LessOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_LessOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_LessOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_LessOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_LessOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_LessOperation;
                            default:
                                return LessOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_less_equal:
                    {
                        #region LessEqualOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_LessEqualOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_LessEqualOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_LessEqualOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_LessEqualOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_LessEqualOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_LessEqualOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_LessEqualOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_LessEqualOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_LessEqualOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_LessEqualOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_LessEqualOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_LessEqualOperation;
                            default:
                                return LessEqualOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_equal:
                    {
                        #region EqualOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_EqualOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_EqualOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_EqualOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_EqualOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_EqualOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_EqualOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_EqualOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_EqualOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_EqualOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_EqualOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_EqualOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_EqualOperation;
                            default:
                                return EqualOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_not_equal:
                    {
                        #region NotEqualOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_NotEqualOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_NotEqualOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_NotEqualOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_NotEqualOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_NotEqualOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_NotEqualOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_NotEqualOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_NotEqualOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_NotEqualOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_NotEqualOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_NotEqualOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_NotEqualOperation;
                            default:
                                return NotEqualOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_greater:
                    {
                        #region GreaterOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_GreaterOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_GreaterOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_GreaterOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_GreaterOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_GreaterOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_GreaterOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_GreaterOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_GreaterOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_GreaterOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_GreaterOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_GreaterOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_GreaterOperation;
                            default:
                                return GreaterOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_greater_equal:
                    {
                        #region GreaterEqualOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                return BOOL_GreaterEqualOperation;
                            case NPY_TYPES.NPY_BYTE:
                                return BYTE_GreaterEqualOperation;
                            case NPY_TYPES.NPY_UBYTE:
                                return UBYTE_GreaterEqualOperation;
                            case NPY_TYPES.NPY_INT16:
                                return INT16_GreaterEqualOperation;
                            case NPY_TYPES.NPY_UINT16:
                                return UINT16_GreaterEqualOperation;
                            case NPY_TYPES.NPY_INT32:
                                return INT32_GreaterEqualOperation;
                            case NPY_TYPES.NPY_UINT32:
                                return UINT32_GreaterEqualOperation;
                            case NPY_TYPES.NPY_INT64:
                                return INT64_GreaterEqualOperation;
                            case NPY_TYPES.NPY_UINT64:
                                return UINT64_GreaterEqualOperation;
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_GreaterEqualOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_GreaterEqualOperation;
                            case NPY_TYPES.NPY_DECIMAL:
                                return DECIMAL_GreaterEqualOperation;
                            default:
                                return GreaterEqualOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_isnan:
                    {
                        #region IsNaNOperation
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_IsNaNOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_IsNaNOperation;
                            default:
                                return FLOAT_IsNaNOperation;
                        }
                        #endregion
                    }
                case NpyArray_Ops.npy_op_floor_divide:
                    {
                        return FloorDivideOperation;
                    }
                case NpyArray_Ops.npy_op_true_divide:
                    {
                        return TrueDivideOperation;
                    }
                case NpyArray_Ops.npy_op_logical_or:
                    {
                        return LogicalOrOperation;
                    }
                case NpyArray_Ops.npy_op_logical_and:
                    {
                        return LogicalAndOperation;
                    }
                case NpyArray_Ops.npy_op_floor:
                    {
                        return FloorOperation;
                    }
                case NpyArray_Ops.npy_op_ceil:
                    {
                        return CeilOperation;
                    }
                case NpyArray_Ops.npy_op_maximum:
                    {
                        return MaximumOperation;
                    }
                case NpyArray_Ops.npy_op_fmax:
                    {
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_FMaxOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_FMaxOperation;
                            default:
                                return MaximumOperation;
                        }
                    }
                case NpyArray_Ops.npy_op_minimum:
                    {
                        return MinimumOperation;
                    }
                case NpyArray_Ops.npy_op_fmin:
                    {
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_FMinOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_FMinOperation;
                            default:
                                return MinimumOperation;
                        }
                    }

                case NpyArray_Ops.npy_op_heaviside:
                    {
                        switch (ItemType)
                        {
                            case NPY_TYPES.NPY_FLOAT:
                                return FLOAT_HeavisideOperation;
                            case NPY_TYPES.NPY_DOUBLE:
                                return DOUBLE_HeavisideOperation;
                            default:
                                return DOUBLE_HeavisideOperation;
                        }
                    }

                case NpyArray_Ops.npy_op_rint:
                    {
                        return RintOperation;
                    }
                case NpyArray_Ops.npy_op_conjugate:
                    {
                        return ConjugateOperation;
                    }
  
                default:
                    return null;
            }
        }

        internal static NpyArray NpyArray_PerformNumericOpScalar(NpyArray srcArray, NpyArray_Ops operationType, double operand, bool UseSrcAsDest)
        {
            NpyArray destArray = null;
            if (UseSrcAsDest)
            {
                destArray = srcArray;
            }
            else
            {
                destArray = NpyArray_NumericOpArraySelection(srcArray, null, operationType);
            }

            PerformNumericOpScalar(srcArray, destArray, operand, operationType);
            return destArray;
        }
        internal static NpyArray NpyArray_PerformNumericOpArray(NpyArray srcArray, NpyArray_Ops operationType, NpyArray operandArray, bool UseSrcAsDest)
        {
            // why doesn't this function work?
            //var A1 = NpyArray_PerformNumericOpArray_NpyArray_GenericBinaryFunction(srcArray, operationType, operandArray, UseSrcAsDest);

            NpyArray destArray = null;
            if (UseSrcAsDest)
            {
                destArray = srcArray;
            }
            else
            {
                destArray = NpyArray_NumericOpArraySelection(srcArray, operandArray, operationType);

                //srcArray = NpyArray_NumericOpUpscaleSourceArray(srcArray, operandArray);
                //destArray = NpyArray_NumericOpUpscaleSourceArray(destArray, operandArray);
            }
                
            PerformNumericOpArray(srcArray, destArray, operandArray, operationType);

            //CompareArrays(A1, destArray);
  
            return destArray;
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


        private static void CompareArrays(NpyArray a1, NpyArray a2)
        {
            if (a1.nd != a2.nd)
            {
                throw new Exception("gotcha");
            }

            for (int i = 0; i < a1.nd; i++)
            {
                if (a1.dimensions[i] !=a2.dimensions[i])
                {
                    throw new Exception("gotcha");
                }
            }

            for (int i = 0; i < a1.nd; i++)
            {
                if (a1.strides[i] != a2.strides[i])
                {
                    throw new Exception("gotcha");
                }
            }

            var Length = a1.GetElementCount();
            for (int i = 0; i < Length; i++)
            {
                var a1data = GetIndex(a1.data, i);
                var a2data = GetIndex(a2.data, i);

                if (Convert.ToDouble(a1data) != Convert.ToDouble(a2data))
                {
                    throw new Exception("gotcha");
                }
            }

        }

        // this function does not currently work.  Don't know why.
        internal static NpyArray NpyArray_PerformNumericOpArray_NpyArray_GenericBinaryFunction(NpyArray srcArray, NpyArray_Ops operationType, NpyArray operandArray, bool UseSrcAsDest)
        {

            NpyArray destArray = null;
            if (UseSrcAsDest)
            {
                destArray = srcArray;
            }
            else
            {
                destArray = NpyArray_NumericOpArraySelection(srcArray, operandArray, operationType);
            }

            var UFuncOp = get_op_loc(operationType);
            UFuncOp.nin = 3;
            UFuncOp.nargs = 3;
     
            destArray = NpyArray_GenericBinaryFunction(srcArray, operandArray, UFuncOp, destArray);
            return destArray;
        }

        private static NpyArray NpyArray_NumericOpArraySelection(NpyArray srcArray, NpyArray operandArray, NpyArray_Ops operationType)
        {
            NpyArray_Descr newtype = srcArray.descr;
            NPYARRAYFLAGS flags = srcArray.flags | NPYARRAYFLAGS.NPY_ENSURECOPY | NPYARRAYFLAGS.NPY_FORCECAST;

            if (operandArray != null)
            {
                if (NpyArray_ISFLOAT(operandArray))
                {
                    newtype = NpyArray_DescrFromType(NPY_TYPES.NPY_DOUBLE);
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
                        newtype = NpyArray_DescrFromType(NPY_TYPES.NPY_DOUBLE);
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
                        if (GetTypeSize(srcArray.ItemType) > 4)
                        {
                            newtype = NpyArray_DescrFromType(NPY_TYPES.NPY_DOUBLE);
                        }
                        else
                        {
                            newtype = NpyArray_DescrFromType(NPY_TYPES.NPY_FLOAT);
                        }
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
                        newtype = NpyArray_DescrFromType(NPY_TYPES.NPY_DOUBLE);
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
                newArray = NpyArray_FromArray(srcArray, newtype, flags);
            }
            else
            {
                newArray = NpyArray_FromArray(operandArray, newtype, flags);
            }
            return newArray;
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
                        NpyArray_DescrFromType(NPY_TYPES.NPY_INT64),
                        1, new npy_intp[] { 1 }, false, null);

                    Int64[] Data = repeatArray.data.datap as Int64[];
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
        public static void PerformNumericOpScalar(NpyArray srcArray, NpyArray destArray, double operand, NpyArray_Ops operationType)
        {
            NumericOperation operation = GetOperation(ref srcArray, operationType);

            //PerformNumericOpScalarIter(srcArray, destArray, operand, operation);

            PerformNumericOpScalar(srcArray, destArray, operand, destArray.dimensions, 0, 0, 0, operation);


            //PerformNumericOpScalar2(srcArray, destArray, operand, operation);

        }

    
        private static void PerformNumericOpScalar(NpyArray srcArray, NpyArray destArray, double operand, npy_intp[] dimensions, int dimIdx, long src_offset, long dest_offset, NumericOperation operation)
        {
            if (dimIdx == destArray.nd)
            {
                var srcValue = srcArray.descr.f.getitem(src_offset, srcArray);
                object destValue = null;

                destValue = operation(srcValue, operand);

                try
                {
                    destArray.descr.f.setitem(dest_offset, destValue, destArray);
                }
                catch
                {
                    destArray.descr.f.setitem(dest_offset, 0, destArray);
                }
            }
            else
            {
                for (int i = 0; i < dimensions[dimIdx]; i++)
                {
                    long lsrc_offset = src_offset + srcArray.strides[dimIdx] * i;
                    long ldest_offset = dest_offset + destArray.strides[dimIdx] * i;

                    PerformNumericOpScalar(srcArray, destArray, operand, dimensions, dimIdx + 1, lsrc_offset, ldest_offset, operation);
                }
            }
        }

        private static void PerformNumericOpScalarIter(NpyArray srcArray, NpyArray destArray, double operand, NumericOperation operation)
        {
            var srcSize = NpyArray_Size(srcArray);
            var SrcIter = NpyArray_BroadcastToShape(srcArray, srcArray.dimensions, srcArray.nd);
            var DestIter = NpyArray_BroadcastToShape(destArray, destArray.dimensions, destArray.nd);

            for (long i = 0; i < srcSize; i++)
            {
                var srcValue = srcArray.descr.f.getitem(SrcIter.dataptr.data_offset-srcArray.data.data_offset, srcArray);
                object destValue = null;

                destValue = operation(srcValue, operand);

                try
                {
                    destArray.descr.f.setitem(DestIter.dataptr.data_offset-destArray.data.data_offset, destValue, destArray);
                }
                catch
                {
                    destArray.descr.f.setitem(DestIter.dataptr.data_offset - destArray.data.data_offset, 0, destArray);
                }

                NpyArray_ITER_NEXT(SrcIter);
                NpyArray_ITER_NEXT(DestIter);
            }
        }


        private static void PerformNumericOpScalarIter(NpyArray srcArray, NpyArray destArray, NpyArray operArray, NumericOperation operation)
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

            for (long i = 0; i < destSize; i++)
            {
                var srcValue = srcArray.descr.f.getitem(SrcIter.dataptr.data_offset - srcArray.data.data_offset, srcArray);
                var operValue = operArray.descr.f.getitem(OperIter.dataptr.data_offset - operArray.data.data_offset, operArray);

                object destValue = null;

                destValue = operation(srcValue, Convert.ToDouble(operValue));

                try
                {
                    destArray.descr.f.setitem(DestIter.dataptr.data_offset - destArray.data.data_offset, destValue, destArray);
                }
                catch
                {
                    destArray.descr.f.setitem(DestIter.dataptr.data_offset - destArray.data.data_offset, 0, destArray);
                }

                NpyArray_ITER_NEXT(SrcIter);
                NpyArray_ITER_NEXT(DestIter);
                NpyArray_ITER_NEXT(OperIter);
            }
        }

        private static void PerformOuterOpArrayIter(NpyArray a,  NpyArray b, NpyArray destArray, NumericOperation operation)
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
                var aValue = a.descr.f.getitem(aIter.dataptr.data_offset - a.data.data_offset, a);
                var bIter = NpyArray_IterNew(b);

                for (long j = 0; j < bSize; j++)
                {
                    var bValue = b.descr.f.getitem(bIter.dataptr.data_offset - b.data.data_offset, b);

                    object destValue = operation(aValue, Convert.ToDouble(bValue));

                    try
                    {
                        destArray.descr.f.setitem(DestIter.dataptr.data_offset - destArray.data.data_offset, destValue, destArray);
                    }
                    catch
                    {
                        destArray.descr.f.setitem(DestIter.dataptr.data_offset - destArray.data.data_offset, 0, destArray);
                    }
                    NpyArray_ITER_NEXT(bIter);
                    NpyArray_ITER_NEXT(DestIter);
                }

                NpyArray_ITER_NEXT(aIter);
            }
        }



        internal static int PerformNumericOpScalar2(NpyArray srcArray, NpyArray destArray, double operand, NumericOperation operation)
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
                    var r = operation(srcArray.descr.f.getitem(srcPtr.data_offset, srcArray), operand);
                    try
                    {
                        destArray.descr.f.setitem(destPtr.data_offset, r, destArray);
                    }
                    catch (Exception ex)
                    {
                        destArray.descr.f.setitem(destPtr.data_offset, 0, destArray);
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
                    var r = operation(srcArray.descr.f.getitem(srcIter.dataptr.data_offset, srcArray), operand);
                    destArray.descr.f.setitem(destIter.dataptr.data_offset, r, destArray);
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

            PerformNumericOpScalarIter(srcArray, destArray, operandArray, operation);
        }

        public static NpyArray PerformOuterOpArray(NpyArray srcArray,  NpyArray operandArray, NpyArray destArray, NpyArray_Ops operationType)
        {
            NumericOperation operation = GetOperation(ref srcArray, operationType);
            PerformOuterOpArrayIter(srcArray, operandArray, destArray, operation);
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

        /// <summary>
        /// Recursively walks the array and appends a representation of each element
        /// to the passed string builder.  Square brackets delimit each array dimension.
        /// </summary>
        /// <param name="dimensions">Array of size of each dimension</param>
        /// <param name="src_strides">Offset in bytes to reach next element in each dimension</param>
        /// <param name="dimIdx">Index of the current dimension (starts at 0, recursively counts up)</param>
        /// <param name="src_offset">Byte offset into data array, starts at 0</param>
        private static void PerformNumericOpArray(NpyArray srcArray, NpyArray destArray, NpyArrayWrapper operandArray, npy_intp[] dimensions, int dimIdx, long src_offset, long dest_offset, long operand_offset, NumericOperation operation)
        {
            if (dimIdx == destArray.nd)
            {

                var srcValue = srcArray.descr.f.getitem(src_offset, srcArray);

                var operandValue = operandArray.array.descr.f.getitem(operandArray.GetIndex(), operandArray.array);

                object destValue = operation(srcValue, Convert.ToDouble(operandValue));

                try
                {
                    destArray.descr.f.setitem(dest_offset, destValue, destArray);
                }
                catch
                {
                    destArray.descr.f.setitem(dest_offset, 0, destArray);
                }
            }
            else
            {
                for (int i = 0; i < dimensions[dimIdx]; i++)
                {
                    long lsrc_offset = src_offset + srcArray.strides[dimIdx] * i;
                    long ldest_offset = dest_offset + destArray.strides[dimIdx] * i;

                    operandArray.operand_offset = operand_offset;
                    operandArray.dimIdx = dimIdx;
                    operandArray.i = i;

                    PerformNumericOpArray(srcArray, destArray, operandArray, dimensions, dimIdx + 1, lsrc_offset, ldest_offset, operandArray.GetIndex(), operation);
                }
            }
        }

        private class NpyArrayWrapper
        {
            public NpyArray array = null;
            public long offset = 0;
            public long operand_offset = 0;
            public int dimIdx = 0;
            public int i = 0;

            public NpyArrayWrapper(NpyArray array)
            {
                this.array = array;
                this.offset = 0;
            }

            public long GetIndex()
            {
                if (array == null || array.strides == null || array.strides.Length == 0)
                    return 0;

                if (dimIdx >= array.strides.Length)
                {
                    dimIdx = dimIdx % array.strides.Length;
                    operand_offset = 0;
                }

                long calculatedOffset =  operand_offset + array.strides[dimIdx] * i;
                //Console.WriteLine("x:{0},{1}, {2}", calculatedOffset, operand_offset, i);

                npy_intp ArraySize = NpyArray_SIZE(array);
                npy_intp ItemSize = NpyArray_ITEMSIZE(array);
                npy_intp MaxOffset = ArraySize * ItemSize;

                if (calculatedOffset >= MaxOffset)
                {
                    calculatedOffset = calculatedOffset % MaxOffset;
                }

                return calculatedOffset;
            }
        }

        #endregion

        #region AddOperation
        private static object BOOL_AddOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        private static object BYTE_AddOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue + (double)operand;
        }
        private static object UBYTE_AddOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue + (double)operand;
        }
        private static object INT16_AddOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue + (double)operand;
        }
        private static object UINT16_AddOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue + (double)operand;
        }
        private static object INT32_AddOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue + (double)operand;
        }
        private static object UINT32_AddOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue + (double)operand;
        }
        private static object INT64_AddOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue + (double)operand;
        }
        private static object UINT64_AddOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue + (double)operand;
        }
        private static object FLOAT_AddOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue + (double)operand;
        }
        private static object DOUBLE_AddOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue + (double)operand;
        }
        private static object DECIMAL_AddOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue + (decimal)operand;
        }
        private static T AddOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue + operand;
        }
        #endregion

        #region SubtractOperation
        private static object BOOL_SubtractOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue | Convert.ToBoolean(operand);
        }
        private static object BYTE_SubtractOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue - (double)operand;
        }
        private static object UBYTE_SubtractOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue - (double)operand;
        }
        private static object INT16_SubtractOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue - (double)operand;
        }
        private static object UINT16_SubtractOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue - (double)operand;
        }
        private static object INT32_SubtractOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue - (double)operand;
        }
        private static object UINT32_SubtractOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue - (double)operand;
        }
        private static object INT64_SubtractOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue - (double)operand;
        }
        private static object UINT64_SubtractOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue - (double)operand;
        }
        private static object FLOAT_SubtractOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue - (double)operand;
        }
        private static object DOUBLE_SubtractOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue - (double)operand;
        }
        private static object DECIMAL_SubtractOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue - (decimal)operand;
        }
        private static T SubtractOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue - operand;
        }
        #endregion

        #region MultiplyOperation
        private static object BOOL_MultiplyOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        private static object BYTE_MultiplyOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue * (double)operand;
        }
        private static object UBYTE_MultiplyOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue * (double)operand;
        }
        private static object INT16_MultiplyOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue * (double)operand;
        }
        private static object UINT16_MultiplyOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue * (double)operand;
        }
        private static object INT32_MultiplyOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue * (double)operand;
        }
        private static object UINT32_MultiplyOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue * (double)operand;
        }
        private static object INT64_MultiplyOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue * (double)operand;
        }
        private static object UINT64_MultiplyOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue * (double)operand;
        }
        private static object FLOAT_MultiplyOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue * (double)operand;
        }
        private static object DOUBLE_MultiplyOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue * (double)operand;
        }
        private static object DECIMAL_MultiplyOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue * (decimal)operand;
        }

        private static T MultiplyOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue * operand;
        }
        #endregion

        #region DivideOperation
        private static object BOOL_DivideOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        private static object BYTE_DivideOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / doperand;
        }
        private static object UBYTE_DivideOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / doperand;
        }
        private static object INT16_DivideOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / doperand;
        }
        private static object UINT16_DivideOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / doperand;
        }
        private static object INT32_DivideOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / doperand;
        }
        private static object UINT32_DivideOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / doperand;
        }
        private static object INT64_DivideOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / doperand;
        }
        private static object UINT64_DivideOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / doperand;
        }
        private static object FLOAT_DivideOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / doperand;
        }
        private static object DOUBLE_DivideOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / doperand;
        }
        private static object DECIMAL_DivideOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            decimal doperand = (decimal)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / doperand;
        }

        private static T DivideOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            if (operand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / operand;
        }
        #endregion

        #region FloorDivideOperation
        private static T FloorDivideOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            if (operand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return Math.Floor(dValue / operand);
        }
        #endregion

        #region RemainderOperation
        private static object BOOL_RemainderOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        private static object BYTE_RemainderOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            var rem = dValue % doperand;
            if ((dValue > 0) == (doperand > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + doperand;
            }
        }
        private static object UBYTE_RemainderOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            var rem = dValue % doperand;
            if ((dValue > 0) == (doperand > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + doperand;
            }
        }
        private static object INT16_RemainderOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            var rem = dValue % doperand;
            if ((dValue > 0) == (doperand > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + doperand;
            }
        }
        private static object UINT16_RemainderOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            var rem = dValue % doperand;
            if ((dValue > 0) == (doperand > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + doperand;
            }
        }
        private static object INT32_RemainderOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            var rem = dValue % doperand;
            if ((dValue > 0) == (doperand > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + doperand;
            }
        }
        private static object UINT32_RemainderOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            var rem = dValue % doperand;
            if ((dValue > 0) == (doperand > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + doperand;
            }
        }
        private static object INT64_RemainderOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            var rem = dValue % doperand;
            if ((dValue > 0) == (doperand > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + doperand;
            }
        }
        private static object UINT64_RemainderOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            var rem = dValue % doperand;
            if ((dValue > 0) == (doperand > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + doperand;
            }
        }
        private static object FLOAT_RemainderOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            var rem = dValue % doperand;
            if ((dValue > 0) == (doperand > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + doperand;
            }
        }
        private static object DOUBLE_RemainderOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            var rem = dValue % doperand;
            if ((dValue > 0) == (doperand > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + doperand;
            }
        }
        private static object DECIMAL_RemainderOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            decimal doperand = (decimal)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            var rem = dValue % doperand;
            if ((dValue > 0) == (doperand > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + doperand;
            }
        }

        private static T RemainderOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            if (operand == 0)
            {
                dValue = 0;
                return dValue;
            }
            var rem = dValue % operand;
            if ((dValue > 0) == (operand > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + operand;
            }

        }
        #endregion

        #region FModOperation
        private static object BOOL_FModOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        private static object BYTE_FModOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % doperand;
        }
        private static object UBYTE_FModOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % doperand;
        }
        private static object INT16_FModOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % doperand;
        }
        private static object UINT16_FModOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % doperand;
        }
        private static object INT32_FModOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % doperand;
        }
        private static object UINT32_FModOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % doperand;
        }
        private static object INT64_FModOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % doperand;
        }
        private static object UINT64_FModOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % doperand;
        }
        private static object FLOAT_FModOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % doperand;
        }
        private static object DOUBLE_FModOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            double doperand = (double)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % doperand;
        }
        private static object DECIMAL_FModOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            decimal doperand = (decimal)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % doperand;
        }

        private static T FModOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            if (operand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % operand;
        }
        #endregion

        #region PowerOperation
        private static object BOOL_PowerOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        private static object BYTE_PowerOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object UBYTE_PowerOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object INT16_PowerOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object UINT16_PowerOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object INT32_PowerOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object UINT32_PowerOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object INT64_PowerOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object UINT64_PowerOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object FLOAT_PowerOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object DOUBLE_PowerOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object DECIMAL_PowerOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return Math.Pow(Convert.ToDouble(dValue), Convert.ToDouble(operand));
        }

        private static T PowerOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return Math.Pow(dValue, operand);
        }
        #endregion

        #region SquareOperation
        private static object BOOL_SquareOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ dValue;
        }
        private static object BYTE_SquareOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue * dValue;
        }
        private static object UBYTE_SquareOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue * dValue;
        }
        private static object INT16_SquareOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue * dValue;
        }
        private static object UINT16_SquareOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue * dValue;
        }
        private static object INT32_SquareOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue * dValue;
        }
        private static object UINT32_SquareOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue * dValue;
        }
        private static object INT64_SquareOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue * dValue;
        }
        private static object UINT64_SquareOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue * dValue;
        }
        private static object FLOAT_SquareOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue * dValue;
        }
        private static object DOUBLE_SquareOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue * dValue;
        }
        private static object DECIMAL_SquareOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue * dValue;
        }
        private static T SquareOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue * dValue;
        }
        #endregion

        #region ReciprocalOperation
        private static object BOOL_ReciprocalOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ dValue;
        }
        private static object BYTE_ReciprocalOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return 1 / dValue;
        }
        private static object UBYTE_ReciprocalOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return 1 / dValue;
        }
        private static object INT16_ReciprocalOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return 1 / dValue;
        }
        private static object UINT16_ReciprocalOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return 1 / dValue;
        }
        private static object INT32_ReciprocalOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return 1 / dValue;
        }
        private static object UINT32_ReciprocalOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return 1 / dValue;
        }
        private static object INT64_ReciprocalOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return 1 / dValue;
        }
        private static object UINT64_ReciprocalOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return 1 / dValue;
        }
        private static object FLOAT_ReciprocalOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return 1 / dValue;
        }
        private static object DOUBLE_ReciprocalOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return 1 / dValue;
        }
        private static object DECIMAL_ReciprocalOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return 1 / dValue;
        }
        private static T ReciprocalOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return 1 / dValue;
        }
        #endregion

        #region OnesLikeOperation
        private static object OnesLikeOperation<T>(T bValue, object operand)
        {
            double dValue = 1;
            return dValue;
        }
        #endregion

        #region SqrtOperation
        private static object BOOL_SqrtOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        private static object BYTE_SqrtOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return Math.Sqrt(dValue);
        }
        private static object UBYTE_SqrtOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return Math.Sqrt(dValue);
        }
        private static object INT16_SqrtOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return Math.Sqrt(dValue);
        }
        private static object UINT16_SqrtOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return Math.Sqrt(dValue);
        }
        private static object INT32_SqrtOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return Math.Sqrt(dValue);
        }
        private static object UINT32_SqrtOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return Math.Sqrt(dValue);
        }
        private static object INT64_SqrtOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return Math.Sqrt(dValue);
        }
        private static object UINT64_SqrtOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return Math.Sqrt(dValue);
        }
        private static object FLOAT_SqrtOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return Math.Sqrt(dValue);
        }
        private static object DOUBLE_SqrtOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return Math.Sqrt(dValue);
        }
        private static object DECIMAL_SqrtOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return Math.Sqrt(Convert.ToDouble(dValue));
        }

        private static T SqrtOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return Math.Sqrt(dValue);
        }
        #endregion

        #region NegativeOperation
        private static object BOOL_NegativeOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        private static object BYTE_NegativeOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return -dValue;
        }
        private static object UBYTE_NegativeOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return -dValue;
        }
        private static object INT16_NegativeOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return -dValue;
        }
        private static object UINT16_NegativeOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return -dValue;
        }
        private static object INT32_NegativeOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return -dValue;
        }
        private static object UINT32_NegativeOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return -dValue;
        }
        private static object INT64_NegativeOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return -dValue;
        }
        private static object UINT64_NegativeOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)(UInt64)bValue;
            return (UInt64)(-dValue);
        }
        private static object FLOAT_NegativeOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return -dValue;
        }
        private static object DOUBLE_NegativeOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return -dValue;
        }
        private static object DECIMAL_NegativeOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return -dValue;
        }

        private static T NegativeOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return -dValue;
        }
        #endregion

        #region AbsoluteOperation
        private static object BOOL_AbsoluteOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return false;
        }
        private static object BYTE_AbsoluteOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return Math.Abs(dValue);
        }
        private static object UBYTE_AbsoluteOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return Math.Abs(dValue);
        }
        private static object INT16_AbsoluteOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return Math.Abs(dValue);
        }
        private static object UINT16_AbsoluteOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return Math.Abs(dValue);
        }
        private static object INT32_AbsoluteOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return Math.Abs(dValue);
        }
        private static object UINT32_AbsoluteOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return Math.Abs(dValue);
        }
        private static object INT64_AbsoluteOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return Math.Abs(dValue);
        }
        private static object UINT64_AbsoluteOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return Math.Abs(dValue);
        }
        private static object FLOAT_AbsoluteOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return Math.Abs(dValue);
        }
        private static object DOUBLE_AbsoluteOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return Math.Abs(dValue);
        }
        private static object DECIMAL_AbsoluteOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return Math.Abs(dValue);
        }

        private static T AbsoluteOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return Math.Abs(dValue);
        }
        #endregion

        #region InvertOperation
        private static object BOOL_InvertOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return !dValue;
        }
        private static object BYTE_InvertOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return ~dValue;
        }
        private static object UBYTE_InvertOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return ~dValue;
        }
        private static object INT16_InvertOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return ~dValue;
        }
        private static object UINT16_InvertOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return ~dValue;
        }
        private static object INT32_InvertOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return ~dValue;
        }
        private static object UINT32_InvertOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return ~dValue;
        }
        private static object INT64_InvertOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return ~dValue;
        }
        private static object UINT64_InvertOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return ~dValue;
        }
        private static object FLOAT_InvertOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue;
        }
        private static object DOUBLE_InvertOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue;
        }
        private static object DECIMAL_InvertOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue;
        }

        private static T InvertOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return ~dValue;
        }
        #endregion

        #region LeftShiftOperation
        private static object BOOL_LeftShiftOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue;
        }
        private static object BYTE_LeftShiftOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        private static object UBYTE_LeftShiftOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        private static object INT16_LeftShiftOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        private static object UINT16_LeftShiftOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        private static object INT32_LeftShiftOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        private static object UINT32_LeftShiftOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        private static object INT64_LeftShiftOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        private static object UINT64_LeftShiftOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        private static object FLOAT_LeftShiftOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(float)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        private static object DOUBLE_LeftShiftOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(double)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        private static object DECIMAL_LeftShiftOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(decimal)bValue;
            return dValue << Convert.ToInt32(operand);
        }

        private static T LeftShiftOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue << Convert.ToInt32(operand);
        }
        #endregion

        #region RightShiftOperation
        private static object BOOL_RightShiftOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue;
        }
        private static object BYTE_RightShiftOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        private static object UBYTE_RightShiftOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        private static object INT16_RightShiftOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        private static object UINT16_RightShiftOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        private static object INT32_RightShiftOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        private static object UINT32_RightShiftOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        private static object INT64_RightShiftOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        private static object UINT64_RightShiftOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        private static object FLOAT_RightShiftOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(float)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        private static object DOUBLE_RightShiftOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(double)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        private static object DECIMAL_RightShiftOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(decimal)bValue;
            return dValue >> Convert.ToInt32(operand);
        }

        private static T RightShiftOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        #endregion

        #region BitWiseAndOperation
        private static object BOOL_BitWiseAndOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue & Convert.ToBoolean(operand);
        }
        private static object BYTE_BitWiseAndOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue & Convert.ToSByte(operand);
        }
        private static object UBYTE_BitWiseAndOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue & Convert.ToByte(operand);
        }
        private static object INT16_BitWiseAndOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue & Convert.ToInt16(operand);
        }
        private static object UINT16_BitWiseAndOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue & Convert.ToUInt16(operand);
        }
        private static object INT32_BitWiseAndOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue & Convert.ToInt32(operand);
        }
        private static object UINT32_BitWiseAndOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue & Convert.ToUInt32(operand);
        }
        private static object INT64_BitWiseAndOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue & Convert.ToInt64(operand);
        }
        private static object UINT64_BitWiseAndOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue & Convert.ToUInt64(operand);
        }
        private static object FLOAT_BitWiseAndOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(float)bValue;
            return dValue & Convert.ToUInt64(operand);
        }
        private static object DOUBLE_BitWiseAndOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(double)bValue;
            return dValue & Convert.ToUInt64(operand);
        }
        private static object DECIMAL_BitWiseAndOperation(object bValue, object operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue & Convert.ToUInt64(operand);
        }

        private static T BitWiseAndOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue & Convert.ToUInt64(operand);
        }
        #endregion

        #region BitWiseXorOperation
        private static object BOOL_BitWiseXorOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ Convert.ToBoolean(operand);
        }
        private static object BYTE_BitWiseXorOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue ^ Convert.ToSByte(operand);
        }
        private static object UBYTE_BitWiseXorOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue ^ Convert.ToByte(operand);
        }
        private static object INT16_BitWiseXorOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue ^ Convert.ToInt16(operand);
        }
        private static object UINT16_BitWiseXorOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue ^ Convert.ToUInt16(operand);
        }
        private static object INT32_BitWiseXorOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue ^ Convert.ToInt32(operand);
        }
        private static object UINT32_BitWiseXorOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue ^ Convert.ToUInt32(operand);
        }
        private static object INT64_BitWiseXorOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue ^ Convert.ToInt64(operand);
        }
        private static object UINT64_BitWiseXorOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue ^ Convert.ToUInt64(operand);
        }
        private static object FLOAT_BitWiseXorOperation(object bValue, object operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue ^ Convert.ToUInt64(operand);
        }
        private static object DOUBLE_BitWiseXorOperation(object bValue, object operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue ^ Convert.ToUInt64(operand);
        }
        private static object DECIMAL_BitWiseXorOperation(object bValue, object operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue ^ Convert.ToUInt64(operand);
        }

        private static T BitWiseXorOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue ^ operand;
        }
        #endregion

        #region BitWiseOrOperation
        private static object BOOL_BitWiseOrOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue | Convert.ToBoolean(operand);
        }
        private static object BYTE_BitWiseOrOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue | Convert.ToSByte(operand);
        }
        private static object UBYTE_BitWiseOrOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue | Convert.ToByte(operand);
        }
        private static object INT16_BitWiseOrOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue | Convert.ToInt16(operand);
        }
        private static object UINT16_BitWiseOrOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue | Convert.ToUInt16(operand);
        }
        private static object INT32_BitWiseOrOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue | Convert.ToInt32(operand);
        }
        private static object UINT32_BitWiseOrOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue | Convert.ToUInt32(operand);
        }
        private static object INT64_BitWiseOrOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue | Convert.ToInt64(operand);
        }
        private static object UINT64_BitWiseOrOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue | Convert.ToUInt64(operand);
        }
        private static object FLOAT_BitWiseOrOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(float)bValue;
            return dValue | Convert.ToUInt64(operand);
        }
        private static object DOUBLE_BitWiseOrOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(double)bValue;
            return dValue | Convert.ToUInt64(operand);
        }
        private static object DECIMAL_BitWiseOrOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(decimal)bValue;
            return dValue | Convert.ToUInt64(operand);
        }

        private static T BitWiseOrOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue | (Int64)operand;
        }
        #endregion

        #region LessOperation
        private static object BOOL_LessOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return false;
        }
        private static object BYTE_LessOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue < (double)operand;
        }
        private static object UBYTE_LessOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue < (double)operand;
        }
        private static object INT16_LessOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue < (double)operand;
        }
        private static object UINT16_LessOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue < (double)operand;
        }
        private static object INT32_LessOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue < (double)operand;
        }
        private static object UINT32_LessOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue < (double)operand;
        }
        private static object INT64_LessOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue < (double)operand;
        }
        private static object UINT64_LessOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue < (double)operand;
        }
        private static object FLOAT_LessOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue < (double)operand;
        }
        private static object DOUBLE_LessOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue < (double)operand;
        }
        private static object DECIMAL_LessOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue < Convert.ToDecimal(operand);
        }

        private static T LessOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue < operand;
        }
        #endregion

        #region LessEqualOperation
        private static object BOOL_LessEqualOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return false;
        }
        private static object BYTE_LessEqualOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue <= (double)operand;
        }
        private static object UBYTE_LessEqualOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue <= (double)operand;
        }
        private static object INT16_LessEqualOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue <= (double)operand;
        }
        private static object UINT16_LessEqualOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue <= (double)operand;
        }
        private static object INT32_LessEqualOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue <= (double)operand;
        }
        private static object UINT32_LessEqualOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue <= (double)operand;
        }
        private static object INT64_LessEqualOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue <= (double)operand;
        }
        private static object UINT64_LessEqualOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue <= (double)operand;
        }
        private static object FLOAT_LessEqualOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue <= (double)operand;
        }
        private static object DOUBLE_LessEqualOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue <= (double)operand;
        }
        private static object DECIMAL_LessEqualOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue <= Convert.ToDecimal(operand);
        }

        private static T LessEqualOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue <= operand;
        }
        #endregion

        #region EqualOperation
        private static object BOOL_EqualOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue == Convert.ToBoolean(operand);
        }
        private static object BYTE_EqualOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue == (double)operand;
        }
        private static object UBYTE_EqualOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue == (double)operand;
        }
        private static object INT16_EqualOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue == (double)operand;
        }
        private static object UINT16_EqualOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue == (double)operand;
        }
        private static object INT32_EqualOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue == (double)operand;
        }
        private static object UINT32_EqualOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue == (double)operand;
        }
        private static object INT64_EqualOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue == (double)operand;
        }
        private static object UINT64_EqualOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue == (double)operand;
        }
        private static object FLOAT_EqualOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue == (double)operand;
        }
        private static object DOUBLE_EqualOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue == (double)operand;
        }
        private static object DECIMAL_EqualOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue == Convert.ToDecimal(operand);
        }
        private static T EqualOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue == operand;
        }
        #endregion

        #region NotEqualOperation
        private static object BOOL_NotEqualOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue != Convert.ToBoolean(operand);
        }
        private static object BYTE_NotEqualOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue != (double)operand;
        }
        private static object UBYTE_NotEqualOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue != (double)operand;
        }
        private static object INT16_NotEqualOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue != (double)operand;
        }
        private static object UINT16_NotEqualOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue != (double)operand;
        }
        private static object INT32_NotEqualOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue != (double)operand;
        }
        private static object UINT32_NotEqualOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue != (double)operand;
        }
        private static object INT64_NotEqualOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue != (double)operand;
        }
        private static object UINT64_NotEqualOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue != (double)operand;
        }
        private static object FLOAT_NotEqualOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue != (double)operand;
        }
        private static object DOUBLE_NotEqualOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue != (double)operand;
        }
        private static object DECIMAL_NotEqualOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue != Convert.ToDecimal(operand);
        }
        private static T NotEqualOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue != operand;
        }
        #endregion

        #region GreaterOperation
        private static object BOOL_GreaterOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return true;
        }
        private static object BYTE_GreaterOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue > (double)operand;
        }
        private static object UBYTE_GreaterOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue > (double)operand;
        }
        private static object INT16_GreaterOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue > (double)operand;
        }
        private static object UINT16_GreaterOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue > (double)operand;
        }
        private static object INT32_GreaterOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue > (double)operand;
        }
        private static object UINT32_GreaterOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue > (double)operand;
        }
        private static object INT64_GreaterOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue > (double)operand;
        }
        private static object UINT64_GreaterOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue > (double)operand;
        }
        private static object FLOAT_GreaterOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue > (double)operand;
        }
        private static object DOUBLE_GreaterOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue > (double)operand;
        }
        private static object DECIMAL_GreaterOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue > Convert.ToDecimal(operand);
        }
        private static T GreaterOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue > operand;
        }
        #endregion

        #region GreaterEqualOperation
        private static object BOOL_GreaterEqualOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return true;
        }
        private static object BYTE_GreaterEqualOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue >= (double)operand;
        }
        private static object UBYTE_GreaterEqualOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue >= (double)operand;
        }
        private static object INT16_GreaterEqualOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue >= (double)operand;
        }
        private static object UINT16_GreaterEqualOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue >= (double)operand;
        }
        private static object INT32_GreaterEqualOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue >= (double)operand;
        }
        private static object UINT32_GreaterEqualOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue >= (double)operand;
        }
        private static object INT64_GreaterEqualOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue >= (double)operand;
        }
        private static object UINT64_GreaterEqualOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue >= (double)operand;
        }
        private static object FLOAT_GreaterEqualOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue >= (double)operand;
        }
        private static object DOUBLE_GreaterEqualOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue >= (double)operand;
        }
        private static object DECIMAL_GreaterEqualOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue >= Convert.ToDecimal(operand);
        }
        private static T GreaterEqualOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue >= operand;
        }
        #endregion

        #region IsNaNOperation
        private static object FLOAT_IsNaNOperation(object bValue, object operand)
        {
            return float.IsNaN((float)bValue);
        }
        private static object DOUBLE_IsNaNOperation(object bValue, object operand)
        {
            return double.IsNaN((double)bValue);
        }
        #endregion

        private static T TrueDivideOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue / operand;
        }
        private static T LogicalOrOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue || Convert.ToBoolean(operand);
        }
        private static T LogicalAndOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue && Convert.ToBoolean(operand);
        }
        private static T FloorOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;

            if (bValue is decimal)
            {
                return Math.Floor(Convert.ToDecimal(dValue));
            }
            else
            {
                return Math.Floor(Convert.ToDouble(dValue));
            }
        }
        private static T CeilOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;

            if (bValue is decimal)
            {
                return Math.Ceiling(Convert.ToDecimal(dValue));
            }
            else
            {
                return Math.Ceiling(Convert.ToDouble(dValue));
            }

        }
        private static T MaximumOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;

            if (bValue is decimal)
            {
                return Math.Max(Convert.ToDecimal(dValue), operand);
            }
            else
            {
                return Math.Max(Convert.ToDouble(dValue), operand);
            }

        }

        private static object FLOAT_FMaxOperation(object bValue, dynamic operand)
        {

            if (float.IsNaN(Convert.ToSingle(operand)))
                return bValue;
            if (float.IsNaN(Convert.ToSingle(bValue)))
                return operand;

            return Math.Max(Convert.ToSingle(bValue), Convert.ToSingle(operand));
      
        }

        private static object DOUBLE_FMaxOperation(object bValue, dynamic operand)
        {

            if (double.IsNaN(Convert.ToDouble(operand)))
                return bValue;
            if (double.IsNaN(Convert.ToDouble(bValue)))
                return operand;

            return Math.Max(Convert.ToDouble(bValue), Convert.ToDouble(operand));

        }
   
        private static T MinimumOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;

            if (bValue is decimal)
            {
                return Math.Min(Convert.ToDecimal(dValue), operand);
            }
            else
            {
                return Math.Min(Convert.ToDouble(dValue), operand);
            }

        }

        private static object FLOAT_FMinOperation(object bValue, dynamic operand)
        {

            if (float.IsNaN(Convert.ToSingle(operand)))
                return bValue;
            if (float.IsNaN(Convert.ToSingle(bValue)))
                return operand;

            return Math.Min(Convert.ToSingle(bValue), Convert.ToSingle(operand));

        }

        private static object DOUBLE_FMinOperation(object bValue, dynamic operand)
        {

            if (double.IsNaN(Convert.ToDouble(operand)))
                return bValue;
            if (double.IsNaN(Convert.ToDouble(bValue)))
                return operand;

            return Math.Min(Convert.ToDouble(bValue), Convert.ToDouble(operand));

        }

        private static object FLOAT_HeavisideOperation(object bValue, dynamic operand)
        {
            float x = Convert.ToSingle(bValue);

            if (float.IsNaN(x))
                return float.NaN;

            if (x == 0.0f)
                return Convert.ToSingle(operand);

            if (x < 0.0f)
                return 0.0f;

            return 1.0f;
        }

        private static object DOUBLE_HeavisideOperation(object bValue, dynamic operand)
        {

            double x = Convert.ToDouble(bValue);

            if (double.IsNaN(x))
                return double.NaN;

            if (x == 0.0)
                return Convert.ToDouble(operand);

            if (x < 0.0)
                return 0.0;

            return 1.0;

        }



        private static T RintOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;

            if (bValue is decimal)
            {
                return Math.Round(Convert.ToDecimal(dValue));
            }
            else
            {
                return Math.Round(Convert.ToDouble(dValue));
            }
        }
        private static T ConjugateOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue;
        }
 
  
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
                arg_func(ip, ip.data_offset / elsize, m, ref rptr[i], ap);
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
                arg_func(ip, ip.data_offset / elsize, m, ref rptr[i], ap);
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


        internal static NpyArray NpyArray_Conjugate(NpyArray self, NpyArray outPtr)
        {
            if (NpyArray_ISCOMPLEX(self))
            {
                return NpyArray_GenericUnaryFunction(
                    self,
                    numpyAPI.NpyArray_GetNumericOp(NpyArray_Ops.npy_op_conjugate),
                    outPtr);
            }
            else
            {
                NpyArray ret;
                if (null != outPtr)
                {
                    if (NpyArray_CopyAnyInto(outPtr, self) < 0)
                    {
                        return null;
                    }
                    ret = outPtr;
                }
                else
                {
                    ret = self;
                }
                Npy_INCREF(ret);
                return ret;
            }
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

            object floor = 0;
            PerformUFunc(srcArray, outPtr, ref floor, outPtr.dimensions, 0, 0, 0, operation);


            Npy_DECREF(outPtr);
            return NpyArray_Flatten(outPtr, NPY_ORDER.NPY_ANYORDER);
        }

        internal static NpyArray NpyArray_IsNaN(NpyArray srcArray)
        {
            NumericOperation operation = GetOperation(ref srcArray, NpyArray_Ops.npy_op_isnan);

            NpyArray outPtr = NpyArray_FromArray(srcArray, NpyArray_DescrFromType(NPY_TYPES.NPY_BOOL), NPYARRAYFLAGS.NPY_CONTIGUOUS | NPYARRAYFLAGS.NPY_FORCECAST);

            object floor = 0;
            PerformUFunc(srcArray, outPtr, ref floor, outPtr.dimensions, 0, 0, 0, operation);


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

        private static void PerformUFunc(NpyArray srcArray, NpyArray destArray, ref object cumsum, npy_intp[] dimensions, int dimIdx, long src_offset, long dest_offset, NumericOperation operation)
        {
            if (dimIdx == destArray.nd)
            {
                var srcValue = srcArray.descr.f.getitem(src_offset, srcArray);

                cumsum = operation(srcValue, Convert.ToDouble(cumsum));

                try
                {
                    destArray.descr.f.setitem(dest_offset, cumsum, destArray);
                }
                catch
                {
                    destArray.descr.f.setitem(dest_offset, 0, destArray);
                }
            }
            else
            {
                for (int i = 0; i < dimensions[dimIdx]; i++)
                {
                    long lsrc_offset = src_offset + srcArray.strides[dimIdx] * i;
                    long ldest_offset = dest_offset + destArray.strides[dimIdx] * i;

                    PerformUFunc(srcArray, destArray, ref cumsum, dimensions, dimIdx + 1, lsrc_offset, ldest_offset, operation);
                }
            }
        }
 




    }
}
