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
    internal partial class numpyinternal
    {
        #region NumericOP
        internal static NpyUFuncObject NpyArray_GetNumericOp(UFuncOperation op)
        {
            NpyUFuncObject loc = get_op_loc(op);
            return (null != loc) ? loc : null;
        }

        internal static NpyUFuncObject NpyArray_SetNumericOp(UFuncOperation op, NpyUFuncObject func)
        {
            throw new NotImplementedException();
        }

        private static npy_intp AdjustNegativeIndex<T>(VoidPtr data, npy_intp index)
        {
            if (index < 0)
            {
                T[] dp = data.datap as T[];
                index = dp.Length - -index;
            }
            return index;
        }

        static NpyUFuncObject get_op_loc(UFuncOperation op)
        {
            NpyUFuncObject loc = null;

            switch (op)
            {
                case UFuncOperation.add:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.maximum:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.minimum:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.multiply:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.logical_or:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.logical_and:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.subtract:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.divide:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.remainder:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.fmod:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.power:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.square:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.reciprocal:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.ones_like:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.sqrt:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.negative:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.absolute:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.invert:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.left_shift:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.right_shift:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.bitwise_and:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.bitwise_xor:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.bitwise_or:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.less:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.less_equal:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.equal:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.not_equal:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.greater:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.greater_equal:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.floor_divide:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.true_divide:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.floor:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.ceil:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.rint:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.conjugate:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.isnan:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.fmax:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.fmin:
                    return DefaultOpControl(op, UFuncCommon);
                case UFuncOperation.heaviside:
                    return DefaultOpControl(op, UFuncCommon);
                default:
                    loc = null;
                    break;
            }
            return loc;
        }

        private static NpyUFuncObject DefaultOpControl(UFuncOperation ops, NpyUFuncGenericFunction UFunc)
        {
            var loc = new NpyUFuncObject(UFunc)
            {
                ops = ops,
                name = "add",
                identity = NpyUFuncIdentity.NpyUFunc_Zero,
                nin = 1,
                nargs = 1,
                types = new NPY_TYPES[] {NPY_TYPES.NPY_BYTE, NPY_TYPES.NPY_UBYTE,
                                         NPY_TYPES.NPY_INT16, NPY_TYPES.NPY_UINT16,
                                         NPY_TYPES.NPY_INT32, NPY_TYPES.NPY_UINT32,
                                         NPY_TYPES.NPY_INT64, NPY_TYPES.NPY_UINT64,
                                         NPY_TYPES.NPY_FLOAT, NPY_TYPES.NPY_DOUBLE,
                                         NPY_TYPES.NPY_DECIMAL, NPY_TYPES.NPY_COMPLEX,
                                         NPY_TYPES.NPY_BIGINT, NPY_TYPES.NPY_OBJECT, NPY_TYPES.NPY_STRING},

            };

            return loc;
        }
        #endregion

        #region UFUNC Handlers

        internal static UFUNC_Bool Instance_UFUNC_Bool = new UFUNC_Bool();
        internal static UFUNC_SByte Instance_UFUNC_SByte = new UFUNC_SByte();
        internal static UFUNC_UByte Instance_UFUNC_UByte = new UFUNC_UByte();
        internal static UFUNC_Int16 Instance_UFUNC_Int16 = new UFUNC_Int16();
        internal static UFUNC_UInt16 Instance_UFUNC_UInt16 = new UFUNC_UInt16();
        internal static UFUNC_Int32 Instance_UFUNC_Int32 = new UFUNC_Int32();
        internal static UFUNC_UInt32 Instance_UFUNC_UInt32 = new UFUNC_UInt32();
        internal static UFUNC_Int64 Instance_UFUNC_Int64 = new UFUNC_Int64();
        internal static UFUNC_UInt64 Instance_UFUNC_UInt64 = new UFUNC_UInt64();
        internal static UFUNC_Float Instance_UFUNC_Float = new UFUNC_Float();
        internal static UFUNC_Double Instance_UFunc_Double = new UFUNC_Double();
        internal static UFUNC_Decimal Instance_UFUNC_Decimal = new UFUNC_Decimal();
        internal static UFUNC_Complex Instance_UFUNC_Complex = new UFUNC_Complex();
        internal static UFUNC_BigInt Instance_UFUNC_BigInt = new UFUNC_BigInt();
        internal static UFUNC_Object Instance_UFUNC_Object = new UFUNC_Object();
        internal static UFUNC_String Instance_UFUNC_String = new UFUNC_String();

        private static IUFUNC_Operations GetUFuncHandler(NPY_TYPES npy_type)
        {
            switch (npy_type)
            {
                case NPY_TYPES.NPY_BOOL:
                    return Instance_UFUNC_Bool;

                case NPY_TYPES.NPY_BYTE:
                    return Instance_UFUNC_SByte;

                case NPY_TYPES.NPY_UBYTE:
                    return Instance_UFUNC_UByte;

                case NPY_TYPES.NPY_INT16:
                    return Instance_UFUNC_Int16;

                case NPY_TYPES.NPY_UINT16:
                    return Instance_UFUNC_UInt16;

                case NPY_TYPES.NPY_INT32:
                    return Instance_UFUNC_Int32;

                case NPY_TYPES.NPY_UINT32:
                    return Instance_UFUNC_UInt32;

                case NPY_TYPES.NPY_INT64:
                    return Instance_UFUNC_Int64;

                case NPY_TYPES.NPY_UINT64:
                    return Instance_UFUNC_UInt64;

                case NPY_TYPES.NPY_FLOAT:
                    return Instance_UFUNC_Float;

                case NPY_TYPES.NPY_DOUBLE:
                    return Instance_UFunc_Double;

                case NPY_TYPES.NPY_DECIMAL:
                    return Instance_UFUNC_Decimal;

                case NPY_TYPES.NPY_COMPLEX:
                    return Instance_UFUNC_Complex;

                case NPY_TYPES.NPY_BIGINT:
                    return Instance_UFUNC_BigInt;

                case NPY_TYPES.NPY_OBJECT:
                    return Instance_UFUNC_Object;

                case NPY_TYPES.NPY_STRING:
                    return Instance_UFUNC_String;

                default:
                    return null;
            }
        }

        internal static UFuncGeneralReductionHandler GetGeneralReductionUFuncHandler(GenericReductionOp op, VoidPtr[] bufPtr)
        {
            VoidPtr Result = bufPtr[2];

            if (Result.type_num == bufPtr[0].type_num && Result.type_num == bufPtr[0].type_num)
            {
                IUFUNC_Operations UFunc = GetUFuncHandler(Result.type_num);
                if (UFunc != null)
                {
                    if (op == GenericReductionOp.NPY_UFUNC_REDUCE)
                    {
                        return UFunc.PerformReduceOpArrayIter;
                    }
                    if (op == GenericReductionOp.NPY_UFUNC_ACCUMULATE)
                    {
                        return UFunc.PerformAccumulateOpArrayIter;
                    }
                    if (op == GenericReductionOp.NPY_UFUNC_REDUCEAT)
                    {
                        return UFunc.PerformReduceAtOpArrayIter;
                    }
                }
            }

            return null;
        }

        internal static void UFuncCommon(GenericReductionOp op, VoidPtr[] bufPtr, npy_intp N, npy_intp[] steps, UFuncOperation Ops)
        {
            VoidPtr Result = bufPtr[2];

            if (Result.type_num == bufPtr[0].type_num && Result.type_num == bufPtr[0].type_num)
            {
                IUFUNC_Operations UFunc = GetUFuncHandler(Result.type_num);
                if (UFunc != null)
                {
                    if (op == GenericReductionOp.NPY_UFUNC_REDUCE)
                    {
                        UFunc.PerformReduceOpArrayIter(bufPtr, steps, Ops, N);
                        return;
                    }
                    if (op == GenericReductionOp.NPY_UFUNC_ACCUMULATE)
                    {
                        UFunc.PerformAccumulateOpArrayIter(bufPtr, steps, Ops, N);
                        return;
                    }
                    if (op == GenericReductionOp.NPY_UFUNC_REDUCEAT)
                    {
                        UFunc.PerformReduceAtOpArrayIter(bufPtr, steps, Ops, N);
                        return;
                    }
                }
            }

            throw new Exception(string.Format("Unexpected UFUNC Type : {0}", Result.type_num.ToString()));

            //switch (Result.type_num)
            //{
            //    case NPY_TYPES.NPY_BOOL:
            //        UFuncCommon_R<bool>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_BYTE:
            //        UFuncCommon_R<sbyte>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_UBYTE:
            //        UFuncCommon_R<byte>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_INT16:
            //        UFuncCommon_R<Int16>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_UINT16:
            //        UFuncCommon_R<UInt16>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_INT32:
            //        UFuncCommon_R<Int32>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_UINT32:
            //        UFuncCommon_R<UInt32>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_INT64:
            //        UFuncCommon_R<Int64>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_UINT64:
            //        UFuncCommon_R<UInt64>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_FLOAT:
            //        UFuncCommon_R<float>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_DOUBLE:
            //        UFuncCommon_R<double>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_DECIMAL:
            //        UFuncCommon_R<decimal>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_COMPLEX:
            //        UFuncCommon_R<System.Numerics.Complex>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_BIGINT:
            //        UFuncCommon_R<System.Numerics.BigInteger>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_OBJECT:
            //        UFuncCommon_R<object>(op, bufPtr, N, steps, Ops);
            //        break;
            //    case NPY_TYPES.NPY_STRING:
            //        UFuncCommon_R<string>(op, bufPtr, N, steps, Ops);
            //        break;
            //}

            return;
        }

        //private static void UFuncCommon_R<R>(GenericReductionOp op, VoidPtr[] bufPtr, npy_intp N, npy_intp[] steps, UFuncOperation Ops)
        //{
        //    VoidPtr Result = bufPtr[0];

        //    switch (Result.type_num)
        //    {
        //        case NPY_TYPES.NPY_BOOL:
        //            UFuncCommon_RO<R, bool>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_BYTE:
        //            UFuncCommon_RO<R, sbyte>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_UBYTE:
        //            UFuncCommon_RO<R, byte>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_INT16:
        //            UFuncCommon_RO<R, Int16>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_UINT16:
        //            UFuncCommon_RO<R, UInt16>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_INT32:
        //            UFuncCommon_RO<R, Int32>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_UINT32:
        //            UFuncCommon_RO<R, UInt32>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_INT64:
        //            UFuncCommon_RO<R, Int64>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_UINT64:
        //            UFuncCommon_RO<R, UInt64>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_FLOAT:
        //            UFuncCommon_RO<R, float>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_DOUBLE:
        //            UFuncCommon_RO<R, double>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_DECIMAL:
        //            UFuncCommon_RO<R, decimal>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_COMPLEX:
        //            UFuncCommon_RO<R, System.Numerics.Complex>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_BIGINT:
        //            UFuncCommon_RO<R, System.Numerics.BigInteger>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_OBJECT:
        //            UFuncCommon_RO<R, object>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_STRING:
        //            UFuncCommon_RO<R, string>(op, bufPtr, N, steps, Ops);
        //            break;
        //    }

        //    return;
        //}

        //private static void UFuncCommon_RO<R, O1>(GenericReductionOp op, VoidPtr[] bufPtr, npy_intp N, npy_intp[] steps, UFuncOperation Ops)
        //{
        //    VoidPtr Result = bufPtr[1];

        //    switch (Result.type_num)
        //    {
        //        case NPY_TYPES.NPY_BOOL:
        //            UFuncCommon_ROO<R, O1, bool>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_BYTE:
        //            UFuncCommon_ROO<R, O1, sbyte>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_UBYTE:
        //            UFuncCommon_ROO<R, O1, byte>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_INT16:
        //            UFuncCommon_ROO<R, O1, Int16>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_UINT16:
        //            UFuncCommon_ROO<R, O1, UInt16>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_INT32:
        //            UFuncCommon_ROO<R, O1, Int32>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_UINT32:
        //            UFuncCommon_ROO<R, O1, UInt32>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_INT64:
        //            UFuncCommon_ROO<R, O1, Int64>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_UINT64:
        //            UFuncCommon_ROO<R, O1, UInt64>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_FLOAT:
        //            UFuncCommon_ROO<R, O1, float>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_DOUBLE:
        //            UFuncCommon_ROO<R, O1, double>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_DECIMAL:
        //            UFuncCommon_ROO<R, O1, decimal>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_COMPLEX:
        //            UFuncCommon_ROO<R, O1, System.Numerics.Complex>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_BIGINT:
        //            UFuncCommon_ROO<R, O1, System.Numerics.BigInteger>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_OBJECT:
        //            UFuncCommon_ROO<R, O1, object>(op, bufPtr, N, steps, Ops);
        //            break;
        //        case NPY_TYPES.NPY_STRING:
        //            UFuncCommon_ROO<R, O1, string>(op, bufPtr, N, steps, Ops);
        //            break;
        //    }
        //}

        //private static void UFuncCommon_ROO<R, O1, O2>(GenericReductionOp op, VoidPtr[] bufPtr, npy_intp N, npy_intp[] steps, UFuncOperation ops)
        //{
        //    if (op == GenericReductionOp.NPY_UFUNC_REDUCE)
        //    {
        //        UFuncCommon_REDUCE<R, O1, O2>(bufPtr, N, steps, ops);
        //        return;
        //    }

        //    if (op == GenericReductionOp.NPY_UFUNC_ACCUMULATE)
        //    {
        //        UFuncCommon_ACCUMULATE<R, O1, O2>(bufPtr, N, steps, ops);
        //        return;
        //    }

        //    if (op == GenericReductionOp.NPY_UFUNC_REDUCEAT)
        //    {
        //        UFuncCommon_REDUCEAT<R, O1, O2>(bufPtr, N, steps, ops);
        //        return;
        //    }

        //    if (op == GenericReductionOp.NPY_UFUNC_OUTER)
        //    {
        //        UFuncCommon_OUTER<R, O1, O2>(bufPtr, N, steps, ops);
        //        return;
        //    }

        //    throw new Exception("Unexpected UFUNC TYPE");

        //}

        //private static void UFuncCommon_REDUCE<R, O1, O2>(VoidPtr[] bufPtr, npy_intp N, npy_intp[] steps, UFuncOperation ops)
        //{
        //    VoidPtr Operand1 = bufPtr[0];
        //    VoidPtr Operand2 = bufPtr[1];
        //    VoidPtr Result = bufPtr[2];

        //    npy_intp O1_Step = steps[0];
        //    npy_intp O2_Step = steps[1];
        //    npy_intp R_Step = steps[2];

        //    if (Operand2 == null)
        //    {
        //        Operand2 = Operand1;
        //        O2_Step = O1_Step;
        //    }
        //    if (Result == null)
        //    {
        //        Result = Operand1;
        //        R_Step = O1_Step;
        //    }



        //    NumericOperation Operation = GetOperation(Operand1, ops);
        //    var Operand1Handler = DefaultArrayHandlers.GetArrayHandler(Operand1.type_num);
        //    var Operand2Handler = DefaultArrayHandlers.GetArrayHandler(Operand2.type_num);
        //    var ResultHandler = DefaultArrayHandlers.GetArrayHandler(Result.type_num);

        //    int O1_sizeDiv = Operand1Handler.ItemDiv;
        //    int O2_sizeDiv = Operand2Handler.ItemDiv;
        //    int R_sizeDiv = ResultHandler.ItemDiv;

        //    npy_intp O1_Offset = Operand1.data_offset;
        //    npy_intp O2_Offset = Operand2.data_offset;
        //    npy_intp R_Offset = Result.data_offset;

        //    npy_intp R_Index = AdjustNegativeIndex<R>(Result, R_Offset >> R_sizeDiv);
        //    npy_intp O1_Index = AdjustNegativeIndex<O1>(Operand1, O1_Offset >> O1_sizeDiv);

        //    R[] retArray = Result.datap as R[];
        //    O1[] Op1Array = Operand1.datap as O1[];
        //    O2[] Op2Array = Operand2.datap as O2[];

        //    npy_intp O2_CalculatedStep = (O2_Step >> O2_sizeDiv);
        //    npy_intp O2_CalculatedOffset = (O2_Offset >> O2_sizeDiv);

        //    bool ThrewException = false;

        //    npy_intp O2_Index = ((0 * O2_CalculatedStep) + O2_CalculatedOffset);

        //    for (int i = 0; i < N; i++)
        //    {

        //        try
        //        {
        //            var O1Value = Op1Array[O1_Index];
        //            var O2Value = Op2Array[O2_Index];                                            // get operand 2

        //            retry:
        //            if (ThrewException)
        //            {
        //                var RRValue = Operation(O1Value, Operand1Handler.MathOpConvertOperand(O1Value, O2Value));    // calculate result
        //                ResultHandler.SetIndex(Result, R_Index, RRValue);
        //            }
        //            else
        //            {
        //                try
        //                {
        //                    R RValue = (R)Operation(O1Value, O2Value);    // calculate result
        //                    retArray[R_Index] = RValue;
        //                }
        //                catch
        //                {
        //                    ThrewException = true;
        //                    goto retry;
        //                }

        //            }

        //        }
        //        catch (System.OverflowException oe)
        //        {
        //            NpyErr_SetString(npyexc_type.NpyExc_OverflowError, oe.Message);
        //        }
        //        catch (Exception ex)
        //        {
        //            NpyErr_SetString(npyexc_type.NpyExc_ValueError, ex.Message);
        //        }

        //        O2_Index += O2_CalculatedStep;
        //    }

        //    return;
        //}

        //private static void UFuncCommon_ACCUMULATE<R, O1, O2>(VoidPtr[] bufPtr, long N, long[] steps, UFuncOperation ops)
        //{
        //    VoidPtr Operand1 = bufPtr[0];
        //    VoidPtr Operand2 = bufPtr[1];
        //    VoidPtr Result = bufPtr[2];

        //    npy_intp O1_Step = steps[0];
        //    npy_intp O2_Step = steps[1];
        //    npy_intp R_Step = steps[2];

        //    if (Operand2 == null)
        //    {
        //        Operand2 = Operand1;
        //        O2_Step = O1_Step;
        //    }
        //    if (Result == null)
        //    {
        //        Result = Operand1;
        //        R_Step = O1_Step;
        //    }

        //    NumericOperation Operation = GetOperation(Operand1, ops);
        //    var Operand1Handler = DefaultArrayHandlers.GetArrayHandler(Operand1.type_num);
        //    var Operand2Handler = DefaultArrayHandlers.GetArrayHandler(Operand2.type_num);
        //    var ResultHandler = DefaultArrayHandlers.GetArrayHandler(Result.type_num);

        //    int O1_sizeDiv = Operand1Handler.ItemDiv;
        //    int O2_sizeDiv = Operand2Handler.ItemDiv;
        //    int R_sizeDiv = ResultHandler.ItemDiv;

        //    npy_intp O1_Offset = Operand1.data_offset;
        //    npy_intp O2_Offset = Operand2.data_offset;
        //    npy_intp R_Offset = Result.data_offset;


        //    R[] retArray = Result.datap as R[];
        //    O1[] Op1Array = Operand1.datap as O1[];
        //    O2[] Op2Array = Operand2.datap as O2[];

        //    npy_intp O1_CalculatedStep = (O1_Step >> O1_sizeDiv);
        //    npy_intp O1_CalculatedOffset = (O1_Offset >> O1_sizeDiv);

        //    npy_intp O2_CalculatedStep = (O2_Step >> O2_sizeDiv);
        //    npy_intp O2_CalculatedOffset = (O2_Offset >> O2_sizeDiv);

        //    npy_intp R_CalculatedStep = (R_Step >> R_sizeDiv);
        //    npy_intp R_CalculatedOffset = (R_Offset >> R_sizeDiv);

        //    bool ThrewException = false;

        //    npy_intp O1_Index = ((0 * O1_CalculatedStep) + O1_CalculatedOffset);
        //    npy_intp O2_Index = ((0 * O2_CalculatedStep) + O2_CalculatedOffset);
        //    npy_intp R_Index = ((0 * R_CalculatedStep) + R_CalculatedOffset);

        //    for (int i = 0; i < N; i++)
        //    {
        //        var O1Value = Op1Array[O1_Index];                                            // get operand 1
        //        var O2Value = Op2Array[O2_Index];                                            // get operand 2

        //        try
        //        {
        //            retry:
        //            if (ThrewException)
        //            {
        //                var RRValue = Operation(O1Value, Operand1Handler.MathOpConvertOperand(O1Value, O2Value));    // calculate result
        //                ResultHandler.SetIndex(Result, R_Index, RRValue);
        //            }
        //            else
        //            {
        //                try
        //                {
        //                    R RValue = (R)Operation(O1Value, Operand1Handler.MathOpConvertOperand(O1Value, O2Value));    // calculate result
        //                    retArray[R_Index] = RValue;
        //                }
        //                catch
        //                {
        //                    ThrewException = true;
        //                    goto retry;
        //                }
        //            }

        //        }
        //        catch (System.OverflowException oe)
        //        {
        //            NpyErr_SetString(npyexc_type.NpyExc_OverflowError, oe.Message);
        //        }
        //        catch (Exception ex)
        //        {
        //            NpyErr_SetString(npyexc_type.NpyExc_ValueError, ex.Message);
        //        }

        //        O1_Index += O1_CalculatedStep;
        //        O2_Index += O2_CalculatedStep;
        //        R_Index += R_CalculatedStep;
        //    }
        //}

        //private static void UFuncCommon_REDUCEAT<R, O1, O2>(VoidPtr[] bufPtr, long N, long[] steps, UFuncOperation ops)
        //{
        //    VoidPtr Operand1 = bufPtr[0];
        //    VoidPtr Operand2 = bufPtr[1];
        //    VoidPtr Result = bufPtr[2];

        //    npy_intp O1_Step = steps[0];
        //    npy_intp O2_Step = steps[1];
        //    npy_intp R_Step = steps[2];

        //    if (Operand2 == null)
        //    {
        //        Operand2 = Operand1;
        //        O2_Step = O1_Step;
        //    }
        //    if (Result == null)
        //    {
        //        Result = Operand1;
        //        R_Step = O1_Step;
        //    }

        //    NumericOperation Operation = GetOperation(Operand1, ops);
        //    var Operand1Handler = DefaultArrayHandlers.GetArrayHandler(Operand1.type_num);
        //    var Operand2Handler = DefaultArrayHandlers.GetArrayHandler(Operand2.type_num);
        //    var ResultHandler = DefaultArrayHandlers.GetArrayHandler(Result.type_num);

        //    int O1_sizeDiv = Operand1Handler.ItemDiv;
        //    int O2_sizeDiv = Operand2Handler.ItemDiv;
        //    int R_sizeDiv = ResultHandler.ItemDiv;

        //    npy_intp O1_Offset = Operand1.data_offset;
        //    npy_intp O2_Offset = Operand2.data_offset;
        //    npy_intp R_Offset = Result.data_offset;


        //    R[] retArray = Result.datap as R[];
        //    O1[] Op1Array = Operand1.datap as O1[];
        //    O2[] Op2Array = Operand2.datap as O2[];

        //    npy_intp O1_CalculatedStep = (O1_Step >> O1_sizeDiv);
        //    npy_intp O1_CalculatedOffset = (O1_Offset >> O1_sizeDiv);

        //    npy_intp O2_CalculatedStep = (O2_Step >> O2_sizeDiv);
        //    npy_intp O2_CalculatedOffset = (O2_Offset >> O2_sizeDiv);

        //    npy_intp R_CalculatedStep = (R_Step >> R_sizeDiv);
        //    npy_intp R_CalculatedOffset = (R_Offset >> R_sizeDiv);

        //    bool ThrewException = false;

        //    npy_intp O1_Index = ((0 * O1_CalculatedStep) + O1_CalculatedOffset);
        //    npy_intp O2_Index = ((0 * O2_CalculatedStep) + O2_CalculatedOffset);
        //    npy_intp R_Index = ((0 * R_CalculatedStep) + R_CalculatedOffset);

        //    for (int i = 0; i < N; i++)
        //    {
        //        var O1Value = Op1Array[O1_Index];                                            // get operand 1
        //        var O2Value = Op2Array[O2_Index];                                            // get operand 2

        //        try
        //        {
        //            retry:
        //            if (ThrewException)
        //            {
        //                var RRValue = Operation(O1Value, Operand1Handler.MathOpConvertOperand(O1Value, O2Value));    // calculate result
        //                ResultHandler.SetIndex(Result, R_Index, RRValue);
        //            }
        //            else
        //            {
        //                try
        //                {
        //                    R RValue = (R)Operation(O1Value, Operand1Handler.MathOpConvertOperand(O1Value, O2Value));    // calculate result
        //                    retArray[R_Index] = RValue;
        //                }
        //                catch
        //                {
        //                    ThrewException = true;
        //                    goto retry;
        //                }
        //            }

        //        }
        //        catch (System.OverflowException oe)
        //        {
        //            NpyErr_SetString(npyexc_type.NpyExc_OverflowError, oe.Message);
        //        }
        //        catch (Exception ex)
        //        {
        //            NpyErr_SetString(npyexc_type.NpyExc_ValueError, ex.Message);
        //        }

        //        O1_Index += O1_CalculatedStep;
        //        O2_Index += O2_CalculatedStep;
        //        R_Index += R_CalculatedStep;
        //    }
        //}

        //private static void UFuncCommon_OUTER<R, O1, O2>(VoidPtr[] bufPtr, long N, long[] steps, UFuncOperation ops)
        //{
        //    VoidPtr Operand1 = bufPtr[0];
        //    VoidPtr Operand2 = bufPtr[1];
        //    VoidPtr Result = bufPtr[2];

        //    npy_intp O1_Step = steps[0];
        //    npy_intp O2_Step = steps[1];
        //    npy_intp R_Step = steps[2];

        //    if (Operand2 == null)
        //    {
        //        Operand2 = Operand1;
        //        O2_Step = O1_Step;
        //    }
        //    if (Result == null)
        //    {
        //        Result = Operand1;
        //        R_Step = O1_Step;
        //    }

        //    NumericOperation Operation = GetOperation(Operand1, ops);
        //    var Operand1Handler = DefaultArrayHandlers.GetArrayHandler(Operand1.type_num);
        //    var Operand2Handler = DefaultArrayHandlers.GetArrayHandler(Operand2.type_num);
        //    var ResultHandler = DefaultArrayHandlers.GetArrayHandler(Result.type_num);

        //    int O1_sizeDiv = Operand1Handler.ItemDiv;
        //    int O2_sizeDiv = Operand2Handler.ItemDiv;
        //    int R_sizeDiv = ResultHandler.ItemDiv;

        //    npy_intp O1_Offset = Operand1.data_offset;
        //    npy_intp O2_Offset = Operand2.data_offset;
        //    npy_intp R_Offset = Result.data_offset;


        //    R[] retArray = Result.datap as R[];
        //    O1[] Op1Array = Operand1.datap as O1[];
        //    O2[] Op2Array = Operand2.datap as O2[];

        //    npy_intp O2_CalculatedStep = (O2_Step >> O2_sizeDiv);
        //    npy_intp O2_CalculatedOffset = (O2_Offset >> O2_sizeDiv);

        //    npy_intp O1_Index = ((0 * O1_Step) + O1_Offset);
        //    npy_intp O2_Index = ((0 * O2_Step) + O2_Offset);
        //    npy_intp R_Index = ((0 * R_Step) + R_Offset);

        //    for (int i = 0; i < N; i++)
        //    {
        //        try
        //        {
        //            var O1Value = Operand1Handler.GetIndex(Operand1, O1_Index >> O1_sizeDiv);    // get operand 1
        //            var O2Value = Operand2Handler.GetIndex(Operand2, O2_Index >> O2_sizeDiv);    // get operand 2
        //            var RValue = Operation(O1Value, Operand1Handler.MathOpConvertOperand(O1Value, O2Value));    // calculate result
        //            ResultHandler.SetIndex(Result, R_Index >> R_sizeDiv, RValue);
        //        }
        //        catch (System.OverflowException oe)
        //        {
        //            NpyErr_SetString(npyexc_type.NpyExc_OverflowError, oe.Message);
        //        }
        //        catch (Exception ex)
        //        {
        //            NpyErr_SetString(npyexc_type.NpyExc_ValueError, ex.Message);
        //        }

        //        O1_Index += O1_Step;
        //        O2_Index += O2_Step;
        //        R_Index += R_Step;
        //    }
        //}

        #endregion

        internal abstract class UFUNC_BASE<T>
        {
            public UFUNC_BASE(int sizeOfItem)
            {
                this.ItemSize = sizeOfItem;
                ItemDiv = GetDivSize(sizeOfItem);
            }
            protected int ItemSize;
            protected int ItemDiv;


            #region SCALAR CALCULATIONS
            public void PerformScalarOpArrayIter(NpyArray destArray, NpyArray srcArray, NpyArray operArray, UFuncOperation op)
            {
                var destSize = NpyArray_Size(destArray);

                var SrcIter = NpyArray_BroadcastToShape(srcArray, destArray.dimensions, destArray.nd);
                var DestIter = NpyArray_BroadcastToShape(destArray, destArray.dimensions, destArray.nd);
                var OperIter = NpyArray_BroadcastToShape(operArray, destArray.dimensions, destArray.nd);

                if (!SrcIter.requiresIteration && !DestIter.requiresIteration && !operArray.IsASlice)
                {
                    PerformNumericOpScalarIterContiguousNoIter(srcArray, destArray, operArray, op, SrcIter, DestIter, OperIter);
                    return;
                }

                if (!SrcIter.requiresIteration && !DestIter.requiresIteration)
                {
                    PerformNumericOpScalarIterContiguousNoIter(srcArray, destArray, operArray, op, SrcIter, DestIter, OperIter);
                    return;
                }

                PerformNumericOpScalarSmallIter(srcArray, destArray, operArray, op, SrcIter, DestIter, OperIter, destSize);
                return;

            }

            protected void PerformNumericOpScalarSmallIter(NpyArray srcArray, NpyArray destArray, NpyArray operArray, UFuncOperation op, NpyArrayIterObject srcIter, NpyArrayIterObject destIter, NpyArrayIterObject operIter, npy_intp taskSize)
            {
                T[] src = srcArray.data.datap as T[];
                T[] dest = destArray.data.datap as T[];
                T[] oper = operArray.data.datap as T[];

                srcIter = NpyArray_ITER_ConvertToIndex(srcIter, srcArray.ItemDiv);
                destIter = NpyArray_ITER_ConvertToIndex(destIter, destArray.ItemDiv);
                operIter = NpyArray_ITER_ConvertToIndex(operIter, operArray.ItemDiv);


                var UFuncOperation = GetUFuncOperation(op);
                if (UFuncOperation == null)
                {
                    throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                }



                if (NpyArray_Size(operArray) == 1)
                {
                    if (destIter.contiguous)
                    {
                        var srcParallelIters = NpyArray_ITER_ParallelSplit(srcIter, numpyinternal.maxNumericOpParallelSize);
                        var destParallelIters = NpyArray_ITER_ParallelSplit(destIter, numpyinternal.maxNumericOpParallelSize);

                        T operValue = oper[operIter.dataptr.data_offset];

                        Parallel.For(0, destParallelIters.Count(), index =>
                        //for (int index = 0; index < destParallelIters.Count(); index++) // 
                        {
                            var ldestIter = destParallelIters.ElementAt(index);
                            var lsrcIter = srcParallelIters.ElementAt(index);

                            while (ldestIter.index < ldestIter.size)
                            {
                                npy_intp cacheSize = ldestIter.size - ldestIter.index;
                                NpyArray_ITER_CACHE(lsrcIter, cacheSize);


                                UFuncScalerIterTemplate(UFuncOperation,
                                    src, lsrcIter.internalCache, operValue,
                                    dest, ldestIter.index,
                                    lsrcIter.internalCacheLength);

                                ldestIter.index += lsrcIter.internalCacheLength;
                            }

                        } );
                    }
                    else
                    {
                        var srcParallelIters = NpyArray_ITER_ParallelSplit(srcIter, numpyinternal.maxNumericOpParallelSize);
                        var destParallelIters = NpyArray_ITER_ParallelSplit(destIter, numpyinternal.maxNumericOpParallelSize);

                        T operValue = oper[operIter.dataptr.data_offset];

                        Parallel.For(0, destParallelIters.Count(), index =>
                        //for (int index = 0; index < destParallelIters.Count(); index++) // 
                        {
                            var ldestIter = destParallelIters.ElementAt(index);
                            var lsrcIter = srcParallelIters.ElementAt(index);

                            while (ldestIter.index < ldestIter.size)
                            {
                                npy_intp cacheSize = ldestIter.size - ldestIter.index;
                                NpyArray_ITER_CACHE(ldestIter, cacheSize);
                                NpyArray_ITER_CACHE(lsrcIter, cacheSize);


                                UFuncScalerIterTemplate(UFuncOperation,
                                    src, lsrcIter.internalCache, operValue,
                                    dest, ldestIter.internalCache,
                                    ldestIter.internalCacheLength);
                            }

                        });
                    }
                }
                else
                {
                    if (destIter.contiguous)
                    {
                        var srcParallelIters = NpyArray_ITER_ParallelSplit(srcIter, numpyinternal.maxNumericOpParallelSize);
                        var destParallelIters = NpyArray_ITER_ParallelSplit(destIter, numpyinternal.maxNumericOpParallelSize);
                        var operParallelIters = NpyArray_ITER_ParallelSplit(operIter, numpyinternal.maxNumericOpParallelSize);


                        //Parallel.For(0, destParallelIters.Count(), index =>
                        for (int index = 0; index < destParallelIters.Count(); index++) // 
                        {
                            var ldestIter = destParallelIters.ElementAt(index);
                            var lsrcIter = srcParallelIters.ElementAt(index);
                            var loperIter = operParallelIters.ElementAt(index);

                            while (ldestIter.index < ldestIter.size)
                            {
                                npy_intp cacheSize = ldestIter.size - ldestIter.index;
                                NpyArray_ITER_CACHE(lsrcIter, cacheSize);
                                NpyArray_ITER_CACHE(loperIter, cacheSize);


                                UFuncScalerIterTemplate(UFuncOperation,
                                    src, lsrcIter.internalCache, oper,
                                    loperIter.internalCache,
                                    dest, ldestIter.index,
                                    lsrcIter.internalCacheLength);

                                ldestIter.index += lsrcIter.internalCacheLength;
                            }

                        } //);
                    }
                    else
                    {
                        var srcParallelIters = NpyArray_ITER_ParallelSplit(srcIter, numpyinternal.maxNumericOpParallelSize);
                        var destParallelIters = NpyArray_ITER_ParallelSplit(destIter, numpyinternal.maxNumericOpParallelSize);
                        var operParallelIters = NpyArray_ITER_ParallelSplit(operIter, numpyinternal.maxNumericOpParallelSize);


                        Parallel.For(0, destParallelIters.Count(), index =>
                        //for (int index = 0; index < destParallelIters.Count(); index++) // 
                        {
                            var ldestIter = destParallelIters.ElementAt(index);
                            var lsrcIter = srcParallelIters.ElementAt(index);
                            var loperIter = operParallelIters.ElementAt(index);

                            while (ldestIter.index < ldestIter.size)
                            {
                                npy_intp cacheSize = ldestIter.size - ldestIter.index;
                                NpyArray_ITER_CACHE(ldestIter, cacheSize);
                                NpyArray_ITER_CACHE(lsrcIter, cacheSize);
                                NpyArray_ITER_CACHE(loperIter, cacheSize);


                                UFuncScalerIterTemplate(UFuncOperation,
                                    src, lsrcIter.internalCache, oper,
                                    loperIter.internalCache,
                                    dest, ldestIter.internalCache,
                                    ldestIter.internalCacheLength);
                            }

                        });
                    }

    
                }

            }


            protected void UFuncScalerIterTemplate(opFunction UFuncOperation,
                T[] src, npy_intp[] srcOffsets,
                T[] oper, npy_intp[] operOffsets,
                T[] dest, npy_intp[] destOffsets, npy_intp offsetsLen)
            {

                if (numpyinternal.getEnableTryCatchOnCalculations)
                {
                    for (npy_intp i = 0; i < offsetsLen; i++)
                    {
                        T srcValue, operand;
                        npy_intp destIndex;

                        srcValue = src[srcOffsets[i]];
                        operand = oper[operOffsets[i]];
                        destIndex = destOffsets[i];

                        try
                        {
                            dest[destIndex] = UFuncOperation(srcValue, operand);
                        }
                        catch
                        {
                            dest[destIndex] = default(T);
                        }

                    }
                }
                else
                {
                    try
                    {
                        for (npy_intp i = 0; i < offsetsLen; i++)
                        {
                            T srcValue, operand;
                            npy_intp destIndex;

                            srcValue = src[srcOffsets[i]];
                            operand = oper[operOffsets[i]];
                            destIndex = destOffsets[i];
 
                            dest[destIndex] = UFuncOperation(srcValue, operand);
                        }
                    }
                    catch (Exception ex)
                    {
                        string Message = numpyinternal.GenerateTryCatchExceptionMessage(ex.Message);
                        throw new Exception(Message);
                    }
                }

  
            }


            protected void UFuncScalerIterTemplate(opFunction UFuncOperation,
                T[] src, npy_intp[] srcOffsets,
                T[] oper, npy_intp[] operOffsets,
                T[] dest, npy_intp destOffset, npy_intp offsetsLen)
            {
                if (numpyinternal.getEnableTryCatchOnCalculations)
                {
                    for (npy_intp i = 0; i < offsetsLen; i++)
                    {
                        T srcValue, operand;
                        npy_intp destIndex;

                        srcValue = src[srcOffsets[i]];
                        operand = oper[operOffsets[i]];
                        destIndex = destOffset + i;

                        try
                        {
                            dest[destIndex] = UFuncOperation(srcValue, operand);
                        }
                        catch
                        {
                            dest[destIndex] = default(T);
                        }
                    }

                }
                else
                {
                    try
                    {
                        for (npy_intp i = 0; i < offsetsLen; i++)
                        {
                            T srcValue, operand;
                            npy_intp destIndex;

                            srcValue = src[srcOffsets[i]];
                            operand = oper[operOffsets[i]];
                            destIndex = destOffset + i;
               
                            dest[destIndex] = UFuncOperation(srcValue, operand);
                        }
                    }
                    catch (Exception ex)
                    {
                        string Message = numpyinternal.GenerateTryCatchExceptionMessage(ex.Message);
                        throw new Exception(Message);
                    }
     
                }

     
            }



            protected void UFuncScalerIterTemplate(opFunction UFuncOperation,
                T[] src, npy_intp[] srcOffsets,
                T operValue,
                T[] dest, npy_intp[] destOffsets, npy_intp offsetsLen)
            {
                if (numpyinternal.getEnableTryCatchOnCalculations)
                {
                    for (npy_intp i = 0; i < offsetsLen; i++)
                    {
                        T srcValue;
                        npy_intp destIndex;

                        srcValue = src[srcOffsets[i]];
                        destIndex = destOffsets[i];

                        try
                        {
                            dest[destIndex] = UFuncOperation(srcValue, operValue);
                        }
                        catch
                        {
                            dest[destIndex] = default(T);
                        }
                    }
                }
                else
                {
                    try
                    {
                        for (npy_intp i = 0; i < offsetsLen; i++)
                        {
                            T srcValue;
                            npy_intp destIndex;

                            srcValue = src[srcOffsets[i]];
                            destIndex = destOffsets[i];
     
                            dest[destIndex] = UFuncOperation(srcValue, operValue);
                        }
                    }
                    catch (Exception ex)
                    {
                        string Message = numpyinternal.GenerateTryCatchExceptionMessage(ex.Message);
                        throw new Exception(Message);
                    }
                }

     
            }

            protected void UFuncScalerIterTemplate(opFunction UFuncOperation,
                T[] src, npy_intp[] srcOffsets,
                T operValue,
                T[] dest, npy_intp destOffset, npy_intp offsetsLen)
            {
                if (numpyinternal.getEnableTryCatchOnCalculations)
                {
                    for (npy_intp i = 0; i < offsetsLen; i++)
                    {
                        T srcValue;
                        npy_intp destIndex;

                        srcValue = src[srcOffsets[i]];
                        destIndex = destOffset + i;

                        try
                        {
                            dest[destIndex] = UFuncOperation(srcValue, operValue);
                        }
                        catch
                        {
                            dest[destIndex] = default(T);
                        }
                    }
                }
                else
                {
                    try
                    {
                        for (npy_intp i = 0; i < offsetsLen; i++)
                        {
                            T srcValue;
                            npy_intp destIndex;

                            srcValue = src[srcOffsets[i]];
                            destIndex = destOffset + i;

                            dest[destIndex] = UFuncOperation(srcValue, operValue);
                        }
                    }
                    catch (Exception ex)
                    {
                        string Message = numpyinternal.GenerateTryCatchExceptionMessage(ex.Message);
                        throw new Exception(Message);
                    }
                }

  
            }


            protected void PerformNumericOpScalarIterContiguousNoIter(NpyArray srcArray, NpyArray destArray, NpyArray operArray, UFuncOperation op, NpyArrayIterObject srcIter, NpyArrayIterObject destIter, NpyArrayIterObject operIter)
            {
                T[] src = srcArray.data.datap as T[];
                T[] dest = destArray.data.datap as T[];
                T[] oper = operArray.data.datap as T[];


                int srcAdjustment = (int)srcArray.data.data_offset >> srcArray.ItemDiv;
                int destAdjustment = (int)destArray.data.data_offset >> destArray.ItemDiv;

                var exceptions = new ConcurrentQueue<Exception>();

                var loopCount = NpyArray_Size(destArray);


                var UFuncOperation = GetUFuncOperation(op);
                if (UFuncOperation == null)
                {
                    throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                }

                if (NpyArray_Size(operArray) == 1 && !operArray.IsASlice)
                {
                    var ScalarIterContiguousNoIterAccelerator = GetUFuncScalarIterContiguousNoIter(op);

                    T operand = oper[0];

                    var segments = NpyArray_SEGMENT_ParallelSplit(loopCount, numpyinternal.maxNumericOpParallelSize);

                    if (numpyinternal.getEnableTryCatchOnCalculations)
                    {
                        Parallel.For(0, segments.Count(), seg_index =>
                        //for (npy_intp index = 0; index < loopCount; index++)
                        {
                            var segment = segments.ElementAt(seg_index);

                            try
                            {
                                if (ScalarIterContiguousNoIterAccelerator != null)
                                {
                                    ScalarIterContiguousNoIterAccelerator(src, dest, operand,
                                            segment.start, segment.end, srcAdjustment, destAdjustment,
                                            op);
                                }
                                else
                                {
                                    PerformNumericOpScalarIterContiguousNoIter(src, dest, operand,
                                            segment.start, segment.end, srcAdjustment, destAdjustment,
                                            UFuncOperation);
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
                        try
                        {
                            Parallel.For(0, segments.Count(), seg_index =>
                            //for (npy_intp index = 0; index < loopCount; index++)
                            {
                                var segment = segments.ElementAt(seg_index);

                                if (ScalarIterContiguousNoIterAccelerator != null)
                                {
                                    ScalarIterContiguousNoIterAccelerator(src, dest, operand,
                                            segment.start, segment.end, srcAdjustment, destAdjustment,
                                            op);
                                }
                                else
                                {
                                    PerformNumericOpScalarIterContiguousNoIter(src, dest, operand,
                                            segment.start, segment.end, srcAdjustment, destAdjustment,
                                            UFuncOperation);
                                }

                            });
                        }
                        catch (Exception ex)
                        {
                            string Message = numpyinternal.GenerateTryCatchExceptionMessage(ex.Message);
                            throw new Exception(Message);
                        }
                    }

                }
                else
                {
                    var ScalarIterContiguousIterAccelerator = GetUFuncScalarIterContiguousIter(op);


                    var ParallelIters = NpyArray_ITER_ParallelSplit(operIter, numpyinternal.maxNumericOpParallelSize);

                    if (numpyinternal.getEnableTryCatchOnCalculations)
                    {
                        Parallel.For(0, ParallelIters.Count(), index =>
                        {
                            var Iter = ParallelIters.ElementAt(index);

                            try
                            {
                                if (ScalarIterContiguousIterAccelerator != null)
                                {
                                    ScalarIterContiguousIterAccelerator(Iter, src, dest, oper, srcAdjustment, destAdjustment, op);
                                }
                                else
                                {
                                    PerformNumericOpScalarIterContiguousIter(Iter, src, dest, oper, srcAdjustment, destAdjustment, UFuncOperation);
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
                        try
                        {
                            Parallel.For(0, ParallelIters.Count(), index =>
                            {
                                var Iter = ParallelIters.ElementAt(index);

                                if (ScalarIterContiguousIterAccelerator != null)
                                {
                                    ScalarIterContiguousIterAccelerator(Iter, src, dest, oper, srcAdjustment, destAdjustment, op);
                                }
                                else
                                {
                                    PerformNumericOpScalarIterContiguousIter(Iter, src, dest, oper, srcAdjustment, destAdjustment, UFuncOperation);
                                }
                            });
                        }
                        catch (Exception ex)
                        {
                            string Message = numpyinternal.GenerateTryCatchExceptionMessage(ex.Message);
                            throw new Exception(Message);
                        }
                    }

 
                }

                if (exceptions.Count > 0)
                {
                    throw exceptions.ElementAt(0);
                }

            }

            private void PerformNumericOpScalarIterContiguousIter(NpyArrayIterObject Iter, T[] src, T[] dest, T[]oper, npy_intp srcAdjustment, npy_intp destAdjustment, opFunction UFuncOperation)
            {
                if (numpyinternal.getEnableTryCatchOnCalculations)
                {
                    while (Iter.index < Iter.size)
                    {
                        try
                        {
                            T operand = oper[Iter.dataptr.data_offset >> ItemDiv];
                            T srcValue = src[Iter.index - srcAdjustment];

                            T retValue = retValue = UFuncOperation(srcValue, operand);
                            dest[Iter.index - destAdjustment] = retValue;
                        }
                        catch (System.OverflowException of)
                        {
                            dest[Iter.index - destAdjustment] = default(T);
                        }

                        NpyArray_ITER_NEXT(Iter);
                    }
                }
                else
                {
                    try
                    {
                        while (Iter.index < Iter.size)
                        {

                            T operand = oper[Iter.dataptr.data_offset >> ItemDiv];
                            T srcValue = src[Iter.index - srcAdjustment];

                            T retValue = retValue = UFuncOperation(srcValue, operand);
                            dest[Iter.index - destAdjustment] = retValue;

                            NpyArray_ITER_NEXT(Iter);
                        }
                    }
                    catch (Exception ex)
                    {
                        string Message = numpyinternal.GenerateTryCatchExceptionMessage(ex.Message);
                        throw new Exception(Message);
                    }
                }

 
            }

            private void PerformNumericOpScalarIterContiguousNoIter(T []src, T []dest, T operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, opFunction UFuncOperation)
            {
                npy_intp srcIndex = start - srcAdjustment;
                npy_intp destIndex = start - destAdjustment;

                if (numpyinternal.getEnableTryCatchOnCalculations)
                {
                    for (npy_intp index = start; index < end; index++)
                    {
                        try
                        {
                            T srcValue = src[srcIndex];
                            dest[destIndex] = UFuncOperation(srcValue, operand);
                        }
                        catch (System.OverflowException of)
                        {
                            dest[destIndex] = default(T);
                        }

                        srcIndex++;
                        destIndex++;
                    }
                }
                else
                {
                    try
                    {
                        for (npy_intp index = start; index < end; index++)
                        {
                            T srcValue = src[srcIndex];
                            dest[destIndex] = UFuncOperation(srcValue, operand);

                            srcIndex++;
                            destIndex++;
                        }
                    }
                    catch (Exception ex)
                    {
                        string Message = numpyinternal.GenerateTryCatchExceptionMessage(ex.Message);
                        throw new Exception(Message);
                    }
                }

            }

            #endregion
            // kevin
            #region UFUNC Outer
            public void PerformOuterOpArrayIter(NpyArray a, NpyArray b, NpyArray destArray, NumericOperations operations, UFuncOperation op)
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

                T[] aValues = new T[aSize];
                for (long i = 0; i < aSize; i++)
                {
                    aValues[i] = ConvertToTemplate(operations.srcGetItem(aIter.dataptr.data_offset));
                    NpyArray_ITER_NEXT(aIter);
                }

                T[] bValues = new T[bSize];
                for (long i = 0; i < bSize; i++)
                {
                    bValues[i] = ConvertToTemplate(operations.operandGetItem(bIter.dataptr.data_offset));
                    NpyArray_ITER_NEXT(bIter);
                }


                T[] dp = destArray.data.datap as T[];

                var UFuncOperation = GetUFuncOperation(op);
                if (UFuncOperation == null)
                {
                    throw new Exception(string.Format("UFunc op:{0} is not implemented", op.ToString()));
                }
                
                if (DestIter.contiguous && destSize > UFUNC_PARALLEL_DEST_MINSIZE && aSize > UFUNC_PARALLEL_DEST_ASIZE)
                {
                    List<Exception> caughtExceptions = new List<Exception>();

                    var UFuncOuterContigAccelerator = GetUFuncOuterContigOperation(op);

                    Parallel.For(0, aSize, i =>
                    {
                        try
                        {
                            var aValue = aValues[i];

                            npy_intp destIndex = (destArray.data.data_offset >> destArray.ItemDiv) + i * bSize;

                            if (UFuncOuterContigAccelerator != null)
                            {
                                UFuncOuterContigAccelerator(operations, aValue, bValues, bSize, dp, destIndex, destArray, op);
                            }
                            else
                            {
                                PerformOuterOp(operations, aValue, bValues, bSize, dp, destIndex, destArray, UFuncOperation);
                            }
                        }
                        catch (Exception ex)
                        {
                            caughtExceptions.Add(ex);
                        }


                    });

                    if (caughtExceptions.Count > 0)
                    {
                        Exception ex = caughtExceptions[0];
                        if (ex is System.OverflowException)
                        {
                            NpyErr_SetString(npyexc_type.NpyExc_OverflowError, ex.Message);
                            return;
                        }

                        NpyErr_SetString(npyexc_type.NpyExc_ValueError, ex.Message);
                        return;
                    }
                }
                else
                {
                    try
                    {
                        var UFuncOuterIterAccelerator = GetUFuncOuterIterOperation(op);

                        for (long i = 0; i < aSize; i++)
                        {
                            var aValue = aValues[i];

                            if (UFuncOuterIterAccelerator != null)
                            {
                                UFuncOuterIterAccelerator(operations, aValue, bValues, bSize, dp, DestIter, destArray, op);
                            }
                            else
                            {
                                PerformOuterOp(operations, aValue, bValues, bSize, dp, DestIter, destArray, UFuncOperation);
                            }
                        }
                    }
                    catch (System.OverflowException ex)
                    {
                        NpyErr_SetString(npyexc_type.NpyExc_OverflowError, ex.Message);
                        return;
                    }
                    catch (Exception ex)
                    {
                        NpyErr_SetString(npyexc_type.NpyExc_ValueError, ex.Message);
                        return;
                    }

                }


            }

            void PerformOuterOp(NumericOperations operations, T aValue, T[] bValues, npy_intp bSize, T []dp, npy_intp destIndex, NpyArray destArray, opFunction UFuncOperation)
            {
                if (numpyinternal.getEnableTryCatchOnCalculations)
                {
                    for (npy_intp j = 0; j < bSize; j++)
                    {
                        var bValue = bValues[j];

                        try
                        {
                            dp[destIndex] = UFuncOperation(aValue, bValue);
                        }
                        catch
                        {
                            operations.destSetItem(destIndex, 0);
                        }
                        destIndex++;
                    }
                }
                else
                {
                    try
                    {
                        for (npy_intp j = 0; j < bSize; j++)
                        {
                            var bValue = bValues[j];
                            dp[destIndex] = UFuncOperation(aValue, bValue);
                            destIndex++;
                        }
                    }
                    catch (Exception ex)
                    {
                        string Message = numpyinternal.GenerateTryCatchExceptionMessage(ex.Message);
                        throw new Exception(Message);
                    }
                }

   
            }

            void PerformOuterOp(NumericOperations operations, T aValue, T[] bValues, npy_intp bSize, T[] dp, NpyArrayIterObject DestIter, NpyArray destArray, opFunction UFuncOperation)
            {
                if (numpyinternal.getEnableTryCatchOnCalculations)
                {
                    for (npy_intp j = 0; j < bSize; j++)
                    {
                        var bValue = bValues[j];

                        T destValue = UFuncOperation(aValue, bValue);
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
                else
                {
                    try
                    {
                        for (npy_intp j = 0; j < bSize; j++)
                        {
                            var bValue = bValues[j];

                            T destValue = UFuncOperation(aValue, bValue);
                            npy_intp AdjustedIndex = DestIter.dataptr.data_offset >> ItemDiv;
                            dp[AdjustedIndex] = destValue;
                 
                            NpyArray_ITER_NEXT(DestIter);
                        }
                    }
                    catch (Exception ex)
                    {
                        string Message = numpyinternal.GenerateTryCatchExceptionMessage(ex.Message);
                        throw new Exception(Message);
                    }
                }

  
            }

            #endregion

            #region UFUNC Reduce

            public void PerformReduceOpArrayIter(VoidPtr[] bufPtr, npy_intp[] steps, UFuncOperation ops, npy_intp N)
            {
                VoidPtr Operand1 = bufPtr[0];
                VoidPtr Operand2 = bufPtr[1];
                VoidPtr Result = bufPtr[2];

                npy_intp O1_Step = steps[0];
                npy_intp O2_Step = steps[1];
                npy_intp R_Step = steps[2];

                npy_intp O1_Offset = Operand1.data_offset;
                npy_intp O2_Offset = Operand2.data_offset;
                npy_intp R_Offset = Result.data_offset;


                T[] retArray = Result.datap as T[];
                T[] Op1Array = Operand1.datap as T[];
                T[] Op2Array = Operand2.datap as T[];

                npy_intp R_Index = AdjustNegativeIndex(retArray, R_Offset >> ItemDiv);
                npy_intp O1_Index = AdjustNegativeIndex(Op1Array, O1_Offset >> ItemDiv);

                npy_intp O2_CalculatedStep = (O2_Step >> ItemDiv);
                npy_intp O2_CalculatedOffset = (O2_Offset >> ItemDiv);


                T retValue = retArray[R_Index];
                npy_intp O2_Index = ((0 * O2_CalculatedStep) + O2_CalculatedOffset);

                var UFuncReduceOperation = GetUFuncReduceOperation(ops);
                if (UFuncReduceOperation != null)
                {
                    retArray[R_Index] = UFuncReduceOperation(retValue, Op2Array, O2_Index, O2_CalculatedStep, N);
                    return;
                }


                var UFuncOperation = GetUFuncOperation(ops);
                if (UFuncOperation == null)
                {
                    throw new Exception(string.Format("UFunc op:{0} is not implemented", ops.ToString()));
                }

                // note: these can't be parallelized.
                for (npy_intp i = 0; i < N; i++)
                {
                    var Op1Value = retValue;
                    var Op2Value = Op2Array[O2_Index];

                    retValue = UFuncOperation(Op1Value, Op2Value);

                    O2_Index += O2_CalculatedStep;
                }

                retArray[R_Index] = retValue;
                return;
            }

            #endregion

            #region UFUNC Accumulate

            public void PerformAccumulateOpArrayIter(VoidPtr[] bufPtr, npy_intp[] steps, UFuncOperation ops, npy_intp N)
            {
                VoidPtr Operand1 = bufPtr[0];
                VoidPtr Operand2 = bufPtr[1];
                VoidPtr Result = bufPtr[2];

                npy_intp O1_Step = steps[0];
                npy_intp O2_Step = steps[1];
                npy_intp R_Step = steps[2];

                if (Operand2 == null)
                {
                    Operand2 = Operand1;
                    O2_Step = O1_Step;
                }
                if (Result == null)
                {
                    Result = Operand1;
                    R_Step = O1_Step;
                }

                npy_intp O1_Offset = Operand1.data_offset;
                npy_intp O2_Offset = Operand2.data_offset;
                npy_intp R_Offset = Result.data_offset;


                T[] retArray = Result.datap as T[];
                T[] Op1Array = Operand1.datap as T[];
                T[] Op2Array = Operand2.datap as T[];

                npy_intp O1_CalculatedStep = (O1_Step >> ItemDiv);
                npy_intp O1_CalculatedOffset = (O1_Offset >> ItemDiv);

                npy_intp O2_CalculatedStep = (O2_Step >> ItemDiv);
                npy_intp O2_CalculatedOffset = (O2_Offset >> ItemDiv);

                npy_intp R_CalculatedStep = (R_Step >> ItemDiv);
                npy_intp R_CalculatedOffset = (R_Offset >> ItemDiv);

                npy_intp O1_Index = ((0 * O1_CalculatedStep) + O1_CalculatedOffset);
                npy_intp O2_Index = ((0 * O2_CalculatedStep) + O2_CalculatedOffset);
                npy_intp R_Index = ((0 * R_CalculatedStep) + R_CalculatedOffset);

                var UFuncAccumulateOperation = GetUFuncAccumulateOperation(ops);
                if (UFuncAccumulateOperation != null)
                {
                    UFuncAccumulateOperation(
                        Op1Array, O1_Index, O1_CalculatedStep,
                        Op2Array, O2_Index, O2_CalculatedStep,
                        retArray, R_Index, R_CalculatedStep,N);
                    return;
                }

                var UFuncOperation = GetUFuncOperation(ops);
                if (UFuncOperation == null)
                {
                    throw new Exception(string.Format("UFunc op:{0} is not implemented", ops.ToString()));
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

            }
 

            #endregion

            #region REDUCEAT

            public void PerformReduceAtOpArrayIter(VoidPtr[] bufPtr, npy_intp[] steps, UFuncOperation ops, npy_intp N)
            {
                PerformAccumulateOpArrayIter(bufPtr, steps, ops, N);
            }

            #endregion

            protected npy_intp AdjustNegativeIndex(T[] data, npy_intp index)
            {
                if (index < 0)
                {
                    index = data.Length - -index;
                }
                return index;
            }

            protected opFunction GetUFuncOperation(UFuncOperation ops)
            {
                switch (ops)
                {
                    case UFuncOperation.add:
                        return Add;

                    case UFuncOperation.subtract:
                        return Subtract;

                    case UFuncOperation.multiply:
                        return Multiply;

                    case UFuncOperation.divide:
                        return Divide;

                    case UFuncOperation.remainder:
                        return Remainder;

                    case UFuncOperation.fmod:
                        return FMod;

                    case UFuncOperation.power:
                        return Power;

                    case UFuncOperation.square:
                        return Square;

                    case UFuncOperation.reciprocal:
                        return Reciprocal;

                    case UFuncOperation.ones_like:
                        return OnesLike;

                    case UFuncOperation.sqrt:
                        return Sqrt;

                    case UFuncOperation.negative:
                        return Negative;

                    case UFuncOperation.absolute:
                        return Absolute;

                    case UFuncOperation.invert:
                        return Invert;

                    case UFuncOperation.left_shift:
                        return LeftShift;

                    case UFuncOperation.right_shift:
                        return RightShift;

                    case UFuncOperation.bitwise_and:
                        return BitWiseAnd;

                    case UFuncOperation.bitwise_xor:
                        return BitWiseXor;

                    case UFuncOperation.bitwise_or:
                        return BitWiseOr;

                    case UFuncOperation.less:
                        return Less;

                    case UFuncOperation.less_equal:
                        return LessEqual;

                    case UFuncOperation.equal:
                        return Equal;

                    case UFuncOperation.not_equal:
                        return NotEqual;

                    case UFuncOperation.greater:
                        return Greater;

                    case UFuncOperation.greater_equal:
                        return GreaterEqual;

                    case UFuncOperation.floor_divide:
                        return FloorDivide;

                    case UFuncOperation.true_divide:
                        return TrueDivide;

                    case UFuncOperation.logical_or:
                        return LogicalOr;

                    case UFuncOperation.logical_and:
                        return LogicalAnd;

                    case UFuncOperation.floor:
                        return Floor;

                    case UFuncOperation.ceil:
                        return Ceiling;

                    case UFuncOperation.maximum:
                        return Maximum;

                    case UFuncOperation.minimum:
                        return Minimum;

                    case UFuncOperation.rint:
                        return Rint;

                    case UFuncOperation.conjugate:
                        return Conjugate;

                    case UFuncOperation.isnan:
                        return IsNAN;

                    case UFuncOperation.fmax:
                        return FMax;

                    case UFuncOperation.fmin:
                        return FMin;

                    case UFuncOperation.heaviside:
                        return Heaviside;

                }

                return null;
            }

            protected opFunctionReduce GetUFuncReduceOperation(UFuncOperation ops)
            {
                // each individual data type can support accelerator functions if
                // it chooses.  This call will return a delegate if the operation
                // is supported, else null.
                return GetUFuncReduceHandler(ops);
            }

            protected opFunctionAccumulate GetUFuncAccumulateOperation(UFuncOperation ops)
            {
                // each individual data type can support accelerator functions if
                // it chooses.  This call will return a delegate if the operation
                // is supported, else null.
                return GetUFuncAccumulateHandler(ops);
            }

            protected opFunctionScalerIter GetUFuncScalarIterOperation(UFuncOperation ops)
            {
                // each individual data type can support accelerator functions if
                // it chooses.  This call will return a delegate if the operation
                // is supported, else null.
                return GetUFuncScalarIterHandler(ops);
            }

            protected opFunctionOuterOpContig GetUFuncOuterContigOperation(UFuncOperation ops)
            {
                // each individual data type can support accelerator functions if
                // it chooses.  This call will return a delegate if the operation
                // is supported, else null.

                return GetUFuncOuterContigHandler(ops);
            }
            protected opFunctionOuterOpIter GetUFuncOuterIterOperation(UFuncOperation ops)
            {
                // each individual data type can support accelerator functions if
                // it chooses.  This call will return a delegate if the operation
                // is supported, else null.

                return GetUFuncOuterIterHandler(ops);
            }
            protected opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIter(UFuncOperation ops)
            {
                // each individual data type can support accelerator functions if
                // it chooses.  This call will return a delegate if the operation
                // is supported, else null.

                return GetUFuncScalarIterContiguousNoIterHandler(ops);
            }
            protected opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIter(UFuncOperation ops)
            {
                // each individual data type can support accelerator functions if
                // it chooses.  This call will return a delegate if the operation
                // is supported, else null.

                return GetUFuncScalarIterContiguousIterHandler(ops);
            }

            protected delegate T opFunction(T o1, T o2);
            protected delegate T opFunctionReduce(T Op1Value, T[] Op2Values, npy_intp O2_Index, npy_intp O2_Step, npy_intp N);
            protected delegate void opFunctionAccumulate(T[] Op1Array, npy_intp O1_Index, npy_intp O1_Step,
                                                         T[] Op2Array, npy_intp O2_Index, npy_intp O2_Step,
                                                         T[] retArray, npy_intp R_Index, npy_intp R_Step, npy_intp N);
            protected delegate void opFunctionScalerIter(T[] src, npy_intp[] srcOffsets,
                                                         T[] oper, npy_intp[] operOffsets,
                                                         T[] dest, npy_intp[] destOffsets, 
                                                         npy_intp OffetLength, UFuncOperation ops);
            protected delegate void opFunctionOuterOpContig(NumericOperations operations, T aValue, T[] bValues, npy_intp bSize, T[] dp, npy_intp destIndex, NpyArray destArray, UFuncOperation ops);
            protected delegate void opFunctionOuterOpIter(NumericOperations operations, T aValue, T[] bValues, npy_intp bSize, T[] dp, NpyArrayIterObject DestIter, NpyArray destArray, UFuncOperation ops);
            protected delegate void opFunctionScalarIterContiguousNoIter(T[] src, T[] dest, T operand, npy_intp start, npy_intp end, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops);
            protected delegate void opFunctionScalarIterContiguousIter(NpyArrayIterObject Iter, T[] src, T[] dest, T[] oper, npy_intp srcAdjustment, npy_intp destAdjustment, UFuncOperation ops);

            protected abstract opFunctionReduce GetUFuncReduceHandler(UFuncOperation ops);
            protected abstract opFunctionAccumulate GetUFuncAccumulateHandler(UFuncOperation ops);
            protected abstract opFunctionScalerIter GetUFuncScalarIterHandler(UFuncOperation ops);
            protected abstract opFunctionOuterOpContig GetUFuncOuterContigHandler(UFuncOperation ops);
            protected abstract opFunctionOuterOpIter GetUFuncOuterIterHandler(UFuncOperation ops);
            protected abstract opFunctionScalarIterContiguousNoIter GetUFuncScalarIterContiguousNoIterHandler(UFuncOperation ops);
            protected abstract opFunctionScalarIterContiguousIter GetUFuncScalarIterContiguousIterHandler(UFuncOperation ops);


            protected abstract T Add(T o1, T o2);
            protected abstract T Subtract(T o1, T o2);
            protected abstract T Multiply(T o1, T o2);
            protected abstract T Divide(T o1, T o2);
            protected abstract T Power(T o1, T o2);
            protected abstract T Remainder(T o1, T o2);
            protected abstract T FMod(T o1, T o2);
            protected abstract T Square(T o1, T o2);
            protected abstract T Reciprocal(T o1, T o2);
            protected abstract T OnesLike(T o1, T o2);
            protected abstract T Sqrt(T o1, T o2);
            protected abstract T Negative(T o1, T o2);
            protected abstract T Absolute(T o1, T o2);
            protected abstract T Invert(T o1, T o2);
            protected abstract T LeftShift(T o1, T o2);
            protected abstract T RightShift(T o1, T o2);
            protected abstract T BitWiseAnd(T o1, T o2);
            protected abstract T BitWiseXor(T o1, T o2);
            protected abstract T BitWiseOr(T o1, T o2);
            protected abstract T Less(T o1, T o2);
            protected abstract T LessEqual(T o1, T o2);
            protected abstract T Equal(T o1, T o2);
            protected abstract T NotEqual(T o1, T o2);
            protected abstract T Greater(T o1, T o2);
            protected abstract T GreaterEqual(T o1, T o2);
            protected abstract T FloorDivide(T o1, T o2);
            protected abstract T TrueDivide(T o1, T o2);
            protected abstract T LogicalOr(T o1, T o2);
            protected abstract T LogicalAnd(T o1, T o2);
            protected abstract T Floor(T o1, T o2);
            protected abstract T Ceiling(T o1, T o2);
            protected abstract T Maximum(T o1, T o2);
            protected abstract T Minimum(T o1, T o2);
            protected abstract T Rint(T o1, T o2);
            protected abstract T Conjugate(T o1, T o2);
            protected abstract T IsNAN(T o1, T o2);
            protected abstract T FMax(T o1, T o2);
            protected abstract T FMin(T o1, T o2);
            protected abstract T Heaviside(T o1, T o2);

            protected abstract T PerformUFuncOperation(UFuncOperation op, T o1, T o2);

            protected abstract T ConvertToTemplate(object v);


        }




    }
}
