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
        internal static NpyUFuncObject NpyArray_GetNumericOp(NpyArray_Ops op)
        {
            NpyUFuncObject loc = get_op_loc(op);
            return (null != loc) ? loc : null;
        }

        internal static NpyUFuncObject NpyArray_SetNumericOp(NpyArray_Ops op, NpyUFuncObject func)
        {
            throw new NotImplementedException();
        }

        internal static NpyArray NpyArray_GenericBinaryFunction(NpyArray m1, NpyArray m2, NpyUFuncObject op, NpyArray outArray)
        {
            NpyArray[] mps = new NpyArray[npy_defs.NPY_MAXARGS];
            NpyArray result;
            int i;

            Debug.Assert(Validate(op));
            Debug.Assert(Validate(m1));
            Debug.Assert(Validate(m2));
            if (outArray != null)
                Debug.Assert(Validate(outArray));
 

            mps[0] = m1;
            mps[1] = m2;
            Npy_XINCREF(outArray);
            mps[2] = outArray;

            for (i = 0; i < 3; i++)
            {
                Npy_XINCREF(mps[i]);
            }
            if (0 > NpyUFunc_GenericFunction(op, 3, mps, 0, null, false, null, null))
            {
                result = null;
                goto finish;
            }

            if (outArray != null) {
                result = outArray;
            } else {
                result = mps[2];
            }

            finish:
            Npy_XINCREF(result);
            for (i = 0; i < 3; i++)
            {
                if (mps[i] != null)
                {
                    if ((mps[i].flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) != 0)
                    {
                        NpyArray_ForceUpdate(mps[i]);
                    }
                    Npy_DECREF(mps[i]);
                }
            }
            return result;
        }

        internal static NpyArray NpyArray_GenericUnaryFunction(NpyArray inputArray, NpyUFuncObject op, NpyArray retArray)
        {
            NpyArray[] mps = new NpyArray[npy_defs.NPY_MAXARGS];
            NpyArray result;

            if (retArray == null)
                retArray = inputArray;

            Debug.Assert(Validate(op));
            Debug.Assert(Validate(inputArray));
            Debug.Assert(Validate(retArray));

            mps[0] = inputArray;
            Npy_XINCREF(retArray);
            mps[1] = retArray;
            if (0 > NpyUFunc_GenericFunction(op, 2, mps, 0, null, false, null, null))
            {
                result = null;
                goto finish;
            }

            if (retArray != null)
            {
                result = retArray;
            }
            else
            {
                result = mps[op.nin];
            }
            finish:
            Npy_XINCREF(result);
            if (mps[1] != null && (mps[1].flags & NPYARRAYFLAGS.NPY_UPDATEIFCOPY) != 0)
            {
                NpyArray_ForceUpdate(mps[1]);
            }
            Npy_XDECREF(mps[1]);
            return result;
        }

        internal static int NpyArray_Bool(NpyArray mp)
        {
            npy_intp n;

            n = NpyArray_SIZE(mp);
            if (n == 1)
            {
                var vp = NpyArray_BYTES(mp);
                return NpyArray_DESCR(mp).f.nonzero(vp, vp.data_offset/ mp.ItemSize) ? 1 : 0;
            }
            else if (n == 0)
            {
                return 0;
            }
            else
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError,
                                "The truth value of an array with more than one element is ambiguous. Use np.any() or np.all()");
                return -1;
            }
        }

        static NpyUFuncObject get_op_loc(NpyArray_Ops op)
        {
            NpyUFuncObject loc = null;

            switch (op)
            {
                case NpyArray_Ops.npy_op_add:
                    return DefaultOpControl(op, UFuncAdd);
                case NpyArray_Ops.npy_op_maximum:
                    return DefaultOpControl(op, UFuncMax);
                case NpyArray_Ops.npy_op_minimum:
                    return DefaultOpControl(op, UFuncMin);
                case NpyArray_Ops.npy_op_multiply:
                    return DefaultOpControl(op, UFuncMultiply);
                case NpyArray_Ops.npy_op_logical_or:
                    return DefaultOpControl(op, UFuncLogicalOr);
                case NpyArray_Ops.npy_op_logical_and:
                    return DefaultOpControl(op, UFuncLogicalAnd);
                case NpyArray_Ops.npy_op_subtract:
                    return DefaultOpControl(op, UFuncSubtract);
                case NpyArray_Ops.npy_op_divide:
                    return DefaultOpControl(op, UFuncDivide);
                case NpyArray_Ops.npy_op_remainder:
                    return DefaultOpControl(op, UFuncRemainder);
                case NpyArray_Ops.npy_op_fmod:
                    return DefaultOpControl(op, UFuncFMod);
                case NpyArray_Ops.npy_op_power:
                    return DefaultOpControl(op, UFuncPower);
                case NpyArray_Ops.npy_op_square:
                    return DefaultOpControl(op, UFuncSquare);
                case NpyArray_Ops.npy_op_reciprocal:
                    return DefaultOpControl(op, UFuncReciprocal);
                case NpyArray_Ops.npy_op_ones_like:
                    return DefaultOpControl(op, UFuncOnesLike);
                case NpyArray_Ops.npy_op_sqrt:
                    return DefaultOpControl(op, UFuncSqrt);
                case NpyArray_Ops.npy_op_negative:
                    return DefaultOpControl(op, UFuncNegative);
                case NpyArray_Ops.npy_op_absolute:
                    return DefaultOpControl(op, UFuncAbsolute);
                case NpyArray_Ops.npy_op_invert:
                    return DefaultOpControl(op, UFuncInvert);
                case NpyArray_Ops.npy_op_left_shift:
                    return DefaultOpControl(op, UFuncLeftShift);
                case NpyArray_Ops.npy_op_right_shift:
                    return DefaultOpControl(op, UFuncRightShift);
                case NpyArray_Ops.npy_op_bitwise_and:
                    return DefaultOpControl(op, UFuncBitWiseAnd);
                case NpyArray_Ops.npy_op_bitwise_xor:
                    return DefaultOpControl(op, UFuncBitWiseXor);
                case NpyArray_Ops.npy_op_bitwise_or:
                    return DefaultOpControl(op, UFuncBitWiseOr);
                case NpyArray_Ops.npy_op_less:
                    return DefaultOpControl(op, UFuncLess);
                case NpyArray_Ops.npy_op_less_equal:
                    return DefaultOpControl(op, UFuncLessEqual);
                case NpyArray_Ops.npy_op_equal:
                    return DefaultOpControl(op, UFuncEqual);
                case NpyArray_Ops.npy_op_not_equal:
                    return DefaultOpControl(op, UFuncNotEqual);
                case NpyArray_Ops.npy_op_greater:
                    return DefaultOpControl(op, UFuncGreater);
                case NpyArray_Ops.npy_op_greater_equal:
                    return DefaultOpControl(op, UFuncGreaterEqual);
                case NpyArray_Ops.npy_op_floor_divide:
                    return DefaultOpControl(op, UFuncFloorDivide);
                case NpyArray_Ops.npy_op_true_divide:
                    return DefaultOpControl(op, UFuncTrueDivide);
                case NpyArray_Ops.npy_op_floor:
                    return DefaultOpControl(op, UFuncFloor);
                case NpyArray_Ops.npy_op_ceil:
                    return DefaultOpControl(op, UFuncCeil);
                case NpyArray_Ops.npy_op_rint:
                    return DefaultOpControl(op, UFuncRint);
                case NpyArray_Ops.npy_op_conjugate:
                    return DefaultOpControl(op, UFuncConjugate);
                default:
                    loc = null;
                    break;
            }
            return loc;
        }

        private static NpyUFuncObject DefaultOpControl(NpyArray_Ops ops, NpyUFuncGenericFunction UFunc)
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

        internal static void UFuncCommon(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData, NumericOperation Operation)
        {
            VoidPtr Result = bufPtr[2];

            switch (Result.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    UFuncCommon_R<bool>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    UFuncCommon_R<sbyte>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    UFuncCommon_R<byte>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_INT16:
                    UFuncCommon_R<Int16>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    UFuncCommon_R<UInt16>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_INT32:
                    UFuncCommon_R<Int32>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    UFuncCommon_R<UInt32>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_INT64:
                    UFuncCommon_R<Int64>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    UFuncCommon_R<UInt64>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    UFuncCommon_R<float>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    UFuncCommon_R<double>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    UFuncCommon_R<decimal>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    UFuncCommon_R<System.Numerics.Complex>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    UFuncCommon_R<System.Numerics.BigInteger>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    UFuncCommon_R<object>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_STRING:
                    UFuncCommon_R<string>(bufPtr, N, steps, Operation);
                    break;
            }
   
            return;
        }

        private static void UFuncCommon_R<R>(VoidPtr[] bufPtr, npy_intp N, npy_intp[] steps, NumericOperation Operation)
        {
            VoidPtr Result = bufPtr[0];

            switch (Result.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    UFuncCommon_RO<R,bool>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    UFuncCommon_RO<R, sbyte>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    UFuncCommon_RO<R, byte>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_INT16:
                    UFuncCommon_RO<R, Int16>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    UFuncCommon_RO<R, UInt16>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_INT32:
                    UFuncCommon_RO<R, Int32>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    UFuncCommon_RO<R, UInt32>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_INT64:
                    UFuncCommon_RO<R, Int64>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    UFuncCommon_RO<R, UInt64>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    UFuncCommon_RO<R, float>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    UFuncCommon_RO<R, double>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    UFuncCommon_RO<R, decimal>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    UFuncCommon_RO<R, System.Numerics.Complex>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    UFuncCommon_RO<R, System.Numerics.BigInteger>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    UFuncCommon_RO<R, object>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_STRING:
                    UFuncCommon_RO<R, string>(bufPtr, N, steps, Operation);
                    break;
            }

            return;
        }

        private static void UFuncCommon_RO<R,O1>(VoidPtr[] bufPtr, npy_intp N, npy_intp[] steps, NumericOperation Operation)
        {
            VoidPtr Result = bufPtr[1];

            switch (Result.type_num)
            {
                case NPY_TYPES.NPY_BOOL:
                    UFuncCommon_ROO<R,O1,bool>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_BYTE:
                    UFuncCommon_ROO<R, O1, sbyte>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    UFuncCommon_ROO<R, O1, byte>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_INT16:
                    UFuncCommon_ROO<R, O1, Int16>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_UINT16:
                    UFuncCommon_ROO<R, O1, UInt16>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_INT32:
                    UFuncCommon_ROO<R, O1, Int32>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_UINT32:
                    UFuncCommon_ROO<R, O1, UInt32>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_INT64:
                    UFuncCommon_ROO<R, O1, Int64>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_UINT64:
                    UFuncCommon_ROO<R, O1, UInt64>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    UFuncCommon_ROO<R, O1, float>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    UFuncCommon_ROO<R, O1, double>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    UFuncCommon_ROO<R, O1, decimal>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    UFuncCommon_ROO<R, O1, System.Numerics.Complex>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_BIGINT:
                    UFuncCommon_ROO<R, O1, System.Numerics.BigInteger>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_OBJECT:
                    UFuncCommon_ROO<R, O1, object>(bufPtr, N, steps, Operation);
                    break;
                case NPY_TYPES.NPY_STRING:
                    UFuncCommon_ROO<R, O1, string>(bufPtr, N, steps, Operation);
                    break;
            }
        }

        private static void UFuncCommon_ROO<R, O1, O2>(VoidPtr[] bufPtr, npy_intp N, npy_intp[] steps, NumericOperation Operation)
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

            var Operand1Handler = DefaultArrayHandlers.GetArrayHandler(Operand1.type_num);
            var Operand2Handler = DefaultArrayHandlers.GetArrayHandler(Operand2.type_num);
            var ResultHandler = DefaultArrayHandlers.GetArrayHandler(Result.type_num);

            npy_intp O1_sizeData = Operand1Handler.ItemSize;
            npy_intp O2_sizeData = Operand2Handler.ItemSize;
            npy_intp R_sizeData = ResultHandler.ItemSize;

            npy_intp O1_Offset = Operand1.data_offset;
            npy_intp O2_Offset = Operand2.data_offset;
            npy_intp R_Offset = Result.data_offset;
                   

            if (R_Step == 0 && O1_Step == 0)
            {

                npy_intp R_Index = AdjustNegativeIndex<R>(Result, R_Offset / R_sizeData);
                npy_intp O1_Index = AdjustNegativeIndex<O1>(Operand1, O1_Offset / O1_sizeData);

                R[] retArray = Result.datap as R[];
                O1[] Op1Array = Operand1.datap as O1[];
                O2[] Op2Array = Operand2.datap as O2[];

                npy_intp O2_CalculatedStep = (O2_Step / O2_sizeData);
                npy_intp O2_CalculatedOffset = (O2_Offset / O2_sizeData);

                bool ThrewException = false;
                

                for (int i = 0; i < N; i++)
                {

                    try
                    {
                        var O1Value = Op1Array[O1_Index];

                        npy_intp O2_Index = ((i * O2_CalculatedStep) + O2_CalculatedOffset);
                        var O2Value = Op2Array[O2_Index];                                            // get operand 2

                    retry:                   
                        if (ThrewException)
                        {
                            var RRValue = Operation(O1Value, Operand1Handler.MathOpConvertOperand(O1Value, O2Value));    // calculate result
                            ResultHandler.SetIndex(Result, R_Index, RRValue);
                        }
                        else
                        {
                            try
                            {
                                R RValue = (R)Operation(O1Value, O2Value);    // calculate result
                                retArray[R_Index] = RValue;
                            }
                            catch
                            {
                                ThrewException = true;
                                goto retry;
                            }

                        }
                    
                    }
                    catch (System.OverflowException oe)
                    {
                        NpyErr_SetString(npyexc_type.NpyExc_OverflowError, oe.Message);
                    }
                    catch (Exception ex)
                    {
                        NpyErr_SetString(npyexc_type.NpyExc_ValueError, ex.Message);
                    }
                }

                return;

            }

            // todo: try idea of allocation an array objects, do the set and convert when all done.
            // todo: try next level down of optimization like above.

            for (int i = 0; i < N; i++)
            {
                npy_intp O1_Index = ((i * O1_Step) + O1_Offset) / O1_sizeData;
                npy_intp O2_Index = ((i * O2_Step) + O2_Offset) / O2_sizeData;
                npy_intp R_Index = ((i * R_Step) + R_Offset) / R_sizeData;

                try
                {
                    var O1Value = Operand1Handler.GetIndex(Operand1, O1_Index);                  // get operand 1
                    var O2Value = Operand2Handler.GetIndex(Operand2, O2_Index);                  // get operand 2
                    var RValue = Operation(O1Value, Operand1Handler.MathOpConvertOperand(O1Value, O2Value));    // calculate result
                    ResultHandler.SetIndex(Result, R_Index, RValue);
                }
                catch (System.OverflowException oe)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_OverflowError, oe.Message);
                }
                catch (Exception ex)
                {
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, ex.Message);
                }
            }

            return;
        }

        private static npy_intp AdjustNegativeIndex<T>(VoidPtr data, npy_intp index)
        {
            if (index < 0)
            {
                T[] dp = data.datap as T[];
                index = dp.Length - Math.Abs(index);
            }
            return index;
        }


        internal static void UFuncAdd(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_add));
            return;
        }

        internal static void UFuncMultiply(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_multiply));
            return;
        }

        internal static void UFuncMax(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_maximum));
            return;
        }
 
 
        internal static void UFuncMin(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_minimum));
            return;
        }
  
 
        internal static void UFuncLogicalOr(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_logical_or));
            return;
        }

        internal static void UFuncLogicalAnd(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_logical_and));
            return;
        }

        internal static void UFuncSubtract(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_subtract));
            return;
        }

        internal static void UFuncDivide(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_divide));
            return;
        }

        internal static void UFuncRemainder(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_remainder));
            return;
        }

        internal static void UFuncFMod(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_fmod));
            return;
        }

        internal static void UFuncPower(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_power));
            return;
        }

        internal static void UFuncSquare(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_square));
            return;
        }

        internal static void UFuncReciprocal(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_reciprocal));
            return;
        }

        internal static void UFuncOnesLike(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_ones_like));
            return;
        }

        internal static void UFuncSqrt(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_sqrt));
            return;
        }

        internal static void UFuncNegative(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_negative));
            return;
        }

        internal static void UFuncAbsolute(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_absolute));
            return;
        }

        internal static void UFuncInvert(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_invert));
            return;
        }

        internal static void UFuncLeftShift(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_left_shift));
            return;
        }

        internal static void UFuncRightShift(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_right_shift));
            return;
        }

        internal static void UFuncBitWiseAnd(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_bitwise_and));
            return;
        }
        internal static void UFuncBitWiseXor(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_bitwise_xor));
            return;
        }
        internal static void UFuncBitWiseOr(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0],NpyArray_Ops.npy_op_bitwise_or));
            return;
        }

        internal static void UFuncLess(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_less));
            return;
        }
        internal static void UFuncLessEqual(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_less_equal));
            return;
        }
        internal static void UFuncEqual(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_equal));
            return;
        }
        internal static void UFuncNotEqual(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_not_equal));
            return;
        }
        internal static void UFuncGreater(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_greater));
            return;
        }
        internal static void UFuncGreaterEqual(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_greater_equal));
            return;
        }
        internal static void UFuncFloorDivide(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_floor_divide));
            return;
        }
        internal static void UFuncTrueDivide(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_true_divide));
            return;
        }
        internal static void UFuncFloor(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_floor));
            return;
        }
        internal static void UFuncCeil(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_ceil));
            return;
        }
        internal static void UFuncRint(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_rint));
            return;
        }
        internal static void UFuncConjugate(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData)
        {
            UFuncCommon(ref bufPtr, ref N, ref steps, funcData, GetOperation(bufPtr[0], NpyArray_Ops.npy_op_conjugate));
            return;
        }
    }
}
