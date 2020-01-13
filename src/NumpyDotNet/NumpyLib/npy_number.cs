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
                                         NPY_TYPES.NPY_BIGINT},

             };

            return loc;
        }

        internal static void UFuncCommon(ref VoidPtr[] bufPtr, ref npy_intp N, ref npy_intp[] steps, object funcData, NumericOperation Operation)
        {
            VoidPtr Operand1 = bufPtr[0];
            VoidPtr Operand2 = bufPtr[1];
            VoidPtr Result = bufPtr[2];

            long O1_Step = steps[0];
            long O2_Step = steps[1];
            long R_Step = steps[2];

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

            long O1_sizeData = Operand1Handler.ItemSize;
            long O2_sizeData = Operand2Handler.ItemSize;
            long R_sizeData = ResultHandler.ItemSize;

            long O1_Offset = Operand1.data_offset;
            long O2_Offset = Operand2.data_offset;
            long R_Offset = Result.data_offset;



            for (int i = 0; i < N; i++)
            {
                long O1_Index = ((i * O1_Step) + O1_Offset) / O1_sizeData;
                long O2_Index  = ((i * O2_Step) + O2_Offset) / O2_sizeData;
                long R_Index = ((i * R_Step) + R_Offset) / R_sizeData;

                try
                {
                    var O1 = Operand1Handler.GetIndex(Operand1, O1_Index);                  // get operand 1
                    var O2 = Operand2Handler.GetIndex(Operand2, O2_Index);                  // get operand 2
                    var R = Operation(O1, Operand1Handler.MathOpConvertOperand(O1, O2));    // calculate result
                    ResultHandler.SetIndex(Result, R_Index, R);
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
