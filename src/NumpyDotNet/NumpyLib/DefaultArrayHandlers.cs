using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;

namespace NumpyLib
{
    public interface IArrayHandlers
    {
        NumericOperation AddOperation { get; set; }
        NumericOperation SubtractOperation { get; set; }
        NumericOperation MultiplyOperation { get; set; }
        NumericOperation DivideOperation { get; set; }
        NumericOperation RemainderOperation { get; set; }
        NumericOperation FModOperation { get; set; }
        NumericOperation PowerOperation { get; set; }
        NumericOperation SquareOperation { get; set; }
        NumericOperation ReciprocalOperation { get; set; }
        NumericOperation OnesLikeOperation { get; set; }
        NumericOperation SqrtOperation { get; set; }
        NumericOperation NegativeOperation { get; set; }
        NumericOperation AbsoluteOperation { get; set; }
        NumericOperation InvertOperation { get; set; }
        NumericOperation LeftShiftOperation { get; set; }
        NumericOperation RightShiftOperation { get; set; }
        NumericOperation BitWiseAndOperation { get; set; }
        NumericOperation BitWiseXorOperation { get; set; }
        NumericOperation BitWiseOrOperation { get; set; }
        NumericOperation LessOperation { get; set; }
        NumericOperation LessEqualOperation { get; set; }
        NumericOperation EqualOperation { get; set; }
        NumericOperation NotEqualOperation { get; set; }
        NumericOperation GreaterOperation { get; set; }
        NumericOperation GreaterEqualOperation { get; set; }
        NumericOperation IsNANOperation { get; set; }
        NumericOperation FloorDivideOperation { get; set; }
        NumericOperation TrueDivideOperation { get; set; }
        NumericOperation LogicalOrOperation { get; set; }
        NumericOperation LogicalAndOperation { get; set; }
        NumericOperation FloorOperation { get; set; }
        NumericOperation CeilingOperation { get; set; }
        NumericOperation MaximumOperation { get; set; }
        NumericOperation FMaxOperation { get; set; }
        NumericOperation MinimumOperation { get; set; }
        NumericOperation FMinOperation { get; set; }
        NumericOperation HeavisideOperation { get; set; }
        NumericOperation RintOperation { get; set; }
        NumericOperation ConjugateOperation { get; set; }

    }

    public delegate object NumericOperation(object bValue, object operand);


    public class DefaultArrayHandlers
    {
        public static void Initialize()
        {
            SetArrayHandler(NPY_TYPES.NPY_BOOL, new BoolHandlers());
            SetArrayHandler(NPY_TYPES.NPY_BYTE, new ByteHandlers());
            SetArrayHandler(NPY_TYPES.NPY_UBYTE, new UByteHandlers());
            SetArrayHandler(NPY_TYPES.NPY_INT16, new Int16Handlers());
            SetArrayHandler(NPY_TYPES.NPY_UINT16, new UInt16Handlers());
            SetArrayHandler(NPY_TYPES.NPY_INT32, new Int32Handlers());
            SetArrayHandler(NPY_TYPES.NPY_UINT32, new UInt32Handlers());
            SetArrayHandler(NPY_TYPES.NPY_INT64, new Int64Handlers());
            SetArrayHandler(NPY_TYPES.NPY_UINT64, new UInt64Handlers());
            SetArrayHandler(NPY_TYPES.NPY_FLOAT, new FloatHandlers());
            SetArrayHandler(NPY_TYPES.NPY_DOUBLE, new DoubleHandlers());
            SetArrayHandler(NPY_TYPES.NPY_DECIMAL, new DecimalHandlers());
            SetArrayHandler(NPY_TYPES.NPY_COMPLEX, new ComplexHandlers());
        }

        public static void SetArrayHandler(NPY_TYPES ItemType, IArrayHandlers Handlers)
        {
            ArrayHandlers[(int)ItemType] = Handlers;
        }

        public static IArrayHandlers GetArrayHandler(NPY_TYPES ItemType)
        {
            int index = (int)ItemType;

            if (index < 0)
            {
                throw new Exception("ItemType can't be a negative value");
            }
            if (index >= ArrayHandlers.Length)
            {
                throw new Exception("Specified ItemType is not found");
            }

            IArrayHandlers Handler = ArrayHandlers[(int)ItemType];
            if (Handler == null)
            {
                throw new Exception("The specified ItemType does not have a registered handler");
            }

            return Handler;
        }

        private static IArrayHandlers[] ArrayHandlers = new IArrayHandlers[255];

    }

    internal class NumericOperations
    {
        public NumericOperations()
        {

        }
        public NpyArray_GetItemFunc srcGetItem;
        public NpyArray_SetItemFunc srcSetItem;

        public NpyArray_GetItemFunc destGetItem;
        public NpyArray_SetItemFunc destSetItem;


        public NpyArray_GetItemFunc operandGetItem;
        public NpyArray_SetItemFunc operandSetItem;

        public NumericOperation operation;

        public static NumericOperations GetOperations(NumericOperation operation, NpyArray srcArray, NpyArray destArray, NpyArray operandArray)
        {
            NumericOperations operations = new NumericOperations();
            operations.operation = operation;

            if (srcArray != null)
            {
                operations.srcGetItem = srcArray.descr.f.getitem;
                operations.srcSetItem = srcArray.descr.f.setitem;
            }

            if (destArray != null)
            {
                operations.destGetItem = destArray.descr.f.getitem;
                operations.destSetItem = destArray.descr.f.setitem;
            }

            if (operandArray != null)
            {
                operations.operandGetItem = operandArray.descr.f.getitem;
                operations.operandSetItem = operandArray.descr.f.setitem;
            }


            return operations;
        }
    }

    internal class ArrayHandlerBase
    {
        internal ArrayHandlerBase()
        {
            AddOperation = Add;
            SubtractOperation = Subtract;
            MultiplyOperation = Multiply;
            DivideOperation = Divide;
            RemainderOperation = Remainder;
            FModOperation = FMod;
            PowerOperation = Power;
            SquareOperation = Square;
            ReciprocalOperation = Reciprocal;
            OnesLikeOperation = OnesLike;
            SqrtOperation = Sqrt;
            NegativeOperation = Negative;
            AbsoluteOperation = Absolute;
            InvertOperation = Invert;
            LeftShiftOperation = LeftShift;
            RightShiftOperation = RightShift;
            BitWiseAndOperation = BitWiseAnd;
            BitWiseXorOperation = BitWiseXor;
            BitWiseOrOperation = BitWiseOr;
            LessOperation = Less;
            LessEqualOperation = LessEqual;
            EqualOperation = Equal;
            NotEqualOperation = NotEqual;
            GreaterOperation = Greater;
            GreaterEqualOperation = GreaterEqual;
            IsNANOperation = IsNAN;
            FloorDivideOperation = FloorDivide;
            TrueDivideOperation = TrueDivide;
            LogicalOrOperation = LogicalOr;
            LogicalAndOperation = LogicalAnd;
            FloorOperation = Floor;
            CeilingOperation = Ceiling;
            MaximumOperation = Maximum;
            FMaxOperation = FMax;
            MinimumOperation = Minimum;
            FMinOperation = FMin;
            HeavisideOperation = Heaviside;
            RintOperation = Rint;
            ConjugateOperation = Conjugate;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }
        public NumericOperation DivideOperation { get; set; }
        public NumericOperation RemainderOperation { get; set; }
        public NumericOperation FModOperation { get; set; }
        public NumericOperation PowerOperation { get; set; }
        public NumericOperation SquareOperation { get; set; }
        public NumericOperation ReciprocalOperation { get; set; }
        public NumericOperation OnesLikeOperation { get; set; }
        public NumericOperation SqrtOperation { get; set; }
        public NumericOperation NegativeOperation { get; set; }
        public NumericOperation AbsoluteOperation { get; set; }
        public NumericOperation InvertOperation { get; set; }
        public NumericOperation LeftShiftOperation { get; set; }
        public NumericOperation RightShiftOperation { get; set; }
        public NumericOperation BitWiseAndOperation { get; set; }
        public NumericOperation BitWiseXorOperation { get; set; }
        public NumericOperation BitWiseOrOperation { get; set; }
        public NumericOperation LessOperation { get; set; }
        public NumericOperation LessEqualOperation { get; set; }
        public NumericOperation EqualOperation { get; set; }
        public NumericOperation NotEqualOperation { get; set; }
        public NumericOperation GreaterOperation { get; set; }
        public NumericOperation GreaterEqualOperation { get; set; }
        public NumericOperation IsNANOperation { get; set; }
        public NumericOperation FloorDivideOperation { get; set; }
        public NumericOperation TrueDivideOperation { get; set; }
        public NumericOperation LogicalOrOperation { get; set; }
        public NumericOperation LogicalAndOperation { get; set; }
        public NumericOperation FloorOperation { get; set; }
        public NumericOperation CeilingOperation { get; set; }
        public NumericOperation MaximumOperation { get; set; }
        public NumericOperation FMaxOperation { get; set; }
        public NumericOperation MinimumOperation { get; set; }
        public NumericOperation FMinOperation { get; set; }
        public NumericOperation HeavisideOperation { get; set; }
        public NumericOperation RintOperation { get; set; }
        public NumericOperation ConjugateOperation { get; set; }



        protected virtual object Add(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue + operand;
        }
        protected virtual object Subtract(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue - operand;
        }
        protected virtual object Multiply(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue * operand;
        }
        protected virtual object Divide(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            if (operand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / operand;
        }
        protected virtual object Remainder(dynamic bValue, dynamic operand)
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
        protected virtual object FMod(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            if (operand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % operand;
        }
        protected virtual object Power(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return Math.Pow(dValue, operand);
        }
        protected virtual object Square(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue * dValue;
        }
        protected virtual object Reciprocal(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return 1 / dValue;
        }
        protected virtual object OnesLike(dynamic bValue, object operand)
        {
            double dValue = 1;
            return dValue;
        }
        protected virtual object Sqrt(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return Math.Sqrt(dValue);
        }
        protected virtual object Negative(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return -dValue;
        }
        protected virtual object Absolute(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return Math.Abs(dValue);
        }
        protected virtual object Invert(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return ~dValue;
        }
        protected virtual object LeftShift(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected virtual object RightShift(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected virtual object BitWiseAnd(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue & Convert.ToUInt64(operand);
        }
        protected virtual object BitWiseXor(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue ^ operand;
        }
        protected virtual object BitWiseOr(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue | (Int64)operand;
        }
        protected virtual object Less(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue < operand;
        }
        protected virtual object LessEqual(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue <= operand;
        }
        protected virtual object Equal(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue == operand;
        }
        protected virtual object NotEqual(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue != operand;
        }
        protected virtual object Greater(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue > operand;
        }
        protected virtual object GreaterEqual(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue >= operand;
        }
        protected virtual object IsNAN(object bValue, object operand)
        {
            return float.IsNaN((float)bValue);
        }
        protected virtual object FloorDivide(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            if (operand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return Math.Floor(dValue / operand);
        }
        protected virtual object TrueDivide(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue / operand;
        }
        protected virtual object LogicalOr(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue || Convert.ToBoolean(operand);
        }
        protected virtual object LogicalAnd(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue && Convert.ToBoolean(operand);
        }
        protected virtual object Floor(dynamic bValue, dynamic operand)
        {
            return Math.Floor(Convert.ToDouble(bValue));
        }
        protected virtual object Ceiling(dynamic bValue, dynamic operand)
        {
            return Math.Ceiling(Convert.ToDouble(bValue));
        }
        protected virtual object Maximum(dynamic bValue, dynamic operand)
        {
            return Math.Max(Convert.ToDouble(bValue), operand);
        }
        protected virtual object FMax(dynamic bValue, dynamic operand)
        {
            return Math.Max(Convert.ToDouble(bValue), operand);
        }
        protected virtual object Minimum(dynamic bValue, dynamic operand)
        {
            return Math.Min(Convert.ToDouble(bValue), operand);
        }
        protected virtual object FMin(dynamic bValue, dynamic operand)
        {
            return Math.Min(Convert.ToDouble(bValue), operand);
        }
        protected virtual object Heaviside(dynamic bValue, dynamic operand)
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
        protected virtual object Rint(dynamic bValue, dynamic operand)
        {
            return Math.Round(Convert.ToDouble(bValue));
        }
        protected virtual object Conjugate(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue;
        }

    }

    internal class BoolHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public BoolHandlers()
        {
        }


        protected override object Add(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue + (double)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue | Convert.ToBoolean(operand);
        }
        protected override object Multiply(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        protected override object Remainder(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        protected override object FMod(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        protected override object Power(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        protected override object Square(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        protected override object Negative(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        protected override object Absolute(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return false;
        }
        protected override object Invert(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return !dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue;
        }
        protected override object RightShift(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue;
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue & Convert.ToBoolean(operand);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ Convert.ToBoolean(operand);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue | Convert.ToBoolean(operand);
        }
        protected override object Less(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return false;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return false;
        }
        protected override object Equal(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue == Convert.ToBoolean(operand);
        }
        protected override object NotEqual(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue != Convert.ToBoolean(operand);
        }
        protected override object Greater(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return true;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return true;
        }
    }

    internal class ByteHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public ByteHandlers()
        {
  
        }

        protected override object Add(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue + (double)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue - (double)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue * (double)operand;
        }
        protected override object Divide(object bValue, object operand)
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
        protected override object Remainder(object bValue, object operand)
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
        protected override object FMod(object bValue, object operand)
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
        protected override object Power(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        protected override object Square(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue * dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return 1 / dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return Math.Sqrt(dValue);
        }
        protected override object Negative(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return -dValue;
        }
        protected override object Absolute(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return Math.Abs(dValue);
        }
        protected override object Invert(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return ~dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected override object RightShift(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue & Convert.ToSByte(operand);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue ^ Convert.ToSByte(operand);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue | Convert.ToSByte(operand);
        }
        protected override object Less(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue < (double)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue <= (double)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue == (double)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue != (double)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue > (double)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue >= (double)operand;
        }
    }

    internal class UByteHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public UByteHandlers()
        {

        }

        protected override object Add(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue + (double)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue - (double)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue * (double)operand;
        }
        protected override object Divide(object bValue, object operand)
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
        protected override object Remainder(object bValue, object operand)
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
        protected override object FMod(object bValue, object operand)
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
        protected override object Power(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        protected override object Square(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue * dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return 1 / dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return Math.Sqrt(dValue);
        }
        protected override object Negative(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return -dValue;
        }
        protected override object Absolute(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return Math.Abs(dValue);
        }
        protected override object Invert(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return ~dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected override object RightShift(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue & Convert.ToByte(operand);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue ^ Convert.ToByte(operand);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue | Convert.ToByte(operand);
        }
        protected override object Less(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue < (double)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue <= (double)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue == (double)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue != (double)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue > (double)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue >= (double)operand;
        }
    }

    internal class Int16Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public Int16Handlers()
        {
 
        }

        protected override object Add(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue + (double)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue - (double)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue * (double)operand;
        }
        protected override object Divide(object bValue, object operand)
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
        protected override object Remainder(object bValue, object operand)
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
        protected override object FMod(object bValue, object operand)
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
        protected override object Power(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        protected override object Square(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue * dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return 1 / dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return Math.Sqrt(dValue);
        }
        protected override object Negative(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return -dValue;
        }
        protected override object Absolute(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return Math.Abs(dValue);
        }
        protected override object Invert(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return ~dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected override object RightShift(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue & Convert.ToInt16(operand);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue ^ Convert.ToInt16(operand);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue | Convert.ToInt16(operand);
        }
        protected override object Less(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue < (double)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue <= (double)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue == (double)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue != (double)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue > (double)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue >= (double)operand;
        }
    }

    internal class UInt16Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public UInt16Handlers()
        {
 
        }


        protected override object Add(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue + (double)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue - (double)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue * (double)operand;
        }
        protected override object Divide(object bValue, object operand)
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
        protected override object Remainder(object bValue, object operand)
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
        protected override object FMod(object bValue, object operand)
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
        protected override object Power(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        protected override object Square(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue * dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return 1 / dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return Math.Sqrt(dValue);
        }
        protected override object Negative(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return -dValue;
        }
        protected override object Absolute(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return Math.Abs(dValue);
        }
        protected override object Invert(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return ~dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected override object RightShift(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue & Convert.ToUInt16(operand);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue ^ Convert.ToUInt16(operand);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue | Convert.ToUInt16(operand);
        }
        protected override object Less(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue < (double)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue <= (double)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue == (double)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue != (double)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue > (double)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue >= (double)operand;
        }
    }

    internal class Int32Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public Int32Handlers()
        {
  
        }


        protected override object Add(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue + (double)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue - (double)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue * (double)operand;
        }
        protected override object Divide(object bValue, object operand)
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
        protected override object Remainder(object bValue, object operand)
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
        protected override object FMod(object bValue, object operand)
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
        protected override object Power(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        protected override object Square(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue * dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return 1 / dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return Math.Sqrt(dValue);
        }
        protected override object Negative(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return -dValue;
        }
        protected override object Absolute(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return Math.Abs(dValue);
        }
        protected override object Invert(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return ~dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected override object RightShift(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue & Convert.ToInt32(operand);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue ^ Convert.ToInt32(operand);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue | Convert.ToInt32(operand);
        }
        protected override object Less(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue < (double)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue <= (double)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue == (double)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue != (double)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue > (double)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue >= (double)operand;
        }
    }

    internal class UInt32Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public UInt32Handlers()
        {
 
        }

        protected override object Add(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue + (double)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue - (double)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue * (double)operand;
        }
        protected override object Divide(object bValue, object operand)
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
        protected override object Remainder(object bValue, object operand)
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
        protected override object FMod(object bValue, object operand)
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
        protected override object Power(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        protected override object Square(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue * dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return 1 / dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return Math.Sqrt(dValue);
        }
        protected override object Negative(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return -dValue;
        }
        protected override object Absolute(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return Math.Abs(dValue);
        }
        protected override object Invert(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return ~dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected override object RightShift(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue & Convert.ToUInt32(operand);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue ^ Convert.ToUInt32(operand);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue | Convert.ToUInt32(operand);
        }
        protected override object Less(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue < (double)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue <= (double)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue == (double)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue != (double)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue > (double)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue >= (double)operand;
        }
    }

    internal class Int64Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public Int64Handlers()
        {

        }

        protected override object Add(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue + (double)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue - (double)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue * (double)operand;
        }
        protected override object Divide(object bValue, object operand)
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
        protected override object Remainder(object bValue, object operand)
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
        protected override object FMod(object bValue, object operand)
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
        protected override object Power(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        protected override object Square(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue * dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return 1 / dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return Math.Sqrt(dValue);
        }
        protected override object Negative(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return -dValue;
        }
        protected override object Absolute(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return Math.Abs(dValue);
        }
        protected override object Invert(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return ~dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected override object RightShift(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue & Convert.ToInt64(operand);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue ^ Convert.ToInt64(operand);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue | Convert.ToInt64(operand);
        }
        protected override object Less(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue < (double)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue <= (double)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue == (double)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue != (double)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue > (double)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue >= (double)operand;
        }
    }

    internal class UInt64Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public UInt64Handlers()
        {
   
        }

        protected override object Add(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue + (double)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue - (double)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue * (double)operand;
        }
        protected override object Divide(object bValue, object operand)
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
        protected override object Remainder(object bValue, object operand)
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
        protected override object FMod(object bValue, object operand)
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
        protected override object Power(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        protected override object Square(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue * dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return 1 / dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return Math.Sqrt(dValue);
        }
        protected override object Negative(object bValue, object operand)
        {
            Int64 dValue = (Int64)(UInt64)bValue;
            return (UInt64)(-dValue);
        }
        protected override object Absolute(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return Math.Abs(dValue);
        }
        protected override object Invert(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return ~dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected override object RightShift(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue & Convert.ToUInt64(operand);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue ^ Convert.ToUInt64(operand);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue | Convert.ToUInt64(operand);
        }
        protected override object Less(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue < (double)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue <= (double)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue == (double)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue != (double)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue > (double)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue >= (double)operand;
        }
    }

    internal class FloatHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public FloatHandlers()
        {
 
        }

        protected override object Add(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue + (double)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue - (double)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue * (double)operand;
        }
        protected override object Divide(object bValue, object operand)
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
        protected override object Remainder(object bValue, object operand)
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
        protected override object FMod(object bValue, object operand)
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
        protected override object Power(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        protected override object Square(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue * dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return 1 / dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return Math.Sqrt(dValue);
        }
        protected override object Negative(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return -dValue;
        }
        protected override object Absolute(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return Math.Abs(dValue);
        }
        protected override object Invert(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(float)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected override object RightShift(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(float)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(float)bValue;
            return dValue & Convert.ToUInt64(operand);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue ^ Convert.ToUInt64(operand);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(float)bValue;
            return dValue | Convert.ToUInt64(operand);
        }
        protected override object Less(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue < (double)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue <= (double)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue == (double)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue != (double)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue > (double)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue >= (double)operand;
        }
        protected override object FMax(object bValue, object operand)
        {

            if (float.IsNaN(Convert.ToSingle(operand)))
                return bValue;
            if (float.IsNaN(Convert.ToSingle(bValue)))
                return operand;

            return Math.Max(Convert.ToSingle(bValue), Convert.ToSingle(operand));

        }
        protected override object FMin(object bValue, dynamic operand)
        {

            if (float.IsNaN(Convert.ToSingle(operand)))
                return bValue;
            if (float.IsNaN(Convert.ToSingle(bValue)))
                return operand;

            return Math.Min(Convert.ToSingle(bValue), Convert.ToSingle(operand));

        }
        protected override object Heaviside(object bValue, object operand)
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
    }

    internal class DoubleHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public DoubleHandlers()
        {

        }

        protected override object Add(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue + (double)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue - (double)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue * (double)operand;
        }
        protected override object Divide(object bValue, object operand)
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
        protected override object Remainder(object bValue, object operand)
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
        protected override object FMod(object bValue, object operand)
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
        protected override object Power(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        protected override object Square(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue * dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return 1 / dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return Math.Sqrt(dValue);
        }
        protected override object Negative(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return -dValue;
        }
        protected override object Absolute(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return Math.Abs(dValue);
        }
        protected override object Invert(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(double)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected override object RightShift(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(double)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(double)bValue;
            return dValue & Convert.ToUInt64(operand);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue ^ Convert.ToUInt64(operand);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(double)bValue;
            return dValue | Convert.ToUInt64(operand);
        }
        protected override object Less(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue < (double)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue <= (double)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue == (double)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue != (double)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue > (double)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue >= (double)operand;
        }
        protected override object IsNAN(object bValue, object operand)
        {
            return double.IsNaN((double)bValue);
        }
        protected override object FMax(object bValue, object operand)
        {

            if (double.IsNaN(Convert.ToDouble(operand)))
                return bValue;
            if (double.IsNaN(Convert.ToDouble(bValue)))
                return operand;

            return Math.Max(Convert.ToDouble(bValue), Convert.ToDouble(operand));

        }
        protected override object FMin(object bValue, dynamic operand)
        {

            if (double.IsNaN(Convert.ToDouble(operand)))
                return bValue;
            if (double.IsNaN(Convert.ToDouble(bValue)))
                return operand;

            return Math.Min(Convert.ToDouble(bValue), Convert.ToDouble(operand));
        }
        protected override object Heaviside(object bValue, object operand)
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
    }

    internal class DecimalHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public DecimalHandlers()
        {
 
        }

        protected override object Add(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue + (decimal)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue - (decimal)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue * (decimal)operand;
        }
        protected override object Divide(object bValue, object operand)
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
        protected override object Remainder(object bValue, object operand)
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
        protected override object FMod(object bValue, object operand)
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
        protected override object Power(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return Math.Pow(Convert.ToDouble(dValue), Convert.ToDouble(operand));
        }
        protected override object Square(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue * dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return 1 / dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            decimal epsilon = 0.0M;

            if (dValue < 0)
                throw new OverflowException("Cannot calculate square root from a negative number");

            decimal current = (decimal)Math.Sqrt((double)dValue), previous;
            do
            {
                previous = current;
                if (previous == 0.0M)
                    return 0;

                current = (previous + dValue / previous) / 2;
            }
            while (Math.Abs(previous - current) > epsilon);
            return current;
        }
        protected override object Negative(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return -dValue;
        }
        protected override object Absolute(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return Math.Abs(dValue);
        }
        protected override object Invert(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(decimal)bValue;
            return dValue << Convert.ToInt32(operand);
        }
        protected override object RightShift(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(decimal)bValue;
            return dValue >> Convert.ToInt32(operand);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue & Convert.ToUInt64(operand);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            UInt64 dValue = Convert.ToUInt64(bValue);
            return dValue ^ Convert.ToUInt64(operand);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)(decimal)bValue;
            return dValue | Convert.ToUInt64(operand);
        }
        protected override object Less(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue < (decimal)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue <= (decimal)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue == (decimal)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue != (decimal)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue > (decimal)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue >= (decimal)operand;
        }
        protected override object IsNAN(object bValue, object operand)
        {
            return false;
        }
        protected override object Floor(object bValue, object operand)
        {
            return Math.Floor(Convert.ToDecimal(bValue));
        }
        protected override object Ceiling(object bValue, object operand)
        {
            return Math.Ceiling(Convert.ToDecimal(bValue));
        }
        protected override object Maximum(object bValue, object operand)
        {
            return Math.Max(Convert.ToDecimal(bValue), Convert.ToDecimal(operand));
        }
        protected override object FMax(object bValue, object operand)
        {
            return Math.Max(Convert.ToDecimal(bValue), Convert.ToDecimal(operand));
        }
        protected override object Minimum(object bValue, object operand)
        {
            return Math.Min(Convert.ToDecimal(bValue), Convert.ToDecimal(operand));
        }
        protected override object FMin(object bValue, object operand)
        {
            return Math.Min(Convert.ToDecimal(bValue), Convert.ToDecimal(operand));
        }
        protected override object Rint(dynamic bValue, dynamic operand)
        {
            return Math.Round(Convert.ToDecimal(bValue));
        }
    }

    internal class ComplexHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public ComplexHandlers()
        {

        }

        System.Numerics.Complex ConvertToComplex(object o)
        {
            if (o is System.Numerics.Complex)
            {
                System.Numerics.Complex c = (System.Numerics.Complex)o;
                return c;
            }
            else
            {
                return new System.Numerics.Complex(Convert.ToDouble(o), 0);
            }
        }

        protected override object Add(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return dValue + (Complex)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return dValue - (Complex)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return dValue * (Complex)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            Complex doperand = (Complex)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / doperand;
        }
        protected override object Remainder(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            Complex doperand = (Complex)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            var rem = dValue.Real % doperand.Real;
            if ((dValue.Real > 0) == (doperand.Real > 0) || rem == 0)
            {
                return rem;
            }
            else
            {
                return rem + doperand;
            }
        }
        protected override object FMod(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            Complex doperand = (Complex)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue.Real % doperand.Real;
        }
        protected override object Power(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return Complex.Pow(dValue, Convert.ToDouble(operand));
        }
        protected override object Square(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return dValue * dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return 1 / dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            return Complex.Sqrt((Complex)bValue);
        }
        protected override object Negative(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return -dValue;
        }
        protected override object Absolute(object bValue, object operand)
        {
            return Complex.Abs((Complex)bValue);
        }
        protected override object Invert(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            return bValue;
            //UInt64 dValue = (UInt64)(Complex)bValue;
            //return dValue << Convert.ToInt32(operand);
        }
        protected override object RightShift(object bValue, object operand)
        {
            return bValue;
            //UInt64 dValue = (UInt64)(Complex)bValue;
            //return dValue >> Convert.ToInt32(operand);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            return bValue;
            //UInt64 dValue = Convert.ToUInt64(bValue);
            //return dValue & Convert.ToUInt64(operand);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            return bValue;
            //UInt64 dValue = Convert.ToUInt64(bValue);
            //return dValue ^ Convert.ToUInt64(operand);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            return bValue;

            //UInt64 dValue = (UInt64)(decimal)bValue;
            //return dValue | Convert.ToUInt64(operand);
        }
        protected override object Less(object bValue, object operand)
        {
            return false;
            //Complex dValue = (Complex)bValue;
            //return dValue < (Complex)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            return false;

            //Complex dValue = (Complex)bValue;
            //return dValue <= (Complex)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return dValue == (Complex)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return dValue != (Complex)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            return false;
            //Complex dValue = (Complex)bValue;
            //return dValue > (Complex)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            return false;
            //Complex dValue = (Complex)bValue;
            //return dValue >= (Complex)operand;
        }
        protected override object IsNAN(object bValue, object operand)
        {
            return false;
        }
        protected override object Floor(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return Math.Floor(dValue.Real);
        }
        protected override object Ceiling(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return Math.Ceiling(dValue.Real);
        }
        protected override object Maximum(object bValue, object operand)
        {
            Complex a = (Complex)bValue;
            Complex b = (Complex)operand;

            return Math.Max(a.Real, b.Real);
        }
        protected override object FMax(object bValue, object operand)
        {
            Complex a = (Complex)bValue;
            Complex b = (Complex)operand;

            return Math.Max(a.Real, b.Real);
        }
        protected override object Minimum(object bValue, object operand)
        {
            Complex a = (Complex)bValue;
            Complex b = (Complex)operand;

            return Math.Min(a.Real, b.Real);
        }
        protected override object FMin(object bValue, object operand)
        {
            Complex a = (Complex)bValue;
            Complex b = (Complex)operand;

            return Math.Min(a.Real, b.Real);
        }
        protected override object Rint(dynamic bValue, dynamic operand)
        {
            Complex a = (Complex)bValue;
            Complex b = (Complex)operand;

            return Math.Round(a.Real);
        }
    }
}
