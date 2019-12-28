using System;
using System.Collections.Generic;
using System.Text;

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

    }
}
