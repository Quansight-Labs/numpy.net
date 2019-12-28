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
        protected static T AddOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue + operand;
        }
        protected static T SubtractOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue - operand;
        }
        private static T MultiplyOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue * operand;
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
        private static T PowerOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return Math.Pow(dValue, operand);
        }
        private static T SquareOperation<T>(T bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue * dValue;
        }
    }

    internal class BoolHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public BoolHandlers()
        {
            AddOperation = INT32_AddOperation;
            SubtractOperation = BOOL_SubtractOperation;
            MultiplyOperation = BOOL_MultiplyOperation;
            DivideOperation = BOOL_DivideOperation;
            RemainderOperation = BOOL_RemainderOperation;
            FModOperation = BOOL_FModOperation;
            PowerOperation = BOOL_PowerOperation;
            SquareOperation = BOOL_SquareOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }
        public NumericOperation DivideOperation { get; set; }
        public NumericOperation RemainderOperation { get; set; }
        public NumericOperation FModOperation { get; set; }
        public NumericOperation PowerOperation { get; set; }
        public NumericOperation SquareOperation { get; set; }


        private object INT32_AddOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue + (double)operand;
        }
        private static object BOOL_SubtractOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue | Convert.ToBoolean(operand);
        }
        private static object BOOL_MultiplyOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        private static object BOOL_DivideOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        private static object BOOL_RemainderOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        private static object BOOL_FModOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        private static object BOOL_PowerOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ (bool)operand;
        }
        private static object BOOL_SquareOperation(object bValue, object operand)
        {
            bool dValue = (bool)bValue;
            return dValue ^ dValue;
        }
    }

    internal class ByteHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public ByteHandlers()
        {
            AddOperation = BYTE_AddOperation;
            SubtractOperation = BYTE_SubtractOperation;
            MultiplyOperation = BYTE_MultiplyOperation;
            DivideOperation = BYTE_DivideOperation;
            RemainderOperation = BYTE_RemainderOperation;
            FModOperation = BYTE_FModOperation;
            PowerOperation = BYTE_PowerOperation;
            SquareOperation = BYTE_SquareOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }
        public NumericOperation DivideOperation { get; set; }
        public NumericOperation RemainderOperation { get; set; }
        public NumericOperation FModOperation { get; set; }
        public NumericOperation PowerOperation { get; set; }
        public NumericOperation SquareOperation { get; set; }


        private object BYTE_AddOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue + (double)operand;
        }
        private static object BYTE_SubtractOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue - (double)operand;
        }
        private static object BYTE_MultiplyOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue * (double)operand;
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
        private static object BYTE_PowerOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object BYTE_SquareOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue * dValue;
        }
    }

    internal class UByteHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public UByteHandlers()
        {
            AddOperation = UBYTE_AddOperation;
            SubtractOperation = UBYTE_SubtractOperation;
            MultiplyOperation = UBYTE_MultiplyOperation;
            DivideOperation = UBYTE_DivideOperation;
            RemainderOperation = UBYTE_RemainderOperation;
            FModOperation = UBYTE_FModOperation;
            PowerOperation = UBYTE_PowerOperation;
            SquareOperation = UBYTE_SquareOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }
        public NumericOperation DivideOperation { get; set; }
        public NumericOperation RemainderOperation { get; set; }
        public NumericOperation FModOperation { get; set; }
        public NumericOperation PowerOperation { get; set; }
        public NumericOperation SquareOperation { get; set; }

        private object UBYTE_AddOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue + (double)operand;
        }
        private static object UBYTE_SubtractOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue - (double)operand;
        }
        private static object UBYTE_MultiplyOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue * (double)operand;
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
        private static object UBYTE_PowerOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object UBYTE_SquareOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue * dValue;
        }
    }

    internal class Int16Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public Int16Handlers()
        {
            AddOperation = INT16_AddOperation;
            SubtractOperation = INT16_SubtractOperation;
            MultiplyOperation = INT16_MultiplyOperation;
            DivideOperation = INT16_DivideOperation;
            RemainderOperation = INT16_RemainderOperation;
            FModOperation = INT16_FModOperation;
            PowerOperation = INT16_PowerOperation;
            SquareOperation = INT16_SquareOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }
        public NumericOperation DivideOperation { get; set; }
        public NumericOperation RemainderOperation { get; set; }
        public NumericOperation FModOperation { get; set; }
        public NumericOperation PowerOperation { get; set; }
        public NumericOperation SquareOperation { get; set; }

        private static object INT16_AddOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue + (double)operand;
        }
        private static object INT16_SubtractOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue - (double)operand;
        }
        private static object INT16_MultiplyOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue * (double)operand;
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
        private static object INT16_PowerOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object INT16_SquareOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue * dValue;
        }
    }

    internal class UInt16Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public UInt16Handlers()
        {
            AddOperation = UINT16_AddOperation;
            SubtractOperation = UINT16_SubtractOperation;
            MultiplyOperation = UINT16_MultiplyOperation;
            DivideOperation = UINT16_DivideOperation;
            RemainderOperation = UINT16_RemainderOperation;
            FModOperation = UINT16_FModOperation;
            PowerOperation = UINT16_PowerOperation;
            SquareOperation = UINT16_SquareOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }
        public NumericOperation DivideOperation { get; set; }
        public NumericOperation RemainderOperation { get; set; }
        public NumericOperation FModOperation { get; set; }
        public NumericOperation PowerOperation { get; set; }
        public NumericOperation SquareOperation { get; set; }


        private static object UINT16_AddOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue + (double)operand;
        }
        private static object UINT16_SubtractOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue - (double)operand;
        }
        private static object UINT16_MultiplyOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue * (double)operand;
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
        private static object UINT16_PowerOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object UINT16_SquareOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue * dValue;
        }
    }

    internal class Int32Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public Int32Handlers()
        {
            AddOperation = INT32_AddOperation;
            SubtractOperation = INT32_SubtractOperation;
            MultiplyOperation = INT32_MultiplyOperation;
            DivideOperation = INT32_DivideOperation;
            RemainderOperation = INT32_RemainderOperation;
            FModOperation = INT32_FModOperation;
            PowerOperation = INT32_PowerOperation;
            SquareOperation = INT32_SquareOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }
        public NumericOperation DivideOperation { get; set; }
        public NumericOperation RemainderOperation { get; set; }
        public NumericOperation FModOperation { get; set; }
        public NumericOperation PowerOperation { get; set; }
        public NumericOperation SquareOperation { get; set; }

        private object INT32_AddOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue + (double)operand;
        }
        private static object INT32_SubtractOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue - (double)operand;
        }
        private static object INT32_MultiplyOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue * (double)operand;
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
        private static object INT32_PowerOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object INT32_SquareOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue * dValue;
        }
    }

    internal class UInt32Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public UInt32Handlers()
        {
            AddOperation = UINT32_AddOperation;
            SubtractOperation = UINT32_SubtractOperation;
            MultiplyOperation = UINT32_MultiplyOperation;
            DivideOperation = UINT32_DivideOperation;
            RemainderOperation = UINT32_RemainderOperation;
            FModOperation = UINT32_FModOperation;
            PowerOperation = UINT32_PowerOperation;
            SquareOperation = UINT32_SquareOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }
        public NumericOperation DivideOperation { get; set; }
        public NumericOperation RemainderOperation { get; set; }
        public NumericOperation FModOperation { get; set; }
        public NumericOperation PowerOperation { get; set; }
        public NumericOperation SquareOperation { get; set; }

        private object UINT32_AddOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue + (double)operand;
        }
        private static object UINT32_SubtractOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue - (double)operand;
        }
        private static object UINT32_MultiplyOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue * (double)operand;
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
        private static object UINT32_PowerOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object UINT32_SquareOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue * dValue;
        }
    }

    internal class Int64Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public Int64Handlers()
        {
            AddOperation = INT64_AddOperation;
            SubtractOperation = INT64_SubtractOperation;
            MultiplyOperation = INT64_MultiplyOperation;
            DivideOperation = INT64_DivideOperation;
            RemainderOperation = INT64_RemainderOperation;
            FModOperation = INT64_FModOperation;
            PowerOperation = INT64_PowerOperation;
            SquareOperation = INT64_SquareOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }
        public NumericOperation DivideOperation { get; set; }
        public NumericOperation RemainderOperation { get; set; }
        public NumericOperation FModOperation { get; set; }
        public NumericOperation PowerOperation { get; set; }
        public NumericOperation SquareOperation { get; set; }

        private object INT64_AddOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue + (double)operand;
        }
        private static object INT64_SubtractOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue - (double)operand;
        }
        private static object INT64_MultiplyOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue * (double)operand;
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
        private static object INT64_PowerOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object INT64_SquareOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue * dValue;
        }
    }

    internal class UInt64Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public UInt64Handlers()
        {
            AddOperation = UINT64_AddOperation;
            SubtractOperation = UINT64_SubtractOperation;
            MultiplyOperation = UINT64_MultiplyOperation;
            DivideOperation = UINT64_DivideOperation;
            RemainderOperation = UINT64_RemainderOperation;
            FModOperation = UINT64_FModOperation;
            PowerOperation = UINT64_PowerOperation;
            SquareOperation = UINT64_SquareOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }
        public NumericOperation DivideOperation { get; set; }
        public NumericOperation RemainderOperation { get; set; }
        public NumericOperation FModOperation { get; set; }
        public NumericOperation PowerOperation { get; set; }
        public NumericOperation SquareOperation { get; set; }

        private object UINT64_AddOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue + (double)operand;
        }
        private static object UINT64_SubtractOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue - (double)operand;
        }
        private static object UINT64_MultiplyOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue * (double)operand;
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
        private static object UINT64_PowerOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object UINT64_SquareOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue * dValue;
        }
    }

    internal class FloatHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public FloatHandlers()
        {
            AddOperation = Float_AddOperation;
            SubtractOperation = FLOAT_SubtractOperation;
            MultiplyOperation = FLOAT_MultiplyOperation;
            DivideOperation = FLOAT_DivideOperation;
            RemainderOperation = FLOAT_RemainderOperation;
            FModOperation = FLOAT_FModOperation;
            PowerOperation = FLOAT_PowerOperation;
            SquareOperation = FLOAT_SquareOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }
        public NumericOperation DivideOperation { get; set; }
        public NumericOperation RemainderOperation { get; set; }
        public NumericOperation FModOperation { get; set; }
        public NumericOperation PowerOperation { get; set; }
        public NumericOperation SquareOperation { get; set; }

        private object Float_AddOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue + (double)operand;
        }
        private static object FLOAT_SubtractOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue - (double)operand;
        }
        private static object FLOAT_MultiplyOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue * (double)operand;
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
        private static object FLOAT_PowerOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object FLOAT_SquareOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue * dValue;
        }
    }

    internal class DoubleHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public DoubleHandlers()
        {
            AddOperation = Double_AddOperation;
            SubtractOperation = DOUBLE_SubtractOperation;
            MultiplyOperation = DOUBLE_MultiplyOperation;
            DivideOperation = DOUBLE_DivideOperation;
            RemainderOperation = DOUBLE_RemainderOperation;
            FModOperation = DOUBLE_FModOperation;
            PowerOperation = DOUBLE_PowerOperation;
            SquareOperation = DOUBLE_SquareOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }
        public NumericOperation DivideOperation { get; set; }
        public NumericOperation RemainderOperation { get; set; }
        public NumericOperation FModOperation { get; set; }
        public NumericOperation PowerOperation { get; set; }
        public NumericOperation SquareOperation { get; set; }

        private object Double_AddOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue + (double)operand;
        }
        private static object DOUBLE_SubtractOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue - (double)operand;
        }
        private static object DOUBLE_MultiplyOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue * (double)operand;
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
        private static object DOUBLE_PowerOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return Math.Pow(dValue, (double)operand);
        }
        private static object DOUBLE_SquareOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue * dValue;
        }
    }

    internal class DecimalHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public DecimalHandlers()
        {
            AddOperation = Decimal_AddOperation;
            SubtractOperation = DECIMAL_SubtractOperation;
            MultiplyOperation = DECIMAL_MultiplyOperation;
            DivideOperation = DECIMAL_DivideOperation;
            RemainderOperation = DECIMAL_RemainderOperation;
            FModOperation = DECIMAL_FModOperation;
            PowerOperation = DECIMAL_PowerOperation;
            SquareOperation = DECIMAL_SquareOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }
        public NumericOperation DivideOperation { get; set; }
        public NumericOperation RemainderOperation { get; set; }
        public NumericOperation FModOperation { get; set; }
        public NumericOperation PowerOperation { get; set; }
        public NumericOperation SquareOperation { get; set; }

        private object Decimal_AddOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue + (decimal)operand;
        }
        private static object DECIMAL_SubtractOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue - (decimal)operand;
        }
        private static object DECIMAL_MultiplyOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue * (decimal)operand;
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
        private static object DECIMAL_PowerOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return Math.Pow(Convert.ToDouble(dValue), Convert.ToDouble(operand));
        }
        private static object DECIMAL_SquareOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue * dValue;
        }

    }
}
