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
    }

    internal class BoolHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public BoolHandlers()
        {
            AddOperation = INT32_AddOperation;
            SubtractOperation = BOOL_SubtractOperation;
            MultiplyOperation = BOOL_MultiplyOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }

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
    }

    internal class ByteHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public ByteHandlers()
        {
            AddOperation = BYTE_AddOperation;
            SubtractOperation = BYTE_SubtractOperation;
            MultiplyOperation = BYTE_MultiplyOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }


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
    }

    internal class UByteHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public UByteHandlers()
        {
            AddOperation = UBYTE_AddOperation;
            SubtractOperation = UBYTE_SubtractOperation;
            MultiplyOperation = UBYTE_MultiplyOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }

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
    }

    internal class Int16Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public Int16Handlers()
        {
            AddOperation = INT16_AddOperation;
            SubtractOperation = INT16_SubtractOperation;
            MultiplyOperation = INT16_MultiplyOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }

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
    }

    internal class UInt16Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public UInt16Handlers()
        {
            AddOperation = UINT16_AddOperation;
            SubtractOperation = UINT16_SubtractOperation;
            MultiplyOperation = UINT16_MultiplyOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }


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
    }

    internal class Int32Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public Int32Handlers()
        {
            AddOperation = INT32_AddOperation;
            SubtractOperation = INT32_SubtractOperation;
            MultiplyOperation = INT32_MultiplyOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }

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
    }

    internal class UInt32Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public UInt32Handlers()
        {
            AddOperation = UINT32_AddOperation;
            SubtractOperation = UINT32_SubtractOperation;
            MultiplyOperation = UINT32_MultiplyOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }

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
    }

    internal class Int64Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public Int64Handlers()
        {
            AddOperation = INT64_AddOperation;
            SubtractOperation = INT64_SubtractOperation;
            MultiplyOperation = INT64_MultiplyOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }

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

    }

    internal class UInt64Handlers : ArrayHandlerBase, IArrayHandlers
    {
        public UInt64Handlers()
        {
            AddOperation = UINT64_AddOperation;
            SubtractOperation = UINT64_SubtractOperation;
            MultiplyOperation = UINT64_MultiplyOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }

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
    }

    internal class FloatHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public FloatHandlers()
        {
            AddOperation = Float_AddOperation;
            SubtractOperation = FLOAT_SubtractOperation;
            MultiplyOperation = FLOAT_MultiplyOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }

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
    }

    internal class DoubleHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public DoubleHandlers()
        {
            AddOperation = Double_AddOperation;
            SubtractOperation = DOUBLE_SubtractOperation;
            MultiplyOperation = DOUBLE_MultiplyOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }

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
    }

    internal class DecimalHandlers : ArrayHandlerBase, IArrayHandlers
    {
        public DecimalHandlers()
        {
            AddOperation = Decimal_AddOperation;
            SubtractOperation = DECIMAL_SubtractOperation;
            MultiplyOperation = DECIMAL_MultiplyOperation;
        }

        public NumericOperation AddOperation { get; set; }
        public NumericOperation SubtractOperation { get; set; }
        public NumericOperation MultiplyOperation { get; set; }

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
    }
}
