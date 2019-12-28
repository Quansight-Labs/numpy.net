using System;
using System.Collections.Generic;
using System.Text;

namespace NumpyLib
{
    public interface IArrayHandlers
    {
        NumericOperation AddOperation { get; set; }
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
            if (index >= ArrayHandlers.Length )
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

    internal class BoolHandlers : IArrayHandlers
    {
        public BoolHandlers()
        {
            AddOperation = INT32_AddOperation;
        }

        public NumericOperation AddOperation { get; set; }

        private object INT32_AddOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue + (double)operand;
        }
    }

    internal class ByteHandlers : IArrayHandlers
    {
        public ByteHandlers()
        {
            AddOperation = BYTE_AddOperation;
        }

        public NumericOperation AddOperation { get; set; }


        private object BYTE_AddOperation(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue + (double)operand;
        }
    }

    internal class UByteHandlers : IArrayHandlers
    {
        public UByteHandlers()
        {
            AddOperation = UBYTE_AddOperation;
        }

        public NumericOperation AddOperation { get; set; }


        private object UBYTE_AddOperation(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue + (double)operand;
        }
    }

    internal class Int16Handlers : IArrayHandlers
    {
        public Int16Handlers()
        {
            AddOperation = INT16_AddOperation;
        }

        public NumericOperation AddOperation { get; set; }

        private static object INT16_AddOperation(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue + (double)operand;
        }
    }

    internal class UInt16Handlers : IArrayHandlers
    {
        public UInt16Handlers()
        {
            AddOperation = UINT16_AddOperation;
        }

        public NumericOperation AddOperation { get; set; }


        private static object UINT16_AddOperation(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue + (double)operand;
        }
    }

    internal class Int32Handlers : IArrayHandlers
    {
        public Int32Handlers()
        {
            AddOperation = INT32_AddOperation;
        }

        public NumericOperation AddOperation { get; set; }


        private object INT32_AddOperation(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue + (double)operand;
        }
    }

    internal class UInt32Handlers : IArrayHandlers
    {
        public UInt32Handlers()
        {
            AddOperation = UINT32_AddOperation;
        }

        public NumericOperation AddOperation { get; set; }


        private object UINT32_AddOperation(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue + (double)operand;
        }
    }

    internal class Int64Handlers : IArrayHandlers
    {
        public Int64Handlers()
        {
            AddOperation = INT64_AddOperation;
        }

        public NumericOperation AddOperation { get; set; }


        private object INT64_AddOperation(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue + (double)operand;
        }
    }

    internal class UInt64Handlers : IArrayHandlers
    {
        public UInt64Handlers()
        {
            AddOperation = UINT64_AddOperation;
        }

        public NumericOperation AddOperation { get; set; }

        private object UINT64_AddOperation(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue + (double)operand;
        }
    }

    internal class FloatHandlers : IArrayHandlers
    {
        public FloatHandlers()
        {
            AddOperation = Float_AddOperation;
        }

        public NumericOperation AddOperation { get; set; }


        private object Float_AddOperation(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue + (double)operand;
        }
    }

    internal class DoubleHandlers : IArrayHandlers
    {
        public DoubleHandlers()
        {
            AddOperation = Double_AddOperation;
        }

        public NumericOperation AddOperation { get; set; }


        private object Double_AddOperation(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue + (double)operand;
        }
    }

    internal class DecimalHandlers : IArrayHandlers
    {
        public DecimalHandlers()
        {
            AddOperation = Decimal_AddOperation;
        }

        public NumericOperation AddOperation { get; set; }


        private object Decimal_AddOperation(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue + (decimal)operand;
        }
    }
}
