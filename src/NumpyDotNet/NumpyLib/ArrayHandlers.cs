using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Linq;
#if NPY_INTP_64
using npy_intp = System.Int64;
using npy_ucs4 = System.Int64;
#else
using npy_intp = System.Int32;
using npy_ucs4 = System.Int32;
#endif

namespace NumpyLib
{
    internal interface IArrayHandlers
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

        System.Array ToArray(Array ssrc);
        int ItemSize { get; }
        int ItemDiv { get; }
        Type ItemType { get; }
        int GetLength(VoidPtr vp);
        object AllocateNewArray(int size);
        object AllocateAndCopy(object datap, int startingOffset, int numElements);
        void SortArray(VoidPtr data, int offset, int length);
        int CompareTo(object invalue, object comparevalue);
        npy_intp ArgMax(object ip, npy_intp startIndex, npy_intp endIndex);
        npy_intp ArgMin(object ip, npy_intp startIndex, npy_intp endIndex);
        bool IsNan(object o);
        bool IsInfinity(object o);
        object ConvertToUpgradedValue(object o);
        VoidPtr GetArrayCopy(VoidPtr vp);
        void ArrayFill(VoidPtr vp, object FillValue);
        int ArrayFill(VoidPtr dest, VoidPtr scalar, int length, int dest_offset, int fill_offset);
        NPY_TYPES MathOpReturnType(UFuncOperation Operation);
        NPY_TYPES MathOpFloatingType(UFuncOperation Operation);
        object MathOpConvertOperand(object operValue);
        object GetArgSortMinValue();
        object GetArgSortMaxValue();
        object GetPositiveInfinity();
        object GetNegativeInfinity();
        object GetNaN();
        bool NonZero(VoidPtr vp, npy_intp index);
        object GetItem(VoidPtr vp, npy_intp index);
        object GetIndex(VoidPtr vp, npy_intp index);
        object GetItemDifferentType(VoidPtr vp, npy_intp index, NPY_TYPES ItemType, int ItemSize);
        int SetItem(VoidPtr vp, npy_intp index, object value);
        int SetIndex(VoidPtr obj, npy_intp index, object invalue);
        int SetItemDifferentType(VoidPtr vp, npy_intp index, object value);
    }

    internal delegate object NumericOperation(object bValue, object operand);
    internal delegate object NumericConversion(object operand);

    internal class DefaultArrayHandlers
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
            SetArrayHandler(NPY_TYPES.NPY_COMPLEXREAL, new DoubleHandlers());
            SetArrayHandler(NPY_TYPES.NPY_COMPLEXIMAG, new DoubleHandlers());
            SetArrayHandler(NPY_TYPES.NPY_BIGINT, new BigIntHandlers());
            SetArrayHandler(NPY_TYPES.NPY_OBJECT, new ObjectHandlers());
            SetArrayHandler(NPY_TYPES.NPY_STRING, new StringHandlers());

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

        public static NPY_TYPES GetArrayType(object o)
        {
            Type otype = o.GetType();
            for (int index = 0; index < ArrayHandlers.Length; index++)
            {
                var ah = ArrayHandlers[index];
                if (ah != null)
                {
                    if (ah.ItemType == otype)
                    {
                        return (NPY_TYPES)index;
                    }
                }
    
            }

            return NPY_TYPES.NPY_NOTSET;
        }

        private static IArrayHandlers[] ArrayHandlers = new IArrayHandlers[255];

    }

    internal interface INumericOperationsHelper
    {
        object GetItem(npy_intp offset);
        void SetItem(npy_intp offset, object Item);
    }

    #region NumericOperationsHelper
    internal abstract class NumericOperationsHelper<T>
    {
        protected NpyArray srcArray;
        protected T[] Array;
        protected int ItemDiv;
        protected bool ArraySameType;

        public NumericOperationsHelper(NpyArray srcArray)
        {
            this.srcArray = srcArray;
            this.ItemDiv = srcArray.ItemDiv;
            this.Array = srcArray.data.datap as T[];
            this.ArraySameType = srcArray.ItemType == srcArray.data.type_num;
        }

        virtual public object GetItem(npy_intp offset)
        {
            if (ArraySameType)
            {
                return Array[offset >> ItemDiv];
            }
            else
            {
                return numpyinternal.DifferentSizes_GetItemFunc(offset, srcArray);
            }
        }

        virtual public void SetItem(npy_intp offset, dynamic Item)
        {
            if (ArraySameType)
            {
                Array[offset >> ItemDiv] = (T)Item;
            }
            else
            {
                numpyinternal.DifferentSizes_SetItemFunc(offset, Item, srcArray);
            }

        }
    }
    internal class NumericOperationsHelper_BOOL : NumericOperationsHelper<bool>, INumericOperationsHelper
    {
        public NumericOperationsHelper_BOOL(NpyArray srcArray) : base(srcArray)
        {
       
        }
    }
    internal class NumericOperationsHelper_SBYTE : NumericOperationsHelper<sbyte>, INumericOperationsHelper
    {
        public NumericOperationsHelper_SBYTE(NpyArray srcArray) : base(srcArray)
        {

        }
    }
    internal class NumericOperationsHelper_UBYTE : NumericOperationsHelper<byte>, INumericOperationsHelper
    {
        public NumericOperationsHelper_UBYTE(NpyArray srcArray) : base(srcArray)
        {

        }
    }
  
    internal class NumericOperationsHelper_Int16 : NumericOperationsHelper<Int16>, INumericOperationsHelper
    {
        public NumericOperationsHelper_Int16(NpyArray srcArray) : base(srcArray)
        {

        }
    }
    internal class NumericOperationsHelper_UInt16 : NumericOperationsHelper<UInt16>, INumericOperationsHelper
    {
        public NumericOperationsHelper_UInt16(NpyArray srcArray) : base(srcArray)
        {

        }
    }
    internal class NumericOperationsHelper_Int32 : NumericOperationsHelper<Int32>, INumericOperationsHelper
    {
        public NumericOperationsHelper_Int32(NpyArray srcArray) : base(srcArray)
        {

        }
    }
    internal class NumericOperationsHelper_UInt32 : NumericOperationsHelper<UInt32>, INumericOperationsHelper
    {
        public NumericOperationsHelper_UInt32(NpyArray srcArray) : base(srcArray)
        {

        }
    }
    internal class NumericOperationsHelper_Int64 : NumericOperationsHelper<Int64>, INumericOperationsHelper
    {
        public NumericOperationsHelper_Int64(NpyArray srcArray) : base(srcArray)
        {

        }
    }
    internal class NumericOperationsHelper_UInt64 : NumericOperationsHelper<UInt64>, INumericOperationsHelper
    {
        public NumericOperationsHelper_UInt64(NpyArray srcArray) : base(srcArray)
        {

        }
    }
    internal class NumericOperationsHelper_FLOAT : NumericOperationsHelper<float>, INumericOperationsHelper
    {
        public NumericOperationsHelper_FLOAT(NpyArray srcArray) : base(srcArray)
        {

        }
    }
    internal class NumericOperationsHelper_DOUBLE : NumericOperationsHelper<double>, INumericOperationsHelper
    {
        public NumericOperationsHelper_DOUBLE(NpyArray srcArray) : base(srcArray)
        {

        }
    }
    internal class NumericOperationsHelper_DECIMAL : NumericOperationsHelper<decimal>, INumericOperationsHelper
    {
        public NumericOperationsHelper_DECIMAL(NpyArray srcArray) : base(srcArray)
        {

        }
    }
    internal class NumericOperationsHelper_COMPLEX : NumericOperationsHelper<System.Numerics.Complex>, INumericOperationsHelper
    {
        public NumericOperationsHelper_COMPLEX(NpyArray srcArray) : base(srcArray)
        {

        }
    }
    internal class NumericOperationsHelper_BIGINT : NumericOperationsHelper<System.Numerics.BigInteger>, INumericOperationsHelper
    {
        public NumericOperationsHelper_BIGINT(NpyArray srcArray) : base(srcArray)
        {

        }
    }
    internal class NumericOperationsHelper_OBJECT : NumericOperationsHelper<System.Object>, INumericOperationsHelper
    {
        public NumericOperationsHelper_OBJECT(NpyArray srcArray) : base(srcArray)
        {

        }
    }
    internal class NumericOperationsHelper_STRING : NumericOperationsHelper<System.String>, INumericOperationsHelper
    {
        public NumericOperationsHelper_STRING(NpyArray srcArray) : base(srcArray)
        {

        }

        override public void SetItem(npy_intp offset, dynamic Item)
        {
            if (ArraySameType)
            {
                Array[offset >> ItemDiv] = Item != null ? Item.ToString() : null;
            }
            else
            {
                return;
            }

        }
    }
    #endregion

    internal class NumericOperations
    {
        public NumericOperations()
        {

        }

        private INumericOperationsHelper srcHelper;
        private INumericOperationsHelper destHelper;
        private INumericOperationsHelper operHelper;

        public NumericConversion _ConvertOperand;

        public UFuncOperation operationType;
        public NumericOperation operation;
        public NPY_TYPES destItemType;
        public bool destTypeIsFloat;

        private bool IsSrcAndOperandSameType = false;

        public object srcGetItem(npy_intp offset)
        {
            return srcHelper.GetItem(offset);
        }
        public object operandGetItem(npy_intp offset)
        {
            return operHelper.GetItem(offset);
        }
        public void destSetItem(npy_intp offset, object Item)
        {
            destHelper.SetItem(offset, Item);
        }
        public object ConvertOperand(object Operand)
        {
            if (!destTypeIsFloat && IsSrcAndOperandSameType)
            {
                return Operand;
            }

            return _ConvertOperand(Operand);
        }

        public static NumericOperations GetOperations(UFuncOperation operationType, NumericOperation operation, NpyArray srcArray, NpyArray destArray, NpyArray operandArray)
        {
            NumericOperations operations = new NumericOperations();
            operations.operationType = operationType;
            operations.operation = operation;

            if (srcArray != null)
            {
                operations.srcHelper = GetNumericOperationsHelper(srcArray);
                operations._ConvertOperand = DefaultArrayHandlers.GetArrayHandler(srcArray.ItemType).MathOpConvertOperand;
            }

            if (destArray != null)
            {
                operations.destHelper = GetNumericOperationsHelper(destArray);
                operations.destItemType = destArray.ItemType;
                operations.destTypeIsFloat = numpyinternal.NpyTypeNum_ISFLOAT(destArray.ItemType);
            }

            if (operandArray != null)
            {
                operations.operHelper = GetNumericOperationsHelper(operandArray);
                operations.IsSrcAndOperandSameType = srcArray.ItemType == operandArray.ItemType;
            }


            return operations;
        }

        private static INumericOperationsHelper GetNumericOperationsHelper(NpyArray Array)
        {
            switch (Array.ItemType)
            {
                case NPY_TYPES.NPY_BOOL:
                    return new NumericOperationsHelper_BOOL(Array);

                case NPY_TYPES.NPY_BYTE:
                    return new NumericOperationsHelper_SBYTE(Array);

                case NPY_TYPES.NPY_UBYTE:
                    return new NumericOperationsHelper_UBYTE(Array);

                case NPY_TYPES.NPY_INT16:
                    return new NumericOperationsHelper_Int16(Array);

                case NPY_TYPES.NPY_UINT16:
                    return new NumericOperationsHelper_UInt16(Array);

                case NPY_TYPES.NPY_INT32:
                    return new NumericOperationsHelper_Int32(Array);

                case NPY_TYPES.NPY_UINT32:
                    return new NumericOperationsHelper_UInt32(Array);

                case NPY_TYPES.NPY_INT64:
                    return new NumericOperationsHelper_Int64(Array);

                case NPY_TYPES.NPY_UINT64:
                    return new NumericOperationsHelper_UInt64(Array);

                case NPY_TYPES.NPY_FLOAT:
                    return new NumericOperationsHelper_FLOAT(Array);

                case NPY_TYPES.NPY_DOUBLE:
                    return new NumericOperationsHelper_DOUBLE(Array);

                case NPY_TYPES.NPY_DECIMAL:
                    return new NumericOperationsHelper_DECIMAL(Array);

                case NPY_TYPES.NPY_COMPLEX:
                    return new NumericOperationsHelper_COMPLEX(Array);

                case NPY_TYPES.NPY_BIGINT:
                    return new NumericOperationsHelper_BIGINT(Array);

                case NPY_TYPES.NPY_OBJECT:
                    return new NumericOperationsHelper_OBJECT(Array);

                case NPY_TYPES.NPY_STRING:
                    return new NumericOperationsHelper_STRING(Array);

                default:
                    return null;
            }
        }

    }

}
