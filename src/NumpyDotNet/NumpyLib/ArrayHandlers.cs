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

        System.Array ToArray(Array ssrc);
        int ItemSize { get; }
        Type ItemType { get; }
        int GetLength(VoidPtr vp);
        object AllocateNewArray(int size);
        object AllocateAndCopy(object datap, int startingOffset, int numElements);
        void dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n);
        void SortArray(VoidPtr data, int offset, int length);
        int CompareTo(object invalue, object comparevalue);
        npy_intp ArgMax(object ip, npy_intp startIndex, npy_intp endIndex);
        npy_intp ArgMin(object ip, npy_intp startIndex, npy_intp endIndex);
        bool IsNan(object o);
        bool IsInfinity(object o);
        bool IsInRange(object o, double low, double high);
        object ConvertToUpgradedValue(object o);
        VoidPtr GetArrayCopy(VoidPtr vp);
        void ArrayFill(VoidPtr vp, object FillValue);
        int ArrayFill(VoidPtr dest, VoidPtr scalar, int length, int dest_offset, int fill_offset);
        NPY_TYPES MathOpReturnType(NpyArray_Ops Operation);
        NPY_TYPES MathOpFloatingType(NpyArray_Ops Operation);
        object MathOpConvertOperand(object srcValue, object operValue);
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

    internal class NumericOperations
    {
        public NumericOperations()
        {

        }
        public NpyArray_GetItemFunc srcGetItem;
        public NpyArray_SetItemFunc srcSetItem;
        public NumericOperation ConvertOperand;

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
                operations.ConvertOperand = DefaultArrayHandlers.GetArrayHandler(srcArray.ItemType).MathOpConvertOperand;
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

}
