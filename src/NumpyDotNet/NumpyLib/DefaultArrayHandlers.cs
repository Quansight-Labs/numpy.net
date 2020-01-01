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
        object ConvertToUpgradedValue(object o);
        VoidPtr GetArrayCopy(VoidPtr vp);
        void ArrayFill(VoidPtr vp, object FillValue);
        NPY_TYPES MathOpReturnType(NpyArray_Ops Operation);
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
            SetArrayHandler(NPY_TYPES.NPY_OBJECT, new ObjectHandlers());
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

    internal class ArrayHandlerBase<T>
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

        public virtual int ItemSize {get;}

        protected T[] _t = new T[1] { default(T) };
        public Type ItemType
        {
            get
            {
                return _t[0].GetType();
            }
        }

        public System.Array ToArray(Array ssrc)
        {
            return ssrc.Cast<T>().ToArray();
        }
        public int GetLength(VoidPtr vp)
        {
            var dbool = vp.datap as T[];
            return dbool.Length;
        }
        public object AllocateNewArray(int size)
        {
            return new T[size];
        }
        public object AllocateAndCopy(object datap, int startingOffset, int numElements)
        {
            T[] data = new T[numElements];
            Array.Copy(datap as T[], startingOffset, data, 0, numElements);
            return data;
        }
        public virtual void SortArray(VoidPtr data, int offset, int length)
        {
            var arr = data.datap as T[];
            Array.Sort(arr, offset, length);
        }
        public virtual int CompareTo(dynamic invalue, dynamic comparevalue)
        {
            if (invalue is IComparable && comparevalue is IComparable)
            {
                if (invalue == comparevalue)
                    return 0;
                if (invalue < comparevalue)
                    return -1;
                return 1;

                //IComparable inx = (IComparable)invalue;
               //return inx.CompareTo(comparevalue);
            }

            return 0;
        }
        public virtual npy_intp ArgMax(object oip, npy_intp startIndex, npy_intp endIndex)
        {
            T[] ip = oip as T[];
            T mp = ip[0 + startIndex];

            npy_intp max_ind = 0;
            for (npy_intp i = 1 + startIndex; i < endIndex + startIndex; i++)
            {
                if (CompareTo(ip[i], mp) > 0)
                {
                    mp = ip[i];
                    max_ind = i - startIndex;
                }
            }
            return max_ind;
        }
        public virtual npy_intp ArgMin(object oip, npy_intp startIndex, npy_intp endIndex)
        {
            T[] ip = oip as T[];
            T mp = ip[0 + startIndex];

            npy_intp max_ind = 0;
            for (npy_intp i = 1 + startIndex; i < endIndex + startIndex; i++)
            {
                if (CompareTo(ip[i], mp) < 0)
                {
                    mp = ip[i];
                    max_ind = i - startIndex;
                }
            }
            return max_ind;
        }
        public virtual void ArrayFill(VoidPtr vp, object FillValue)
        {
            var adata = vp.datap as object[];
            Fill(adata, Convert.ToBoolean(FillValue), 0, adata.Length);
            return;
        }
        protected void Fill<T>(T[] array, T fillValue, int startIndex, int count)
        {
            for (int i = startIndex; i < count; i++)
            {
                array[i] = fillValue;
            }
        }


        public virtual bool IsNan(object o)
        {
            return false;
        }
        public virtual bool IsInfinity(object o)
        {
            return false;
        }
        public virtual object ConvertToUpgradedValue(object o)
        {
            return Convert.ToDouble(o);
        }
        public virtual VoidPtr GetArrayCopy(VoidPtr vp)
        {
            var src = vp.datap as T[];

            var copy = new T[src.Length];
            Array.Copy(src, copy, src.Length);
            return new VoidPtr(src, vp.type_num);
        }




        public void dot(VoidPtr _ip1, npy_intp is1, VoidPtr _ip2, npy_intp is2, VoidPtr _op, npy_intp n)
        {
            T tmp = default(T);
            npy_intp i;

            T[] ip1 = _ip1.datap as T[];
            T[] ip2 = _ip2.datap as T[];
            T[] op = _op.datap as T[];

            npy_intp ip1_index = _ip1.data_offset;
            npy_intp ip2_index = _ip2.data_offset;

            npy_intp ip1Size = DefaultArrayHandlers.GetArrayHandler(_ip1.type_num).ItemSize;
            npy_intp ip2Size = DefaultArrayHandlers.GetArrayHandler(_ip2.type_num).ItemSize;
            npy_intp opSize = DefaultArrayHandlers.GetArrayHandler(_op.type_num).ItemSize;

            for (i = 0; i < n; i++, ip1_index += is1, ip2_index += is2)
            {
                tmp = (T)T_dot(tmp, ip1, ip2, ip1_index, ip2_index, ip1Size, ip2Size);
            }
            op[_op.data_offset / opSize] = tmp;
        }
        protected virtual object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            System.Numerics.Complex tmp = (System.Numerics.Complex)otmp;
            System.Numerics.Complex[] ip1 = op1 as System.Numerics.Complex[];
            System.Numerics.Complex[] ip2 = op2 as System.Numerics.Complex[];

            tmp += (ip1[ip1_index / ip1Size] * ip2[ip2_index / ip2Size]);
            return tmp;
        }

        public virtual NPY_TYPES MathOpReturnType(NpyArray_Ops Operation)
        {
            switch (Operation)
            {
                case NpyArray_Ops.npy_op_sqrt:
                {
                    if (ItemSize > 4)
                        return NPY_TYPES.NPY_DOUBLE;
                    else
                        return NPY_TYPES.NPY_FLOAT;
                }
            }
            return NPY_TYPES.NPY_DOUBLE;
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

    internal class ObjectHandlers : ArrayHandlerBase<object>, IArrayHandlers
    {
        public ObjectHandlers()
        {
            _t = new object[1] { new object() };
        }

        public int ItemSize
        {
            get { return IntPtr.Size; }
        }

        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            var adata = vp.datap as object[];
            Fill(adata, FillValue, 0, adata.Length);
            return;
        }

        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            throw new NotImplementedException("This array handler does not implement DOT");
            return otmp;
        }

    }

    internal class BoolHandlers : ArrayHandlerBase<bool>, IArrayHandlers
    {
        public BoolHandlers()
        {
        }

        public override int ItemSize
        {
            get { return sizeof(bool); }
        }
        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            var adata = vp.datap as bool[];
            Fill(adata, Convert.ToBoolean(FillValue), 0, adata.Length);
            return;
        }


        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            bool tmp = (bool)otmp;
            bool[] ip1 = op1 as bool[];
            bool[] ip2 = op2 as bool[];

            if ((ip1[ip1_index / ip1Size] == true) && (ip2[ip2_index / ip2Size] == true))
            {
                tmp = true;
                return tmp;
            }
            return tmp;
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

    internal class ByteHandlers : ArrayHandlerBase<sbyte>, IArrayHandlers
    {
        public ByteHandlers()
        {
  
        }
 
        public override int ItemSize
        {
            get { return sizeof(sbyte); }
        }
        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            sbyte[] adata = vp.datap as sbyte[];
            Fill(adata, Convert.ToSByte(FillValue), 0, adata.Length);
            return;
        }

        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            sbyte tmp = (sbyte)otmp;
            sbyte[] ip1 = op1 as sbyte[];
            sbyte[] ip2 = op2 as sbyte[];

            tmp += (sbyte)((sbyte)ip1[ip1_index / ip1Size] * (sbyte)ip2[ip2_index / ip2Size]);
            return tmp;
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

    internal class UByteHandlers : ArrayHandlerBase<byte>, IArrayHandlers
    {
        public UByteHandlers()
        {

        }

        public override int ItemSize
        {
            get { return sizeof(byte); }
        }
        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            byte[] adata = vp.datap as byte[];
            Fill(adata, Convert.ToByte(FillValue), 0, adata.Length);
            return;
        }


        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            byte tmp = (byte)otmp;
            byte[] ip1 = op1 as byte[];
            byte[] ip2 = op2 as byte[];

            tmp += (byte)((byte)ip1[ip1_index / ip1Size] * (byte)ip2[ip2_index / ip2Size]);
            return tmp;
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

    internal class Int16Handlers : ArrayHandlerBase<Int16>, IArrayHandlers
    {
        public Int16Handlers()
        {
 
        }

        public override int ItemSize
        {
            get { return sizeof(Int16); }
        }
        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            Int16[] adata = vp.datap as Int16[];
            Fill(adata, Convert.ToInt16(FillValue), 0, adata.Length);
            return;
        }

        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            Int16 tmp = (Int16)otmp;
            Int16[] ip1 = op1 as Int16[];
            Int16[] ip2 = op2 as Int16[];

            tmp += (Int16)((Int16)ip1[ip1_index / ip1Size] * (Int16)ip2[ip2_index / ip2Size]);
            return tmp;
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

    internal class UInt16Handlers : ArrayHandlerBase<UInt16>, IArrayHandlers
    {
        public UInt16Handlers()
        {
 
        }

        public override int ItemSize
        {
            get { return sizeof(UInt16); }
        }
        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            UInt16[] adata = vp.datap as UInt16[];
            Fill(adata, Convert.ToUInt16(FillValue), 0, adata.Length);
            return;
        }

        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            UInt16 tmp = (UInt16)otmp;
            UInt16[] ip1 = op1 as UInt16[];
            UInt16[] ip2 = op2 as UInt16[];

            tmp += (UInt16)((UInt16)ip1[ip1_index / ip1Size] * (UInt16)ip2[ip2_index / ip2Size]);
            return tmp;
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

    internal class Int32Handlers : ArrayHandlerBase<Int32>, IArrayHandlers
    {
        public Int32Handlers()
        {
  
        }
        public override int ItemSize
        {
            get { return sizeof(Int32); }
        }
        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            Int32[] adata = vp.datap as Int32[];
            Fill(adata, Convert.ToInt32(FillValue), 0, adata.Length);
            return;
        }

        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            Int32 tmp = (Int32)otmp;
            Int32[] ip1 = op1 as Int32[];
            Int32[] ip2 = op2 as Int32[];

            tmp += (Int32)((Int32)ip1[ip1_index / ip1Size] * (Int32)ip2[ip2_index / ip2Size]);
            return tmp;
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

    internal class UInt32Handlers : ArrayHandlerBase<UInt32>, IArrayHandlers
    {
        public UInt32Handlers()
        {
 
        }

        public override int ItemSize
        {
            get { return sizeof(UInt32); }
        }
        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            UInt32[] adata = vp.datap as UInt32[];
            Fill(adata, Convert.ToUInt32(FillValue), 0, adata.Length);
            return;
        }

        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            UInt32 tmp = (UInt32)otmp;
            UInt32[] ip1 = op1 as UInt32[];
            UInt32[] ip2 = op2 as UInt32[];

            tmp += (UInt32)((UInt32)ip1[ip1_index / ip1Size] * (UInt32)ip2[ip2_index / ip2Size]);
            return tmp;
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

    internal class Int64Handlers : ArrayHandlerBase<Int64>, IArrayHandlers
    {
        public Int64Handlers()
        {

        }
        public override int ItemSize
        {
            get { return sizeof(Int64); }
        }
        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            Int64[] adata = vp.datap as Int64[];
            Fill(adata, Convert.ToInt64(FillValue), 0, adata.Length);
            return;
        }
        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            Int64 tmp = (Int64)otmp;
            Int64[] ip1 = op1 as Int64[];
            Int64[] ip2 = op2 as Int64[];

            tmp += (Int64)((Int64)ip1[ip1_index / ip1Size] * (Int64)ip2[ip2_index / ip2Size]);
            return tmp;
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

    internal class UInt64Handlers : ArrayHandlerBase<UInt64>, IArrayHandlers
    {
        public UInt64Handlers()
        {
   
        }
        public override int ItemSize
        {
            get { return sizeof(UInt64); }
        }
        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            UInt64[] adata = vp.datap as UInt64[];
            Fill(adata, Convert.ToUInt64(FillValue), 0, adata.Length);
            return;
        }
        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            UInt64 tmp = (UInt64)otmp;
            UInt64[] ip1 = op1 as UInt64[];
            UInt64[] ip2 = op2 as UInt64[];

            tmp += (UInt64)((UInt64)ip1[ip1_index / ip1Size] * (UInt64)ip2[ip2_index / ip2Size]);
            return tmp;
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

    internal class FloatHandlers : ArrayHandlerBase<float>, IArrayHandlers
    {
        public FloatHandlers()
        {
 
        }
        public override int ItemSize
        {
            get { return sizeof(float); }
        }
        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            float[] adata = vp.datap as float[];
            Fill(adata, Convert.ToSingle(FillValue), 0, adata.Length);
            return;
        }
        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            float tmp = (float)otmp;
            float[] ip1 = op1 as float[];
            float[] ip2 = op2 as float[];

            tmp += (float)((float)ip1[ip1_index / ip1Size] * (float)ip2[ip2_index / ip2Size]);
            return tmp;
        }
        public override bool IsNan(object o)
        {
            float f = (float)o;
            return float.IsNaN(f);
        }
        public override bool IsInfinity(object o)
        {
            float f = (float)o;
            return float.IsInfinity(f);
        }
        public override object ConvertToUpgradedValue(object o)
        {
            return Convert.ToSingle(o);
        }

        public override NPY_TYPES MathOpReturnType(NpyArray_Ops Operation)
        {
            switch (Operation)
            {
                case NpyArray_Ops.npy_op_power:
                    return NPY_TYPES.NPY_DOUBLE;
                case NpyArray_Ops.npy_op_true_divide:
                    return NPY_TYPES.NPY_DOUBLE;
                case NpyArray_Ops.npy_op_special_operand_is_float:
                    return NPY_TYPES.NPY_DOUBLE;

            }

            return NPY_TYPES.NPY_FLOAT;
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

    internal class DoubleHandlers : ArrayHandlerBase<double>, IArrayHandlers
    {
        public DoubleHandlers()
        {

        }
        public override int ItemSize
        {
            get { return sizeof(double); }
        }
        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            double[] adata = vp.datap as double[];
            Fill(adata, Convert.ToDouble(FillValue), 0, adata.Length);
            return;
        }
        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            double tmp = (double)otmp;
            double[] ip1 = op1 as double[];
            double[] ip2 = op2 as double[];

            tmp += (double)((double)ip1[ip1_index / ip1Size] * (double)ip2[ip2_index / ip2Size]);
            return tmp;
        }
        public override bool IsNan(object o)
        {
            double d = (double)o;
            return double.IsNaN(d);
        }
        public override bool IsInfinity(object o)
        {
            double d = (double)o;
            return double.IsInfinity(d);
        }
        public override object ConvertToUpgradedValue(object o)
        {
            return Convert.ToDouble(o);
        }

        public override NPY_TYPES MathOpReturnType(NpyArray_Ops Operation)
        {
            return NPY_TYPES.NPY_DOUBLE;
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

    internal class DecimalHandlers : ArrayHandlerBase<Decimal>, IArrayHandlers
    {
        public DecimalHandlers()
        {
 
        }

        public override int ItemSize
        {
            get { return sizeof(decimal); }
        }
        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            decimal[] adata = vp.datap as decimal[];
            Fill(adata, Convert.ToDecimal(FillValue), 0, adata.Length);
            return;
        }
        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            decimal tmp = (decimal)otmp;
            decimal[] ip1 = op1 as decimal[];
            decimal[] ip2 = op2 as decimal[];

            tmp += (decimal)((decimal)ip1[ip1_index / ip1Size] * (decimal)ip2[ip2_index / ip2Size]);
            return tmp;
        }
        public override object ConvertToUpgradedValue(object o)
        {
            return Convert.ToDecimal(o);
        }

        public override NPY_TYPES MathOpReturnType(NpyArray_Ops Operation)
        {
            return NPY_TYPES.NPY_DECIMAL;
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

    internal class ComplexHandlers : ArrayHandlerBase<System.Numerics.Complex>, IArrayHandlers
    {
        public ComplexHandlers()
        {

        }

        public override int ItemSize
        {
            get { return sizeof(double) * 2; }
        }
        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            System.Numerics.Complex[] adata = vp.datap as System.Numerics.Complex[];

            if (FillValue is System.Numerics.Complex)
                Fill(adata, (System.Numerics.Complex)FillValue, 0, adata.Length);
            else
                Fill(adata, new System.Numerics.Complex(Convert.ToDouble(FillValue), 0), 0, adata.Length);

            return;
        }
        protected override object T_dot(object otmp, object op1, object op2, npy_intp ip1_index, npy_intp ip2_index, npy_intp ip1Size, npy_intp ip2Size)
        {
            var tmp = (System.Numerics.Complex)otmp;
            var ip1 = op1 as System.Numerics.Complex[];
            var ip2 = op2 as System.Numerics.Complex[];

            tmp += (System.Numerics.Complex)((System.Numerics.Complex)ip1[ip1_index / ip1Size] * (System.Numerics.Complex)ip2[ip2_index / ip2Size]);
            return tmp;
        }
        public override object ConvertToUpgradedValue(object o)
        {
            return ConvertToComplex(o);
        }
        public override NPY_TYPES MathOpReturnType(NpyArray_Ops Operation)
        {
            return NPY_TYPES.NPY_COMPLEX;
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
            return Complex.Pow(dValue, ConvertToComplex(operand));
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
            return Complex.Negate((Complex)bValue);
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
            Complex cValue = (Complex)bValue;
            Complex oValue = (Complex)operand;

            UInt64 rValue = (UInt64)cValue.Real;
            rValue = rValue << Convert.ToInt32(oValue.Real);

            UInt64 iValue = (UInt64)cValue.Imaginary;
            iValue = iValue << Convert.ToInt32(oValue.Imaginary);

            return new Complex((double)rValue, (double)iValue);
        }
        protected override object RightShift(object bValue, object operand)
        {
            Complex cValue = (Complex)bValue;
            Complex oValue = (Complex)operand;

            UInt64 rValue = (UInt64)cValue.Real;
            rValue = rValue >> Convert.ToInt32(oValue.Real);

            UInt64 iValue = (UInt64)cValue.Imaginary;
            iValue = iValue >> Convert.ToInt32(oValue.Imaginary);

            return new Complex((double)rValue, (double)iValue);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            Complex cValue = (Complex)bValue;
            Complex oValue = (Complex)operand;

            UInt64 rValue = (UInt64)cValue.Real;
            rValue = rValue & Convert.ToUInt64(oValue.Real);

            UInt64 iValue = (UInt64)cValue.Imaginary;
            iValue = iValue & Convert.ToUInt64(oValue.Imaginary);

            return new Complex((double)rValue, (double)iValue);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            Complex cValue = (Complex)bValue;
            Complex oValue = (Complex)operand;

            UInt64 rValue = (UInt64)cValue.Real;
            rValue = rValue ^ Convert.ToUInt64(oValue.Real);

            UInt64 iValue = (UInt64)cValue.Imaginary;
            iValue = iValue ^ Convert.ToUInt64(oValue.Imaginary);

            return new Complex((double)rValue, (double)iValue);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            Complex cValue = (Complex)bValue;
            Complex oValue = (Complex)operand;

            UInt64 rValue = (UInt64)cValue.Real;
            rValue = rValue | Convert.ToUInt64(oValue.Real);

            UInt64 iValue = (UInt64)cValue.Imaginary;
            iValue = iValue | Convert.ToUInt64(oValue.Imaginary);

            return new Complex((double)rValue, (double)iValue);
        }
        protected override object Less(object bValue, object operand)
        {
            Complex cOperand = (Complex)operand;
            Complex dValue = (Complex)bValue;

            if (cOperand.Imaginary == 0)
            {
                return dValue.Real < cOperand.Real;
            }
            return false;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            Complex cOperand = (Complex)operand;
            Complex dValue = (Complex)bValue;

            if (cOperand.Imaginary == 0)
            {
                return dValue.Real <= cOperand.Real;
            }
            return false;
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
            Complex cOperand = (Complex)operand;
            Complex dValue = (Complex)bValue;

            if (cOperand.Imaginary == 0)
            {
                return dValue.Real > cOperand.Real;
            }
            return false;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            Complex cOperand = (Complex)operand;
            Complex dValue = (Complex)bValue;

            if (cOperand.Imaginary == 0)
            {
                return dValue.Real >= cOperand.Real;
            }
            return false;
        }
        protected override object IsNAN(object bValue, object operand)
        {
            return false;
        }
        protected override object FloorDivide(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            Complex oValue = (Complex)operand;
            if (oValue == 0)
            {
                dValue = 0;
                return dValue;
            }

            double Real = 0;
            if (oValue.Real != 0)
                Real = Math.Floor(dValue.Real / oValue.Real);

            double Imaginary = 0;
            if (oValue.Imaginary != 0)
                Imaginary = Math.Floor(dValue.Imaginary / oValue.Imaginary);

            return new Complex(Real, Imaginary);
        }
        protected override object Floor(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return new Complex(Math.Floor(dValue.Real), Math.Floor(dValue.Imaginary));
        }
        protected override object Ceiling(object bValue, object operand)
        {
            Complex dValue = (Complex)bValue;
            return new Complex(Math.Ceiling(dValue.Real), Math.Ceiling(dValue.Imaginary));
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

            return new Complex(Math.Min(a.Real, b.Real), 0);

        }
        protected override object Rint(object bValue, object operand)
        {
            Complex a = (Complex)bValue;
            Complex b = (Complex)operand;

            return new Complex(Math.Round(a.Real), Math.Round(a.Imaginary));
        }
        protected override object Heaviside(object bValue, object operand)
        {
            Complex x = ConvertToComplex(bValue);

    
            if (x == 0.0)
                return ConvertToComplex(operand);

            if (x.Real < 0.0)
                return 0.0;

            return 1.0;

        }
        public override void SortArray(VoidPtr data, int offset, int length)
        {
            var arr = data.datap as System.Numerics.Complex[];
            Quick_Sort(arr, offset, offset + length-1);
        }
        static void Quick_Sort(System.Numerics.Complex[] array, int low, int high)
        {
            if (low < high)
            {
                int partitionIndex = Partition(array, low, high);

                //3. Recursively continue sorting the array
                Quick_Sort(array, low, partitionIndex - 1);
                Quick_Sort(array, partitionIndex + 1, high);
            }
        }
        static int Partition(System.Numerics.Complex[] array, int low, int high)
        {
            //1. Select a pivot point.
            double pivot = array[high].Real;

            int lowIndex = (low - 1);

            //2. Reorder the collection.
            for (int j = low; j < high; j++)
            {
                if (array[j].Real <= pivot)
                {
                    lowIndex++;

                    var temp = array[lowIndex];
                    array[lowIndex] = array[j];
                    array[j] = temp;
                }
            }

            var temp1 = array[lowIndex + 1];
            array[lowIndex + 1] = array[high];
            array[high] = temp1;

            return lowIndex + 1;
        }

        public override int CompareTo(object invalue, object comparevalue)
        {
            if (invalue is System.Numerics.Complex)
            {
                System.Numerics.Complex cin = (System.Numerics.Complex)invalue;

                if (comparevalue is System.Numerics.Complex)
                {
                    System.Numerics.Complex ccin = (System.Numerics.Complex)comparevalue;
                    return cin.Real.CompareTo(ccin.Real);
                }
                if (comparevalue is IComparable)
                {
                    return cin.Real.CompareTo(comparevalue);
                }
            }

            return 0;
        }
    }
}
