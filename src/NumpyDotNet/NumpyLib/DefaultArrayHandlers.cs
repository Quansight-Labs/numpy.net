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

            ItemDiv = numpyinternal.GetDivSize(ItemSize);
        }

        public virtual int ItemSize {get;}
        public virtual int ItemDiv { get; }

        protected T[] _t = new T[1] { default(T) };
        public Type ItemType
        {
            get
            {
                return _t[0].GetType();
            }
        }

        public virtual object GetItem(VoidPtr data, npy_intp index)
        {
            T[] dp = data.datap as T[];

            long AdjustedIndex = (data.data_offset + index) >> ItemDiv;

            if (AdjustedIndex < 0)
            {
                AdjustedIndex = dp.Length - -AdjustedIndex;
            }

            return dp[AdjustedIndex];
        }

        public virtual object GetIndex(VoidPtr data, npy_intp index)
        {
            index = AdjustNegativeIndex(data, index);

            T[] dp = data.datap as T[];
            return dp[index];
        }

        public virtual object GetItemDifferentType(VoidPtr vp, npy_intp index, NPY_TYPES ItemType, int ItemSize)
        {
            return 0;
        }


        public virtual int SetItem(VoidPtr data, npy_intp index, object value)
        {
            T[] dp = data.datap as T[];

            long AdjustedIndex = (data.data_offset + index) >> ItemDiv;
            if (AdjustedIndex < 0)
            {
                AdjustedIndex = dp.Length - -AdjustedIndex;
            }

            dp[AdjustedIndex] = (T)value;
            return 1;
        }
        public virtual int SetItemDifferentType(VoidPtr data, npy_intp index, object value)
        {
            throw new Exception(string.Format("Arrays of {0} are not programmed to set items of this type", data.type_num));
        }
        public virtual int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            T[] dp = data.datap as T[];
            dp[index] = (T)value;
            return 1;
        }
        protected npy_intp AdjustNegativeIndex(VoidPtr data, npy_intp index)
        {
            if (index < 0)
            {
                T[] dp = data.datap as T[];
                index = dp.Length - -index;
            }
            return index;
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
            Fill(adata, FillValue, 0, adata.Length);
            return;
        }
        protected void Fill<T>(T[] array, T fillValue, int startIndex, int count)
        {
            for (int i = startIndex; i < count; i++)
            {
                array[i] = fillValue;
            }
        }
        public virtual int ArrayFill(VoidPtr dest, VoidPtr scalar, int length, int dest_offset, int fill_offset)
        {
            T[] destp = dest.datap as T[];
            T[] scalarp = scalar.datap as T[];
            if (destp == null || scalarp == null)
                return -1;

            Fill<T>(destp, (T)(scalarp[fill_offset]), dest_offset, length + dest_offset);
            return 0;
        }


        public virtual object GetArgSortMinValue()
        {
            return double.MinValue;
        }
        public virtual object GetArgSortMaxValue()
        {
            return double.MaxValue;
        }
        public virtual object GetPositiveInfinity()
        {
            return double.PositiveInfinity;
        }
        public virtual object GetNegativeInfinity()
        {
            return double.NegativeInfinity;
        }
        public virtual object GetNaN()
        {
            return double.NaN;
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
            return new VoidPtr(copy, vp.type_num);
        }

        public virtual bool NonZero(VoidPtr vp, npy_intp index)
        {
            T[] bp = vp.datap as T[];
            return !(bp[index].Equals(0));
        }

        public virtual NPY_TYPES MathOpReturnType(UFuncOperation Operation)
        {
            switch (Operation)
            {
                case UFuncOperation.sqrt:
                {
                    if (ItemSize > 4)
                        return NPY_TYPES.NPY_DOUBLE;
                    else
                        return NPY_TYPES.NPY_FLOAT;
                }
            }
            return NPY_TYPES.NPY_DOUBLE;
        }

        public virtual NPY_TYPES MathOpFloatingType(UFuncOperation Operation)
        {
            return NPY_TYPES.NPY_DOUBLE;
        }

        public virtual object MathOpConvertOperand(object srcValue, object operValue)
        {
            if (operValue is System.Numerics.Complex)
            {
                System.Numerics.Complex c = (System.Numerics.Complex)operValue;
                return Convert.ToDouble(c.Real);
            }
            if (operValue is System.Numerics.BigInteger)
            {
                System.Numerics.BigInteger c = (System.Numerics.BigInteger)operValue;
                return c;
            }

            return Convert.ToDouble(operValue);
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
            if (dValue == 0)
                return 0;

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
            
            try
            {
                return Convert.ToBoolean(bValue) || Convert.ToBoolean(operand);
            }
            catch
            {
                return false;
            }
        }
        protected virtual object LogicalAnd(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            try
            {
                return Convert.ToBoolean(dValue) && Convert.ToBoolean(operand);
            }
            catch
            {
                return false;
            }
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

        public override int ItemSize
        {
            get { return IntPtr.Size; }
        }

        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            var adata = vp.datap as object[];
            Fill(adata, FillValue, 0, adata.Length);
            return;
        }
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            object[] dp = data.datap as object[];
            dp[index] = (object)value;
            return 1;
        }

        public override bool NonZero(VoidPtr vp, npy_intp index)
        {
            object[] bp = vp.datap as object[];
            return !(bp[index] == null);
        }

        public override object MathOpConvertOperand(object srcValue, object operValue)
        {
            return operValue;
        }
        public override NPY_TYPES MathOpFloatingType(UFuncOperation Operation)
        {
            return NPY_TYPES.NPY_OBJECT;
        }

        public override object ConvertToUpgradedValue(object o)
        {
            return o;
        }

        public override NPY_TYPES MathOpReturnType(UFuncOperation Operation)
        {
            return NPY_TYPES.NPY_DOUBLE;
        }

        protected override object Divide(dynamic bValue, dynamic operand)
        {
            return bValue / operand;
        }

        protected override object BitWiseAnd(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue & operand;
        }
        protected override object BitWiseXor(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue ^ operand;
        }
        protected override object BitWiseOr(dynamic bValue, dynamic operand)
        {
            dynamic dValue = bValue;
            return dValue | operand;
        }
        protected override object Remainder(dynamic bValue, dynamic operand)
        {
            return bValue % operand;
        }
        protected override object Floor(dynamic bValue, dynamic operand)
        {
            if (bValue is decimal)
            {
                return Math.Floor(Convert.ToDecimal(bValue));
            }
            return Math.Floor(Convert.ToDouble(bValue));
        }
        protected override object Ceiling(dynamic bValue, dynamic operand)
        {
            if (bValue is decimal)
            {
                return Math.Ceiling(Convert.ToDecimal(bValue));
            }
            return Math.Ceiling(Convert.ToDouble(bValue));
        }
        protected override object Maximum(dynamic bValue, dynamic operand)
        {
            if (bValue >= operand)
                return bValue;
            return operand;
        }
        protected override object FMax(dynamic bValue, dynamic operand)
        {
            if (bValue >= operand)
                return bValue;
            return operand;
        }
        protected override object Minimum(dynamic bValue, dynamic operand)
        {
            if (bValue <= operand)
                return bValue;
            return operand;
        }
        protected override object FMin(dynamic bValue, dynamic operand)
        {
            if (bValue <= operand)
                return bValue;
            return operand;
        }
        protected override object Rint(dynamic bValue, dynamic operand)
        {
            if (bValue is decimal)
            {
                return Math.Round(Convert.ToDecimal(bValue));
            }
            return Math.Round(Convert.ToDouble(bValue));
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
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var dbool = data.datap as bool[];
            dbool[index] = Convert.ToBoolean(value);
            return 1;
        }

        public override bool NonZero(VoidPtr vp, npy_intp index)
        {
            bool[] bp = vp.datap as bool[];
            return (bp[index]);
        }

        protected override object Add(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue + (dynamic)operand;
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
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var dsbyte = data.datap as sbyte[];
            dsbyte[index] = Convert.ToSByte(value);
            return 1;
        }

        protected override object Add(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue + (dynamic)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue - (dynamic)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue * (dynamic)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            if (dValue == 0)
                return 0;

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
            return dValue < (dynamic)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue <= (dynamic)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue == (dynamic)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue != (dynamic)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue > (dynamic)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            sbyte dValue = (sbyte)bValue;
            return dValue >= (dynamic)operand;
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
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var dbyte = data.datap as byte[];
            dbyte[index] = Convert.ToByte(value);
            return 1;
        }

        protected override object Add(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue + (dynamic)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue - (dynamic)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue * (dynamic)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            if (dValue == 0)
                return 0;

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
            return dValue < (dynamic)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue <= (dynamic)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue == (dynamic)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue != (dynamic)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue > (dynamic)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            byte dValue = (byte)bValue;
            return dValue >= (dynamic)operand;
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
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var dint16 = data.datap as Int16[];
            dint16[index] = Convert.ToInt16(value);
            return 1;
        }

        protected override object Add(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue + (dynamic)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue - (dynamic)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue * (dynamic)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            if (dValue == 0)
                return 0;

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
            return dValue < (dynamic)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue <= (dynamic)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue == (dynamic)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue != (dynamic)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue > (dynamic)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            Int16 dValue = (Int16)bValue;
            return dValue >= (dynamic)operand;
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
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var duint16 = data.datap as UInt16[];
            duint16[index] = Convert.ToUInt16(value);
            return 1;
        }

         protected override object Add(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue + (dynamic)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue - (dynamic)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue * (dynamic)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            if (dValue == 0)
                return 0;

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
            return dValue < (dynamic)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue <= (dynamic)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue == (dynamic)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue != (dynamic)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue > (dynamic)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            UInt16 dValue = (UInt16)bValue;
            return dValue >= (dynamic)operand;
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
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var dint32 = data.datap as Int32[];
            dint32[index] = Convert.ToInt32(value);
            return 1;
        }

        protected override object Add(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue + (dynamic)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue - (dynamic)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue * (dynamic)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            if (dValue == 0)
                return 0;

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
            return dValue < (dynamic)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue <= (dynamic)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue == (dynamic)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue != (dynamic)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue > (dynamic)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            Int32 dValue = (Int32)bValue;
            return dValue >= (dynamic)operand;
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
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var duint32 = data.datap as UInt32[];
            duint32[index] = Convert.ToUInt32(value);
            return 1;
        }

        protected override object Add(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue + (dynamic)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue - (dynamic)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue * (dynamic)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            if (dValue == 0)
                return 0;

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
            return dValue < (dynamic)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue <= (dynamic)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue == (dynamic)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue != (dynamic)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue > (dynamic)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            UInt32 dValue = (UInt32)bValue;
            return dValue >= (dynamic)operand;
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
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var dint64 = data.datap as Int64[];
            dint64[index] = Convert.ToInt64(value);
            return 1;
        }

        protected override object Add(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue + (dynamic)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue - (dynamic)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue * (dynamic)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            if (dValue == 0)
                return 0;

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
            return dValue < (dynamic)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue <= (dynamic)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue == (dynamic)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue != (dynamic)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue > (dynamic)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            Int64 dValue = (Int64)bValue;
            return dValue >= (dynamic)operand;
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
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var duint64 = data.datap as UInt64[];
            duint64[index] = Convert.ToUInt64(value);
            return 1;
        }

        protected override object Add(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue + (dynamic)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue - (dynamic)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue * (dynamic)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            if (dValue == 0)
                return 0;
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
            double dValue = Convert.ToDouble(bValue);
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
            return dValue < (dynamic)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue <= (dynamic)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue == (dynamic)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue != (dynamic)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue > (dynamic)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            UInt64 dValue = (UInt64)bValue;
            return dValue >= (dynamic)operand;
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
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var float1 = data.datap as float[];
            float1[index] = Convert.ToSingle(value);
            return 1;
        }

        public override object GetArgSortMinValue()
        {
            return float.MinValue;
        }
        public override object GetArgSortMaxValue()
        {
            return float.MaxValue;
        }
        public override object GetPositiveInfinity()
        {
            return float.PositiveInfinity;
        }
        public override object GetNegativeInfinity()
        {
            return float.NegativeInfinity;
        }
        public override object GetNaN()
        {
            return float.NaN;
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

        public override NPY_TYPES MathOpReturnType(UFuncOperation Operation)
        {
            switch (Operation)
            {
                case UFuncOperation.power:
                    return NPY_TYPES.NPY_DOUBLE;
                case UFuncOperation.true_divide:
                    return NPY_TYPES.NPY_DOUBLE;
                case UFuncOperation.special_operand_is_float:
                    return NPY_TYPES.NPY_DOUBLE;

            }

            return NPY_TYPES.NPY_FLOAT;
        }

        public override NPY_TYPES MathOpFloatingType(UFuncOperation Operation)
        {
             return NPY_TYPES.NPY_FLOAT;
        }

        protected override object Add(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue + (dynamic)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue - (dynamic)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue * (dynamic)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            float dValue = (float)bValue;
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            if (dValue == 0)
                return 0;

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
            return dValue < (dynamic)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue <= (dynamic)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue == (dynamic)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue != (dynamic)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue > (dynamic)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            float dValue = (float)bValue;
            return dValue >= (dynamic)operand;
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
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var double1 = data.datap as double[];
            double1[index] = Convert.ToDouble(value);
            return 1;
        }

        public override object GetArgSortMinValue()
        {
            return double.MinValue;
        }
        public override object GetArgSortMaxValue()
        {
            return double.MaxValue;
        }
        public override object GetPositiveInfinity()
        {
            return double.PositiveInfinity;
        }
        public override object GetNegativeInfinity()
        {
            return double.NegativeInfinity;
        }
        public override object GetNaN()
        {
            return double.NaN;
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

        public override NPY_TYPES MathOpReturnType(UFuncOperation Operation)
        {
            return NPY_TYPES.NPY_DOUBLE;
        }
        public override NPY_TYPES MathOpFloatingType(UFuncOperation Operation)
        {
            return NPY_TYPES.NPY_DOUBLE;
        }

        protected override object Add(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue + (dynamic)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue - (dynamic)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue * (dynamic)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            double dValue = (double)bValue;
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            double doperand = (dynamic)operand;
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
            if (dValue == 0)
                return 0;

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
            return dValue < (dynamic)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue <= (dynamic)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue == (dynamic)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue != (dynamic)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue > (dynamic)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            double dValue = (double)bValue;
            return dValue >= (dynamic)operand;
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
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var decimal1 = data.datap as decimal[];
            decimal1[index] = Convert.ToDecimal(value);
            return 1;
        }

        public override object GetArgSortMinValue()
        {
            return decimal.MinValue;
        }
        public override object GetArgSortMaxValue()
        {
            return decimal.MaxValue;
        }
        public override object GetPositiveInfinity()
        {
            return double.PositiveInfinity;
        }
        public override object GetNegativeInfinity()
        {
            return double.NegativeInfinity;
        }
        public override object GetNaN()
        {
            return double.NaN;
        }

        public override object ConvertToUpgradedValue(object o)
        {
            return Convert.ToDecimal(o);
        }

        public override NPY_TYPES MathOpReturnType(UFuncOperation Operation)
        {
            return NPY_TYPES.NPY_DECIMAL;
        }
        public override NPY_TYPES MathOpFloatingType(UFuncOperation Operation)
        {
            return NPY_TYPES.NPY_DECIMAL;
        }

        public override object MathOpConvertOperand(object srcValue, object operValue)
        {
            return Convert.ToDecimal(operValue);
        }

        public override bool NonZero(VoidPtr vp, npy_intp index)
        {
            decimal[] bp = vp.datap as decimal[];
            return (bp[index] != 0);
        }
 
        protected override object Add(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue + (dynamic)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue - (dynamic)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue * (dynamic)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            decimal doperand = (dynamic)operand;
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
            decimal doperand = (dynamic)operand;
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
            decimal doperand = (dynamic)operand;
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
            if (dValue == 0)
                return 0;

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
            return dValue < (dynamic)operand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue <= (dynamic)operand;
        }
        protected override object Equal(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue == (dynamic)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue != (dynamic)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue > (dynamic)operand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            decimal dValue = (decimal)bValue;
            return dValue >= (dynamic)operand;
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

        public override int SetItem(VoidPtr data, npy_intp index, object value)
        {
            System.Numerics.Complex[] dp = data.datap as System.Numerics.Complex[];

            long AdjustedIndex = (data.data_offset + index) >> ItemDiv;
            if (AdjustedIndex < 0)
            {
                AdjustedIndex = dp.Length - -AdjustedIndex;
            }
            if (value is System.Numerics.Complex)
            {
                dp[AdjustedIndex] = (System.Numerics.Complex)value;
            }
            else
            {
                dp[AdjustedIndex] = new System.Numerics.Complex(Convert.ToDouble(value), 0);
            }

            return 1;
        }
        public override int SetItemDifferentType(VoidPtr data, npy_intp index, object value)
        {
            System.Numerics.Complex cvalue;

            if (value is System.Numerics.Complex)
            {
                cvalue = (System.Numerics.Complex)value;
            }
            else
            {
                try
                {
                    cvalue = new System.Numerics.Complex(Convert.ToDouble(value), 0);
                }
                catch
                {
                    throw new Exception("unable to convert {0} to a Complex value");
                }

            }

            return SetItem(data, index, cvalue);
        }
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var complex1 = data.datap as System.Numerics.Complex[];
            if (value is System.Numerics.Complex)
            {
                complex1[index] = (System.Numerics.Complex)value;
            }
            else
            {
                complex1[index] = new System.Numerics.Complex(Convert.ToDouble(value), 0);
            }
            return 1;
        }

        public override object GetArgSortMinValue()
        {
            return (System.Numerics.Complex)double.MinValue;
        }
        public override object GetArgSortMaxValue()
        {
            return (System.Numerics.Complex)double.MaxValue;
        }
        public override object GetPositiveInfinity()
        {
            return (System.Numerics.Complex)double.PositiveInfinity;
        }
        public override object GetNegativeInfinity()
        {
            return (System.Numerics.Complex)double.NegativeInfinity;
        }
        public override object GetNaN()
        {
            return (System.Numerics.Complex)double.NaN;
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
 
        public override object ConvertToUpgradedValue(object o)
        {
            return ConvertToComplex(o);
        }
        public override NPY_TYPES MathOpReturnType(UFuncOperation Operation)
        {
            return NPY_TYPES.NPY_COMPLEX;
        }
        public override NPY_TYPES MathOpFloatingType(UFuncOperation Operation)
        {
            return NPY_TYPES.NPY_COMPLEX;
        }

        public override object MathOpConvertOperand(object srcValue, object operValue)
        {
            if (operValue is System.Numerics.Complex)
            {
                return operValue;
            }
            else
            {
                return new System.Numerics.Complex(Convert.ToDouble(operValue), 0);
            }
        }

        public override bool NonZero(VoidPtr vp, npy_intp index)
        {
            System.Numerics.Complex[] bp = vp.datap as System.Numerics.Complex[];
            return (bp[index] != System.Numerics.Complex.Zero);
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
            if (dValue == 0)
                return 0;

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
        void Quick_Sort(System.Numerics.Complex[] array, int low, int high)
        {
            if (low < high)
            {
                int partitionIndex = Partition(array, low, high);

                //3. Recursively continue sorting the array
                Quick_Sort(array, low, partitionIndex - 1);
                Quick_Sort(array, partitionIndex + 1, high);
            }
        }
        int Partition(System.Numerics.Complex[] array, int low, int high)
        {
            //1. Select a pivot point.
            System.Numerics.Complex pivot = array[high];

            int lowIndex = (low - 1);

            //2. Reorder the collection. sort by Real first, then Imaginary.
            for (int j = low; j < high; j++)
            {
                if (array[j].Real == pivot.Real && array[j].Imaginary <= pivot.Imaginary)
                {
                    lowIndex++;

                    var temp = array[lowIndex];
                    array[lowIndex] = array[j];
                    array[j] = temp;
                }
                else
                if (array[j].Real < pivot.Real)
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
        protected override object Conjugate(object invalue, object operand)
        {
            if (invalue is System.Numerics.Complex)
            {
                var cc = (System.Numerics.Complex)invalue;
                cc = new System.Numerics.Complex(cc.Real, -cc.Imaginary);
                return cc;
            }
            return invalue;
        }


    }

    internal class BigIntHandlers : ArrayHandlerBase<System.Numerics.BigInteger>, IArrayHandlers
    {
        public BigIntHandlers()
        {

        }

        public override int ItemSize
        {
            get { return sizeof(double) * 4; }
        }

        public override int SetItem(VoidPtr data, npy_intp index, object value)
        {
            System.Numerics.BigInteger[] dp = data.datap as System.Numerics.BigInteger[];

            long AdjustedIndex = (data.data_offset + index) >> ItemDiv;
            if (AdjustedIndex < 0)
            {
                AdjustedIndex = dp.Length - -AdjustedIndex;
            }
            if (value is System.Numerics.BigInteger)
            {
                dp[AdjustedIndex] = (System.Numerics.BigInteger)value;
            }
            else
            {
                dp[AdjustedIndex] = new System.Numerics.BigInteger(Convert.ToDouble(value));
            }

            return 1;
        }
        public override int SetItemDifferentType(VoidPtr data, npy_intp index, object value)
        {
            System.Numerics.BigInteger bivalue;

            if (value is System.Numerics.BigInteger)
            {
                bivalue = (System.Numerics.BigInteger)value;
            }
            else
            {
                try
                {
                    bivalue = new System.Numerics.BigInteger(Convert.ToDouble(value));
                }
                catch
                {
                    throw new Exception("unable to convert {0} to a BigInt value");
                }

            }

            return SetItem(data, index, bivalue);
        }
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            var bigint1 = data.datap as System.Numerics.BigInteger[];
            if (value is System.Numerics.BigInteger)
            {
                bigint1[index] = (System.Numerics.BigInteger)value;
            }
            else if (value is System.Numerics.Complex)
            {
                var cc = (System.Numerics.Complex)value;
                bigint1[index] = (System.Numerics.BigInteger)cc.Real;
            }
            else
            {
                bigint1[index] = new System.Numerics.BigInteger(Convert.ToDouble(value));
            }
            return 1;
        }

        public override object GetArgSortMinValue()
        {
            return (System.Numerics.BigInteger)double.MinValue;
        }
        public override object GetArgSortMaxValue()
        {
            return (System.Numerics.BigInteger)double.MaxValue;
        }
        public override object GetPositiveInfinity()
        {
            return (System.Numerics.BigInteger)double.PositiveInfinity;
        }
        public override object GetNegativeInfinity()
        {
            return (System.Numerics.BigInteger)double.NegativeInfinity;
        }
        public override object GetNaN()
        {
            return (System.Numerics.BigInteger)double.NaN;
        }

        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            System.Numerics.BigInteger[] adata = vp.datap as System.Numerics.BigInteger[];

            if (FillValue is System.Numerics.BigInteger)
                Fill(adata, (System.Numerics.BigInteger)FillValue, 0, adata.Length);
            else
                Fill(adata, new System.Numerics.BigInteger(Convert.ToDouble(FillValue)), 0, adata.Length);

            return;
        }
 
        public override object ConvertToUpgradedValue(object o)
        {
            return ConvertToBigInt(o);
        }
        public override NPY_TYPES MathOpReturnType(UFuncOperation Operation)
        {
            return NPY_TYPES.NPY_BIGINT;
        }
        public override NPY_TYPES MathOpFloatingType(UFuncOperation Operation)
        {
            return NPY_TYPES.NPY_DOUBLE;
        }

        public override object MathOpConvertOperand(object srcValue, object operValue)
        {
            if (operValue is System.Numerics.BigInteger)
            {
                return operValue;
            }
            else
            {
                return new System.Numerics.BigInteger(Convert.ToDouble(operValue));
            }
        }

        public override bool NonZero(VoidPtr vp, npy_intp index)
        {
            System.Numerics.BigInteger[] bp = vp.datap as System.Numerics.BigInteger[];
            return (bp[index] != System.Numerics.BigInteger.Zero);
        }


        System.Numerics.BigInteger ConvertToBigInt(object o)
        {
            if (o is System.Numerics.BigInteger)
            {
                System.Numerics.BigInteger c = (System.Numerics.BigInteger)o;
                return c;
            }
            else
            {
                return new System.Numerics.BigInteger(Convert.ToDouble(o));
            }
        }

        protected override object Add(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            return dValue + (BigInteger)operand;
        }
        protected override object Subtract(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            return dValue - (BigInteger)operand;
        }
        protected override object Multiply(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            return dValue * (BigInteger)operand;
        }
        protected override object Divide(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            BigInteger doperand = (BigInteger)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue / doperand;
        }
        protected override object Remainder(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            BigInteger doperand = (BigInteger)operand;
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
            BigInteger dValue = (BigInteger)bValue;
            BigInteger doperand = (BigInteger)operand;
            if (doperand == 0)
            {
                dValue = 0;
                return dValue;
            }
            return dValue % doperand;
        }
        protected override object Power(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            return BigInteger.Pow(dValue, (int)ConvertToBigInt(operand));
        }
        protected override object Square(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            return dValue * dValue;
        }
        protected override object Reciprocal(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            if (dValue == 0)
                return 0;

            return 1 / dValue;
        }
        protected override object Sqrt(object bValue, object operand)
        {
            return Math.Round(Math.Pow(Math.E, BigInteger.Log((BigInteger)bValue) / 2));
        }
  
        protected override object Negative(object bValue, object operand)
        {
            return BigInteger.Negate((BigInteger)bValue);
        }
        protected override object Absolute(object bValue, object operand)
        {
            return BigInteger.Abs((BigInteger)bValue);
        }
        protected override object Invert(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            return dValue;
        }
        protected override object LeftShift(object bValue, object operand)
        {
            BigInteger cValue = (BigInteger)bValue;
            BigInteger oValue = (BigInteger)operand;

            UInt64 rValue = (UInt64)cValue;
            rValue = rValue << Convert.ToInt32((Int64)oValue);


            return new BigInteger(rValue);
        }
        protected override object RightShift(object bValue, object operand)
        {
            BigInteger cValue = (BigInteger)bValue;
            BigInteger oValue = (BigInteger)operand;

            UInt64 rValue = (UInt64)cValue;
            rValue = rValue >> Convert.ToInt32((Int64)oValue);

            return new BigInteger(rValue);
        }
        protected override object BitWiseAnd(object bValue, object operand)
        {
            BigInteger cValue = (BigInteger)bValue;
            BigInteger oValue = (BigInteger)operand;

            UInt64 rValue = (UInt64)cValue;
            rValue = rValue & Convert.ToUInt64((UInt64)oValue);

            return new BigInteger(rValue);
        }
        protected override object BitWiseXor(object bValue, object operand)
        {
            BigInteger cValue = (BigInteger)bValue;
            BigInteger oValue = (BigInteger)operand;

            UInt64 rValue = (UInt64)cValue;
            rValue = rValue ^ Convert.ToUInt64((UInt64)oValue);

            return new BigInteger(rValue);
        }
        protected override object BitWiseOr(object bValue, object operand)
        {
            BigInteger cValue = (BigInteger)bValue;
            BigInteger oValue = (BigInteger)operand;

            UInt64 rValue = (UInt64)cValue;
            rValue = rValue | Convert.ToUInt64((UInt64)oValue);

            return new BigInteger(rValue);
        }
        protected override object Less(object bValue, object operand)
        {
            BigInteger cOperand = (BigInteger)operand;
            BigInteger dValue = (BigInteger)bValue;

            return dValue < cOperand;
        }
        protected override object LessEqual(object bValue, object operand)
        {
            BigInteger cOperand = (BigInteger)operand;
            BigInteger dValue = (BigInteger)bValue;
 
            return dValue <= cOperand;
        }
        protected override object Equal(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            return dValue == (BigInteger)operand;
        }
        protected override object NotEqual(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            return dValue != (BigInteger)operand;
        }
        protected override object Greater(object bValue, object operand)
        {
            BigInteger cOperand = (BigInteger)operand;
            BigInteger dValue = (BigInteger)bValue;
            return dValue > cOperand;
        }
        protected override object GreaterEqual(object bValue, object operand)
        {
            BigInteger cOperand = (BigInteger)operand;
            BigInteger dValue = (BigInteger)bValue;
            return dValue >= cOperand;
        }
        protected override object IsNAN(object bValue, object operand)
        {
            return false;
        }
        protected override object FloorDivide(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            BigInteger oValue = (BigInteger)operand;
            if (oValue == 0)
            {
                dValue = 0;
                return dValue;
            }
            return (dValue / oValue);
        }
        protected override object Floor(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            return dValue;
        }
        protected override object Ceiling(object bValue, object operand)
        {
            BigInteger dValue = (BigInteger)bValue;
            return dValue;
        }
        protected override object Maximum(object bValue, object operand)
        {
            BigInteger a = (BigInteger)bValue;
            BigInteger b = (BigInteger)operand;

            return BigInteger.Max(a, b);
        }
        protected override object FMax(object bValue, object operand)
        {
            BigInteger a = (BigInteger)bValue;
            BigInteger b = (BigInteger)operand;

            return BigInteger.Max(a, b);
        }
        protected override object Minimum(object bValue, object operand)
        {
            BigInteger a = (BigInteger)bValue;
            BigInteger b = (BigInteger)operand;

            return BigInteger.Min(a, b);
        }
        protected override object FMin(object bValue, object operand)
        {
            BigInteger a = (BigInteger)bValue;
            BigInteger b = (BigInteger)operand;

            return BigInteger.Min(a, b);
        }
        protected override object Rint(object bValue, object operand)
        {
            BigInteger a = (BigInteger)bValue;
            BigInteger b = (BigInteger)operand;
            return a;
        }
        protected override object Heaviside(object bValue, object operand)
        {
            BigInteger x = ConvertToBigInt(bValue);

            if (x == 0)
                return ConvertToBigInt(operand);

            if (x < 0)
                return (BigInteger)0;

            return (BigInteger)1;

        }
        public override void SortArray(VoidPtr data, int offset, int length)
        {
            var arr = data.datap as System.Numerics.BigInteger[];
            Quick_Sort(arr, offset, offset + length - 1);
        }
        void Quick_Sort(System.Numerics.BigInteger[] array, int low, int high)
        {
            if (low < high)
            {
                int partitionIndex = Partition(array, low, high);

                //3. Recursively continue sorting the array
                Quick_Sort(array, low, partitionIndex - 1);
                Quick_Sort(array, partitionIndex + 1, high);
            }
        }
        int Partition(System.Numerics.BigInteger[] array, int low, int high)
        {
            //1. Select a pivot point.
            BigInteger pivot = array[high];

            int lowIndex = (low - 1);

            //2. Reorder the collection.
            for (int j = low; j < high; j++)
            {
                if (array[j] <= pivot)
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
            if (invalue is System.Numerics.BigInteger)
            {
                System.Numerics.BigInteger cin = (System.Numerics.BigInteger)invalue;

                if (comparevalue is System.Numerics.BigInteger)
                {
                    System.Numerics.BigInteger ccin = (System.Numerics.BigInteger)comparevalue;
                    return cin.CompareTo(ccin);
                }
                if (comparevalue is IComparable)
                {
                    return cin.CompareTo(comparevalue);
                }
            }

            return 0;
        }


    }

    internal class StringHandlers : ArrayHandlerBase<string>, IArrayHandlers
    {
        public StringHandlers()
        {
            _t = new string[1] { "X" };
        }

        public override int ItemSize
        {
            get { return IntPtr.Size; }
        }

        private string SetCheck(object value)
        {
            return value != null ? value.ToString() : null;
        }

        public override int SetItem(VoidPtr data, npy_intp index, object value)
        {
            string[] dp = data.datap as string[];

            long AdjustedIndex = (data.data_offset + index) >> ItemDiv;
            if (AdjustedIndex < 0)
            {
                AdjustedIndex = dp.Length - -AdjustedIndex;
            }

            dp[AdjustedIndex] = SetCheck(value);
            return 1;
        }
        public override int SetItemDifferentType(VoidPtr data, npy_intp index, object value)
        {
            return SetItem(data, index, value);
        }
        public override int SetIndex(VoidPtr data, npy_intp index, object value)
        {
            index = AdjustNegativeIndex(data, index);

            string[] dp = data.datap as string[];
            dp[index] = SetCheck(value);
            return 1;
        }

        public override void ArrayFill(VoidPtr vp, object FillValue)
        {
            var adata = vp.datap as string[];
            Fill(adata, SetCheck(FillValue), 0, adata.Length);
            return;
        }
  
        public override int ArrayFill(VoidPtr dest, VoidPtr scalar, int length, int dest_offset, int fill_offset)
        {
            string[] destp = dest.datap as string[];
            string[] scalarp = scalar.datap as string[];
            if (destp == null || scalarp == null)
                return -1;

            Fill(destp, SetCheck(scalarp[fill_offset]), dest_offset, length + dest_offset);
            return 0;
        }


        public override object GetArgSortMinValue()
        {
            return "\u0000";
        }
        public override object GetArgSortMaxValue()
        {
            return "\u10FFFF";
        }
        public override int CompareTo(dynamic invalue, dynamic comparevalue)
        {
            if (invalue == null)
            {
                if (comparevalue == null)
                {
                    return 0;
                }
                return 1;
            }

            return string.Compare(invalue.ToString(), comparevalue.ToString());
        }


        public override object ConvertToUpgradedValue(object o)
        {
            return o;
        }

        public override bool NonZero(VoidPtr vp, npy_intp index)
        {
            string[] bp = vp.datap as string[];
            return !(bp[index] == null);
        }

        public override object MathOpConvertOperand(object srcValue, object operValue)
        {
            if (operValue == null)
                return null;

            return operValue.ToString();
        }
        public override NPY_TYPES MathOpFloatingType(UFuncOperation Operation)
        {
            return NPY_TYPES.NPY_STRING;
        }

        public override NPY_TYPES MathOpReturnType(UFuncOperation Operation)
        {
            return NPY_TYPES.NPY_STRING;
        }

        protected override object Add(dynamic bValue, dynamic operand)
        {
            if (operand == null)
                return bValue;

            return bValue + operand;
        }
        protected override object Subtract(dynamic bValue, dynamic operand)
        {
            string sValue = (string)bValue;
            if (sValue == null)
                return sValue;

            return sValue.Replace(operand.ToString(), "");
        }
        protected override object Multiply(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
        protected override object Divide(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
        protected override object Remainder(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
        protected override object FMod(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
        protected override object Power(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
        protected override object Square(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
        protected override object Reciprocal(dynamic bValue, dynamic operand)
        {
            return bValue;
        }

        protected override object OnesLike(dynamic bValue, object operand)
        {
            return "1";
        }
        protected override object Sqrt(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
        protected override object Negative(dynamic bValue, dynamic operand)
        {
            if (bValue == null)
                return bValue;

            char[] arr = bValue.ToCharArray();
            Array.Reverse(arr);
            return new string(arr);
        }
        protected override object Absolute(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
        protected override object Invert(dynamic bValue, dynamic operand)
        {
            string sValue = bValue;

            if (sValue == null)
                return sValue;

            string lowercase = sValue.ToLower();
            if (lowercase != sValue)
                return lowercase;
            else
                return sValue.ToUpper();
        }

        protected override object LeftShift(dynamic bValue, dynamic operand)
        {
            if (bValue == null || operand == null)
                return bValue;

            string sValue = bValue;
            int shiftCount = Convert.ToInt32(operand);

            if (string.IsNullOrEmpty(sValue))
                return sValue;

            for (int i = 0; i < shiftCount; i++)
            {
                string first = sValue.Substring(0,1);
                sValue = sValue.Substring(1) + first;
            }
            return sValue;
        }
        protected override object RightShift(dynamic bValue, dynamic operand)
        {
            if (bValue == null || operand == null)
                return bValue;

            string sValue = bValue;
            int shiftCount = Convert.ToInt32(operand);

            if (string.IsNullOrEmpty(sValue))
                return sValue;

            for (int i = 0; i < shiftCount; i++)
            {
                string last = sValue.Substring(sValue.Length-1, 1);
                sValue = last + sValue.Substring(0, sValue.Length - 1);
            }
            return sValue;
        }
        protected override object BitWiseAnd(dynamic bValue, dynamic operand)
        {
            return bValue;

        }
        protected override object BitWiseXor(dynamic bValue, dynamic operand)
        {
            return bValue;

        }
        protected override object BitWiseOr(dynamic bValue, dynamic operand)
        {
            return bValue;
        }

        protected override object Less(dynamic bValue, dynamic operand)
        {
            return CompareTo(bValue, operand) < 0;
        }
        protected override object LessEqual(dynamic bValue, dynamic operand)
        {
            return CompareTo(bValue, operand) <= 0;
        }
        protected override object Equal(dynamic bValue, dynamic operand)
        {
            return CompareTo(bValue, operand) == 0;
        }
        protected override object NotEqual(dynamic bValue, dynamic operand)
        {
            return CompareTo(bValue, operand) != 0;
        }
        protected override object Greater(dynamic bValue, dynamic operand)
        {
            return CompareTo(bValue, operand) > 0;
        }
        protected override object GreaterEqual(dynamic bValue, dynamic operand)
        {
            return CompareTo(bValue, operand) >= 0;
        }

        protected override object IsNAN(object bValue, object operand)
        {
            return bValue;
        }
        protected override object FloorDivide(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
        protected override object TrueDivide(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
        protected override object LogicalOr(dynamic bValue, dynamic operand)
        {
            if (bValue == null || operand == null)
                return bValue;

            string sValue = (string)bValue;
            return !sValue.Contains(operand.ToString());
        }
        protected override object LogicalAnd(dynamic bValue, dynamic operand)
        {
            if (bValue == null || operand == null)
                return bValue;

            string sValue = (string)bValue;
            return sValue.Contains(operand.ToString());
        }
        protected override object Floor(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
        protected override object Ceiling(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
        protected override object Maximum(dynamic bValue, dynamic operand)
        {
            if (CompareTo(bValue, operand) >= 0)
                return bValue.ToString();
            return operand.ToString();
        }
        protected override object FMax(dynamic bValue, dynamic operand)
        {
            if (CompareTo(bValue, operand) >= 0)
                return bValue.ToString();
            return operand.ToString();
        }
        protected override object Minimum(dynamic bValue, dynamic operand)
        {
            if (CompareTo(bValue, operand) <= 0)
                return bValue.ToString();
            return operand.ToString();
        }
        protected override object FMin(dynamic bValue, dynamic operand)
        {
            if (CompareTo(bValue, operand) <= 0)
                return bValue.ToString();
            return operand.ToString();
        }

        protected override object Heaviside(dynamic bValue, dynamic operand)
        {
            return bValue;
        }

        protected override object Rint(dynamic bValue, dynamic operand)
        {
            return bValue;
        }
 

    }
}
