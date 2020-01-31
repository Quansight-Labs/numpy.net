using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace MultiThreadingWithObjects
{
    #region ObjectDemoData

    public class ObjectDemoData
    {
        public System.Int64 iInt64;
        public System.Numerics.Complex iComplex;
        public System.Numerics.BigInteger iBigInt;
        public System.Double iDouble;
        public System.String iString;

        public ObjectDemoData(Int64 iValue)
        {
            this.iInt64 = iValue;
            this.iDouble = iValue;
            this.iComplex = new Complex((double)iValue, 0);
            this.iBigInt = iValue;
            this.iString = iValue.ToString()+":";
        }

        private static ObjectDemoData Copy(ObjectDemoData a)
        {
            ObjectDemoData b = new ObjectDemoData(0);
            b.iInt64 = a.iInt64;
            b.iDouble = a.iDouble;
            b.iComplex = a.iComplex;
            b.iBigInt = a.iBigInt;
            b.iString = a.iString;
            return b;
        }

        public override string ToString()
        {
            return string.Format("{0}:{1}:{2}:{3}:{4}",
                iInt64.ToString(), iDouble.ToString(), iComplex.ToString(), iBigInt.ToString(), iString);
        }
        #region ADD operations
        public static ObjectDemoData operator +(ObjectDemoData a, Int64 iValue)
        {
            var b = Copy(a);

            b.iInt64 += iValue;
            b.iDouble += iValue;
            b.iComplex += new Complex((double)iValue, 0);
            b.iBigInt += iValue;
            b.iString += iValue.ToString();

            return b;
        }

        public static ObjectDemoData operator +(ObjectDemoData a, double dValue)
        {
            var b = Copy(a);

            b.iInt64 += (Int64)dValue;
            b.iDouble += dValue;
            b.iComplex += new Complex((double)dValue, 0);
            b.iBigInt += (Int64)dValue;
            b.iString += b.iInt64.ToString();

            return b;
        }

        public static ObjectDemoData operator +(ObjectDemoData a, Complex iValue)
        {
            var b = Copy(a);

            b.iInt64 += (Int64)iValue.Real;
            b.iDouble += iValue.Real;
            b.iComplex += iValue;
            b.iBigInt += (Int64)iValue.Real;
            b.iString += b.iInt64.ToString();

            return b;
        }

        public static ObjectDemoData operator +(ObjectDemoData a, BigInteger iValue)
        {
            var b = Copy(a);

            b.iInt64 += (Int64)iValue;
            b.iDouble += (Int64)iValue;
            b.iComplex += new Complex((double)iValue, 0);
            b.iBigInt += iValue;
            b.iString += b.iInt64.ToString();

            return b;
        }

        public static ObjectDemoData operator +(ObjectDemoData a, string sValue)
        {
            var b = Copy(a);

            Int64 iValue = 0;
            Int64.TryParse(sValue, out iValue);

            b.iInt64 += iValue;
            b.iDouble += iValue;
            b.iComplex += new Complex((double)iValue, 0);
            b.iBigInt += iValue;
            b.iString += sValue;

            return b;
        }

        public static ObjectDemoData operator +(ObjectDemoData a, ObjectDemoData iValue)
        {
            var b = Copy(a);

            b.iInt64 += iValue.iInt64;
            b.iDouble += iValue.iDouble;
            b.iComplex += iValue.iComplex;
            b.iBigInt += iValue.iBigInt;
            b.iString += b.iInt64.ToString();

            return b;
        }
        #endregion

        #region SUBTRACT operations
        public static ObjectDemoData operator -(ObjectDemoData a, Int64 iValue)
        {
            var b = Copy(a);

            b.iInt64 -= iValue;
            b.iDouble -= iValue;
            b.iComplex -= new Complex((double)iValue, 0);
            b.iBigInt -= iValue;
            b.iString = b.iString.Substring(0, (int)Math.Min(iValue, b.iString.Length));

            return b;
        }

        public static ObjectDemoData operator -(ObjectDemoData a, double dValue)
        {
            var b = Copy(a);

            b.iInt64 -= (Int64)dValue;
            b.iDouble -= dValue;
            b.iComplex -= new Complex((double)dValue, 0);
            b.iBigInt -= (Int64)dValue;
            b.iString = b.iString.Substring(0, (int)Math.Min(b.iInt64, b.iString.Length));

            return b;
        }

        public static ObjectDemoData operator -(ObjectDemoData a, Complex iValue)
        {
            var b = Copy(a);

            b.iInt64 -= (Int64)iValue.Real;
            b.iDouble -= iValue.Real;
            b.iComplex -= iValue;
            b.iBigInt -= (Int64)iValue.Real;
            b.iString = b.iString.Substring(0, (int)Math.Min(b.iInt64, b.iString.Length));

            return b;
        }

        public static ObjectDemoData operator -(ObjectDemoData a, BigInteger iValue)
        {
            var b = Copy(a);

            b.iInt64 -= (Int64)iValue;
            b.iDouble -= (Int64)iValue;
            b.iComplex -= new Complex((double)iValue, 0);
            b.iBigInt -= iValue;
            b.iString = b.iString.Substring(0, (int)Math.Min(b.iInt64, b.iString.Length));

            return b;
        }

        public static ObjectDemoData operator -(ObjectDemoData a, string sValue)
        {
            var b = Copy(a);

            Int64 iValue = 0;
            Int64.TryParse(sValue, out iValue);

            b.iInt64 -= iValue;
            b.iDouble -= iValue;
            b.iComplex -= new Complex((double)iValue, 0);
            b.iBigInt -= iValue;

            b.iString.Replace(sValue, "");
            return b;
        }

        public static ObjectDemoData operator -(ObjectDemoData a, ObjectDemoData iValue)
        {
            var b = Copy(a);

            b.iInt64 -= iValue.iInt64;
            b.iDouble -= iValue.iDouble;
            b.iComplex -= iValue.iComplex;
            b.iBigInt -= iValue.iBigInt;
            b.iString = b.iString.Substring(0, (int)Math.Min(b.iInt64, b.iString.Length));

            return b;
        }

        #endregion

        #region MULTIPLY operations
        public static ObjectDemoData operator *(ObjectDemoData a, Int64 iValue)
        {
            var b = Copy(a);

            b.iInt64 *= iValue;
            b.iDouble *= iValue;
            b.iComplex *= new Complex((double)iValue, 0);
            b.iBigInt *= iValue;
            b.iString += "*" + iValue.ToString();
            return b;
        }

        public static ObjectDemoData operator *(ObjectDemoData a, double dValue)
        {
            var b = Copy(a);

            b.iInt64 *= (Int64)dValue;
            b.iDouble *= dValue;
            b.iComplex *= new Complex((double)dValue, 0);
            b.iBigInt *= (Int64)dValue;
            b.iString += "*" + ((Int64)dValue).ToString();

            return b;
        }

        public static ObjectDemoData operator *(ObjectDemoData a, Complex iValue)
        {
            var b = Copy(a);

            b.iInt64 *= (Int64)iValue.Real;
            b.iDouble *= iValue.Real;
            b.iComplex *= iValue;
            b.iBigInt *= (Int64)iValue.Real;

            return b;
        }

        public static ObjectDemoData operator *(ObjectDemoData a, BigInteger iValue)
        {
            var b = Copy(a);

            b.iInt64 *= (Int64)iValue;
            b.iDouble *= (Int64)iValue;
            b.iComplex *= new Complex((double)iValue, 0);
            b.iBigInt *= iValue;

            return b;
        }

        public static ObjectDemoData operator *(ObjectDemoData a, string sValue)
        {
            var b = Copy(a);

            Int64 iValue = 0;
            Int64.TryParse(sValue, out iValue);

            b.iInt64 *= iValue;
            b.iDouble *= iValue;
            b.iComplex *= new Complex((double)iValue, 0);
            b.iBigInt *= iValue;

            return b;
        }
        public static ObjectDemoData operator *(ObjectDemoData a, ObjectDemoData iValue)
        {
            var b = Copy(a);

            b.iInt64 *= iValue.iInt64;
            b.iDouble *= iValue.iDouble;
            b.iComplex *= iValue.iComplex;
            b.iBigInt *= iValue.iBigInt;

            return b;
        }
        #endregion

        #region DIVIDE operations
        public static ObjectDemoData operator /(ObjectDemoData a, Int64 iValue)
        {
            var b = Copy(a);

            b.iInt64 /= iValue;
            b.iDouble /= iValue;
            b.iComplex /= new Complex((double)iValue, 0);
            b.iBigInt /= iValue;

            return b;
        }

        public static ObjectDemoData operator /(ObjectDemoData a, double dValue)
        {
            var b = Copy(a);

            b.iInt64 /= (Int64)dValue;
            b.iDouble /= dValue;
            b.iComplex /= new Complex((double)dValue, 0);
            b.iBigInt /= (Int64)dValue;

            return b;
        }

        public static ObjectDemoData operator /(ObjectDemoData a, Complex iValue)
        {
            var b = Copy(a);

            b.iInt64 /= (Int64)iValue.Real;
            b.iDouble /= iValue.Real;
            b.iComplex /= iValue;
            b.iBigInt /= (Int64)iValue.Real;

            return b;
        }

        public static ObjectDemoData operator /(ObjectDemoData a, BigInteger iValue)
        {
            var b = Copy(a);

            b.iInt64 /= (Int64)iValue;
            b.iDouble /= (Int64)iValue;
            b.iComplex /= new Complex((double)iValue, 0);
            b.iBigInt /= iValue;

            return b;
        }

        public static ObjectDemoData operator /(ObjectDemoData a, string sValue)
        {
            var b = Copy(a);

            Int64 iValue = 0;
            Int64.TryParse(sValue, out iValue);

            b.iInt64 /= iValue;
            b.iDouble /= iValue;
            b.iComplex /= new Complex((double)iValue, 0);
            b.iBigInt /= iValue;

            return b;
        }
        public static ObjectDemoData operator /(ObjectDemoData a, ObjectDemoData iValue)
        {
            var b = Copy(a);

            b.iInt64 /= iValue.iInt64;
            b.iDouble /= iValue.iDouble;
            b.iComplex /= iValue.iComplex;
            b.iBigInt /= iValue.iBigInt;

            return b;
        }
        #endregion

        #region LEFTSHIFT operations
        public static ObjectDemoData operator <<(ObjectDemoData a, int iValue)
        {
            var b = Copy(a);

            b.iInt64 <<= iValue;
            //b.iDouble <<= iValue;
            //b.iComplex <<= new Complex((double)iValue, 0);
            b.iBigInt <<= iValue;

            return b;
        }

        #endregion

        #region RIGHTSHIFT operations
        public static ObjectDemoData operator >>(ObjectDemoData a, int iValue)
        {
            var b = Copy(a);

            b.iInt64 >>= iValue;
            //b.iDouble >>= iValue;
            //b.iComplex >>= new Complex((double)iValue, 0);
            b.iBigInt >>= iValue;

            return b;
        }

        #endregion

        #region BITWISEAND operations
        public static ObjectDemoData operator &(ObjectDemoData a, Int32 iValue)
        {
            var b = Copy(a);

            b.iInt64 &= iValue;
            //b.iDouble &= iValue;
            //b.iComplex &= new Complex((double)iValue, 0);
            b.iBigInt &= iValue;

            return b;
        }
        public static ObjectDemoData operator &(ObjectDemoData a, Int64 iValue)
        {
            var b = Copy(a);

            b.iInt64 &= iValue;
            //b.iDouble &= iValue;
            //b.iComplex &= new Complex((double)iValue, 0);
            b.iBigInt &= iValue;

            return b;
        }
        #endregion

        #region BITWISEOR operations
        public static ObjectDemoData operator |(ObjectDemoData a, Int32 iValue)
        {
            var b = Copy(a);

            b.iInt64 |= (UInt32)iValue;
            //b.iDouble &= iValue;
            //b.iComplex &= new Complex((double)iValue, 0);
            b.iBigInt |= iValue;

            return b;
        }
        public static ObjectDemoData operator |(ObjectDemoData a, Int64 iValue)
        {
            var b = Copy(a);

            b.iInt64 |= iValue;
            //b.iDouble &= iValue;
            //b.iComplex &= new Complex((double)iValue, 0);
            b.iBigInt |= iValue;

            return b;
        }
        #endregion

        #region BITWISEXOR operations
        public static ObjectDemoData operator ^(ObjectDemoData a, Int32 iValue)
        {
            var b = Copy(a);

            b.iInt64 ^= iValue;
            //b.iDouble &= iValue;
            //b.iComplex &= new Complex((double)iValue, 0);
            b.iBigInt ^= iValue;

            return b;
        }
        public static ObjectDemoData operator ^(ObjectDemoData a, Int64 iValue)
        {
            var b = Copy(a);

            b.iInt64 ^= iValue;
            //b.iDouble &= iValue;
            //b.iComplex &= new Complex((double)iValue, 0);
            b.iBigInt ^= iValue;

            return b;
        }
        #endregion

        #region REMAINDER operations
        public static ObjectDemoData operator %(ObjectDemoData a, double iValue)
        {
            var b = Copy(a);

            b.iInt64 %= (Int64)iValue;
            b.iDouble %= iValue;
            //b.iComplex %= iValue; 
            b.iBigInt %= (Int64)iValue;

            return b;
        }
        public static ObjectDemoData operator %(ObjectDemoData a, Int64 iValue)
        {
            var b = Copy(a);

            b.iInt64 %= iValue;
            b.iDouble %= iValue;
            //b.iComplex %= new Complex((double)iValue, 0);
            b.iBigInt %= iValue;

            return b;
        }
        public static ObjectDemoData operator %(ObjectDemoData a, ObjectDemoData iValue)
        {
            var b = Copy(a);

            b.iInt64 %= iValue.iInt64;
            b.iDouble %= iValue.iDouble;
            //b.iComplex %= iValue.iComplex;
            b.iBigInt %= iValue.iBigInt;

            return b;
        }
        #endregion

        #region NEGATIVE operations
        public static ObjectDemoData operator -(ObjectDemoData a)
        {
            var b = Copy(a);

            b.iInt64 = -b.iInt64;
            b.iDouble = -b.iDouble;
            b.iComplex = -b.iComplex;
            b.iBigInt = -b.iBigInt;

            return b;
        }
        #endregion

        #region INVERT operations
        public static ObjectDemoData operator ~(ObjectDemoData a)
        {
            var b = Copy(a);

            b.iInt64 = ~b.iInt64;
            //b.iDouble = ~b.iDouble;
            //b.iComplex = ~b.iComplex;
            b.iBigInt = ~b.iBigInt;

            return b;
        }
        #endregion

        #region LESS operations
        public static bool operator <(ObjectDemoData a, Int64 iValue)
        {
            return a.iInt64 < iValue;
        }

        public static bool operator <(ObjectDemoData a, double dValue)
        {
            return a.iDouble < dValue;
        }

        public static bool operator <(ObjectDemoData a, Complex iValue)
        {
            return a.iComplex.Real < iValue.Real;
        }

        public static bool operator <(ObjectDemoData a, BigInteger iValue)
        {
            return a.iBigInt < iValue;
        }

        public static bool operator <(ObjectDemoData a, ObjectDemoData iValue)
        {
            return a.iInt64 < iValue.iInt64;
        }
        #endregion

        #region LESSEQUAL operations
        public static bool operator <=(ObjectDemoData a, Int64 iValue)
        {
            return a.iInt64 <= iValue;
        }

        public static bool operator <=(ObjectDemoData a, double dValue)
        {
            return a.iDouble <= dValue;
        }

        public static bool operator <=(ObjectDemoData a, Complex iValue)
        {
            return a.iComplex.Real <= iValue.Real;
        }

        public static bool operator <=(ObjectDemoData a, BigInteger iValue)
        {
            return a.iBigInt <= iValue;
        }

        public static bool operator <=(ObjectDemoData a, ObjectDemoData iValue)
        {
            return a.iInt64 <= iValue.iInt64;
        }
        #endregion

        #region EQUAL operations
        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public static bool operator ==(ObjectDemoData a, Int64 iValue)
        {
            return a.iInt64 == iValue;
        }

        public static bool operator ==(ObjectDemoData a, double dValue)
        {
            return a.iDouble == dValue;
        }

        public static bool operator ==(ObjectDemoData a, Complex iValue)
        {
            return a.iComplex.Real == iValue.Real;
        }

        public static bool operator ==(ObjectDemoData a, BigInteger iValue)
        {
            return a.iBigInt == iValue;
        }

        public static bool operator ==(ObjectDemoData a, ObjectDemoData iValue)
        {
            return a.iInt64 == iValue.iInt64;
        }
        #endregion

        #region NOTEQUAL operations


        public static bool operator !=(ObjectDemoData a, Int64 iValue)
        {
            return a.iInt64 != iValue;
        }

        public static bool operator !=(ObjectDemoData a, double dValue)
        {
            return a.iDouble != dValue;
        }

        public static bool operator !=(ObjectDemoData a, Complex iValue)
        {
            return a.iComplex.Real != iValue.Real;
        }

        public static bool operator !=(ObjectDemoData a, BigInteger iValue)
        {
            return a.iBigInt != iValue;
        }

        public static bool operator !=(ObjectDemoData a, ObjectDemoData iValue)
        {
            return a.iInt64 != iValue.iInt64;
        }
        #endregion

        #region GREATER operations
        public static bool operator >(ObjectDemoData a, Int64 iValue)
        {
            return a.iInt64 > iValue;
        }

        public static bool operator >(ObjectDemoData a, double dValue)
        {
            return a.iDouble > dValue;
        }

        public static bool operator >(ObjectDemoData a, Complex iValue)
        {
            return a.iComplex.Real > iValue.Real;
        }

        public static bool operator >(ObjectDemoData a, BigInteger iValue)
        {
            return a.iBigInt > iValue;
        }

        public static bool operator >(ObjectDemoData a, ObjectDemoData iValue)
        {
            return a.iInt64 > iValue.iInt64;
        }
        #endregion

        #region GREATEREQUAL operations
        public static bool operator >=(ObjectDemoData a, Int64 iValue)
        {
            return a.iInt64 >= iValue;
        }

        public static bool operator >=(ObjectDemoData a, double dValue)
        {
            return a.iDouble >= dValue;
        }

        public static bool operator >=(ObjectDemoData a, Complex iValue)
        {
            return a.iComplex.Real >= iValue.Real;
        }

        public static bool operator >=(ObjectDemoData a, BigInteger iValue)
        {
            return a.iBigInt >= iValue;
        }

        public static bool operator >=(ObjectDemoData a, ObjectDemoData iValue)
        {
            return a.iInt64 >= iValue.iInt64;
        }
        #endregion
    }



    #endregion
}
