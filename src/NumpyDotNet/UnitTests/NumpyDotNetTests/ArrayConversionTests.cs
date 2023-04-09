using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumpyLib;


namespace NumpyDotNetTests
{
    [TestClass]
    public class ArrayConversionsTests : TestBaseClass
    {
        [TestMethod]
        public void test_AsBool()
        {
            var TestData = new bool[] { true, false, false, true, false };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, false, true, false });
            AssertArray(np.array(abool.AsBoolArray()), new bool[] { true, false, false, true, false });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 1, 0, 0, 1, 0 });
            AssertArray(np.array(aSByte.AsBoolArray()), TestData);
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 1, 0, 0, 1, 0 });
            AssertArray(np.array(aUByte.AsBoolArray()), TestData);
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 1, 0, 0, 1, 0 });
            AssertArray(np.array(aInt16.AsBoolArray()), TestData);
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 1, 0, 0, 1, 0 });
            AssertArray(np.array(aUInt16.AsBoolArray()), TestData);
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 1, 0, 0, 1, 0 });
            AssertArray(np.array(aInt32.AsBoolArray()), TestData);
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 1, 0, 0, 1, 0 });
            AssertArray(np.array(aUInt32.AsBoolArray()), TestData);
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 1, 0, 0, 1, 0 });
            AssertArray(np.array(aInt64.AsBoolArray()), TestData);
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 1, 0, 0, 1, 0 });
            AssertArray(np.array(aUInt64.AsBoolArray()), TestData);
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 1f, 0f, 0f, 1f, 0f });
            AssertArray(np.array(afloat.AsBoolArray()), TestData);
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 1, 0, 0, 1, 0 });
            AssertArray(np.array(adouble.AsBoolArray()), TestData);
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 1, 0, 0, 1, 0 });
            AssertArray(np.array(adecimal.AsBoolArray()), TestData);
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 1, 0, 0, 1, 0 });
            AssertArray(np.array(acomplex.AsBoolArray()), TestData);
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 1, 0, 0, 1, 0 });
            AssertArray(np.array(abigint.AsBoolArray()), TestData);
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { 1, 0, 0, 1, 0 });
            aobject[1] = null; aobject[2] = null; aobject[4] = null;
            AssertArray(np.array(aobject.AsBoolArray()), new bool[] { true, false, false, true, false });
            print(aobject);

            var astring = np.array(a.AsStringArray());
            AssertArray(astring, new string[] { "True", "False", "False", "True", "False" });
            astring[1] = null; astring[2] = null; astring[4] = null;
            AssertArray(np.array(astring.AsBoolArray()), new bool[] { true, false, false, true, false });
            print(aobject);
        }

        [TestMethod]
        public void test_AsByte()
        {
            var TestData = new Byte[] { 6, 0, 10, 23, 11 };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsByteArray()), new byte[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 6, 0, 10, 23, 11 });
            AssertArray(np.array(aSByte.AsByteArray()), TestData);
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 6, 0, 10, 23, 11 });
            AssertArray(np.array(aUByte.AsByteArray()), TestData);
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 6, 0, 10, 23, 11 });
            AssertArray(np.array(aInt16.AsByteArray()), TestData);
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 6, 0, 10, 23, 11 });
            AssertArray(np.array(aUInt16.AsByteArray()), TestData);
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 6, 0, 10, 23, 11 });
            AssertArray(np.array(aInt32.AsByteArray()), TestData);
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 6, 0, 10, 23, 11 });
            AssertArray(np.array(aUInt32.AsByteArray()), TestData);
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 6, 0, 10, 23, 11 });
            AssertArray(np.array(aInt64.AsByteArray()), TestData);
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 6, 0, 10, 23, 11 });
            AssertArray(np.array(aUInt64.AsByteArray()), TestData);
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 6f, 0f, 10f, 23f, 11f });
            AssertArray(np.array(afloat.AsByteArray()), TestData);
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 6, 0, 10, 23, 11 });
            AssertArray(np.array(adouble.AsByteArray()), TestData);
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 6, 0, 10, 23, 11 });
            AssertArray(np.array(adecimal.AsByteArray()), TestData);
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 6, 0, 10, 23, 11 });
            AssertArray(np.array(acomplex.AsByteArray()), TestData);
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 6, 0, 10, 23, 11 });
            AssertArray(np.array(abigint.AsByteArray()), TestData);
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { (byte)6, (byte)0, (byte)10, (byte)23, (byte)(11) });
            AssertArray(np.array(aobject.AsByteArray()), new byte[] { 0, 0, 0, 0, 0 });
            print(aobject);

        }

        [TestMethod]
        public void test_AsSByte()
        {
            var TestData = new sbyte[] { 6, 0, 10, 23, -23 };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsSByteArray()), new sbyte[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(aSByte.AsSByteArray()), TestData);
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 6, 0, 10, 23, 233 });
            AssertArray(np.array(aUByte.AsSByteArray()), new sbyte[] { 6, 0, 10, 23, -23 });
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(aInt16.AsSByteArray()), TestData);
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 6, 0, 10, 23, 65513 });
            AssertArray(np.array(aUInt16.AsSByteArray()), new sbyte[] { 6, 0, 10, 23, -23 });
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(aInt32.AsSByteArray()), TestData);
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 6, 0, 10, 23, 4294967273 });
            AssertArray(np.array(aUInt32.AsSByteArray()), new sbyte[] { 6, 0, 10, 23, -23 });
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(aInt64.AsSByteArray()), new sbyte[] { 6, 0, 10, 23, -23 });
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 6, 0, 10, 23, 18446744073709551593 });
            AssertArray(np.array(aUInt64.AsSByteArray()), new sbyte[] { 6, 0, 10, 23, -23 });
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 6f, 0f, 10f, 23f, -23f });
            AssertArray(np.array(afloat.AsSByteArray()), new sbyte[] { 6, 0, 10, 23, -23 });
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(adouble.AsSByteArray()), new sbyte[] { 6, 0, 10, 23, -23 });
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(adecimal.AsSByteArray()), new sbyte[] { 6, 0, 10, 23, -23 });
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(acomplex.AsSByteArray()), new sbyte[] { 6, 0, 10, 23, -23 });
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(abigint.AsSByteArray()), new sbyte[] { 6, 0, 10, 23, -23 });
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { (sbyte)6, (sbyte)0, (sbyte)10, (sbyte)23, (sbyte)(-23) });
            AssertArray(np.array(aobject.AsSByteArray()), new sbyte[] { 0, 0, 0, 0, 0 });
            print(aobject);

        }

        [TestMethod]
        public void test_AsInt16()
        {
            var TestData = new Int16[] { 6, 0, 10, 23, -23 };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsInt16Array()), new Int16[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(aSByte.AsInt16Array()), TestData);
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 6, 0, 10, 23, 233 });
            AssertArray(np.array(aUByte.AsInt16Array()), new Int16[] { 6, 0, 10, 23, 233 });
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(aInt16.AsInt16Array()), TestData);
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 6, 0, 10, 23, 65513 });
            AssertArray(np.array(aUInt16.AsInt16Array()), new Int16[] { 6, 0, 10, 23, -23 });
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(aInt32.AsInt16Array()), TestData);
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 6, 0, 10, 23, 4294967273 });
            AssertArray(np.array(aUInt32.AsInt16Array()), new Int16[] { 6, 0, 10, 23, -23 });
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(aInt64.AsInt16Array()), new Int16[] { 6, 0, 10, 23, -23 });
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 6, 0, 10, 23, 18446744073709551593 });
            AssertArray(np.array(aUInt64.AsInt16Array()), new Int16[] { 6, 0, 10, 23, -23 });
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 6f, 0f, 10f, 23f, -23f });
            AssertArray(np.array(afloat.AsInt16Array()), new Int16[] { 6, 0, 10, 23, -23 });
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(adouble.AsInt16Array()), new Int16[] { 6, 0, 10, 23, -23 });
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(adecimal.AsInt16Array()), new Int16[] { 6, 0, 10, 23, -23 });
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(acomplex.AsInt16Array()), new Int16[] { 6, 0, 10, 23, -23 });
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(abigint.AsInt16Array()), new Int16[] { 6, 0, 10, 23, -23 });
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { (Int16)6, (Int16)0, (Int16)10, (Int16)23, (Int16)(-23) });
            AssertArray(np.array(aobject.AsInt16Array()), new Int16[] { 0, 0, 0, 0, 0 });
            print(aobject);

        }

        [TestMethod]
        public void test_AsUInt16()
        {
            var TestData = new UInt16[] { 6, 0, 10, 23, UInt16.MaxValue };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsUInt16Array()), new UInt16[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 6, 0, 10, 23, -1 });
            AssertArray(np.array(aSByte.AsUInt16Array()), TestData);
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 6, 0, 10, 23, 255 });
            AssertArray(np.array(aUByte.AsUInt16Array()), new UInt16[] { 6, 0, 10, 23, 255 });
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 6, 0, 10, 23, -1 });
            AssertArray(np.array(aInt16.AsUInt16Array()), TestData);
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 6, 0, 10, 23, UInt16.MaxValue });
            AssertArray(np.array(aUInt16.AsUInt16Array()), new UInt16[] { 6, 0, 10, 23, 65535 });
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 6, 0, 10, 23, UInt16.MaxValue });
            AssertArray(np.array(aInt32.AsUInt16Array()), TestData);
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 6, 0, 10, 23, UInt16.MaxValue });
            AssertArray(np.array(aUInt32.AsUInt16Array()), TestData);
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 6, 0, 10, 23, UInt16.MaxValue });
            AssertArray(np.array(aInt64.AsUInt16Array()), TestData);
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 6, 0, 10, 23, UInt16.MaxValue });
            AssertArray(np.array(aUInt64.AsUInt16Array()), TestData);
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 6f, 0f, 10f, 23f, (float)UInt16.MaxValue });
            AssertArray(np.array(afloat.AsUInt16Array()), TestData);
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 6, 0, 10, 23, UInt16.MaxValue });
            AssertArray(np.array(adouble.AsUInt16Array()), TestData);
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 6, 0, 10, 23, UInt16.MaxValue });
            AssertArray(np.array(adecimal.AsUInt16Array()), TestData);
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 6, 0, 10, 23, UInt16.MaxValue });
            AssertArray(np.array(acomplex.AsUInt16Array()), TestData);
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 6, 0, 10, 23, UInt16.MaxValue });
            AssertArray(np.array(abigint.AsUInt16Array()), TestData);
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { (UInt16)6, (UInt16)0, (UInt16)10, (UInt16)23, (UInt16)(UInt16.MaxValue) });
            AssertArray(np.array(aobject.AsUInt16Array()), new UInt16[] { 0, 0, 0, 0, 0 });
            print(aobject);

        }

        [TestMethod]
        public void test_AsInt32()
        {
            var TestData = new Int32[] { 6, 0, 10, 23, -23 };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsInt32Array()), new Int32[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(aSByte.AsInt32Array()), TestData);
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 6, 0, 10, 23, 233 });
            AssertArray(np.array(aUByte.AsInt32Array()), new Int32[] { 6, 0, 10, 23, 233 });
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(aInt16.AsInt32Array()), TestData);
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 6, 0, 10, 23, 65513 });
            AssertArray(np.array(aUInt16.AsInt32Array()), new Int32[] { 6, 0, 10, 23, 65513 });
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(aInt32.AsInt32Array()), TestData);
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 6, 0, 10, 23, 4294967273 });
            AssertArray(np.array(aUInt32.AsInt32Array()), new Int32[] { 6, 0, 10, 23, -23 });
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(aInt64.AsInt32Array()), new Int32[] { 6, 0, 10, 23, -23 });
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 6, 0, 10, 23, 18446744073709551593 });
            AssertArray(np.array(aUInt64.AsInt32Array()), new Int32[] { 6, 0, 10, 23, -23 });
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 6f, 0f, 10f, 23f, -23f });
            AssertArray(np.array(afloat.AsInt32Array()), new Int32[] { 6, 0, 10, 23, -23 });
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(adouble.AsInt32Array()), new Int32[] { 6, 0, 10, 23, -23 });
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(adecimal.AsInt32Array()), new Int32[] { 6, 0, 10, 23, -23 });
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(acomplex.AsInt32Array()), new Int32[] { 6, 0, 10, 23, -23 });
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 6, 0, 10, 23, -23 });
            AssertArray(np.array(abigint.AsInt32Array()), new Int32[] { 6, 0, 10, 23, -23 });
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { (Int32)6, (Int32)0, (Int32)10, (Int32)23, (Int32)(-23) });
            AssertArray(np.array(aobject.AsInt32Array()), new Int32[] { 0, 0, 0, 0, 0 });
            print(aobject);

        }

        [TestMethod]
        public void test_AsUInt32()
        {
            var TestData = new UInt32[] { 6, 0, 10, 23, UInt32.MaxValue };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsUInt32Array()), new UInt32[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 6, 0, 10, 23, -1 });
            AssertArray(np.array(aSByte.AsUInt32Array()), TestData);
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 6, 0, 10, 23, 255 });
            AssertArray(np.array(aUByte.AsUInt32Array()), new UInt32[] { 6, 0, 10, 23, 255 });
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 6, 0, 10, 23, -1 });
            AssertArray(np.array(aInt16.AsUInt32Array()), TestData);
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 6, 0, 10, 23, 65535 });
            AssertArray(np.array(aUInt16.AsUInt32Array()), new UInt32[] { 6, 0, 10, 23, 65535 });
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 6, 0, 10, 23, -1 });
            AssertArray(np.array(aInt32.AsUInt32Array()), TestData);
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 6, 0, 10, 23, UInt32.MaxValue });
            AssertArray(np.array(aUInt32.AsUInt32Array()), TestData);
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 6, 0, 10, 23, UInt32.MaxValue });
            AssertArray(np.array(aInt64.AsUInt32Array()), TestData);
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 6, 0, 10, 23, UInt32.MaxValue });
            AssertArray(np.array(aUInt64.AsUInt32Array()), TestData);
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 6f, 0f, 10f, 23f, (float)UInt32.MaxValue });
            AssertArray(np.array(afloat.AsUInt32Array()), new UInt32[] { 6, 0, 10, 23, 0 });
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 6, 0, 10, 23, UInt32.MaxValue });
            AssertArray(np.array(adouble.AsUInt32Array()), TestData);
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 6, 0, 10, 23, UInt32.MaxValue });
            AssertArray(np.array(adecimal.AsUInt32Array()), TestData);
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 6, 0, 10, 23, UInt32.MaxValue });
            AssertArray(np.array(acomplex.AsUInt32Array()), TestData);
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 6, 0, 10, 23, UInt32.MaxValue });
            AssertArray(np.array(abigint.AsUInt32Array()), TestData);
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { (UInt32)6, (UInt32)0, (UInt32)10, (UInt32)23, (UInt32)(UInt32.MaxValue) });
            AssertArray(np.array(aobject.AsUInt32Array()), new UInt32[] { 0, 0, 0, 0, 0 });
            print(aobject);

        }

        [TestMethod]
        public void test_AsInt64()
        {
            var TestData = new Int64[] { 6, 0, 10, 23, Int64.MaxValue };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsInt64Array()), new Int64[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 6, 0, 10, 23, -1 });
            AssertArray(np.array(aSByte.AsInt64Array()), new Int64[] { 6, 0, 10, 23, -1});
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 6, 0, 10, 23, 255 });
            AssertArray(np.array(aUByte.AsInt64Array()), new Int64[] { 6, 0, 10, 23, 255 });
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 6, 0, 10, 23, -1 });
            AssertArray(np.array(aInt16.AsInt64Array()), new Int64[] { 6, 0, 10, 23, -1 });
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 6, 0, 10, 23, 65535 });
            AssertArray(np.array(aUInt16.AsInt64Array()), new Int64[] { 6, 0, 10, 23, 65535 });
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 6, 0, 10, 23, -1 });
            AssertArray(np.array(aInt32.AsInt64Array()), new Int64[] { 6, 0, 10, 23, -1 });
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 6, 0, 10, 23, UInt32.MaxValue });
            AssertArray(np.array(aUInt32.AsInt64Array()), new Int64[] { 6, 0, 10, 23, UInt32.MaxValue });
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, TestData);
            AssertArray(np.array(aInt64.AsInt64Array()), TestData);
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 6, 0, 10, 23, Int64.MaxValue });
            AssertArray(np.array(aUInt64.AsInt64Array()), new Int64[] { 6, 0, 10, 23, Int64.MaxValue });
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 6f, 0f, 10f, 23f, (float)Int64.MaxValue });
            AssertArray(np.array(afloat.AsInt64Array()), new Int64[] { 6, 0, 10, 23, Int64.MinValue });
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 6, 0, 10, 23, Int64.MaxValue });
            AssertArray(np.array(adouble.AsInt64Array()), new Int64[] { 6, 0, 10, 23, Int64.MinValue });
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 6, 0, 10, 23, Int64.MaxValue });
            AssertArray(np.array(adecimal.AsInt64Array()), TestData);
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 6, 0, 10, 23, Int64.MaxValue });
            AssertArray(np.array(acomplex.AsInt64Array()), new Int64[] { 6, 0, 10, 23, Int64.MinValue });
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 6, 0, 10, 23, Int64.MaxValue });
            AssertArray(np.array(abigint.AsInt64Array()), TestData);
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { (Int64)6, (Int64)0, (Int64)10, (Int64)23, (Int64)(Int64.MaxValue) });
            AssertArray(np.array(aobject.AsInt64Array()), new Int64[] { 0, 0, 0, 0, 0 });
            print(aobject);

        }

        [TestMethod]
        public void test_AsUInt64()
        {
            var TestData = new UInt64[] { 6, 0, 10, 23, UInt64.MaxValue };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsUInt64Array()), new UInt64[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 6, 0, 10, 23, -1 });
            AssertArray(np.array(aSByte.AsUInt64Array()), new UInt64[] { 6, 0, 10, 23, UInt64.MaxValue });
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 6, 0, 10, 23, 255 });
            AssertArray(np.array(aUByte.AsUInt64Array()), new UInt64[] { 6, 0, 10, 23, 255 });
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 6, 0, 10, 23, -1 });
            AssertArray(np.array(aInt16.AsUInt64Array()), new UInt64[] { 6, 0, 10, 23, UInt64.MaxValue });
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 6, 0, 10, 23, 65535 });
            AssertArray(np.array(aUInt16.AsUInt64Array()), new UInt64[] { 6, 0, 10, 23, 65535 });
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 6, 0, 10, 23, -1 });
            AssertArray(np.array(aInt32.AsUInt64Array()), new UInt64[] { 6, 0, 10, 23, UInt64.MaxValue });
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 6, 0, 10, 23, UInt32.MaxValue });
            AssertArray(np.array(aUInt32.AsUInt64Array()), new UInt64[] { 6, 0, 10, 23, UInt32.MaxValue });
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 6, 0, 10, 23, -1 });
            AssertArray(np.array(aInt64.AsUInt64Array()), TestData);
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 6, 0, 10, 23, UInt64.MaxValue });
            AssertArray(np.array(aUInt64.AsUInt64Array()), new UInt64[] { 6, 0, 10, 23, UInt64.MaxValue });
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 6f, 0f, 10f, 23f, (float)UInt64.MaxValue });
            AssertArray(np.array(afloat.AsUInt64Array()), new UInt64[] { 6, 0, 10, 23, UInt64.MinValue });
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 6, 0, 10, 23, UInt64.MaxValue });
            AssertArray(np.array(adouble.AsUInt64Array()), new UInt64[] { 6, 0, 10, 23, UInt64.MinValue });
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 6, 0, 10, 23, UInt64.MaxValue });
            AssertArray(np.array(adecimal.AsUInt64Array()), TestData);
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 6, 0, 10, 23, UInt64.MaxValue });
            AssertArray(np.array(acomplex.AsUInt64Array()), new UInt64[] { 6, 0, 10, 23, UInt64.MinValue });
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 6, 0, 10, 23, UInt64.MaxValue });
            AssertArray(np.array(abigint.AsUInt64Array()), TestData);
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { (UInt64)6, (UInt64)0, (UInt64)10, (UInt64)23, (UInt64)(UInt64.MaxValue) });
            AssertArray(np.array(aobject.AsUInt64Array()), new UInt64[] { 0, 0, 0, 0, 0 });
            print(aobject);

        }

        [TestMethod]
        public void test_AsFloat()
        {
            var TestData = new float[] { 6, 0, 10, 23, -25};

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsFloatArray()), new float[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aSByte.AsFloatArray()), new float[] { 6, 0, 10, 23, -25 });
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 6, 0, 10, 23, 231 });
            AssertArray(np.array(aUByte.AsFloatArray()), new float[] { 6, 0, 10, 23, 231 });
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt16.AsFloatArray()), new float[] { 6, 0, 10, 23, -25 });
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 6, 0, 10, 23, 65511 });
            AssertArray(np.array(aUInt16.AsFloatArray()), new float[] { 6, 0, 10, 23, 65511 });
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt32.AsFloatArray()), new float[] { 6, 0, 10, 23, -25 });
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 6, 0, 10, 23, 4294967271 });
            AssertArray(np.array(aUInt32.AsFloatArray()), new float[] { 6, 0, 10, 23, UInt32.MaxValue });
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt64.AsFloatArray()), TestData);
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 6, 0, 10, 23, 18446744073709551591 });
            AssertArray(np.array(aUInt64.AsFloatArray()), new float[] { 6, 0, 10, 23, UInt64.MaxValue });
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 6f, 0f, 10f, 23f, -25 });
            AssertArray(np.array(afloat.AsFloatArray()), TestData);
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(adouble.AsFloatArray()), TestData);
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 6, 0, 10, 23,-25 });
            AssertArray(np.array(adecimal.AsFloatArray()), TestData);
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(acomplex.AsFloatArray()), TestData);
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(abigint.AsFloatArray()), TestData);
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { (float)6, (float)0, (float)10, (float)23, (float)(-25) });
            AssertArray(np.array(aobject.AsFloatArray()), new float[] { 0, 0, 0, 0, 0 });
            print(aobject);

        }

        [TestMethod]
        public void test_AsDouble()
        {
            var TestData = new double[] { 6, 0, 10, 23, -25 };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsDoubleArray()), new double[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aSByte.AsDoubleArray()), TestData);
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 6, 0, 10, 23, 231 });
            AssertArray(np.array(aUByte.AsDoubleArray()), new double[] { 6, 0, 10, 23, 231 });
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt16.AsDoubleArray()), TestData);
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 6, 0, 10, 23, 65511 });
            AssertArray(np.array(aUInt16.AsDoubleArray()), new double[] { 6, 0, 10, 23, 65511 });
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt32.AsDoubleArray()), TestData);
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 6, 0, 10, 23, 4294967271 });
            AssertArray(np.array(aUInt32.AsDoubleArray()), new double[] { 6, 0, 10, 23, 4294967271 });
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt64.AsDoubleArray()), TestData);
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 6, 0, 10, 23, 18446744073709551591 });
            AssertArray(np.array(aUInt64.AsDoubleArray()), new double[] { 6, 0, 10, 23, UInt64.MaxValue });
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 6f, 0f, 10f, 23f, -25 });
            AssertArray(np.array(afloat.AsDoubleArray()), TestData);
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(adouble.AsDoubleArray()), TestData);
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(adecimal.AsDoubleArray()), TestData);
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(acomplex.AsDoubleArray()), TestData);
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(abigint.AsDoubleArray()), TestData);
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { (double)6, (double)0, (double)10, (double)23, (double)(-25) });
            AssertArray(np.array(aobject.AsDoubleArray()), new double[] { 0, 0, 0, 0, 0 });
            print(aobject);

        }

        [TestMethod]
        public void test_AsDecimal()
        {
            var TestData = new decimal[] { 6, 0, 10, 23, -25 };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsDecimalArray()), new decimal[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aSByte.AsDecimalArray()), TestData);
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 6, 0, 10, 23, 0 });
            AssertArray(np.array(aUByte.AsDecimalArray()), new decimal[] { 6, 0, 10, 23, 0 });
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt16.AsDecimalArray()), TestData);
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 6, 0, 10, 23, 0 });
            AssertArray(np.array(aUInt16.AsDecimalArray()), new decimal[] { 6, 0, 10, 23, 0 });
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt32.AsDecimalArray()), TestData);
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 6, 0, 10, 23, 0 });
            AssertArray(np.array(aUInt32.AsDecimalArray()), new decimal[] { 6, 0, 10, 23, 0 });
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt64.AsDecimalArray()), TestData);
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 6, 0, 10, 23, 0 });
            AssertArray(np.array(aUInt64.AsDecimalArray()), new decimal[] { 6, 0, 10, 23, 0 });
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 6f, 0f, 10f, 23f, -25 });
            AssertArray(np.array(afloat.AsDecimalArray()), TestData);
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(adouble.AsDecimalArray()), TestData);
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(adecimal.AsDecimalArray()), TestData);
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(acomplex.AsDecimalArray()), TestData);
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(abigint.AsDecimalArray()), TestData);
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { (Decimal)6, (Decimal)0, (Decimal)10, (Decimal)23, (Decimal)(-25) });
            AssertArray(np.array(aobject.AsDecimalArray()), new Decimal[] { 0, 0, 0, 0, 0 });
            print(aobject);

        }

        [TestMethod]
        public void test_AsComplex()
        {
            var TestData = new Complex[] { 6, 0, 10, 23, -25 };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsComplexArray()), new Complex[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aSByte.AsComplexArray()), TestData);
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 6, 0, 10, 23, 0 });
            AssertArray(np.array(aUByte.AsComplexArray()), new Complex[] { 6, 0, 10, 23, 0 });
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt16.AsComplexArray()), TestData);
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 6, 0, 10, 23, 0 });
            AssertArray(np.array(aUInt16.AsComplexArray()), new Complex[] { 6, 0, 10, 23, 0 });
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt32.AsComplexArray()), TestData);
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 6, 0, 10, 23, 0 });
            AssertArray(np.array(aUInt32.AsComplexArray()), new Complex[] { 6, 0, 10, 23, 0 });
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt64.AsComplexArray()), TestData);
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 6, 0, 10, 23, 0 });
            AssertArray(np.array(aUInt64.AsComplexArray()), new Complex[] { 6, 0, 10, 23, 0 });
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 6f, 0f, 10f, 23f, -25 });
            AssertArray(np.array(afloat.AsComplexArray()), TestData);
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(adouble.AsComplexArray()), TestData);
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(adecimal.AsComplexArray()), TestData);
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(acomplex.AsComplexArray()), TestData);
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(abigint.AsComplexArray()), TestData);
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { (Complex)6, (Complex)0, (Complex)10, (Complex)23, (Complex)(-25) });
            AssertArray(np.array(aobject.AsComplexArray()), new Complex[] { 0, 0, 0, 0, 0 });
            print(aobject);

        }

        [TestMethod]
        public void test_AsBigInt()
        {
            var TestData = new BigInteger[] { 6, 0, 10, 23, -25 };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsBigIntArray()), new BigInteger[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aSByte.AsBigIntArray()), TestData);
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 6, 0, 10, 23, 0 });
            AssertArray(np.array(aUByte.AsBigIntArray()), new BigInteger[] { 6, 0, 10, 23, 0 });
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt16.AsBigIntArray()), TestData);
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 6, 0, 10, 23, 0 });
            AssertArray(np.array(aUInt16.AsBigIntArray()), new BigInteger[] { 6, 0, 10, 23, 0 });
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt32.AsBigIntArray()), TestData);
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 6, 0, 10, 23, 0 });
            AssertArray(np.array(aUInt32.AsBigIntArray()), new BigInteger[] { 6, 0, 10, 23, 0 });
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(aInt64.AsBigIntArray()), TestData);
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 6, 0, 10, 23, 0 });
            AssertArray(np.array(aUInt64.AsBigIntArray()), new BigInteger[] { 6, 0, 10, 23, 0 });
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 6f, 0f, 10f, 23f, -25 });
            AssertArray(np.array(afloat.AsBigIntArray()), TestData);
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(adouble.AsBigIntArray()), TestData);
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(adecimal.AsBigIntArray()), TestData);
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(acomplex.AsBigIntArray()), TestData);
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 6, 0, 10, 23, -25 });
            AssertArray(np.array(abigint.AsBigIntArray()), TestData);
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, new object[] { (BigInteger)6, (BigInteger)0, (BigInteger)10, (BigInteger)23, (BigInteger)(-25) });
            AssertArray(np.array(aobject.AsBigIntArray()), new BigInteger[] { 0, 0, 0, 0, 0 });
            print(aobject);

        }

        [TestMethod]
        public void test_AsObject()
        {
            var TestData = new object[] { "A", null, "10", "23", "-25" };

            var a = np.array(TestData);

            var abool = np.array(a.AsBoolArray());
            AssertArray(abool, new bool[] { true, false, true, true, true });
            AssertArray(np.array(abool.AsObjectArray()), new object[] { 1, 0, 1, 1, 1 });
            print(abool);

            var aSByte = np.array(a.AsSByteArray());
            AssertArray(aSByte, new sbyte[] { 0, 0, 0, 0, 0 });
            //AssertArray(np.array(aSByte.AsObjectArray()), new object[] { 0, 0, 0, 0, 0 });
            print(aSByte);

            var aUByte = np.array(a.AsByteArray());
            AssertArray(aUByte, new byte[] { 0, 0, 0, 0, 0 });
            //AssertArray(np.array(aUByte.AsObjectArray()), new object[] { 0, 0, 0, 0, 0 });
            print(aUByte);

            var aInt16 = np.array(a.AsInt16Array());
            AssertArray(aInt16, new Int16[] { 0, 0, 0, 0, 0 });
            //AssertArray(np.array(aInt16.AsObjectArray()), new object[] { 0, 0, 0, 0, 0 });
            print(aInt16);

            var aUInt16 = np.array(a.AsUInt16Array());
            AssertArray(aUInt16, new UInt16[] { 0, 0, 0, 0, 0 });
            //AssertArray(np.array(aUInt16.AsObjectArray()), new object[] { 0, 0, 0, 0, 0 });
            print(aUInt16);

            var aInt32 = np.array(a.AsInt32Array());
            AssertArray(aInt32, new Int32[] { 0, 0, 0, 0, 0 });
            //AssertArray(np.array(aInt32.AsObjectArray()), new object[] { 0, 0, 0, 0, 0 });
            print(aInt32);

            var aUInt32 = np.array(a.AsUInt32Array());
            AssertArray(aUInt32, new UInt32[] { 0, 0, 0, 0, 0 });
            //AssertArray(np.array(aUInt32.AsObjectArray()), new object[] { 0, 0, 0, 0, 0 });
            print(aUInt32);

            var aInt64 = np.array(a.AsInt64Array());
            AssertArray(aInt64, new Int64[] { 0, 0, 0, 0, 0 });
            //AssertArray(np.array(aInt64.AsObjectArray()), new object[] { 0, 0, 0, 0, 0 });
            print(aInt64);

            var aUInt64 = np.array(a.AsUInt64Array());
            AssertArray(aUInt64, new UInt64[] { 0, 0, 0, 0, 0 });
            //AssertArray(np.array(aUInt64.AsObjectArray()), new object[] { 0, 0, 0, 0, 0 });
            print(aUInt64);

            var afloat = np.array(a.AsFloatArray());
            AssertArray(afloat, new float[] { 0, 0, 0, 0, 0 });
            //AssertArray(np.array(afloat.AsObjectArray()), new object[] { 0, 0, 0, 0, 0 });
            print(afloat);

            var adouble = np.array(a.AsDoubleArray());
            AssertArray(adouble, new double[] { 0, 0, 0, 0, 0 });
            //AssertArray(np.array(adouble.AsObjectArray()), new object[] { 0, 0, 0, 0, 0 });
            print(adouble);

            var adecimal = np.array(a.AsDecimalArray());
            AssertArray(adecimal, new decimal[] { 0, 0, 0, 0, 0 });
            //AssertArray(np.array(adecimal.AsObjectArray()), new object[] { 0, 0, 0, 0, 0 });
            print(adecimal);

            var acomplex = np.array(a.AsComplexArray());
            AssertArray(acomplex, new Complex[] { 0, 0, 0, 0, 0 });
            //AssertArray(np.array(acomplex.AsObjectArray()), new object[] { 0, 0, 0, 0, 0 });
            print(acomplex);

            var abigint = np.array(a.AsBigIntArray());
            AssertArray(abigint, new BigInteger[] { 0, 0, 0, 0, 0 });
            //AssertArray(np.array(abigint.AsObjectArray()), new object[] { 0, 0, 0, 0, 0 });
            print(abigint);

            var aobject = np.array(a.AsObjectArray());
            AssertArray(aobject, TestData);
            //AssertArray(np.array(aobject.AsObjectArray()), TestData);
            print(aobject);

        }

        [TestMethod]
        public void test_ToArray()
        {
            // 2D array tests
            var adata = new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            ndarray a = np.array(adata);

            var a1 = (int[])a.ToArray();
            AssertArray(np.array(a1), new int[] { 1, 2, 3, 4, 5, 6 });

            // 3D array test
            var bdata = new float[,,] { { { 14 }, { 13 }, { 12 }, { 11 } }, { { 18 }, { 17 }, { 16 }, { 15 } }, { { 22 }, { 21 }, { 20 }, { 19 } } };
            ndarray b = np.array(bdata);

            var b1 = (float[])b.ToArray();
            AssertArray(np.array(b1), new float[] { 14,13,12,11,18,17,16,15,22,21,20,19 });

            // 4D array test
            var cdata = new double[,,,] { { { { 1, 0 }, { 3, 2 } }, { { 5, 4 }, { 7, 6 } } }, { { { 9, 8 }, { 11, 10 } }, { { 13, 12 }, { 15, 14 } } } };
            ndarray c = np.array(cdata);

            var c1 = (double[])c.ToArray();
            AssertArray(np.array(c1), new double[] {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14 });

            // 5D array test
            var ddata = new System.Numerics.Complex[,,,,] { { { { { 1, 0 }, { 3, 2 } }, { { 5, 4 }, { 7, 6 } } }, { { { 9, 8 }, { 11, 10 } }, { { 13, 12 }, { 15, 14 } } } } };
            ndarray d = np.array(ddata);

            var d1 = (System.Numerics.Complex[])d.ToArray();
            AssertArray(np.array(d1), new System.Numerics.Complex[] { 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14 });

        }

        [TestMethod]
        public void test_ToSystemArray()
        {
            // 2D array tests
            var adata = new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            ndarray a = np.array(adata);

            var a1 = (int[,])a.ToSystemArray();
            AssertArray(a, adata);
            AssertArray(a, a1);

            // 3D array test
            var bdata = new float[,,] { { { 14 }, { 13 }, { 12 }, { 11 } }, { { 18 }, { 17 }, { 16 }, { 15 } }, { { 22 }, { 21 }, { 20 }, { 19 } } };
            ndarray b = np.array(bdata);

            var b1 = (float[,,])b.ToSystemArray();
            AssertArray(b, bdata);
            AssertArray(b, b1);

            // 4D array test
            var cdata = new double[,,,] { { { { 1, 0 }, { 3, 2 } }, { { 5, 4 }, { 7, 6 } } }, { { { 9, 8 }, { 11, 10 } }, { { 13, 12 }, { 15, 14 } } } };
            ndarray c = np.array(cdata);

            var c1 = (double[,,,])c.ToSystemArray();
            AssertArray(c, cdata);
            AssertArray(c, c1);

            // 5D array test
            var ddata = new System.Numerics.Complex[,,,,] { { { { { 1, 0 }, { 3, 2 } }, { { 5, 4 }, { 7, 6 } } }, { { { 9, 8 }, { 11, 10 } }, { { 13, 12 }, { 15, 14 } } } } };
            ndarray d = np.array(ddata);

            var d1 = (System.Numerics.Complex[,,,,])d.ToSystemArray();
            AssertArray(d, ddata);
            AssertArray(d, d1);


            // crazy array dims
            var e = np.arange(0, 20, dtype: np.BigInt);
            for (int x = 0; x < 31; x++)
            {
                e = np.expand_dims(e, 0);
                var e1 = e.ToSystemArray();
                Assert.AreEqual(e.ndim, e1.Rank);
            }


        }

        [TestMethod]
        public void test_ToSystemArray_Reverse()
        {
            // 2D array tests
            var adata = new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            var adata_revers = new int[,] { { 2, 1 }, { 4, 3 }, { 6, 5 } };
            ndarray a = np.array(adata);
            ndarray ar = a[":", "::-1"] as ndarray;

            var a1 = (int[,])ar.ToSystemArray();
            AssertArray(ar, adata_revers);

            // 3D array test
            var bdata = new float[,,] { { { 14 }, { 13 }, { 12 }, { 11 } }, { { 18 }, { 17 }, { 16 }, { 15 } }, { { 22 }, { 21 }, { 20 }, { 19 } } };
            var bdata_revers = new float[,,] { { { 11 }, { 12 }, { 13 }, { 14 } }, { { 15 }, { 16 }, { 17 }, { 18 } }, { { 19 }, { 20 }, { 21 }, { 22 } } };
            ndarray b = np.array(bdata);

            ndarray br = b[":", "::-1", "::-1"] as ndarray;
            var b1 = (float[,,])br.ToSystemArray();
            AssertArray(br, bdata_revers);

            var bdata_revers2 = new float[,,] { { { 22 }, { 21 }, { 20 }, { 19 } }, { { 18 }, { 17 }, { 16 }, { 15 } }, { { 14 }, { 13 }, { 12 }, { 11 } } };
            br = b["::-1", ":", "::-1"] as ndarray;
            b1 = (float[,,])br.ToSystemArray();
            AssertArray(br, bdata_revers2);

            var c = np.arange(0, 256, dtype: np.Int16).reshape(4, 4, 4, 4);
            var cr = c["::-2", "::-1", "::-2", "::-1"] as ndarray;
            //print(cr);

            var crr = cr["::-1", "::-2", "::-1", "::-2"] as ndarray;
            print(crr);

            var crr_revers = new Int16[,,,] {{{{ 68, 70 },{ 76, 78 }},
                                              {{ 100, 102 },{ 108, 110 }}},
                                              {{{ 196, 198 },{ 204, 206 }},
                                              {{ 228, 230 },{ 236, 238 }}}};

            AssertArray(crr, crr_revers);


            crr[0, 1, 0, 1] = -55;
            crr[1, 1, 1, 1] = -77;
            crr[1, 0, 0, 1] = -88;

            crr_revers = new Int16[,,,] {{{{ 68, 70 },{ 76, 78 }},
                                              {{ 100, -55 },{ 108, 110 }}},
                                              {{{ 196, -88 },{ 204, 206 }},
                                              {{ 228, 230 },{ 236, -77 }}}};

            AssertArray(crr, crr_revers);

            Assert.AreEqual((Int16)(-55), c[1, 2, 1, 2]);
            Assert.AreEqual((Int16)(-77), c[3, 2, 3, 2]);
            Assert.AreEqual((Int16)(-88), c[3, 0, 1, 2]);

        }



        [TestMethod]
        public void test_setitem_byIndex()
        {
            var bdata = new int[,,] { { { 14 }, { 13 }, { 12 }, { 11 } }, { { 18 }, { 17 }, { 16 }, { 15 } }, { { 22 }, { 21 }, { 20 }, { 19 } } };
            ndarray b = np.array(bdata);

            int OriginalValue = (int)b[2, 2, 0];
            Assert.AreEqual(20, OriginalValue);

            b.itemset(2, 2, 0, 99);
            int UpdatedValue = (int)b[2, 2, 0];
            Assert.AreEqual(99, UpdatedValue);

            b.itemset_byindex(new int[] { 2, 2, 0 }, 88);
            UpdatedValue = (int)b[2, 2, 0];
            Assert.AreEqual(88, UpdatedValue);




        }



    }
}
