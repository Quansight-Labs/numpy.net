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

        }
    }
}
