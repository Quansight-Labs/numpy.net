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
        public void test_AsInt32()
        {
            var TestData = new Int32[] { 6, 0, 10, 23, -23 };
            var UTestData = new Int32[] { 6, 0, 10, 23, 233 };

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
    }
}
