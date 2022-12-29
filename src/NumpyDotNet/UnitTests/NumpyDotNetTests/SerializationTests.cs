using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumpyLib;

#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNetTests
{
    [TestClass]
    public class SerializationTests : TestBaseClass
    {
        [TestMethod]
        public void test_shape_serialization_newtonsoft()
        {
            var a = np.arange(9).reshape(3, 3);
            AssertArray(a, new int [,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });

            var ashapeSerialized = SerializationHelper.SerializeNewtonsoftJSON(a.shape);

            var newShape = SerializationHelper.DeSerializeNewtonsoftJSON<shape>(ashapeSerialized);

            var b = a.reshape(newShape);
            AssertArray(b, new int[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });

            var c = a.reshape(newShape.iDims);
            AssertArray(c, new int[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });

        }

        [TestMethod]
        public void test_shape_serialization_XML()
        {
            var a = np.arange(9).reshape(3, 3);
            AssertArray(a, new int[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });

            var ashapeSerialized = SerializationHelper.SerializeXml(a.shape);

            var newShape = SerializationHelper.DeserializeXml<shape>(ashapeSerialized);

            var b = a.reshape(newShape);
            AssertArray(b, new int[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });

            var c = a.reshape(newShape.iDims);
            AssertArray(c, new int[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });

        }

        [TestMethod]
        public void test_dtype_serialization_newtonsoft()
        {
            var a = np.arange(9).reshape(3, 3);
            AssertArray(a, new int[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });

            var A_DtypeSerializedFormat = a.Dtype.ToSerializable();

            var A_Serialized = SerializationHelper.SerializeNewtonsoftJSON(A_DtypeSerializedFormat);
            var A_Deserialized = SerializationHelper.DeSerializeNewtonsoftJSON<dtype_serializable>(A_Serialized);

            dtype b = new dtype(A_Deserialized);

            Assert.AreEqual(a.Dtype.TypeNum, b.TypeNum);
            Assert.AreEqual(a.Dtype.str, b.str);
            Assert.AreEqual(a.Dtype.alignment, b.alignment);
            Assert.AreEqual(a.Dtype.ElementSize, b.ElementSize);
            Assert.AreEqual(a.Dtype.Kind, b.Kind);

        }

        [TestMethod]
        public void test_dtype_serialization_XML()
        {
            var a = np.arange(9).reshape(3, 3);
            AssertArray(a, new int[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });

            dtype_serializable A_DtypeSerializedFormat = np.ToSerializable(a.Dtype);

            var A_Serialized = SerializationHelper.SerializeXml(A_DtypeSerializedFormat);
            var A_Deserialized = SerializationHelper.DeserializeXml<dtype_serializable>(A_Serialized);

            dtype b = np.FromSerializable(A_Deserialized);

            Assert.AreEqual(a.Dtype.TypeNum, b.TypeNum);
            Assert.AreEqual(a.Dtype.str, b.str);
            Assert.AreEqual(a.Dtype.alignment, b.alignment);
            Assert.AreEqual(a.Dtype.ElementSize, b.ElementSize);
            Assert.AreEqual(a.Dtype.Kind, b.Kind);

        }

        [TestMethod]
        public void test_ndarray_serialization_newtonsoft()
        {
            var a = np.array(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 }).reshape(3,3);
            AssertArray(a, new int[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });

            var A_ArraySerializedFormat = a.ToSerializable();
            var A_Serialized = SerializationHelper.SerializeNewtonsoftJSON(A_ArraySerializedFormat);
            var A_Deserialized = SerializationHelper.DeSerializeNewtonsoftJSON<ndarray_serializable>(A_Serialized);

            Console.WriteLine("AA");
            print(A_Serialized);

            var b = new ndarray(A_Deserialized);

            var B_ArraySerializedFormat = b.ToSerializable();
            var B_Serialized = SerializationHelper.SerializeNewtonsoftJSON(B_ArraySerializedFormat);
            var B_Deserialized = SerializationHelper.DeSerializeNewtonsoftJSON<ndarray_serializable>(B_Serialized);
            Console.WriteLine("\n\nBB");
            print(B_Serialized);

            Assert.AreEqual(0, string.Compare(A_Serialized, B_Serialized));
            Assert.AreEqual(a.Dtype.TypeNum, b.Dtype.TypeNum);
            Assert.AreEqual(a.Dtype.str, b.Dtype.str);
            Assert.AreEqual(a.Dtype.alignment, b.Dtype.alignment);
            Assert.AreEqual(a.Dtype.ElementSize, b.Dtype.ElementSize);
            Assert.AreEqual(a.Dtype.Kind, b.Dtype.Kind);

        }

        [TestMethod]
        public void test_ndarray_inf_serialization_newtonsoft()
        {
            var a = np.array(new double[] { double.NegativeInfinity, double.PositiveInfinity,
                                            double.NegativeInfinity, double.PositiveInfinity }).reshape(2, 2);

            var A_ArraySerializedFormat = a.ToSerializable();
            var A_Serialized = SerializationHelper.SerializeNewtonsoftJSON(A_ArraySerializedFormat);
            var A_Deserialized = SerializationHelper.DeSerializeNewtonsoftJSON<ndarray_serializable>(A_Serialized);

            Console.WriteLine("AA");
            print(A_Serialized);

            var b = new ndarray(A_Deserialized);
            AssertArray(b, new double[,] { { double.NegativeInfinity, double.PositiveInfinity }, { double.NegativeInfinity, double.PositiveInfinity } });
            //print(b);

            var B_ArraySerializedFormat = b.ToSerializable();
            var B_Serialized = SerializationHelper.SerializeNewtonsoftJSON(B_ArraySerializedFormat);
            var B_Deserialized = SerializationHelper.DeSerializeNewtonsoftJSON<ndarray_serializable>(B_Serialized);
            Console.WriteLine("\n\nBB");
            print(B_Serialized);

            Assert.AreEqual(0, string.Compare(A_Serialized, B_Serialized));
            Assert.AreEqual(a.Dtype.TypeNum, b.Dtype.TypeNum);
            Assert.AreEqual(a.Dtype.str, b.Dtype.str);
            Assert.AreEqual(a.Dtype.alignment, b.Dtype.alignment);
            Assert.AreEqual(a.Dtype.ElementSize, b.Dtype.ElementSize);
            Assert.AreEqual(a.Dtype.Kind, b.Dtype.Kind);

        }

        [TestMethod]
        public void test_ndarray_inf_serialization_xml()
        {
            var a = np.array(new double[] { double.NegativeInfinity, double.PositiveInfinity,
                                            double.NegativeInfinity, double.PositiveInfinity }).reshape(2, 2);

            var A_ArraySerializedFormat = a.ToSerializable();
            var A_Serialized = SerializationHelper.SerializeXml(A_ArraySerializedFormat);
            var A_Deserialized = SerializationHelper.DeserializeXml<ndarray_serializable>(A_Serialized);

            Console.WriteLine("AA");
            print(A_Serialized);

            var b = new ndarray(A_Deserialized);
            AssertArray(b, new double[,] { { double.NegativeInfinity, double.PositiveInfinity }, { double.NegativeInfinity, double.PositiveInfinity } });

            var B_ArraySerializedFormat = b.ToSerializable();
            var B_Serialized = SerializationHelper.SerializeXml(B_ArraySerializedFormat);
            var B_Deserialized = SerializationHelper.DeserializeXml<ndarray_serializable>(B_Serialized);
            Console.WriteLine("\n\nBB");
            print(B_Serialized);

            //Assert.AreEqual(0, string.Compare(A_Serialized, B_Serialized));
            Assert.AreEqual(a.Dtype.TypeNum, b.Dtype.TypeNum);
            Assert.AreEqual(a.Dtype.str, b.Dtype.str);
            Assert.AreEqual(a.Dtype.alignment, b.Dtype.alignment);
            Assert.AreEqual(a.Dtype.ElementSize, b.Dtype.ElementSize);
            Assert.AreEqual(a.Dtype.Kind, b.Dtype.Kind);

        }

        [TestMethod]
        public void test_ndarray_serialization_XML()
        {
            var a = np.array(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 }).reshape(3, 3);
            AssertArray(a, new int[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });

            var A_ArraySerializedFormat = a.ToSerializable();
            var A_Serialized = SerializationHelper.SerializeXml(A_ArraySerializedFormat);
            var A_Deserialized = SerializationHelper.DeserializeXml<ndarray_serializable>(A_Serialized);

            Console.WriteLine("AA");
            print(A_Serialized);

            var b = new ndarray(A_Deserialized);

            var B_ArraySerializedFormat = b.ToSerializable();
            var B_Serialized = SerializationHelper.SerializeXml(B_ArraySerializedFormat);
            var B_Deserialized = SerializationHelper.DeserializeXml<ndarray_serializable>(B_Serialized);
            Console.WriteLine("\n\nBB");
            print(B_Serialized);

            //Assert.AreEqual(0, string.Compare(A_Serialized, B_Serialized));
            Assert.AreEqual(a.Dtype.TypeNum, b.Dtype.TypeNum);
            Assert.AreEqual(a.Dtype.str, b.Dtype.str);
            Assert.AreEqual(a.Dtype.alignment, b.Dtype.alignment);
            Assert.AreEqual(a.Dtype.ElementSize, b.Dtype.ElementSize);
            Assert.AreEqual(a.Dtype.Kind, b.Dtype.Kind);

        }

        [TestMethod]
        public void test_ndarray_serialization_newtonsoft_3layer()
        {
            var a = np.array(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 }).reshape(3, 3);

            var A_ArraySerializedFormat = a.ToSerializable();
            var A_Serialized = SerializationHelper.SerializeNewtonsoftJSON(A_ArraySerializedFormat);
            var A_Deserialized = SerializationHelper.DeSerializeNewtonsoftJSON<ndarray_serializable>(A_Serialized);

            Console.WriteLine("AA");
            print(A_Serialized);

            var b = new ndarray(A_Deserialized);
            AssertArray(b, new int[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });
            b = b.reshape(9, 1);
            //print(b);
            AssertArray(b, new int[,] { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 } });

            var B_ArraySerializedFormat = b.ToSerializable();
            var B_Serialized = SerializationHelper.SerializeNewtonsoftJSON(B_ArraySerializedFormat);
            var B_Deserialized = SerializationHelper.DeSerializeNewtonsoftJSON<ndarray_serializable>(B_Serialized);
            Console.WriteLine("\n\nBB");
            print(B_Serialized);

            var c = np.FromSerializable(B_Deserialized);
            AssertArray(c, new int[,] { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 } });
            c = c.reshape(1, 9);
            //print(c);
            AssertArray(c, new int[,] { { 0, 1, 2, 3, 4, 5, 6, 7, 8 } });

            var C_ArraySerializedFormat = c.ToSerializable();
            var C_Serialized = SerializationHelper.SerializeNewtonsoftJSON(C_ArraySerializedFormat);
            var C_Deserialized = SerializationHelper.DeSerializeNewtonsoftJSON<ndarray_serializable>(C_Serialized);
            Console.WriteLine("\n\nCC");
            print(C_Serialized);
            var d = np.FromSerializable(C_Deserialized);
            AssertArray(d, new int[,] { { 0, 1, 2, 3, 4, 5, 6, 7, 8 } });

            var D_ArraySerializedFormat = d.ToSerializable();
            var D_Serialized = SerializationHelper.SerializeNewtonsoftJSON(D_ArraySerializedFormat);

            Assert.AreEqual(0, string.Compare(C_Serialized, D_Serialized));

            Assert.AreEqual(a.Dtype.TypeNum, c.Dtype.TypeNum);
            Assert.AreEqual(a.Dtype.str, c.Dtype.str);
            Assert.AreEqual(a.Dtype.alignment, c.Dtype.alignment);
            Assert.AreEqual(a.Dtype.ElementSize, c.Dtype.ElementSize);
            Assert.AreEqual(a.Dtype.Kind, c.Dtype.Kind);

        }

        [TestMethod]
        public void test_nprandom_serialization_newtonsoft()
        {
            var Rand1 = new np.random();
            Rand1.seed(1234);

            var Rand1Serialized = SerializationHelper.SerializeNewtonsoftJSON(Rand1.ToSerialization());
            print(Rand1Serialized);

            double fr = Rand1.randn();
            print(fr);
            Assert.AreEqual(0.47143516373249306, fr);
            fr = Rand1.randn();
            print(fr);
            Assert.AreEqual(-1.1909756947064645, fr);

            var Rand1Deserialized = SerializationHelper.DeSerializeNewtonsoftJSON<np.random_serializable>(Rand1Serialized);
            var Rand2 = new np.random();
            Rand2.FromSerialization(Rand1Deserialized);
            fr = Rand2.randn();
            print(fr);
            Assert.AreEqual(0.47143516373249306, fr);
            fr = Rand2.randn();
            print(fr);
            Assert.AreEqual(-1.1909756947064645, fr);


            Rand1Serialized = SerializationHelper.SerializeNewtonsoftJSON(Rand1.ToSerialization());
            print(Rand1Serialized);

            var Rand2Serialized = SerializationHelper.SerializeNewtonsoftJSON(Rand2.ToSerialization());
            print(Rand2Serialized);

            Assert.AreEqual(0, string.Compare(Rand1Serialized, Rand2Serialized));

        }

        [TestMethod]
        public void test_nprandom_serialization_xml()
        {
            var Rand1 = new np.random();
            Rand1.seed(1234);

            var Rand1Serialized = SerializationHelper.SerializeXml(Rand1.ToSerialization());
            print(Rand1Serialized);

            double fr = Rand1.randn();
            print(fr);
            Assert.AreEqual(0.47143516373249306, fr);
            fr = Rand1.randn();
            print(fr);
            Assert.AreEqual(-1.1909756947064645, fr);

            var Rand1Deserialized = SerializationHelper.DeserializeXml<np.random_serializable>(Rand1Serialized);
            var Rand2 = new np.random();
            Rand2.FromSerialization(Rand1Deserialized);
            fr = Rand2.randn();
            print(fr);
            Assert.AreEqual(0.47143516373249306, fr);
            fr = Rand2.randn();
            print(fr);
            Assert.AreEqual(-1.1909756947064645, fr);


            Rand1Serialized = SerializationHelper.SerializeXml(Rand1.ToSerialization());
            print(Rand1Serialized);

            var Rand2Serialized = SerializationHelper.SerializeXml(Rand2.ToSerialization());
            print(Rand2Serialized);

            Assert.AreEqual(0, string.Compare(Rand1Serialized, Rand2Serialized));

        }

        [TestMethod]
        public void test_nprandom_serialization_newtonsoft_2()
        {
            var Rand1 = new np.random();
            Rand1.seed(701);
            ndarray arr1 = Rand1.randint(2, 3, new shape(4), dtype: np.Int32);

            var Rand1Serialized = SerializationHelper.SerializeNewtonsoftJSON(Rand1.ToSerialization());
            var Rand1Deserialized = SerializationHelper.DeSerializeNewtonsoftJSON<np.random_serializable>(Rand1Serialized);
            var Rand2 = new np.random();
            Rand2.FromSerialization(Rand1Deserialized);


            ndarray arr = Rand1.randint(9, 128000, new shape(5000000), dtype: np.Int32);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT32);
            var amax = np.amax(arr);
            Assert.AreEqual((Int32)127999, amax.GetItem(0));

            arr = Rand2.randint(9, 128000, new shape(5000000), dtype: np.Int32);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT32);
            amax = np.amax(arr);
            Assert.AreEqual((Int32)127999, amax.GetItem(0));

            Rand1Serialized = SerializationHelper.SerializeNewtonsoftJSON(Rand1.ToSerialization());
            print(Rand1Serialized);

            var Rand2Serialized = SerializationHelper.SerializeNewtonsoftJSON(Rand2.ToSerialization());
            print(Rand2Serialized);

            Assert.AreEqual(0, string.Compare(Rand1Serialized, Rand2Serialized));

        }


    }


    public static class SerializationHelper
    {
        public static T DeserializeXml<T>(this string toDeserialize)
        {
            System.Xml.Serialization.XmlSerializer xmlSerializer = new System.Xml.Serialization.XmlSerializer(typeof(T));
            using (System.IO.StringReader textReader = new System.IO.StringReader(toDeserialize))
            {
                return (T)xmlSerializer.Deserialize(textReader);
            }
        }

        public static string SerializeXml<T>(this T toSerialize)
        {
            System.Xml.Serialization.XmlSerializer xmlSerializer = new System.Xml.Serialization.XmlSerializer(typeof(T));
            using (System.IO.StringWriter textWriter = new System.IO.StringWriter())
            {
                xmlSerializer.Serialize(textWriter, toSerialize);
                return textWriter.ToString();
            }
        }

        public static string SerializeNewtonsoftJSON<T>(this T toSerialize)
        {
            return Newtonsoft.Json.JsonConvert.SerializeObject(toSerialize);
        }

        public static T DeSerializeNewtonsoftJSON<T>(this string toDeserialize)
        {
            return Newtonsoft.Json.JsonConvert.DeserializeObject<T>(toDeserialize);
        }

    }
}


