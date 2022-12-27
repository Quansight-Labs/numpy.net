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


