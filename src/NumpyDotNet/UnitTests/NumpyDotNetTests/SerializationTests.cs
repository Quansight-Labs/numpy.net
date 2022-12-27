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

            var DtypeInSerializedFormat = a.Dtype.ToSerializable();

            var adtypeSerialized = SerializationHelper.SerializeNewtonsoftJSON(DtypeInSerializedFormat);
            var newDtypeSerializedFormat = SerializationHelper.DeSerializeNewtonsoftJSON<dtype_serializable>(adtypeSerialized);

            dtype newDtype = new dtype(newDtypeSerializedFormat);

            Assert.AreEqual(a.Dtype.TypeNum, newDtype.TypeNum);
            Assert.AreEqual(a.Dtype.str, newDtype.str);
            Assert.AreEqual(a.Dtype.alignment, newDtype.alignment);
            Assert.AreEqual(a.Dtype.ElementSize, newDtype.ElementSize);
            Assert.AreEqual(a.Dtype.Kind, newDtype.Kind);

        }

        [TestMethod]
        public void test_dtype_serialization_XML()
        {
            var a = np.arange(9).reshape(3, 3);
            AssertArray(a, new int[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });

            dtype_serializable DtypeInSerializedFormat = np.ToSerializable(a.Dtype);

            var adtypeSerialized = SerializationHelper.SerializeXml(DtypeInSerializedFormat);
            var newDtypeSerializedFormat = SerializationHelper.DeserializeXml<dtype_serializable>(adtypeSerialized);

            dtype newDtype = np.FromSerializable(newDtypeSerializedFormat);

            Assert.AreEqual(a.Dtype.TypeNum, newDtype.TypeNum);
            Assert.AreEqual(a.Dtype.str, newDtype.str);
            Assert.AreEqual(a.Dtype.alignment, newDtype.alignment);
            Assert.AreEqual(a.Dtype.ElementSize, newDtype.ElementSize);
            Assert.AreEqual(a.Dtype.Kind, newDtype.Kind);

        }

        [TestMethod]
        public void test_ndarray_serialization_newtonsoft()
        {
            var a = np.arange(9).reshape(3,3);
            AssertArray(a, new int[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });

            var ndArraySerializedFormat = a.ToSerializable();
            var adtypeSerialized = SerializationHelper.SerializeNewtonsoftJSON(ndArraySerializedFormat);
            var newDtypeSerializedFormat = SerializationHelper.DeSerializeNewtonsoftJSON<ndarray_serializable>(adtypeSerialized);

            Console.WriteLine("AA");
            print(adtypeSerialized);

            var b = new ndarray(newDtypeSerializedFormat);

            var ndArraySerializedFormatB = b.ToSerializable();
            var adtypeSerializedB = SerializationHelper.SerializeNewtonsoftJSON(ndArraySerializedFormatB);
            var newDtypeSerializedFormatB = SerializationHelper.DeSerializeNewtonsoftJSON<ndarray_serializable>(adtypeSerializedB);
            Console.WriteLine("\n\nBB");
            print(adtypeSerializedB);


            Assert.AreEqual(a.Dtype.TypeNum, b.Dtype.TypeNum);
            Assert.AreEqual(a.Dtype.str, b.Dtype.str);
            Assert.AreEqual(a.Dtype.alignment, b.Dtype.alignment);
            Assert.AreEqual(a.Dtype.ElementSize, b.Dtype.ElementSize);
            Assert.AreEqual(a.Dtype.Kind, b.Dtype.Kind);

        }

        [TestMethod]
        public void test_ndarray_serialization_XML()
        {
            var a = np.arange(9).reshape(3, 3);
            AssertArray(a, new int[,] { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } });

            ndarray_serializable ndArraySerializedFormat = np.ToSerializable(a);

            var adtypeSerialized = SerializationHelper.SerializeXml(ndArraySerializedFormat);
            var newDtypeSerializedFormat = SerializationHelper.DeserializeXml<ndarray_serializable>(adtypeSerialized);

            ndarray b = np.FromSerializable(newDtypeSerializedFormat);


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


