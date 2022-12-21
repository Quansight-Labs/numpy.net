using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumpyLib;
using Newtonsoft.Json;
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
            print(a);
            AssertShape(a, 3, 3);

            var ashapeSerialized = JsonConvert.SerializeObject(a.shape);

            var b = a.reshape(JsonConvert.DeserializeObject<shape>(ashapeSerialized));
            print(b);

        }


    }
}
