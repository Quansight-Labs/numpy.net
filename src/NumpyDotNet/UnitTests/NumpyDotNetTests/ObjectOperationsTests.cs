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
    /// <summary>
    /// functions that test numpy support for objects
    /// </summary>
    [TestClass]
    public class ObjectOperationsTests : TestBaseClass
    {
        [Ignore]
        [TestMethod]
        public void xxx_Test_ObjectOperations_Placeholder()
        {
            string[] TestData = new string[] { "A", "B", "C", "D" };

            var a = np.array(TestData, dtype: np.Object);

            print(a);

            a = a.reshape((2, 2));
            print(a);

            a = a * 2;
            print(a);

        }
    }
}
