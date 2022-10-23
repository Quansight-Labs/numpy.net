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
    public class IterToolsTests  
    {
        [TestMethod]
        public void IterTools_Product_1()
        {
            List<object> products = new List<object>();

            foreach (var iter in IterTools.products(new string[] { "ABC", "xy", "12" }, 1))
            {
                products.Add(iter);
            }

            products.Clear();
            foreach (var iter in IterTools.products(new object[] { 0, 1 }, 3))
            {
                products.Add(iter);
            }

        }
    }
}
