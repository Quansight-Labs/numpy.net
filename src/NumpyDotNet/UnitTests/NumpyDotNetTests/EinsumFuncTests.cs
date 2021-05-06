using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumpyDotNetTests
{
    [TestClass]
    public class EinsumFuncTests : TestBaseClass
    {

        [Ignore] // not implemented yet
        [TestMethod]
        public void test_einsumpath_1()
        {
            var xx = np.einsum_path();
            return;

        }


        [Ignore] // not implemented yet
        [TestMethod]
        public void test_einsum_1()
        {
            var xx = np.einsum();
            return;
        }


    }
}
