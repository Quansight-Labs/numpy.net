using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumpyLib;
using System.Threading.Tasks;

namespace NumpyDotNetTests
{
    [TestClass]
    public class ReportedBugs : TestBaseClass
    {
        [TestMethod]
        public void test_multi_index_selection_Jan52020()
        {
            var x = np.arange(10, dtype: np.Int32);
            var y = x.reshape(new shape(2, 5));
            print(y);
            Assert.AreEqual(3, (int)y[0, 3]);
            Assert.AreEqual(8, (int)y[1, 3]);
        }
    }
}
