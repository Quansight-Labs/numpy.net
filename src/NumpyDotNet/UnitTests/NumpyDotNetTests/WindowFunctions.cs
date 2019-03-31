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
    public class WindowFunctions : TestBaseClass
    {
        [TestMethod]
        public void test_bartlett_1()
        {
            var b = np.bartlett(5);
            AssertArray(b, new double[] { 0.0, 0.5, 1.0, 0.5, 0.0 });
            print(b);

            b = np.bartlett(10);
            AssertArray(b, new double[] { 0.0, 0.222222222222222, 0.444444444444444, 0.666666666666667, 0.888888888888889,
                                          0.888888888888889, 0.666666666666667, 0.444444444444444, 0.222222222222222, 0.0 });
            print(b);

            b = np.bartlett(12);
            AssertArray(b, new double[] { 0.0, 0.181818181818182, 0.363636363636364, 0.545454545454545, 0.727272727272727, 0.909090909090909,
                                          0.909090909090909, 0.727272727272727, 0.545454545454545, 0.363636363636364, 0.181818181818182, 0.0});
            print(b);

            return;
        }


        [Ignore]
        [TestMethod]
        public void test_blackman_placeholder()
        {

        }

   
        [Ignore]
        [TestMethod]
        public void test_hanning_placeholder()
        {

        }

        [Ignore]
        [TestMethod]
        public void test_hamming_placeholder()
        {

        }

    }
}
