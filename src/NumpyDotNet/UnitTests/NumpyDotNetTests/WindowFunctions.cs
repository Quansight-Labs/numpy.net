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


        [TestMethod]
        public void test_blackman_1()
        {
            var b = np.blackman(5);
            AssertArray(b, new double[] { -1.38777878078145E-17, 3.40000000e-01, 1.00000000e+00, 3.40000000e-01, -1.38777878078145E-17 });
            print(b);

            b = np.blackman(10);
            AssertArray(b, new double[] { -1.38777878e-17,  5.08696327e-02,  2.58000502e-01,  6.30000000e-01, 9.51129866e-01,
                                           9.51129866e-01,  6.30000000e-01,  2.58000502e-01,  5.08696327e-02, -1.38777878e-17});
            print(b);

            b = np.blackman(12);
            AssertArray(b, new double[] { -1.38777878e-17,  3.26064346e-02,  1.59903635e-01,  4.14397981e-01,  7.36045180e-01,  9.67046769e-01,
                                           9.67046769e-01,  7.36045180e-01,  4.14397981e-01,  1.59903635e-01,  3.26064346e-02, -1.38777878e-17});
            print(b);
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
