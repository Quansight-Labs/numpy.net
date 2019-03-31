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


        [TestMethod]
        public void test_hamming_1()
        {
            var b = np.hamming(5);
            AssertArray(b, new double[] { 0.08, 0.54, 1.0, 0.54, 0.08 });
            print(b);

            b = np.hamming(10);
            AssertArray(b, new double[] { 0.08, 0.18761955616527, 0.460121838273212, 0.77, 0.972258605561518,
                                          0.972258605561518, 0.77, 0.460121838273212, 0.18761955616527, 0.08 });
            print(b);

            b = np.hamming(12);
            AssertArray(b, new double[] { 0.08, 0.153023374897657, 0.348909094019132, 0.605464825605711, 0.841235937614831, 0.981366767862669,
                                          0.981366767862669, 0.841235937614831, 0.605464825605711, 0.348909094019132, 0.153023374897657, 0.08 });
            print(b);
        }


        [TestMethod]
        public void test_hanning_1()
        {
            var b = np.hanning(5);
            AssertArray(b, new double[] { 0.0, 0.5, 1.0, 0.5, 0.0 });
            print(b);

            b = np.hanning(10);
            AssertArray(b, new double[] { 0.0, 0.116977778440511, 0.413175911166535, 0.75, 0.969846310392954, 0.969846310392954,
                                          0.75, 0.413175911166535, 0.116977778440511, 0.0 });
            print(b);

            b = np.hanning(12);
            AssertArray(b, new double[] { 0.0, 0.0793732335844094, 0.292292493499057, 0.571157419136643, 0.827430366972642, 0.979746486807249,
                                          0.979746486807249, 0.827430366972643, 0.571157419136643, 0.292292493499057, 0.0793732335844094, 0.0 });
            print(b);
        }

        [TestMethod]
        public void test_kaiser_1()
        {
            var a = np.kaiser(12, 14);
            AssertArray(a, new double[] {7.72686684e-06, 3.46009194e-03, 4.65200189e-02, 2.29737120e-01, 5.99885316e-01, 9.45674898e-01,
                                         9.45674898e-01, 5.99885316e-01, 2.29737120e-01, 4.65200189e-02, 3.46009194e-03, 7.72686684e-06 });
            print(a);

            a = np.kaiser(3, 5);
            AssertArray(a, new double[] { 0.03671089, 1.0, 0.03671089 });
            print(a);
        }

    }
}
