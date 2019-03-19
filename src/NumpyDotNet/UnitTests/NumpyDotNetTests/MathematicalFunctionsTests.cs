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
    public class MathematicalFunctionsTests : TestBaseClass
    {
        [TestMethod]
        public void test_sin_1()
        {
            var ExpectedResult = new double[] { 0, 0.909297426825682, -0.756802495307928, -0.279415498198926, 0.989358246623382 };

            var a = np.arange(0, 10, dtype : np.Float64);
            a = a["::2"] as ndarray;
            var b = np.sin(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype : np.Float32);
            a = a["::2"] as ndarray;
            b = np.sin(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype : np.Int16);
            a = a["::2"] as ndarray;
            b = np.sin(a);
            AssertArray(b, ExpectedResult);
            print(b);

            print("********");

            a = np.arange(0, 10, dtype: np.Float64).reshape((1,2,5));
            a = a["::2"] as ndarray;
            b = np.sin(a);

            var ExpectedDataB = new double[,,]
                {{{ 0,                  0.841470984807897, 0.909297426825682, 0.141120008059867, -0.756802495307928},
                  {-0.958924274663138, -0.279415498198926, 0.656986598718789, 0.989358246623382,  0.412118485241757}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.array(new int[,] {{0, 1, 2, 3, 4},{5,6,7,8,9}});
            a = a["::2"] as ndarray;
            b = np.sin(a, where: a > 2);
            AssertArray(b, new double[] { 0.141120008059867, -0.756802495307928 });
            print(b);

            a = np.array(new int[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.sin(a, where: new bool[,] { { false, false, false, true, true } });
            AssertArray(b, new double[] { 0.141120008059867, -0.756802495307928 });
            print(b);

        }

        [Ignore]
        [TestMethod]
        public void xxx_Test_MathematicalFunctions_Placeholder()
        {
        }


    }
}
