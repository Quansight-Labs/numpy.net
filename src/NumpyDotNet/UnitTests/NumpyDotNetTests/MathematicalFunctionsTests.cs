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
        #region Trigonometric Functions

        [TestMethod]
        public void test_sin_1()
        {
            var ExpectedResult = new double[] { 0, 0.909297426825682, -0.756802495307928, -0.279415498198926, 0.989358246623382 };

            var a = np.arange(0, 10, dtype: np.Float64);
            a = a["::2"] as ndarray;
            var b = np.sin(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype: np.Float32);
            a = a["::2"] as ndarray;
            b = np.sin(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype: np.Int16);
            a = a["::2"] as ndarray;
            b = np.sin(a);
            AssertArray(b, ExpectedResult);
            print(b);

            print("********");

            a = np.arange(0, 10, dtype: np.Float64).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.sin(a);

            var ExpectedDataB = new double[,,]
                {{{ 0,                  0.841470984807897, 0.909297426825682, 0.141120008059867, -0.756802495307928},
                  {-0.958924274663138, -0.279415498198926, 0.656986598718789, 0.989358246623382,  0.412118485241757}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.array(new int[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.sin(a, where: a > 2);
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, 0.141120008059867, -0.756802495307928 } });
            print(b);

            a = np.array(new int[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.sin(a, where: new bool[,] { { false, false, false, true, true } });
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, 0.141120008059867, -0.756802495307928 } });
            print(b);

        }

        [TestMethod]
        public void test_sin_2()
        {

            var a = np.arange(0, 1024 * 1024, dtype: np.Float64).reshape((256, 64, 32, 2));

            var sw1 = new System.Diagnostics.Stopwatch();
            sw1.Start();
            ndarray b = np.sin(a);
            sw1.Stop();

            var sw2 = new System.Diagnostics.Stopwatch();
            sw2.Start();
            ndarray c = np.sin(a);
            sw2.Stop();


            Console.WriteLine("Entries1: {0} Elapsed1: {1}", b.size, sw1.ElapsedMilliseconds);
            Console.WriteLine("Entries2: {0} Elapsed2: {1}", c.size, sw2.ElapsedMilliseconds);

            //Assert.IsTrue(CompareArrays(b, c));
        }

        [TestMethod]
        public void test_sin_3()
        {
            var a = np.arange(0, 5, dtype: np.Float64);
 
            ndarray b = np.sin(a);
            ndarray c = np.sin(a.A("::-1"));

            AssertArray(b, new double[] { 0.0, 0.841470984807897, 0.909297426825682, 0.141120008059867, -0.756802495307928 });
            print(b);
            AssertArray(c, new double[] { -0.756802495307928, 0.141120008059867, 0.909297426825682, 0.841470984807897, 0.0 });
            print(c);
       
        }
        [TestMethod]
        public void test_cos_1()
        {
            var ExpectedResult = new double[] { 1.0, -0.416146836547142, -0.653643620863612, 0.960170286650366, -0.145500033808614 };

            var a = np.arange(0, 10, dtype: np.Float64);
            a = a["::2"] as ndarray;
            var b = np.cos(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype: np.Float32);
            a = a["::2"] as ndarray;
            b = np.cos(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype: np.Int16);
            a = a["::2"] as ndarray;
            b = np.cos(a);
            AssertArray(b, ExpectedResult);
            print(b);

            print("********");

            a = np.arange(0, 10, dtype: np.Float64).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.cos(a);

            var ExpectedDataB = new double[,,]
                {{{ 1.0,               0.54030230586814, -0.416146836547142, -0.989992496600445, -0.653643620863612},
                  { 0.283662185463226, 0.960170286650366, 0.753902254343305, -0.145500033808614, -0.911130261884677}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.array(new int[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.cos(a, where: a > 2);
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, -0.989992496600445, -0.65364362086361 } });
            print(b);

            a = np.array(new int[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.cos(a, where: new bool[,] { { false, false, false, true, true } });
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, -0.989992496600445, -0.65364362086361 } });
            print(b);

        }

        [TestMethod]
        public void test_tan_1()
        {
            var ExpectedResult = new double[] { 0.0, -2.18503986326152, 1.15782128234958, -0.291006191384749, -6.79971145522038 };

            var a = np.arange(0, 10, dtype: np.Float64);
            a = a["::2"] as ndarray;
            var b = np.tan(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype: np.Float32);
            a = a["::2"] as ndarray;
            b = np.tan(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype: np.Int16);
            a = a["::2"] as ndarray;
            b = np.tan(a);
            AssertArray(b, ExpectedResult);
            print(b);

            print("********");

            a = np.arange(0, 10, dtype: np.Float64).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.tan(a);

            var ExpectedDataB = new double[,,]
                {{{ 0.0, 1.5574077246549, -2.18503986326152, -0.142546543074278, 1.15782128234958},
                  { -3.38051500624659, -0.291006191384749, 0.871447982724319, -6.79971145522038, -0.45231565944181}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.array(new int[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.tan(a, where: a > 2);
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, -0.142546543074278, 1.15782128234958 } });
            print(b);

            a = np.array(new int[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.tan(a, where: new bool[,] { { false, false, false, true, true } });
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, -0.142546543074278, 1.15782128234958 } });
            print(b);

        }

        [TestMethod]
        public void test_arcsin_1()
        {
            var ExpectedResult = new double[] { -1.5707963267949, -0.958241588455558, -0.6897750007855, -0.471861837279642,
                                                -0.276226630763592, -0.091034778037415, 0.091034778037415, 0.276226630763592,
                                                 0.471861837279642, 0.6897750007855, 0.958241588455558, 1.5707963267949 };

            double ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            var b = np.arcsin(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arcsin(a);

            var ExpectedDataB = new double[,,]
                {{{ -1.5707963267949, -0.958241588455558, -0.6897750007855},
                  { -0.471861837279642, -0.276226630763592, -0.091034778037415}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            a = a.A("::2");
            b = np.arcsin(a, where: a > -0.5);
            AssertArray(b, new double[] { np.NaN, np.NaN, -0.276226630763592, 0.091034778037415, 0.471861837279642, 0.958241588455558 });
            print(b);

            a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            a = a.A("::2");
            b = np.arcsin(a, where: new bool[] { false, false, true, true, true, true });
            AssertArray(b, new double[] { np.NaN, np.NaN, -0.276226630763592, 0.091034778037415, 0.471861837279642, 0.958241588455558 });
            print(b);

        }

        [TestMethod]
        public void test_arccos_1()
        {
            var ExpectedResult = new double[] { 3.14159265358979, 2.52903791525045, 2.2605713275804, 2.04265816407454,
                                                1.84702295755849, 1.66183110483231, 1.47976154875748, 1.29456969603131,
                                                1.09893448951525, 0.881021326009397, 0.612554738339339, 0.0 };

            double ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            var b = np.arccos(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arccos(a);

            var ExpectedDataB = new double[,,]
                {{{3.14159265358979, 2.52903791525045, 2.2605713275804},
                  {2.04265816407454, 1.84702295755849, 1.66183110483231}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            a = a.A("::2");
            b = np.arccos(a, where: a > -0.5);
            AssertArray(b, new double[] { np.NaN, np.NaN, 1.84702295755849, 1.47976154875748, 1.09893448951525, 0.612554738339339 });
            print(b);

            a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            a = a.A("::2");
            b = np.arccos(a, where: new bool[] { false, false, true, true, true, true });
            AssertArray(b, new double[] { np.NaN, np.NaN, 1.84702295755849, 1.47976154875748, 1.09893448951525, 0.612554738339339 });
            print(b);

        }

        [TestMethod]
        public void test_arctan_1()
        {
            var ExpectedResult = new double[] { -0.785398163397448, -0.685729510906286, -0.566729217523506, -0.426627493126876,
                                                -0.266252049150925, -0.090659887200745, 0.090659887200745,   0.266252049150925,
                                                 0.426627493126876, 0.566729217523506, 0.685729510906286, 0.785398163397448 };

            double ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            var b = np.arctan(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arctan(a);

            var ExpectedDataB = new double[,,]
                {{{-0.785398163397448, -0.685729510906286, -0.566729217523506},
                  {-0.426627493126876, -0.266252049150925, -0.090659887200745}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            a = a.A("::2");
            b = np.arctan(a, where: a > -0.5);
            AssertArray(b, new double[] { np.NaN, np.NaN, -0.266252049150925, 0.090659887200745, 0.426627493126876, 0.685729510906286 });
            print(b);

            a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            a = a.A("::2");
            b = np.arctan(a, where: new bool[] { false, false, true, true, true, true });
            AssertArray(b, new double[] { np.NaN, np.NaN, -0.266252049150925, 0.090659887200745, 0.426627493126876, 0.685729510906286 });
            print(b);

        }

        [TestMethod]
        public void test_hypot_1()
        {

            var a = np.hypot(np.ones((3, 3)) * 3, np.ones((3, 3)) * 4);
            print(a);
            AssertArray(a, new double[,] { { 5, 5, 5 }, { 5, 5, 5 }, { 5, 5, 5 } });

            var b = np.hypot(np.ones((3, 3)) * 3, new int[] { 4 });
            print(b);
            AssertArray(b, new double[,] { { 5, 5, 5 }, { 5, 5, 5 }, { 5, 5, 5 } });

        }

        [TestMethod]
        public void test_arctan2_1()
        {
            var x = np.array(new double[] { -1, +1, +1, -1 });
            var y = np.array(new double[] { -1, -1, +1, +1 });
            var z = np.arctan2(y, x) * 180 / Math.PI;
            AssertArray(z, new double[] { -135.0, -45.0, 45.0, 135.0 });
            print(z);

            var a = np.arctan2(new double[] { 1.0, -1.0 }, new double[] { 0.0, 0.0 });
            AssertArray(a, new double[] { 1.5707963267949, -1.5707963267949 });
            print(a);

            var b = np.arctan2(new double[] { 0.0, 0.0, double.PositiveInfinity }, new double[] { +0.0, -0.0, double.PositiveInfinity });
            AssertArray(b, new double[] { 0.0, 3.14159265358979, double.NaN });
            print(b);

        }

        [TestMethod]
        public void test_degrees_1()
        {
            var rad = np.arange(12.0, dtype: np.Float64) * Math.PI / 6;
            var a = np.degrees(rad);
            AssertArray(a, new double[] { 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330 });
            print(a);

            //var _out = np.zeros((rad.shape));
            //var r = np.degrees(rad, _out);
            //print(np.all(r == _out));

        }

        [TestMethod]
        public void test_radians_1()
        {
            var deg = np.arange(12.0, dtype: np.Float64) * 30.0;
            var a = np.radians(deg);
            AssertArray(a, new double[] { 0.0, 0.523598775598299, 1.0471975511966, 1.5707963267949, 2.0943951023932,
                                         2.61799387799149, 3.14159265358979, 3.66519142918809, 4.18879020478639,
                                        4.71238898038469, 5.23598775598299, 5.75958653158129 });
            print(a);

            //var _out = np.zeros((deg.shape));
            //var r = np.radians(deg, _out);
            //print(np.all(r == _out));

        }
 
        [TestMethod]
        public void test_rad2deg_1()
        {
            var rad = np.arange(12.0, dtype: np.Float64) * Math.PI / 6;
            var a = np.rad2deg(rad);
            AssertArray(a, new double[] { 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330 });
            print(a);

            //var _out = np.zeros((rad.shape));
            //var r = np.degrees(rad, _out);
            //print(np.all(r == _out));

        }

        [TestMethod]
        public void test_deg2rad_1()
        {
            var deg = np.arange(12.0, dtype: np.Float64) * 30.0;
            var a = np.deg2rad(deg);
            AssertArray(a, new double[] { 0.0, 0.523598775598299, 1.0471975511966, 1.5707963267949, 2.0943951023932,
                                         2.61799387799149, 3.14159265358979, 3.66519142918809, 4.18879020478639,
                                        4.71238898038469, 5.23598775598299, 5.75958653158129 });
            print(a);

            //var _out = np.zeros((deg.shape));
            //var r = np.radians(deg, _out);
            //print(np.all(r == _out));

        }

        #endregion

        #region Hyperbolic functions

        [TestMethod]
        public void test_sinh_1()
        {
            var ExpectedResult = new double[] { 0.0, 3.62686040784702, 27.2899171971278, 201.713157370279, 1490.47882578955 };

            var a = np.arange(0, 10, dtype: np.Float64);
            a = a["::2"] as ndarray;
            var b = np.sinh(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype: np.Float32);
            a = a["::2"] as ndarray;
            b = np.sinh(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype: np.Int16);
            a = a["::2"] as ndarray;
            b = np.sinh(a);
            AssertArray(b, ExpectedResult);
            print(b);

            print("********");

            a = np.arange(0, 10, dtype: np.Float64).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.sinh(a);

            var ExpectedDataB = new double[,,]
                {{{ 0.0, 1.1752011936438, 3.62686040784702, 10.0178749274099, 27.2899171971278},
                  {74.2032105777888, 201.713157370279, 548.316123273246, 1490.47882578955, 4051.54190208279}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.array(new int[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.sinh(a, where: a > 2);
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, 10.0178749274099, 27.2899171971278 } });
            print(b);

            a = np.array(new int[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.sinh(a, where: new bool[,] { { false, false, false, true, true } });
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, 10.0178749274099, 27.2899171971278 } });
            print(b);

        }

        [TestMethod]
        public void test_cosh_1()
        {
            var ExpectedResult = new double[] { 1.0, 3.76219569108363, 27.3082328360165, 201.715636122456, 1490.47916125218 };

            var a = np.arange(0, 10, dtype: np.Float64);
            a = a["::2"] as ndarray;
            var b = np.cosh(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype: np.Float32);
            a = a["::2"] as ndarray;
            b = np.cosh(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype: np.Int16);
            a = a["::2"] as ndarray;
            b = np.cosh(a);
            AssertArray(b, ExpectedResult);
            print(b);

            print("********");

            a = np.arange(0, 10, dtype: np.Float64).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.cosh(a);

            var ExpectedDataB = new double[,,]
                {{{ 1.0,               1.54308063481524, 3.76219569108363, 10.0676619957778, 27.3082328360165},
                  { 74.2099485247878, 201.715636122456, 548.317035155212, 1490.47916125218, 4051.54202549259}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.array(new int[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.cosh(a, where: a > 2);
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, 10.0676619957778, 27.3082328360165 } });
            print(b);

            a = np.array(new int[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.cosh(a, where: new bool[,] { { false, false, false, true, true } });
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, 10.0676619957778, 27.3082328360165 } });
            print(b);

        }

        [TestMethod]
        public void test_tanh_1()
        {
            var ExpectedResult = new double[] { 0.0, 0.964027580075817, 0.999329299739067, 0.999987711650796, 0.999999774929676 };

            var a = np.arange(0, 10, dtype: np.Float64);
            a = a["::2"] as ndarray;
            var b = np.tanh(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype: np.Float32);
            a = a["::2"] as ndarray;
            b = np.tanh(a);
            AssertArray(b, ExpectedResult);
            print(b);

            a = np.arange(0, 10, dtype: np.Int16);
            a = a["::2"] as ndarray;
            b = np.tanh(a);
            AssertArray(b, ExpectedResult);
            print(b);

            print("********");

            a = np.arange(0, 10, dtype: np.Float64).reshape((1, 2, 5));
            a = a["::2"] as ndarray;
            b = np.tanh(a);

            var ExpectedDataB = new double[,,]
                {{{ 0.0, 0.761594155955765, 0.964027580075817, 0.99505475368673, 0.999329299739067},
                  { 0.999909204262595, 0.999987711650796, 0.999998336943945, 0.999999774929676, 0.999999969540041}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.array(new int[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.tanh(a, where: a > 2);
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, 0.99505475368673, 0.999329299739067 } });
            print(b);

            a = np.array(new int[,] { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 } });
            a = a["::2"] as ndarray;
            b = np.tanh(a, where: new bool[,] { { false, false, false, true, true } });
            AssertArray(b, new double[,] { { np.NaN, np.NaN, np.NaN, 0.99505475368673, 0.999329299739067 } });
            print(b);
        }

        [TestMethod]
        public void test_arcsinh_1()
        {
            var ExpectedResult = new double[] { -0.881373587019543, -0.7468029948789, -0.599755399970846, -0.440191235352683,
                                                -0.26945474934928, -0.090784335188522, 0.0907843351885222, 0.269454749349279,
                                                 0.440191235352683, 0.599755399970846, 0.7468029948789, 0.881373587019543 };

            double ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            var b = np.arcsinh(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arcsinh(a);

            var ExpectedDataB = new double[,,]
                {{{ -0.881373587019543, -0.7468029948789, -0.599755399970846},
                  { -0.440191235352683, -0.26945474934928, -0.090784335188522}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            a = a.A("::2");
            b = np.arcsinh(a, where: a > -0.5);
            AssertArray(b, new double[] { np.NaN, np.NaN, -0.26945474934928, 0.0907843351885222, 0.440191235352683, 0.7468029948789 });
            print(b);

            a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            a = a.A("::2");
            b = np.arcsinh(a, where: new bool[] { false, false, true, true, true, true });
            AssertArray(b, new double[] { np.NaN, np.NaN, -0.26945474934928, 0.0907843351885222, 0.440191235352683, 0.7468029948789 });
            print(b);

        }

        [TestMethod]
        public void test_arccosh_1()
        {
            var ExpectedResult = new double[] { 0.0, 0.423235459210748, 0.594240703336901, 0.722717193587915,
                                                0.82887090230963, 0.920606859928063, 1.00201733044986, 1.07555476344184,
                                                1.1428302089675, 1.20497120816827, 1.26280443110946, 1.31695789692482 };

            double ref_step = 0;
            var a = np.linspace(1.0, 2.0, ref ref_step, 12);
            var b = np.arccosh(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(1.0, 2.0, ref ref_step, 12).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arccosh(a);

            var ExpectedDataB = new double[,,]
                {{{0.0, 0.423235459210748, 0.594240703336901},
                  {0.722717193587915, 0.82887090230963, 0.920606859928063}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.linspace(1.0, 2.0, ref ref_step, 12);
            a = a.A("::2");
            b = np.arccosh(a, where: a > 1.5);
            AssertArray(b, new double[] { np.NaN, np.NaN, np.NaN, 1.00201733044986, 1.1428302089675, 1.26280443110946 });
            print(b);

            a = np.linspace(1.0, 2.0, ref ref_step, 12);
            a = a.A("::2");
            b = np.arccosh(a, where: new bool[] { false, false, false, true, true, true });
            AssertArray(b, new double[] { np.NaN, np.NaN, np.NaN, 1.00201733044986, 1.1428302089675, 1.26280443110946 });
            print(b);

        }

        [TestMethod]
        public void test_arctanh_1()
        {
            var ExpectedResult = new double[] { double.NegativeInfinity, -1.15129254649702, -0.752038698388137, -0.490414626505863,
                                                     -0.279807893967711, -0.0911607783969772, 0.0911607783969772, 0.279807893967711,
                                                      0.490414626505863, 0.752038698388137, 1.15129254649702, double.PositiveInfinity };

            double ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            var b = np.arctanh(a);
            AssertArray(b, ExpectedResult);
            print(b);


            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12).reshape((2, 2, 3));
            a = a["::2"] as ndarray;
            b = np.arctanh(a);

            var ExpectedDataB = new double[,,]
                {{{double.NegativeInfinity, -1.15129254649702, -0.752038698388137},
                  {-0.490414626505863, -0.279807893967711, -0.0911607783969772}}};

            AssertArray(b, ExpectedDataB);
            print(b);

            print("********");

            a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            a = a.A("::2");
            b = np.arctanh(a, where: a > -0.5);
            AssertArray(b, new double[] { np.NaN, np.NaN, -0.279807893967711, 0.0911607783969772, 0.490414626505863, 1.15129254649702 });
            print(b);

            a = np.linspace(-1.0, 1.0, ref ref_step, 12);
            a = a.A("::2");
            b = np.arctanh(a, where: new bool[] { false, false, true, true, true, true });
            AssertArray(b, new double[] { np.NaN, np.NaN, -0.279807893967711, 0.0911607783969772, 0.490414626505863, 1.15129254649702 });
            print(b);

        }

        #endregion

        #region Rounding Functions

        [TestMethod]
        public void test_around_1()
        {
            ndarray a = np.around(np.array(new double[] { 0.37, 1.64 }));
            print(a);
            AssertArray(a, new double[] { 0, 2 });

            ndarray b = np.around(np.array(new double[] { 0.37, 1.64 }), decimals: 1);
            print(b);
            AssertArray(b, new double[] { 0.4, 1.6 });

            ndarray c = np.around(np.array(new double[] { .5, 1.5, 2.5, 3.5, 4.5 })); // rounds to nearest even value
            print(c);
            AssertArray(c, new double[] { 0.0, 2.0, 2.0, 4.0, 4.0 });

            ndarray d = np.around(np.array(new int[] { 1, 2, 3, 11 }), decimals: 1); // ndarray of ints is returned
            print(d);
            AssertArray(d, new Int32[] { 1, 2, 3, 11 });

            ndarray e = np.around(np.array(new int[] { 1, 2, 3, 11 }), decimals: -1);
            print(e);
            AssertArray(e, new Int32[] { 0, 0, 0, 10 });
        }

        [TestMethod]
        public void test_round_1()
        {
            double ref_step = 0;
            var a = np.linspace(-1.0, 1.0, ref ref_step, 12).reshape((2, 2, 3));
            print(a);

            var ExpectedData1 = new double[,,] { { { -1.0, -0.82, -0.64 }, { -0.45, -0.27, -0.09 } }, { { 0.09, 0.27, 0.45 }, { 0.64, 0.82, 1.0 } } };

            print("********");
            var b = np.round_(a, 2);
            AssertArray(b, ExpectedData1);
            print(b);

            print("********");

            var c = np.round(a, 2);
            AssertArray(c, ExpectedData1);
            print(c);

            var ExpectedData2 = new double[,,] { { { -1.0, -0.8182, -0.6364 }, { -0.4545, -0.2727, -0.0909 } }, { { 0.0909, 0.2727, 0.4545 }, { 0.6364, 0.8182, 1.0 } } };

            print("********");
            b = np.round_(a, 4);
            AssertArray(b, ExpectedData2);
            print(b);

            print("********");

            c = np.round(a, 4);
            AssertArray(c, ExpectedData2);
            print(c);

        }

        [TestMethod]
        public void test_rint_1()
        {
            var a = np.array(new double[] { -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, -4.2 });
            var b = np.rint(a);
            AssertArray(b, new double[] { -2.0, -2.0, 0.0, 0.0, 2.0, 2.0, 2.0, -4.0 });
            print(b);

            b = np.rint(a.reshape((2, 4)));
            AssertArray(b, new double[,] { { -2.0, -2.0, 0.0, 0.0 }, { 2.0, 2.0, 2.0, -4.0 } });
            print(b);

            var x = a > 0.0;
            print(x);

            b = np.rint(a, where: x);
            AssertArray(b, new double[] { double.NaN, double.NaN, double.NaN, 0.0, 2.0, 2.0, 2.0, double.NaN });
            print(b);
        }

        [TestMethod]
        public void test_fix_1()
        {
            var a = np.fix(3.14);
            Assert.AreEqual((double)3.0, a.GetItem(0));
            print(a);

            var b = np.fix(3);
            Assert.AreEqual((int)3, b.GetItem(0));
            print(b);

            var c = np.fix(new double[] { 2.1, 2.9, -2.1, -2.9 });
            AssertArray(c, new double[] { 2.0, 2.0, -2.0, -2.0 });
            print(c);
        }

        [TestMethod]
        public void test_floor_1()
        {
            float[] TestData = new float[] { -1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f };
            var x = np.array(TestData);
            var y = np.floor(x);

            print(x);
            print(y);

            AssertArray(y, new float[] { -2.0f, -2.0f, -1.0f, 0.0f, 1.0f, 1.0f, 2.0f });

        }

        [TestMethod]
        public void test_ceil_1()
        {
            float[] TestData = new float[] { -1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f };
            var x = np.array(TestData);
            var y = np.ceil(x);

            print(x);
            print(y);

            AssertArray(y, new float[] { -1.0f, -1.0f, -0.0f, 1.0f, 2.0f, 2.0f, 2.0f });

        }

        [TestMethod]
        public void test_trunc_1()
        {
            var a = np.trunc(3.14);
            Assert.AreEqual((double)3.0, a.GetItem(0));
            print(a);

            var b = np.trunc(3);
            Assert.AreEqual((int)3, b.GetItem(0));
            print(b);

            var c = np.trunc(new double[] { 2.1, 2.9, -2.1, -2.9 });
            AssertArray(c, new double[] { 2.0, 2.0, -2.0, -2.0 });
            print(c);
        }

        #endregion

        #region Sums, products, differences

        [TestMethod]
        public void test_prod_1()
        {
            //UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            //var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));

            UInt64[] TestData = new UInt64[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt64);

            var y = np.prod(x);

            print(x);
            print(y);
            Assert.AreEqual((UInt64)1403336390625000000, y.GetItem(0));

        }

        [TestMethod]
        public void test_prod_2()
        {
            ndarray a = np.prod(np.array(new double[] { 1.0, 2.0 }));
            print(a);
            Assert.AreEqual((double)2, a.GetItem(0));
            print("*****");

            ndarray b = np.prod(np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }));
            print(b);
            Assert.AreEqual((double)24, b.GetItem(0));
            print("*****");

            ndarray c = np.prod(np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }), axis: 1);
            print(c);
            AssertArray(c, new double[] { 2, 12 });
            print("*****");

            ndarray d = np.array(new byte[] { 1, 2, 3 }, dtype: np.UInt8);
            bool e = np.prod(d).Dtype.TypeNum == NPY_TYPES.NPY_UINT32; 
            print(e);
            Assert.AreEqual(true, e);
            print("*****");

            ndarray f = np.array(new sbyte[] { 1, 2, 3 }, dtype: np.Int8);
            bool g = np.prod(f).Dtype.TypeNum == NPY_TYPES.NPY_INT32; 
            print(g);
            Assert.AreEqual(true, g);

            print("*****");

        }
 
        [TestMethod]
        public void test_prod_3()
        {
            ndarray a = np.array(new sbyte[] { 1, 2, 3 });
            ndarray b = np.prod(a);          // intermediate results 1, 1*2
                                             // total product 1*2*3 = 6
            print(b);
            Assert.AreEqual((Int32)6, b.GetItem(0));
            print("*****");

            a = np.array(new Int32[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            ndarray c = np.prod(a, dtype: np.Float32); //specify type of output
            print(c);
            Assert.AreEqual((float)720, c.GetItem(0));
            print("*****");

            ndarray d = np.prod(a, axis: 0);
            print(d);
            AssertArray(d, new Int64[] { 4, 10, 18 });
            print("*****");


            ndarray e = np.prod(a, axis: 1);
            print(e);
            AssertArray(e, new Int64[] { 6, 120 });
            print("*****");

        }

        [TestMethod]
        public void test_sum_1()
        {
            UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));
            x = x * 3;
            var y = np.sum(x);

            print(x);
            print(y);

            Assert.AreEqual(y.GetItem(0), (UInt32)1578);

        }

        [TestMethod]
        public void test_sum_2()
        {
            UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));
            x = x * 3;

            var y = np.sum(x, axis: 0);
            print(y);
            AssertArray(y, new UInt32[,] { { 339, 450 }, { 339, 450 } });

            print("*****");

            y = np.sum(x, axis: 1);
            print(y);
            AssertArray(y, new UInt32[,] { { 105, 180 }, { 264, 315 }, { 309, 405 } });

            print("*****");

            y = np.sum(x, axis: 2);
            print(y);
            AssertArray(y, new UInt32[,] { { 75, 210 }, { 504, 75 }, { 210, 504 } });

            print("*****");

        }

        [TestMethod]
        public void test_sum_3()
        {
            double[] TestData = new double[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.Float64).reshape(new shape(3, 2, -1));
            x = x * 3.456;

            var y = np.sum(x, axis: 0);
            print(y);
            AssertArray(y, new double[,] { { 390.528, 518.4 }, { 390.528, 518.4 } });
            print("*****");

            y = np.sum(x, axis: 1);
            print(y);
            AssertArray(y, new double[,] { { 120.96, 207.36 }, { 304.128, 362.88 }, { 355.968, 466.56 } });
            print("*****");

            y = np.sum(x, axis: 2);
            print(y);
            AssertArray(y, new double[,] { { 86.4, 241.92 }, { 580.608, 86.4 }, { 241.92, 580.608 } });

            print("*****");

        }

        [TestMethod]
        public void test_sum_keepdims()
        {
            double[] TestData = new double[] { 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.Float64);

            var y = np.sum(x);
            Assert.AreEqual(263.0, y.GetItem(0));
            print(y);
            print(y.shape);
            print("*****");

            print("keepdims");
            y = np.sum(x, keepdims:true);
            AssertArray(y, new double[] { 263 });
            print(y);
            print(y.shape);
            print("*****");

            x = np.array(TestData, dtype: np.Float64).reshape((3, 2, -1));
            y = np.sum(x, axis: 1);
            AssertArray(y, new double[,] { { 25 }, { 70 }, { 168 } });
            print(y);
            print(y.shape);
            print("*****");

            print("keepdims");
            y = np.sum(x, axis: 1, keepdims: true);
            AssertArray(y, new double[,,] { { { 25 } }, { { 70 } }, { { 168 } } });
            print(y);
            print(y.shape);
            print("*****");

            x = np.array(TestData, dtype: np.Float64).reshape((-1, 3, 2));
            y = np.sum(x, axis: 2);
            AssertArray(y, new double[,] {{ 25 , 70 , 168 } });
            print(y);
            print(y.shape);
            print("*****");

            print("keepdims");
            y = np.sum(x, axis: 2, keepdims: true);
            AssertArray(y, new double[,,] { { { 25 }, { 70 }, { 168 } } });
            print(y);
            print(y.shape);
            print("*****");

        }

        [TestMethod]
        public void test_nanprod_placeholder()
        {
            // see the NANFunctionsTest version
        }

        [TestMethod]
        public void test_nansum_placeholder()
        {
            // see the NANFunctionsTest version
        }

        [TestMethod]
        public void test_cumprod_1()
        {
            UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));
            x = x * 3;

            var y = np.cumprod(x);
            print(y);

            AssertArray(y, new UInt32[] {30,1350,101250,13668750,3198487500,303198504,
                            506020528,1296087280,2717265488,1758620720,3495355360,3148109376 });


            Int32[] TestData2 = new Int32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            x = np.array(TestData2, dtype: np.Int32).reshape(new shape(3, 2, -1));
            x = x * 3;

            y = np.cumprod(x);

            print(y);

            AssertArray(y, new Int32[] { 30, 1350, 101250, 13668750, -1096479796, 303198504,
                            506020528, 1296087280, -1577701808, 1758620720, -799611936, -1146857920});
        }

        [TestMethod]
        public void test_cumprod_1a()
        {

            bool CaughtException = false;

            try
            {
                UInt64[] TestData = new UInt64[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
                var x = np.array(TestData, dtype: np.UInt64).reshape(new shape(3, 2, -1));
                x = x * 1;

                var y = np.cumprod(x);
                print(y);

                AssertArray(y, new UInt64[] {10, 150, 3750, 168750, 13162500, 1184625000,
                                11846250000, 177693750000, 4442343750000,199905468750000,
                                15592626562500000, 1403336390625000000  });
            }
            catch (Exception ex)
            {
                CaughtException = true;
            }
            Assert.IsFalse(CaughtException);

            try
            {
                Int64[] TestData2 = new Int64[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
                var x = np.array(TestData2, dtype: np.Int64).reshape(new shape(3, 2, -1));
                x = x * 1;

                var y = np.cumprod(x);

                print(y);

                AssertArray(y, new Int64[] {10, 150, 3750, 168750, 13162500, 1184625000,
                                11846250000, 177693750000, 4442343750000,199905468750000,
                                15592626562500000, 1403336390625000000  });
            }
            catch (Exception ex)
            {
                CaughtException = true;
            }
            Assert.IsFalse(CaughtException);

        }

        [TestMethod]
        public void test_cumprod_2()
        {
            ndarray a = np.array(new Int32[] { 1, 2, 3 });
            ndarray b = np.cumprod(a);          // intermediate results 1, 1*2
                                                // total product 1*2*3 = 6
            print(b);
            AssertArray(b, new Int32[] { 1, 2, 6 });
            print("*****");

            a = np.array(new Int32[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            ndarray c = np.cumprod(a, dtype: np.Float32); //specify type of output
            print(c);
            AssertArray(c, new float[] { 1f, 2f, 6f, 24f, 120f, 720f });
            print("*****");

            ndarray d = np.cumprod(a, axis: 0);
            print(d);
            AssertArray(d, new Int32[,] { { 1, 2, 3 }, { 4, 10, 18 } });
            print("*****");

            ndarray e = np.cumprod(a, axis: 1);
            print(e);
            AssertArray(e, new Int32[,] { { 1, 2, 6 }, { 4, 20, 120 } });
            print("*****");

        }

        [TestMethod]
        public void test_cumsum_1()
        {
            UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));
            x = x * 3;
            var y = np.cumsum(x);

            print(x);
            print(y);

            AssertArray(y, new UInt32[] { 30, 75, 150, 285, 519, 789, 819, 864, 939, 1074, 1308, 1578 });

        }

        [TestMethod]
        public void test_cumsum_2()
        {
            ndarray a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            print(a);
            print("*****");

            ndarray b = np.cumsum(a);
            print(b);
            AssertArray(b, new Int32[] { 1, 3, 6, 10, 15, 21 });
            print("*****");

            ndarray c = np.cumsum(a, dtype: np.Float32);     // specifies type of output value(s)
            print(c);
            AssertArray(c, new float[] { 1f, 3f, 6f, 10f, 15f, 21f });

            print("*****");

            ndarray d = np.cumsum(a, axis: 0);     // sum over rows for each of the 3 columns
            print(d);
            AssertArray(d, new Int32[,] { { 1, 2, 3 }, { 5, 7, 9 } });
            print("*****");


            ndarray e = np.cumsum(a, axis: 1);    // sum over columns for each of the 2 rows
            print(e);
            AssertArray(e, new Int32[,] { { 1, 3, 6 }, { 4, 9, 15 } });
        }

        [TestMethod]
        public void test_cumsum_3()
        {
            ndarray a = np.array(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }).reshape(new shape(2, 3, -1));
            print(a);
            print("*****");

            ndarray b = np.cumsum(a);
            print(b);
            AssertArray(b, new Int32[] { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78 });
            print("*****");

            ndarray c = np.cumsum(a, dtype: np.Float32);     // specifies type of output value(s)
            print(c);
            AssertArray(c, new float[] { 1f, 3f, 6f, 10f, 15f, 21f, 28f, 36f, 45f, 55f, 66f, 78f });
            print("*****");

            ndarray d = np.cumsum(a, axis: 0);     // sum over rows for each of the 3 columns
            print(d);

            var ExpectedDataD = new int[,,]
            {{{1,  2},
              {3,  4},
              {5,  6}},

             {{ 8, 10},
              {12, 14},
              {16, 18}}};

            AssertArray(d, ExpectedDataD);
            print("*****");



            ndarray e = np.cumsum(a, axis: 1);    // sum over columns for each of the 2 rows
            print(e);

            var ExpectedDataE = new int[,,]
            {{{1,  2},
              {4,  6},
              {9,  12}},

             {{ 7, 8},
              {16, 18},
              {27, 30}}};

            AssertArray(e, ExpectedDataE);
            print("*****");

            ndarray f = np.cumsum(a, axis: 2);    // sum over columns for each of the 2 rows
            print(f);

            var ExpectedDataF = new int[,,]
            {{{1,  3},
              {3,  7},
              {5,  11}},

             {{7, 15},
              {9, 19},
              {11, 23}}};

            AssertArray(f, ExpectedDataF);
            print("*****");

        }

        [TestMethod]
        public void test_nancumproduct_placeholder()
        {
            // see the NANFunctionsTest version
        }

        [TestMethod]
        public void test_nancumsum_placeholder()
        {
            // see the NANFunctionsTest version
        }

        [TestMethod]
        public void test_diff_1()
        {
            UInt32[] TestData = new UInt32[6] { 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt32);
            x = x * 3;
            var y = np.diff(x.A("1:"));

            print(x);
            print(y);

            AssertArray(y, new UInt32[] { 30, 60, 99, 36 });
        }

        [TestMethod]
        public void test_diff_2()
        {
            UInt32[] TestData = new UInt32[6] { 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(2, -1));
            x = x * 3;
            var y = np.diff(x, axis: 0);

            print(x);
            print(y);

            AssertArray(y, new UInt32[,] { { 105, 189, 195 } });

        }

        [TestMethod]
        public void test_diff_3()
        {
            UInt32[] TestData = new UInt32[] { 10, 15, 25, 45, 78, 90, 10, 15, 25, 45, 78, 90 };
            var x = np.array(TestData, dtype: np.UInt32).reshape(new shape(3, 2, -1));
            x = x * 3;
            var y = np.diff(x, axis: 2);

            print(x);
            print(y);

            var ExpectedData = new UInt32[,,]
                {
                 {{15},
                  {60}},

                 {{36},
                  {15}},

                 {{60},
                  {36}}
                };

            AssertArray(y, ExpectedData);

        }

        [TestMethod]
        public void test_ediff1d_1()
        {
            ndarray x = np.array(new int[] { 1, 2, 4, 7, 0 });
            ndarray y = np.ediff1d(x);
            print(y);
            AssertArray(y, new int[] { 1, 2, 3, -7 });

            y = np.ediff1d(x, to_begin: np.array(new int[] { -99 }), to_end: np.array(new int[] { 88, 99 }));
            print(y);
            AssertArray(y, new int[] { -99, 1, 2, 3, -7, 88, 99 });

            x = np.array(new int[,] { { 1, 2, 4 }, { 1, 6, 24 } });
            y = np.ediff1d(x);
            print(y);
            AssertArray(y, new int[] { 1, 2, -3, 5, 18 });

        }

        [TestMethod]
        public void test_gradient_1()
        {
            var f = np.array(new double[] { 1, 2, 4, 7, 11, 16 }, dtype: np.Float64);
            var a = np.gradient(f);
            AssertArray(a[0], new double[] { 1, 1.5, 2.5, 3.5, 4.5, 5 });
            print(a[0]);
            print("***********");

            var b = np.gradient(f, new object[] { 2 });
            AssertArray(b[0], new double[] { 0.5, 0.75, 1.25, 1.75, 2.25, 2.5 });
            print(b[0]);
            print("***********");

            // Spacing can be also specified with an array that represents the coordinates
            // of the values F along the dimensions.
            // For instance a uniform spacing:

            var x = np.arange(f.size);
            var c = np.gradient(f, new object[] { x });
            AssertArray(c[0], new double[] { 1.0, 1.5, 2.5, 3.5, 4.5, 5.0 });
            print(c[0]);
            print("***********");

            // Or a non uniform one:

            x = np.array(new double[] { 0.0f, 1.0f, 1.5f, 3.5f, 4.0f, 6.0f }, dtype: np.Float64);
            var d = np.gradient(f, new object[] { x });
            AssertArray(d[0], new double[] { 1.0,  3.0,  3.5, 6.7, 6.9, 2.5 });
            print(d[0]);
        }

        [TestMethod]
        public void test_gradient_2()
        {
            // For two dimensional arrays, the return will be two arrays ordered by
            // axis. In this example the first array stands for the gradient in
            // rows and the second one in columns direction:

            var a = np.gradient(np.array(new int[,] { { 1, 2, 6 }, { 3, 4, 5 } }, dtype: np.Float32));
            AssertArray(a[0], new float[,] { { 2.0f, 2.0f, -1.0f}, { 2.0f, 2.0f, -1.0f} });
            AssertArray(a[1], new float[,] { { 1.0f , 2.5f, 4.0f }, { 1.0f , 1.0f , 1.0f } });
  
            print(a[0]);
            print(a[1]);
            print("***********");

            // In this example the spacing is also specified:
            // uniform for axis=0 and non uniform for axis=1

            var dx = 2.0;
            var y = new double[] { 1.0, 1.5, 3.5 };
            var b = np.gradient(np.array(new int[,] { { 1, 2, 6 }, { 3, 4, 5 } }, dtype: np.Float64), new object[] { dx, y });
            AssertArray(b[0], new double[,] { { 1.0, 1.0, -0.5 }, { 1.0, 1.0, -0.5 } });
            AssertArray(b[1], new double[,] { { 2.0, 2.0, 2.0 }, { 2.0, 1.6999999999999993, 0.5 } });

            print(b[0]);
            print(b[1]);

            print("***********");

            // It is possible to specify how boundaries are treated using `edge_order`

            var x = np.array(new int[] { 0, 1, 2, 3, 4 });
            var f = np.power(x,2);
            var c = np.gradient(f, edge_order: 1);
            AssertArray(c[0], new double[] { 1, 2, 4, 6, 7 });

            print(c[0]);
            print("***********");

            var d = np.gradient(f, edge_order: 2);
            AssertArray(d[0], new double[] { 0, 2, 4, 6, 8 });

            print(d[0]);
            print("***********");

            // The `axis` keyword can be used to specify a subset of axes of which the
            // gradient is calculated

            var e = np.gradient(np.array(new int[,] { { 1, 2, 6 }, { 3, 4, 5 } }, dtype: np.Float32), axes: 0);
            AssertArray(e[0], new float[,] { { 2, 2, -1 }, { 2, 2, -1 } });
            print(e[0]);

        }

        [TestMethod]
        public void test_cross_1()
        {
            // Vector cross-product.
            var x = new int[] { 1, 2, 3 };
            var y = new int[] { 4, 5, 6 };
            var a = np.cross(x, y);
            AssertArray(a, new int[] { -3, 6, -3 });
            print(a);

            // One vector with dimension 2.
            x = new int[] { 1, 2 };
            y = new int[] { 4, 5, 6 };
            var b = np.cross(x, y);
            AssertArray(b, new int[] { 12, -6, -3 });
            print(b);

            // Equivalently:
            x = new int[] { 1, 2, 0 };
            y = new int[] { 4, 5, 6 };
            b = np.cross(x, y);
            AssertArray(b, new int[] { 12, -6, -3 });
            print(b);

            // Both vectors with dimension 2.
            x = new int[] { 1, 2 };
            y = new int[] { 4, 5 };
            var c = np.cross(x, y);
            Assert.AreEqual(-3, c.GetItem(0));
            print(c);

            return;
        }

        [TestMethod]
        public void test_cross_2()
        {
            // Multiple vector cross-products. Note that the direction of the cross
            // product vector is defined by the `right-hand rule`.

            var x = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var y = np.array(new int[,] { { 4, 5, 6 }, { 1, 2, 3 } });
            var a = np.cross(x, y);
            AssertArray(a, new int[,] { { -3, 6, -3 }, { 3, -6, 3 } });
            print(a);


            // The orientation of `c` can be changed using the `axisc` keyword.

            var b = np.cross(x, y, axisc : 0);
            AssertArray(b, new int[,]{{-3,3}, {6,-6}, {-3,3}});
            print(b);

            // Change the vector definition of `x` and `y` using `axisa` and `axisb`.

            x = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
            y = np.array(new int[,] { { 7, 8, 9 }, { 4, 5, 6 }, { 1, 2, 3 } });
            a = np.cross(x, y);
            AssertArray(a, new int[,]{{-6,12, -6}, {0,0,0}, {6,-12, 6}});
            print(a);

            b = np.cross(x, y, axisa: 0, axisb: 0);
            AssertArray(b, new int[,]{{-24, 48, -24},{-30, 60, -30}, {-36,72, -36}});
            print(b);

            return;
        }

        [TestMethod]
        public void test_trapz_1()
        {
            var a = np.trapz(new int[] { 1, 2, 3 });
            Assert.AreEqual((double)4.0, a.GetItem(0));
            print(a);

            var b = np.trapz(new int[] { 1, 2, 3 }, x : new int[] { 4, 6, 8 });
            Assert.AreEqual(8.0, b.GetItem(0));
            print(b);

            var c = np.trapz(new int[] { 1, 2, 3 }, dx: 2);
            Assert.AreEqual(8.0, c.GetItem(0));
            print(c);

            a = np.arange(6).reshape((2, 3));
            b = np.trapz(a, axis: 0);
            AssertArray(b, new double[] { 1.5, 2.5, 3.5 });
            print(b);

            c = np.trapz(a, axis: 1);
            AssertArray(c, new double[] { 2.0, 8.0 });
            print(c);
        }

        #endregion

        #region Exponents and logarithms

        [TestMethod]
        public void test_exp_1()
        {
            var x = np.array(new double[] { 1e-10, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, -4.2 });
            var a = np.exp(x);
            AssertArray(a, new double[] { 1.0, 0.22313016014843, 0.818730753077982, 1.22140275816017,
                                          4.48168907033806, 5.4739473917272, 7.38905609893065, 0.0149955768204777 });
            print(a);


            a = np.exp(x.reshape((2, -1)));
            AssertArray(a, new double[,] { {1.0, 0.22313016014843, 0.818730753077982, 1.22140275816017 },
                                           {4.48168907033806, 5.4739473917272, 7.38905609893065, 0.0149955768204777  } });
            print(a);

            a = np.exp(x, where: x > 0);
            AssertArray(a, new double[] { 1.0, double.NaN, double.NaN, 1.22140275816017,
                                          4.48168907033806, 5.4739473917272, 7.38905609893065, double.NaN });
            print(a);

        }

        [TestMethod]
        public void test_expm1_1()
        {
            var x = np.array(new double[] { 1e-10, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, -4.2 });
            var a = np.expm1(x);
            AssertArray(a, new double[] { 1.00000000005E-10, -0.77686983985157, -0.181269246922018, 0.22140275816017,
                                          3.48168907033806, 4.4739473917272, 6.38905609893065, -0.985004423179522 });
            print(a);


            a = np.expm1(x.reshape((2, -1)));
            AssertArray(a, new double[,] { {1.00000000005E-10, -0.77686983985157, -0.181269246922018, 0.22140275816017 },
                                           {3.48168907033806, 4.4739473917272, 6.38905609893065, -0.985004423179522  } });
            print(a);

            a = np.expm1(x, where: x > 0);
            AssertArray(a, new double[] { 1.00000000005E-10, double.NaN, double.NaN, 0.22140275816017,
                                          3.48168907033806, 4.4739473917272, 6.38905609893065, double.NaN });
            print(a);

        }

        [TestMethod]
        public void test_exp2_1()
        {
            var x = np.array(new double[] { -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, -4.2 });
            var a = np.exp2(x);
            AssertArray(a, new double[] { 0.307786103336229, 0.353553390593274, 0.870550563296124, 1.14869835499704,
                                          2.82842712474619,  3.24900958542494,  4.0,               0.0544094102060078 });
            print(a);


            a = np.exp2(x.reshape((2, -1)));
            AssertArray(a, new double[,] { {0.307786103336229, 0.353553390593274, 0.870550563296124, 1.14869835499704, },
                                           {2.82842712474619,  3.24900958542494,  4.0,               0.0544094102060078  } });
            print(a);

            a = np.exp2(x, where: x > 0);
            AssertArray(a, new double[] { double.NaN, double.NaN, double.NaN, 1.14869835499704,
                                          2.82842712474619,  3.24900958542494,  4.0, double.NaN });
            print(a);

        }

        [TestMethod]
        public void test_log_1()
        {
            var x = np.array(new double[] { 1, Math.E, Math.Pow(Math.E, 2), 0 });
            var a = np.log(x);
            AssertArray(a, new double[] { 0.0, 1.0, 2.0, double.NegativeInfinity });
            print(a);


            a = np.log(x.reshape((2, -1)));
            AssertArray(a, new double[,] { {0.0, 1.0 },
                                           {2.0, double.NegativeInfinity } });
            print(a);

            a = np.log(x, where: x > 0);
            AssertArray(a, new double[] { 0.0, 1.0, 2.0, double.NaN });
            print(a);

        }

        [TestMethod]
        public void test_log10_1()
        {
            var x = np.array(new double[] { 1, Math.E, Math.Pow(Math.E, 2), 0 });
            var a = np.log10(x);
            AssertArray(a, new double[] { 0.0, 0.434294481903252, 0.868588963806504, double.NegativeInfinity });
            print(a);


            a = np.log10(x.reshape((2, -1)));
            AssertArray(a, new double[,] { {0.0, 0.434294481903252 },
                                           {0.868588963806504, double.NegativeInfinity } });
            print(a);

            a = np.log10(x, where: x > 0);
            AssertArray(a, new double[] { 0.0, 0.434294481903252, 0.868588963806504, double.NaN });
            print(a);

        }

        [TestMethod]
        public void test_log2_1()
        {
            var x = np.array(new double[] { 1, Math.E, Math.Pow(Math.E, 2), 0 });
            var a = np.log2(x);
            AssertArray(a, new double[] { 0.0, 1.44269504088896, 2.88539008177793, double.NegativeInfinity });
            print(a);


            a = np.log2(x.reshape((2, -1)));
            AssertArray(a, new double[,] { {0.0, 1.44269504088896 },
                                           {2.88539008177793, double.NegativeInfinity } });
            print(a);

            a = np.log2(x, where: x > 0);
            AssertArray(a, new double[] { 0.0, 1.44269504088896, 2.88539008177793, double.NaN });
            print(a);

        }

        [TestMethod]
        public void test_logn_1()
        {
            var x = np.array(new double[] { 1, Math.E, Math.Pow(Math.E, 2), 0 });
            var a = np.logn(x, 2);
            AssertArray(a, new double[] { 0.0, 1.44269504088896, 2.88539008177793, double.NegativeInfinity });
            print(a);


            a = np.logn(x.reshape((2, -1)), 16);
            AssertArray(a, new double[,] { {0.0, 0.360673760222241 },
                                           {0.721347520444482, double.NegativeInfinity } });
            print(a);

            a = np.logn(x, 32, where: x > 0);
            AssertArray(a, new double[] { 0.0, 0.288539008177793, 0.577078016355585, double.NaN });
            print(a);

        }

        [TestMethod]
        public void test_log1p_1()
        {
            var x = np.array(new double[] { 1, Math.E, Math.Pow(Math.E, 2), 0 });
            var a = np.log1p(x);
            AssertArray(a, new double[] { 0.0, 1.0, 2.0, double.NegativeInfinity });
            print(a);


            a = np.log(x.reshape((2, -1)));
            AssertArray(a, new double[,] { {0.0, 1.0 },
                                           {2.0, double.NegativeInfinity } });
            print(a);

            a = np.log(x, where: x > 0);
            AssertArray(a, new double[] { 0.0, 1.0, 2.0, double.NaN });
            print(a);

        }

        [TestMethod]
        public void test_logaddexp_1()
        {
            var prob1 = np.log(1e-50);
            var prob2 = np.log(2.5e-50);
            var a = np.logaddexp(prob1, prob2);
            AssertArray(a, new double[] { -113.876491681207 });
            print(a);
            var b = np.exp(a);
            AssertArray(b, new double[] { 3.50000000000001E-50 });
            print(b);

        }

        [TestMethod]
        public void test_logaddexp2_1()
        {
            var prob1 = np.log2(1e-50);
            var prob2 = np.log2(2.5e-50);
            var a = np.logaddexp2(prob1, prob2);
            AssertArray(a, new double[] { -164.289049822311 });
            print(a);
            var b = Math.Pow(2, (double)a.GetItem(0));
            Assert.AreEqual(b, 3.4999999999999914E-50);
            print(b);

        }

        [TestMethod]
        public void test_logaddexpn_1()
        {
            var prob1 = np.log2(1e-50);
            var prob2 = np.log2(2.5e-50);
            var a = np.logaddexpn(prob1, prob2, 4);
            AssertArray(a, new double[] { -329.334828493609 });
            print(a);
            var b = Math.Pow(2, (double)a.GetItem(0));
            Assert.AreEqual(b, 7.2500000000000423E-100);
            print(b);

        }

        #endregion

        #region Other special Functions

        [TestMethod]
        public void test_i0_1()
        {
            var a = np.i0(5);
            Assert.AreEqual(27.239871823604442, a.GetItem(0));
            print(a);

            a = np.i0(5.0);
            Assert.AreEqual(27.239871823604442, a.GetItem(0));
            print(a);

            a = np.i0(new double[] { 5.0, 6.0 } );
            AssertArray(a, new double[] { 27.2398718236044, 67.234406976478 });
            print(a);

            a = np.i0(new double[,] { { 27.2398718236044, 67.234406976478 }, { 389.40628328, 427.56411572 } });
            print(a);

            return;

        }

        [TestMethod]
        public void test_sinc_1()
        {
            double retstep = 0;
            var x = np.linspace(-4, 4, ref retstep, 10);
            var a = np.sinc(x);
            AssertArray(a, new double[] {-3.89817183e-17, -3.49934120e-02,  9.20725429e-02, -2.06748336e-01, 7.05316598e-01,
                                          7.05316598e-01, -2.06748336e-01,  9.20725429e-02, -3.49934120e-02, -3.89817183e-17 });
            print(a);

            print("********");

            var xx = np.outer(x, x);
            var b = np.sinc(xx);

            var ExpectedDataB = new double[,]

                {{-3.89817183e-17,  2.51898785e-02,  1.22476942e-02, -5.16870839e-02, -1.15090679e-01,
                  -1.15090679e-01, -5.16870839e-02,  1.22476942e-02,  2.51898785e-02, -3.89817183e-17},
                 { 2.51898785e-02, -2.78216241e-02,  1.23470027e-02,  3.44387931e-02, -2.14755666e-01,
                  -2.14755666e-01,  3.44387931e-02,  1.23470027e-02, -2.78216241e-02,  2.51898785e-02},
                 { 1.22476942e-02,  1.23470027e-02,  1.24217991e-02,  1.24718138e-02,  1.24968663e-02,
                   1.24968663e-02,  1.24718138e-02,  1.24217991e-02,  1.23470027e-02,  1.22476942e-02},
                 {-5.16870839e-02,  3.44387931e-02,  1.24718138e-02, -1.15090679e-01,  5.14582086e-01,
                   5.14582086e-01, -1.15090679e-01,  1.24718138e-02,  3.44387931e-02, -5.16870839e-02},
                 {-1.15090679e-01, -2.14755666e-01,  1.24968663e-02,  5.14582086e-01,  9.37041792e-01,
                   9.37041792e-01,  5.14582086e-01,  1.24968663e-02, -2.14755666e-01, -1.15090679e-01},
                 {-1.15090679e-01, -2.14755666e-01,  1.24968663e-02,  5.14582086e-01,  9.37041792e-01,
                   9.37041792e-01,  5.14582086e-01,  1.24968663e-02, -2.14755666e-01, -1.15090679e-01},
                 {-5.16870839e-02,  3.44387931e-02,  1.24718138e-02, -1.15090679e-01,  5.14582086e-01,
                   5.14582086e-01, -1.15090679e-01,  1.24718138e-02,  3.44387931e-02, -5.16870839e-02},
                 { 1.22476942e-02,  1.23470027e-02,  1.24217991e-02,  1.24718138e-02,  1.24968663e-02,
                   1.24968663e-02,  1.24718138e-02,  1.24217991e-02,  1.23470027e-02,  1.22476942e-02},
                 { 2.51898785e-02, -2.78216241e-02,  1.23470027e-02,  3.44387931e-02, -2.14755666e-01,
                  -2.14755666e-01,  3.44387931e-02,  1.23470027e-02, -2.78216241e-02,  2.51898785e-02},
                 { -3.89817183e-17,  2.51898785e-02,  1.22476942e-02, -5.16870839e-02, -1.15090679e-01,
                  -1.15090679e-01, -5.16870839e-02,  1.22476942e-02,  2.51898785e-02, -3.89817183e-17} };

            AssertArray(b, ExpectedDataB);

            print(b);

        }

        #endregion

        #region Floating point routines

        [TestMethod]
        public void test_signbit_1()
        {
            var a = np.signbit(-1.2);
            Assert.AreEqual(true, a.GetItem(0));
            print(a);

            var b = np.signbit(np.array(new double[] {1, -2.3, 2.1}));
            AssertArray(b, new bool[] {false, true, false});
            print(b);

            var c = np.signbit(np.array(new double[] { +0.0, -0.0}));  // note: different result than python.  No such thing as -0.0
            AssertArray(c, new bool[] { false, false });
            print(c);

            var d = np.signbit(np.array(new float[] { float.NegativeInfinity, float.PositiveInfinity }));
            AssertArray(d, new bool[] { true, false });
            print(d);

            var e = np.signbit(np.array(new double[] { -double.NaN, double.NaN })); // note: different result.  No such thing as -NaN
            AssertArray(e, new bool[] { false, false });
            print(e);

            var f = np.signbit(np.array(new int[] { -1, 0, 1}));
            AssertArray(f, new bool[] { true, false, false });
            print(f);
        }

        [TestMethod]
        public void test_copysign_1()
        {
            var a = np.copysign(1.3, -1);
            Assert.AreEqual(-1.3, a.GetItem(0));
            print(a);

            var b = np.divide(1 , np.copysign(0, 1));
            Assert.AreEqual(0, b.GetItem(0));  // note: python gets a np.inf value here
            print(b);

            var c = 1 / np.copysign(0, -1);
            Assert.AreEqual(0, c.GetItem(0));  // note: python gets a -np.inf value here
            print(c);


            var d = np.copysign(new int[] { -1, 0, 1 }, -1.1);
            AssertArray(d, new int[] { -1, 0, -1 });
            print(d);

            var e = np.copysign(new int[] { -1, 0, 1 }, np.arange(3) - 1);
            AssertArray(e, new int[] { -1, 0, 1});
            print(e);
        }

        [TestMethod]
        public void test_frexp_1()
        {
            var x = np.arange(9);
            var results = np.frexp(x);

            AssertArray(results[0], new double[] { 0.0, 0.5, 0.5, 0.75, 0.5, 0.625, 0.75, 0.875, 0.5 });
            AssertArray(results[1], new int[] { 0, 1, 2, 2, 3, 3, 3, 3, 4 });

            print(results[0]);
            print(results[1]);

            print("***************");

            x = np.arange(9, dtype: np.Float32).reshape((3,3));
            results = np.frexp(x);

            //AssertArray(results[0], new float[,] { { 0.0f, 0.5f, 0.5f }, { 0.75f, 0.5f, 0.625f }, { 0.75f, 0.875f, 0.5f } });
            //AssertArray(results[1], new int[,] { { 0, 1, 2 }, { 2, 3, 3 }, { 3, 3, 4 } });

            print(results[0]);
            print(results[1]);

            print("***************");

            x = np.arange(9, dtype: np.Float64).reshape((3, 3));
            results = np.frexp(x, where: x < 5);

            AssertArray(results[0], new double[,] { { 0.0, 0.5, 0.5 }, { 0.75, 0.5, double.NaN }, { double.NaN, double.NaN, double.NaN } });
            AssertArray(results[1], new int[,] { { 0, 1, 2 }, { 2, 3, 0 }, { 0, 0, 0 } });

            print(results[0]);
            print(results[1]);
        }

        [TestMethod]
        public void test_ldexp_1()
        {
            var a = np.ldexp(5, np.arange(4));
            //AssertArray(a, new float[] { 5.0f, 10.0f, 20.0f, 40.0f });
            //print(a);

            var b = np.ldexp(np.arange(4, dtype: np.Int64), 5);
            AssertArray(b, new double[] { 0.0, 32.0, 64.0, 96.0 });
            print(b);
        }

        [TestMethod]
        public void test_nextafter_1()
        {
            var a = np.nextafter(1, 2);
            double d = (double)a.GetItem(0);
            Assert.AreEqual(d, 1.0000000000000002);
            print(d);

            var b = np.nextafter(new int[] { 1, 2 }, new int[] { 2, 1 });
            double d1 = (double)b.GetItem(0);
            double d2 = (double)b.GetItem(1);
            Assert.AreEqual(d1, 1.0000000000000002);
            Assert.AreEqual(d2, 1.9999999999999998);

            print(d1);
            print(d2);

            var c = np.nextafter(new float[] { 1f, 2f }, new float[] { 2f, 1f });
            float f1 = (float)c.GetItem(0);
            float f2 = (float)c.GetItem(1);
            //Assert.AreEqual(f1, 1.0000001f);
            //Assert.AreEqual(f2, 1.99999988f);

            print(f1);
            print(f2);


        }

        #endregion

        #region Rational routines

        [TestMethod]
        public void test_lcm_1()
        {
            var a = np.lcm(12, 20);
            Assert.AreEqual(60, a.GetItem(0));
            print(a);

            //var b = np.lcm.reduce(new int[] { 3, 12, 20 }); // todo: need to implement reduce functionality
            //print(b);

            //var c = np.lcm.reduce(new int[] { 40, 12, 20 }); // tod: need to implement reduce functionality
            //print(c);

            var d = np.lcm(np.arange(6), new int[] { 20 });
            AssertArray(d, new int[] { 0, 20, 20, 60, 20, 20 });
            print(d);

            var e = np.lcm(new int[] { 20, 21 }, np.arange(6).reshape((3, 2)));
            AssertArray(e, new int[,] { { 0, 21 }, { 20, 21 }, { 20, 105 } });
            print(e);

            var f = np.lcm(new long[] { 20, 21 }, np.arange(6, dtype: np.Int64).reshape((3, 2)));
            AssertArray(f, new long[,] { { 0, 21 }, { 20, 21 }, { 20, 105 } });
            print(f);
        }

        [TestMethod]
        public void test_gcd_1()
        {

            var a = np.gcd(12, 20);
            Assert.AreEqual(4, a.GetItem(0));
            print(a);

            //var b = np.gcd.reduce(new int[] { 3, 12, 20 }); // todo: need to implement reduce functionality
            //print(b);

            //var c = np.gcd.reduce(new int[] { 40, 12, 20 }); // tod: need to implement reduce functionality
            //print(c);

            var d = np.gcd(np.arange(6), new int[] { 20 });
            AssertArray(d, new int[] { 20, 1, 2, 1, 4, 5 });
            print(d);

            var e = np.gcd(new int[] { 20, 20 }, np.arange(6).reshape((3, 2)));
            AssertArray(e, new int[,] { { 20, 1 }, { 2, 1 }, { 4, 5 } });
            print(e);

            var f = np.gcd(new long[] { 20, 20 }, np.arange(6, dtype: np.Int64).reshape((3, 2)));
            AssertArray(f, new long[,] { { 20, 1 }, { 2, 1 }, { 4, 5 } });
            print(f);
        }

        #endregion

        #region Arithmetic operations

        [TestMethod]
        public void test_add_1()
        {
            var a = np.add(1.0, 4.0);
            Assert.AreEqual(5.0, a.GetItem(0));
            print(a);

            var b = np.arange(9.0).reshape((3, 3));
            var c = np.arange(3.0);
            var d = np.add(b, c);
            AssertArray(d, new float[,] { { 0, 2, 4 }, { 3, 5, 7 }, { 6, 8, 10 } });
            print(d);

        }

        [TestMethod]
        public void test_reciprocal_operations()
        {
            var a = np.arange(1, 32, 1, dtype: np.Float32);
            print(a);

            var b = np.reciprocal(a);
            print(b);

            var ExpectedDataB1 = new float[]
            {
                1.0f, 0.5f,        0.33333334f, 0.25f,       0.2f,        0.16666667f,
                0.14285715f,       0.125f,      0.11111111f, 0.1f,        0.09090909f, 0.08333334f,
                0.07692308f,       0.07142857f, 0.06666667f, 0.0625f,     0.05882353f, 0.05555556f,
                0.05263158f,       0.05f,       0.04761905f, 0.04545455f, 0.04347826f, 0.04166667f,
                0.04f,             0.03846154f, 0.03703704f, 0.03571429f, 0.03448276f, 0.03333334f,
                0.03225806f
            };

            AssertArray(b, ExpectedDataB1);


            a = np.arange(2048, 2048 + 32, 1, dtype: np.Float64);
            print(a);

            b = np.reciprocal(a);
            print(b);

            var ExpectedDataB2 = new double[]
            {
                0.00048828, 0.00048804, 0.0004878,  0.00048757, 0.00048733, 0.00048709,
                0.00048685, 0.00048662, 0.00048638, 0.00048614, 0.00048591, 0.00048567,
                0.00048544, 0.0004852,  0.00048497, 0.00048473, 0.0004845,  0.00048426,
                0.00048403, 0.00048379, 0.00048356, 0.00048333, 0.00048309, 0.00048286,
                0.00048263, 0.00048239, 0.00048216, 0.00048193, 0.0004817,  0.00048146,
                0.00048123, 0.000481
            };
            AssertArray(b, ExpectedDataB2);
        }

        [TestMethod]
        public void test_positive_1()
        {
            var d = np.positive(new int[] { -1, -0, 1 });
            AssertArray(d, new int[] { -1, -0, 1 });
            print(d);

            var e = np.positive(new int[,] { { 1, 0, -1 }, { -2, 3, -4 } });
            AssertArray(e, new int[,] { { 1, 0, -1 }, { -2, 3, -4 } });
            print(e);
        }

        [TestMethod]
        public void test_negative_1()
        {
            var d = np.negative(new int[] { -1, -0, 1 });
            AssertArray(d, new int[] { 1, 0, -1 });
            print(d);

            var e = np.negative(new int[,] { { 1, 0, -1 }, { -2, 3, -4 } });
            AssertArray(e, new int[,] { { -1, 0, 1 }, { 2, -3, 4 } });
            print(e);
        }
 
        [TestMethod]
        public void test_multiply_1()
        {
            var a = np.multiply(2.0, 4.0);
            Assert.AreEqual(8.0, a.GetItem(0));
            print(a);

            var b = np.arange(9.0, dtype: np.Float64).reshape((3, 3));
            var c = np.arange(3.0, dtype: np.Float64);
            var d = np.multiply(b, c);
            AssertArray(d, new double[,] { { 0, 1, 4 }, { 0, 4, 10 }, { 0, 7, 16 } });
            print(d);
        }

        [TestMethod]
        public void test_divide()
        {
            var a = np.divide(7, 3);
            Assert.AreEqual(2, a.GetItem(0));
            print(a);

            var b = np.divide(new double[] { 1.0, 2.0, 3.0, 4.0 }, 2.5);
            AssertArray(b, new double[] { 0.4, 0.8, 1.2, 1.6 });
            print(b);

            var c = np.divide(new double[] { 1.0, 2.0, 3.0, 4.0 }, new double[] { 0.5, 2.5, 2.5, 3.5 });
            AssertArray(c, new double[] { 2.0, 0.8, 1.2, 1.14285714 });
            print(c);

            return;
        }

        [TestMethod]
        public void test_power_operations()
        {
            var a = np.arange(0, 32, 1, dtype: np.Int16);
            print(a);

            var b = np.power(a, 3.23);
            print(b);

            var ExpectedDataB1 = new double[]
                { 0.0, 1.0, 9.38267959385503, 34.7617516700826, 88.0346763609436, 180.997724101542,
                  326.15837804154, 536.619770563306, 826.001161443457, 1208.37937917249, 1698.24365246174,
                  2310.45956851781, 3060.23955801521, 3963.11822364251, 5034.9313709235, 6291.79793806794,
                  7750.10424197608, 9426.49010646868, 11337.8365425969, 13501.2547250997, 15934.0760633466,
                  18653.8432055784, 21678.3018459592, 25025.3932276144, 28713.2472532973, 32760.176129938,
                  37184.6684850056, 42005.3839020428, 47241.1478304245, 52910.9468307066, 59033.9241221693,
                  65629.3754035258 };

            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Int64);
            print(a);

            b = np.power(a, 4);
            print(b);

            var ExpectedDataB2 = new double[]
            {
             17592186044416, 17626570956801, 17661006250000, 17695491973201,
             17730028175616, 17764614906481, 17799252215056, 17833940150625,
             17868678762496, 17903468100001, 17938308212496, 17973199149361,
             18008140960000, 18043133693841, 18078177400336, 18113272128961,
             18148417929216, 18183614850625, 18218862942736, 18254162255121,
             18289512837376, 18324914739121, 18360368010000, 18395872699681,
             18431428857856, 18467036534241, 18502695778576, 18538406640625,
             18574169170176, 18609983417041, 18645849431056, 18681767262081
            };

            AssertArray(b, ExpectedDataB2);

            b = np.power(a, 0);
            print(b);
            var ExpectedDataB3 = new double[]
            {
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            };
            AssertArray(b, ExpectedDataB3);


            b = np.power(a, 0.5);
            print(b);

            var ExpectedDataB4 = new double[]
            {
                45.254834,   45.26588119, 45.27692569, 45.2879675,  45.29900661, 45.31004304,
                45.32107677, 45.33210783, 45.3431362,  45.35416188, 45.36518489, 45.37620522,
                45.38722287, 45.39823785, 45.40925016, 45.4202598,  45.43126677, 45.44227107,
                45.45327271, 45.46427169, 45.475268,   45.48626166, 45.49725266, 45.50824101,
                45.51922671, 45.53020975, 45.54119015, 45.5521679,  45.563143,   45.57411546,
                45.58508528, 45.59605246
            };

            AssertArray(b, ExpectedDataB4);

        }

        [TestMethod]
        public void test_subtract_1()
        {
            var a = np.subtract(1.0, 4.0);
            Assert.AreEqual(-3.0, a.GetItem(0));
            print(a);

            var b = np.arange(9.0, dtype: np.Float64).reshape((3, 3));
            var c = np.arange(3.0, dtype: np.Float64);
            var d = np.subtract(b, c);
            AssertArray(d, new double[,] { { 0, 0, 0 }, { 3, 3, 3 }, { 6, 6, 6 } });
            print(d);
        }

        [TestMethod]
        public void test_true_divide()
        {
            var a = np.true_divide(7, 3);
            Assert.AreEqual(2.3333333333333335, a.GetItem(0));
            print(a);

            var b = np.true_divide(new double[] { 1.0, 2.0, 3.0, 4.0 }, 2.5);
            AssertArray(b, new double[] { 0.4, 0.8, 1.2, 1.6 });
            print(b);

            var c = np.true_divide(new double[] { 1.0, 2.0, 3.0, 4.0 }, new double[] { 0.5, 2.5, 2.5, 3.5 });
            AssertArray(c, new double[] { 2.0, 0.8, 1.2, 1.14285714 });
            print(c);

            return;
        }

        [TestMethod]
        public void test_floor_divide()
        {
            var a = np.floor_divide(7, 3);
            Assert.AreEqual(2, a.GetItem(0));
            print(a);

            var b = np.floor_divide(new double[] { 1.0, 2.0, 3.0, 4.0 }, 2.5);
            AssertArray(b, new double[] { 0, 0, 1, 1 });
            print(b);

            var c = np.floor_divide(new double[] { 1.0, 2.0, 3.0, 4.0 }, new double[] { 0.5, 2.5, 2.5, 3.5 });
            AssertArray(c, new double[] { 2, 0, 1, 1 });
            print(c);

            return;

        }

        [TestMethod]
        public void test_float_power()
        {
            var x1 = new int[] { 0, 1, 2, 3, 4, 5 };

            var a = np.float_power(x1, 3);
            AssertArray(a, new double[] { 0.0, 1.0, 8.0, 27.0, 64.0, 125.0 });
            print(a);

            var x2 = new double[] { 1.0, 2.0, 3.0, 3.0, 2.0, 1.0 };
            var b = np.float_power(x1, x2);
            AssertArray(b, new double[] { 0.0, 1.0, 8.0, 27.0, 16.0, 5.0 });
            print(b);

            var x3 = np.array(new double[,] { { 1, 2, 3, 3, 2, 1 }, { 1, 2, 3, 3, 2, 1 } });
            var c = np.float_power(x1, x3);
            AssertArray(c, new double[,] { { 0.0, 1.0, 8.0, 27.0, 16.0, 5.0 }, { 0.0, 1.0, 8.0, 27.0, 16.0, 5.0 } });
            print(c);

            return;
        }

        [TestMethod]
        public void test_fmod_1()
        {
            var x = np.fmod(new int[] { 4, 7 }, new int[] { 2, 3 });
            AssertArray(x, new int[] { 0, 1 });
            print(x);

            var y = np.fmod(np.arange(7), 5);
            AssertArray(y, new int[] { 0, 1, 2, 3, 4, 0, 1 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_fmod_2()
        {
            var x = np.fmod(new int[] { -4, -7 }, new int[] { 2, 3 });
            AssertArray(x, new int[] { 0, -1 });
            print(x);

            var y = np.fmod(np.arange(7), -5);
            AssertArray(y, new int[] { 0, 1, 2, 3, 4, 0, 1 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_mod_1()
        {
            var x = np.mod(new int[] { 4, 7 }, new int[] { 2, 3 });
            AssertArray(x, new int[] { 0, 1 });
            print(x);

            var y = np.mod(np.arange(7), 5);
            AssertArray(y, new int[] { 0, 1, 2, 3, 4, 0, 1 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_modf_1()
        {
            var x = np.modf(new double[] { 0, 3.5 });
            AssertArray(x[0], new double[] { 0, 0.5 });
            AssertArray(x[1], new double[] { 0, 3.0 });
            print(x);

            var y = np.modf(np.arange(7));
            AssertArray(y[0], new double[] { 0, 0, 0,0,0,0,0 });
            AssertArray(y[1], new double[] { 0, 1,2,3,4,5,6 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_remainder_1()
        {
            var x = np.remainder(new int[] { 4, 7 }, new int[] { 2, 3 });
            AssertArray(x, new int[] { 0, 1 });
            print(x);

            var y = np.remainder(np.arange(7), 5);
            AssertArray(y, new int[] { 0, 1, 2, 3, 4, 0, 1 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_remainder_2()
        {
            var x = np.remainder(new int[] { -4, -7 }, new int[] { 2, 3 });
            AssertArray(x, new int[] { 0, 2 });
            print(x);

            var y = np.remainder(np.arange(7), -5);
            AssertArray(y, new int[] { 0, -4, -3, -2, -1, 0, -4 });
            print(y);

            return;
        }

        [TestMethod]
        public void test_divmod_1()
        {
            var a = np.divmod(7, 3);
            Assert.AreEqual(2, a[0].GetItem(0));
            Assert.AreEqual(1, a[1].GetItem(0));

            print(a);

            var b = np.divmod(new double[] { 1.0, 2.0, 3.0, 4.0 }, 2.5);
            AssertArray(b[0], new double[] { 0, 0, 1, 1 });
            AssertArray(b[1], new double[] { 1, 2, 0.5, 1.5 });
            print(b);

            var c = np.divmod(new double[] { 1.0, 2.0, 3.0, 4.0 }, new double[] { 0.5, 2.5, 2.5, 3.5 });
            AssertArray(c[0], new double[] { 2, 0, 1, 1 });
            AssertArray(c[1], new double[] { 0, 2, 0.5, 0.5 });
            print(c);

            return;

        }

 
        #endregion

        #region Handling complex numbers

        // see the functions in ComplexNumbersTests

        #endregion


        #region Miscellaneous



        [TestMethod]
        public void test_convolve_1()
        {
            var a = np.convolve(new int[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f });
            AssertArray(a, new double[] { 0.0, 1.0, 2.5, 4.0, 1.5 });
            print(a);

            var b = np.convolve(new int[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_SAME);
            AssertArray(b, new double[] { 1.0, 2.5, 4.0 });
            print(b);

            var c = np.convolve(new int[] { 1, 2, 3 }, new float[] { 0, 1, 0.5f }, mode: NPY_CONVOLE_MODE.NPY_CONVOLVE_VALID);
            AssertArray(c, new double[] { 2.5 });
            print(c);

            return;
        }

        [TestMethod]
        public void test_clip_1()
        {
            ndarray a = np.arange(10);
            print(a);
            print("*****");

            ndarray b = np.clip(a, 1, 8);
            print(b);
            print("*****");
            AssertArray(b, new Int32[] { 1, 1, 2, 3, 4, 5, 6, 7, 8, 8 });

            ndarray c = np.clip(a, 3, 6, @out: a);
            print(c);
            AssertArray(c, new Int32[] { 3, 3, 3, 3, 4, 5, 6, 6, 6, 6, });
            print(a);
            AssertArray(a, new Int32[] { 3, 3, 3, 3, 4, 5, 6, 6, 6, 6, });
            print("*****");


            a = np.arange(10);
            print(a);
            b = np.clip(a, np.array(new Int32[] { 3, 4, 1, 1, 1, 4, 4, 4, 4, 4 }), 8);
            print(b);
            AssertArray(a, new Int32[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            AssertArray(b, new Int32[] { 3, 4, 2, 3, 4, 5, 6, 7, 8, 8 });
            print("*****");
        }

        [TestMethod]
        public void test_clip_2()
        {
            ndarray a = np.arange(16).reshape(new shape(4, 4));
            print(a);
            print("*****");

            ndarray b = np.clip(a, 1, 8);
            print(b);
            print("*****");
            AssertArray(b, new Int32[,] { { 1, 1, 2, 3 }, { 4, 5, 6, 7 }, { 8, 8, 8, 8 }, { 8, 8, 8, 8 } });

            ndarray c = np.clip(a, 3, 6, @out: a);
            print(c);
            AssertArray(c, new Int32[,] { { 3, 3, 3, 3 }, { 4, 5, 6, 6 }, { 6, 6, 6, 6 }, { 6, 6, 6, 6 } });
            print(a);
            AssertArray(a, new Int32[,] { { 3, 3, 3, 3 }, { 4, 5, 6, 6 }, { 6, 6, 6, 6 }, { 6, 6, 6, 6 } });
            print("*****");

            a = np.arange(16).reshape(new shape(4, 4));
            print(a);
            b = np.clip(a, np.array(new Int32[] { 3, 4, 1, 1 }), 8);
            print(b);
            AssertArray(b, new Int32[,] { { 3, 4, 2, 3 }, { 4, 5, 6, 7 }, { 8, 8, 8, 8 }, { 8, 8, 8, 8 } });

        }
 
        [TestMethod]
        public void test_square_operations()
        {
            var a = np.arange(0, 32, 1, dtype: np.Int32);
            print(a);

            var b = np.square(a);
            print(b);

            var ExpectedDataB1 = new Int32[]
            {
                0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144, 169, 196, 225, 256, 289,
                324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961
            };
            AssertArray(b, ExpectedDataB1);

            a = np.arange(2048, 2048 + 32, 1, dtype: np.Int64);
            print(a);

            b = np.square(a);
            print(b);

            var ExpectedDataB2 = new Int64[]
            {
                4194304, 4198401, 4202500, 4206601, 4210704, 4214809, 4218916, 4223025, 4227136,
                4231249, 4235364, 4239481, 4243600, 4247721, 4251844, 4255969, 4260096, 4264225,
                4268356, 4272489, 4276624, 4280761, 4284900, 4289041, 4293184, 4297329, 4301476,
                4305625, 4309776, 4313929, 4318084, 4322241
            };
            AssertArray(b, ExpectedDataB2);

        }

        [TestMethod]
        public void test_absolute_operations()
        {
            var a = np.arange(-32, 32, 1, dtype: np.Int16);
            print(a);

            var b = np.absolute(a);
            print(b);

            var ExpectedDataB = new Int16[]
            {
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,
                2,  1,  0,  1,  2,  3,  4,  5,   6,  7,  8,  9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31
            };
        }

        [TestMethod]
        public void test_fabs_1()
        {
            var a = np.arange(-32, 32, 1, dtype: np.Int16);
            print(a);

            var b = np.fabs(a);
            print(b);

            var ExpectedDataB = new Int16[]
            {
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,
                2,  1,  0,  1,  2,  3,  4,  5,   6,  7,  8,  9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31
            };
        }

        [TestMethod]
        public void test_sign_1()
        {
            var a = np.sign(-1.2f);
            Assert.AreEqual(-1.0f, a.GetItem(0));
            print(a);

            var b = np.sign(np.array(new double[] { 1, -2.3, 2.1 }));
            AssertArray(b, new double[] { 1, -1, 1 });
            print(b);

            var c = np.sign(np.array(new double[] { +0.0, -0.0 }));  
            AssertArray(c, new double[] { 0, 0 });
            print(c);

            var d = np.sign(np.array(new float[] { float.NegativeInfinity, float.PositiveInfinity }));
            AssertArray(d, new float[] { -1, 1 });
            print(d);

            var e = np.sign(np.array(new double[] { -double.NaN, double.NaN })); // note: different result.  No such thing as -NaN
            AssertArray(e, new double[] { double.NaN , double.NaN });
            print(e);

            var f = np.sign(np.array(new int[] { -1, 0, 1 }));
            AssertArray(f, new Int64[] { -1, 0, 1 });
            print(f);
        }

        [TestMethod]
        public void test_heaviside_1()
        {
            var a = np.heaviside(new float[] { -1.5f, 0.0f, 2.0f }, 0.5f);
            AssertArray(a, new float[] {0.0f, 0.5f, 1.0f });
            print(a);

            var b = np.heaviside(new double[] { -1.5, 0, 2.0 }, 1);
            AssertArray(b, new double[] { 0.0, 1.0, 1.0 });
            print(b);

            var c = np.heaviside(new int[] { -1, 0, 2 }, 1);
            AssertArray(c, new Int32[] { 0, 1, 1 });
            print(c);

        }

        [TestMethod]
        public void test_maximum_1()
        {
            var a = np.maximum(new int[] { 2, 3, 4 }, new int[] { 1, 5, 2 });
            AssertArray(a, new int[] { 2, 5, 4 });
            print(a);

            var b = np.maximum(np.eye(2), new double[] { 0.5, 2 }); // broadcasting
            AssertArray(b, new double[,] { { 1, 2 }, { 0.5, 2.0 } });
            print(b);

            var c = np.maximum(new float[] { float.NaN, 0, float.NaN }, new float[] { 0, float.NaN, float.NaN });
            AssertArray(c, new float[] { float.NaN, float.NaN, float.NaN });
            print(c);

            var d = np.maximum(double.PositiveInfinity, 1);
            Assert.AreEqual(double.PositiveInfinity, d.GetItem(0));
            print(d);
        }

        [TestMethod]
        public void test_minimum_1()
        {
            var a = np.minimum(new int[] { 2, 3, 4 }, new int[] { 1, 5, 2 });
            AssertArray(a, new int[] { 1, 3, 2 });
            print(a);

            var b = np.minimum(np.eye(2), new double[] { 0.5, 2 }); // broadcasting
            AssertArray(b, new double[,] { { 0.5, 0.0 }, { 0.0, 1.0 } });
            print(b);

            var c = np.minimum(new float[] { float.NaN, 0, float.NaN }, new float[] { 0, float.NaN, float.NaN });
            AssertArray(c, new float[] { float.NaN, float.NaN, float.NaN });
            print(c);

            var d = np.minimum(double.PositiveInfinity, 1);
            Assert.AreEqual((double)1, d.GetItem(0));
            print(d);
        }

        [TestMethod]
        public void test_fmax_1()
        {
            var a = np.fmax(new int[] { 2, 3, 4 }, new int[] { 1, 5, 2 });
            AssertArray(a, new int[] { 2, 5, 4 });
            print(a);

            var b = np.fmax(np.eye(2), new double[] { 0.5, 2 }); // broadcasting
            AssertArray(b, new double[,] { { 1, 2 }, { 0.5, 2.0 } });
            print(b);

            var c = np.fmax(new float[] { float.NaN, 0, float.NaN }, new float[] { 0, float.NaN, float.NaN });
            AssertArray(c, new float[] { 0.0f, 0.0f, float.NaN });
            print(c);

            var d = np.fmax(double.PositiveInfinity, 1);
            Assert.AreEqual(double.PositiveInfinity, d.GetItem(0));
            print(d);
        }

        [TestMethod]
        public void test_fmin_1()
        {
            var a = np.fmin(new int[] { 2, 3, 4 }, new int[] { 1, 5, 2 });
            AssertArray(a, new int[] { 1, 3, 2 });
            print(a);

            var b = np.fmin(np.eye(2), new double[] { 0.5, 2 }); // broadcasting
            AssertArray(b, new double[,] { { 0.5, 0.0 }, { 0.0, 1.0 } });
            print(b);

            var c = np.fmin(new float[] { float.NaN, 0, float.NaN }, new float[] { 0, float.NaN, float.NaN });
            AssertArray(c, new float[] { 0.0f, 0.0f, float.NaN });
            print(c);

            var d = np.fmin(double.PositiveInfinity, 1);
            Assert.AreEqual((double)1, d.GetItem(0));
            print(d);
        }

        [TestMethod]
        public void test_nan_to_num_1()
        {
            double a1 = (double)np.nan_to_num(double.PositiveInfinity);
            Assert.AreEqual(a1, 1.7976931348623157e+308);
            print(a1);

            double b1 = (double)np.nan_to_num(double.NegativeInfinity);
            Assert.AreEqual(b1, -1.7976931348623157e+308);
            print(b1);

            double c1 = (double)np.nan_to_num(double.NaN);
            Assert.AreEqual(c1, 0.0);
            print(c1);

            double a2 = (float)np.nan_to_num(float.PositiveInfinity);
            Assert.AreEqual(a2, 3.40282346638529E+38f);
            print(a2);

            double b2 = (float)np.nan_to_num(float.NegativeInfinity);
            Assert.AreEqual(b2, -3.40282346638529E+38f);
            print(b2);

            double c2 = (float)np.nan_to_num(float.NaN);
            Assert.AreEqual(c2, 0.0f);
            print(c2);

            ndarray x = np.array(new double[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN, -128, 128 });
            ndarray d = np.nan_to_num(x);
            AssertArray(d, new double[] { 1.7976931348623157E+308, -1.7976931348623157E+308,  0.00000000e+000, -1.28000000e+002, 1.28000000e+002 });
            print(d);

            //e = np.nan_to_num(x, nan=-9999, posinf=33333333, neginf=33333333);
            //print(e);

            //y = np.array([complex(np.inf, np.nan), np.nan, complex(np.nan, np.inf)]);
            //print(y);

            //f = np.nan_to_num(y);
            //print(f);

            //g = np.nan_to_num(y, nan=111111, posinf=222222);
            //print(g);
        }

        [TestMethod]
        public void test_interp_1()
        {
            var xp = new float[] { 1, 2, 3 };
            var fp = new float[] { 3, 2, 0 };

            var a = np.interp(2.5, xp, fp);
            AssertArray(a, new double[] { 1.0 });
            print("a: ", a);  // 1.0

            // 3.0, 3.0, 2.5, 0.56, 0.0
            var xb = new double[] { 0, 1, 1.5, 2.72, 3.14 };
            var b = np.interp(xb, xp, fp);
            AssertArray(b, new double[] { 3.0  , 3.0 , 2.5, 0.56, 0.0 });
            print(b);
   

            float UNDEF = -99.0f;

            var c = np.interp(new float[] { 3.14f }, xp, fp, right: UNDEF);
            AssertArray(c, new double[] { -99 });

            var d = np.interp(new float[] { 3.14f, -1f }, xp, fp, left: UNDEF, right: UNDEF);
            AssertArray(d, new double[] { -99, -99 });

        }

        [TestMethod]
        public void test_interp_1a()
        {
            var xp = new double[] { 1, 2, 3 };
            var fp = new double[] { 3, 2, 0 };

            var a = np.interp(2.5, xp, fp);
            AssertArray(a, new double[] { 1.0 });
            print("a: ", a);  // 1.0

            // 3.0, 3.0, 2.5, 0.56, 0.0
            var xb = new double[] { 0, 1, 1.5, 2.72, 3.14 };
            var b = np.interp(xb, xp, fp);
            AssertArray(b, new double[] { 3.0, 3.0, 2.5, 0.56, 0.0 });
            print(b);


            float UNDEF = -99.0f;

            var c = np.interp(new float[] { 3.14f }, xp, fp, right: UNDEF);
            AssertArray(c, new double[] { -99 });

            var d = np.interp(new float[] { 3.14f, -1f }, xp, fp, left: UNDEF, right: UNDEF);
            AssertArray(d, new double[] { -99, -99 });

        }

        [TestMethod]
        public void test_interp_1b()
        {
            var xp = new int[] { 1, 2, 3 };
            var fp = new int[] { 3, 2, 0 };

            var a = np.interp(2.5, xp, fp);
            AssertArray(a, new double[] { 1.0 });
            print("a: ", a);  // 1.0

            // 3.0, 3.0, 2.5, 0.56, 0.0
            var xb = new double[] { 0, 1, 1.5, 2.72, 3.14 };
            var b = np.interp(xb, xp, fp);
            AssertArray(b, new double[] { 3.0, 3.0, 2.5, 0.56, 0.0 });
            print(b);


            float UNDEF = -99.0f;

            var c = np.interp(new float[] { 3.14f }, xp, fp, right: UNDEF);
            AssertArray(c, new double[] { -99 });

            var d = np.interp(new float[] { 3.14f, -1f }, xp, fp, left: UNDEF, right: UNDEF);
            AssertArray(d, new double[] { -99, -99 });

        }

        [TestMethod]
        public void test_interp_2()
        {
            //// with period
            var x = new float[] { -180, -170, -185, 185, -10, -5, 0, 365 };
            var xp = new float[] { 190, -190, 350, -350 };
            var fp = new float[] { 5, 10, 3, 4 };
            var a = np.interp(x, xp, fp, period: 360);
            AssertArray(a, new double[] { 7.5, 5.0, 8.75, 6.25, 3.0, 3.25, 3.5, 3.75 });
            print(a);
        }


        #endregion

        #region UFunc Special Calls

        [Ignore]
        [TestMethod]
        public void xxx_UFunc_At_Placeholder()
        {

        }
        #endregion

   
    }
}
