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

            var sw1 = new  System.Diagnostics.Stopwatch();
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

            a = np.linspace(-1.0, 1.0, ref ref_step, 12).reshape((2,2,3));
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
            AssertArray(b, new double[] { np.NaN, np.NaN, -0.276226630763592, 0.091034778037415, 0.471861837279642, 0.958241588455558});
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

        [Ignore]  // too much work.
        [TestMethod]
        public void test_hypot_1()
        {

        }

        [TestMethod]
        public void test_arctan2_1()
        {
            var x = np.array(new double[] { -1, +1, +1, -1 });
            var y = np.array(new double[] { -1, -1, +1, +1 });
            var z = np.arctan2(y, x) * 180 / Math.PI;
            AssertArray(z, new double[] { -135.0, -45.0, 45.0, 135.0 });
            print(z);

            var a = np.arctan2(new double[] { 1.0, -1.0}, new double[] { 0.0, 0.0} );
            AssertArray(a, new double[] { 1.5707963267949, -1.5707963267949 });
            print(a);

            var b = np.arctan2(new double[] { 0.0, 0.0, double.PositiveInfinity}, new double[] { +0.0, -0.0, double.PositiveInfinity});
            AssertArray(b, new double[] { 0.0, 3.14159265358979, double.NaN });
            print(b);

        }

        #endregion

        private bool CompareArrays(ndarray a1, ndarray a2)
        {
            if (a1.size != a2.size)
                return false;

            if (a1.Dtype.TypeNum != a2.Dtype.TypeNum)
                return false;

            long ElementCount = a1.size;

            for (int i = 0; i < ElementCount; i++)
            {
                var a1d = a1.GetItem(i);
                var a2d = a2.GetItem(i);

                if (!a1d.Equals(a2d))
                    return false;
            }

            return true;

        }

        [Ignore]
        [TestMethod]
        public void xxx_Test_MathematicalFunctions_Placeholder()
        {
        }


    }
}
