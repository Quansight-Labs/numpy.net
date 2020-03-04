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
    public class MatrixProductTests : TestBaseClass
    {
        [TestMethod]
        public void test_matmul_1()
        {
            var a = np.arange(9).reshape(3, 3);
            var b = np.arange(9).reshape(3, 3);

            var ret = np.matmul(a, b);
            print(ret);

            AssertArray(ret, new Int32[,] { { 15, 18, 21 }, { 42, 54, 66 }, { 69, 90, 111 } });
        }

        [TestMethod]
        public void test_matmul_2()
        {
            var a = np.full((3, 3),2, dtype: np.Int32);
            var b = np.full((3, 3),2, dtype: np.Int32);
            var ret = np.matmul(a, b);
            print(ret);

            AssertArray(ret, new Int32[,] { { 12, 12, 12 }, { 12, 12, 12 }, { 12, 12, 12 } });
        }

        [TestMethod]
        public void test_matmul_3()
        {
            var a = np.arange(9).reshape(3, 3);
            var b = np.arange(3).reshape(3);

            var ret = np.matmul(a, b);
            print(ret);

            AssertArray(ret, new Int32[] { 5, 14, 23 });
        }

        [TestMethod]
        public void test_matmul_4()
        {
            var a = np.full((3, 3), 2, dtype: np.Int32);
            var b = np.full((3), 3, dtype: np.Int32);
            var ret = np.matmul(a, b);
            print(ret);

            AssertArray(ret, new Int32[] { 18, 18, 18 });
        }


        [TestMethod]
        public void test_matmul_5()
        {
            var a = np.full((3, 2,2), 2, dtype: np.Int32);
            var b = np.full((3,2,2), 3, dtype: np.Int32);
            var ret = np.matmul(a, b);
            print(ret.shape);
            print(ret);

            AssertArray(ret, new Int32[,,] { { { 12, 12 }, { 12, 12 } }, { { 12, 12 }, { 12, 12 } },{ { 12, 12 }, { 12, 12 } } });

        }

        [TestMethod]
        public void test_matmul_6()
        {
            var a = np.full((3, 1, 2, 2), 2, dtype: np.Int32);
            var b = np.full((3, 2, 2), 3, dtype: np.Int32);

            try
            {
                var ret = np.matmul(a, b);
                Assert.Fail("Should have thrown an exception");
            }
            catch
            {

            }
   
        }

        [TestMethod]
        public void test_matmul_7()
        {
            var a = np.full((3, 3), 2, dtype: np.Int32);
            var b = np.full((3, 1), 3, dtype: np.Int32);
            var ret = np.matmul(a, b);
            print(ret.shape);
            print(ret);

            AssertArray(ret, new Int32[,] { { 18 }, { 18 }, { 18 } });

        }


        [TestMethod]
        public void test_matmul_8()
        {
            var a = np.full((3, 3), 2, dtype: np.Int32);
            var b = np.full((3), 3, dtype: np.Int32);
            var ret = np.matmul(a, b);
            print(ret.shape);
            print(ret);

            AssertArray(ret, new Int32[] { 18, 18, 18 });

        }



        [TestMethod]
        public void test_matmul_bad1()
        {
            var a = np.full((3, 2, 2), 2, dtype: np.Int32);
            var b = np.full((3, 2, 2), 3, dtype: np.Int32);

            try
            {
                var ret = np.matmul(a, 3);
                Assert.Fail("Should have thrown an exception");
            }
            catch
            {

            }


        }


        [TestMethod]
        public void test_matmul_bad2()
        {
            var a = np.full((12), 2, dtype: np.Int32);
            var b = np.full((3, 4), 3, dtype: np.Int32);

            try
            {
                var ret = np.matmul(a, b);
                Assert.Fail("Should have thrown an exception");
            }
            catch
            {

            }

        }

        [TestMethod]
        public void KEVIN_matmul_DOUBLE()
        {
            float scaling = 5.0f;
            int width = 256;
            int height = 256;
            double ret_step = 0;

            var x_range = np.linspace(-1 * scaling, scaling, ref ret_step, width, dtype: np.Float32);

            var x_mat = np.matmul(np.ones(new shape(height, 1)), x_range.reshape(1, width));
            //print(x_mat);

            var sum = np.sum(x_mat);
            return;

        }

        [TestMethod]
        public void test_maxtrix_99_WORKS()
        {
            double ret_step = 0;

            var a = np.linspace(0.0, 1.0, ref ret_step, num: 16).reshape(1, 16);

            var b = np.reshape(a, new shape(1, 1, 16)) * np.ones((32, 1)) * 1;
            //print(b);
            var c = np.sum(b);
            print(c);
        }


         //this is the root cause of the customer failure.  Why does test above work but not this one?

        //the tuning parameter seems to help fix it
        [TestMethod]
        public void test_maxtrix_99_BROKEN()
        {
            double ret_step = 0;

            var a = np.linspace(0.0, 1.0, ref ret_step, num : 32).reshape(1, 32);
            print(a);

            var b = np.reshape(a, new shape(1, 1, 32)) * np.ones((65536, 1)); // * 1;
            //print(b);
            var c = np.sum(b);
            print(c);
        }

        [TestMethod]
        public void test_maxtrix_100_BROKEN()
        {
            var a = np.arange(00, 32).reshape(1, 32);
            print(a);

            var b = np.reshape(a, new shape(1, 1, 32)) * np.ones((65536, 1)); // * 1;
            //print(b);
            var c = np.sum(b);
            print(c);
        }


        [TestMethod]
        public void test_maxtrix_101_BROKEN()
        {
            var a = np.arange(00, 32).reshape(1, 32);
            print(a);

            var b = np.full((1, 1, 32),2 ) * np.full((65536, 1), 3); // * 1;
            //print(b);

            var d = np.where(b != 6);

            var kevin = b.AsDoubleArray();

            var c = np.sum(b);
            print(c);
        }


    }
}
