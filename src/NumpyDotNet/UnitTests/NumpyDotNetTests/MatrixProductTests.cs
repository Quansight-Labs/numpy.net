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

    }
}
