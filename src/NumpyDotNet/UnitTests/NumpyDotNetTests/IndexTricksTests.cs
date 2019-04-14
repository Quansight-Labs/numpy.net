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
    public class IndexTricksTests : TestBaseClass
    {
        #if NOT_PLANNING_TODO
        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void xxx_test_ravel_multi_index_1()
        {

        }

        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void xxx_test_unravel_index_1()
        {

        }

        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void xxx_test_mgrid_1()
        {

        }

        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void xxx_test_ogrid_1()
        {

        }

        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void xxx_test_c_1()
        {

        }

        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void xxx_test_r_1()
        {

        }

        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void xxx_test_s_1()
        {

        }

        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void xxx_test_index_exp_1()
        {

        }

        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void xxx_test_ix_1()
        {

        }

        #endif

        [TestMethod]
        public void test_fill_diagonal_1()
        {
            var a = np.zeros((3, 3), np.Int32);
            np.fill_diagonal(a, 5);
            AssertArray(a, new int[,] { { 5, 0, 0 }, { 0, 5, 0 }, { 0, 0, 5 } });
            print(a);

            a = np.zeros((3, 3, 3, 3), np.Int32);
            np.fill_diagonal(a, 4);
            AssertArray(a[0, 0] as ndarray, new int[,] { { 4, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } });
            print(a[0, 0]);
            AssertArray(a[1, 1] as ndarray, new int[,] { { 0, 0, 0 }, { 0, 4, 0 }, { 0, 0, 0 } });
            print(a[1, 1]);
            AssertArray(a[2, 2] as ndarray, new int[,] { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 4 } });
            print(a[2, 2]);

            // tall matrices no wrap
            a = np.zeros((5, 3), np.Int32);
            np.fill_diagonal(a, 4);
            AssertArray(a, new int[,] { { 4, 0, 0 }, { 0, 4, 0 }, { 0, 0, 4 }, { 0, 0, 0 }, { 0, 0, 0 } });
            print(a);

            // tall matrices wrap
            a = np.zeros((5, 3), np.Int32);
            np.fill_diagonal(a, 4, wrap:true);
            AssertArray(a, new int[,] { { 4, 0, 0 }, { 0, 4, 0 }, { 0, 0, 4 }, { 0, 0, 0 }, { 4, 0, 0 } });
            print(a);

            // wide matrices wrap
            a = np.zeros((3, 5), np.Int32);
            np.fill_diagonal(a, 4, wrap: true);
            AssertArray(a, new int[,] { { 4, 0, 0, 0, 0 }, { 0, 4, 0, 0, 0 }, { 0, 0, 4, 0, 0 } });
            print(a);


        }

        [TestMethod]
        public void test_diag_indices_1()
        {
            var di = np.diag_indices(4);
            AssertArray(di[0], new Int32[] { 0, 1, 2, 3 });
            AssertArray(di[1], new Int32[] { 0, 1, 2, 3 });
            print(di);

            var a = np.arange(16).reshape((4, 4));
            a[di] = 100;

            AssertArray(a, new Int32[,] { {100,1,2,3 }, {4,100,6,7 }, {8, 9, 100, 11 }, { 12,13,14,100 } });
            print(a);

            return;

        }

        [TestMethod]
        public void test_diag_indices_from_1()
        {
            var a = np.arange(16).reshape((4, 4));
            var di = np.diag_indices_from(a);
            AssertArray(di[0], new Int32[] { 0, 1, 2, 3 });
            AssertArray(di[1], new Int32[] { 0, 1, 2, 3 });
            print(di);
        }


    }
}
