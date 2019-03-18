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

        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void xxx_test_ndenumerate_1()
        {

        }

        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void xxx_test_ndindex_1()
        {

        }
        #endif

        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void xxx_test_fill_diagonal_1()
        {

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
