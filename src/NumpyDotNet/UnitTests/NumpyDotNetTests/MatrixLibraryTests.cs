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
    public class MatrixLibraryTests : TestBaseClass
    {

        // apparently maxtrix operations are targeted for discontinuation. 
        //
        // matrix are really just forced 2D ndarrays that override certain operations like multiply and power.
        // However these operations can be done using the np.dot function instead.

 #if NOT_PLANNING_TODO
        [Ignore]
        [TestMethod]
        public void xxx_Test_MatrixLibrary_PlaceHolder()
        {
        }


        [Ignore]  // waiting to implement matrix code
        [TestMethod]
        public void xxx_test_asmatrix_1()
        {
            var x = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            var m = np.asmatrix(x);

            x[0, 0] = 5;

            AssertArray(m, new int[,] { { 5, 2 }, { 3, 4 } });

            print(m);

            return;
        }

        [Ignore] // not implemented yet
        [TestMethod]
        public void xxx_test_mat_1()
        {

            return;
        }


        [Ignore] // not implemented yet
        [TestMethod]
        public void xxx_test_bmat_1()
        {

            return;
        }
#endif
    }


}
