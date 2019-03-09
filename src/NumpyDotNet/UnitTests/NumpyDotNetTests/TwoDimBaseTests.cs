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
    public class TwoDimBaseTests : TestBaseClass
    {
        [TestMethod]
        public void test_diag_1()
        {
            ndarray m = np.arange(9);
            var n = np.diag(m);

            print(m);
            print(n);

            var ExpectedDataN = new Int32[,]
                {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                 {0, 1, 0, 0, 0, 0, 0, 0, 0},
                 {0, 0, 2, 0, 0, 0, 0, 0, 0},
                 {0, 0, 0, 3, 0, 0, 0, 0, 0},
                 {0, 0, 0, 0, 4, 0, 0, 0, 0},
                 {0, 0, 0, 0, 0, 5, 0, 0, 0},
                 {0, 0, 0, 0, 0, 0, 6, 0, 0},
                 {0, 0, 0, 0, 0, 0, 0, 7, 0},
                 {0, 0, 0, 0, 0, 0, 0, 0, 8}};

            AssertArray(n, ExpectedDataN);

            m = np.arange(9).reshape(new shape(3, 3));
            n = np.diag(m);

            print(m);
            print(n);
            AssertArray(n, new int[] { 0, 4, 8 });
        }

        [TestMethod]
        public void test_diagflat_1()
        {
            ndarray m = np.arange(1, 5).reshape(new shape(2, 2));
            var n = np.diagflat(m);

            print(m);
            print(n);

            var ExpectedDataN = new Int32[,]
            {
             {1, 0, 0, 0},
             {0, 2, 0, 0},
             {0, 0, 3, 0},
             {0, 0, 0, 4}
            };
            AssertArray(n, ExpectedDataN);

            m = np.arange(1, 3);
            n = np.diagflat(m, 1);

            print(m);
            print(n);

            ExpectedDataN = new Int32[,]
            {
             {0, 1, 0},
             {0, 0, 2},
             {0, 0, 0},
            };

            AssertArray(n, ExpectedDataN);

            m = np.arange(1, 3);
            n = np.diagflat(m, -1);

            print(m);
            print(n);

            ExpectedDataN = new Int32[,]
            {
             {0, 0, 0},
             {1, 0, 0},
             {0, 2, 0},
            };

            AssertArray(n, ExpectedDataN);

        }


        [TestMethod]
        public void test_eye_1()
        {
            ndarray a = np.eye(2, dtype: np.Int32);

            print(a);
            print(a.shape);
            print(a.strides);

            var ExpectedDataA = new Int32[2, 2]
            {
                { 1,0 },
                { 0,1 },
            };
            AssertArray(a, ExpectedDataA);
            AssertShape(a, 2, 2);
            AssertStrides(a, 8, 4);


            ndarray b = np.eye(3, k: 1);

            print(b);
            print(b.shape);
            print(b.strides);

            var ExpectedDataB = new double[3, 3]
            {
                { 0,1,0 },
                { 0,0,1 },
                { 0,0,0 },
            };

            AssertArray(b, ExpectedDataB);
            AssertShape(b, 3, 3);
            AssertStrides(b, 24, 8);


        }


        [TestMethod]
        public void test_fliplr_1()
        {
            ndarray m = np.arange(8).reshape(new shape(2, 2, 2));
            var n = np.fliplr(m);

            print(m);
            print(n);

            AssertArray(n, new Int32[,,] { { { 2, 3 }, { 0, 1 } }, { { 6, 7 }, { 4, 5 } } });
        }

        [TestMethod]
        public void test_flipud_1()
        {
            ndarray m = np.arange(8).reshape(new shape(2, 2, 2));
            var n = np.flipud(m);

            print(m);
            print(n);

            AssertArray(n, new Int32[,,] { { { 4, 5 }, { 6, 7 } }, { { 0, 1 }, { 2, 3 } } });
        }



        [TestMethod]
        public void test_tri_1()
        {
            ndarray a = np.tri(3, 5, 2, dtype: np.Int32);
            print(a);

            var ExpectedDataA = new Int32[,]
            {
             {1, 1, 1, 0, 0},
             {1, 1, 1, 1, 0},
             {1, 1, 1, 1, 1}
            };
            AssertArray(a, ExpectedDataA);

            print("***********");
            ndarray b = np.tri(3, 5, -1);
            print(b);

            var ExpectedDataB = new float[,]
            {
             {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
             {1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
             {1.0f, 1.0f, 0.0f, 0.0f, 0.0f}
            };
            AssertArray(b, ExpectedDataB);
        }

        [TestMethod]
        public void test_tril_1()
        {
            ndarray a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } });
            ndarray b = np.tril(a, -1);
            print(a);
            print("***********");
            print(b);

            var ExpectedDataB = new Int32[,]
            {
             {0, 0, 0},
             {4, 0, 0},
             {7, 8, 0},
             {10, 11, 12},
            };
            AssertArray(b, ExpectedDataB);

        }

        [TestMethod]
        public void test_triu_1()
        {
            ndarray a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } });
            ndarray b = np.triu(a, -1);
            print(a);
            print("***********");
            print(b);

            var ExpectedDataB = new Int32[,]
            {
             {1, 2, 3},
             {4, 5, 6},
             {0, 8, 9},
             {0, 0, 12},
            };
            AssertArray(b, ExpectedDataB);

        }


        [Ignore] // not implemented yet
        [TestMethod]
        public void test_vander_1()
        {

        }


        [Ignore] // not implemented yet
        [TestMethod]
        public void test_histogram2d()
        {

        }

        [TestMethod]
        public void test_mask_indices()
        {
            var iu = np.mask_indices(3, np.triu);
            AssertArray(iu[0], new Int64[] { 0, 0, 0, 1, 1, 2 });
            AssertArray(iu[1], new Int64[] { 0, 1, 2, 1, 2, 2 });
            print(iu);

            var a = np.arange(9).reshape((3, 3));
            var b = a[iu] as ndarray;
            AssertArray(b, new Int64[] { 0, 1, 2, 4, 5, 8 });
            print(b);

            var iu1 = np.mask_indices(3, np.triu, 1);

            var c = a[iu1] as ndarray;
            AssertArray(c, new Int64[] { 1, 2, 5 });
            print(c);

            return;
        }

        [TestMethod]
        public void test_tril_indices()
        {
            var il1 = np.tril_indices(4);
            var il2 = np.tril_indices(4, 2);

            var a = np.arange(16).reshape((4, 4));
            var b = a[il1] as ndarray;
            AssertArray(b, new int[] { 0, 4, 5, 8, 9, 10, 12, 13, 14, 15 });
            print(b);

            a[il1] = -1;

            var ExpectedDataA1 = new int[,]
                {{-1,  1, 2,  3}, {-1, -1,  6,  7}, 
                 {-1, -1,-1, 11}, {-1, -1, -1, -1}};
            AssertArray(a, ExpectedDataA1);
            print(a);

            a[il2] = -10;

            var ExpectedDataA2 = new int[,]
                {{-10, -10, -10,  3}, {-10, -10, -10, -10},
                 {-10, -10,-10, -10}, {-10, -10, -10, -10}};
            AssertArray(a, ExpectedDataA2);
            print(a);

            return;
        }

        [Ignore] // not implemented yet
        [TestMethod]
        public void test_tril_indices_from()
        {

        }

        [Ignore] // not implemented yet
        [TestMethod]
        public void test_triu_indices()
        {

        }

        [Ignore] // not implemented yet
        [TestMethod]
        public void test_triu_indices_from()
        {

        }

    }
}
