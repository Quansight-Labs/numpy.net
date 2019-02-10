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
    public class ShapeBaseTests : TestBaseClass
    {
        [TestMethod]
        public void test_atleast_1d()
        {
            var a = np.atleast_1d(1.0);
            print(a);
            AssertArray(a.ElementAt(0), new double[] { 1.0 });

            print("**************");
            var x = np.arange(9.0).reshape(new shape(3, 3));
            var b = np.atleast_1d(x);
            print(b);

            var ExpectedB = new double[,]
                {{0.0, 1.0, 2.0},
                 {3.0, 4.0, 5.0},
                 {6.0, 7.0, 8.0}};
            AssertArray(b.ElementAt(0), ExpectedB);

            print("**************");

            var c = np.atleast_1d(new object[] { 1, new int[] { 3, 4 } });

            AssertArray(c.ElementAt(0), new Int32[] { 1 });
            AssertArray(c.ElementAt(1), new Int32[] { 3, 4 });
            print(c);

        }

        [TestMethod]
        public void test_atleast_2d()
        {
            var a = np.atleast_2d(1.0);
            print(a);
            AssertArray(a.ElementAt(0), new double[,] { { 1.0 } });

            print("**************");
            var x = np.arange(9.0).reshape(new shape(3, 3));
            var b = np.atleast_2d(x);
            print(b);

            var ExpectedB = new double[,]
                {{0.0, 1.0, 2.0},
                 {3.0, 4.0, 5.0},
                 {6.0, 7.0, 8.0}};
            AssertArray(b.ElementAt(0), ExpectedB);

            print("**************");

            var c = np.atleast_2d(new object[] { 1, new int[] { 3, 4 }, new int[] { 5, 6 } });

            AssertArray(c.ElementAt(0), new Int32[,] { { 1 } });
            AssertArray(c.ElementAt(1), new Int32[,] { { 3, 4 } });
            AssertArray(c.ElementAt(2), new Int32[,] { { 5, 6 } });
            print(c);

        }

        [TestMethod]
        public void test_atleast_3d()
        {
            var a = np.atleast_3d(1.0);
            print(a);
            AssertArray(a.ElementAt(0), new double[,,] { { { 1.0 } } });

            print("**************");
            var x = np.arange(9.0).reshape(new shape(3, 3));
            var b = np.atleast_3d(x);
            print(b);

            var ExpectedB = new double[,,]
             {{{0.0},
               {1.0},
               {2.0}},
              {{3.0},
               {4.0},
               {5.0}},
              {{6.0},
               {7.0},
               {8.0}}};

            AssertArray(b.ElementAt(0), ExpectedB);

            print("**************");

            var c = np.atleast_3d(new object[] { new int[] { 1, 2 }, new int[] { 3, 4 }, new int[] { 5, 6 } });

            AssertArray(c.ElementAt(0), new Int32[,,] { { { 1 }, { 2 } } });
            AssertArray(c.ElementAt(1), new Int32[,,] { { { 3 }, { 4 } } });
            AssertArray(c.ElementAt(2), new Int32[,,] { { { 5 }, { 6 } } });
            print(c);


        }

        [TestMethod]
        public void test_vstack_1()
        {
            var a = np.array(new Int32[] { 1, 2, 3 });
            var b = np.array(new Int32[] { 2, 3, 4 });
            var c = np.vstack(new object[] { a, b });

            AssertArray(c, new Int32[,] { { 1, 2, 3 }, { 2, 3, 4 } });

            print(c);
        }

        [TestMethod]
        public void test_vstack_2()
        {
            var a = np.array(new Int32[] {  1 ,  2 ,  3 }).reshape(new shape(-1,1));
            var b = np.array(new Int32[] {  2 ,  3 ,  4 }).reshape(new shape(-1,1)); 
            var c = np.vstack(new object[] { a, b });

            AssertArray(c, new Int32[,] { { 1 }, { 2 }, { 3 } , { 2 }, { 3 }, { 4 }  });

            print(c);
        }

        [TestMethod]
        public void test_hstack_1()
        {
            var a = np.array(new Int32[] { 1, 2, 3 });
            var b = np.array(new Int32[] { 2, 3, 4 });
            var c = np.hstack(new object[] { a, b });

            AssertArray(c, new Int32[] { 1, 2, 3, 2, 3, 4 });

            print(c);
        }

        [TestMethod]
        public void test_hstack_2()
        {
            var a = np.array(new Int32[] { 1, 2, 3 }).reshape(new shape(-1, 1));
            var b = np.array(new Int32[] { 2, 3, 4 }).reshape(new shape(-1, 1));
            var c = np.hstack(new object[] { a, b });

            AssertArray(c, new Int32[,] { { 1, 2 }, { 2, 3 }, { 3, 4 } });

            print(c);
        }

        [TestMethod]
        public void test_stack_1()
        {
            var a = np.array(new Int32[] { 1, 2, 3 }).reshape(new shape(-1, 1));
            var b = np.array(new Int32[] { 2, 3, 4 }).reshape(new shape(-1, 1));

            var c = np.stack(new object[] { a, b }, axis: 0);
            AssertArray(c, new Int32[,,] { { { 1 }, { 2 }, { 3 } }, { { 2 }, { 3 }, { 4 } } });
            print(c);
            print("**************");

            var d = np.stack(new object[] { a, b }, axis: 1);
            AssertArray(d, new Int32[,,] { { { 1 }, { 2 } }, { { 2 }, { 3 } }, { { 3 }, { 4 } } });
            print(d);
            print("**************");

            var e = np.stack(new object[] { a, b }, axis: 2);
            AssertArray(e, new Int32[,,] { { { 1, 2 } }, { { 2, 3 } }, { { 3, 4 } } });
            print(e);

        }

        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void test_block_1()
        {
            var A = np.eye(2) * 2;
            var B = np.eye(3) * 3;
            var C = np.block(new object[] {new object[] {A, np.zeros(new shape(2, 3))}, new object[] {np.ones(new shape(3, 2)), B}});

            var ExpectedDataC = new double[,]
                {{2.0, 0.0, 0.0, 0.0, 0.0},
                 {0.0, 2.0, 0.0, 0.0, 0.0,},
                 {1.0, 1.0, 3.0, 0.0, 0.0,},
                 {1.0, 1.0, 0.0, 3.0, 0.0,},
                 {1.0, 1.0, 0.0, 0.0, 3.0,}};

            AssertArray(C, ExpectedDataC);

            print(C);
        }

        [Ignore] // not implemented yet.  Too much work
        [TestMethod]
        public void test_block_2()
        {
            var a = np.array(new Int32[] { 1, 2, 3 });
            var b = np.array(new Int32[] { 2, 3, 4 });
            var c = np.block(new object[] { a, b, 10 });    // hstack([a, b, 10])

            AssertArray(c, new Int32[] { 1,  2,  3,  2,  3,  4, 10 });
            print(c);
            print("**************");

            a = np.array(new Int32[] { 1, 2, 3 });
            b = np.array(new Int32[] { 2, 3, 4 });
            c = np.block(new object[]{new object[]{a}, new object[] {b}});    // vstack([a, b])

            AssertArray(c, new Int32[,] { { 1, 2, 3 }, { 2, 3, 4 } });
            print(c);

        }

        [TestMethod]
        public void test_expand_dims_1()
        {
            var a = np.array(new Int32[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }).reshape(new shape(2,-1, 2));
            var b = np.expand_dims(a, axis: 0);

            var ExpectedDataB = new Int32[,,,] 
            {{{{1,  2}, {3,  4}, {5,  6}},
              {{7,  8}, {9, 10}, {11, 12}}}};

            AssertArray(b, ExpectedDataB);
            print(b);
            print("**************");

            var c = np.expand_dims(a, axis: 1);
            var ExpectedDataC = new Int32[,,,]
                {{{{1,  2}, {3,  4}, {5,  6}}},
                {{{ 7,  8},{ 9, 10}, {11, 12}}}};
            AssertArray(c, ExpectedDataC);
            print(c);
            print("**************");

            var d = np.expand_dims(a, axis: 2);
            var ExpectedDataD = new Int32[,,,]
            {{{{1,  2}},{{3,  4}},{{5,  6}}},
             {{{7,  8}},{{9, 10}},{{11, 12}}}};

            AssertArray(d, ExpectedDataD);
            print(d);

        }

        [TestMethod]
        public void test_column_stack_1()
        {
            var a = np.array(new Int32[] { 1, 2, 3 });
            var b = np.array(new Int32[] { 2, 3, 4 });
            var c = np.column_stack(new object[] { a, b });

            AssertArray(c, new Int32[,] { { 1, 2 }, { 2, 3 }, { 3, 4 } });
            print(c);
        }

        [TestMethod]
        public void test_row_stack_1()
        {
            var a = np.array(new Int32[] { 1, 2, 3 });
            var b = np.array(new Int32[] { 2, 3, 4 });
            var c = np.row_stack(new object[] { a, b });

            AssertArray(c, new Int32[,] { { 1, 2, 3 }, { 2, 3, 4 } });

            print(c);
        }

    }
}
