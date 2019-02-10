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

        [TestMethod]
        public void test_dstack_1()
        {
            var a = np.array(new Int32[] { 1, 2, 3 });
            var b = np.array(new Int32[] { 2, 3, 4 });
            var c = np.dstack(new object[] { a, b });

            AssertArray(c, new Int32[,,] { { { 1, 2 }, { 2, 3 }, { 3, 4 } } });
            print(c);

            a = np.array(new Int32[] { 1, 2, 3 }).reshape(new shape(3,1));
            b = np.array(new Int32[] { 2, 3, 4 }).reshape(new shape(3,1));
            c = np.dstack(new object[] { a, b });

            AssertArray(c, new Int32[,,] { { { 1, 2 } }, { { 2, 3 } }, { { 3, 4 } } });
            
            print(c);
        }

        [TestMethod]
        public void test_array_split_1()
        {
            var x = np.arange(8.0);
            var y = np.array_split(x, 3);
 
            AssertArray(y.ElementAt(0), new double[] { 0, 1, 2 });
            AssertArray(y.ElementAt(1), new double[] { 3, 4, 5 });
            AssertArray(y.ElementAt(2), new double[] { 6, 7 });
            print(y);

            print("**************");

            x = np.arange(7.0);
            y = np.array_split(x, 3);

            AssertArray(y.ElementAt(0), new double[] { 0, 1, 2 });
            AssertArray(y.ElementAt(1), new double[] { 3, 4 });
            AssertArray(y.ElementAt(2), new double[] { 5, 6 });
            print(y);
        }

        [TestMethod]
        public void test_array_split_2()
        {
            var x = np.arange(16.0).reshape(new shape(2, 8, 1));
            var y = np.array_split(x, 3, axis : 0);


            AssertArray(y.ElementAt(0), new double[,,] { { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 } } } );
            AssertArray(y.ElementAt(1), new double[,,] { { { 8 }, { 9 }, { 10 }, { 11 }, { 12 }, { 13 }, { 14 }, { 15 } } } );
            AssertShape(y.ElementAt(2), 0,8,1 );

            print(y);

            print("**************");

            x = np.arange(16.0).reshape(new shape(2, 8, 1));
            y = np.array_split(x, 3, axis : 1);

            AssertArray(y.ElementAt(0), new double[,,] { { { 0 }, { 1 }, { 2 } }, { { 8 }, { 9 }, { 10 } } } );
            AssertArray(y.ElementAt(1), new double[,,] { { { 3 }, { 4 }, { 5 } }, { { 11 }, { 12 }, { 13 } } } );
            AssertArray(y.ElementAt(2), new double[,,]  { { { 6 }, { 7 } }, { { 14 }, { 15 } } } );


            print(y);

            print("**************");

            x = np.arange(16.0).reshape(new shape(2, 8, 1));
            y = np.array_split(x, 3, axis : 2);

            AssertArray(y.ElementAt(0), new double[,,] { { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 } }, { { 8 }, { 9 }, { 10 }, { 11 }, { 12 }, { 13 }, { 14 }, { 15 } } } );
            AssertShape(y.ElementAt(1), 2,8,0);
            AssertShape(y.ElementAt(2), 2,8,0);
            print(y);
        }

        [TestMethod]
        public void test_split_1()
        {
            var x = np.arange(9.0);
            var y = np.split(x, 3);

            AssertArray(y.ElementAt(0), new double[] { 0, 1, 2 });
            AssertArray(y.ElementAt(1), new double[] { 3, 4, 5 });
            AssertArray(y.ElementAt(2), new double[] { 6, 7, 8 });
            print(y);

            print("**************");

            x = np.arange(8.0);
            y = np.split(x, new int[] { 3, 5, 6, 10 });

            AssertArray(y.ElementAt(0), new double[] { 0, 1, 2 });
            AssertArray(y.ElementAt(1), new double[] { 3, 4 });
            AssertArray(y.ElementAt(2), new double[] { 5 });
            AssertArray(y.ElementAt(3), new double[] { 6,7 });
            AssertShape(y.ElementAt(4), 0);
            print(y);
        }

        [TestMethod]
        public void test_split_2()
        {
            var x = np.arange(16.0).reshape(new shape(8, 2, 1));
            var y = np.split(x, new int[] { 2, 3 }, axis: 0);

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new double[,,] { { { 0 }, { 1 } }, { { 2 }, { 3 } } } );
            AssertArray(y.ElementAt(1), new double[,,] { { { 4 }, { 5 } } });
            AssertArray(y.ElementAt(2), new double[,,] { { { 6 }, { 7 } }, { { 8 }, { 9 } }, { { 10 }, { 11 } }, { { 12 }, { 13 } }, { { 14 }, { 15 } } });


            print(y);

            print("**************");

            x = np.arange(16.0).reshape(new shape(8, 2, 1));
            y = np.split(x, new int[] { 2, 3 }, axis: 1);

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new double[,,] {{{0},{1}},{{2}, {3}}, {{4}, {5}}, {{6}, { 7}},
                                                        {{8},{9}},{{10},{11}}, {{12}, {13}}, {{14}, {15}}});
            AssertShape(y.ElementAt(1), 8, 0, 1);
            AssertShape(y.ElementAt(2), 8, 0, 1);

            print(y);

            print("**************");

            x = np.arange(16.0).reshape(new shape(8, 2, 1));
            y = np.split(x, new int[] { 2, 3 }, axis: 2);

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new double[,,] {{{ 0},{ 1}},{{ 2}, { 3}}, {{ 4}, { 5}}, {{ 6}, { 7}},
                                                        {{ 8},{ 9}},{{10}, {11}}, {{12}, {13}}, {{14}, {15}}});
            AssertShape(y.ElementAt(1), 8, 2, 0);
            AssertShape(y.ElementAt(2), 8, 2, 0);
  
            print(y);
        }

        [TestMethod]
        public void test_hsplit_1()
        {
            var x = np.arange(16).reshape(new shape(4, 4));
            var y = np.hsplit(x, 2);

            Assert.AreEqual(2, y.Count);
            AssertArray(y.ElementAt(0), new double[,] { { 0, 1 }, { 4, 5 }, { 8, 9 }, { 12, 13 } });
            AssertArray(y.ElementAt(1), new double[,] { { 2, 3 }, { 6, 7 }, { 10, 11 }, { 14, 15 } });
            print(y);

            print("**************");

            x = np.arange(16).reshape(new shape(4, 4));
            y = np.hsplit(x, new int[] { 3, 6 });

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new double[,] { { 0, 1, 2 }, { 4, 5, 6 }, { 8, 9, 10 }, { 12, 13, 14 } });
            AssertArray(y.ElementAt(1), new double[,] { { 3 }, { 7 }, { 11 }, { 15 } });
            AssertShape(y.ElementAt(2), 4, 0);
            print(y);
        }

        [TestMethod]
        public void test_hsplit_2()
        {
            var x = np.arange(8).reshape(new shape(2, 2, 2));
            var y = np.hsplit(x, 2);

            Assert.AreEqual(2, y.Count);
            AssertArray(y.ElementAt(0), new double[,,] { { { 0, 1 } }, { { 4, 5 } } });
            AssertArray(y.ElementAt(1), new double[,,] { { { 2, 3 } }, { { 6, 7 } } });
            print(y);
     
            print("**************");

            x = np.arange(8).reshape(new shape(2, 2, 2));
            y = np.hsplit(x, new int[] { 3, 6 });

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new double[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } });
            AssertShape(y.ElementAt(1), 2, 0, 2);
            AssertShape(y.ElementAt(2), 2, 0, 2);
 
            print(y);
        }

        [TestMethod]
        public void test_vsplit_1()
        {
            var x = np.arange(16).reshape(new shape(4, 4));
            var y = np.vsplit(x, 2);

            Assert.AreEqual(2, y.Count);
            AssertArray(y.ElementAt(0), new double[,] { { 0, 1, 2, 3 }, { 4, 5, 6, 7 } });
            AssertArray(y.ElementAt(1), new double[,] { { 8, 9, 10, 11 }, { 12, 13, 14, 15 } } );
            print(y);

            print("**************");

            x = np.arange(16).reshape(new shape(4, 4));
            y = np.vsplit(x, new int[] { 3, 6 });

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new double[,] { { 0, 1, 2, 3 }, { 4, 5, 6, 7 }, { 8, 9, 10, 11 } });
            AssertArray(y.ElementAt(1), new double[,] { { 12, 13, 14, 15 } });
            AssertShape(y.ElementAt(2), 0, 4);
            print(y);
        }

        [TestMethod]
        public void test_vsplit_2()
        {
            var x = np.arange(8).reshape(new shape(2, 2, 2));
            var y = np.vsplit(x, 2);

            Assert.AreEqual(2, y.Count);
            AssertArray(y.ElementAt(0), new double[,,] { { { 0, 1 }, { 2, 3 } } });
            AssertArray(y.ElementAt(1), new double[,,] { { { 4, 5 }, { 6, 7 } } });
            print(y);

            print("**************");

            x = np.arange(8).reshape(new shape(2, 2, 2));
            y = np.vsplit(x, new int[] { 3, 6 });

            Assert.AreEqual(3, y.Count);
            AssertArray(y.ElementAt(0), new double[,,] { { { 0, 1 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } });
            AssertShape(y.ElementAt(1), 0, 2, 2);
            AssertShape(y.ElementAt(2), 0, 2, 2);

            print(y);
        }

    }
}
