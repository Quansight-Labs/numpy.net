import unittest
import numpy as np
import time as tm
from nptest import nptest

class Test_CustomerReportedIssues(unittest.TestCase):

    
    def test_tensordot_asiamartini_bugreport(self):  
        
        alpha_0 = np.array([ 1.0, 1.0, 1.0 ]);
        temp = np.array([ [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ], 
                         [ [ -18.0, 12.0, 12.0, -10.0 ], [ 12.0, 10.0, 18.0, 12.0 ], [ 12.0, 18.0, -10.0, -12.0 ], [ -10.0, 12.0, -12.0, 18.0 ] ], 
                         [ [ -3.5, 21.5, 5.5, -2.5 ], [ 21.5, 2.5, 5.5, 3.5 ], [ 5.5, 5.5, -20.5, -5.5 ], [ -2.5, 3.5, -5.5, 21.5 ] ] ])


        matrix = nptest.tensordot(alpha_0, temp, axes=([0],[0]))
        print(matrix)
        return


    def test_matmul_asiamartini_bugreport(self):

        rq = np.array([0.5,0.5,0.5,0.5]);
        am = np.array([[-21.5, 33.5, 17.5, -12.5], [33.5, 12.5, 23.5, 15.5], [17.5,23.5,-30.5,-17.5], [-12.5, 15.5, -17.5, 39.5] ]);
   
        temp1 = np.matmul(rq.T, am);
        print(temp1)


    def test_goodgood_test(self):

        img = np.fromfile('test.dat',sep=',')
        img = img.reshape(10,10,3)
        print(img)
        output = img.copy()

        valid = np.all(img == [ 255, 255, 240], axis = -1)
        rs, cs = valid.nonzero()
        output[rs, cs, :] = [255, 255, 255]
        print(output)

    def test_goodgood_test_2(self):

        imgnd = np.arange(1,301).reshape(10,10,3)
        R = imgnd[:, :, 0]
        G = imgnd[:, :, 1]
        B = imgnd[:, :, 2]
        print(R.shape,G.shape,B.shape)
        print(R)
        print(G)
        print(B)

    def test_append_msever_1(self):

        arr = np.array([[1,2,3],[4,5,6]])
        row = np.array([7,8,9])
        arr = np.append(arr,[row],axis= 0)
        print(arr)

    def test_tuple_msever_2(self):

        a = np.array((1,2,3))
        print(a)
        b = np.array((2,3,4))
        print(b)
          
        c = np.column_stack((a,b))
        print(c)

    def test_slice_msever_1(self):

        a=np.array([[1,3,0],[0,0,5]])

        col1 = a[:,0] 
        col2 = a[:,1]
        col3 = a[:,2]

        print(col1)
        print(col2)
        print(col3)

    def test_hsplit_msever_1(self):

        a=np.array([[1,3,0],[0,0,5]])

        row, col = np.hsplit(np.argwhere(a),2)

        print(row)
        print(col)


    def test_take_msever_1(self):

        testVector = np.array([ 1.011163, 1.01644999999999, 1.01220500000001, 1.01843699999999, 1.00985100000001, 1.018964, 1.005825, 1.016707, 8.11556899999999, 1.010744, 1.01700600000001, 1.01323099999999, 1.010389, 1.015216, 1.015418, 1.01704600000001, 1.01191, 1.01164299999999, 1.01062400000001, 1.014199, 1.012952, 1.017645, 1.01591999999999, 1.018655, 1.00942400000001, 1.012852, 1.010543, 1.02000700000001, 1.008196, 1.01396099999999 ]);
        testVector2 = testVector.reshape(15, 2);
        testDataMode1 = np.array([ 1, 2, 2, 3, 4, 7, 9 ]);

        print(testVector2);
        print(testDataMode1);

        print("np.take()");
        testTake = np.take(testVector2, testDataMode1.astype(np.intp), axis=0);
        print(testTake);

        testVector3 = np.arange(0.0, 30000.0, 0.5, dtype= np.float64);
        testVector4 = testVector3.reshape(30000, 2);
        testIndex = np.arange(0, 30000, 100, dtype= np.intp);

        print("test BIG np.take()");
        # testBigTake = np.take(testVector4, testIndex, axis: 0);
        testBigTake = np.zeros((300, 2), dtype= np.float64);
        testBigTake = np.take(testVector4, testIndex, axis= 0);
        print(testIndex);
        print(testBigTake);
        print(np.diff(testIndex));
        print(np.diff(testBigTake, axis= 0));
   
    def test_HadrianTang_1(self):

       x = np.array_equal([], [])
       print(x)

       x = np.array_equiv([], [])
       print(x)



    def test_HadrianTang_2(self):

       x = np.all([])
       print(x)
       x = np.any([])
       print(x)

    def test_HadrianTang_3(self):

       x = np.sum([])
       print(x)
       x = np.prod([])
       print(x)

    def test_HadrianTang_4(self):

       x = np.logical_and([],[])
       print(x)
       x = np.logical_or([], [])
       print(x)


    def test_HadrianTang_5(self):

        x = np.not_equal([], [])
        print(x)

    def test_HadrianTang_6(self):

        x = np.concatenate([[], []])
        print(x)

        x = np.stack(([],[]))
        print(x)

    def test_HadrianTang_7(self):

        x = np.array([])
        x = x * 5
        print(x)
        x = x + 5
        print(x)


    def test_HadrianTang_8(self):

        x = np.array([["0","1"],["0","0"]]).astype(np.float);
     
        print(x)
    

    def test_HadrianTang_9(self):

        x = np.delete(np.array([["0", "1", "@"],["1", "0", "@"]]), 1, 1);
     
        print(x)
    
    def test_HadrianTang_10(self):

        x = np.logical_and(np.array([0, 1, 2]), np.array([1, 0, 2]))
        print(x)

        x = np.logical_or(np.array([1, 0, 2]), np.array([1, 0, 0]))
        print(x)

    def test_HadrianTang_11(self):

        a = np.argmax(np.arange(5))
        print(a)
        b = np.arange(7)[a]
        print(b)

    def test_HadrianTang_12(self):

        print(np.array(4).shape)
        print(np.arange(7)[np.array(4)])

    def test_HadrianTang_13(self):

       objecta = 9.234
       objectb = 33

       a = np.min(objecta)
       b = np.max(objectb)
       #c =  ndarray.this[objectb]

       print(a)
       print(b)
       #print(c)

    def test_HadrianTang_14(self):

        a = np.array(2);
        b = np.array(2, np.int32);

        print(a.ndim)
        print(b.ndim)

    def test_ChengYenTang_1(self):

        a = np.array([1,2,3]);
        b = np.less(-np.Inf, a)
        print(b)

        b = -np.Inf < a
        print(b)
        

        c = a > -np.Inf
        print(c)

    def test_ChengYenTang_2(self):

        low = np.array([[30,8,7],[2, -np.Inf, 3]]);
        high = np.array([[30,22,10],[np.Inf, 5, 3]]);

        a = low < high
        print(a)

        b = low > high
        print(b)

        c = low <= high
        print(c)

        d = low >= high
        print(d)

    def test_ChengYenTang_3(self):

        a = np.arange(0,32);
        b = np.reshape(a, (2,)+(16,))
        print(b.shape)

        c = np.reshape(a, (2,2)+(8,))
        print(c.shape)

        d = np.reshape(a, (2,2)+(2,4))
        print(d.shape)

        g = np.reshape(a, (2,2)+(2,4)+(1,1))
        print(g.shape)

    def test_ChengYenTang_4(self):

        low = np.array([[9, 8, 7], [2, -np.inf, 1]])
        print(low)

        stack_low = np.repeat(low, 3, axis=0)
        print(stack_low)

        observation = np.array([[[9, 8, 7], [2, -np.inf, 1]], [[30, 22, 10], [np.inf, 5, 3]]])
        print(observation)

        stackedobs = np.zeros((2,) + stack_low.shape);
        print(stackedobs)

        stackedobs[:, -observation.shape[1] :, ...] = observation
        print(stackedobs)

    def test_ChengYenTang_5(self):

        a = np.array([[0], [0], [0]])
        print(a.shape)
        print(a.shape[-1])

    def test_ChengYenTang_6(self):

        a = np.array([[1,2],[3,4],[5,6]])
        print(a.shape)

        b = a[0, ...]
        print(b)

        c = a[0, ..., :-1]
        print(c)

    def test_ChengYenTang_7(self):

        stackedobs = np.arange(0, 3*2*2*4).reshape(3, 2, 2, 4)
        A = stackedobs[..., -2:]
        print("A")
        print(A.shape)
        print(A)


        B = stackedobs[..., 1, -2:]
        print("----------------")
        print("B")
        print(B.shape)
        print(B)

    def test_ChengYenTang_8(self):

        A = np.arange(0, 3*2*2*4).reshape(3, 2, 2, 4)
        i = A.shape[1:]
        print(i)

        j = A.shape[1:2]
        print(j)

        k = A.shape[0::2]
        print(k)

        
        l = A.shape[1::2]
        print(l)

        m = A.shape[:]
        print(m)

        n = A.shape[::]
        print(n)

    def test_SimonCraenen_1(self):

         a = np.arange(0,4, dtype=np.float32);
         b = np.getbuffer(a)
         print(len(b))
         c = np.frombuffer(b, dtype=np.float32);
         print(c)


    def test_Rainyl_1a(self):

         arr = np.array([ -9, 7, 5, 3, 1, -1, -3, -5,-11,13,17,21 ]).reshape(4, 3);
         idx2 = np.argsort(arr)
         print(arr)
         print(idx2)

         x = arr.T[0, :]
         y = arr.T[1, :]

         idx = np.argsort(x)
         print(x);
         print(idx);

         idx = np.argsort(y)
         print(y);
         print(idx);

    def test_Rainyl_1b(self):

         arr = np.array([ -9, 7, 5, 3, 1, -1, -3, -5,-11,13,17,21 ], dtype=np.int16).reshape(4, 3);
         idx2 = np.argsort(arr)
         print(arr)
         print(idx2)

         x = arr.T[0, :]
         y = arr.T[1, :]

         idx = np.argsort(x)
         print(x);
         print(idx);

    def test_Rainyl_2a(self):

         arr = np.array([ 7, -9, -5, -3, 1, -1, 33, 5,-11,13,17,-21 ]).reshape(4, 3);
         idx2 = np.argsort(arr)
         print(arr)
         print(idx2)

         x = arr.T[:, 0]
         y = arr.T[:, 1]

         idx = np.argsort(x)
         print(x);
         print(idx);

         idx = np.argsort(y)
         print(y);
         print(idx);

    def test_Rainyl_3(self):

        a = np.asarray([[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]]);
        b = a[:, ::-1]
        print(b);
       
        
        print("output:");print(b[0, 0]);print(b[0, 1]);print(b[0, 2]);print(b[0, 3]);print(b[0, 4]);
        print("output:");print(b[1, 0]);print(b[1, 1]);print(b[1, 2]);print(b[1, 3]);print(b[1, 4]);

    def test_Rainyl_3a(self):

        a = np.asarray([[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]]);
        b = a[::-1, ::-1]
        print(b);
       
        
        print("output:");print(b[0, 0]);print(b[0, 1]);print(b[0, 2]);print(b[0, 3]);print(b[0, 4]);
        print("output:");print(b[1, 0]);print(b[1, 1]);print(b[1, 2]);print(b[1, 3]);print(b[1, 4]);

    def test_Rainyl_3b(self):

        a = np.asarray([[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]]);
        b = a[::-2, ::-2]
        print(b);
       
        print("output:");print(b[0, 0]);print(b[0, 1]);print(b[0, 2]);

        b[0, 0] = 88;
        b[0, 1] = 77;
        b[0, 2] = 66;

        print(a)


    def test_Taz145_1(self):

        arr1 = np.array([0,1,1,0,1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,1,0]).reshape((2, 3, 4));
        arr2 = np.arange(24).reshape((2, 3, 4));

        print(arr2)
        arr2 = np.rot90(arr2, k=2, axes= (0, 2));

        print(arr2)
        arr2[arr1 > 0] = 0;
        print(arr2)

    def test_Taz145_1a(self):

        arr1 = np.array([0,1,1,0,1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,1,0]).reshape((2, 3, 4));
        arr2 = np.arange(24).reshape((2, 3, 4));

        print(arr2)
        arr2 = arr2[::-1]
        print(arr2)
        arr2[arr1 > 0] = 0;
        print(arr2)

    def test_Taz145_2(self):

        arr1 = np.array([0,1,1,0,1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,1,0]).reshape((2, 3, 4));
        arr2 = np.arange(24).reshape((2, 3, 4));

        print(arr2)
        arr2 = np.rot90(arr2, k=2, axes= (0, 2));

        print(arr2)
        arr3 = arr2[arr1 > 0];
        print(arr3)

    def test_Taz145_2a(self):

        arr1 = np.array([0,1,1,0,1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,1,0]).reshape((2, 3, 4));
        arr2 = np.arange(24).reshape((2, 3, 4));

        print(arr2)
        arr2 = arr2[::-1]

        print(arr2)
        arr3 = arr2[arr1 > 0];
        print(arr3)

    def test_Taz145_3(self):

        arr1 = np.array([2,4,6], dtype=np.intp);
        arr2 = np.arange(24);

        arr2 = arr2[::-1];
        print(arr2)
        arr3 = arr2[arr1];
        print(arr3)

    def test_Taz145_3a(self):

        arr2 = np.arange(24).reshape((2, 3, 4));
        print(arr2)

        arr2 = np.rot90(arr2, k=2, axes= (0, 2));
        print(arr2)

        arr3 = arr2[np.array([0, 1, -1, -2])];
        print("arr3");
        print(arr3)

    def test_Taz145_4(self):

        arr2 = np.arange(24).reshape((2, 3, 4));
        arr2 = np.rot90(arr2, k=2, axes= (0, 2));

        arr3 = np.array(arr2.tobytes());
        print(arr3)

        arr4 = np.array(arr2.tobytes(order='F'));
        print(arr4)

    def test_Taz145_4b(self):

        arr2 = np.arange(24).reshape((2, 3, 4));
        arr2 = arr2[::-1];

        arr3 = np.array(arr2.tobytes());
        print(arr3)

        arr4 = np.array(arr2.tobytes(order='F'));
        print(arr4)

    def test_Sundarrajan06295_DifferentTypes(self):

        x = np.arange(0, 4000 * 10 * 4000, dtype=np.uint32).reshape(-1, 4000);
        y = np.arange(0, 4000 * 10 * 4000, dtype=np.float64).reshape(-1, 4000);

        start = tm.time()
   
        z = np.multiply(x, y)

        end = tm.time()

        diff = end-start
        print("DifferentTypes calculations took %f milliseconds" %(diff))

    def test_Sundarrajan06295_SameTypes(self):

        x = np.arange(0, 4000 * 10 * 4000, dtype=np.float64).reshape(-1, 4000);
        y = np.arange(0, 4000 * 10 * 4000, dtype=np.float64).reshape(-1, 4000);

        start = tm.time()
   
        z = np.multiply(x, y)

        end = tm.time()

        diff = end-start
        print("SameTypes calculations took %f milliseconds" %(diff))

    def test_Sundarrajan06295_SameTypes_2(self):

        x = np.arange(0, 4000 * 10 * 4000, dtype=np.float64);
        y = np.arange(0, 4000 * 10 * 4000, dtype=np.float64);

        start = tm.time()
   
        z = np.multiply(x, y)

        end = tm.time()

        diff = end-start
        print("SameTypes calculations took %f milliseconds" %(diff))

    def test_Sundarrajan06295_Quantile_1(self):

        x = np.arange(0, 4000 * 10 * 4000, dtype=np.float64);

        start = tm.time()
   
        z = np.quantile(x, 0.5)

        end = tm.time()

        diff = end-start
        print("SameTypes calculations took %f milliseconds" %(diff))

    def test_Sundarrajan06295_var_1(self):

        data = np.arange(3815 * 2800, dtype = np.float32).reshape(3815,2800);
   
        start = tm.time()
   
        variance = np.var(data, axis=1)
        #print(variance)

        variance = np.var(data, axis=0)
        #print(variance)


        end = tm.time()

        diff = end-start
        print("var calculations took %f milliseconds" %(diff))

    def test_DeanZhuo_convolve_1(self):

        hlpf = np.arange(27, dtype = np.int64);
        hhpf = np.arange(35, dtype = np.int64);

        a = np.convolve(hlpf, hhpf, "full")
        print(a.size)
        print(a)


    def test_lintao185_2(self):

        gn = np.ones((10, 10, 10)).astype(np.int);
        bytes1 = gn.tobytes();
        bytes1[0] = 99;

 
        print(gn)   
      

    def test_GregTheDev_1(self):

       sampleData = np.random.rand(496, 682);
       filter = sampleData > 0.5;

       filteredData = np.where(filter, 0, sampleData);
       # filteredData2 = np.where(filter, 0d, sampleData);
       filteredData3 = np.where(filter, sampleData, sampleData);

    def test_ByteSwap_ReturnsCorrectValues_ForFloat32(self):
        A = np.array([1.0, 256.0, 8755.0], dtype=np.float32)
        B = A.byteswap(False)
        print(B)    # B = array([4.6006030e-41, 4.6011635e-41, 1.8737409e-38], dtype=float32)

    def test_ByteSwap_ReturnsCorrectValues_ForFloat64(self):
        A = np.array([1.0, 256.0, 8755.0], dtype=np.float64)
        B = A.byteswap(False)
        print(B)    # B = array([3.03865194e-319 1.41974704e-319 1.06183182e-314], dtype=float64)


    def test_Where_MaintainsOriginalDimensions(self):
    
        # This is testing whether different shapes for x & y arguments affect the outcome (answer: they don't)
        np.random.seed(5555);

        sampleData = np.random.rand(3, 4, 6);
        sampleData2 = np.random.rand(3, 4, 6);
        filter = sampleData > 0.5;
        #print(sampleData)
        #print(sampleData2)
        #print(filter)

        # scalar vs multi dimensional
        filteredData = np.where(filter, 0, sampleData);
        # Assert.That(filteredData.shape.iDims.Length, Is.EqualTo(3)); // filter.shape = (3, 496, 682), filteredData.shape = (3, 496, 682)
        #print(filter)
        #print(filteredData)

        # multi dimensional vs multi dimensional
        filteredData2 = np.where(filter, sampleData, sampleData2);
        #print(filteredData2)
        #Assert.That(filteredData2.shape.iDims.Length, Is.EqualTo(3)); // filter.shape = (3, 496, 682), filteredData2.shape = (3, 496, 682)

        #single dimensional vs multi dimensional (fails - shape of result drops a dimension)
        filter = np.max(sampleData, axis= 0) > 0.5; # shape = 496, 682
        print(filter)
        filteredData3 = np.where(filter, 0, sampleData2);
        print(filteredData3)

        filteredData4 = np.where(filter, sampleData2, 0);
        print(filteredData4)
        #Assert.That(filteredData3.shape.iDims.Length, Is.EqualTo(3)); // filter.shape = (496, 682), filteredData3.shape = (496, 682)

    def test_Where_DoesNotDuplicateResults(self):
        sampleData = np.array([ 1, 2, 3, 4, 5, 6, 7, 8 ]).reshape(2, 2, 2);
        filter = np.array([ True, False, True, False ]).reshape(2, 2);

        # 'split' the layers of sampleData into two seperate arrays of 2*2
        # dimA & dimB reflect expected values (1,2,3,4) & (5,6,7,8)
        dimA = sampleData[0];
        #print(dimA);
        dimB = sampleData[1];
        #print(dimB);

        # Use the same filter, but on each seperate array
        # In this case 'b' ends up with the same values as 'a'
        a = np.where(filter, dimA, dimA);
        print(a);
        b = np.where(filter, dimB, dimB);
        print(b);


    def test_williamlzw(self):

        arr = np.zeros(134)
        bb = np.array([True])
        print(bb)
        print("")


        xx = arr[1:] != arr[:-1]
        print(xx)
        print("")

        mask = np.append(bb, xx)
        print(mask)


    def test_nsmith29_1(self):

        a = np.arange(0,9, dtype = np.int).reshape(3,3)
        b = np.arange(0,9, dtype = np.int).reshape(3,3)

        row_nbytes = a.shape[1] * a.itemsize
        # costum dtype which represents a row as a single block.
        row_dtype = np.dtype((np.void, row_nbytes)) 

        # Views of both a & b with each row as a single element
        a_view = a.view(row_dtype).ravel()

        b_view = b.view(row_dtype).ravel()

        # From views find rows of a that are in b
        mask = np.isin(a_view, b_view)
        print(mask);

        return

if __name__ == '__main__':
    unittest.main()
