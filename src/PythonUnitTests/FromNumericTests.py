import unittest
import numpy as np
from nptest import nptest


class FromNumericTests(unittest.TestCase):

    def test_take_1(self):
        a = [4, 3, 5, 7, 6, 8, 9, 12, 14, 16, 18, 20, 22, 24, 26, 28]
        indices = [0, 1, 4]
        b = np.take(a, indices)

        print("B")
        print(b)
        print(b.shape)

        a = np.array(a)
        print(a[indices])

        c = np.take(a, [[0, 1], [2, 3]])
        print("C")
        print(c)
        print(c.shape)
        print(c.strides)

        d = np.take(a.reshape(4,-1), [[0, 1], [2, 3]], axis=0)
        print("D")
        print(d)
        print(d.shape)
        print(d.strides)

        e = np.take(a.reshape(4,-1), [[0, 1], [2, 3]], axis=1)
        print("E")
        print(e)
        print(e.shape)
        print(e.strides)

    def test_take_along_axis_1(self):

        a = np.array([[10, 30, 20], [60, 40, 50]])
        b = np.sort(a, axis=1)
        #print(b)

        ai = np.argsort(a, axis=1)
        print(ai)
        print("*********")

        c = np.take_along_axis(a, ai, axis=1)
        print(c)

        print("*********")

        d = np.take(a, ai)
        print(d)

        print("*********")

        aa = a[0,:]
        e = np.take(a[0,:], ai)
        print(e)

        return;

    def test_reshape_1(self):
        a = np.arange(6).reshape((3, 2))
        print(a)
        print("")

        print(np.reshape(a, (2, 3)))    # C-like index ordering
        print("")
        print(np.reshape(np.ravel(a), (2, 3))) # equivalent to C ravel then C reshape
        print("")
        print(np.reshape(a, (2, 3), order='F')) # Fortran-like index ordering
        print("")
        print(np.reshape(np.ravel(a, order='F'), (2, 3), order='F'))

    def test_ravel_1(self):
        
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.ravel(a)
        print(b)

        c = a.reshape(-1)
        print(c)

        d = np.ravel(a, order='F')
        print(d)

        # When order is 'A', it will preserve the array's 'C' or 'F' ordering:
        e = np.ravel(a.T)
        print(e)

        f = np.ravel(a.T, order='A');
        print(f)

    def test_ravel_2(self):
        # When order is 'K', it will preserve orderings that are neither 'C' nor 'F', but won't reverse axes:

        a = np.arange(3)[::-1];
        print(a)

        b = a.ravel(order='C')
        print(b)

        c = a.ravel(order='K')
        print(c)

    def test_ravel_3(self):

        a = np.arange(12).reshape(2,3,2).swapaxes(1,2);
        print(a)

        b = a.ravel(order='C')
        print(b)

        c = a.ravel(order='K')
        print(c)

    def test_choose_1(self):

        choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        a = np.choose([2, 3, 1, 0], choices)
   
        print(a)

        return

    def test_choose_2(self):

        choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        a = np.choose([2, 4, 1, 0], choices, mode='clip')
        print(a)

        a = np.choose([2, 4, 1, 0], choices, mode='wrap')
        print(a)

        try:
            a = np.choose([2, 4, 1, 0], choices, mode='raise')
            print(a)
        except:
            return

        assert(False)
        return

    def test_choose_3(self):

        a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        choices = [-10, 10]
        b = np.choose(a, choices)

        print(b)
    
    def test_choose_4(self):

        a = np.array([0, 1]).reshape((2,1,1))
        c1 = np.array([1, 2, 3]).reshape((1,3,1))
        c2 = np.array([-1, -2, -3, -4, -5]).reshape((1,1,5))
        b = np.choose(a, (c1, c2)) # result is 2x3x5, res[0,:,:]=c1, res[1,:,:]=c2

        print(b)

    def test_select_1(self):

        x = np.arange(10)
        condlist = [x<3, x>5]
        choicelist = [x, x**2]
        y = np.select(condlist, choicelist)
        print(y)

    def test_repeat_1(self):

        x = np.array([1,2,3,4]).reshape(2,2)
        y = np.array([2])
        z = np.repeat(x, y)
        print(z)
        print("")

        z = np.repeat(3, 4)
        print(z)
        print("")

        z = np.repeat(x, 3, axis=0)
        print(z)
        print("")

        z = np.repeat(x, 3, axis=1)
        print(z)
        print("")

        z = np.repeat(x, [1, 2], axis=0)
        print(z)

        return

    def test_put_1(self):

        a = np.arange(5)
        np.put(a, [0, 2], [-44, -55])
        print(a)

        a = np.arange(5)
        np.put(a, 22, -5, mode='clip')
        print(a)
        
        a = np.arange(5)
        np.put(a, 22, -5, mode='wrap')
        print(a)

        try:
            a = np.arange(5)
            np.put(a, 22, -5, mode='raise')
            print(a)    
        except:
            return

        raise Exception("this should have caused exception")

    def test_put_2(self):

        a = np.arange(15)
        np.put(a[:5], [0, 2], [-44, -55])
        print(a)

        a = np.arange(15)
        np.put(a[:5], 22, -5, mode='clip')
        print(a)
        
        a = np.arange(15)
        np.put(a[:5], 22, -5, mode='wrap')
        print(a)

        try:
            a = np.arange(15)
            np.put(a[:5], 22, -5, mode='raise')
            print(a)    
        except:
            return

        raise Exception("this should have caused exception")

    def test_putmask_1(self):

        x = np.arange(6).reshape(2, 3)
        np.putmask(x, x>2, x**2)
        print(x)


        # If values is smaller than a it is repeated:

        x = np.arange(5)
        np.putmask(x, x>1, [-33, -44])
        print(x)

        return

    def test_swapaxes_1(self):

        x = np.array([[1,2,3]])
        print(x)
        print("********")

        y = np.swapaxes(x,0,1)
        print(y)
        print("********")

        x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
        print(x)
        print("********")

        y = np.swapaxes(x,0,2)
        print(y)

    def test_ndarray_T_1(self):
      x = np.arange(0,32, dtype= np.int16).reshape(8,4)
      print("X")
      print(x)
      print(x.shape)
 
      y = x.T
  
      print("Y")
      print(y)
      print(y.shape)

    def test_ndarray_T_2(self):
      x = np.arange(0,32, dtype= np.int16)
      print("X")
      print(x)
      print(x.shape)
 
      y = x.T
  
      print("Y")
      print(y)
      print(y.shape)

    def test_ndarray_T_3(self):
      x = np.arange(0,32, dtype= np.int16).reshape(2,-1,4)
      print("X")
      print(x)
      print(x.shape)
 
      y = x.T
  
      print("Y")
      print(y)
      print(y.shape)

    def test_ndarray_T_4(self):
      x = np.arange(0,64, dtype= np.int16).reshape(2,4,-1,4)
      print("X")
      print(x)
      print(x.shape)
 
      y = x.T
  
      print("Y")
      print(y)
      print(y.shape)

    def test_ndarray_transpose_1(self):
      x = np.arange(0,64, dtype= np.int16).reshape(2,4,-1,4)
      print("X")
      print(x)
      print(x.shape)
 
      y = np.transpose(x, (1,2,3,0))
  
      print("Y")
      print(y)
      print(y.shape)

    def test_ndarray_transpose_2(self):
      x = np.arange(0,64, dtype= np.int16).reshape(2,4,-1,4)
      print("X")
      print(x)
      print(x.shape)
 
      y = np.transpose(x, (3,2,1,0))
  
      print("Y")
      print(y)
      print(y.shape)

    def test_partition_1(self):

     a = np.array([3, 4, 2, 1])
     b = np.partition(a, 3)
     print(a)
     print(b)
     print("********")
     a = np.array([3, 4, 2, 1])
     b = np.partition(a, 0)
     print(a)
     print(b)
 
     print("********")

     c = np.partition(a, (1, 3))
     print(c)

    def test_partition_2(self):

     a = np.array([3.1, 4.1, 2.1, 1.1])
     b = np.partition(a, 3)
     print(a)
     print(b)
     print("********")
     a = np.array([3.1, 4.1, 2.1, 1.1])
     b = np.partition(a, 0)
     print(a)
     print(b)
 
     print("********")

     c = np.partition(a, (1, 3))
     print(c)

    def test_partition_3(self):

        a = np.arange(22, 10, -1).reshape(3,4,1)
        b = np.partition(a, 1, axis=0)
        print(b)
        print("********")
        c = np.partition(a, 2, axis=1)
        print(c)
        print("********")
        d = np.partition(a, 0, axis=2)
        print(d)

    def test_partition_4(self):

        a = np.arange(22, 10, -1, dtype=np.float64).reshape(3,4,1)
        b = np.partition(a, 1, axis=0)
        print(b)
        print("********")
        c = np.partition(a, 2, axis=1)
        print(c)
        print("********")
        d = np.partition(a, 0, axis=2)
        print(d)
    
    def test_argpartition_1(self):

     a = np.array([3, 4, 2, 1])
     b = np.argpartition(a, 3)
     print(b)
 
     print("********")

     c = np.argpartition(a, (1, 3))
     print(c)

    def test_argpartition_2(self):

     a = np.array([3.1, 4.1, 2.1, 1.1])
     b = np.argpartition(a, 3)
     print(a)
     print(b)
     print("********")
     a = np.array([3.1, 4.1, 2.1, 1.1])
     b = np.argpartition(a, 0)
     print(a)
     print(b)
 
     print("********")

     c = np.argpartition(a, (1, 3))
     print(c)

    def test_argpartition_3(self):

        a = np.arange(22, 10, -1).reshape(3,4,1)
        b = np.argpartition(a, 1, axis=0)
        print(b)
        print("********")
        c = np.argpartition(a, 2, axis=1)
        print(c)
        print("********")
        d = np.argpartition(a, 0, axis=2)
        print(d)

    def test_argpartition_4(self):

        a = np.arange(22, 10, -1, dtype=np.float64).reshape(3,4,1)
        b = np.argpartition(a, 1, axis=0)
        print(b)
        print("********")
        c = np.argpartition(a, 2, axis=1)
        print(c)
        print("********")
        d = np.argpartition(a, 0, axis=2)
        print(d)

    def test_sort_1(self):

        a = np.array([[1,4],[3,1]])
        b = np.sort(a)                # sort along the last axis
        print(b)
        print("********")

        c = np.sort(a, axis=None)     # sort the flattened array
        print(c)
        print("********")

        d = np.sort(a, axis=0)        # sort along the first axis
        print(d)
        print("********")


    def test_sort_2(self):

        a = np.arange(32.2, 0.2, -1.0)

        print(a)


        b = np.sort(a)                # sort along the last axis
        print(b)
        print("********")

        c = np.sort(a, axis=None)     # sort the flattened array
        print(c)
        print("********")

        d = np.sort(a, axis=0)        # sort along the first axis
        print(d)
        print("********")

    def test_msort_1(self):

        a = np.array([[1,4],[3,1]])
        b = np.msort(a)               
        print(b)
        print("********")

        a = np.arange(32.2, 0.2, -1.0)
        b = np.msort(a)               
        print(b)



    def test_ndarray_argsort_1(self):

      x = np.array([1,2,3,1,3,4,5,4,4])
      ar = np.array([3,2,1]);
      perm1 = np.argsort(x, kind = 'mergesort')
      perm2 = np.argsort(ar, kind = 'quicksort')
      perm3 = np.argsort(ar)

      print(perm1)
      print(perm2)
      print(perm3)

    def test_ndarray_argsort_2(self):

      ar = np.array([1,2,3,1,3,4,5,4,4,1,9,6,9,11,23,9,5,0,11,12]).reshape(5,4);
      perm1 = np.argsort(ar, kind = 'mergesort')
      perm2 = np.argsort(ar, kind = 'quicksort')
      perm3 = np.argsort(ar)

      print(perm1)
      print(perm2)
      print(perm3)

    def test_argmax_1(self):
        a = np.array([32,33,45,98,11,02]).reshape(2,3)
        print(a)

        b = np.argmax(a)
        print(b)
        print("********")

        c = np.argmax(a, axis=0)
        print(c)
        print("********")

        d = np.argmax(a, axis=1)
        print(d)
        print("********")

    def test_argmin_1(self):
        a = np.array([32,33,45,98,11,02]).reshape(2,3)
        print(a)

        b = np.argmin(a)
        print(b)
        print("********")

        c = np.argmin(a, axis=0)
        print(c)
        print("********")

        d = np.argmin(a, axis=1)
        print(d)
        print("********")

    def test_searchsorted_1(self):

        a = np.searchsorted([1,2,3,4,5], 3)
        print(a)
        b = np.searchsorted([1,2,3,4,5], 3, side='right')
        print(b)
        c = np.searchsorted([1,2,3,4,5], [-10, 10, 2, 3])
        print(c)

        d = np.searchsorted([15,14,13,12,11], 13)
        print(d)

    def test_resize_1(self):

        a=np.array([[0,1],[2,3]])
        print(a)

        b = np.resize(a,(2,3))
        print(b)

        c = np.resize(a,(1,4))
        print(c)

        d = np.resize(a,(2,4))
        print(d)

    def test_squeeze_1(self):

        x = np.array([[[0], [1], [2]]])
        print(x)
        
        a = np.squeeze(x)
        print(a)

        b = np.squeeze(x, axis=0)
        print(b)

        caughtException = False
        try:
            c = np.squeeze(x, axis=1)
            print(c)
        except:
             caughtException = True

        assert(caughtException == True)

        d = np.squeeze(x, axis=2)
        print(d)

    def test_squeeze_2(self):

        x = np.arange(0,32, 1, dtype=np.float32).reshape(-1,1,8,1)
        print(x)
        
        a = np.squeeze(x)
        print(a)

        b = np.squeeze(x, axis=1)
        print(b)

        caughtException = False
        try:
            c = np.squeeze(x, axis=0)
            print(c)
        except:
             caughtException = True

        assert(caughtException == True)

        caughtException = False
        try:
            d = np.squeeze(x, axis=2)
            print(d)
        except:
             caughtException = True

        assert(caughtException == True)

        e = np.squeeze(x, axis=3)
        print(e)


    def test_diagonal_1(self):

        a = np.arange(4).reshape(2,2)
        print(a)
        print("*****")
    
        b = a.diagonal()
        print(b)
        print("*****")

        c = a.diagonal(1)
        print(c)
        print("*****")

        a = np.arange(8).reshape(2,2,2); 
        print(a)
        print("*****")
        b = a.diagonal(0, # Main diagonals of two arrays created by skipping
                       0, # across the outer(left)-most axis last and
                       1) # the "middle" (row) axis first.

        print(b)
        print("*****")

        print(a[:,:,0])
        print("*****")

        print(a[:,:,1])
        print("*****")
  
    def test_trace_1(self):

       a = np.trace(np.eye(3))
       print(a) 
       print("*****")
 
       a = np.arange(8).reshape((2,2,2))
       b = np.trace(a)
       print(b)
       print("*****")

       a = np.arange(24).reshape((2,2,2,3))
       c = np.trace(a);
       print(c)

    def test_nonzero_1(self):

        x = np.array([[1,0,0], [0,2,0], [1,1,0]])
        print(x)
        print("*****")

        y = np.nonzero(x)
        print(y)
        print("*****")

        z = x[np.nonzero(x)]
        print(z)
        print("*****")

        q = np.transpose(np.nonzero(x))
        print(q)

    def test_compress_1(self):
 
        a = np.array([[1, 2], [3, 4], [5, 6]])
        print(a)
        print("*****")

        b = np.compress([0, 1], a, axis=0)
        print(b)
        print("*****")

        c = np.compress([False, True, True], a, axis=0)
        print(c)
        print("*****")

        d = np.compress([False, True], a, axis=1)
        print(d);
        print("*****")

        e = np.compress([False, True], a);
        print(e)

  
 

    def test_any_1(self):

        x = np.array([2.5, -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        y = np.any(x)

        print(x)
        print(y)

        x = np.array([0.0,0.0,0.0,0.0])
        y = np.any(x)

        print(x)
        print(y)

        return

    def test_any_2(self):

        a = np.any([[True, False], [True, True]])
        print(a)
        print("*****") 
        
        b = np.any([[True, False], [False, False]], axis=0)
        print(b)
        print("*****") 

        c = np.any([-1, 0, 5])
        print(c)
        print("*****") 

        d = np.any(np.nan)
        print(d)
        print("*****")
        
        #o=np.array([False])
        #z=np.any([-1, 4, 5], out=o)
        #print(z, o)

    def test_any_3(self):

        x = np.array([2.5, -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, 0]).reshape(3,3)
        print(x)

        y = np.any(x, axis = 0)
        print(y)
   
        y = np.any(x, axis = 1)
        print(y)

        x = np.array([2.5, -1.7, -1.5, -0.2, 0.2, 1.5, 0, 0, 0]).reshape(3,3)
        print(x)

        y = np.any(x, axis = 0)
        print(y)
   
        y = np.any(x, axis = 1)
        print(y)
   

        return


    def test_all_1(self):

        x = np.array([2.5, -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        y = np.all(x)

        print(x)
        print(y)

        x = np.array([1.0,1.0,0.0,1.0])
        y = np.all(x)

        print(x)
        print(y)

        return

    def test_all_2(self):

        a = np.all([[True,False],[True,True]])
        print(a)

        b = np.all([[True,False],[True,True]], axis=0)
        print(b)

        b = np.all([[True,False],[True,True]], axis=1)
        print(b)

        c = np.all([-1, 4, 5])
        print(c)

        d = np.all([1.0, np.nan])
        print(d)

        #o=np.array([False])
        #z=np.all([-1, 4, 5], out=o)
        #print(o)
        #print(z)

        
  


    def test_size_2(self):

        a = np.array([[1,2,3],[4,5,6]])
        print(np.size(a))
        print(np.size(a,1))
        print(np.size(a,0))

    def test_around_1(self):

        a = np.around([0.37, 1.64])
        print(a)

        b = np.around([0.37, 1.64], decimals=1)
        print(b)

        c = np.around([.5, 1.5, 2.5, 3.5, 4.5]) # rounds to nearest even value
        print(c)

        d = np.around([1,2,3,11], decimals=1) # ndarray of ints is returned
        print(d)

        e = np.around([1,2,3,11], decimals=-1)
        print(e)

        
    def test_ndarray_mean_1(self):
      x = np.arange(0,12, dtype=np.uint8).reshape(3,-1)
      print("X")
      print(x)
      y = np.mean(x);
      print("Y")
      print(y)

      y = np.mean(x, axis= 0);
      print("Y")
      print(y)

      y = np.mean(x, axis= 1);
      print("Y")
      print(y)
   


    def test_place_1(self):

        arr = np.arange(6).reshape(2, 3)
        np.place(arr, arr>2, [44, 55])
        print(arr)


    def test_extract_1(self):

        arr = np.arange(12).reshape((3, 4))
        condition = np.mod(arr, 3)==0
        print(condition)

        b = np.extract(condition, arr)
        print(b)
 
        
    def test_indicesfromaxis_1(self):
        TestData = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ]
        a = np.zeros_like(TestData, dtype= np.uint32).reshape((3, 2, -1));
         #print(a);


        a[:, 0, 0] = 99;
        #print(a);
        #UpdateArrayByAxis(a, 0, 99);
        #print(a.ravel());
        print(np.sum(a, axis=0))
        print("************")

        a[0, :, 0] = 11;
        #print(a);
        #UpdateArrayByAxis(a, 1, 11);
        #print(a.ravel());
        print(np.sum(a, axis=1))
        print("************")

        a[0, 0, :] = 22;
        #print(a);
        #UpdateArrayByAxis(a, 1, 22);
        #print(a.ravel());
        print(np.sum(a, axis=2))

        print(np.sum(a))


if __name__ == '__main__':
    unittest.main()
