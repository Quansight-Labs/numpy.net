import unittest
import numpy as np
from nptest import nptest

class NumericTests(unittest.TestCase):

    def test_zeros_1(self):
      x = np.zeros(10)
      print(x)
      print("Update sixth value to 11")
      x[6] = 11
      print(x)
      print(x.shape)
      print(x.strides)

    def test_zeros_like_1(self):    

        a = [1, 2, 3, 4, 5, 6]
        b = np.zeros_like(a, dtype=None)

        b[2] = 99
        print(b)
        return

    def test_zeros_like_2(self):    

        a = [[1, 2, 3], [4, 5, 6]]
        b = np.zeros_like(a)

        b[1,2] = 99
        print(b)
        return

    def test_zeros_like_3(self):    

        a = [[[1, 2, 3], [4, 5, 6]]]
        b = np.zeros_like(a)

        b[0,0,2] = 99
        b[0,1,1] = 88

        print(b)
        return

    def test_ones_1(self):
      x = np.ones(10)
      print(x)
      print("Update sixth value to 11")
      x[6] = 11
      print(x)
      print(x.shape)
      print(x.strides)

      
    def test_ones_like_1(self):    

        a = [1, 2, 3, 4, 5, 6]
        b = np.ones_like(a, dtype=None)

        b[2] = 99
        print(b)
        return


    def test_ones_like_2(self):    

        a = [[1, 2, 3], [4, 5, 6]]
        b = np.ones_like(a)

        b[1,2] = 99
        print(b)
        return

    def test_ones_like_3(self):    

        a = [[[1, 2, 3], [4, 5, 6]]]
        b = np.ones_like(a)

        b[0,0,2] = 99
        b[0,1,1] = 88

        print(b)
        return
    
    def test_empty(self):

        a = np.empty((2,3))
        print(a)

        b = np.empty((2,4), np.int32)
        print(b)

    def test_empty_like_1(self):    

        a = [1, 2, 3, 4, 5, 6]
        b = np.empty_like(a, dtype=None)

        b[2] = 99
        print(b)
        return


    def test_empty_like_2(self):    

        a = [[1, 2, 3], [4, 5, 6]]
        b = np.empty_like(a)

        b[1,2] = 99
        print(b)
        return

    def test_empty_like_3(self):    

        a = [[[1, 2, 3], [4, 5, 6]]]
        b = np.empty_like(a)

        b[0,0,2] = 99
        b[0,1,1] = 88

        print(b)
        return
   

    def test_full_1(self):
      x = np.full(10, 99)
      print(x)
      print("Update sixth value to 11")
      x[6] = 11
      print(x)
      print(x.shape)
      print(x.strides)

  
    def test_full_2(self):
      x = np.arange(0,100).reshape(10,10)
      print(x)
      print("Update sixth value to 11")
      x[6] = 55
      print(x)
      print(x.shape)
      print(x.strides)
      #x[5,5] = 12
      #print(x)
      #print(x.shape)
      #print(x.strides)

    def test_full_3(self):
      x = np.arange(0,100)
      print(x)
      kevin = x[62]
      print(kevin)

    def test_full_4(self):
      x = np.arange(0,100).reshape(10,10)

      y = np.arange(1000,1010)

      x[2] = y;
      print(x)

    def test_full_5(self):
      x = np.arange(0,100).reshape(10,10)

      y = np.arange(1000,1010)

      np.put(x, 6, y)
      print(x)


    def test_full_like_1(self):    

        a = [1, 2, 3, 4, 5, 6]
        b = np.full_like(a, 66, dtype=None)

        b[2] = 99
        print(b)
        return


    def test_full_like_2(self):    

        a = [[1, 2, 3], [4, 5, 6]]
        b = np.full_like(a, 55)

        b[1,2] = 99
        print(b)
        return

    def test_full_like_3(self):    

        a = [[[1, 2, 3], [4, 5, 6]]]
        b = np.full_like(a, 33)

        b[0,0,2] = 99
        b[0,1,1] = 88

        print(b)
        return

    def test_count_nonzero_1(self):
        
        a = np.count_nonzero(np.eye(4))
        print(a)

        b = np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]])
        print(b)

        c = np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=0)
        print(c)

        d = np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=1)
        print(d)

        return

    
    def test_asarray_1(self):
        
        a = [1, 2]
        b = np.asarray(a)
        print(b)

        c = np.array([1, 2], dtype=np.float32)
        d = np.asarray(c, dtype=np.float32)
        print(d)

        e = np.asarray(a, dtype=np.float64)
        print(e)

    def test_asanyarray_1(self):
        
        a = [1, 2]
        b = np.asanyarray(a)
        print(b)

        c = np.array([1, 2], dtype=np.float32)
        d = np.asanyarray(c, dtype=np.float32)
        print(d)

        e = np.asanyarray(c, dtype=np.float64)
        print(e)

    def test_ascontiguousarray_1(self):
        
        x = np.arange(6).reshape(2,3)
        y = np.ascontiguousarray(x, dtype=np.float32)
        print(y)

        print(x.flags['C_CONTIGUOUS'])
        print(y.flags['C_CONTIGUOUS'])

    def test_asfortranarray_1(self):
        
        x = np.arange(6).reshape(2,3)
        y = np.asfortranarray(x, dtype=np.float32)
        print(y)

        print(x.flags['F_CONTIGUOUS'])
        print(y.flags['F_CONTIGUOUS'])

        
    def test_require_1(self):

        x = np.arange(6).reshape(2,3)
        print(x.flags)
 
        y = np.require(x, dtype=np.float32, requirements=['A', 'O', 'W', 'F'])
        print(y.flags)

        
    def test_isfortran_1(self): 
        
        a = np.array([[1, 2, 3], [4, 5, 6]], order='C')
        a1 = np.isfortran(a)
        print(a1)

        b = np.array([[1, 2, 3], [4, 5, 6]], order='FORTRAN')
        b1 = np.isfortran(b)
        print(b1)

        c = np.array([[1, 2, 3], [4, 5, 6]], order='C')
        c1 = np.isfortran(c)
        print(c1)

        d = a.T
        d1 = np.isfortran(d)
        print(d1)

        # C-ordered arrays evaluate as False even if they are also FORTRAN-ordered.

        e1 = np.isfortran(np.array([1, 2], order='FORTRAN'))
        print(e1)

        return

       
    def test_argwhere_1(self): 
        
        x = np.arange(6).reshape(2, 3)

        x1 = np.nonzero(x > 1)
        x2 = np.transpose(x1)

        y = np.argwhere(x > 1)
        print(y)

        a = np.arange(12).reshape(2, 3, 2)
        b = np.argwhere(a > 1)
        print(b)

        return
           
    def test_flatnonzero_1(self): 
        
        x = np.arange(-2, 3)

        y = np.flatnonzero(x)
        print(y)

        # Use the indices of the non-zero elements as an index array to extract these elements:

        z = x.ravel()[np.flatnonzero(x)]
        print(z)

        return
           
    def test_correlate_1(self):    
        return
           
    def test_convolve_1(self):    
        return
           
    def test_outer_1(self):
        
        a = np.arange(2,10).reshape(2,4)
        b = np.arange(12,20).reshape(2,4)
        c = np.outer(a,b)
        print(c)

        return
           
    def test_tensordot_1(self):    
        return

    
    def test_roll_forward(self):
        a =  np.arange(10, dtype=np.uint16)
        print("A")
        print(a)
        print(a.shape)
        print(a.strides)

        b = np.roll(a, 2)
        print("B")
        print(b)
        print(b.shape)
        print(b.strides)

        c = np.roll(b, 2)
        print("C")
        print(c)
        print(c.shape)
        print(c.strides)

    def test_roll_backward(self):
        a =  np.arange(10, dtype=np.uint16)
        print("A")
        print(a)
        print(a.shape)
        print(a.strides)

        b = np.roll(a, -2)
        print("B")
        print(b)
        print(b.shape)
        print(b.strides)

        c = np.roll(b, -6)
        print("C")
        print(c)
        print(c.shape)
        print(c.strides)

    def test_ndarray_rollaxis(self):

        a = np.ones((3,4,5,6))
        b = np.rollaxis(a, 3, 1).shape
        print(b)

        c = np.rollaxis(a, 2).shape
        print(c)

        d = np.rollaxis(a, 1, 4).shape
        print(d)

    def test_ndarray_moveaxis(self):

        x = np.zeros((3, 4, 5))
        b = np.moveaxis(x, 0, -1).shape
        print(b)

        c = np.moveaxis(x, -1, 0).shape
        print(c)

        #These all achieve the same result:
        d = np.transpose(x).shape
        print(d)
    
        e = np.swapaxes(x, 0, -1).shape
        print(e)

        f = np.moveaxis(x, [0, 1], [-1, -2]).shape
        print(f)

        g = np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
        print(g)


    def test_cross_1(self):    
        return

    def test_indices_1(self):    
        return

    def test_fromfunction_1(self):    
        return

    def test_isscalar_1(self):    
        return

    def test_binary_repr(self):    
        return

    def test_base_repr(self):    
        return


    def test_identity_1(self):

        a = np.identity(2, dtype = np.float)

        print(a)
        print(a.shape)
        print(a.strides)
        print("")

        b = np.identity(5, dtype = np.int8)

        print(b)
        print(b.shape)
        print(b.strides)

    def test_allclose_1(self):    
        return

    def test_isclose_1(self):    
        return


    def test_array_equal_1(self):
        a =np.array_equal([1, 2], [1, 2])
        print(a)
        b = np.array_equal(np.array([1, 2]), np.array([1, 2]))
        print(b)
        c = np.array_equal([1, 2], [1, 2, 3])
        print(c)
        d = np.array_equal([1, 2], [1, 4])
        print(d)


    def test_array_equiv_1(self):
        
        a =np.array_equiv([1, 2], [1, 2])
        print(a)
        b = np.array_equiv([1, 2], [1, 3])
        print(b)
        c = np.array_equiv([1, 2], [[1, 2], [1, 2]])
        print(c)
        d = np.array_equiv([1, 2], [[1, 2, 1, 2], [1, 2, 1, 2]])
        print(d)
        
        e = np.array_equiv([1, 2], [[1, 2], [1, 3]])
        print(e)

if __name__ == '__main__':
    unittest.main()
