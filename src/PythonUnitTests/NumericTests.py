import unittest
import numpy as np
import itertools
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
      x = np.full((100), 99).reshape(10,10)
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
      x = np.full((100),100)
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


    def test_asanyarray_2(self):
        
        a = [1, 2, 3, 4]
        b = [10, 20, 30, 40]

        c = np.asanyarray([a,b,a])
        print(c)
  
        aa = np.array(a);
        ba = np.array(b)

        ca = np.asanyarray([aa, ba])
        print(ca)

    def test_asanyarray_3(self):
        
        a = [1, 2, 3, 4]
        b = [10, 20, 30, 40]

        c = np.asanyarray([a,b, a])
        print(c)
  
        aa = np.array(a).reshape(2,2);
        ba = np.array(b).reshape(2,2);

        ca = np.asanyarray([aa, ba])
        print(ca)

    def test_asanyarray_4(self):
        
        a = [1, 2, 3, 4]
        b = [10, 20, 30, 40, 50]

        c = np.asanyarray([a,b, a])
        print(c)


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
 
           
    def test_outer_1(self):
       
        a = np.arange(2,10).reshape(2,4,1,1)
        b = np.arange(12,20).reshape(2,4)
        c = np.outer(a,b)
        print(c)
        print(c.shape)

        return

    def test_inner_1(self):

        a = np.arange(1,5, dtype = np.int16).reshape(2,2)
        b = np.arange(11,15, dtype = np.int32).reshape(2,2)
        c = np.inner(a,b)
        print(c)
       
        a = np.arange(2,10).reshape(2,4)
        b = np.arange(12,20).reshape(2,4)
        c = np.inner(a,b)
        print(c)
        print(c.shape)

        return

    def test_inner_2(self):

        a = np.array([True,False, False, True]).reshape(2,2)
        b = np.array([True,False, True, True]).reshape(2,2)
        c = np.inner(a,b)
        print(c)

        b = np.arange(11,15, dtype = np.int16).reshape(2,2)
        c = np.inner(a,b)
        print(c)

        c = np.inner(b,a)
        print(c)
       
        a = np.arange(0,80, dtype = np.int32).reshape(-1,4,5,2)
        b = np.arange(100,180, dtype= np.float).reshape(-1,4,5,2)
        c = np.inner(a,b)
        #print(c)
        print(c.shape)

        print(c.sum())
        print(c.sum(axis=1))
        print(c.sum(axis=2))

        return
           
    def test_tensordot_1(self):  
        
        a = np.arange(60.).reshape(3,4,5)
        b = np.arange(24.).reshape(4,3,2)
        c = np.tensordot(a,b, axes=([1,0],[0,1]))
        print(c.shape)
        print(c)
        return

    def test_tensordot_2(self):  
        
        a = np.arange(12).reshape(3,4)
        b = np.arange(24).reshape(4,3,2)
        c = nptest.tensordot(a,b, axes=1)
        print(c.shape)
        print(c)

        c = np.tensordot(a,b, axes=0)
        print(c.shape)
        #print(c)

        return


    def test_dot_1(self):

        a = [[1, 0], [0, 1]]
        b = [[4, 1], [2, 2]]
        c = np.dot(a, b)
        print(c)

        d = np.dot(3,4)
        print(d)

        e = np.arange(3*4*5*6).reshape((3,4,5,6))
        f = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
        g = np.dot(e, f)
        print(g.shape)
        print(g.sum())

        g = g[2,3,2,1,2,2]
        print(g)

    
    def test_roll_forward(self):
        a =  np.arange(10, dtype=np.uint16)
        print("A")
        print(a)
        print(a.shape)
        print(a.strides)

        b = np.roll(a, 2)
        print("B1")
        print(b)
        print(b.shape)
        print(b.strides)

        b = nptest.roll(a, 2)
        print("B2")
        print(b)
        print(b.shape)
        print(b.strides)

        #########################

        c = np.roll(b, 2)
        print("C1")
        print(c)
        print(c.shape)
        print(c.strides)

        c = nptest.roll(b, 2)
        print("C2")
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
        print("B1")
        print(b)
        print(b.shape)
        print(b.strides)

        b = nptest.roll(a, -2)
        print("B2")
        print(b)
        print(b.shape)
        print(b.strides)

        ###################

        c = np.roll(b, -6)
        print("C1")
        print(c)
        print(c.shape)
        print(c.strides)

        c = nptest.roll(b, -6)
        print("C2")
        print(c)
        print(c.shape)
        print(c.strides)

    def test_roll_with_axis(self):

        x = np.arange(10)
        A = np.roll(x, 2)
        print(A)
        #A = nptest.roll(x, 2)
        print(A)
        print("---------------")

        x2 = np.reshape(x, (2,5))
        B = np.roll(x2, 1)
        print(B)
        #B = nptest.roll(x2, 1)
        print(B)
        print("---------------")

        C = np.roll(x2, 1, axis=0)
        print(C)
        C = nptest.roll(x2, 1, axis=0)
        print(C)
        print("---------------")

        D = np.roll(x2, 1, axis=1)
        print(D)
        D = nptest.roll(x2, 1, axis=1)
        print(D)

    def test_roll_with_axis_2(self):
 
        x = np.arange(12)
        A = np.roll(x, 2)
        print("A")
        print(A)
 
        x2 = np.reshape(x, (2,2,3))
        B = np.roll(x2, 1)
        print("B")
        print(B)
   
        C = np.roll(x2, 1, axis=0)
        print("C")
        print(C)
  
        D = np.roll(x2, 1, axis=1)
        print("D")
        print(D)

        print("E")
        E = nptest.roll(x2, 1, axis=2)
        print(E)

    def test_roll_with_axis_3(self):
 
        x = np.arange(16)
        A = np.roll(x, 2)
        print("A")
        print(A)
 
        x2 = np.reshape(x, (2,2,2,2))
        B = np.roll(x2, 1)
        print("B")
        print(B)
   
        C = np.roll(x2, 1, axis=0)
        print("C")
        print(C)
  
        D = np.roll(x2, 1, axis=1)
        print("D")
        print(D)

        print("E")
        E = nptest.roll(x2, 1, axis=2)
        print(E)

        print("F")
        F = nptest.roll(x2, 1, axis=3)
        print(F)
 

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
        
        # Vector cross-product.
        x = [1, 2, 3]
        y = [4, 5, 6]
        a = np.cross(x, y)

        print(a)

        # One vector with dimension 2.
        x = [1, 2]
        y = [4, 5, 6]
        b = np.cross(x, y)
        print(b)

        # Equivalently:
        x = [1, 2, 0]
        y = [4, 5, 6]
        b = np.cross(x, y)
        print(b)

       # Both vectors with dimension 2.
        x = [1,2]
        y = [4,5]
        c = np.cross(x, y)
        print(c)

        return

    def test_cross_2(self): 

        # Multiple vector cross-products. Note that the direction of the cross
        # product vector is defined by the `right-hand rule`.

        x = np.array([[1,2,3], [4,5,6]])
        y = np.array([[4,5,6], [1,2,3]])
        a = np.cross(x, y)
        print(a)
        print("*********")

        # The orientation of `c` can be changed using the `axisc` keyword.

        b = np.cross(x, y, axisc=0)
        print(b)
        print("*********")

        # Change the vector definition of `x` and `y` using `axisa` and `axisb`.

        x = np.array([[1,2,3], [4,5,6], [7, 8, 9]])
        y = np.array([[7, 8, 9], [4,5,6], [1,2,3]])
        a = np.cross(x, y)
        print(a)
        print("*********")

        b = np.cross(x, y, axisa=0, axisb=0)
        print(b)
 
        return

    def test_indices_1(self):  
        
        grid = np.indices((2, 3))
        print(grid.shape)
        print(grid[0])
        print(grid[1])

        x = np.arange(20).reshape(5, 4)
        y = x[grid[0], grid[1]]
        print(y)


        return

    def test_fromfunction_1(self):    
        return

    def test_isscalar_1(self): 
        
        a = np.isscalar(3.1)
        print(a)

        b = np.isscalar(np.array(3.1))
        print(b)

        c = np.isscalar([3.1])
        print(c)

        d = np.isscalar(False)
        print(d)

        e = np.isscalar('numpy')
        print(e)

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
        
        a = np.allclose([1e10,1e-7], [1.00001e10,1e-8])
        print(a)

        b = np.allclose([1e10,1e-8], [1.00001e10,1e-9])
        print(b)

        c = np.allclose([1e10,1e-8], [1.0001e10,1e-9])
        print(c)

        d = np.allclose([1.0, np.nan], [1.0, np.nan])
        print(d)

        e = np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
        print(e)

        return

    def test_isclose_1(self): 
  
        a = np.isclose([1e10,1e-7], [1.00001e10,1e-8])
        print(a)

        b = np.isclose([1e10,1e-8], [1.00001e10,1e-9])
        print(b)

        c = np.isclose([1e10,1e-8], [1.0001e10,1e-9])
        print(c)

        d = np.isclose([1.0, np.nan], [1.0, np.nan])
        print(d)

        e = np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
        print(e)

        f = np.isclose([1e-8, 1e-7], [0.0, 0.0])
        print(f)

        g = np.isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0)
        print(g)

        h = np.isclose([1e-10, 1e-10], [1e-20, 0.0])
        print(h)

        i = np.isclose([1e-10, 1e-10], [1e-20, 0.999999e-10], atol=0.0)
        print(i)


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
