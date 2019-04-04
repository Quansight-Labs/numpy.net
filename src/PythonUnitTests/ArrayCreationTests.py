import unittest
import numpy as np
import matplotlib.pyplot as plt
from nptest import nptest
import operator

class ArrayCreationTests(unittest.TestCase):
    def test_PrintVersionString(self):
        print(np.__version__)


 

    def test_asfarray_1(self):

        a = np.asfarray([2, 3])
        print(a)

        b = np.asfarray([2, 3], dtype='float')
        print(b)

        c = np.asfarray([2, 3], dtype='int8')
        print(c)



    def test_asmatrix_1(self):

        x = np.array([[1, 2], [3, 4]])
        m = np.asmatrix(x)

        x[0,0] = 5

        print(m)
  

    def test_copy_1(self):
        x = np.array([1, 2, 3])
        y = x
        z = np.copy(x)
        # Note that, when we modify x, y changes, but not z:

        x[0] = 10
        print(x[0] == y[0])
        #True
        print(x[0] == z[0])
        #False

        
    def test_linspace_1(self):
        a = np.linspace(2.0, 3.0, num=5)
        print(a)

        b = np.linspace(2.0, 3.0, num=5, endpoint=False)
        print(b)

        c = np.linspace(2.0, 3.0, num=5, retstep=True)
        print(c)

    def test_logspace_1(self):
        a = np.logspace(2.0, 3.0, num=4)
        print(a)

        b = np.logspace(2.0, 3.0, num=4, endpoint=False)
        print(b)

        c = np.logspace(2.0, 3.0, num=4, base=2.0)
        print(c)


    def test_geomspace_1(self):
        a = np.geomspace(1, 1000, num=4)
        print(a)

        b = np.geomspace(1, 1000, num=3, endpoint=False)
        print(b)

        c = np.geomspace(1, 1000, num=4, endpoint=False)
        print(c)

        d = np.geomspace(1, 256, num=9)
        print(d)
 
    def test_OneDimensionalArray(self):
        l = [12.23, 13.32, 100, 36.32]
        print("Original List:",l)
        a = np.array(l)
        print("One-dimensional numpy array: ",a)
        print(a.shape)
        print(a.strides)
    
    def test_arange_2to11(self):
        a =  np.arange(2, 11, 1, dtype = np.int8)
        print(a)

        print(a.shape)
        print(a.strides)

    def test_arange_2to11_double(self):
        a =  np.arange(2.5, 11.5, 2)
        print(a)

        print(a.shape)
        print(a.strides)

    def test_arange_2to11_float(self):
        a =  np.arange(2.5, 37.7, 2.2, dtype=np.float32)
        print(a)

        print(a.shape)
        print(a.strides)
        
    def test_arange_reshape_33(self):
        a =  np.arange(2, 11).reshape(3,3)
        print(a)

        print(a.shape)
        print(a.strides)

    def test_arange_reshape_53(self):
        a =  np.arange(0, 15).reshape(5,3)
        print(a)

        print(a.shape)
        print(a.strides)



    def test_reverse_array(self):
      x = np.arange(0,40)
      print("Original array:")
      print(x)
      print("Reverse array:")
      x = x[::-1]
      print(x)

      y = x + 100
      print(y)

      z = x.reshape(5,-1)
      print(z)


    def test_1_OnBorder_0Inside(self):
      x = np.ones((15,15), dtype= np.double)
      print("Original array:")
      print(x)
      print(x.shape)
      print(x.strides)
      print("1 on the border and 0 inside in the array")
      x[1:-1,1:-1] = 0
      print(x)
      print(x.shape)
      print(x.strides)

    def test_1_OnBorder_0Inside_2(self):
      x = np.arange(0,225, dtype= np.double).reshape(15,15)
      print("Original array:")
      print(x)
      print(x.shape)
      print(x.strides)
      print("1 on the border and 0 inside in the array")
      x = x[1:-1,1:-1];
      print(x)
      print(x.shape)
      print(x.strides)

    def test_checkerboard_1(self):
        x = np.ones((3,3))
        print("Checkerboard pattern:")
        x = np.zeros((8,8),dtype=int)
        x[1::2,::2] = 1
        x[::2,1::2] = 1
        print(x)

    def test_F2C_1(self):
       fvalues = [0, 12, 45.21, 34, 99.91]
       F = np.array(fvalues)
       print("Values in Fahrenheit degrees:")
       print(F)
       print("Values in  Centigrade degrees:") 
       print(5*F/9 - 5*32/9)


    def test_RealImage_1(self):
        x = np.sqrt([1+0j])
        y = np.sqrt([0+1j])
        print("Original array:x ",x)
        print("Original array:y ",y)
        print("Real part of the array:")
        print(x.real)
        print(y.real)
        print("Imaginary part of the array:")
        print(x.imag)
        print(y.imag)

    def test_ArrayStats_1(self):
        x = np.array([1,2,3], dtype=np.float64)
        print("Size of the array: ", x.size)
        print("Length of one array element in bytes: ", x.itemsize)
        print("Total bytes consumed by the elements of the array: ", x.nbytes)


    def test_ndarray_flatten(self):
      x = np.arange(0.73,25.73, dtype= np.double).reshape(5,5)
      y = x.flatten()
      print(x)
      print(y)

      y = x.flatten(order='F')
      print(y)

      y = x.flatten(order='K')
      print(y)

 
 

    def test_ndarray_byteswap(self):
      x = np.arange(32,64, dtype= np.int16)
      print(x)
      y = x.byteswap(True)
      print(y)

      x = np.arange(32,64, dtype= np.int32)
      print(x)
      y = x.byteswap(True)
      print(y)

      x = np.arange(32,64, dtype= np.int64)
      print(x)
      y = x.byteswap(True)
      print(y)

    def test_ndarray_view(self):
      x = np.arange(256+32,256+64, dtype= np.int16)
      print(x)
      print(x.shape)
      print(x.dtype)

      y = x.view(np.uint8)
      print(y)
      print(y.shape)
      print(y.dtype)
     
      print("modifying data")
      y[1] = 99
      print(x)

      
    def test_ndarray_view_1(self):
      x = np.arange(0,32, dtype= np.int16).reshape(2,-1,4)
      print("X")
      print(x)
      print(x.shape)
 
      y = x.T
  
      print("Y")
      print(y)
      print(y.shape)

      z = y.view()
      z[0] = 99
      print("Z")
      print(z)
      print(z.shape)

      print("X")
      print(x)
      print("Y") 
      print(y)


    def test_ndarray_view2(self):
      x = np.arange(256+32,256+64, dtype= np.int16)
      print(x)
      print(x.shape)
      print(x.dtype)

      y = x.view(np.uint32)
      print(y)
      print(y.shape)
      print(y.dtype)
     
      print("modifying data")
      y[1] = 99
      y[5] = 88
      print(y)
      print(x)

      
    def test_ndarray_view2_reshape(self):
      x = np.arange(65470+32,65470+64, dtype= np.uint16).reshape(2,2,-1)
      print(x)
      print(x.shape)
      print(x.dtype)

      z = x[:,:,[2]]
      print(z)

      y = z.view().reshape(-1);
      print(y)
      print(y.shape)
      print(y.dtype)
     
    
    def test_ndarray_view3(self):
      x = np.arange(256+32,256+64, dtype= np.int16)
      print(x)
      print(x.shape)
      print(x.dtype)

      y = x.view(np.uint64)
      print(y)
      print(y.shape)
      print(y.dtype)
     
      print("modifying data")
      y[1] = 99
      y[5] = 88
      print(y)
      print(x)

    def test_ndarray_delete1(self):
      x = np.arange(0,32, dtype= np.int16).reshape(8,4)
      print("X")
      print(x)
      print(x.shape)
 
      y = np.delete(x, 0, axis=1)
      y[1] = 99
      print("Y")
      print(y)
      print(y.shape)


      print("X")
      print(x)


    def test_ndarray_delete2(self):
      x = np.arange(0,32, dtype= np.int16)
      print("X")
      print(x)
      print(x.shape)
 
      y = np.delete(x, 1,0)
  
      print("Y")
      print(y)
      print(y.shape)


      print("X")
      print(x)


    def test_ndarray_delete3(self):
      x = np.arange(0,32, dtype= np.int16).reshape(8,4)
      print("X")
      print(x)
      print(x.shape)
 
      mask = np.ones_like(x, dtype=np.bool)
      mask[:,[0]] = False
      print(mask)
      y = x[mask].reshape(8,3)
  
      print("Y")
      print(y)
      print(y.shape)


      print("X")
      print(x)

 


    def test_ndarray_unique_1(self):

      x = np.array([1,2,3,1,3,4,5,4,4]);

      print("X")
      print(x)

      uvalues, indexes, inverse, counts = np.unique(x, return_counts = True, return_index=True, return_inverse=True);

      print("uvalues")
      print(uvalues)
      print("indexes")
      print(indexes)
      print("inverse")
      print(inverse)
      print("counts")
      print(counts)

    def test_ndarray_unique_2(self):

      x = np.array([1,2,3,1,98,97,96,94,3,4,5,4,4,1,9,6,9,11,23,9,5,0,11,12]).reshape(6,4);
  
      print("X")
      print(x)
      uvalues, indexes, inverse, counts = np.unique(x, return_counts = True, return_index=True, return_inverse=True, axis=0);

      print("uvalues")
      print(uvalues)
      print("indexes")
      print(indexes)
      print("inverse")
      print(inverse)
      print("counts")
      print(counts)

      uvalues, indexes, inverse, counts = np.unique(x, return_counts = True, return_index=True, return_inverse=True, axis=1);

      print("uvalues")
      print(uvalues)
      print("indexes")
      print(indexes)
      print("inverse")
      print(inverse)
      print("counts")
      print(counts)




    def test_ndarray_where_1(self):
      x = np.array([1,2,3,1,3,4,5,4,4]).reshape(3,3)
      print("X")
      print(x)
      y = np.where(x == 3)
      print("Y")
      print(y)

    def test_ndarray_where_2(self):
      x = np.array([1,2,3,1,3,4,5,4,4], dtype=np.int32).reshape(3,3)
      print("X")
      print(x)
      y = np.where(x == 3)
      print("Y")
      print(y)

      z = x[y]
      print("Z")
      print(z)


    def test_ndarray_where_3(self):
      x = np.arange(0, 1000, dtype=np.int32).reshape(-1,10)
      #print("X")
      #print(x)
      y = np.where(x % 10 == 0)
      #print("Y")
      #print(y)

      z = x[y]
      print("Z")
      print(z)

    def test_ndarray_where_4(self):
      x = np.arange(0, 3000000, dtype=np.int32)
      #print("X")
      #print(x)
      y = np.where(x % 7 == 0)
      print("Y")
      print(y)

      z = x[y]
      m = np.mean(z);
      print("M")
      print(m)

    def test_ndarray_where_5(self):
        a = np.arange(10)

        b = np.where(a < 5, a, 10*a)
        print(b)

        a = np.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]])
        b = np.where(a < 4, a, -1)  # -1 is broadcast
        print(b)

        c = np.where([[True, False], [True, True]], 
                     [[1, 2], [3, 4]], 
                     [[9, 8], [7, 6]])
        print(c)

    def test_ndarray_unpackbits_1(self):
      x = np.arange(0,12, dtype=np.uint8).reshape(3,-1)
      print("X")
      print(x)
      y = np.unpackbits(x, 1);
      print("Y")
      print(y)

      z = np.packbits(y, 1)
      print("Z")
      print(z)

    def test_arange_slice_1(self):
        a =  np.arange(0, 1024, dtype=np.int16).reshape(2,4, -1)
        print("A")
        #print(a)
        print(a.shape)
        print(a.strides)

        b = a[:,:,122]
        print("B")
        print(b)
        print(b.shape)
        print(b.strides)

        c = a[:,:,[122]]
        print("C")
        print(c)
        print(c.shape)
        print(c.strides)

        c2 = np.arange(0,8, dtype=np.int16).reshape(2,4,1);
        print("C2")
        print(c2)
        print(c2.shape)
        print(c2.strides)

        d = a[:,:,[122,123]]
        print("D")
        print(d)
        print(d.shape)
        print(d.strides)


    def test_arange_slice_2(self):
        a =  np.arange(0, 32, dtype=np.int16).reshape(2,4, -1)
        print("A")
        #print(a)
        print(a.shape)
        print(a.strides)

        # b has unexpected strides.  If a copy from A is made first
        b = a[:,:,[2]]
        print("B")
        print(b)
        print(b.shape)
        print(b.strides)

  
        
    def test_arange_slice_2A(self):
        a =  np.arange(0, 32, dtype=np.int16).reshape(2,4, -1)
        print("A")
        #print(a)
        print(a.shape)
        print(a.strides)

        # b has unexpected strides.  If a copy from A is made first
        b = a[:,:, np.where(a > 20)]
        print("B")
        print(b)
        print(b.shape)
        print(b.strides)



    def test_arange_slice_2B(self):
        a =  np.arange(0, 32, dtype=np.int16).reshape(2,4, -1)
        b =  np.arange(100, 132, dtype=np.int16).reshape(2,4, -1)
        print("A")
        #print(a)
        print(a.shape)
        print(a.strides)

        # b has unexpected strides.  If a copy from A is made first
        b[:,:,[2]] = a[:,:,[2]]
        print("B")
        print(b)
        print(b.shape)
        print(b.strides)

    def test_arange_slice_2C(self):
        a =  np.arange(0, 32, dtype=np.int16).reshape(2,4, -1)
        b =  np.arange(100, 132, dtype=np.int16).reshape(2,4, -1)
        print("A")
        #print(a)
        print(a.shape)
        print(a.strides)

        b[:,:,[2]] |= a[:,:,[2]]
        print("B")
        print(b)
        print(b.shape)
        print(b.strides)


    def test_arange_slice_2C2(self):
        a =  np.arange(0, 32, dtype=np.int16).reshape(2,4, -1)
        b =  np.arange(100, 132, dtype=np.int16).reshape(2,4, -1)
        print("A")
        #print(a)
        print(a.shape)
        print(a.strides)

        # b has unexpected strides.  If a copy from A is made first
        aarray = a[:, :, [2]]
        barray = b[:, :, [2]]
        carray = barray | aarray
        print("B")
        print(carray)
        print(carray.shape)
        print(carray.strides)


    def test_ndarray_NAN(self):

        _max = 5
        output = np.ndarray(shape=(_max,), dtype = np.float);
        output[:] = np.NaN;

        print(output)
        print(output.shape)

    def test_insert_1(self):

        a = np.array([[1, 1], [2, 2], [3, 3]])
        b = np.insert(a, 1, 5);

        c = nptest.insert(a, 0, [999,100,101])

        print(a)
        print(a.shape)

        print("B")
        print(b)
        print(b.shape)
        print(b.strides)

        print("C")
        print(c)
        print(c.shape)
        print(c.strides)

    def test_insert_2(self):

        #print(np.source(np.insert))

        a = np.array([1, 1, 2, 2, 3, 3])
        b = np.array([90, 91, 92, 92, 93, 93])
        c = np.insert(a, slice(None), b);
        #d = nptest.insert(a, slice(None), b);

        print(a)
        print(a.shape)

        print("B")
        print(b)
        print(b.shape)
        print(b.strides)

        print("C")
        print(c)
        print(c.shape)
        print(c.strides)

        #print(d)
        #print(d.shape)


    def test_append_1(self):

        a = np.array([[1, 1], [2, 2], [3, 3]])
        b = np.append(a, 1);

        print(a)
        print(a.shape)

        print(b)
        print(b.shape)
        print(b.strides)

    def test_append_2(self):

        a = np.array([[1, 1], [2, 2], [3, 3]])
        b = np.append(a, [4,4]);

        print(a)
        print(a.shape)

        print(b)
        print(b.shape)
        print(b.strides)

    def test_append_3(self):

        a = np.array([[1, 1], [2, 2], [3, 3]])
        b = np.array([[4, 4], [5, 5], [6, 6]])
        c = np.append(a, b);

        print(a)
        print(a.shape)

        print(b)
        print(b.shape)

        print(c)
        print(c.shape)
        print(c.strides)

    def test_append_4(self):

        a = np.array([1, 1, 2, 2, 3, 3]).reshape(2,-1)
        b = np.array([4, 4, 5, 5, 6, 6]).reshape(2,-1)
        c = np.append(a, b, axis=1)

        print(a)
        print(a.shape)
        print("")

        print(b)
        print(b.shape)
        print("")

        print(c)
        print(c.shape)
        print(c.strides)
        print("")



    def test_flat_1(self):

        x = np.arange(10, 16).reshape(2,3);

        print(x)

        x.flat[3] = 9
        print(x)
        print(x.shape)
        print(x.strides)

        z = x.flat[3]
        print(z)
        print("")
        print("indexes")
        print("")

        for zz in x.flat:
            print(zz);
 
    def test_flat_2(self):

        x = np.arange(1, 7).reshape(2, 3)
        print(x)

        print(x.flat[3])

        print(x.T)
        print(x.T.flat[3])

        x.flat = 3
        print(x)

        x.flat[[1,4]] = 1
        print(x)


    def test_intersect1d_1(self):

        a = np.array([ 1, 3, 4, 3 ])
        b = np.array([ 3, 1, 2, 1 ])

        c = np.intersect1d(a,b)
        print(c)

    def test_setxor1d_1(self):

        a = np.array([1, 2, 3, 2, 4])
        b = np.array([2, 3, 5, 7, 5])
        c = np.setxor1d(a,b)

        print(c)

    def test_in1d_1(self):

        test = np.array([0, 1, 2, 5, 0])
        states = [0, 2]
        mask = np.in1d(test, states)
        print(mask)
        print(test[mask])

        mask = np.in1d(test, states, invert=True)
        print(mask)
        print(test[mask])

    def test_isin_1(self):

        element = 2*np.arange(4).reshape((2, 2));
        print(element)
  
        test_elements = [1, 2, 4, 8]
        mask = np.isin(element, test_elements)
        print(mask)
        print(element[mask])

        print("***********")

        mask = np.isin(element, test_elements, invert=True)
        print(mask)
        print(element[mask])

    def test_union1d_1(self):

        a = np.union1d([-1, 0, 1], [-2, 0, 2])
        print(a)
 

    def test_Ellipsis_indexing_1(self):
        
        a = np.array([10.0, 7, 4, 3, 2, 1])
        b = a[..., -1]
        print(b)
        print("********")

        a = np.array([[10.0, 7, 4], [3, 2, 1]])
        c = a[..., -1]
        print(c)
        print("********")

        TestData = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ]
        a = np.array(TestData, dtype= np.uint32).reshape((1, 3, 2, -1, 1));
        d = a[..., -1]
        print(d)
        print("********")

        e = a[0, ..., -1]
        print(e)
        print("********")

        f = a[0, :,:,:, -1]
        print(f)
        print("********")

        g = a[0, 1, ..., -1]
        print(g)
        print("********")

        h = a[0, 2, 1, ..., -1]
        print(h)
        print("********")
        
        i = a[:, 2, 1, 1, ...]
        print(i)


if __name__ == '__main__':
    unittest.main()

 