import unittest
import numpy as np
from nptest import nptest
from scipy.interpolate import interp1d

class Test_NumericOperationsTests(unittest.TestCase):
 
    def test_add_operations(self):
        a =  np.arange(0, 20, 1, dtype = np.int16)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a + 8
        print(b)
        print(b.shape)
        print(b.strides)

        a = np.arange(0, 20, 1, dtype = np.int64)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a + 2400
        print(b)
        print(b.shape)
        print(b.strides)

    def test_add_operations_2(self):
        a =  np.arange(0, 20, 1, dtype = np.int16)
        a = a.reshape(5,-1)
        print(a)
        
        b = np.array([2], dtype = np.int16);
        c = a + b;
        print(c)

        b = np.array([10,20,30,40], dtype = np.int16);
        c = a + b;
        print(c)



    def test_subtract_operations(self):
        a =  np.arange(0, 20, 1, dtype = np.int16)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a - 8
        print(b)
        print(b.shape)
        print(b.strides)

        a = np.arange(0, 20, 1, dtype = np.int64)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a - 2400
        print(b)
        print(b.shape)
        print(b.strides)

    #def test_subtract_operations_2(self):

    #    a =  np.arange(0, 4, 1, dtype = np.int16)
    #    b =  np.array([1,2], dtype = np.int16)
    #    c = a-b
    #    print(a)
    #    print(b)
    #    print(c)

    #    a =  np.arange(0, 4, 1, dtype = np.int16).reshape((2,2))
    #    b =  np.array([1,2], dtype = np.int16).reshape(1,2)
    #    c = a-b
    #    print(a)
    #    print(b)
    #    print(c)


    def test_multiply_operations(self):
        a =  np.arange(0, 20, 1, dtype = np.int16)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a * 8
        print(b)
        print(b.shape)
        print(b.strides)

        a = np.arange(0, 20, 1, dtype = np.int64)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a * 2400
        print(b)
        print(b.shape)
        print(b.strides)

    def test_division_operations(self):
        a =  np.arange(20000, 20020, 1, dtype = np.int16)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a / 8
        print(b)
        print(b.shape)
        print(b.strides)

        a = np.arange(2000000, 2000020, 1, dtype = np.int64)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a / 2400
        print(b)
        print(b.shape)
        print(b.strides)

    def test_leftshift_operations(self):
        a =  np.arange(0, 20, 1, dtype = np.int16)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a << 8
        print(b)
        print(b.shape)
        print(b.strides)

        a = np.arange(0, 20, 1, dtype = np.int64)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a << 24
        print(b)
        print(b.shape)
        print(b.strides)

    def test_leftshift_operations2(self):
        a =  np.arange(0, 20, 1, dtype = np.int8)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a << 16
        print(b)
        print(b.shape)
        print(b.strides)

        a = np.arange(0, 20, 1, dtype = np.int64)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a << 48
        print(b)
        print(b.shape)
        print(b.strides)



    def test_rightshift_operations(self):
        a =  np.arange(20000, 20020, 1, dtype = np.int16)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a >> 8
        print(b)
        print(b.shape)
        print(b.strides)

        a = np.arange(2123450, 2123470, 1, dtype = np.int64)
        a = a.reshape(5,-1)
        print(a)
        print(a.shape)
        print(a.strides)

        b = a >> 8
        print(b)
        print(b.shape)
        print(b.strides)

    def test_bitwiseand_operations(self):
        a =  np.arange(0, 32, 1, dtype = np.int16)
        print(a)

        b = a & 0x0f
        print(b)
        
        a = np.arange(2048, 2048+32, 1, dtype = np.int64)
        print(a)

        b = a & 0xFF
        print(b)

    def test_bitwiseor_operations(self):
        a =  np.arange(0, 32, 1, dtype = np.int16)
        print(a)

        b = a | 0x100
        print(b)
        
        a = np.arange(2048, 2048+32, 1, dtype = np.int64)
        print(a)

        b = a | 0x1000
        print(b)

    def test_bitwisexor_operations(self):
        a =  np.arange(0, 32, 1, dtype = np.int16)
        print(a)

        b = a ^ 0xAAA
        print(b)
        
        a = np.arange(2048, 2048+32, 1, dtype = np.int64)
        print(a)

        b = a ^ 0xAAAA
        print(b)

    def test_remainder_operations(self):
        a =  np.arange(0, 32, 1, dtype = np.int16)
        print(a)

        b = a % 6
        print(b)
        
        a = np.arange(2048, 2048+32, 1, dtype = np.int64)
        print(a)

        b = a % 6
        print(b)

    def test_power_operations(self):
        a =  np.arange(0, 32, 1, dtype = np.int16)
        print(a)

        b = np.power(a, 3.23)
        print(b)
        
        a = np.arange(2048, 2048+32, 1, dtype = np.int64)
        print(a)

        b = np.power(a, 4)
        print(b)

        b = np.power(a, 0)
        print(b)

        b = np.power(a, 0.5)
        print(b)

    def test_square_operations(self):
        a =  np.arange(0, 32, 1, dtype = np.int16)
        print(a)

        b = np.square(a)
        print(b)
        
        a = np.arange(2048, 2048+32, 1, dtype = np.int64)
        print(a)

        b = np.square(a)
        print(b)

    def test_reciprocal_operations(self):
        a =  np.arange(1, 32, 1, dtype = np.float32)
        print(a)

        b = np.reciprocal(a)
        print(b)
        
        a = np.arange(2048, 2048+32, 1, dtype = np.float64)
        print(a)

        b = np.reciprocal(a)
        print(b)

    def test_sqrt_operations(self):
        a =  np.arange(0, 32, 1, dtype = np.int16)
        print(a)

        b = np.sqrt(a)
        print(b)
        
        a = np.arange(2048, 2048+32, 1, dtype = np.int64)
        print(a)

        b = np.sqrt(a)
        print(b)

    def test_negative_operations(self):
        a =  np.arange(0, 32, 1, dtype = np.int16)
        print(a)

        b = -a
        print(b)

    def test_absolute_operations(self):
        a =  np.arange(-32, 32, 1, dtype = np.int16)
        print(a)

        b = np.absolute(a)
        print(b)
  
    def test_invert_operations(self):
        a =  np.arange(-32, 32, 1, dtype = np.int16)
        print(a)

        b = ~a
        print(b)


    def test_LESS_operations(self):
        a =  np.arange(-5, 5, 1, dtype = np.int16)
        print(a)

        b = a < -2
        print(b)

        
    def test_LESSEQUAL_operations(self):
        a =  np.arange(-5, 5, 1, dtype = np.int16)
        print(a)

        b = a <= -2
        print(b)

    def test_EQUAL_operations(self):
        a =  np.arange(-5, 5, 1, dtype = np.int16)
        print(a)

        b = a == -2
        print(b)

    def test_NOTEQUAL_operations(self):
        a =  np.arange(-5, 5, 1, dtype = np.int16)
        print(a)

        b = a != -2
        print(b)

    def test_GREATER_operations(self):
        a =  np.arange(-5, 5, 1, dtype = np.int16)
        print(a)

        b = a > -2
        print(b)

    def test_GREATEREQUAL_operations(self):
        a =  np.arange(-5, 5, 1, dtype = np.int16)
        print(a)

        b = a >= -2
        print(b)

    def test_LOGICALOR_operations(self):
        a =  np.arange(-5, 5, 1, dtype = np.int16)
        print(a)

        b = np.logical_or(a == 0, False)
        print(b)

    def test_arrayarray_or(self):
        a =  np.arange(0, 32, 1, dtype = np.int16)
        b =  np.arange(33, 33+32, 1, dtype = np.int16)
        c = a | b

        print("A")
        print(a)
        print("B")
        print(b)
        print("C")
        print(c)

    def test_bitwise_and(self):

        x = np.arange(1023, 1039, dtype= np.uint32).reshape(2, -1)
        y = np.bitwise_and(x, 0x3FF);
        z = x & 0x3FF;

        print(x)
        print(y)
        print(z)

        return


    def test_bitwise_or(self):

        x = np.arange(1023, 1039, dtype= np.uint32).reshape(2, -1)
        y = np.bitwise_or(x, 0x10);
        z = x | 0x10;

        print(x)
        print(y)
        print(z)

        return

    def test_right_shift(self):

        x = np.arange(1023, 1039, dtype= np.uint32).reshape(2, -1)
        y = np.right_shift(x, 2);
        z = x >> 2;

        print(x)
        print(y)
        print(z)

        return

    def test_left_shift(self):

        x = np.arange(1023, 1039, dtype= np.uint32).reshape(2, -1)
        y = np.left_shift(x, 2);
        z = x << 2;

        print(x)
        print(y)
        print(z)

        return

    def test_NAN(self):

        x = np.arange(1023, 1039, dtype= np.float).reshape(2, -1)
        x[:] = np.NaN

        print(x)

        return

    def test_diff_1(self):

        x = np.array([10,15,25,45,78,90], dtype= np.uint32)
        x = x * 3
        y = np.diff(x[1:]);


        print(x)
        print(y)

        return

    def test_diff_2(self):

        x = np.array([10,15,25,45,78,90], dtype= np.uint32).reshape(2, -1)
        x = x * 3
        y = np.diff(x, axis=0);


        print(x)
        print(y)

        return

    def test_diff_3(self):

        x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.uint32).reshape(3, 2, -1)
        x = x * 3
        y = np.diff(x,axis=2);


        print(x)
        print(y)

        return



    def test_average(self):

        x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.uint32).reshape(3, 2, -1)
        x = x * 3
        y = np.average(x);


        print(x)
        print(y)

        return

    def test_floor(self):

        x = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        y = np.floor(x);


        print(x)
        print(y)

        return
    
    def test_min(self):

        x = np.array([2.5, -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        y = np.min(x)

        print(x)
        print(y)

        return

    def test_max(self):

        x = np.array([2.5, -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        y = np.max(x)

        print(x)
        print(y)

        return

  

    def test_isnan(self):

        x = np.array([-1.7, np.nan, np.nan, 0.2, 1.5, np.nan, 2.0], dtype=np.float)
        y = np.isnan(x)
        z = x == np.nan

        print(x)
        print(y)
        print(z);

        return

    def test_setdiff1d(self):

        a = np.array([1, 2, 3, 2, 4, 1])
        b = np.array([3, 4, 5, 6])
        c = np.setdiff1d(a, b)

        print(a)
        print(b)
        print(c)

        return

    def test_setdiff1d_2(self):

        a = np.arange(1, 39, dtype= np.uint32).reshape(2, -1)
        b = np.array([3, 4, 5, 6])
        c = np.setdiff1d(a, b)

        print(a)
        print(b)
        print(c)

        return

    def test_interp1d(self):

        x = np.arange(2, 12, 2)
        #y = np.arange(1, 6, 1)

        y = np.exp(-x/3.0)

        #y = x/2.0
 
        f = interp1d(x, y)

        #xnew = np.arange(0,9, 1)
        ynew = f(x);
 
        print(x)
        print(y)
        print(ynew)

        #plt.plot(x, y, 'o', xnew, ynew, '-')
        #plt.show()

        return

    def test_interp1d_2(self):

        x = np.arange(0, 10)
        y = np.exp(-x/3.0)
        f = interp1d(x, y)

        xnew = np.arange(0, 9, 0.1)
        ynew = f(xnew)   # use interpolation function returned by `interp1d`

        print(x)
        print(y)

        print(xnew)
        print(ynew)

        return


    def test_rot90_1(self):

        m = np.array([[1,2],[3,4]], int)
        print(m)
        
        print("************")
        n = np.rot90(m)
        print(n)
        print("************")

        n = np.rot90(m, 2)
        print(n)
        print("************")

        m = np.arange(8).reshape((2,2,2))
        n = np.rot90(m, 1, (1,2))
        print(n)

    def test_flip_1(self):

        A = np.arange(8).reshape((2,2,2))
        B = np.flip(A, 0)
        print(A)
        print("************")
        print(B)
        print("************")
        C = np.flip(A, 1)
        print(C)
        print("************")

    def test_iterable_1(self):
        print(np.iterable([1, 2, 3]))
        print(np.iterable(2))

    def test_trim_zeros_1(self):
        a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
        print(np.trim_zeros(a))
        print(np.trim_zeros(a, 'b'))

    def test_fliplr_1(self):
        m = np.arange(8).reshape((2,2,2))
        n = np.fliplr(m)

        print(m)
        print(n)

    def test_flipud_1(self):
        m = np.arange(8).reshape((2,2,2))
        n = np.flipud(m)

        print(m)
        print(n)

    def test_diag_1(self):
        m = np.arange(9);
        n = np.diag(m)

        print(m)
        print(n)

        m = np.arange(9).reshape(3,3);
        n = np.diag(m)

        print(m)
        print(n)

    def test_diagflat_1(self):
        m = np.arange(1,5).reshape(2,2);
        n = np.diagflat(m)

        print(m)
        print(n)

        m = np.arange(1,3)
        n = np.diagflat(m, 1)

        print(m)
        print(n)

        m = np.arange(1,3)
        n = np.diagflat(m, -1)

        print(m)
        print(n)


    def test_tri_1(self):
        print(np.tri(3, 5, 2, dtype=int))
        print("***********")
        print(np.tri(3, 5, -1))


    def test_tril_1(self):

        a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
        b = np.tril(a, -1)
        print(a)
        print("***********")
        print(b)
  
    def test_triu_1(self):

        a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
        b = np.triu(a, -1)
        print(a)
        print("***********")
        print(b)


    def test_ediff1d_1(self):
        x = np.array([1, 2, 4, 7, 0])
        y = np.ediff1d(x)
        print(y)

        print(np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99])))

        y = [[1, 2, 4], [1, 6, 24]]
        print(np.ediff1d(y))



if __name__ == '__main__':
    unittest.main()
