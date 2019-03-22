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
        d = a + b;
        print(d)
         

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

    def test_subtract_operations_2(self):

        a =  np.arange(100, 102, 1, dtype = np.int16)
        b =  np.array([1,63], dtype = np.int16)
        c = a-b
        print(a)
        print("****")
        print(b)
        print("****")
        print(c)
        print("****")

        a =  np.arange(0, 4, 1, dtype = np.int16).reshape((2,2))
        b =  np.array([65,78], dtype = np.int16).reshape(1,2)
        c = a-b
        print(a)
        print("****")
        print(b)
        print("****")
        print(c)


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

    def test_bitwise_xor(self):

        a = np.bitwise_xor(13, 17)
        print(a)

        b = np.bitwise_xor(31, 5)
        print(b)

        c = np.bitwise_xor([31,3], 5)
        print(c)

        d = np.bitwise_xor([31,3], [5,6])
        print(d)

        e = np.bitwise_xor([True, True], [False, True])
        print(e)

        return

    def test_bitwise_not(self):

        a = np.bitwise_not(13)
        print(a)

        b = np.bitwise_not(31)
        print(b)

        c = np.bitwise_not([31,3])
        print(c)

        d = np.bitwise_not([31,3])
        print(d)

        e = np.bitwise_not([True, False])
        print(e)

        return

    def test_invert(self):

        a = np.invert(13)
        print(a)

        b = np.invert(31)
        print(b)

        c = np.invert([31,3])
        print(c)

        d = np.invert([31,3])
        print(d)

        e = np.invert([True, False])
        print(e)

        return

    def test_right_shift(self):

        x = np.arange(1023, 1039, dtype= np.uint32).reshape(2, -1)
        y = np.right_shift(x, 2);
        z = x >> 2;

        print(x)
        print(y)
        print(z)

        return

    def test_right_shift_2(self):

        a = np.right_shift([10], [1,2,3]);
        print(a)

    def test_left_shift(self):

        x = np.arange(1023, 1039, dtype= np.uint32).reshape(2, -1)
        y = np.left_shift(x, 2);
        z = x << 2;

        print(x)
        print(y)
        print(z)

        return

    def test_left_shift_2(self):

        a = np.left_shift([10], [1,2,3]);
        print(a)

    def test_NAN(self):

        x = np.arange(1023, 1039, dtype= np.float).reshape(2, -1)
        x[:] = np.NaN

        print(x)

        return



    def test_average(self):

        x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.uint32).reshape(3, 2, -1)
        x = x * 3
        y = np.average(x);


        print(x)
        print(y)

        return


    def test_divide(self):

        a = np.divide(7,3)
        print(a)

        b = np.divide([1., 2., 3., 4.], 2.5)
        print(b)

        c = np.divide([1., 2., 3., 4.], [0.5, 2.5, 2.5, 3.5 ])
        print(c)

        return

    def test_true_divide(self):

        a = np.true_divide(7,3)
        print(a)

        b = np.true_divide([1., 2., 3., 4.], 2.5)
        print(b)

        c = np.true_divide([1., 2., 3., 4.], [0.5, 2.5, 2.5, 3.5 ])
        print(c)

        return

    def test_floor_divide(self):

        a = np.floor_divide(7,3)
        print(a)

        b = np.floor_divide([1., 2., 3., 4.], 2.5)
        print(b)

        c = np.floor_divide([1., 2., 3., 4.], [0.5, 2.5, 2.5, 3.5 ])
        print(c)

        return

    def test_divmod(self):

        a = np.divmod(7,3)
        print(a)

        b = np.divmod([1., 2., 3., 4.], 2.5)
        print(b)

        c = np.divmod([1., 2., 3., 4.], [0.5, 2.5, 2.5, 3.5 ])
        print(c)

        return

    def test_mod_1(self):

        x = np.mod([4, 7], [2, 3])
        print(x)

        y = np.mod(np.arange(7), 5)
        print(y)

        return

    def test_remainder_1(self):

        x = np.remainder([4, 7], [2, 3])
        print(x)

        y = np.remainder(np.arange(7), 5)
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


    def test_logical_and_1(self):
        a = np.logical_and(True, False)
        print(a)

        b = np.logical_and([True, False], [False, False])
        print(b)

        x = np.arange(5)
        c = np.logical_and(x>1, x<4)
        print(c)

        y = np.arange(6).reshape(2,3)
        d = np.logical_and(y>1, y<4)
        print(d)

    def test_logical_or_1(self):
        a = np.logical_or(True, False)
        print(a)

        b = np.logical_or([True, False], [False, False])
        print(b)

        x = np.arange(5)
        c = np.logical_or(x<1, x>3)
        print(c)

        y = np.arange(6).reshape(2,3)
        d = np.logical_or(y<1, y>3)
        print(d)

    def test_logical_xor_1(self):
        a = np.logical_xor(True, False)
        print(a)

        b = np.logical_xor([True, False], [False, False])
        print(b)

        x = np.arange(5)
        c = np.logical_xor(x<1, x>3)
        print(c)

        y = np.arange(6).reshape(2,3)
        d = np.logical_xor(y<1, y>3)
        print(d)

        e = np.logical_xor(0, np.eye(2))
        print(e)


    def test_logical_not_1(self):
 
        a = np.logical_not(3)
        print(a)

        b = np.logical_not([0, -1, 0, 1])
        print(b)

        x = np.arange(5)
        c = np.logical_not(x<3)
        print(c)

    def test_greater_1(self):

        a = np.greater([4, 2, 1], [2, 2, 2])
        print(a)

        b = np.greater([4, 2, 1], 1)
        print(b)

        c = np.greater(2, [4, 2, 1])
        print(c)

    def test_greater_equal_1(self):

        a = np.greater_equal([4, 2, 1], [2, 2, 2])
        print(a)

        b = np.greater_equal([4, 2, 1], 1)
        print(b)

        c = np.greater_equal(2, [4, 2, 1])
        print(c)

    def test_less_1(self):

        a = np.less([4, 2, 1], [2, 2, 2])
        print(a)

        b = np.less([4, 2, 1], 1)
        print(b)

        c = np.less(2, [4, 2, 1])
        print(c)


    def test_less_equal_1(self):

        a = np.less_equal([4, 2, 1], [2, 2, 2])
        print(a)

        b = np.less_equal([4, 2, 1], 1)
        print(b)

        c = np.less_equal(2, [4, 2, 1])
        print(c)

    def test_equal_1(self):

        a = np.equal([4, 2, 1], [2, 2, 2])
        print(a)

        b = np.equal([4, 2, 1], 1)
        print(b)

        c = np.equal(2, [4, 2, 1])
        print(c)

    def test_not_equal_1(self):

        a = np.not_equal([4, 2, 1], [2, 2, 2])
        print(a)

        b = np.not_equal([4, 2, 1], 1)
        print(b)

        c = np.not_equal(2, [4, 2, 1])
        print(c)

 

    def test_subtract_1(self):        

        a = np.subtract(2.0, 4.0)
        print(a)

        b = np.arange(9.0).reshape((3, 3))
        c = np.arange(3.0)
        d = np.subtract(b, c)
        print(d)

 

    def test_isfinite_1(self):

        a = np.isfinite(1)
        print(a)

        b = np.isfinite(0)
        print(b)

        c = np.isfinite(np.nan)
        print(c)
        
        d = np.isfinite(np.inf)
        print(d)

        e = np.isfinite(np.NINF)
        print(e)

        f = np.isfinite([np.log(-1.),1.,np.log(0)])
        print(f)


        x = np.array([-np.inf, 0., np.inf, np.inf]).reshape(2,2)
        y = np.array([2, 2, 2])
        g = np.isfinite(x)
        print(g)
        print(y)

    def test_isinf_1(self):

        a = np.isinf(1)
        print(a)

        b = np.isinf(0)
        print(b)

        c = np.isinf(np.nan)
        print(c)
        
        d = np.isinf(np.inf)
        print(d)

        e = np.isinf(np.NINF)
        print(e)

        f = np.isinf([np.log(-1.),1.,np.log(0)])
        print(f)


        x = np.array([-np.inf, 0., np.inf, np.inf]).reshape(2,2)
        y = np.array([2, 2, 2])
        g = np.isinf(x)
        print(g)
        print(y)

    def test_isneginf_1(self):

        a = np.isneginf(1)
        print(a)

        b = np.isneginf(0)
        print(b)

        c = np.isneginf(np.nan)
        print(c)
        
        d = np.isneginf(np.inf)
        print(d)

        e = np.isneginf(np.NINF)
        print(e)

        f = np.isneginf([np.log(-1.),1.,np.log(0)])
        print(f)


        x = np.array([-np.inf, 0., np.inf, np.inf]).reshape(2,2)
        y = np.array([2, 2, 2])
        g = np.isneginf(x)
        print(g)
        print(y)

    def test_isposinf_1(self):

        a = np.isposinf(1)
        print(a)

        b = np.isposinf(0)
        print(b)

        c = np.isposinf(np.nan)
        print(c)
        
        d = np.isposinf(np.inf)
        print(d)

        e = np.isposinf(np.NINF)
        print(e)

        f = np.isposinf([np.log(-1.),1.,np.log(0)])
        print(f)


        x = np.array([-np.inf, 0., np.inf, np.inf]).reshape(2,2)
        y = np.array([2, 2, 2])
        g = np.isposinf(x)
        print(g)
        print(y)

    def test_mat_1(self):

        a = np.matrix('1 2; 3 4')
        print(a)

        x = np.array([[1, 2], [3, 4]])
        m = np.asmatrix(x);
        n = np.mat(x)

        print(m)
        print(n)


if __name__ == '__main__':
    unittest.main()
