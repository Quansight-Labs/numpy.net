import unittest
import numpy as np
from nptest import nptest

class MathematicalFunctionsTests(unittest.TestCase):

    #region Trigonometric Functions

    def test_sin_1(self):

        a = np.arange(0, 10, dtype = np.float64)
        a = a[::2]
        b = np.sin(a)
        print(b)

        a = np.arange(0, 10, dtype = np.float32)
        a = a[::2]
        b = np.sin(a)
        print(b)

        a = np.arange(0, 10, dtype = np.int16)
        a = a[::2]
        b = np.sin(a)
        print(b)
        
        print("********")

        a = np.arange(0, 10, dtype = np.float64).reshape((1,2,5))
        a = a[::2]
        b = np.sin(a)
        print(b)

        print("********")

        a = np.array([[0,1,2,3,4],[5,6,7,8,9]])
        a = a[::2]

        x = a>2

        out = np.zeros_like(a, dtype=np.float64)

        b = np.sin(a, where= x, out = out )
        print(b)

    def test_sin_3(self):

        a = np.arange(0, 5, dtype = np.float64)
        b = np.sin(a)
        c = np.sin(a[::-1])

        print(b)
        print(c)

    def test_cos_1(self):

        a = np.arange(0, 10, dtype = np.float64)
        a = a[::2]
        b = np.cos(a)
        print(b)

        a = np.arange(0, 10, dtype = np.float32)
        a = a[::2]
        b = np.cos(a)
        print(b)

        a = np.arange(0, 10, dtype = np.int16)
        a = a[::2]
        b = np.cos(a)
        print(b)
        
        print("********")

        a = np.arange(0, 10, dtype = np.float64).reshape((1,2,5))
        a = a[::2]
        b = np.cos(a)
        print(b)

        print("********")

        a = np.array([[0,1,2,3,4],[5,6,7,8,9]])
        a = a[::2]

        x = a>2

        out = np.zeros_like(a, dtype=np.float64)

        b = np.cos(a, where= x, out = out )
        print(b)

    def test_tan_1(self):

        a = np.arange(0, 10, dtype = np.float64)
        a = a[::2]
        b = np.tan(a)
        print(b)

        a = np.arange(0, 10, dtype = np.float32)
        a = a[::2]
        b = np.tan(a)
        print(b)

        a = np.arange(0, 10, dtype = np.int16)
        a = a[::2]
        b = np.tan(a)
        print(b)
        
        print("********")

        a = np.arange(0, 10, dtype = np.float64).reshape((1,2,5))
        a = a[::2]
        b = np.tan(a)
        print(b)

        print("********")

        a = np.array([[0,1,2,3,4],[5,6,7,8,9]])
        a = a[::2]

        x = a>2

        out = np.zeros_like(a, dtype=np.float64)

        b = np.tan(a, where= x, out = out )
        print(b)

    def test_arcsin_1(self):

        a = np.linspace(-1.0, 1.0, 12)
        print(a)
        b = np.arcsin(a)
        print(b)

       
        print("********")

        a = np.linspace(-1.0, 1.0, 12).reshape((2,2,3))
        a = a[::2]
        b = np.arcsin(a)
        print(b)

        print("********")

        a = np.linspace(-1.0, 1.0, 12)
        a = a[::2]

        x = a > -0.5
        print(x)

        out = np.zeros_like(a, dtype=np.float64)

        b = np.arcsin(a, where= x, out = out )
        print(b)

    def test_arccos_1(self):

        a = np.linspace(-1.0, 1.0, 12)
        print(a)
        b = np.arccos(a)
        print(b)

       
        print("********")

        a = np.linspace(-1.0, 1.0, 12).reshape((2,2,3))
        a = a[::2]
        b = np.arccos(a)
        print(b)

        print("********")

        a = np.linspace(-1.0, 1.0, 12)
        a = a[::2]

        x = a > -0.5
        print(x)

        out = np.zeros_like(a, dtype=np.float64)

        b = np.arccos(a, where= x, out = out )
        print(b)

    def test_arctan_1(self):

        a = np.linspace(-1.0, 1.0, 12)
        print(a)
        b = np.arctan(a)
        print(b)

       
        print("********")

        a = np.linspace(-1.0, 1.0, 12).reshape((2,2,3))
        a = a[::2]
        b = np.arctan(a)
        print(b)

        print("********")

        a = np.linspace(-1.0, 1.0, 12)
        a = a[::2]

        x = a > -0.5
        print(x)

        out = np.zeros_like(a, dtype=np.float64)

        b = np.arctan(a, where= x, out = out )
        print(b)

    def test_hypot_1(self):

        a = np.hypot(np.ones((3, 3)) * 3, np.ones((3, 3)) * 4)
        print(a)

        b = np.hypot(np.ones((3, 3)) * 3, [4])
        print(b)

    def test_arctan2_1(self):

        x = np.array([-1, +1, +1, -1])
        y = np.array([-1, -1, +1, +1])
        z = np.arctan2(y, x) * 180 / np.pi
        print(z)

        a = np.arctan2([1., -1.], [0., 0.])
        print(a)

        b = np.arctan2([0., 0., np.inf], [+0., -0., np.inf])
        print(b)

    def test_degrees_1(self):

        rad = np.arange(12.)*np.pi/6
        a = np.degrees(rad)
        print(a)

        out = np.zeros((rad.shape))
        r = np.degrees(rad, out)
        print(np.all(r == out))

    def test_radians_1(self):

        deg = np.arange(12.0, dtype=np.float64) * 30.0;
        a = np.radians(deg)
        print(a)

        out = np.zeros((deg.shape))
        r = np.radians(deg, out)
        print(np.all(r == out))

    def test_rad2deg_1(self):

        rad = np.arange(12.)*np.pi/6
        a = np.rad2deg(rad)
        print(a)

        out = np.zeros((rad.shape))
        r = np.rad2deg(rad, out)
        print(np.all(r == out))

    def test_deg2rad_1(self):

        deg = np.arange(12.0, dtype=np.float64) * 30.0;
        a = np.deg2rad(deg)
        print(a)

        out = np.zeros((deg.shape))
        r = np.deg2rad(deg, out)
        print(np.all(r == out))

    #endregion

    #region Hyperbolic functions

    def test_sinh_1(self):

        a = np.arange(0, 10, dtype = np.float64)
        a = a[::2]
        b = np.sinh(a)
        print(b)

        a = np.arange(0, 10, dtype = np.float32)
        a = a[::2]
        b = np.sinh(a)
        print(b)

        a = np.arange(0, 10, dtype = np.int16)
        a = a[::2]
        b = np.sinh(a)
        print(b)
        
        print("********")

        a = np.arange(0, 10, dtype = np.float64).reshape((1,2,5))
        a = a[::2]
        b = np.sinh(a)
        print(b)

        print("********")

        a = np.array([[0,1,2,3,4],[5,6,7,8,9]])
        a = a[::2]

        x = a>2

        out = np.zeros_like(a, dtype=np.float64)

        b = np.sinh(a, where= x, out = out )
        print(b)

    def test_cosh_1(self):

        a = np.arange(0, 10, dtype = np.float64)
        a = a[::2]
        b = np.cosh(a)
        print(b)

        a = np.arange(0, 10, dtype = np.float32)
        a = a[::2]
        b = np.cosh(a)
        print(b)

        a = np.arange(0, 10, dtype = np.int16)
        a = a[::2]
        b = np.cosh(a)
        print(b)
        
        print("********")

        a = np.arange(0, 10, dtype = np.float64).reshape((1,2,5))
        a = a[::2]
        b = np.cosh(a)
        print(b)

        print("********")

        a = np.array([[0,1,2,3,4],[5,6,7,8,9]])
        a = a[::2]

        x = a>2

        out = np.zeros_like(a, dtype=np.float64)

        b = np.cosh(a, where= x, out = out )
        print(b)

    def test_tanh_1(self):

        a = np.arange(0, 10, dtype = np.float64)
        a = a[::2]
        b = np.tanh(a)
        print(b)

        a = np.arange(0, 10, dtype = np.float32)
        a = a[::2]
        b = np.tanh(a)
        print(b)

        a = np.arange(0, 10, dtype = np.int16)
        a = a[::2]
        b = np.tanh(a)
        print(b)
        
        print("********")

        a = np.arange(0, 10, dtype = np.float64).reshape((1,2,5))
        a = a[::2]
        b = np.tanh(a)
        print(b)

        print("********")

        a = np.array([[0,1,2,3,4],[5,6,7,8,9]])
        a = a[::2]

        x = a>2

        out = np.zeros_like(a, dtype=np.float64)

        b = np.tanh(a, where= x, out = out )
        print(b)

    def test_arcsinh_1(self):

        a = np.linspace(-1.0, 1.0, 12)
        b = np.arcsinh(a)
        print(b)

       
        print("********")

        a = np.linspace(-1.0, 1.0, 12).reshape((2,2,3))
        a = a[::2]
        b = np.arcsinh(a)
        print(b)

        print("********")

        a = np.linspace(-1.0, 1.0, 12)
        a = a[::2]

        x = a > -0.5
        print(x)

        out = np.zeros_like(a, dtype=np.float64)

        b = np.arcsinh(a, where= x, out = out )
        print(b)

    def test_arccosh_1(self):

        a = np.linspace(1.0, 2.0, 12)
        b = np.arccosh(a)
        print(b)

       
        print("********")

        a = np.linspace(1.0, 2.0, 12).reshape((2,2,3))
        a = a[::2]
        b = np.arccosh(a)
        print(b)

        print("********")

        a = np.linspace(1.0, 2.0, 12)
        a = a[::2]

        x = a > 1.5
        print(x)

        out = np.zeros_like(a, dtype=np.float64)

        b = np.arccosh(a, where= x, out = out )
        print(b)

    def test_arctanh_1(self):

        a = np.linspace(-1.0, 1.0, 12)
        b = np.arctanh(a)
        print(b)

       
        print("********")

        a = np.linspace(-1.0, 1.0, 12).reshape((2,2,3))
        a = a[::2]
        b = np.arctanh(a)
        print(b)

        print("********")

        a = np.linspace(-1.0, 1.0, 12)
        a = a[::2]

        x = a > -0.5
        print(x)

        out = np.zeros_like(a, dtype=np.float64)

        b = np.arctanh(a, where= x, out = out )
        print(b)

        #endregion

    #region Rounding Functions

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

    def test_round_1(self):

        a = np.linspace(-1.0, 1.0, 12).reshape((2,2,3))
        print(a)

        print("********")
        b = np.round_(a, 2)
        print(b)

        print("********")

        c = np.round(a,2)
        print(c)

        print("********")
        b = np.round_(a, 4)
        print(b)

        print("********")

        c = np.round(a,4)
        print(c)

    def test_rint_1(self):

        a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, -4.2])
        b = np.rint(a)
        print(b)

        b = np.rint(a.reshape(2,4))
        print(b)

        x = a > 0.0
        print(x)

        b = np.rint(a, where = x)
        print(b)

    def test_fix_1(self):

        a = np.fix(3.14)
        print(a)

        b = np.fix(3)
        print(b)

        c = np.fix([2.1, 2.9, -2.1, -2.9])
        print(c)

        d = np.fix([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        print(d)

        
    def test_floor_1(self):

        x = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        y = np.floor(x);


        print(x)
        print(y)

        return

    def test_ceil_1(self):

        a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        b = np.ceil(a)
        print(b)


    def test_trunc_1(self):

        a = np.trunc(3.14)
        print(a)

        b = np.trunc(3)
        print(b)

        c = np.trunc([2.1, 2.9, -2.1, -2.9])
        print(c)

        d = np.trunc([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        print(d)

    #endregion

    #region  Sums, products, differences             
    def test_prod_1(self):

        #x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.uint32).reshape(3, 2, -1)
        x = np.array([10,15,25,45,78,90, 10, 15, 25, 45, 78, 90 ], dtype= np.uint64)

        y = np.prod(x);


        print(x)
        print(y)

        return

    def test_prod_2(self):

        a = np.prod([1.,2.])
        print(a)
        print("*****")

        b = np.prod([[1.,2.],[3.,4.]])
        print(b)
        print("*****")

        c = np.prod([[1.,2.],[3.,4.]], axis=1)
        print(c)
        print("*****")

        d = np.array([1, 2, 3], dtype=np.uint8)
        e = np.prod(d).dtype == np.uint
        print(e)
        print("*****")

        f = np.array([1, 2, 3], dtype=np.int8)
        g = np.prod(f).dtype == int
        print(g)
        print("*****")


    def test_prod_3(self):

        a = np.array([1,2,3])
        b = np.prod(a)   # intermediate results 1, 1*2
                            # total product 1*2*3 = 6
        print(b)
        print("*****")

        a = np.array([[1, 2, 3], [4, 5, 6]])
        c = np.prod(a, dtype=float)  # specify type of output
        print(c)
        print("*****")

        d = np.prod(a, axis=0)
        print(d)
        print("*****")
  
        e = np.prod(a,axis=1)
        print(e)
        print("*****")
      
    def test_sum_1(self):

        x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.uint32).reshape(3, 2, -1)
        x = x * 3
        y = np.sum(x);


        print(x)
        print(y)

        return
       
    def test_sum_2(self):

        x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.uint32).reshape(3, 2, -1)
        x = x * 3

        y = np.sum(x, axis=0);
        print(y)
        print("*****")

        y = np.sum(x, axis=1);
        print(y)
        print("*****")

        y = np.sum(x, axis=2);
        print(y)
        print("*****")

        return

    def test_sum_3(self):

        x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.float64).reshape(3, 2, -1)
        x = x * 3.456

        y = np.sum(x, axis=0);
        print(y)
        print("*****")

        y = np.sum(x, axis=1);
        print(y)
        print("*****")

        y = np.sum(x, axis=2);
        print(y)
        print("*****")

        return

    def test_sum_keepdims(self):

        x = np.array([10,15,25,45,78,90], dtype= np.float64)

        y = np.sum(x);
        print(y)
        print(y.shape)
        print("*****")

        print("keepdims")
        y = np.sum(x, keepdims = True);
        print(y)
        print(y.shape)
        print("*****")

        x = np.array([10,15,25,45,78,90], dtype= np.float64).reshape(3, 2, -1)
        y = np.sum(x, axis=1);
        print(y)
        print(y.shape)
        print("*****")

        print("keepdims")
        y = np.sum(x, axis=1, keepdims = True);
        print(y)
        print(y.shape)
        print("*****")

        x = np.array([10,15,25,45,78,90], dtype= np.float64).reshape(-1, 3, 2)
        y = np.sum(x, axis=2);
        print(y)
        print(y.shape)
        print("*****")

        print("keepdims")
        y = np.sum(x, axis=2, keepdims = True);
        print(y)
        print(y.shape)
        print("*****")

        return


    def test_cumprod_1(self):

        x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.uint32).reshape(3, 2, -1)
        x = x * 3
        y = np.cumprod(x);
        print(y)

        x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.int32).reshape(3, 2, -1)
        x = x * 3
        y = np.cumprod(x);
        print(y)

        return

    def test_cumprod_1a(self):

        x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.uint64).reshape(3, 2, -1)
        x = x * 1
        y = np.cumprod(x);
        print(y)

        x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.int64).reshape(3, 2, -1)
        x = x * 1
        y = np.cumprod(x);
        print(y)

        return

    def test_cumprod_2(self):

        a = np.array([1,2,3])
        b = np.cumprod(a)   # intermediate results 1, 1*2
                            # total product 1*2*3 = 6
        print(b)
        print("*****")

        a = np.array([[1, 2, 3], [4, 5, 6]])
        c = np.cumprod(a, dtype=float)  # specify type of output
        print(c)
        print("*****")

        d = np.cumprod(a, axis=0)
        print(d)
        print("*****")
  
        e = np.cumprod(a,axis=1)
        print(e)
        print("*****")

    def test_cumsum_1(self):

        x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.uint32).reshape(3, 2, -1)
        x = x * 3
        y = np.cumsum(x);


        print(x)
        print(y)

        return
    
    def test_cumsum_2(self):

        a = np.array([[1,2,3], [4,5,6]])
        print(a)
        print("*****")

        b = np.cumsum(a)
        print(b)
        print("*****")

        c = np.cumsum(a, dtype=float)     # specifies type of output value(s)
        print(c)
        print("*****")

        d = np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
        print(d)
        print("*****")
 
        e = np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
        print(e)

        return

    def test_cumsum_3(self):

        a = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]]).reshape(2,3,-1)
        print(a)
        print("*****")

        b = np.cumsum(a)
        print(b)
        print("*****")

        c = np.cumsum(a, dtype=float)     # specifies type of output value(s)
        print(c)
        print("*****")

        d = np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
        print(d)
        print("*****")
 
        e = np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
        print(e)
        print("*****")

        f = np.cumsum(a,axis=2)      # sum over columns for each of the 2 rows
        print(f)
        print("*****")

        #g = np.cumsum(a,axis=3)      # sum over columns for each of the 2 rows
        #print(g)

    def test_cumsum_4(self):

        a = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]]).reshape(3,2,-1)
        print(a)
        print("*****")

        b = np.cumsum(a)
        print(b)
        print("*****")

        c = np.cumsum(a, dtype=float)     # specifies type of output value(s)
        print(c)
        print("*****")

        d = np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
        print(d)
        print("*****")
 
        e = np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
        print(e)

        f = np.cumsum(a,axis=2)      # sum over columns for each of the 2 rows
        print(f)

        #g = np.cumsum(a,axis=3)      # sum over columns for each of the 2 rows
        #print(g)

        
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

    

    def test_ediff1d_1(self):
        x = np.array([1, 2, 4, 7, 0])
        y = np.ediff1d(x)
        print(y)

        print(np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99])))

        y = [[1, 2, 4], [1, 6, 24]]
        print(np.ediff1d(y))


    def test_gradient_1(self):
        f = np.array([1, 2, 4, 7, 11, 16], dtype=float)
        a = nptest.gradient(f)
        print(a)
        print("***********")

        b = nptest.gradient(f, 2)
        print(b)
        print("***********")

        #Spacing can be also specified with an array that represents the coordinates
        #of the values F along the dimensions.
        #For instance a uniform spacing:

        x = np.arange(f.size)
        c = nptest.gradient(f, x)
        print(c)
        print("***********")

        #Or a non uniform one:

        x = np.array([0., 1., 1.5, 3.5, 4., 6.], dtype=float)
        d = nptest.gradient(f, x)
        print(d)

    def test_gradient_STRING_1(self):
        f = np.array(["1", "2", "4", "7", "11", "16"])
        a = nptest.gradient(f)
        print(a)
  
   

    def test_gradient_2(self):

        #For two dimensional arrays, the return will be two arrays ordered by
        #axis. In this example the first array stands for the gradient in
        #rows and the second one in columns direction:

        a = nptest.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=float))
        print(a)
        print("***********")
 
        #In this example the spacing is also specified:
        #uniform for axis=0 and non uniform for axis=1

        dx = 2.
        y = [1., 1.5, 3.5]
        b = nptest.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=float), dx, y)
        print(b)
        print("***********")
 
        #It is possible to specify how boundaries are treated using `edge_order`

        x = np.array([0, 1, 2, 3, 4])
        f = x**2
        c = nptest.gradient(f, edge_order=1)
        print(c)
        print("***********")

        d = nptest.gradient(f, edge_order=2)
        print(d)
        print("***********")

        #The `axis` keyword can be used to specify a subset of axes of which the
        #gradient is calculated

        e = nptest.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=float), axis=0)
        print(e)
 


    def test_trapz_1(self):

        a = np.trapz([1,2,3])
        print(a)

        b = np.trapz([1,2,3], x=[4,6,8])
        print(b)

        c = np.trapz([1,2,3], dx=2)
        print(c)

        a = np.arange(6).reshape(2, 3)
        b = np.trapz(a, axis=0)
        print(b)

        c = np.trapz(a, axis=1)
        print(c)


    #endregion       
    #region Exponents and logarithms

    def test_exp_1(self):

        x = np.array([1e-10, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, -4.2])
        a = np.exp(x)
        print(a)

        a = np.exp(x.reshape(2,-1))
        print(a)

        b = x > 0
        a = np.exp(x, where= b)
        print(a)
        return

    def test_expm1_1(self):

        x = np.array([1e-10, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, -4.2])
        a = np.expm1(x)
        print(a)

        a = np.expm1(x.reshape(2,-1))
        print(a)

        b = x > 0
        a = np.expm1(x, where= b)
        print(a)
        return

    def test_exp2_1(self):

        x = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, -4.2])
        a = np.exp2(x)
        print(a)

        a = np.exp2(x.reshape(2,-1))
        print(a)

        b = x > 0
        a = np.exp2(x, where= b)
        print(a)
        return


    def test_log_1(self):

        x = np.array([1, np.e, np.e**2, 0])
        a = np.log(x)
        print(a)

        a = np.log(x.reshape(2,-1))
        print(a)

        b = x > 0
        a = np.log(x, where= b)
        print(a)
        return

    def test_log10_1(self):

        x = np.array([1, np.e, np.e**2, 0])
        a = np.log10(x)
        print(a)

        a = np.log10(x.reshape(2,-1))
        print(a)

        b = x > 0
        a = np.log10(x, where= b)
        print(a)
        return

    def test_log2_1(self):

        x = np.array([1, np.e, np.e**2, 0])
        a = np.log2(x)
        print(a)

        a = np.log2(x.reshape(2,-1))
        print(a)

        b = x > 0
        a = np.log2(x, where= b)
        print(a)
        return

    def test_log1p_1(self):

        x = np.array([1, np.e, np.e**2, 0])
        a = np.log1p(x)
        print(a)

        a = np.log1p(x.reshape(2,-1))
        print(a)

        b = x > 0
        a = np.log1p(x, where= b)
        print(a)
        return

    def test_logaddexp_1(self):

        prob1 = np.log(1e-50)
        prob2 = np.log(2.5e-50)
        a = np.logaddexp(prob1, prob2)
        print(a)
        b = np.exp(a)
        print(b)

    def test_logaddexp2_1(self):

        prob1 = np.log2(1e-50)
        prob2 = np.log2(2.5e-50)
        a = np.logaddexp2(prob1, prob2)
        print(a)
        b = 2 ** a
        print(b)

    #endregion    

    #region Other special Functions

    def test_i0_1(self):

        a = np.i0(5)
        print(a)

        a = np.i0(5.0)
        print(a)

        a = np.i0([5.0, 6.0])
        print(a)

        a = np.i0([[5.0, 6.0],[7.9, 8.0]])
        print(a)

        return;

    def test_sinc_1(self):

        x = np.linspace(-4, 4, 10)
        a = np.sinc(x)
        print(a)

        print("********")

        xx = np.outer(x, x)
        b = np.sinc(xx)
        print(b)


    #endregion

    #region Floating point routines

    def test_signbit_1(self):

        a =np.signbit(-1.2)
        print(a)
        
        b = np.signbit(np.array([1, -2.3, 2.1]))
        print(b)

        c = np.signbit(np.array([+0.0, -0.0]))
        print(c)

        d = np.signbit(np.array([-np.inf, np.inf]))
        print(d)

        e = np.signbit(np.array([-np.nan, np.nan]))
        print(e)


        f = np.signbit(np.array([-1, 0, 1]))
        print(f)

    def test_copysign_1(self):

        a = np.copysign(1.3, -1)
        print(a)

        b = 1/np.copysign(0, 1)
        print(b)

        c = 1/np.copysign(0, -1)
        print(c)


        d = np.copysign([-1, 0, 1], -1.1)
        print(d)

        e  = np.copysign([-1, 0, 1], np.arange(3)-1)
        print(e)

    def test_frexp_1(self):

        x = np.arange(9)
        y1, y2 = np.frexp(x)
        print(y1)
        print(y2)

        print("***************")

        x = np.arange(9, dtype = np.float32).reshape(3,3)
        y1, y2 = np.frexp(x)
        print(y1)
        print(y2)

        print("***************")

        x = np.arange(9, dtype = np.float64).reshape(3,3)
        y1, y2 = np.frexp(x, where = x < 5)
        print(y1)
        print(y2)

        
    def test_ldexp_1(self):

        a = np.ldexp(5, np.arange(4))
        print(a)

        b = np.ldexp(np.arange(4), 5);
        print(b)

    def test_nextafter_1(self):

        a = np.nextafter(1, 2)
        print(a)

        b = np.nextafter([1, 2], [2, 1])
        d1 = b[0]
        d2 = b[1]
        print(d1)
        print(d2)

        c1 = np.array([1, 2], dtype=np.float32)
        c2 = np.array([2, 1], dtype=np.float32)

        c = np.nextafter(c1,c2)
        f1 = c[0]
        f2 = c[1]
        print(f1)
        print(f2)


    #endregion

    #region Rational routines

    def test_lcm_1(self):

        a = np.lcm(12, 20)
        print(a)

        b = np.lcm.reduce([3, 12, 20])
        print(b)

        c = np.lcm.reduce([40, 12, 20])
        print(c)
 
        d = np.lcm(np.arange(6), [20])
        print(d)

        e = np.lcm([20, 21], np.arange(6).reshape(3, 2))
        print(e)

        #f = np.lcm(np.arange(8).reshape(2,4), np.arange(16).reshape(4, 4))
        #print(f)

    def test_gcd_1(self):

        a = np.gcd(12, 20)
        print(a)

        b = np.gcd.reduce([3, 12, 20])
        print(b)

        c = np.gcd.reduce([40, 12, 20])
        print(c)
 
        d = np.gcd(np.arange(6), [20])
        print(d)

        e = np.gcd([20, 20], np.arange(6).reshape(3, 2))
        print(e)

        #f = np.lcm(np.arange(8).reshape(2,4), np.arange(16).reshape(4, 4))
        #print(f)

    #endregion

    #region Arithmetic operations

    def test_add_1(self):        

        a = np.add(1.0, 4.0)
        print(a)

        b = np.arange(9.0).reshape((3, 3))
        c = np.arange(3.0)
        d = np.add(b, c)
        print(d)

    def test_reciprocal_operations(self):

        a =  np.arange(1, 32, 1, dtype = np.float32)
        print(a)

        b = np.reciprocal(a)
        print(b)
        
        a = np.arange(2048, 2048+32, 1, dtype = np.float64)
        print(a)

        b = np.reciprocal(a)
        print(b)

    def test_positive_1(self):
 
        d = np.positive([-1, -0, 1])
        print(d)

        e  = np.positive([[1, 0, -1], [-2, 3, -4]])
        print(e)

        
    def test_negative_1(self):
 
        d = np.negative([-1, -0, 1])
        print(d)

        e  = np.negative([[1, 0, -1], [-2, 3, -4]])
        print(e)

    def test_multiply_1(self):        

        a = np.multiply(1.0, 4.0)
        print(a)

        b = np.arange(9.0).reshape((3, 3))
        c = np.arange(3.0)
        d = np.multiply(b, c)
        print(d)


    def test_divide(self):

        a = np.divide(7,3)
        print(a)

        b = np.divide([1., 2., 3., 4.], 2.5)
        print(b)

        c = np.divide([1., 2., 3., 4.], [0.5, 2.5, 2.5, 3.5 ])
        print(c)

        return

    
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

        
    def test_subtract_1(self):        

        a = np.subtract(2.0, 4.0)
        print(a)

        b = np.arange(9.0).reshape((3, 3))
        c = np.arange(3.0)
        d = np.subtract(b, c)
        print(d)

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

    def test_float_power(self):

        x1 = range(6)

        a = np.float_power(x1, 3)
        print(a)

        x2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
        b = np.float_power(x1, x2)
        print(b)

        x3 = np.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])

        c = np.float_power(x1, x3)
        print(c)

        return

    def test_fmod_1(self):

        x = np.fmod([4, 7], [2, 3])
        print(x)

        y = np.fmod(np.arange(7), 5)
        print(y)

        return

    def test_fmod_2(self):

        x = np.fmod([-4, -7], [2, 3])
        print(x)

        y = np.fmod(np.arange(7), -5)
        print(y)

        return   

    def test_mod_1(self):

        x = np.mod([4, 7], [2, 3])
        print(x)

        y = np.mod(np.arange(7), 5)
        print(y)

        return

    def test_modf_1(self):

        x = np.modf([0, 3.5])
        print(x)

        y = np.modf(np.arange(7))
        print(y)

        return

    def test_remainder_1(self):

        x = np.remainder([4, 7], [2, 3])
        print(x)

        y = np.remainder(np.arange(7), 5)
        print(y)

        return

    def test_remainder_2(self):

        x = np.remainder([-4, -7], [2, 3])
        print(x)

        y = np.remainder(np.arange(7), -5)
        print(y)

        return   

    def test_divmod_1(self):

        a = np.divmod(7,3)
        print(a)

        b = np.divmod([1., 2., 3., 4.], 2.5)
        print(b)

        c = np.divmod([1., 2., 3., 4.], [0.5, 2.5, 2.5, 3.5 ])
        print(c)

        return
    

    #endregion

    #region Miscellaneous

              
    def test_convolve_1(self):    

        a = np.convolve([1, 2, 3], [0, 1, 0.5])
        print(a)

        # Only return the middle values of the convolution. Contains boundary effects, where zeros are taken into account:

        b = np.convolve([1,2,3],[0,1,0.5], 'same')
        print(b)

        # The two arrays are of the same length, so there is only one position where they completely overlap:

        c = np.convolve([1,2,3],[0,1,0.5], 'valid')
        print(c)

        return

    def test_clip_1(self):

        a = np.arange(10)
        print(a)
        print("*****")

        b = np.clip(a, 1, 8)
        print(b)
        print("*****")

        c = np.clip(a, 3, 6, out=a)
        print(c)
        print(a)
        print("*****")

        a = np.arange(10)
        print(a)
        b = np.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
        print(b)
        print("*****")

    def test_clip_2(self):

        a = np.arange(16).reshape(4,4)
        print(a)
        print("*****")

        b = np.clip(a, 1, 8)
        print(b)
        print("*****")

        c = np.clip(a, 3, 6, out=a)
        print(c)
        print(a)
        print("*****")

        a = np.arange(16).reshape(4,4)
        print(a)
        b = np.clip(a, [3, 4, 1, 1], 8)
        print(b)
        print("*****")

         
    def test_sqrt_operations(self):
        a =  np.arange(0, 32, 1, dtype = np.int16)
        print(a)

        b = np.sqrt(a)
        print(b)
        
        a = np.arange(2048, 2048+32, 1, dtype = np.int64)
        print(a)

        b = np.sqrt(a)
        print(b)

                
    def test_cbrt_operations(self):
        a =  np.arange(0, 32, 1, dtype = np.int16)
        print(a)

        b = np.cbrt(a)
        print(b)
        
        a = np.arange(2048, 2048+32, 1, dtype = np.int64)
        print(a)

        b = np.cbrt(a)
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

    def test_absolute_operations(self):
        a =  np.arange(-32, 32, 1, dtype = np.int16)
        print(a)

        b = np.absolute(a)
        print(b)

    def test_fabs_1(self):
        a =  np.arange(-32, 32, 1, dtype = np.int16)
        print(a)

        b = np.fabs(a)
        print(b)

    def test_sign_1(self):

        a =np.sign(-1.2)
        print(a)
        
        b = np.sign(np.array([1, -2.3, 2.1]))
        print(b)

        c = np.sign(np.array([+0.0, -0.0]))
        print(c)

        d = np.sign(np.array([-np.inf, np.inf]))
        print(d)

        e = np.sign(np.array([-np.nan, np.nan]))
        print(e)


        f = np.sign(np.array([-1, 0, 1]))
        print(f)

    def test_heaviside_1(self):

        a = np.heaviside([-1.5, 0, 2.0], 0.5)
        print(a)

        b = np.heaviside([-1.5, 0, 2.0], 1)
        print(b)

        c = np.heaviside([-1, 0, 2], 1)
        print(c)


    def test_maximum_1(self):

        a = np.maximum([2, 3, 4], [1, 5, 2])
        print(a)

        b = np.maximum(np.eye(2), [0.5, 2]) # broadcasting
        print(b)

        c = np.maximum([np.nan, 0, np.nan], [0, np.nan, np.nan])
        print(c)

        d = np.maximum(np.Inf, 1)
        print(d)


    def test_minimum_1(self):

        a = np.minimum([2, 3, 4], [1, 5, 2])
        print(a)

        b = np.minimum(np.eye(2), [0.5, 2]) # broadcasting
        print(b)

        c = np.minimum([np.nan, 0, np.nan], [0, np.nan, np.nan])
        print(c)

        d = np.minimum(np.Inf, 1)
        print(d)


    def test_fmax_1(self):

        a = np.fmax([2, 3, 4], [1, 5, 2])
        print(a)

        b = np.fmax(np.eye(2), [0.5, 2]) # broadcasting
        print(b)

        c = np.fmax([np.nan, 0, np.nan], [0, np.nan, np.nan])
        print(c)

        d = np.fmax(np.Inf, 1)
        print(d)

    def test_fmin_1(self):

        a = np.fmin([2, 3, 4], [1, 5, 2])
        print(a)

        b = np.fmin(np.eye(2), [0.5, 2]) # broadcasting
        print(b)

        c = np.fmin([np.nan, 0, np.nan], [0, np.nan, np.nan])
        print(c)

        d = np.fmin(np.Inf, 1)
        print(d)


    def test_nan_to_num_1(self):

        a = np.nan_to_num(np.inf)
        print(a)

        b = np.nan_to_num(-np.inf)
        print(b)

        c = np.nan_to_num(np.nan)
        print(c)

        x = np.array([np.inf, -np.inf, np.nan, -128, 128])
        d = np.nan_to_num(x)
        print(d)

        #e = np.nan_to_num(x, nan=-9999, posinf=33333333, neginf=33333333)
        #print(e)

        y = np.array([complex(np.inf, np.nan), np.nan, complex(np.nan, np.inf)])
        print(y)

        f = np.nan_to_num(y)
        print(f)

        #g = np.nan_to_num(y, nan=111111, posinf=222222)
        #print(g)

    def test_interp_1(self):

        xp = xp = [1, 2, 3]
        fp = [3, 2, 0]
        a = np.interp(2.5, xp, fp)
        print(a)

        b = np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
        print(b)

        UNDEF = -99.0
        c = np.interp(3.14, xp, fp, right=UNDEF)
        print(c)

        d = np.interp([3.14, -1], xp, fp, left=UNDEF, right=UNDEF)
        print(d)

    def test_interp_NAN_1(self):

        xp = xp = [1, 2, 3]
        fp = [3, 2, 0]
        a = np.interp(np.NaN, xp, fp)
        print(a)

        b = np.interp([np.NaN, 1, 1.5, np.NaN, 3.14], xp, fp)
        print(b)

        UNDEF = np.NaN
        c = np.interp(3.14, xp, fp, right=UNDEF)
        print(c)

        d = np.interp([3.14, -1], xp, fp, left=UNDEF, right=UNDEF)
        print(d)

    def test_interp_NAN_2(self):

        xp = xp = [np.NaN, 2, 3]
        fp = [3, 2, np.NaN]
        a = np.interp(2.5, xp, fp)
        print(a)

        b = np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
        print(b)

        UNDEF = -99.0
        c = np.interp(3.14, xp, fp, right=UNDEF)
        print(c)

        d = np.interp([3.14, -1], xp, fp, left=UNDEF, right=UNDEF)
        print(d)

    def test_interp_INF_1(self):

        xp = xp = [1, 2, 3]
        fp = [3, 2, 0]
        a = np.interp(np.Inf, xp, fp)
        print(a)

        b = np.interp([np.Inf, 1, 1.5, np.Inf, 3.14], xp, fp)
        print(b)

        UNDEF = np.Inf
        c = np.interp(3.14, xp, fp, right=UNDEF)
        print(c)

        d = np.interp([3.14, -1], xp, fp, left=UNDEF, right=UNDEF)
        print(d)

    def test_interp_INF_1a(self):

        xp = xp = [1, 2, 3]
        fp = [3, 2, 0]
        a = np.interp(-np.Inf, xp, fp)
        print(a)

        b = np.interp([-np.Inf, 1, 1.5, -np.Inf, 3.14], xp, fp)
        print(b)

        UNDEF = -np.Inf
        c = np.interp(3.14, xp, fp, right=UNDEF)
        print(c)

        d = np.interp([3.14, -1], xp, fp, left=UNDEF, right=UNDEF)
        print(d)

    def test_interp_2(self):

        x = [-180, -170, -185, 185, -10, -5, 0, 365]
        xp = [190, -190, 350, -350]
        fp = [5, 10, 3, 4]
        a = np.interp(x, xp, fp, period=360)
        print(a)


    def test_interp_COMPLEX_1(self):

        xp = xp = [1, 2, 3]
        fp = [1.0j, 0, 2+3j]
        a = np.interp(2.5, xp, fp)
        print(a)

        b = np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
        print(b)

        UNDEF = -99+88j
        c = np.interp(3.14, xp, fp, right=UNDEF)
        print(c)

        d = np.interp([3.14, -1], xp, fp, left=UNDEF, right=UNDEF)
        print(d)

        UNDEF = 66
        c = np.interp(3.14, xp, fp, right=UNDEF)
        print(c)

        d = np.interp([3.14, -1], xp, fp, left=UNDEF, right=UNDEF)
        print(d)


    def test_interp_COMPLEX_2(self):

        x = [-180, -170, -185, 185, -10, -5, 0, 365]
        xp = [190, -190, 350, -350]
        fp = [5+0j, 10+0j, 3+0j, 4+0j]
        a = np.interp(x, xp, fp, period=360)
        print(a)

    #endregion

if __name__ == '__main__':
    unittest.main()
