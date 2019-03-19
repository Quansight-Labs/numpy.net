import unittest
import numpy as np
from nptest import nptest

class MathematicalFunctionsTests(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()
