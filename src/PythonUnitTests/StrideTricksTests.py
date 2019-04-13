import unittest
import numpy as np
import matplotlib.pyplot as plt
from nptest import nptest
import operator

class Test_StrideTricksTests(unittest.TestCase):

    def test_broadcast_1(self):

        x = np.array([[11], [2], [3]])
        y = np.array([4, 5, 6])
        b = np.broadcast(x, y)
        print(b.shape)

        print(b.index)
        for u,v in b:
            print(u, v)
        print(b.index)

        #c = np.empty(b.shape)
        #c.flat = [u+v for (u,v) in b]
        #print(c)

    def test_broadcast_2(self):

        x = np.array([[11], [2], [3]])
        y = np.array([4, 5, 6, 7, 8, 9])
        b = np.broadcast(x, y)
        print(b.shape)
        print(b.size)

        print(b.index)
        for u,v in b:
            print(u, v)
        print(b.index)

        #c = np.empty(b.shape)
        #c.flat = [u+v for (u,v) in b]
        #print(c)

    def test_broadcast_3(self):

        x = np.array([[11], [2], [3]])
        y = np.array([4, 5, 6, 7, 8, 9])
        z = np.array([[21], [22], [23]])
        b = np.broadcast(x, y, z)
        print(b.numiter)
        print(b.shape)
        print(b.size)

        print(b.index)
        for u,v,w in b:
            if True:
                print(u, v, w)
        print(b.index)

        #c = np.empty(b.shape)
        #c.flat = [u+v for (u,v) in b]
        #print(c)


    def test_broadcast_to_1(self):

        a = np.broadcast_to(5, (4,4))
        print(a)
        print(a.shape)
        print(a.strides)
        print("*************")
 
        b = np.broadcast_to([1, 2, 3], (3, 3))
        print(b)
        print(b.shape)
        print(b.strides)
        print("*************")

        #for aa in np.nditer(b):
        #    print(aa)


    def test_broadcast_to_2(self):

        #a = np.array([1,2,3,1,2,3,1,2,3,1,2,3]).reshape(4,3)
        #print(a)
        #print(a.shape)
        #print(a.strides)
        #print("*************")

        x = np.array([[1, 2, 3]])
        #print(x)
        #print(x.shape)
        #print(x.strides)
        #print("*************")

        b = np.broadcast_to(x, (4, 3))
        print(b)
        print(b.shape)
        print(b.strides)

    def test_broadcast_to_3(self):

        a = np.array([1,2,3,1,2,3,1,2,3,1,2,3]).reshape(2,2,3)
   
        b = np.broadcast_to(a, (4, 2, 3))
        print(b)
        print(b.shape)
        print(b.strides)

    def test_broadcast_arrays_1(self):

        x = np.array([[1,2,3]])
        y = np.array([[4],[5]])
        z = np.broadcast_arrays(x, y)

        print(z)

        print(np.array(a) for a in np.broadcast_arrays(x, y))

    def test_as_strided_1(self):

        y = np.zeros((10, 10))
        print(y.strides)

        n = 1000
        a = np.arange(n)

        b = np.lib.stride_tricks.as_strided(a, (n, n), (0, 8))

        print(b)

        print(b.size)
        print(b.shape)
        print(b.nbytes)




if __name__ == '__main__':
    unittest.main()
