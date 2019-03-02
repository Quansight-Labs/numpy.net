import unittest
import numpy as np
import matplotlib.pyplot as plt
from nptest import nptest
import operator

class Test_StrideTricksTests(unittest.TestCase):

    def test_broadcast_1(self):

        x = np.array([[1], [2], [3]])
        y = np.array([4, 5, 6])
        b = np.broadcast(x, y)
        print(b)

        c = np.empty(b.shape)
        c.flat = [u+v for (u,v) in b]
        print(c)

    def test_broadcast_to_1(self):

        x = np.array([1, 2, 3])
        b = np.broadcast_to(x, (3, 3))
        print(b)

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
