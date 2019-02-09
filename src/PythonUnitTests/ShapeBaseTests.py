import unittest
import numpy as np
from nptest import nptest


class Test_ShapeBaseTests(unittest.TestCase):
    def test_atleast_1d(self):

        a = np.atleast_1d(1.0)
        print(a)

        print("**************")
        x = np.arange(9.0).reshape(3,3)
        b = np.atleast_1d(x)
        print(b)

        print("**************")

        c = np.atleast_1d(1, [3,4])
        print(c)


    def test_atleast_2d(self):

        a = np.atleast_2d(1.0)
        print(a)

        print("**************")
        x = np.arange(9.0).reshape(3,3)
        b = np.atleast_2d(x)
        print(b)

        print("**************")

        c = np.atleast_2d(1, [3,4], [5.6])
        print(c)

    def test_atleast_3d(self):

        a = np.atleast_3d(1.0)
        print(a)

        print("**************")
        x = np.arange(9.0).reshape(3,3)
        b = np.atleast_3d(x)
        print(b)

        print("**************")

        c = np.atleast_3d([1,2], [[3,4]], [[5,6]])
        print(c)

        for arr in c:
            print(arr, arr.shape)

    def test_vstack_1(self):

        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        c = np.vstack((a,b))

        print(c)

    def test_vstack_2(self):

        a = np.array([[1], [2], [3]])
        b = np.array([[2], [3], [4]])
        c  = np.vstack((a,b))

        print(c)

    def test_hstack_1(self):

        a = np.array((1, 2, 3))
        b = np.array((2, 3, 4))
        c = np.hstack((a,b))

        print(c)

    def test_hstack_2(self):

        a = np.array([[1],[2],[3]])
        b = np.array([[2],[3],[4]])
        c = np.hstack((a,b))

        print(c)


if __name__ == '__main__':
    unittest.main()
