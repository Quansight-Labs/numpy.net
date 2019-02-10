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

    def test_stack_1(self):

        a = np.array([[1],[2],[3]])
        b = np.array([[2],[3],[4]])

        c = np.stack((a,b), axis=0)
        print(c)
        print("**************")
        
        d = np.stack((a,b), axis=1)
        print(d)
        print("**************")

        e = np.stack((a,b), axis=2)
        print(e)

    def test_block_1(self):

        A = np.eye(2) * 2
        B = np.eye(3) * 3
        C = np.block([[A, np.zeros((2, 3))], [np.ones((3, 2)), B]])
        print(C)

    def test_block_2(self):

        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        c = np.block([a, b, 10])             # hstack([a, b, 10])
        print(c)

        print("**************")

        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        c = np.block([[a], [b]])             # vstack([a, b])
        print(c)

    def test_expand_dims_1(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(2,-1, 2)
        b = np.expand_dims(a, axis=0)
        print(b)
        print("**************")

        c = np.expand_dims(a, axis=1)
        print(c)
        print("**************")

        d = np.expand_dims(a, axis=2)
        print(d)

    def test_column_stack_1(self):

        a = np.array((1, 2, 3))
        b = np.array((2, 3, 4))
        c = np.column_stack((a, b))
        print(c)

        print("**************")

        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        c = np.column_stack([a, b])
        print(c)
 
    def test_row_stack_1(self):

        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        c = np.row_stack((a,b))

        print(c)

    def test_dstack_1(self):

        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        c = np.dstack((a,b))

        print(c)
        print("**************")

        a = np.array([[1], [2], [3]])
        b = np.array([[2], [3], [4]])
        c = np.dstack((a,b))
        print(c)

    def test_array_split_1(self):

        x = np.arange(8.0)
        y = np.array_split(x, 3)
        print(y)

        print("**************")

        x = np.arange(7.0)
        y = np.array_split(x, 3)
        print(y)

    def test_array_split_2(self):

        x = np.arange(16.0).reshape(2,8,1)
        y = np.array_split(x, 3, axis=0)
        print(y)

        print("**************")

        x = np.arange(16.0).reshape(2,8,1)
        y = np.array_split(x, 3, axis=1)
        print(y)

        print("**************")

        x = np.arange(16.0).reshape(2,8,1)
        y = np.array_split(x, 3, axis=2)
        print(y)

    def test_split_1(self):

        x = np.arange(9.0)
        y = np.split(x, 3)
        print(y)

        print("**************")

        x = np.arange(8.0)
        y = np.split(x, [3,5,6,10])
        print(y)

    def test_split_2(self):

        x = np.arange(16.0).reshape(8,2,1)
        y = np.split(x, [2,3], axis=0)
        print(y)

        print("**************")

        x = np.arange(16.0).reshape(8,2,1)
        y = np.split(x, [2,3], axis=1)
        print(y)

        print("**************")

        x = np.arange(16.0).reshape(8,2,1)
        y = np.split(x, [2,3], axis=2)
        print(y)

    def test_hsplit_1(self):

        x = np.arange(16).reshape(4,4)
        y = np.hsplit(x, 2)
        print(y)

        print("**************")

        x = np.arange(16).reshape(4,4)
        y = np.hsplit(x, [3,6])
        print(y)

    def test_hsplit_2(self):

        x = np.arange(8).reshape(2,2,2)
        y = np.hsplit(x, 2)
        print(y)

        print("**************")

        x = np.arange(8).reshape(2,2,2)
        y = np.hsplit(x, [3,6])
        print(y)
   
    def test_vsplit_1(self):

        x = np.arange(16).reshape(4,4)
        y = np.vsplit(x, 2)
        print(y)

        print("**************")

        x = np.arange(16).reshape(4,4)
        y = np.vsplit(x, [3,6])
        print(y)

    def test_vsplit_2(self):

        x = np.arange(8).reshape(2,2,2)
        y = np.vsplit(x, 2)
        print(y)

        print("**************")

        x = np.arange(8).reshape(2,2,2)
        y = np.vsplit(x, [3,6])
        print(y)

if __name__ == '__main__':
    unittest.main()
