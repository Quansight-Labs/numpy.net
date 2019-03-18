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

    def test_dsplit_1(self):

        x = np.arange(16).reshape(2,2,4)
        y = np.dsplit(x, 2)
        print(y)

        print("**************")

        x = np.arange(16).reshape(2,2,4)
        y = np.dsplit(x, [3,6])
        print(y)

    def test_kron_1(self):

        a =  np.kron([1,10,100], [5,6,7])
        print(a)

        b = np.kron([5,6,7], [1,10,100])
        print(b)

        c = np.kron(np.eye(2, dtype=np.int32), np.ones((2,2), dtype=np.int32))
        print(c)

        d = np.kron(np.ones((5,7,9, 11), dtype=np.int32), np.ones((3,4, 6, 8), dtype=np.int32))
        print(d.shape)

    def test_kron_2(self):
        a = np.arange(100).reshape((2,5,2,5))
        b = np.arange(24).reshape((2,3,4))
        c = np.kron(a,b)
        print(c.shape)

        d = c.sum()
        print(d)

    def test_tile_1(self):

        a = np.array([0, 1, 2])
  
        b = np.tile(a, 2)
        print(b)
        print("**************")

        c = np.tile(a, (2,2))
        print(c)
        print("**************")

        d = np.tile(a, (2,1,2))
        print(d)

        e = np.arange(100).reshape((2,5,2,5))
        f = np.tile(e, (2,1,2))
        print(f.shape)

    def test_tile_2(self):

        a = np.array([[1, 2], [3, 4]])
        b = np.tile(a, 2)
        print(b)
        print("**************")
   
        c = np.tile(a, (2, 1))
        print(c)
        print("**************")
        d = np.array([1,2,3,4])
        e = np.tile(d,(4,1))
        print(e)

    def test_apply_along_axis_1(self):

        def my_func(a):
            #Average first and last element of a 1-D array"""
            return (a[0] + a[-1]) * 0.5

        def my_func2(a):
            #Average first and last element of a 1-D array"""
            return (a[0]  * 10)

        b = np.array([[1,2,3], [4,5,6], [7,8,9]])
        c = np.apply_along_axis(my_func2, 0, b)
        print(c)

        d = np.apply_along_axis(my_func, 1, b);
        print(d)
        print(b)

    def test_apply_along_axis_2(self):

        b = np.array([[[8,1,7], [4,3,9], [5,2,6]]])
        c = np.apply_along_axis(sorted, 1, b)
        print(c)

        c = np.apply_along_axis(sorted, 0, b[:,0,0])
        print(c)
        
        c = np.apply_along_axis(sorted, 0, b[0,:,0])
        print(c)

        c = np.apply_along_axis(sorted, 0, b[0,0,:])
        print(c)

    def test_apply_along_axis_3(self):

        b = np.array([[1,2,3], [4,5,6], [7,8,9]])

        c = np.diag(b)

        c = np.apply_along_axis(np.diag, 1, b)
        print(c)

    def test_apply_over_axis_1(self):

        a = np.arange(24).reshape(2,3,4)
        print(a)

        # Sum over axes 0 and 2. The result has same number of dimensions as the original array:

        b = np.apply_over_axes(np.sum, a, [0,2])
        print(b)

        # Tuple axis arguments to ufuncs are equivalent:

        c = np.sum(a, axis=(0,2), keepdims=True)
        print(c)



if __name__ == '__main__':
    unittest.main()
