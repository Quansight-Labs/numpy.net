import unittest
import numpy as np
from nptest import nptest

class Test_test1(unittest.TestCase):

    def test_ravel_multi_index_1(self):

          return

    def test_unravel_index_1(self):

          return

    def test_mgrid_1(self):

         return;

    def test_ogrid_1(self):

         return;

    def test_c_1(self):

        a = np.c_[np.array([1,2,3]), np.array([4,5,6])]
        print(a)

        b = np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
        print(b)


    def test_r_1(self):

        a = np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])]
        print(a)

        b = np.r_[-1:1:6j, [0]*3, 5, 6]
        print(b)

        c = np.r_['0,2,0', [1,2,3], [4,5,6]]
        print(c)

        d = np.r_['1,2,0', [1,2,3], [4,5,6]]
        print(d)

        e = np.r_['r',[1,2,3], [4,5,6]]
        print(e)

    def test_s_1(self):
        return

    def test_index_exp_1(self):
        return

    def test_ix_1(self):
        return
    
    def test_ndenumerate_1(self):
        return

    def test_ndindex_1(self):
        return

    def test_fill_diagonal_1(self):
        return

    def test_diag_indices_1(self):
        return
    
    def test_diag_indices_from_1(self):
        return


if __name__ == '__main__':
    unittest.main()
