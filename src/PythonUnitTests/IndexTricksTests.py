import unittest
import numpy as np
from nptest import nptest

class Test_test1(unittest.TestCase):

    def test_ravel_multi_index_1(self):

          return

    def test_unravel_index_1(self):

          return

    def test_mgrid_1(self):
        
         a = nptest.mgrid[0:5]
         print(a)

         print("************")

         b = nptest.mgrid[0.0:5.5]
         print(b)

         print("************")


         c = np.mgrid[0:5,0:5]
         print(c)

         print("************")

         d = np.mgrid[0:5.5,0:5.5]
         print(d)

         print("************")

         e = nptest.mgrid[3:5,4:6, 2:4.2]
         print(e)

         return;

    def test_ogrid_1(self):

         a = nptest.ogrid[0:5]
         print(a)

         print("************")

         b = nptest.ogrid[0.0:5.5]
         print(b)

         print("************")


         c = nptest.ogrid[0:5,0:5]
         print(c)

         print("************")

         d = np.ogrid[0:5.5,0:5.5]
         print(d)

         print("************")

         e = nptest.ogrid[3:5,4:6, 2:4.2]
         print(e)

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
    
 

    def test_fill_diagonal_1(self):

        a = np.zeros((3, 3), int)
        np.fill_diagonal(a, 5)
        print(a)

        a = np.zeros((3, 3, 3, 3), int)
        np.fill_diagonal(a, 4)
        print(a[0,0])
        print(a[1,1])
        print(a[2,2])

        # tall matrices no wrap
        a = np.zeros((5, 3),int)
        np.fill_diagonal(a, 4)
        print(a)

        # tall matrices wrap
        a = np.zeros((5, 3),int)
        np.fill_diagonal(a, 4, wrap = True)
        print(a)

        # wide matrices wrap
        a = np.zeros((3, 5),int)
        np.fill_diagonal(a, 4, wrap = True)
        print(a)

        return

    def test_diag_indices_1(self):

        di = np.diag_indices(4)
        print(di)

        a = np.arange(16).reshape(4, 4)
        a[di] = 100
        print(a)

        return
    
    def test_diag_indices_from_1(self):

        a = np.arange(16).reshape(4, 4)
        di = np.diag_indices_from(a)
        print(di)
        return


if __name__ == '__main__':
    unittest.main()
