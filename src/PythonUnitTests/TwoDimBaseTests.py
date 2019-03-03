import unittest
import numpy as np
from nptest import nptest

class Test_test1(unittest.TestCase):
  
    def test_diag_1(self):
        m = np.arange(9);
        n = np.diag(m)

        print(m)
        print(n)

        m = np.arange(9).reshape(3,3);
        n = np.diag(m)

        print(m)
        print(n)

    def test_diagflat_1(self):
        m = np.arange(1,5).reshape(2,2);
        n = np.diagflat(m)

        print(m)
        print(n)

        m = np.arange(1,3)
        n = np.diagflat(m, 1)

        print(m)
        print(n)

        m = np.arange(1,3)
        n = np.diagflat(m, -1)

        print(m)
        print(n)

        
    def test_eye_1(self):

        a = np.eye(2, dtype = np.int32)

        print(a)
        print(a.shape)
        print(a.strides)
        print("")

        b = np.eye(3, k = 1)

        print(b)
        print(b.shape)
        print(b.strides)

        
    def test_fliplr_1(self):
        m = np.arange(8).reshape((2,2,2))
        n = np.fliplr(m)

        print(m)
        print(n)

    def test_flipud_1(self):
        m = np.arange(8).reshape((2,2,2))
        n = np.flipud(m)

        print(m)
        print(n)


    def test_tri_1(self):
        print(np.tri(3, 5, 2, dtype=int))
        print("***********")
        print(np.tri(3, 5, -1))


    def test_tril_1(self):

        a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
        b = np.tril(a, -1)
        print(a)
        print("***********")
        print(b)
  
    def test_triu_1(self):

        a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
        b = np.triu(a, -1)
        print(a)
        print("***********")
        print(b)

    def test_vander_1(self):

        return
    
    def test_histogram2d(self):

        return
    
    def test_mask_indices(self):

        return

    
    def test_tril_indices(self):

        return
    
    def test_tril_indices_from(self):

        return
    
    def test_triu_indices(self):

        return
    
    def test_triu_indices_from(self):

        return
 

if __name__ == '__main__':
    unittest.main()
