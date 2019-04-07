import unittest
import numpy as np
from nptest import nptest


class IteratorTests(unittest.TestCase):
    def test_nditer_1(self):
      
        a = np.arange(0, 6).reshape(2,3)
        b = np.array([7,8,9])

        for aa in np.nditer(a):
            print(aa)


        for aa in np.nditer((a,b)):
            print(aa)

            
        for aa in np.nditer((a,b,a,b)):
            print(aa)

        #for bb in np.nditer((a,b), flags=['multi_index', 'refs_ok', 'zerosize_ok']).itviews[0]:
        #    print(bb)

        #for cc in np.nditer((a,b), flags=['multi_index', 'refs_ok', 'zerosize_ok']).itviews[1]:
        #    print(cc)


    def test_ndindex_1(self):

        for aa in np.ndindex((2,3)):
            print(aa)

    def test_ndenumerate_1(self):

        a = np.arange(0, 6).reshape(2,3)

        for aa in np.ndenumerate(a):
            print(aa)


if __name__ == '__main__':
    unittest.main()
