import unittest
import numpy as np
from nptest import nptest

class NpyIOTests(unittest.TestCase):

    def test_save_and_load(self):

        t1 = np.arange(1,7);
        np.save('c:/temp/t1', t1)
        kev = np.load('c:/temp/t1.npy')

        t2 = t1.reshape(2,3)
        np.save('c:/temp/t2', t2)
        kev = np.load('c:/temp/t2.npy')

        t3 = t2.reshape(3,2)
        np.save('c:/temp/t3', t3)
        kev = np.load('c:/temp/t3.npy')

if __name__ == '__main__':
    unittest.main()
