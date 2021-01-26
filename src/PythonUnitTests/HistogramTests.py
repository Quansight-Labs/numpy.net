import unittest
import numpy as np
from nptest import nptest


class HistogramTests(unittest.TestCase):


    def test_bincount_1(self):

        x = np.arange(5)
        a = np.bincount(x)
        print(a)

        x = np.array([0, 1, 1, 3, 2, 1, 7])
        a = np.bincount(x)
        print(a)

        x = np.array([0, 1, 1, 3, 2, 1, 7, 23])
        a = np.bincount(x)
        print(a)

        print(a.size == np.amax(x)+1)

    def test_bincount_2(self):

        x = np.arange(5, dtype=np.int64)
        a = np.bincount(x)
        print(a)

        x = np.array([0, 1, 1, 3, 2, 1, 7], dtype=np.int16)
        a = np.bincount(x)
        print(a)

        x = np.array([0, 1, 1, 3, 2, 1, 7, 23], dtype=np.int8)
        a = np.bincount(x)
        print(a)

        print(a.size == np.amax(x)+1)

    def test_bincount_3(self):

        w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
        x = np.arange(6, dtype=np.int64)
        a = np.bincount(x, weights=w)
        print(a)

        x = np.array([0, 1, 3, 2, 1, 7], dtype=np.int16)
        a = np.bincount(x,weights=w)
        print(a)

        x = np.array([0, 1, 3, 2, 1, 7], dtype=np.int8)
        a = np.bincount(x, weights=w)
        print(a)

    def test_bincount_4(self):

        x = np.arange(5, dtype=np.int64)
        a = np.bincount(x, minlength=8)
        print(a)

        x = np.array([0, 1, 1, 3, 2, 1, 7], dtype=np.int16)
        a = np.bincount(x, minlength=10)
        print(a)

        x = np.array([0, 1, 1, 3, 2, 1, 7, 23], dtype=np.int8)
        a = np.bincount(x, minlength=32)
        print(a)

        print(a.size == np.amax(x)+1)

    def test_bincount_uint64(self):

        try :
            x = np.arange(5, dtype=np.uint64)
            a = np.bincount(x)
            print(a)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_bincount_double(self):

        try :
            x = np.arange(5, dtype=np.float64)
            a = np.bincount(x)
            print(a)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_bincount_not1d(self):

        try :
            x = np.arange(100, dtype=np.int64).reshape(10,10);
            a = np.bincount(x)
            print(a)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

if __name__ == '__main__':
    unittest.main()
