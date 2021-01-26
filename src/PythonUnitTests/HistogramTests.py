import unittest
import numpy as np
from nptest import nptest


class HistogramTests(unittest.TestCase):

#region bincount
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

    def test_bincount_slice(self):

        w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6, .19, -0.8, 0.3, 0.5 ]) # weights
        x = np.arange(10, dtype=np.int64)
        a = np.bincount(x[::2], weights=w[::2])
        print(a)
  

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
#endregion

#region digitize
    def test_digitize_1(self):

        x = np.array([0.2, 6.4, 3.0, 1.6])
        bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        inds = np.digitize(x, bins)
        print(inds)

  
  
    def test_digitize_2(self):

        x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
        bins = np.array([0, 5, 10, 15, 20])

        inds = np.digitize(x, bins, right=True)
        print(inds)
  
        inds = np.digitize(x, bins, right=False)
        print(inds)
  
    def test_digitize_3(self):

        x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
        bins = np.array([20, 15, 10, 5, 0])

        inds = np.digitize(x, bins, right=True)
        print(inds)
  
        inds = np.digitize(x, bins, right=False)
        print(inds)
#endregion

if __name__ == '__main__':
    unittest.main()
