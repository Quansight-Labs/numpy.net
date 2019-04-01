import unittest
import numpy as np
from nptest import nptest


class NANFunctionsTests(unittest.TestCase):

    def test_nanmin_1(self):

        a = np.array([[1, 2], [3, np.nan]])

        aa = np.isnan(a)

        b = np.nanmin(a)
        print(b)
    
        c = np.nanmin(a, axis=0)
        print(c)

        d = np.nanmin(a, axis=1)
        print(d)

        # When positive infinity and negative infinity are present:

        e = np.nanmin([1, 2, np.nan, np.inf])
        print(e)

        f = np.nanmin([1, 2, np.nan, np.NINF])
        print(f)

        g = np.amin([1, 2, -3, np.NINF])
        print(g)


if __name__ == '__main__':
    unittest.main()
