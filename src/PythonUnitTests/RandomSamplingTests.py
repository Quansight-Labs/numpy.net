import unittest
import numpy as np
from nptest import nptest


class Test_test1(unittest.TestCase):

    def test_rand_1(self):

        #np.random.seed(1234);

        f = np.random.rand()
        print(f)

        arr = np.random.rand(5000000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

    def test_randn_1(self):

        #np.random.seed(1234);

        f = np.random.randn()
        print(f)

        arr = np.random.randn(5000000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

    def test_standard_normal_1(self):

        #np.random.seed(1234);
        arr = np.random.standard_normal(5000000);
        print(np.max(arr));
        print(np.min(arr));
        print(np.average(arr));


if __name__ == '__main__':
    unittest.main()
