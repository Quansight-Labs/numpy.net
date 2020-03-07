import unittest
import numpy as np
from nptest import nptest


class Test_test1(unittest.TestCase):

      def test_standard_normal_1(self):

        #np.random.seed(1234);
        arr = np.random.standard_normal(5000000);
        print(np.max(arr));
        print(np.min(arr));
        print(np.average(arr));

if __name__ == '__main__':
    unittest.main()
