import unittest
import numpy as np
from nptest import nptest


class ObjectOperationsTests(unittest.TestCase):

    
    def test_ObjectArray_Test1(self):

        TestData = ['A', 'B', 'C', 'D']

        a = np.array(TestData, dtype=object)
        print(a)

        a = a.reshape((2,2))
        print(a)

        a = a * 2
        print(a)

        a = a / 20
        print(a)


if __name__ == '__main__':
    unittest.main()
