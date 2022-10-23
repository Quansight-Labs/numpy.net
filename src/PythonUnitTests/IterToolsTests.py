import unittest
import numpy as np
from nptest import nptest
from IterTools import IterTools

class Test_IterToolsTests(unittest.TestCase):

    def test_IterTools_Product_1(self):

        for iter in IterTools.product('ABC', 'xy', '12'):
            print(iter)

        for iter in IterTools.product(range(2)):
            print(iter)

       

if __name__ == '__main__':
    unittest.main()
