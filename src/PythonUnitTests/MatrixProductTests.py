import unittest
import numpy as np
from nptest import nptest


class NANFunctionsTests(unittest.TestCase):

 
    def test_matmul_1(self):

        a = np.arange(9).reshape(3, 3);
        b = np.arange(9).reshape(3, 3);

        ret = np.matmul(a, b);
        print(ret)

    def test_matmul_2(self):
        
        a = np.full((3, 3), 2);
        b = np.full((3, 3), 2);
        ret = np.matmul(a, b);
        print(ret)

    def test_matmul_3(self):
        
        a = np.arange(9).reshape(3, 3);
        b = np.arange(3).reshape(3);
        ret = np.matmul(a, b);
        print(ret)

    def test_matmul_4(self):
        
        a = np.full((3, 3), 2);
        b = np.full((3), 3);
        ret = np.matmul(a, b);
        print(ret)

    def test_matmul_5(self):
        
        a = np.full((3,2,2), 2);
        b = np.full((3,2,2), 3);
        ret = np.matmul(a, b);
        print(ret.shape)
        print(ret)

    def test_matmul_6(self):
        
        a = np.full((3,1,2,2), 2);
        b = np.full((3,2,2), 3);
        ret = np.matmul(a, b);
        print(ret.shape)
        print(ret)

    def test_matmul_bad1(self):
        
        a = np.full((3,2,2), 2);
        b = np.full((3,2,2), 3);
        ret = np.matmul(a, 3);
        print(ret) 

    def test_matmul_bad2(self):
        
        a = np.full((12), 2);
        b = np.full((3,4), 3);
        ret = np.matmul(a, b);
        print(ret) 

    def test_rand_1(self):
        
        a = np.random.randn(2,3,4);
        print(a)

    def test_rand_uniform_1(self):
        
        a = np.random.uniform(-1, 1,40);
        print(a)
    
 

if __name__ == '__main__':
    unittest.main()
