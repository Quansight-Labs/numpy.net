import unittest
import numpy as np
from nptest import nptest


class NANFunctionsTests(unittest.TestCase):

    def test_matmul_asiamartini_bugreport(self):

        rq = np.array([0.5,0.5,0.5,0.5]);
        am = np.array([[-21.5, 33.5, 17.5, -12.5], [33.5, 12.5, 23.5, 15.5], [17.5,23.5,-30.5,-17.5], [-12.5, 15.5, -17.5, 39.5] ]);
   
        temp1 = np.matmul(rq.T, am);
        print(temp1)

 
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

    def test_matmul_7(self):
        
        a = np.full((3,3), 2);
        b = np.full((3,1), 3);
        ret = np.matmul(a, b);
        print(ret.shape)
        print(ret)

    def test_matmul_8(self):
        
        a = np.full((3,3), 2);
        b = np.full((3), 3);
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

    def test_maxtrix_99_WORKS(self):
        a = np.linspace(0.0, 1.0, num=16).reshape(1,16)

        b = np.reshape(a, (1,1,16)) * np.ones((32, 1)) * 1
        #print(b)
        c = np.sum(b)
        print(c)

    def test_maxtrix_99_BROKEN(self):
        a = np.linspace(0.0, 1.0, num=32).reshape(1,32)
        print(a)

        b = np.reshape(a, (1,1,32)) * np.ones((65536, 1)) # * 1
        #print(b)
        c = np.sum(b)
        print(c)

    def test_maxtrix_100_BROKEN(self):
        a = np.arange(0, 32).reshape(1,32);
        print(a)

        b = np.reshape(a, (1,1,32)) * np.ones((65536, 1)) # * 1
        #print(b)
        c = np.sum(b)
        print(c)


    def test_maxtrix_101_BROKEN(self):
        a = np.arange(0, 32).reshape(1,32);
        print(a)

        b = np.full((1, 1, 32), 2) * np.full((65536, 1), 3) # * 1
        #print(b)

        d = np.where(b != 6)
        c = np.sum(b)
        print(c)

 

if __name__ == '__main__':
    unittest.main()
