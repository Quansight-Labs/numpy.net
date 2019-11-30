import unittest
import numpy as np
from nptest import nptest

class Test_UFUNCTests(unittest.TestCase):

    #region UFUNC ADD tests
    def test_UFUNC_AddReduce_1(self):

        x = np.arange(8);

        a = np.add.reduce(x)
        print(a)

        x = np.arange(8).reshape((2,2,2))
        b = np.add.reduce(x)
        print(b)

        c = np.add.reduce(x, 0)
        print(c)

        d = np.add.reduce(x, 1)
        print(d)

        e = np.add.reduce(x, 2)
        print(e)

    def test_UFUNC_AddReduce_2(self):

  
        x = np.arange(8).reshape((2,2,2))
        b = np.add.reduce(x)
        print(b)

        c = np.add.reduce(x, (0,1))
        print(c)

        d = np.add.reduce(x, (1,2))
        print(d)

        e = np.add.reduce(x, (2,1))
        print(e)

    def test_UFUNC_AddReduceAt_1(self):

        a =np.add.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]
        print(a)
        print("**********")

        x = np.linspace(0, 15, 16).reshape(4,4)
        b = np.add.reduceat(x, [0, 3, 1, 2, 0])
        print(b)
        print("**********")

        c = np.multiply.reduceat(x, [0, 3], axis = 1)
        print(c)

    def test_UFUNC_AddOuter_1(self):

        x = np.arange(4);

        a = np.add.outer(x, x)
        print(a.shape)
        print(a)

        x = np.arange(6).reshape((3,2))
        y = np.arange(6).reshape((2,3))
        b = np.add.outer(x, y)
        print(b.shape)
        print(b)
    
    #endregion

    #region UFUNC SUBTRACT tests

    def test_UFUNC_SubtractReduce_1(self):

        x = np.arange(8);

        a = np.subtract.reduce(x)
        print(a)

        x = np.arange(8).reshape((2,2,2))
        b = np.subtract.reduce(x)
        print(b)

        c = np.subtract.reduce(x, 0)
        print(c)

        d = np.subtract.reduce(x, 1)
        print(d)

        e = np.subtract.reduce(x, 2)
        print(e)


    def test_UFUNC_SubtractReduceAt_1(self):

        a =np.subtract.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]
        print(a)
        print("**********")

        x = np.linspace(0, 15, 16).reshape(4,4)
        b = np.subtract.reduceat(x, [0, 3, 1, 2, 0])
        print(b)
        print("**********")

        c = np.multiply.reduceat(x, [0, 3], axis = 1)
        print(c)

    def test_UFUNC_SubtractOuter_1(self):

        x = np.arange(4);

        a = np.subtract.outer(x, x)
        print(a.shape)
        print(a)

        x = np.arange(6).reshape((3,2))
        y = np.arange(6).reshape((2,3))
        b = np.subtract.outer(x, y)
        print(b.shape)
        print(b)

    #endregion
 

if __name__ == '__main__':
    unittest.main()
