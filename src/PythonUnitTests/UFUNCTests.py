import unittest
import numpy as np
from nptest import nptest

class Test_UFUNCTests(unittest.TestCase):

    #region UFUNC ADD tests
    def test_UFUNC_AddAccumlate_1(self):

        x = np.arange(8);

        a = np.add.accumulate(x)
        print(a)

        x = np.arange(8).reshape((2,2,2))
        b = np.add.accumulate(x)
        print(b)

        c = np.add.accumulate(x, 0)
        print(c)

        d = np.add.accumulate(x, 1)
        print(d)

        e = np.add.accumulate(x, 2)
        print(e)

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

    def test_UFUNC_SubtractAccumulate_1(self):

        x = np.arange(8);

        a = np.subtract.accumulate(x)
        print(a)

        x = np.arange(8).reshape((2,2,2))
        b = np.subtract.accumulate(x)
        print(b)

        c = np.subtract.accumulate(x, 0)
        print(c)

        d = np.subtract.accumulate(x, 1)
        print(d)

        e = np.subtract.accumulate(x, 2)
        print(e)

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


    #region UFUNC MULTIPLY tests

    def test_UFUNC_MultiplyAccumulate_1(self):

        x = np.arange(8);

        a = np.multiply.accumulate(x)
        print(a)

        x = np.arange(8).reshape((2,2,2))
        b = np.multiply.accumulate(x)
        print(b)

        c = np.multiply.accumulate(x, 0)
        print(c)

        d = np.multiply.accumulate(x, 1)
        print(d)

        e = np.multiply.accumulate(x, 2)
        print(e)

    def test_UFUNC_MultiplyReduce_1(self):

        x = np.arange(8);

        a = np.multiply.reduce(x)
        print(a)

        x = np.arange(8).reshape((2,2,2))
        b = np.multiply.reduce(x)
        print(b)

        c = np.multiply.reduce(x, 0)
        print(c)

        d = np.multiply.reduce(x, 1)
        print(d)

        e = np.multiply.reduce(x, 2)
        print(e)


    def test_UFUNC_MultiplyReduceAt_1(self):

        a =np.multiply.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]
        print(a)
        print("**********")

        x = np.linspace(0, 15, 16).reshape(4,4)
        b = np.multiply.reduceat(x, [0, 3, 1, 2, 0])
        print(b)
        print("**********")

        c = np.multiply.reduceat(x, [0, 3], axis = 1)
        print(c)

    def test_UFUNC_MultiplyOuter_1(self):

        x = np.arange(4);

        a = np.multiply.outer(x, x)
        print(a.shape)
        print(a)

        x = np.arange(6).reshape((3,2))
        y = np.arange(6).reshape((2,3))
        b = np.multiply.outer(x, y)
        print(b.shape)
        print(b)

    #endregion


    #region UFUNC DIVIDE tests

    def test_UFUNC_DivideAccumulate_1(self):

        x = np.arange(8, 16, dtype=np.float64);

        a = np.divide.accumulate(x)
        print(a)
        x = np.arange(8, 16, dtype=np.float64).reshape((2,2,2))
        b = np.divide.accumulate(x)
        print(b)

        c = np.divide.accumulate(x, 0)
        print(c)

        d = np.divide.accumulate(x, 1)
        print(d)

        e = np.divide.accumulate(x, 2)
        print(e)

    def test_UFUNC_DivideReduce_1(self):

        x = np.arange(8, 16, dtype=np.float64);

        a = np.divide.reduce(x)
        print(a)
        print("*****")


        x = np.arange(8, 16, dtype=np.float64).reshape((2,2,2))
        b = np.divide.reduce(x)
        print(b)
        print("*****")

        c = np.divide.reduce(x, 0)
        print(c)
        print("*****")

        d = np.divide.reduce(x, 1)
        print(d)
        print("*****")

        e = np.divide.reduce(x, 2)
        print(e)


    def test_UFUNC_DivideReduceAt_1(self):

        a =np.divide.reduceat(np.arange(8, 16, dtype=np.float64),[0,4, 1,5, 2,6, 3,7])[::2]
        print(a)
        print("**********")

        x = np.linspace(0, 15, 16).reshape(4,4)
        b = np.divide.reduceat(x, [0, 3, 1, 2, 0])
        print(b)
        print("**********")

        c = np.divide.reduceat(x, [0, 3], axis = 1)
        print(c)

    def test_UFUNC_DivideOuter_1(self):

        x = np.arange(4, 8, dtype=np.float64);

        a = np.divide.outer(x, x)
        print(a.shape)
        print(a)

        x = np.arange(8,14, dtype=np.float64).reshape((3,2))
        y = np.arange(8,14, dtype=np.float64).reshape((2,3))
        b = np.divide.outer(x, y)
        print(b.shape)
        print(b)

    #endregion
 
     #region UFUNC REMAINDER tests

    def test_UFUNC_RemainderAccumulate_1(self):

        x = np.arange(16, 8, -1, dtype=np.float64);

        a = np.remainder.accumulate(x)
        print(a)
        x = np.arange(16, 8, -1, dtype=np.float64).reshape((2,2,2))
        b = np.remainder.accumulate(x)
        print(b)

        c = np.remainder.accumulate(x, 0)
        print(c)

        d = np.remainder.accumulate(x, 1)
        print(d)

        e = np.remainder.accumulate(x, 2)
        print(e)

    def test_UFUNC_RemainderReduce_1(self):

        x = np.arange(16, 8, -1, dtype=np.float64);

        a = np.remainder.reduce(x)
        print(a)
        print("*****")


        x = np.arange(16, 8, -1, dtype=np.float64).reshape((2,2,2))
        b = np.remainder.reduce(x)
        print(b)
        print("*****")

        c = np.remainder.reduce(x, 0)
        print(c)
        print("*****")

        d = np.remainder.reduce(x, 1)
        print(d)
        print("*****")

        e = np.remainder.reduce(x, 2)
        print(e)


    def test_UFUNC_RemainderReduceAt_1(self):

        a =np.remainder.reduceat(np.arange(16,8,-1, dtype=np.float64),[0,4, 1,5, 2,6, 3,7])[::2]
        print(a)
        print("**********")

        x = np.linspace(0, 15, 16).reshape(4,4)
        b = np.remainder.reduceat(x, [0, 3, 1, 2, 0])
        print(b)
        print("**********")

        c = np.remainder.reduceat(x, [0, 3], axis = 1)
        print(c)

    def test_UFUNC_RemainderOuter_1(self):

        x = np.arange(4, 8, dtype=np.float64);

        a = np.remainder.outer(x, x)
        print(a.shape)
        print(a)

        x = np.arange(14,8,-1, dtype=np.float64).reshape((3,2))
        y = np.arange(14,8,-1, dtype=np.float64).reshape((2,3))
        b = np.remainder.outer(x, y)
        print(b.shape)
        print(b)

    #endregion

    #region UFUNC FMOD tests

    def test_UFUNC_FModAccumulate_1(self):

        x = np.arange(16, 8, -1, dtype=np.float64);

        a = np.fmod.accumulate(x)
        print(a)
        x = np.arange(16, 8, -1, dtype=np.float64).reshape((2,2,2))
        b = np.fmod.accumulate(x)
        print(b)

        c = np.fmod.accumulate(x, 0)
        print(c)

        d = np.fmod.accumulate(x, 1)
        print(d)

        e = np.fmod.accumulate(x, 2)
        print(e)

    def test_UFUNC_FModReduce_1(self):

        x = np.arange(16, 8, -1, dtype=np.float64);

        a = np.fmod.reduce(x)
        print(a)
        print("*****")


        x = np.arange(16, 8, -1, dtype=np.float64).reshape((2,2,2))
        b = np.fmod.reduce(x)
        print(b)
        print("*****")

        c = np.fmod.reduce(x, 0)
        print(c)
        print("*****")

        d = np.fmod.reduce(x, 1)
        print(d)
        print("*****")

        e = np.fmod.reduce(x, 2)
        print(e)


    def test_UFUNC_FModReduceAt_1(self):

        a =np.fmod.reduceat(np.arange(16,8,-1, dtype=np.float64),[0,4, 1,5, 2,6, 3,7])[::2]
        print(a)
        print("**********")

        x = np.linspace(0, 15, 16).reshape(4,4)
        b = np.fmod.reduceat(x, [0, 3, 1, 2, 0])
        print(b)
        print("**********")

        c = np.fmod.reduceat(x, [0, 3], axis = 1)
        print(c)

    def test_UFUNC_FModOuter_1(self):

        x = np.arange(4, 8, dtype=np.float64);

        a = np.fmod.outer(x, x)
        print(a.shape)
        print(a)

        x = np.arange(14,8,-1, dtype=np.float64).reshape((3,2))
        y = np.arange(14,8,-1, dtype=np.float64).reshape((2,3))
        b = np.fmod.outer(x, y)
        print(b.shape)
        print(b)

    #endregion
if __name__ == '__main__':
    unittest.main()
