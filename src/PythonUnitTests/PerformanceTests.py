import unittest
import numpy as np
import time as tm
from nptest import nptest


class PerformanceTests(unittest.TestCase):


    def test_ScalarOperationPerformance(self):

        LoopCount = 200;

        matrix = np.arange(1600000).astype(np.int64).reshape((40, -1));

        start = tm.time()

        #matrix = matrix[1:3:2, 1:-2:3]

        for i in range(LoopCount):
            matrix = matrix / 3;
            matrix = matrix + i;
            

        output = matrix[15:25:2, 15:25:2]

        end = tm.time()

        diff = end-start
        print("Int64 calculations took %f milliseconds" %(diff))
        print(output)

        
    def test_ScalarOperationPerformance_NotContiguous(self):

        LoopCount = 200;

        matrix = np.arange(16000000, dtype=np.int64).reshape((40, -1));

        start = tm.time()

        matrix = matrix[1:40:2, 1:-2:3]

        for i in range(LoopCount):
            matrix = matrix / 3;
            matrix = matrix + i;
            

        output = matrix[15:25:2, 15:25:2]

        end = tm.time()

        diff = end-start
        print("Int64 calculations took %f milliseconds" %(diff))
        print(output)

    def test_MathOperation_Sin(self):

        LoopCount = 200;

        a = np.arange(10000000, dtype=np.float64);

        start = tm.time()


        b = np.sin(a)

        end = tm.time()

        diff = end-start
        print("1000000 sin calculations took %f milliseconds" %(diff))


    def test_xxx(self):

        AASize = 16000000;
        AADim1 = 4000;

        AA = np.arange(AASize, dtype= np.int32).reshape((AADim1, -1))
        BB = np.arange(AASize/AADim1, dtype= np.int16);


        #AA1 = AA[1:40:2, 1:-2:3]

        AA2 = AA / 3
        AA3 = AA2 + 1
        AABB = (AA * BB)

        indexes = np.where(AABB < 100)

        masked = AABB.ravel()[np.flatnonzero(indexes)]

        print(masked)


if __name__ == '__main__':
    unittest.main()
