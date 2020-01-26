import unittest
import numpy as np
import time as tm
from nptest import nptest


class PerformanceTests(unittest.TestCase):


    def test_ScalarOperationPerformance(self):

        LoopCount = 200;

        start = tm.time()

        matrix = np.arange(1600000).astype(np.int64).reshape((40, -1));

        #matrix = matrix[1:3:2, 1:-2:3]

        for x in range(LoopCount):
            matrix = matrix / 3;
            matrix = matrix + x;
            

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



if __name__ == '__main__':
    unittest.main()
