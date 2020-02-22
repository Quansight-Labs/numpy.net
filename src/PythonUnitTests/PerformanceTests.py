import unittest
import numpy as np
import time as tm
from nptest import nptest


class PerformanceTests(unittest.TestCase):


    def test_ScalarOperationPerformance(self):

        LoopCount = 200;

        matrix = np.arange(1600000).astype(np.float64).reshape((40, -1));

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

        matrix = np.arange(16000000, dtype=np.float64).reshape((40, -1));

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

    def test_AddReduce_Performance(self):

        LoopCount = 200;

        a = np.arange(10000000, dtype=np.float64);

        start = tm.time()
        
        for i in range(LoopCount):
            b = np.add.reduce(a)
            print(b)

        end = tm.time()

        diff = end-start
        print("1000000 add calculations took %f milliseconds" %(diff))

    def test_AddReduce_Performance_FLOAT(self):

        LoopCount = 200;

        a = np.arange(10000000, dtype=np.float32);

        start = tm.time()
        
        for i in range(LoopCount):
            b = np.add.reduce(a)
            print(b)

        end = tm.time()

        diff = end-start
        print("1000000 add calculations took %f milliseconds" %(diff))

    def test_AddReduce_Performance2(self):

        LoopCount = 200;

        a = np.arange(0, 4000 * 10 * 4000, dtype=np.float64).reshape(-1, 4000);

        start = tm.time()
        
        for i in range(LoopCount):
            b = np.add.reduce(a)
            c = np.sum(b)
            print(c)

        end = tm.time()

        diff = end-start
        print("1000000 add calculations took %f milliseconds" %(diff))

    def test_AddReduce_Performance2_FLOAT(self):

        LoopCount = 200;

        a = np.arange(0, 4000 * 10 * 4000, dtype=np.float32).reshape(-1, 4000);

        start = tm.time()
        
        for i in range(LoopCount):
            b = np.add.reduce(a)
            c = np.sum(b)
            print(c)

        end = tm.time()

        diff = end-start
        print("1000000 add calculations took %f milliseconds" %(diff))

    def test_AddAccumulate_Performance(self):

        LoopCount = 200;

        a = np.arange(10000000, dtype=np.float64);

        start = tm.time()
        
        for i in range(LoopCount):
            b = np.add.accumulate(a)

        end = tm.time()

        diff = end-start
        print("1000000 add calculations took %f milliseconds" %(diff))

    def test_AddAccumulate_Performance2(self):

        LoopCount = 200;

        a = np.arange(4000 * 4000, dtype=np.float64).reshape(4000,4000);

        start = tm.time()
        
        for i in range(LoopCount):
            b = np.add.accumulate(a)

        end = tm.time()

        diff = end-start
        print("1000000 add calculations took %f milliseconds" %(diff))


    def test_AddReduceAt_Performance(self):

        LoopCount = 200;

        a = np.arange(10000000, dtype=np.float64).reshape((40, -1));

        start = tm.time()
        
        for i in range(LoopCount):
            b = np.add.reduceat(a, [10, 20, 30, 39])
            print(b.shape)

        end = tm.time()

        diff = end-start
        print("1000000 add calculations took %f milliseconds" %(diff))

    def test_AddOuter_Performance(self):

        LoopCount = 200;

        a = np.arange(1000, dtype=np.float64);

        start = tm.time()
        
        for i in range(LoopCount):
            b = np.add.outer(a,a)
            #print(b.shape)

        end = tm.time()

        diff = end-start
        print("1000000 add calculations took %f milliseconds" %(diff))

    def test_AddOuter_Performance_NotSameType(self):

        LoopCount = 200;

        a1 = np.arange(1000, dtype=np.float64);
        a2 = np.arange(1000, dtype=np.int16);

        start = tm.time()
        
        for i in range(LoopCount):
            b = np.add.outer(a2,a1)
            print(b.shape)

        end = tm.time()

        diff = end-start
        print("1000000 add calculations took %f milliseconds" %(diff))




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

        masked = AABB.ravel()[np.flatnonzero(indexes[0])]

        print(masked)

    def test_KEVIN(self):

        sigma = 0.4
        size = int(8 * sigma + 1)

        if size % 2 == 0:
            size = size + 1
        
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis] * 4
        x0 = y0 = size // 2
   
        gaus = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)
        print(gaus)

    def test_copyto_kev(self):

        x = np.arange(0, 6, 1, float).reshape((3,2))
        y = x[:,:, np.newaxis] * 4
        y = x[:,np.newaxis,:] * 4
        #y = y[:,:,:,np.newaxis]
        z = x + y
        print(z)

    def test_Performance_WhereOperation_DOUBLE(self):

        LoopCount = 200;

        matrix = np.arange(16000000, dtype=np.float64).reshape((40, -1));
        x1comp = np.arange(0,16000000,5, dtype=np.float64).reshape((40, -1));
        x2comp = np.arange(0,16000000,10, dtype=np.float64).reshape((40, -1));

        start = tm.time()

        matrix = matrix[1:40:2, 1:-2:3]

        for i in range(LoopCount):
            x1 = np.where(matrix % 5 == 0);
            x2 = np.where(matrix % 10 == 0);

            b1 = np.where(x1 != x1comp)
            b2 = np.where(x2 != x2comp)
            

        output = matrix[15:25:2, 15:25:2]

        end = tm.time()

        diff = end-start
        print("Int64 calculations took %f milliseconds" %(diff))
        print(output)


if __name__ == '__main__':
    unittest.main()
