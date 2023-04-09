import unittest
import numpy as np
from nptest import nptest

class Test_SerializationTests(unittest.TestCase):
    
    def test_ToSystemArray_Reverse(self):

        adata =  [[ 1, 2 ], [ 3, 4 ], [ 5, 6 ]];
        a = np.array(adata);
        ar = a[:, ::-1]
        #print(ar)

        bdata = [[[ 14 ], [ 13 ], [ 12 ], [ 11 ] ], [ [ 18 ], [ 17 ], [ 16 ], [ 15 ] ], [ [ 22 ], [ 21 ], [ 20 ], [ 19 ] ] ]
        b = np.array(bdata);
        br = b[:, ::-1, ::-1]
        #print(br)

        br = b[::-1, :, ::-1]
        #print(br)

        c = np.arange(0,256, dtype=np.int16).reshape(4,4,4,4);
        cr = c[::-2,::-1,::-2,::-1]
        #print(cr)

        crr = cr[::-1, ::-2, ::-1, ::-2]
        #print(crr)

        crr[0,1,0,1] = -55
        crr[1,1,1,1] = -77
        crr[1,0,0,1] = -88

        print(c)

        print(c[1,2,1,2])

        print(c[3,2,3,2])

        print(c[3,0,1,2])


if __name__ == '__main__':
    unittest.main()
