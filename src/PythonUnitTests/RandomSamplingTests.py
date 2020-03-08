import unittest
import numpy as np
from nptest import nptest


class Test_test1(unittest.TestCase):

    def test_rand_1(self):

        np.random.seed(8765);

        f = np.random.rand()
        print(f)

        arr = np.random.rand(5000000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

    def test_randn_1(self):

        np.random.seed(1234);

        f = np.random.randn()
        print(f)

        arr = np.random.randn(5000000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

    def test_randbool_1(self):

        np.random.seed(8188);

        f = np.random.randint(False,True+1,4, dtype=np.bool)
        print(f)

        arr = np.random.randint(False,True+1,5000000, dtype=np.bool);
        cnt = arr == True
        print(cnt.size);

    def test_randint8_1(self):

        np.random.seed(9292);

        f = np.random.randint(2,3,4, dtype=np.int8)
        print(f)

        arr = np.random.randint(2,8,5000000, dtype=np.int8);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.randint(-2,3,5000000, dtype=np.int8);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_randuint8_1(self):

        np.random.seed(1313);

        f = np.random.randint(2,3,4, dtype=np.uint8)
        print(f)

        arr = np.random.randint(2,128,5000000, dtype=np.uint8);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

        first10 = arr[0:10:1]
        print(first10)

    def test_randint16_1(self):

        np.random.seed(8381);

        f = np.random.randint(2,3,4, dtype=np.int8)
        print(f)

        arr = np.random.randint(2,2478,5000000, dtype=np.int16);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.randint(-2067,3000,5000000, dtype=np.int16);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_randuint16_1(self):

        np.random.seed(5555);

        f = np.random.randint(2,3,4, dtype=np.uint16)
        print(f)

        arr = np.random.randint(23,12801,5000000, dtype=np.uint16);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

        first10 = arr[0:10:1]
        print(first10)

    def test_randint_1(self):

        #np.random.seed(1234);

        f = np.random.randint(2,3,4)
        print(f)

        arr = np.random.randint(2,3,5000000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));


        arr = np.random.randint(-2,3,5000000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

    def test_randuint_1(self):

        #np.random.seed(1234);

        f = np.random.randint(2,3,4, dtype=np.uint32)
        print(f)

        arr = np.random.randint(2,5,5000000, dtype=np.uint32);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

    def test_randint64_1(self):

        #np.random.seed(1234);

        f = np.random.randint(2,3,4, dtype=np.int64)
        print(f)

        arr = np.random.randint(2,3,5000000, dtype=np.int64);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));


        arr = np.random.randint(-2,3,5000000, dtype=np.int64);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

    def test_randuint64_1(self):

        #np.random.seed(1234);

        f = np.random.randint(2,3,4, dtype=np.uint64)
        print(f)

        arr = np.random.randint(2,5,5000000, dtype=np.uint64);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

    def test_standard_normal_1(self):

        #np.random.seed(1234);
        arr = np.random.standard_normal(5000000);
        print(np.max(arr));
        print(np.min(arr));
        print(np.average(arr));

        
    def test_beta_1(self):

        a = np.arange(1,11, dtype=np.float64);
        b = np.arange(1,11, dtype= np.float64);

        arr = np.random.beta(b, b, 10);
        print(arr);

       
    def test_rand_binomial_1(self):

        np.random.seed(123)

        arr = np.random.binomial(9, 0.1, 20);
        s = np.sum(arr== 0);
        print(s);
        print(arr);

        arr = np.random.binomial(9, 0.1, 20000);
        s = np.sum(arr== 0);
        print(s)


if __name__ == '__main__':
    unittest.main()
