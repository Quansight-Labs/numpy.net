import unittest
import numpy as np
from nptest import nptest


class LargeArrayTests(unittest.TestCase):

    def test_largearray_matmul_INT64_1(self):
        width = 1024
        height = 1024

        x_range = np.arange(0,width, 1, dtype = np.int64)
        y_range = np.arange(0,height*2, 2, dtype = np.int64)

        x_mat = np.matmul(x_range.reshape(width,1), y_range.reshape(1, height))
        z = np.sum(x_mat)
        print(z)


    def test_largearray_matmul_INT64_2(self):
        width = 1024
        height = 1024

        x_range = np.arange(0,width, 1, dtype = np.int64)
        y_range = np.arange(0,height*2, 2, dtype = np.int64)

        x_mat = np.matmul(x_range.reshape(width,1), y_range.reshape(1, height))

        z = np.sum(x_mat, axis=0)
        z1 = np.sum(z)
        print(z1)

        z = np.sum(x_mat, axis=1)
        z1 = np.sum(z)
        print(z1)


    def test_largearray_add_INT64_1(self):
        width = 1024
        height = 1024

        x_range = np.arange(0,width, 1, dtype = np.int64)
        y_range = np.arange(0,height*2, 2, dtype = np.int64)

        x_mat = np.add(x_range.reshape(width,1), y_range.reshape(1, height))

        z = np.sum(x_mat, axis=0)
        z1 = np.sum(z)
        print(z1)

        z = np.sum(x_mat, axis=1)
        z1 = np.sum(z)
        print(z1)


    def test_largearray_add_INT64_2(self):
        width = 1024
        height = 1024

        x_range = np.arange(0,width, 1, dtype = np.int64)
        y_range = np.arange(0,height*2, 2, dtype = np.int64)

        x_mat = np.add(x_range.reshape(width,1), y_range.reshape(1, height))
        x_mat = np.expand_dims(x_mat, 0)

        z = np.sum(x_mat, axis=0)
        z1 = np.sum(z)
        print(z1)

        z = np.sum(x_mat, axis=1)
        z1 = np.sum(z)
        print(z1)

        z = np.sum(x_mat, axis=2)
        z1 = np.sum(z)
        print(z1)

    def test_largearray_multiply_INT64_1(self):
        width = 2048
        height = 2048

        x_range = np.arange(0,width, 1, dtype = np.int64)
        y_range = np.arange(0,height*2, 2, dtype = np.int64)

        x_mat = np.multiply(x_range.reshape(width,1), y_range.reshape(1, height))

        z = np.sum(x_mat, axis=0)
        z1 = np.sum(z)
        print(z1)

        z = np.sum(x_mat, axis=1)
        z1 = np.sum(z)
        print(z1)

    def test_largearray_multiply_INT64_2(self):
        width = 4096
        height = 4096

        x_range = np.arange(0,width, 1, dtype = np.int64)
        y_range = np.arange(0,height*2, 2, dtype = np.int64)

        x_mat = np.multiply(x_range.reshape(1, width), y_range.reshape(height, 1))
        x_mat = np.expand_dims(x_mat, 0)

        z = np.sum(x_mat, axis=0)
        z1 = np.sum(z)
        print(z1)

        z = np.sum(x_mat, axis=1)
        z1 = np.sum(z)
        print(z1)

        z = np.sum(x_mat, axis=2)
        z1 = np.sum(z)
        print(z1)

    def test_largearray_copy_int64_1(self):

        length = 268435435 # (Int32.MaxValue) / sizeof(double) - 20;
        x = np.arange(0, length, 1, dtype = np.int64);
        z = np.sum(x);
        print(z)
        y = x.copy()
        z = np.sum(y)
        print(z)

    def test_largearray_copy_int64_2(self):

        length = 268435434 # (Int32.MaxValue) / sizeof(double) - 21;
        x = np.arange(0, length, 1, dtype = np.int64).reshape(2,-1);
        z = np.sum(x, axis=0);
        z = np.sum(z)
        print(z)
        y = x.copy()
        z = np.sum(y, axis=1)
        z = np.sum(z)
        print(z)

    def test_largearray_meshgrid_int64_2(self):
        length = 100 * 100

        x = np.arange(0,length, 1, dtype = np.int64)
        x1, x2 = np.meshgrid(x,x)
        print(x1.shape)
        print(x2.shape)

        z = np.sum(x1)
        print(z)

        z = np.sum(x2)
        print(z)

    def test_largearray_checkerboard_1(self):
 
        x = np.zeros((2048,2048),dtype=int)
        x[1::2,::2] = 1
        x[::2,1::2] = 1


        print(np.sum(x))

    def test_largearray_byteswap_int64_2(self):

        length = 1024 * 1024* 32 # (Int32.MaxValue) / sizeof(double) - 21;
        x = np.arange(0, length, 1, dtype = np.int64).reshape(2,-1);
        y = x.byteswap();

        z = np.sum(y, axis=0);
        z = np.sum(z)
        print(z)

        z = np.sum(y, axis=1)
        z = np.sum(z)
        print(z)

    def test_largearray_unique_INT32(self):

        matrix = np.arange(16000000, dtype=np.int32).reshape((40, -1));

        matrix = matrix[1:40:2, 1:-2:1]

        uvalues, indexes, inverse, counts = np.unique(matrix, return_counts = True, return_index=True, return_inverse=True);
             
        print(np.sum(uvalues))
        print(np.sum(indexes))
        print(np.sum(inverse))
        print(np.sum(counts))



if __name__ == '__main__':
    unittest.main()

