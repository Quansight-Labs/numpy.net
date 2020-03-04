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

        x_mat = np.add(x_range.reshape(1, width,1), y_range.reshape(1, height, 1))

        z = np.sum(x_mat, axis=0)
        z1 = np.sum(z)
        print(z1)

        z = np.sum(x_mat, axis=1)
        z1 = np.sum(z)
        print(z1)

        z = np.sum(x_mat, axis=2)
        z1 = np.sum(z)
        print(z1)


if __name__ == '__main__':
    unittest.main()

