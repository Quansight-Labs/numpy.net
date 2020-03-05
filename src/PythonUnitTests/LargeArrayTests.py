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

    def test_largearray_where_INT32(self):

        matrix = np.arange(16000000, dtype=np.int32).reshape((40, -1));
        print(np.sum(matrix))
        indices = np.where(matrix % 2 == 0);

        m1 = matrix[indices]
        print(np.sum(m1))

    def test_largearray_insert_INT64(self):

        matrix = np.arange(16000000, dtype=np.int64).reshape((40, -1));
        print(np.sum(matrix))

        m1 = np.insert(matrix, 0, [999,100,101])
        print(np.sum(m1))

    def test_largearray_append_INT64(self):

        matrix = np.arange(16000000, dtype=np.int64).reshape((40, -1));
        print(np.sum(matrix))

        m1 = np.append(matrix, [999,100,101])
        print(np.sum(m1))

    def test_largearray_concatenate_INT64(self):

        a = np.arange(16000000, dtype=np.int64).reshape((40, -1));
        b = np.arange(1, 16000001, dtype=np.int64).reshape((40, -1));

        c = np.concatenate((a, b), axis=0)
        print(np.sum(c))

        #d = np.concatenate((a.T, b), axis=1)
        #print(np.sum(d))

        e = np.concatenate((a, b), axis=None)
        print(np.sum(e))

    def test_largearray_min_INT64(self):

        a = np.arange(16000000, dtype=np.int64).reshape((40, -1));

        b = np.amin(a)
        print(np.sum(b))

        b = np.amin(a, axis=0)
        print(np.sum(b))

        b = np.amin(a, axis=1)
        print(np.sum(b))

    def test_largearray_max_INT64(self):

        a = np.arange(16000000, dtype=np.int64).reshape((40, -1));

        b = np.amax(a)
        print(np.sum(b))

        b = np.amax(a, axis=0)
        print(np.sum(b))

        b = np.amax(a, axis=1)
        print(np.sum(b))

    def test_largearray_setdiff1d_INT64(self):

        a = np.arange(16000000, dtype=np.int64);

        b = np.array([3, 4, 5, 6])
        c = np.setdiff1d(a, b)
        print(np.sum(a))
        print(np.sum(b))
        print(np.sum(c))

    def test_largearray_copyto_INT64(self):

        a = np.arange(16000000, dtype=np.int64).reshape(-1, 5);
        print(np.sum(a))

        b = np.array([1, 2, 3, 4, 5])
        np.copyto(a, b)
        print(np.sum(a))

        a = np.arange(16000000, dtype=np.int64).reshape(-1, 5);
        b = np.array([1, 2, 3, 4, 5])
        np.copyto(a, b, where = b % 2 == 0)
        print(np.sum(a))
 
    def test_largearray_sin_DOUBLE(self):

        a = np.ones(16000000, dtype=np.float64).reshape(-1, 5);
        b = np.sin(a)
        print(np.sum(b))

    def test_largearray_diff_INT64(self):

        a = np.arange(0, 16000000 * 3, 3, dtype=np.int64).reshape(-1, 5);
        b = np.diff(a)
        print(np.sum(b))

    def test_largearray_ediff1d_INT64(self):

        a = np.arange(0, 16000000 * 3, 3, dtype=np.int64).reshape(-1, 5);
        b = np.ediff1d(a)
        print(np.sum(b))

    def test_largearray_gradient_INT64(self):

        a = np.arange(0, 16000000 * 3, 3, dtype=np.int64).reshape(-1, 5);
        b = np.gradient(a)
        print(np.sum(b[0]))
        print(np.sum(b[1]))

    def test_largearray_cross_INT64(self):

        a = np.arange(16000000, dtype=np.int64).reshape((-1, 2));
        b = np.arange(1, 16000001, dtype=np.int64).reshape((-1, 2));

        c = np.cross(a, b)
        print(np.sum(c))

    def test_largearray_convolve_INT64(self):

        a = np.arange(160000, dtype=np.int64);
        b = np.arange(1, 160001, dtype=np.int64);

        c = np.convolve(a, b)
        print(np.sum(c))

    def test_largearray_clip_INT64(self):

        a = np.arange(16000000, dtype=np.int64).reshape((-1, 2));

        c = np.clip(a, 1, 1000);
        print(np.sum(c))

        
    def test_largearray_take_INT64(self):

        a = np.arange(16000000, dtype=np.int64).reshape((-1, 2));
        indices = np.arange(0,a.size, 2, np.intp)
        c = np.take(a, indices);
        print(np.sum(c))

    def test_largearray_choose_INT64(self):

        choice1 = np.arange(16000000, dtype=np.int64);
        choice2 = np.arange(16000000, dtype=np.int64);
        choice3 = np.arange(16000000, dtype=np.int64);
        choice4 = np.arange(16000000, dtype=np.int64);

        selection = np.repeat([0,1,2,3], choice1.size/4)

        c = np.choose(selection, [choice1, choice2, choice3, choice4])
        print(np.sum(c))



if __name__ == '__main__':
    unittest.main()

