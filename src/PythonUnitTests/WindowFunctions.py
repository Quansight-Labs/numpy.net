import unittest
import numpy as np
from nptest import nptest

class WindowTests(unittest.TestCase):


    def test_bartlett_1(self):

        b = np.bartlett(5)
        print(b)

        b = np.bartlett(10)
        print(b)

        b = np.bartlett(12)
        print(b)

        return

    def test_blackman_1(self):

        b = np.blackman(5)
        print(b)

        b = np.blackman(10)
        print(b)

        b = np.blackman(12)
        print(b)

        return

    def test_hamming_1(self):

        b = np.hamming(5)
        print(b)

        b = np.hamming(10)
        print(b)

        b = np.hamming(12)
        print(b)

        return

    def test_hanning_1(self):

        b = np.hanning(5)
        print(b)

        b = np.hanning(10)
        print(b)

        b = np.hanning(12)
        print(b)

        return

    def test_kaiser_1(self):

        a = np.kaiser(12, 14)
        print(a)

        a = np.kaiser(3, 5)
        print(a)

        return

if __name__ == '__main__':
    unittest.main()
