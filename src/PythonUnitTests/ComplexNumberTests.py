import unittest
import numpy as np
from nptest import nptest


class ComplexNumbersTests(unittest.TestCase):

    def test_angle_1(self):

        a = np.angle([1.0, 1.0j, 1+1j])               # in radians
        print(a)

        b = np.angle(1+1j, deg=True)                  # in degrees
        print(b)

        c = np.angle([-1,2,-3])
        print(c)


if __name__ == '__main__':
    unittest.main()
