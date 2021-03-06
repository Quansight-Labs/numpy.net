import unittest
import numpy as np
from nptest import nptest


class ComplexNumbersTests(unittest.TestCase):

    
    def test_F2C_1_COMPLEX(self):

       fvalues = [0-1j, 12-1.1j, 45.21+45.3456j, 34+87j, 99.91+789J]
       F = np.array(fvalues, dtype=np.complex)
       print("Values in Fahrenheit degrees:")
       print(F)
       print("Values in  Centigrade degrees:") 
       print(5*F/9 - 5*32/9)
    
    def test_multiply_1x_COMPLEX(self):

       fvalues = [0-1j, 12-1.1j, 45.21+45.3456j, 34+87j, 99.91+789J]
       F = np.array(fvalues, dtype=np.complex)

       evalues = [5-1.567j, -12.56-2.1j, 145.21+415.3456j, -34+87j, 99.91+7189J]
       E = np.array(evalues, dtype=np.complex)

       G = F * E

       print(G)

    def test_conj_1(self):

        fvalues = [0-1j, 12-1.1j, 45.21+45.3456j, 34+87j, 99.91+789J]
        F = np.array(fvalues, dtype=np.complex)

        E = np.conj(F);

        print(F)
        print(E)


    def test_angle_1(self):

        a = np.angle([1.0, 1.0j, 1+1j])               # in radians
        print(a)

        b = np.angle(1+1j, deg=True)                  # in degrees
        print(b)

        c = np.angle([-1,2,-3])
        print(c)

    def test_sort_complex_1(self):

        IntTestData = [5, 3, 6, 2, 1]
        ComplexTextData = [1 + 2j, 2 - 1j, 3 - 2j, 3 - 3j, 3 + 5j]

        a = np.sort(IntTestData)
        print(a)

        b = np.sort_complex(IntTestData)
        print(b)

        c = np.sort(ComplexTextData)
        print(c)

        d = np.sort_complex(ComplexTextData)
        print(d)



if __name__ == '__main__':
    unittest.main()
