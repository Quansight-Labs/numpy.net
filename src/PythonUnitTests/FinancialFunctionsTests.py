import unittest
import numpy as np
import time as tm
from decimal import Decimal
import math
from _financial import npf

class FinancialFunctionsTests(unittest.TestCase):

    def test_fv_int(self):

        x = npf.fv(75, 20, -2000, 0, 0)
        print(x)

    def test_fv_float(self):

        x = npf.fv(0.075, 20, -2000, 0, 0)
        print(x)

    def test_fv_decimal(self):

        x = npf.fv(Decimal('0.075'), Decimal('20'), Decimal('-2000'), 0, 0)
        print(x)

    def test_fv_when_is_begin_float(self):

        x = npf.fv(0.075, 20, -2000, 0, 'begin')
        print(x)

    def test_fv_when_is_begin_decimal(self):

        x = npf.fv(Decimal('0.075'), Decimal('20'), Decimal('-2000'), 0, 'begin')
        print(x)

    def test_fv_when_is_end_float(self):

        x = npf.fv(0.075, 20, -2000, 0, 'end')
        print(x)

    def test_fv_when_is_end_decimal(self):

        x = npf.fv(Decimal('0.075'), Decimal('20'), Decimal('-2000'), 0, 'end')
        print(x)

    def test_fv_broadcast(self):

        result = npf.fv([[0.1], [0.2]], 5, 100, 0, [0, 1])
        print(result)

    def test_fv_some_rates_zero(self):

        result = npf.fv([0, 0.1], 5, 100, 0)
        print(result)

if __name__ == '__main__':
    unittest.main()
