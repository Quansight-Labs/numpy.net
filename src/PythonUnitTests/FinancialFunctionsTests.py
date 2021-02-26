import unittest
import numpy as np
import time as tm
from decimal import Decimal
import math
from _financial import npf

class FinancialFunctionsTests(unittest.TestCase):

#region npf.fv

    def test_fv_int(self):

        x = npf.fv(75, 20, -2000, 0, 0)
        print(x)

    def test_fv_float(self):

        x = npf.fv(0.075, 20, -2000, 0, 0)
        print(x)

    def test_fv_decimal(self):

        x = npf.fv(Decimal('0.075'), Decimal('20'), Decimal('-2000'), 0, 0)
        print(x)

    def test_fv_complex(self):

        x = npf.fv(0.075j, 20, -2000, 0, 0)
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


    def test_fv_float_array_1(self):

        x = npf.fv([-0.075,1.075,-1.075], [20], [-2100,2000,-2500], 0, ['begin','end', 'begin'])
        print(x)

    def test_fv_float_array_1A(self):

        x = npf.fv([-0.075,1.075,-1.075], [20], [-2100,2000,-2500], 0, [1,0,1])
        print(x)

    def test_fv_float_array_1B(self):

        x = npf.fv([-0.075,1.075,-1.075], [20], [-2100,2000,-2500], 0, [1,'end', 'begin'])
        print(x)

    def test_fv_float_array_2(self):

        try:
            x = npf.fv([-0.075,1.075,-1.075], [20], [-2100,2000,-2500], 0, ['begin','end', 'xxx'])
            print(x)
        except:
            print("exception caught")

    def test_fv_float_array_3(self):

        try:
            x = npf.fv([-0.075,1.075,-1.075], [20], [-2100,2000,-2500], 0, ['begin','end'])
            print(x)
        except:
            print("exception caught")
#endregion

#region pmt

    def test_pmt_1_DOUBLE(self):

        res = npf.pmt(0.08 / 12, 5 * 12, 15000)
        print(res)

        res = npf.pmt(0.0, 5 * 12, 15000)
        print(res)

        res = npf.pmt([[0.0, 0.8], [0.3, 0.8]], [12, 3], [2000, 20000])
        print(res)


    def test_pmt_1_DECIMAL(self):

        res = npf.pmt(Decimal('0.08') / Decimal('12'), Decimal('60'), Decimal('15000'))
        print(res)

        res = npf.pmt(Decimal('0.0'), Decimal('60'), Decimal('15000'))
        print(res)

        res = npf.pmt([[Decimal('0.0'), Decimal('0.8')], [Decimal('0.3'), Decimal('0.8')]], [Decimal('12'), Decimal('3')], [Decimal('2000'), Decimal('20000')])
        print(res)

    def test_pmt_1_COMPLEX(self):

        res = npf.pmt(0.08j / 12j, 5 * 12, 15000)
        print(res)

        res = npf.pmt(0.0j, 5 * 12, 15000)
        print(res)

        res = npf.pmt([[0.0j, 0.8j], [0.3j, 0.8j]], [12, 3], [2000, 20000])
        print(res)

  
    def test_pmt_when_DOUBLE(self):

        res = npf.pmt(0.08 / 12, 5 * 12, 15000, 0, 0)
        print(res)

        res = npf.pmt(0.08 / 12, 5 * 12, 15000., 0, 'end')
        print(res)

        res = npf.pmt(0.08 / 12, 5 * 12, 15000., 0, 1)
        print(res)

        res = npf.pmt(0.08 / 12, 5 * 12, 15000., 0, 'begin')
        print(res)

    def test_pmt_when_DECIMAL(self):

        res = npf.pmt(Decimal('0.08') / Decimal('12'), 5 * 12, 15000, 0, 0)
        print(res)

        res = npf.pmt(Decimal('0.08') / Decimal('12'), 5 * 12, 15000, 0, 'end')
        print(res)

        res = npf.pmt(Decimal('0.08') / Decimal('12'), 5 * 12, 15000, 0, 1)
        print(res)

        res = npf.pmt(Decimal('0.08') / Decimal('12'), 5 * 12, 15000, 0,'begin')
        print(res)

#endregion

#region nper 
    def test_nper_broadcast_DOUBLE(self):

        res = npf.nper(0.075, -2000, 0, 100000., [0, 1])
        print(res)

    def test_nper_basic_values_DOUBLE(self):

        res = npf.nper([0, 0.075], -2000, 0, 100000)
        print(res)
  
    def test_nper_gh_18_DOUBLE(self):

        res = npf.nper(0.1, 0, -500, 1500)
        print(res)

    def test_nper_infinite_payments_DOUBLE(self):

        res = npf.nper(0, -0.0, 1000)
        print(res)

    def test_nper_no_interest_DOUBLE(self):

        res = npf.nper(0, -100, 1000)
        print(res)
  
#endregion

#region ipmt

    def test_ipmt_DOUBLE(self):

        res = npf.ipmt(0.1 / 12, 1, 24, 2000)
        print(res)

    def test_ipmt_DECIMAL(self):

        res = npf.ipmt(Decimal('0.1') / Decimal('12'), 1, 24, 2000)
        print(res)  
        
    def test_ipmt_when_is_begin_DOUBLE(self):

        res = npf.ipmt(0.1 / 12, 1, 24, 2000, 0, 'begin')
        print(res) 

        res = npf.ipmt(0.1 / 12, 1, 24, 2000, 0, 1)
        print(res) 

    def test_ipmt_when_is_end_DOUBLE(self):

        res = npf.ipmt(0.1 / 12, 1, 24, 2000, 0, 'end')
        print(res)  
        
        res = npf.ipmt(0.1 / 12, 1, 24, 2000, 0, 0)
        print(res)         

    def test_ipmt_gh_17_DOUBLE(self):

        rate = 0.001988079518355057

        res = npf.ipmt(rate, 0, 360, 300000, when="begin")
        print(res)  

        res = npf.ipmt(rate, 1, 360, 300000, when="begin")
        print(res)  

        res = npf.ipmt(rate, 2, 360, 300000, when="begin")
        print(res)  

        res = npf.ipmt(rate, 3, 360, 300000, when="begin")
        print(res)  
 
    def test_ipmt_broadcasting_DOUBLE(self):

        res = npf.ipmt(0.1 / 12, np.arange(5), 24, 2000)
        print(res)  

#endregion

#region ppmt
    
    def test_ppmt_DOUBLE(self):

        res = npf.ppmt(0.1 / 12, 1, 60, 55000)
        print(res)

    def test_ppmt_begin_DOUBLE(self):

        res = npf.ppmt(0.1 / 12, 1, 60, 55000, 0, 1)
        print(res)

        res = npf.ppmt(0.1 / 12, 1, 60, 55000, 0, 'begin')
        print(res)

    def test_ppmt_end_DOUBLE(self):

        res = npf.ppmt(0.1 / 12, 1, 60, 55000, 0, 0)
        print(res)

        res = npf.ppmt(0.1 / 12, 1, 60, 55000, 0, 'end')
        print(res)

    def test_ppmt_invalid_per_DOUBLE(self):

        res = npf.ppmt(0.1 / 12, 0, 60, 15000)
        print(res)
 

    def test_ppmt_broadcast_DOUBLE(self):

        res = npf.ppmt(0.1 / 12, np.arange(1,5), 24, 2000, 0)
        print(res)

        res = npf.ppmt(0.1 / 12, np.arange(1,5), 24, 2000, 0, 'end')
        print(res)

        res = npf.ppmt(0.1 / 12, np.arange(1,5), 24, 2000, 0,'begin')
        print(res)
#endregion

#region pv

    def test_pv_DOUBLE(self):
        res = npf.pv(0.07, 20, 12000)
        print(res)

        res = npf.pv(0.07, 20, 12000, 0)
        print(res)

        res = npf.pv(0.07, 20, 12000, 222220)
        print(res)

    def test_pv_begin_DOUBLE(self):
        res = npf.pv(0.07, 20, 12000, 0, 1)
        print(res)

        res = npf.pv(0.07, 20, 12000, 0, 'begin')
        print(res)

    def test_pv_end_DOUBLE(self):
        res = npf.pv(0.07, 20, 12000, 0, 0)
        print(res)

        res = npf.pv(0.07, 20, 12000, 0, 'end')
        print(res)     
#endregion

#region rate

    def test_rate_DOUBLE(self):

        res = npf.rate(10, 0, -3500, 10000)
        print(res)

    def test_rate_begin_DOUBLE(self):

        res = npf.rate(10, 0, -3500, 10000, 1)
        print(res)

        res = npf.rate(10, 0, -3500, 10000, 'begin')
        print(res)

    def test_rate_end_DOUBLE(self):

        res = npf.rate(10, 0, -3500, 10000, 0)
        print(res)

        res = npf.rate(10, 0, -3500, 10000, 'end')
        print(res)

    def test_rate_infeasable_solution_DOUBLE(self):

        res = npf.rate(12.0,400.0,10000.0,5000.0, when=0)
        print(res)

        res = npf.rate(12.0,400.0,10000.0,5000.0, when=1)
        print(res)

        res = npf.rate(12.0,400.0,10000.0,5000.0, when='end')
        print(res)

        res = npf.rate(12.0,400.0,10000.0,5000.0, when='begin')
        print(res)
 

#endregion
if __name__ == '__main__':
    unittest.main()
