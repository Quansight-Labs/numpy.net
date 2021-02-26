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

#region irr

    def test_irr_basic_values(self):

        cashflows = np.array([-150000, 15000, 25000, 35000, 45000, 60000])
        res = npf.irr(cashflows);
        print(res)

        cashflows = np.array([-100, 0, 0, 74])
        res = npf.irr(cashflows);
        print(res)

        cashflows = np.array([-100, 39, 59, 55, 20])
        res = npf.irr(cashflows);
        print(res)

        cashflows = np.array([-100, 100, 0, -7])
        res = npf.irr(cashflows);
        print(res)

        cashflows = np.array([-100, 100, 0, 7])
        res = npf.irr(cashflows);
        print(res)

        cashflows = np.array([-5, 10.5, 1, -8, 1])
        res = npf.irr(cashflows);
        print(res)

    def test_irr_trailing_zeros(self):

        cashflows = np.array([-5, 10.5, 1, -8, 1, 0, 0, 0])
        res = npf.irr(cashflows);
        print(res)

    def test_irr_gh_6744(self):

        cashflows = np.array([-1, -2, -3])
        res = npf.irr(cashflows);
        print(res)

    def test_irr_gh_15(self):

        v = [
            -3000.0,
            2.3926932267015667e-07,
            4.1672087103345505e-16,
            5.3965110036378706e-25,
            5.1962551071806174e-34,
            3.7202955645436402e-43,
            1.9804961711632469e-52,
            7.8393517651814181e-62,
            2.3072565113911438e-71,
            5.0491839233308912e-81,
            8.2159177668499263e-91,
            9.9403244366963527e-101,
            8.942410813633967e-111,
            5.9816122646481191e-121,
            2.9750309031844241e-131,
            1.1002067043497954e-141,
            3.0252876563518021e-152,
            6.1854121948207909e-163,
            9.4032980015353301e-174,
            1.0629218520017728e-184,
            8.9337141847171845e-196,
            5.5830607698467935e-207,
            2.5943122036622652e-218,
            8.9635842466507006e-230,
            2.3027710094332358e-241,
            4.3987510596745562e-253,
            6.2476630372575209e-265,
            6.598046841695288e-277,
            5.1811095266842017e-289,
            3.0250999925830644e-301,
            1.3133070599585015e-313,
        ]

        res = npf.irr(v);
        print(res)

#endregion

if __name__ == '__main__':
    unittest.main()
