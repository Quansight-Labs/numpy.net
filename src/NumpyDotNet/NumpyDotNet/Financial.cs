/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018-2021
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

using NumpyLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{
    public static class npf
    {
        private static object _convert_when(object when)
        {
            if (when is ndarray)
                return when;

            if (when is string)
            {
                string _swhen = when as string;
                if (_swhen == "begin")
                    return 1;
                if (_swhen == "end")
                    return 0;

                throw new Exception(string.Format("only 'begin' and 'end' are supported as when parameters"));
            }

            if (when.GetType().IsArray)
            {
                try
                {
                    System.Array whenarray = when as System.Array;

                    List<int> whenvals = new List<int>();
                    foreach (var _when in whenarray)
                    {
                        whenvals.Add(Convert.ToInt32(_convert_when(_when)));
                    }

                    return whenvals.ToArray();
                }
                catch (Exception ex)
                {
                    throw new Exception("unrecognized when parameter.  Must be integers (0,1) or 'begin' or 'end'");
                }

            }

            try
            {
                Int32 iValue = Convert.ToInt32(when);
                if (iValue != 0 && iValue != 1)
                {
                    throw new Exception("unrecognized when parameter.  Must be integers (0,1) or 'begin' or 'end'");
                }
                return iValue;
            }
            catch (Exception ex)
            {
                throw new Exception("unrecognized when parameter.  Must be integers (0,1) or 'begin' or 'end'");
            }

        }

        #region fv
        /*
        Compute the future value.

        Given:
         * a present value, `pv`
         * an interest `rate` compounded once per period, of which
           there are
         * `nper` total
         * a (fixed) payment, `pmt`, paid either
         * at the beginning(`when` = {'begin', 1}) or the end
           (`when` = { 'end', 0}) of each period

        Return:
           the value at the end of the `nper` periods

        Parameters
        ----------
        rate : scalar or array_like of shape(M, )
            Rate of interest as decimal (not per cent) per period
        nper : scalar or array_like of shape(M, )
            Number of compounding periods
        pmt : scalar or array_like of shape(M, )
            Payment
        pv : scalar or array_like of shape(M, )
            Present value
        when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
        When payments are due('begin' (1) or 'end' (0)).
            Defaults to {'end', 0}.

        Returns
        -------
        out : ndarray
            Future values.If all input is scalar, returns a scalar float.  If
          any input is array_like, returns future values for each input element.
          If multiple inputs are array_like, they all must have the same shape.

      Notes
        -----
        The future value is computed by solving the equation::

         fv +
         pv* (1+rate)** nper +
         pmt* (1 + rate* when)/rate* ((1 + rate)** nper - 1) == 0

        or, when ``rate == 0``::

         fv + pv + pmt* nper == 0

        References
        ----------
        .. [WRW] Wheeler, D.A., E.Rathke, and R.Weir(Eds.) (2009, May).
           Open Document Format for Office Applications(OpenDocument)v1.2,
           Part 2: Recalculated Formula(OpenFormula) Format - Annotated Version,
          Pre-Draft 12. Organization for the Advancement of Structured Information
          Standards(OASIS). Billerica, MA, USA. [ODT Document].
           Available:
           http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula
           OpenDocument-formula-20090508.odt

        Examples
        --------
        >>> import numpy as np
        >>> import numpy_financial as npf

        What is the future value after 10 years of saving $100 now, with
        an additional monthly savings of $100.  Assume the interest rate is
        5% (annually) compounded monthly?

        >>> npf.fv(0.05/12, 10*12, -100, -100)
        15692.928894335748

        By convention, the negative sign represents cash flow out (i.e.money not
        available today).  Thus, saving $100 a month at 5% annual interest leads
        to $15,692.93 available to spend in 10 years.

        If any input is array_like, returns an array of equal shape.Let's
        compare different interest rates from the example above.

        >>> a = np.array((0.05, 0.06, 0.07)) / 12
        >>> npf.fv(a, 10 * 12, -100, -100)
        array([15692.92889434, 16569.87435405, 17509.44688102]) # may vary

        */

        /// <summary>
        /// 
        /// </summary>
        /// <param name="rate">Rate of interest as decimal (not per cent) per period</param>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pmt">Payment</param>
        /// <param name="pv">Present value</param>
        /// <param name="when">When payments are due</param>
        /// <returns></returns>
        public static ndarray fv(object rate, object nper, object pmt, object pv, object when)
        {
            when = _convert_when(when);

            List<ndarray> inputArrays = new List<ndarray>();
            inputArrays.Add(np.asanyarray(rate));
            inputArrays.Add(np.asanyarray(nper));
            inputArrays.Add(np.asanyarray(pmt));
            inputArrays.Add(np.asanyarray(pv));
            inputArrays.Add(np.asanyarray(when));

            var outputArrays = np.broadcast_arrays(true, inputArrays.ToArray());
            if (outputArrays.Count() != 5)
            {
                throw new Exception("broadcast_arrays did not produced expected result");
            }

            ndarray _rate = outputArrays.ElementAt(0);
            ndarray _nper = outputArrays.ElementAt(1);
            ndarray _pmt = outputArrays.ElementAt(2);
            ndarray _pv = outputArrays.ElementAt(3);
            ndarray _when = outputArrays.ElementAt(4);

            ndarray fv_array = np.empty_like(_rate);
            ndarray zero = _rate == 0;
            ndarray nonzero = ~zero;

            fv_array[zero] = -(_pv.A(zero) + _pmt.A(zero) * _nper.A(zero));

            ndarray rate_nonzero = _rate.A(nonzero);
            ndarray temp = np.power((1 + rate_nonzero), _nper.A(nonzero));

            fv_array[nonzero] =
                (-_pv.A(nonzero) * temp
                - _pmt.A(nonzero) * (1 + rate_nonzero * _when.A(nonzero)) / rate_nonzero
                * (temp - 1)
                );

            return fv_array;
        }
        #endregion

        #region pmt

        /*
        Compute the payment against loan principal plus interest.

        Given:
         * a present value, `pv` (e.g., an amount borrowed)
         * a future value, `fv` (e.g., 0)
         * an interest `rate` compounded once per period, of which
           there are
         * `nper` total
         * and(optional) specification of whether payment is made
          at the beginning(`when` = { 'begin', 1}) or the end
        (`when` = { 'end', 0}) of each period

        Return:
           the(fixed) periodic payment.

       Parameters
       ----------

       rate : array_like

           Rate of interest(per period)

       nper : array_like

           Number of compounding periods

       pv : array_like

           Present value

       fv : array_like, optional

           Future value(default = 0)

       when : {{'begin', 1}, {'end', 0}}, {string, int}
            When payments are due('begin' (1) or 'end' (0))

        Returns
        -------
        out : ndarray
            Payment against loan plus interest.If all input is scalar, returns a
            scalar float.  If any input is array_like, returns payment for each
            input element.If multiple inputs are array_like, they all must have
            the same shape.

        Notes
        -----
        The payment is computed by solving the equation::

         fv +
         pv*(1 + rate)** nper +
         pmt* (1 + rate* when)/rate* ((1 + rate)** nper - 1) == 0

        or, when ``rate == 0``::

          fv + pv + pmt* nper == 0

        for ``pmt``.

        Note that computing a monthly mortgage payment is only
        one use for this function.For example, pmt returns the
        periodic deposit one must make to achieve a specified
        future balance given an initial deposit, a fixed,
        periodically compounded interest rate, and the total
        number of periods.

        References
        ----------
        .. [WRW] Wheeler, D.A., E.Rathke, and R. Weir (Eds.) (2009, May).
           Open Document Format for Office Applications (OpenDocument)v1.2,
           Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,
           Pre-Draft 12. Organization for the Advancement of Structured Information
           Standards (OASIS). Billerica, MA, USA. [ODT Document].
           Available:
           http://www.oasis-open.org/committees/documents.php
           ?wg_abbrev=office-formulaOpenDocument-formula-20090508.odt

        Examples
        --------
        >>> import numpy_financial as npf

        What is the monthly payment needed to pay off a $200,000 loan in 15
        years at an annual interest rate of 7.5%?

        >>> npf.pmt(0.075/12, 12*15, 200000)
        -1854.0247200054619

        In order to pay-off (i.e., have a future-value of 0) the $200,000 obtained
        today, a monthly payment of $1,854.02 would be required.  Note that this
        example illustrates usage of `fv` having a default value of 0.

        */

        /// <summary>
        /// Compute the payment against loan principal plus interest.
        /// </summary>
        /// <param name="rate">Rate of interest (per period)</param>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">uture value (default = 0)</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <returns></returns>
        public static ndarray pmt(object rate, object nper, object pv)
        {
            return _pmt(rate, nper, pv, 0, "end");
        }
        /// <summary>
        /// Compute the payment against loan principal plus interest.
        /// </summary>
        /// <param name="rate">Rate of interest (per period)</param>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">uture value (default = 0)</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <returns></returns>
        public static ndarray pmt(object rate, object nper, object pv, object fv)
        {
            return _pmt(rate, nper, pv, fv, "end");
        }
        /// <summary>
        /// Compute the payment against loan principal plus interest.
        /// </summary>
        /// <param name="rate">Rate of interest (per period)</param>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">uture value (default = 0)</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <returns></returns>
        public static ndarray pmt(object rate, object nper, object pv, object fv, object when)
        {
            return _pmt(rate, nper, pv, fv, when);
        }

        private static ndarray _pmt(object rate, object nper, object pv, object fv, object when)
        {
            when = _convert_when(when);

            ndarray _rate = np.asanyarray(rate);
            ndarray _nper = np.asanyarray(nper);
            ndarray _pv = np.asanyarray(pv);
            ndarray _fv = np.asanyarray(fv);
            ndarray _when = np.asanyarray(when);

            dtype retType = np.Float64;
            if (_rate.TypeNum == NPY_TYPES.NPY_DECIMAL)
            {
                retType = np.Decimal;
            }
            if (_rate.TypeNum == NPY_TYPES.NPY_COMPLEX)
            {
                retType = np.Complex;
            }

            ndarray temp = np.power((1 + _rate), _nper);
            ndarray mask = (_rate == 0);
            ndarray masked_rate = (ndarray)np.where(mask, np.asanyarray(1).astype(retType), _rate);
            ndarray fact = (ndarray)np.where(mask != 0, _nper.astype(retType),
                 (1 + masked_rate * _when) * (temp - 1) / masked_rate);
            return (-(_fv + _pv * temp) / fact) as ndarray;
        }

        #endregion

        #region nper

        /*
        Compute the number of periodic payments.

        :class:`decimal.Decimal` type is not supported.

        Parameters
        ----------
        rate : array_like
            Rate of interest (per period)
        pmt : array_like
            Payment
        pv : array_like
            Present value
        fv : array_like, optional
            Future value
        when : { { 'begin', 1}, { 'end', 0} }, {string, int}, optional
            When payments are due('begin' (1) or 'end' (0))

        Notes
        -----
        The number of periods ``nper`` is computed by solving the equation::

         fv + pv* (1+rate)** nper + pmt* (1+rate* when)/rate* ((1+rate)** nper-1) = 0

        but if ``rate = 0`` then::

         fv + pv + pmt* nper = 0

        Examples
        --------
        >>> import numpy as np
        >>> import numpy_financial as npf

        If you only had $150/month to pay towards the loan, how long would it take
        to pay-off a loan of $8,000 at 7% annual interest?

        >>> print(np.round(npf.nper(0.07/12, -150, 8000), 5))
        64.07335

        So, over 64 months would be required to pay off the loan.

        The same analysis could be done with several different interest rates
        and/or payments and/or total amounts to produce an entire table.

        >>> npf.nper(*(np.ogrid[0.07 / 12: 0.08 / 12: 0.01 / 12,
        ...                     -150   : -99    : 50,
        ...                     8000   : 9001   : 1000]))
        array([[[ 64.07334877,  74.06368256],
                [108.07548412, 127.99022654]],
               [[ 66.12443902,  76.87897353],
                [114.70165583, 137.90124779]]])

        */

        /// <summary>
        /// Compute the number of periodic payments.
        /// </summary>
        /// <param name="rate">Rate of interest (per period)</param>
        /// <param name="pmt">Payment</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due('begin' (1) or 'end' (0))</param>
        /// <returns></returns>
        public static ndarray nper(object rate, object pmt, object pv)
        {
            return _nper(rate, pmt, pv, 0, "end");
        }
        /// <summary>
        /// Compute the number of periodic payments.
        /// </summary>
        /// <param name="rate">Rate of interest (per period)</param>
        /// <param name="pmt">Payment</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due('begin' (1) or 'end' (0))</param>
        /// <returns></returns>
        public static ndarray nper(object rate, object pmt, object pv, object fv)
        {
            return _nper(rate, pmt, pv, fv, "end");
        }

        /// <summary>
        /// Compute the number of periodic payments.
        /// </summary>
        /// <param name="rate">Rate of interest (per period)</param>
        /// <param name="pmt">Payment</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due('begin' (1) or 'end' (0))</param>
        /// <returns></returns>
        public static ndarray nper(object rate, object pmt, object pv, object fv, object when)
        {
            return _nper(rate, pmt, pv, fv, when);
        }

        private static ndarray _nper(object rate, object pmt, object pv, object fv, object when)
        {
            when = _convert_when(when);

            List<ndarray> inputArrays = new List<ndarray>();
            inputArrays.Add(np.asanyarray(rate));
            inputArrays.Add(np.asanyarray(pmt));
            inputArrays.Add(np.asanyarray(pv));
            inputArrays.Add(np.asanyarray(fv));
            inputArrays.Add(np.asanyarray(when));

            var outputArrays = np.broadcast_arrays(true, inputArrays.ToArray());
            if (outputArrays.Count() != 5)
            {
                throw new Exception("broadcast_arrays did not produced expected result");
            }

            ndarray _rate = outputArrays.ElementAt(0);
            ndarray _pmt = outputArrays.ElementAt(1);
            ndarray _pv = outputArrays.ElementAt(2);
            ndarray _fv = outputArrays.ElementAt(3);
            ndarray _when = outputArrays.ElementAt(4);

            dtype retType = np.Float64;
            if (_rate.TypeNum == NPY_TYPES.NPY_DECIMAL)
                retType = np.Decimal;

            var nper_array = np.empty_like(_rate, dtype : retType);

            var zero = _rate == 0;
            var nonzero = ~zero;

            try
            {
                nper_array[zero] = -(_fv.A(zero) + _pv.A(zero)) / _pmt.A(zero);
            }
            catch (DivideByZeroException dbz)
            {
                Console.WriteLine("");
            }

            var nonzero_rate = _rate.A(nonzero);
            var z = _pmt[nonzero] * (1 + nonzero_rate * _when.A(nonzero)) / nonzero_rate;

            nper_array[nonzero] = (np.log((-_fv.A(nonzero) + z) / (_pv.A(nonzero) + z)) / np.log(1 + nonzero_rate));

            return nper_array;
        }

        #endregion

        #region ipmt

        /*
        Compute the interest portion of a payment.

        Parameters
        ----------
        rate : scalar or array_like of shape(M, )
            Rate of interest as decimal (not per cent) per period
        per : scalar or array_like of shape(M, )
            Interest paid against the loan changes during the life or the loan.
            The `per` is the payment period to calculate the interest amount.
        nper : scalar or array_like of shape(M, )
            Number of compounding periods
        pv : scalar or array_like of shape(M, )
            Present value
        fv : scalar or array_like of shape(M, ), optional
            Future value
        when : { { 'begin', 1}, { 'end', 0} }, {string, int }, optional
            When payments are due('begin' (1) or 'end' (0)).
            Defaults to {'end', 0}.

        Returns
        -------
        out : ndarray
            Interest portion of payment.If all input is scalar, returns a scalar
            float.  If any input is array_like, returns interest payment for each
            input element.If multiple inputs are array_like, they all must have
            the same shape.

        See Also
        --------
        ppmt, pmt, pv

        Notes
        -----
        The total payment is made up of payment against principal plus interest.

        ``pmt = ppmt + ipmt``

        Examples
        --------
        >>> import numpy as np
        >>> import numpy_financial as npf

        What is the amortization schedule for a 1 year loan of $2500 at
        8.24% interest per year compounded monthly?

        >>> principal = 2500.00

        The 'per' variable represents the periods of the loan.Remember that
        financial equations start the period count at 1!

        >>> per = np.arange(1 * 12) + 1
        >>> ipmt = npf.ipmt(0.0824 / 12, per, 1 * 12, principal)
        >>> ppmt = npf.ppmt(0.0824 / 12, per, 1 * 12, principal)

        Each element of the sum of the 'ipmt' and 'ppmt' arrays should equal
        'pmt'.

        >>> pmt = npf.pmt(0.0824 / 12, 1 * 12, principal)
        >>> np.allclose(ipmt + ppmt, pmt)
        True

        >>> fmt = '{0:2d} {1:8.2f} {2:8.2f} {3:8.2f}'
        >>> for payment in per:
        ...     index = payment - 1
        ...principal = principal + ppmt[index]
        ...print(fmt.format(payment, ppmt[index], ipmt[index], principal))
         1  -200.58   -17.17  2299.42
         2  -201.96   -15.79  2097.46
         3  -203.35   -14.40  1894.11
         4  -204.74   -13.01  1689.37
         5  -206.15   -11.60  1483.22
         6  -207.56   -10.18  1275.66
         7  -208.99    -8.76  1066.67
         8  -210.42    -7.32   856.25
         9  -211.87    -5.88   644.38
        10  -213.32    -4.42   431.05
        11  -214.79    -2.96   216.26
        12  -216.26    -1.49    -0.00

        >>> interestpd = np.sum(ipmt)
        >>> np.round(interestpd, 2)
        - 112.98

        */

        /// <summary>
        /// Compute the interest portion of a payment.
        /// </summary>
        /// <param name="rate">Rate of interest as decimal (not per cent) per period</param>
        /// <param name="per">Interest paid against the loan changes during the life or the loan. The 'per' is the payment period to calculate the interest amount.</param>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <returns></returns>
        public static ndarray ipmt(object rate, object per, object nper, object pv)
        {
            return _ipmt(rate, per, nper, pv, 0, "end");
        }
        /// <summary>
        /// Compute the interest portion of a payment.
        /// </summary>
        /// <param name="rate">Rate of interest as decimal (not per cent) per period</param>
        /// <param name="per">Interest paid against the loan changes during the life or the loan. The 'per' is the payment period to calculate the interest amount.</param>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <returns></returns>
        public static ndarray ipmt(object rate, object per, object nper, object pv, object fv)
        {
            return _ipmt(rate, per, nper, pv, fv, "end");
        }

        /// <summary>
        /// Compute the interest portion of a payment.
        /// </summary>
        /// <param name="rate">Rate of interest as decimal (not per cent) per period</param>
        /// <param name="per">Interest paid against the loan changes during the life or the loan. The 'per' is the payment period to calculate the interest amount.</param>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <returns></returns>
        public static ndarray ipmt(object rate, object per, object nper, object pv, object fv, object when)
        {
            return _ipmt(rate, per, nper, pv, fv, when);
        }


        private static object _value_like(ndarray arr, object value)
        {
            if (arr.TypeNum == NPY_TYPES.NPY_DECIMAL)
            {
                try
                {
                    return Convert.ToDecimal(value);
                }
                catch
                {
                    return 0m;
                }

            }

            var temp = np.array(value, dtype: arr.Dtype);
            return temp.GetItem(0);
        }
        private static ndarray _ipmt(object rate, object per, object nper, object pv, object fv, object when)
        {
            when = _convert_when(when);

            List<ndarray> inputArrays = new List<ndarray>();
            inputArrays.Add(np.asanyarray(rate));
            inputArrays.Add(np.asanyarray(per));
            inputArrays.Add(np.asanyarray(nper));
            inputArrays.Add(np.asanyarray(pv));
            inputArrays.Add(np.asanyarray(fv));
            inputArrays.Add(np.asanyarray(when));

            var outputArrays = np.broadcast_arrays(true, inputArrays.ToArray());
            if (outputArrays.Count() != 6)
            {
                throw new Exception("broadcast_arrays did not produced expected result");
            }

            ndarray _rate = outputArrays.ElementAt(0);
            ndarray _per = outputArrays.ElementAt(1);
            ndarray _nper = outputArrays.ElementAt(2);
            ndarray _pv = outputArrays.ElementAt(3);
            ndarray _fv = outputArrays.ElementAt(4);
            ndarray _when = outputArrays.ElementAt(5);

            var total_pmt = pmt(rate, nper, pv, fv, when);
            var ipmt_array = np.array(_rbl(_rate, _per, total_pmt, _pv, _when) * _rate);

            // Payments start at the first period, so payments before that
            // don't make any sense.
            ipmt_array[_per < 1] = _value_like(ipmt_array, np.NaN);
            //If payments occur at the beginning of a period and this is the
            //first period, then no interest has accrued.
            var per1_and_begin = (_when == 1) & (_per == 1);
            ipmt_array[per1_and_begin] = _value_like(ipmt_array, 0);
            // If paying at the beginning we need to discount by one period.
            var per_gt_1_and_begin = (_when == 1) & (_per > 1);
            ipmt_array[per_gt_1_and_begin] = (ipmt_array.A(per_gt_1_and_begin) / (1 + _rate.A(per_gt_1_and_begin)));

            return ipmt_array;
        }
    

        //This function is here to simply have a different name for the 'fv'
        //function to not interfere with the 'fv' keyword argument within the 'ipmt'
        //function.It is the 'remaining balance on loan' which might be useful as
        //it's own function, but is easily calculated with the 'fv' function.
        private static ndarray _rbl(ndarray rate, ndarray per, ndarray pmt, ndarray pv, ndarray when)
        {
            return fv(rate, (per - 1), pmt, pv, when);
        }

        #endregion
    }
}
