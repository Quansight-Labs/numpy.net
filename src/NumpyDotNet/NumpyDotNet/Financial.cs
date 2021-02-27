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

        #region ppmt

        /*
        Compute the payment against loan principal.

        Parameters
        ----------
        rate : array_like
            Rate of interest (per period)
        per : array_like, int
            Amount paid against the loan changes.The `per` is the period of
            interest.
        nper : array_like
            Number of compounding periods
        pv : array_like
            Present value
        fv : array_like, optional
            Future value
        when : { { 'begin', 1}, { 'end', 0} }, {string, int }
            When payments are due('begin' (1) or 'end' (0)) 

        See Also
        --------
        pmt, pv, ipmt
        */

        /// <summary>
        /// Compute the payment against loan principal.
        /// </summary>
        /// <param name="rate">Rate of interest (per period)</param>
        /// <param name="per">Amount paid against the loan changes.  The `per` is the period of interest.</param>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <returns></returns>       
        public static ndarray ppmt(object rate, object per, object nper, object pv)
        {
            return _ppmt(rate, per, nper, pv, 0, "end");
        }

         /// <summary>
        /// Compute the payment against loan principal.
        /// </summary>
        /// <param name="rate">Rate of interest (per period)</param>
        /// <param name="per">Amount paid against the loan changes.  The `per` is the period of interest.</param>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <returns></returns>      
        public static ndarray ppmt(object rate, object per, object nper, object pv, object fv)
        {
            return _ppmt(rate, per, nper, pv, fv, "end");
        }

        /// <summary>
        /// Compute the payment against loan principal.
        /// </summary>
        /// <param name="rate">Rate of interest (per period)</param>
        /// <param name="per">Amount paid against the loan changes.  The `per` is the period of interest.</param>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <returns></returns>
        public static ndarray ppmt(object rate, object per, object nper, object pv, object fv, object when)
        {
            return _ppmt(rate, per, nper, pv, fv, when);
        }

        private static ndarray _ppmt(object rate, object per, object nper, object pv, object fv, object when)
        {
            var total = npf.pmt(rate, nper, pv, fv, when);
            return total - npf.ipmt(rate, per, nper, pv, fv, when);
        }

        #endregion

        #region pv

        /*
        Compute the present value.

        Given:
         * a future value, `fv`
         * an interest `rate` compounded once per period, of which
           there are
         * `nper` total
         * a (fixed) payment, `pmt`, paid either
         * at the beginning(`when` = {'begin', 1}) or the end
            (`when` = { 'end', 0}) of each period

        Return:
           the value now

        Parameters
        ----------
        rate : array_like
            Rate of interest(per period)
        nper : array_like
            Number of compounding periods
        pmt : array_like
            Payment
        fv : array_like, optional
            Future value
        when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
            When payments are due('begin' (1) or 'end' (0))

        Returns
        -------
        out : ndarray, float
            Present value of a series of payments or investments.

        Notes
        -----
        The present value is computed by solving the equation::

         fv +
         pv*(1 + rate)** nper +
         pmt* (1 + rate* when)/rate* ((1 + rate)** nper - 1) = 0

        or, when ``rate = 0``::

         fv + pv + pmt* nper = 0

        for `pv`, which is then returned.

        References
        ----------
        .. [WRW] Wheeler, D.A., E.Rathke, and R. Weir (Eds.) (2009, May).
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

        What is the present value(e.g., the initial investment)
        of an investment that needs to total $15692.93
        after 10 years of saving $100 every month? Assume the
        interest rate is 5% (annually) compounded monthly.

        >>> npf.pv(0.05/12, 10*12, -100, 15692.93)
        -100.00067131625819

        By convention, the negative sign represents cash flow out
        (i.e., money not available today).  Thus, to end up with
        $15,692.93 in 10 years saving $100 a month at 5% annual
        interest, one's initial deposit should also be $100.

        If any input is array_like, ``pv`` returns an array of equal shape.
        Let's compare different interest rates in the example above:

        >>> a = np.array((0.05, 0.04, 0.03)) / 12
        >>> npf.pv(a, 10 * 12, -100, 15692.93)
        array([-100.00067132, -649.26771385, -1273.78633713]) # may vary

        So, to end up with the same $15692.93 under the same $100 per month
        "savings plan," for annual interest rates of 4% and 3%, one would
        need initial investments of $649.27 and $1273.79, respectively.

        */

        /// <summary>
        /// Compute the present value.
        /// </summary>
        /// <param name="rate">Rate of interest (per period)</param>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pmt">Payment</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <returns></returns>
        public static ndarray pv(object rate, object nper, object pmt)
        {
            return _pv(rate, nper, pmt, 0, "end");
        }

        /// <summary>
        /// Compute the present value.
        /// </summary>
        /// <param name="rate">Rate of interest (per period)</param>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pmt">Payment</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <returns></returns>
        public static ndarray pv(object rate, object nper, object pmt, object fv)
        {
            return _pv(rate, nper, pmt, fv, "end");
        }

        /// <summary>
        /// Compute the present value.
        /// </summary>
        /// <param name="rate">Rate of interest (per period)</param>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pmt">Payment</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <returns></returns>
        public static ndarray pv(object rate, object nper, object pmt, object fv, object when)
        {
            return _pv(rate, nper, pmt, fv, when);
        }

        private static ndarray _pv(object rate, object nper, object pmt, object fv, object when)
        {
            when = _convert_when(when);

            ndarray _rate = np.asanyarray(rate);
            ndarray _nper = np.asanyarray(nper);
            ndarray _pmt = np.asanyarray(pmt);
            ndarray _fv = np.asanyarray(fv);
            ndarray _when = np.asanyarray(when);

            if (_rate.TypeNum == NPY_TYPES.NPY_DECIMAL)
            {
                _nper = _nper.astype(np.Decimal);
            }
            else
            {
                _nper = _nper.astype(np.Float64);
            }
 

            var temp = np.power((1 + _rate),_nper);
            var fact = np.where(_rate == 0, _nper, (1 + _rate * _when) * (temp - 1) / _rate) as ndarray;

            var result = -(_fv + _pmt * fact) / temp;
            return (ndarray)result;


        }
        #endregion

        #region rate

        /*
        Compute the rate of interest per period.

        Parameters
        ----------
        nper : array_like
            Number of compounding periods
        pmt : array_like
            Payment
        pv : array_like
            Present value
        fv : array_like
            Future value
        when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
            When payments are due('begin' (1) or 'end' (0))
        guess : Number, optional
            Starting guess for solving the rate of interest, default 0.1
        tol : Number, optional
            Required tolerance for the solution, default 1e-6
        maxiter : int, optional
            Maximum iterations in finding the solution

        Notes
        -----
        The rate of interest is computed by iteratively solving the
        (non-linear) equation::

         fv + pv* (1+rate)** nper + pmt* (1+rate* when)/rate* ((1+rate)** nper - 1) = 0

        for ``rate``.

        References
        ----------
        Wheeler, D.A., E.Rathke, and R.Weir(Eds.) (2009, May). Open Document
        Format for Office Applications(OpenDocument)v1.2, Part 2: Recalculated
       Formula(OpenFormula) Format - Annotated Version, Pre-Draft 12.
        Organization for the Advancement of Structured Information Standards
        (OASIS). Billerica, MA, USA. [ODT Document]. Available:
        http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula
        OpenDocument-formula-20090508.odt

        */


        private static double _g_div_gp(double? r, ndarray n, ndarray p, ndarray x, ndarray y, ndarray w)
        {
            // Evaluate g(r_n)/g'(r_n), where g =
            // fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1)

            var t1 = np.power(r + 1,n);
            var t2 = np.power(r + 1, n - 1);
            var g = y + t1 * x + p * (t1 - 1) * (r * w + 1) / r;
            var gp = (n * t2 * x
                 - p * (t1 - 1) * (r * w + 1) / (np.power(r,2))
                 + n * p * t2 * (r * w + 1) / r
                 + p * (t1 - 1) * w / r);

            ndarray t3 = (ndarray)(g / gp);
            return (double)t3;
        }

        private static decimal _g_div_gp(decimal? r, ndarray n, ndarray p, ndarray x, ndarray y, ndarray w)
        {
            // Evaluate g(r_n)/g'(r_n), where g =
            // fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1)

            var t1 = np.power(r + 1, n);
            var t2 = np.power(r + 1, n - 1);
            var g = y + t1 * x + p * (t1 - 1) * (r * w + 1) / r;
            var gp = (n * t2 * x
                 - p * (t1 - 1) * (r * w + 1) / (np.power(r, 2))
                 + n * p * t2 * (r * w + 1) / r
                 + p * (t1 - 1) * w / r);

            ndarray t3 = (ndarray)(g / gp);
            return (decimal)t3;
        }

        /// <summary>
        /// Compute the rate of interest per period.
        /// </summary>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pmt">Payment</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <param name="guess">Starting guess for solving the rate of interest, default 0.1</param>
        /// <param name="tol">Required tolerance for the solution, default 1e-6</param>
        /// <param name="maxiter">Maximum iterations in finding the solution</param>
        /// <returns></returns>
        public static ndarray rate(object nper, object pmt, object pv, object fv)
        {
            return _rate(nper, pmt, pv, fv, "end", null, null, 100);
        }
        /// <summary>
        /// Compute the rate of interest per period.
        /// </summary>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pmt">Payment</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <param name="guess">Starting guess for solving the rate of interest, default 0.1</param>
        /// <param name="tol">Required tolerance for the solution, default 1e-6</param>
        /// <param name="maxiter">Maximum iterations in finding the solution</param>
        /// <returns></returns>       
        public static ndarray rate(object nper, object pmt, object pv, object fv, object when)
        {
            return _rate(nper, pmt, pv, fv, when, null, null, 100);
        }
         /// <summary>
        /// Compute the rate of interest per period.
        /// </summary>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pmt">Payment</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <param name="guess">Starting guess for solving the rate of interest, default 0.1</param>
        /// <param name="tol">Required tolerance for the solution, default 1e-6</param>
        /// <param name="maxiter">Maximum iterations in finding the solution</param>
        /// <returns></returns>     
        public static ndarray rate(object nper, object pmt, object pv, object fv, object when, double guess)
        {
            return _rate(nper, pmt, pv, fv, when, guess, null, 100);
        }
        /// <summary>
        /// Compute the rate of interest per period.
        /// </summary>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pmt">Payment</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <param name="guess">Starting guess for solving the rate of interest, default 0.1</param>
        /// <param name="tol">Required tolerance for the solution, default 1e-6</param>
        /// <param name="maxiter">Maximum iterations in finding the solution</param>
        /// <returns></returns>      
        public static ndarray rate(object nper, object pmt, object pv, object fv, object when, double guess, double tol)
        {
            return _rate(nper, pmt, pv, fv, when, guess, tol, 100);
        }

        /// <summary>
        /// Compute the rate of interest per period.
        /// </summary>
        /// <param name="nper">Number of compounding periods</param>
        /// <param name="pmt">Payment</param>
        /// <param name="pv">Present value</param>
        /// <param name="fv">Future value</param>
        /// <param name="when">When payments are due ('begin' (1) or 'end' (0))</param>
        /// <param name="guess">Starting guess for solving the rate of interest, default 0.1</param>
        /// <param name="tol">Required tolerance for the solution, default 1e-6</param>
        /// <param name="maxiter">Maximum iterations in finding the solution</param>
        /// <returns></returns>
        public static ndarray rate(object nper, object pmt, object pv, object fv, object when, double guess, double tol, Int32 maxiter)
        {
            return _rate(nper, pmt, pv, fv, when, guess, tol, maxiter);
        }

        private static ndarray _rate(object nper, object pmt, object pv, object fv, object when, double? guess, double? tol, Int32 maxiter)
        {
            when = _convert_when(when);

            ndarray _nper = np.asanyarray(nper);
            ndarray _pmt = np.asanyarray(pmt);
            ndarray _pv = np.asanyarray(pv);
            ndarray _fv = np.asanyarray(fv);
            ndarray _when = np.asanyarray(when);

            if (_pmt.TypeNum == NPY_TYPES.NPY_DECIMAL)
            {
                decimal? dguess = null;
                if (guess.HasValue)
                    dguess = Convert.ToDecimal(guess);

                decimal? dtol = null;
                if (tol.HasValue)
                    dtol = Convert.ToDecimal(tol);

                if (dguess == null)
                {
                    dguess = 0.1m;
                }
                if (dtol == null)
                {
                    dtol = 1e-6m;
                }

                var rn = dguess;
                Int32 iterator = 0;
                bool close = false;

                while ((iterator < maxiter) && !close)
                {
                    var rnp1 = rn - npf._g_div_gp(rn, _nper, _pmt, _pv, _fv, _when);
                    var diff = Math.Abs(rnp1.Value - rn.Value);
                    close = np.allb(diff < dtol);
                    iterator += 1;
                    rn = rnp1;
                }
                if (!close)
                {
                    throw new Exception("Decimal numbers don't support NaN values");
                }
                else
                {
                    return np.array(rn, _pmt.Dtype);
                }
            }
            else
            {
                _pmt = _pmt.astype(np.Float64);

                if (guess == null)
                {
                    guess = 0.1;
                }
                if (tol == null)
                {
                    tol = 1e-6;
                }

                var rn = guess;
                Int32 iterator = 0;
                bool close = false;

                while ((iterator < maxiter) && !close)
                {
                    var rnp1 = rn - npf._g_div_gp(rn, _nper, _pmt, _pv, _fv, _when);
                    var diff = Math.Abs(rnp1.Value - rn.Value);
                    close = np.allb(diff < tol);
                    iterator += 1;
                    rn = rnp1;
                }
                if (!close)
                {
                    // Return nan's in array of the same shape as rn
                    return np.array(double.NaN + rn, _pmt.Dtype);
                }
                else
                {
                    return np.array(rn, _pmt.Dtype);
                }
            }


        }

        #endregion

        #region irr

        /*   
        Return the Internal Rate of Return(IRR).

        This is the "average" periodically compounded rate of return
        that gives a net present value of 0.0; for a more complete explanation,
        see Notes below.

        :class:`decimal.Decimal` type is not supported.

        Parameters
        ----------
        values : array_like, shape(N,)
            Input cash flows per time period.By convention, net "deposits"
            are negative and net "withdrawals" are positive.  Thus, for
            example, at least the first element of `values`, which represents
            the initial investment, will typically be negative.

        Returns
        -------
        out : float
            Internal Rate of Return for periodic input values.

        Notes
        -----
        The IRR is perhaps best understood through an example (illustrated
        using np.irr in the Examples section below).  Suppose one invests 100
        units and then makes the following withdrawals at regular(fixed)
        intervals: 39, 59, 55, 20.  Assuming the ending value is 0, one's 100
        unit investment yields 173 units; however, due to the combination of
        compounding and the periodic withdrawals, the "average" rate of return
        is neither simply 0.73/4 nor(1.73)^0.25-1.  Rather, it is the solution
       (for :math:`r`) of the equation:

        .. math:: -100 + \\frac{39}{1+r} + \\frac{59}{(1+r)^2}
         + \\frac{55}{(1+r)^3} + \\frac{20}{(1+r)^4} = 0

        In general, for `values` :math:`= [v_0, v_1, ...v_M]`,
        irr is the solution of the equation: [G]
        _

        ..math:: \\sum_{t=0}^M{\\frac{v_t}{(1+irr)^{t}}} = 0

        References
        ----------
        .. [G] L.J.Gitman, "Principles of Managerial Finance, Brief," 3rd ed.,
         Addison-Wesley, 2003, pg. 348.


      Examples
        --------
        >>> import numpy_financial as npf

        >>> round(npf.irr([-100, 39, 59, 55, 20]), 5)
        0.28095
        >>> round(npf.irr([-100, 0, 0, 74]), 5)
        -0.0955
        >>> round(npf.irr([-100, 100, 0, -7]), 5)
        -0.0833
        >>> round(npf.irr([-100, 100, 0, 7]), 5)
        0.06206
        >>> round(npf.irr([-5, 10.5, 1, -8, 1]), 5)
        0.0886

        */

        /// <summary>
        /// Return the Internal Rate of Return (IRR).
        /// </summary>
        /// <param name="values">Input cash flows per time period.</param>
        /// <returns></returns>
        public static ndarray irr(object values)
        {
            var _values = np.atleast_1d(values).ElementAt(0);
            if (_values.ndim != 1)
            {
                throw new ValueError("Cashflows must be a rank-1 array");
            }

            // Strip leading and trailing zeros. Since we only care about
            // positive roots we can neglect roots at zero.
            var non_zero = np.nonzero(np.ravel(_values))[0];

            _values = _values.A(string.Format("{0}:{1}",(npy_intp)non_zero[0],(npy_intp)non_zero[-1] + 1));

            var res = _roots(_values.A("::-1"));

            ndarray mask;
            if (res.IsComplex)
            {
                mask = (res.Imag == 0) & (res.Real > 0);
            }
            else
            {
                mask = (res > 0);
            }

            if (!mask.Anyb())
            {
                return np.array(double.NaN);
            }

            res = res.A(mask).Real;
            // NPV(rate) = 0 can have more than one solution so we return
            // only the solution closest to zero.
            var rate = 1 / res - 1;
            rate = np.array(rate.item((npy_intp)np.argmin(np.absolute(rate))));
            return rate;
        }

        private static ndarray _roots(ndarray p)
        {
            var t1 = np.frexp(p);
            var e = t1[1];

            // Balance the most extreme exponents e_max and e_min by solving
            // the equation
            //
            // |c + e_max| = |c + e_min|.
            //
            //Round the exponent to an integer to avoid rounding errors.

            var k1 = np.max(e);
            var k2 = np.min(e);

            var c = (int)(-0.5 * (Convert.ToDouble(np.max(e)) + Convert.ToDouble(np.min(e))));
            p = np.ldexp(p, c);

            var A = np.diag(np.full(p.size - 2, p[0]), k : -1);
            A[0, ":"] = -p.A("1:");

            // todo:  We need to implement np.linalg.eigvals(A) before we can make this work.
            var eigenvalues = np.array(new double[] { 1.92605857, -0.30194819, 0.22965312, 0.1462365 }); // np.linalg.eigvals(A);
            return eigenvalues / p[0];
        }

        #endregion

        #region npv

        /*
        Returns the NPV(Net Present Value) of a cash flow series.

        Parameters
        ----------
        rate : scalar
            The discount rate.
        values : array_like, shape(M, )
            The values of the time series of cash flows.  The (fixed) time
            interval between cash flow "events" must be the same as that for
            which `rate` is given (i.e., if `rate` is per year, then precisely
            a year is understood to elapse between each cash flow event).  By
            convention, investments or "deposits" are negative, income or
            "withdrawals" are positive; `values` must begin with the initial
            investment, thus `values [0]` will typically be negative.

        Returns
        -------
        out : float
            The NPV of the input cash flow series `values` at the discount
            `rate`.

        Warnings
        --------
        ``npv`` considers a series of cashflows starting in the present(t = 0).
        NPV can also be defined with a series of future cashflows, paid at the
        end, rather than the start, of each period.If future cashflows are used,
        the first cashflow `values[0]` must be zeroed and added to the net
        present value of the future cashflows.This is demonstrated in the
        examples.

        Notes
        -----
        Returns the result of: [G]_

        .. math :: \\sum_{ t = 0}^{M-1}{\\frac{values_t}{(1+rate)^{t}}}

        References
        ----------
        .. [G] L.J.Gitman, "Principles of Managerial Finance, Brief," 3rd ed.,
         Addison-Wesley, 2003, pg. 346.


      Examples
        --------
        >>> import numpy as np
        >>> import numpy_financial as npf

      Consider a potential project with an initial investment of $40 000 and
      projected cashflows of $5 000, $8 000, $12 000 and $30 000 at the end of

      each period discounted at a rate of 8% per period. To find the project's

      net present value:

        >>> rate, cashflows = 0.08, [-40_000, 5_000, 8_000, 12_000, 30_000]
        >>> npf.npv(rate, cashflows).round(5)
        3065.22267

        It may be preferable to split the projected cashflow into an initial
        investment and expected future cashflows.In this case, the value of
        the initial cashflow is zero and the initial investment is later added
        to the future cashflows net present value:

        >>> initial_cashflow = cashflows[0]
        >>> cashflows[0] = 0
        >>> np.round(npf.npv(rate, cashflows) + initial_cashflow, 5)
        3065.22267

        */

        /// <summary>
        /// Returns the NPV (Net Present Value) of a cash flow series.
        /// </summary>
        /// <param name="rate">The discount rate.</param>
        /// <param name="values">The values of the time series of cash flows.</param>
        /// <returns></returns>
        public static ndarray npv(object rate, object values)
        {
            var _values = np.asanyarray(values);
            var _rate = np.asanyarray(rate);

            ndarray npvArr = (ndarray)(_values / np.power((1 + _rate), np.arange(0,_values.Dim(0))));
            ndarray npvValue = npvArr.Sum(axis: 0);
            return np.atleast_1d(npvValue).ElementAt(0);
        }

        #endregion
    }
}
