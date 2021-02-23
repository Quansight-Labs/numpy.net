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
                    throw new Exception("unrecognized when parameter.  Must be integers or 'begin' or 'end'");
                }

            }

            try
            {
                return Convert.ToInt32(when);
            }
            catch (Exception ex)
            {
                throw new Exception("unrecognized when parameter.  Must be integers or 'begin' or 'end'");
            }

        }

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


    }
}
