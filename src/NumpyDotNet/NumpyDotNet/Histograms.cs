/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018-2019
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
    public static partial class np
    {
        #region bin selectors
        //Square root histogram bin estimator.

        //Bin width is inversely proportional to the data size.Used by many
        //programs for its simplicity.

        //Parameters
        //----------
        //x : array_like
        //    Input data that is to be histogrammed, trimmed to range.May not
        //    be empty.

        //Returns
        //-------
        //h : An estimate of the optimal bin width for the given data.

        private static double _hist_bin_sqrt(object _x)
        {
            var x = np.asanyarray(_x);
            return (double)(ndarray)(x.ptp() / np.sqrt(x.size));
        }

        // Sturges histogram bin estimator.
        //
        //A very simplistic estimator based on the assumption of normality of
        //
        //the data. This estimator has poor performance for non-normal data,
        //which becomes especially obvious for large data sets.The estimate
        //
        //depends only on size of the data.
        //
        //Parameters
        //    ----------
        //
        //x : array_like
        //    Input data that is to be histogrammed, trimmed to range.May not
        //    be empty.
        //
        //Returns
        //    -------
        //h : An estimate of the optimal bin width for the given data.

        private static double _hist_bin_sturges(object _x)
        {
            var x = np.asanyarray(_x);
            return (double)(ndarray)(x.ptp() / (np.log2(x.size) + 1.0));
        }

        //Rice histogram bin estimator.
        // Another simple estimator with no normality assumption. It has better
        // performance for large data than Sturges, but tends to overestimate
        //
        // the number of bins. The number of bins is proportional to the cube
        //
        // root of data size (asymptotically optimal). The estimate depends
        // only on size of the data.
        //
        // Parameters
        //    ----------
        //
        // x : array_like
        //     Input data that is to be histogrammed, trimmed to range.May not
        //
        //     be empty.
        //
        // Returns
        //    -------
        // h : An estimate of the optimal bin width for the given data.

        private static double _hist_bin_rice(object _x)
        {
            var x = np.asanyarray(_x);
            return (double)(ndarray)(x.ptp() / (2.0 * np.power(x.size, (1.0 / 3))));
        }

        //Scott histogram bin estimator.
        //
        //The binwidth is proportional to the standard deviation of the data
        //and inversely proportional to the cube root of data size
        //(asymptotically optimal).
        //
        //Parameters
        //    ----------
        //x : array_like
        //    Input data that is to be histogrammed, trimmed to range.May not
        //    be empty.
        //
        //Returns
        //    -------
        //h : An estimate of the optimal bin width for the given data.

        private static double _hist_bin_scott(object _x)
        {
            var x = np.asanyarray(_x);
            return (double)(ndarray)(Math.Pow((24.0 * Math.Pow(Math.PI, 0.5) / x.size), (1.0 / 3.0)) * np.std(x));
        }

        //Doane's histogram bin estimator.

        //Improved version of Sturges' formula which works better for
        //non-normal data.See
        //stats.stackexchange.com/questions/55134/doanes-formula-for-histogram-binning
        //
        //Parameters
        //----------
        //x : array_like
        //    Input data that is to be histogrammed, trimmed to range.May not
        //    be empty.
        //
        //Returns
        //-------
        //h : An estimate of the optimal bin width for the given data.
        //

        private static double _hist_bin_doane(object _x)
        {
            var x = np.asanyarray(_x);

            if (x.size > 2)
            {
                var sg1 = np.sqrt(6.0 * (x.size - 2) / ((x.size + 1.0) * (x.size + 3)));
                var sigma = (double)np.std(x);
                if (sigma > 0.0)
                {
                    //These three operations add up to
                    // g1 = np.mean(((x - np.mean(x)) / sigma)**3)
                    // but use only one temp array instead of three
                    var temp = x - np.mean(x);
                    np.true_divide(temp, sigma, temp);
                    np.power(temp, 3, temp);
                    var g1 = np.mean(temp);
                    return (double)(ndarray)(x.ptp() / (1.0 + np.log2(x.size) + np.log2(1.0 + (ndarray)(np.absolute(g1) / sg1))));
                }

            }

            return 0.0;

        }

        //The Freedman-Diaconis histogram bin estimator.

        //The Freedman-Diaconis rule uses interquartile range (IQR) to
        //estimate binwidth. It is considered a variation of the Scott rule
        //with more robustness as the IQR is less affected by outliers than
        //the standard deviation. However, the IQR depends on fewer points
        //than the standard deviation, so it is less accurate, especially for
        //long tailed distributions.

        //If the IQR is 0, this function returns 1 for the number of bins.
        //Binwidth is inversely proportional to the cube root of data size
        //(asymptotically optimal).

        //Parameters
        //----------
        //x : array_like
        //    Input data that is to be histogrammed, trimmed to range.May not
        //    be empty.
        //Returns
        //-------
        //h : An estimate of the optimal bin width for the given data.

        private static double _hist_bin_fd(object _x)
        {
            var x = np.asanyarray(_x);

            var bins = np.percentile(x, new float[] { 75, 25 });
            var iqr = (double)bins.GetItem(0) - (double)bins.GetItem(1);
            return 2.0 * iqr * Math.Pow(x.size, (-1.0 / 3.0));
        }

        // Histogram bin estimator that uses the minimum width of the
        // Freedman-Diaconis and Sturges estimators.
        //
        // The FD estimator is usually the most robust method, but its width
        // estimate tends to be too large for small `x`. The Sturges estimator
        // is quite good for small (<1000) datasets and is the default in the R
        // language.This method gives good off the shelf behaviour.
        //
        //Parameters
        // ----------
        // x : array_like
        //    Input data that is to be histogrammed, trimmed to range.May not
        //    be empty.
        //
        //Returns
        // -------
        //h : An estimate of the optimal bin width for the given data.
        //
        //See Also
        // --------
        //_hist_bin_fd, _hist_bin_sturges

        private static double _hist_bin_auto(object _x)
        {
            // There is no need to check for zero here. If ptp is, so is IQR and
            // vice versa. Either both are zero or neither one is.
            return (double)np.minimum(_hist_bin_fd(_x), _hist_bin_sturges(_x));
        }
        #endregion

        /*
        *
        * bincount accepts one, two or three arguments. The first is an array of
        * non-negative integers The second, if present, is an array of weights,
        * which must be promotable to double. Call these arguments list and
        * weight. Both must be one-dimensional with len(weight) == len(list). If
        * weight is not present then bincount(list)[i] is the number of occurrences
        * of i in list.  If weight is present then bincount(self,list, weight)[i]
        * is the sum of all weight[j] where list [j] == i.  Self is not used.
        * The third argument, if present, is a minimum length desired for the
        * output array.
        */
        public static ndarray bincount(object x, object weights = null, npy_intp? minlength = null)
        {
            ndarray list = np.asanyarray(x).ravel();
            ndarray weight = weights != null ? np.asanyarray(weights).ravel() : null;
            ndarray ans;

            #region validation
            if (list.ndim != 1)
            {
                throw new Exception("Histograms only supported on 1d arrays");
            }
            if (weight != null)
            {
                if (weight.ndim != 1)
                {
                    throw new Exception("weights array must be 1d");
                }

                if (list.Size != weight.Size)
                {
                    throw new Exception("the list and weights must of the same length");
                }
            }
            if (minlength != null)
            {
                if (minlength < 0)
                {
                    throw new Exception("minlength must not be a negative value");
                }
            }
      
            switch (list.TypeNum)
            {
                case NPY_TYPES.NPY_BYTE:
                case NPY_TYPES.NPY_INT16:
                case NPY_TYPES.NPY_INT32:
                case NPY_TYPES.NPY_INT64:
                    break;
                case NPY_TYPES.NPY_UBYTE:
                case NPY_TYPES.NPY_UINT16:
                case NPY_TYPES.NPY_UINT32:
                case NPY_TYPES.NPY_UINT64:
                    // not sure why these are not supported in python. I will support them.
                    break;

                default:
                    throw new Exception("Histograms only supported on integer arrays");

            }
            #endregion

            // convert input array to intp if not already
            ndarray lst = np.asarray(list, np.intp);
            npy_intp len = lst.Size;

            /* handle empty list */
            if (len == 0)
            {
                ans = np.zeros(new shape(1), dtype: np.intp);
                return ans;
            }


            // get pointer to raw input data
            npy_intp[] numbers = lst.ToArray<npy_intp>();

            // get min and max values of the input data
            npy_intp mn = (npy_intp)np.amin(lst).GetItem(0);
            npy_intp mx = (npy_intp)np.amax(lst).GetItem(0);
            if (mn < 0)
            {
                throw new Exception("histogram arrays must not contain negative numbers");
            }

            // determine size of return array.
            npy_intp ans_size = mx + 1;
            if (minlength != null)
            {
                if (ans_size < minlength.Value)
                {
                    ans_size = minlength.Value;
                }
            }

            // if weight is null, we return array of npy_intp, else doubles.
            if (weight == null)
            {
                npy_intp[] ians = new npy_intp[ans_size];
                for (npy_intp i = 0; i < len; i++)
                {
                    ians[numbers[i]] += 1;
                }

                ans = np.array(ians, dtype: np.intp);
                return ans;
            }
            else
            {
                ndarray wts = np.asarray(weight, dtype: np.Float64);
                double[] _weights = wts.ToArray<double>();

                double[] dans = new double[ans_size];

                for (npy_intp i = 0; i < len; i++)
                {
                    dans[numbers[i]] += _weights[i];
                }

                ans = np.array(dans, dtype: np.Float64);
                return ans;
            }

        }


        /*
        * digitize(x, bins, right=False) returns an array of integers the same length
        * as x. The values i returned are such that bins[i - 1] <= x < bins[i] if
        * bins is monotonically increasing, or bins[i - 1] > x >= bins[i] if bins
        * is monotonically decreasing.  Beyond the bounds of bins, returns either
        * i = 0 or i = len(bins) as appropriate. If right == True the comparison
        * is bins [i - 1] < x <= bins[i] or bins [i - 1] >= x > bins[i]
        */
        public static ndarray digitize(object x, object bins, bool right= false)
        {
            ndarray arr_x = np.asanyarray(x).ravel();
            ndarray arr_bins = np.asanyarray(bins).ravel();

            arr_x = np.asarray(arr_x, np.Float64);
            arr_bins = np.asarray(arr_bins, np.Float64);

            var len_bins = arr_bins.Size;
            if (len_bins == 0)
            {
                throw new Exception("bins must have non-zero length");
            }

            double[] arr_bins_data = arr_bins.ToArray<double>();

            int monotonic = check_array_monotonic(arr_bins_data, len_bins);

            if (monotonic == 0)
            {
                throw new Exception("bins must be monotonically increasing or decreasing");
            }

            /* PyArray_SearchSorted needs an increasing array */
            if (monotonic == -1)
            {
                // reverse the array
                arr_bins = arr_bins["::-1"] as ndarray;
            }

            var ret =  np.searchsorted(arr_bins, arr_x, right ?  NPY_SEARCHSIDE.NPY_SEARCHLEFT : NPY_SEARCHSIDE.NPY_SEARCHRIGHT, null);
            if (ret == null)
            {
                return null;
            }

            /* If bins is decreasing, ret has bins from end, not start */
            if (monotonic == -1)
            {
                npy_intp[] ret_data = ret.ToArray<npy_intp>();
                npy_intp len_ret = ret.Size;


                npy_intp index = 0;
                while (len_ret-- > 0)
                {
                    ret_data[index] = len_bins - ret_data[index];
                    index++;
                }
    
            }

            return ret;
        }

        private static int check_array_monotonic(double[] a, long lena)
        {
            npy_intp i;
            double next;
            double last = a[0];

            /* Skip repeated values at the beginning of the array */
            for (i = 1; (i < lena) && (a[i] == last); i++) ;

            if (i == lena)
            {
                /* all bin edges hold the same value */
                return 1;
            }

            next = a[i];
            if (last < next)
            {
                /* Possibly monotonic increasing */
                for (i += 1; i < lena; i++)
                {
                    last = next;
                    next = a[i];
                    if (last > next)
                    {
                        return 0;
                    }
                }
                return 1;
            }
            else
            {
                /* last > next, possibly monotonic decreasing */
                for (i += 1; i < lena; i++)
                {
                    last = next;
                    next = a[i];
                    if (last < next)
                    {
                        return 0;
                    }
                }
                return -1;
            }
        }
    }
}
