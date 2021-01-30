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
    public enum Histogram_BinSelector
    {
        auto,
        doane,
        fd,
        rice,
        scott,
        sqrt,
        sturges,
    }


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

        #region histogram
        public static (ndarray hist, ndarray bin_edges) histogram<T>(object a, T[] bins = null, (float Low, float High)? range = null, object weights = null, bool? density = null)
        {
            return _histogram<T>(a, bins, range, weights, density);
        }
        public static (ndarray hist, ndarray bin_edges) histogram(object a, ndarray bins = null, (float Low, float High)? range = null, object weights = null, bool? density = null)
        {
            return _histogram<Int32>(a, bins, range, weights, density);
        }
        public static (ndarray hist, ndarray bin_edges) histogram(object a, int? bins = null, (float Low, float High)? range = null, object weights = null, bool? density = null)
        {
            return _histogram<Int32>(a, bins, range, weights, density);
        }
        public static (ndarray hist, ndarray bin_edges) histogram(object a, Histogram_BinSelector bins, (float Low, float High)? range = null, object weights = null, bool? density = null)
        {
            return _histogram<Int32>(a, bins, range, weights, density);
        }

        private static (ndarray hist, ndarray bin_edges) _histogram<T>(object _a, object bins, (float Low, float High)? range, object _weights, bool? density)
        {
            dtype ntype;
            ndarray a;
            ndarray weights;
            ndarray cum_n;
            ndarray n;
            ndarray bin_edges;
            float first_edge;
            float last_edge;
            int? n_equal_bins;

            var t1 = _ravel_and_check_weights(_a, _weights);
            a = t1.a;
            weights = t1.weights;
            
            var _bin_edges_data = _get_bin_edges<T>(a, bins, range, weights);
            bin_edges = _bin_edges_data.bin_edges;
            first_edge = _bin_edges_data.first_edge;
            last_edge = _bin_edges_data.last_edge;
            n_equal_bins = _bin_edges_data.n_equal_bins;

            // Histogram is an integer or a float array depending on the weights.
            if (_weights == null)
                ntype = np.intp;
            else
                ntype = weights.Dtype;

            // We set a block size, as this allows us to iterate over chunks when
            // computing histograms, to minimize memory usage.
            Int32 BLOCK = 65536;

            //The fast path uses bincount, but that only works for certain types of weight
            bool simple_weights =
                (_weights == null ||
                np.can_cast(weights.Dtype, np.Float64) ||
                np.can_cast(weights.Dtype, np.Complex));

            if (n_equal_bins != null && simple_weights)
            {
                // Fast algorithm for equal bins
                // We now convert values of a to bin indices, under the assumption of
                // equal bin widths (which is valid here).

                // Initialize empty histogram
                n = np.zeros(n_equal_bins, ntype);

                // Pre-compute histogram scaling factor
                var norm = n_equal_bins / (last_edge - first_edge);

                // We iterate over blocks here for two reasons: the first is that for
                // large arrays, it is actually faster (for example for a 10^8 array it
                // is 2x as fast) and it results in a memory footprint 3x lower in the
                // limit of large arrays.
                for (int i = 0; i < len(a); i+= BLOCK)
                {
                    ndarray tmp_w;

                    var tmp_a = a.A(string.Format("{0}:{1}", i, i + BLOCK));
                    if (weights == null)
                        tmp_w = null;
                    else
                        tmp_w = weights.A(string.Format("{0}:{1}", i, i + BLOCK));


                    // Only include values in the right range
                    var keep = (tmp_a >= first_edge);
                    keep &= (tmp_a <= last_edge);
                    if ((bool)np.ufunc.reduce(UFuncOperation.logical_and, keep) == false)
                    {
                        tmp_a = tmp_a.A(keep);
                        if (tmp_w != null)
                            tmp_w = tmp_w.A(keep);
                    }

                    // This cast ensures no type promotions occur below, which gh-10322
                    // make unpredictable. Getting it wrong leads to precision errors
                    // like gh-8123.
                    tmp_a = tmp_a.astype(bin_edges.Dtype, copy: false);


                    // Compute the bin indices, and for values that lie exactly on
                    // last_edge we need to subtract one
                    var f_indices = (tmp_a - first_edge) * norm;
                    var indices = f_indices.astype(np.intp);

                    var mask = np.equal(indices, n_equal_bins);
                    indices[mask] = indices.A(mask) - 1;

                    // The index computation is not guaranteed to give exactly
                    // consistent results within ~1 ULP of the bin edges.
                    var decrement = tmp_a < bin_edges[indices];
                    indices[decrement] = indices.A(decrement) - 1;
                    // The last bin includes the right edge. The other bins do not.
                    var increment = ((tmp_a >= bin_edges[indices + 1])
                         & (np.not_equal(indices,n_equal_bins - 1)));
                    indices[increment] = indices.A(increment) + 1;

                    // We now compute the histogram using bincount
                    //if (ntype.kind == "c")
                    //{
                    //    n.Real += np.bincount(indices, weights = tmp_w.Real, minlength: n_equal_bins);
                    //    n.Imag += np.bincount(indices, weights = tmp_w.Imag, minlength: n_equal_bins);
                    //}
                    //else
                    {
                        n += np.bincount(indices, weights = tmp_w, minlength: n_equal_bins).astype(ntype);
                    }
                }
            }
            else
            {
                // Compute via cumulative histogram
                cum_n = np.zeros(bin_edges.shape, ntype);
                if (weights == null)
                {
                    for (int i = 0; i < len(a); i+= BLOCK)
                    {
                        var sa = np.sort(a[string.Format("{0}:{1}", i, i + BLOCK)]);
                        cum_n += _search_sorted_inclusive(sa, bin_edges);
                    }
                }
      
                else
                {
                    var zero = np.zeros(1, dtype : ntype);
                    for (int i = 0; i < len(a); i += BLOCK)
                    {
                        var tmp_a = a.A(string.Format("{0}:{1}", i, i + BLOCK));
                        var tmp_w = weights.A(string.Format("{0}:{1}", i, i + BLOCK));
                        var sorting_index = np.argsort(tmp_a);
                        var sa = tmp_a.A(sorting_index);
                        var sw = tmp_w.A(sorting_index);
                        var cw = np.concatenate((zero, sw.cumsum()));
                        var bin_index = _search_sorted_inclusive(sa, bin_edges);
                        cum_n += cw.A(bin_index);
                    }
   
                }


                n = np.diff(cum_n);
            }

            bool normed = false;
            // density overrides the normed keyword
            if (density != null)
                normed = false;

            if (density == true)
            {
                var db = np.array(np.diff(bin_edges), dtype: np.Float32);
                return (n / db / np.sum(n), bin_edges);
            }
            else if (normed)
            {
                // deprecated, buggy behavior. Remove for NumPy 2.0.0
                var db = np.array(np.diff(bin_edges), dtype: np.Float32);
                return ((ndarray)(n / np.sum(n * db)), bin_edges);
            }
            else
            {
                return (n, bin_edges);
            }

        }

        private static ndarray _search_sorted_inclusive(ndarray a, ndarray v)
        {
            //Like `searchsorted`, but where the last item in `v` is placed on the right.

            //In the context of a histogram, this makes the last bin edge inclusive

            return np.concatenate((
                a.searchsorted(v[":-1"], NPY_SEARCHSIDE.NPY_SEARCHLEFT),
                a.searchsorted(v["-1:"], NPY_SEARCHSIDE.NPY_SEARCHRIGHT)));
        }

        private static (ndarray a, ndarray weights) _ravel_and_check_weights(object a, object weights)
        {
            ndarray _a = np.asarray(a);
            ndarray _weights = null;
            if (weights != null)
            {
                _weights = np.asarray(weights);
                if (!_weights.shape.Equals(_a.shape))
                {
                    throw new Exception("weights should have the same shape as a!");
                }

                _weights = _weights.ravel();
            }

            _a = _a.ravel();
            return (_a, _weights);
        }

        #endregion

        #region histogramdd

        public static (ndarray hist, ndarray[] bin_edges) histogramdd<T>(object sample, T[] bins = null, T[] range = null, object weights = null, bool? density = null)
        {
            return _histogramdd<T>(sample, bins, range, weights, density);
        }
        public static (ndarray hist, ndarray[] bin_edges) histogramdd<T>(object sample, ndarray bins = null, T[] range = null, object weights = null, bool? density = null)
        {
            return _histogramdd<T>(sample, bins, range, weights, density);
        }
        public static (ndarray hist, ndarray[] bin_edges) histogramdd<T>(object sample, int? bins = null, T[] range = null, object weights = null, bool? density = null)
        {
            return _histogramdd<T>(sample, bins, range, weights, density);
        }
        public static (ndarray hist, ndarray[] bin_edges) histogramdd<T>(object sample, Histogram_BinSelector bins, T[] range = null, object weights = null, bool? density = null)
        {
            return _histogramdd<T>(sample, bins, range, weights, density);
        }

        private static (ndarray hist, ndarray[] bin_edges) _histogramdd<T>(object _sample, object _bins, T[] range, object _weights, bool? density)
        {
            npy_intp N, D;
            npy_intp M;

            ndarray sample = np.asanyarray(_sample);
            ndarray weights = null;
            T[] bins = null;
            ndarray smin, smax;
            dtype edge_dt;

            try
            {
                N = sample.Dim(0);
                D = sample.Dim(1);
            }
            catch (Exception ex)
            {
                sample = np.atleast_2d(sample).ElementAt(0).T;
                N = sample.Dim(0);
                D = sample.Dim(1);
            }


            ndarray nbin = np.empty(D, np.Int32);
            var edges = new ndarray[D];
            var dedges = new ndarray[D];

            if (_weights != null)
            {
                weights = np.asarray(_weights);
            }

            if (_bins is T[])
            {
                bins = _bins as T[];
                M = bins.Length;
                if (M != D)
                {
                    throw new Exception("The dimension of bins must be equal to the dimension of the sample x.");
                }
            }
            else if (_bins is Int32)
            {
                bins = new T[D];
            }

            // Select range for each dimension
            // Used only if number of bins is given.
            if (range == null)
            {
                // Handle empty input. Range can't be determined in that case, use 0-1.
                if (N == 0)
                {
                    smin = np.zeros(D);
                    smax = np.ones(D);
                }
                else
                {
                    smin = np.atleast_1d(np.array(np.amin(sample, axis: 0), np.Float32)).ElementAt(0);
                    smax = np.atleast_1d(np.array(np.amax(sample, axis: 0), np.Float32)).ElementAt(0);
                }
            }
            else
            {
                if (!np.allb(np.isfinite(range)))
                {
                    throw new Exception("range parameter must be finite.");
                }
                smin = np.zeros(D);
                smax = np.zeros(D);
                for (int i = 0; i < D; i++)
                {
                    smin[i] = smax[i] = range[i];
                }
            }


            // Make sure the bins have a finite width.
            for (int i = 0; i < len(smin); i++)
            {
                if (smin[i] == smax[i])
                {
                    smin[i] = (dynamic)smin[i] - .5;
                    smax[i] = (dynamic)smax[i] + .5;
                }
            }
 
            if (sample.IsInexact)
                edge_dt = sample.Dtype;
            else
                edge_dt = np.Float32;


            // Create edge arrays
            for (int i = 0; i < D; i++)
            {
                if (np.isscalar(bins[i]))
                {
                    if ((dynamic)bins[i] < 1)
                    {
                        throw new Exception(string.Format("Element at index {0} in 'bins' should be a positive integer.", i));
                    }
                    nbin[i] = (dynamic)bins[i] + 2;   // +2 for outlier bins

                    double ret_step = 0;
                    edges[i] = np.linspace((dynamic)smin[i], (dynamic)smax[i], ref ret_step, (dynamic)nbin[i] - 1, dtype: edge_dt);
                }
                else
                {
                    edges[i] = np.asarray(bins[i], edge_dt);
                    nbin[i] = len(edges[i]) + 1;   // +1 for outlier bins
                }

                dedges[i] = np.diff(edges[i]);
                if (np.anyb(np.asarray(dedges[i]) <= 0))
                {
                    throw new Exception("Found bin edge of size <= 0. Did you specify 'bins' with non-monotonic sequence?");
                }

            }

            nbin = np.asarray(nbin);

            //Handle empty input.
            if (N == 0)
                return (np.zeros(new shape((nbin - 2))), edges);

            List<ndarray> Ncount = new List<ndarray>();
            for (int i = 0; i < D; i++)
            {
                Ncount.Add(np.digitize(sample[":", i], edges[i]));
            }

            // Using digitize, values that fall on an edge are put in the right bin.
            // For the rightmost bin, we want values equal to the right edge to be
            // counted in the last bin, and not as an outlier.
            for (int i = 0; i < D; i++)
            {
                // Rounding precision
                dynamic mindiff = np.min(dedges[i]);
                if (!np.isinf(mindiff).GetItem(0))
                {
                    var log10 = -np.log10(mindiff).GetItem(0);
                    int _decimal = (int)log10 +6;
                    //Find which points are on the rightmost edge.
                    ndarray not_smaller_than_edge = (sample.A(":", i) >= edges[i].A("-1"));
                    ndarray on_edge = (np.around(sample.A(":", i), _decimal).Equals(np.around(edges[i].A("-1"), _decimal)));
                    //# Shift these points one bin to the left.

                    ndarray mask = np.nonzero(on_edge & not_smaller_than_edge)[0];
                    Ncount[i][mask] = Ncount[i].A(mask) - 1;
                }
  
            }

            // Flattened histogram matrix (1D)
            // Reshape is used so that overlarge arrays
            // will raise an error.
            ndarray hist = np.zeros(new shape(nbin), np.Float32).reshape(-1);

            // Compute the sample indices in the flattened histogram matrix.
            var ni = np.argsort(nbin);
            var xy = np.zeros(N, np.Int32);
            for (int i = 0; i < D - 1; i++)
            {
                var k1 = Ncount[(int)(npy_intp)ni.GetItem(i)];
                var k2 = nbin.A(ni[string.Format("{0}:", i+1)]);

                xy += k1 * np.prod(k2);
            }

            xy += Ncount[(int)(npy_intp)ni.GetItem(-1)];

            // Compute the number of repetitions in xy and assign it to the
            // flattened histmat.
            if (len(xy) == 0)
            {
                return (np.zeros(nbin - 2, np.Int32), edges);
            }

            var flatcount = np.bincount(xy, weights);
            var a = np.arange(len(flatcount));
            hist[a] = flatcount;


            // Shape into a proper matrix
            hist = hist.reshape(new shape(np.sort(nbin)));
            for (int i = 0; i < nbin.size; i++)
            {
                var j = (npy_intp)np.argsort(ni)[i];
                hist = np.swapaxes(hist, i, (int)j);

                var temp = ni[i];
                ni[j] = ni[i];
                ni[j] = temp;
            }

            var core = BuildSliceArray(new Slice(1, -1), (int)D);
            hist = hist.A(core);

            // Normalize if density is True
            if (density == true)
            {
                var s = hist.Sum();
                for (int i = 0; i < D; i++)
                {
                    var shape = np.ones(D, np.Int32);
                    shape[i] = np.subtract(nbin[i],2);
                    hist = np.divide(hist, dedges[i].reshape(new shape(shape.ToArray<Int32>())));
                }

                hist = np.divide(hist, s);
            }

            if (!hist.shape.Equals(new shape(np.subtract(nbin,2))))
            {
                throw new RuntimeError("Internal Shape Error");
            }

            return (hist, edges);

        }
#endregion

#region histogram_bin_edges
        public static ndarray histogram_bin_edges<T>(object a, T[] bins = null, (float Low, float High)? range = null, object weights = null)
        {
            ndarray _a = np.asanyarray(a);
            ndarray _weights = np.asanyarray(weights);
            var bin_edges = _get_bin_edges<T>(_a, bins, range, _weights);
            return bin_edges.bin_edges;
        }
        public static ndarray histogram_bin_edges(object a, ndarray bins = null, (float Low, float High)? range = null, object weights = null)
        {
            ndarray _a = np.asanyarray(a);
            ndarray _weights = np.asanyarray(weights);
            var bin_edges = _get_bin_edges<Int32>(_a, bins, range, _weights);
            return bin_edges.bin_edges;
        }
        public static ndarray histogram_bin_edges(object a, int? bins = null, (float Low, float High)? range = null, object weights = null)
        {
            ndarray _a = np.asanyarray(a);
            ndarray _weights = np.asanyarray(weights);
            var bin_edges = _get_bin_edges<Int32>(_a, bins, range, _weights);
            return bin_edges.bin_edges;
        }
        public static ndarray histogram_bin_edges(object a, Histogram_BinSelector bins, (float Low, float High)? range = null, object weights = null)
        {
            ndarray _a = np.asanyarray(a);
            ndarray _weights = np.asanyarray(weights);
            var bin_edges = _get_bin_edges<Int32>(_a, bins, range, _weights);
            return bin_edges.bin_edges;
        }

        private class bin_edges_data
        {
            public ndarray bin_edges;
            public float first_edge;
            public float last_edge;
            public Int32? n_equal_bins;
        }

  
        private static bin_edges_data _get_bin_edges<T>(ndarray a, object bins, (float Low, float High)? range, ndarray weights)
        {
            int n_equal_bins = -1;
            float first_edge = 0;
            float last_edge = 0;
            ndarray bin_edges = null;

            if (bins is Histogram_BinSelector)
            {
                Histogram_BinSelector binSelector = (Histogram_BinSelector)bins;

                if (weights != null)
                {
                    throw new Exception("Automated estimation of the number of bins is not supported for weighted data");
                }

                var edges = _get_outer_edges(a, range);
                first_edge = edges.first_edge;
                last_edge = edges.last_edge;

                if (range != null)
                {
                    ndarray keep = (a >= first_edge);
                    keep &= (a <= last_edge);
                    if ((bool)np.ufunc.reduce(UFuncOperation.logical_and, keep) == false)
                    {
                        a = a[keep] as ndarray;
                    }
                }

                if (a.size == 0)
                {
                    n_equal_bins = 1;
                }
                else
                {
                    // Do not call selectors on empty arrays
                    double width = 0.0;

                    switch (binSelector)
                    {
                        case Histogram_BinSelector.auto:
                            width = _hist_bin_auto(a);
                            break;
                        case Histogram_BinSelector.doane:
                            width = _hist_bin_doane(a);
                            break;
                        case Histogram_BinSelector.fd:
                            width = _hist_bin_fd(a);
                            break;
                        case Histogram_BinSelector.rice:
                            width = _hist_bin_rice(a);
                            break;
                        case Histogram_BinSelector.scott:
                            width = _hist_bin_scott(a);
                            break;
                        case Histogram_BinSelector.sqrt:
                            width = _hist_bin_sqrt(a);
                            break;
                        case Histogram_BinSelector.sturges:
                            width = _hist_bin_sturges(a);
                            break;
                        default:
                            throw new Exception(string.Format("{0} is not a valid estimator for 'bins'", binSelector.ToString()));
                    }

                    if (width > 0)
                    {
                        n_equal_bins = Convert.ToInt32(np.ceil((last_edge - first_edge) / width).GetItem(0));
                    }
                    else
                    {
                        // Width can be zero for some estimators, e.g. FD when
                        // the IQR of the data is zero.
                        n_equal_bins = 1;
                    }
 
                }
            }
            else if (bins is Int32)
            {
                n_equal_bins = Convert.ToInt32(bins);
                if (n_equal_bins < 1)
                {
                    throw new Exception("'bins' must be a positive number");
                }

                var edges = _get_outer_edges(a, range);
                first_edge = edges.first_edge;
                last_edge = edges.last_edge;
            }
            else if (bins is T[] || bins is ndarray)
            {
                bin_edges = np.asarray(bins);
                if (np.anyb((ndarray)bin_edges[":-1"] > (ndarray)bin_edges["1:"]))
                {
                    throw new Exception("'bins' must increase monotonically");
                }
            }
            else
            {
                throw new Exception("unrecognized 'bin' value");
            }

            if (n_equal_bins > 0)
            {
                // gh-10322 means that type resolution rules are dependent on array
                // shapes. To avoid this causing problems, we pick a type now and stick
                // with it throughout.
                double ret_step = 0;
                bin_edges = np.linspace(first_edge, last_edge, ref ret_step, n_equal_bins + 1, endpoint: true, dtype: np.Float64);
                return new bin_edges_data() { bin_edges = bin_edges, first_edge = first_edge, last_edge = last_edge, n_equal_bins = n_equal_bins };
            }

            return new bin_edges_data() { bin_edges = bin_edges, first_edge = -1, last_edge = -1, n_equal_bins = null };
        }

        private static (float first_edge, float last_edge) _get_outer_edges(ndarray a, (float Low, float High)? range)
        {
            float first_edge;
            float last_edge;

            //Determine the outer bin edges to use, from either the data or the range argument

            if (range != null)
            {
                first_edge = range.Value.Low;
                last_edge = range.Value.High;
                if (first_edge > last_edge)
                {
                    throw new Exception("max must be larger than min in range parameter.");
                }
                if (!((bool)np.isfinite(first_edge) && (bool)np.isfinite(last_edge)))
                {
                    throw new Exception(string.Format("supplied range of [{0}, {1}] is not finite", first_edge, last_edge));
                }
            }
            else if (a.Size == 0)
            {
                first_edge = 0;
                last_edge = 1;
            }
            else
            {
                first_edge = Convert.ToSingle(np.min(a));
                last_edge = Convert.ToSingle(np.max(a));

                if (!((bool)np.isfinite(first_edge) && (bool)np.isfinite(last_edge)))
                {
                    throw new Exception(string.Format("autodetected range of [{0}, {1}] is not finite", first_edge, last_edge));
                }

            }

  

            // expand empty range to avoid divide by zero
            if (first_edge == last_edge)
            {
                first_edge = first_edge - 0.5f;
                last_edge = last_edge + 0.5f;
            }

            return (first_edge, last_edge);
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
