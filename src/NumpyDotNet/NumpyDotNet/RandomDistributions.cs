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
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif



namespace NumpyDotNet
{
    internal class rk_state
    {
        public const int RK_STATE_LEN = 624;
        public Int64[]key = new long[RK_STATE_LEN];
        public int pos;
        public bool has_gauss; /* !=0: gauss contains a gaussian deviate */
        public double gauss;

        /* The rk_state structure has been extended to store the following
         * information for the binomial generator. If the input values of n or p
         * are different than nsave and psave, then the other parameters will be
         * recomputed. RTK 2005-09-02 */

        public bool has_binomial; /* !=0: following parameters initialized for binomial */
        public double psave;
        public long nsave;
        public double r;
        public double q;
        public double fm;
        public long m;
        public double p1;
        public double xm;
        public double xl;
        public double xr;
        public double c;
        public double laml;
        public double lamr;
        public double p2;
        public double p3;
        public double p4;

    }

    internal static class RandomDistributions
    {
      /*
      * log-gamma function to support some of these distributions. The
      * algorithm comes from SPECFUN by Shanjie Zhang and Jianming Jin and their
      * book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.
      */
        static double loggam(double x)
        {
            double x0, x2, xp, gl, gl0;
            long k, n;

            double []a = {8.333333333333333e-02,-2.777777777777778e-03,
                            7.936507936507937e-04,-5.952380952380952e-04,
                            8.417508417508418e-04,-1.917526917526918e-03,
                            6.410256410256410e-03,-2.955065359477124e-02,
                            1.796443723688307e-01,-1.39243221690590e+00};
            x0 = x;
            n = 0;
            if ((x == 1.0) || (x == 2.0))
            {
                return 0.0;
            }
            else if (x <= 7.0)
            {
                n = (long)(7 - x);
                x0 = x + n;
            }
            x2 = 1.0 / (x0 * x0);
            xp = 2 * Math.PI;
            gl0 = a[9];
            for (k = 8; k >= 0; k--)
            {
                gl0 *= x2;
                gl0 += a[k];
            }
            gl = gl0 / x0 + 0.5 * Math.Log(xp) + (x0 - 0.5) * Math.Log(x0) - x0;
            if (x <= 7.0)
            {
                for (k = 1; k <= n; k++)
                {
                    gl -= Math.Log(x0 - 1.0);
                    x0 -= 1.0;
                }
            }
            return gl;
        }


        internal static double rk_normal(rk_state state, double loc, double scale)
        {
            return loc + scale * rk_gauss(state);
        }

        internal static double rk_standard_exponential(rk_state state)
        {
            /* We use -log(1-U) since U is [0, 1) */
            return -Math.Log(1.0 - rk_double(state));
        }

        internal static double rk_exponential(rk_state state, double scale)
        {
            return scale * rk_standard_exponential(state);
        }
 
        internal static double rk_uniform(rk_state state, double loc, double scale)
        {
            return loc + scale * rk_double(state);
        }

        internal static double rk_standard_gamma(rk_state state, double shape)
        {
            double b, c;
            double U, V, X, Y;

            if (shape == 1.0)
            {
                return rk_standard_exponential(state);
            }
            else if (shape < 1.0)
            {
                for (; ; )
                {
                    U = rk_double(state);
                    V = rk_standard_exponential(state);
                    if (U <= 1.0 - shape)
                    {
                        X = Math.Pow(U, 1.0 / shape);
                        if (X <= V)
                        {
                            return X;
                        }
                    }
                    else
                    {
                        Y = -Math.Log((1 - U) / shape);
                        X = Math.Pow(1.0 - shape + shape * Y, 1.0/ shape);
                        if (X <= (V + Y))
                        {
                            return X;
                        }
                    }
                }
            }
            else
            {
                b = shape - 1.0/ 3.0;
                c = 1.0/ Math.Sqrt(9 * b);
                for (; ; )
                {
                    do
                    {
                        X = rk_gauss(state);
                        V = 1.0 + c * X;
                    } while (V <= 0.0);

                    V = V * V * V;
                    U = rk_double(state);
                    if (U < 1.0 - 0.0331 * (X * X) * (X * X)) return (b * V);
                    if (Math.Log(U) < 0.5 * X * X + b * (1.0 - V + Math.Log(V))) return (b * V);
                }
            }
        }

        internal static double rk_gamma(rk_state state, double shape, double scale)
        {
            return scale * rk_standard_gamma(state, shape);
        }


        internal static double rk_beta(rk_state state, double a, double b)
        {
            double Ga, Gb;

            if ((a <= 1.0) && (b <= 1.0))
            {
                double U, V, X, Y;
                /* Use Johnk's algorithm */

                while (true)
                {
                    U = rk_double(state);
                    V = rk_double(state);
                    X = Math.Pow(U, 1.0 / a);
                    Y = Math.Pow(V, 1.0 / b);

                    if ((X + Y) <= 1.0)
                    {
                        if (X + Y > 0)
                        {
                            return X / (X + Y);
                        }
                        else
                        {
                            double logX = Math.Log(U) / a;
                            double logY = Math.Log(V) / b;
                            double logM = logX > logY ? logX : logY;
                            logX -= logM;
                            logY -= logM;

                            return Math.Exp(logX - Math.Log(Math.Exp(logX) + Math.Exp(logY)));
                        }
                    }
                }
            }
            else
            {
                Ga = rk_standard_gamma(state, a);
                Gb = rk_standard_gamma(state, b);
                return Ga / (Ga + Gb);
            }
        }

        internal static double rk_chisquare(rk_state state, double df)
        {
            return 2.0 * rk_standard_gamma(state, df / 2.0);
        }

        internal static double rk_noncentral_chisquare(rk_state state, double df, double nonc)
        {
            if (nonc == 0)
            {
                return rk_chisquare(state, df);
            }
            if (1 < df)
            {
                double Chi2 = rk_chisquare(state, df - 1);
                double N = rk_gauss(state) + Math.Sqrt(nonc);
                return Chi2 + N * N;
            }
            else
            {
                long i = rk_poisson(state, nonc / 2.0);
                return rk_chisquare(state, df + 2 * i);
            }
        }

        internal static double rk_f(rk_state state, double dfnum, double dfden)
        {
            return ((rk_chisquare(state, dfnum) * dfden) /
                    (rk_chisquare(state, dfden) * dfnum));
        }


        internal static double rk_noncentral_f(rk_state state, double dfnum, double dfden, double nonc)
        {
            double t = rk_noncentral_chisquare(state, dfnum, nonc) * dfden;
            return t / (rk_chisquare(state, dfden) * dfnum);
        }


        internal static long rk_binomial_btpe(rk_state state, long n, double p)
        {
            double r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
            double a, u, v, s, F, rho, t, A, nrq, x1, x2, f1, f2, z, z2, w, w2, x;
            long m, y, k, i;

            if (!(state.has_binomial) ||
                 (state.nsave != n) ||
                 (state.psave != p))
            {
                /* initialize */
                state.nsave = n;
                state.psave = p;
                state.has_binomial = true;
                state.r = r = Math.Min(p, 1.0 - p);
                state.q = q = 1.0 - r;
                state.fm = fm = n * r + r;
                state.m = m = (long)Math.Floor(state.fm);
                state.p1 = p1 = Math.Floor(2.195 * Math.Sqrt(n * r * q) - 4.6 * q) + 0.5;
                state.xm = xm = m + 0.5;
                state.xl = xl = xm - p1;
                state.xr = xr = xm + p1;
                state.c = c = 0.134 + 20.5 / (15.3 + m);
                a = (fm - xl) / (fm - xl * r);
                state.laml = laml = a * (1.0 + a / 2.0);
                a = (xr - fm) / (xr * q);
                state.lamr = lamr = a * (1.0 + a / 2.0);
                state.p2 = p2 = p1 * (1.0 + 2.0 * c);
                state.p3 = p3 = p2 + c / laml;
                state.p4 = p4 = p3 + c / lamr;
            }
            else
            {
                r = state.r;
                q = state.q;
                fm = state.fm;
                m = state.m;
                p1 = state.p1;
                xm = state.xm;
                xl = state.xl;
                xr = state.xr;
                c = state.c;
                laml = state.laml;
                lamr = state.lamr;
                p2 = state.p2;
                p3 = state.p3;
                p4 = state.p4;
            }

            /* sigh ... */
            Step10:
            nrq = n * r * q;
            u = rk_double(state) * p4;
            v = rk_double(state);
            if (u > p1) goto Step20;
            y = (long)Math.Floor(xm - p1 * v + u);
            goto Step60;

            Step20:
            if (u > p2) goto Step30;
            x = xl + (u - p1) / c;
            v = v * c + 1.0 - Math.Abs(m - x + 0.5) / p1;
            if (v > 1.0) goto Step10;
            y = (long)Math.Floor(x);
            goto Step50;

            Step30:
            if (u > p3) goto Step40;
            y = (long)Math.Floor(xl + Math.Log(v) / laml);
            if (y < 0) goto Step10;
            v = v * (u - p2) * laml;
            goto Step50;

            Step40:
            y = (long)Math.Floor(xr - Math.Log(v) / lamr);
            if (y > n) goto Step10;
            v = v * (u - p3) * lamr;

            Step50:
            k = Math.Abs(y - m);
            if ((k > 20) && (k < ((nrq) / 2.0 - 1))) goto Step52;

            s = r / q;
            a = s * (n + 1);
            F = 1.0;
            if (m < y)
            {
                for (i = m + 1; i <= y; i++)
                {
                    F *= (a / i - s);
                }
            }
            else if (m > y)
            {
                for (i = y + 1; i <= m; i++)
                {
                    F /= (a / i - s);
                }
            }
            if (v > F) goto Step10;
            goto Step60;

            Step52:
            rho = (k / (nrq)) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
            t = -k * k / (2 * nrq);
            A = Math.Log(v);
            if (A < (t - rho)) goto Step60;
            if (A > (t + rho)) goto Step10;

            x1 = y + 1;
            f1 = m + 1;
            z = n + 1 - m;
            w = n - y + 1;
            x2 = x1 * x1;
            f2 = f1 * f1;
            z2 = z * z;
            w2 = w * w;
            if (A > (xm * Math.Log(f1 / x1)
                   + (n - m + 0.5) * Math.Log(z / w)
                   + (y - m) * Math.Log(w * r / (x1 * q))
                   + (13680.0- (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320
        
                   + (13680.0- (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0

                   + (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0

                   + (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.0))
            {
                goto Step10;
            }

            Step60:
            if (p > 0.5)
            {
                y = n - y;
            }

            return y;
        }


        static long rk_binomial_inversion(rk_state state, long n, double p)
        {
            double q, qn, np, px, U;
            long X, bound;

            if (!(state.has_binomial) ||
                 (state.nsave != n) ||
                 (state.psave != p))
            {
                state.nsave = n;
                state.psave = p;
                state.has_binomial = true;
                state.q = q = 1.0 - p;
                state.r = qn = Math.Exp(n * Math.Log(q));
                state.c = np = n * p;
                state.m = bound = (npy_intp)Math.Min(n, np + 10.0 * Math.Sqrt(np * q + 1));
            }
            else
            {
                q = state.q;
                qn = state.r;
                np = state.c;
                bound = state.m;
            }
            X = 0;
            px = qn;
            U = rk_double(state);
            while (U > px)
            {
                X++;
                if (X > bound)
                {
                    X = 0;
                    px = qn;
                    U = rk_double(state);
                }
                else
                {
                    U -= px;
                    px = ((n - X + 1) * p * px) / (X * q);
                }
            }
            return X;
        }


        static long rk_binomial(rk_state state, long n, double p)
        {
            double q;

            if (p <= 0.5)
            {
                if (p * n <= 30.0)
                {
                    return rk_binomial_inversion(state, n, p);
                }
                else
                {
                    return rk_binomial_btpe(state, n, p);
                }
            }
            else
            {
                q = 1.0 - p;
                if (q * n <= 30.0)
                {
                    return n - rk_binomial_inversion(state, n, q);
                }
                else
                {
                    return n - rk_binomial_btpe(state, n, q);
                }
            }

        }

        static long rk_negative_binomial(rk_state state, double n, double p)
        {
            double Y;

            Y = rk_gamma(state, n, (1 - p) / p);
            return rk_poisson(state, Y);
        }


        static long rk_poisson_mult(rk_state state, double lam)
        {
            long X;
            double prod, U, enlam;

            enlam = Math.Exp(-lam);
            X = 0;
            prod = 1.0;
            while (true)
            {
                U = rk_double(state);
                prod *= U;
                if (prod > enlam)
                {
                    X += 1;
                }
                else
                {
                    return X;
                }
            }
        }


        /*
         * The transformed rejection method for generating Poisson random variables
         * W. Hoermann
         * Insurance: Mathematics and Economics 12, 39-45 (1993)
         */
        const double LS2PI = 0.91893853320467267;
        const double TWELFTH = 0.083333333333333333333333;
        static long rk_poisson_ptrs(rk_state state, double lam)
        {
            long k;
            double U, V, slam, loglam, a, b, invalpha, vr, us;

            slam = Math.Sqrt(lam);
            loglam = Math.Log(lam);
            b = 0.931 + 2.53 * slam;
            a = -0.059 + 0.02483 * b;
            invalpha = 1.1239 + 1.1328 / (b - 3.4);
            vr = 0.9277 - 3.6224 / (b - 2);

            while (true)
            {
                U = rk_double(state) - 0.5;
                V = rk_double(state);
                us = 0.5 - Math.Abs(U);
                k = (long)Math.Floor((2 * a / us + b) * U + lam + 0.43);
                if ((us >= 0.07) && (V <= vr))
                {
                    return k;
                }
                if ((k < 0) ||
                    ((us < 0.013) && (V > us)))
                {
                    continue;
                }
                if ((Math.Log(V) + Math.Log(invalpha) - Math.Log(a / (us * us) + b)) <=
                    (-lam + k * loglam - loggam(k + 1)))
                {
                    return k;
                }


            }

        }


        static long rk_poisson(rk_state state, double lam)
        {
            if (lam >= 10)
            {
                return rk_poisson_ptrs(state, lam);
            }
            else if (lam == 0)
            {
                return 0;
            }
            else
            {
                return rk_poisson_mult(state, lam);
            }
        }

        static double rk_standard_cauchy(rk_state state)
        {
            return rk_gauss(state) / rk_gauss(state);
        }


        static double rk_standard_t(rk_state state, double df)
        {
            double N, G, X;

            N = rk_gauss(state);
            G = rk_standard_gamma(state, df / 2);
            X = Math.Sqrt(df / 2) * N / Math.Sqrt(G);
            return X;
        }

        /* Uses the rejection algorithm compared against the wrapped Cauchy
           distribution suggested by Best and Fisher and documented in
           Chapter 9 of Luc's Non-Uniform Random Variate Generation.
           http://cg.scs.carleton.ca/~luc/rnbookindex.html
           (but corrected to match the algorithm in R and Python)
        */
        static double rk_vonmises(rk_state state, double mu, double kappa)
        {
            double s;
            double U, V, W, Y, Z;
            double result, mod;
            bool neg;

            if (kappa < 1e-8)
            {
                return Math.PI * (2 * rk_double(state) - 1);
            }
            else
            {
                /* with double precision rho is zero until 1.4e-8 */
                if (kappa < 1e-5)
                {
                    /*
                     * second order taylor expansion around kappa = 0
                     * precise until relatively large kappas as second order is 0
                     */
                    s = (1.0/ kappa + kappa);
                }
                else
                {
                    double r = 1 + Math.Sqrt(1 + 4 * kappa * kappa);
                    double rho = (r - Math.Sqrt(2 * r)) / (2 * kappa);
                    s = (1 + rho * rho) / (2 * rho);
                }

                while (true)
                {
                    U = rk_double(state);
                    Z = Math.Cos(Math.PI * U);
                    W = (1 + s * Z) / (s + Z);
                    Y = kappa * (s - W);
                    V = rk_double(state);
                    if ((Y * (2 - Y) - V >= 0) || (Math.Log(Y / V) + 1 - Y >= 0))
                    {
                        break;
                    }
                }

                U = rk_double(state);

                result = Math.Acos(W);
                if (U < 0.5)
                {
                    result = -result;
                }
                result += mu;
                neg = (result < 0);
                mod = Math.Abs(result);
                mod = (Math.IEEERemainder(mod + Math.PI, 2 * Math.PI) - Math.PI);
                if (neg)
                {
                    mod *= -1;
                }

                return mod;
            }
        }


        static double rk_pareto(rk_state state, double a)
        {
            return Math.Exp(rk_standard_exponential(state) / a) - 1;
        }

        static double rk_weibull(rk_state state, double a)
        {
            return Math.Pow(rk_standard_exponential(state), 1.0/ a);
        }

        static double rk_power(rk_state state, double a)
        {
            return Math.Pow(1 - Math.Exp(-rk_standard_exponential(state)), 1.0/ a);
        }


        static double rk_laplace(rk_state state, double loc, double scale)
        {
            double U;

            U = rk_double(state);
            if (U < 0.5)
            {
                U = loc + scale * Math.Log(U + U);
            }
            else
            {
                U = loc - scale * Math.Log(2.0 - U - U);
            }
            return U;
        }

        static double rk_gumbel(rk_state state, double loc, double scale)
        {
            double U;

            U = 1.0 - rk_double(state);
            return loc - scale * Math.Log(-Math.Log(U));
        }


        static double rk_logistic(rk_state state, double loc, double scale)
        {
            double U;

            U = rk_double(state);
            return loc + scale * Math.Log(U / (1.0 - U));
        }

        static double rk_lognormal(rk_state state, double mean, double sigma)
        {
            return Math.Exp(rk_normal(state, mean, sigma));
        }

        static double rk_rayleigh(rk_state state, double mode)
        {
            return mode * Math.Sqrt(-2.0 * Math.Log(1.0 - rk_double(state)));
        }

        static double rk_wald(rk_state state, double mean, double scale)
        {
            double U, X, Y;
            double mu_2l;

            mu_2l = mean / (2 * scale);
            Y = rk_gauss(state);
            Y = mean * Y * Y;
            X = mean + mu_2l * (Y - Math.Sqrt(4 * scale * Y + Y * Y));
            U = rk_double(state);
            if (U <= mean / (mean + X))
            {
                return X;
            }
            else
            {
                return mean * mean / X;
            }
        }


        static long rk_zipf(rk_state state, double a)
        {
            double am1, b;

            am1 = a - 1.0;
            b = Math.Pow(2.0, am1);
            while (true)
            {
                double T, U, V, X;

                U = 1.0 - rk_double(state);
                V = rk_double(state);
                X = Math.Floor(Math.Pow(U, -1.0 / am1));
                /*
                 * The real result may be above what can be represented in a signed
                 * long. Since this is a straightforward rejection algorithm, we can
                 * just reject this value. This function then models a Zipf
                 * distribution truncated to sys.maxint.
                 */
                if (X > Int64.MaxValue || X < 1.0)
                {
                    continue;
                }

                T = Math.Pow(1.0 + 1.0 / X, am1);
                if (V * X * (T - 1.0) / (b - 1.0) <= T / b)
                {
                    return (long)X;
                }
            }
        }


        static long rk_geometric_search(rk_state state, double p)
        {
            double U;
            long X;
            double sum, prod, q;

            X = 1;
            sum = prod = p;
            q = 1.0 - p;
            U = rk_double(state);
            while (U > sum)
            {
                prod *= q;
                sum += prod;
                X++;
            }
            return X;
        }

        static long rk_geometric_inversion(rk_state state, double p)
        {
            return (long)Math.Ceiling(Math.Log(1.0 - rk_double(state)) / Math.Log(1.0 - p));
        }

        static long rk_geometric(rk_state state, double p)
        {
            if (p >= 0.333333333333333333333333)
            {
                return rk_geometric_search(state, p);
            }
            else
            {
                return rk_geometric_inversion(state, p);
            }
        }


        static long rk_hypergeometric_hyp(rk_state state, long good, long bad, long sample)
        {
            long d1, K, Z;
            double d2, U, Y;

            d1 = bad + good - sample;
            d2 = (double)Math.Min(bad, good);

            Y = d2;
            K = sample;
            while (Y > 0.0)
            {
                U = rk_double(state);
                Y -= (long)Math.Floor(U + Y / (d1 + K));
                K--;
                if (K == 0) break;
            }
            Z = (long)(d2 - Y);
            if (good > bad) Z = sample - Z;
            return Z;
        }


        /* D1 = 2*sqrt(2/e) */
        /* D2 = 3 - 2*sqrt(3/e) */

        static long rk_hypergeometric_hrua(rk_state state, long good, long bad, long sample)
        {
            const double D1 = 1.7155277699214135;
            const double D2 = 0.8989161620588988;

            long mingoodbad, maxgoodbad, popsize, m, d9;
            double d4, d5, d6, d7, d8, d10, d11;
            long Z;
            double T, W, X, Y;

            mingoodbad = Math.Min(good, bad);
            popsize = good + bad;
            maxgoodbad = Math.Max(good, bad);
            m = Math.Min(sample, popsize - sample);
            d4 = ((double)mingoodbad) / popsize;
            d5 = 1.0 - d4;
            d6 = m * d4 + 0.5;
            d7 = Math.Sqrt((double)(popsize - m) * sample * d4 * d5 / (popsize - 1) + 0.5);
            d8 = D1 * d7 + D2;
            d9 = (long)Math.Floor((double)(m + 1) * (mingoodbad + 1) / (popsize + 2));
            d10 = (loggam(d9 + 1) + loggam(mingoodbad - d9 + 1) + loggam(m - d9 + 1) +
                   loggam(maxgoodbad - m + d9 + 1));
            d11 = Math.Min(Math.Min(m, mingoodbad) + 1.0, Math.Floor(d6 + 16 * d7));
            /* 16 for 16-decimal-digit precision in D1 and D2 */

            while (true)
            {
                X = rk_double(state);
                Y = rk_double(state);
                W = d6 + d8 * (Y - 0.5) / X;

                /* fast rejection: */
                if ((W < 0.0) || (W >= d11)) continue;

                Z = (long)Math.Floor(W);
                T = d10 - (loggam(Z + 1) + loggam(mingoodbad - Z + 1) + loggam(m - Z + 1) +
                           loggam(maxgoodbad - m + Z + 1));

                /* fast acceptance: */
                if ((X * (4.0 - X) - 3.0) <= T) break;

                /* fast rejection: */
                if (X * (X - T) >= 1) continue;

                if (2.0 * Math.Log(X) <= T) break;  /* acceptance */
            }

            /* this is a correction to HRUA* by Ivan Frohne in rv.py */
            if (good > bad) Z = m - Z;

            /* another fix from rv.py to allow sample to exceed popsize/2 */
            if (m < sample) Z = good - Z;

            return Z;
        }


        static long rk_hypergeometric(rk_state state, long good, long bad, long sample)
        {
            if (sample > 10)
            {
                return rk_hypergeometric_hrua(state, good, bad, sample);
            }
            else
            {
                return rk_hypergeometric_hyp(state, good, bad, sample);
            }
        }


        static double rk_triangular(rk_state state, double left, double mode, double right)
        {
            double _base, leftbase, ratio, leftprod, rightprod;
            double U;

            _base = right - left;
            leftbase = mode - left;
            ratio = leftbase / _base;
            leftprod = leftbase * _base;
            rightprod = (right - mode) * _base;

            U = rk_double(state);
            if (U <= ratio)
            {
                return left + Math.Sqrt(U * leftprod);
            }
            else
            {
                return right - Math.Sqrt((1.0 - U) * rightprod);
            }
        }


        static long rk_logseries(rk_state state, double p)
        {
            double q, r, U, V;
            long result;

            r = Math.Log(1.0 - p);

            while (true)
            {
                V = rk_double(state);
                if (V >= p)
                {
                    return 1;
                }
                U = rk_double(state);
                q = 1.0 - Math.Exp(r * U);
                if (V <= q * q)
                {
                    result = (long)Math.Floor(1 + Math.Log(V) / Math.Log(q));
                    if (result < 1)
                    {
                        continue;
                    }
                    else
                    {
                        return result;
                    }
                }
                if (V >= q)
                {
                    return 1;
                }
                return 2;
            }
        }

        #region randomkit




        /*
         * Slightly optimised reference implementation of the Mersenne Twister
         * Note that regardless of the precision of long, only 32 bit random
         * integers are produced
         */
        static ulong rk_random(rk_state state)
        {
            /* Magic Mersenne Twister constants */
            const int N = 624;
            const int M = 397;
            const long MATRIX_A = 0x9908b0df;
            const long UPPER_MASK = 0x80000000;
            const long LOWER_MASK = 0x7fffffff;

            long y;

            if (state.pos == rk_state.RK_STATE_LEN)
            {
                int i;

                
                for (i = 0; i < N - M; i++)
                {
                    y = (state.key[i] & UPPER_MASK) | (state.key[i + 1] & LOWER_MASK);
                    state.key[i] = state.key[i + M] ^ (y >> 1) ^ (-((y & 1) & MATRIX_A));
                }
                for (; i < N - 1; i++)
                {
                    y = (state.key[i] & UPPER_MASK) | (state.key[i + 1] & LOWER_MASK);
                    state.key[i] = state.key[i + (M - N)] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);
                }
                y = (state.key[N - 1] & UPPER_MASK) | (state.key[0] & LOWER_MASK);
                state.key[N - 1] = state.key[M - 1] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);

                state.pos = 0;
            }
            y = state.key[state.pos++];

            /* Tempering */
            y ^= (y >> 11);
            y ^= (y << 7) & 0x9d2c5680;
            y ^= (y << 15) & 0xefc60000;
            y ^= (y >> 18);

            return (ulong)y;
        }
        static UInt64 rk_uint64(rk_state state)
        {
            UInt64 upper = (UInt64)rk_random(state) << 32;
            UInt64 lower = (UInt64)rk_random(state);
            return upper | lower;
        }


        /*
         * Returns an unsigned 32 bit random integer.
         */
        static UInt32 rk_uint32(rk_state state)
        {
            return (UInt32)rk_random(state);
        }


        /*
         * Fills an array with cnt random npy_uint64 between off and off + rng
         * inclusive. The numbers wrap if rng is sufficiently large.
         */
        static void rk_random_uint64(UInt64 off, UInt64 rng, npy_intp cnt,
                         UInt64 []_out, rk_state state)
        {
            UInt64 val, mask = rng;
            npy_intp i;

            if (rng == 0)
            {
                for (i = 0; i < cnt; i++)
                {
                    _out[i] = off;
                }
                return;
            }

            /* Smallest bit mask >= max */
            mask |= mask >> 1;
            mask |= mask >> 2;
            mask |= mask >> 4;
            mask |= mask >> 8;
            mask |= mask >> 16;
            mask |= mask >> 32;

            for (i = 0; i < cnt; i++)
            {
                if (rng <= 0xffffffffUL)
                {
                    while ((val = (rk_uint32(state) & mask)) > rng) ;
                }
                else
                {
                    while ((val = (rk_uint64(state) & mask)) > rng) ;
                }
                _out[i] = off + val;
            }
        }


        /*
         * Fills an array with cnt random npy_uint32 between off and off + rng
         * inclusive. The numbers wrap if rng is sufficiently large.
         */
        static void rk_random_uint32(UInt32 off, UInt32 rng, npy_intp cnt,
                         UInt32[] _out, rk_state state)
        {
            UInt32 val, mask = rng;
            npy_intp i;

            if (rng == 0)
            {
                for (i = 0; i < cnt; i++)
                {
                    _out[i] = off;
                }
                return;
            }

            /* Smallest bit mask >= max */
            mask |= mask >> 1;
            mask |= mask >> 2;
            mask |= mask >> 4;
            mask |= mask >> 8;
            mask |= mask >> 16;

            for (i = 0; i < cnt; i++)
            {
                while ((val = (rk_uint32(state) & mask)) > rng) ;
                _out[i] = off + val;
            }
        }


        /*
         * Fills an array with cnt random npy_uint16 between off and off + rng
         * inclusive. The numbers wrap if rng is sufficiently large.
         */
        static void rk_random_uint16(UInt16 off, UInt16 rng, npy_intp cnt,
                         UInt16[] _out, rk_state state)
        {
            UInt16 val, mask = rng;
            npy_intp i;
            UInt32 buf = 0;
            int bcnt = 0;

            if (rng == 0)
            {
                for (i = 0; i < cnt; i++)
                {
                    _out[i] = off;
                }
                return;
            }

            /* Smallest bit mask >= max */
            mask |= (UInt16)(mask >> 1);
            mask |= (UInt16)(mask >> 2);
            mask |= (UInt16)(mask >> 4);
            mask |= (UInt16)(mask >> 8);

            for (i = 0; i < cnt; i++)
            {
                do
                {
                    if (bcnt == 0)
                    {
                        buf = rk_uint32(state);
                        bcnt = 1;
                    }
                    else
                    {
                        buf >>= 16;
                        bcnt--;
                    }
                    val = (UInt16)(buf & mask);
                } while (val > rng);
                _out[i] = (UInt16)(off + val);
            }
        }

        /*
         * Fills an array with cnt random npy_uint8 between off and off + rng
         * inclusive. The numbers wrap if rng is sufficiently large.
         */
        static void rk_random_uint8(byte off, byte rng, npy_intp cnt,
                        byte[] _out, rk_state state)
        {
            byte val, mask = rng;
            npy_intp i;
            UInt32 buf = 0;
            int bcnt = 0;

            if (rng == 0)
            {
                for (i = 0; i < cnt; i++)
                {
                    _out[i] = off;
                }
                return;
            }

            /* Smallest bit mask >= max */
            mask |= (byte)(mask >> 1);
            mask |= (byte)(mask >> 2);
            mask |= (byte)(mask >> 4);

            for (i = 0; i < cnt; i++)
            {
                do
                {
                    if (bcnt == 0)
                    {
                        buf = rk_uint32(state);
                        bcnt = 3;
                    }
                    else
                    {
                        buf >>= 8;
                        bcnt--;
                    }
                    val = (byte)(buf & mask);
                } while (val > rng);
                _out[i] = (byte)(off + val);
            }
        }


        /*
         * Fills an array with cnt random npy_bool between off and off + rng
         * inclusive.
         */
        static void rk_random_bool(bool off, bool rng, npy_intp cnt,
                        bool[] _out, rk_state state)
        {
            npy_intp i;
            UInt32 buf = 0;
            int bcnt = 0;

            if (rng == false)
            {
                for (i = 0; i < cnt; i++)
                {
                    _out[i] = off;
                }
                return;
            }

            /* If we reach here rng and mask are one and off is zero */
            System.Diagnostics.Debug.Assert(rng == true && off == false);
            for (i = 0; i < cnt; i++)
            {
                if (bcnt == 0)
                {
                    buf = rk_uint32(state);
                    bcnt = 31;
                }
                else
                {
                    buf >>= 1;
                    bcnt--;
                }
                _out[i] = (buf & 0x00000001) != 0;
            }
        }


        static long rk_long(rk_state state)
        {
            return (long)rk_ulong(state) >> 1;
        }

        static ulong rk_ulong(rk_state state)
        {
            return (rk_random(state) << 32) | (rk_random(state));
        }


        static ulong rk_interval(ulong max, rk_state state)
        {
            ulong mask = max, value;

            if (max == 0)
            {
                return 0;
            }
            /* Smallest bit mask >= max */
            mask |= mask >> 1;
            mask |= mask >> 2;
            mask |= mask >> 4;
            mask |= mask >> 8;
            mask |= mask >> 16;
            mask |= mask >> 32;

            /* Search a random value in [0..mask] <= max */
            if (max <= 0xffffffffUL)
            {
                while ((value = (rk_random(state) & mask)) > max) ;
            }
            else
            {
                while ((value = (rk_ulong(state) & mask)) > max) ;
            }

            return value;
        }


        static double rk_double(rk_state state)
        {
            /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
            long a = (long)rk_random(state) >> 5;
            long b = (long)rk_random(state) >> 6;
            return (a * 67108864.0 + b) / 9007199254740992.0;
        }


    //    static void rk_fill(byte buffer, size_t size, rk_state* state)
    //    {
    //        unsigned long r;
    //        unsigned char* buf = buffer;

    //        for (; size >= 4; size -= 4)
    //        {
    //            r = rk_random(state);
    //            *(buf++) = r & 0xFF;
    //            *(buf++) = (r >> 8) & 0xFF;
    //            *(buf++) = (r >> 16) & 0xFF;
    //            *(buf++) = (r >> 24) & 0xFF;
    //        }

    //        if (!size)
    //        {
    //            return;
    //        }
    //        r = rk_random(state);
    //        for (; size; r >>= 8, size--)
    //        {
    //            *(buf++) = (unsigned char)(r & 0xFF);
    //    }
    //}

    static double rk_gauss(rk_state state)
        {
            if (state.has_gauss)
            {
                double tmp = state.gauss;
                state.gauss = 0;
                state.has_gauss = false;
                return tmp;
            }
            else
            {
                double f, x1, x2, r2;

                do
                {
                    x1 = 2.0 * rk_double(state) - 1.0;
                    x2 = 2.0 * rk_double(state) - 1.0;
                    r2 = x1 * x1 + x2 * x2;
                }
                while (r2 >= 1.0 || r2 == 0.0);

                /* Box-Muller transform */
                f = Math.Sqrt(-2.0 * Math.Log(r2) / r2);
                /* Keep for next call */
                state.gauss = f * x1;
                state.has_gauss = true;
                return f * x2;
            }
        }

        #endregion

    }


}
