import unittest
import numpy as np
from nptest import nptest

class StatisticsTests(unittest.TestCase):

    #region amin/amax
    def test_amin_1(self):

        a = np.arange(1,5).reshape((2,2))
        print(a)
        print("*****")

        b = np.amin(a)           # Minimum of the flattened array
        print(b)
        print("*****")

        c = np.amin(a, axis=0)   # Minima along the first axis
        print(c)
        print("*****")

        d = np.amin(a, axis=1)   # Minima along the second axis
        print(d)
        print("*****")

        e = np.arange(5, dtype=float)
        e[2] = np.NaN
        f = np.amin(e)
        print(f)
        print("*****")

        g = np.nanmin(b)
        print(g)

    def test_amin_2(self):

        a = np.arange(30.25,46.25).reshape((4,4))
        print(a)
        print("*****")

        b = np.amin(a)           # Minimum of the flattened array
        print(b)
        print("*****")

        c = np.amin(a, axis=0)   # Minima along the first axis
        print(c)
        print("*****")

        d = np.amin(a, axis=1)   # Minima along the second axis
        print(d)
        print("*****")

        e = np.arange(5, dtype=float)
        e[2] = np.NaN
        f = np.amin(e)
        print(f)
        print("*****")

        g = np.nanmin(b)
        print(g)

    def test_amax_1(self):

        a = np.arange(4).reshape((2,2))
        print(a)
        print("*****")

        b = np.amax(a)           # Maximum of the flattened array
        print(b)
        print("*****")

        c = np.amax(a, axis=0)   # Maxima along the first axis
        print(c)
        print("*****")

        d = np.amax(a, axis=1)   # Maxima along the second axis
        print(d)
        print("*****")

        e = np.arange(5, dtype=float)
        e[2] = np.NaN
        f = np.amax(e)
        print(f)
        print("*****")

        g = np.nanmax(b)
        print(g)

    def test_amax_2(self):

        a = np.arange(30.25,46.25).reshape((4,4))
        print(a)
        print("*****")

        b = np.amax(a)           # Maximum of the flattened array
        print(b)
        print("*****")

        c = np.amax(a, axis=0)   # Maxima along the first axis
        print(c)
        print("*****")

        d = np.amax(a, axis=1)   # Maxima along the second axis
        print(d)
        print("*****")

        e = np.arange(5, dtype=float)
        e[2] = np.NaN
        f = np.amax(e)
        print(f)
        print("*****")

        g = np.nanmax(b)
        print(g)
    #endregion

    #region nanmin/nanmax
        # see NANFunctionsTests
    #endregion

    #region ptp
    def test_ptp_1(self):

        a = np.arange(4).reshape((2,2))
        print(a)
        print("*****")

        b = np.ptp(a, axis=0)
        print(b)
        print("*****")

        c = np.ptp(a, axis=1)
        print(c)

        d = np.ptp(a)
        print(d)

    #endregion

    #region percentile/quantile

    def test_percentile_1(self):
  

        a = np.array([[10.0, 7.0, 4.0], [3.0, 2.0, 1.0]])

        b = np.percentile(a, 50)
        print(b)

        c = np.percentile(a, 50, axis=0)
        print(c)

        d = np.percentile(a, 50, axis=1)
        print(d)

        e = np.percentile(a, 50, axis=1, keepdims=True)
        print(e)

        m = np.percentile(a, 50, axis=0)
        n = np.zeros_like(m)
        o = np.percentile(a, 50, axis=0, out=n)
        print(o)
        print(n)

        b = a.copy()
        c = np.percentile(b, 50, axis=1, overwrite_input=True)
        print(c)

        assert not np.all(a == b)

        return

    def test_percentile_2(self):

        a = np.array([[10, 7, 4], [3, 2, 1]])

        b = np.percentile(a, [50, 75])
        print(b)

        c = np.percentile(a, [50, 75], axis=0)
        print(c)

        d = np.percentile(a, [50, 75], axis=1)
        print(d)

        e = np.percentile(a, [50, 75], axis=1, keepdims=True)
        print(e)

        m = np.percentile(a, [50, 75], axis=0)
        n = np.zeros_like(m)
        o = np.percentile(a, [50, 75], axis=0, out=n)
        print(o)
        print(n)

        b = a.copy()
        c = np.percentile(b, [50, 75], axis=1, overwrite_input=True)
        print(c)

        assert not np.all(a == b)

        return


    def test_quantile_1(self):

        a = np.array([[10, 7, 4], [3, 2, 1]])

        b = np.quantile(a, 0.5)
        print(b)

        c = np.quantile(a, 0.5, axis=0)
        print(c)

        d = np.quantile(a, 0.5, axis=1)
        print(d)

        e = np.quantile(a, 0.5, axis=1, keepdims=True)
        print(e)


        m = np.quantile(a, 0.5, axis=0)
        out = np.zeros_like(m)
        np.quantile(a, 0.5, axis=0, out=out)

        print(out)
        print(m)

        b = a.copy()
        c = np.quantile(b, 0.5, axis=1, overwrite_input=True)
        print(c)

        assert not np.all(a == b)
        return

    #endregion

    #region nanpercentile/nanquantile
        # see NANFunctionsTests
    #endregion

    #region median/average/mean

    def test_median_1(self):

        a = np.array([[10, 7, 4], [3, 2, 1]])

        b = np.median(a)
        print(b)

        c = np.median(a, axis=0)
        print(c)

        d = np.median(a, axis=1)
        print(d)

        m = np.median(a, axis=0)
        out = np.zeros_like(m)
        n = np.median(a, axis=0, out=m)
        print(n)
        print(m)

        b = a.copy()
        c = np.median(b, axis=1, overwrite_input=True)
        print(c)

        assert not np.all(a==b)

        b = a.copy()
        c = np.median(b, axis=None, overwrite_input=True)
        print(c)

        assert not np.all(a==b)

        return

    def test_median_2(self):

        shape = [1,2,3,4,5,6,7,8]
        shape2 = shape[:4]


        a = np.arange(0,64,1).reshape(4,4,4)
        #nd = a.ndim
        #axis = [0,2]

        #keep = set(range(nd)) - set(axis)
        #nkeep = len(keep)
        ## swap axis that should not be reduced to front
        #for i, s in enumerate(sorted(keep)):
        #    a = a.swapaxes(i, s);
        ## merge reduced axis
        #a = a.reshape(a.shape[:nkeep] + (-1,))
     
        #keepdim = tuple(keepdim)


        b = np.median(a,axis= [0,2], keepdims = True)
        print(b)

        c = np.median(a, axis= [0,1], keepdims = True)
        print(c)

        d = np.median(a, axis=[1,2], keepdims = True)
        print(d)

  
        return

    def test_median_3(self):

        a = np.array([[10.0, 7.2, 4.2], [3.2, 2.2, 1.2]])

        b = np.median(a)
        print(b)

        c = np.median(a, axis=0)
        print(c)

        d = np.median(a, axis=1)
        print(d)

        m = np.median(a, axis=0)
        out = np.zeros_like(m)
        n = np.median(a, axis=0, out=m)
        print(n)
        print(m)

        b = a.copy()
        c = np.median(b, axis=1, overwrite_input=True)
        print(c)

        assert not np.all(a==b)

        b = a.copy()
        c = np.median(b, axis=None, overwrite_input=True)
        print(c)

        assert not np.all(a==b)

        return

    def test_average_1(self):

        x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.uint32).reshape(3, 2, -1)
        x = x * 3
        y = np.average(x);


        print(x)
        print(y)

        return

    
    def test_average_2(self):

        a = np.array([1,2,3,4,5,6,7,8,9,10]);
        w = [10,9,8,7,6,5,4,3,2,1]

        x = np.average(a)
        print(x)
        print("********")

        x = np.average(a, weights=w)
        print(x)
        print("********")

        x = np.average(a.reshape(2,-1), weights=np.array(w).reshape(2,-1))
        print(x)
        print("********")

        x = np.average(a.reshape(2,-1), axis = 0, weights=np.array(w).reshape(2,-1))
        print(x)
        print("********")

        x = np.average(a.reshape(2,-1), axis = 1, weights=np.array(w).reshape(2,-1))
        print(x)
        print("********")

        x = np.average(a.reshape(1,2, -1,1), axis = 1, weights=np.array(w).reshape(1,2, -1,1))
        print(x)
        print("********")

        x = np.average(a.reshape(1,-1,2,1), axis = 1, weights=np.array(w).reshape(1,-1,2,1))
        print(x)
        print("********")

        x = np.average(a.reshape(2,-1,1,1), axis = 1, weights=np.array(w).reshape(2,-1,1,1))
        print(x)


        return

    def test_average_3(self):

        a = np.array([1,2,3,4,5,6,7,8,9,10]);
        w = [10,9,8,7,6,5,4,3,2,1]

        x = np.average(a, returned = True)
        print(x)
        print("********")

        x = np.average(a, weights=w, returned = True)
        print(x)
        print("********")

        x = np.average(a.reshape(2,-1), weights=np.array(w).reshape(2,-1), returned = True)
        print(x)
        print("********")

        x = np.average(a.reshape(2,-1), axis = 0, weights=np.array(w).reshape(2,-1), returned = True)
        print(x)
        print("********")

        x = np.average(a.reshape(2,-1), axis = 1, weights=np.array(w).reshape(2,-1), returned = True)
        print(x)
        print("********")

        x = np.average(a.reshape(1,2, -1,1), axis = 1, weights=np.array(w).reshape(1,2, -1,1), returned = True)
        print(x)
        print("********")

        x = np.average(a.reshape(1,-1,2,1), axis = 1, weights=np.array(w).reshape(1,-1,2,1), returned = True)
        print(x)
        print("********")

        x = np.average(a.reshape(2,-1,1,1), axis = 1, weights=np.array(w).reshape(2,-1,1,1), returned = True)
        print(x)


        return

    def test_mean_1(self):

        x = np.array([10,15,25,45,78,90,10,15,25,45,78,90], dtype= np.uint32).reshape(3, 2, -1)
        x = x * 3
        print(x)

        y = np.mean(x);
        print(y)

        y = np.mean(x, axis=0);
        print(y)

        y = np.mean(x, axis=1);
        print(y)

        y = np.mean(x, axis=2);
        print(y)

        return

    def test_mean_2(self):

        a = np.zeros((2, 512*512), dtype=np.float32)
        a[0, :] = 1.0
        a[1, :] = 0.1
        b = np.mean(a)
        print(b)

        c = np.mean(a, dtype=np.float64)
        print(c)


    #endregion

    #region std/var
 
    def test_std_1(self):

        a = np.array([[1, 2], [3, 4]])
        b = np.std(a)
        print(b)

        c = np.std(a, axis=0)
        print(c)

        d = np.std(a, axis=1)
        print(d)

        #In single precision, std() can be inaccurate:

        a = np.zeros((2, 512*512), dtype=np.float32)
        a[0, :] = 1.0
        a[1, :] = 0.1
        b = np.std(a)
        print(b)

        # Computing the standard deviation in float64 is more accurate:

        c = np.std(a, dtype=np.float64)
        print(c)

    def test_var_1(self):

        a = np.array([[1, 2], [3, 4]])
        b = np.var(a)
        print(b)

        c = np.var(a, axis=0)
        print(c)

        d = np.var(a, axis=1)
        print(d)

        #In single precision, std() can be inaccurate:

        a = np.zeros((2, 512*512), dtype=np.float32)
        a[0, :] = 1.0
        a[1, :] = 0.1
        b = np.var(a)
        print(b)

        # Computing the standard deviation in float64 is more accurate:

        c = np.var(a, dtype=np.float64)
        print(c)

    #endregion

    #region nanmedian/nanmean
        # see NANFunctionsTests
    #endregion

    #region nanstd/nanvar
        # see NANFunctionsTests
    #endregion

    #region Correlating

              
    def test_correlate_1(self): 
        
        a = np.correlate([1, 2, 3], [0, 1, 0.5])
        print(a)

        b = np.correlate([1, 2, 3], [0, 1, 0.5], "same")
        print(b)

        c = np.correlate([1, 2, 3], [0, 1, 0.5], "full")
        print(c)

        return

    def test_correlate_BIGINT(self): 
        
        a = np.correlate([1, 2, 3], [0, 1, 5])
        print(a)

        b = np.correlate([1, 2, 3], [0, 1, 5], "same")
        print(b)

        c = np.correlate([1, 2, 3], [0, 1, 5], "full")
        print(c)

        return
 
    def test_cov_1(self): 

        x1 = np.array([[0, 2], [1, 1], [2, 0]]).T
        print(x1)

        # Note how  increases while  decreases. The covariance matrix shows this clearly:

        a = np.cov(x1)
        print(a)

        x = [-2.1, -1,  4.3]
        y = [3,  1.1,  0.12]
        X = np.stack((x, y), axis=0)
        a = np.cov(X)
        print(a)

        b = np.cov(x, y)
        print(b)

        c = np.cov(x)
        print(c)

        d = np.cov(X, rowvar=False)
        print(d)

        e = np.cov(X, rowvar=False, bias=True)
        print(e)

        f = np.cov(X, rowvar=False, bias=True, fweights = [1,2])
        print(f)

        g = np.cov(X, rowvar=False, bias=True, fweights = [1,2], aweights = [1,2])
        print(g)


        return

    def test_cov_1_BIGINT(self): 

        x1 = np.array([[0, 2], [1, 1], [2, 0]]).T
        print(x1)

        # Note how  increases while  decreases. The covariance matrix shows this clearly:

        a = np.cov(x1)
        print(a)

        x = [-21, -1,  43]
        y = [3,  11,  12]
        X = np.stack((x, y), axis=0)
        a = np.cov(X)
        print(a)

        b = np.cov(x, y)
        print(b)

        c = np.cov(x)
        print(c)

        d = np.cov(X, rowvar=False)
        print(d)

        e = np.cov(X, rowvar=False, bias=True)
        print(e)

        f = np.cov(X, rowvar=False, bias=True, fweights = [1,2])
        print(f)

        g = np.cov(X, rowvar=False, bias=True, fweights = [1,2], aweights = [1,2])
        print(g)


        return

    def test_corrcoef_1(self): 

        x1 = np.array([[0, 2], [1, 1], [2, 0]]).T
        print(x1)

        # Note how  increases while  decreases. The covariance matrix shows this clearly:

        a = np.corrcoef(x1)
        print(a)

        x = [-2.1, -1,  4.3]
        y = [3,  1.1,  0.12]
        X = np.stack((x, y), axis=0)
        a = np.corrcoef(X)
        print(a)

        b = np.corrcoef(x, y)
        print(b)

        c = np.corrcoef(x,y, rowvar=False)
        print(c)

        return

    #endregion

    #region Histograms

    #endregion

if __name__ == '__main__':
    unittest.main()
