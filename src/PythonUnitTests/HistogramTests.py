import unittest
import numpy as np
from nptest import nptest


class HistogramTests(unittest.TestCase):

#region bincount
    def test_bincount_1(self):

        x = np.arange(5)
        a = np.bincount(x)
        print(a)

        x = np.array([0, 1, 1, 3, 2, 1, 7])
        a = np.bincount(x)
        print(a)

        x = np.array([0, 1, 1, 3, 2, 1, 7, 23])
        a = np.bincount(x)
        print(a)

        print(a.size == np.amax(x)+1)

    def test_bincount_2(self):

        x = np.arange(5, dtype=np.int64)
        a = np.bincount(x)
        print(a)

        x = np.array([0, 1, 1, 3, 2, 1, 7], dtype=np.int16)
        a = np.bincount(x)
        print(a)

        x = np.array([0, 1, 1, 3, 2, 1, 7, 23], dtype=np.int8)
        a = np.bincount(x)
        print(a)

        print(a.size == np.amax(x)+1)

    def test_bincount_3(self):

        w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
        x = np.arange(6, dtype=np.int64)
        a = np.bincount(x, weights=w)
        print(a)

        x = np.array([0, 1, 3, 2, 1, 7], dtype=np.int16)
        a = np.bincount(x,weights=w)
        print(a)

        x = np.array([0, 1, 3, 2, 1, 7], dtype=np.int8)
        a = np.bincount(x, weights=w)
        print(a)

    def test_bincount_4(self):

        x = np.arange(5, dtype=np.int64)
        a = np.bincount(x, minlength=8)
        print(a)

        x = np.array([0, 1, 1, 3, 2, 1, 7], dtype=np.int16)
        a = np.bincount(x, minlength=10)
        print(a)

        x = np.array([0, 1, 1, 3, 2, 1, 7, 23], dtype=np.int8)
        a = np.bincount(x, minlength=32)
        print(a)

        print(a.size == np.amax(x)+1)

    def test_bincount_slice(self):

        w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6, .19, -0.8, 0.3, 0.5 ]) # weights
        x = np.arange(10, dtype=np.int64)
        a = np.bincount(x[::2], weights=w[::2])
        print(a)
  

    def test_bincount_uint64(self):

        try :
            x = np.arange(5, dtype=np.uint64)
            a = np.bincount(x)
            print(a)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_bincount_double(self):

        try :
            x = np.arange(5, dtype=np.float64)
            a = np.bincount(x)
            print(a)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_bincount_not1d(self):

        try :
            x = np.arange(100, dtype=np.int64).reshape(10,10);
            a = np.bincount(x)
            print(a)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
#endregion

#region digitize
    def test_digitize_1(self):

        x = np.array([0.2, 6.4, 3.0, 1.6])
        bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        inds = np.digitize(x, bins)
        print(inds)

  
  
    def test_digitize_2(self):

        x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
        bins = np.array([0, 5, 10, 15, 20])

        inds = np.digitize(x, bins, right=True)
        print(inds)
  
        inds = np.digitize(x, bins, right=False)
        print(inds)
  
    def test_digitize_3(self):

        x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
        bins = np.array([20, 15, 10, 5, 0])

        inds = np.digitize(x, bins, right=True)
        print(inds)
  
        inds = np.digitize(x, bins, right=False)
        print(inds)
#endregion

#region Histogram


    def test_histogram_1(self):

        x = np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
        print(x)

 
        x= np.histogram(np.arange(4), bins=np.arange(5), density=True)
        print(x)

        x= np.histogram(np.arange(4), bins=np.arange(5), density=False)
        print(x)

        x = np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])
        print(x)

    def test_histogram_1a(self):

        x = np.histogram([1, 2.0, 1.0], bins=[0.0, 1.0, 2.0, 3.0])
        print(x)

 
        x= np.histogram(np.arange(4, dtype = np.float), bins=np.arange(5, dtype=np.float64), density=True)
        print(x)

        x= np.histogram(np.arange(4, dtype=np.int16), bins=np.arange(5, dtype=np.float64), density=False)
        print(x)

        x = np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])
        print(x)

    def test_histogram_2(self):

        x = np.histogram([1, 2, 1], bins=4)
        print(x)

 
        x= np.histogram(np.arange(4), bins=5, density=True)
        print(x)

        x= np.histogram(np.arange(4), bins=5, density=False)
        print(x)

        x = np.histogram([[1, 2, 1], [1, 0, 1]], bins=8)
        print(x)

    def test_histogram_3(self):

        x = np.histogram([1, 2, 1], bins="auto")
        print(x)

 
        x= np.histogram(np.arange(4), bins="doane", density=True)
        print(x)

        x= np.histogram(np.arange(4), bins="fd", density=False)
        print(x)

        x = np.histogram([[1, 2, 1], [1, 0, 1]], bins="rice")
        print(x)

        x= np.histogram(np.arange(4), bins="scott", density=True)
        print(x)

        x= np.histogram(np.arange(4), bins="sqrt", density=False)
        print(x)

        x = np.histogram([[1, 2, 1], [1, 0, 1]], bins="sturges")
        print(x)


    def test_histogram_4(self):


        x= np.histogram(np.arange(40),  bins=np.arange(5), range=(15.0, 30.0), density=True)
        print(x)

        x= np.histogram(np.arange(40),  bins=np.arange(4), range=(15.0, 30.0), density=False)
        print(x)

        x= np.histogram(np.arange(40), bins=6, range=(15.0, 30.0), density=True)
        print(x)

        x= np.histogram(np.arange(40), bins=4, range=(15.0, 30.0), density=False)
        print(x)


    def test_histogram_5(self):

        weights = np.arange(40, dtype=np.float64);
        weights.fill(0.5)

        x= np.histogram(np.arange(40),  bins=np.arange(5), range=(15.0, 30.0), weights=weights, density=True)
        print(x)

        x= np.histogram(np.arange(40),  bins=np.arange(4), range=(15.0, 30.0),  weights=weights, density=False)
        print(x)

        x= np.histogram(np.arange(40), bins=6, range=(15.0, 30.0),  weights=weights, density=True)
        print(x)

        x= np.histogram(np.arange(40), bins=4, range=(15.0, 30.0),  weights=weights, density=False)
        print(x)

#endregion

#region histogram_bin_edges


    def test_histogram_bin_edges_1(self):

        arr = np.arange(40, dtype=np.float64);

        x = np.histogram_bin_edges(arr, bins='auto', range=(10, 30))
        print(x)

        x = np.histogram_bin_edges(arr, bins=4, range=(10, 30))
        print(x)

        x = np.histogram_bin_edges(arr, bins=np.arange(5), range=(10, 30))
        print(x)

#endregion

#region histogramdd


    def test_histogramdd_1(self):

        np.random.seed(8765);
        r = np.random.randint(10, 30, 3000)

        x = np.histogramdd(r.reshape(-1,4), bins=[2,2,2,2])
        print(x)

        x = nptest.histogramdd(r.reshape(-1,4), bins=[2.0,2.0,2.0,2.0])
        print(x)
 
        x= np.histogramdd(r.reshape(-1,2), bins=[3,3], density=True)
        print(x)

        x= np.histogramdd(r.reshape(-1,3), bins=[4,4,4], density=False)
        print(x)

 
    def test_histogramdd_2(self):

        np.random.seed(8765);
        r = np.random.randint(10, 30, 300000)

        x = nptest.histogramdd(r.reshape(-1,4), bins=[2,2,2,2], range=[(15, 25), (15,25), (15,25), (15,25)])
        print(x)

        x = nptest.histogramdd(r.reshape(-1,4), bins=[2.0,2.0,2.0,2.0], range=[(20, 20), (20,20), (20,20), (20,20)])
        print(x)
 
        x= np.histogramdd(r.reshape(-1,2), bins=[3,3], density=True, range=[(15, 25), (15,25)])
        print(x)

        x= np.histogramdd(r.reshape(-1,3), bins=[4,4,4], density=False, range=[(15, 25), (15,25), (15,25)])
        print(x)


    def test_histogramdd_3(self):

        np.random.seed(8765);
        r = np.random.randint(10, 30, 300000)

        weights = np.arange(300000/4, dtype=np.float64);
        weights.fill(0.5)

        x = nptest.histogramdd(r.reshape(-1,4), bins=[2,2,2,2], range=[(15, 25), (15,25), (15,25), (15,25)], weights=weights)
        print(x)

  
    def test_histogramdd_4(self):

        np.random.seed(8765);
        r = np.random.randint(10, 30, 300000)

        x = nptest.histogramdd(r.reshape(-1,4), bins=3, range=[(15, 25), (15,25), (15,25), (15,25)])
        print(x)

        print("")
        print("*******************")
        print("")

        x= nptest.histogramdd(r.reshape(-1,2), bins=2, normed=True, range=[(15, 25), (15,25)])
        print(x)

    def test_histogramdd_5(self):

        np.random.seed(8765);
        r = np.random.randint(10, 30, 300000)

        x = nptest.histogramdd(r.reshape(-1,4), bins= np.array([2,2,2,2]), range=[(15, 25), (15,25), (15,25), (15,25)])
        print(x)

        x = nptest.histogramdd(r.reshape(-1,4), bins= np.array([2.0,2.0,2.0,2.0]), range=[(20, 20), (20,20), (20,20), (20,20)])
        print(x)
 
        x= np.histogramdd(r.reshape(-1,2), bins= np.array([3,3]), density=True, range=[(15, 25), (15,25)])
        print(x)

        x= np.histogramdd(r.reshape(-1,3), bins= np.array([4,4,4]), density=False, range=[(15, 25), (15,25), (15,25)])
        print(x)


#endregion

#region histogram2d

    def test_histogram2d_1(self):

        np.random.seed(8765);
        x = np.random.normal(2, 1, 100)
        y = np.random.normal(1, 1, 100)

        xedges = [0, 1, 3, 5]
        yedges = [0, 2, 3, 4, 6]

        weights = np.arange(300000/4, dtype=np.float64);
        weights.fill(0.5)

        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        print(H)
        print(xedges)
        print(yedges)
        
   

#endregion

    def test_histogram_prep_1(self):

        x = np.arange(1, 21, dtype=np.int64)

        binsize = nptest._hist_bin_sqrt(x);
        print(binsize);

        binsize = nptest._hist_bin_sturges(x);
        print(binsize);

        binsize = nptest._hist_bin_rice(x);
        print(binsize);

        binsize = nptest._hist_bin_scott(x);
        print(binsize);

        binsize = nptest._hist_bin_doane(x);
        print(binsize);

        binsize = nptest._hist_bin_fd(x);
        print(binsize);

        binsize = nptest._hist_bin_auto(x);
        print(binsize);



if __name__ == '__main__':
    unittest.main()
