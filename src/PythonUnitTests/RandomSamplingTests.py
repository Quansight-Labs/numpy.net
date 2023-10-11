import unittest
import numpy as np
from nptest import nptest


class Test_test1(unittest.TestCase):

    def test_rand_1(self):

        np.random.seed(8765);

        f = np.random.rand()
        print(f)

        arr = np.random.rand(5000000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

    def test_randn_1(self):

        np.random.seed(1234);

        f = np.random.randn()
        print(f)

        arr = np.random.randn(5000000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

    def test_randbool_1(self):

        np.random.seed(8188);

        f = np.random.randint(False,True+1,4, dtype=np.bool)
        print(f)

        arr = np.random.randint(False,True+1,5000000, dtype=np.bool);
        cnt = arr == True
        print(cnt.size);

    def test_randint8_1(self):

        np.random.seed(9292);

        f = np.random.randint(2,3,4, dtype=np.int8)
        print(f)

        arr = np.random.randint(2,8,5000000, dtype=np.int8);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.randint(-2,3,5000000, dtype=np.int8);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_randuint8_1(self):

        np.random.seed(1313);

        f = np.random.randint(2,3,4, dtype=np.uint8)
        print(f)

        arr = np.random.randint(2,128,5000000, dtype=np.uint8);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

        first10 = arr[0:10:1]
        print(first10)

    def test_randint16_1(self):

        np.random.seed(8381);

        f = np.random.randint(2,3,4, dtype=np.int8)
        print(f)

        arr = np.random.randint(2,2478,5000000, dtype=np.int16);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.randint(-2067,3000,5000000, dtype=np.int16);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_randuint16_1(self):

        np.random.seed(5555);

        f = np.random.randint(2,3,4, dtype=np.uint16)
        print(f)

        arr = np.random.randint(23,12801,5000000, dtype=np.uint16);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

        first10 = arr[0:10:1]
        print(first10)

    def test_randint_1(self):

        np.random.seed(701);

        f = np.random.randint(2,3,4, dtype=np.int32)
        print(f)

        arr = np.random.randint(9,128000,5000000, dtype=np.int32);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.randint(-20000,300000,5000000, dtype=np.int32);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_randuint_1(self):

        np.random.seed(8357);

        f = np.random.randint(2,3,4, dtype=np.uint32)
        print(f)

        arr = np.random.randint(29,13000,5000000, dtype=np.uint32);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

        first10 = arr[0:10:1]
        print(first10)

    def test_randint64_1(self):

        np.random.seed(10987);

        f = np.random.randint(2,3,4, dtype=np.int64)
        print(f)

        arr = np.random.randint(20,9999999,5000000, dtype=np.int64);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.randint(-9999999,9999999,5000000, dtype=np.int64);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_randuint64_1(self):

        np.random.seed(1990);

        f = np.random.randint(2,3,4, dtype=np.uint64)
        print(f)

        arr = np.random.randint(64,64000,5000000, dtype=np.uint64);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

        first10 = arr[0:10:1]
        print(first10)
    
    def test_rand_shuffle_1(self):

        np.random.seed(1964);

        arr = np.arange(10);
        np.random.shuffle(arr);
        print(arr);

        arr = np.arange(10).reshape((-1,1));
        print(arr);

        np.random.shuffle(arr);
        print(arr);

    def test_rand_permutation_1(self):

        np.random.seed(1963);

        arr = np.random.permutation(10);
        print(arr);

        arr = arr = np.random.permutation(np.arange(5));
        print(arr);


        
    def test_beta_1(self):

        np.random.seed(5566);

        a = np.arange(1,11, dtype=np.float64);
        b = np.arange(1,11, dtype= np.float64);

        arr = np.random.beta(b, b, 10);
        print(arr);

       
    def test_rand_binomial_1(self):

        np.random.seed(123)

        arr = np.random.binomial(9, 0.1, 20);
        s = np.sum(arr== 0);
        print(s);
        print(arr);

        arr = np.random.binomial(9, 0.1, 20000);
        s = np.sum(arr== 0);
        print(s)

    def test_rand_negative_binomial_1(self):

        np.random.seed(123)

        arr = np.random.negative_binomial(1, 0.1, 20);
        s = np.sum(arr== 0);
        print(s);
        print(arr);

        arr = np.random.negative_binomial(1, 0.1, 20000);
        s = np.sum(arr== 0);
        print(s)

    def test_rand_chisquare_1(self):

        np.random.seed(904)

        arr = np.random.chisquare(2, 40);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.chisquare(np.arange(1,(25*25)+1), 25*25);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_dirichlet_1(self):

        np.random.seed(904)

        arr = np.random.dirichlet((2,20), 40);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.dirichlet((25,1,25), 25*25);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_exponential_1(self):

        np.random.seed(914)

        arr = np.random.exponential(2.0, 40);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.exponential([1.75, 2.25, 3.5, 4.1], 4);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.exponential(1.75, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_exponential_2(self):

        np.random.seed(914)

        arr = np.random.exponential();
 
        print(arr)


    def test_rand_f_1(self):

        np.random.seed(94)

        arr = np.random.f(1, 48, 1000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.f([1.75, 2.25, 3.5, 4.1], 48, 4);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.f(1.75, 53, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_gamma_1(self):

        np.random.seed(99)

        arr = np.random.gamma([4,4], 2);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.gamma([1.75, 2.25, 3.5, 4.1], 48, 4);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.gamma(1.75, 53, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_geometric_1(self):

        np.random.seed(101)

        arr = np.random.geometric(0.35);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        #first10 = arr[0:10:1]
        #print(first10)

        arr = np.random.geometric([.75, .25, .5, .1], [100, 4]);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.geometric(.75, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)


    def test_rand_gumbel_1(self):

        np.random.seed(1431)

        arr = np.random.gumbel(0.32);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        #first10 = arr[0:10:1]
        #print(first10)

        arr = np.random.gumbel([.75, .25, .5, .1], [4]);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.gumbel(.75, 0.5, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_hypergeometric_1(self):

        np.random.seed(1631)

        arr = np.random.hypergeometric(100, 2, 10, 1000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.hypergeometric([75, 25, 5, 1], [5], [80, 30, 10, 6]);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.hypergeometric(15, 15, 15, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_laplace_1(self):

        np.random.seed(1400)

        arr = np.random.laplace(0.32);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        #first10 = arr[0:10:1]
        #print(first10)

        arr = np.random.laplace([.75, .25, .5, .1], [4]);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.laplace(.75, 0.5, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_logistic_1(self):

        np.random.seed(1400)

        arr = np.random.logistic(0.32);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        #first10 = arr[0:10:1]
        #print(first10)

        arr = np.random.logistic([.75, .25, .5, .1], [4]);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.logistic(.75, 0.5, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_lognormal_1(self):

        np.random.seed(990)

        arr = np.random.lognormal([4,4], 2);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.lognormal([1.75, 2.25, 3.5, 4.1], 48, 4);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.lognormal(1.75, 53, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_logseries_1(self):

        np.random.seed(9909)

        arr = np.random.logseries([0.1,0.99], [20, 2]);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.logseries([.75, .25, .5, .1], [400, 4]);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.logseries(.334455, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_multinomial_1(self):

        np.random.seed(9909)

        arr = np.random.multinomial(20, [1/6.]*6, size=1000)
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.multinomial(100, [1/7.]*5 + [2/7.]);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.multinomial(20, [1/6.]*6, size=20000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_noncentral_chisquare_1(self):

        np.random.seed(904)

        arr = np.random.noncentral_chisquare(3, 20, 100000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.noncentral_chisquare(np.arange(1,(25*25)+1), 25*25);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_noncentral_f_1(self):

        np.random.seed(95)

        arr = np.random.noncentral_f(1, 20, 48, 1000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.noncentral_f([1.75, 2.25, 3.5, 4.1], 20, 48, 4);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.noncentral_f(1.75, 3, 53, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_normal_1(self):

        np.random.seed(96)

        arr = np.random.normal(1, 48, 1000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.normal([1.75, 2.25, 3.5, 4.1], 48, 4);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.normal(1.75, 53, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_pareto_1(self):

        np.random.seed(993)

        arr = np.random.pareto(3.0, 1000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.pareto([1.75, 2.25, 3.5, 4.1], 4);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.pareto(1.75, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)


    def test_rand_poisson_1(self):

        np.random.seed(993)

        arr = np.random.poisson(3.0, 1000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.poisson([1.75, 2.25, 3.5, 4.1], 4);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.poisson(1.75, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)


    def test_rand_power_1(self):

        np.random.seed(339)

        arr = np.random.power(3.0, 1000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.power([1.75, 2.25, 3.5, 4.1], 4);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.power(1.75, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_rayleigh_1(self):

        np.random.seed(340)

        arr = np.random.rayleigh(3.0, 1000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.rayleigh([1.75, 2.25, 3.5, 4.1], 4);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.rayleigh(1.75, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_standard_cauchy_1(self):

        np.random.seed(341)

        arr = np.random.standard_cauchy(1000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.standard_cauchy((40, 40, 40));
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        #print(first10)

        arr = np.random.standard_cauchy(200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_standard_exponential_1(self):

        np.random.seed(342)

        arr = np.random.standard_exponential(1000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.standard_exponential((40, 40, 40));
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        #print(first10)

        arr = np.random.standard_exponential(200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_standard_gamma_1(self):

        np.random.seed(343)

        arr = np.random.standard_gamma(2, 1000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.standard_gamma(4, (40, 40, 40));
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        #print(first10)

        arr = np.random.standard_gamma(.25, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_standard_normal_1(self):

        np.random.seed(8877);
        arr = np.random.standard_normal(5000000);
        print(np.max(arr));
        print(np.min(arr));
        print(np.average(arr));

        first10 = arr[0:10:1]
        print(first10)


    def test_rand_standard_t_1(self):

        np.random.seed(344)

        arr = np.random.standard_t(10, 1000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.standard_t(40, (40, 40, 40));
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        #print(first10)

        arr = np.random.standard_t(20000, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_triangular_1(self):

        np.random.seed(967)

        arr = np.random.triangular(1, 20, 48, 1000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.triangular([1.75, 2.25, 3.5, 4.1], 20, 48, 4);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.triangular(1.75, 3, 53, 200000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_uniform_1(self):

        np.random.seed(5461);
        arr = np.random.uniform(-1, 1, 5000000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

        first10 = arr[0:10:1]
        print(first10)

    def test_rand_uniform_2(self):

        np.random.seed(5461);
        low = np.array([9.0, 8.0, 7.0, 1.0])
        high = np.array([30.0, 22.0, 10.0, 3.0])
        shape = (4,)
        arr = np.random.uniform(low, high, shape)
      
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));

        first10 = arr[0:10:1]
        print(first10)

    def test_rand_vonmises_1(self):

        np.random.seed(909)

        arr = np.random.vonmises(0.0, 4.0, 100000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.vonmises(np.arange(1,(25*25)+1), 25*25);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_wald_1(self):

        np.random.seed(964)

        arr = np.random.wald(3, 20, 100000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.wald(np.arange(1,(25*25)+1), 25*25);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        
    def test_rand_weibull_1(self):

        np.random.seed(974)

        arr = np.random.weibull(5, 100000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.weibull(np.arange(1,(25*25)+1), 25*25);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

       
    def test_rand_zipf_1(self):

        np.random.seed(979)

        arr = np.random.zipf(5.2, 100000);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

        arr = np.random.zipf(np.arange(2,(25*25)+2), 25*25);
        print(np.amax(arr));
        print(np.amin(arr));
        print(np.average(arr));
        first10 = arr[0:10:1]
        print(first10)

    def test_rand_choice_1(self):


        np.random.seed(979)

        a = np.random.choice(2)
        print(a)

        b = np.random.choice(65)
        print(b)

        b1 = np.random.choice(65, size = [3,4,5] )
        print(b1)

        #c = np.random.choice(-1)
        #print(c)

    def test_rand_choice_2(self):


        np.random.seed(979)

        a = np.random.choice([22,33,44])
        print(a)



    def test_rand_choice_3(self):


        np.random.seed(979)

        a = np.random.choice(5,3)
        print(a)

        b = np.random.choice(5,3, p=[0.1, 0, 0.3, 0.6, 0])
        print(b)
    
    def test_rand_choice_4(self):


        np.random.seed(979)

        x = np.arange(1,9)

        a = np.random.choice(x,3)
        print(a)

        b = np.random.choice(x,3, p=[0.1, 0, 0.3, 0.2, 0.1, 0.2, 0.0, 0.1])
        print(b)

    def test_rand_choice_5(self):

        np.random.seed(979)

        x = aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']

        a = np.random.choice(x,5)
        print(a)

        b = np.random.choice(x,5, p=[0.5, 0.1, 0.1, 0.3])
        print(b)

    def test_rand_choice_6(self):


        np.random.seed(979)

        x = np.arange(1,9)

        a = np.random.choice(x,3, replace=False)
        print(a)

        b = np.random.choice(x,3, replace = False, p=[0.1, 0, 0.3, 0.2, 0.1, 0.2, 0.0, 0.1])
        print(b)

if __name__ == '__main__':
    unittest.main()
