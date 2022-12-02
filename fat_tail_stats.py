import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Sample the sum of absolute deviation
def sample_ad(gen, sz, mean, sample_times = 500):
    t = [np.abs(gen(sz).sum() - mean * sz) for _ in range(sample_times)]
    return np.mean(t)

def Mn_Mn0(n, n0, gen, mean, sample_times=500):
    M_n = sample_ad(gen, n, mean, sample_times)
    M_n0 = sample_ad(gen, n0, mean, sample_times)
    return M_n / M_n0

def Mn_Mn0_range(n_range, gen, mean, n0=1, sample_times=500):
    mn_m0 = [Mn_Mn0(n, n0, gen, mean, sample_times) for n in n_range]

    kn_n0 = \
        [2 - (np.log(n_range[i]) - np.log(n0)) / np.log(mn_m0[i]) \
            for i in range(len(n_range))]
    return mn_m0, kn_n0

def kn_kn0(mn, mn0, n, n0):
    return 2 - (np.log10(n) - np.log10(n0)) / np.log10(mn / mn0)

def kappa_func(x, u):
    return x ** u

def kappa_metric(u):
    return 2 - 1 / u

def fit_kappa_curve(x, y):
    opt, _ = curve_fit(kappa_func, x, y)
    kappa = kappa_metric(opt[0])    
    return kappa, opt[0]

def ms_plot(v, label=' '):
    v = pd.Series(v).abs()
    c_sum = v.cumsum()
    c_max = v.cummax()
    plt.plot(np.abs(c_max / c_sum), label=label)

def mean_plot(v, label=' '):
    v = pd.Series(v)
    c_mean = v.cumsum() / np.arange(1, len(v)+1)
    plt.plot(c_mean, label=label)  

def cdf_plot(v, label=' '):
    v = np.sort(v)
    N = len(v)
    cdf =  np.arange(N) / N
    plt.plot(v, cdf, label=label)

def survival_log_plot(v, label=' '):
    v = np.log(np.sort(v))
    N = len(v)
    cdf = np.log(1 - np.arange(N) / N)
    plt.plot(v, cdf, '.', label=label)
    x = np.linspace(np.min(v), np.max(v), 10)
    y = np.linspace(np.min(cdf), np.max(cdf), 10)
    plt.xticks(x, [('%.1f%%' % (t * 100)) for t in np.exp(x)], rotation=70)
    plt.yticks(y, ['%.2f' % t for t in np.exp(y)])

class YahooData:
    def __init__(self, csv_path, symbol):
        self.csv_path = csv_path
        self.sym = symbol
        self.data = self.load(self.csv_path)
        self.daily_r = (self.data['Adj Close'].diff() / self.data['Adj Close'].shift(1))[1:]      
        self.daily_r_mean = self.daily_r.mean()
        self.daily_r_std = self.daily_r.std()
        self.mn_mn0_cache_ = None
        self.n_range_cache_ = None

    def load(self, path):
        return pd.read_csv(path)

    def Mn_Mn0(self, n, n0=1):
        return Mn_Mn0(n, n0,
            gen = lambda n: self.daily_r.sample(n),
            mean = self.daily_r_mean)

    def Mn_Mn0_range(self, n_range, n0=1):
        self.mn_mn0_cache_ = [self.Mn_Mn0(n, n0) for n in n_range]
        self.kn_n0_cache_ = \
            [2 - (np.log(self.n_range_cache_[i]) - self.log(n0)) / np.log(self.mn_mn0_cache_[i]) \
                for i in range(len(self.n_range_cache_))]
        self.n_range_cache_ = n_range
        return self.mn_mn0_cache_

    def kappa(self, n_range=None, n0=1):
        assert(n_range is not None or self.mn_mn0_cache_ is not None)
        if n_range is not None:
            self.Mn_Mn0_range(n_range, n0)
        self.kappa_, self.kappa_exp_ = fit_kappa_curve(self.n_range_cache_, self.mn_mn0_cache_)
        return self.kappa_

    def kappa_plot(self, n_range=None, n0=1):
        if n_range is not None:
            self.kappa(n_range, n0)
        plt.plot(self.n_range_cache_, self.mn_mn0_cache_, 'x')
        plt.plot(self.n_range_cache_,
            kappa_func(self.n_range_cache_, self.kappa_exp_),
            label="%s, k = %.2f" % (self.sym, self.kappa_))

    def cum_mean_plot(self, first_n=None):
        if first_n is None:
            first_n = len(self.daily_r)
        r = self.daily_r[:first_n]
        mean_plot(r, label=self.sym)

    def ms_plot(self, moment, n_range=None):
        if n_range is None:
            n_range = (0, len(self.daily_r))
        s, e = n_range       
        v = self.daily_r[s:e] ** moment
        ms_plot(v, label=self.sym)

    def neg_ms_plot(self, moment, n_range=None):
        if n_range is None:
            n_range = (0, len(self.daily_r))
        s, e = n_range
        v = self.daily_r.values
        v = v[v <= 0][s:e] ** moment
        ms_plot(v, label=self.sym)

    def pos_ms_plot(self, moment, n_range=None):
        if n_range is None:
            n_range = (0, len(self.daily_r))
        s, e = n_range
        v = self.daily_r.values
        v = v[v >= 0][s:e] ** moment
        ms_plot(v, label=self.sym)

    def z_score_plot(self, bins=100):
        z = stats.zscore(self.daily_r)
        self.z_score_ = z
        plt.hist(z, bins, histtype='step', label = '%s [%.2f, %.2f]' % (self.sym, z.min(), z.max()))             
