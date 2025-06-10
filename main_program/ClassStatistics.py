import math
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis 
import tensorflow as tf
import tensorflow_probability as tfp
from statsmodels.tsa.stattools import acf
sample_rate=22050
class RowStatistics:
    def __init__(self, arr):
        self.arr = arr     
        
    def mean(self):
        return np.mean(self.arr, axis=1)    
    
    def variance(self):
        return np.var(self.arr, axis=1, ddof=0)
    
    def range_(self):
        return np.ptp(self.arr, axis=1)      
    
    def skewness(self):
        return skew(self.arr, axis=1, bias=False)
    
    def kurt(self):
        return kurtosis(self.arr, axis=1, fisher=False, bias=False)
    
    def summary(self):
        return {
            "mfcc_mean": self.mean(),
            "mfcc_var": self.variance(),
            "mfcc_range": self.range_(),
            "mfcc_skew": self.skewness(),
            "mfcc_kurt": self.kurt()
        }
class Timeseriesfeatures:
    def __init__(self,serie,seconds):
        self.audio=serie
        self.sec=seconds
        
    def mean_second(self):
        self.mu=[]
        for i in range(int(20/self.sec)):
            adesso=np.mean(self.audio[math.floor(i*sample_rate*self.sec):math.floor((i+1)*sample_rate*self.sec)])
            self.mu.append(adesso)
        return self.mu
    
    def var_second(self):
        self.sigma=[]
        for i in range(int(20/self.sec)):
            adesso=np.var(self.audio[math.floor(i*sample_rate*self.sec):math.floor((i+1)*sample_rate*self.sec)])
            self.sigma.append(adesso)
        return self.sigma  
    
    def maxpool_second(self):
        self.maxpool=[]
        for i in range(int(20/self.sec)):
            adesso=np.max(self.audio[math.floor(i*sample_rate*self.sec):math.floor((i+1)*sample_rate*self.sec)])
            self.maxpool.append(adesso)
        return self.maxpool  
    
    def median_second(self):
        self.me=[]
        for i in range(int(20/self.sec)):
            adesso=np.median(self.audio[math.floor(i*sample_rate*self.sec):math.floor((i+1)*sample_rate*self.sec)])
            self.me.append(adesso)
        return self.me 
    
    def skew_second(self):
        self.sk=[]
        for i in range(int(20/self.sec)):
            adesso=pd.Series(self.audio[math.floor(i*sample_rate*self.sec):math.floor((i+1)*sample_rate*self.sec)]).skew()
            self.sk.append(adesso)
        return self.sk 
    
    def kurt_second(self):
        self.kurt=[]
        for i in range(int(20/self.sec)):
            adesso=pd.Series(self.audio[math.floor(i*sample_rate*self.sec):math.floor((i+1)*sample_rate*self.sec)]).kurtosis()
            self.kurt.append(adesso)
        return self.kurt 
    
    def acf_second(self):
        self.acf=[]
        for i in range(int(20/self.sec)):
            adesso=acf(self.audio[math.floor(i*sample_rate*self.sec):math.floor((i+1)*sample_rate*self.sec)],nlags=10)[1:11].tolist()
            self.acf.append(adesso)
        return self.acf
    
    def summary(self):
        return{
            "ts_mean": self.mean_second(),
            "ts_var": self.var_second(),
            "ts_range": self.median_second(),
            "ts_skew": self.skew_second(),
            "ts_kurt": self.kurt_second(),
            "ts_maxpool":self.maxpool_second()
            }

class TimeseriesFeaturesTF:
    def __init__(self, serie, seconds):

        self.audio = tf.convert_to_tensor(serie, dtype=tf.float32)
        self.sec = seconds
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * seconds)
        self.n_windows = tf.shape(self.audio)[0] // self.window_size
        
        
        self.window_starts = tf.range(self.n_windows) * self.window_size
        self.window_ends = (tf.range(self.n_windows) + 1) * self.window_size
        
    def _get_windows(self):
        pad_size = (self.window_size - (tf.shape(self.audio)[0] % self.window_size)) % self.window_size
        audio_padded = tf.pad(self.audio, [[0, pad_size]])  
        return tf.reshape(audio_padded, [self.n_windows, self.window_size])
    
    def mean_second(self):
        windows = self._get_windows()
        return tf.reduce_mean(windows, axis=1).numpy()
    
    def var_second(self):
        windows = self._get_windows()
        return tf.math.reduce_variance(windows, axis=1).numpy()
    
    def maxpool_second(self):
        windows = self._get_windows()
        return tf.reduce_max(windows, axis=1).numpy()
    
    def median_second(self):
        windows = self._get_windows()
        return tfp.stats.percentile(windows, 50.0, axis=1).numpy()
    
    def skew_second(self):
        windows = self._get_windows()
        mean = tf.reduce_mean(windows, axis=1, keepdims=True)
        std = tf.math.reduce_std(windows, axis=1, keepdims=True)
        skew = tf.reduce_mean(((windows - mean) / std)**3, axis=1)
        return skew.numpy()
    
    def kurt_second(self):
        windows = self._get_windows()
        mean = tf.reduce_mean(windows, axis=1, keepdims=True)
        std = tf.math.reduce_std(windows, axis=1, keepdims=True)
        kurt = tf.reduce_mean(((windows - mean) / std)**4, axis=1) - 3.0
        return kurt.numpy()
    
    def acf_second(self, nlags=10):
        windows = self._get_windows()
        acfs = []
        for i in range(self.n_windows):
            window = windows[i]
            acf = []
            for lag in range(1, nlags+1):
                cov = tf.reduce_mean(window[lag:] * window[:-lag])
                var = tf.math.reduce_variance(window)
                acf.append((cov / var).numpy())
            acfs.append(acf)
        return acfs
    
    def summary(self):
        return {
            "ts_mean": self.mean_second(),
            "ts_var": self.var_second(),
            "ts_range": self.median_second(),
            "ts_skew": self.skew_second(),
            "ts_kurt": self.kurt_second(),
            "ts_maxpool": self.maxpool_second()
        }

