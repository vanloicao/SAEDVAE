# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:06:19 2016

@author: caoloi
"""
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
import scipy.spatial
import numpy as np
"******************************* CENTROID *************************************"
class CentroidBasedOneClassClassifier:
    def __init__(self, threshold = 0.95, metric="euclidean", scale = "standard"):
        
        self.threshold = threshold
        """only CEN used StandardScaler because the centroid of training set need
        to be move to origin"""
        self.scaler = preprocessing.StandardScaler()                
        self.metric = metric

    def fit(self, X):
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        # because we are using StandardScaler, the centroid is a
        # vector of zeros, but we save it in shape (1, n) to allow
        # cdist to work happily later.
        self.centroid = np.zeros((1, X.shape[1]))
        # no need to scale again
        dists = self.get_density(X, scale=False) 
        # transform relative threshold (eg 95%) to absolute
        self.abs_threshold = np.percentile(dists, 100 * self.threshold)
    #It is actually the mean of the distances from each points in training set
    #to the centroid zero.
    def get_density(self, X, scale=True):
        if scale:
            X = self.scaler.transform(X)
        dists = scipy.spatial.distance.cdist(X, self.centroid, metric=self.metric)
        dists = np.mean(dists, axis=1)
        return dists
   
    def predict(self, X):
        dists = self.get_density(X)
        return dists > self.abs_threshold
        
"****************** CENTROID WITHOUT STANDARD SCALER *************************"
class Centroid_Classifier:
    def __init__(self, threshold = 0.95, metric="euclidean"):
        
        self.threshold = threshold
        """only CEN used StandardScaler because the centroid of training set need
        to be move to origin"""
        #self.scaler = preprocessing.StandardScaler()                
        self.metric = metric

    def fit(self, X):
        #self.scaler.fit(X)
        #X = self.scaler.transform(X)
        # because we are using StandardScaler, the centroid is a
        # vector of zeros, but we save it in shape (1, n) to allow
        # cdist to work happily later.
        self.centroid = np.zeros((1, X.shape[1]))
        # no need to scale again
        dists = self.get_density(X) 
        # transform relative threshold (eg 95%) to absolute
        self.abs_threshold = np.percentile(dists, 100 * self.threshold)
    #It is actually the mean of the distances from each points in training set
    #to the centroid zero.
    def get_density(self, X):
        #if scale:
        #    X = self.scaler.transform(X)
        dists = scipy.spatial.distance.cdist(X, self.centroid, metric=self.metric)
        dists = np.mean(dists, axis=1)
        return dists
   
    def predict(self, X):
        dists = self.get_density(X)
        return dists > self.abs_threshold



        
"************************** NEGATIVE DISTANCE ********************************"
class NegativeMeanDistance:    
    def __init__(self, metric="euclidean"):
        self.metric = metric
        
    def fit(self, X):
        self.X = X
       
    def score_samples(self, X):
        dists = scipy.spatial.distance.cdist(X, self.X, metric=self.metric)
        return -np.mean(dists, axis=1)
     
     
"*************************** DENSITY *****************************************"
class DensityBasedOneClassClassifier:
    def __init__(self, threshold=0.95, 
                 kernel="gaussian", 
                 bandwidth=1.0,
                 metric="euclidean", 
                 should_downsample=False, 
                 downsample_count=1000,
                 scale = "standard"):

        self.should_downsample = should_downsample
        self.downsample_count = downsample_count
        self.threshold = threshold
        
        #Load dataset, standard or maxabs[-1,1], minmax[0,1], No
        if (scale == "standard"):
            self.scaler = preprocessing.StandardScaler()
        elif (scale == "minmax"):    
            self.scaler = preprocessing.MinMaxScaler()
        elif (scale == "maxabs"):
            self.scaler = preprocessing.MaxAbsScaler()
        
        if kernel == "really_linear":
            self.dens = NegativeMeanDistance(metric=metric)
        else:
            self.dens = KernelDensity(bandwidth=bandwidth, kernel=kernel, metric=metric)

    def fit(self, X):
        # scale
        self.scaler.fit(X)
        self.X = self.scaler.transform(X)
        
        # downsample?
        if self.should_downsample:
            self.X = self.downsample(self.X, self.downsample_count)
    
        self.dens.fit(self.X)
        # transform relative threshold (eg 95%) to absolute
        dens = self.get_density(self.X, scale=False) # no need to scale again
        self.abs_threshold = np.percentile(dens, 100 * (1 - self.threshold))

    def get_density(self, X, scale=True):
        if scale:
            X = self.scaler.transform(X)
        # in negative log-prob (for KDE), in negative distance (for NegativeMeanDistance)
        return self.dens.score_samples(X)

    def predict(self, X):
        dens = self.get_density(X)
        return dens < self.abs_threshold # in both KDE and NMD, lower values are more anomalous

    def downsample(self, X, n):
        # we've already fit()ted, but we're worried that our X is so
        # large our classifier will be too slow in practice. we can
        # downsample by running a kde on X and sampling from it (this
        # will be slow, but happens only once), and then using those
        # points as the new X.
        if len(X) < n:
            return X
        kde = KernelDensity()
        kde.fit(X)
        return kde.sample(n)