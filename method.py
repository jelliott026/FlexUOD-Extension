import numpy as np
import pandas as pd
import timeit
import copy
import scipy
import pyod
import scipy.stats
from scipy import stats
from numpy.linalg import norm
from scipy.special import kl_div


def dist(data, c=None):
    if c is None:
        c = np.mean(data, axis=0)
    d = np.linalg.norm(data - c, axis=1)**2
    return d

def normIt(data, m=None):
    nData = data.copy()
    if m is None:
        m = np.mean(nData)
    nData = nData - m
    nData = nData / np.linalg.norm(nData, axis =1, keepdims=True)
    return nData 

def estShell(data):
    m_ = np.mean(data, axis=0, keepdims=True)
    d = np.linalg.norm(data - m_, axis=1)
    var = np.mean(d)

    err = np.absolute(d-var)
    MAD =  np.median(err)
    eSig = 1.4826*MAD

    return m_, var, eSig

def projectMean(data, m, var):
    d = np.linalg.norm(data - m, axis=1)
    err = d-var
    return err

def robustMean(featTrain, globalMean, thres=1, numIter=10):
    
    feat = normIt(featTrain, globalMean)
    m_, var, eSig = estShell(feat)
    err = projectMean(feat, m_, var)
    mask = err>eSig*thres
    meanInlier = np.mean(featTrain[mask,:], axis=0)
    meanOutlier = np.mean(featTrain[~mask,:], axis=0)
    globalMean = (meanInlier + meanOutlier)/2

    for i in range(numIter):
        feat = normIt(featTrain, globalMean)
        m_, var, eSig = estShell(feat[mask])
        err = projectMean(feat, m_, var)
        mask = err>eSig*thres

        meanInlier = np.mean(featTrain[mask,:], axis=0)
        meanOutlier = np.mean(featTrain[~mask,:], axis=0)
        globalMean = (meanInlier + meanOutlier)/2
        #newMean = meanOutlier
    return err


class DaDTAnomalyDetector():
    def normZscore(self, data, m=None):
        nData = data.copy()
        if m is None:
            m = np.mean(nData)
        nData = nData - m
        nData = nData / np.std(nData)
        return nData, m

    def normErgo(self, data, m=None):
        nData = data.copy()
        if m is None:
            m = np.mean(nData)
        nData = nData - m
        nData = nData / np.linalg.norm(nData, axis =1, keepdims=True)
        return nData, m
    
    def KL_(self, d1, d2):
        KL = scipy.stats.entropy(d1, d2)
        return KL

    def three_sigma(self, d, reverse=True):
        mean = np.mean(d)
        std = np.std(d)
        if reverse:
            thres = mean + 3 * std
        else:
            thres = mean - 3 * std
        return thres

    def cos_sim(self, data_norm, m = None):
        if m is None:
            m = np.mean(data_norm, axis = 0)
        d = []
        for i in range(data_norm.shape[0]):
            cosine = np.dot(data_norm[i], m)/(norm(data_norm[i])*norm(m))
            d.append(cosine)
        return d
    
    def dist(self, data, c=None):
        if c is None:
            c = np.mean(data, axis=0)
        d = np.linalg.norm(data - c, axis=1)**2
        return d

    def brayCurtis_dist(self, data, c=None):
        if c is None:
            c = np.mean(data, axis=0)
        d = np.sum(np.abs((data - c)), axis=1) / np.sum(np.abs((data + c)), axis=1)
        return d

    ## f_low
    def dadt_simple_(self, data, metric = 'l2'):
        data_ins, _ = self.normErgo(data)
        data_, _ = self.normZscore(data)
        
        ss = np.mean(np.abs((data_ - np.mean(data_, axis=0))), axis=1)
        ss_ = np.mean(np.abs((data_ + np.mean(data_, axis=0))), axis=1)
        
        if self.KL_(ss, ss_) < 0.05 and self.three_sigma(ss_, reverse=False) > self.three_sigma(ss, reverse=True):
            score_dadt = self.brayCurtis_dist(data_)
        else:
            score_dadt = self.dist(data_ins)
                   
        globalMean = np.mean(data_ins, axis = 0)
        score_re = robustMean(data_ins, globalMean, thres=1, numIter=10)

        score_bc_norm = (score_dadt-np.min(score_dadt))/(np.max(score_dadt)-np.min(score_dadt))
        score_re_norm = (score_re-np.min(score_re))/(np.max(score_re)-np.min(score_re))
        score = (2*score_re_norm+1*score_bc_norm)/2
        return score
    

    def dadt_(self, data, metric = 'l2'):
        data_ins, _ = self.normErgo(data)
        data_, _ = self.normZscore(data)
        
        ss = np.mean(np.abs((data_ - np.mean(data_, axis=0))), axis=1)
        ss_ = np.mean(np.abs((data_ + np.mean(data_, axis=0))), axis=1)
        
        if self.KL_(ss, ss_) < 0.05 and self.three_sigma(ss_, reverse=False) > self.three_sigma(ss, reverse=True):
            score_dadt = self.brayCurtis_dist(data_)
        else:
            score_dadt = self.dist(data_ins)
                 
        globalMean = np.mean(data_ins, axis = 0)
        score_re = robustMean(data_ins, globalMean, thres=1, numIter=10)
        
        sort_list_bc = np.argsort(score_dadt) 
        sort_list_re = np.argsort(score_re) 
        
        spearmanr_simi = stats.spearmanr(sort_list_bc, sort_list_re).statistic
        
        if round(spearmanr_simi, 2) >= 0.3:
            score = score_re
        elif round(spearmanr_simi, 2) >= 0.1:
            score_bc_norm = (score_dadt-np.min(score_dadt))/(np.max(score_dadt)-np.min(score_dadt))
            score_re_norm = (score_re-np.min(score_re))/(np.max(score_re)-np.min(score_re))
            score = (score_re_norm+score_bc_norm)/2
        else:
            score = score_dadt
            
        return score