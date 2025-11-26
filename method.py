import copy
import timeit

import numpy as np
import pandas as pd
import pyod
import scipy
import scipy.stats
from numpy.linalg import norm
from scipy import stats
from scipy.special import kl_div


def dist(data, c=None):
    if c is None:
        c = np.mean(data, axis=0)
    d = np.linalg.norm(data - c, axis=1) ** 2
    return d


def normIt(data, m=None):
    nData = data.copy()
    if m is None:
        m = np.mean(nData)
    nData = nData - m
    nData = nData / np.linalg.norm(nData, axis=1, keepdims=True)
    return nData


def estShell(data):
    mean = np.mean(data, axis=0, keepdims=True)
    d = np.linalg.norm(data - mean, axis=1)
    var = np.mean(d)

    err = np.absolute(d - var)
    MAD = np.median(err)
    eSig = 1.4826 * MAD

    return mean, var, eSig


def projectMean(data, m, var):
    d = np.linalg.norm(data - m, axis=1)
    err = d - var
    return err


def robustMean(featTrain, globalMean, thres=1, numIter=10):

    feat = normIt(featTrain, globalMean)
    m_, var, eSig = estShell(feat)
    err = projectMean(feat, m_, var)
    mask = err > eSig * thres
    meanInlier = np.mean(featTrain[mask, :], axis=0)
    meanOutlier = np.mean(featTrain[~mask, :], axis=0)
    globalMean = (meanInlier + meanOutlier) / 2

    for _ in range(numIter):
        feat = normIt(featTrain, globalMean)
        m_, var, eSig = estShell(feat[mask])
        err = projectMean(feat, m_, var)
        mask = err > eSig * thres

        meanInlier = np.mean(featTrain[mask, :], axis=0)
        meanOutlier = np.mean(featTrain[~mask, :], axis=0)
        globalMean = (meanInlier + meanOutlier) / 2
        # newMean = meanOutlier
    return err


class DaDTAnomalyDetector:
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
        nData = nData / np.linalg.norm(nData, axis=1, keepdims=True)
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

    def cos_sim(self, data_norm, m=None):
        if m is None:
            m = np.mean(data_norm, axis=0)
        d = []
        for i in range(data_norm.shape[0]):
            cosine = np.dot(data_norm[i], m) / (norm(data_norm[i]) * norm(m))
            d.append(cosine)
        return d

    def dist(self, data, c=None):
        if c is None:
            c = np.mean(data, axis=0)
        d = np.linalg.norm(data - c, axis=1) ** 2
        return d

    def brayCurtis_dist(self, data, c=None):
        if c is None:
            c = np.mean(data, axis=0)
        d = np.sum(np.abs((data - c)), axis=1) / np.sum(np.abs((data + c)), axis=1)
        return d

    ## f_low
    def dadt_simple_(self, data, metric="l2"):
        data_ins, _ = self.normErgo(data)
        data_, _ = self.normZscore(data)

        ss = np.mean(np.abs((data_ - np.mean(data_, axis=0))), axis=1)
        ss_ = np.mean(np.abs((data_ + np.mean(data_, axis=0))), axis=1)

        if self.KL_(ss, ss_) < 0.05 and self.three_sigma(
            ss_, reverse=False
        ) > self.three_sigma(ss, reverse=True):
            score_dadt = self.brayCurtis_dist(data_)
        else:
            score_dadt = self.dist(data_ins)

        globalMean = np.mean(data_ins, axis=0)
        score_re = robustMean(data_ins, globalMean, thres=1, numIter=10)

        score_bc_norm = (score_dadt - np.min(score_dadt)) / (
            np.max(score_dadt) - np.min(score_dadt)
        )
        score_re_norm = (score_re - np.min(score_re)) / (
            np.max(score_re) - np.min(score_re)
        )
        score = (2 * score_re_norm + 1 * score_bc_norm) / 2
        return score

    def dadt_(self, data, metric="l2"):
        data_ins, _ = self.normErgo(data)
        data_, _ = self.normZscore(data)

        ss = np.mean(np.abs((data_ - np.mean(data_, axis=0))), axis=1)
        ss_ = np.mean(np.abs((data_ + np.mean(data_, axis=0))), axis=1)

        if self.KL_(ss, ss_) < 0.05 and self.three_sigma(
            ss_, reverse=False
        ) > self.three_sigma(ss, reverse=True):
            score_dadt = self.brayCurtis_dist(data_)
        else:
            score_dadt = self.dist(data_ins)

        globalMean = np.mean(data_ins, axis=0)
        score_re = robustMean(data_ins, globalMean, thres=1, numIter=10)

        sort_list_bc = np.argsort(score_dadt)
        sort_list_re = np.argsort(score_re)

        spearmanr_simi = stats.spearmanr(sort_list_bc, sort_list_re).statistic

        if round(spearmanr_simi, 2) >= 0.3:
            score = score_re
        elif round(spearmanr_simi, 2) >= 0.1:
            score_bc_norm = (score_dadt - np.min(score_dadt)) / (
                np.max(score_dadt) - np.min(score_dadt)
            )
            score_re_norm = (score_re - np.min(score_re)) / (
                np.max(score_re) - np.min(score_re)
            )
            score = (score_re_norm + score_bc_norm) / 2
        else:
            score = score_dadt

        return score

    def dadt_alt(self, data, metric="l2"):
        data_ins, _ = self.normErgo(data)
        data_, _ = self.normZscore(data)

        ss = np.mean(np.abs((data_ - np.mean(data_, axis=0))), axis=1)
        ss_ = np.mean(np.abs((data_ + np.mean(data_, axis=0))), axis=1)

        if self.KL_(ss, ss_) < 0.05 and self.three_sigma(
            ss_, reverse=False
        ) > self.three_sigma(ss, reverse=True):
            score_dadt = self.brayCurtis_dist(data_)
        else:
            score_dadt = self.dist(data_ins)

        globalMean = np.mean(data_ins, axis=0)
        score_re = robustMean(data_ins, globalMean, thres=1, numIter=10)

        sort_list_bc = np.argsort(score_dadt)
        sort_list_re = np.argsort(score_re)

        spearmanr_simi = stats.spearmanr(sort_list_bc, sort_list_re).statistic

        if round(spearmanr_simi, 2) >= 0.3:
            score = score_re
        elif round(spearmanr_simi, 2) >= 0.1:
            score_bc_norm = (score_dadt - np.min(score_dadt)) / (
                np.max(score_dadt) - np.min(score_dadt)
            )
            score_re_norm = (score_re - np.min(score_re)) / (
                np.max(score_re) - np.min(score_re)
            )
            score = (score_re_norm + score_bc_norm) / 2
        else:
            score = score_dadt

        return score, spearmanr_simi

    def predict_labels(self, data):
        scores, contamination_factor = self.dadt_alt(data)

        # this is the threshold calculation mentioned in the paper
        # basically, this sets the threshold such that the resultant outlier ratio
        # of our predictions matches the contamination factor calculate by the method
        threshold = np.sort(scores)[
            int(len(scores) * (1 - np.abs(contamination_factor)))
        ]
        # threshold = self.compute_boundary(scores)
        print(f"Threshold: {threshold}")
        labels = (scores > threshold).astype(int)
        return scores, labels


# CHATGPT possible implementation idea to have a constantly updating model (needs testing, not currently in use)--->
class OnlineFlexUOD:
    """
    Streaming version of FlexUOD.
    Maintains a rolling window of feature vectors,
    updates the scoring model continuously,
    and returns the anomaly score of the most recent sample.
    """

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
        nData = nData / np.linalg.norm(nData, axis=1, keepdims=True)
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

    def cos_sim(self, data_norm, m=None):
        if m is None:
            m = np.mean(data_norm, axis=0)
        d = []
        for i in range(data_norm.shape[0]):
            cosine = np.dot(data_norm[i], m) / (norm(data_norm[i]) * norm(m))
            d.append(cosine)
        return d

    def dist(self, data, c=None):
        if c is None:
            c = np.mean(data, axis=0)
        d = np.linalg.norm(data - c, axis=1) ** 2
        return d

    def brayCurtis_dist(self, data, c=None):
        if c is None:
            c = np.mean(data, axis=0)
        d = np.sum(np.abs((data - c)), axis=1) / np.sum(np.abs((data + c)), axis=1)
        return d

    def __init__(self, window=128, min_samples=20):
        self.window = window
        self.min_samples = min_samples
        self.buffer = []

    # ---- Normalization helpers ---- #

    def norm_z(self, X):
        return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    def norm_ergo(self, X):
        X = X - np.mean(X, axis=0)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    def bray_curtis(self, X, mean=None):
        if mean is None:
            mean = np.mean(X, axis=0)
        num = np.sum(np.abs(X - mean), axis=1)
        den = np.sum(np.abs(X + mean), axis=1) + 1e-8
        return num / den

    # ---- Main scoring method (adapted FlexUOD) ---- #

    def compute_scores(self, X):
        X_ins = self.norm_ergo(X)
        X_z = self.norm_z(X)

        ss = np.mean(np.abs(X_z - np.mean(X_z, axis=0)), axis=1)
        ss_ = np.mean(np.abs(X_z + np.mean(X_z, axis=0)), axis=1)

        # KL decision rule
        KL = stats.entropy(ss, ss_)

        if KL < 0.05:
            score_dadt = self.bray_curtis(X_z)
        else:
            score_dadt = np.linalg.norm(X_ins - np.mean(X_ins, axis=0), axis=1) ** 2

        # robust mean scoring
        gm = np.mean(X_ins, axis=0)
        score_re = robustMean(X_ins, gm)

        # choose based on ranking similarity
        r1 = np.argsort(score_dadt)
        r2 = np.argsort(score_re)
        spear = stats.spearmanr(r1, r2).statistic

        if spear >= 0.3:
            return score_re
        elif spear >= 0.1:
            bc = (score_dadt - np.min(score_dadt)) / (np.ptp(score_dadt) + 1e-8)
            re = (score_re - np.min(score_re)) / (np.ptp(score_re) + 1e-8)
            return (bc + re) / 2
        else:
            return score_dadt

    # ---- Public API ---- #

    def add_sample(self, feat):
        """
        Add a new D-dimensional feature vector.
        Returns anomaly score (float) or None if warming up.
        """
        self.buffer.append(feat)

        # maintain fixed window
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

        if len(self.buffer) < self.min_samples:
            return None  # Not enough data yet

        X = np.array(self.buffer)
        scores = self.compute_scores(X)

        return float(scores[-1])  # score of newest sample
