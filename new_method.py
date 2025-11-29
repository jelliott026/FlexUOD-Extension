import copy
import timeit

import cupy as cp
import numpy as np
import pyod
import scipy
import scipy.stats
from numpy.linalg import norm
from scipy import stats
from scipy.special import kl_div


def dist(data, c=None):
    if c is None:
        c = cp.mean(data, axis=0)
    d = cp.linalg.norm(data - c, axis=1) ** 2
    return d


def normIt(data, m=None):
    if m is None:
        m = cp.mean(data)
    nData = data - m
    nData = nData / cp.linalg.norm(nData, axis=1, keepdims=True)
    return nData


def estShell(data):
    mean = cp.mean(data, axis=0, keepdims=True)
    d = cp.linalg.norm(data - mean, axis=1)
    var = cp.mean(d)

    err = cp.absolute(d - var)
    MAD = cp.median(err)
    eSig = 1.4826 * MAD

    return mean, var, eSig


def projectMean(data, m, var):
    d = cp.linalg.norm(data - m, axis=1)
    err = d - var
    return err


def robustMean(featTrain, globalMean, thres=1, numIter=10):

    feat = normIt(featTrain, globalMean)
    m_, var, eSig = estShell(feat)
    err = projectMean(feat, m_, var)
    mask = err > eSig * thres

    meanInlier = cp.mean(featTrain[mask, :], axis=0)
    meanOutlier = cp.mean(featTrain[~mask, :], axis=0)
    globalMean = (meanInlier + meanOutlier) / 2

    for _ in range(numIter):
        feat = normIt(featTrain, globalMean)
        m_, var, eSig = estShell(feat[mask])
        err = projectMean(feat, m_, var)
        mask = err > eSig * thres

        meanInlier = cp.mean(featTrain[mask, :], axis=0)
        meanOutlier = cp.mean(featTrain[~mask, :], axis=0)
        globalMean = (meanInlier + meanOutlier) / 2
        # newMean = meanOutlier
    return err


class DaDTAnomalyDetector:
    def normZscore(self, data, m=None):
        if m is None:
            m = cp.mean(data)
        nData = data - m
        nData = nData / cp.std(nData)
        return nData, m

    def normErgo(self, data, m=None):
        if m is None:
            m = cp.mean(data)
        nData = data - m
        nData = nData / cp.linalg.norm(nData, axis=1, keepdims=True)
        return nData, m

    def KL_(self, d1, d2):
        return scipy.stats.entropy(d1.get(), d2.get())

    def three_sigma(self, d, reverse=True):
        mean = cp.mean(d)
        std = cp.std(d)
        if reverse:
            thres = mean + 3 * std
        else:
            thres = mean - 3 * std
        return thres

    def cos_sim(self, data_norm, m=None):
        if m is None:
            m = cp.mean(data_norm, axis=0)
        d = []
        for i in range(data_norm.shape[0]):
            cosine = cp.dot(data_norm[i], m) / (norm(data_norm[i]) * norm(m))
            d.append(cosine)
        return d

    def dist(self, data, c=None):
        if c is None:
            c = cp.mean(data, axis=0)
        d = cp.linalg.norm(data - c, axis=1) ** 2
        return d

    def brayCurtis_dist(self, data, c=None):
        if c is None:
            c = cp.mean(data, axis=0)
        d = cp.sum(cp.abs(data - c), axis=1) / cp.sum(cp.abs(data + c), axis=1)
        return d

    ## f_low
    def dadt_simple_(self, data, metric="l2"):
        data_ins, _ = self.normErgo(data)
        data_, _ = self.normZscore(data)

        ss = cp.mean(cp.abs((data_ - cp.mean(data_, axis=0))), axis=1)
        ss_ = cp.mean(cp.abs((data_ + cp.mean(data_, axis=0))), axis=1)

        if self.KL_(ss, ss_) < 0.05 and self.three_sigma(
            ss_, reverse=False
        ) > self.three_sigma(ss, reverse=True):
            score_dadt = self.brayCurtis_dist(data_)
        else:
            score_dadt = self.dist(data_ins)

        globalMean = cp.mean(data_ins, axis=0)
        score_re = robustMean(data_ins, globalMean, thres=1, numIter=10)

        score_bc_norm = (score_dadt - cp.min(score_dadt)) / (
            cp.max(score_dadt) - cp.min(score_dadt)
        )
        score_re_norm = (score_re - cp.min(score_re)) / (
            cp.max(score_re) - cp.min(score_re)
        )
        score = (2 * score_re_norm + 1 * score_bc_norm) / 2
        return score

    def dadt_(self, data, metric="l2"):
        data_ins, _ = self.normErgo(data)
        data_, _ = self.normZscore(data)

        ss = cp.mean(cp.abs(data_ - cp.mean(data_, axis=0)), axis=1)
        ss_ = cp.mean(cp.abs(data_ + cp.mean(data_, axis=0)), axis=1)

        if self.KL_(ss, ss_) < 0.05 and self.three_sigma(
            ss_, reverse=False
        ) > self.three_sigma(ss, reverse=True):
            score_dadt = self.brayCurtis_dist(data_)
        else:
            score_dadt = self.dist(data_ins)

        globalMean = cp.mean(data_ins, axis=0)
        score_re = robustMean(data_ins, globalMean, thres=1, numIter=10)

        sort_list_bc = cp.argsort(score_dadt)
        sort_list_re = cp.argsort(score_re)

        spearmanr_simi = stats.spearmanr(sort_list_bc, sort_list_re).statistic

        if round(spearmanr_simi, 2) >= 0.3:
            score = score_re
        elif round(spearmanr_simi, 2) >= 0.1:
            score_bc_norm = (score_dadt - cp.min(score_dadt)) / (
                cp.max(score_dadt) - cp.min(score_dadt)
            )
            score_re_norm = (score_re - cp.min(score_re)) / (
                cp.max(score_re) - cp.min(score_re)
            )
            score = (score_re_norm + score_bc_norm) / 2
        else:
            score = score_dadt

        return score

    def dadt_alt(self, data, metric="l2", cuda_out=False):
        data = cp.array(data)
        singularMean = cp.mean(data)

        data_ins, _ = self.normErgo(data, singularMean)
        data_, _ = self.normZscore(data, singularMean)

        dataMean = cp.mean(data_, axis=0)

        ss = cp.mean(cp.abs((data_ - dataMean)), axis=1)
        ss_ = cp.mean(cp.abs((data_ + dataMean)), axis=1)

        if self.KL_(ss, ss_) < 0.05 and self.three_sigma(
            ss_, reverse=False
        ) > self.three_sigma(ss, reverse=True):
            score_dadt = self.brayCurtis_dist(data_, dataMean)
        else:
            score_dadt = self.dist(data_ins, dataMean)

        globalMean = cp.mean(data_ins, axis=0)
        score_re = robustMean(data_ins, globalMean, thres=1, numIter=10)

        sort_list_bc = cp.argsort(score_dadt)
        sort_list_re = cp.argsort(score_re)

        spearmanr_simi = stats.spearmanr(
            sort_list_bc.get(), sort_list_re.get()
        ).statistic

        if round(spearmanr_simi, 2) >= 0.3:
            score = score_re
        elif round(spearmanr_simi, 2) >= 0.1:
            dadt_min = cp.min(score_dadt)
            re_min = cp.min(score_re)
            score_bc_norm = (score_dadt - dadt_min) / (cp.max(score_dadt) - dadt_min)
            score_re_norm = (score_re - re_min) / (cp.max(score_re) - re_min)
            score = (score_re_norm + score_bc_norm) / 2
        else:
            score = score_dadt

        if cuda_out:
            return score, spearmanr_simi
        else:
            return score.get(), spearmanr_simi

    def predict_labels(self, data):
        scores, contamination_factor = self.dadt_alt(data, cuda_out=True)

        # this is the threshold calculation mentioned in the paper
        # basically, this sets the threshold such that the resultant outlier ratio
        # of our predictions matches the contamination factor calculate by the method
        threshold = cp.sort(scores)[
            int(scores.size * (1 - cp.abs(contamination_factor)))
        ]
        # threshold = self.compute_boundary(scores)
        # print(f"Threshold: {threshold}")
        labels = (scores > threshold).astype(int)
        return scores.get(), labels.get()
