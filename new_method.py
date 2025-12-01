import cupy as cp
import numpy as np
import scipy
import scipy.stats
from numpy.typing import ArrayLike, NDArray
from scipy import stats


def dist(data, c=None):
    if c is None:
        c = cp.mean(data, axis=0)
    d = cp.linalg.norm(data - c, axis=1) ** 2
    return d


def norm_it(data, mean=None):
    if mean is None:
        mean = cp.mean(data)
    n_data = data - mean
    n_data = n_data / cp.linalg.norm(n_data, axis=1, keepdims=True)
    return n_data


def est_shell(data):
    mean = cp.mean(data, axis=0, keepdims=True)
    d = cp.linalg.norm(data - mean, axis=1)
    var = cp.mean(d)

    err = cp.absolute(d - var)
    MAD = cp.median(err)
    e_sig = 1.4826 * MAD

    return mean, var, e_sig


def project_mean(data, m, var):
    d = cp.linalg.norm(data - m, axis=1)
    err = d - var
    return err


def robust_mean(feat_train, global_mean, thres=1, numIter=10):

    feat = norm_it(feat_train, global_mean)
    m_, var, e_sig = est_shell(feat)
    err = project_mean(feat, m_, var)
    mask = err > e_sig * thres

    mean_inlier = cp.mean(feat_train[mask, :], axis=0)
    mean_outlier = cp.mean(feat_train[~mask, :], axis=0)
    global_mean = (mean_inlier + mean_outlier) / 2

    for _ in range(numIter):
        feat = norm_it(feat_train, global_mean)

        # prevent errors from occurring if the masked features return an empty array
        if len(feat[mask]) == 0:
            return err

        m_, var, e_sig = est_shell(feat[mask])
        err = project_mean(feat, m_, var)
        mask = err > e_sig * thres

        mean_inlier = cp.mean(feat_train[mask, :], axis=0)
        mean_outlier = cp.mean(feat_train[~mask, :], axis=0)
        global_mean = (mean_inlier + mean_outlier) / 2

    return err


def norm_zscore(data, mean=None):
    if mean is None:
        mean = cp.mean(data)
    n_data = data - mean
    n_data = n_data / cp.std(n_data)
    return n_data, mean


def norm_ergo(data, mean=None):
    if mean is None:
        mean = cp.mean(data)
    n_data = data - mean
    n_data = n_data / cp.linalg.norm(n_data, axis=1, keepdims=True)
    return n_data, mean


def KL_(d1, d2):
    return scipy.stats.entropy(d1.get(), d2.get())


def three_sigma(d, reverse=True):
    mean = cp.mean(d)
    std = cp.std(d)
    if reverse:
        thres = mean + 3 * std
    else:
        thres = mean - 3 * std
    return thres


def cos_similarity(data_norm, mean=None):
    if mean is None:
        mean = cp.mean(data_norm, axis=0)
    d = []
    for i in range(data_norm.shape[0]):
        cosine = cp.dot(data_norm[i], mean) / (
            cp.linalg.norm(data_norm[i]) * cp.linalg.norm(mean)
        )
        d.append(cosine)
    return d


def dist(data, centroid=None):
    if centroid is None:
        centroid = cp.mean(data, axis=0)
    d = cp.linalg.norm(data - centroid, axis=1) ** 2
    return d


def bray_curtis_dist(data, center=None):
    if center is None:
        center = cp.mean(data, axis=0)
    d = cp.sum(cp.abs(data - center), axis=1) / cp.sum(cp.abs(data + center), axis=1)
    return d


def predict_scores(
    data: ArrayLike, cuda_out: bool = False
) -> tuple[NDArray[np.float32], float]:
    """Predicts the outlier scores for the given features.

    Args:
        data: The features to process. Should be a 2D array or something similar. Axis 0 corresponds to
            features and axis 1 feature elements.
        cuda_out: Whether to output cupy arrays (True) or numpy arrays (False). Defaults to False.

    Returns:
        The tuple (scores, contamination_factor). The scores element may either be a cupy or numpy array depending
            on the value of `cuda_out`.
    """
    data = cp.array(data)
    singular_mean = cp.mean(data)

    data_ins, _ = norm_ergo(data, singular_mean)
    data_zscore, _ = norm_zscore(data, singular_mean)

    data_zcenter = cp.mean(data_zscore, axis=0)

    ss = cp.mean(cp.abs((data_zscore - data_zcenter)), axis=1)
    ss_ = cp.mean(cp.abs((data_zscore + data_zcenter)), axis=1)

    if KL_(ss, ss_) < 0.05 and three_sigma(ss_, reverse=False) > three_sigma(
        ss, reverse=True
    ):
        score_dadt = bray_curtis_dist(data_zscore, data_zcenter)
    else:
        score_dadt = dist(data_ins, data_zcenter)

    data_ins_center = cp.mean(data_ins, axis=0)
    score_re = robust_mean(data_ins, data_ins_center, thres=1, numIter=10)

    sort_list_bc = cp.argsort(score_dadt)
    sort_list_re = cp.argsort(score_re)

    spearmanr_simi = stats.spearmanr(sort_list_bc.get(), sort_list_re.get()).statistic

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


def predict_labels(data: ArrayLike) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Implementation of labeling method mentioned in paper but not implemented.

    Effectively, we assume the given contamination factor is correct and select the
    outlier score within `outlier_scores` that results in a matching outlier ratio.

    Args:
        data: An array of features (see `predict_scores`).

    Returns:
        The tuple (outlier_scores, outlier_labels). Both arrays correspond directly with input data;
            a label of `1` corresponds to outliers and `0` inliers.
    """
    scores, contamination_factor = predict_scores(data, cuda_out=True)

    # this is the threshold calculation mentioned in the paper
    # basically, this sets the threshold such that the resultant outlier ratio
    # of our predictions matches the contamination factor calculate by the method
    threshold = cp.sort(scores)[int(scores.size * (1 - cp.abs(contamination_factor)))]
    labels = (scores > threshold).astype(np.int32)
    return scores.get(), labels.get()
