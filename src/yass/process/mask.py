from scipy.stats import chi2
import numpy as np


def getmask(score, group, mask_th, n_features):
    """
    """
    th = 1.5*(chi2.ppf(mask_th, 1)*n_features)
    n_data, n_features, n_channels = score.shape
    if n_data > 0:
        n_group = np.max(group) + 1

        # find the average score per group
        score_group = np.zeros((n_group, n_features, n_channels))
        n_per_group = np.zeros(n_group)
        for j in range(n_data):
            score_group[group[j]] += score[j]
            n_per_group[group[j]] += 1
        for j in range(n_group):
            score_group[j] = score_group[j]/n_per_group[j]

        # find mask for each averaged score
        maskTemp = np.minimum(np.maximum(
            ((np.sum(np.square(score_group), axis=1) - np.min(th))/(np.max(th)-np.min(th))), 0), 1)

        # match the mask per group to each data
        mask = np.zeros((n_data, n_channels))
        for j in range(n_data):
            mask[j] = maskTemp[group[j]]

    return mask

def getmask_depreciated(score, group, mask_th, n_features, n_channels, do_coreset):
    """
    """
    th = 1.5*(chi2.ppf(mask_th, 1)*n_features)
    mask_all = list()

    for c in range(n_channels):

        if score[c].shape[0] > 0:
            ndata, nfeat, nchan = score[c].shape
            if do_coreset:
                ngroup = np.max(group[c]) + 1
                score_temp = score[c]
                group_temp = group[c]
                score_group = np.zeros((ngroup, nfeat, nchan))
                n_per_group = np.zeros(ngroup)
                for j in range(score[c].shape[0]):
                    score_group[group_temp[j]] += score_temp[j]
                    n_per_group[group_temp[j]] += 1
                for j in range(ngroup):
                    score_group[j] = score_group[j]/n_per_group[j]

                maskTemp = np.minimum(np.maximum(
                    ((np.sum(np.square(score_group), axis=1) - np.min(th))/(np.max(th)-np.min(th))), 0), 1)
                mask = np.zeros((ndata, nchan))
                for j in range(ndata):
                    mask[j] = maskTemp[group_temp[j]]
                mask_all.append(mask)

            else:
                score_group = score[c]
                mask = np.minimum(np.maximum(
                    ((np.sum(np.square(score_group), axis=1) - np.min(th))/(np.max(th)-np.min(th))), 0), 1)
                mask_all.append(mask)

        else:
            mask_all.append(np.zeros(0))

    return mask_all