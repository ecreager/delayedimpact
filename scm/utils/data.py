"""Utilities for loading data and formatting relevant distns."""
import gin
import numpy as np
import pandas as pd

import fico

@gin.configurable
def get_data_args(data_dir=gin.REQUIRED):
    """Returns objects that specify distns p(A), p(X|A), p(Y|X,A)."""

    all_cdfs, performance, totals = fico.get_FICO_data(data_dir)
    # NOTE: we drop last column to make the CDFs invertible ####################
    all_cdfs = all_cdfs.drop(all_cdfs.index[-1])
    performance = performance.drop(performance.index[-1])
    ############################################################################
    cdfs = all_cdfs[["White", "Black"]]

    cdf_B = cdfs['White'].values
    cdf_A = cdfs['Black'].values

    repay_B = performance['White']
    repay_A = performance['Black']

    inv_cdf_series = [
        pd.Series(cdfs[key].index.values, index=cdfs[key].values)
        for key in ('Black', 'White')
        ]
    inv_cdf_indices = [
        ics.index for ics in inv_cdf_series
        ]

    scores = cdfs.index
    scores_list = scores.tolist()

    # to populate group distributions
    def get_pmf(cdf):
        """Convert CDF into PMF."""
        pis = np.zeros(cdf.size)
        pis[0] = cdf[0]
        for score in range(cdf.size-1):
            pis[score+1] = cdf[score+1] - cdf[score]
        return pis

    # to get loan repay probabilities for a given score
    loan_repaid_probs = [
        lambda i: repay_A[scores[scores.get_loc(i, method='nearest')]],
        lambda i: repay_B[scores[scores.get_loc(i, method='nearest')]]
        ]

    # to get score at a given probabilities
    idx_A, idx_B = inv_cdf_indices
    srs_A, srs_B = inv_cdf_series
    inv_cdfs = [
        lambda i: srs_A[idx_A[idx_A.get_loc(i, method='nearest')]],
        lambda i: srs_B[idx_B[idx_B.get_loc(i, method='nearest')]],
        ]

    # get probability mass functions of each group
    pi_A = get_pmf(cdf_A)
    pi_B = get_pmf(cdf_B)
    pis = np.vstack([pi_A, pi_B])

    # demographic statistics
    group_ratio = np.array((totals["Black"], totals["White"]))
    group_size_ratio = group_ratio/group_ratio.sum()

    return inv_cdfs, loan_repaid_probs, pis, group_size_ratio, scores_list
