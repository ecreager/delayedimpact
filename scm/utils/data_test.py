"""Unit tests for data utils."""
import unittest

import numpy as np
import pandas as pd

import fico
from utils.data import get_data_args

DATA_DIR = './data'

def old_code(data_dir):
    """Return inv_cdfs and loan_repaid_probs without batching."""
    all_cdfs, performance, _ = fico.get_FICO_data(data_dir)
    # NOTE: we drop last column to make the CDFs invertible ####################
    all_cdfs = all_cdfs.drop(all_cdfs.index[-1])
    performance = performance.drop(performance.index[-1])
    ############################################################################
    cdfs = all_cdfs[["White", "Black"]]

    cdf_B = cdfs['White'].values  # pylint: disable=unused-variable
    cdf_A = cdfs['Black'].values  # pylint: disable=unused-variable

    repay_B = performance['White']
    repay_A = performance['Black']

    scores = cdfs.index
    scores_list = scores.tolist()  # pylint: disable=unused-variable

    loan_repaid_probs = [
        lambda i: repay_A[scores[scores.get_loc(i, method='nearest')]],
        lambda i: repay_B[scores[scores.get_loc(i, method='nearest')]]
        ]
    inv_cdf_series = [
        pd.Series(cdfs[key].index.values, index=cdfs[key].values)
        for key in ('Black', 'White')
        ]
    inv_cdf_indices = [
        ics.index for ics in inv_cdf_series
        ]
    idx_A, idx_B = inv_cdf_indices
    srs_A, srs_B = inv_cdf_series
    inv_cdfs = [
        lambda i: srs_A[idx_A[idx_A.get_loc(i, method='nearest')]],
        lambda i: srs_B[idx_B[idx_B.get_loc(i, method='nearest')]],
        ]

    return inv_cdfs, loan_repaid_probs



class TestSameFns(unittest.TestCase):
    """Check that functions before/after refactor are same."""
    NUM_SAMPS = 100
    def __init__(self, *args, **kwargs):
        super(TestSameFns, self).__init__(*args, **kwargs)
        self.inv_cdfs_old, self.loan_repaid_probs_old = \
                    old_code(data_dir=DATA_DIR)
        self.inv_cdfs_new, self.loan_repaid_probs_new, *args = \
                    get_data_args(data_dir=DATA_DIR)

    def test_same_fns(self):
        """Make sure functions before/after match."""
        scores = np.random.uniform(low=300., high=850., size=self.NUM_SAMPS)
        probs = np.random.uniform(low=0., high=1., size=self.NUM_SAMPS)

        for group in range(2):
            for score in scores:
                self.assertEqual(
                    self.loan_repaid_probs_old[group](score),
                    self.loan_repaid_probs_old[group](score)
                )
            for prob in probs:
                self.assertEqual(
                    self.inv_cdfs_old[group](prob),
                    self.inv_cdfs_new[group](prob)
                )

        for group in range(2):
            batched_prob_repaid = []  # outputs of loan_prob_repaid
            batched_scores = []  # outputs of inv_cdfs
            for score in scores:
                batched_prob_repaid.append(
                    self.loan_repaid_probs_old[group](score)
                )
            batched_prob_repaid = np.array(batched_prob_repaid)
            self.assertAlmostEqual(np.linalg.norm(
                batched_prob_repaid - self.loan_repaid_probs_new[group](scores)
            ), 0., places=3)
            for prob in probs:
                batched_scores.append(
                    self.inv_cdfs_old[group](prob)
                )
            batched_scores = np.array(batched_scores)
            self.assertAlmostEqual(np.linalg.norm(
                batched_scores - self.inv_cdfs_new[group](probs)
            ), 0., places=3)
