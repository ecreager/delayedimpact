#!/usr/bin/python
"""Basic SCM refactor produces same thresholds as original jupyter notebook."""

import pickle
import unittest

import numpy as np
from parameterized import parameterized

BASELINE_RESULTS_FILENAME = 'results/python/figure-3.p'
RESULTS_FILENAME_FMT = \
        'results/python/simulation/{situation:}/{policy_name:}/results.p'

class TestSameThresholds(unittest.TestCase):
    """Check that thresholds are same in this as that."""

    def __init__(self, *args, **kwargs):
        # format baseline threshold results
        super(TestSameThresholds, self).__init__(*args, **kwargs)
        with open(BASELINE_RESULTS_FILENAME, 'rb') as f:
            baseline_results = pickle.load(f)
        baseline_thresholds_nested_tuple = baseline_results['threshes']
        self.baseline_thresholds = dict()
        for situation, thresholds in zip(
                ('situation1', 'situation2'),
                baseline_thresholds_nested_tuple):
            thresholds_dict = dict(zip(
                ('dempar', 'eqopp', 'maxprof', 'downward'),
                thresholds
                ))
            self.baseline_thresholds[situation] = thresholds_dict

    @parameterized.expand([
        ('situation1', 'maxprof'),
        ('situation1', 'dempar'),
        ('situation1', 'eqopp'),
        ('situation2', 'maxprof'),
        ('situation2', 'dempar'),
        ('situation2', 'eqopp')
        ])
    def test_same_thresholds(self, situation, policy_name):  # pylint: disable=missing-docstring
        """Test that reuslts stored in filename are consistent with baseline."""
        kwargs = dict(situation=situation, policy_name=policy_name)
        filename = RESULTS_FILENAME_FMT.format(**kwargs)
        with open(filename, 'rb') as f:
            these_results = pickle.load(f)
        these_thresholds = these_results['threshes']
        result1 = np.array(these_thresholds[situation][policy_name])
        result2 = np.array(self.baseline_thresholds[situation][policy_name])
        msg = ('Tresholds in {situation:} with policy {policy_name:} '
               'don\'t match!').format(**kwargs)
        norm = np.linalg.norm
        self.assertAlmostEqual(norm(result1 - result2), 0., places=0, msg=msg)  # pylint: disable=deprecated-method

if __name__ == '__main__':
    unittest.main()
