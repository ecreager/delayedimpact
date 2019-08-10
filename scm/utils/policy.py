"""Utilities for initializing policies as StructuralEqns."""
import gin

import distribution_to_loans_outcomes as dlo
from structural_eqns import ThresholdLoanPolicy

@gin.configurable
def get_policy(loan_repaid_probs,
               pis,
               group_size_ratio,
               utils,
               score_change_fns,
               scores,
               policy_name=gin.REQUIRED):
    """Get named threshold policy StructuralEqn."""

    thresh_dempar, thresh_eqopp, thresh_maxprof, thresh_downwards = \
        dlo.get_thresholds(loan_repaid_probs, pis, group_size_ratio, utils,
                           score_change_fns, scores)

    if policy_name.lower() == 'maxprof':
        return ThresholdLoanPolicy(*thresh_maxprof)  # pylint: disable=no-value-for-parameter

    if policy_name.lower() == 'dempar':
        return ThresholdLoanPolicy(*thresh_dempar)  # pylint: disable=no-value-for-parameter

    if policy_name.lower() == 'eqopp':
        return ThresholdLoanPolicy(*thresh_eqopp)  # pylint: disable=no-value-for-parameter

    raise ValueError('Bad policy name: {}'.format(policy_name))
