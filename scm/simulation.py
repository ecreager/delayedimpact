"""Single simulation of one-step FICO dynamics under Liu et al 2018 SCM."""

import os
import pickle

from absl import app
from absl import flags
import gin
import torch

import structural_eqns as se
from utils.policy import get_policy
from utils.data import get_data_args


class Simulation:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Runs simulation for one step of dynamics under Liu et al 2018 SCM."""
    def __init__(self,
                 f_A,  # stochastic structural eqn for group membership
                 f_X,  # stochastic structural eqn for indiv scores
                 f_Y,  # stochastic structural eqn for potential repayment
                 f_T,  # structural eqn for threshold loan policy
                 f_Delta,  # structural eqn for indiv score change
                 f_u,  # structural eqn for individual utility
                 f_Umathcal,  # structural eqn for avg institutional utility
                 f_Xtilde,  # structural eqn for next-step individual score
                 f_muDeltaj,  # structural eqn for per-group avg score change
                 ):
        self.f_A = f_A
        self.f_X = f_X
        self.f_Y = f_Y
        self.f_T = f_T
        self.f_Delta = f_Delta
        self.f_u = f_u
        self.f_Xtilde = f_Xtilde
        self.f_muDeltaj = f_muDeltaj
        self.f_Umathcal = f_Umathcal

    def run(self, num_steps, num_samps):
        """Run simulation forward for num_steps and return all observables."""
        if num_steps != 1:
            raise ValueError('Only one-step dynamics are currently supported.')
        blank_tensor = torch.zeros(num_samps)
        A = self.f_A(blank_tensor)
        X = self.f_X(A)
        Y = self.f_Y(X, A)
        T = self.f_T(X, A)
        Delta = self.f_Delta(Y, T)
        u = self.f_u(Y, T)
        Xtilde = self.f_Xtilde(X, Delta)
        muDeltaj = self.f_muDeltaj(Delta, A)
        Umathcal = self.f_Umathcal(u)
        return_dict = dict(
            A=A,
            X=X,
            Y=Y,
            T=T,
            Delta=Delta,
            u=u,
            Xtilde=Xtilde,
            muDeltaj=muDeltaj,
            Umathcal=Umathcal,
            )
        return return_dict

    def intervene(self, **kwargs):
        """Update attributes via intervention."""
        for k, v in kwargs.items():
            setattr(self, k, v)


def main(unused_argv):
    """Produces figures from Liu et al 2018 and save results."""
    del unused_argv
    gin.parse_config_files_and_bindings([FLAGS.gin_file], FLAGS.gin_param)

    seed = gin.query_parameter('%seed')
    results_dir = gin.query_parameter('%results_dir')
    num_steps = gin.query_parameter('%num_steps')
    num_samps = gin.query_parameter('%num_samps')
    utility_repay = gin.query_parameter('%utility_repay')
    utility_default = gin.query_parameter('%utility_default')
    score_change_repay = gin.query_parameter('%score_change_repay')
    score_change_default = gin.query_parameter('%score_change_default')

    torch.manual_seed(seed)

    inv_cdfs, loan_repaid_probs, pis, group_size_ratio, scores_list, _ = \
            get_data_args()
    utils = (utility_default, utility_repay)
    impact = (score_change_default, score_change_repay)
    prob_A_equals_1 = group_size_ratio[-1]
    f_A = se.IndivGroupMembership(prob_A_equals_1)
    f_X = se.InvidScore(*inv_cdfs)
    f_Y = se.RepayPotentialLoan(*loan_repaid_probs)
    f_T = get_policy(loan_repaid_probs, pis, group_size_ratio, utils, impact,
                     scores_list)
    f_Delta = se.ScoreChange(*impact)
    f_u = se.InstitUtil(*utils)
    f_Umathcal = se.AvgInstitUtil()
    f_Xtilde = se.NewScore()
    f_muDeltaj = se.AvgGroupScoreChange()

    simulation = Simulation(
        f_A, f_X, f_Y, f_T, f_Delta, f_u, f_Umathcal, f_Xtilde, f_muDeltaj,
        )

    results = simulation.run(num_steps, num_samps)

    # Finally, write results to disk
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # add thresholds determined by solver
    policy_name = gin.query_parameter('%policy_name')
    situation = 'situation1' if (utility_default == -4) else 'situation2'
    these_thresholds = {
        situation:
        {policy_name: [f_T.threshold_group_0, f_T.threshold_group_1]}
    }
    results['threshes'] = these_thresholds

    results_filename = os.path.join(results_dir, 'results.p')
    with open(results_filename, 'wb') as f:
        _ = pickle.dump(results, f)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        'gin_file', './config/one-quarter.gin', 'Path of config file.')
    flags.DEFINE_multi_string(
        'gin_param', None, 'Newline separated list of Gin parameter bindings.')

    app.run(main)
