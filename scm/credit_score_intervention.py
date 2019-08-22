"""Plot results under credit score intervention."""

import os
import pickle
import sys
from typing import Dict

from absl import app
from absl import flags
import numpy as np
import gin
import torch
from tqdm import tqdm

import structural_eqns as se
from utils.data import get_data_args
from utils.plots import plot_figure4
from utils.plots import plot_new_plot
from utils.policy import get_policy
from utils.policy import get_dempar_policy_from_selection_rate
from utils.policy import get_eqopp_policy_from_selection_rate

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'gin_file', './config/credit_score_intervention.gin',
    'Path of config file.')
flags.DEFINE_multi_string(
    'gin_param', None, 'Newline separated list of Gin parameter bindings.')


class OneStepWithThresholdSimulation:  # pylint: disable=too-many-instance-attributes
    """Runs simulation for one step of dynamics under Liu et al 2018 SCM."""
    def __init__(self,
                 f_A: se.StructuralEqn,  # stochastic SE for group membership
                 f_X: se.StructuralEqn,  # stochastic SE for indiv scores
                 f_Xhat: se.StructuralEqn,  # SE for credit score threshold
                 f_Y: se.StructuralEqn,  # stochastic SE for potential repayment
                 f_T: se.StructuralEqn,  # SE for threshold loan policy
                 f_Xtilde: se.StructuralEqn,  # SE for indiv score change
                 f_u: se.StructuralEqn,  # SE for individual utility
                 f_Umathcal: se.StructuralEqn,  # SE for avg instit. utility
                 f_Deltaj: se.StructuralEqn,  # SE per-group avg score change
                 ) -> None:
        self.f_A = f_A
        self.f_X = f_X
        self.f_Xhat = f_Xhat
        self.f_Y = f_Y
        self.f_T = f_T
        self.f_Xtilde = f_Xtilde
        self.f_u = f_u
        self.f_Deltaj = f_Deltaj
        self.f_Umathcal = f_Umathcal

    def run(self, num_steps: int, num_samps: int) -> Dict:
        """Run simulation forward for num_steps and return all observables."""
        if num_steps != 1:
            raise ValueError('Only one-step dynamics are currently supported.')
        blank_tensor = torch.zeros(num_samps)
        A = self.f_A(blank_tensor)
        X = self.f_X(A)
        Y = self.f_Y(X, A)
        Xhat = self.f_Xhat(X)
        T = self.f_T(Xhat, A)
        Xtilde = self.f_Xtilde(X, Y, T)
        u = self.f_u(Y, T)
        Deltaj = self.f_Deltaj(X, Xtilde, A)
        Umathcal = self.f_Umathcal(u)
        return_dict = dict(
            A=A,
            X=X,
            Xhat=Xhat,
            Y=Y,
            T=T,
            u=u,
            Xtilde=Xtilde,
            Deltaj=Deltaj,
            Umathcal=Umathcal,
            )
        return return_dict

    def intervene(self, **kwargs):
        """Update attributes via intervention."""
        for k, v in kwargs.items():
            setattr(self, k, v)



@gin.configurable
def get_simulation(
        utility_repay=gin.REQUIRED,
        utility_default=gin.REQUIRED,
        score_change_repay=gin.REQUIRED,
        score_change_default=gin.REQUIRED,
        score_threshold=gin.REQUIRED):
    """Get a basic one-step simulation going."""
    data_args = get_data_args()
    inv_cdfs, loan_repaid_probs, pis, group_size_ratio, scores_list, \
            rate_indices = data_args  # pylint: disable=unused-variable
    utils = (utility_default, utility_repay)
    impact = (score_change_default, score_change_repay)
    prob_A_equals_1 = group_size_ratio[-1]
    f_A = se.IndivGroupMembership(prob_A_equals_1)
    f_X = se.InvidScore(*inv_cdfs)
    f_Xhat = se.ThresholdScore(score_threshold)
    f_Y = se.RepayPotentialLoan(*loan_repaid_probs)
    f_T = get_policy(loan_repaid_probs, pis, group_size_ratio, utils, impact,
                     scores_list)
    f_Xtilde = se.ScoreUpdate(*impact)
    f_u = se.InstitUtil(*utils)
    f_Umathcal = se.AvgInstitUtil()
    f_Deltaj = se.AvgGroupScoreChange()

    simulation = OneStepWithThresholdSimulation(
        f_A, f_X, f_Xhat, f_Y, f_T, f_Xtilde, f_u, f_Umathcal, f_Deltaj,
        )

    return simulation, data_args, utils, impact



def main(unused_argv):
    """Get results by sweeping inverventions"""
    del unused_argv
    gin.parse_config_files_and_bindings([FLAGS.gin_file], FLAGS.gin_param)

    seed = gin.query_parameter('%seed')
    results_dir = gin.query_parameter('%results_dir')
    num_samps = gin.query_parameter('%num_samps')

    results_dir = os.path.normpath(results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    torch.manual_seed(seed)

    simulation, data_args, _, _ = get_simulation()

    inv_cdfs, loan_repaid_probs, pis, _, scores, \
            rate_indices = data_args
    rate_index_A, rate_index_B = rate_indices

    ############################################################################
    # Outcome and utility curves
    ############################################################################
    def check(results):
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    msg = 'NaN spotted in results for variable ' + k
                    raise ValueError(msg)

    outcome_curve_A = []
    outcome_curve_B = []
    utility_curve_A = []
    utility_curve_B = []
    # NOTE: To match fidelity of Liu et al plots we sweep twice, with each group
    #       evaluting results at a different selection_rate grid.
    for selection_rate in tqdm(rate_index_A):
        f_T = get_dempar_policy_from_selection_rate(selection_rate, inv_cdfs)
        #print(selection_rate, f_T.threshold_group_0, f_T.threshold_group_1)
        simulation.intervene(f_T=f_T)
        results = simulation.run(1, num_samps)
        check(results)
        DeltaA, _ = [mdj.item() for mdj in results['Deltaj']]
        if (results['A'] != 0).all():  # no members of this group
            UmathcalA = 0.
        else:
            UmathcalA = torch.mean(results['u'][results['A'] == 0]).item()
        outcome_curve_A.append(DeltaA)
        utility_curve_A.append(UmathcalA)

    for selection_rate in tqdm(rate_index_B):
        f_T = get_dempar_policy_from_selection_rate(selection_rate, inv_cdfs)
        simulation.intervene(f_T=f_T)
        results = simulation.run(1, num_samps)
        check(results)
        _, DeltaB = [mdj.item() for mdj in results['Deltaj']]
        if (results['A'] != 1).all():  # no members of this group
            UmathcalB = 0.
        else:
            UmathcalB = torch.mean(results['u'][results['A'] == 1]).item()
        outcome_curve_B.append(DeltaB)
        utility_curve_B.append(UmathcalB)

    outcome_curve_A = np.array(outcome_curve_A)
    outcome_curve_B = np.array(outcome_curve_B)
    utility_curves = np.array([
        utility_curve_A,
        utility_curve_B,
        ])
    util_MP = np.amax(utility_curves, axis=1)
    utility_curves_MP = np.vstack(
        [utility_curves[0] + util_MP[1], utility_curves[1]+ util_MP[0]]
        )

    # collect DemPar results
    utility_curves_DP = [[], []]
    for i in tqdm(range(len(rate_index_A))):
        beta_A = rate_index_A[i]
        beta_B = rate_index_B[i]
        # get global util results under dempar at selection rate beta_A
        f_T_at_beta_A = get_dempar_policy_from_selection_rate(
            beta_A, inv_cdfs)
        simulation.intervene(f_T=f_T_at_beta_A)
        results = simulation.run(1, num_samps)
        check(results)
        empirical_beta_A = results['T'][results['A'] == 0].float().mean().item()
        print('DP', 'A', beta_A, f_T_at_beta_A.threshold_group_0,
                f_T_at_beta_A.threshold_group_1, empirical_beta_A)
        Umathcal_at_beta_A = results['Umathcal'].item()
        utility_curves_DP[0].append(Umathcal_at_beta_A)
        # get global util results under dempar at selection rate beta_B
        f_T_at_beta_B = get_dempar_policy_from_selection_rate(
            beta_B, inv_cdfs)
        simulation.intervene(f_T=f_T_at_beta_B)
        results = simulation.run(1, num_samps)
        check(results)
        empirical_beta_B = results['T'][results['A'] == 1].float().mean().item()
        print('DP', 'B', beta_B, f_T_at_beta_B.threshold_group_0,
                f_T_at_beta_B.threshold_group_1, empirical_beta_B)
        Umathcal_at_beta_B = results['Umathcal'].item()
        utility_curves_DP[1].append(Umathcal_at_beta_B)
    utility_curves_DP = np.array(utility_curves_DP)

    # collect EqOpp results
    utility_curves_EO = [[], []]
    for i in tqdm(range(len(rate_index_A))):
        beta_A = rate_index_A[i]
        beta_B = rate_index_B[i]
        # get global util results under dempar at selection rate beta_A
        f_T_at_beta_A = get_eqopp_policy_from_selection_rate(
            beta_A, loan_repaid_probs, pis, scores)
        simulation.intervene(f_T=f_T_at_beta_A)
        results = simulation.run(1, num_samps)
        check(results)
        # TODO: fix this
        empirical_beta_A = results['T'][
                (results['A'] == 1) & (results['Y'] == 1)
                ].float().mean().item()
        print('EO', 'A', beta_A, f_T_at_beta_A.threshold_group_0,
                f_T_at_beta_A.threshold_group_1, empirical_beta_A)
        Umathcal_at_beta_A = results['Umathcal'].item()
        utility_curves_EO[0].append(Umathcal_at_beta_A)
        # get global util results under dempar at selection rate beta_B
        f_T_at_beta_B = get_eqopp_policy_from_selection_rate(
            beta_B, loan_repaid_probs, pis, scores, A=False)
        simulation.intervene(f_T=f_T_at_beta_B)
        results = simulation.run(1, num_samps)
        check(results)
        empirical_beta_B = results['T'][
                (results['A'] == 1) & (results['Y'] == 1)
                ].float().mean().item()
        print('EO', 'B', beta_B, f_T_at_beta_B.threshold_group_0,
                f_T_at_beta_B.threshold_group_1, empirical_beta_B)
#        import pdb
#        pdb.set_trace()
        Umathcal_at_beta_B = results['Umathcal'].item()
        utility_curves_EO[1].append(Umathcal_at_beta_B)
    utility_curves_EO = np.array(utility_curves_EO)

    ############################################################################
    # Plot results
    ############################################################################
    plot_figure4(
        rate_index_A,
        rate_index_B,
        outcome_curve_A,
        outcome_curve_B,
        utility_curves_MP,
        utility_curves_DP,
        utility_curves_EO,
        results_dir)

    threshold = gin.query_parameter('get_simulation.score_threshold')
    plot_new_plot(
        threshold,
        scores,
        rate_index_A,
        rate_index_B,
        outcome_curve_A,
        outcome_curve_B,
        utility_curves_MP,
        utility_curves_DP,
        utility_curves_EO,
        results_dir)

    results.update(dict(
        rate_index_A=rate_index_A,
        rate_index_B=rate_index_B,
        outcome_curve_A=outcome_curve_A,
        outcome_curve_B=outcome_curve_B,
        utility_curves_MP=utility_curves_MP,
        utility_curves_DP=utility_curves_DP,
        utility_curves_EO=utility_curves_EO))

    ############################################################################
    # Finally, write commands, script, and results to disk
    ############################################################################
    # for reproducibility, copy command and script contents to results
    DEFAULT_RESULTS_DIR = '/scratch/gobi1/creager/delayedimpact'
    if results_dir not in ('.', 'results/python', DEFAULT_RESULTS_DIR):
        cmd = 'python ' + ' '.join(sys.argv)
        with open(os.path.join(results_dir, 'command.sh'), 'w') as f:
            f.write(cmd)
        this_script = open(__file__, 'r').readlines()
        with open(os.path.join(results_dir, __file__), 'w') as f:
            f.write(''.join(this_script))

    results_filename = os.path.join(results_dir, 'results.p')
    with open(results_filename, 'wb') as f:
        _ = pickle.dump(results, f)

    # Finally, write gin config to disk
    with open(os.path.join(results_dir, 'config.gin'), 'w') as f:
        f.write(gin.operative_config_str())

    # TODO(creager): possibly change plotting code

if __name__ == "__main__":
    app.run(main)
