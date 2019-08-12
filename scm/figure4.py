"""Reproduce figure 4 using empirical samples from the SCM."""

import os
import pickle

from absl import app
from absl import flags
import numpy as np
import gin
import torch
from tqdm import tqdm

from simulation import Simulation
import structural_eqns as se
from utils.policy import get_policy
from utils.policy import get_dempar_policy_from_selection_rate
from utils.policy import get_eqopp_policy_from_selection_rate
from utils.data import get_data_args

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'gin_file', './config/figure4.gin', 'Path of config file.')
flags.DEFINE_multi_string(
    'gin_param', None, 'Newline separated list of Gin parameter bindings.')

@gin.configurable
def plot_figure4(
        rate_index_A,
        rate_index_B,
        outcome_curveA,
        outcome_curveB,
        utility_curves_MP,
        utility_curves_DP,
        utility_curves_EO,
        results_dir,
        basename='figure-4-empirical.pdf'
        ):
    """Reproduce figure 4 given computed results."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # set plotting parameters
    sns.set_context("talk")
    sns.set_style("white")

    # this needs to be here so we can edit figures later
    plt.rcParams['pdf.fonttype'] = 42


    c = lambda n: isinstance(n, np.ndarray)  # check type
    assert (c(utility_curves_MP)
            and c(utility_curves_DP)
            and c(utility_curves_EO))

    # NOTE: the code I eventually want to use
    _, ax = plt.subplots(2, 2, figsize=(16, 16))
    for i in range(2):
        for j in range(2):
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].axhline(0, LineStyle='-', color='grey', alpha=0.4)

    ax[0, 0].plot(rate_index_A, outcome_curveA, color='black')
    ax[0, 0].set_xlabel('selection rate')
    ax[0, 0].set_ylabel('expected score change')
    ax[0, 0].set_title('Black')
    ax[0, 0].set_ylim([-70, 50])
    ax[0, 0].set_xlim([0, 1])

    ax[0, 1].plot(rate_index_B, outcome_curveB, color='black')
    ax[0, 1].set_xlabel('selection rate')
    ax[0, 1].set_title('White')
    ax[0, 1].set_ylim([-70, 50])
    ax[0, 1].set_xlim([0, 1])

    ax[1, 0].plot(rate_index_A, utility_curves_MP[0], label='MP')
    ax[1, 0].plot(rate_index_A, utility_curves_DP[0], label='DP')
    ax[1, 0].plot(rate_index_A, utility_curves_EO[0], label='EO')
    ax[1, 0].set_xlabel('selection rate')
    ax[1, 0].set_ylabel('expected profit')
    ax[1, 0].set_ylim([-1, 1])
    ax[1, 0].legend()
    ax[1, 0].set_xlim([0, 1])

    ax[1, 1].plot(rate_index_B, utility_curves_MP[1], label='MP')
    ax[1, 1].plot(rate_index_B, utility_curves_DP[1], label='DP')
    ax[1, 1].plot(rate_index_B, utility_curves_EO[1], label='EO')
    ax[1, 1].set_xlabel('selection rate')
    ax[1, 1].set_ylabel('expected profit')
    ax[1, 1].set_ylim([-1, 1])
    ax[1, 1].legend()
    ax[1, 1].set_xlim([0, 1])
    plt.suptitle("")

    plt.savefig(os.path.join(results_dir, basename))

@gin.configurable
def get_simulation(
        utility_repay_1=gin.REQUIRED,
        utility_default_1=gin.REQUIRED,
        utility_repay_2=gin.REQUIRED,
        utility_default_2=gin.REQUIRED,
        score_change_repay=gin.REQUIRED,
        score_change_default=gin.REQUIRED):
    """Get a basic one-step simulation going."""
    data_args = get_data_args()
    inv_cdfs, loan_repaid_probs, pis, group_size_ratio, scores_list, \
            rate_indices = data_args  # pylint: disable=unused-variable
    utils = (
                (utility_default_1, utility_repay_1),
                (utility_default_2, utility_repay_2),
        )
    impact = (score_change_default, score_change_repay)
    prob_A_equals_1 = group_size_ratio[-1]
    f_A = se.IndivGroupMembership(prob_A_equals_1)
    f_X = se.InvidScore(*inv_cdfs)
    f_Y = se.RepayPotentialLoan(*loan_repaid_probs)
    f_T = get_policy(loan_repaid_probs, pis, group_size_ratio, utils[0], impact,
                     scores_list)
    f_Delta = se.ScoreChange(*impact)
    f_u = se.InstitUtil(*utils[0])
    f_Umathcal = se.AvgInstitUtil()
    f_Xtilde = se.NewScore()
    f_muDeltaj = se.AvgGroupScoreChange()

    simulation = Simulation(
        f_A, f_X, f_Y, f_T, f_Delta, f_u, f_Umathcal, f_Xtilde, f_muDeltaj,
        )

    return simulation, data_args



def main(unused_argv):
    """Get results by sweeping inverventions"""
    del unused_argv
    gin.parse_config_files_and_bindings([FLAGS.gin_file], FLAGS.gin_param)

    seed = gin.query_parameter('%seed')
    results_dir = gin.query_parameter('%results_dir')
    num_samps = gin.query_parameter('%num_samps')

    torch.manual_seed(seed)

    simulation, data_args = get_simulation()

    inv_cdfs, loan_repaid_probs, pis, _, scores, \
            rate_indices = data_args
    rate_index_A, rate_index_B = rate_indices

    ############################################################################
    # outcome and utility curves
    ############################################################################

    # for top half of figure 4, iterate over selection_rate, find threshold
    # policy at each SR value, simulate under intervention, and compute average
    # per-group Delta

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
        simulation.intervene(f_T=f_T)
        results = simulation.run(1, num_samps)
        check(results)
        muDeltaA, _ = [mdj.item() for mdj in results['muDeltaj']]
        if (results['A'] != 0).all():  # no members of this group
            UmathcalA = 0.
        else:
            UmathcalA = torch.mean(results['u'][results['A'] == 0]).item()
        outcome_curve_A.append(muDeltaA)
        utility_curve_A.append(UmathcalA)

    for selection_rate in tqdm(rate_index_B):
        f_T = get_dempar_policy_from_selection_rate(selection_rate, inv_cdfs)
        simulation.intervene(f_T=f_T)
        results = simulation.run(1, num_samps)
        check(results)
        _, muDeltaB = [mdj.item() for mdj in results['muDeltaj']]
        if (results['A'] != 1).all():  # no members of this group
            UmathcalB = 0.
        else:
            UmathcalB = torch.mean(results['u'][results['A'] == 1]).item()
        outcome_curve_B.append(muDeltaB)
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
    # TODO(creager): possibly iterate over scores insead of selection rates?
    for i in tqdm(range(len(rate_index_A))):
        beta_A = rate_index_A[i]
        beta_B = rate_index_B[i]
        # get global util results under dempar at selection rate beta_A
        f_T_at_beta_A = get_dempar_policy_from_selection_rate(
            beta_A, inv_cdfs)
        simulation.intervene(f_T=f_T_at_beta_A)
        results = simulation.run(1, num_samps)
        check(results)
        Umathcal_at_beta_A = results['Umathcal'].item()
        utility_curves_DP[0].append(Umathcal_at_beta_A)
        # get global util results under dempar at selection rate beta_B
        f_T_at_beta_B = get_dempar_policy_from_selection_rate(
            beta_B, inv_cdfs)
        simulation.intervene(f_T=f_T_at_beta_B)
        results = simulation.run(1, num_samps)
        check(results)
        Umathcal_at_beta_B = results['Umathcal'].item()
        utility_curves_DP[1].append(Umathcal_at_beta_B)
    utility_curves_DP = np.array(utility_curves_DP)

    # collect EqOpp results
    utility_curves_EO = [[], []]
    # TODO(creager): possibly iterate over scores insead of selection rates?
    for i in tqdm(range(len(rate_index_A))):
        beta_A = rate_index_A[i]
        beta_B = rate_index_B[i]
        # get global util results under dempar at selection rate beta_A
        f_T_at_beta_A = get_eqopp_policy_from_selection_rate(
            beta_A, loan_repaid_probs, pis, scores)
        simulation.intervene(f_T=f_T_at_beta_A)
        results = simulation.run(1, num_samps)
        check(results)
        Umathcal_at_beta_A = results['Umathcal'].item()
        utility_curves_EO[0].append(Umathcal_at_beta_A)
        # get global util results under dempar at selection rate beta_B
        f_T_at_beta_B = get_dempar_policy_from_selection_rate(
            beta_B, inv_cdfs)
        simulation.intervene(f_T=f_T_at_beta_B)
        results = simulation.run(1, num_samps)
        check(results)
        Umathcal_at_beta_B = results['Umathcal'].item()
        utility_curves_EO[1].append(Umathcal_at_beta_B)
    utility_curves_EO = np.array(utility_curves_EO)

    with open(os.path.join(results_dir, 'figure-4.p'), 'rb') as f:
        old_results = pickle.load(f)

    # NOTE: for consistency with new results, which omit the top bin
    del old_results['rate_index_A'][0]
    del old_results['rate_index_B'][0]

    nd = np.array
    norm = np.linalg.norm
    norm_diff = lambda a, b: norm(nd(a) - nd(b))

    assert norm_diff(rate_indices[0], old_results['rate_index_A']) == 0.
    assert norm_diff(rate_indices[1], old_results['rate_index_B']) == 0.

    plot_figure4(
        rate_index_A,
        rate_index_B,
        outcome_curve_A,
        outcome_curve_B,
        utility_curves_MP,
        utility_curves_DP,
        utility_curves_EO,
        results_dir)


if __name__ == "__main__":
    app.run(main)
