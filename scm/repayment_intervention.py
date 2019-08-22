"""Intervene by sampling Y from P(Y|X) rather than P(Y|X,A)."""

import os
import pickle
import sys

from absl import app
from absl import flags
import numpy as np
import gin
import torch
from tqdm import tqdm

from figure4 import get_simulation
from utils.data import get_marginal_loan_repaid_probs
from utils.plots import plot_figure4
from utils.policy import get_policy
from utils.policy import get_dempar_policy_from_selection_rate
from utils.policy import get_eqopp_policy_from_selection_rate





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

    simulation, data_args, utils, impact = get_simulation()
    inv_cdfs, _, pis, group_size_ratio, scores, rate_indices = data_args
    rate_index_A, rate_index_B = rate_indices
    marginal_loan_repaid_probs = get_marginal_loan_repaid_probs()
    f_T_marginal = get_policy(marginal_loan_repaid_probs, pis, group_size_ratio,
                              utils, impact, scores)

    simulation.intervene(f_T=f_T_marginal)

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
    for i in tqdm(range(len(rate_index_A))):
        beta_A = rate_index_A[i]
        beta_B = rate_index_B[i]
        # get global util results under dempar at selection rate beta_A
        f_T_at_beta_A = get_eqopp_policy_from_selection_rate(
            beta_A, marginal_loan_repaid_probs, pis, scores)
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

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        'gin_file', './config/repayment_intervention.gin',
        'Path of config file.')
    flags.DEFINE_multi_string(
        'gin_param', None, 'Newline separated list of Gin parameter bindings.')

    app.run(main)
