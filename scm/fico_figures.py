#!/usr/bin/python
"""Basic reproducibility test."""

import os
import pickle

from absl import app
from absl import flags
import gin
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import fico
import distribution_to_loans_outcomes as dlo


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'gin_file', './config/default.gin', 'Path of config file.')
flags.DEFINE_multi_string(
    'gin_param', None, 'Newline separated list of Gin parameter bindings.')


def main(unused_argv):
    """Produces figures from Liu et al 2018 and save results."""
    del unused_argv
    gin.parse_config_files_and_bindings([FLAGS.gin_file], FLAGS.gin_param)


    DATA_DIR = gin.query_parameter('%data_dir')
    RESULTS_DIR = gin.query_parameter('%results_dir')
    seed = gin.query_parameter('%seed')
    np.random.seed(seed)


    # set plotting parameters
    sns.set_context("talk")
    sns.set_style("white")

    # this needs to be here so we can edit figures later
    plt.rcParams['pdf.fonttype'] = 42

    all_cdfs, performance, totals = fico.get_FICO_data(data_dir=DATA_DIR)

    cdfs = all_cdfs[["White", "Black"]]

    cdf_B = cdfs['White'].values
    cdf_A = cdfs['Black'].values

    repay_B = performance['White']
    repay_A = performance['Black']

    scores = cdfs.index
    scores_list = scores.tolist()
    scores_repay = cdfs.index  # pylint: disable=unused-variable


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


    # basic parameters
    N_scores = cdf_B.size  # pylint: disable=unused-variable
    N_groups = 2  # pylint: disable=unused-variable

    # get probability mass functions of each group
    pi_A = get_pmf(cdf_A)
    pi_B = get_pmf(cdf_B)
    pis = np.vstack([pi_A, pi_B])

    # demographic statistics
    group_ratio = np.array((totals["Black"], totals["White"]))
    group_size_ratio = group_ratio/group_ratio.sum()
    print(group_size_ratio)


    # profit and impact
    utility_repay_1 = gin.query_parameter('%utility_repay_1')
    utility_default_1 = gin.query_parameter('%utility_default_1')
    utility_repay_2 = gin.query_parameter('%utility_repay_2')
    utility_default_2 = gin.query_parameter('%utility_default_2')

    score_change_repay = gin.query_parameter('%score_change_repay')
    score_change_default = gin.query_parameter('%score_change_default')

    # considering several utility ratios to understand sensitivity of
    # qualitative results
    util_repay = [
        [utility_default_1, utility_repay_1],
        [utility_default_2, utility_repay_2]
        ]

    impact = [score_change_default, score_change_repay]


    # plot the repay probabilities
    _, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].plot(scores_list, repay_A, color='black', label='black')
    ax[0].plot(scores_list, repay_B, label='white', color="grey")
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_title("repay probabilities")
    ax[0].set_xlabel("score")
    ax[0].set_ylabel("repay probability")
    ax[0].legend()

    ax[1].plot(cdf_A, repay_A, color='black', label='black')
    ax[1].plot(cdf_B, repay_B, label='white', color="grey")
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_title("repay probabilities")
    ax[1].set_xlabel("CDF")
    ax[1].set_ylabel("repay probability")
    ax[1].legend()

    plt.savefig(os.path.join(RESULTS_DIR, 'scores-and-cdfs.pdf'))

    with open(os.path.join(RESULTS_DIR, 'scores-and-cdfs.p'), 'wb') as f:
        pickle.dump(
            dict(
                scores_list=scores_list,
                cdf_A=cdf_A,
                cdf_B=cdf_B,
                repay_A=repay_A,
                repay_B=repay_B), f
        )


    threshes = []
    for utils in util_repay:
        threshes.append(dlo.get_thresholds(
            loan_repaid_probs, pis, group_size_ratio, utils, impact, scores_list
            ))


    # plot the threshes
    _, ax = plt.subplots(1, len(threshes), figsize=(16, 8))
    plt.title("")
    for i, thresh in enumerate(threshes):
	# unpack
        threshes_DP, threshes_EO, threshes_MP, threshes_downward = thresh
        ax[i].plot(scores, 1 - cdf_A, label='black', color="black")
        ax[i].plot(scores, 1 - cdf_B, label='white', color="grey", alpha=0.4)

        ax[i].set_xlabel("score")
        ax[i].axvline(threshes_downward[0], LineStyle='-', color='yellow',
                      label="thresh-active-harm A")
        ax[i].axvline(threshes_MP[0], LineStyle='-', color='orange',
                      label="MP A")
        ax[i].axvline(threshes_DP[0], LineStyle='-', color='cyan',
                      label="DP A")
        ax[i].axvline(threshes_EO[0], LineStyle='-', color='purple',
                      label="EO A")
        ax[i].set_title("[u-,u+] = {0}".format(util_repay[i]))
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].set_ylim([0, 1])
        ax[i].set_xlim([300, 850])

        plt.suptitle(
            "Thresholds under different institution utility ratios (u-/u+)")

    ax[0].set_ylabel("Fraction of group above")
    plt.legend(loc='lower left')

    plt.savefig(os.path.join(RESULTS_DIR, 'figure-3.pdf'))

    with open(os.path.join(RESULTS_DIR, 'figure-3.p'), 'wb') as f:
        pickle.dump(
            dict(
                scores_list=scores_list,
                util_repay=util_repay,
                cdf_A=cdf_A,
                cdf_B=cdf_B,
                threshes=threshes,
            ), f
        )


    # get outcome curves

    outcome_curveA = \
        dlo.get_outcome_curve(loan_repaid_probs[0], pis[0], scores, impact)
    outcome_curveB = \
        dlo.get_outcome_curve(loan_repaid_probs[1], pis[1], scores, impact)

    rate_index_A = list(reversed(1- cdf_A))
    rate_index_B = list(reversed(1- cdf_B))

    # get utility curves
    utility_curves = dlo.get_utility_curve(
        loan_repaid_probs, pis, scores, utils=util_repay[0]
        )
    util_MP = np.amax(utility_curves, axis=1)

    utility_curves_MP = np.vstack(
        [utility_curves[0] + util_MP[1], utility_curves[1]+ util_MP[0]]
        )
    utility_curves_DP = dlo.get_utility_curves_dempar(
        utility_curves, np.vstack([cdf_A, cdf_B]), group_size_ratio, scores
        )
    utility_curves_EO = dlo.get_utility_curves_eqopp(
        utility_curves, loan_repaid_probs, pis, group_size_ratio, scores
        )

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

    plt.savefig(os.path.join(RESULTS_DIR, 'figure-4.pdf'))

    with open(os.path.join(RESULTS_DIR, 'figure-4.p'), 'wb') as f:
        pickle.dump(
            dict(
                rate_index_A=rate_index_A,
                outcome_curveA=outcome_curveA,
                rate_index_B=rate_index_B,
                outcome_curveB=outcome_curveB,
                utility_curves_MP=utility_curves_MP,
                utility_curves_DP=utility_curves_DP,
                utility_curves_EO=utility_curves_EO,
                utility_curves=utility_curves,
                util_MP=util_MP
            ), f
        )


if __name__ == "__main__":
    app.run(main)
