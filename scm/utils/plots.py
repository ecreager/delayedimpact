"""Plotting utilities."""
import os

import gin
import numpy as np

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
