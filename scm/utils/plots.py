"""Plotting utilities."""
import os

import gin
import numpy as np

FONTSIZE = 40

def _plot_cdfs(scores, cdfs, tag='cdfs'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cdf_A = cdfs[0]
    cdf_B = cdfs[1]

    # plot the threshes
    _, ax = plt.subplots(figsize=(8, 8))
    plt.title("")
    # unpack
    ax.plot(scores, 1 - cdf_A, label='black', color="black")
    ax.plot(scores, 1 - cdf_B, label='white', color="grey", alpha=0.4)
    ax.set_xlabel("score")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0, 1])
    ax.set_xlim([300, 900])

    plt.suptitle(
        "Thresholds under different institution utility ratios (u-/u+)")

    ax.set_ylabel("Fraction of group above")
    plt.legend(loc='lower left')

    filename = '/scratch/gobi1/creager/delayedimpact/{}.pdf'.format(tag)
    plt.savefig(filename)
    plt.close()


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
    ax[1, 0].legend(prop=dict(size=int(FONTSIZE / 2.5)))
    ax[1, 0].set_xlim([0, 1])

    ax[1, 1].plot(rate_index_B, utility_curves_MP[1], label='MP')
    ax[1, 1].plot(rate_index_B, utility_curves_DP[1], label='DP')
    ax[1, 1].plot(rate_index_B, utility_curves_EO[1], label='EO')
    ax[1, 1].set_xlabel('selection rate')
    ax[1, 1].set_ylabel('expected profit')
    ax[1, 1].set_ylim([-1, 1])
    ax[1, 1].legend(prop=dict(size=int(FONTSIZE / 2.5)))
    ax[1, 1].set_xlim([0, 1])
    plt.suptitle("")

    plt.savefig(os.path.join(results_dir, basename))

@gin.configurable
def plot_new_plot(
        thresh,
        scores,
        outcome_curveA,
        outcome_curveB,
        utility_curves_MP,
        utility_curves_DP,
        utility_curves_EO,
        threshes_MP,
        threshes_DP,
        threshes_EO,
        results_dir,
        basename_a='new-plot-a.pdf',
        basename_b='new-plot-a.pdf',
        basename_c='new-plot-c.pdf',
        basename_d='new-plot-d.pdf',
        ):
    """Reproduce figure 4 given computed results."""
    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('text', usetex=True)
    font = {'family' : 'stix',
            'weight' : 'bold',
            'size'   : 50}
    rc('font', **font)
    rc('mathtext', fontset='stix')
    import seaborn as sns

    # set plotting parameters
    sns.set_context("talk")
    sns.set_style("white")

    # this needs to be here so we can edit figures later
    #plt.rcParams['pdf.fonttype'] = 42


    c = lambda n: isinstance(n, np.ndarray)  # check type
    assert (c(utility_curves_MP)
            and c(utility_curves_DP)
            and c(utility_curves_EO))

    def get_ax():
        _, ax = plt.subplots(figsize=(6, 6))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(0, LineStyle='-', color='grey', alpha=0.4)
        return ax

    # reverse scores to match rate indices
    scores = np.flipud(scores)

    ###########################################################################
    # FIRST SUBFIGURE
    ###########################################################################
    ax = get_ax()
    ax.plot(scores, outcome_curveA, color='black',
            label=r'$\Delta_{\textrm{Black}}$')
    plt.axvline(
        x=thresh, ymin=-70, ymax=50, linestyle='--', color='c',
        label=r'$\tau_{\textrm{CB}}=600$')
    ax.set_xlabel(r'$\tau_{\textrm{Black}}$', fontsize=FONTSIZE)
    xlabel = r'$\tau_{\textrm{Black}}$'
    ylabel = 'Avg score change'
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.legend(loc='upper left', prop=dict(size=int(FONTSIZE / 2.5)))
    ax.set_ylim([-70, 50])
    ax.set_xlim([300, 850])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, basename_a))
    plt.close()

    ###########################################################################
    # SECOND SUBFIGURE
    ###########################################################################
    ax = get_ax()
    ax.plot(scores, outcome_curveB, color='black',
            label=r'$\Delta_{\textrm{White}}$')
    plt.axvline(
        x=thresh, ymin=-70, ymax=50, linestyle='--', color='c',
        label=r'$\tau_{\textrm{CB}}=600$')
    ax.set_xlabel(r'$\tau_{\textrm{Black}}$', fontsize=FONTSIZE)
    xlabel = r'$\tau_{\textrm{White}}$'
    ylabel = 'Avg score change'
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.legend(loc='upper left', prop=dict(size=int(FONTSIZE / 2.5)))
    ax.set_ylim([-70, 50])
    ax.set_xlim([300, 850])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, basename_b))
    plt.close()


    ###########################################################################
    # THIRD SUBFIGURE
    ###########################################################################
    ax = get_ax()
    ax.plot(threshes_MP[0], utility_curves_MP[0], label='MaxProf')
    ax.plot(threshes_DP[0], utility_curves_DP[0], label='DemPar')
    ax.plot(threshes_EO[0], utility_curves_EO[0], label='EqOpp')
    plt.axvline(
        x=thresh, ymin=-70, ymax=50, linestyle='--', color='c',
        label=r'$\tau_{\textrm{CB}}=600$')
    ax.set_ylim([-1, 1])
    ylabel = 'Institutional profit'
    xlabel = r'$\tau_{\textrm{Black}}$'
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.legend(prop=dict(size=int(FONTSIZE / 3.0)))
    ax.set_xlim([300, 850])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, basename_c))
    plt.close()

    ###########################################################################
    # FOURTH SUBFIGURE
    ###########################################################################
    ax = get_ax()
    ax.plot(threshes_MP[1], utility_curves_MP[1], label='MaxProf')
    ax.plot(threshes_DP[1], utility_curves_DP[1], label='DemPar')
    ax.plot(threshes_EO[1], utility_curves_EO[1], label='EqOpp')
    plt.axvline(
        x=thresh, ymin=-70, ymax=50, linestyle='--', color='c',
        label=r'$\tau_{\textrm{CB}}=600$')
    ax.set_ylim([-1, 1])
    ylabel = 'Institutional profit'
    xlabel = r'$\tau_{\textrm{White}}$'
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.legend(prop=dict(size=int(FONTSIZE / 3.0)))
    ax.set_xlim([300, 850])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, basename_d))
    plt.close()
