import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

from .stats import fit_nbinom, stats1d

logger = logging.getLogger(__name__)


def plot1d(
    rvs: np.ndarray,
    dist: str,
    *,
    title: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> None:
    """Generates random variables from distribution `dist` and plots a
    histogram in kde mode.
    """
    # For better plot view in case there are a few extreme outliers
    params = norm.fit(rvs)
    z1 = norm.ppf(0.01, *params)
    z2 = norm.ppf(0.99, *params)
    to_keep = (rvs >= z1) & (rvs <= z2)
    to_remove = (~to_keep).sum()
    if 0 < to_remove <= 10:
        logger.warning(f"Removing {to_remove} points from view.")
        rvs_to_plot = rvs[to_keep]
    else:
        rvs_to_plot = rvs

    sns.violinplot(rvs_to_plot, color='#b56576')
    ax = sns.stripplot(rvs_to_plot, color='black', size=1, jitter=1)
    ax.set_title(title)
    params = fit_nbinom(rvs) if dist == 'nbinom' else None
    stats = stats1d(rvs, dist, params=params, pprint=True)
    ax.axhline(stats['dist_q05'], color='#e56b6f', zorder=100)
    ax.axhline(stats['dist_q95'], color='#e56b6f', zorder=100)

    y2 = ax.twinx()
    y2.set_ylim(ax.get_ylim())
    y2.set_yticks([stats['dist_q05'], stats['dist_q95']])
    y2.set_yticklabels(["dist_q=0.05", "dist_q=0.95"])
