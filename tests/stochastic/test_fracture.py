import numpy as np
import pytest
from bgem.stochastic import fracture as frac
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def test_PowerLawSize():
    powers = [0.8, 1.6, 2.9, 3, 3.2]
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(powers))]

    fig = plt.figure(figsize = (16, 9))
    axes = fig.subplots(1, 2, sharey=True)
    for i, power in enumerate(powers):
        diam_range = (0.1, 10)
        distr = frac.PowerLawSize(power, diam_range, 1000)
        sizes = distr.sample(volume=1, size=10000)
        sizes.sort()
        x = np.geomspace(*diam_range, 30)
        y = [distr.cdf(xv, diam_range) for xv in x]
        z = [distr.ppf(yv, diam_range) for yv in y]
        np.allclose(x, z)
        axes[0].set_xscale('log')
        axes[0].plot(x, y, label=str(power), c=colors[i])

        axes[0].plot(sizes[::100], np.linspace(0, 1, len(sizes))[::100], c=colors[i], marker='+')
        sample_range = [0.1, 1]
        x1 = np.geomspace(*sample_range, 200)
        y1 = [distr.cdf(xv, sample_range) for xv in x1]
        axes[1].set_xscale('log')
        axes[1].plot(x1, y1, label=str(power))
    fig.legend()
    plt.show()
