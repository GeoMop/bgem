"""
Test DFN stochastic description.

TODO:
- test functionality of individual classes
- test Familly and Population interface

- test stochastic properties of individual distributions
- visulalization tests for fracture distribution
- calculation of estimates of stochastic parameters from the sample(s)
  (way to inversions)
"""

import pytest
import os
import attr
import numpy as np
import collections
# import matplotlib.pyplot as plt

from bgem.stochastic import dfn
from fixtures import sandbox_fname
#script_dir = os.path.dirname(os.path.realpath(__file__))



geometry_dict = {
    'box_dimensions': [100, 100, 100],
    'center_depth': 5000,
    'fracture_mesh_step': 15,
    'n_frac_limit': 200,
    'well_distance': 200,
    'well_effective_radius': 10,
    'well_openning': [-50, 50]}

from fixtures import fracture_stats


def test_PowerLawSize():
    #dfn.PowerLawSize.from_mean_area()
    pass

def test_UniformBoxPosition():
    center = [-10, -20, -40]
    dimensions = [20, 30, 40]
    pos = dfn.UniformBoxPosition(dimensions, center)
    assert pos.volume == 24000
    unit_pos_sample =  (pos.sample(1000) - center) / dimensions
    assert np.all(unit_pos_sample > -0.5)
    assert np.all(unit_pos_sample < 0.5)

def to_polar(x, y, z):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    if z > 0:
        phi += np.pi
    return (phi, rho)


def plot_fr_orientation(fractures):
    family_dict = collections.defaultdict(list)
    for fr in fractures:
        x, y, z = \
        dfn.FisherOrientation.rotate(np.array([0, 0, 1]), axis=fr.rotation_axis, angle=fr.rotation_angle)[0]
        family_dict[fr.region].append([
            to_polar(z, y, x),
            to_polar(z, x, -y),
            to_polar(y, x, z)
        ])

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
    for name, data in family_dict.items():
        # data shape = (N, 3, 2)
        data = np.array(data)
        for i, ax in enumerate(axes):
            phi = data[:, i, 0]
            r = data[:, i, 1]
            c = ax.scatter(phi, r, cmap='hsv', alpha=0.75, label=name)
    axes[0].set_title("X-view, Z-north")
    axes[1].set_title("Y-view, Z-north")
    axes[2].set_title("Z-view, Y-north")
    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
    fig.legend(loc=1)
    fig.savefig("fracture_orientation.pdf")
    plt.close(fig)
    # plt.show()

def test_population():
    pass
