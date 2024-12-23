"""
Visualize meaning of the exponent in power law size distribution.
Estimate Hausdorf dimension numerically for a large set of fractures.


Conclusion:
- fractures with homogeneous position can not have Hausdorf dimension given by the power
  parameter of the power law and have dimension close to 2 according to their density
- we need to include the correlation between the positions (and possibly orientations)
"""
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from bgem.stochastic import dfn


def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[1], k), axis=1),
        np.arange(0, Z.shape[0], k), axis=0)

    # We count non-empty (0) and non-full boxes (k*k)
    cover = S > 0
    count = np.sum(cover)
    return count


def fractal_dimension(img, threshold=1):
    a = np.array(img)
    Z = a[:, :, 0]
    # Only for 2d image
    assert (len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = [boxcount(Z, size) for size in sizes]
    print("sizes: ", np.log(sizes))
    print("counts: ", np.log(counts) / np.log(sizes))
    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    plt.scatter(np.log(sizes), np.log(counts))
    plt.show()
    return -coeffs[0]


def scale_cut(x, s):
    return tuple([max(0, min(sv, int(sv * xv))) for xv, sv in zip(x[:2], s)])


def plot_sizes(sizes, sample_range, size_distr):
    N = 10000
    fig = plt.figure(figsize=(100, 100))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale('log')
    # ax.hist(sizes, bins=100, cumulative=True)
    x = np.geomspace(*sample_range, 100)
    # powlaw_scale = [x ** (1- power) / (1- power) for x in reversed(sample_range)]
    # powlaw_scale = powlaw_scale[0] - powlaw_scale[1]
    # ax.plot(x, 1/powlaw_scale * N * x**-power)
    sizes.sort()
    ax.plot(sizes, np.linspace(0, N, len(sizes)), c='red')
    ax.plot(x, [N * size_distr.cdf(xv, sample_range) for xv in x], c='green')
    plt.show()


def plot_dfn(power):
    s = (1000, 1000)

    im = Image.new('RGBA', s, (255, 255, 255, 255))
    draw = ImageDraw.Draw(im)

    fracture_box = [1, 1, 1]
    sample_range = (0.001, 1)
    power = 2.1
    conf_range = [0.001, 1]
    p_32 = 100
    # p_32 = 0.094
    size = dfn.PowerLawSize.from_mean_area(power - 1, conf_range, p_32, power)
    family = dfn.FrFamily(
        orientation=dfn.FisherOrientation(0, 90, 0),
        size=size,
        shape_angle=dfn.VonMisesOrientation(0, 0)
    )
    pop = dfn.Population(
        domain=(fracture_box[0], fracture_box[1], 0),
        families=[family]
    )
    pop.set_sample_range(sample_range)
    pos_gen = dfn.UniformBoxPosition(fracture_box)
    print("total mean size: ", pop.mean_size())

    fractures = pop.sample(pos_distr=pos_gen, keep_nonempty=True)
    print("N frac:", len(fractures))
    sizes = []
    for fr in fractures:
        t = 0.5 * fr.r * np.array([-fr.normal[2], fr.normal[1], 0])
        a = scale_cut(0.5 + fr.center - t, s)
        b = scale_cut(0.5 + fr.center + t, s)
        draw.line((a, b), fill=(0, 0, 0), width=1)
        sizes.append(fr.r)

    # plot_sizes(sizes, sample_range, pop.families[0].size)

    # ax.imshow(np.asarray(im),  origin='lower')
    im.show()

    print("Minkowski–Bouligand dimension (computed): ", fractal_dimension(im))
    print("Hausdorff dim (exp.): ", power - 2)

    # ax.line
    # pos_gen = fracture.UniformBoxPosition(fracture_box)
    # fractures = pop.sample(pos_distr=pos_gen, keep_nonempty=True)
    # for i, fr in enumerate(fractures):
    #     fr.region = reg


plot_dfn(2)
