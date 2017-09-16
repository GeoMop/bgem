"""
Functions to plot Bspline curves and surfaces.
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np



def plot_curve_2d(curve, n_points=100, **kwargs):
    """
    Plot a 2d Bspline curve.
    :param curve: Curve t -> x,y
    :param n_points: Number of evaluated points.
    :param: kwargs: Additional parameters passed to the mtplotlib plot command.
    :return: Plot object.
    """

    basis = curve.basis
    t_coord = np.linspace(basis.domain[0], basis.domain[1], n_points)

    coords = [curve.eval(t) for t in t_coord]
    x_coord, y_coord = zip(*coords)
    return plt.plot(x_coord, y_coord, **kwargs)


def plot_curve_poles_2d(curve, **kwargs):
    """
    Plot poles of the B-spline curve.
    :param curve: Curve t -> x,y
    :param: kwargs: Additional parameters passed to the mtplotlib plot command.
    :return: Plot object.
    """
    x_poles, y_poles = curve.poles.T[0:2, :]    # remove weights
    return plt.plot(x_poles, y_poles, 'bo', color='red', **kwargs)


def plot_surface_3d(surface, fig_ax, n_points=(100, 100), **kwargs):
    """
    Plot a surface in 3d.
    Usage:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plot_surface_3d(surface, ax)
        plt.show()

    :param surface: Parametric surface in 3d.
    :param fig_ax: Axes object:
    :param n_points: (nu, nv), nu*nv - number of evaluation point
    :param kwargs: surface_plot additional options
    :return: The plot object.
    """

    u_basis, v_basis = surface.u_basis, surface.v_basis

    u_coord = np.linspace(u_basis.domain[0], u_basis.domain[1], n_points[0])
    v_coord = np.linspace(v_basis.domain[0], v_basis.domain[1], n_points[1])

    U, V = np.meshgrid(u_coord, v_coord)
    points = np.stack( [U.ravel(), V.ravel()], axis = 1 )

    xyz = surface.eval_array(points)
    X, Y, Z = xyz.T

    X = X.reshape(U.shape)
    Y = Y.reshape(U.shape)
    Z = Z.reshape(U.shape)

    # Plot the surface.
    return fig_ax.plot_surface(X, Y,  Z, **kwargs)


def plot_surface_poles_3d(surface, fig_ax, **kwargs):
    """
    Plot poles of the B-spline curve.
    :param curve: Curve t -> x,y
    :param: kwargs: Additional parameters passed to the mtplotlib plot command.
    :return: Plot object.
    """
    x_poles, y_poles, z_poles = surface.poles[:, :, 0:3].reshape(-1, 3).T          # remove weights and flatten nu, nv
    return fig_ax.scatter(x_poles, y_poles, z_poles, color='red', **kwargs)



