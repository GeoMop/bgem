import pyvista as pv
import numpy as np

from bgem.upscale import Grid
import matplotlib.pyplot as plt
import numpy as np


def plot_pressure_fields(grid:Grid, pressure):
    """
    x_grid: shape (M,)
    y_grid: shape (N,)
    Plots K scalar fields stored in a (K, M, N) shaped array `pressure`.

    Parameters:
    pressure (numpy.ndarray): An array of shape (K, M, N) representing K scalar fields.
    """
    x_grid, y_grid = [np.linspace(0, grid.dimensions[i], grid.dofs_shape[i]) for i in range(2)]
    K, n_dofs = pressure.shape
    assert n_dofs == grid.n_dofs
    M, N = grid.shape + 1
    pressure = pressure.reshape(K, M, N)

    # Create a figure and K subplots in a single row
    fig, axes = plt.subplots(1, K, figsize=(K * 5, 5), sharey=True)

    # Setting the limits for all plots
    #x_limit = (0, M)
    #y_limit = (0, N)

    # Find the global min and max values for a common color scale
    vmin, vmax = pressure.min(), pressure.max()
    X, Y = np.meshgrid(x_grid, y_grid)
    for i in range(K):
        # Plot each scalar field
        im = axes[i].pcolormesh(X, Y, pressure[i].transpose(), vmin=vmin, vmax=vmax, shading='gouraud')

    #im = axes[i].imgshow(pressure[i, :, :], vmin=vmin, vmax=vmax,
        #                    origin='lower', extent=x_limit + y_limit)

        # Set title for each subplot
        axes[i].set_title(f"Field {i + 1}")

        # Set limits for x and y axis
        #axes[i].set_xlim(x_limit)
        #axes[i].set_ylim(y_limit)

    # Add a common color bar
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    # Show the plot
    #plt.tight_layout()
    plt.show()




def plot_grid(n=5):
    """
    Create
    :param n:
    :return:
    """
    # Create a PyVista mesh from the points
    points = np.mgrid[:n, :n, :n] / (n - 1.0)
    mesh = pv.StructuredGrid(*points[::-1])
    points = points.reshape((3, -1))
    return points, mesh


def scatter_3d(mesh, values, n=5):
    # Normalize the function values for use in scaling
    scaled_values = (values - np.min(values)) / (np.max(values) - np.min(values))

    mesh['scalars'] = scaled_values

    # Create the glyphs: scale and color by the scalar values
    geom = pv.Sphere(phi_resolution=8, theta_resolution=8)
    glyphs = mesh.glyph(geom=geom, scale='scalars', factor=0.3)

    # Create a plotting object
    p = pv.Plotter()

    # Add the glyphs to the plotter
    p.add_mesh(glyphs, cmap='coolwarm', show_scalar_bar=True)

    # Add axes and bounding box for context
    p.add_axes()
    p.show_grid()
    p.add_bounding_box()

    # Show the plot
    p.show()


def plot_fn_3d(fn, n=5):
    points, mesh = plot_grid(n)
    values = fn(*points[::-1])
    scatter_3d(mesh, values)


def f(x, y, z):
    return x * (1 - y) * z * (1 - z) * 4


#plot_fn_3d(f)