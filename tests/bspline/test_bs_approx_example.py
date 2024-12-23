import numpy as np
from bgem.bspline import bspline as bs, bspline_plot as bs_plot, bspline_approx as bs_approx
import os

script_dir = os.path.dirname(os.path.realpath(__file__))


def make_a_test_grid(output_path, func, nuv):
    """
    Create a test grid file. Function evaluated on the unit square.
    Args:
        output_path: target file
        func: f(x,y) to use to create the grid.
        nuv: (nu, nv) - shape of the full grid.

    Returns:
        File with subset of the full grid and with added noise.
    """
    grid = bs.make_function_grid(func, *nuv)
    grid = grid.reshape((-1, 3))
    n_points = grid.shape[0]
    subindices = np.random.choice(n_points, size=int(0.7 * n_points))
    grid = grid[subindices, :]
    dx = 0.2 * 1 / nuv[0]
    dy = 0.2 * 1 / nuv[1]
    dz = 0.01 * np.ptp(grid[:, 2])  # value range
    grid += np.random.randn(*grid.shape) * np.array([dx, dy, dz])[None, :]
    np.savetxt(output_path, grid)


def function_sin_cos(x):
    return np.sin(x[0] * 4) * np.cos(x[1] * 4)


# def plot_cmp(a_grid, b_grid):
#     plt = bs_plot.Plotting()
#     #plt.scatter_3d(a_grid[:, 0], a_grid[:, 1], a_grid[:,  2])
#     plt.plot_surface(a_grid[:, 0], a_grid[:, 1], a_grid[:,  2])
#     plt.plot_surface(b_grid[:, 0], b_grid[:, 1], b_grid[:,  2])
#     plt.show()


def verify_approximation(func, surf):
    # Verification.
    # Evaluate approximation and the function on the same grid.

    nu = 4 * surf.u_basis.size
    nv = 4 * surf.v_basis.size
    V_grid, U_grid = np.meshgrid(np.linspace(0.0, 1.0, nu), np.linspace(0.0, 1.0, nv))
    xy_probe = np.stack([U_grid.ravel(), V_grid.ravel()], axis=1)
    # xyz_approx = surf.eval_xy_array(xy_probe).reshape(-1, 3)
    #
    z_func_eval = np.array([func([u, v]) for u, v in xy_probe], dtype=float)
    xyz_func = np.concatenate((xy_probe, z_func_eval[:, None]), axis=1).reshape(-1, 3)

    # plot_cmp(xyz_approx, xyz_func)
    plt = bs_plot.Plotting()
    plt.plot_surface_3d(surf.make_full_surface(), (nu, nv), poles=False)
    plt.scatter_3d(xyz_func[:, 0], xyz_func[:, 1], xyz_func[:, 2])
    plt.show()


def test_grid_approx_example():
    nuv = (50, 50)
    os.chdir(script_dir)
    # Create an input grid file.
    grid_path = "_grid_data.xyz"
    make_a_test_grid(grid_path, function_sin_cos, nuv)

    ### Make an approximation.

    # Load point set from the grid file.
    surf_point_set = bs_approx.SurfacePointSet.from_file(grid_path)

    # Compute minimal surface bounding rectangular of points projected to the XY plane.
    # or use own XY rectangle given as array of shape (4,2) of the four vertices.
    # quad = surf_point_set.compute_default_quad()

    # Crate the approximation object.
    surf_approx = bs_approx.SurfaceApprox(surf_point_set)

    # Try to guess dimensions of a (semi regular) grid.
    nuv = surf_approx._compute_default_nuv()
    # We want usually  much sparser approximation.
    nuv = nuv / 5

    # 4. Compute the approximation.
    surface = surf_approx.compute_approximation()
    # Own or computed initial number of patches `nuv=(nu, nv)` can be provided.
    # surface = surf_approx.compute_approximation(nuv=nuv)

    # Verification.
    # Evaluate approximation and the function on the same grid.
    verify_approximation(function_sin_cos, surface)


# Replace call through the pytest, allow execution as a script.
if __name__ == "__main__":
    test_grid_approx_example()

    # def gen_uv_grid(nu, nv):
    #     # surface on unit square
    #     U = np.linspace(0.0, 1.0, nu)
    #     V = np.linspace(0.0, 1.0, nv)
    #     V_grid, U_grid = np.meshgrid(V, U)
    #
    #     return np.stack([U_grid.ravel(), V_grid.ravel()], axis=1)
    #
    # def eval_func_on_grid(func, xy_mat, xy_shift, shape=(50, 50)):
    #     nu, nv = shape
    #     UV = gen_uv_grid(nu, nv)
    #     XY = xy_mat.dot(UV.T).T + xy_shift
    #     # z_func_eval = np.array([z_mat[0] * func([u, v]) + z_mat[1] for u, v in UV])
    #     z_func_eval = np.array([func([u, v]) for u, v in UV], dtype=float)
    #     return np.concatenate((XY, z_func_eval[:, None]), axis=1).reshape(nu, nv, 3)
