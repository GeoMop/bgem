import numpy as np

from bgem.bspline import \
    bspline as bs, \
    bspline_approx as bs_approx, \
    brep_writer as bw

import os

script_dir = os.path.dirname(os.path.realpath(__file__))


def function_sin_cos(x):
    return np.sin(x[0] * 4) * np.cos(x[1] * 4)


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


def make_surface_approx(path):
    """
    Make a full XYZ bspline surface approximation a point grid from a file.
    """

    ### Make an approximation.

    # Load point set from the grid file.
    surf_point_set = bs_approx.SurfacePointSet.from_file(path)

    # Compute minimal surface bounding rectangular of points projected to the XY plane.
    # or use own XY rectangle given as array of shape (4,2) of the four vertices.
    # quad = surf_point_set.compute_default_quad()

    # Crate the approximation object.
    surf_approx = bs_approx.SurfaceApprox(surf_point_set)

    # Try to guess dimensions of a (semi regular) grid.
    # nuv = surf_approx._compute_default_nuv()
    # We want usually  much sparser approximation.
    # nuv = nuv / 5

    # 4. Compute the approximation.
    surface = surf_approx.compute_approximation()
    # Own or computed initial number of patches `nuv=(nu, nv)` can be provided.
    # surface = surf_approx.compute_approximation(nuv=nuv)

    return surface.make_full_surface()


def make_shell_brep(surface_3d):
    # Make surface
    bw_surface = bw.surface_from_bs(surface_3d)

    # make Vertices for the corners
    vertices = [bw.Vertex.on_surface(u, v, bw_surface)
                for u, v in [(0, 0), (1, 0), (1, 1), (0, 1)]]
    vertices.append(vertices[0])
    edges = [bw.Edge(*vertices[i:i + 2]).attach_to_surface(bw_surface) for i in range(4)]
    face = bw.Face([bw.Wire(edges)], bw_surface)
    shell = bw.Shell([face])
    compound = bw.Compound([shell])
    compound.set_free_shapes()
    return compound


def test_make_shell():
    nuv = (50, 50)
    os.chdir(script_dir)
    # Create an input grid file.
    grid_path = "_grid_data.xyz"
    make_a_test_grid(grid_path, function_sin_cos, nuv)
    surface_3d = make_surface_approx(grid_path)
    brep_obj = make_shell_brep(surface_3d)
    with open(grid_path + ".brep", 'wt') as f:
        bw.write_model(f, brep_obj)


# Replace call through the pytest, allow execution as a script.
if __name__ == "__main__":
    test_make_shell()
