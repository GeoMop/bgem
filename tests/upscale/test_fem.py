import numpy as np
import pytest

from bgem.stochastic import dfn
from bgem.upscale import fem, fields, fem_plot

def basis_1():
    points = np.array([0.0, 1.0])
    basis = fem.Q1_1d_basis(points)
    return basis, points

def basis_2():
    points = np.array([0.0, 0.5, 1.0])
    basis = fem.Q1_1d_basis(points)
    return basis, points




def test_Q1_1D_basis():
    basis_order_1, points = basis_1()
    assert basis_order_1.shape == (2, 2)
    np.allclose(fem.eval_1d(basis_order_1, points), np.eye(2,2))
    print("Q1 order 1 basis: \n", basis_order_1)

    basis_order_2, points = basis_2()
    assert basis_order_2.shape == (3, 3)
    np.allclose(fem.eval_1d(basis_order_2, points), np.eye(3, 3))

    print("Q1 order 2 basis: \n", basis_order_2)




def test_poly_diff_1d():
    diff_order_1 = fem.poly_diff_1d(basis_1()[0])
    assert diff_order_1.shape == (2, 1)
    print("Q1 order 1 diff basis: \n", diff_order_1)
    diff_order_2 = fem.poly_diff_1d(basis_2()[0])
    assert diff_order_2.shape == (3, 2)
    print("Q1 order 2 diff basis: \n", diff_order_2)

def test_eval_1d():
    basis_order_1, _ = basis_1()
    points = [0.2, 0.7]
    values = [[0.2, 0.7], [0.8, 0.3]]
    np.allclose(fem.eval_1d(basis_order_1, points), values)

def test_Fe_Q1():
    for dim in range(1, 4):
        order = 1
        f = fem.Fe.Q1(dim, order)
        points_1d = np.linspace(0, 1, 2*order + 1)
        points = np.stack([
            points_1d,
            *(dim - 1) * [np.zeros_like(points_1d)]
        ])
        basis = f.eval(points)
        assert basis.shape == ((order + 1)**dim, len(points_1d))
        grad = f.grad_eval(points)
        assert grad.shape == (dim, (order + 1)**dim, len(points_1d))


def test_flatten_dim():
    x = np.outer([1, 2, 3, 4, 5, 6, 7, 8], [10, 100, 1000])
    tensor_x = fem.tensor_dim(x, 3, 2)
    assert tensor_x.shape == (2, 2, 2, 3)
    #print(tensor_x)
    flat_x = fem.flat_dim(tensor_x, 3)
    assert flat_x.shape == x.shape
    assert np.allclose(flat_x, x)


def test_grid_numbering():
    # Test Grid numbering

    dim = 1
    order = 2
    g = fem.Grid(100.0, 4, fem.Fe.Q1(dim, order))
    print(g)

    dim = 2
    order = 1
    g = fem.Grid(100.0, 4, fem.Fe.Q1(dim, order))
    print(g)

    dim = 3
    order = 1
    g = fem.Grid(100.0, 3, fem.Fe.Q1(dim, order))
    print(g)

def test_grid_bc():
    g = fem.Grid(10, 2, fem.Fe.Q1(1, 1))
    assert np.all(g.bc_coords == np.array([[0, 2]]))
    assert np.allclose(g.bc_points, np.array([[0, 10]]))

    g = fem.Grid(10, 2, fem.Fe.Q1(2, 1))
    ref = np.array([[0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2]])
    assert np.all(g.bc_coords == ref)

def test_laplace():
    order = 1
    N = 3
    dim = 2
    g = fem.Grid(N, N, fem.Fe.Q1(dim, order))
    l = g.laplace.reshape((-1, g.fe.n_dofs, g.fe.n_dofs))
    print("\nlaplace, 2d:\n", l)
def test_grid_assembly():
    for dim in range(1, 4):
        order = 1
        N = 3
        g = fem.Grid(30, N, fem.Fe.Q1(dim, order))
        K_const = np.diag(np.arange(1, dim + 1))
        K_const = fem.tn_to_voigt(K_const[None, :, :])
        K_field = K_const * np.ones(g.n_elements)[:, None]
        A = g.assembly_dense(K_field)
        n_dofs = (N+1)**dim
        assert A.shape == (n_dofs, n_dofs)

@pytest.mark.skip
def test_solve_system():
    for dim in range(1, 4):
        order = 1
        N = 3
        g = fem.Grid(30, N, fem.Fe.Q1(dim, order))
        K_const = np.diag(np.arange(1, dim + 1))
        K_const = fem.tn_to_voigt(K_const[:, :, None])
        K_field = K_const.T * np.ones(g.n_elements)[:, None]
        p_grads = np.eye(dim)
        pressure = g.solve_system(K_field, p_grads)
        assert pressure.shape == (dim, *dim * [N + 1])


@pytest.mark.skip
def test_solve_2d():
    dim = 2
    order = 1
    N = 30
    g = fem.Grid(100, N, fem.Fe.Q1(dim, order))
    x = g.barycenters()[0, :]
    K_const = np.diag([1, 1])
    #K_const = np.ones((dim, dim))
    K_const = fields.tn_to_voigt(K_const[None, :, :])
    K_field = K_const * x[:, None]
    #K_field = K_const.T * np.ones_like(x)[:, None]
    p_grads = np.eye(dim)
    pressure = g.solve_system(K_field, p_grads)
    xy_grid = [np.linspace(0, g.size[i], g.ax_dofs[i]) for i in range(2)]
    fem_plot.plot_pressure_fields(*xy_grid, pressure)

@pytest.mark.skip()
def test_upsacale_2d():
    K_const = np.diag([10, 100])
    K_const = fields.tn_to_voigt(K_const[None, :, :])
    K_field = K_const * np.ones((8, 8))[:, :, None]
    K_eff = fem.upscale(K_field)
    assert np.allclose(K_eff, K_const[0, :])


def test_upscale_parallel_plates():
    cube = [1, 1, 1]
    for dim in [2, 3]:
        plates = dfn.FractureSet.parallel_plates(
            box = cube,
            normal = [1, 0, 0]
        )


def single_fracture_distance_function():
    """
    Determine effective tensor as a function of the voxel center distance from
    the fracture plane and angle.
    lattitude : 0 - pi/4 : 9
    longitude : 0 - pi/4, up to pi/2 for validation : 9
    distance : 9 levels
    :return: about 1000 runs, also test of perfromance
    use 128^3 grid
    """
    pass