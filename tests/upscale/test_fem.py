import numpy as np
import pytest

#from bgem.stochastic import dfn
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
        f = fem.Fe.Q(dim, order)
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


def test_grid_init():
    g = fem.Grid((100, 150, 200), (4, 3, 2), fem.Fe.Q(3, 1), origin=(-4, -5, -6))
    assert g.dim == 3
    assert np.allclose(g.origin, [-4, -5, -6])
    assert np.allclose(g.dimensions, [100, 150, 200])
    assert np.allclose(g.shape, [4, 3, 2])

    # basic properties
    assert np.allclose(g.step, [25, 50, 100])
    assert g.n_loc_dofs == 8
    assert np.allclose(g.dofs_shape, [5, 4, 3])
    assert g.n_dofs == 60
    assert g.n_elements == 24
    assert np.allclose(g.dof_coord_coef, [12, 3, 1])

    # numberings
    assert g.n_bc_dofs == 60 - 6
    ref_natur_map = [
        0, 1, 2,      3, 4, 5,     6, 7, 8,     9, 10, 11,
        12, 13, 14,   15,   17,    18,   20,    21, 22, 23,
        24, 25, 26,   27,   29,    30,   32,    33, 34, 35,
        36, 37, 38,   39,   41,    42,   44,  45, 46, 47,
        48, 49, 50,   51, 52, 53,  54, 55, 56,  57, 58, 59,
        16, 19, 28, 31, 40, 43
    ]
    assert np.allclose(g.natur_map, ref_natur_map)
    # gives natural dof index for given calculation dof index
    # natural numbering comes from flattened (ix, iy, iz) dof coordinates
    # calculation numbering puts Dirichlet DOFs at the begining
    nx, ny, nz = g.shape
    for ix in range(g.shape[0]):
        for iy in range(g.shape[1]):
            for iz in range(g.shape[2]):
                i_dof_0 = (ix * (ny+1) + iy) * (nz+1) + iz

                natur_el_dofs = i_dof_0 + np.array([0, 1, (nz+1), (nz+1) + 1,
                                                    (nz+1)*(ny+1), (nz+1)*(ny+1) + 1, (nz+1)*(ny+2), (nz+1)*(ny+2) + 1])

                i_el = (ix * ny + iy) * nz + iz
                assert np.allclose(g.natur_map[g.el_dofs[i_el]], natur_el_dofs)
    # shape (n_elements, n_local_dofs), DOF indices in calculation numbering


def test_barycenters():
    origin = [-4, -5, -6]
    g = fem.Grid((100, 150, 200), (4, 3, 2), fem.Fe.Q(3, 1), origin=origin)
    xyz_grid = np.meshgrid(*[np.arange(n_els) for n_els in g.shape], indexing='ij')
    ref_barycenters = (np.stack(xyz_grid, axis=-1).reshape(-1, 3) + 0.5) * g.step + origin
    assert np.allclose(g.barycenters(), ref_barycenters)

# def test_nodes():
#     origin = [-4, -5, -6]
#     g = fem.Grid((100, 150, 200), (4, 3, 2), fem.Fe.Q(3, 1), origin=origin)
#     xyz_grid = np.meshgrid(*[np.arange(n_dofs) for n_dofs in g.dofs_shape], indexing='ij')
#     ref_barycenters = (np.stack(xyz_grid, axis=-1).reshape(-1, 3) + 0.5) * g.step + origin
#     assert np.allclose(g.nodes(), ref_barycenters)

def test_bc_all():
    origin = [-4, -5, -6]
    g = fem.Grid((100, 150, 200), (4, 3, 2), fem.Fe.Q(3, 1), origin=origin)
    ref_bc_coord = np.zeros(())
    nx,ny,nz = g.dofs_shape
    dof_grid = np.meshgrid(*[np.arange(n_dofs) for n_dofs in g.dofs_shape], indexing='ij')
    dof_coords = np.stack(dof_grid, axis=-1).reshape(-1, 3)
    ref_bc_coord = [dof_coords[g.natur_map[i_dof]] for i_dof in range(g.n_bc_dofs)]
    assert np.all(g.bc_coords == ref_bc_coord)
    assert np.allclose(g.bc_points[0], origin)
    max_corner = g.origin + g.dimensions
    bc_points = g.bc_points
    assert np.allclose(bc_points[g.dofs_shape[2]-1], [origin[0], origin[1], max_corner[2]])
    assert np.allclose(bc_points[g.dofs_shape[1]*g.dofs_shape[2]-1], [origin[0], max_corner[1], max_corner[2]])
    assert np.allclose(bc_points[-1], max_corner)

def grid_numbering_Q1(dim):
    order = 1
    g = fem.Grid(100.0, 4, fem.Fe.Q(dim, order))
    idx_to_coord = g.dof_coord_coef * g.step[None, :]
    ref_barycenters = np.arange(g.n_elements)
    np.allclose(g.barycenters(), )



#
# def test_grid_numbering():
#     # Test Grid numbering
#     for dim in [1, 2, 3]:
#         grid_numbering_Q1(dim)
#     dim = 1
#     order = 2
#     g = fem.Grid(100.0, 4, fem.Fe.Q(dim, order))
#     print(g)
#
#     dim = 2
#     order = 1
#     g = fem.Grid(100.0, 4, fem.Fe.Q(dim, order))
#     print(g)
#
#     dim = 3
#     order = 1
#     g = fem.Grid(100.0, 3, fem.Fe.Q(dim, order))
#     print(g)
#
# def test_grid_nodes():


def test_grid_bc():
    g = fem.Grid(10, 2, fem.Fe.Q(1, 1))
    assert np.all(g.bc_coords == np.array([[0], [2]]))
    assert np.allclose(g.bc_points, np.array([[0], [10]]))

    g = fem.Grid(10, 2, fem.Fe.Q(2, 1))
    ref = np.array([[0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2]]).T
    assert np.all(g.bc_coords == ref)

def test_laplace():
    order = 1
    N = 3
    dim = 2
    g = fem.Grid(N, N, fem.Fe.Q(dim, order))
    l = g.laplace.reshape((-1, g.fe.n_dofs, g.fe.n_dofs))
    print("\nlaplace, 2d:\n", l)

def test_grid_assembly():
    for dim in range(1, 4):
        order = 1
        N = 3
        g = fem.Grid(30, N, fem.Fe.Q(dim, order))
        K_const = np.diag(np.arange(1, dim + 1))
        K_const = fem.tn_to_voigt(K_const[None, :, :])
        K_field = K_const * np.ones(g.n_elements)[:, None]
        A = g.assembly_dense(K_field)
        n_dofs = (N+1)**dim
        assert A.shape == (n_dofs, n_dofs)

#@pytest.mark.skip
def test_solve_system():
    for dim in range(1, 4):
        order = 1
        N = 3
        g = fem.Grid(30, N, fem.Fe.Q(dim, order))
        K_const = np.diag(np.arange(1, dim + 1))
        K_const = fem.tn_to_voigt(K_const[None, :, :])
        K_field = K_const * np.ones(g.n_elements)[:, None]
        p_grads = np.eye(dim)
        pressure = g.solve_direct(K_field, p_grads)
        assert pressure.shape == (dim, *dim * [N + 1])


#@pytest.mark.skip
def test_solve_2d():
    dim = 2
    order = 1
    N = 30
    g = fem.Grid(100, N, fem.Fe.Q(dim, order))
    x = g.barycenters()[:, 0]
    K_const = np.diag([1, 1])
    #K_const = np.ones((dim, dim))
    K_const = fields.tn_to_voigt(K_const[None, :, :])
    K_field = K_const * x[:, None]
    #K_field = K_const.T * np.ones_like(x)[:, None]
    p_grads = np.eye(dim)
    pressure = g.solve_direct(K_field, p_grads)
    xy_grid = [np.linspace(0, g.dimensions[i], g.dofs_shape[i]) for i in range(2)]
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