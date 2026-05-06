"""
Exact FEM based homogenization using regular d-dimansional grid.
"""
from functools import cached_property
import numpy as np
from .fields import tn_to_voigt, voigt_to_tn, voigt_coords
from .homogenization import equivalent_posdef_tensor
#from bgem.stochastic import dfn
import scipy.sparse as sp
import pyamg

def Q1_1d_basis(points):
    """
    Coeeficients of 1D Q1 basis.

    return a_ij matrix
    where i row is i-th basis function with coefficients:
    p_i(x) = a_i0 + a_i1 * x + a_i2 * x**2 + ...
    """
    order = len(points)
    res = np.empty((order, order))
    prod = np.ones(order)
    for i in range(order):
        res[i] = prod
        prod = prod * points
    monomial_at_point = res
    return np.linalg.inv(monomial_at_point)

def poly_diff_1d(poly_functions):
    """
    poly_functions (n, m) shape array with m coefficients for n functions
    Returns derivatives of the functions, shape: (n, m-1)
    """
    order = poly_functions.shape[1]
    return poly_functions[:, 1:] * np.arange(1, order)



def eval_1d(poly_functions, x):
    """
    Evaluate polynomials `poly_functions`, shape (n, order) for
    vector of values `x`.
    return shape: (n, len(x))
    """
    #x = np.atleast_1d(x)
   # print((order, *x.shape))
    x = np.array(x)
    abs_coef = poly_functions[:, np.full_like(x, -1, dtype=np.int64)]
    # broadcast abs term to the result shape
    res = abs_coef
    for coef in poly_functions.T[-2::-1]:
        res = res * x[None, :] + coef[:, None]
    return res


# evaluation of tensor product basis functions
#
def flat_dim(x, dim):
    """
    Flatten dimension related axes, i.e. first $d$ axes of the $x$ array.
    x shape (n_1, ... n_d, other) -> (n_1 * .. * n_d, other)
    """
    return x.reshape((-1, *x.shape[dim:]))


def tensor_dim(x, dim, order):
    """
    x shape (order**dim, other) -> (order, ... , order, other)

    """
    assert x.shape[0] == order ** dim
    new_shape = (*(dim * [order]), *x.shape[1:])
    return x.reshape(new_shape)


def outer_product_along_first_axis(arrays):
    """
    arrays: list of `k` arrays a_i with shape [n_i, m]
    :param arrays:
    :return: res with shape [n_1, ... n_k, m]
    """
    _, n = arrays[0].shape
    result = arrays[0]
    for arr in arrays[1:] :
        assert arr.shape[1] == n
        result = result[..., np.newaxis, :] * arr
    return result


class Fe:
    """
    Tensor product basis.
    """

    @classmethod
    def Q(cls, dim, order=1):
        order = order + 1
        points = np.linspace(0, 1, order)
        basis = Q1_1d_basis(points)
        return cls(dim, basis)

    def __init__(self, dim, basis_1d):
        """
        """
        n, m = basis_1d.shape
        assert n == m
        self.n_dofs_1d = n
        self.dim = dim
        self.basis = basis_1d
        self.diff_basis = poly_diff_1d(basis_1d)

    @property
    def n_dofs(self):
        return self.n_dofs_1d ** self.dim

    def eval(self, points):
        """
        Evaluate all tensor product basis functions at given points.
        """
        dim, n_points = points.shape
        assert dim == self.dim
        #print(self.basis)
        #print(points.ravel())
        dim_basis_values = eval_1d(self.basis, points.ravel()).reshape(-1, self.dim, n_points)
        # shape (order, dim , n_points))
        tensor_values = outer_product_along_first_axis(dim_basis_values.transpose([1, 0, 2]))
        # shape (order, ... order, n_points)
        return flat_dim(tensor_values, self.dim)

    def grad_eval(self, points):
        """
        points: (dim, n_points)
        Evaluate gradients of all tensor product basis functions at given points.
        return: shape (dim, n_basis_fn, n_points)
        """
        dim, n_points = points.shape
        assert dim == self.dim
        dim_basis_values = eval_1d(self.basis, points.ravel()).reshape(-1, self.dim, n_points)
        # shape (order, dim , n_points))
        diff_vals = eval_1d(self.diff_basis, points.ravel())
        dim_diff_values = diff_vals.reshape(-1, self.dim, n_points)
        # shape (order, dim , n_points))

        result = []
        for i_dim in range(dim):
            diff_product_basis_list = [
                dim_diff_values[:, j_dim, :] if j_dim == i_dim else dim_basis_values[:, j_dim, :]
                for j_dim in range(dim)]
            prod_basis = outer_product_along_first_axis(diff_product_basis_list)
            result.append(flat_dim(prod_basis, self.dim))
        result = np.stack(result)
        # print(result.shape)
        return result

    def ref_el_dofs(self):
        """
        Positions of the DOFs on the reference element.
        ref_el_dofs[:, i] .. position of the i-th dofs
        :return: ndarray shape (dim, n_dofs)
        """
        n = self.n_dofs_1d
        grid_slice = tuple(self.dim * [slice(0, n)])
        return np.mgrid[grid_slice].reshape(self.dim, -1)

    def __repr__(self):
        return f"Q1(d={self.dim}, order={self.n_dofs_1d - 1})"



class Grid:
    """
    Regular anisotropic grid for a box in D dimensional ambient space.
    The cells are linearly numbered as np.empty(nx,ny,nz).flatten
    Cells numbered as C-style numpy array, last dimension running the fastest.
    """
    @classmethod
    def from_step(cls, dimensions, step, origin=0):
        shape = np.ceil(dimensions / step).astype(int)
        return cls(dimensions, shape, origin=origin)

    def __init__(self, dimensions, n_steps, origin=0):
        """
        dimensions: sequence of the physical dimensions of the grid box
        n_steps: sequence, number of cells in each axis, e.g. (nx, ny, nz)
                 or just int for same number of cells in each axis (n,n,n)
        origin: position of the grid node zero
        """
        self.dimensions = dimensions
        # Array with physital dimensions of the homogenization domain.
        self.dim = len(dimensions)
        # Dimension of the ambient space, e.g. 1d, 2d, 3d.
        self.shape = n_steps * np.ones(self.dim, dtype=np.int64)
        # Int Array with number of cells in each axis, i.e. shape of the grid
        self.origin = origin * np.ones(self.dim)
        # Absolute position of the node zero.

    @property
    def __len__(self):
        """
        Number of cells
        We undrestood the grid as a collection of cells.
        :return:
        """
        return np.prod(self.shape)

    @property
    def n_elements(self):
        return np.prod(self.shape)

    @cached_property
    def step(self):
        # Array with step size in each axis.
        return self.dimensions / self.shape

    def grid_center(self):
        """
        Return cooridinates of the barycenter of the whole grid.
        Result: np.array, shape = (3,)
        """
        return (2*self.origin + self.dimensions) / 2

    def barycenters(self):
        """
        Barycenters of elements.
        n_els = prod( n_steps )
        :return: shape (n_els, dim)
        """
        bary_axes = [self.step[i] * (np.arange(self.shape[i]) + 0.5) for i in range(self.dim)]
        mesh_grid = np.meshgrid(*bary_axes, indexing='ij')
        mesh_grid_array = np.stack(mesh_grid, axis=-1)
        return mesh_grid_array.reshape(-1, self.dim) + self.origin


    def __repr__(self):
        msg = f"Grid({self.shape}, domain: {self.dimensions}@{self.origin})"
        return msg

    def points_to_cell_coords(self, points):
        """
        For a point array of shape (N, 3) return
        array (N,3) of cell indices containing the points.
        Indices set to -1 if point is out of the grid.
        :param points:
        :return:
        """
        centers_ijk_grid = (points - self.origin) // self.step[None, :]
        centers_ijk_grid = centers_ijk_grid.astype(np.int32)
        out_of_grid = np.logical_or( centers_ijk_grid < 0, centers_ijk_grid >=self.shape[None,:])
        centers_ijk_grid[out_of_grid, :] = -1
        return centers_ijk_grid

    def coord_aabb(self, aabb):
        i_aabb = (aabb - self.origin) // self.step[None, :]
        i_aabb = i_aabb.astype(np.int32)
        i_aabb = np.maximum(i_aabb, 0)
        i_aabb = np.minimum(i_aabb, self.shape-1)
        return i_aabb

    def project_points(self, points):
        """
        :param points: array of shape (N, dim)
        :return: For each point the index of the containing grid cell.
        """
        #grid_min_corner = -grid.dimensions / 2
        centers_ijk_grid = (points - self.origin) // self.step[None, :]
        centers_ijk_grid = centers_ijk_grid.astype(np.int32)
        assert np.alltrue(centers_ijk_grid < self.shape[None, :])
        grid_cell_idx = centers_ijk_grid[:, 0] + self.shape[0] * (
                    centers_ijk_grid[:, 1] + self.shape[1] * centers_ijk_grid[:, 2])
        return grid_cell_idx

    def cell_box(self, min_cell, max_cell):
        """For given cell coords return array of cell ids in the box"""
        pass

    def axes_linspace(self):
        """
        Return list of linspaces, on for each axis.
        :return:
        """
        return [
            np.linspace(self.origin[ax], self.origin[ax] + self.dimensions[ax], self.shape[ax] + 1, dtype=np.float32)
            for ax in range(self.dim)
        ]

    def axes_cell_coords(self):
        """
        Return list of self.dim arrays specifying positions of the cell barycenters
        in each axis.
        :return:
        """
        return [
            np.arange(self.origin[ax] + self.step[ax]/2, self.origin[ax] + self.dimensions[ax], self.step[ax], dtype=np.float32)
            for ax in range(self.dim)
        ]

    def cell_field_C_like(self, cell_array_F_like):
        """
        :param cell_array: shape (n_elements, *value_dim) in F-like numbering
        :return: Same values rearranged for a C-like indexing, Z index running the fastest
        ... used in self
        """
        value_shape = cell_array_F_like.shape[1:]
        grid_field = cell_array_F_like.reshape(*reversed(self.shape), *value_shape)
        transposed = grid_field.transpose(*reversed(range(self.dim)))
        return transposed.reshape(-1, *value_shape)

    def cell_field_F_like(self, cell_array_C_like):
        """
        :param cell_array: shape (n_elements, *value_dim) in C-like numbering
        :return: Same values rearranged for a F-like indexing, X index running the fastest
        ... used in PyVista.
        """
        value_shape = cell_array_C_like.shape[1:]
        grid_field = cell_array_C_like.reshape(*self.shape, *value_shape)
        transposed = grid_field.transpose(*reversed(range(self.dim)),-1)
        return transposed.reshape(-1, *value_shape)


def fem_grid(dimensions, shape, fe, origin=None):
    """
    More compact construction of the FEM problem.
    :param dimensions:
    :param shape:
    :param fe:
    :param origin:
    :return:

    """
    dimensions = dimensions = dimensions * np.ones(fe.dim)
    if origin is None:
        origin = 0
    grid = Grid(dimensions, shape, origin)
    return FEM(grid, fe)

class FEM:
    def __init__(self, grid, fe):
        self.grid = grid
        # The grid of finite elements.
        self.fe = fe
        # Tensor product finite element class.
        assert self.grid.dim == self.fe.dim

        self.n_bc_dofs = 0
        # Number of bounday DOFs, first part of the calculation numbering of DOFs.
        self.natur_map = None
        # gives natural dof index for given calculation dof index
        # natural numbering comes from flattened (ix, iy, iz) dof coordinates
        # calculation numbering puts Dirichlet DOFs at the begining
        self.el_dofs = None
        # shape (n_elements, n_local_dofs), DOF indices in calculation numbering

        self.make_numbering(self.grid.dim)

    def make_numbering(self, dim):
        # grid of integers, set to (-1)
        # go through boundary, enumerate, skip filled values
        # go through internal nodes, enumerate remaining
        # reshape -> computation_from_natural
        assert self.dofs_shape.shape == (dim,)
        n_dofs = np.prod(self.dofs_shape)
        # mark boundary dofs -1, interior dofs -2
        calc_map = np.full(self.dofs_shape, -1, dtype=np.int64)
        interior_slice = tuple(dim * [slice(1, -1)])
        calc_map[interior_slice] = -2

        # construct new numbering of dofs
        el_indices = np.where(calc_map == -1)
        self.n_bc_dofs = len(el_indices[0])
        # print(self.n_bc_dofs, indices)
        calc_map[el_indices] = np.arange(0, self.n_bc_dofs, dtype=np.int64)
        el_indices = np.where(calc_map == -2)
        calc_map[el_indices] = np.arange(self.n_bc_dofs, n_dofs, dtype=np.int64)
        calc_map = calc_map.flatten()
        self.natur_map = np.empty(len(calc_map), dtype=np.int64)
        self.natur_map[calc_map[:]] = np.arange(len(calc_map), dtype=np.int64)
        assert len(self.natur_map) == self.n_dofs

        # create element dofs mapping in natural dofs numbering
        ref_dofs = self.fe.ref_el_dofs()  # shape (dim, n_local_dofs)
        assert ref_dofs.shape == (dim, self.fe.n_dofs_1d ** dim)

        #print(ax.shape, ref_dofs.shape)
        # Dof indices on the first cell.
        cell_0_dofs = (self.dof_coord_coef[None, :] @ ref_dofs).ravel()
        #print(ref_dofs.shape)

        # Creating a meshgrid for each dimension
        el_indices = np.meshgrid(*[np.arange(n) for n in self.grid.shape], indexing='ij')

        # Calculating the tensor values based on the formula and axes
        el_dofs = np.zeros(self.grid.shape, dtype=np.int64)
        o = self.fe.n_dofs_1d - 1
        for d in range(dim):
            el_dofs += (self.dof_coord_coef[d] * o ** (d + 1)) * el_indices[d]
        #print(el_dofs)
        el_dofs = el_dofs[..., None] + cell_0_dofs[None, :]  # shape: nx, nY, nz, loc_dofs
        self.el_dofs = calc_map[el_dofs.reshape(-1, el_dofs.shape[-1])]
        assert self.el_dofs.shape == (self.grid.n_elements, self.fe.n_dofs)


    @property
    def n_loc_dofs(self):
        return self.fe.n_dofs

    @property
    def dofs_shape(self):
        """
        Shape of DOFs grid.
        :return:
        """
        return self.grid.shape * (np.array(self.fe.n_dofs_1d) - 1) + 1

    @property
    def n_dofs(self):
        return np.prod(self.dofs_shape)

    # @property
    # def ax_dofs(self):
    #     """
    #     Number of dofs in each axis.
    #     :return:
    #     """
    #     return self.n_steps * (self.fe.n_dofs_1d - 1) + 1  # shape (dim, )

    @property
    def dof_coord_coef(self):
        # Array for computiong global dof index from dof int coords.
        #
        # idx = sum(coord * coord_coef)
        # 1D: [1]
        # 2D: [ny, 1]
        # 3D: [ny*nz, nz, 1]
        return np.cumprod([1, *self.dofs_shape[:0:-1]])[::-1]

    def get_dof_grid(self):
        return Grid(self.grid.dimensions,
                    self.grid.step / (self.fe.n_dofs_1d - 1),
                    self.origin)

    def nodes(self):
        """
        Nodes of the grid.
        n_nodes = prod( n_steps + 1 )
        :return: shape (n_nodes, dim)
        """
        node_coords = self.dof_idx_to_coord(np.arange(self.n_dofs, dtype=np.int64))
        return node_coords * self.grid.step[None, :] + self.grid.origin[None, :]  #mesh_grid_array.reshape(-1, self.dim) + self.origin

    @cached_property
    def bc_coords(self):
        """
        ?? todo transpose, refactor
        :return:
        """
        bc_natur_indeces = self.natur_map[np.arange(self.n_bc_dofs, dtype=np.int64)]
        return self.dof_idx_to_coord(bc_natur_indeces)

    @cached_property
    def bc_points(self):
        """
        todo refactor
        :return:
        """
        return self.bc_coords * self.grid.step[None, :] + self.grid.origin[None, :]


    def dof_idx_to_coord(self, dof_natur_indices):
        """
        Produce index coordinates (ix,iy,iz) for given natural dof indices.
        :param dof_natur_indices: np.int64 array, shape (n_dofs,)
        :return: integer coordinates: (len(dof_natur_indeces), self.dim)
        """
        indices = dof_natur_indices
        coords = np.empty((*dof_natur_indices.shape, self.grid.dim), dtype=np.int64)
        for i in range(self.grid.dim-1, 0,  -1):
            indices, coords[:, i] = np.divmod(indices, self.dofs_shape[i])
            #indices, coords[:, i] = np.divmod(indices, self.dof_to_coord[i])
        coords[:, 0] = indices
        return coords

    def __repr__(self):
        msg = f"FEM({self.fe} {self.grid})"
        return msg


    @cached_property
    def laplace(self):
        """
        Return matrix M. Shape (n_voigt, n_loc_dofs * n_loc_dofs).

        This should be used to assmebly the local matrices like:
        A_loc = M[None, :, :] @ K[:, None, :]
        where A_loc is array of local matrices (n_elements, n_loc_dofs**2)
        and K is (n_elements, K_tn_size), where K_tn_size is just upper triangle values
        i.e. 1, 3, 6 for dim=1, 2, 3.
        """
        # we integrate square of gradients, which is poly of degree 2*(deg -1) = 2deg - 2
        # Gaussian quadrature integrates exactly degree 2*deg -1
        deg = 2* (self.fe.n_dofs_1d - 1)   # 2 * degeree of base function polynomials

        # points and wights on [0, 1] interval
        points, weights = np.polynomial.legendre.leggauss(deg)
        points = 0.5 * (points + 1.0)
        weights = 0.5 * weights

        msh_params = self.grid.dim * [points]
        points_tn = np.stack(np.meshgrid(*msh_params)).reshape((self.grid.dim, -1))
        outer_params = [jac * weights[:, None] for jac in self.grid.step]
        weights_tn = outer_product_along_first_axis(outer_params).ravel()
        grad = self.fe.grad_eval(points_tn)  # shape: (dim, n_loc_dofs, n_quads)
        # dim, n_loc_dofs, n_quad = grad.shape
        weight_grad = weights_tn[None, None :] * grad[:, :, :] # (dim, n_loc_dofs, n_quads)
        full_tn_laplace = grad[:, None, :, :] @ weight_grad[None, :, :, :].transpose(0, 1, 3, 2)

        M = [
            full_tn_laplace[i, j, :, :]
            if i==j else
            full_tn_laplace[i, j, :, :] + full_tn_laplace[j, i, :, :]
            for i, j in voigt_coords[self.grid.dim]
        ]
        # M shape [n_voight, n_loc_dofs, n_loc_dofs]
        # return np.reshape(M, (len(M), -1)).T
        M = np.stack(M)
        assert M.shape == (len(voigt_coords[self.grid.dim]), self.fe.n_dofs, self.fe.n_dofs)
        return M.reshape((M.shape[0], -1))

    @cached_property
    def loc_mat_ij(self):
        """
        returns: rows, cols
        both with shape: (loc_dofs, loc_dofs, n_elements)
        Provides rows and cols for the local matrices.
        """
        n_elements, n_loc_dofs = self.el_dofs.shape
        rows = np.tile(self.el_dofs[:, :, None], [1, 1, n_loc_dofs])
        cols = np.tile(self.el_dofs[:, None, :], [1, n_loc_dofs, 1])
        return rows, cols

    def _loc_matrices(self, K_voight_tensors):
        """
        K_voight_tensors, shape: (n_elements, n_voight)
        """
        assert K_voight_tensors.shape == (self.grid.n_elements, len(voigt_coords[self.grid.dim]))
        laplace = self.laplace
        # n_voight, locmat_dofs  == laplace.shape
        # Use transpositions of intputs and output in order to enforce cache efficient storage.
        #loc_matrices = np.zeros((self.n_loc_dofs self.n_loc_dofs, self.n_elements))
        #np.matmul(.T[:, None, :], laplace[None, :, :], out=loc_matrices.reshape(-1, self.n_elements).T)
        loc_matrices = K_voight_tensors[:, None, :] @ laplace[None, :, :]
        return loc_matrices.reshape((self.grid.n_elements, self.fe.n_dofs, self.fe.n_dofs))

    def assembly_dense(self, K_voight_tensors):
        """
        K_voight_tensors, shape: (n_elements, n_voight)
        """
        loc_matrices = self._loc_matrices(K_voight_tensors)
        A = np.zeros((self.n_dofs, self.n_dofs))
        # Use advanced indexing to add local matrices to the global matrix
        np.add.at(A, self.loc_mat_ij, loc_matrices)
        return A

    def assembly_csr(self, K_voight_tensors):
        """
        K_voight_tensors, shape: (n_elements, n_voight)
        """
        rows, cols = self.loc_mat_ij
        values = self._loc_matrices(K_voight_tensors)
        rows, cols, values = map(np.ravel, [rows, cols, values])

        # Drop boundary rows.
        bulk_rows = rows >= self.n_bc_dofs
        rows, cols, values = [array[bulk_rows] for array in (rows, cols, values)]
        rows = rows - self.n_bc_dofs

        # Split boundary and bulk rows
        # See also np.split  for possible faster implementation
        n_dofs = self.n_dofs  - self.n_bc_dofs
        def sub_mat(rows, cols, vals, condition, n_cols, idx_shift):
            sub_cols = cols[condition] - idx_shift
            return sp.csr_matrix((vals[condition], (rows[condition], sub_cols)), shape=(n_dofs, n_cols))

        bulk_cols = (cols >= self.n_bc_dofs)
        A_bulk = sub_mat(rows, cols, values, bulk_cols, self.n_dofs - self.n_bc_dofs, self.n_bc_dofs)
        A_bc = sub_mat(rows, cols, values, ~bulk_cols, self.n_bc_dofs, 0)
        return A_bulk, A_bc

    def solve_direct(self, K, p_grad_bc):
        """
        :param K: array, shape: (n_elements, n_voight)
                  K = array of shape (*self.shape, n_voight).reshape(-1, n_voight)
                  cell at position (iX, iY, iZ) has index
                  (iX * self.shape[1] + iY) * self.shape[2]  +  iZ
                  i.e. the Z index is running fastest,
        :param p_grad_bc: array, shape: (n_vectors, dim)
        usually n_vectors >= dim
        :return: pressure, shape: (n_vectors, n_dofs)
        """
        n_rhs, d = p_grad_bc.shape
        assert d == self.grid.dim
        A = self.assembly_dense(K)
        bc_points_rel = self.bc_coords * self.grid.step[None, :]
        pressure_bc = p_grad_bc @ bc_points_rel.T # (n_vectors, n_bc_dofs)
        B = pressure_bc @ A[:self.n_bc_dofs, self.n_bc_dofs:]  # (n_vectors, n_interior_dofs)
        pressure = np.empty((n_rhs, self.n_dofs))
        pressure[:, :self.n_bc_dofs] = pressure_bc
        pressure[:, self.n_bc_dofs:] = np.linalg.solve(A[self.n_bc_dofs:, self.n_bc_dofs:], -B.T).T
        pressure_natur = np.empty_like(pressure)
        pressure_natur[:, self.natur_map[:]] = pressure[:, :]
        return pressure_natur.reshape((n_rhs, -1))


    def solve_sparse(self, K, p_grad_bc):
        """
        :param K: array, shape: (n_elements, n_voight)
                  K = array of shape (*self.shape, n_voight).reshape(-1, n_voight)
                  cell at position (iX, iY, iZ) has index
                  (iX * self.shape[1] + iY) * self.shape[2]  +  iZ
                  i.e. the Z index is running fastest,
        :param p_grad_bc: array, shape: (n_vectors, dim)
        usually n_vectors >= dim
        :return: pressure, shape: (n_vectors, n_dofs)
        """
        n_rhs, d = p_grad_bc.shape
        assert d == self.grid.dim
        A_bulk, A_bc = self.assembly_csr(K)
        bc_points_rel = self.bc_coords * self.grid.step[None, :]
        pressure_bc = p_grad_bc @ bc_points_rel.T # (n_vectors, n_bc_dofs)
        B =  (A_bc @ pressure_bc.T).T  # (n_vectors, n_interior_dofs)

        #pyamg.ruge_stuben_solver(A_bulk)
        solver = pyamg.smoothed_aggregation_solver(A_bulk, symmetry='symmetric')
        pressure = np.zeros((n_rhs, self.n_dofs))
        pressure[:, :self.n_bc_dofs] = pressure_bc
        for pressure_comp, b in zip(pressure, B):
            pressure_comp[self.n_bc_dofs:] = solver.solve(-b)
            #pressure[:, ] = np.linalg.solve(A[self.n_bc_dofs:, self.n_bc_dofs:], -B.T).T
        pressure_natur = np.empty_like(pressure)
        pressure_natur[:, self.natur_map[:]] = pressure[:, :]
        return pressure_natur.reshape((n_rhs, -1))

    def field_grad(self, dof_vals):
        """
        Compute solution gradient in element barycenters.
        :param dof_vals: (n_vec, n_dofs)
        :return: (n_vec, n_el, dim)
        """
        el_dof_vals = dof_vals[:, self.natur_map[self.el_dofs[:, :]]]  # (n_vec, n_el, n_loc_dofs)
        quads = np.full((self.grid.dim, 1), 0.5)   # Zero order Gaussian quad. Integrates up to deg = 1.
        grad_basis = self.fe.grad_eval(quads) # (dim, n_loc_dofs, 1)
        grad_els = grad_basis[None,None,:, :,0] @ el_dof_vals[:,:, :, None]
        return grad_els[:, :,:,0]


def upscale(K, domain=None):
    """

    :param K: array (nx, ny, nz, n_voigt) or similar for dim=1, 2
    :param domain: domain size array, default np.ones(dim)
    :return: Effective tensor.
    """
    dim = len(K.shape) - 1
    if domain is None:
        domain = np.ones(dim)

    order = 1
    fem = FEM(Grid(domain, K.shape[:-1]), Fe.Q(dim, order))
    p_grads = np.eye(dim)
    K_els = K.reshape((fem.grid.n_elements, -1))
    pressure = fem.solve_direct(K_els, p_grads)
    #xy_grid = [np.linspace(0, g.size[i], g.ax_dofs[i]) for i in range(2)]
    #fem_plot.plot_pressure_fields(*xy_grid, pressure)
    #pressure_flat = pressure.reshape((len(p_grads), -1))
    grad = fem.field_grad(pressure)   # (n_vectors, n_els, dim)
    loads = np.average(grad, axis=1) # (n_vectors, dim)
    full_K_els = voigt_to_tn(K_els)
    responses_els =  grad[:, :, None, :] @ full_K_els[None, :, :, :]   #(n_vec, n_els, 1, dim)
    responses = np.average(responses_els[:, :, 0, :], axis=1)
    return equivalent_posdef_tensor(loads, responses)


# def rasterize_dfn(fractures: dfn.FractureSet, step):
#     """
#     Rasterize given fracture to the grid with `step`
#     :param fractures:
#     :param step:
#     :return:
#     """
#     pass