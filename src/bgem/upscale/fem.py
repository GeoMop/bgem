"""
Exact FEM based homogenization using regular d-dimansional grid.
"""
from functools import cached_property
import numpy as np
from .fields import tn_to_voigt, voigt_coords


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
    def Q1(cls, dim, order):
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
        Evaluate gradients of all tensor product basis functions at given points.
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
        n = self.n_dofs_1d
        grid_slice = tuple(self.dim * [slice(0, n)])
        return np.mgrid[grid_slice].reshape(self.dim, -1)

    def __repr__(self):
        return f"Q1(d={self.dim}, order={self.n_dofs_1d - 1})"



class Grid:
    """
    Regular grid, distribution of DOFs, System matrix assembly.
    """
    def __init__(self, size, n_steps, fe: Fe):
        """
        dim - dimension 1, 2, 3
        size    - domain size (Lx, Ly, Lz) or just scalar L for a cube domain
        n_steps - number of elements in each axis (nx, ny, nz) or just `n` for isotropic division
        """
        self.dim = fe.dim
        # Spatial dimension.
        self.size = size * np.ones(self.dim)
        # Array with physital dimensions of the homogenization domain.
        self.n_steps = n_steps * np.ones(self.dim, dtype=np.int64)
        # Int Array with number of elements for each axis.
        self.fe = fe
        # Tensor product finite element class.

        self.n_bc_dofs = 0
        # Number of bounday DOFs, first part of the calculation numbering of DOFs.
        self.natur_map = None
        # gives natural dof index for given calculation dof index
        # natural numbering comes from flattened (ix, iy, iz) dof coordinates
        # calculation numbering puts Dirichlet DOFs at the begining
        self.el_dofs = None
        # shape (n_local_dofs, n_elements), DOF indices in calculation numbering

        self.make_numbering(self.dim)

    @cached_property
    def step(self):
        # Array with step size in each axis.
        return self.size / self.n_steps

    @property
    def n_loc_dofs(self):
        return self.fe.n_dofs

    @property
    def dofs_shape(self):
        return self.n_steps * (np.array(self.fe.n_dofs_1d) - 1) + 1
    @property
    def n_dofs(self):
        return np.prod(self.dofs_shape)

    @property
    def n_elements(self):
        return np.prod(self.n_steps)

    @property
    def ax_dofs(self):
        """
        Number of dofs in each axis.
        :return:
        """
        return self.n_steps * (self.fe.n_dofs_1d - 1) + 1  # shape (dim, )

    @property
    def dof_to_coord(self):
        # Array for computiong global dof index from dof int coords.
        return np.cumprod([1, *self.ax_dofs[:-1]])

    @cached_property
    def bc_coords(self):
        bc_natur_indeces = self.natur_map[np.arange(self.n_bc_dofs, dtype=np.int64)]
        return self.idx_to_coord(bc_natur_indeces)


    @cached_property
    def bc_points(self):
        return self.bc_coords * self.step[:, None]

    def make_numbering(self, dim):
        # grid of integers, set to (-1)
        # go through boundary, enumerate, skip filled values
        # go through internal nodes, enumerate remaining
        # reshape -> computation_from_natural
        assert self.ax_dofs.shape == (self.dim,)
        n_dofs = np.prod(self.ax_dofs)
        # mark boundary dofs -1, interior dofs -2
        calc_map = np.full(self.ax_dofs, -1, dtype=np.int64)
        interior_slice = tuple(self.dim * [slice(1, -1)])
        calc_map[interior_slice] = -2

        # construct new numbering of dofs
        indices = np.where(calc_map == -1)
        self.n_bc_dofs = len(indices[0])
        # print(self.n_bc_dofs, indices)
        calc_map[indices] = np.arange(0, self.n_bc_dofs, dtype=np.int64)
        indices = np.where(calc_map == -2)
        calc_map[indices] = np.arange(self.n_bc_dofs, n_dofs, dtype=np.int64)
        calc_map = calc_map.flatten()
        self.natur_map = np.empty(len(calc_map), dtype=np.int64)
        self.natur_map[calc_map[:]] = np.arange(len(calc_map), dtype=np.int64)
        assert len(self.natur_map) == self.n_dofs

        # create element dofs mapping in natural dofs numbering
        ref_dofs = self.fe.ref_el_dofs()  # shape (dim, n_local_dofs)
        assert ref_dofs.shape == (self.dim, self.fe.n_dofs_1d ** self.dim)

        #print(ax.shape, ref_dofs.shape)
        ref_dofs = (self.dof_to_coord[None, :] @ ref_dofs).ravel()
        #print(ref_dofs.shape)

        # Creating a meshgrid for each dimension
        indices = np.meshgrid(*[np.arange(n) for n in self.n_steps], indexing='ij')

        # Calculating the tensor values based on the formula and axes
        el_dofs = np.zeros(self.n_steps, dtype=np.int64)
        o = self.fe.n_dofs_1d - 1
        for d in range(dim):
            el_dofs += (self.dof_to_coord[d] * o ** (d + 1)) * indices[d]
        #print(el_dofs)
        el_dofs = el_dofs[..., None] + ref_dofs[None, :]  # shape: nx, nY, nz, loc_dofs
        self.el_dofs = calc_map[el_dofs.reshape(-1, el_dofs.shape[-1])]

    def barycenters(self):
        """
        Barycenters of elements
        :return: shape (dim, n_els)
        """
        bary_axes = [self.step[i] * (np.arange(self.n_steps[i]) + 0.5) for i in range(self.dim)]
        return np.stack(np.meshgrid(*bary_axes)).reshape(self.dim, -1)

    def idx_to_coord(self, dof_natur_indices):
        """
        Produce physical coordinates for given natural dof indices.
        :param dof_natur_indices: np.int64 array
        :return: integer coordinates: (dim, *dof_natur_indeces.shape)
        """
        indices = dof_natur_indices
        coords = np.empty((self.dim, *dof_natur_indices.shape), dtype=np.int64)
        for i in range(self.dim-1, 0,  -1):
            indices, coords[i] = np.divmod(indices, self.dof_to_coord[i])
        coords[0] = indices
        return coords

    def __repr__(self):
        msg = \
            f"{self.fe} Grid: {self.n_steps} Domain: {self.size}\n" + \
            f"Natur Map:\n{self.natur_map}\n" + \
            f"El_DOFs:\n{self.el_dofs}\n"
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

        msh_params = self.dim * [points]
        points_tn = np.stack(np.meshgrid(*msh_params)).reshape((self.dim, -1))
        outer_params = [jac * weights[:, None] for jac in self.step]
        weights_tn = outer_product_along_first_axis(outer_params).ravel()
        grad = self.fe.grad_eval(points_tn)  # shape: (dim, n_loc_dofs, n_quads)
        # dim, n_loc_dofs, n_quad = grad.shape
        weight_grad = weights_tn[None, None :] * grad[:, :, :] # (dim, n_loc_dofs, n_quads)
        full_tn_laplace = grad[:, None, :, :] @ weight_grad[None, :, :, :].transpose(0, 1, 3, 2)

        M = [
            full_tn_laplace[i, j, :, :]
            if i==j else
            full_tn_laplace[i, j, :, :] + full_tn_laplace[j, i, :, :]
            for i, j in voigt_coords[self.dim]
        ]
        # M shape [n_voight, n_loc_dofs, n_loc_dofs]
        # return np.reshape(M, (len(M), -1)).T
        M = np.stack(M)
        assert M.shape == (len(voigt_coords[self.dim]), self.fe.n_dofs, self.fe.n_dofs)
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


    def assembly_dense(self, K_voight_tensors):
        """
        K_voight_tensors, shape: (n_elements, n_voight)
        """
        assert K_voight_tensors.shape == (self.n_elements, len(voigt_coords[self.dim]))
        laplace = self.laplace
        # locmat_dofs, n_voight = laplace.shape
        # Use transpositions of intputs and output in order to enforce cache efficient storage.
        #loc_matrices = np.zeros((self.n_loc_dofs self.n_loc_dofs, self.n_elements))
        #np.matmul(.T[:, None, :], laplace[None, :, :], out=loc_matrices.reshape(-1, self.n_elements).T)
        loc_matrices = K_voight_tensors[:, None, :] @ laplace[None, :, :]
        loc_matrices = loc_matrices.reshape((self.n_elements, self.fe.n_dofs, self.fe.n_dofs))
        A = np.zeros((self.n_dofs, self.n_dofs))
        # Use advanced indexing to add local matrices to the global matrix
        np.add.at(A, self.loc_mat_ij, loc_matrices)
        return A
    def solve_system(self, K, p_grad_bc):
        """
        :param K: array, shape: (n_voight, n_elements)
        :param p_grad_bc: array, shape: (dim, n_vectors)
        usually n_vectors >= dim
        :return: pressure, shape: (n_vectors, n_dofs)
        """
        d, n_rhs = p_grad_bc.shape
        assert d == self.dim
        A = self.assembly_dense(K)
        pressure_bc = p_grad_bc.T @ self.bc_points # (n_vectors, n_bc_dofs)
        B = pressure_bc @ A[:self.n_bc_dofs, self.n_bc_dofs:]  # (n_vectors, n_interior_dofs)
        pressure = np.empty((n_rhs, self.n_dofs))
        pressure[:, :self.n_bc_dofs] = pressure_bc
        pressure[:, self.n_bc_dofs:] = np.linalg.solve(A[self.n_bc_dofs:, self.n_bc_dofs:], -B.T).T
        pressure_natur = np.empty_like(pressure)
        pressure_natur[:, self.natur_map[:]] = pressure[:, :]
        return pressure_natur.reshape((n_rhs, *self.dofs_shape))

