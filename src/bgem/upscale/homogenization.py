import numpy as np
import logging


def equivalent_scalar(loads, responses):
    assert loads.shape[1] == responses.shape[1] == 1
    return np.dot(loads[:, 0], responses[:, 0]) / np.dot(loads[:, 0], loads[:, 0])

def equivalent_sym_tensor_3d(loads, responses):
    """
    :param loads: array, (N, dim), e.g grad pressure  grad_p(x)
    :param responses:  (N, dim), e.g. Darcian velocity -v(x) = K(x) grad_p(x)
    :return:
    """
    # from LS problem for 6 unknowns in Voigt notation: X, YY, ZZ, YZ, XZ, XY
    # the matrix has three blocks for Vx, Vy, Vz component of the responses
    # each block has different sparsity pattern
    n_loads = loads.shape[0]
    zeros = np.zeros(n_loads)
    ls_mat_vx = np.stack([loads[:, 0], zeros, zeros, zeros, loads[:, 2], loads[:, 1]], axis=1)
    rhs_vx = responses[:, 0]
    ls_mat_vy = np.stack([zeros, loads[:, 1], zeros, loads[:, 2], zeros, loads[:, 0]], axis=1)
    rhs_vy = responses[:, 1]
    ls_mat_vz = np.stack([zeros, zeros, loads[:, 2], loads[:, 1], loads[:, 0], zeros], axis=1)
    rhs_vz = responses[:, 2]
    ls_mat = np.concatenate([ls_mat_vx, ls_mat_vy, ls_mat_vz], axis=0)
    rhs = np.concatenate([rhs_vx, rhs_vy, rhs_vz], axis=0)
    assert ls_mat.shape == (3 * n_loads, 6)
    assert rhs.shape == (3 * n_loads,)
    result = np.linalg.lstsq(ls_mat, rhs, rcond=None)
    cond_tn_voigt, residuals, rank, singulars = result
    condition_number = singulars[0] / singulars[-1]
    if condition_number > 1e3:
        logging.warning(f"Badly conditioned inversion. Residual: {residuals}, max/min sing. : {condition_number}")
    return cond_tn_voigt


def equivalent_sym_tensor_2d(loads, responses):
    """
    :param loads: array, (N, dim), e.g grad pressure  grad_p(x)
    :param responses:  (N, dim), e.g. Darcian velocity -v(x) = K(x) grad_p(x)
    :return:
    """
    # from LS problem for 6 unknowns in Voigt notation: X, YY, ZZ, YZ, XZ, XY
    # the matrix has three blocks for Vx, Vy, Vz component of the responses
    # each block has different sparsity pattern
    n_loads = loads.shape[0]
    zeros = np.zeros(n_loads)
    ls_mat_vx = np.stack([loads[:, 0], zeros, loads[:, 1]], axis=1)
    rhs_vx = responses[:, 0]
    ls_mat_vy = np.stack([zeros, loads[:, 1], loads[:, 0]], axis=1)
    rhs_vy = responses[:, 1]
    ls_mat = np.concatenate([ls_mat_vx, ls_mat_vy], axis=0)
    rhs = np.concatenate([rhs_vx, rhs_vy], axis=0)
    assert ls_mat.shape == (2 * n_loads, 3)
    assert rhs.shape == (2 * n_loads,)
    result = np.linalg.lstsq(ls_mat, rhs, rcond=None)
    cond_tn_voigt, residuals, rank, singulars = result
    condition_number = singulars[0] / singulars[-1]
    if condition_number > 1e3:
        logging.warning(f"Badly conditioned inversion. Residual: {residuals}, max/min sing. : {condition_number}")
    return cond_tn_voigt


_equivalent_sym_tensor = {
    1: equivalent_scalar,
    2: equivalent_sym_tensor_2d,
    3: equivalent_sym_tensor_3d
}



def equivalent_posdef_tensor(loads, responses):
    # tensor pos. def.  <=> load . response > 0
    # ... we possibly modify responses to satisfy
    dim = loads.shape[0]
    assert dim == responses.shape[0]
    unit_loads = loads / np.linalg.norm(loads, axis=1)[:, None]
    load_components = np.sum(responses * unit_loads, axis=1)
    responses_fixed = responses + (np.maximum(0, load_components) - load_components)[:, None] * unit_loads

    return _equivalent_sym_tensor[dim](loads, responses_fixed)
