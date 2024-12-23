import pytest
from pathlib import Path
from bgem import stochastic
import matplotlib.pyplot as plt
import numpy as np
import fixtures

script_dir = Path(__file__).absolute().parent
workdir = script_dir / "sandbox"

"""
Test base shapes.
"""


def test_ellipse_shape():
    shape = stochastic.EllipseShape


def plot_aabb(aabb, points, inside):
    # Prepare the AABB rectangle coordinates
    aabb_rect = np.array([
        aabb[ii, (0, 1)] for ii in [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    ])

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = np.where(inside, 'red', 'grey')
    ax.scatter(points[:, 0], points[:, 1], s=1, color=colors, label='Points')
    ax.plot(aabb_rect[:, 0], aabb_rect[:, 1], color='blue', linestyle='--', label='AABB')
    ax.set_aspect('equal', 'box')
    # Labels and Title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Scatter Plot with AABB')
    ax.legend()

    # Display the plot
    ax.grid(True)
    plt.show()


@pytest.mark.parametrize("base_shape",
                         [stochastic.EllipseShape(), stochastic.RectangleShape(), stochastic.PolygonShape(6),
                          stochastic.PolygonShape(8)]
                         )
def test_base_shapes(base_shape):
    """
    Use MC integration to:
    - confirm the shape has unit area
    - check it could determine interior points (but not that this check is correct)
    - check that the corresponding primitive could be made in GMSH interface
    - confirm that aabb is correct for that primitive
    :param shape:
    :return:
    """

    # Take AABB of the reference shape
    aabb = base_shape.aabb
    assert aabb.shape == (2, 2)
    assert np.allclose(aabb[0] + aabb[1], 0)
    safe_aabb = 2 * aabb

    # equivalence of is_point_inside ande are_points_inside
    N = 1000
    points = np.random.random((N, 2)) * (safe_aabb[1] - safe_aabb[0]) + safe_aabb[0]
    inside_single = [base_shape.is_point_inside(*pt) for pt in points]
    inside_vector = base_shape.are_points_inside(points)

    assert np.all(np.array(inside_single, dtype=bool) == inside_vector)
    out_of_aabb = np.logical_or.reduce((*(points < aabb[0]).T, *(aabb[1] < points).T))
    any_out = np.any(inside_vector & out_of_aabb)
    # if any_out:
    # plot_aabb(aabb, points, inside_vector)
    assert not any_out

    N = 100000
    points = np.random.random((N, 2)) * (aabb[1] - aabb[0]) + aabb[0]
    N_in = sum(base_shape.are_points_inside(points))
    aabb_area = np.prod(aabb[1] - aabb[0])
    area_estimate = N_in / N * aabb_area
    assert abs(area_estimate - 1.0) < 0.01


def check_orthogonal_columns(mat):
    product = mat.T @ mat
    product_offdiag = product - np.diag(np.diag(product))
    assert np.allclose(product_offdiag, np.zeros_like(product))


def check_fractures_transform_mat(fr_list):
    dfn = stochastic.FractureSet.from_list(fr_list)
    # BREP of the fractures
    # dfn.make_fractures_brep(workdir / "transformed")

    dfn_base = dfn.transform_mat @ np.eye(3)

    for i, fr in enumerate(fr_list):
        print(f"fr #{i}")
        base_vectors = dfn_base[i]
        assert base_vectors.shape == (3, 3)
        ref_base_1 = (fr.transform(np.eye(3)) - fr.center).T
        # Original fracture transform with respect to DFN transform matrix.
        assert np.allclose(dfn.center[i], fr.center)

        # Test base is perpendicular
        check_orthogonal_columns(ref_base_1)
        check_orthogonal_columns(base_vectors)

        assert np.allclose(base_vectors, ref_base_1), f"fr #{i} diff:\n {base_vectors}\n {ref_base_1}\n"

        # Tak a single fracture from DFN and compare its transform to the DFN transform.
        fr_2 = dfn[i]
        ref_base_2 = (fr_2.transform(np.eye(3)) - fr.center).T
        assert np.allclose(dfn.center[i], fr_2.center)
        assert np.allclose(base_vectors, ref_base_2)

    # Check rotation matrix
    assert np.allclose(dfn.rotation_mat.transpose((0, 2, 1)) @ dfn.rotation_mat, np.eye(3))

    # Check inverse transform
    assert np.allclose(dfn.inv_transform_mat @ dfn.transform_mat, np.eye(3))


def test_transform_mat():
    """
    Apply transform for
    :return:
    """
    # Tests without shape rotation.

    # shape_id = stochastic.EllipseShape.id
    shape_id = stochastic.RectangleShape.id
    fr = lambda s, c, n: stochastic.Fracture(shape_id, np.array(s), np.array(c), np.array(n) / np.linalg.norm(n))
    fractures = [
        fr([1, 1], [1, 2, 3], [0, 0, 1]),
        fr([2, 2], [1, 2, 3], [0, 0, 1]),
        fr([2, 3], [1, 2, 3], [1, 1, 0.2]),
        fr([2, 3], [1, 2, 3], [-1, -1, 0.2]),
        fr([2, 3], [1, 2, 3], [0, 0, -1]),
        fr([2, 3], [0, 0, 0], [0, 1, 0]),
        fr([2, 3], [0, 0, 0], [0, -1, 0]),
        fr([2, 3], [0, 0, 0], [1, 0, 0]),
        fr([2, 3], [0, 0, 0], [-1, 0, 0]),
        fr([2, 3], [0, 0, 0], [1, 2, 3]),
    ]
    check_fractures_transform_mat(fractures)

    fr = lambda s, c, n, ax: stochastic.Fracture(shape_id, np.array(s), np.array(c), np.array(n) / np.linalg.norm(n),
                                                 np.array(ax) / np.linalg.norm(ax))
    s = [2, 2]  # [2, 3]
    fractures = [
        fr(s, [0, 0, 0], [0, 0, 1], [1, 1]),
        fr(s, [0, 0, 0], [0, 0, 1], [-1, 1]),
        fr(s, [0, 0, 0], [0, 0, 1], [-1, -1]),
        fr(s, [0, 0, 0], [0, 0, 1], [1, -1]),
        fr(s, [0, 0, 0], [0, 0, 1], [1, 2]),
        fr(s, [0, 0, 0], [1, -0.5, -3], [1, 0]),

        fr(s, [1, 2, 3], [0, 1, 1], [1, 2]),  # out of order
        fr(s, [1, 2, 3], [1, -0.5, -3], [1, 2]),
        fr(s, [1, 2, 3], [1, -0.5, -3], [-1, -1]),  # out of order

        fr(s, [1, 2, 3], [0, 0, -1], [1, 0]),
        fr(s, [0, 0, 0], [0, 1, 0], [1, 0]),
        fr(s, [0, 0, 0], [0, -1, 0], [1, 0]),
        fr(s, [0, 0, 0], [1, 0, 0], [1, 0]),
        fr(s, [0, 0, 0], [-1, 0, 0], [1, 0]),
    ]
    check_fractures_transform_mat(fractures)

    fractures = fixtures.get_dfn_sample()
    check_fractures_transform_mat(fractures)

    # fracture.fr_intersect(fractures)

    # stochastic.Fracture(shape_id, np.array(s), np.array(c), np.array(n))


@pytest.mark.parametrize("base_shape",
                         [stochastic.EllipseShape(), stochastic.RectangleShape(), stochastic.PolygonShape(6),
                          stochastic.PolygonShape(8)]
                         )
def test_fracture_set_AABB(base_shape):
    fractures = fixtures.get_dfn_sample()
    base_polygon = base_shape.vertices(256 * 256)
    tight = 0
    for i, fr in enumerate(fractures):
        boundary_points = fractures.transform_mat[i] @ base_polygon[:, :, None] + fr.center
        min_corner, max_corner = fractures.AABB[i]
        assert np.all(min_corner[None, :] <= boundary_points)
        assert np.all(max_corner[None, :] >= boundary_points)
        l_tight = np.min(boundary_points - min_corner[None, :])
        u_tight = np.min(max_corner[None, :] - boundary_points)
        rel_tight = max(l_tight, u_tight) / np.linalg.norm(fractures.radius[i])
        tight = max(tight, rel_tight)
        print("Tight:", tight, rel_tight)
