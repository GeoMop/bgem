"""Reproduce incomplete flat-tetrahedron healing with four tetrahedra around AB.

The six nodes and four tetrahedra are the same as in reproduce_minimal.py.
ABCD is nearly flat; ABDE and ABFC share its faces, while ABEF shares only AB.
The old face-neighbour-only repair leaves ABEF unsplit after inserting M on AB.
These self-contained tests require no mesh files or Gmsh session. They check
complete subdivision, physical element tags, and the topology of the healed mesh.
"""
from collections import Counter
from types import SimpleNamespace

import numpy as np
from bgem.gmsh.heal_mesh import HealMesh


def minimal_mesh():
    """Build a fresh four-tetrahedron ring, with node IDs 1–6 denoting A–F."""
    h = 0.001
    # AB lies just above CD. Flattening ABCD makes their intersection M=(0,0,0).
    return SimpleNamespace(
        nodes=dict(enumerate(np.array([
            [-1, 0, h], [1, 0, h], [0, -1, -h], [0, 1, -h],
            [0, 1, 1], [0, -1, 1],
        ]), start=1)),
        elements={
            1: (4, [1, 1], [1, 2, 3, 4]),  # ABCD: sliver to remove
            2: (4, [1, 1], [1, 2, 4, 5]),  # ABDE: shares face ABD
            3: (4, [1, 1], [1, 2, 5, 6]),  # ABEF: shares only edge AB
            4: (4, [1, 1], [1, 2, 6, 3]),  # ABFC: shares face ABC
        },
    )


def faces(mesh):
    """Count tetrahedral face incidences using orientation-independent triples.

    One incident tetrahedron marks a boundary face, two an interior face, and
    more than two a nonmanifold face. Gmsh element type 4 denotes a tetrahedron.
    """
    return Counter(tuple(sorted(nodes[i] for i in face))
                   for typ, _, nodes in mesh.elements.values() if typ == 4
                   for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))


def test_quad_splits_complete_edge_stars():
    """Split edge-only neighbours, physical lines, and a physical triangle."""
    mesh = minimal_mesh()
    # Lines on both diagonals and triangle ABE share no sliver face. Their full
    # physical/geometrical tag pairs must survive in both children.
    mesh.elements.update({
        5: (1, [100, 200], [1, 2]),  # AB
        6: (1, [101, 201], [3, 4]),  # CD
        7: (2, [102, 202], [1, 2, 5]),  # ABE
    })

    # Invoke exactly the operation in the minimal reproducer, without other
    # repairs masking the missing subdivision. The new intersection node is 7.
    healer = HealMesh(mesh)
    healer.gamma_tol = 0.01
    healer._check_flat_tetra(healer._make_element(1))
    np.testing.assert_allclose(mesh.nodes[7], [0, 0, 0], atol=1e-14)
    for edge in ({1, 2}, {3, 4}):
        assert not any(edge <= set(nodes) for _, _, nodes in mesh.elements.values())

    # Three remaining tetrahedra must each become two, including ABEF, which
    # the old repair misses because it shares only an edge with the sliver.
    tetrahedra = [nodes for typ, _, nodes in mesh.elements.values() if typ == 4]
    assert len(tetrahedra) == 6
    assert all(7 in nodes for nodes in tetrahedra)
    assert all(tags == [1, 1] for typ, tags, _ in mesh.elements.values() if typ == 4)

    # Compare exact child connectivities and full tags, ignoring orientation.
    expected = {
        (1, (100, 200)): {(1, 7), (2, 7)},
        (1, (101, 201)): {(3, 7), (4, 7)},
        (2, (102, 202)): {(1, 5, 7), (2, 5, 7)},
    }
    actual = Counter((typ, tuple(tags), tuple(sorted(nodes)))
                     for typ, tags, nodes in mesh.elements.values() if typ != 4)
    assert actual == Counter((typ, tags, nodes)
                             for (typ, tags), children in expected.items()
                             for nodes in children)


def test_quad_healing_preserves_conforming_topology():
    """Check the whole healing loop against the hand-derived boundary faces."""
    mesh = minimal_mesh()
    healer = HealMesh(mesh)
    healer.gamma_tol = 0.01
    healer._check_flat_tetra(healer._make_element(1))
    healer.heal_mesh(gamma_tol=0.01)

    # Unlike a sliver enclosed inside a larger cavity, this sliver touches the
    # exterior at ACD and BCD. Flattening replaces these two boundary faces by
    # AMC, BMC, AMD, BMD. The other six exterior faces retain their connectivity.
    expected_boundary = {
        (1, 4, 5), (2, 4, 5),  # ADE, BDE
        (1, 5, 6), (2, 5, 6),  # AEF, BEF
        (1, 3, 6), (2, 3, 6),  # AFC, BFC
        (1, 3, 7), (2, 3, 7),  # AMC, BMC
        (1, 4, 7), (2, 4, 7),  # AMD, BMD
    }
    incidence = faces(mesh)
    assert max(incidence.values()) == 2
    assert {face for face, count in incidence.items() if count == 1} == expected_boundary

    # Signed volumes detect inversion, which gamma (using absolute volume)
    # cannot. All six children must also exceed the requested quality threshold.
    shapes = [healer._make_element(eid).shape for eid in mesh.elements]
    assert len(shapes) == 6
    assert min(shape.measure for shape in shapes) > 0
    assert min(shape.gmsh_gamma() for shape in shapes) > 0.01
