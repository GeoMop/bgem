from bgem.gmsh.gmsh_io import GmshIO

import os
import filecmp


MESHES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")


def test_read():
    gio = GmshIO(os.path.join(MESHES_DIR, "cube_1x1x1_frac_coarse.msh2"))
    assert len(gio.nodes) == 75
    assert gio.nodes[1] == [-0.5, -0.5, -0.5]
    assert len(gio.elements) == 325
    assert gio.elements[1] == (1, [1, 9], [1, 19])
    assert len(gio.physical) == 5
    assert gio.physical[".1d_chanel"] == (1, 1)

    gio = GmshIO(os.path.join(MESHES_DIR, "flow.msh"))
    assert len(gio.element_data) == 2
    assert "pressure_p0" in gio.element_data
    assert len(gio.element_data["pressure_p0"]) == 14
    assert gio.element_data["pressure_p0"][0].time == 0.0
    assert len(gio.element_data["pressure_p0"][0].values) == 356


def test_write():
    gio = GmshIO(os.path.join(MESHES_DIR, "cube_1x1x1_frac_coarse.msh2"))
    gio.write(os.path.join(MESHES_DIR, "cube_test.msh"))
    gio = GmshIO(os.path.join(MESHES_DIR, "cube_test.msh"))
    gio.write(os.path.join(MESHES_DIR, "cube_test2.msh"))
    filecmp.cmp(os.path.join(MESHES_DIR, "cube_test.msh"), os.path.join(MESHES_DIR, "cube_test2.msh"))
    os.remove(os.path.join(MESHES_DIR, "cube_test.msh"))
    os.remove(os.path.join(MESHES_DIR, "cube_test2.msh"))
