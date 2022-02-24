import pytest
import numpy as np
import os
import filecmp
from bgem.bspline import brep_writer as bw
this_source_dir = os.path.dirname(os.path.realpath(__file__))

class TestLocation:
    def test_Location(self):
        print( "test locations")
        la = bw.Location([[1, 0, 0, 4], [0, 1, 0, 8], [0, 0, 1, 12]])
        lb = bw.Location([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        lc = bw.ComposedLocation([ (la, 2), (lb, 1) ])
        ld = bw.ComposedLocation([ (lb, 2), (lc, 3) ])
        with pytest.raises(bw.ParamError):
            bw.Location([1,2,3])
        with pytest.raises(bw.ParamError):
            bw.Location([[1], [2], [3]])
        with pytest.raises(bw.ParamError):
            a = 1
            b = 'a'
            lb = bw.Location([[a, b, a, b], [a, b, a, b], [a, b, a, b]])

    def test_transforms(self):
        points = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,1]]).T
        # Translate
        shift = [1,2,3]
        translate_loc = bw.Location.Translate([1,2,3])
        assert np.alltrue(translate_loc.matrix == np.array([[1,0,0,1],[0,1,0,2],[0,0,1,3]]))
        # Apply
        t_points = translate_loc.apply(points)
        assert np.alltrue(np.array([[2,1,1,2],[2,3,2,3],[3,3,4,4]]) == t_points)
        # Rotate
        rotate_loc = bw.Location.Rotate([1,1,1], 2 * np.pi / 3)
        r_points = rotate_loc.apply(points)
        ref_r_points = np.array([[0,1,0], [0,0,1], [1,0,0], [1,1,1]]).T
        assert np.allclose(ref_r_points, r_points)
        rotate_c_loc = bw.Location.Rotate([1, 1, 1], 2 * np.pi / 3, center=[1,0,0])
        rc_points = rotate_c_loc.apply(points)
        ref_rc_points = ref_r_points + np.array([1,-1,0])[:, None]
        assert np.allclose(ref_rc_points, rc_points)
        # Scale
        scale_loc = bw.Location.Scale([1,2,3], center=[0,1,0])
        s_points = scale_loc.apply(points)
        assert np.allclose(np.array([[1,0,0,1],[-1,1,-1,1],[0,0,3,3]]), s_points)


class TestConstructors:

    def test_Shape(self):

        # check sub types
        with pytest.raises(bw.ParamError):
            bw.Wire(['a', 'b'])
        v=bw.Vertex([1,2,3])
        with pytest.raises(bw.ParamError):
            bw.Wire([v, v])


    def test_Vertex(self):
        with pytest.raises(bw.ParamError):
            bw.Vertex(['a','b','c'])

def compare_brep(compound, ref_brep, location):
    full_ref_brep = os.path.join(this_source_dir,'ref_brepwriter', ref_brep)
    new_brep = '_'.join(ref_brep.split('_')[:-1]) + '_tst.brep'
    full_new_brep = os.path.join(this_source_dir,'ref_brepwriter', new_brep)
    with open(full_new_brep, "w") as f:
        bw.write_model(f, compound, location)

    return filecmp.cmp(full_new_brep, full_ref_brep)


def composed_location():
    loc1=bw.Location([[0,0,1,0],[1,0,0,0],[0,1,0,0]])
    loc2=bw.Location([[0,0,1,0],[1,0,0,0],[0,1,0,0]])
    cloc=bw.ComposedLocation([(loc1,1),(loc2,1)])
    return cloc


def prism():
    # 0, 0; top, bottom
    v1=bw.Vertex((0.0, 0.0, 1.0))
    v2=bw.Vertex((0.0, 0.0, 0.0))

    v3=bw.Vertex((0.0, 1.0, 1.0))
    v4=bw.Vertex((0.0, 1.0, 0.0))

    v5=bw.Vertex((1.0, 0.0, 1.0))
    v6=bw.Vertex((1.0, 0.0, 0.0))

    # vertical edges
    e1=bw.Edge([v1,v2])
    e2=bw.Edge([v3,v4])
    e3=bw.Edge([v5,v6])

    # face v12 - v34
    # top
    e4=bw.Edge([v1,v3])
    # bottom
    e5=bw.Edge([v2,v4])
    f1 = bw.Face([e1.m(), e4, e2, e5.m()])

    # face v34 - v56
    # top
    e6=bw.Edge([v3, v5])
    # bottom
    e7=bw.Edge([v4, v6])
    f2 = bw.Face([e2.m(), e6, e3, e7.m()])

    # face v56 - v12
    # top
    e8=bw.Edge([v5, v1])
    # bottom
    e9=bw.Edge([v6, v2])
    f3 = bw.Face([e3.m(), e8, e1, e9.m()])

    # top cup
    f4 = bw.Face([e4, e6, e8])
    # bot cup
    w5=bw.Wire([e5, e7, e9])
    f5 = bw.Face([w5.m()])

    shell = bw.Shell([ f1, f2, f3, f4, f5.m() ])
    cube_solid = bw.Solid([ shell ])
    model = bw.Compound([cube_solid])
    return model, composed_location()


def prism_perturbed():
    # prism with different numbering of edges and faces
    # 0, 0; top, bottom
    v1=bw.Vertex((0.0, 0.0, 1.0))
    v2=bw.Vertex((0.0, 0.0, 0.0))

    v3=bw.Vertex((0.0, 1.0, 1.0))
    v4=bw.Vertex((0.0, 1.0, 0.0))

    v5=bw.Vertex((1.0, 0.0, 1.0))
    v6=bw.Vertex((1.0, 0.0, 0.0))

    # face v12 - v34
    # top
    e4=bw.Edge([v1,v3])
    # top
    e6=bw.Edge([v3, v5])
    # top
    e8=bw.Edge([v5, v1])

    # top cup
    f4 = bw.Face([e4, e6, e8])

    # bottom
    e5=bw.Edge([v2,v4])
    # face v34 - v56
    # bottom
    e7=bw.Edge([v4, v6])
    # face v56 - v12
    # bottom
    e9=bw.Edge([v6, v2])

    # bot cup
    w5=bw.Wire([e5, e7, e9])
    f5 = bw.Face([w5.m()])


    # vertical edges
    e1=bw.Edge([v1,v2])
    e2=bw.Edge([v3,v4])
    e3=bw.Edge([v5,v6])

    f1 = bw.Face([e1.m(), e4, e2, e5.m()])
    f2 = bw.Face([e2.m(), e6, e3, e7.m()])
    f3 = bw.Face([e3.m(), e8, e1, e9.m()])

    shell = bw.Shell([ f4, f5.m(), f1, f2, f3 ])
    cube_solid = bw.Solid([ shell ])
    model = bw.Compound([cube_solid])
    return model, composed_location()


def tetrahedron():
    v0 = bw.Vertex([0,0,0])
    v1 = bw.Vertex([1,0,0])
    v2 = bw.Vertex([0,1,0])
    v3 = bw.Vertex([0,0,1])

    e01 = bw.Edge([v0, v1])
    e02 = bw.Edge([v0, v2])
    e03 = bw.Edge([v0, v3])
    e12 = bw.Edge([v1, v2])
    e23 = bw.Edge([v2, v3])
    e13 = bw.Edge([v1, v3])

    f1 = bw.Face([e01, e12, e02.m()])
    f2 = bw.Face([e02, e23, e03.m()])
    f3 = bw.Face([e03, e13.m(), e01.m()])
    f4 = bw.Face([e12, e23, e13.m()])

    shell = bw.Shell([f1, f2, f3, f4])
    s = bw.Solid([shell])
    return bw.Compound([s]), bw.Identity





@pytest.mark.parametrize("compound_fn",
    [prism, prism_perturbed, tetrahedron])
def test_geometries(compound_fn):
    name = compound_fn.__name__ + '_ref.brep'
    compound, location = compound_fn()
    assert compare_brep(compound, name, location)

