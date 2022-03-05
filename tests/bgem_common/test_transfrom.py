import pytest
import numpy as np
from bgem import Transform, ParamError

class TestLocation:

    def test_transforms(self):
        points = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,1]]).T
        # Translate
        translate_loc = Transform.Translate([1,2,3])
        assert np.alltrue(translate_loc.matrix == np.array([[1,0,0,1],[0,1,0,2],[0,0,1,3]]))
        # Apply
        t_points = translate_loc(points)
        assert np.alltrue(np.array([[2,1,1,2],[2,3,2,3],[3,3,4,4]]) == t_points)
        # Rotate
        rotate_loc = Transform.Rotate([0,0,1], np.pi / 4)
        r_points = rotate_loc(rotate_loc(points))
        ref_r_points = np.array([[0,1,0], [-1,0,0], [0,0,1], [-1,1,1]]).T
        assert np.allclose(ref_r_points, r_points)
        rotate_loc = Transform.Rotate([1,1,1], 2 * np.pi / 3)
        r_points = rotate_loc(points)
        ref_r_points = np.array([[0,1,0], [0,0,1], [1,0,0], [1,1,1]]).T
        assert np.allclose(ref_r_points, r_points)
        rotate_c_loc = Transform.Rotate([1, 1, 1], 2 * np.pi / 3, center=[1,0,0])
        rc_points = rotate_c_loc(points)
        ref_rc_points = ref_r_points + np.array([1,-1,0])[:, None]
        assert np.allclose(ref_rc_points, rc_points)
        # Scale
        scale_loc = Transform.Scale([1,2,3], center=[0,1,0])
        s_points = scale_loc(points)
        assert np.allclose(np.array([[1,0,0,1],[-1,1,-1,1],[0,0,3,3]]), s_points)

        # Compose
        la = Transform([[1, 0, 0, 4], [0, 1, 0, 8], [0, 0, 1, 12]])
        lb = Transform([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        l_pow = la ** 2
        assert np.allclose(la(la(points)), l_pow(points))

        lc = (lb @ l_pow)
        lc_points = lc(points)
        lc_points_ref = lb(la(la(points)))
        assert np.allclose(lc_points_ref, lc_points)

        ld = Transform(lb.matrix, 1, Transform(la.matrix, 2))
        ld_points = ld(points)
        assert np.allclose(lc_points_ref, ld_points)

        # Errors
        with pytest.raises(ParamError):
            Transform([1,2,3])
        with pytest.raises(ParamError):
            Transform([[1], [2], [3]])
        with pytest.raises(ParamError):
            a = 1
            b = 'a'
            Transform([[a, b, a, b], [a, b, a, b], [a, b, a, b]])
