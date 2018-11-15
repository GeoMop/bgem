
import numpy as np
import numpy.linalg as la

class IsecCurveSurf:
    """
    Class which provides intersection of the given surface and curve
    """

    def __init__(self, surf, curv):
        self.surf = surf
        self.curv = curv


    def _compute_jacobian_and_delta(self, uvt, iu, iv, it):
        """
        Computes Jacobian matrix and delta vector of the function
        :param uvt: vector of local coordinates [u,v,t] (array 3x1)
        :param iu: index of the knot interval of the coordinate u
        :param iv: index of the knot interval of the coordinate v
        :param it: index of the knot interval of the coordinate t
        :return: J: jacobian matrix (array 3x3) , deltaXYZ: vector of deltas in R^3 space (array 3x1)
        """

        surf = self.surf
        curv = self.curv

        surf_poles = surf.poles[iu:iu + 3, iv:iv + 3, :]
        t_poles = curv.poles[it:it + 3, :]


        uf = surf.u_basis.eval_vector(iu, uvt[0, 0])
        vf = surf.v_basis.eval_vector(iv, uvt[1, 0])
        ufd = surf.u_basis.eval_diff_vector(iu, uvt[0, 0])
        vfd = surf.v_basis.eval_diff_vector(iv, uvt[1, 0])
        tf = curv.basis.eval_vector(it, uvt[2, 0])
        tfd = curv.basis.eval_diff_vector(it, uvt[2, 0])

        dXYZt = np.tensordot(tfd, t_poles, axes=([0], [0]))
        #print(dXYZt)
        dXYZu1 = self._energetic_inner_product(ufd, vf, surf_poles)
        dXYZv1 = self._energetic_inner_product(uf, vfd, surf_poles)
        J = np.column_stack((dXYZu1, dXYZv1, -dXYZt))

        XYZ1 = self._energetic_inner_product(uf, vf, surf_poles)
        XYZ2 = np.tensordot(tf, t_poles, axes=([0], [0]))
        #print(XYZ2)
        #return
        XYZ2 = XYZ2[:, None]
        XYZ1 = XYZ1[:, None]

        deltaXYZ = XYZ1 - XYZ2

        return J, deltaXYZ

    def get_intersection(self, uvt, iu, iv, it, max_it, rel_tol, abs_tol):
        """
        Main solving method for solving
        :param uvt: vector of initial guess of local coordinates [u,v,t] (array 3x1)
        :param iu: index of the knot interval of the coordinate u
        :param iv: index of the knot interval of the coordinate v
        :param it: index of the knot interval of the coordinate t
        :param max_it: maximum number of iteration
        :param rel_tol: relative tolerance (absolute in parametric space)
        :param abs_tol: absolute tolerance (in R3 space)
        :return: uvt: vector of initial guess of local coordinates [u,v,t] (array 3x1),
        conv as "0" if the methog does not achive desired accuracy
                "1" if the methog achive desired accuracy
        flag as intersection specification
        """
        conv = 0
        #flag = -1
        flag = -np.ones([3],'int')

        ui = self._compute_bounds(self.surf.u_basis.knots, iu)
        vi = self._compute_bounds(self.surf.v_basis.knots, iv)
        ti = self._compute_bounds(self.curv.basis.knots, it)

        for i in range(max_it):
            J, deltaXYZ = self._compute_jacobian_and_delta(uvt, iu, iv, it)
            if la.norm(deltaXYZ) < abs_tol:
                break
            uvt = uvt - la.solve(J, deltaXYZ)
            uvt = self._range_test(uvt, ui, vi, ti, 0.0)

        conv, XYZ = self._test_intesection_tolerance(uvt, iu, iv, it, abs_tol)

        if conv == 1:
            flag[0] = self._patch_boundary_intersection(uvt[0, 0], ui, rel_tol)
            flag[1] = self._patch_boundary_intersection(uvt[1, 0], vi, rel_tol)
            flag[2] = self._curve_boundary_intersection(uvt[2, 0], ti, rel_tol)


        return uvt, conv, flag, XYZ

    @staticmethod
    def _curve_boundary_intersection(t,ti,rel_tol):
        """

        :param t: parameter value
        :param it: interval array (2x1)
        :return:
        flag as "0" corresponds to lower bound of the curve
                "1" corresponds to upper bound of the curve
                "-1" corresponds to the interior points of the curve
        """
        # interior boundary
        if abs(ti[0] - t) < rel_tol:
            flag = 0
        elif abs(ti[1] - t) < rel_tol:
            flag = 1
        else:
            flag = -1

        return flag

    @staticmethod
    def _patch_boundary_intersection(t,ti,rel_tol):
        """

        :param t: parameter value
        :param it: interval array (2x1)
        :return:
        flag as "-1" corresponds to lower bound of the curve
                "1" corresponds to upper bound of the curve
                "0" corresponds to the boundary points of the curve
        """
        # interior boundary

        flag = -1

        if ti[0] != 0:
            if abs(ti[0] - t) < rel_tol:
                flag = 0
        elif ti[1] != 1:
            if abs(ti[1] - t) < rel_tol:
                flag = 1
        #else:
        #    flag = -1

        return flag


    @staticmethod
    def _compute_bounds(knots, idx):
        """
        Computes boundary of the given area (lower and upper) in parametric space
        :param knots: knot vector
        :param idx: index of the interval
        :return: bounds as (array 2x1) (lower bound, upper bound)
        """
        s = knots[idx + 2]
        e = knots[idx + 3]
        bounds = np.array([s, e])
        return bounds

    @staticmethod
    def _range_test(uvt, ui, vi, ti, rel_tol):
        """
        Test of the entries of uvt, lies in given intervals
        with a given tolerance
        :param uvt: vector of local coordinates [u,v,t] (array 3x1)
        :param ui: knot interval of the coordinate u (array 2x1)
        :param vi: knot interval of the coordinate v (array 2x1)
        :param ti: knot interval of the coordinate t (array 2x1)
        :param rel_tol: given relative tolerance (absolute in [u,v,t])
        :return: uvt: vector of local coordinates [u,v,t] (array 3x1)
        """

        test = 0

        du = np.array([ ui[0] - uvt[0, 0],uvt[0, 0] - ui[1]])
        dv = np.array([ vi[0] - uvt[1, 0],uvt[1, 0] - vi[1]])
        dt = np.array([ ti[0] - uvt[2, 0],uvt[2, 0] - ti[1]])

        for i in range(0, 2):
            if (du[i] > rel_tol):
                uvt[0, 0] = ui[i]

        for i in range(0, 2):
            if (dv[i] > rel_tol):
                uvt[1, 0] = vi[i]

        for i in range(0, 2):
            if (dt[i] > rel_tol):
                uvt[2, 0] = ti[i]

        #if np.logical_and(uvt[0, 0] >= ui[0], uvt[0, 0] <= ui[1]):
        #    if np.logical_and(uvt[1, 0] >= vi[0], uvt[1, 0] <= vi[1]):
        #        if np.logical_and(uvt[2, 0] >= ti[0], uvt[2, 0] <= ti[1]):
        #            test = 1

        return uvt

    def _test_intesection_tolerance(self, uvt, iu, iv, it, abs_tol):
        """
        Test of the tolerance of the intersections in R3
        :param uvt: vector of local coordinates [u,v,t] (array 3x1)
        :param iu: index of the knot interval of the coordinate u
        :param iv: index of the knot interval of the coordinate v
        :param it: index of the knot interval of the coordinate t
        :param abs_tol: given absolute tolerance in R^3 space
        :return: conv as "0" if the methog does not achive desired accuracy
         or "1" if the methog achive desired accuracy
        """

        conv = 0

        surf_R3 = self.surf.eval_local(uvt[0, 0], uvt[1, 0], iu, iv)
        curv_R3 = self.curv.eval_local(uvt[2, 0], it)
        XYZ = (surf_R3 + curv_R3)/2
        dist = la.norm(surf_R3 - curv_R3)
        #print('distance =', dist)
        if dist <= abs_tol:
            conv = 1

        return conv, XYZ

    def get_initial_condition(self, iu, iv, it):
        """
        Computes initial guess as mean value of the considered area
        :param iu: index of the knot interval of the coordinate u
        :param iv: index of the knot interval of the coordinate v
        :param it: index of the knot interval of the coordinate t
        :return: uvt as array (3x1)
        """

        uvt = np.zeros([3, 1])

        uvt[0, 0] = self._get_mean_value(self.surf.u_basis.knots, iu)
        uvt[1, 0] = self._get_mean_value(self.surf.v_basis.knots, iv)
        uvt[2, 0] = self._get_mean_value(self.curv.basis.knots, it)

        return uvt

    @staticmethod
    def _get_mean_value(knots, int):
        """
        Computes mean value of the local coordinate in the given interval
        :param knots: knot vector
        :param idx: index of the patch
        :return: mean
        """
        mean = (knots[int + 2] + knots[int + 3])/2

        return mean

    @staticmethod
    def _energetic_inner_product(u, v, surf_poles):
        """
        Computes energetic inner product u^T X v
        :param u: vector of nonzero basis function in u
        :param v: vector of nonzero basis function in v
        :param X: tensor of poles in x,y,z
        :return: xyz as array (3x1)
        """
        uX = np.tensordot(u, surf_poles, axes=([0], [0]))
        xyz = np.tensordot(uX, v, axes=([0], [0]))
        return xyz



#class IsecCurvSurf:
#    '''
#    Class for calculation and representation of the intersection of
#    B-spline curve and B-spline surface.
#    Currently only (multi)-point intersections are assumed.
#    '''
#    def __init__(self, curve, surface):
#        pass