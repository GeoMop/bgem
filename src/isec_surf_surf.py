import sys
build_path="/home/jiri/Soft/Geomop/Intersections/external/bih/build"
sys.path+=[build_path]
print(sys.path)

import bih
import numpy as np
import numpy.linalg as la

import bspline as bs


class IsecPoint:

    def __init__(self, surf1,iu1,iv1,uv1,surf2,iu2,iv2,uv2):

        self.surf1 = surf1
        self.iu1 = iu1
        self.iv1 = iv1
        self.uv1 = uv1


        self.surf2 = surf2
        self.iu2 = iu2
        self.iv2 = iv2
        self.uv2 = uv2

        self.tol = 1

        R3_coor = 1

        #self.curv = curv
        #self.it = it


class IsecCurveSurf:
    """
    Class which provides intersection of the surface (patch with given interval  iu,iv) and curve with a given
    interval it
    """
    def __init__(self, surf,iu,iv,curv,it):
        self.surf = surf
        self.iu = iu
        self.iv = iv
        self.curv = curv
        self.it = it
        self.point = 1

    def _compute_jacobian_and_delta(self,uvt):
        """
        Computes Jacobian matrix and delta vector of the function
        :param uvt: vector of unknowns [u,v,t] (array 3x1)
        :return: J: jacobian  , deltaXYZ: vector of deltas  (array 3x3)
        """
        surf = self.surf
        curv = self.curv
        iu = self.iu
        iv = self.iv
        it = self.it

        surf_poles = surf.poles[iu:iu + 3, iv:iv + 3, :]
        t_poles = curv.poles[it:it+3,:]

        uf = surf.u_basis.eval_vector(iu, uvt[0, 0])
        vf = surf.v_basis.eval_vector(iv, uvt[1, 0])
        ufd = surf.u_basis.eval_diff_vector(iu, uvt[0, 0])
        vfd = surf.v_basis.eval_diff_vector(iv, uvt[1, 0])
        tf = curv.basis.eval_vector(it, uvt[2, 0])
        tfd = curv.basis.eval_diff_vector(it, uvt[2, 0])

        dXYZt = np.tensordot(tfd, t_poles, axes=([0], [0]))
        dXYZu1 = self._energetic_inner_product(ufd, vf, surf_poles)
        dXYZv1 = self._energetic_inner_product(uf, vfd, surf_poles)
        J = np.column_stack((dXYZu1, dXYZv1, -dXYZt))

        XYZ1 = self._energetic_inner_product(uf, vf, surf_poles)
        XYZ2 = np.tensordot(tf, t_poles, axes=([0], [1]))
        XYZ1 = XYZ1[:,None]
        deltaXYZ = XYZ1 - XYZ2

        return J, deltaXYZ

    def get_intersection(self,uvt,nit,rel_tol,abs_tol):
        """
        computes intersection of the patch and curve
        :param uvt: vector of initial condition (array 3x1)
        :param nit: number of iteration (scalar)
        :param rel_tol: relative tolerance (absolute in parametric space)
        :param abs_tol: absolute tolerance (in R3 space)
        :return: point as array (3x1) [u,v,t]
        """

        iu = self.iu
        iv = self.iv
        it = self.it

        ui = self._compute_bounds(self.surf.u_basis.knots, iu)
        vi = self._compute_bounds(self.surf.v_basis.knots, iv)
        ti = self._compute_bounds(self.curv.basis.knots, it)

        for i in range(nit):
            J, deltaXYZ = self._compute_jacobian_and_delta(uvt)
            uvt = uvt - la.solve(J, deltaXYZ)
            test, uvt = self._range_test(uvt, ui, vi, ti, 0.0)

        test, uvt = self._range_test(uvt, ui, vi, ti, rel_tol)

        if test == 1:
            conv = self._test_intesection_tolerance(uvt, iu, iv, it, abs_tol)

        return uvt, conv

    @staticmethod
    def _compute_bounds(knots,idx):
        """
        Computes bounds (lower and upper) of the patch (in parametric space)
        :param knots: knot vector
        :param idx: index of tha patch
        :return: bounds as array 2x1 (lower bound, upper bound)
        """
        s = knots[idx +2]
        e = knots[idx +3]
        bounds = np.array([s, e])
        return bounds

    @staticmethod
    def _range_test(uvt,ui, vi, ti, rel_tol):
        """
        Test of the entries of uvt, lies in given intervals
        with a given tolerance
        :param uvt:
        :param ui:
        :param vi:
        :param ti:
        :param rel_tol:
        :return:
        """

        test = 0

        du = np.array([uvt[0,0] - ui[0], ui[1] - uvt[0,0]])
        dv = np.array([uvt[1,0] - vi[0], vi[1] - uvt[1,0]])
        dt = np.array([uvt[2,0] - ti[0], ti[1] - uvt[2,0]])

        for i in range(0,2):
            if (du[i] < -rel_tol):
                uvt[0,0] = ui[i]

        for i in range(0,2):
            if (dv[i] < -rel_tol):
                uvt[1,0] = vi[i]

        for i in range(0,2):
            if (dt[i] < -rel_tol):
                uvt[2,0] = ti[i]

        if np.logical_and(uvt[0,0] >= ui[0],uvt[0,0] <= ui[1]):
            if np.logical_and(uvt[1,0] >= vi[0], uvt[1,0] <= vi[1]):
                if np.logical_and(uvt[2,0] >= ti[0], uvt[2,0] <= ti[1]):
                    test = 1

        return test, uvt

    def _test_intesection_tolerance(self,uvt, iu, iv, it,abs_tol):
        """
        Test of the intersections in R3
        :param uvt:
        :param iu:
        :param iv:
        :param it:
        :param abs_tol:
        :return:
        """

        conv = 0

        surf_R3 = self.surf.eval_local(uvt[0,0],uvt[1,0],iu, iv)
        curv_R3 = self.curv.eval_local(uvt[2,0],it)
        dist = la.norm(surf_R3 - curv_R3)

        if dist <= abs_tol:
            conv = 1

        return conv


    @staticmethod
    def _energetic_inner_product(u,v,surf_poles):
        """
        Computes energetic inner product u^T X v
        :param u: vector of nonzero basis function in u
        :param v: vector of nonzero basis function in v
        :param X: tensor of poles in x,y,z
        :return: xyz
        """
        #xyz = np.zeros([3,1])
        uX = np.tensordot(u, surf_poles, axes=([0], [0]))
        xyz = np.tensordot(uX, v, axes=([0], [0]))
        return xyz

class IsecSurfSurf:

    def __init__(self, surf1, surf2, nt=2,nit=10):
        self.surf1 = surf1
        self.surf2 = surf2
        self.box1, self.tree1 = self.bounding_boxes(self.surf1)
        self.box2, self.tree2 = self.bounding_boxes(self.surf2)
        self.nt = nt
        self.nit = nit
        self._ipoint_list = [] # append
        #tolerance

    def bounding_boxes(self,surf):
        """
        Compute bounding boxes and construct BIH tree for a given surface
        :param surf:
        :return:
        """
        tree = bih.BIH()
        n_patch = (surf.u_basis.n_intervals)*(surf.v_basis.n_intervals)


        patch_poles = np.zeros([9, 3, n_patch])
        i_patch = 0
        for iu in range(surf.u_basis.n_intervals):
            for iv in range(surf.v_basis.n_intervals):
                n_points = 0
                for i in range(0,3):
                    for j in range(0,3):
                        patch_poles[n_points,:,i_patch] = surf.poles[iu+i, iv+j, :]
                        n_points += 1
                assert i_patch == (iu * surf.v_basis.n_intervals + iv)
                i_patch += 1

        boxes = [bih.AABB(patch_poles[:,:,p].tolist()) for p in range(n_patch)]
        #print(patch_poles[:, :, 0])
        tree.add_boxes( boxes )
        tree.construct()
        #print(boxes)
        #for b in boxes:
        #    print(b.min()[0:2],b.max()[0:2])
        return boxes, tree


    def get_intersection(self):
        """
        Main method to get intersection points
        :return:
        """

        point_list = self._intersection(self.surf1,self.surf2,self.tree1,self.box2) # patches of surf 1 with respect threads of the surface 2
        point_list2 = self._intersection(self.surf2,self.surf1,self.tree2,self.box1)

        print(point_list.__len__())
        print(point_list2.__len__())


    def _intersection(self,surf1,surf2,tree1,box2):
        """
        Tries to compute intersection for every patches which does not have empty intersection of the bounding boxes
        :param surf1: surface which will be intersected
        :param surf2: surface which patches are attempted
        :param tree1: BIH tree of the bounding boxes of the patches of the surface 1
        :param box2: coordinates of the bounding boxes of the surface 2
        :return: list of points (intersections)
        """
        point_list = []
        for iu1 in range(surf1.u_basis.n_intervals):
            for iv1 in range(surf1.v_basis.n_intervals):
                s=0
                box_id = iv1 + surf1.v_basis.n_intervals * iu1
                intersectioned_patches1 = tree1.find_box(box2[box_id])
                for ipatch2 in intersectioned_patches1:
                    iu2 = int(np.floor(ipatch2/ surf2.v_basis.n_intervals))
                    iv2 = int(ipatch2 - (iu2 * surf2.v_basis.n_intervals))
                    assert ipatch2 == iu2 * surf2.v_basis.n_intervals + iv2
                    points = self._patch_patch_intersection(surf1,iu1, iv1, surf2, iu2, iv2)
                    if points.__len__() != 0:
                        point_list.append(points)

        return point_list

    @staticmethod
    def _compute_bounds(knots,idx):
        """
        Computes bounds of the intervals of the knot vector
        :param knots: knot vector
        :param idx: index of the interval
        :return: s,e,c (lower bound, upper bound, center)
        """
        s = knots[idx +2]
        e = knots[idx +3]
        c = (s + e)/2
        return s,e,c

    def _patch_patch_intersection( self,surf1,iu1, iv1, surf2, iu2, iv2):
        """
        Intersection of two patches and its reduction to curve patch intersection

        :param surf1: intersected surface
        :param iu1: u coordinate of the patch
        :param iv1: v coordinate of the patch
        :param surf2: surface which is broken into main curves
        :param iu2: u coordinate of the patch
        :param iv2: v coordinate of the patc
        :return:
        """

        nit = self.nit
        nt = self.nt

        r3_coor = np.zeros([3, 1])
        uv1 = np.zeros([2, 1])
        uv2 = np.zeros([2, 1])

        abs_tol = 1e-6 # in x,y,z
        rel_tol = 1e-4 # in u,v

        u2s, u2e, u2c = self._compute_bounds(surf2.u_basis.knots, iu2)
        v2s, v2e, v2c = self._compute_bounds(surf2.v_basis.knots, iv2)
        u1c = (surf1.u_basis.knots[iu1+2] + surf1.u_basis.knots[iu1+3])/2
        v1c = (surf1.v_basis.knots[iv1+2] + surf1.v_basis.knots[iv1+3])/2

        points = []

        sum_idx = 0 # u2_fixed
        for w in np.linspace(u2s,u2e,nt):

            # initial condition
            uvt = np.zeros([3, 1])
            uvt[0, 0] = u1c
            uvt[1, 0] = v1c
            uvt[2, 0] = v2c

            # curve init
            u2f = surf2.u_basis.eval_vector(iu2,w)
            surf_pol = surf2.poles[iu2:iu2 + 3, :,:]
            curv_pol = np.tensordot(u2f, surf_pol, axes=([0], [sum_idx]))
            curv_basis = surf2.v_basis
            it = iv2

            curv = bs.Curve(curv_basis,curv_pol)
            curve_surf_isec = IsecCurveSurf(surf1,iu1,iv1,curv,it)
            uvt, conv = curve_surf_isec.get_intersection(uvt,nit,rel_tol,abs_tol)

            if conv == 1:
                uv1 = uvt[0:1, :]
                uv2[0, 0] = w
                uv2[1, 0] = uvt[2, 0]
                point = IsecPoint(surf1,iu1,iv1,uv1,surf2,iu2,iv2,uv2)
                points.append(point)

        sum_idx = 1 # v2_fixed
        for w in np.linspace(v2s,v2e,nt):

            # initial condition
            uvt = np.zeros([3, 1])
            uvt[0, 0] = u1c
            uvt[1, 0] = v1c
            uvt[2, 0] = u2c

            # curve init
            v2f = surf2.v_basis.eval_vector(iv2,w)
            surf_pol = surf2.poles[:,iv2:iv2 + 3,:]
            curv_pol = np.tensordot(v2f, surf_pol, axes=([0], [sum_idx]))
            it = iu2

            curv = bs.Curve(curv_basis,curv_pol)
            patch_curve_isec = IsecCurveSurf(surf1,iu1,iv1,curv,it)
            uvt, conv = patch_curve_isec.get_intersection(uvt,nit,rel_tol,abs_tol)

            if conv == 1:
                uv1 = uvt[0:1, :]
                uv2[0, 0] = uvt[2, 0]
                uv2[1, 0] = w
                point = IsecPoint(surf1,iu1,iv1,uv1,surf2,iu2,iv2,uv2)
                points.append(point)

        return points
    '''
    Calculation and representation of intersection of two B-spline surfaces.
    
    Result is set of B-spline segments approximating the intersection branches.
    Every segment consists of: a 3D curve and two 2D curves in parametric UV space of the surfaces.
    '''
    