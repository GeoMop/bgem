import sys
build_path="/home/jiri/Soft/Geomop/Intersections/external/bih/build"
sys.path+=[build_path]
print(sys.path)

import bih
import numpy as np
import numpy.linalg as la

import bspline as bs


class IsecPoint:

    def __init__(self, surf1,iu1,iv1,surf2,iu2,iv2):

        self.surf1 = surf1
        self.iu1 = iu1
        self.iv1 = iv1

        self.surf2 = surf2
        self.iu2 = iu2
        self.iv2 = iv2

        self.tol = 1

        R3_coor = 1

        self.curv = curv
        self.it = it


class IsecCurveSurf:

    def __init__(self, surf,iu,iv,curv,it):
        self.surf = surf
        self.iu = iu
        self.iv = iv
        self.curv = curv
        self.it = it
        self.point = 1

    def _compute_jacobian_and_delta(self,uvt):
        """
        :param uvt: vector of unknowns [u,v,t]
        :return: J: jacobian  , deltaXYZ: vector of deltas
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
        #tf = t_basis.eval_vector(it, uvt[2, 0])
        tf = curv.basis.eval_vector(it, uvt[2, 0])
        tfd = curv.basis.eval_diff_vector(it, uvt[2, 0])
        #print('jacobi')
        dXYZt = np.tensordot(tfd, t_poles, axes=([0], [0]))
        #print(dXYZt)
        dXYZu1 = self._energetic_inner_product(ufd, vf, surf_poles)
        dXYZv1 = self._energetic_inner_product(uf, vfd, surf_poles)
        J = np.column_stack((dXYZu1, dXYZv1, -dXYZt))
        #print(J)

        XYZ1 = self._energetic_inner_product(uf, vf, surf_poles)
        XYZ2 = np.tensordot(tf, t_poles, axes=([0], [1]))

        XYZ1 = XYZ1[:,None]

        deltaXYZ = XYZ1 - XYZ2

        return J, deltaXYZ

    def get_intersection(self,uvt,nit,rel_tol,abs_tol):
        """

        :param uvt: vector of initial condition (array 3x1)
        :param nit: number of iteration (scalar)
        :param rel_tol: relative tolerance (absolute in parametric space)
        :param abs_tol: absolute tolerance (in R3 space)
        :return: point as array (3x1)
        """

        ui = self._compute_bounds(self.surf.u_basis.knots, self.iu)
        vi = self._compute_bounds(self.surf.v_basis.knots, self.iv)
        ti = self._compute_bounds(self.curv.basis.knots, self.it)

        for i in range(nit):
            J, deltaXYZ = self._compute_jacobian_and_delta(uvt)
            uvt = uvt - la.solve(J, deltaXYZ)
            test, uvt = self._rangetest(uvt, ui, vi, ti, 0.0)

        test, uvt = self._rangetest(uvt, ui, vi, ti, rel_tol)

        if test == 1:
            conv = self._test_intesection_tolerance(uvt, abs_tol)

        if conv != 1:
            uvt = -np.ones([3,1])

        return uvt

    @staticmethod
    def _compute_bounds(knots,idx):
        """
        Computes bounds of the patch (in parametric space)
        :param knots: knot vector
        :param idx: index of tha patch
        :return: bounds as array 2x1 (lower bound, upper bound)
        """
        s = knots[idx +2]
        e = knots[idx +3]
        bounds = np.array([s, e])
        return bounds

    @staticmethod
    def _rangetest(uvt,ui, vi, ti, rel_tol):
    # test if paramaters does not lie outside current patch, otherwise they are returned to the boundary
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

    def _test_intesection_tolerance(self,uvt,abs_tol):

        conv = 0

        surf_R3 = self.surf.eval(uvt[0,0],uvt[1,0])
        curv_R3 = self.curv.eval(uvt[2,0])
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
        #tolerance

    def bounding_boxes(self,surf):
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



    #@staticmethod
    #def get_intersection(surf1,surf2,tree1,tree2,box1,box2,nt,nit):
    def get_intersection(self):
        # surf1, surf2, tree1, tree2, box1, box2, nt, nit
    # nt - number of threads (integer)
            #X,Xs, u_n_intervs,v_n_intervs,u_knots,v_knots,
     #us_n_intervs,vs_n_intervs,us_knots,vs_knots,isec,n_isec, nit,nt )
        # computes intersection of BSpline patch with BSpline thread


        point, n_points, ninter = self._intersection(self.surf1,self.surf2,self.tree1,self.tree2,self.box1,self.box2)
        # surf1, surf2, tree1, tree2, box1, box2, nt, nit

    def _intersection(self,surf1,surf2,tree1,tree2,box1,box2): # use

        #surf1, surf2, tree1, tree2, box1, box2, nt, nit
        n_points = 0
        point = np.zeros([surf1.u_basis.n_intervals * surf1.v_basis.n_intervals, 11])
        #ninter =  np.zeros([surf1.u_basis.n_intervals,surf1.v_basis.n_intervals])
        for iu1 in range(surf1.u_basis.n_intervals):
            for iv1 in range(surf1.v_basis.n_intervals):
                s=0
                box_id = iv1 + surf1.v_basis.n_intervals * iu1
                intersectioned_patches1 = tree1.find_box(box2[box_id])
                for ipatch2 in intersectioned_patches1:
                    iu2 = int(np.floor(ipatch2/ surf2.v_basis.n_intervals))
                    iv2 = int(ipatch2 - (iu2 * surf2.v_basis.n_intervals))
                    assert ipatch2 == iu2 * surf2.v_basis.n_intervals + iv2
                    uv1, uv2, r3_coor, conv = self._patch_patch_intersection(surf1,iu1, iv1, surf2, iu2, iv2)
                    print(uv1)
                    print(uv2)

                    if np.not_equal(conv, 0):
                        s = s+1
                        n_points = n_points +1
                        print(r3_coor[0], r3_coor[1], r3_coor[2])
                        print(iu2, iv2, iu1, iv1)
                        print(uv1[0], uv1[1], uv2[0], uv2[1], r3_coor[0], r3_coor[1], r3_coor[2], iu2, iv2, iu1, iv1)
                        point[n_points,:] = uv1[0], uv1[1], uv2[0], uv2[1], r3_coor[0], r3_coor[1], r3_coor[2], iu2, iv2, iu1, iv1
                #ninter[iu1,iv1] = s

        return point, n_points, ninter

    # @staticmethod
    # def _energetic_inner_product(u,v,X):
    #     """
    #     Computes energetic inner product u^T X v
    #     :param u: vector of nonzero basis function in u
    #     :param v: vector of nonzero basis function in v
    #     :param X: tensor of poles in x,y,z
    #     :return: xyz
    #     """
    #     #xyz = np.zeros([3,1])
    #     uX = np.tensordot(u, X, axes=([0], [0]))
    #     xyz = np.tensordot(uX, v, axes=([0], [0]))
    #     return xyz

    @staticmethod
    def _compute_bounds(knots,idx):
        """
        Computes bounds of the patch
        :param knots: knot vector
        :param idx: index of tha patch
        :return: s,e,c (lower bound, upper bound, center)
        """
        s = knots[idx +2]
        e = knots[idx +3]
        c = (s + e)/2
        return s,e,c

    def _compute_jacobian_and_delta(self,uvt,surf1,iu1,iv1,X,t_poles,t_basis,it):
        """

        :param uvt: vector of unknowns [u1,v1,t]
        #:param sum_idx: index which determines sumation - u fixed = 0, v fixed = 1
        :param surf1: intersected surface
        :param iu1: index of the patch of the surface (surf1) in u
        :param iv1: index of the patch of the surface (surf1) in u
        :param X: poles of the surface (surf1)
        :param t_poles: poles of curve
        :param t_basis: function basis of the curve
        :param it: index of the interval of the curve
        :return: J: jacobian  , deltaXYZ: vector of deltas
        """

        uf = surf1.u_basis.eval_vector(iu1, uvt[0, 0])
        vf = surf1.v_basis.eval_vector(iv1, uvt[1, 0])
        ufd = surf1.u_basis.eval_diff_vector(iu1, uvt[0, 0])
        vfd = surf1.v_basis.eval_diff_vector(iv1, uvt[1, 0])
        tf = t_basis.eval_vector(it, uvt[2, 0])
        tfd = t_basis.eval_diff_vector(it, uvt[2, 0])
        #print('jacobi')
        dXYZt = np.tensordot(tfd, t_poles, axes=([0], [0]))
        #print(dXYZt)
        dXYZu1 = self._energetic_inner_product(ufd, vf, X)
        dXYZv1 = self._energetic_inner_product(uf, vfd, X)
        J = np.column_stack((dXYZu1, dXYZv1, -dXYZt))
        #print(J)

        XYZ1 = self._energetic_inner_product(uf, vf, X)
        XYZ2 = np.tensordot(tf, t_poles, axes=([0], [1]))

        XYZ1 = XYZ1[:,None]

        deltaXYZ = XYZ1 - XYZ2

        return J, deltaXYZ

    def _patch_patch_intersection( self,surf1,iu1, iv1, surf2, iu2, iv2):
        """

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

        conv = 0
        abs_tol = 1e-6 # in x,y,z
        rel_tol = 1e-4 # in u,v


        u1s, u1e, u1c = self._compute_bounds(surf1.u_basis.knots,iu1)
        v1s, v1e, v1c = self._compute_bounds(surf1.v_basis.knots, iv1)
        u2s, u2e, u2c = self._compute_bounds(surf2.u_basis.knots, iu2)
        v2s, v2e, v2c = self._compute_bounds(surf2.v_basis.knots, iv2)

        u1i = np.array([u1s, u1e])
        v1i = np.array([v1s, v1e])
        u2i = np.array([u2s, u2e])
        v2i = np.array([v2s, v2e])


        sum_idx = 0 # u2_fixed
        for w in np.linspace(u2i[0],u2i[1],nt):

            # initial condition
            uvt = np.zeros([3, 1])
            uvt[0, 0] = u1c
            uvt[1, 0] = v1c

            # curve init
            u2f = surf2.u_basis.eval_vector(iu2,w)
            surf_pol = surf2.poles[iu2:iu2 + 3, :,:]
            curv_pol = np.tensordot(u2f, surf_pol, axes=([0], [sum_idx]))
            curv_basis = surf2.v_basis

            it = iv2
            uvt[2, 0] = v2c

            curv = bs.Curve(curv_basis,curv_pol)
            curve_surf_isec = IsecCurveSurf(surf1,iu1,iv1,curv,it)

            uvt = curve_surf_isec.get_intersection(uvt,nit,rel_tol,abs_tol)

            uv1 = uvt[0:1, :]
            uv2[0, 0] = w
            uv2[1, 0] = uvt[2, 0]


        sum_idx = 1 # v2_fixed
        for w in np.linspace(v2i[0],v2i[1],nt):

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

            uvt = patch_curve_isec.get_intersection(uvt,nit,rel_tol,abs_tol)


            uv1 = uvt[0:1, :]
            uv2[0, 0] = uvt[2, 0]
            uv2[1, 0] = w

        return uv1, uv2, r3_coor, conv

    # @staticmethod
    # def _test_intesection_tolerance(surf1,surf2,sum_idx,uvt,w,abs_tol):
    #
    #     if sum_idx == 0:
    #         surf2_pos = surf2.eval(w, uvt[2, 0])  # may be faster, patches indices are known
    #         uv2 = np.array([w, uvt[2, 0]])
    #     if sum_idx == 1:
    #         surf2_pos = surf2.eval(uvt[2, 0], w)
    #         uv2 = np.array(uvt[2, 0], w)
    #     r3_coor = surf1.eval(uvt[0, 0], uvt[1, 0])
    #     dist = la.norm(r3_coor - surf2_pos)
    #     if dist <= abs_tol:
    #         uv1 = np.array(uvt[0, 0], uvt[1, 0])
    #         conv = 1
    #     else:
    #         uv1 = np.array([0, 0])
    #         conv = 0
    #
    #     return uv1, uv2, r3_coor, conv

    # def _rangetest(self,uvt,ui, vi, ti, tol):
    # # test if paramaters does not lie outside current patch, otherwise they are returned to the boundary
    #     test = 0
    #
    #     du = np.array([uvt[0,0] - ui[0], ui[1] - uvt[0,0]])
    #     dv = np.array([uvt[1,0] - vi[0], vi[1] - uvt[1,0]])
    #     dt = np.array([uvt[2,0] - ti[0], ti[1] - uvt[2,0]])
    #
    #     for i in range(0,2):
    #         if (du[i] < -tol):
    #             uvt[0,0] = ui[i]
    #
    #     for i in range(0,2):
    #         if (dv[i] < -tol):
    #             uvt[1,0] = vi[i]
    #
    #     for i in range(0,2):
    #         if (dt[i] < -tol):
    #             uvt[2,0] = ti[i]
    #
    #     if np.logical_and(uvt[0,0] >= ui[0],uvt[0,0] <= ui[1]):
    #         if np.logical_and(uvt[1,0] >= vi[0], uvt[1,0] <= vi[1]):
    #             if np.logical_and(uvt[2,0] >= ti[0], uvt[2,0] <= ti[1]):
    #                 test = 1
    #
    #     return test, uvt

    '''
    Calculation and representation of intersection of two B-spline surfaces.
    
    Result is set of B-spline segments approximating the intersection branches.
    Every segment consists of: a 3D curve and two 2D curves in parametric UV space of the surfaces.
    '''
    