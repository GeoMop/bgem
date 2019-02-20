import sys
import os

src_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(src_dir, "../external/bih/build")
sys.path += [build_path]
print(sys.path)

import bih
import numpy as np

import bspline as bs
import isec_point as IP
import surface_point as SP
import isec_curve_surf as ICS




class IsecSurfSurf:
    def __init__(self, surf1, surf2, nt=2, max_it=10, rel_tol = 1e-16, abs_tol = 1e-14):
        self.surf1 = surf1
        self.surf2 = surf2
        self.box1, self.tree1 = self.bounding_boxes(self.surf1)
        self.box2, self.tree2 = self.bounding_boxes(self.surf2)
        self.nt = nt
        self.max_it = max_it
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

        self._ipoint_list = []  # append
        # tolerance

    def bounding_boxes(self, surf):
        """
        Compute bounding boxes and construct BIH tree for a given surface
        :param surf:
        :return:
        """
        tree = bih.BIH()
        n_patch = (surf.u_basis.n_intervals) * (surf.v_basis.n_intervals)

        patch_poles = np.zeros([9, 3, n_patch])
        i_patch = 0
        for iu in range(surf.u_basis.n_intervals):
            for iv in range(surf.v_basis.n_intervals):
                n_points = 0
                for i in range(0, 3):
                    for j in range(0, 3):
                        patch_poles[n_points, :, i_patch] = surf.poles[iu + i, iv + j, :]
                        n_points += 1
                #assert i_patch == (iu * surf.v_basis.n_intervals + iv)
                assert i_patch == self._patch_pos2id(surf,iu,iv)
                i_patch += 1

        boxes = [bih.AABB(patch_poles[:, :, p].tolist()) for p in range(n_patch)]
        # print(patch_poles[:, :, 0])
        tree.add_boxes(boxes)
        tree.construct()
        # print(boxes)
        # for b in boxes:
        #    print(b.min()[0:2],b.max()[0:2])
        return boxes, tree

    def get_intersection(self):
        """
        Main method to get intersection points
        :return:
        """
        point_list1 = self.get_intersections(self.surf1, self.surf2, self.tree2)  # patches of surf 2 with respect threads of the surface 1
        point_list2 = self.get_intersections(self.surf2, self.surf1, self.tree1) # patches of surf 1 with respect threads of the surface 2

        #print('points')
        #for point in point_list1:
        #    print(point.uv1)
        #print('points2')
        #for point in point_list2:
        #    print(point.uv2)
        #print(point_list1.__len__(),point_list2.__len__())

        #assert point_list1.__len__() == 9
        #assert point_list2.__len__() == 9

        connected_points = self._connect_points(point_list1, point_list2)
        print(len(point_list1))
        print(len(point_list2))

        return point_list1, point_list2


    @staticmethod
    def _patch_pos2id(surf, iu, iv):

        id = iu * surf.v_basis.n_intervals + iv
        return id

    @staticmethod
    def _patch_pos2id2(surf_point, k):

        iu = surf_point.iuv[k][0]
        iv = surf_point.iuv[k][1]
        v_int = surf_point.surf.v_basis.n_intervals

        id = iu * v_int + iv
        return id



    @staticmethod
    def _patch_id2pos(surf, patch_id):

        iu = int(np.floor(patch_id / surf.v_basis.n_intervals))
        iv = int(patch_id - (iu * surf.v_basis.n_intervals))

        return iu, iv


    @staticmethod
    def _main_curves(surf, axis):
        """
        Construction of the main threads
        Todo: what is "thread", describe.
        :param surf: surface which is used to construction of the main threads
        :param axis: sum_idx == 0 --> u fixed, sum_idx == 1 --> v fixed
        :return: curves as list of curves, w_val as list of value of the fixed local coordinates , patches as list of neighbour patches
        """

        poles = surf.poles

        if axis == IP.Axis.u:
            fix_basis = surf.u_basis
            curv_basis = surf.v_basis
        elif axis == IP.Axis.v:
            fix_basis = surf.v_basis
            curv_basis = surf.u_basis

        curves = []
        w_val = []
        patch = []

        patch.append(0)
        for iw in range(0, fix_basis.n_intervals):
            patch.append(iw)

        for iw in range(0, fix_basis.n_intervals+1):
            w1f = fix_basis.eval_vector(patch[iw], fix_basis.knots[iw + 2])
            #ind = [slice(0, surf.u_basis.size), slice(0, surf.v_basis.size), slice(0, 3)]
            # slice(None) !!!!!  np.s_[]
            ind = [slice(None), slice(None), slice(None)]
            ind[axis] = slice(patch[iw], patch[iw] + 3)
            surf_pol = poles[tuple(ind)]
            curv_pol = np.tensordot(w1f, surf_pol, axes=([0], [axis]))
            w_val.append(fix_basis.knots[iw + 2])
            curv = bs.Curve(curv_basis, curv_pol)
            curves.append(curv)

        return curves, w_val, patch

    def get_intersections(self, surf1, surf2, tree2):
        """
        Tries to compute intersection of the main curves from surface1 and patches of the surface2 which have a
         non-empty intersection of corresponding bonding boxes
        :param surf1: Surface used to construction of the main threads
        :param surf2: Intersected surface
        :param tree2: Bih tree of the patches of the surface 2
        :return: point_list as list of points of intersection
        """

        point_list = []
        crossing = np.zeros([surf1.u_basis.n_intervals + 1, surf1.v_basis.n_intervals + 1])

        for axis in [IP.Axis.u, IP.Axis.v]:
            curves, w_val, patch = self._main_curves(surf1, axis)
            curve_id = -1
            for curve in curves:
                curve_id += 1
                interval_id = -1
                interval_intersections = 0
                for it in range(curve.basis.n_intervals):
                    interval_id += 1
                    if self._already_found(crossing, interval_id, curve_id, axis) == 1: #?
                        print('continue')
                        continue
                    curv_surf_isec = ICS.IsecCurveSurf(surf2, curve)
                    boxes = bih.AABB(curve.poles[it:it+3, :].tolist())
                    intersectioned_patches2 = tree2.find_box(boxes)
                    for ipatch2 in intersectioned_patches2:
                        iu2, iv2 = self._patch_id2pos(surf2, ipatch2)
                        uvt,  conv, xyz = curv_surf_isec.get_intersection(iu2, iv2, it, self.max_it,
                                                                            self.rel_tol, self.abs_tol)
                        if conv == 1:
                            # Point A
                            uv_a = np.zeros([2])
                            uv_a[axis] = w_val[curve_id]
                            uv_a[1 - axis] = uvt[2]
                            iuv_a = np.zeros([2], dtype=int)
                            iuv_a[axis] = patch[curve_id]
                            iuv_a[1 - axis] = it
                            surf_point_a = SP.SurfacePoint(surf1, iuv_a, uv_a)

                            # Point B
                            uv_b = uvt[0:2]
                            iuv_b = np.array([iu2, iv2])
                            surf_point_b = SP.SurfacePoint(surf2, iuv_b, uv_b)

                            point = IP.IsecPoint(surf_point_a, surf_point_b, xyz)

                            if interval_intersections == 0:
                                point_list.append(point)
                                interval_intersections += 1
                            else:
                                a = 1
                                #check


                            direction = surf_point_a.interface_flag[1-axis]
                            if direction != 0:
                                ind = [curve_id, curve_id]
                                ind[1-axis] = interval_id + int(0.5 * (direction + 1))
                                crossing[tuple(ind)] = 1
                            #break  # we consider only one point as an intersection of segment of a curve and a patch
                            #check duplicities

        return point_list



    @staticmethod
    def _already_found(crossing, interval_id, curve_id, axis):

        found = 0
        ind1 = [curve_id, curve_id]
        ind2 = [curve_id, curve_id]
        ind1[1-axis] = interval_id
        ind2[1-axis] = interval_id + 1

        if np.logical_or(crossing[tuple(ind1)] == 1, crossing[tuple(ind2)] == 1):
            found = 1

        return found

    # Connecting of the points

    def _connect_points(self, point_list1, point_list2):

        patch_point_list1, boundary_points1 = self.make_patch_point_list(point_list1)
        patch_point_list2, boundary_points2 = self.make_patch_point_list(point_list2)

        # print(boundary_points2)

        patch_point_list = []
        boundary_points = []
        point_list = []

        #  boundary_points[0].append(1)
        #  print(boundary_points )
        #  print(boundary_points[0] )
        #  print(boundary_points[1] )
        #  print('done')

        point_list.append(point_list1)
        point_list.append(point_list2)
        patch_point_list.append(patch_point_list1)
        patch_point_list.append(patch_point_list2)
        boundary_points.append(boundary_points1)
        boundary_points.append(boundary_points2)

        # line = self._make_orderings(point_list,patch_point_list, boundary_points)
        line = []

        return line

    def make_patch_point_list(self, point_list):

        surf = point_list[0].surface_point_a.surf

        list_len = surf.u_basis.n_intervals * surf.v_basis.n_intervals
        patch_points = []
        boundary_points = []

        for i in range(list_len):
            patch_points.append([])

        idp = -1
        for point in point_list:
            idp += 1
            for i_patch in range(point.surface_point_a.iuv.__len__()):
                id = self._patch_pos2id2(point.surface_point_a, i_patch)
                patch_points[id].append(idp)
            if point.surface_point_a.surface_boundary_flag == 1:
                boundary_points.append(idp)

        return patch_points, boundary_points

    def _get_start_point(self, boundary_points, point_list):
        """
        Returns first unconnected boundary point from the list
        :param boundary_points: list of the id's of the boundary points
        :param point_list: list of the intersection points
        :return: intersection point which lies on the boundary of the surface & id of the point_list
        """
        # obtain start point of the curve
        for i in range(0, 2):
            for id_point in boundary_points[i]:
                point = point_list[i][id_point]
                if point.connected == 0:
                    i_surf = i
                    return point, i_surf

    def _get_patch_id(self, point):
        """
        :param point: as isec_point
        :return: neighbouring patches id's
        """
        patch_id = np.zeros([len(point.surface_point_a.iuv)], dtype=int)
        for k in range(0, len(point.surface_point_a.iuv)):
            patch_id[k] = self._patch_pos2id2(point.surface_point_a, k)

        return patch_id

    def _get_out_point(self, point_list, patch_point_list, i_surf, patch_id):

        # patch_pos = self._patch_pos2id(surf, patch_id)
        next_point = []
        print(type(patch_id))
        for i in range(0, patch_id.size):
            for id_point in patch_point_list[i_surf][patch_id]:
                print(id_point)
                print(point_list[i_surf][id_point].connected)
                print('xx')
                point = point_list[i_surf][id_point]
                if point.connected == 0:
                    next_point.append(point)

        return next_point

    def _get_out_point2(self, point_list, patch_point_list, i_surf, patch_id):
        """

        :param point_list: list of all points
        :param patch_point_list:
        :param i_surf: index of the surface "0" or "1"
        :param patch_id:
        :return:
        """

        next_point = []
        for p_id in patch_id:
            for id_point in patch_point_list[i_surf][p_id]:
                point = point_list[i_surf][id_point]
                patch_id2 = self._get_patch_id(point)
                for p2_id in patch_id2:
                    if p2_id == p_id:
                        if point.connected == 0:
                            next_point.append(point)
        return next_point

    def _make_orderings(self, point_list, patch_point_list, boundary_points):
        """
        TODO: split into smaller functions.
        :param point_list:
        :param patch_point_list:
        :param boundary_points:
        :return:
        """
        n_curves = 0

        line = [[]]
        line_surf_info = [[]]

        #n_unconnected_points = len(patch_point_list[0]) + len(patch_point_list[1])
        # enf found  = n_curves = n+1
        # make_open_curves
        #while n_unconnected_points != 0:



        # obtain start point
        point, i_surf = self._get_start_point(boundary_points, point_list)
        point.connected = 1
        line[n_curves].append(point)
        line_surf_info[n_curves].append(i_surf)

        #n_unconnected_points -= 1
        end_found = 0
        # in_patch_id = self._get_patch_id(line[n_curves][-1])

        # if in_patch_id.size > 2:   # should be strange for surface boundary point
        #   print('problem')

        while end_found == 0:

            # get ID's of all corresponding patches
            patch_id = self._get_patch_id(line[n_curves][-1])

            next_point = self._get_out_point2(point_list, patch_point_list, i_surf, patch_id)

            # for i in range(0, patch_id.size):
            #    next_point = self._get_out_point(point_list, patch_point_list, i_surf, patch_id[i])

            n_points = len(next_point)

            if n_points == 1:
                next_point = next_point[0]
            else:
                print('problem')  # should be solved

            if n_points == 1:  # full cut (two sides cut, no points between)
                in_point = line[n_curves][-1]

                #points_between = 0
                line[n_curves].append(next_point)
                line_surf_info[n_curves].append(i_surf)
                if next_point.surface_boundary_flag == 1:
                    end_found = 1
                ##
                patch_id = self._patch_pos2id2(in_point.surface_point_b, 0)
                inter_points = self._get_out_point(point_list, patch_point_list, 1 - i_surf, patch_id)
                if len(inter_points) == 0:
                    continue
                dist = self._compute_distances(in_point, next_point, inter_points)



        return line

    # @staticmethod


    '''
    Calculation and representation of intersection of two B-spline surfaces.

    Result is set of B-spline segments approximating the intersection branches.
    Every segment consists of: a 3D curve and two 2D curves in parametric UV space of the surfaces.
    '''
