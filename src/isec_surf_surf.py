import sys
import os

src_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(src_dir, "../external/bih/build")
sys.path += [build_path]
print(sys.path)

import bih
import numpy as np
import numpy.linalg as la

import bspline as bs
import isec_point as IP
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

        return point_list1, point_list2




    def _connect_points(self,point_list1,point_list2):

        patch_point_list1, boundary_points1 = self.make_patch_point_list(point_list1)
        patch_point_list2, boundary_points2 = self.make_patch_point_list(point_list2)

        #print(boundary_points2)

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



        line = self._make_orderings(point_list,patch_point_list, boundary_points)

        return line


    def _make_orderings(self,point_list,patch_point_list, boundary_points):
        """
        TODO: split into smaller functions.
        :param point_list:
        :param patch_point_list:
        :param boundary_points:
        :return:
        """
        n_curves = 0

        line = []

        n_intersections = patch_point_list[0].__len__() + patch_point_list[1].__len__()

        # enf found  = n_curves = n+1

        while n_intersections != 0:

            # obtain start point
            for i in range(0, 2):
                for bpatchpoint in boundary_points[i]:
                    #print(bpatchpoint)
                    #print(type(bpatchpoint[0]))
                    if(bpatchpoint.connected() == 0):
                        bpatchpoint.connect()
                        n_intersections -= 1
                        line[n_curves].append(bpatchpoint)
                        i_surf = i
                        surf = bpatchpoint.surf1
                        in_point = bpatchpoint
                        patch_id = self._patch_pos2id(surf, bpatchpoint.iu1[0], bpatchpoint.iv1[0])

            patch_pos = self._patch_id2pos(surf, patch_id)
            n_points = 0
            for pp in patch_point_list[i_surf][patch_pos]:
                print(pp)
                n_points += 1
                if pp.connected == 0:
                    out_point = pp

            if n_points > 2:
                print('problem')

            if n_points == 2: # test of the points between
                if np.logical_and(in_point.iu2[0] == out_point.iu2[0], in_point.iv2[0] == out_point.iv2[0]):
                    points_between = 1
                else:
                    points_between = 0

            if np.logical_or(points_between == 1, n_points == 1):
                patch_id = self._patch_pos2id(in_point.surf2, in_point.iu2[0], in_point.iv2[0])
                i_point = patch_point_list[1-i_surf][patch_id]


            for i in range(0,2):
                for patchpoint in patch_point_list[i]:
                    if np.logical_and(patchpoint.connected == 0, patchpoint.boundary_flag == 1 ):
                        patchpoint.connect()
                        n_intersections -= 1
                        line[n_curves].append(patchpoint)




                        while end_found == 0:
                            #id  = self._patch_pos2id(patchpoint.surf2,patchpoint.iu2[k],patchpoint.iv2[k])
                            pt = np.zeros([patchpoint.iu1.___len___])
                            for k in range(patchpoint.iu1.___len___):
                                ids = self._patch_pos2id(patchpoint.surf1,patchpoint.iu1[k],patchpoint.iv1[k])
                                for pp in patch_point_list[1-i][ids]:
                                    if pp.connected == 0:
                                        pt[k] += 1
                            if np.sum(pt) > 1:
                                print('problem')
                            #else:
#
#            #found_start
#            end_found = 0
#            while end_found == 0:
#                #search next
#                check_duplicities
#
#
#
#
#            n_curves += 1



        return line




    #@staticmethod
    def make_patch_point_list(self, point_list):

        surf1 = point_list[0].surf1

        list_len = surf1.u_basis.n_intervals * surf1.v_basis.n_intervals
        patch_points = []
        boundary_points = []

        #boundary_points.append([[], []])

        for i in range(list_len):
            patch_points.append([])

        idp = -1
        for point in point_list:
            idp += 1
            for i_patch in range(point.iu1.__len__()):
                #print(point.iu1)
                #print(point.iv1)
                #print('done')
                id = self._patch_pos2id(surf1, point.iu1[i_patch], point.iv1[i_patch])
                patch_points[id].append(idp)
            if point.boundary_flag == 1:
                boundary_points.append(idp)


        return patch_points, boundary_points

    @staticmethod
    def _main_threads(surf,sum_idx):
        """
        Construction of the main threads
        Todo: what is "thread", describe.
        Todo: use eunum to mark u/v direction instead of 'sum_idx'
        :param surf: surface which is used to construction of the main threads
        :param sum_idx: sum_idx == 0 --> u fixed, sum_idx == 1 --> v fixed
        :return: curves as list of curves, w_val as list of value of the fixed local coordinates , patches as list of neighbour patches
        """

        poles = surf.poles

        if sum_idx == 0:
            fix_basis = surf.u_basis
            curv_basis = surf.v_basis
        elif sum_idx == 1:
            fix_basis = surf.v_basis
            curv_basis = surf.u_basis

        curves = []
        w_val = []
        patches = []

        if sum_idx == 0:
            curv_pol = poles[0, :, :]
        elif sum_idx == 1:
            curv_pol = poles[:, 0, :]

        w_val.append(0.0)
        patches.append([0])

        curv = bs.Curve(curv_basis, curv_pol)
        curves.append(curv)

        for iw in range(1,fix_basis.n_intervals):
            w1f = fix_basis.eval_vector(iw, fix_basis.knots[iw + 2])
            if sum_idx == 0:
                surf_pol = poles[iw:iw + 3, :, :]
                curv_pol = np.tensordot(w1f, surf_pol, axes=([0], [sum_idx]))
            elif sum_idx == 1:
                surf_pol = poles[:,iw:iw + 3, :]
                curv_pol = np.tensordot(w1f, surf_pol, axes=([0], [sum_idx]))
            w_val.append(fix_basis.knots[iw + 2])
            patches.append([iw-1, iw])

            curv = bs.Curve(curv_basis, curv_pol)
            curves.append(curv)

        if sum_idx == 0:
            curv_pol = poles[poles.shape[sum_idx]-1,:, :]
            #print(poles.shape[sum_idx]-1,fix_basis.n_intervals - 1)
        elif sum_idx == 1:
            curv_pol = poles[:,poles.shape[sum_idx]-1 , :]  # fix_basis.n_intervals - 1
            #print(poles.shape[sum_idx] - 1,fix_basis.n_intervals - 1)

        w_val.append(1.0)
        patches.append([fix_basis.n_intervals-1])

        curv = bs.Curve(curv_basis, curv_pol)
        curves.append(curv)

        return curves, w_val, patches

    @staticmethod
    def _patch_pos2id(surf,iu,iv):

        id = iu * surf.v_basis.n_intervals + iv
        return id

    @staticmethod
    def _patch_id2pos(surf,id):

        iu = int(np.floor(id / surf.v_basis.n_intervals))
        iv = int(id - (iu * surf.v_basis.n_intervals))

        return iu, iv

    def get_intersections(self,surf1,surf2,tree2):
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
        #print([surf1.u_basis.n_intervals+1,surf1.v_basis.n_intervals+1])

        for sum_idx in range(2): # sum_idx = 0 ==> fixed u, sum_idx = 1 ==> fixed v
            curves, w_val, patches = self._main_threads(surf1, sum_idx)
            curve_id = -1
            for curve in curves:
                curve_id += 1
                interval_id = -1
                for it in range(curve.basis.n_intervals):
                    interval_id += 1
                    if self._already_found(crossing,interval_id,curve_id,sum_idx) == 1:
                        print('continue')
                        #print(crossing)
                        print(point.R3_coor)
                        continue
                    curv_surf_isec = ICS.IsecCurveSurf(surf2, curve)
                    boxes = bih.AABB(curve.poles[it:it+3, :].tolist())
                    uv1 = np.zeros([2,1])
                    #if np.logical_or(it == 0, it == curve.basis.n_intervals - 1):

                    # todo: len(patches[curve_id])
                    # todo: (n+1)*[it]
                    if (patches[curve_id].__len__() == 1):
                        interval_list = [it]
                    else:
                        interval_list = [it, it]
                    uv1[sum_idx,0] = w_val[curve_id]
                    intersectioned_patches2 = tree2.find_box(boxes)
                    #print("curve_id=", curve_id)
                    #print("sum_idx=", sum_idx)
                    #print("intersectioned_patches2=",intersectioned_patches2)
                    for ipatch2 in intersectioned_patches2:
                        iu2, iv2 = self._patch_id2pos(surf2, ipatch2)
                        uvt = curv_surf_isec.get_initial_condition(iu2, iv2, it)
                        uvt,  conv, flag, XYZ = curv_surf_isec.get_intersection(uvt, iu2, iv2, it, self.max_it,
                                                                            self.rel_tol, self.abs_tol)
                        if conv == 1:
                            #check second surf
                            #else break
                            #if np.logical_or(np.logical_and(flag[2] == 0, it == 0),
                            #                 np.logical_and(flag[2] == 1, it == curve.basis.n_intervals - 1)):
                            if np.logical_or(curve_id == 0, curve_id == curves.__len__() - 1):
                                boundary_flag = 1
                            else:
                                boundary_flag = 0

                            uv1[1 - sum_idx, 0] = uvt[2, 0]
                            if sum_idx == 0:
                                point = IP.IsecPoint(surf1, patches[curve_id], interval_list, uv1, boundary_flag, flag,
                                                  sum_idx, surf2, [iu2], [iv2], uvt[0:2, :], XYZ)
                            elif sum_idx == 1:
                                point = IP.IsecPoint(surf1, interval_list, patches[curve_id], uv1, boundary_flag, flag,
                                                  sum_idx, surf2, [iu2], [iv2], uvt[0:2, :], XYZ)

                            point_list.append(point)

                            #if boundary_flag == 0:
                            #    point_list[- 1].add_patches(sum_idx, flag[2]) # move to constructor
                            #print('flag')
                            #print(flag.shape)
                            #print(flag[2])
                            if np.logical_or(flag[2] == 0,flag[2] == 1):
                                if sum_idx == 0:
                                    crossing[curve_id, interval_id + flag[2]] = 1
                                elif sum_idx == 1:
                                    crossing[interval_id + flag[2],curve_id] = 1 # xxx
                                break



        return point_list

    @staticmethod
    def _already_found(crossing,interval_id,curve_id,sum_idx):

        found = 0


        if sum_idx == 1:
            for i in range(2):
                if crossing[interval_id + i, curve_id] == 1:
                    found = 1
        elif sum_idx == 0:
            for i in range(2):
                if crossing[curve_id, interval_id + i] == 1:
                    found = 1

        return found



# np.logical_not(np.logical_and(crossing[interval_id+flag,curve_id] == 1,sum_idx == 1))
    '''
    Calculation and representation of intersection of two B-spline surfaces.

    Result is set of B-spline segments approximating the intersection branches.
    Every segment consists of: a 3D curve and two 2D curves in parametric UV space of the surfaces.
    '''
