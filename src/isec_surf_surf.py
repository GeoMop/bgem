import sys
build_path="/home/jiri/Soft/Geomop/Intersections/external/bih/build"
sys.path+=[build_path]
print(sys.path)

import bih
import numpy as np
import numpy.linalg as la





class IsecSurfSurf:

    def __init__(self, surf1, surf2, nt=2,nit=10):
        self.surf1 = surf1
        self.surf2 = surf2
        self.tree1 = self.bounding_boxes(self.surf1)
        self.tree2 = self.bounding_boxes(self.surf2)
        self.nt = nt
        self.nit = nit

    @staticmethod
    def bounding_boxes(surf):
        tree = bih.BIH()
        n_patch = (surf.u_basis.n_intervals)*(surf.v_basis.n_intervals)


        patch_poles = np.zeros([9, 3, n_patch])
        i_patch = 0
        for k in range(surf.u_basis.n_intervals):
            for l in range(surf.v_basis.n_intervals):
                n_points = 0
                for i in range(0,3):
                    for j in range(0,3):
                        patch_poles[n_points,:,i_patch] = surf.poles[k+i, l+j, :]
                        n_points += 1
                i_patch += 1
        boxes = [bih.AABB(patch_poles[:,:,p].tolist()) for p in range(n_patch)]
        #print(patch_poles[:, :, 0])
        tree.add_boxes( boxes )
        tree.construct()
        #print(boxes)
        #for b in boxes:
        #    print(b.min()[0:2],b.max()[0:2])
        return boxes, tree



    @staticmethod
    def get_intersection(surf1,surf2,tree1,tree2,box1,box2,nt,nit):
    # nt - number of threads (integer)
            #X,Xs, u_n_intervs,v_n_intervs,u_knots,v_knots,
     #us_n_intervs,vs_n_intervs,us_knots,vs_knots,isec,n_isec, nit,nt )
        # computes intersection of BSpline patch with BSpline thread





        def _patch_patch_intersection( surf1, surf2,uvt, ui,vi,u2i,v2i, nit ,m,o,k,l):
            #returns coordinetes of the intersection of the patches (if exist), one
            #parameter must be fixed
            pt = np.zeros([3,1])
            conv =0
            tol = 1e-6; # in x,y,z
            tol2 = 1e-4; # in u,v
            # nt = self.nt

            Xs = surf2.poles[m:m + 3, o:o + 3, 0]
            X = surf1.poles[k:k + 3, l:l + 3, 0]

            if u2i.shape[0] == 1:
                u2f = surf2.u_basis.eval_vector(m,u2i) # eval_vector(self, i_base, t):

            if v2i.shape[0] == 1:
                v2f = surf2.v_basis.eval_vector(o,v2i)


            for i in range(nit):
                uf = surf1.u_basis.eval_vector(k,uvt[0])
                vf = surf1.v_basis.eval_vector(l,uvt[1])
                ufd = surf1.u_basis.eval_diff_vector(k,uvt[0])
                vfd = surf1.v_basis.eval_diff_vector(l,uvt[1])

            if u2i.shape[0] == 1:
                v2f = surf2.v_basis.eval_vector(o,uvt[2])
                v2fd = surf2.v_basis.eval_diff_vector(o,uvt[2])
                dXYZp2 = la.multi_dot([u2f,Xs,v2fd])
                #(kron(v2fd',u2f')*Xs)';

            if v2i.shape[0] == 1:
                u2f = surf2.u_basis.eval_vector(m,uvt[2])  #splinebasevec(us_knots,uvt(3),0,m)
                u2fd = surf2.u_basis.eval_diff_vector(m,uvt[2]) #splinebasevec(us_knots,uvt(3),1,m)
                dXYZp2 = la.multi_dot([u2fd, Xs, v2f])



            dXYZu1 = la.multi_dot([ufd, X, vf])   #(kron(vf',ufd')*X)'
            dXYZv1 = la.multi_dot([uf,X,vfd])# (kron(vfd',uf')*X)'
            J = np.array([[dXYZu1.transpose()], [dXYZv1.transpose()], [-dXYZp2.transpose()]])

            print(dXYZu1)
            print(dXYZv1)
            print(-dXYZp2)


            print(J)
            return
            deltaXYZ = la.multi_dot([uf,X,vf])  - la.multi_dot([u2f,Xs,vf])  #  kron(vf',uf') * X)' - (kron(v2f',u2f') * Xs)'
            uvt = uvt - la.solve(J,deltaXYZ)   #uvt = uvt- J\deltaXYZ
            test,uvt = _rangetest(uvt,ui,vi,u2i,v2i,0.0)

            test,uvt = _rangetest(uvt,ui,vi,u2i,v2i,tol2);
            if test == 1:
                if np.length(u2i) == 1:
                    surf2_pos = surf2.eval(u2i,uvt[2])
                if np.length(v2i) == 1:
                    surf2_pos = surf2.eval(uvt[2],v2i)

            dist = la.norm(surf1.eval(uvt[0],uvt[1]) - surf2_pos) # may be faster, indices of patches are known



            if test == 1:
                if dist <= tol:
                    pt = la.multi_dot(uf,X,vf) #kron(vf',uf')*X
                    conv =1
            else:
                uvt = np.zeros(3,1)

            return uvt, pt, conv

        def _rangetest(uvt, ui, vi, u2i, v2i, tol):
        # test if paramaters does not lie outside current patch, otherwise they are returned to the boundary

            test = 0

            du = np.array([uvt[0] - ui[0], ui[1] - uvt[0]])
            dv = np.array([uvt[1] - vi[0], vi[1] - uvt[1]])

            if np.length(v2i) == 1:
                d2p = [uvt[2] - u2i[0], u2i[1] - uvt[2]]
                pi = u2i

            if np.length(u2i) == 1:
                d2p = np.array([uvt[2] - v2i[0], v2i[1] - uvt[2]])
                pi = v2i

            for i in range(0,2):
                if (du[i] < -tol):
                    uvt[0] = ui[i]

            for i in range(0,2):
                if (dv[i] < -tol):
                    uvt[1] = vi[i]

            for i in range(0,2):
                if (d2p[i] < -tol):
                    uvt[2] = pi[i]


            if np.logical_and(uvt[0] >= ui[0],uvt[0] <= ui[1]):
                if np.logical_and(uvt[1] >= vi[0], uvt[1] <= vi[1]):
                    if np.logical_and(uvt[2] >= pi[0], uvt[2] <= pi[1]):
                        test = 1

            return uvt, test



        def _compute_intersection(surf1,surf2,tree1,tree2,box1,box2,nt,nit ):
            n_points = 0
            point = np.zeros([surf1.u_basis.n_intervals * surf1.v_basis.n_intervals, 11])
            ninter =  np.zeros([surf1.u_basis.n_intervals,surf1.v_basis.n_intervals])
            for k in range(surf1.u_basis.n_intervals):
                us1 = surf1.u_basis.knots[k + 1]
                ue1 = surf1.u_basis.knots[k + 2]
                u1_c =(us1 + ue1)/2
                ui = np.array([us1, ue1])
                for l in range(surf1.v_basis.n_intervals):
                    vs1 = surf1.v_basis.knots[l+1]
                    ve1 = surf1.v_basis.knots[l+2]
                    v1_c = (vs1  + ve1)/2
                    vi = np.array([vs1, ve1])
                    s=0
                    box_id = l + surf1.u_basis.n_intervals * k
                    n_isec = tree1.find_box(box2[box_id])

                    #if n_isec.shape[1] > 0:
                    for p in n_isec:
                        m = int(np.ceil(p / surf2.u_basis.n_intervals))
                        o = int(p - (m-1)*surf2.u_basis.n_intervals)
                        sp_i = (m-1)*(surf2.u_basis.n_intervals) + o
                        # v2 fixed
                        u2i = np.array([surf2.u_basis.knots[m+1], surf2.u_basis.knots[m+2]])
                        v_knot = np.linspace(surf2.v_basis.knots[o+1],surf2.v_basis.knots[o+2],nt)
                        for h  in range(nt):  #=1:length(v_knot)
                            u2_c = (u2i[0] + u2i[1])/2
                            v2i = np.array([v_knot[h]])
                            uvu2v2 = np.array([u1_c, v1_c, u2_c]) # initial condition
                            uvu2v2,ptl,conv  = _patch_patch_intersection( surf1, surf2,uvu2v2, ui,vi,u2i,v2i, nit ,m,o,k,l)

                            if np.not_equal(conv, 0):
                                s = s+1
                                n_points = n_points +1
                                point[n_points,:] = [uvu2v2,v_knot[h],ptl(1),ptl(2),ptl(3),k,l,m,o]

                        # u2 fixed
                        #v2i = [vs_knots(o+2) vs_knots(o+3)]
                        #u_knot = linspace(us_knots(m+2),us_knots(m+3),nt)
                        #for h =1:length(u_knot)
                        #    v2_c = (vs_knots(o+2) + vs_knots(o+3))/2
                        #    u2i = u_knot(h)
                        #    uvu2v2 = [u1_c,v1_c,v2_c]; % initial condition
                        #    [ uvu2v2,ptl,conv ] = patch_patch_intersection( uvu2v2, ui,vi,u2i,v2i, u_knots, v_knots, X,us_knots, vs_knots, Xs, nit ,m,o,k,l)
                        #    if conv ~= 0
                        #        s = s+1
                        #        n_points = n_points +1
                        #        point(n_points,:) = [uvu2v2(1:2)',u_knot(h),uvu2v2(3),ptl(1),ptl(2),ptl(3),k,l,m,o]

                    ninter[k,l] = s

            return point, n_points, ninter

        point, n_points, ninter = _compute_intersection(surf1, surf2, tree1, tree2, box1, box2, nt,nit)


#    def get_intersections(tree1,tree2):
#$        for k in range( self.surf.u_basis.n_intervals - 2):
 #           for l in range(self.surf.v_basis.n_intervals - 2):
 #           self.tree1.






    # def compute_intersection(self.tree1,self.tree2):
    #
    #
    #     intersect_box_ids = tree.find_point([0.7, 0.5, 0.5])
    #     intersect_box_ids = tree.find_box(box)


    '''
    Calculation and representation of intersection of two B-spline surfaces.
    
    Result is set of B-spline segments approximating the intersection branches.
    Every segment consists of: a 3D curve and two 2D curves in parametric UV space of the surfaces.
    '''
    