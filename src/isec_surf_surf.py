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
        self.box1, self.tree1 = self.bounding_boxes(self.surf1)
        self.box2, self.tree2 = self.bounding_boxes(self.surf2)
        self.nt = nt
        self.nit = nit
        #tolerance

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
                assert i_patch == (k * surf.v_basis.n_intervals + l)
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


        point, n_points, ninter = self._compute_intersection(self.surf1,self.surf2,self.tree1,self.tree2,self.box1,self.box2)
        # surf1, surf2, tree1, tree2, box1, box2, nt, nit

    def _compute_intersection(self,surf1,surf2,tree1,tree2,box1,box2):

        nt = self.nt
        nit = self.nit

        #surf1, surf2, tree1, tree2, box1, box2, nt, nit
        n_points = 0
        point = np.zeros([surf1.u_basis.n_intervals * surf1.v_basis.n_intervals, 11])
        ninter =  np.zeros([surf1.u_basis.n_intervals,surf1.v_basis.n_intervals])
        for u_index in range(surf1.u_basis.n_intervals):
            us1 = surf1.u_basis.knots[u_index + 2]
            ue1 = surf1.u_basis.knots[u_index + 3]
            u1_center =(us1 + ue1)/2
            ui = np.zeros([2,1])
            print(ui.shape)
            ui[0:1,0] = ([us1, ue1])
            #ui = np.array([us1, ue1] )
            for v_index in range(surf1.v_basis.n_intervals):
                vs1 = surf1.v_basis.knots[v_index+2]
                ve1 = surf1.v_basis.knots[v_index+3]
                v1_center = (vs1  + ve1)/2
                vi = np.array([vs1, ve1])
                s=0
                box_id = v_index + surf1.u_basis.n_intervals * u_index
                intersectioned_boxes1 = tree1.find_box(box2[box_id])

                #if n_isec.shape[1] > 0:
                for i_boxes1 in intersectioned_boxes1:
                    u2_index = int(np.floor(i_boxes1 / surf2.v_basis.n_intervals))
                    v2_index = int(i_boxes1 - (u2_index * surf2.v_basis.n_intervals))
                    sp_i = u2_index * surf2.v_basis.n_intervals + v2_index
                    # v2 fixed
                    u2i = np.array([surf2.u_basis.knots[u2_index+2], surf2.u_basis.knots[u2_index+3]])
                    v2_thread = np.linspace(surf2.v_basis.knots[v2_index+2],surf2.v_basis.knots[v2_index+3],nt)
                    for v2_val  in v2_thread:  #=1:length(v_knot)
                        u2_center = (u2i[0] + u2i[1])/2
                        v2i = np.array([v2_val])
                        uvu2v2 = np.array([u1_center, v1_center, u2_center])#initial condition
                        uvu2v2,ptl,conv  = self._patch_patch_intersection( surf1, surf2,uvu2v2, ui,vi,u2i,v2i, nit ,u2_index,v2_index,u_index,v_index)

                        if np.not_equal(conv, 0):
                            s = s+1
                            n_points = n_points +1
                            point[n_points,:] = [uvu2v2,v2_val,ptl(1),ptl(2),ptl(3),u2_index,v2_index,u_index,v_index]

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

                ninter[u_index,v_index] = s

        return point, n_points, ninter

    def _patch_patch_intersection( self,surf1, surf2,uvt, ui,vi,u2i,v2i, nit ,m,o,k,l):
        #returns coordinetes of the intersection of the patches (if exist), one
        #parameter must be fixed
        pt = np.zeros([3,1])
        conv =0
        tol = 1e-6 # in x,y,z
        tol2 = 1e-4 # in u,v
        # nt = self.nt
        #print(uvt.shape)
        #print(uvt[0])
        #print(uvt)
        #print(m,o,k,l)

        X2 = surf2.poles[m:m + 3, o:o + 3,:]
        X = surf1.poles[k:k + 3, l:l + 3,:]


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
                uX2 = np.tensordot(u2f,X2,axes=([0],[0]))
                dXYZp2 = np.tensordot(uX2,v2fd,axes=([0],[0]))

            if v2i.shape[0] == 1:
                u2f = surf2.u_basis.eval_vector(m,uvt[2])
                u2fd = surf2.u_basis.eval_diff_vector(m,uvt[2])
                uX2 = np.tensordot(u2fd,X2,axes=([0],[0]))
                dXYZp2 = np.tensordot(uX2,v2f,axes=([0],[0]))

            X1v = np.tensordot(X, vf,axes=([0],[0]))
            dXYZu1 = np.tensordot(ufd,X1v,axes=([0],[0]))
            X1vd = np.tensordot(X,vfd,axes=([0],[0]))
            dXYZv1 = np.tensordot(uf,X1vd,axes=([0],[0]))
            J = np.column_stack((dXYZu1, dXYZv1, -dXYZp2))

            uX1 = np.tensordot(uf,X,axes=([0],[0]))
            XYZ1 = np.tensordot(uX1,vf,axes=([0],[0]))
            uX2 = np.tensordot(u2f,X,axes=([0],[0]))
            XYZ2 = np.tensordot(uX2,v2f,axes=([0],[0]))[:,0]
            #print(uf.shape)
            #print(vf.shape)
            #print(u2f.shape)
            #print(v2f.shape)
            #print(XYZ1.shape)
            #print(XYZ2.shape)
            deltaXYZ = XYZ1 - XYZ2


            #print(XYZ1)
            #print(XYZ2)

            #print(deltaXYZ)
            uvt = uvt - la.solve(J,deltaXYZ)   #uvt = uvt- J\deltaXYZ
            #print(uvt)
            test,uvt = self._rangetest(uvt,ui,vi,u2i,v2i,0.0)

            test,uvt = self._rangetest(uvt,ui,vi,u2i,v2i,tol2)
            if test == 1:
                if u2i.shape[0] == 1:
                    surf2_pos = surf2.eval(u2i,uvt[2])
                if v2i.shape[0] == 1:
                    surf2_pos = surf2.eval(uvt[2],v2i)

            dist = la.norm(surf1.eval(uvt[0],uvt[1]) - surf2_pos) # may be faster, indices of patches are known

        if test == 1:
            if dist <= tol:
                #pt = la.multi_dot(uf,X,vf) #kron(vf',uf')*X
                uX1 = np.tensordot(uf,X,axes=([0],[0]))
                pt = np.dot(uX1,vf,axes=([0],[0]))
                conv =1
        else:
            uvt = np.zeros(3,1)

        return uvt, pt, conv

    def _rangetest(self,uvt, ui, vi, u2i, v2i, tol):
    # test if paramaters does not lie outside current patch, otherwise they are returned to the boundary
        test = 0
        #print(uvt)
        print(ui)

        du = np.array([uvt[0] - ui[0], ui[1] - uvt[0]])
        dv = np.array([uvt[1] - vi[0], vi[1] - uvt[1]])
        #print(uvt.shape)
        #print(uvt[0])
        #print(ui[0])
        #print(ui[1])


        if v2i.shape[0] == 1:
            d2p = [uvt[2] - u2i[0], u2i[1] - uvt[2]]
            pi = u2i

        if u2i.shape[0] == 1:
            d2p = np.array([uvt[2] - v2i[0], v2i[1] - uvt[2]])
            pi = v2i
        #print(du)
        for i in range(0,2):
            #print(du[i])
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
    