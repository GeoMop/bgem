import sys
build_path="/home/jiri/Soft/Geomop/Intersections/external/bih/build"
sys.path+=[build_path]
print(sys.path)

import bih
import numpy as np
import numpy.linalg as la





class IsecSurfSurf:

    def __init__(self, surf1, surf2):
        self.surf1 = surf1
        self.surf2 = surf2
        self.tree1 = self.bounding_boxes(self.surf1)
        self.tree2 = self.bounding_boxes(self.surf2)

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



    def get_intersection(surf1,surf2,nt):
    # nt - number of threads (integer)
            #X,Xs, u_n_intervs,v_n_intervs,u_knots,v_knots,
     #us_n_intervs,vs_n_intervs,us_knots,vs_knots,isec,n_isec, nit,nt )
        # computes intersection of BSpline patch with BSpline thread

        def patch_patch_intersection( uvt, ui,vi,u2i,v2i, u_knots, v_knots, X,us_knots, vs_knots, Xs, nit ,m,o,k,l):
            #returns coordinetes of the intersection of the patches (if exist), one
            #parameter must be fixed
            pt = np.zeros(3,1)
            conv =0
            tol = 1e-6; # in x,y,z
            tol2 = 1e-4; # in u,v

            if np.length(u2i) == 1:
                u2f = surf2.u_basis._eval_vector_deg_2(m,u2i)

            if np.length(v2i) == 1:
                v2f = surf2.v_basis._eval_vector_deg_2(o,v2i)


            for i in range(nit):
                uf = surf1.u_basis._eval_vector_deg_2(k,uvt(1))
                vf = surf1.v_basis._eval_vector_deg_2(l,uvt(2))
                ufd = surf1.u_basis._eval_diff_vector_deg_2(k,uvt(1))
                vfd = surf1.v_basis._eval_diff_vector_deg_2(l,uvt(2))

            if np.length(u2i) == 1:
                v2f = surf2.v_basis._eval_vector_deg_2(o,uvt(3))
                v2fd = surf2.v_basis._eval_diff_vector_deg_2(o,uvt(3))
                dXYZp2 = (kron(v2fd',u2f')*Xs)';

            if np.length(v2i) == 1:
                u2f = splinebasevec(us_knots,uvt(3),0,m)
                u2fd = splinebasevec(us_knots,uvt(3),1,m)
                dXYZp2 = (kron(v2f',u2fd')*Xs)';


            dXYZu1 = (kron(vf',ufd')*X)'
            dXYZv1 = (kron(vfd',uf')*X)'
            J = [dXYZu1, dXYZv1, -dXYZp2]
            deltaXYZ = (kron(vf',uf') * X)' - (kron(v2f',u2f') * Xs)'
            uvt = uvt- J\deltaXYZ
            test,uvt = rangetest(uvt,ui,vi,u2i,v2i,0.0)

            test,uvt = rangetest(uvt,ui,vi,u2i,v2i,tol2);
            if test == 1
            dist = get_delta(u_knots, v_knots,us_knots, vs_knots,uvt,u2i,v2i,k,l,m,o,X,Xs);


            if test == 1:
                if dist <= tol:
                    pt =kron(vf',uf')*X
                    conv =1
            else:
                uvt = np.zeros(3,1)

            return uvu2v2, ptl, conv


        n_points = 0
        ninter =  np.zeros(surf1.u_basis.n_intervals,surf1.v_basis.n_intervals)
        for k in range(surf1.u_basis.n_intervals):
            us1 = surf1.u_basis.knots[k + 1]
            ue1 = surf1.u_basis.knots[k + 2]
            u1_c =(us1 + ue1)/2
            ui = np.array([us1, ue1])
            for l in range(surf1.v_basis.n_intervals):
                vs1 = surf1.v_basis.knots(l+1)
                ve1 = surf1.v_basis.knots(l+2)
                v1_c = (vs1  + ve1)/2
                vi = np.array([vs1, ve1])
                s=0
                if n_isec(k,l) ~= 0
                    for p =1: n_isec(k,l)
                        m = ceil(isec(k,l,p) / surf2.u_basis.n_intervals)
                        o = isec(k,l,p) - (m-1)*surf2.u_basis.n_intervals
                        sp_i = (m-1)*(surf2.u_basis.n_intervals) + o
                        # v2 fixed
                        u2i = [surf2.u_basis.knots(m+1), surf2.u_basis.knots(m+2)]
                        v_knot = np.linspace(surf2.v_basis.knots(o+1),surf2.v_basis(o+2),nt)
                        for h  in range(nt):  #=1:length(v_knot)
                            u2_c = (u2i[0] + u2i[1])/2
                            v2i = v_knot[h]
                            uvu2v2 = [u1_c, v1_c, u2_c] # initial condition
                            uvu2v2,ptl,conv  = patch_patch_intersection( uvu2v2, ui,vi,u2i,v2i, u_knots, v_knots, X,us_knots, vs_knots, Xs, nit ,m,o,k,l);
                            if conv ~= 0
                                s = s+1;
                                n_points = n_points +1;
                                point(n_points,:) = [uvu2v2',v_knot(h),ptl(1),ptl(2),ptl(3),k,l,m,o];

                        # u2 fixed
                        v2i = [vs_knots(o+2) vs_knots(o+3)];
                        u_knot = linspace(us_knots(m+2),us_knots(m+3),nt);
                        for h =1:length(u_knot)
                            v2_c = (vs_knots(o+2) + vs_knots(o+3))/2;
                            u2i = u_knot(h);
                            uvu2v2 = [u1_c;v1_c;v2_c]; % initial condition
                            [ uvu2v2,ptl,conv ] = patch_patch_intersection( uvu2v2, ui,vi,u2i,v2i, u_knots, v_knots, X,us_knots, vs_knots, Xs, nit ,m,o,k,l);
                            if conv ~= 0
                                s = s+1;
                                n_points = n_points +1;
                                point(n_points,:) = [uvu2v2(1:2)',u_knot(h),uvu2v2(3),ptl(1),ptl(2),ptl(3),k,l,m,o];
                            end
                        end
                        ninter(k,l) = s;

        return point , n_points,ninter




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
    