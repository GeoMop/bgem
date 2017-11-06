import sys
build_path="/home/jiri/Soft/Geomop/Intersections/external/bih"
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
        n_patch = (surf.u_basis.n_intervals - 2)*(surf.v_basis.n_intervals - 2)
        patch_poles = np.zeros([9, 3, n_patch])
        i_patch = 0
        for k in range( surf.u_basis.n_intervals - 2):
            for l in range(surf.v_basis.n_intervals - 2):
                n_points = 0
                i_patch += 1
                for i in range(0,3):
                    for j in range(0,3):
                        patch_poles[n_points,:,i_patch] = surf.poles[k+1, l+j, :]

        boxes = [bih.AABB(patch_poles[:,:,p]) for p in range(n_patch)]
        tree.add_boxes( boxes )
        tree.construct()
        return tree

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
    