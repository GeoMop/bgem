# bgem.bspline subpackage

Modules:
- **brep_writer** - hierarchical construction of the boundary representation with management of shared shapes 
  (edges, faces) which is crucial for correct meshing (e.g. via gmsh), own writer to the 
  [BREP file format](https://www.opencascade.com/doc/occt-6.7.0/overview/html/occt_brep_format.html)
  
- **bspline** - B-spline curve and B-spline surface classes and related, mainly for internal use
- **bspline_approx** - Approximation of a point grid (close to a plane) by the B-spline surface (B-spline function)
- **bspline_plot** - auxiliary plotting tools
- **isec_curve_surf** - computing intersection of a B-spline curve and B-spline surface
- **isec_curve_surf** - computing intersection of two B-spline surfaces, approximate by a one or more B-spline curves  

## bspline_approx
Creating a B-spline surface from a point grid.
    
1. Create the approximation object from the points in the grid file.
   
        surf_approx = bs_approx.SurfaceApprox.approx_from_file(grid_path)

2. Compute minimal surface bounding rectangular of points projected to the XY plane.
   or use own XY rectangle given as array of shape (4,2) of the four vertices.
    
        quad = surf_approx.compute_default_quad()

3. Try to guess dimensions of the (semi regular) input grid.
        
        nuv = surf_approx.compute_default_nuv()
    We usually want much sparser approximation.
    
        nuv = nuv / 5

4. Compute the approximation.
    
        surface = surf_approx.compute_approximation()
        
Result is a ZSurface, that is only Z coordinate is B-spline while XY are just linear 
transformation of the UV coordinates (given by the bounding quad).
In order to create the fully parametric surface one can use:

    full_surface = surface.make_full_surface()
    
    
See the file `tests/bspline/test_bs_approx_example.py' for the full usage example.
 
