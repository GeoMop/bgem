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
Example:
