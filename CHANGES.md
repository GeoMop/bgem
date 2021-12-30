# Release 0.2.0
- ObjectSet.mesh_step can be used to associate a mesh step with a the object set.
  Setting the mesh step is postponed right before the mesh is created otherwise the mesh step is forgotten by GMSH
  during later geometric operations.
- GeometryOCC new methods:
  operations: extrude, revolve
  primitives: circle, disc, cylinder_discrete, disc_discrete
- Complex geometry of the Greet experiment is presented as a tutorial 'tutorials/01_Greet_experiment'.


# Release 0.1.1

- imporoved testing and publishing actions

### bspline

- bspline_approx - faster approximation of point grids by bspline surfaces
- preliminary version of working surface-surface intersections

### added geometry subpackage
Creation of a layered BREP geometry from the LayerEditor format.


# Release 0.1.0

### bspline
Longterm goal to have own CAD like library producing only compatible geometries (indepdent of OCC).

- representation of B-spline curves and surfaces
- approximation of point clouds by B-spline curves and surfaces
- intersections of curves and surfaces
- composition of compatible 3D geometries using BREP format and B-splines
- ultimate goal: fast algorithms for B-spline logical operations
- work in progress

### gmsh
Wrapping 'gmsh-sdk' meat and bones into enjoyable being.

- documented interface to usefule GMSH options
- documented and usable wrapper for 'Fields'
- operations with groups of shapes
- own association of shapes with regions, assigned just before meshing or even after meshing
- work in progress

### polygons
Decomposition of the plane into disjoint polygons by line segments. 

- keep compatibility (i.e. single segment separting two polygons)
- support for merging close points and segments with sharp angles (enhance regularity of resulting mesh)
- support for assignment of regions (or other data) to the shapes (points, segments, polygons)
- support for undo/redo of the operations





