
# B-spline GEometry Modeling package

<!---
#[![Build Status](https://travis-ci.org/GeoMop/Intersections.svg?branch=master)](https://travis-ci.org/GeoMop/Intersections)
#[![Code Health](https://landscape.io/github/GeoMop/Intersections/master/landscape.svg?style=flat)](https://landscape.io/github/GeoMop/Intersections/master)
#[![Code Climate](https://codeclimate.com/github/GeoMop/Intersections/badges/gpa.svg)](https://codeclimate.com/github/GeoMop/Intersections)
#[![Test Coverage](https://codeclimate.com/github/GeoMop/Intersections/badges/coverage.svg)](https://codeclimate.com/github/GeoMop/Intersections/coverage)
--->

**Goal**: Robust open source tool for creation of parametric geometries and computational meshes via. Python code. 
Primary focus are hydrogeological applications with geometries including both random fractures and deterministic natural or antropogenic features.

## Rationale
GMSH is a mature meshing tool recently complemented by the 'gmsh-sdk' intreface library. However its practical usage (from Python) have several issues:

- Geometry bolean operations (based on OCC) doesn't preserve "compatible" geometry, i.e. boundary separation two volumes exists only once. This
  makes major problems in meshing.
- Lack of support to work with "groups of shapes". E.g. a group of subdomains fragmented by a fracture  network results in a mess of all resulting subshapes.
  Note, that we must make a single fragmenting operation in order to get "compatible" geometry.
- Compatible boundary shapes can only be retrieved from the final volume, loosing information about it's parts (e.g. boundary of internal hole vs. outer boundary).
- The regions (physical groups in GMSH) are composed from the shapes, while the compatible geometry requests a single region per shape. So it seems logical to assigne regions to the shapes.
- For thousands of regions, the internal GMSH/OCC algorithms are extremaly slow (probably do to quadratic complexity).
- 'gmsh-sdk' builds on semantics of GMSH scripting language leading to cumbersome usage from Python, namely for Fields and Options.



## Features:
### bgem.bspline
Longterm goal to have own CAD like library producing only compatible geometries (indepdent of OCC).
- representation of B-spline curves and surfaces
- approximation of point clouds by B-spline curves and surfaces
- intersections of curves and surfaces
- composition of compatible 3D geometries using BREP format and B-splines
- ultimate goal: fast algorithms for B-spline logical operations
- work in progress

### bgem.gmsh
Wrapping 'gmsh-sdk' meat and bones into enjoyable being.
- documented interface to usefule GMSH options
- documented and usable wrapper for 'Fields'
- operations with groups of shapes
- own association of shapes with regions, assigned just before meshing or even after meshing
- work in progress

### bgem.polygons
Decomposition of the plane into disjoint polygons by line segments. 
- keep compatibility (i.e. single segment separting two polygons)
- support for merging close points and segments with sharp angles (enhance regularity of resulting mesh)
- support for assignment of regions (or other data) to the shapes (points, segments, polygons)
- support for undo/redo of the operations

 
## Authors

Jan Březina, Jiří Kopal, Radek Srb, Jana Ehlerová, Jiří Hnídek
 
## Dependencies

* [bih](https://github.com/flow123d/bih) package
* [gmsh-sdk](https://pypi.org/project/gmsh-sdk/) package



## Theory references
[Patrikalakis-Maekawa-Cho](http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/mathe.html)


## Similar libraries (for B-splines]

- [NURBS++](http://libnurbs.sourceforge.net/old/documentation.shtml) - unmantained, last updates from 2002, seems there is no support for intersections
- [libnurbs](https://sourceforge.net/projects/libnurbs/) - effort to add intersections and other features to the [openNURBBS](https://www.rhino3d.com/opennurbs)
  library provided by Rhino
- [SINTEF SISL](https://www.sintef.no/sisl) - mature, mantained, features, C lib:
    - approximation for curves
    - intersection of curves
    - closest point problems for curves
    - evaluation and manipulation of curves
    - approximation of surfaces
    - intersection of surfaces: topology and inspection of the intersection curve
    - evaluation and manipulation of surfaces
- [Ayam](http://ayam.sourceforge.net/) - under development, 3d modeling tool
   .

