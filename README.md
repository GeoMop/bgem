
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
GMSH is a mature meshing tool providing a minimalistic API common for several languages. For Python the GMSH library and its Python API is accessible
through the 'gmsh' package. However usage for practical applications is nontrivial partly due to intrinsic limitations of the underlaying OpenCASCADE 
library and partly due to generic API lacking the Python idioms. In particular, we have identified following sort of problems:

- Geometry bolean operations (based on OCC) doesn't preserve "compatible" geometry, e.g. a surface separating two volumes can be duplicate, once for each volume. 
  This produces a mesh with duplicated nodes loosing the coupling in the FEM simulations. Function for removal of duplicities is provided, but not reliable.
- API only operates with atomic geometric entities lacking a support to organise them into logical groups, especialy after fragmenting due to bolean oeprations.
- GMSH forms "physical groups" from the geometric entities so an entity can be part of two physical groups. That leads to duplicate elements after meshing, 
  so we rather want "physical groups" (called "regions" for distinction) assigned to the geometric objects. We also want to preserve assigned regions during set operation if possible.
- Consider an extracted boundary A' (e.g. for prescribing a boundary condition) of an object A. Once the object A is part of a set operation it becomes fragmented to a set B and 
  there is no way how to get boundary corresponding to A'.
- For thousands of "physical groups", the internal GMSH/OCC algorithms are extremaly slow (probably do to quadratic complexity).
- The generic GMSH API is cumbersome, namely for Fields and Options.
- For Fields related to the geometic entites these must be the final geometric entities, their fragmentation by boolean operations 
  leads to spurious results. 


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

## Installation

### Installation from PYPI

System-wide installation of a last version from PYPI. Need root/admin access (or sudo). 

    pip install bgem

Installation from PYPI into the user's directory, no admin access necessary. E.g. on a cluster.
However prefered is usage of a [virtual environment](https://docs.python.org/3/tutorial/venv.html).

    pip install --user bgem

### Instalation from sources

Installation from sources located in DIR (copy of sources is performed). 

    pip install DIR

Installation from sources located in DIR. It uses links to the editable sources (usefull for debugging).

    pip install -e DIR


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

