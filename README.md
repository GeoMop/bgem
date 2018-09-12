Intersections
==============

[![Build Status](https://travis-ci.org/GeoMop/Intersections.svg?branch=master)](https://travis-ci.org/GeoMop/Intersections)
[![Code Health](https://landscape.io/github/GeoMop/Intersections/master/landscape.svg?style=flat)](https://landscape.io/github/GeoMop/Intersections/master)
[![Code Climate](https://codeclimate.com/github/GeoMop/Intersections/badges/gpa.svg)](https://codeclimate.com/github/GeoMop/Intersections)
[![Test Coverage](https://codeclimate.com/github/GeoMop/Intersections/badges/coverage.svg)](https://codeclimate.com/github/GeoMop/Intersections/coverage)


Computing intersections of B-spline curves and surfaces.

Library focus on fast intersection algorithms for non-degenerate intersections of B-spline curves and surfaces
of small degree (especially quadratic). 

Requirements
------------

* g++ 4.x or newer
* cmake 3.x

In order to install BIH package locally for development just run the 'bih_install.sh' script.

Theory
------
[Patrikalakis-Maekawa-Cho](http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/mathe.html)


Similar libraries
-----------------

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

