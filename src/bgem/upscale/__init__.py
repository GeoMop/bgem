from .fem import Fe, flat_dim, tensor_dim, Grid, FEM, upscale
from .fields import voigt_to_tn, tn_to_voigt
from .voxelize import FracturedDomain, Intersection, FracturedMedia, \
                       intersection_decovalex, intersection_cell_corners, fr_conductivity