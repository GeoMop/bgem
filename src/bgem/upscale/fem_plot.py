import pyvista as pv
import numpy as np
import pathlib
from bgem.upscale import Grid
import matplotlib.pyplot as plt
import numpy as np

"""
Custom plotting function, mainly for debugging and test purpose.
- VTK output of cell and vector fields on a structured grid
- PyVista plot of a given cell / point field
- 3d Scatter glyph visualization of a vector/tensor field 
"""



def grid_fields_vtk(grid:Grid,
                cell_fields = None,
                point_fields = None,
                vtk_path: pathlib.Path=None):
    """
    Output given cell and point fields to VTK.
    Return: the pv grid object with the cell and point data arrays
    """
    x, y, z = np.meshgrid(*grid.axes_linspace(), indexing='ij')
    pv_grid = pv.StructuredGrid(x, y, z)
    if cell_fields is not None:
        for k, v in cell_fields.items():
            if pv_grid.GetNumberOfCells() != v.shape[0]:
                raise ValueError(f"Cell field size {v.shape[0]} mismatch number of cells {pv_grid.GetNumberOfCells()}")
            pv_grid.cell_data[k] = v
    if point_fields is not None:
        for k, v in point_fields.items():
            if pv_grid.GetNumberOfPoints() != v.shape[0]:
                raise ValueError(f"Point field size {v.shape[0]} mismatch number of points {pv_grid.GetNumberOfPoints()}")
            pv_grid.point_data[k] = v
    if vtk_path is not None:
        pv_grid.save(str(vtk_path))
    return pv_grid

def create_plotter(**options):
    #pv.start_xvfb()
    font_size = 20
    #pv.global_theme.font.size = font_size
    plotter = pv.Plotter(**options)
    # Add axes and bounding box for context
    plotter.add_axes()
    plotter.show_grid()
    plotter.add_bounding_box()
    return plotter
#
# def pv_plot_mesh(pv_grid, color='grey', opacity=1.0, plotter = None):
#     """
#     Usage:
#     plotter = pv_plot_mesh(mesh_one)
#     plotter = pv_plot_mesh(mesh_two, plotter=plotter)
#     plotter.show()
#     """
#     if plotter is None:
#         pv.start_xvfb()
#         font_size = 20
#         pv.global_theme.font.size = font_size
#         plotter = pv.Plotter(off_screen=True, window_size=(1024, 768))
#         # Add axes and bounding box for context
#         plotter.add_axes()
#         plotter.show_grid()
#         plotter.add_bounding_box()
#
#     #plotter.set_font(font_size=font_size)
#     plotter.add_mesh(pv_grid, color=color, opacity=opacity)
#
#     return plotter


def plot_grid(n):
    """
    Create
    :param n:
    :return:
    """
    # Create a PyVista mesh from the points
    points = np.mgrid[:n, :n, :n] / (n - 1.0)
    mesh = pv.StructuredGrid(*points[::-1])
    points = points.reshape((3, -1))
    return points, mesh

def pv_plotter(meshes):
    # Create a plotting object
    p = pv.Plotter()

    # Add axes and bounding box for context
    p.add_axes()
    p.show_grid()
    p.add_bounding_box()

    # Show the plot
    p.show()


def scatter_3d(mesh, values, n=5):
    # Normalize the function values for use in scaling
    scaled_values = (values - np.min(values)) / (np.max(values) - np.min(values))

    mesh['scalars'] = scaled_values

    # Create the glyphs: scale and color by the scalar values
    geom = pv.Sphere(phi_resolution=8, theta_resolution=8)
    glyphs = mesh.glyph(geom=geom, scale='scalars', factor=0.3)


    # Add the glyphs to the plotter
    p.add_mesh(glyphs, cmap='coolwarm', show_scalar_bar=True)



def plot_fn_3d(fn, n=5):
    points, mesh = plot_grid(n)
    values = fn(*points[::-1])
    scatter_3d(mesh, values)


def f(x, y, z):
    return x * (1 - y) * z * (1 - z) * 4


#plot_fn_3d(f)