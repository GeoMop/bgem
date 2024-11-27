"""
Linear transformation in 3d space.
"""
import copy
from typing import *
import numpy as np
from bgem import ParamError

def check_matrix(mat, shape, values, idx=()):
    '''
    Check shape and type of scalar, vector or matrix.
    :param mat: Scalar, vector, or vector of vectors (i.e. matrix). Vector may be list or other iterable.
    :param shape: List of dimensions: [] for scalar, [ n ] for vector, [n_rows, n_cols] for matrix.
    If a value in this list is None, the dimension can be arbitrary. The shape list is set fo actual dimensions
    of the matrix.
    :param values: Type or tuple of  allowed types of elements of the matrix. E.g. ( int, float )
    :param idx: Internal. Used to pass actual index in the matrix for possible error messages.
    :return:
    TODO: replace check_matrix by conversion to appropriate numpy array.
    '''
    try:

        if len(shape) == 0:
            if not isinstance(mat, values):
                raise ParamError("Element at index {} of type {}, expected instance of {}.".format(idx, type(mat), values))
        else:

            if shape[0] is None:
                shape[0] = len(mat)
            l=None
            if not hasattr(mat, '__len__'):
                l=0
            elif len(mat) != shape[0]:
                l=len(mat)
            if not l is None:
                raise ParamError("Wrong len {} of element {}, should be  {}.".format(l, idx, shape[0]))
            for i, item in enumerate(mat):
                sub_shape = shape[1:]
                check_matrix(item, sub_shape, values, idx = (i, *idx))
                shape[1:] = sub_shape
        return shape
    except ParamError:
        raise
    except Exception as e:
        raise ParamError(e)


Matrix = np.array   #shape 3x4
Power = int
class Transform:
    """
    Defines an affine transformation in 3D space. (Corresponds to the Location inthe BREP file).
    """

    @staticmethod
    def _identity_matrix():
        return np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)

    @staticmethod
    def _flat(composition):
        """
        Combine transfromations from self._composition.
        """
        result = np.eye(4)
        for t, p in composition:
            full_matrix = Transform._matrix_expand(t.matrix)
            for _ in range(p):
                result = full_matrix @ result
        return Transform._matrix_compress(result)

    @staticmethod
    def _matrix_expand(matrix):
        return np.concatenate([matrix, np.array([[0, 0, 0, 1]])], axis=0)

    @staticmethod
    def _matrix_compress(matrix):
        return matrix[:-1, :]

    def __init__(self, matrix: Matrix = None):
        """
        Constructor for elementary afine transformation.
        :param matrix: Transformation matrix 3x4. First three columns forms the linear transformation matrix.
        Last column is the translation vector.
        The full affine transform matrix is available through the full_affine_matrix property.
        TODO: allow passing the full affine transfrom
        """
        self._composition = []
        if matrix is None:
            self._matrix = None
        else:
            if matrix.shape == (4, 4):
                assert np.allclose(matrix[3],   [0, 0, 0, 1])
                matrix = matrix[:3]
            check_matrix(matrix, [3, 4], (int, float))
            self._matrix = np.array(matrix, dtype=float)

    def is_composed(self) -> bool :
        """
        Composed of singel matrix with power one.
        """
        return len(self._composition) > 0

    def is_identity(self):
        return self._matrix is None

    @property
    def matrix(self):
        if self.is_identity():
            return Transform._identity_matrix()
        else:
            return self._matrix

    @property
    def affine_matrix(self):
        return np.concatenate((self._matrix, np.array([[0,0,0,1]])))

    def __call__(self, points:np.array) -> np.array:
        """
        :param points: shape (3, N)
        return: transformed array, shape (3, N)
        """
        return self.matrix[:, :3] @ points + (self.matrix[:, 3])[:, None]


    def __pow__(self, power:int):
        """
        Return power of the transform.
        """
        if self.is_composed():
            composition = [(t, p * power) for t, p  in self._composition]
        else:
            composition = [(self, power)]
        result = Transform()
        result._composition = composition
        result._matrix = result._flat(composition)
        return result

    def __matmul__(self, other: 'Transform') -> 'Transform':
        """
        Can use matrix multiplication operator '@' to compose Locations.
        E.g.

        location = Identity @ Location.Rotate([0,1,0], angle) @ Location.Translate([1,2,3])

        is equivalent to

        location = Identity.rotate([0,1,0], angle).translate([1,2,3])

        Return ComposedLocation.
        """
        if other.is_identity():
            return self
        if self.is_identity():
            return other
        a = self ** 1
        b = other ** 1
        result = Transform()
        result._composition = b._composition + a._composition
        result._matrix = result._flat(result._composition)
        return result


    def translate(self, vector):
        """
        Apply translation by the shift 'vector'.
        Return a composed location.
        """
        matrix = Transform._identity_matrix()
        matrix[:, 3] += np.array(vector, dtype=float)
        return Transform(matrix) @ self

    def rotate(self, axis, angle, center=(0, 0, 0)):
        """
        Assuming the coordinate system:

        ^ Y
        |
        Z --> X

        Create a rotation anticlockwise (right hand rule) by the `angle` (radians)
        around the (normalised) `axis` vector pointing to you.
        Optionally the center of the rotation can be specified. This first shift center to origin,
        rotate, and then shift back.
        """
        matrix = Transform._identity_matrix()
        if angle != 0.0:
            center = np.array(center, dtype=float)
            axis = np.array(axis, dtype=float)
            axis /= np.linalg.norm(axis)

            W = np.array(
                [[0, -axis[2], axis[1]],
                 [axis[2], 0, -axis[0]],
                 [-axis[1], axis[0], 0]])
            M = np.eye(3) +  np.sin(angle) * W + 2 * np.sin(angle/2) ** 2 * W @ W
            matrix[:, 3] -= center
            matrix = M @ matrix
            matrix[:, 3] += center
        return Transform(matrix) @ self

    def scale(self, scale_vector, center=(0, 0, 0)):
        """
        Create a scaling the 'scale_vector' keeping 'center' unmodified.
        """
        matrix = Transform._identity_matrix()
        center = np.array(center, dtype=float)
        scale_vector = np.array(scale_vector, dtype=float)
        matrix[:, 3] -= center
        matrix = np.diag(scale_vector) @ matrix
        matrix[:, 3] += center
        return Transform(matrix) @ self

