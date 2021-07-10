"""In flight test applications, the frame of reference that is used is always that of the aircraft, so that all sensor
and measurements are done with the same criteria. Because the lidar simulation is done with the sensor as the frame of
reference, as that is the way that all lidar technologies work, there exists a need to easily change these coordinates
from one base to the other. For that, a transformation matrix between the two is first needed, requiring at least four
points whose coordinates are know in both bases. From that transformation matrix, the pointcloud that would be generated
by the lidar from its point of view could easily be transformed into a pointcloud in the aircraft base."""

import numpy as np

def transf_params(coord_b1, coord_b2):
    """Obtains the transformation matrix and shift of origin between two different basis from >=4 different
    points

    Arguments:
        coord_b1: 2D np.array,
            The point coordinates on the first base for 4 or more points
        coord_b2: 2D np.array
            The point coordinated on the second base for 4 or more points

    Returns:
        transf_matrix: 2D np.array
            The transformation matrix from b1 to b2
        shift_origin: 2D np.array
            The shift of origin points using the canonic coordinates
    """

    # The change of basis is defined as v' = [M]*v + c, where:
    # - v' and v define the position for one point in the new and old basis respectively, b1 and b2.
    # - M is the transformation matrix from b1 to b2.
    # - c is the shift of origins between the two basis from b1 to b2.
    # Given the positions for n>3 points in both basis, the transformation matrix and shift of origins can be obtained
    # by solving the following linear system:
    #
    # [ v1 000 000   1 0 0 ][ M11 ]     [ v1'1 ]
    # | 000 v1 000   0 1 0 || M12 |     | v1'2 |
    # | 000 000 v1   0 0 1 || ... |     | v1'3 |
    # | v2 000 000   1 0 0 || M33 |  =  | v2'1 |    ,     A*x = B
    # |       ...          || ... |     | ...  |
    # | 000 vn 000   0 1 0 || c2  |     | vn'2 |
    # [ 000 000 vn   0 0 1 ][ c3  ]     [ vn'3 ]

    # Both arrays must contain the same number of points
    if np.shape(coord_b1)[0] != np.shape(coord_b2)[0] or len(np.shape(coord_b1)) !=2 or len(np.shape(coord_b2)) !=2:
        print('Values given do not match!')
        return

    # number of points
    n = np.shape(coord_b1)[0]
    if n <4:
        print('Not enough values given!')
        return

    # Building the matrix A
    # Constructing the left part first
    A_left = np.zeros((3*n,9))
    for i in range(n):
        for j in range (3):
            A_left[3*i + j, 3*j : 3*(j + 1)] = coord_b1[i,:]
    # Constructing the right part
    A_right = np.identity(3)
    for i in range(n-1):
        A_right = np.vstack((A_right, np.identity(3)))
    A = np.hstack((A_left, A_right))

    # The coordinates on the second base serve as the independent matrix
    b = coord_b2.flatten()

    # Solving for least squares
    values = np.linalg.lstsq(A, b, rcond=None)[0]
    transf_matrix = np.reshape(values[:-3], (3,3))
    shift_origin = values[-3:]

    return transf_matrix, shift_origin


def coordinate_transf(point_cloud_b1, transf_matrix, shift_origin):
    """Changes a set of points from one base to another

    Inputs:
        point_cloud_b1: 2D np.array
            Pointcloud in the first base
        transf_matrix: 2D np.array
            The transformation matrix between the two basis
        shift_origin: 1D np.array
            The shift of origin points between the two basis
    Returns:
        point_cloud_b1: 2D np.array
            Pointcloud in the second base
    """

    # The change of basis is defined as v' = [M]*v + c, where:
    # - v' and v define the position for one point in the new and old basis respectively, b1 and b2.
    # - M is the transformation matrix from b1 to b2.
    # - c is the shift of origins between the two basis from b1 to b2.
    # Given both the transformation matrix and the shift of origins, to transform a point on base b1 into its
    # representation in base b2 it is as simple as computing the expression from the definition.

    # If the input is a single point, returns the 1D array for that point in the second base
    if len(np.shape(point_cloud_b1)) == 1:
        return np.matmul(transf_matrix, point_cloud_b1) + shift_origin

    # For a point cloud as input in the form of a 2D array
    point_cloud_b2 = np.zeros(np.shape(point_cloud_b1))
    for i in range(np.shape(point_cloud_b1)[0]):
        # Constructs a matrix of the same dimensions but with the corresponding points in the second base
        point_cloud_b2[i] = np.matmul(transf_matrix,point_cloud_b1[i]) + shift_origin
    return point_cloud_b2


