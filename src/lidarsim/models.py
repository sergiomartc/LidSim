"""Contains models to calculate trajectories from the pointcloud. These models must be defined as classes, that take
as many initialization parameters as necessary, and that contain a fit() method that takes the pointcloud data and
returns the detected trajectory"""
import numpy as np

class Centroid:
    """Calculates the detected trajectory point as the average point of the pointcloud"""

    def fit(self, pointcloud):
        detected_pose = np.append(np.nanmean(pointcloud, axis=0), [np.nan, np.nan, np.nan])
        return detected_pose


class SphereRegression:
    """Least squares sphere fit of the pointcloud, only relevant when the object is a sphere

    Code adapted from: https://jekel.me/2015/Least-Squares-Sphere-Fit/
    """

    def fit(self, pointcloud):

        # Selects only the non nan rows of the pointcloud
        points = np.copy(pointcloud)[~np.isnan(pointcloud).any(axis=1)]
        # Number of points available
        n_points = len(points)

        # Assemble the A matrix
        sp_x = points[:, 0]
        sp_y = points[:, 1]
        sp_z = points[:, 2]
        A = np.zeros((n_points, 4))
        A[:, 0] = sp_x * 2
        A[:, 1] = sp_y * 2
        A[:, 2] = sp_z * 2
        A[:, 3] = 1

        #   Assemble the f matrix
        f = np.zeros((n_points, 1))
        f[:, 0] = (sp_x * sp_x) + (sp_y * sp_y) + (sp_z * sp_z)

        # Least squares fit
        C, residules, rank, singval = np.linalg.lstsq(A, f, rcond=None)

        # Construct the detected pose array
        detected_pose = np.array([C[0][0], C[1][0], C[2][0], np.nan, np.nan, np.nan])

        return detected_pose
