import vtk
import os
import numpy as np
from math import sin, cos, radians
from stl import mesh


class Scenario:

    def __init__(self, object_file, trajectory=None):
        """Defines an object moving along a predetermined path, and calculates all its positions as .stl files

        This is the simulated case environment for the lidar to scan. The predetermined path is defined as the series of
        coordinates that its center point has at a series of time stamps, as well as the orientation of the object with
        respect to its original state at these time stamps. The more points defined, the smoother any future
        calculations or animations will be. The so called "reference point" is the part of the object, not necessarily
        a point, that sits in the center in the original .stl file.

        Args:
            object_file: str
                File path of the triangulated mesh of the object of interest, in either .obj/.stl file (uses .stl
                native libraries). The object must be centered around the origin for rotations to be performed, unless
                no trajectory is provided, in which case the object must be placed far enough from the origin for the
                sensor to be able to detect it.
            trajectory: 2D np.array (7 columns)
                The coordinates of the reference point and the orientation of the object at specified times,
                (t, x, y, z, psi, theta, phi). The time coordinate must always be increasing.
                Defaults to None, the script will interpret this as a stationary object. If the trajectory of the object
                is not defined, it needs to be situated far enough from the sensor to be detected.

        Attributes:
            self.object_file: str
                File path of the triangulated mesh of the object of interest
            self.mesh:
                Original numpy-stl mesh
            self.trajectory: 2D np.array (7 columns)
                The coordinates of the reference point and the orientation of the object at specified times,
                (t, x, y, z, psi, theta, phi)
        """

        # The object file (.stl) has to be situated with its reference point at (0, 0, 0), since rotations will be
        # made around the origin. After rotation, the corresponding translation will be done.
        self.object_file = object_file
        self.mesh = mesh.Mesh.from_file(object_file)  # Original numpy-stl mesh
        self.trajectory = trajectory
        # The mesh is loaded as an object in numpy-stl, modified internally, and returned as vtk polydata

    def _loadSTL(self, object_file):
        """Generates a vtkPolyData object for the triangulated mesh

        Aruments:
            object_file: str
                File path of the triangulated mesh in .stl/.obj format

        Returns:
            polydata: stored triangulated mesh of the object
        """

        # Libraries used are .stl native, does not directly support .obj files without transforming them first
        if object_file.endswith('.obj'):
            polydata = self._obj_to_stl(object_file)
            # Currently unsupported
        else:
            readerSTL = vtk.vtkSTLReader()
            readerSTL.SetFileName(object_file)
            # Update the reader, ie, read the .stl file
            readerSTL.Update()
            polydata = readerSTL.GetOutput()

        # If there are no points in polydata something went wrong
        if polydata.GetNumberOfPoints() == 0:
            # The vtk module does not contemplate many exceptions, it is best practice to test for successful operations
            raise ValueError(
                "No point data could be loaded from " + object_file
            )

        # Otherwise, returns the polydata object
        return polydata

    def _obj_to_stl(self, file):
        """Transforms a .obj file to a .stl file, and returns its vtk polydata

        Arguments:
            file: str
                File path of the triangulated mesh in .obj format

        Returns:
            polydata: stored triangulated mesh of the object
        """
        # NOT YET SUPPORTED
        # use pymesh to save the obj file as an stl
        # then load the stl file using numpy-stl
        pass

    def _get_mesh(self, pose):
        """Changes both the position and orientation of a triangulated mesh and returns the new numpy stl mesh object

        Arguments:
            pose: array-like
                The position and orientation of the object at a given point, (x, y, z, psi, theta, phi)

        Returns:
            new_mesh: numpy-stl mesh object
                New mesh representing the current state of the object
        """

        # Position and orientation are defined by a translation and a rotation with Tait-Bryan angles, from its original
        # state, meaning the file is loaded and then moved for every frame

        # If no pose is given, it returns the original file as a vtk polydata object
        if pose is None:
            return mesh.Mesh.from_file(self.object_file)  # original numpy-stl mesh

        # Define the numpy-stl mesh object for rotation and translations
        new_mesh = mesh.Mesh.from_file(self.object_file)

        # Define positional arguments
        translation = pose[:3]
        # Angles are given in degrees
        psi = radians(pose[3])
        theta = radians(pose[4])
        phi = radians(pose[5])
        # The object must be centered around the origin at all times for the rotation to be made possible
        # The rotation is defined as follows:
        # The original frame of reference is rotated an angle psi around its z axis, then an angle theta around its y
        # axis, and a final angle phi around its x axis.
        # This is the Tait-Bryan (zyx) convention, used for aerospace vehicle orientation. The orientation matrix
        # from the lidar reference to the original frame of reference would be:
        rotation_matrix = np.array([
            [cos(theta) * cos(psi), cos(theta) * sin(psi), -sin(theta)],
            [sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),
             sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi), sin(phi) * cos(theta)],
            [cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi),
             cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi), cos(phi) * cos(theta)]
        ]).T
        # The matrix is transposed because of how the multiplication will be done: M*v = v.T*M.T, to ease computation

        for i in range(3):
            # The rotation is applied first
            new_mesh.vectors[:, i] = new_mesh.vectors[:, i].dot(rotation_matrix)
            # Vectors are already transposed in the array and stacked vertically, so the whole column can be calculated
            # at once
            # Then the translation, simply adding new position to the starting position, that is, the origin
            new_mesh.vectors[:, i] += translation

        # Return the new, reoriented, numpy-stl mesh
        return new_mesh

    def _get_polydata(self, pose):
        """Changes both position and orientation of a triangulated mesh and returns the new vtk polydata object

        Arguments:
            pose: array-like
                The position and orientation of the object at a given point, (x, y, z, psi, theta, phi)

        Returns:
            polydata: vtk PolyData object
                New mesh object for vtk representing the current state of the object
        """

        # Gets the mesh for the current pose
        mesh_object = self._get_mesh(pose)

        # The new stl object is saved under __bog.stl to read again with vtk
        mesh_object.save('__bog.stl')

        # New saved file is loaded as a vtkmesh
        polydata = self._loadSTL('__bog.stl')
        # The file then gets deleted
        os.remove("__bog.stl")

        # It was found that this approach of saving and then reading the file again was more efficient than simply
        # feeding the data of one numpy-stl mesh object to a vtk polydata object

        # The polydata gets finally returned
        return polydata

    def _get_pose(self, time):
        """Returns the object pose at a given time interpolating between instants in self.trajectory

        Arguments:
            time: float
                The precise time for the object pose measurement

        Returns:
            pose: np.array
                The pose of the object at that moment of time
        """

        if self.trajectory is None:
            # If the trajectory is not defined, then the pose cannot be inferred
            return None

        def _find_nearest(array, value):
            """Finds the array indexes of the numbers that most closely bound the given value"""
            idx = (np.abs(array - value)).argmin()
            if array[idx] > value:
                return idx-1, idx
            elif array[idx] < value:
                return idx, idx+1

        if time in self.trajectory[:,0]:
            # If the time specified is in the array, get the corresponding index and the pose to return
            idx = np.where(self.trajectory[:,0] == time)[0][0]
            pose = self.trajectory[idx,1:]
            return pose

        # The time specified is not in the array, need to interpolate:
        else:
            # These indexes bound the timestamp inside the array
            indexes = _find_nearest(self.trajectory[:,0], time)
            # If the first index equates to the length of the trajectory array, that timestamp is not covered in the
            # trajectory
            if indexes[0] == len(self.trajectory[:,0]):
                # The time given falls outside the given trajectory
                raise IndexError("Time value given falls outside the defined trajectory")
            else:
                pose = np.empty(6)
                for i in range(6):  # For every coordinate the interpolation is performed
                    dt = (self.trajectory[indexes[1],0] - self.trajectory[indexes[0],0])
                    # i+1 is used instead of i because the first element of the self.trajectory array is the timestamp
                    slope = (self.trajectory[indexes[1],i+1] - self.trajectory[indexes[0],i+1])/dt
                    pose[i] = self.trajectory[indexes[0],i+1] + (time - self.trajectory[indexes[0],0])*slope
                return pose

    def _get_frames(self, start, duration=0, fps=1):
        """Calculates the precise poses of a smaller chunk of the trajectory for analysis

        Since a Scenario object is defined as an object following a defined and discrete path, individual frames that
        are not covered directly by the trajectory data are interpolated.

        Arguments:
            start: float
                Starting time for the snapshot
            duration: float
                Total time that the snapshot lasts. Defaults to 0 meaning a single frame
            fps: float
                Number of frames that comprise this snapshot. Defaults to 1 meaning a single frame

        Returns:
            snapshot_frames: generator
                The positions (and orientations) that the object has at the required frames
        """

        if duration == 0:
            yield self._get_pose(start)
        else:
            t = 0
            while t < duration:
                yield self._get_pose(start + t)
                t += 1/fps

    def plotting_playout(self, start, no_frames=1, fps=1):
        """Calculates the precise timestamps and poses with their corresponding meshes of a smaller chunk of the
        trajectory for analysis

        Since a Scenario object is defined as an object following a defined and discrete path, individual frames that
        are not covered directly by the trajectory data are interpolated.

        Arguments:
            start: float
                Starting time for the snapshot
            no_frames: int
                Total number of frames that the snapshot lasts. Defaults to 1 meaning a single frame
            fps: float
                Number of frames per second of the playout

        Returns:
            playout: polydata generator
                The positions and orientations that the object has at the frame as the equivalent mesh
        """

        if no_frames == 1:
            pose = self._get_pose(start)
            yield self._get_mesh(pose)
        else:
            f = 0 # Starting frame is at the starting time
            while f <= no_frames:  # Number of frames count starts at zero
                pose = self._get_pose(start + f/fps)
                yield self._get_mesh(pose)
                f += 1  # Next frame


    def playout(self, start, no_frames=1, fps=1):
        """Calculates the precise timestamps and poses with their corresponding meshes of a smaller chunk of the
        trajectory for analysis

        Since a Scenario object is defined as an object following a defined and discrete path, individual frames that
        are not covered directly by the trajectory data are interpolated.

        Arguments:
            start: float
                Starting time for the snapshot
            no_frames: int
                Total number of frames that the snapshot lasts. Defaults to 1 meaning a single frame
            fps: float
                Number of frames per second of the playout

        Returns:
            playout: dictionary generator
                The timestamp and mesh that the object has at the required frames, has two keys,
                timestamp: list (of one item for later compatibility) containing the timestamp of the frame
                polydata: the positions and orientations that the object has at the frame as the equivalent mesh
        """

        if no_frames == 1:
            pose = self._get_pose(start)
            yield {'timestamps': [start], 'polydata': self._get_polydata(pose)}
        else:
            f = 0 # Starting frame is at the starting time
            while f <= no_frames:  # Number of frames count starts at zero
                pose = self._get_pose(start + f/fps)
                yield {'timestamps': [start + f / fps], 'polydata': self._get_polydata(pose)}
                f += 1  # Next frame

    def _subplayout(self, start, no_frames=1, fps=1):
        """Calculates the precise poses with their corresponding meshes of a smaller chunk of the trajectory to be used
        by _playout_subplayouts_generator

        Since a Scenario object is defined as an object following a defined and discrete path, individual frames that
        are not covered directly by the trajectory data are interpolated.

        Args:
            start: float
                Starting time for the snapshot
            no_frames: int
                Total number of frames that the snapshot lasts. Defaults to 1 meaning a single frame
            fps: float
                Number of frames per second of the playout

        Returns:
            playout: polydata generator
                The positions and orientations that the object has at the frame as the equivalent mesh
        """

        if no_frames == 1:
            pose = self._get_pose(start)
            yield self._get_polydata(pose)
        else:
            f = 0 # Starting frame is at the starting time
            while f <= no_frames:  # Number of frames count starts at zero
                pose = self._get_pose(start + f/fps)
                yield self._get_polydata(pose)
                f += 1 # Next frame

    def _playout_subplayouts_generator(self, start, no_frames=1, fps=1, subframes=1):  # OLD, NOT USED
        """Returns the polydata generator to be used for playout_subplayouts

        Arguments:
            start: float
                Starting time for the snapshot
            no_frames: int
                Total number of frames that the snapshot lasts. Defaults to 1 meaning a single frame
            fps: float
                Number of frames per second of the playout
            subframes: int
                Used for mechanical spinning lidar, it defines the number of object state frames that are needed for a
                spinning lidar to capture a single frame. It is equal to the Sensor.hpoints_number

        Returns:
            subplayouts: generator object containing all the subplayouts
        """
        # OLD
        # NOT USED
        if no_frames == 1:
            yield self._subplayout(start=start, no_frames=subframes, fps=subframes * fps)
        else:
            f = 0  # Starting frame is at the starting time
            while f <= no_frames - 1:  # Number of frames count starts at zero
                yield self._subplayout(start=start + f / fps, no_frames=subframes, fps=subframes * fps)
                f += 1

    def playout_subplayouts(self, start, no_frames=1, fps=1, subframes=1):
        """Calculates the precise timestamps and poses with their corresponding meshes of a smaller chunk of the
        trajectory for analysis, including the poses between frames that are needed to simulate a mechanical lidar

        Since a Scenario object is defined as an object following a defined and discrete path, individual frames that
        are not covered directly by the trajectory data are interpolated.

        Arguments:
            start: float
                Starting time for the snapshot
            no_frames: int
                Total number of frames that the snapshot lasts. Defaults to 1 meaning a single frame
            fps: float
                Number of frames per second of the playout
            subframes: int
                Used for mechanical spinning lidar, it defines the number of object state frames that are needed for a
                spinning lidar to capture a single frame. It is equal to the Sensor.hpoints_number

        Returns:
            playout_subplayouts: dictionary generator
                The timestamps and meshes that the object has at the required frames, has two keys,
                timestamps: list containing the timestamps of every instant data was gathered
                polydatas: generator object containing all of the subplayouts
        """
        if no_frames == 1:
            timestamps = np.array([start + i / (fps * subframes) for i in range(int(subframes))])
            yield {
                'timestamps': timestamps,
                'polydatas': self._subplayout(start=start, no_frames=subframes, fps=subframes * fps)  # Generator object
            }
        else:
            f = 0  # Starting frame is at the starting time
            generator = self._playout_subplayouts_generator(start, no_frames, fps, subframes)
            while f <= no_frames-1:  # Number of frames count starts at zero
                timestamps = [start + f/fps + i/(fps*subframes) for i in range(int(subframes))]
                yield {
                    'timestamps': timestamps,
                    'polydatas': self._subplayout(start=start + f / fps, no_frames=subframes, fps=subframes * fps)
                }
                f += 1

