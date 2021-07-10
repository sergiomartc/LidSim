import vtk
import numpy as np
from math import sin, cos, tan, radians


class Sensor:

    def __init__(self, config):
        """Defines a simulated lidar sensor with certain parameters

        In its simples form, the lidar is defined by its technology, minimum and maximum range, angular resolution,
        angular and range accuracy, horizontal by vertical field of view and frame rate. All measurements are done in
        the lidar frame of reference, considering it as stationary. Additional specs will be added as the emulator
        evolves.

        Arguments:
            config: dictionary
                Defined as follows: {
                    'lidarKind': technology used, either 'flash' or 'scanning',
                    'lidarRange': {'minimum': _, 'maximum': _, 'standardDeviation': _},
                    'lidarAngular': {
                        'resolution': {'horizontal': _, 'vertical': _},
                        'accuracy': {'horizontal': _, 'vertical': _}
                        }
                    'lidarFOV': {'horizontal': _, 'vertical': _},
                    'lidarFPS': _
                }
                ***Refer to the documentation for more information on what these values mean

        Attributes:
            self.kind: str
                Defines the techology of the model, either flash or scanning (which is done horizontally)
            self.minrange: float
                Defines the minimum range of the sensor
            self.maxrange: float
                Defines the maximum range of the sensor
            self.hres: tuple
                Horizontal resolution and horizontal angular accuracy
            self.vres: tuple
                Vertical resolution and vertical angular accuracy
            self.rangeres: float
                Range standard deviation error
            self.fps: float
                Sensor's frequency as frames per second
            self.fov: tuple
                Sensor's field of view, horizontal by vertical
            self.hpoints_number: int
                Defines the number of points in the horizontal field as in the number of distinct laser beams that are
                sent in the horizontal direction
            self.vpoints_number: int
                Defines the number of points in the vertical field as in the number of distinct laser beams that are
                sent in the vertical direction
            self.ptargets: 2D numpy array
                The point targets for each laser beam
        """

        # >>>NOTE ON FUTURE IMPLEMENTATIONS AND REALISTIC SIMULATIONS
        """
        >range: To better simulate the physics of a lidar sensor, the range parameter may otherwise be defined as the
        minimum and maximum amount of light that the sensor is capable of detecting. This would more accurately
        represent the range limitations that appear in an actual sensor. For this to be implemented, the object file
        would need to have a reflectivity value for each of the triangles that make up the mesh. Then, the energy that
        comes back to the sensor could be simulated by a losses through the air, and loses that get absorbed by the
        object. This way the pointcloud data could also have information on the detected reflectivity of the object.
        Similarly, the standard deviation of the range could be defined as the sensitivity that this energy detection 
        sensor has. Either way, this implementation, although making the simulation more realistic, would not suppose a
        major difference in the values that are gathered, since the applications that this program is meant to 
        simulate would not realistically pose a challenge to any lidar in terms of the reflectivity of the objects of
        interest
        """

        # Loading the lidar parameters from the configuration file
        self.kind = config['lidarKind']
        self.minrange = config['lidarRange']['minimum']
        self.maxrange = config['lidarRange']['maximum']
        self.hres = config['lidarAngular']['resolution']['horizontal'], config['lidarAngular']['accuracy']['horizontal']
        self.vres = config['lidarAngular']['resolution']['vertical'], config['lidarAngular']['accuracy']['vertical']
        self.rangeres = config['lidarRange']['standardDeviation']
        self.fps = config['lidarFPS']

        # Because of the way the target points are defined, FOV implementation does not support angles greater than 180
        # degrees, and is hard capped at 175 for safety.
        if config['lidarFOV']['horizontal'] >= 175 or config['lidarFOV']['vertical'] >= 175:
            raise ValueError('The FOV you provided is not currently supported')
        else:
            self.fov = config['lidarFOV']['horizontal'], config['lidarFOV']['vertical']

        # Number of points in each axis towards which lasers are shone at
        self.hpoints_number = int(self.fov[0] / self.hres[0]) + 1
        self.vpoints_number = int(self.fov[1] / self.vres[0]) + 1

        # Point targets for each laser beam are defined by the sensor's fov, resolution and pose
        # First the target points are calculated for the lidar's frame of reference
        self.ptargets = np.empty((self.vpoints_number * self.hpoints_number, 3))
        self.ptargets[:] = np.NaN

        # Empty ptargets is filled with the coordinates corresponding to the points that are at the max range distance
        # in the lidar's x direction that each laser travels through. The x axis is defined as the direction the lidar
        # is oriented to.
        # Loop starts in the upper left corner of this 'screen'
        for j in range(self.hpoints_number):
            for i in range(self.vpoints_number):
                # Calculating the angles this point is at from the reference that is the x axis
                # Angle formed by the projection of the radial direction over the xy plane and the x axis
                theta = radians(-self.fov[0] / 2 + j * self.hres[0])
                # Angle formed by the radial direction and the xy plane
                phi = radians(self.fov[1] / 2 - i * self.vres[0])

                # >>>NOTE ON THE ALGORITHMIC APPROACH AND ITS SHORTCOMINGS:
                """The algorithm it uses to calculate target points consists of setting up an imaginary screen at 
                a distance of maxrange in the x direction from the sensor, and with sides such that the angle that
                they form with this x direction is equal to half of the field of view. A more comprehensive
                implementation would consists of setting up suck target points following spherical coordinates,
                such that a singularity would not be found at 180º of field of view. Essentially, this implementation
                is less computationally expensive because calculating these target points results in an easier task.
                It was decided to take this simpler approach because the applications that this simulator is intended
                to replicate often do not require more than 90 degrees in field of view. The target "screen" that these 
                target points define is tangent to the sphere that would be necessary to calculate these target points.
                After they have been defined, an error component is added, following a normal distribution, to the
                horizontal and vertical coordinates of these target points. What this component is trying to simulate is
                an error on the manufacturing side, meaning that a laser beam that is supposed to be aiming in a
                particular direction, is not doing so, and it deviates from this direction following a defined normal
                distribution. As such, these errors are only calculated once, when the lidar is defined, because they 
                aim to model the error sources that may arise during the manufacturing process."""

                # The sensor's angular inaccuracies are accounted for with a normal distribution
                self.ptargets[j * self.vpoints_number + i, :] = (
                    self.maxrange,
                    self.maxrange * tan(theta + radians(np.random.normal(0, self.hres[1]))),
                    self.maxrange * tan(phi + radians(np.random.normal(0, self.vres[1])))
                )

    def _send_laser(self, ptarget, obbTree):
        """Calculates the intersection point of the laser with the object

        The simulation only has support for single return lidar, so only the first intersection of a line with the
        surface of the object will be calculated.

        Arguments:
            ptarget: 3D numpy array
                Defines the point at which the laser beam originating from the lidar is pointing at.
            obbTree: OBBTree object
                Defines the OBB tree for intersection testing of this instant in time

        Returns:
            point: tuple
                The position at which the laser intersected the object. If no intersection is found, defaults to None.

        Future implementation:
        >max and min range based off light reflection principles
        >return reflectivity index of the shone surface
        >calculate only the first intersection for better optimization
        """

        # OOBTree object must be initialized beforehand to ease computation
        # Stores the intersection points
        intersection_points = vtk.vtkPoints()
        # We want information about the intersections point, not the cells
        code = obbTree.IntersectWithLine([0, 0, 0], ptarget, intersection_points, None)

        # The code value is 0 where no intersections are found, -1 if psource lies inside the closed surface, or +1 if
        # psource lies outside the closed surface
        if code == 0:
            return None
        else:
            points_data = intersection_points.GetData()
            # Returns the first intersection point, if any
            point = points_data.GetTuple3(0)
            return point

    def _flash_frame_capture(self, frame):
        """Scans the environment with a flash lidar for just one sweep

        Arguments:
            frame: vtkPolyData object
                Defines the position and orientation of the object of interest at a particular instant in time
        Returns:
            point_cloud: 2D np array that corresponds to a single frame scan of the simulated flash lidar
        """

        # Initializes the obbTree of the mesh for intersections
        obbTree = vtk.vtkOBBTree()
        obbTree.SetDataSet(frame)
        obbTree.BuildLocator()

        # Initializes empty array that will contain the point cloud
        point_cloud = np.empty((self.vpoints_number * self.hpoints_number, 3))
        point_cloud[:] = np.NaN

        for j in range(self.hpoints_number):
            for i in range(self.vpoints_number):
                # Calculates the intersection point, if any
                point = self._send_laser(self.ptargets[j * self.vpoints_number + i, :], obbTree)
                if point is not None:
                    # Calculates the distance said point is at
                    distance = np.linalg.norm(point)
                    if distance < self.minrange or point[0] < 0:
                        # Point falls outside the lidar's range
                        point_cloud[j * self.vpoints_number + i, :] = None
                    else:
                        # Point is inside lidar's range

                        # Angle formed by the projection of the radial direction over the xy plane and the x axis
                        theta = radians(-self.fov[0] / 2 + j * self.hres[0])
                        # Angle formed by the radial direction and the xy plane
                        phi = radians(self.fov[1] / 2 - i * self.vres[0])

                        # Range standard deviation error is added
                        # This standard deviation intends to simulate the errors that may arise from measuring the
                        # distance to an object, which is computed for every intersection, since every time a laser beam
                        # is sent the conditions are not the same, unlike the way that the angular error is defined.
                        point_cloud[j * self.vpoints_number + i, 0] = point[0] + \
                                                                      np.random.normal(0, self.rangeres) * cos(
                            phi) * cos(theta)
                        point_cloud[j * self.vpoints_number + i, 1] = point[1] + \
                                                                      np.random.normal(0, self.rangeres) * cos(
                            phi) * sin(theta)
                        point_cloud[j * self.vpoints_number + i, 2] = point[2] + \
                                                                      np.random.normal(0, self.rangeres) * sin(phi)

                else:
                    # No point is found
                    point_cloud[j * self.vpoints_number + i, :] = None

        return point_cloud

    def _scanning_frame_capture(self, subplayout, direction):
        """Scans the environment with a mechanical lidar for just one sweep

        Arguments:
            subplayout: Scenario.playout generator
                Generates the position and orientation of the object of interest at the precise instants in time that
                correspond to the simulated mechanical lidar scanning instants. The number of subplayouts must be equal
                to the number of horizontal points in the scan, hpoints_number
            direction: int (-1/1)
                Defines the direction in which the lidar is spinning, as one with <360º FOV has to spin back and forth

        Returns:
            point_cloud: 2D np array that corresponds to a single frame scan of the simulated flash lidar
        """

        # Initializes empty array that will contain the point cloud
        point_cloud = np.empty((self.vpoints_number * self.hpoints_number, 3))
        point_cloud[:] = np.NaN

        if direction == 1:  # Counterclockwise rotation when viewed from above
            jrange = reversed(range(self.hpoints_number))
        elif direction == -1:  # Clockwise rotation when viewed from above
            jrange = range(self.hpoints_number)

        for j in jrange:

            # Initializes the obbTree of the mesh for intersections
            obbTree = vtk.vtkOBBTree()
            obbTree.SetDataSet(next(subplayout))
            obbTree.BuildLocator()

            for i in range(self.vpoints_number):
                # Calculates the intersection point, if any
                point = self._send_laser(self.ptargets[j * self.vpoints_number + i, :], obbTree)
                if point is not None:
                    # Calculates the distance said point is at
                    distance = np.linalg.norm(point)
                    if distance < self.minrange or point[0] < 0:
                        # Point falls outside the lidar's range
                        point_cloud[j * self.vpoints_number + i, :] = None
                    else:
                        # Point is inside lidar's range

                        # Angle formed by the projection of the radial direction over the xy plane and the x axis
                        theta = radians(-self.fov[0] / 2 + j * self.hres[0])
                        # Angle formed by the radial direction and the xy plane
                        phi = radians(self.fov[1] / 2 - i * self.vres[0])

                        # Range standard deviation error is added
                        # This standard deviation intends to simulate the errors that may arise from measuring the
                        # distance to an object, which is computed for every intersection, since every time a laser beam
                        # is sent the conditions are not the same, unlike the way that the angular error is defined.
                        point_cloud[j * self.vpoints_number + i, 0] = point[0] + \
                                                                      np.random.normal(0, self.rangeres) * cos(
                            phi) * cos(theta)
                        point_cloud[j * self.vpoints_number + i, 1] = point[1] + \
                                                                      np.random.normal(0, self.rangeres) * cos(
                            phi) * sin(theta)
                        point_cloud[j * self.vpoints_number + i, 2] = point[2] + \
                                                                      np.random.normal(0, self.rangeres) * sin(phi)
                else:
                    # No point is found
                    point_cloud[j * self.vpoints_number + i, :] = None

        return point_cloud

    def _flash_record(self, playout):
        """Records its environment for the duration of a Scenario.playout generator with a flash lidar

        Arguments:
            playout: Scenario.playout generator
                Defines the object polydata at every frame that is to be scanned

        Returns:
            point_clouds: generator object
                Generates the simulated flash lidar point clouds for every frame in , where each point cloud is a 2D
                numpy array with the coordinates of the points the simulated lidar detected
        """

        for frame in playout:
            yield [frame['timestamps'], self._flash_frame_capture(frame['polydata'])]

    def _scanning_record(self, playout_subplayouts):
        """Scans its environment for the duration of a Scenario.playout generator with a mechanical lidar

        Arguments:
            playout_subplayouts: Scenario.playout_subplayouts generator generator
                Defines the generators that cointain the generators that define the polydata at every subframe that is
                to be scanned
        Returns:
            point_clouds: generator object
                Generates the simulated flash lidar point clouds for every frame in , where each point cloud is a 2D
                numpy array with the coordinates of the points the simulated lidar detected
        """

        # Need to account for the scanning rotation
        # Defines whether the scanning is done back and forth or constantly rotating
        if self.fov[0] == 360: # Not yet supported
            m = 1
        else:
            m = -1

        # Starting direction is always counterclockwise when viewed from above, from left to right
        direction = -1

        for subplayout in playout_subplayouts:
            yield [subplayout['timestamps'], self._scanning_frame_capture(subplayout['polydatas'], direction)]
            # Changes the direction every sweep
            direction *= m

    def record(self, playout):
        """Scans its environment for the duration of a Scenario.playout generator with the defined simulated lidar

        Arguments:
            playout: Scenario.playout generator
                Defines the object polydata at every frame that is to be scanned
        Returns:
            point_clouds: generator object
                Generates the simulated flash lidar point clouds for every frame in , where each point cloud is a 2D
                numpy array with the coordinates of the points the simulated lidar detected
        """

        if self.kind.lower() == 'flash':
            return self._flash_record(playout)
        elif self.kind.lower() == 'scanning':
            return self._scanning_record(playout)

