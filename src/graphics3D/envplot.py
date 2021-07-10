import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits import mplot3d


class PlotTrajectory():

    def __init__(self, params, view_params='default', plotting_playout=None,
                 pointclouds=None, real_trajectory=None, detected_trajectory=None):
        """Generates a matplotlib animation of all the provided data, with option to show it and store it

        Arguments:
            params: dictionary
                Dictionary of fixed parameters for plotting, determined by LidarRecorder
            view_params: dictionary
                Dictionary of viewing parameters, refer below or to the documentation for its default values and meaning
            plotting_playout: Scenario.plotting_playout generator
                Scenario that defines the trajectory of the object
            pointclouds: dictionary generator
                The simulated point clouds that the sensor captured
            trajectory: 2D numpy array
                Contains the position and orientation of the object for the corresponding time stamps
            detected_trajectory: 2D numpy array
                The poses of the object that the class was able to calculate from the simulated lidar data

        Attributes:
            self.params: dictionary
                Dictionary of fixed parameters for plotting, determined by LidarRecorder
            self.mesh_frames: list
                Contains each of the meshes from the plotting playout
            self.anim_frames: int
                Number of frames in the plotting playout
            self.pointclouds: list
                Contains the pointclouds that the lidar gathered
            self.real_trajectory: 2D numpy array
                The trajectory that the object followed
            self.detected_trajectory: 2D numpy array
                The trajectory that the lidar detected
            self.view_params: dictionary
                Dictionary of viewing paramters for plotting, refer below for its default values
        """

        # Attribute instantiation
        self.params = params
        self.mesh_frames = list(plotting_playout)
        self.anim_frames = len(self.mesh_frames)
        self.pointclouds = list(pointclouds)
        self.real_trajectory = real_trajectory
        self.detected_trajectory = detected_trajectory

        # If no viewing parameters are defined, defaults to these values
        if view_params is None:
            self.view_params = {
                'meshAlpha': 0.4,
                'pointcloudColor': 'r', 'pointcloudSize': 0.5, 'pointcloudAlpha': 1,
                'realColor': 'y', 'realMarker': 'o',
                'detectedColor': 'y', 'detectedMarker': 's'
            }
        else:
            self.view_params = view_params

    def _get_pose(self, trajectory, time):
        """Returns the object pose at a given time interpolating between instants in self.trajectory

        Arguments:
            trajectory: 2D numpy array
                Array containing the position and orientation at specifit time intervals
            time: float
                The precise time for the object pose measurement

        Returns:
            pose: np.array
                The pose of the object at that moment of time
        """

        trajectory = np.array(trajectory)

        def _find_nearest(array, value):
            # Finds the array indexes of the numbers that most closely bound the given value
            idx = (np.abs(array - value)).argmin()
            if array[idx] > value:
                return idx - 1, idx
            elif array[idx] < value:
                return idx, idx + 1

        if time in trajectory[:, 0]:
            # If the time specified is in the array, get the corresponding index and the pose to return
            idx = np.where(trajectory[:, 0] == time)[0][0]
            pose = trajectory[idx, 1:]

        elif time > trajectory[-1, 0]:
            # The time specified falls outside the timestamps of the array
            # Use the last available timestamp
            pose = trajectory[-1, 1:]

        else:
            # The time specified is not in the array, but it can be interpolated
            # These indexes bound the timestamp inside the array
            indexes = _find_nearest(trajectory[:, 0], time)
            # If the first index equates to the length of the trajectory array, that timestamp is not covered in the
            # trajectory
            if indexes[0] == len(trajectory[:, 0]):
                # The time given falls outside the given trajectory
                raise IndexError("Time value given falls outside the defined trajectory")
            else:
                # Degrees of freedom that are described in the trajectory
                dof = len(trajectory[0, 1:])
                pose = np.empty(dof)
                for i in range(dof):  # For every coordinate the interpolation is performed
                    dt = (trajectory[indexes[1], 0] - trajectory[indexes[0], 0])
                    # i+1 is used instead of i because the first element of the self.trajectory array is the timestamp
                    slope = (trajectory[indexes[1], i + 1] - trajectory[indexes[0], i + 1]) / dt
                    pose[i] = trajectory[indexes[0], i + 1] + (time - trajectory[indexes[0], 0]) * slope

        return pose

    def _update_mesh(self, anim_frame, mesh_plot):
        """Updates the triangulated mesh position for a given timestamp

        Arguments:
            anim_frame: int
                Current frame that the animation is in
            mesh_plot: numpy-tsl mesh object
                Triangulated mesh that is being plotted

        Returns:
            mesh_plot: numpy-tsl mesh object
                This value gets returned to ensure proper plotting functioning
        """

        # Because everything is a list, the plotting will loop itself once it hits the beginning again
        frame_mesh = self.mesh_frames[anim_frame]
        # Gets the array of vectors of the next triangulated mesh frame
        _vec = np.array([
            [item for tpl in frame_mesh.x for item in tpl],
            [item for tpl in frame_mesh.y for item in tpl],
            [item for tpl in frame_mesh.z for item in tpl]
        ])
        # Updates the vectors of the triangulated mesh that is being plotted each frame
        mesh_plot._vec[0][:] = _vec[0][:]
        mesh_plot._vec[1][:] = _vec[1][:]
        mesh_plot._vec[2][:] = _vec[2][:]
        return mesh_plot

    def _update_pointcloud(self, anim_frame, pointcloud_plot):
        """Updates the pointcloud for a given timestamp

        Arguments:
            anim_frame: int
                Current frame that the animation is in
            pointcloud_plot: 2D numpy array
                Pointcloud that is being plotted

        Returns:
            pointcloud_plot: 2D numpy array
                This value gets returned to ensure proper plotting functioning
        """

        # The animation frame does not correspond with the lidar frame, nor do their refresh rates
        time_elapsed = anim_frame / self.params['FPS']
        # This current time elapsed corresponds to the lidar frame of number
        lidar_frame = int(self.params['lidarFPS'] * time_elapsed)

        # Getting the pointcloud corresponding to this current lidar frame
        if lidar_frame == len(self.pointclouds):
            # If the time instant corresponds to a lidar frame that is not included in the data, because of it being
            # greater, it uses the last lidar frame available
            pointcloud_current = self.pointclouds[-1]
        else:
            # Otherwise it gets the corresponding pointcloud
            pointcloud_current = self.pointclouds[lidar_frame]

        # Number of steps the lidar takes to do one sweep is the length of the timestamp array in the pointcloud
        steps = len(pointcloud_current[0])

        if steps == 1:
            # If there is only one timestamp, and thus only one step, it is a flash lidar
            # The pointcloud corresponding to this last lidar frame gets plotted altogether, by defining the coordinate
            # values for every point first
            x = pointcloud_current[1][:, 0]
            y = pointcloud_current[1][:, 1]
            z = pointcloud_current[1][:, 2]

            # And then updating the plot, changing the pointcloud data
            pointcloud_plot.set_data(x, y)
            pointcloud_plot.set_3d_properties(z)

        elif steps > 1 and anim_frame == 0:
            # If there are more than one timestamp, and thus more than one step, it is a scanning lidar, in which case
            # the points that were gathered during the last lidar frame duration must be plotted
            # Since the animation frame in this case is also zero, the only points that will be plotted will be the
            # initial column of the lidar sweep, meaning the very first points it gathers in the first step
            points_per_step = int(len(pointcloud_current[1][:, 0]) / steps)
            x = pointcloud_current[1][:points_per_step, 0]
            y = pointcloud_current[1][:points_per_step, 1]
            z = pointcloud_current[1][:points_per_step, 2]

            # Updating the plot and changing the pointcloud data
            pointcloud_plot.set_data(x, y)
            pointcloud_plot.set_3d_properties(z)

        else:
            # It is a scanning lidar, in which case the points that were gathered during the last lidar frame duration
            # must be plotted, which will give it the "scanning" effect when viewing
            def _find_nearest(array, value):
                """Finds the index for the value in the array that is below and closest to the inputted value"""
                idx = (np.abs(array - value)).argmin()
                if array[idx] > value:
                    return idx - 1
                elif array[idx] <= value:
                    return idx

            # Number of horizontal points, total steps, or subframes of the scanning lidar
            subframes = self.params['lidarSubframes']
            # Number of vertical points
            vpoints_number = self.params['lidarVPoints']

            # Takes these many horizontal steps from the current lidar frame
            steps_current = _find_nearest(np.array(pointcloud_current[0]), time_elapsed) + 1

            # Pointclouds from the last frame
            if lidar_frame == 0:
                # If the lidar frame is the first one, this means that there is no such thing as a pointcloud from the
                # last frame, so it is defined as an empty pointcloud
                empty = np.empty((int(vpoints_number * subframes), 3))
                empty[:] = np.NaN
                pointcloud_last = [None, empty]
            else:
                # After the first frame, the previous pointcloud is defined
                pointcloud_last = self.pointclouds[lidar_frame - 1]

            # Number of points taken from the current frame
            points_current = steps_current * vpoints_number

            # In a scanning lidar, each frame consists of consecutive sweeps, first from left to right, then from right
            # to left, and so on. The way a pointcloud matrix is defined in the Sensor class is such that every item in
            # the array is representative of a fixed direction in space, so there is a need to account for the sweeping
            # that goes on when plotting, the same way that the sweeping is incorporated into the way this matrix has
            # its values defined. Each frame sweep direction can then be gathered from the frame number it is at,
            # starting from left to right, and so it follows:

            if lidar_frame % 2 == 0:
                # These are the left-to-right-sweep lidar frames, as the first one, which would be equivalent to the
                # odd number of lidar frames (bear in mind that the count for lidar_frame starts at zero)
                # In short, it takes the number of steps from the last lidar frame from right to left, and the number of
                # steps from the current lidar frame from left to right.
                x = np.copy(pointcloud_last[1][:,0])
                x[:points_current] = pointcloud_current[1][:points_current, 0]

                y = np.copy(pointcloud_last[1][:, 1])
                y[:points_current] = pointcloud_current[1][:points_current, 1]

                z = np.copy(pointcloud_last[1][:, 2])
                z[:points_current] = pointcloud_current[1][:points_current, 2]

            elif lidar_frame % 2 == 1:
                # These are the right-to-left-sweep lidar frames, that will always come after a left-to-right-sweep.
                # In short, it takes the number of steps from the last lidar frame from left to right, and the number of
                # steps from the current lidar frame from right to left.
                x = np.copy(pointcloud_last[1][:,0])
                x[-points_current:] = pointcloud_current[1][-points_current:, 0]

                y = np.copy(pointcloud_last[1][:,1])
                y[-points_current:] = pointcloud_current[1][-points_current:, 1]

                z = np.copy(pointcloud_last[1][:,2])
                z[-points_current:] = pointcloud_current[1][-points_current:, 2]

            # Updating the plot and changing the pointcloud data
            pointcloud_plot.set_data(x, y)
            pointcloud_plot.set_3d_properties(z)

        return pointcloud_plot

    def _update_real_trajectory(self, anim_frame, real_trajectory_plot):
        """Updates the real trajectory plot for a given timestamp

        Arguments:
            anim_frame: int
                Current frame that the animation is in
            real_trajectory_plot: numpy array
                Point corresponding to the real trajectory that is being plotted

        Returns:
            real_trajectory_plot: numpy array
                This value gets returned to ensure proper plotting functioning
        """

        # Gets the current pose interpolating between trajectory timestamp data
        time_elapsed = anim_frame / self.params['FPS']
        pose = self._get_pose(self.real_trajectory, time_elapsed)

        # Updating the plot and changing the trajectory data
        real_trajectory_plot.set_data(pose[0], pose[1])
        real_trajectory_plot.set_3d_properties(pose[2])

        return real_trajectory_plot

    def _update_detected_trajectory(self, anim_frame, detected_trajectory_plot, interpolate=True):
        """Updates the detected trajectory plot for a given timestamp

        Arguments:
            anim_frame: int
                Current frame that the animation is in
            detected_trajectory_plot: numpy array
                Point corresponding to the detected trajectory that is being plotted
            interpolate: bool
                Whether or not to interpolate between values or only show the calculated data

        Returns:
            detected_trajectory_plot: numpy array
                This value gets returned to ensure proper plotting functioning
        """

        def _find_nearest(array, value):
            """Finds the index for the value in the array that is below and closest to the inputted value"""
            idx = (np.abs(array - value)).argmin()
            if array[idx] > value:
                return idx - 1
            elif array[idx] <= value:
                return idx

        # Time that has elapsed since the animation began
        time_elapsed = anim_frame / self.params['FPS']

        if self.detected_trajectory[0][0] > time_elapsed:
            # The first detected trajectory timestamp is greater than the time that has elapsed, which means that it is
            # a scanning lidar, and the point has not yet been computed: uses the future one
            detected_trajectory_plot.set_data(self.detected_trajectory[0][1], self.detected_trajectory[0][2])
            detected_trajectory_plot.set_3d_properties(self.detected_trajectory[0][3])
        else:
            if interpolate:
                # The timestamp is already in detected_trajectory or can be interpolated from it
                pose = self._get_pose(self.detected_trajectory, time_elapsed)

                # Updating the plot and changing the detected trajectory data
                detected_trajectory_plot.set_data(pose[0], pose[1])
                detected_trajectory_plot.set_3d_properties(pose[2])
            else:
                # Since these values will not be interpolated, it only plots the most recent one
                idx = _find_nearest(np.array(self.detected_trajectory), time_elapsed)
                pose = self.detected_trajectory[idx]

                # Updating the plot and changing the detected trajectory data
                detected_trajectory_plot.set_data(pose[0], pose[1])
                detected_trajectory_plot.set_3d_properties(pose[2])

        return detected_trajectory_plot

    def plot(self):
        """Generates the animation of all of the given information together"""

        # These two following funcitons taken from: https://stackoverflow.com/q/13685386
        # Matplotlib default parameters make it so 3d plotting often entails a distorted view. These two functions are
        # then used to stablish the same scale in all axes so that the view of the objects is not distorted but rather
        # represents more faithfully its dimensions.
        def set_axes_equal(ax: plt.Axes):
            """Set 3D plot axes to equal scale.

            Make axes of 3D plot have equal scale so that spheres appear as
            spheres and cubes as cubes.  Required since `ax.axis('equal')`
            and `ax.set_aspect('equal')` don't work on 3D.
            """
            limits = np.array([
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ])
            origin = np.mean(limits, axis=1)
            radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
            _set_axes_radius(ax, origin, radius)

        def _set_axes_radius(ax, origin, radius):
            x, y, z = origin
            ax.set_xlim3d([x - radius, x + radius])
            ax.set_ylim3d([y + radius, y - radius])
            ax.set_zlim3d([z - radius, z + radius])

        def update_graph(anim_frame):
            """Updates the graph for a given animation frame"""

            # The timestamp is calculated using the current animation frame and the animation FPS
            timestamp = anim_frame / self.params['FPS']
            time_elapsed = anim_frame / self.params['FPS']
            lidar_frame = int(self.params['lidarFPS'] * time_elapsed)
            title.set_text('Time: %.2fs' % timestamp)

            # Plots each of the four different items, if they have been requested
            # For now the GUI does not support this "picky" plotting, but it is included in the underlying code
            if self.mesh_frames is not None:
                self._update_mesh(anim_frame, mesh_plot)
            if self.pointclouds is not None:
                self._update_pointcloud(anim_frame, pointcloud_plot)
            if self.real_trajectory is not None:
                self._update_real_trajectory(anim_frame, real_trajectory_plot)
            if self.detected_trajectory is not None:
                self._update_detected_trajectory(anim_frame, detected_trajectory_plot,
                                                 self.params['interpolateDetected'])

            return title, mesh_plot, pointcloud_plot, real_trajectory_plot, detected_trajectory_plot,

        # Plotting parameters definition
        fig = plt.figure()
        ax = mplot3d.Axes3D(fig, auto_add_to_figure=False)
        title = ax.text2D(0.05, 0.95, "Time: 0.00s", transform=ax.transAxes)
        fig.add_axes(ax)

        # >>>THE FRAME 0 PLOT STATE IS DEFINED OUTSIDE THE UPDATE GRAPH FUNCTION
        # Triangulated mesh
        if self.mesh_frames is not None:
            # If requested, plots the first one in the list
            your_mesh = self.mesh_frames[0]
            mesh_plot = ax.add_collection3d(mplot3d.art3d.Poly3DCollection(
                your_mesh.vectors, alpha=self.view_params['meshAlpha'], linewidths=1
            ))
        else:
            mesh_plot = ax.add_collection3d(mplot3d.art3d.Poly3DCollection([[[0, 0, 0]]]))

        # Simulated point cloud
        if self.pointclouds is not None:
            # If requested, plots the first step of the pointcloud data
            pointcloud = self.pointclouds[0]
            steps = len(pointcloud[0])
            points_per_step = int(len(pointcloud[1]) / steps)
            x = pointcloud[1][:points_per_step, 0]
            y = pointcloud[1][:points_per_step, 1]
            z = pointcloud[1][:points_per_step, 2]
            pointcloud_plot, = ax.plot(
                x, y, z, color=self.view_params['pointcloudColor'], ms=self.view_params['pointcloudSize'],
                alpha=self.view_params['pointcloudAlpha'], linestyle="", marker="o"
            )
        else:
            pointcloud_plot, = ax.plot(0, 0, 0, alpha=0)

        # Real object trajectory
        if self.real_trajectory is not None:
            # If requested, plots the first value for the trajectory
            real_trajectory_plot, = ax.plot(
                self.real_trajectory[0][1], self.real_trajectory[0][2], self.real_trajectory[0][3],
                color=self.view_params['realColor'], linestyle="", marker=self.view_params['realMarker']
            )
        else:
            real_trajectory_plot, = ax.plot(0, 0, 0, alpha=0)

        # Detected object trajectory
        if self.detected_trajectory is not None:
            # If requested, plots the first value for the deteceted trajectory
            detected_trajectory_plot, = ax.plot(
                self.detected_trajectory[0][1], self.detected_trajectory[0][2], self.detected_trajectory[0][3],
                color=self.view_params['detectedColor'], linestyle="", marker=self.view_params['detectedMarker']
            )
        else:
            detected_trajectory_plot, = ax.plot(0, 0, 0, alpha=0)

        # Plotting the lidar as a black cube if requested
        if self.params['showLidar']:
            ax.plot(0, 0, 0, color='k', marker='s', alpha=0.7)

        # Setting box aspect on axes
        ax.set_box_aspect([1, 1, 1])

        # Changes the starting camera position
        ax.view_init(elev=self.params['elevation'], azim=self.params['azimuth'])

        # Setting the axes limits
        ax.axes.set_xlim3d(left=self.params['xmin'], right=self.params['xmax'])
        # Getting the y-axis go incrementally from left to right, as is defined in the code
        ax.axes.set_ylim3d(bottom=self.params['ymin'], top=self.params['ymax'])
        ax.axes.set_zlim3d(bottom=self.params['zmin'], top=self.params['zmax'])

        # Setting equal aspect on axes
        set_axes_equal(ax)

        # Hiding axis if requested
        if self.params['hideAxis']:
            ax.set_axis_off()

        # Defining the animation
        ani = animation.FuncAnimation(fig, update_graph, self.anim_frames, interval=1 / self.params['FPS'])

        return ani

    def show(self):
        """Shows the animation as a matplotlib interactive animation"""

        plt.show()


    def save(self):
        """Saves the generated animation under a specified file"""

        # If the savefile is not defined, generates one with a timestamp
        if self.savefile is None or self.savefile == "":
            self.savefile = 'vid_' + str(int(time.time())) + '.mp4'

        # If the savefile is not in mp4 format, reformats it
        if self.savefile.endswith('.mp4') is not True:
            self.savefile += '.mp4'

        # Saving the file
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='LidSim 1.0'), bitrate=1800)
        ani.save(self.savefile, writer=writer)