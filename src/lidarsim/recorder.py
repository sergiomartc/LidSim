import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import lidarsim.models as md
import src.lidarsim.environment as environment
from numpy import loadtxt
from numpy import load
from matplotlib import pyplot as plt
from matplotlib import animation
from graphics3D.envplot import PlotTrajectory


class LidarRecorder:

    def __init__(self, lidar):
        """Produces the simulated 3D point cloud of the environment, leveraging lidar.py and environment.py

        Once a lidar is defined, the virtual sensor is used to scan over a scenario and simulate the 3D point cloud
        that it would see. This environment is defined by the object of interest and the trajectory it follows. The
        reference point for this simulation, ie, the source of the laser beams, is situated at the lidar.

        Arguments:
            lidar: Sensor object
                Defines the lidar sensor that is used to scan the scenario

        Attributes:
            self.sensor: lidar.Sensor object
                Stores the lidar sensor that is used to scan the scenario
            self.kind: str
                The kind of lidar sensor that is used, either 'flash' or 'scanning'
            self.fps: float
                Defines the frame rate of the sensor, the maximum amount of point clouds it is able to create per
                second.
            self.subframes: int
                Number of separate frames that are needed to simulate a lidar sweep
            self.trajectory_model: models.py model
                ML model used to infer the pose of the object from the point cloud, defaults to md.Centroid, that
                gets the trajectory by taking the average point. Also supports md.SphereRegression to fit the points to
                a sphere.
            self.ani: matplotlib animation
                Contains the animation
            self.scenario: environment.Scenario object
                The scenario object that is loaded for interpretation, comprising an .stl file and a trajectory
            self.scenario_duration: float
                The duration of the scenario, as the total duration of the trajectory information
            self.playout: environment.Scenario.playout or environment.Scenario.playout_subplayout attribute
                Smaller segment from the self.scenario for more precise analysis. If no playout is defined, the program
                will default to using the whole of the scenario
            self.pointclouds: list of pointclouds
                The simulated pointclouds that the sensor captured from the last playout it was fed
            self.trajectory: 2D numpy array
                Contains the position and orientation of the object for corresponding time stamps
            self.data: dictionary
                Contains the timestamps for every detected pose, these detected poses and the real poses at those
                timestamps
            """

        # Defines the class attributes about the lidar
        self.sensor = lidar
        self.kind = lidar.kind
        self.fps = lidar.fps

        # Used for scanning lidar
        self.subframes = lidar.hpoints_number

        # Default model for calculating the trajectory, uses the centroid
        self.trajectory_model = md.Centroid()
        #self.trajectory_model = md.SphereRegression() # Fitting to a sphere by least squares

        # Initializes those attributes that are instantiated later
        self.ani = None
        self.scenario = None
        self.scenario_duration = None
        self.playout = None
        self.pointclouds = None
        self.trajectory = None
        self.data = None

    def load_scenario(self, stl_file, trajectory_file):
        """Loads a Scenario from an .stl file centered around the origin and a trajectory and stores it as internal

        Arguments:
            stl_file: str
                Location of the .stl file to create a Scenario with
            trajectory_file: str
                Location of the file that contains the trajectory information, supports .txt and .csv (values must be
                separated by commas), or .npy

        Initializes:
            self.scenario: envirnonment.Scenario object
                The scenario object that is loaded for interpretation, comprising an .stl file and a trajectory
            self.playout: environment.Scenario.playout or environment.Scenario.playout_subplayout attribute
                Smaller segment from the self.scenario for more precise analysis, sets it as None
            self.pointcloud: list of pointclouds
                The simulated pointcloud that the sensor captured from the last playout it was fed
        """

        # The trajectory must be a 2D numpy array comprising 7 columns
        if trajectory_file.endswith('.txt') or trajectory_file.endswith('.csv'):
            self.trajectory = loadtxt(trajectory_file, delimiter=',')
        elif trajectory_file.endswith('.npy'):
            self.trajectory = load(trajectory_file)
        else:
            raise NameError('The file that was provided is not supported')

        self.scenario = environment.Scenario(stl_file, self.trajectory)
        self.scenario_duration = self.trajectory[-1, 0]

        # Every time a new scenario is initialized, the playout attribute is reset to None, as well as the point cloud
        # and the detected trajectory data
        self.playout = None
        self.pointclouds = None

    def set_playout(self, start=None, duration=None):
        """Defines a smaller time segment from the self.scenario as a Scenario.playout

        If no playout gets defined, the program will use the entire playout

        Arguments:
            start: float
                Time at which the playout starts
            duration: float
                Total duration of the playout to be set

        Initializes:
            self.playout: environment.Scenario.playout or environment.Scenario.playout_subplayout attribute
                Smaller segment from the self.scenario for more precise analysis.
        """

        # If the scenario is not set beforehand, raises an error
        if self.scenario is None:
            raise ValueError('The Scenario object has not been defined yet')

        if start is None and duration is None:
            # if no start and duration arguments are provided, the program will use the entire playout
            if self.kind == 'flash':  # flash lidar uses playout
                self.playout = list(self.scenario.playout(
                    start=0, no_frames=self.fps * self.scenario_duration, fps=self.fps
                ))
            elif self.kind == 'scanning':  # scanning lidar uses playout_subplayout
                self.playout = list(self.scenario.playout_subplayouts(
                    start=0, no_frames=self.fps * self.scenario_duration, fps=self.fps, subframes=self.subframes
                ))

        elif start is not None and duration is not None:
            # start and duration arguments are provided
            if duration == 0:
                no_frames = 1  # duration being zero is equivalent to a single frame
            else:
                no_frames = duration * fps
            if self.kind == 'flash':  # flash lidar uses playout
                self.playout = list(self.scenario.playout(
                    start=start, no_frames=no_frames, fps=self.fps
                ))
            if self.kind == 'scanning':  # scanning lidar uses playout_subplayout
                self.playout = list(self.scenario.playout_subplayouts(
                    start=start, no_frames=no_frames, fps=self.fps, subframes=self.subframes
                ))

        else:
            # if only one of the arguments is given it raises an error
            raise TypeError('Both arguments must be provided together or not provided at all')

    def record(self):
        """Records the instantiated playout, using the whole scenario if no playout was defined

        Initializes:
            self.pointcloud: list of pointclouds
                The simulated pointcloud that the sensor captured from the last playout it was fed
        """

        if self.scenario is None:
            raise ValueError('The Scenario object has not been defined yet')

        # If no playout is defined beforehand, it will use the whole scenario
        if self.playout is None:
            self.set_playout()

        self.pointclouds = list(self.sensor.record(self.playout))

    def save_pointcloud(self, file_path):
        """Saves the self.pointcloud into a file after extracting the data

        The self.pointcloud is defined as a list of pointclouds, each containing information about the timestamps
        the data was gathered and the point cloud data itself. This function then unpacks this list and extracts
        these pointclouds and pickles them.

        Arguments:
            file_path: str
                Name of the filepath where the data will be stored. Supports .npy or .txt
        """

        if not file_path.endswith('.pkl'):
            raise FileNotFoundError('The format you provided is not supported by the program')
        else:
            data = []
            for pointcloud in self.pointclouds:
                data.append(pointcloud)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

    def load_pointcloud(self, file_path):
        """Saves the self.pointcloud into a file after extracting the data

        The self.pointcloud is defined as a list of pointclouds, each containing information about the timestamps
        the data was gathered and the point cloud data itself. This function then loads the pickled list into memory

        Arguments:
            file_path: str
                Name of the filepath where the data will be stored. Supports .npy or .txt

        Initializes:
            self.pointclouds: list of pointclouds
                The simulated pointclouds that the sensor captured from the last playout it was fed
        """

        if not file_path.endswith('.pkl'):
            raise FileNotFoundError('The format you provided is not supported by the program')
        else:
            with open(file_path, 'rb') as f:
                self.pointclouds = pickle.load(f)

    def load_models(self, model):
        """Loads the ML model to track the object in the frame

        Arguments:
            model: ML model object
                As the ones defined in models.py, they must have a fit() method that returns the detected pose

        Initializes:
            self.trajectory_model: models.py model
                ML model used to infer the pose of the object from the point cloud, defaults to md.Centroid, that
                gets the trajectory by taking the average point. Also supports md.SphereRegression to fit the points to
                a sphere.
        """

        self.trajectory_model = model

    def get_trajectory(self):
        """Gets the trajectory, as the pose the object has for each timeframe, from the simulated point cloud

        If no ML model is set beforehand on self.trajectory_model, it will simply calculate the average point from
        the point cloud data set, the centroid, without accounting for rotation.

        Initializes:
            self.data: dictionary
                Contains the timestamps for every detected pose, these detected poses and the real poses at those
                timestamps
        """

        # A point cloud must exist
        if self.pointclouds is None:
            raise ValueError('There is no point cloud to get the trajectory from')

        # Initializes the variables to overwrite any previous definitions
        detected_timestamps = []
        detected_trajectory = []
        detected_trajectory_errors = []
        real_distances = []
        real_poses = []

        # Loops through every point cloud
        for i in range(len(self.pointclouds)):

            # Solid state lidar, calculates the detected point by using all the gathered data in a frame
            if len(self.pointclouds[i][0]) == 1:

                # Detected trajectory timestamp, obtained from the simulated point cloud
                detected_timestamp = self.pointclouds[i][0][0]
                detected_timestamps.append(detected_timestamp)

                # Actual pose of the object, from interpolating for this timestamp
                actual_pose = self.scenario._get_pose(time=detected_timestamp)
                actual_distance = np.linalg.norm(actual_pose[:3])
                real_distances.append(actual_distance)
                real_poses.append(actual_pose)

                # Fits the model with the pointcloud data to get the detected pose
                detected_pose = self.trajectory_model.fit(self.pointclouds[i][1])
                detected_point = detected_pose[:3]
                # Adds the distance to the origin and the detected pose to the detected_trajectory list
                detected_trajectory.append(np.append(np.linalg.norm(detected_point), detected_pose))

                # Errors, from the difference between the detected trajectory and the real one
                detected_trajectory_errors.append([
                    np.linalg.norm(detected_point - actual_pose[:3]),
                    abs(detected_pose[0] - actual_pose[0]),
                    abs(detected_pose[1] - actual_pose[1]),
                    abs(detected_pose[2] - actual_pose[2]),
                    abs(detected_pose[3] - actual_pose[3]),
                    abs(detected_pose[4] - actual_pose[4]),
                    abs(detected_pose[5] - actual_pose[5])
                ])

            else:
                # There are several timestamps that correspond to each of the vertical steps of a scanning lidar
                # Instead of using the whole frame to calculate the detected pose, it uses the data that was gathered
                # during the last lidar frame duration to compute a new pose, so that the number of poses that are
                # calculated during each lidar frame is equal to the number of vertical steps the sweep has

                # Vertical points in each lidar sweep
                vpoints_number = self.sensor.vpoints_number

                # Loops through every vertical sweep
                for j in range(len(self.pointclouds[i][0])):

                    # Number of vertical steps the lidar has done so far in this lidar frame
                    steps_current = j + 1
                    # Current pointcloud
                    pointcloud_current = self.pointclouds[i][1]

                    # Detected trajectory current timestamp, obtained from the simulated point cloud
                    detected_timestamp = self.pointclouds[i][0][j]
                    detected_timestamps.append(detected_timestamp)

                    # Actual pose of the object, from interpolating for this timestamp
                    actual_pose = self.scenario._get_pose(time=detected_timestamp)
                    actual_distance = np.linalg.norm(actual_pose[:3])
                    real_distances.append(actual_distance)
                    real_poses.append(actual_pose)

                    if i == 0:
                        # If it's the first frame, there exists no previous pointcloud, and so it uses an empty array
                        pointcloud_last = np.empty((int(self.sensor.vpoints_number * self.subframes), 3))
                        pointcloud_last[:] = np.NaN
                    else:
                        # The previous pointcloud is defined
                        pointcloud_last = self.pointclouds[i-1][1]

                    # Number of points taken from the current frame
                    points_current = steps_current * vpoints_number

                    # In a scanning lidar, each frame consists of consecutive sweeps, first from left to right, then
                    # from right to left, and so on. The way a pointcloud matrix is defined in the Sensor class is such
                    # that every item in the array is representative of a fixed direction in space, so there is a need
                    # to account for the sweeping that goes on when plotting, the same way that the sweeping is
                    # incorporated into the way this matrix has its values defined. Each frame sweep direction can then
                    # be gathered from the frame number it is at, starting from left to right, and so it follows:

                    if i % 2 == 0:

                        # These are the left-to-right-sweep lidar frames, as the first one, which would be equivalent to
                        # the odd number of lidar frames (bear in mind that the count for lidar_frame starts at zero)
                        # In short, it takes the number of steps from the last lidar frame from right to left, and the
                        # number of steps from the current lidar frame from left to right.

                        x = np.copy(pointcloud_last[:, 0])
                        x[:points_current] = np.copy(pointcloud_current)[:points_current, 0]

                        y = np.copy(pointcloud_last[:, 1])
                        y[:points_current] = np.copy(pointcloud_current)[:points_current, 1]

                        z = np.copy(pointcloud_last[:, 2])
                        z[:points_current] = np.copy(pointcloud_current)[:points_current, 2]

                        # Pseudo pointcloud contains all the points that were gathered during a lidar frame time
                        pseudo_pointcloud = np.column_stack((x,y,z))

                    elif i % 2 == 1:

                        # These are the right-to-left-sweep lidar frames, that will always come after a
                        # left-to-right-sweep. In short, it takes the number of steps from the last lidar frame from
                        # left to right, and the number of steps from the current lidar frame from right to left.

                        x = np.copy(pointcloud_last[:, 0])
                        x[-points_current:] = np.copy(pointcloud_current)[-points_current:, 0]

                        y = np.copy(pointcloud_last[:, 1])
                        y[-points_current:] = np.copy(pointcloud_current)[-points_current:, 1]

                        z = np.copy(pointcloud_last[:, 2])
                        z[-points_current:] = np.copy(pointcloud_current)[-points_current:, 2]

                        # Pseudo pointcloud contains all the points that were gathered during a lidar frame time
                        pseudo_pointcloud = np.column_stack((x,y,z))

                    # With the pseudo pointcloud defined, it can now calculate the detected trajectory and its errors
                    # Fits the model with the pointcloud data to get the detected pose
                    detected_pose = self.trajectory_model.fit(pseudo_pointcloud)
                    detected_point = detected_pose[:3]
                    # Adds the distance to the origin and the detected pose to the detected_trajectory list
                    detected_trajectory.append(np.append(np.linalg.norm(detected_point), detected_pose))

                    # Errors, from the difference between the detected trajectory and the real one
                    detected_trajectory_errors.append([
                        np.linalg.norm(detected_point - actual_pose[:3]),
                        abs(detected_pose[0] - actual_pose[0]),
                        abs(detected_pose[1] - actual_pose[1]),
                        abs(detected_pose[2] - actual_pose[2]),
                        abs(detected_pose[3] - actual_pose[3]),
                        abs(detected_pose[4] - actual_pose[4]),
                        abs(detected_pose[5] - actual_pose[5])
                    ])

        # A pandas dataframe is created with the real and detected values and stored as a class attribute
        poses = np.array(real_poses)
        detected = np.array(detected_trajectory)
        errors = np.array(detected_trajectory_errors)
        self.data = pd.DataFrame({
            'timestamp': detected_timestamps,
            'distance': real_distances,
            'x': poses[:, 0],
            'y': poses[:, 1],
            'z': poses[:, 2],
            'psi': poses[:, 3],
            'theta': poses[:, 4],
            'phi': poses[:, 5],
            'detected_distance': detected[:, 0],
            'detected_x': detected[:, 1],
            'detected_y': detected[:, 2],
            'detected_z': detected[:, 3],
            'detected_psi': detected[:, 4],
            'detected_theta': detected[:, 5],
            'detected_phi': detected[:, 6],
            'error_distance': errors[:, 0],
            'error_x': errors[:, 1],
            'error_y': errors[:, 2],
            'error_z': errors[:, 3],
            'error_psi': errors[:, 4],
            'error_theta': errors[:, 5],
            'error_phi': errors[:, 6],
        })

    def save_trajectory_data(self, savefile):
        """Saves the data generated by get_trajectory into a csv file

        Arguments:
            savefile: string
                Name of the file to be saved, supports only csv format
        """

        if savefile.endswith('.csv') is not True:
            savefile += '.csv'

        self.data.to_csv(savefile, index=False)

    def launch_chart(self, variables, ylabel, title):
        """Launches a chart for the requested variable and resets the plotting functionality afterwords

        Arguments:
            variables: list
                Strings of the variables to plot
            ylabel: string
                Name to give the yaxis
            title: string
                Name to give to the chart
        """

        # Seaborn plotting parameters
        plt.figure()
        sns.set_style('dark')
        sns.despine()
        # Plots each of the specified variables
        for variable in variables:
            sns.lineplot(data=self.data, x='timestamp', y=variable)
        # Plotting parameters
        plt.legend(variables)
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def real_detected_charts(self):
        """Plots the charts for the real values and the detected ones"""

        # >>>Does not support orientation yet since the current implementation is unable to calculate it
        # Distance chart
        self.launch_chart(['detected_distance', 'distance'], 'Distance to the sensor (m)', 'Detected and real distance')
        # x coordinate chart
        self.launch_chart(['detected_x', 'x'], 'x coordinate (m)', 'Detected and real x coordinate')
        # y coordinate chart
        self.launch_chart(['detected_y', 'y'], 'y coordinate (m)', 'Detected and real y coordinate')
        # z coordinate chart
        self.launch_chart(['detected_z', 'z'], 'z coordinate (m)', 'Detected and real z coordinate')

    def error_charts(self):
        """Plots the charts for the errors in each of the values"""

        # >>>Does not support orientation yet since the current implementation is unable to calculate it
        # Distance errors
        self.launch_chart(['error_distance'], 'Error (m)', 'Error in the distance measurement')
        # x coordinate errors
        self.launch_chart(['error_x'], 'Error (m)', 'Error in the x coordinate measurement')
        # y coordinate errors
        self.launch_chart(['error_y'], 'Error (m)', 'Error in the y coordinate measurement')
        # z coordinate errors
        self.launch_chart(['error_z'], 'Error (m)', 'Error in the z coordinate measurement')

    def animate(self, anim_config, interpolate_detected=True, view_params=None, which=[1, 1, 1, 1]):
        """Generates the animation for the recorded playout using PlotTrajectory

        Plots the triangulated mesh, the point cloud, the real trajectory and the detected trajectory as they evolve
        over time, with future option to enable/disable parts of it

        Arguments:
            anim_config: dictionary
                Parameters for the animation, constructed as follows: {
                "FPS": _, int, number of FPS of the video to be generated,
                "elevation": _, int, initial camera elevation,
                "azimuth": _, int, initial camera azimuth,
                "hideAxis": _, bool, whether or not to hide the axis,
                "showLidar": _, bool, whether or not to plot a black square at the origin symbolizing the sensor,
                "xmin": _, "xmax": _, floats, values bounding the x axis that will be plotted,
                "ymin": _, "ymax": _, floats, values bounding the x axis that will be plotted,
                "zmin": _, "zmax": _, floats, values bounding the x axis that will be plotted
            }
            interpolate_detected: bool
                Whether to interpolate between the detected trajectory data for plotting or use only detected values
            view_params: dictionary
                Contains secondary viewing parameters. Refer to envplot.py for more informatioin
            which: list of bool
                Which of the parts of the plot to show, these are: triangular mesh, point cloud, real trajectory and
                detected trajectory, in that order.

        Initializes:
            self.ani: matplotlib animation
                Contains the animation
        """

        # Dictionary of fixed parameters for plotting
        params = dict(anim_config)
        params['minrange'] = self.sensor.minrange
        params['maxrange'] = self.sensor.maxrange
        params['fov'] = self.sensor.fov
        params['lidarFPS'] = self.fps
        params['lidarSubframes'] = self.subframes
        params['lidarVPoints'] = self.sensor.vpoints_number
        params['lidarKind'] = self.kind
        params['interpolateDetected'] = interpolate_detected

        # Generates the playout for plotting, containing the specified number of frames
        plotting_playout = self.scenario.plotting_playout(
            start=0, no_frames=anim_config['FPS'] * self.scenario_duration, fps=anim_config['FPS']
        )

        # For the detected trajectory it only uses the position and not the orientation, since it plots a point
        detected_trajectory = np.column_stack((
            self.data['timestamp'],
            self.data['detected_x'], self.data['detected_y'], self.data['detected_z']
        ))

        # Calls plot trajectory to generate a new object
        plot = PlotTrajectory(params, view_params=view_params,
                              plotting_playout=plotting_playout, pointclouds=self.pointclouds,
                              real_trajectory=self.trajectory, detected_trajectory=detected_trajectory
        )

        # Saves the generated animation
        self.ani = plot.plot()

    def show_animation(self):
        """Shows the generated animation"""

        plt.show()

    def save_animation(self, fps, savefile):
        """Saves the generated animation into a specified file

        Arguments
            fps: int
                Number of frames per second that the generated video will have
            savefile: str
                Name of the file to be saved, only supports .mp4
        """

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='LidSim'), bitrate=1800)
        self.ani.save(savefile, writer=writer)
