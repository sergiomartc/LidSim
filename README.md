# LidSim
![LidSim logo](https://github.com/smceron/LidSim/blob/master/images/logos/lidsimlogo.jpg)

**LidSim** is a lidar point cloud gathering and visualization simulator. It supports both flash and scanning lidar technologies, accounting for both angular and range errors, and is capable of simulating point clouds of moving objects. The program also incorporates some basic tracking capabilities, while also allowing for machine learning models to be added in the future to the underlying code.

<p align="center">
  <img src="https://github.com/smceron/LidSim/blob/master/images/logos/gui_screenshot.jpg" />
</p>

## Reasoning behind the software development

The primary aim of the program is to be able to simulate the response of a certain lidar under a specific condition, and therefore knowing beforehand what could be expected of the data it would return. This way, it is possible to make a more informed decision on what lidar model would be best suited for a specific application, without the need to perform these costly tests in the real world and with a real model. Currently, there exists no widely available tool for performing physics-based lidar simulations, and this is specially true in flight test operations, which is what this program is trying to simulate.

<p align="center">
  <img src="https://github.com/smceron/LidSim/blob/master/images/logos/b2_flash.gif" width="380" />
  <img src="https://github.com/smceron/LidSim/blob/master/images/logos/b2_scanning.gif" width="380" />
</p>

In these example gifs, a [B-2 stealth bomber](https://en.wikipedia.org/wiki/Northrop_Grumman_B-2_Spirit) can be seen performing some simulated maneuvers, with the trajectory that its center of mass follows being shown as the yellow sphere. The pointcloud that the simulated lidar generates are the series of red points that appear on the surface of the aircraft, while the trajectory that the simulator is able to detect from these points is represented by the yellow square. The way this trajectory is computed is by calculating the centroid of the point cloud for every frame, as the program does not support any complex tracking algorithms as of yet. The black square represents where the simulated lidar is placed: at the origin and aiming towards the x axis. The animation of the left is from simulating a flash lidar, while the one on the right comes from simulating a scanning lidar.

## Performing a simulation

### Definining a lidar

To generate this video, a lidar must first be defined by setting its parameters under the _Lidar_ part of the GUI. 

<p align="center">
  <img src="https://github.com/smceron/LidSim/blob/master/images/logos/lidar_parameters.png" />
</p>

These parameters include:
- Kind: technology of the model, either flash or scanning lidar.
- Range parameters:
  - Minimum and maximum: distances to the lidar that bound the range where the sensor is able to pick up a signal.
  - Standard deviation: models the error that the lidar has when measuring radial distances from it.
- Angular parameters:
  - Horizontal and vertical resolution: minimum discernible angular difference that the lidar is able to detect, can also be seen as an angular step.
  - Horizontal and vertical accuracy: errors in these resolutions, modelled as a standard deviation.
- Coverage parameters:
  - Horizontal and vertical FOV (field of view): extent of the simulation that is seen at any given moment by the lidar.
  - Frame rate: number of frames a second that the lidar is able to record, can also be seen as the refresh rate.

The field of view and angular resolution parameters define the upper bound of the total number of points that the simulated lidar is able to generate. Once these values have been defined, it is possible to load the lidar into memory by clicking on _Load lidar_. With a lidar model loaded, it is now possible to proceed to the _Scenario_ sector of the GUI.

The program also allows you to load a lidar directly from a file, by going into _Options > Change lidar file_ and selecting a json file [with the appropriate formatting](https://github.com/smceron/LidSim/blob/master/data/lidar_models.json), as well as saving a loaded lidar into the specified lidar file. The two lidars defined in this file were the ones used to generate the animations above.

### Defining a scenario

Once a lidar has been loaded into memory, a scenario can be loaded for recording.

<p align="center">
  <img src="https://github.com/smceron/LidSim/blob/master/images/logos/scenario_definition.png" />
</p>

A scenario is defined as an object following a certain trajectory:
- The object must be a triangulated mesh in stl format, such as the one used for the [example](https://github.com/smceron/LidSim/blob/master/data/B2.stl).
- The trajectory is defined by a csv file containing the timestamp, position (_x_, _y_, _z_) and orientation (_ψ_, _θ_, _φ_) of the object, following the [Tait-Bryan angle convention](https://en.wikipedia.org/wiki/Euler_angles#Tait%E2%80%93Bryan_angles), such as the one used for the [example](https://github.com/smceron/LidSim/blob/master/data/trajectory.txt).

### Camera parameters

These allow you to set the animation FPS, initial camera elevation and azimuth, and whether to hide the axis and show the lidar as a black box at the origin or not. The camera bounding parameters allow you to focus the view over a certain space of interest, although the axes will always default to a square with its side being the biggest range provided. All these parameters can either be modified or left to default values. For the example videos above, the _elevation_ parameter was set to 30 degrees, and the _Show lidar_ box clicked.

### Launching and saving 

Once both the lidar and the scenario have been loaded, it is possible to simulate the point cloud data.

<p align="center">
  <img src="https://github.com/smceron/LidSim/blob/master/images/logos/launching-saving.png" />
</p>

These buttons have the following functionality:
- Launch animation: launches a 3D matplotlib interactive animation, that lets you change the camera position.
- Launch charts: generating charts with the real vs detected trajectories over time, as well as the error in the detected trajectory over time.
- Save animation: letting you save the generated animation into an .mp4 file with the provided filename, or defaulting to a time stamped one.
- Save trajectory: letting you save both the real and detected trajectory into an .csv file with the provided filename, or defaulting to a time stamped one.

These files are saved by default under _savefiles_, that is generated under the [_src_](https://github.com/smceron/LidSim/tree/master/src) directory, but this directory can be changed under _Options > Change save folder_.

## Final note on the code

The code included is well commented with every algorithmic implementation being well explained. It is recommended to run the program in Python 3.9, as other versions have shown some incompatibility with the _vtk_ library. 
