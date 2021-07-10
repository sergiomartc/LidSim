# LidSim
![LidSim logo](https://github.com/smceron/LidSim/blob/master/images/logos/lidsimlogo.jpg)

**LidSim** is a lidar point cloud gathering and visualization simulator. It supports both flash and scanning lidar technologies, accounting for both angular and range errors, and is capable of simulating point clouds of moving objects. The program also incorporates some basic tracking capabilities, while also allowing for machine learning models to be added in the future to the underlying code.

<p align="center">
  <img src="https://github.com/smceron/LidSim/blob/master/images/logos/gui_screenshot.jpg" />
</p>

The primary aim of the program is to be able to simulate the response of a certain lidar under a specific condition, and therefore knowing beforehand what could be expected of the data it would return. This way, it is possible to make a more informed decision on what lidar model would be best suited for a specific application, without the need to perform these costly tests in the real world.

<p align="center">
  <img src="https://github.com/smceron/LidSim/blob/master/images/logos/example.gif" />
</p>

In this example gif, a B2 stealth bomber can be seen performing some simulated maneuvers, with its center of mass being represented by the yellow sphere. The pointcloud that the simulated lidar generates is repersented by the series of red points that appear on the surface of the aircraft, while the trajectory that the simulator is able to detect from this point cloud data is represented by the yellow square. The black square represents where the simulated lidar is placed: at the origin and aiming towards the x axis.

### Performing a simulation

To generate this video, a lidar must first be defined by setting its parameters under the _Lidar_ part of the GUI. These include:

<p align="center">
  <img src="https://github.com/smceron/LidSim/blob/master/images/logos/lidar_parameters.png" />
</p>





It is recommended to run the program in Python 3.9, as other versions have shown some incompatibility with the _vtk_ library. A copy of the environment used when developing the program is included under the _bin_ folder.
