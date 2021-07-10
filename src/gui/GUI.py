import json
import os
import time
import matplotlib

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from .QtDMainForm import Ui_MainWindow
from PyQt5 import QtGui

from lidarsim.lidar import Sensor
from lidarsim.recorder import LidarRecorder


class GUI(QMainWindow, Ui_MainWindow):

    def __init__(self, app, parent=None):
        """Initializes the GUI for LidSim"""

        # Set-up parameters
        super(GUI, self).__init__(parent)
        self.setupUi(self)
        self.app = app
        self.move(0, 0)

        # Change in the predefined lidar combobox activates a function that fills up all the values
        self.comboBox_SelectedPredeterminedLidar.currentTextChanged.connect(self.change_predefined_model)

        # Loading lidar configuration when pressing the button
        self.pushButton_loadLidarConfig.clicked.connect(self.load_lidar_config)

        # Saving lidar configuration when pressing the button, initially unavailable
        self.pushButton_saveLidarConfig.setEnabled(False)
        self.pushButton_saveLidarConfig.clicked.connect(self.save_lidar_config)

        # Changing any lidar parameters resets both the model and the name to default values
        # This way, after selecting a lidar model, if any parameter is changed, it stops it from showing it as the
        # original model
        self.lineEdit_minRange.editingFinished.connect(self.reset_lidar_model_name)
        self.lineEdit_maxRange.editingFinished.connect(self.reset_lidar_model_name)
        self.lineEdit_rangeSTD.editingFinished.connect(self.reset_lidar_model_name)
        self.lineEdit_horizontalResolution.editingFinished.connect(self.reset_lidar_model_name)
        self.lineEdit_verticalResolution.editingFinished.connect(self.reset_lidar_model_name)
        self.lineEdit_horizontalAccuracy.editingFinished.connect(self.reset_lidar_model_name)
        self.lineEdit_verticalAccuracy.editingFinished.connect(self.reset_lidar_model_name)
        self.lineEdit_horizontalFOV.editingFinished.connect(self.reset_lidar_model_name)
        self.lineEdit_verticalFOV.editingFinished.connect(self.reset_lidar_model_name)
        self.lineEdit_lidarFPS.editingFinished.connect(self.reset_lidar_model_name)

        # File selection for the object and trajectory
        self.toolButton_ChangeObjectFile.clicked.connect(self.change_object_file)
        self.toolButton_ChangeTrajectoryFile.clicked.connect(self.change_trajectory_file)

        # These textboxes will only show the file that was selected, so they cannot be written over
        self.lineEdit_objectFile.setReadOnly(True)
        self.lineEdit_trajectoryFile.setReadOnly(True)

        # Animation and trajectory execution/saving cannot be clicked until the data is there to activate them
        self.pushButton_loadScenario.setEnabled(False)
        self.pushButton_launchAnimation.setEnabled(False)
        self.pushButton_launchCharts.setEnabled(False)
        self.pushButton_saveAnimation.setEnabled(False)
        self.pushButton_saveTrajectory.setEnabled(False)

        # Animation parameters have predetermined values
        self.lineEdit_animationFPS.setText("30")
        self.lineEdit_animationElevation.setText("0")
        self.lineEdit_animationAzimuth.setText("-180")
        # The hide axis parameter starts as false, as the box is unchecked, and so does the show lidar parameter
        self.hide_axis = False
        self.show_lidar = False

        # Animation and trajectory graphs execution
        self.checkBox_animationHideAxis.clicked.connect(self.hide_axis_checkbox_clicked)
        self.checkBox_animationShowLidar.clicked.connect(self.show_lidar_checkbox_clicked)
        self.pushButton_loadScenario.clicked.connect(self.load_scenario)
        self.pushButton_launchAnimation.clicked.connect(self.launch_animation)
        self.pushButton_launchCharts.clicked.connect(self.launch_charts)

        # Animation, trajectory, and trajectory graphs saving
        self.pushButton_saveAnimation.clicked.connect(self.save_animation)
        self.pushButton_saveTrajectory.clicked.connect(self.save_trajectory)

        # Options toolbar
        self.actionChange_lidar_file.triggered.connect(self.change_lidar_models_file)
        self.actionChange_save_folder.triggered.connect(self.change_save_folder)
        self.actionViewing_parameters.triggered.connect(self.change_viewing_parameters)

        # Parameters that are used by the methods below
        self.recorder = None
        self.scenario = None
        # Predefined lidar models are undefined, must be loaded by the user from Options>Change lidar file
        self.predefined_lidars = {"--select model--": ""}
        self.lidar_config = None
        self.lidar_models_file = None
        self.save_directory = os.getcwd() + '\\savefiles\\'
        self.viewing_parameters = None

    def _clean_lidar_config(self):
        """Cleans all the lidar input text boxes"""

        self.comboBox_SelectedPredeterminedLidar.setCurrentIndex(0)
        self.lineEdit_lidarName.setText("")
        self.lineEdit_minRange.setText("")
        self.lineEdit_maxRange.setText("")
        self.lineEdit_rangeSTD.setText("")
        self.lineEdit_horizontalResolution.setText("")
        self.lineEdit_verticalResolution.setText("")
        self.lineEdit_horizontalAccuracy.setText("")
        self.lineEdit_verticalAccuracy.setText("")
        self.lineEdit_horizontalFOV.setText("")
        self.lineEdit_verticalFOV.setText("")
        self.lineEdit_lidarFPS.setText("")

    def change_predefined_model(self):
        """Changes the all lidar parameters when a predefined model is chosen from the combo box"""

        # Current selected model
        model = self.comboBox_SelectedPredeterminedLidar.currentText()

        # Default state
        if model not in self.predefined_lidars:
            # Clears the name
            self.lineEdit_lidarName.setText("")
            # Finish execution
            return

        # Otherwise, it fills up the values, first getting the model name

        # Assign the index of the combo box for each supported technologies, flash or scanning
        supported_technologies = {'flash': 0, 'scanning': 1}
        idx = supported_technologies[self.predefined_lidars[model]['lidarKind']]
        self.comboBox_lidarKind.setCurrentIndex(idx)

        # Filling up the rest of the values
        self.lineEdit_minRange.setText(str(self.predefined_lidars[model]['lidarRange']['minimum']))
        self.lineEdit_maxRange.setText(str(self.predefined_lidars[model]['lidarRange']['maximum']))
        self.lineEdit_rangeSTD.setText(str(self.predefined_lidars[model]['lidarRange']['standardDeviation']))
        self.lineEdit_horizontalResolution.setText(
            str(self.predefined_lidars[model]['lidarAngular']['resolution']['horizontal']))
        self.lineEdit_verticalResolution.setText(
            str(self.predefined_lidars[model]['lidarAngular']['resolution']['vertical']))
        self.lineEdit_horizontalAccuracy.setText(
            str(self.predefined_lidars[model]['lidarAngular']['accuracy']['horizontal']))
        self.lineEdit_verticalAccuracy.setText(
            str(self.predefined_lidars[model]['lidarAngular']['accuracy']['vertical']))
        self.lineEdit_horizontalFOV.setText(str(self.predefined_lidars[model]['lidarFOV']['horizontal']))
        self.lineEdit_verticalFOV.setText(str(self.predefined_lidars[model]['lidarFOV']['vertical']))
        self.lineEdit_lidarFPS.setText(str(self.predefined_lidars[model]['lidarFPS']))

        # The name field is set at the end because changing the lines resets it back to an empty string
        self.lineEdit_lidarName.setText(model)

    def load_lidar_config(self):
        """Loads the lidar configuration by reading the text boxes"""

        # Empty dictionary to be filled up
        lidar_config = {'lidarRange': {}, 'lidarAngular': {'resolution': {}, 'accuracy': {}}, 'lidarFOV': {}}

        try:
            # See if values provided are floats
            minrange = float(self.lineEdit_minRange.text())
            maxrange = float(self.lineEdit_maxRange.text())
            stdrange = float(self.lineEdit_rangeSTD.text())

            hres = float(self.lineEdit_horizontalResolution.text())
            vres = float(self.lineEdit_verticalResolution.text())
            hstd = float(self.lineEdit_horizontalAccuracy.text())
            vstd = float(self.lineEdit_verticalAccuracy.text())

            hfov = float(self.lineEdit_horizontalFOV.text())
            vfov = float(self.lineEdit_verticalFOV.text())
            fps = float(self.lineEdit_lidarFPS.text())

        except ValueError:
            # Conversion could not be made, raises a warning box
            self.label_Status.setText('Status: clearing values')
            title = 'Lidar configuration error: values provided are not valid'
            msg = "Please make sure the values provided are all numerical."
            self.warning_box(title, msg)
            self._clean_lidar_config()
            self.label_Status.setText('Status: ')
            return

        else:
            # Test for incompatibility in the data
            if hres > hfov or vres > vfov:
                # Data is not compatible, raises a warning box
                # Each of the fields of view must be greater than their respective resolutions, otherwise the target
                # points cannot be defined
                self.label_Status.setText('Status: clearing values')
                title = 'Lidar configuration error: values provided are not valid'
                msg = "Please make sure that the fields of view are bigger than their respective resolutions."
                self.warning_box(title, msg)
                self._clean_lidar_config()
                self.label_Status.setText('Status: ')
                return

            # The data is valid
            # Load all the parameters into the configuration
            lidar_config['lidarKind'] = str(self.comboBox_lidarKind.currentText())

            lidar_config['lidarRange']['minimum'] = minrange
            lidar_config['lidarRange']['maximum'] = maxrange
            lidar_config['lidarRange']['standardDeviation'] = stdrange

            lidar_config['lidarAngular']['resolution']['horizontal'] = hres
            lidar_config['lidarAngular']['resolution']['vertical'] = vres
            lidar_config['lidarAngular']['accuracy']['horizontal'] = hstd
            lidar_config['lidarAngular']['accuracy']['vertical'] = vstd

            lidar_config['lidarFOV']['horizontal'] = hfov
            lidar_config['lidarFOV']['vertical'] = vfov
            lidar_config['lidarFPS'] = fps

            # Initializes the lidar parameters
            sensor = Sensor(lidar_config)
            self.recorder = LidarRecorder(sensor)
            self.pushButton_loadScenario.setEnabled(True)
            self.lidar_config = lidar_config

            # By changing the loaded lidar these actions will not be available until the scenario is loaded
            # So if they were enabled beforehand, they are disabled now
            self.pushButton_launchAnimation.setEnabled(False)
            self.pushButton_launchCharts.setEnabled(False)
            self.pushButton_saveAnimation.setEnabled(False)
            self.pushButton_saveTrajectory.setEnabled(False)

        self.label_Status.setText('Status: lidar configuration loaded')
        # It is now possible to save the lidar
        self.pushButton_saveLidarConfig.setEnabled(True)

        # Camera bounding parameters have predefined values that are set when the lidar is loaded
        self.lineEdit_boundxmin.setText("0")
        self.lineEdit_boundxmax.setText(str(maxrange))
        self.lineEdit_boundymin.setText(str(maxrange/2))
        self.lineEdit_boundymax.setText(str(-maxrange/2))
        self.lineEdit_boundzmin.setText(str(-maxrange/2))
        self.lineEdit_boundzmax.setText(str(maxrange/2))

    def save_lidar_config(self):
        """Saves the specified lidar configuration that was last loaded.

        If all the parameters, including name, are not given, it raises a warning box. Similarly, if the lidar file is
        not specified, raises a different warning box
        """

        # Checks if the lidar models file is actually defined
        if self.lidar_models_file is None:
            # If there is no file to save it in, raises a warning box
            title = 'Lidar configuration saving error: no file defined'
            msg = 'Please make sure you specify a save file under Options>Change lidar file'
            self.warning_box(title, msg)
            return

        # Checks if the name is empty string
        if self.lineEdit_lidarName.text() == "":
            # If no name is defined for the lidar, it cannot be stored, and raises a warning box
            title = 'Lidar configuration saving error: no name defined'
            msg = 'Please make sure you provide a name for the lidar before saving it'
            self.warning_box(title, msg)
            return

        # Checks if the name is repeated
        if self.lineEdit_lidarName.text() in self.predefined_lidars:
            # If the name is already in the list of lidars, that means it is a dupe, and raises a warning box
            title = 'Lidar configuration saving error: duplicate name'
            msg = 'The name you provided is already in the list'
            self.warning_box(title, msg)
            self.lineEdit_lidarName.setText("")
            return

        # Otherwise, the lidar can be saved
        # Saves this last configuration to the available file
        name = self.lineEdit_lidarName.text()
        self.predefined_lidars[self.lineEdit_lidarName.text()] = self.lidar_config
        with open(self.lidar_models_file, 'w') as f:
            json.dump(self.predefined_lidars, f)
            idx = len(self.predefined_lidars.keys())
        # Changes the GUI viewing to reflect this last saved lidar as the one currently loaded
        self.comboBox_SelectedPredeterminedLidar.addItems({self.lineEdit_lidarName.text(): self.lidar_config})
        self.comboBox_SelectedPredeterminedLidar.setCurrentIndex(idx)

        # Save lidar box is now turned off, as it has already been saved
        self.pushButton_saveLidarConfig.setEnabled(False)
        self.label_Status.setText('Status: last loaded lidar configuration saved as ' + name)

    def reset_lidar_model_name(self):
        """Clears the lidar name value after a change in the other parameters has been made, as well as preventing it
        from being saved"""

        self.lineEdit_lidarName.setText("")
        self.pushButton_saveLidarConfig.setEnabled(False)

    def change_object_file(self):
        """Opens up a file dialog box to select the STL object file and reflects it in the textbox"""

        # By changing the object file these actions will not be available until the new scenario is loaded
        self.pushButton_launchAnimation.setEnabled(False)
        self.pushButton_launchCharts.setEnabled(False)
        self.pushButton_saveAnimation.setEnabled(False)
        self.pushButton_saveTrajectory.setEnabled(False)

        # Opens up the file dialog box
        fil = "STL (*.stl);;All files (*.*)"
        self.label_Status.setText('Status: changing object file')
        dialog = QFileDialog(self, 'Select STL file', "...", fil)
        if dialog.exec_() == QFileDialog.Accepted:
            file = dialog.selectedFiles()[0]
            # Stores the selected value in the textbox
            self.lineEdit_objectFile.setText(file)
        self.label_Status.setText('Status: ')

        if self.lineEdit_trajectoryFile.text() != "":
            # Both values for the scenario have been provided
            self.label_Status.setText('Status: scenario ready to be loaded')

    def change_trajectory_file(self):
        """Opens up a file dialog to select the trajectory file and reflects it in the textbox"""

        # By changing the trajectory file these actions will not be available until the new scenario is loaded
        self.pushButton_launchAnimation.setEnabled(False)
        self.pushButton_launchCharts.setEnabled(False)
        self.pushButton_saveAnimation.setEnabled(False)
        self.pushButton_saveTrajectory.setEnabled(False)

        # Opens up the file dialog box
        fil = "TXT (*.txt);;CSV (*.csv);;NPY (*.npy);;All files (*.*)"
        self.label_Status.setText('Status: changing trajectory file')
        dialog = QFileDialog(self, 'Select trajectory file', "...", fil)
        if dialog.exec_() == QFileDialog.Accepted:
            file = dialog.selectedFiles()[0]
            # Stores the selected value in the textbox
            self.lineEdit_trajectoryFile.setText(file)
        self.label_Status.setText('Status: ')

        if self.lineEdit_objectFile.text() != "":
            # Both values for the scenario have been provided
            self.label_Status.setText('Status: scenario ready to be loaded')

    def hide_axis_checkbox_clicked(self, checked):
        """Changes the state of the hide_axis value every time the checkbox is clicked"""
        self.hide_axis = checked

    def show_lidar_checkbox_clicked(self, checked):
        """Changes the state of the show_lidar value every time the checkbox is clicked"""
        self.show_lidar = checked

    def load_scenario(self):
        """Loads the defined scenario into memory"""

        # Reads the textboxes for the corresponding files
        object_file = self.lineEdit_objectFile.text()
        trajectory_file = self.lineEdit_trajectoryFile.text()

        try:
            # Tries to load the scenario
            self.recorder.load_scenario(object_file, trajectory_file)

        except:
            # Bad approach, I know, but there are countless exceptions that can arise from a wrong scenario being loaded
            # This being a simple reading of two files, this approach, although generally a bad take, works
            # Scenario could not be loaded if any exception is raised. Shows a warning box
            self.label_Status.setText('Status: clearing values')
            title = 'Scenario initialization error: values provided are not valid'
            msg = "Please make sure the files provided are valid"
            self.warning_box(title, msg)
            # Clears the scenario files
            self.lineEdit_objectFile.setText("")
            self.lineEdit_trajectoryFile.setText("")
            self.label_Status.setText('Status: ')
            return

        # Otherwise, the scenario can be loaded
        self.label_Status.setText('Status: scenario loaded')
        # Once the scenario is loaded, it is possible to launch and store animations, launch charts, and store
        # trajectories
        self.pushButton_launchAnimation.setEnabled(True)
        self.pushButton_launchCharts.setEnabled(True)
        self.pushButton_saveAnimation.setEnabled(True)
        self.pushButton_saveTrajectory.setEnabled(True)

    def load_animation(self):
        """Loads the animation into memory, used both for launching and saving the animation"""

        # Checks if the pointclouds have already been generated, if they haven't, it generates them
        if self.recorder.pointclouds is None:
            self.label_Status.setText('Status: generating point cloud')
            self.app.processEvents()
            self.recorder.record()
            self.label_Status.setText('Status: calculating trajectory')
            self.app.processEvents()
            self.recorder.get_trajectory()
            self.label_Status.setText('Status: ')
            self.app.processEvents()

        # Gets the animation parameters as inputted in the GUI
        anim_config_params = {
            "FPS": int(self.lineEdit_animationFPS.text()),
            "elevation": int(self.lineEdit_animationElevation.text()),
            "azimuth": int(self.lineEdit_animationAzimuth.text()),
            "hideAxis": self.hide_axis,
            "showLidar": self.show_lidar,
            "xmin": float(self.lineEdit_boundxmin.text()),
            "xmax": float(self.lineEdit_boundxmax.text()),
            "ymin": float(self.lineEdit_boundymin.text()),
            "ymax": float(self.lineEdit_boundymax.text()),
            "zmin": float(self.lineEdit_boundzmin.text()),
            "zmax": float(self.lineEdit_boundzmax.text()),
        }

        # Generates a new animation with these config parameters and the viewing parameters, which may not be defined
        self.label_Status.setText('Status: generating animation')
        self.recorder.animate(anim_config=anim_config_params, view_params=self.viewing_parameters)

    def launch_animation(self):
        """Launches the generated animation for matplotlib-interactive viewing"""

        # Loads a new animation instance
        self.load_animation()

        # Launches the animation
        self.label_Status.setText('Status: launching animation')
        self.recorder.show_animation()
        self.label_Status.setText('Status: ')

    def launch_charts(self):
        """Launches the chats for the detected trajectory

        These comprise of: real values and detected values for distance, x, y and z coordinates, and the errors in the
        measurements of distance, x, y and z coordinates.
        """

        # Checks if the pointclouds have already been generated, if they haven't, it generates them
        if self.recorder.pointclouds is None:
            self.label_Status.setText('Status: generating point cloud')
            self.recorder.record()
            self.label_Status.setText('Status: calculating trajectory')
            self.recorder.get_trajectory()
            self.label_Status.setText('Status: ')

        # With the pointclouds generated and the data then stored under self.recorder.data, launches the charts
        self.recorder.real_detected_charts()
        self.recorder.error_charts()

        # Goes back to matplotlib defaults to not interfere with the animation plotting
        matplotlib.rc_file_defaults()

    def save_animation(self):
        """Saves the generated animation under the specified file or a default one"""

        # Check if the directory exists beforehand, create it if not
        if not os.path.exists(self.save_directory):
            os.mkdir(self.save_directory)

        # Generates the saving path
        name = self.lineEdit_saveAnimationFilename.text()
        if name == '':
            name = 'animation_' + str(int(time.time()))
        path = self.save_directory + name + '.mp4'

        # Checks if the file already exists, raises a warning box if so
        if os.path.isfile(path):
            title = 'File error'
            msg = 'The file you specified already exists'
            self.warning_box(title, msg)
            return

        # Saves this new animation
        self.label_Status.setText('Status: saving animation')
        self.recorder.save_animation(params['FPS'], path)
        self.label_Status.setText('Status: ')

    def save_trajectory(self):
        """Saves the trajectory data under the specified file or a default one"""

        # Checks if the pointclouds have already been generated, if they haven't, it generates them
        if self.recorder.pointclouds is None:
            self.label_Status.setText('Status: generating point cloud')
            self.recorder.record()
            self.label_Status.setText('Status: calculating trajectory')
            self.recorder.get_trajectory()
            self.label_Status.setText('Status: ')

        # Check if the directory exists beforehand, create it if not
        if not os.path.exists(self.save_directory):
            os.mkdir(self.save_directory)

        # Generates the saving path
        name = self.lineEdit_saveTrajectoryFilename.text()
        if name == '':
            name = 'trajectory_' + str(int(time.time()))
        path = self.save_directory + name + '.csv'

        # Checks if the file already exists, raises a warning box if so
        if os.path.isfile(path):
            title = 'File error'
            msg = 'The file you specified already exists'
            self.warning_box(title, msg)
            return

        # Saves this new animation
        self.label_Status.setText('Status: saving trajectory')
        self.recorder.save_trajectory_data(path)
        self.label_Status.setText('Status: ')

    def load_predefined_lidar(self):
        """Loads the loaded list of predefined lidars and shows them under the combobox"""

        self.comboBox_SelectedPredeterminedLidar.clear()
        self.comboBox_SelectedPredeterminedLidar.addItems({"--select model--": ""})
        self.comboBox_SelectedPredeterminedLidar.addItems(self.predefined_lidars)

    def change_lidar_models_file(self):
        """Opens a json file containing the lidar models of interest"""

        fil = "JSON (*.json);;All files (*.*)"
        self.label_Status.setText('Status: changing lidar models file')
        dialog = QFileDialog(self, 'Select the lidar models file', "...", fil)
        if dialog.exec_() == QFileDialog.Accepted:
            self.lidar_models_file = dialog.selectedFiles()[0]

        self.predefined_lidars = json.load(open(self.lidar_models_file))
        # Loads this empty dictionary into the combobox, only "--select model--" will appear
        self.load_predefined_lidar()
        self.label_Status.setText('Status: ')

    def change_save_folder(self):
        """Selects a folder in which to save the animation and trajectory data"""

        dialog = QFileDialog(self, "Select directory", "...", None)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        if dialog.exec_() == QFileDialog.Accepted:
            self.save_directory = dialog.selectedFiles()[0]

    def change_viewing_parameters(self):
        """Opens a json file containing the animation parameters to use"""

        fil = "JSON (*.json);;All files (*.*)"
        self.label_Status.setText('Status: changing viewing parameters')
        dialog = QFileDialog(self, 'Select the viewing parameters file', "...", fil)
        if dialog.exec_() == QFileDialog.Accepted:
            file = dialog.selectedFiles()[0]

        # Saves it under viewing_parameters
        self.viewing_parameters = json.load(open(file))

    def warning_box(self, title, msg):
        """Generates a pop up warning box

        Args:
            title: str
                The title of the warning box
            msg: str
                Additional message of the warning box
        """

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(msg)
        msgBox.setWindowTitle(title)
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec_()

