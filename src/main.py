"""Launches the GUI"""
import sys
from gui.GUI import GUI
from PyQt5.QtWidgets import QApplication

def my_excepthook(type, value, tback):

    # Call the default handler
    sys.__excepthook__(type, value, tback)

def main():

    sys.excepthook = my_excepthook
    # A new instance of QApplication
    app = QApplication(sys.argv)
    # Set the form to be the main form
    form = GUI(app)
    # Show the form
    form.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    # Running the file
    main()