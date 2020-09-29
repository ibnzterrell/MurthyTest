import sys
from PySide2.QtWidgets import QAction, QApplication, QFileDialog, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide2.QtCore import Slot
from PySide2.QtGui import QKeySequence
import multiprocessing as mp
import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.widget = MainWidget()

        # Menu Bar
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")
        self.model_menu = self.menu.addMenu("Model")

        # File->Open Video Action
        open_action = QAction("Open File...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_video)
        self.file_menu.addAction(open_action)

        # File->Exit Action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)

        # Model->Open Model Action
        open_model = QAction("Open Model...", self)
        open_model.triggered.connect(self.open_model)
        self.model_menu.addAction(open_model)

        # Status Bar
        self.statusBar = self.statusBar()
        self.serverStatus = QLabel("Model Not Loaded")
        self.statusBar.addPermanentWidget(self.serverStatus)

        self.setWindowTitle("Task A")
        self.setCentralWidget(self.widget)

    @Slot()
    def open_video(self):
        result = QFileDialog.getOpenFileName(
            self, "Open Video", "./", "Video Files (*.mp4)")
        fileName = result[0]
        self.widget.loadVideo(fileName)

    @Slot()
    def open_model(self):
        result = QFileDialog.getOpenFileName(
            self, "Open Model", "./", "Model Files (*.h5)")
        fileName = result[0]


class MainWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.figure = plt.figure(figsize=(8, 8))
        self.ax = self.figure.add_axes([0, 0, 1, 1])
        self.canvas = FigureCanvas(self.figure)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.canvas)

    def loadVideo(self, fileName):
        self.reader = cv2.VideoCapture(fileName)
        n_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        self.seekFrame(0)

    def seekFrame(self, frameN):
        self.reader.set(cv2.CAP_PROP_POS_FRAMES, frameN)
        status, img = self.reader.read()
        img = img[:, :, :1]  # convert to grayscale
        self.ax.imshow(img.squeeze(), cmap="gray")
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWidget = MainWidget()
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
