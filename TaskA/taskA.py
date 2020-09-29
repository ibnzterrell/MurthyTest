import sys
from PySide2.QtWidgets import QAction, QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide2.QtCore import Slot
from PySide2.QtGui import QKeySequence
import multiprocessing as mp
import numpy as np
import cv2
import server
import zmq
import json
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

    def stopServerProcess(self):
        self.widget.stopServerProcess()

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
        self.serverStatus.setText("Model Loaded")
        self.widget.loadModel(fileName)


class NavigationWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.layout = QHBoxLayout(self)
        self.frameLabel = QLabel("0")
        self.prevFrameButton = QPushButton("<")
        self.nextFrameButton = QPushButton(">")
        self.layout.addWidget(self.prevFrameButton)
        self.layout.addWidget(self.frameLabel)
        self.layout.addWidget(self.nextFrameButton)


class MainWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.figure = plt.figure(figsize=(8, 8))
        self.ax = self.figure.add_axes([0, 0, 1, 1])
        self.canvas = FigureCanvas(self.figure)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.canvas)
        self.navigation = NavigationWidget()
        self.layout.addWidget(self.navigation)
        self.navigation.nextFrameButton.clicked.connect(self.nextFrame)
        self.navigation.prevFrameButton.clicked.connect(self.prevFrame)
        self.client = Client()
        self.client.runServerProcess()
        self.client.predictCallback = self.showPrediction
        self.frameN = 0
        self.frameMax = 0

    @Slot()
    def nextFrame(self):
        self.frameN = self.frameN + 1
        self.seekFrame()

    @Slot()
    def prevFrame(self):
        self.frameN = self.frameN - 1
        self.seekFrame()

    def loadVideo(self, fileName):
        self.reader = cv2.VideoCapture(fileName)
        self.frameN = 0
        self.frameMax = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        self.seekFrame()

    def seekFrame(self):
        # Clamp frameN to valid frames
        if (self.frameN < 0):
            self.frameN = 0
        if (self.frameN > self.frameMax):
            self.frameN = self.frameMax
        self.navigation.frameLabel.setText(str(self.frameN))

        self.reader.set(cv2.CAP_PROP_POS_FRAMES, self.frameN)
        status, self.img = self.reader.read()
        self.img = self.img[:, :, :1]  # convert to grayscale

        # Show image while we wait for prediction
        self.ax.imshow(self.img.squeeze(), cmap="gray")
        self.canvas.draw()

        self.client.sendCommand("predict", self.img.tolist())

    def showPrediction(self, response):
        heatmap = np.asarray(response["payload"])
        self.ax.imshow(self.img.squeeze(), cmap="gray")
        self.ax.imshow(heatmap.squeeze(),
                       extent=[
            -0.5,
            self.img.shape[1] - 0.5,
            self.img.shape[0] - 0.5,
            -0.5,
        ],  # (left, right, top, bottom),
            alpha=0.5
        )
        self.canvas.draw()

    def loadModel(self, fileName):
        self.client.sendCommand("loadModel", fileName)

    def stopServerProcess(self):
        self.client.stopServerProcess()


class Client:
    def runServerProcess(self):
        self.p = mp.Process(target=server.process)
        self.p.start()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:8888")
        self.loadModelCallback = self.defaultCallback
        self.predictCallback = self.defaultCallback
        self.shutdownCallback = self.defaultCallback

    def defaultCallback(self, message):
        print("Callback Not Assigned")

    def stopServerProcess(self):
        self.sendCommand("shutdown")
        self.p.join()

    def sendCommand(self, opCode, payload=""):
        command = {
            "opCode": opCode,
            "payload": payload
        }
        self.socket.send_json(command)
        response = self.socket.recv_json()
        self.handleResponse(response)

    def handleResponse(self, message):
        {
            "loadModelResponse": self.loadModelCallback,
            "predictResponse": self.predictCallback,
            "shutdownResponse": self.shutdownCallback
        }[message["opCode"]](message)

    def sendPrediction(self, X):
        self.sendCommand("predict", X)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    result = app.exec_()
    mainWindow.stopServerProcess()
    sys.exit(result)
