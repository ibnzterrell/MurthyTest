import sys
from PySide2.QtWidgets import QAction, QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide2.QtCore import Slot, Qt
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

        # File->Open Video File Action
        open_action = QAction("Open Video File...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_video)
        self.file_menu.addAction(open_action)

        # File->Exit Action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)

        # Model->Open Model File Action
        open_model = QAction("Load Model File...", self)
        open_model.triggered.connect(self.open_model)
        self.model_menu.addAction(open_model)

        # Status Bar
        self.statusBar = self.statusBar()
        self.serverStatus = QLabel("Model Not Loaded")
        self.statusBar.addPermanentWidget(self.serverStatus)

        self.setWindowTitle("Task A")
        self.setCentralWidget(self.widget)

    def stopServers(self):
        self.widget.stopServers()

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
        self.serverStatus.setText("Model Loaded: " + fileName)
        self.widget.loadModel(fileName)


class NavigationWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.layout = QHBoxLayout(self)
        self.frameLabel = QLabel("0 / 0")
        self.frameLabel.setAlignment(Qt.AlignCenter)
        self.firstFrameButton = QPushButton("|<")
        self.prevFrameButton = QPushButton("<")
        self.nextFrameButton = QPushButton(">")
        self.prev2FrameButton = QPushButton("<<")
        self.next2FrameButton = QPushButton(">>")
        self.prev3FrameButton = QPushButton("<<<")
        self.next3FrameButton = QPushButton(">>>")
        self.lastFrameButton = QPushButton(">|")
        self.layout.addWidget(self.firstFrameButton)
        self.layout.addWidget(self.prev3FrameButton)
        self.layout.addWidget(self.prev2FrameButton)
        self.layout.addWidget(self.prevFrameButton)
        self.layout.addWidget(self.frameLabel)
        self.layout.addWidget(self.nextFrameButton)
        self.layout.addWidget(self.next2FrameButton)
        self.layout.addWidget(self.next3FrameButton)
        self.layout.addWidget(self.lastFrameButton)


class MainWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.figure = plt.figure(figsize=(8, 8))
        self.ax = self.figure.add_axes([0, 0, 1, 1])
        self.canvas = FigureCanvas(self.figure)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.canvas)

        # Add Navigation and Connect It
        self.navigation = NavigationWidget()
        self.layout.addWidget(self.navigation)
        self.navigation.firstFrameButton.clicked.connect(self.firstFrame)
        self.navigation.prevFrameButton.clicked.connect(self.prevFrame)
        self.navigation.prev2FrameButton.clicked.connect(self.prev2Frame)
        self.navigation.prev3FrameButton.clicked.connect(self.prev3Frame)
        self.navigation.nextFrameButton.clicked.connect(self.nextFrame)
        self.navigation.next2FrameButton.clicked.connect(self.next2Frame)
        self.navigation.next3FrameButton.clicked.connect(self.next3Frame)
        self.navigation.lastFrameButton.clicked.connect(self.lastFrame)

        self.client = InferenceClient()
        self.client.startServer()
        self.client.predictCallback = self.receivePrediction
        self.frameN = 0
        self.frameMax = 0

    @Slot()
    def firstFrame(self):
        self.frameN = 0
        self.seekFrame()

    @Slot()
    def lastFrame(self):
        self.frameN = self.frameMax
        self.seekFrame()

    @Slot()
    def nextFrame(self):
        self.frameN = self.frameN + 1
        self.seekFrame()

    @Slot()
    def next2Frame(self):
        self.frameN = self.frameN + 25
        self.seekFrame()

    @Slot()
    def next3Frame(self):
        self.frameN = self.frameN + 125
        self.seekFrame()

    @Slot()
    def prevFrame(self):
        self.frameN = self.frameN - 1
        self.seekFrame()

    @Slot()
    def prev2Frame(self):
        self.frameN = self.frameN - 25
        self.seekFrame()

    @Slot()
    def prev3Frame(self):
        self.frameN = self.frameN - 125
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

        # Indicate current frame and number of frames
        self.navigation.frameLabel.setText(
            str(self.frameN + 1) + " / " + str(self.frameMax + 1))

        # Decode frame
        self.reader.set(cv2.CAP_PROP_POS_FRAMES, self.frameN)
        status, self.img = self.reader.read()
        self.img = self.img[:, :, :1]  # convert to grayscale

        # Send image for prediction and show it while we wait
        self.ax.imshow(self.img.squeeze(), cmap="gray")
        self.canvas.draw()
        self.client.askForPrediction(self.img.tolist())

    def receivePrediction(self, response):
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
        self.client.loadModel(fileName)

    def stopServers(self):
        self.client.shutdownServer()


class InferenceClient:
    def startServer(self):
        self.p = mp.Process(target=server.inferenceProcess)
        self.p.start()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:8888")
        self.loadModelCallback = self.defaultCallback
        self.predictCallback = self.defaultCallback
        self.shutdownCallback = self.defaultCallback

    def defaultCallback(self, message):
        print("Callback Not Assigned")

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

    def loadModel(self, fileName):
        self.sendCommand("loadModel", fileName)

    def askForPrediction(self, X):
        self.sendCommand("predict", X)

    def shutdownServer(self):
        self.sendCommand("shutdown")
        self.p.join()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    result = app.exec_()
    mainWindow.stopServers()
    sys.exit(result)
