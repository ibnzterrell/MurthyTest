import sys
from PySide2.QtWidgets import QAction, QApplication, QFileDialog, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide2.QtCore import Slot
from PySide2.QtGui import QKeySequence


class MainWindow(QMainWindow):
    def __init__(self, widget):
        QMainWindow.__init__(self)
        self.setWindowTitle("Task A")

        # Menu Bar
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")

        # File->Open Video Action
        open_action = QAction("Open File...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_file)
        self.file_menu.addAction(open_action)

        # File->Exit Action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)

        # Status Bar
        self.statusBar = self.statusBar()
        self.serverStatus = QLabel("Server Not Running")
        self.statusBar.addPermanentWidget(self.serverStatus)

        self.setCentralWidget(widget)

    @Slot()
    def open_file(self):
        fileName = QFileDialog.getOpenFileName(
            self, "Open Video", "./", "Video Files (*.mp4)")


class MainWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        # Create Widgets
        self.label = QLabel("Test")

        # Create Vertical Layout and Add Widgets to It
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)

        self.setLayout(self.layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    mainWidget = MainWidget()
    mainWindow = MainWindow(mainWidget)
    mainWindow.show()
    sys.exit(app.exec_())
