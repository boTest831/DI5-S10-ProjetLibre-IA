# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MLP.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!
import csv

import cv2
import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QMessageBox
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use("Qt5Agg")  # Declare to use QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import Perceptron.MLP as MLP_class
from PyQt5 import QtCore, QtGui, QtWidgets


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)  # Define a signal to send str

    def write(self, text):
        self.textWritten.emit(str(text))


class MyFigure(FigureCanvas):
    def __init__(self, width=10, height=4, dpi=100):
        # Step 1: Create a Create Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # Step 2: Activate the Figure window in the parent class
        super(MyFigure, self).__init__(self.fig)  # This sentence is essential, otherwise graphics cannot be displayed
        self.axes = self.fig.add_subplot(111)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(794, 679)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(440, 200, 371, 441))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 20, 221, 241))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.adjustLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.adjustLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.adjustLabel.setObjectName("adjustLabel")
        self.verticalLayout.addWidget(self.adjustLabel)
        self.imageView = QtWidgets.QGraphicsView(self.verticalLayoutWidget)
        self.imageView.setStyleSheet("border-image: url(:/img/drawing(1).png);")
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.NoBrush)
        self.imageView.setBackgroundBrush(brush)
        self.imageView.setObjectName("imageView")
        self.verticalLayout.addWidget(self.imageView)
        self.selectImageButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.selectImageButton.setObjectName("selectImageButton")
        self.verticalLayout.addWidget(self.selectImageButton)
        self.horizontalScrollBar = QtWidgets.QScrollBar(self.verticalLayoutWidget)
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")
        self.verticalLayout.addWidget(self.horizontalScrollBar)
        self.recognizeButton = QtWidgets.QPushButton(self.frame)
        self.recognizeButton.setGeometry(QtCore.QRect(30, 270, 121, 41))
        self.recognizeButton.setObjectName("recognizeButton")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.frame)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(30, 330, 181, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.resultLabel = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.resultLabel.setObjectName("resultLabel")
        self.horizontalLayout.addWidget(self.resultLabel)
        self.resultText = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.resultText.setObjectName("resultText")
        self.horizontalLayout.addWidget(self.resultText)
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(30, 210, 371, 341))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.tabWidget = QtWidgets.QTabWidget(self.frame_2)
        self.tabWidget.setGeometry(QtCore.QRect(10, 30, 351, 321))
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.consoleBrowser = QtWidgets.QTextBrowser(self.tab)
        self.consoleBrowser.setGeometry(QtCore.QRect(10, 10, 321, 271))
        self.consoleBrowser.setObjectName("consoleBrowser")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.lossView = QtWidgets.QGraphicsView(self.tab_2)
        self.lossView.setGeometry(QtCore.QRect(10, 10, 321, 271))
        self.lossView.setObjectName("lossView")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.performanceBrowser = QtWidgets.QTextBrowser(self.tab_3)
        self.performanceBrowser.setGeometry(QtCore.QRect(10, 10, 321, 271))
        self.performanceBrowser.setObjectName("performanceBrowser")
        self.tabWidget.addTab(self.tab_3, "")
        self.trainResultLabel = QtWidgets.QLabel(self.frame_2)
        self.trainResultLabel.setGeometry(QtCore.QRect(20, 10, 101, 16))
        self.trainResultLabel.setObjectName("trainResultLabel")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(50, 20, 731, 171))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayoutWidget = QtWidgets.QWidget(self.frame_3)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(14, 20, 641, 103))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.nbLayerBox = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.nbLayerBox.setObjectName("nbLayerBox")
        self.gridLayout.addWidget(self.nbLayerBox, 2, 1, 1, 1)
        self.nbLayer3Text = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.nbLayer3Text.setObjectName("nbLayer3Text")
        self.gridLayout.addWidget(self.nbLayer3Text, 2, 3, 1, 1)
        self.nbLayer3Label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.nbLayer3Label.setObjectName("nbLayer3Label")
        self.gridLayout.addWidget(self.nbLayer3Label, 2, 2, 1, 1)
        self.nbLayer2Label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.nbLayer2Label.setObjectName("nbLayer2Label")
        self.gridLayout.addWidget(self.nbLayer2Label, 1, 2, 1, 1)
        self.epochText = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.epochText.setObjectName("epochText")
        self.gridLayout.addWidget(self.epochText, 1, 1, 1, 1)
        self.nbLayerLabel = QtWidgets.QLabel(self.gridLayoutWidget)
        self.nbLayerLabel.setObjectName("nbLayerLabel")
        self.gridLayout.addWidget(self.nbLayerLabel, 2, 0, 1, 1)
        self.dataDirText = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.dataDirText.setObjectName("dataDirText")
        self.gridLayout.addWidget(self.dataDirText, 0, 1, 1, 1)
        self.nbLayer1Label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.nbLayer1Label.setObjectName("nbLayer1Label")
        self.gridLayout.addWidget(self.nbLayer1Label, 0, 2, 1, 1)
        self.nbLayer1Text = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.nbLayer1Text.setObjectName("nbLayer1Text")
        self.gridLayout.addWidget(self.nbLayer1Text, 0, 3, 1, 1)
        self.epochLabel = QtWidgets.QLabel(self.gridLayoutWidget)
        self.epochLabel.setObjectName("epochLabel")
        self.gridLayout.addWidget(self.epochLabel, 1, 0, 1, 1)
        self.dataDirLabel = QtWidgets.QLabel(self.gridLayoutWidget)
        self.dataDirLabel.setObjectName("dataDirLabel")
        self.gridLayout.addWidget(self.dataDirLabel, 0, 0, 1, 1)
        self.nbLayer2Text = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.nbLayer2Text.setObjectName("nbLayer2Text")
        self.gridLayout.addWidget(self.nbLayer2Text, 1, 3, 1, 1)
        self.trainModelButton = QtWidgets.QPushButton(self.frame_3)
        self.trainModelButton.setGeometry(QtCore.QRect(270, 130, 92, 23))
        self.trainModelButton.setObjectName("trainModelButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 794, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionQuit_2 = QtWidgets.QAction(MainWindow)
        self.actionQuit_2.setObjectName("actionQuit_2")
        self.actionHow_to_use = QtWidgets.QAction(MainWindow)
        self.actionHow_to_use.setObjectName("actionHow_to_use")
        self.actionAbout_us = QtWidgets.QAction(MainWindow)
        self.actionAbout_us.setObjectName("actionAbout_us")
        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionQuit_3 = QtWidgets.QAction(MainWindow)
        self.actionQuit_3.setObjectName("actionQuit_3")
        self.actionUsing_help = QtWidgets.QAction(MainWindow)
        self.actionUsing_help.setObjectName("actionUsing_help")

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Getting started with AI-Identifying squares and circles"))
        self.adjustLabel.setText(_translate("MainWindow", "Adjust "))
        self.selectImageButton.setText(_translate("MainWindow", "Select Image..."))
        self.recognizeButton.setText(_translate("MainWindow", "Recognize"))
        self.resultLabel.setText(_translate("MainWindow", "Recognition result"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Console"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Loss"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Performance"))
        self.trainResultLabel.setText(_translate("MainWindow", "Training results"))
        self.nbLayer3Label.setText(_translate("MainWindow", "Number of neurons in 3 layer"))
        self.nbLayer2Label.setText(_translate("MainWindow", "Number of neurons in 2 layer"))
        self.nbLayerLabel.setText(_translate("MainWindow", "Number of layers:"))
        self.nbLayer1Label.setText(_translate("MainWindow", "Number of neurons in 1 layer"))
        self.epochLabel.setText(_translate("MainWindow", "Number of iterations:"))
        self.dataDirLabel.setText(_translate("MainWindow", "Current dataset path:"))
        self.trainModelButton.setText(_translate("MainWindow", "Training model"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionQuit.setText(_translate("MainWindow", "Save As..."))
        self.actionQuit_2.setText(_translate("MainWindow", "Quit"))
        self.actionHow_to_use.setText(_translate("MainWindow", "Using help"))
        self.actionAbout_us.setText(_translate("MainWindow", "About us"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As..."))
        self.actionQuit_3.setText(_translate("MainWindow", "Quit"))
        self.actionUsing_help.setText(_translate("MainWindow", "Using help"))

# import img_rc

    def initialize(self):
        self.filename = ""
        self.dataDirText.setText("/Dataset/basicshapes")
        self.dataDirText.setDisabled(True)
        self.selectImageButton.clicked.connect(self.selectImageClicked)
        self.trainModelButton.clicked.connect(self.trainModelClicked)
        self.recognizeButton.clicked.connect(self.recognizeClicked)

        self.nbLayer2Text.setDisabled(True)
        self.nbLayer3Text.setDisabled(True)

        self.nbLayerBox.addItem('1')
        self.nbLayerBox.addItem('2')
        self.nbLayerBox.addItem('3')
        self.nbLayerBox.currentIndexChanged.connect(self.nbLayerBoxIndexChanged)
        sys.stdout = EmittingStream(textWritten=self.outputWritten)
        sys.stderr = EmittingStream(textWritten=self.outputWritten)

    def plotLoss(self):
        self.F = MyFigure(width=3, height=2, dpi=100)
        x = range(0, self.nn.num_epoch)
        self.F.axes.plot(x, self.nn.array_loss, label='loss')
        self.F.axes.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=3, mode="expand", borderaxespad=0.)

    def performance(self):
        str_train_accuracy = 'Accuracy of the trainset: {}\n'.format(self.nn.train_accuracy)
        str_train_precision = 'Precision of the trainset: {}\n'.format(self.nn.train_precision)
        str_train = str_train_accuracy + str_train_precision

        count_test_error = 0
        for data, label in zip(self.data_test, self.target_test):
            predict = self.nn.predict(data).tolist()
            if not label == round(predict[1][0]):
                count_test_error += 1
        str_test_error = 'Number of prediction errors in the test set = {}/40\n'.format(count_test_error)
        return str_train + str_test_error

    def nbLayerBoxIndexChanged(self):
        if self.nbLayerBox.currentText() == "1":
            self.nbLayer1Text.setDisabled(False)
            self.nbLayer2Text.setDisabled(True)
            self.nbLayer3Text.setDisabled(True)
        elif self.nbLayerBox.currentText() == "2":
            self.nbLayer1Text.setDisabled(False)
            self.nbLayer2Text.setDisabled(False)
            self.nbLayer3Text.setDisabled(True)
        elif self.nbLayerBox.currentText() == "3":
            self.nbLayer1Text.setDisabled(False)
            self.nbLayer2Text.setDisabled(False)
            self.nbLayer3Text.setDisabled(False)

    def selectImageClicked(self):

        filename, _ = QFileDialog.getOpenFileName(self, "Sélectionnez les images à reconnaître")
        self.filename = filename
        print("The path of the selected image:" + self.filename)

        self.imageView.setStyleSheet("border-image: url(" + filename + ");")

    def trainModelClicked(self):
        button = QMessageBox.question(self, "Reminder",
                                      self.tr(
                                          "The training process may take a long time, please wait patiently if the program does not respond"),
                                      QMessageBox.Ok | QMessageBox.Cancel,
                                      QMessageBox.Ok)
        if button == QMessageBox.Ok:
            self.trainModel()
        elif button == QMessageBox.Cancel:
            pass
        else:
            return

    def trainModel(self):
        # self.textEdit.setText("Training, please wait.")
        dataframe = pd.read_csv('newtest.csv')
        y = dataframe.iloc[0:200, 0].values
        y = np.where(y == 0, 0, 1)
        X = dataframe.iloc[0:200, 1:785].values
        for i in range(200):
            for j in range(784):
                X[i][j] = float(X[i][j])
        X_std = np.copy(X)
        X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
        X_train = np.array(X_std)
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(X_train, y,
                                                                                                test_size=0.2)
        # self.textEdit.setText("Training, please wait.")

        nbLayer1 = int(self.nbLayer1Text.text())
        nbLayer2 = 0
        nbLayer3 = 0
        if self.nbLayer2Text.text() != "":
            nbLayer2 = int(self.nbLayer2Text.text())
        if self.nbLayer3Text.text() != "":
            nbLayer3 = int(self.nbLayer3Text.text())

        self.nn = MLP_class.neural_network(self.data_train, self.target_train, self.nbLayerBox.currentIndex() + 2, 784,
                                           2, int(self.epochText.text()), nbLayer1, nbLayer2, nbLayer3, is_bias=True)
        self.nn.learning()

        np.set_printoptions(suppress=True, precision=2)

        self.plotLoss()
        self.scene = QGraphicsScene()  # 创建一个场景
        self.scene.addWidget(self.F)  # 将图形元素添加到场景中
        self.lossView.setScene(self.scene)  # 将创建添加到图形视图显示窗口
        self.performanceBrowser.setText(self.performance())
        self.saveModel()

    def recognizeClicked(self):
        if self.filename == "":
            QMessageBox.warning(self, "Warning", self.tr("You have to choose an image file"))
        elif self.filename[-3:] != 'png':
            QMessageBox.warning(self, "Warning", self.tr("The selected file must be an PNG type"))
        else:
            self.recognize()

    def recognize(self):
        if self.filename != "":
            img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
            # Picture tags
            row_data = []
            # Get picture pixels
            row_data.extend(img.flatten())
            # Write picture data to csv file
            predict = self.nn.predict(np.array(row_data)).tolist()
            if round(predict[1][0]) == 0:
                result = "circle"
            elif (round(predict[1][0]) == 1):
                result = "square"
            print("Prediction Result:" + result)
            self.resultText.setText(result)

    def outputWritten(self, text):
        cursor = self.consoleBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.consoleBrowser.setTextCursor(cursor)
        self.consoleBrowser.ensureCursorVisible()

    def saveModel(self):
        with open("model.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(self.nn.num_layer):
                if i == 0:
                    weight_size = self.nn.input_size
                elif i == 1:
                    weight_size = self.nn.nbneuron1
                elif i == 2:
                    weight_size = self.nn.nbneuron2
                for j in range(weight_size):
                    writer.writerow(self.nn.weights[i][j].tolist())

        print(self.nn.weights)
        print(self.nn.train_accuracy)
        print(self.nn.train_precision)
        print(self.nn.is_bias)
