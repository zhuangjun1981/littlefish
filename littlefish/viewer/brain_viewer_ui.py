# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'brain_viewer_ui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_BrainViewerMainWindow(object):
    def setupUi(self, BrainViewerMainWindow):
        BrainViewerMainWindow.setObjectName("BrainViewerMainWindow")
        BrainViewerMainWindow.resize(351, 219)
        self.centralwidget = QtWidgets.QWidget(BrainViewerMainWindow)
        self.centralwidget.setObjectName("ToolbarWidget")
        self.ChooseFileButton = QtWidgets.QPushButton(self.centralwidget)
        self.ChooseFileButton.setGeometry(QtCore.QRect(90, 140, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.ChooseFileButton.setFont(font)
        self.ChooseFileButton.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.ChooseFileButton.setObjectName("ChooseFileButton")
        self.NoticeBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.NoticeBrowser.setGeometry(QtCore.QRect(20, 20, 321, 101))
        self.NoticeBrowser.setObjectName("NoticeBrowser")
        BrainViewerMainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(BrainViewerMainWindow)
        self.statusbar.setObjectName("statusbar")
        BrainViewerMainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(BrainViewerMainWindow)
        QtCore.QMetaObject.connectSlotsByName(BrainViewerMainWindow)

    def retranslateUi(self, BrainViewerMainWindow):
        _translate = QtCore.QCoreApplication.translate
        BrainViewerMainWindow.setWindowTitle(_translate("BrainViewerMainWindow", "Brain Viewer"))
        self.ChooseFileButton.setText(_translate("BrainViewerMainWindow", "Choose File"))
        self.NoticeBrowser.setHtml(_translate("BrainViewerMainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt;\">Please select a .hdf5 file containing at least one group with name prefix &quot;fish&quot;. These groups should contain the standard data structure of </span><span style=\" font-size:11pt; color:#0055ff;\">littlefish.core.fish.Fish</span><span style=\" font-size:11pt;\"> class. Idealy created by </span><span style=\" font-size:11pt; color:#0055ff;\">littlefish.core.fish.Fish.to_h5_group()</span><span style=\" font-size:11pt;\"> method.</span></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    BrainViewerMainWindow = QtWidgets.QMainWindow()
    ui = Ui_BrainViewerMainWindow()
    ui.setupUi(BrainViewerMainWindow)
    BrainViewerMainWindow.show()
    sys.exit(app.exec_())

