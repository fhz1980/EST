# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DeleteResourse.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DeleteResource(object):
    def setupUi(self, deleteResourse):
        deleteResourse.setObjectName("DeleteResourse")
        deleteResourse.resize(350, 140)
        deleteResourse.setMinimumSize(QtCore.QSize(350, 140))
        deleteResourse.setMaximumSize(QtCore.QSize(350, 140))
        self.cameraLabel = QtWidgets.QLabel(deleteResourse)
        self.cameraLabel.setGeometry(QtCore.QRect(20, 20, 60, 20))
        self.cameraLabel.setObjectName("cameraLabel")
        self.modelLabel = QtWidgets.QLabel(deleteResourse)
        self.modelLabel.setGeometry(QtCore.QRect(20, 60, 60, 20))
        self.modelLabel.setObjectName("modelLabel")
        self.connectionLabel = QtWidgets.QLabel(deleteResourse)
        self.connectionLabel.setGeometry(QtCore.QRect(20, 100, 60, 20))
        self.connectionLabel.setObjectName("connectionLabel")
        self.cameraBox = QtWidgets.QComboBox(deleteResourse)
        self.cameraBox.setGeometry(QtCore.QRect(100, 20, 130, 24))
        self.cameraBox.setObjectName("cameraBox")
        self.modelBox = QtWidgets.QComboBox(deleteResourse)
        self.modelBox.setGeometry(QtCore.QRect(100, 60, 130, 24))
        self.modelBox.setObjectName("modelBox")
        self.connectionBox = QtWidgets.QComboBox(deleteResourse)
        self.connectionBox.setGeometry(QtCore.QRect(100, 100, 130, 24))
        self.connectionBox.setObjectName("connectionBox")
        self.delCameraBtn = QtWidgets.QPushButton(deleteResourse)
        self.delCameraBtn.setGeometry(QtCore.QRect(250, 20, 75, 24))
        self.delCameraBtn.setObjectName("delCameraBtn")
        self.delModelBtn = QtWidgets.QPushButton(deleteResourse)
        self.delModelBtn.setGeometry(QtCore.QRect(250, 60, 75, 24))
        self.delModelBtn.setObjectName("delModelBtn")
        self.delConnnectionBtn = QtWidgets.QPushButton(deleteResourse)
        self.delConnnectionBtn.setGeometry(QtCore.QRect(250, 100, 75, 24))
        self.delConnnectionBtn.setObjectName("delConnnectionBtn")

        self.retranslateUi(deleteResourse)
        QtCore.QMetaObject.connectSlotsByName(deleteResourse)

    def retranslateUi(self, deleteResourse):
        _translate = QtCore.QCoreApplication.translate
        deleteResourse.setWindowTitle(_translate("deleteResourse", "删除资源"))
        self.cameraLabel.setText(_translate("deleteResourse", "摄 像 头"))
        self.modelLabel.setText(_translate("deleteResourse", "模    型"))
        self.connectionLabel.setText(_translate("deleteResourse", "匹    配"))
        self.delCameraBtn.setText(_translate("deleteResourse", "删除"))
        self.delModelBtn.setText(_translate("deleteResourse", "删除"))
        self.delConnnectionBtn.setText(_translate("deleteResourse", "删除"))