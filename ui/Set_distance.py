import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication


class Ui_setDistance(QWidget):
    def __init__(self, main_window):
        super(Ui_setDistance, self).__init__()
        self.main_window = main_window
        self.setObjectName("Form")
        self.resize(567, 392)
        self.widget = QtWidgets.QWidget(self)
        self.widget.setGeometry(QtCore.QRect(11, 11, 204, 37))
        self.widget.setObjectName("widget")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(11, 11, 182, 16))
        self.label.setObjectName("label")
        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setGeometry(QtCore.QRect(10, 50, 511, 81))
        self.groupBox.setObjectName("groupBox")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(30, 20, 451, 51))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setGeometry(QtCore.QRect(150, 330, 211, 51))
        self.pushButton.setObjectName("pushButton")
        self.groupBox_2 = QtWidgets.QGroupBox(self)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 140, 511, 81))
        self.groupBox_2.setObjectName("groupBox_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_2.setGeometry(QtCore.QRect(30, 20, 451, 51))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 230, 511, 81))
        self.groupBox_3.setObjectName("groupBox_3")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_3.setGeometry(QtCore.QRect(30, 20, 451, 51))
        self.lineEdit_3.setObjectName("lineEdit_3")

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

        # 设置触发事件
        self.pushButton.clicked[bool].connect(self.change_distance)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "设置验电距离"))
        self.label.setText(_translate("Form", "验电距离: (默认单位：m）"))
        self.groupBox.setTitle(_translate("Form", "验电距离一"))
        self.pushButton.setText(_translate("Form", "确认设置"))
        self.groupBox_2.setTitle(_translate("Form", "验电距离二"))
        self.groupBox_3.setTitle(_translate("Form", "验电距离三"))

    def change_distance(self):
        text1 = self.lineEdit.text()
        text2 = self.lineEdit_2.text()
        text3 = self.lineEdit_3.text()

        print(text1.isdecimal())
        if self.is_number(text1) and self.is_number(text2) and self.is_number(text3) and text1 and text2 and text3:
            self.main_window.range_distance_main1 = float(text1)
            self.main_window.range_distance_main2 = float(text2)
            self.main_window.range_distance_main3 = float(text3)
            print("验电距离", self.main_window.range_distance_main1, self.main_window.range_distance_main2, self.main_window.range_distance_main3)



    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False
