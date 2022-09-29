import os
import re
import shutil
import threading
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, QRect, Qt, QSize, QRegExp, QFileInfo
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QRegExpValidator
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel, QPushButton, QMainWindow, QFileDialog, \
    QMessageBox
import Model

from BeltDetect_person import BeltDetect_person
from tools import Dao
from Core import Detect, SmokeDetect
from PersonCount import PersonCount
from SecurityDetect import SecurityDetect
from SmokeDetect import SmokeDetect
from FenceDetect import FenceDetect
from Head_new import Head
from EleCheck import EleCheck
from BeltDetect import BeltDetect
from ui.AddCameraWidget import Ui_AddCameraWidget
from ui.AddModel import Ui_AddModel
from ui.Connection import Ui_Connection
from ui.Connections import Ui_Connections
from ui.DeleteResource import Ui_DeleteResource
from ui.MainWindow import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def closeEvent(self, event):
        """
        重写closeEvent方法，实现dialog窗体关闭时执行一些代码
        :param event: close()触发的事件
        :return: None
        """
        reply = QMessageBox.question(self,
                                     '本程序',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            os.system('taskkill /IM python.exe /F')
        else:
            event.ignore()


class AddModelWidget(QWidget, Ui_AddModel):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 设置添加模型窗口里各个按钮的动作
        self.nameBox.setValidator(QRegExpValidator(QRegExp('[0-9a-zA-Z]*'), self.nameBox))
        self.selectBtn.clicked.connect(lambda: self.getFile())
        self.saveSelectBtn.clicked.connect(lambda: self.getDIR())
        self.cancelBtn.clicked.connect(lambda: self.cancel())
        self.comfirmBtn.clicked.connect(lambda: self.comfirm())
        self.DIRBox.editingFinished.connect(lambda: self.autoFill())
        # self.DIRBox.setFocus()
        self.saveDIRBox.setPlaceholderText("未输入则使用默认路径")
        self.defaultDIR = './models/detect_model/'
        self.setWindowModality(Qt.ApplicationModal)

    def getFile(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(), "Model Files(*.pt)")
        if '' != fileName:
            self.DIRBox.setText(fileName)
        self.autoFill()

    def getDIR(self):
        saveDIR = QFileDialog.getExistingDirectory(self, "选择文件夹", "/")
        if '' != saveDIR:
            self.saveDIRBox.setText(saveDIR)

    def autoFill(self):
        url = self.DIRBox.text()
        if '' != url:
            modelName = re.findall(r"/([^/]+?)\.pt$", url)
            if len(modelName):
                modelName = modelName[0]
                self.nameBox.setText(modelName)

    def comfirm(self):
        if '' == self.DIRBox.text():
            QMessageBox(QMessageBox.Warning, 'Warning', '未选择文件!').exec_()
            return
        if '' == self.nameBox.text():
            QMessageBox(QMessageBox.Warning, 'Warning', '未输入模型名称!').exec_()
            return
        if '' == self.saveDIRBox.text():
            self.saveDIRBox.setText(self.defaultDIR)

        url = self.DIRBox.text()
        file = QFileInfo(url)
        if not file.isFile():
            QMessageBox(QMessageBox.Warning, 'Warning', '文件不存在!').exec_()
            return
        if url[len(url) - 3:len(url)] != '.pt':
            QMessageBox(QMessageBox.Warning, 'Warning', '文件类型错误!').exec_()
            return

        url = self.saveDIRBox.text()
        DIR = QFileInfo(url)
        if not DIR.isDir():
            QMessageBox(QMessageBox.Warning, 'Warning', '路径不存在!').exec_()
            return
        temp = [self.DIRBox.text(),
                self.saveDIRBox.text() + self.nameBox.text() + '.pt']
        shutil.copyfile(*temp)
        temp = [self.nameBox.text(),
                self.saveDIRBox.text(),
                self.commentBox.text()]
        model = Model.Model()
        model.setData(None, *temp)
        model.insert()
        QMessageBox.about(self, 'Success', '模型添加成功!')
        self.close()

    def cancel(self):
        self.close()


class AddCameraWidget(QWidget, Ui_AddCameraWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.webCamera.toggled.connect(lambda: self.setRegExp(self.webCamera))
        self.localCamera.toggled.connect(lambda: self.setRegExp(self.localCamera))
        self.webCamera.setChecked(True)
        self.cancelBtn.clicked.connect(self.cancel)
        self.comfirmBtn.clicked.connect(self.comfirm)
        self.addressEdit.setFocus()
        self.setWindowModality(Qt.ApplicationModal)

    def setRegExp(self, btn):
        self.addressEdit.clear()
        if self.webCamera.isChecked() and btn.text() == "网络摄像头":
            inputExp = QRegExp(r'rtsp:\/\/[0-9a-zA-Z]*:[0-9a-zA-Z]*@'
                               r'((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})'
                               r'(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}:[0-9]*'
                               r'\/[a-zA-Z0-9]*\/[a-zA-Z0-9]*\/[0-9]*\?.*')
            self.addressEdit.setValidator(QRegExpValidator(inputExp, self.addressEdit))
        if self.localCamera.isChecked() and btn.text() == "本地摄像头":
            inputExp = QRegExp(r'\d{0,1}')
            self.addressEdit.setValidator(QRegExpValidator(inputExp, self.addressEdit))

    def comfirm(self):
        if '' == self.addressEdit.text():
            QMessageBox(QMessageBox.Warning, 'Warning', '未输入摄像头链接!').exec_()
            return
        if '' == self.nameEdit.text():
            QMessageBox(QMessageBox.Warning, 'Warning', '未输入摄像头名称!').exec_()
            return
        camera = Model.Camera()
        temp = [self.addressedit.text(),
                self.nameEdit.text(),
                self.orderBox.text(),
                self.commentEdit.text()]
        camera.set(None, *temp)
        camera.insert()
        QMessageBox.about(self, 'Success', '摄像头添加成功!')
        self.close()

    def cancel(self):
        self.close()


'''
class infractionQueryWidget(QWidget, Ui_infractionQuery):

'''


class Connection(QWidget, Ui_Connection):
    cameras = {}
    models = {}
    unusedModel = {}

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.cameraBox.addItems(self.cameras.values())
        self.cameraBox.currentIndexChanged.connect(self.setModelBox)
        self.cameraBox.setCurrentIndex(0)
        self.setModelBox()
        self.commentEdit.setPlaceholderText("备注")
        self.setWindowModality(Qt.ApplicationModal)

    @classmethod
    def getData(cls):
        cameraUrl = r'select `cameraID`,`name` from camera'
        modelUrl = r'select `modelID`,`modelName` from model'
        temp = Dao.query(cameraUrl)
        for i in temp:
            cls.cameras[i[0]] = i[1]
            unusedUrl = f'select `modelName` from model where not exists' \
                        f'(select model.`modelID` from detect where ' \
                        f'model.`modelID` = detect.`modelID` and detect.`cameraID` = {i[0]})'
            result = Dao.query(unusedUrl)
            cls.unusedModel[i[1]] = result
        temp = Dao.query(modelUrl)
        for i in temp:
            cls.models[i[0]] = i[1]

    def setData(self, cameraID, modelID, status, comment):
        cameraName = self.cameras[cameraID]
        modelName = self.models[modelID]
        index = self.cameraBox.findText(cameraName)
        self.cameraBox.setCurrentIndex(index)
        self.cameraBox.setEnabled(False)
        self.modelBox.addItems(self.models.values())
        index = self.modelBox.findText(modelName)
        self.modelBox.setCurrentIndex(index)
        self.modelBox.setEnabled(False)
        if status == 1:
            self.statusBox.setChecked(True)
        self.commentEdit.setText(comment)

    def setModelBox(self):
        self.modelBox.clear()
        cameraName = self.cameraBox.currentText()
        modelName = self.unusedModel[cameraName]
        for i in modelName:
            self.modelBox.addItem(i[0])
        self.modelBox.setCurrentIndex(0)

    def insert(self):
        detect = Model.Detect()
        temp = [list(self.cameras.keys())[list(self.cameras.values()).index(self.cameraBox.currentText())],
                list(self.models.keys())[list(self.models.values()).index(self.modelBox.currentText())],
                1 if self.statusBox.isChecked() else 0,
                self.commentEdit.text()]
        detect.setData(*temp, None)
        detect.insert()


class Connections(QWidget, Ui_Connections):
    def __init__(self, superWindow):
        super().__init__()
        self.setupUi(self)
        self.superWindow = superWindow
        self.count = 0
        self.contentArea = QScrollArea(self)
        self.contentArea.setGeometry(QRect(10, 50, 380, 240))
        self.contentArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.contentWidget = QWidget()
        self.contentWidget.setGeometry(0, 0, 358, 0)
        self.contentWidgetLayout = QVBoxLayout(self.contentWidget)
        self.contentArea.setWidget(self.contentWidget)
        self.addBtn.clicked.connect(lambda: self.addConnection(Connection()))
        self.cancelBtn.clicked.connect(lambda: self.cancel())
        self.comfirmBtn.clicked.connect(lambda: self.comfirm())
        Connection.getData()
        self.setData()
        self.setWindowModality(Qt.ApplicationModal)

    def addConnection(self, con):
        self.count += 1
        self.contentWidget.setGeometry(QRect(0, 0, 358, self.count * 80))
        self.contentWidgetLayout.addWidget(con)

    def setData(self):
        detectUrl = r'select `cameraID`,`modelID`,`status`,`comment` from detect'
        temp = Dao.query(detectUrl)
        for i in temp:
            con = Connection()
            con.setData(*i)
            self.addConnection(con)

    def comfirm(self):
        self.superWindow.detects = self.superWindow.getData()
        for i in range(self.contentWidgetLayout.count()):
            self.contentWidgetLayout.itemAt(i).widget().insert()
        self.superWindow.update()
        self.close()

    def cancel(self):
        self.close()


class DeleteResource(QWidget, Ui_DeleteResource):
    def __init__(self, superWindow):
        super().__init__()
        self.superWindow = superWindow
        self.setupUi(self)
        self.delCameraBtn.clicked.connect(lambda: self.delCamera())
        self.delModelBtn.clicked.connect(lambda: self.delModel())
        self.delConnnectionBtn.clicked.connect(lambda: self.delConnection())
        self.setData()
        self.setWindowModality(Qt.ApplicationModal)

    def setData(self):
        cameraUrl = f'select `name` from camera'
        modelUrl = f'select `modelName` from model'
        detectUrl = f'select `name`,`modelName` from show_info'
        self.cameraBox.clear()
        self.modelBox.clear()
        self.connectionBox.clear()
        temp = Dao.query(cameraUrl)
        for i in temp:
            self.cameraBox.addItem(i[0])
        temp = Dao.query(modelUrl)
        for i in temp:
            self.modelBox.addItem(i[0])
        temp = Dao.query(detectUrl)
        for i in temp:
            self.connectionBox.addItem(i[0] + '_' + i[1])

    def delCamera(self):
        delUrl = f'delete from camera where `name`=\'{self.cameraBox.currentText()}\''
        Dao.delete(delUrl)
        self.superWindow.update()
        self.setData()

    def delModel(self):
        delUrl = f'delete from model where `modelName`=\'{self.modelBox.currentText()}\''
        Dao.delete(delUrl)
        self.superWindow.update()
        self.setData()

    def delConnection(self):
        text = self.connectionBox.currentText()
        text = text.split('_')
        delUrl = f'delete from detect where exists ' \
                 f'(select * from camera,model where ' \
                 f'camera.`name`=\'{text[0]}\' and camera.`cameraID`=detect.`cameraID` ' \
                 f'and model.`modelName`=\'{text[1]}\' and model.`modelID`=detect.`modelID`)'
        Dao.delete(delUrl)
        self.superWindow.update()
        self.setData()


class ShowLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cameraList = []
        self.currentCamera = None
        self.fence = FenceSetting()
        self.setText("")
        self.tmepCamera = None

    def setCameras(self, cameraList):
        self.cameraList = cameraList
        self.currentCamera = self.cameraList[0]
        self.currentCamera.setStatus(isShow=True)

    def showImage(self, image):
        image = image.scaled(self.width(), self.height())
        self.setPixmap(QPixmap.fromImage(image))

    def setVideo(self, camera):
        if self.currentCamera is not None:
            self.currentCamera.setStatus(isShow=False)
        self.currentCamera = camera
        self.currentCamera.setStatus(isShow=True)

    def mousePressEvent(self, event):
        if self.currentCamera is not None:
            self.currentCamera.setFence(self.fence)

    def liberateCamera(self, cameraId):
        for i in self.cameraList:
            if id(i) == cameraId:
                self.cameraList.remove(i)


class CameraList:
    def __init__(self, cameraInfoList, showLabel):
        self.cameraList = []
        self.cameraInfoList = cameraInfoList
        self.showLabel = showLabel

    def addCamera(self, main_window, model, modelDIR, url=None, name='', function='', count=''):
        camera = Camera(main_window, self.showLabel, model, modelDIR, url, name, function, count)
        self.cameraList.append(camera)
        if camera.isActive:
            cameraInfo = CameraInfo(camera, self.cameraInfoList)
            self.cameraInfoList.addCameraInfo(cameraInfo)

    def getCameras(self):
        return self.cameraList


class CameraInfo(QWidget):
    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.camera = camera
        self.resize(185, 100)
        self.verticalLayout = QVBoxLayout(self)
        self.cameraName = QLabel(self)
        self.function = QLabel(self)
        self.count = QLabel(self)
        self.verticalLayout.addWidget(self.cameraName)
        self.verticalLayout.addWidget(self.function)
        self.verticalLayout.addWidget(self.count)
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setInfo()

    def setInfo(self):
        self.cameraName.setText(self.camera.name)
        self.function.setText(self.camera.function)
        self.count.setText(self.camera.count)

    def mousePressEvent(self, event):
        for i in self.parent.cameraInfos:
            i.setBackground(240, 240, 240)
        self.setBackground(200, 200, 200)
        self.camera.showLabel.setVideo(self.camera)

    def setBackground(self, r, g, b):
        temp = self.palette()
        temp.setColor(QPalette.Background, QColor(r, g, b))
        self.setPalette(temp)

    def judge(self, url, model):
        if self.camera.url == url and self.camera.model == model:
            return True
        else:
            return False

    def deleteCamera(self):
        self.camera.liberateFromList()


# 打开摄像头
class Camera:
    def __init__(self, main_window, showLabel, model, modelDIR, url=None, name='', function='', count=''):
        self.showLabel = showLabel
        self.model = model
        self.modelDIR = modelDIR
        self.url = url
        self.name = name
        self.function = function
        self.count = count
        self.mask = Mask(name, model)
        self.mask.loadMask()
        self.timeCamera = QTimer()
        self.timeCamera.timeout.connect(self.show)
        self.isActive = False
        self.isShow = False
        self.detectThread = None
        self.active(main_window)

    def getImage(self, returnImage):
        image = self.detectThread.getImage()
        if not returnImage:
            return image
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[0:2]
            qImage = QImage(image.data, width, height, QImage.Format_RGB888)
            return qImage

    def setStatus(self, isShow):
        if isShow:
            self.isShow = True
        else:
            self.isShow = False

    def active(self, main_window):
        file = QFileInfo(self.modelDIR + '\\' + self.model + '.pt')
        if not file.isFile():
            print(self.modelDIR, self.model)
            QMessageBox(QMessageBox.Warning, 'Warning', f'{self.model}模型不存在!').exec_()
            Dao.update(f'update detect set detect.`status`=0 where `modelID` in '
                       f'(select `modelID` from model where '
                       f'`modelDIR`=\'{self.modelDIR}\')')
            return

        # 开启摄像头
        cap = cv2.VideoCapture(self.url)
        if cap is None and not cap.isOpened():
            QMessageBox(QMessageBox.Warning, 'Warning', f'{self.name}不可用!').exec_()
            Dao.update(f'update detect set detect.`status`=0 where `cameraID` in '
                       f'(select `cameraID` from camera where '
                       f'`cameraUrl`=\'{self.url}\')')
            cap.release()
            return
        cap.release()
        self.isActive = True
        self.detectThread = DetectThread(main_window, self.url, self.model, self.modelDIR, self.mask)
        self.detectThread.start()
        self.timeCamera.start(1)

    def show(self):
        if self.isShow:
            self.showLabel.showImage(self.getImage(True))

    def setFence(self, fence):
        fence.popup(self.getImage(False), self.mask)

    def liberateFromList(self):
        self.timeCamera.stop()
        self.timeCamera.deleteLater()
        self.detectThread.stop()
        self.showLabel.liberateCamera(id(self))


class CameraInfoList(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cameraInfos = []
        self.count = 0
        self.setMinimumSize(QSize(0, 0))
        self.setLineWidth(0)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.contentWidget = QWidget()
        self.contentWidget.setGeometry(QRect(0, 0, 189, 0))
        self.contentWidgetLayout = QVBoxLayout(self.contentWidget)
        self.setWidget(self.contentWidget)

    def addCameraInfo(self, cameraInfo):
        self.cameraInfos.append(cameraInfo)
        self.count += 1
        self.contentWidgetLayout.addWidget(cameraInfo)
        self.contentWidget.setGeometry(QRect(0, 0, 189, self.count * 100))

    def deleteCameraInfo(self, cameraUrl, modelName):
        for i in self.cameraInfos[::-1]:
            if i.judge(cameraUrl, modelName):
                i.deleteCamera()
                self.count -= 1
                self.contentWidget.setGeometry((QRect(0, 0, 189, self.count * 100)))
                i.setParent(None)
                self.contentWidgetLayout.removeWidget(i)
                self.cameraInfos.remove(i)
                i.close()


# 用来实现检测的类



class DetectThread(threading.Thread):
    def __init__(self, main_window, url, model, modelDIR, mask):
        threading.Thread.__init__(self)
        self.main_window = main_window
        self.url = url
        self.model = model
        self.modelDIR = modelDIR
        self.isStop = False
        self.detect = None
        self.daemon = True
        self.mask = mask
        # self.weights = './models/detect_model/yolov5s.pt'
        #
        # self.device = '0' if torch.cuda.is_available() else 'cpu'
        # self.device = select_device(self.device)
        # self.model = attempt_load(self.weights, map_location=self.device)

    def run(self):
        # 1：电子围栏违规，2：安全帽违规，3：绝缘手套违规，4：跌倒，5：绝缘靴违规，6：抽烟违规
        print(f'running weight {self.model}')
        if self.model == 'smoke':
            target = []
            self.detect = SmokeDetect(self, target)
        elif self.model == "all":
            target = ['person', 'worker', 'glove', 'shoes', 'helmet']
            self.detect = SecurityDetect(self, target)
        elif self.model == "fence":
            target = ['fence']
            self.detect = FenceDetect(self, target)
        elif self.model == "person":
            self.detect = PersonCount(self)
        elif self.model == "Head":
            target = ["Head"]
            self.detect = Head(self, self.main_window, target)
        elif self.model == "belt1":
            target = ["c1", "c2", "c3", "c4", "c5",'person']
            # self.detect =BeltDetect(self,target)
            self.detect =BeltDetect_person(self, target)

        else:
            self.detect = Detect(self)
        opt = self.detect.parse_opt()
        print("opt-------------", opt)
        self.detect.execDetect(opt)

    def getImage(self):
        return self.detect.getImage()

    def stop(self):
        self.isStop = True


# 电子围栏
class FenceSetting(QWidget):
    def __init__(self):
        super().__init__()
        # points 多边形点的集合列表

        # image 从当前摄像头获取的一帧图像
        # mask 就是生成的 mask 掩码图像
        self.points = []
        self.pointsArray = []
        self.copy_pointsArray = []
        self.image = None
        self.mask = None

        self.resize(500, 400)
        self.setMinimumSize(QSize(500, 400))
        self.setMaximumSize(QSize(500, 400))
        self.setWindowTitle('设置电子围栏')

        self.showArea = QWidget(self)
        self.showArea.setGeometry(QRect(2, 0, 496, 355))
        self.showLabel = QLabel(self.showArea)
        self.showLabel.setGeometry(2, 2, 496, 355)

        self.cancelBtn = QPushButton(self)
        self.cancelBtn.setGeometry(QRect(310, 365, 75, 25))
        self.cancelBtn.setText('撤  销')
        # 绑定响应方法
        self.cancelBtn.clicked.connect(self.cancelPoint)

        self.comfirmBtn = QPushButton(self)
        self.comfirmBtn.setGeometry(QRect(400, 365, 75, 25))
        self.comfirmBtn.setText('确  认')
        # 绑定响应方法
        self.comfirmBtn.clicked.connect(self.comfirmPoints)

        self.setWindowModality(Qt.ApplicationModal)

    # 取消点击的点
    def cancelPoint(self):
        if len(self.pointsArray) == 0:
            # QWidget.close
            self.close()
        else:
            if len(self.pointsArray[len(self.pointsArray) - 1]) == 0:
                self.pointsArray.pop()
            else:
                self.pointsArray[len(self.pointsArray) - 1].pop()
                image = self.mask.settingFence(self.image.copy(), False, self.pointsArray)
                height, width = image.shape[0:2]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = QImage(image.data, width, height, QImage.Format_RGB888)
                image = image.scaled(self.showLabel.width(), self.showLabel.height())
                self.showLabel.setPixmap(QPixmap.fromImage(image))

    # 确认键的响应方法
    def comfirmPoints(self):
        self.mask.clear()
        temp = np.zeros([480, 640, 3], dtype=np.uint8)
        self.copy_pointsArray = self.pointsArray.copy()
        self.mask.settingFence(temp, True, self.pointsArray, (255, 255, 255))
        self.mask.setDataList(temp, self.pointsArray)
        self.mask.storeMask()
        self.close()

    # 鼠标监听函数
    # if (event.button() == 1):
    #     print("左键")
    # if(event.button() == 2 ):
    #     print("右键")\
    # 鼠标按压事件 画电子围栏
    def mousePressEvent(self, event):
        print("self.pointsArray", self.pointsArray)
        point = [event.x() / self.showLabel.width(), event.y() / self.showLabel.height()]
        if (event.button() == 1):
            # print("左键")
            if 1 < event.x() < 497 and event.y() < 366:
                if len(self.pointsArray) == 0:
                    self.pointsArray.append([])
                self.pointsArray[len(self.pointsArray) - 1].append(point)
            image = self.mask.settingFence(self.image.copy(), False, self.pointsArray)  # 在图片上画电子围栏
            # 设置在标签上
            height, width = image.shape[0:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QImage(image.data, width, height, QImage.Format_RGB888)
            image = image.scaled(self.showLabel.width(), self.showLabel.height())
            self.showLabel.setPixmap(QPixmap.fromImage(image))
        if (event.button() == 2):
            # print("右键")
            if len(self.pointsArray[len(self.pointsArray) - 1]) > 0:
                self.pointsArray.append([])
                self.points.clear()

    # 弹出窗口
    def popup(self, image, mask):
        self.points = []
        self.pointsArray = []
        self.mask = mask
        self.image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[0:2]
        image = QImage(image.data, width, height, QImage.Format_RGB888)
        image = image.scaled(self.showLabel.width(), self.showLabel.height())
        self.showLabel.setPixmap(QPixmap.fromImage(image))
        self.show()


class FenceTest(QWidget):
    def __init__(self):
        super().__init__()
        # points 多边形点的集合列表

        # image 从当前摄像头获取的一帧图像
        # mask 就是生成的 mask 掩码图像
        self.points = []
        self.pointsArray = []
        self.image = None
        self.mask = None

        self.resize(500, 400)
        self.setMinimumSize(QSize(500, 400))
        self.setMaximumSize(QSize(500, 400))
        self.setWindowTitle('设置电子围栏')

        self.showArea = QWidget(self)
        self.showArea.setGeometry(QRect(2, 0, 496, 355))
        self.showLabel = QLabel(self.showArea)
        self.showLabel.setGeometry(2, 2, 496, 355)

        self.cancelBtn = QPushButton(self)
        self.cancelBtn.setGeometry(QRect(310, 365, 75, 25))
        self.cancelBtn.setText('撤  销')

        self.comfirmBtn = QPushButton(self)
        self.comfirmBtn.setGeometry(QRect(400, 365, 75, 25))
        self.comfirmBtn.setText('确  认')

        self.setWindowModality(Qt.ApplicationModal)

    def popup(self):
        self.show()


class Mask:
    def __init__(self, cameraName, modelName):
        self.cameraName = cameraName
        self.modelName = modelName
        self.mask = np.zeros([480, 640, 3], dtype=np.uint8)
        self.mask.fill(255)
        self.points = []
        self.pointsList = []

    # 清除 mask
    def clear(self):
        # 用白色填充 mask
        self.mask = np.zeros([480, 640, 3], dtype=np.uint8)
        # 多边形点集列表清空
        self.points = []
        self.pointsList = []

    # 从目录中获取已存储的 mask 和点集
    def setData(self, mask, points):
        self.mask = mask
        self.points = points

    def setDataList(self, mask, pointsList):
        self.mask = mask
        self.pointsList = pointsList

    # 生成响应大小的 mask
    def getMask(self, isScale=True, scale=(480, 640)):
        img = self.mask.copy()
        if isScale:
            img = cv2.resize(img, scale)
        img = np.rollaxis(img, 2, 0)
        img = np.expand_dims(img, axis=0)
        return img

    # 生成响应大小的 mask
    def getOrginMask(self, isScale=True, scale=(480, 640)):
        img = self.mask.copy()
        if isScale:
            img = cv2.resize(img, scale)
        return img

    # 画出围栏
    @staticmethod
    def drawFence(image, isFill, points0, color=(0, 0, 255)):
        height, width = image.shape[0:2]
        points = np.array([points0], np.float32)
        if points.size:
            points[:, :, 0] *= width
            points[:, :, 1] *= height
        points = points.astype(np.int32)
        if not isFill:
            if points.size == 2:  # size用来计算数组中所有元素的个数
                x, y = points[:, :, :][0][0]
                temp = [[x - 5, y - 5], [x - 5, y + 5], [x + 5, y + 5], [x + 5, y - 5]]
                temp = np.array([temp], np.int32)
                cv2.polylines(image, temp, True, color, 2)
            else:
                cv2.polylines(image, points, True, color, 2)  # 这个是画多边形的函数
        else:
            if points.size:
                cv2.fillPoly(image, points, color)
            else:
                image = image.fill(255)
        return image

    # 画电子围栏
    @staticmethod
    def settingFence(image, isFill, pointsArray, color=(0, 0, 255)):
        height, width = image.shape[0:2]
        if len(pointsArray) > 0:
            for i in range(0, len(pointsArray)):
                points = np.array([pointsArray[i]], np.float32)
                if points.size:
                    points[:, :, 0] *= width
                    points[:, :, 1] *= height
                points = points.astype(np.int32)
                if not isFill:
                    if points.size == 2:  # size用来计算数组中所有元素的个数
                        x, y = points[:, :, :][0][0]
                        # print(points[:, :, :])
                        temp = [[x - 5, y - 5], [x - 5, y + 5], [x + 5, y + 5], [x + 5, y - 5]]
                        temp = np.array([temp], np.int32)
                        cv2.polylines(image, temp, True, color, 2)
                    else:
                        cv2.polylines(image, points, True, color, 2)  # 这个是画多边形的函数
                else:
                    if points.size:
                        cv2.fillPoly(image, points, color)
                    else:
                        image = image.fill(255)
        return image

    # 调试使用的方法，展示 mask
    def show(self):
        cv2.imshow('mask', self.mask)
        cv2.waitKey(-1)

    # 存储围栏图片和点集信息
    def storeMask(self):
        storeDIR = '.\\Fence'
        fileName = f'{self.cameraName}_{self.modelName}'
        DIR = QFileInfo(storeDIR)
        if not DIR.isDir():
            QMessageBox(QMessageBox.Warning, 'Warning', '路径不存在!').exec_()
            return
        from PIL import Image
        im = Image.fromarray(self.mask)
        im.save(f'{storeDIR}\\{fileName}.jpg')
        import pickle
        with open(f'{storeDIR}\\{fileName}.pkl', 'wb') as f:
            pickle.dump(self.pointsList, f)

    # 加载 mask 和点集
    def loadMask(self):
        storeDIR = '.\\Fence'
        fileName = f'{self.cameraName}_{self.modelName}'
        file1 = QFileInfo(f'{storeDIR}\\{fileName}.jpg')
        file2 = QFileInfo(f'{storeDIR}\\{fileName}.pkl')
        if not file1.isFile() or not file2.isFile():
            return
        import pickle
        with open(f'{storeDIR}\\{fileName}.pkl', 'rb') as f:
            self.pointsList = pickle.load(f)
        from PIL import Image
        fence = Image.open(f'{storeDIR}\\{fileName}.jpg')
        self.mask = np.array(fence)


if __name__ == "__main__":
    print("Widgets Module.")
