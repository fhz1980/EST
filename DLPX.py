import sys
import threading
import time
import cv2
import os
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication
from tools import sleeptime
from Core import Detect
from MyWidgets import CameraInfoList, ShowLabel, CameraList, AddCameraWidget, MainWindow, AddModelWidget, Connections, \
    DeleteResource
from tools import Dao, word2voice
from ui.Set_distance import Ui_setDistance
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"  #（代表仅使用第0，1号GPU）


# 1：电子围栏违规，2：安全帽违规，3：绝缘手套违规，4：跌倒，5：绝缘靴违规，6：抽烟违规
class DLPX:
    def __init__(self):
        Dao.connectionDB(r'172.26.20.138', r'root', r'abc@123', r'est')
        self.detects = self.getData()
        self.root = os.path.abspath(os.curdir)

        self.range_distance_main1 = 3.0
        self.range_distance_main2 = 3.4
        self.range_distance_main3 = 3.4

        print(self.detects)
        self.app = QApplication(sys.argv)
        self.mainWindow = MainWindow()
        self.showLabel = ShowLabel(self.mainWindow.videoBox)  # 中间展示视频初始化
        self.cameraInfoList = CameraInfoList(self.mainWindow.cameraBox)  # 最右边展示违规信息初始化
        self.cameraList = CameraList(self.cameraInfoList, self.showLabel)  # ？？
        self.addCamera()
        self.setCamera()
        self.mainWindow.gridLayout.addWidget(self.showLabel, 0, 0, 1, 1)
        self.mainWindow.verticalLayout.addWidget(self.cameraInfoList)

        self.set_distance = Ui_setDistance(self)

        # 为设置下拉栏各个动作添加触发事件
        # self.mainWindow.addCamera.triggered.connect(lambda: AddCameraWidget().show())
        # self.mainWindow.addModel.triggered.connect(lambda: AddModelWidget().show())
        # self.mainWindow.connection.triggered.connect(lambda: Connections(self).show())
        # self.mainWindow.deleteResourse.triggered.connect(lambda: DeleteResource(self).show())
        #self.mainWindow.infractionQuery.triggered.connect(lambda: infractionQueryWidget(self).show())
        self.mainWindow.setDistance.triggered.connect(lambda: self.set_distance.show())
        # self.mainWindow.quit.triggered.connect(self.mainWindow.close)

        self.mainWindow.messageThread = MessageThread(self.mainWindow)
        self.mainWindow.messageThread.start()

    @staticmethod
    def getData():
        # 从视图show_info里面查询
        dataUrl = r'select `modelName`,`modelDIR`,`cameraUrl`,`name`,`comment` ' \
                  r'from show_info ' \
                  r'where  `status` = 1'
        return Dao.query(dataUrl)

    def addCamera(self):
        for i in self.detects:
            self.cameraList.addCamera(self, *i, "") # 人数
            time.sleep(sleeptime(0, 0, 10))

    def update(self):
        #real-time modification of monitoring data
        temp = self.getData()
        for i in temp:
            if i not in self.detects:
                self.cameraList.addCamera(self, *i, "人数：")
        for i in self.detects:
            if i not in temp:
                self.cameraInfoList.deleteCameraInfo(i[2], i[0])
        self.detects = temp
        self.setCamera()

    def setCamera(self):
        if len(self.cameraInfoList.cameraInfos):
            self.showLabel.setCameras(self.cameraList.getCameras())
            self.cameraInfoList.cameraInfos[0].setBackground(200, 200, 200)
        else:
            self.showLabel.currentCamera = None

    def show(self):
        self.mainWindow.show()

    def exit(self):
        self.app.exec_()


class MessageThread(threading.Thread):
    def __init__(self, superWindow):
        threading.Thread.__init__(self)
        self.superWindow = superWindow
        self.daemon = True

    def run(self):
        while True:
            message = Detect.getMessage()
            height, width = message.img.shape[0:2]
            image = cv2.cvtColor(message.img, cv2.COLOR_BGR2RGB)
            image = QImage(image.data, width, height, QImage.Format_RGB888)
            image = image.scaled(self.superWindow.displayLabel.width(), self.superWindow.displayLabel.height())
            self.superWindow.displayLabel.setPixmap(QPixmap.fromImage(image))
            self.superWindow.commentLabel.setText(message.name)

            if message.messageType == 1:
                threading.Thread(target=word2voice, args=("您已进入危险区域",)).start()
                # threading.Thread(target=shock, args=(6,)).start()
            if message.messageType == 2:
                threading.Thread(target=word2voice, args=("请佩戴安全帽",)).start()
            if message.messageType == 3:
                threading.Thread(target=word2voice, args=("请戴绝缘手套",)).start()
            if message.messageType == 4:
                threading.Thread(target=word2voice, args=("有人跌倒",)).start()
            if message.messageType == 5:
                threading.Thread(target=word2voice, args=("请穿绝缘靴",)).start()
            if message.messageType == 6:
                threading.Thread(target=word2voice, args=("有人抽烟",)).start()
            if message.messageType == 7:
                threading.Thread(target=word2voice, args=("请佩戴安全带",)).start()
            time.sleep(0.1)


if __name__ == '__main__':
    demo = DLPX()
    demo.show()
    sys.exit(demo.exit())

