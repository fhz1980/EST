import os
import cv2
import queue
import logging
import configparser
import numpy as np

from ctypes import string_at
from multiprocessing import Process, Queue
from HKSdk.pyhikvision.hkws import cm_camera_adpt
from HKSdk.pyhikvision.hkws.model import callbacks
from HKSdk.pyhikvision.hkws.playm4_adpt import PlayM4

NET_DVR_SYSHEAD = 1
NET_DVR_STREAMDATA = 2
NET_DVR_AUDIOSTRAMDATA = 3
NET_DVR_PRIVATE_DATA = 112
COUNT = 0
playm4 = PlayM4()


class CaptureSet:
    capQueue = queue.Queue(50)

    def getQueue(self):
        return self.capQueue

    @staticmethod
    @callbacks.real_data_callback
    def f_real_data_call_back(lRealHandle,
                              dwDataType,
                              pBuffer,
                              dwBufSize,
                              dwUser=None):
        # print("dwDataType:{},pBuffer:{},dwBufSize:{}".format(dwDataType, pBuffer, dwBufSize))
        if dwDataType is NET_DVR_SYSHEAD:  # 系统头数据
            if playm4.get_port() == 0:
                return
            if dwBufSize > 0:
                if playm4.set_stream_open_mode(0) == 0:
                    return
                if playm4.open_stream(pBuffer, dwBufSize, 1024 * 1024) == 0:
                    return
                if playm4.mp4_play() == 0:
                    return
                if playm4.setDecCallBack(CaptureSet.f_dec_call_back_new) == 0:
                    return
            # print("头数据")
        elif dwDataType is NET_DVR_STREAMDATA:  # 流数据
            if dwBufSize > 0:
                isok = playm4.mp4_inputdata(pBuffer, dwBufSize)
                if isok < 1:
                    return
            # print("流数据")
        elif dwDataType is NET_DVR_AUDIOSTRAMDATA:  # 音频数据
            print("音频数据")
        elif dwDataType is NET_DVR_PRIVATE_DATA:  # 私有数据
            print("私有数据")
        return

    @staticmethod
    @callbacks.DecCallBack
    def f_dec_call_back_new(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2):
        frameInfo = pFrameInfo.contents
        buffer = string_at(pBuf, nSize)
        height = frameInfo.nHeight
        width = frameInfo.nWidth
        nparr = np.frombuffer(buffer, np.uint8)
        img = nparr.reshape(height * 3 // 2, width)
        img2 = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YV12)
        if CaptureSet.capQueue.full():
            CaptureSet.capQueue.get()
        CaptureSet.capQueue.put(img2)


class Config:
    def __init__(self,
                 ip="127.0.0.1",
                 user="admin",
                 passWord="123456",
                 sdkPath="G:/ZSL/DLPX/src/HKSdk/pyhikvision/lib/win64/",
                 port=8000,
                 channel=0,
                 plat="1"):

        self.ip = ip
        self.user = user
        self.passWord = passWord
        self.sdkPath = sdkPath
        self.port = port
        self.channel = channel
        if plat == "0":
            self.plat = "0"  # 0-Linux，1-windows
            self.suffix = ".so"
        else:
            self.plat = "1"
            self.suffix = ".dll"

    def InitConfig(self, path):
        cnf = configparser.ConfigParser()
        cnf.read(path)
        self.sdkPath = cnf.get("DEFAULT", "sdkPath")
        self.user = cnf.get("DEFAULT", "user")
        self.passWord = cnf.get("DEFAULT", "passWord")
        self.port = cnf.getint("DEFAULT", "port")
        self.ip = cnf.get("DEFAULT", "ip")
        self.plat = cnf.get("DEFAULT", "plat")
        self.channel = int(cnf.get("DEFAULT", "channel"))
        if self.plat == "0":
            self.suffix = ".so"
        else:
            self.plat = "1"
            self.suffix = ".dll"
        return


class HKCamera:
    def __init__(self, channel):
        # 初始化配置文件
        path = "H:/DL/EST/HKSdk/pyhikvision/config.ini"
        self.cnf = Config()
        self.cnf.InitConfig(path)
        self.cnf.channel = channel

        # 初始化SDK适配器
        self.adapter = cm_camera_adpt.CameraAdapter()
        self.userId = self.adapter.common_start(self.cnf)
        if self.userId < 0:
            logging.error("初始化Adapter失败")
            os._exit(0)

        print("Login successful,the userId is ", self.userId)
        self.lRealPlayHandle = self.adapter.start_preview(None, self.cnf.channel, self.userId)
        if self.lRealPlayHandle < 0:
            self.adapter.logout(self.userId)
            self.adapter.sdk_clean()
            os._exit(2)
        self.queue = None

    def preview(self):
        captureSet = CaptureSet()
        print("start preview 成功", self.lRealPlayHandle)
        callback = self.adapter.callback_standard_data(self.lRealPlayHandle, captureSet.f_real_data_call_back, self.userId)
        print("callback", callback)
        self.queue = captureSet.getQueue()

    def playVideo(self):
        while True:
            if self.queue.not_empty:
                cv2.imshow("sadf", self.queue.get())
                cv2.waitKey(1)
                # print("None")
            else:
                print("None")

    def getCapQueue(self):
        if self.queue is None:
            print("Please execute \"self.preview\" method first!")
            return
        else:
            return self.queue


class CameraProcess(Process):
    def __init__(self, channel, capQueue):
        super().__init__()
        self.channel = channel
        self.capQueue = capQueue

    def run(self):
        hkCamera = HKCamera(self.channel)
        hkCamera.preview()
        tempCapQueue = hkCamera.getCapQueue()
        while True:
            if tempCapQueue.not_empty:
                self.capQueue.put(tempCapQueue.get())


class HKCameraDataSet:
    def __init__(self, channel):
        self.capQueue = Queue(5)
        self.cameraProcess = CameraProcess(int(channel), self.capQueue)
        self.cameraProcess.start()
        self.count = 5

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img = self.capQueue.get()
        return "NO Source", img

    def __len__(self):
        return self.count


if __name__ == "__main__":
    print("HKCamera Module")


