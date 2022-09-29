from HKSdk.pyhikvision.hkws.base_adapter import BaseAdapter
from ctypes import *
from HKSdk.pyhikvision.hkws import cm_camera_adpt


class PlayM4(BaseAdapter):
    def __init__(self):
        self.adapter = cm_camera_adpt.CameraAdapter()
        self.Source_Buf_Min = 1024 * 50
        self.Source_Buf_Max = 1024 * 100000
        self.port = c_uint(0)
        self.__port = 0
        self.__ready = False
        self.count = 0

    def getCount(self):
        return self.count

    def incrementCount(self):
        self.count = self.count + 1

    def ready(self, fun):
        # print("ready", self.__ready)
        if not self.__ready:
            print("port没有准备好，", fun)
        return self.__ready

    # 获取未使用的通道号
    def get_port(self):
        port = c_uint(0)
        res = self.call_cpp("PlayM4_GetPort", byref(port))
        if res == 0:
            self.print_error("PlayM4_SetStreamOpenMode 设置流播放模式失败: the error code is ")
        else:
            self.__port = port.value
            # print("__port", self.__port)
            self.__ready = True
            # print(self.__ready)
        return res

    # 设置流播放模式
    # nport: 播放通道号
    # nmode: 流播放模式  0:会尽量保证实时性，防止数据阻塞;而且数据检查严格   1:按时间戳播放
    def set_stream_open_mode(self, nmode):
        if not self.ready("set_stream_open_mode"):
            return -1
        res = self.call_cpp("PlayM4_SetStreamOpenMode", self.__port, nmode)
        # print("set_stream_open_moderes=======", res)
        if res == 0:
            self.print_error("PlayM4_SetStreamOpenMode 设置流播放模式失败: the error code is ")
        return res

    # 打开流
    # nport: 播放通道号
    # pFileHeadBuf： 文件头数据
    # nSize：文件头长度
    # nBufPoolSize： 设置播放器中存放数据流的缓冲区大小  范围为SOURCE_BUF_MIN ~ SOURCE_BUF_MAX
    def open_stream(self, pFileHeadBuf, nSize, nBufPoolSize):
        # print("open_streamopen_streamopen_streamopen_streamopen_stream")
        if not self.ready("open_stream"):
            return -1
        res = self.call_cpp("PlayM4_OpenStream", self.__port, pFileHeadBuf, nSize, nBufPoolSize)
        if res == 0:
            self.print_error("PlayM4_SetStreamOpenMode 设置流播放模式失败: the error code is ")
        return res

    def mp4_play(self):

        if not self.ready("mp4_play"):
            return -1
        res = self.call_cpp("PlayM4_Play", self.__port, None)
        # print("mp4_playmp4_playmp4_playmp4_playmp4_playmp4_play res=", res)
        if res == 0:
            self.print_error("PlayM4_Play: the error code is ")
        return res

    def mp4_inputdata(self, pFileHeadBuf, nSize):
        # print("port port====",self.__port)
        # if not self.ready("mp4_inputdatamp4_inputdatamp4_inputdata"):
        #     return -1
        res = self.call_cpp("PlayM4_InputData", self.__port, pFileHeadBuf, nSize)
        if res == 0:
            self.print_error("PlayM4_InputData: the error code is ")
        return res

    def setDecCallBack(self, f_dec_call_back):
        if not self.ready("setDecCallBack"):
            return -1
        res = self.adapter.callback_Dec(self.adapter.getHandle(), f_dec_call_back,
                                        self.adapter.getUser())
        if res == 0:
            self.print_error("PlayM4_InputData: the error code is ")
        return res
