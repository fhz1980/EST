import cv2
import numpy as np

from HKSdk.pyhikvision.hkws.core.type_map import *
from HKSdk.pyhikvision.hkws.model import callbacks
from HKSdk.pyhikvision.hkws.playm4_adpt import PlayM4


NET_DVR_SYSHEAD = 1
NET_DVR_STREAMDATA = 2
NET_DVR_AUDIOSTRAMDATA = 3
NET_DVR_PRIVATE_DATA = 112
COUNT = 0
playm4 = PlayM4()


@callbacks.DecCallBack
def f_dec_call_back(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2):
    # count=playm4.getCount()
    # playm4.incrementCount()
    # if count%10==0 :
    print("nPort:", nPort)
    print("Reserved:", nReserved1, nReserved2)
    frameInfo = pFrameInfo.contents
    bbb = string_at(pBuf, nSize)
    height = frameInfo.nHeight
    width = frameInfo.nWidth
    nparr = np.fromstring(bbb, np.uint8)
    img = nparr.reshape(height * 3 // 2, width)
    img2 = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YV12)
    cv2.imshow("sadf", img2)
    cv2.waitKey(1)
    # return


@callbacks.DecCallBack
def f_dec_call_back_new(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2):
    # count=playm4.getCount()
    # playm4.incrementCount()
    # if count%10==0 :
    frameInfo = pFrameInfo.contents
    bbb = string_at(pBuf, nSize)
    height = frameInfo.nHeight
    width = frameInfo.nWidth
    nparr = np.fromstring(bbb, np.uint8)
    img = nparr.reshape(height * 3 // 2, width)
    img2 = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YV12)
    cv2.imshow("sadf", img2)
    cv2.waitKey(1)
    return img2


# 视频流回调函数
# CFUNCTYPE(None, h_LONG, h_DWORD, POINTER(h_BYTE), h_DWORD, h_DWORD)
# dwDataType：数据类型   1：系统头数据 2：流数据 3：音频数据 112：私有数据，包括智能信息
# pBuffer：存放数据的缓冲区指针
# dwBufSize：缓冲区大小
# dwUser： 用户数据
@callbacks.real_data_callback
def f_real_data_call_back(lRealHandle,
                          dwDataType,
                          pBuffer,
                          dwBufSize,
                          dwUser):
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
            if playm4.setDecCallBack(f_dec_call_back) == 0:
                return
        print("头数据")
    elif dwDataType is NET_DVR_STREAMDATA:  # 流数据
        if dwBufSize > 0:
            isok = playm4.mp4_inputdata(pBuffer, dwBufSize)
            if isok < 1:
                return
        # if dwBufSize > 0:
        #     res = bytearray(dwBufSize)
        #     rptr = (c_byte * dwBufSize).from_buffer(res)
        #     if not memmove(rptr, pBuffer, dwBufSize):
        #         raise RuntimeError("memmove failed")
        #     print(res.hex())
        print("流数据")
    elif dwDataType is NET_DVR_AUDIOSTRAMDATA:  # 音频数据
        print("音频数据")
    elif dwDataType is NET_DVR_PRIVATE_DATA:  # 私有数据
        print("私有数据")

    return
