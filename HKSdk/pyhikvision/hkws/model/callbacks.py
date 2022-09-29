from HKSdk.pyhikvision.hkws.core.type_map import *
from HKSdk.pyhikvision.hkws.model import alarm


class FRAME_INFO(Structure):
    _fields_ = [
        ("nWidth", h_LONG),
        ("nHeight", h_LONG),
        ("nStamp", h_LONG),
        ("nType", h_LONG),
        ("nFrameRate", h_LONG),
        ("dwFrameNum", h_DWORD)
    ]


msg_callback_v31 = CFUNCTYPE(h_BOOL, h_LONG, POINTER(alarm.NET_DVR_ALARMER), POINTER(h_CHAR), h_DWORD, h_VOID_P)

# 码流数据回调函数
real_data_callback = CFUNCTYPE(None, h_LONG, h_DWORD, POINTER(h_BYTE), h_DWORD, h_DWORD)

# 码流数据回调函数
standard_data_callback = CFUNCTYPE(None, h_LONG, h_DWORD, POINTER(h_BYTE), h_DWORD, h_DWORD)

# 码流数据回调函数
# DecCallBack =CFUNCTYPE(None, c_long, POINTER(c_char), FRAME_INFO, c_long, c_long)
DecCallBack = CFUNCTYPE(None, h_LONG, POINTER(h_BYTE), h_LONG, POINTER(FRAME_INFO), h_LONG, h_LONG)

# DECCALLBACK = CFUNCTYPE(None, c_long, POINTER(c_char), FRAME_INFO, c_long, c_long)
