import logging
import os

import cv2

from HKSdk.hkUtils.CameraUtils import CaptureSet
from HKSdk.pyhikvision.hkws import cm_camera_adpt, config

# 初始化配置文件
cnf = config.Config()
path = os.path.join('../local_config.ini')
cnf.InitConfig(path)

# 初始化SDK适配器
adapter = cm_camera_adpt.CameraAdapter()
userId = adapter.common_start(cnf)
if userId < 0:
    logging.error("初始化Adapter失败")
    os._exit(0)

print("Login successful,the userId is ", userId)

lRealPlayHandle = adapter.start_preview(None, userId)
if lRealPlayHandle < 0:
    adapter.logout(userId)
    adapter.sdk_clean()
    os._exit(2)


captureSet = CaptureSet()
# print("start preview 成功", lRealPlayHandle)
# callback = adapter.callback_standard_data(lRealPlayHandle, instant_preview_cb.f_real_data_call_back, userId)
callback = adapter.callback_standard_data(lRealPlayHandle, captureSet.f_real_data_call_back, userId)
# print("callback", callback)
queue = captureSet.getQueue()
while True:
    if queue.not_empty:
        cv2.imshow("sadf", queue.get())
        cv2.waitKey(1)
        print("None")
    else:
        print("None")

# deccallback = adapter.callback_Dec(lRealPlayHandle, instant_preview_cb.f_dec_call_back, userId)

time.sleep(36000)
adapter.stop_preview(lRealPlayHandle)
adapter.logout(userId)
adapter.sdk_clean()
