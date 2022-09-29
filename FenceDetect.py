import numpy as np
import torch
import tracker
from HKSdk.hkUtils.CameraUtils import HKCameraDataSet
from detector import Detector
from Core import Detect, Message
from utils.rtspdataset import LoadRTSPStreams
from tools import sendimgtype
import cv2
import os
import time
from utils.plots import colors, plot_one_box
import torch.backends.cudnn as cudnn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FenceDetect(Detect):
    def __init__(self, thread, target):
        super().__init__(thread)
        # 根据视频尺寸，填充一个polygon，供撞线计算使用
        self.target = target
        # 初始化 yolov5

    @torch.no_grad()
    def run(self,
            weights='yolov5s.pt',  # model.pt path(s)
            source='data/images',  # file/dir/URL/glob, 0 for webcam
            imgsz=640,  # inference size (pixels)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            line_thickness=3,  # bounding box thickness (pixels)
            half=False,  # use FP16 half-precision inference
            ):
        detector = Detector(weights)
        # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #     ('rtsp://', 'rtmp://', 'http://', 'https://'))
        # Initialize
        dataset = HKCameraDataSet(source)  # 将视频资源获取
        fenceTime = int(time.time())
        fallTime = int(time.time())
        for path, image in dataset:
            # im0 = image.copy()
            img_mask = self.thread.mask.getOrginMask(scale=image.shape[1::-1])
            # print(img_mask.shape)
            # image = image & img_mask
            # im = cv2.resize(image, (960, 540))
            self.thread.mask.settingFence(image, False, self.thread.mask.pointsList, (255, 0, 0))
            bboxes = detector.detect(image)

            # 如果画面中 有bbox
            points = []
            if len(bboxes) > 0:
                isInFence = False
                # isFall = False
                for (x1, y1, x2, y2, lbl, conf) in bboxes:
                    # plot_one_box([x1, y1, x2, y2], image, label=lbl, line_thickness=line_thickness)
                    if lbl == 'person':
                        width1 = x2 - x1
                        height1 = y2 - y1
                        ratio = width1 / height1
                        # if ratio > 2 and width1 > 240:
                        #     isFall = True
                        plot_one_box([x1, y1, x2, y2], image, label=lbl, line_thickness=line_thickness)
                        # print(label)
                        width, height = image.shape[1], image.shape[0]
                        p = (int(x1) / width, int(y2) / height)
                        points.append(p)
                        p1 = (int(x2) / width, int(y2) / height)
                        points.append(p1)
                for j in points:
                    width, height = int(j[0] * img_mask.shape[1] - 1), int(j[1] * img_mask.shape[0] - 1)
                    if img_mask[height, width, 0] == 255:
                        isInFence = True
                        break
                if isInFence:  # 判断电子围违规
                    fenceNow = int(time.time())
                    fenceTimePeriod = fenceNow - fenceTime
                    if fenceTimePeriod > 2:
                        # imgFilePath = saveImg('h:/voliateimg/', 1, im0)
                        self.putMessage2Queue(Message(1, '电子围栏违规', image))  # Message是用来添加到语音播报队列的
                        sendimgtype(1, image)
                        fenceTime = fenceNow
                # if isFall:
                #     fallNow = int(time.time())
                #     fallTimePeriod = fallNow - fallTime
                #     if fallTimePeriod > 2:
                #         # imgFilePath = saveImg('h:/voliateimg/', 4, im0)
                #         self.putMessage2Queue(Message(4, '作业人员跌倒', image))
                #         # time.sleep(sleeptime(0, 0, 2))
                #         sendimgtype(4, image)
                #         fallTime = fallNow
            self.putImg2Queue(image)
            pass
        pass

        if self.thread.isStop:
            return
