import numpy as np
import torch
import tracker
from detector import Detector
from Core import Detect,Message
from utils.rtspdataset import LoadRTSPStreams
import cv2
import os
from utils.plots import colors, plot_one_box
import torch.backends.cudnn as cudnn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class SmokeDetect(Detect):
    def __init__(self,thread,target):
        super().__init__(thread)
        # 根据视频尺寸，填充一个polygon，供撞线计算使用
        self.target=target

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
        dataset = LoadRTSPStreams(source)
        for path, image in dataset:
            im0 = image.copy()
            img_mask = self.thread.mask.getOrginMask(scale=image.shape[1::-1])
            image = image & img_mask
            # im = cv2.resize(image, (960, 540))
            self.thread.mask.settingFence(im0, False, self.thread.mask.pointsList, (255, 0, 0))
            bboxes = detector.detect(image)
            # 如果画面中 有bbox
            self.target=['person', 'worker','shoes', 'helmet']
            if len(bboxes) > 0:
                for (x1, y1, x2, y2, lbl,conf) in bboxes:
                    plot_one_box([x1, y1, x2, y2], im0, label=lbl, line_thickness=line_thickness)
                    if lbl=='smoke' :
                        self.putMessage2Queue(Message(1,'禁止抽烟', im0))  # Message是用来添加到语音播报队列的
            self.putImg2Queue(im0)
            pass
        pass

        if self.thread.isStop:
            return
