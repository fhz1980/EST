import os

import torch

from Core import Detect, Message
from HKSdk.hkUtils.CameraUtils import HKCameraDataSet
from detector import Detector
from utils.plots import plot_one_box

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SecurityDetect(Detect):
    def __init__(self, thread, target):
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
        dataset = HKCameraDataSet(source)
        for path, image in dataset:
            im0 = image.copy()
            img_mask = self.thread.mask.getOrginMask(scale=image.shape[1::-1])
            image = image & img_mask


            # im = cv2.resize(image, (960, 540))
            self.thread.mask.settingFence(im0, False, self.thread.mask.pointsList, (255, 0, 0))
            bboxes = detector.detect(image)

            # 如果画面中 有bbox
            detectobjetc=[]
            # target=['person', 'worker', 'glove', 'shoes', 'helmet']
            if len(bboxes) > 0:
                for (x1, y1, x2, y2, lbl, conf) in bboxes:
                    if conf > 0.99:
                        plot_one_box([x1, y1, x2, y2], im0, label=lbl, line_thickness=line_thickness)
                        detectobjetc.append(lbl)
                cha = set(self.target).difference(detectobjetc)
                if len(cha) > 0:
                    # print(cha)
                    result = '请佩戴'
                    for name in cha:
                        if name == 'worker':
                            result = result + "工作衣"
                        if name == 'glove':
                            result = result + "绝缘手套"
                        if name == 'shoes':
                            result = result + "绝缘靴"
                        if name == 'helmet':
                            result = result + "安全帽"
                    #self.putMessage2Queue(Message(1, result, im0))  # Message是用来添加到语音播报队列的
            self.putImg2Queue(im0)
            pass
        pass

        if self.thread.isStop:
            return
