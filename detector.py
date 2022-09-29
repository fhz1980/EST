import cv2
import torch
import numpy as np

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


# yolov5处理并获取结果


class Detector:

    def __init__(self, weights):
        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1

        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        print("weights===", weights)
        model = attempt_load(weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()

        self.m = model

        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)  # author zzl 2022.9.1 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
        img = torch.from_numpy(img).to(self.device)# author zzl 2022.9.1 数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
        img = img.half()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    # detect
    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()

        pred = non_max_suppression(pred, self.threshold, 0.4)
        boxes = []

        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *x, conf, cls_id in det:

                    lbl = self.names[int(cls_id)]
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])

                    boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return boxes


