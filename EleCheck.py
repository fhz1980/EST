import threading
import pyttsx3
pt = pyttsx3.init()
import cv2
import numpy
import torch
from detector import Detector
from Core import Detect
import os
import random
from tools import sendimgtype, get_centerpoint
import numpy as np
from RealSenseStreams import getRealSenseStreams
from utils.plots import plot_one_box

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EleCheck(Detect):
    def __init__(self, thread, main_window,  target):
        super().__init__(thread)
        # 根据视频尺寸，填充一个polygon，供撞线计算使用
        self.thread = thread
        self.target = target
        self.range_distance = 3
        self.main_window = main_window
        print(self.main_window.showLabel.fence.pointsArray)


    @torch.no_grad()
    def run(self,
            weights='weights/yolov7.pt',  # model.pt path(s)
            source='data/images',  # file/dir/URL/glob, 0 for webcam
            imgsz=640,  # inference size (pixels)
            conf_thres=0.75,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            line_thickness=3,  # bounding box thickness (pixels)
            half=False,  # use FP16 half-precision inference
            ):

        print("use weight", weights)

        detector = Detector(weights)
        dataset = getRealSenseStreams(source)

        count = set()
        for image, deep_image in dataset:
            print(self.main_window.showLabel.fence.pointsArray)

            if image is None:
                continue
            im0 = image.copy()
            image = numpy.flip(im0, 1).copy()

            bboxes = detector.detect(image)

            if self.main_window.range_distance_main:
                self.range_distance = self.main_window.range_distance_main

            # 验电框
            # plot_one_box((24, 24, 48, 48), image, color=(255, 0, 0))
            # cv2.putText(image, "1", (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            # plot_one_box((320, 24, 344, 48), image, color=(255, 0, 0))
            # cv2.putText(image, "2", (326, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            # plot_one_box((592, 24, 616, 48), image, color=(255, 0, 0))
            # cv2.putText(image, "3", (598, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)



            for polygon in self.main_window.showLabel.fence.pointsArray:
                point_array = []
                for point_i in polygon:
                    point_array.append((int(point_i[0] * image.shape[1]), int(point_i[1] * image.shape[0])))
                if len(point_array) >= 2:
                    for i, point_i in enumerate(point_array):
                        cv2.line(image, point_array[i-1], point_array[i], (0, 0, 255), 5)

            # if box[4] == "Head" and box[5] > 0.6
            for box in bboxes:
                if box[4] == "Head" and box[5] > 0.6:
                    print(box[4], box[5])
                    x_core = int((box[0] + box[2]) / 2)
                    y_core = int((box[1] + box[3]) / 2)
                    distance = self.get_distance(box, deep_image, 24) / 1000

                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255), label=box[4] + "%.2fm" % distance)

                    distance_range = self.range_distance - 0.1 < distance < self.range_distance + 0.1
                    print("distance_range", self.range_distance)

                    if 24 < x_core < 48 and 24 < y_core < 48 and distance_range:
                        if not count:
                            say_thread = threading.Thread(target=self.say, args=("已完成第一次验电",))
                            say_thread.start()
                            count.add(1)
                        else:
                            say_thread = threading.Thread(target=self.say, args=("验电顺序错误,第一次验电失败,请重新开始",))
                            say_thread.start()
                            count = set()

                    elif 320 < x_core < 344 and 24 < y_core < 48 and distance_range:
                        if self.judge(1, count) and not self.judge(2, count) and not self.judge(3, count):
                            say_thread = threading.Thread(target=self.say, args=("已完成第二次验电",))
                            say_thread.start()
                            count.add(2)

                        else:
                            say_thread = threading.Thread(target=self.say, args=("验电顺序错误,第二次验电失败,请重新开始",))
                            say_thread.start()
                            count = set()

                    elif 592 < x_core < 616 and 24 < y_core < 48 and distance_range:
                        if self.judge(1, count) and self.judge(2, count) and not self.judge(3, count):
                            say_thread = threading.Thread(target=self.say, args=("已完成第三次验电,验电成功",))
                            say_thread.start()
                            count.add(3)
                            count = set()

                        else:

                            say_thread = threading.Thread(target=self.say, args=("验电顺序错误,第三次验电失败,请重新开始验电",))
                            say_thread.start()
                            count = set()

            self.putImg2Queue(image)
        if self.thread.isStop:
            return

    def say(self, content):
        pt.say(content)
        pt.runAndWait()

    def judge(self, element, sets):
        """
        功能:判断元素是否在集合中
        element: 元素
        sets: 集合
        return :bool
        """
        for i in sets:
            if i == element:
                return True

        return False

    @classmethod
    def get_distance(cls, box, depth_data, randnum):
        distance_list = []
        mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]  # 确定索引深度的中心像素位置
        min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1]))  # 确定深度搜索范围
        for i in range(randnum):
            bias = random.randint(-min_val // 4, min_val // 4)
            dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
            if dist:
                distance_list.append(dist)
        distance_list = np.array(distance_list)
        distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]  # 冒泡排序+中值滤波
        print(distance_list, np.mean(distance_list))
        return np.mean(distance_list)
