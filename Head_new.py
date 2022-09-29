import queue
import threading
import time

import numpy
import pyttsx3

pt = pyttsx3.init()
import torch
from detector import Detector
from Core import Detect, Message

import os
from HKSdk.hkUtils.CameraUtils import HKCameraDataSet
from utils.plots import plot_one_box
from win32com.client import Dispatch

speaker = Dispatch('SAPI.SpVoice')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

distance1 = 0.23
distance2 = 0.7
distance3 = 0.78
import serial

"""
@author jiaxin_yao at 2022/9/22
"""


class Head(Detect):
    def __init__(self, thread, main_window, target):
        super().__init__(thread)
        # 根据视频尺寸，填充一个polygon，供撞线计算使用
        # self.sayer = Sayer()

        self.target = target
        self.main_window = main_window

        self.my_ser = MySerial()
        print("--------------")
        threading.Thread(target=self.my_ser.get_distance).start()
        print("--------------")
        # 播报线程
        threading.Thread(target=self.say).start()

        self.pre_time = 0

        """
        语音播报相关
        """
        self.pre_content = None
        self.speak_content = None  # 如果不设置长度,默认为无限长
        self.speak_content_queue = queue.Queue(5)


        """
        记录经历过的流程的集合
        """
        self.experienced_process_set = set()

    @torch.no_grad()
    def run(self,
            weights='weights/yolov7.pt',  # model.pt path(s)
            source='data/images',  # file/dir/URL/glob, 0 for webcam
            imgsz=640,  # inference size (pixels)
            conf_thres=0.75,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='cuda:1',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            line_thickness=3,  # bounding box thickness (pixels)
            half=False,  # use FP16 half-precision inference
            ):

        detector_head = Detector(weights)
        dataset = HKCameraDataSet(source)
        print("----------------------source---------------------------", source)

        """
        当前流程所在的阶段
        """
        stage = 1.1

        """
        统计取接地线同一状态出现的次数
        """
        count_num = 0
        for path, image in dataset:
            #  固定
            # ----------------------
            if image is None:
                continue
            image = numpy.fliplr(image)
            # image0用来展示 , image用来预测
            image0 = image.copy()
            img_mask = self.thread.mask.getOrginMask(scale=image.shape[1::-1])
            #image = image & img_mask
            self.thread.mask.settingFence(image0, False, self.thread.mask.pointsList, (255, 0, 0))
            # ---------------------

            # 验电区域划定
            polygon_Array = self.areaDetermination(image)  # 所有框的坐标

            if len(polygon_Array) == 3:
                # 检测
                boxes = detector_head.detect(image)

                # 判断是否在距离内
                is_distance_range1 = distance1 - 0.1 < self.my_ser.distance1 < distance1 + 0.1
                is_distance_range2 = distance2 - 0.1 < self.my_ser.distance2 < distance2 + 0.1
                is_distance_range3 = distance3 - 0.1 < self.my_ser.distance3 < distance3 + 0.1

                # 判断情况 并显示出来   ? 距离信息同步问题
                status_code = self.judge_situation(polygon_Array, boxes, image0, self.my_ser)

                """
                注意,应该先定阶段再判断是否正确
                """
                print("stage", stage, "status_code", status_code)

                """
                第一阶段
                """
                print("self.experienced_process_set", self.experienced_process_set)

                if stage == 1.1:
                    if status_code == 1.1 and is_distance_range1 and 1.1 not in self.experienced_process_set:  # 处于状态1.1并且处于范围内
                        self.transmit_content("已完成第一次验电")
                        self.putMessage2Queue(Message(0, "已完成第一次验电", image0))
                        self.experienced_process_set.add(1.1)  # 删除对其的判断

                        # 进入下一个阶段
                        stage = 1.2
                    # -------------此处可拆分为 1空白 和 2已经做过或者当前情况
                    elif status_code == 0 or status_code == 1.1 or status_code in self.experienced_process_set:
                        pass

                    else:
                        self.transmit_content("流程错误")
                        self.putMessage2Queue(Message(0, "1.1流程错误", image0))

                elif stage == 1.2:
                    if status_code == 1.2 and is_distance_range2 and 1.2 not in self.experienced_process_set:
                        self.transmit_content("已完成第二次验电")
                        self.putMessage2Queue(Message(0, "已完成第二次验电", image0))
                        self.experienced_process_set.add(1.2)
                        # 进入下一个阶段
                        stage = 1.3

                    elif status_code == 0 or status_code == 1.2 or status_code in self.experienced_process_set:
                        pass
                    else:

                        self.transmit_content("流程错误")
                        self.putMessage2Queue(Message(0, "1.2流程错误", image0))

                elif stage == 1.3:
                    if status_code == 1.3 and is_distance_range3 and 1.3 not in self.experienced_process_set:
                        self.transmit_content("已完成第三次验电")
                        self.putMessage2Queue(Message(0, "已完成第三次验电", image0))
                        self.experienced_process_set.add(1.3)
                        # 进入下一个阶段
                        stage = 2.1

                    elif status_code == 0 or status_code == 1.3 or status_code in self.experienced_process_set:

                        pass
                    else:
                        self.transmit_content("流程错误")
                        self.putMessage2Queue(Message(0, "1.3流程错误", image0))
                        """
                        第二阶段
                        """

                elif stage == 2.1:
                    if status_code == 2.1 and is_distance_range1 and 2.1 not in self.experienced_process_set:
                        self.transmit_content("已完成第一次挂接地线")
                        self.putMessage2Queue(Message(0, "已完成第一次挂接地线", image0))
                        self.experienced_process_set.add(2.1)
                        # 进入下一个阶段
                        stage = 2.2

                    elif status_code == 0 or status_code == 2.1 or status_code in self.experienced_process_set:
                        pass
                    else:
                        self.transmit_content("流程错误")
                        self.putMessage2Queue(Message(0, "2.1流程错误", image0))

                elif stage == 2.2:
                    if status_code == 2.2 and is_distance_range2 and 2.2 not in self.experienced_process_set:
                        self.transmit_content("已完成第二次挂接地线")
                        self.putMessage2Queue(Message(0, "已完成第二次挂接地线", image0))
                        self.experienced_process_set.add(2.2)
                        # 进入下一个阶段
                        stage = 2.3

                    elif status_code == 0 or status_code == 2.2 or status_code in self.experienced_process_set:
                        pass
                    else:
                        self.transmit_content("流程错误")
                        self.putMessage2Queue(Message(0, "2.2流程错误", image0))

                elif stage == 2.3:
                    if status_code == 2.3 and is_distance_range2 and 2.3 not in self.experienced_process_set:
                        self.transmit_content("已完成第三次挂接地线")
                        self.putMessage2Queue(Message(0, "已完成第三次挂接地线", image0))
                        self.experienced_process_set.add(2.3)
                        # 进入下一个阶段
                        stage = 3.1

                    elif status_code == 0 or status_code == 2.3 or status_code in self.experienced_process_set:
                        pass
                    else:
                        self.transmit_content("流程错误")
                        self.putMessage2Queue(Message(0, "2.3流程错误", image0))
                        """
                        第三阶段
                        """
                elif stage == 3.1:
                    if status_code == 3.1 and is_distance_range1 and 3.1 not in self.experienced_process_set:
                        self.transmit_content("已完成第二阶段第一次验电")
                        self.putMessage2Queue(Message(0, "已完成第二阶段第一次验电", image0))
                        self.experienced_process_set.add(3.1)
                        # 进入下一个阶段
                        stage = 3.2

                    elif status_code == 0 or status_code == 3.1 or status_code in self.experienced_process_set:
                        pass
                    else:
                        self.transmit_content("流程错误")
                        self.putMessage2Queue(Message(0, "3.1流程错误", image0))
                elif stage == 3.2:
                    if status_code == 3.2 and is_distance_range2 and 3.2 not in self.experienced_process_set:
                        self.transmit_content("已完成第二阶段第二次验电")
                        self.putMessage2Queue(Message(0, "已完成第二阶段第二次验电", image0))
                        self.experienced_process_set.add(3.2)
                        # 进入下一个阶段
                        stage = 3.3

                    elif status_code == 0 or status_code == 3.3 or status_code in self.experienced_process_set:
                        pass
                    else:
                        self.transmit_content("流程错误")
                        self.putMessage2Queue(Message(0, "3.2流程错误", image0))

                elif stage == 3.3:
                    if status_code == 3.3 and is_distance_range3 and 3.3 not in self.experienced_process_set:
                        self.transmit_content("已完成第二阶段第三次验电")
                        self.putMessage2Queue(Message(0, "已完成第二阶段第三次验电", image0))
                        self.experienced_process_set.add(3.3)
                        # 进入下一个阶段
                        stage = 4.1

                    elif status_code == 0 or status_code == 3.2 or status_code in self.experienced_process_set:
                        pass
                    else:
                        self.transmit_content("流程错误")
                        self.putMessage2Queue(Message(0, "3.3流程错误", image0))
                    """
                    第四阶段
                    注意：取接地线很不同,需要进行统计 count_num
                    """
                elif stage == 4.1:
                    if status_code == 4.1 and 4.1 not in self.experienced_process_set and count_num == 60:
                        self.transmit_content("已完成第一次取接地线")
                        self.putMessage2Queue(Message(0, "已完成第一次取接地线", image0))
                        self.experienced_process_set.add(4.1)
                        # 进入下一个阶段
                        stage = 4.2
                    elif status_code == 4.1:
                        count_num += 1

                    elif status_code == 0 or status_code in self.experienced_process_set:
                        count_num = 0
                    else:
                        self.transmit_content("流程错误")
                        self.putMessage2Queue(Message(0, "4.1流程错误", image0))
                elif stage == 4.2:
                    if status_code == 4.2 and 4.2 not in self.experienced_process_set and count_num == 60:
                        self.transmit_content("已完成第二次取接地线")
                        self.putMessage2Queue(Message(0, "已完成第二次取接地线", image0))
                        self.experienced_process_set.add(4.2)
                        # 进入下一个阶段
                        stage = 4.3
                    elif status_code == 4.2:
                        count_num += 1
                    elif status_code == 0 or status_code in self.experienced_process_set:
                        count_num = 0
                    else:
                        self.transmit_content("流程错误")
                        self.putMessage2Queue(Message(0, "4.2流程错误", image0))
                # ----- 这种情况比较特殊--------
                elif stage == 4.3:
                    if status_code == 0 and 4.3 not in self.experienced_process_set and count_num == 60:
                        self.transmit_content("已完成第二次取接地线,任务完成")
                        self.putMessage2Queue(Message(0, "已完成第二次取接地线,任务完成", image0))
                        self.experienced_process_set.add(4.3)
                        # 进入下一个阶段
                        stage = 1.1
                        self.experienced_process_set = set()
                    elif status_code == 0:
                        count_num += 1
                    elif status_code in self.experienced_process_set:
                        count_num = 0
                    else:
                        self.transmit_content("流程错误")
                        self.putMessage2Queue(Message(0, "4.3流程错误", image0))

            self.putImg2Queue(image0)
            pass
        pass
        if self.thread.isStop:
            return

    def transmit_content(self, content):
        """
        @f 语音播报中继站
        @p content: 播报内容
        """
        if self.pre_content != content and not self.speak_content_queue.full():
            print("入队列")
            self.speak_content_queue.put(content)


    def say(self):
        """
        @f 语音播报
        @content: 播报内容
        1.相同内容播报一次
        """
        while True:
            print("正在运行")
            if not self.speak_content_queue.empty():
                content = self.speak_content_queue.get()
                speaker.Speak(content)

                print("wance")

    def areaDetermination(self, image):
        """
        @ function: 获取所有框的坐标信息
        @ p1 存储位置
        @ p2 图片的信息
        """
        print(self.thread.mask.pointsList)
        polygon_Array = []
        for polygon in self.thread.mask.pointsList:
            point_array = []
            for point_i in polygon:
                point_array.append((int(point_i[0] * image.shape[1]), int(point_i[1] * image.shape[0])))
            if len(point_array) != 0:
                polygon_Array.append(point_array)
            if len(point_array) >= 2:
                for i, point_i in enumerate(point_array):
                    pass
        return polygon_Array

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

    def is_in_poly(self, p, poly):
        """
        @f 判断一个点是否在一个多边形中
        @p p: 中心坐标
        @p poly: 数组
        @r bool
        """
        px, py = p
        is_in = False
        for i, corner in enumerate(poly):
            next_i = i + 1 if i + 1 < len(poly) else 0
            x1, y1 = corner
            x2, y2 = poly[next_i]
            if y1 == 0 and y2 == 0:
                if min(x1, x2) < px < max(x1, x2):
                    is_in = True
                    break
            if (x1 == px and y1 == py) or (x2 == px and y2 == py):
                is_in = True
                break
            if min(y1, y2) < py <= max(y1, y2):
                x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if x == px:
                    is_in = True
                    break
                elif x > px:
                    is_in = not is_in
        return is_in

    def judge_situation(self, array, boxes, image, my_ser):
        """
        @f 判断现在框里的情况 并显示出来
        @p  array: 三个框的位置信息
        @p boxes: 三个框检测出来的全部东西 （类别,概率..）
        @p image: 检测的图片的引用
        @p my_ser: 测距对象
        @r status_code
        """

        # 获取情况到字典 {"0",[eleHead,rodHead],}
        # information_dict {0: set(), 1: {'eleHead'}, 2: set()}
        information_dict = {}

        # 为每一个框找到对应的信息
        for i, array_i in enumerate(array):  # @p i：第几个框：框的序号

            classification_set_i = set()  # 第i个框的类别列表
            for box_i in boxes:
                x_core = int((box_i[0] + box_i[2]) / 2)
                y_core = int((box_i[1] + box_i[3]) / 2)
                classification = box_i[4]  # 类别
                if self.is_in_poly((x_core, y_core), array_i):  # 如果在第i个框内
                    classification_set_i.add(classification)  # 添加]
                    # 画框和并显示距离
                    if i == 0:
                        plot_one_box((box_i[0], box_i[1], box_i[2], box_i[3]), image, (0, 0, 255),
                                     label=box_i[4] + "%.2fm" % my_ser.distance1)
                    elif i == 1:
                        plot_one_box((box_i[0], box_i[1], box_i[2], box_i[3]), image, (0, 0, 255),
                                     label=box_i[4] + "%.2fm" % my_ser.distance2)
                    elif i == 2:
                        plot_one_box((box_i[0], box_i[1], box_i[2], box_i[3]), image, (0, 0, 255),
                                     label=box_i[4] + "%.2fm" % my_ser.distance3)

            information_dict[i] = classification_set_i

        # 获取对应的情况号码
        status_code = self.judgeBoxStatus(information_dict)
        #print(status_code)

        return status_code

    def judgeBoxStatus(self, information_dict):
        """
        @f 判断三框的所有可能状况
        @p information_dict 情况字典
        @r 状态号 0表示啥也没有 即4.3状态  -1表示其他状态
        """
        status_code = None
        # 第一阶段 验电
        #print("information_dict", information_dict)
        if len(information_dict) != 0:
            if information_dict[0] == {"eleHead"} and information_dict[1] == set() and information_dict[2] == set():
                status_code = 1.1
            elif information_dict[0] == set() and information_dict[1] == {"eleHead"} and information_dict[2] == set():
                status_code = 1.2
            elif information_dict[0] == set() and information_dict[1] == set() and information_dict[2] == {"eleHead"}:
                status_code = 1.3
            # 第二阶段 挂接地线
            elif information_dict[0] == {"rodHead"} and information_dict[1] == set() and information_dict[2] == set():
                status_code = 2.1
            elif information_dict[0] == {"rodHead"} and information_dict[1] == {"rodHead"} and information_dict[
                2] == set():
                status_code = 2.2
            elif information_dict[0] == {"rodHead"} and information_dict[1] == {"rodHead"} and information_dict[2] == {
                "rodHead"}:
                status_code = 2.3

            # 有接地线在的 第三阶段验电
            elif information_dict[0] == {"rodHead", "eleHead"} and information_dict[1] == {"rodHead"} and \
                    information_dict[
                        2] == {"rodHead"}:
                status_code = 3.1
            elif information_dict[0] == {"rodHead"} and information_dict[1] == {"rodHead", "eleHead"} and \
                    information_dict[
                        2] == {"rodHead"}:
                status_code = 3.2
            elif information_dict[0] == {"rodHead"} and information_dict[1] == {"rodHead"} and information_dict[2] == {
                "rodHead", "eleHead"}:
                status_code = 3.3

            # 第四阶段 取接地线 同第二阶段
            elif information_dict[0] == set() and information_dict[1] == {"rodHead"} and information_dict[2] == {
                "rodHead"}:
                status_code = 4.1
            elif information_dict[0] == set() and information_dict[1] == set() and information_dict[
                2] == {"rodHead"}:
                status_code = 4.2
            elif information_dict[0] == set() and information_dict[1] == set() and information_dict[
                2] == set():
                status_code = 0
            # ---注意:4.3状态和0（空白）状态一样-----
            else:
                status_code = -1
        else:
            status_code = -1
        return status_code


class Sayer(object):
    """
    @c 播报者
    @f 语音播报
    """

    def __init__(self):

        self.all_time_difference = 0  # 累计播报时间差
        self.sayed_time = 0  # 上一次播报完成的时间
        self.sayed_content = None  # 已经播报过的正确信息的集合

    def say(self, content):
        """
        @f 语音播报
        @p content: 播报内容
        1相同连续帧情况只当做一次播报
        2播报一次正确的情况（比如第一次验电成功），就把其剔除，不再播报
        """
        # 不和上一张相同，不是已经播报过的正确帧
        current_time_difference = time.time() - self.sayed_time
        self.all_time_difference += current_time_difference
        # 间隔秒数2,使得上一次语音说完
        if time.time()-self.sayed_time > 6:
            self.sayed_time = time.time()
            print("累计时间差", self.all_time_difference)
            speaker.Speak(content)
            self.sayed_content = content
            self.all_time_difference = 0
            self.sayed_time = time.time()

        print("播报完成")


class MySerial:
    """
    @c 激光雷达
    @f 测距
    """

    def __init__(self):
        self.ser = serial.Serial()
        self.ser.baudrate = 9600
        self.ser.port = 'COM4'
        self.ser.bytesize = 8
        self.ser.stopbits = 1
        self.ser.parity = 'N'
        self.ser.open()
        self.distance1 = -1
        self.distance2 = -1
        self.distance3 = -1

    def get_distance(self):
        while True:
            self.get_distance1()
            self.get_distance2()
            self.get_distance3()

    def get_distance1(self):
        Hex_str = bytes.fromhex('01 03 00 00 00 04 44 09')
        self.ser.write(Hex_str)
        distance_hex = self.ser.read(13).hex()
        self.distance1 = (int(distance_hex[6], 16) * 16 ** 3 + int(distance_hex[7], 16) * 16 ** 2 + int(distance_hex[8],
                                                                                                        16) * 16 + int(
            distance_hex[9], 16)) / 1000
        #print(self.distance1)

    def get_distance2(self):
        Hex_str = bytes.fromhex('02 03 00 00 00 04 44 3A')
        self.ser.write(Hex_str)
        distance_hex = self.ser.read(13).hex()

        self.distance2 = (int(distance_hex[6], 16) * 16 ** 3 + int(distance_hex[7], 16) * 16 ** 2 + int(distance_hex[8],
                                                                                                        16) * 16 + int(
            distance_hex[9], 16)) / 1000
        print(self.distance2)

    def get_distance3(self):
        Hex_str = bytes.fromhex('03 03 00 00 00 04 45 EB')
        self.ser.write(Hex_str)
        distance_hex = self.ser.read(13).hex()

        self.distance3 = (int(distance_hex[6], 16) * 16 ** 3 + int(distance_hex[7], 16) * 16 ** 2 + int(distance_hex[8],
                                                                                                        16) * 16 + int(
            distance_hex[9], 16)) / 1000
        print(self.distance3)
