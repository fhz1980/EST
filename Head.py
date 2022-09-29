import threading
import time
from queue import Queue
from Core import Detect, Message
import cv2
import pyttsx3

pt = pyttsx3.init()
import torch
from detector import Detector
from Core import Detect
import os
from utils.rtspdatasetfast import LoadRTSPStreamsFast
from HKSdk.hkUtils.CameraUtils import HKCameraDataSet
from utils.plots import plot_one_box
from threading import Thread
from win32com.client import Dispatch

speaker = Dispatch('SAPI.SpVoice')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import serial


class Head(Detect):

    def __init__(self, thread, main_window, target):
        super().__init__(thread)
        # 根据视频尺寸，填充一个polygon，供撞线计算使用
        self.say_thread = Say()
        self.say_thread.start()

        self.thread = thread
        self.target = target
        self.main_window = main_window
        self.centerPointMask = []

        self.my_ser = MySerial()
        threading.Thread(target=self.my_ser.get_distance1()).start()
        threading.Thread(target=self.my_ser.get_distance2()).start()
        threading.Thread(target=self.my_ser.get_distance3()).start()
        self.pre_time = 0
        self.say_first = True

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

        weight2 = self.main_window.weight2
        detector_ele_head = Detector(weights)
        detector_rod_head = Detector(weight2)
        detector_head = Detector("D:/code/yolov7/weights/Head.pt")
        dataset = HKCameraDataSet(source)
        print("----------------------source---------------------------", source)

        count = set()
        polygon_last_num_is2 = False
        can_say1 = False

        cuo_lock1 = False
        cuo_lock2 = False
        cuo_lock3 = False
        say_times1 = 0
        say_times2 = 0
        say_times3 = 0

        choice = 0
        choice_weight = 3

        stage = 1

        xu_yao_pan_duan = True

        kong_set_num = 0
        kong_set_pre = None
        kong_set = set()

        # 验电完成的等待时间
        time_ = None

        wait = False

        for path, image in dataset:
            count_list = set()

            if image is None:
                continue
            image = image.copy()
            img_mask = self.thread.mask.getOrginMask(scale=image.shape[1::-1])
            image_detect = image & img_mask

            self.thread.mask.settingFence(image, False, self.thread.mask.pointsList, (255, 0, 0))

            bboxes = None
            if choice_weight == 1:
                bboxes = detector_ele_head.detect(image_detect)
            elif choice_weight == 2:
                bboxes = detector_rod_head.detect(image_detect)
            else:
                bboxes = detector_head.detect(image_detect)

            # 验电区域划定
            polygon_Array = []
            for polygon in self.main_window.showLabel.fence.copy_pointsArray:
                point_array = []
                for point_i in polygon:
                    point_array.append((int(point_i[0] * image.shape[1]), int(point_i[1] * image.shape[0])))
                if len(point_array) != 0:
                    polygon_Array.append(point_array)
                if len(point_array) >= 2:
                    for i, point_i in enumerate(point_array):
                        pass
            time_cha = 100
            if wait:
                time_cha = time.time() - time_

            if len(polygon_Array) == 3 and time_cha > 3:
                if len(bboxes) == 0:
                    kong_set = {0, 1, 2}
                else:
                    for box_id, box in enumerate(bboxes):
                        print("box", box)
                        # 框里是什么东西，判断是否符合流程
                        if xu_yao_pan_duan:
                            if box[4] == "eleHead":
                                print("----", box[4])
                                plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                             label="eleHead")
                                if stage == 1:
                                    xu_yao_pan_duan = False
                                    choice = 1
                                    choice_weight = 1
                                    stage += 1
                                    self.putMessage2Queue(Message(0, "已进入第一次验电", image))
                                    threading.Thread(target=self.say, args=("已经进入第一次验电",)).start()

                                    print("--------------------------------------成功进入1--------------------------")
                                elif stage == 3:
                                    xu_yao_pan_duan = False
                                    choice = 3
                                    choice_weight = 1
                                    stage += 1
                                    self.putMessage2Queue(Message(0, "已进入第二次验电", image))
                                    threading.Thread(target=self.say, args=("已经进入第二次验电",)).start()
                                else:
                                    # 报警

                                    self.putMessage2Queue(Message(0, "流程错误", image))
                                    threading.Thread(target=self.say, args=("流程错误",)).start()

                            if box[4] == "rodHead":
                                plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                             label="rodHead")
                                if stage == 2:
                                    choice = 2
                                    stage += 1
                                    choice_weight = 2
                                    xu_yao_pan_duan = False
                                    self.putMessage2Queue(Message(0, "已进入挂接地线", image))
                                    threading.Thread(target=self.say, args=("已经进入挂接地线",)).start()
                                elif stage == 4:
                                    choice = 4
                                    stage = 1
                                    choice_weight = 2
                                    xu_yao_pan_duan = False
                                    self.putMessage2Queue(Message(0, "已进入取接地线", image))
                                    threading.Thread(target=self.say, args=("已经进入取接地线",)).start()
                                else:
                                    # 报警
                                    self.putMessage2Queue(Message(0, "流程错误", image))
                                    threading.Thread(target=self.say, args=("流程错误",)).start()

                        # xun_huan_next = False
                        if choice == 1:
                            if box[4] == "eleHead":
                                print(box[4], box[5])
                                x_core = int((box[0] + box[2]) / 2)
                                y_core = int((box[1] + box[3]) / 2)

                                # 判断识别到的在哪个框内
                                polygon_num = None
                                print("判断位置", x_core, y_core, polygon_Array)
                                for i, polygon in enumerate(polygon_Array):
                                    if is_in_poly((x_core, y_core), polygon):
                                        polygon_num = i
                                        break

                                print("polygon_num", polygon_num)
                                self.my_ser.distance2
                                # distance1 = self.distance_object1.distance
                                # distance2 = self.distance_object1.distance
                                # distance3 = self.distance_object3.distance
                                # distance_range1 = self.main_window.range_distance_main1 - 0.1 < distance1 < self.main_window.range_distance_main1 + 0.1
                                # distance_range2 = self.main_window.range_distance_main2 - 0.1 < distance2 < self.main_window.range_distance_main2 + 0.1
                                # distance_range3 = self.main_window.range_distance_main3 - 0.1 < distance3 < self.main_window.range_distance_main3 + 0.1

                                if polygon_num == 0:

                                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                                 label="eleHead" + "%.2fm" % self.my_ser.distance1)

                                elif polygon_num == 1:

                                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                                 label="eleHead" + "%.2fm" % self.my_ser.distance2)
                                elif polygon_num == 2:
                                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                                 label="eleHead" + "%.2fm" % self.my_ser.distance3)
                                else:
                                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                                 label="eleHead")
                                print("jaj")
                                if polygon_num == 0 and self.main_window.range_distance_main1 - 0.1 < self.my_ser.distance1 < self.main_window.range_distance_main1 + 0.1:
                                    print(1)
                                    if not count:

                                        # pt.say("已完成第一次验电")
                                        # pt.runAndWait()
                                        # self.say_thread.message.put("已完成第一次验电")
                                        if say_times1 == 0 or can_say1:
                                            say_thread = threading.Thread(target=self.say, args=("已完成第一次验电",))
                                            say_thread.start()
                                            say_times1 = 1
                                            say_times2 = 0
                                            say_times3 = 0
                                            can_say1 = False

                                            cuo_lock1 = False
                                            cuo_lock2 = False
                                            cuo_lock3 = False

                                            print("完成第一次")
                                            self.putMessage2Queue(Message(0, "第一次验电", image))
                                            count.add(1)
                                    else:
                                        print(1.1)
                                        # pt.say("验电顺序错误,第一次验电失败,请重新开始")
                                        # pt.runAndWait()
                                        # self.say_thread.message.put("验电顺序错误,第一次验电失败,请重新开始")

                                        if say_times1 == 0:
                                            if not cuo_lock1:
                                                print("第一次错误，从新开始")
                                                say_thread = threading.Thread(target=self.say,
                                                                              args=("验电顺序错误,验电失败,请重新开始",))
                                                say_thread.start()
                                                say_times1 = 1
                                                say_times2 = 0
                                                say_times3 = 0
                                                cuo_lock2 = True
                                                cuo_lock3 = True
                                                can_say1 = True
                                                count = set()
                                                # self.putMessage2Queue(Message(0, "验电失败", image))

                                            self.putMessage2Queue(Message(0, "验电失败", image))


                                elif polygon_num == 1 and self.main_window.range_distance_main2 - 0.1 < self.my_ser.distance2 < self.main_window.range_distance_main2 + 0.1:
                                    print(2)
                                    if self.judge(1, count) and not self.judge(2, count) and not self.judge(3, count):
                                        # print("已完成第二次验电")
                                        # self.say_thread.message.put("已完成第二次验电")
                                        if say_times2 == 0:
                                            print("已完成第二次验电")
                                            say_thread = threading.Thread(target=self.say, args=("已完成第二次验电",))
                                            say_thread.start()
                                            say_times2 = 1
                                            say_times1 = 0
                                            say_times3 = 0

                                            cuo_lock1 = False
                                            cuo_lock2 = False
                                            cuo_lock3 = False

                                            # can_say2 = False
                                            self.putMessage2Queue(Message(0, "第二次验电", image))
                                            count.add(2)

                                    else:

                                        print("验电顺序错误,第二次验电失败,请重新开始")
                                        if say_times2 == 0:
                                            if not cuo_lock2:
                                                say_thread = threading.Thread(target=self.say,
                                                                              args=("验电顺序错误,验电失败,请重新开始",))
                                                say_thread.start()
                                                say_times2 = 1
                                                say_times1 = 0
                                                say_times3 = 0

                                                cuo_lock1 = True
                                                cuo_lock3 = True
                                                # can_say2 = True
                                                count = set()
                                                # self.putMessage2Queue(Message(0, "验电失败", image))

                                            self.putMessage2Queue(Message(0, "验电失败", image))

                                elif polygon_num == 2 and self.main_window.range_distance_main3 - 0.1 < self.my_ser.distance3 < self.main_window.range_distance_main3 + 0.1:
                                    print(2)
                                    if self.judge(1, count) and self.judge(2, count) and not self.judge(3, count):
                                        # self.say_thread.message.put("已完成第三次验电,验电成功")
                                        if say_times3 == 0:
                                            say_thread = threading.Thread(target=self.say, args=("已完成第三次验电,验电成功",))
                                            say_thread.start()
                                            say_times3 = 1
                                            say_times1 = 0
                                            say_times2 = 0
                                            # can_say3 = False
                                            # can_say1 = False
                                            # can_say2 = False
                                            cuo_lock1 = False
                                            cuo_lock2 = False
                                            cuo_lock3 = False

                                            self.putMessage2Queue(Message(0, "第三次验电", image))
                                            ## 完成验电之后不要那么快识别下一个
                                            wait = True
                                            time_ = time.time()

                                            choice = 0
                                            choice_weight = 3
                                            stage = 2
                                            xu_yao_pan_duan = True
                                            count.add(3)
                                            count = set()
                                            polygon_last_num_is2 = True
                                            break

                                    else:
                                        # self.say_thread.message.put("验电顺序错误,第三次验电失败,请重新开始验电")
                                        if say_times3 == 0:
                                            if not cuo_lock3:
                                                say_thread = threading.Thread(target=self.say,
                                                                              args=("验电顺序错误,验电失败,请重新开始验电",))
                                                say_thread.start()
                                                say_times3 = 1
                                                say_times1 = 0
                                                say_times2 = 0

                                                cuo_lock1 = True
                                                cuo_lock2 = True

                                                # can_say3 = True
                                                count = set()
                                                # self.putMessage2Queue(Message(0, "验电失败", image))

                                            self.putMessage2Queue(Message(0, "验电失败", image))

                                elif polygon_last_num_is2 and polygon_num != 2:
                                    print(4)
                                    say_times3 = 0
                                print("wiwiwi")

                        elif choice == 2:
                            if box[4] == "rod_head" and box[5] > 0.3:
                                print(box[4], box[5])
                                x_core = int((box[0] + box[2]) / 2)
                                y_core = int((box[1] + box[3]) / 2)

                                # 判断识别到的在哪个框内
                                polygon_num = None
                                print("判断位置", x_core, y_core, polygon_Array)
                                for i, polygon in enumerate(polygon_Array):
                                    print()
                                    if is_in_poly((x_core, y_core), polygon):
                                        polygon_num = i
                                        break

                                print("polygon_num", polygon_num)

                                # distance1 = self.distance_object1.distance
                                # distance2 = self.distance_object1.distance
                                # distance3 = self.distance_object3.distance
                                # distance_range1 = self.main_window.range_distance_main1 - 0.1 < distance1 < self.main_window.range_distance_main1 + 0.1
                                # distance_range2 = self.main_window.range_distance_main2 - 0.1 < distance2 < self.main_window.range_distance_main2 + 0.1
                                # distance_range3 = self.main_window.range_distance_main3 - 0.1 < distance3 < self.main_window.range_distance_main3 + 0.1

                                if polygon_num == 0:

                                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                                 label="rodHead" + "%.2fm" % self.my_ser.distance1)

                                elif polygon_num == 1:
                                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                                 label="rodHead" + "%.2fm" % self.my_ser.distance2)
                                elif polygon_num == 2:
                                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                                 label="rodHead" + "%.2fm" % self.my_ser.distance3)
                                else:
                                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                                 label="rodHead")

                                if polygon_num == 0 and self.main_window.range_distance_main1 - 0.1 < self.my_ser.distance1 < self.main_window.range_distance_main1 + 0.1:
                                    if not count:

                                        # pt.say("已完成第一次验电")
                                        # pt.runAndWait()
                                        # self.say_thread.message.put("已完成第一次验电")
                                        if say_times1 == 0 or can_say1:
                                            say_thread = threading.Thread(target=self.say, args=("已完成第一次挂接地线",))
                                            say_thread.start()
                                            say_times1 = 1
                                            say_times2 = 0
                                            say_times3 = 0
                                            can_say1 = False

                                            cuo_lock1 = False
                                            cuo_lock2 = False
                                            cuo_lock3 = False
                                            print("完成第一次")
                                            self.putMessage2Queue(Message(0, "第一次挂接地线", image))
                                            count.add(1)
                                    else:

                                        # pt.say("验电顺序错误,第一次验电失败,请重新开始")
                                        # pt.runAndWait()
                                        # self.say_thread.message.put("验电顺序错误,第一次验电失败,请重新开始")

                                        if say_times1 == 0:
                                            if not cuo_lock1:
                                                print("第一次错误，从新开始")
                                                say_thread = threading.Thread(target=self.say,
                                                                              args=("挂接地线顺序错误,挂接地线失败,请重新开始",))
                                                say_thread.start()
                                                say_times1 = 1
                                                say_times2 = 0
                                                say_times3 = 0
                                                can_say1 = True
                                                cuo_lock2 = True
                                                cuo_lock3 = True
                                                count = set()

                                            self.putMessage2Queue(Message(0, "挂接地线顺序错误", image))

                                elif polygon_num == 1 and self.main_window.range_distance_main2 - 0.1 < self.my_ser.distance2 < self.main_window.range_distance_main2 + 0.1:
                                    if self.judge(1, count) and not self.judge(2, count) and not self.judge(3, count):
                                        # print("已完成第二次验电")
                                        # self.say_thread.message.put("已完成第二次验电")
                                        if say_times2 == 0:
                                            say_thread = threading.Thread(target=self.say, args=("已完成第二次挂接地线",))
                                            say_thread.start()
                                            say_times2 = 1
                                            say_times1 = 0
                                            say_times3 = 0

                                            cuo_lock1 = False
                                            cuo_lock2 = False
                                            cuo_lock3 = False
                                            # can_say2 = False
                                            self.putMessage2Queue(Message(0, "第二次挂接地线", image))
                                            count.add(2)

                                    else:
                                        # self.say_thread.message.put("验电顺序错误,第二次验电失败,请重新开始")
                                        if say_times2 == 0:
                                            if not cuo_lock2:
                                                say_thread = threading.Thread(target=self.say,
                                                                              args=("挂接地线顺序错误,挂接地线失败,请重新开始",))
                                                say_thread.start()
                                                say_times2 = 1
                                                say_times1 = 0
                                                say_times3 = 0
                                                cuo_lock1 = True
                                                cuo_lock3 = True
                                                # can_say2 = True
                                                count = set()
                                            self.putMessage2Queue(Message(0, "挂接地线顺序错误", image))

                                elif polygon_num == 2 and self.main_window.range_distance_main3 - 0.1 < self.my_ser.distance3 < self.main_window.range_distance_main3 + 0.1:
                                    if self.judge(1, count) and self.judge(2, count) and not self.judge(3, count):
                                        # self.say_thread.message.put("已完成第三次验电,验电成功")
                                        if say_times3 == 0:
                                            say_thread = threading.Thread(target=self.say, args=("已完成第三次挂接地线,挂接地线成功",))
                                            say_thread.start()
                                            say_times3 = 1
                                            say_times1 = 0
                                            say_times2 = 0
                                            # can_say3 = False
                                            # can_say2 = False
                                            # can_say1 = False

                                            cuo_lock1 = False
                                            cuo_lock2 = False
                                            cuo_lock3 = False
                                            self.putMessage2Queue(Message(0, "第三次挂接地线", image))
                                            choice = 0
                                            xu_yao_pan_duan = True
                                            choice_weight = 1
                                            stage = 3
                                            count.add(3)
                                            count = set()
                                            polygon_last_num_is2 = True
                                            break

                                    else:
                                        # self.say_thread.message.put("验电顺序错误,第三次验电失败,请重新开始验电")
                                        if say_times3 == 0:
                                            if not cuo_lock3:
                                                say_thread = threading.Thread(target=self.say,
                                                                              args=("挂接地线顺序错误,挂接地线失败,请重新开始挂接地线",))
                                                say_thread.start()
                                                say_times3 = 1
                                                say_times1 = 0
                                                say_times2 = 0
                                                cuo_lock1 = True
                                                cuo_lock2 = True
                                                # can_say3 = False
                                                count = set()
                                            self.putMessage2Queue(Message(0, "挂接地线顺序错误", image))
                                elif polygon_last_num_is2 and polygon_num != 2:
                                    say_times3 = 0

                        elif choice == 3:
                            if box[4] == "eleHead":
                                print(box[4], box[5])
                                x_core = int((box[0] + box[2]) / 2)
                                y_core = int((box[1] + box[3]) / 2)

                                # 判断识别到的在哪个框内
                                polygon_num = None
                                print("判断位置", x_core, y_core, polygon_Array)
                                for i, polygon in enumerate(polygon_Array):
                                    print()
                                    if is_in_poly((x_core, y_core), polygon):
                                        polygon_num = i
                                        break

                                print("polygon_num", polygon_num)

                                # distance1 = self.distance_object1.distance
                                # distance2 = self.distance_object1.distance
                                # distance3 = self.distance_object3.distance
                                # distance_range1 = self.main_window.range_distance_main1 - 0.1 < distance1 < self.main_window.range_distance_main1 + 0.1
                                # distance_range2 = self.main_window.range_distance_main2 - 0.1 < distance2 < self.main_window.range_distance_main2 + 0.1
                                # distance_range3 = self.main_window.range_distance_main3 - 0.1 < distance3 < self.main_window.range_distance_main3 + 0.1

                                if polygon_num == 0:

                                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                                 label="eleHead" + "%.2fm" % self.my_ser.distance1)

                                elif polygon_num == 1:
                                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                                 label="eleHead" + "%.2fm" % self.my_ser.distance2)
                                elif polygon_num == 2:
                                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                                 label="eleHead" + "%.2fm" % self.my_ser.distance3)
                                else:
                                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                                 label="eleHead")

                                if polygon_num == 0 and self.main_window.range_distance_main1 - 0.1 < self.my_ser.distance1 < self.main_window.range_distance_main1 + 0.1:
                                    if not count:

                                        # pt.say("已完成第一次验电")
                                        # pt.runAndWait()
                                        # self.say_thread.message.put("已完成第一次验电")
                                        if say_times1 == 0 or can_say1:
                                            say_thread = threading.Thread(target=self.say, args=("已完成第一次验电",))
                                            say_thread.start()
                                            say_times1 = 1
                                            say_times2 = 0
                                            say_times3 = 0
                                            can_say1 = False

                                            cuo_lock1 = False
                                            cuo_lock2 = False
                                            cuo_lock3 = False
                                            print("完成第一次")
                                            self.putMessage2Queue(Message(0, "第一次验电", image))
                                            count.add(1)
                                    else:

                                        # pt.say("验电顺序错误,第一次验电失败,请重新开始")
                                        # pt.runAndWait()
                                        # self.say_thread.message.put("验电顺序错误,第一次验电失败,请重新开始")

                                        if say_times1 == 0:
                                            if not cuo_lock1:
                                                print("第一次错误，从新开始")
                                                say_thread = threading.Thread(target=self.say,
                                                                              args=("验电顺序错误,验电失败,请重新开始",))
                                                say_thread.start()
                                                say_times1 = 1
                                                say_times2 = 0
                                                say_times3 = 0
                                                can_say1 = True
                                                cuo_lock2 = True
                                                cuo_lock3 = True
                                                count = set()
                                            self.putMessage2Queue(Message(0, "验电顺序错误", image))

                                elif polygon_num == 1 and self.main_window.range_distance_main2 - 0.1 < self.my_ser.distance2 < self.main_window.range_distance_main2 + 0.1:
                                    if self.judge(1, count) and not self.judge(2, count) and not self.judge(3, count):
                                        # print("已完成第二次验电")
                                        # self.say_thread.message.put("已完成第二次验电")
                                        if say_times2 == 0:
                                            say_thread = threading.Thread(target=self.say, args=("已完成第二次验电",))
                                            say_thread.start()
                                            say_times2 = 1
                                            say_times1 = 0
                                            say_times3 = 0

                                            cuo_lock1 = False
                                            cuo_lock2 = False
                                            cuo_lock3 = False
                                            # can_say2 = False
                                            self.putMessage2Queue(Message(0, "第二次验电", image))
                                            count.add(2)

                                    else:
                                        # self.say_thread.message.put("验电顺序错误,第二次验电失败,请重新开始")
                                        if say_times2 == 0:
                                            if not cuo_lock2:
                                                say_thread = threading.Thread(target=self.say,
                                                                              args=("验电顺序错误,验电失败,请重新开始",))
                                                say_thread.start()
                                                say_times2 = 1
                                                say_times1 = 0
                                                say_times3 = 0
                                                cuo_lock1 = True
                                                cuo_lock3 = True
                                                # can_say2 = True
                                                count = set()
                                            self.putMessage2Queue(Message(0, "验电顺序错误", image))

                                elif polygon_num == 2 and self.main_window.range_distance_main3 - 0.1 < self.my_ser.distance3 < self.main_window.range_distance_main3 + 0.1:
                                    if self.judge(1, count) and self.judge(2, count) and not self.judge(3, count):
                                        # self.say_thread.message.put("已完成第三次验电,验电成功")
                                        if say_times3 == 0:
                                            say_thread = threading.Thread(target=self.say, args=("已完成第三次验电,验电成功",))
                                            say_thread.start()
                                            say_times3 = 1
                                            say_times1 = 0
                                            say_times2 = 0
                                            # can_say3 = False
                                            # can_say2 = False
                                            # can_say1 = False

                                            cuo_lock1 = False
                                            cuo_lock2 = False
                                            cuo_lock3 = False
                                            self.putMessage2Queue(Message(0, "第三次验电", image))
                                            choice = 0
                                            choice_weight = 2
                                            stage = 4
                                            xu_yao_pan_duan = True
                                            count.add(3)
                                            count = set()
                                            polygon_last_num_is2 = True
                                            break

                                    else:
                                        # self.say_thread.message.put("验电顺序错误,第三次验电失败,请重新开始验电")
                                        if say_times3 == 0:
                                            if not cuo_lock3:
                                                say_thread = threading.Thread(target=self.say,
                                                                              args=("验电顺序错误,验电失败,请重新开始验电",))
                                                say_thread.start()
                                                say_times3 = 1
                                                say_times1 = 0
                                                say_times2 = 0
                                                cuo_lock1 = True
                                                cuo_lock2 = True
                                                # can_say3 = True
                                                count = set()
                                            self.putMessage2Queue(Message(0, "验电顺序错误", image))
                                elif polygon_last_num_is2 and polygon_num != 2:
                                    say_times3 = 0

                        elif choice == 4:
                            print(choice, box[4], box[5])
                            if len(bboxes) > 0:
                                if box[4] == "rod_head" and box[5] > 0.3:
                                    print(box[4], box[5])
                                    x_core = int((box[0] + box[2]) / 2)
                                    y_core = int((box[1] + box[3]) / 2)

                                    plot_one_box((box[0], box[1], box[2], box[3]), image, (0, 0, 255),
                                                 label="rodHead")

                                    # 判断空白在何处

                                    # print("判断位置", x_core, y_core, polygon_Array)
                                    xun_huan_next = None
                                    break_xun_huan = None
                                    # 如果有框
                                    if len(bboxes) > 0:
                                        for i, polygon in enumerate(polygon_Array):

                                            # 判断在哪个框内
                                            if is_in_poly((x_core, y_core), polygon):
                                                count_list.add(i)
                                                # 扫描完一张图片之后
                                                print("box_id== len(bboxes)-1:", box_id == len(bboxes) - 1, box_id,
                                                      len(bboxes), len(bboxes) - 1)
                                                if box_id == len(bboxes) - 1:
                                                    kong_set = {0, 1, 2} - count_list
                                                    print("kong_set", kong_set)
                                                    if kong_set == kong_set_pre:
                                                        kong_set_pre = {0, 1, 2} - count_list
                                                        kong_set_num += 1
                                                        print("num", kong_set_num)
                                                        if kong_set_num >= 6:
                                                            kong_set_num = 0
                                                            break
                                                        else:
                                                            # 扫描下一张
                                                            break_xun_huan = True
                                                            break

                                                    else:
                                                        # 扫描下一张
                                                        kong_set_pre = {0, 1, 2} - count_list
                                                        kong_set_num = 0
                                                        break_xun_huan = True
                                                        break

                                                else:
                                                    xun_huan_next = True
                                                    break

                                    if xun_huan_next:
                                        continue
                                    if break_xun_huan:
                                        break

                            else:
                                kong_set = {1, 2, 3}

                            print("count", count)
                            # print("kong_set", kong_set)
                            # 没取过,且空集为第三个框
                            if not count:
                                if kong_set == {2}:
                                    # count_image_num1_is += 1  # 空白的次数
                                    # count_image_num1_not -= 1  # 非空白的次数
                                    if say_times1 == 0 or can_say1:  # and count_image_num1_is == 6:
                                        say_thread = threading.Thread(target=self.say, args=("已完成第一次取接地线",))
                                        say_thread.start()
                                        say_times1 = 1
                                        can_say1 = False
                                        say_times2 = 0
                                        say_times3 = 0

                                        cuo_lock1 = False
                                        cuo_lock2 = False
                                        cuo_lock3 = False
                                        print("完成第一次")
                                        # count_image_num1_is = 0
                                        self.putMessage2Queue(Message(0, "第一次取接地线", image))
                                        count.add(3)
                                elif kong_set == set():
                                    say_times1 = 0
                                else:

                                    # count_image_num1_is -= 1
                                    # count_image_num1_not += 1
                                    if say_times1 == 0:
                                        if not cuo_lock1:  # and count_image_num1_not == 8:
                                            print("第一次错误，从新开始")
                                            say_thread = threading.Thread(target=self.say,
                                                                          args=("取接地线顺序错误,取接地线失败,请重新开始",))
                                            say_thread.start()
                                            can_say1 = True
                                            say_times1 = 1
                                            say_times2 = 0
                                            say_times3 = 0
                                            cuo_lock2 = True
                                            cuo_lock3 = True
                                            # count_image_num1_not = 0  # 错误清零
                                            count = set()
                                        self.putMessage2Queue(Message(0, "取接地线顺序错误", image))

                            # 取过第三个
                            elif not self.judge(1, count) and not self.judge(2, count) and self.judge(3, count):
                                if kong_set == {1, 2}:
                                    if say_times2 == 0:  # and count_image_num2_is == 6:
                                        say_thread = threading.Thread(target=self.say, args=("已完成第二次取接地线",))
                                        say_thread.start()
                                        say_times2 = 1
                                        say_times1 = 1
                                        say_times3 = 0

                                        cuo_lock1 = False
                                        cuo_lock2 = False
                                        cuo_lock3 = False
                                        # can_say2 = False
                                        # count_image_num2_is = 0
                                        self.putMessage2Queue(Message(0, "第二次取接地线", image))
                                        count.add(2)

                                elif kong_set == {2}:
                                    pass
                                    # count_image_num2_is -= 1
                                    # count_image_num2_not += 1
                                else:
                                    if say_times2 == 0:
                                        if not cuo_lock2:  # and count_image_num2_not == 6:
                                            say_thread = threading.Thread(target=self.say,
                                                                          args=("取接地线顺序错误,取接地线失败,请重新开始",))
                                            say_thread.start()
                                            say_times2 = 1
                                            say_times1 = 0
                                            say_times3 = 0
                                            cuo_lock1 = True
                                            cuo_lock3 = True
                                            # can_say2 = True
                                            # count_image_num2_not = 0
                                            count = set()
                                        self.putMessage2Queue(Message(0, "取接地线顺序错误", image))
                if choice == 4:
                    print("kongji", kong_set)
                    if not self.judge(1, count) and self.judge(2, count) and self.judge(3, count):
                        print("kongji", kong_set)
                        if kong_set == {0, 1, 2}:
                            # count_image_num3_is += 1
                            # count_image_num3_not -= 1
                            if say_times3 == 0:  # and count_image_num3_is == 6:
                                say_thread = threading.Thread(target=self.say, args=("已完成第三次取接地线,取接地线成功,任务完成",))
                                say_thread.start()
                                say_times3 = 1
                                say_times1 = 0
                                say_times2 = 0
                                # can_say3 = False
                                # can_say2 = False
                                # can_say1 = False

                                cuo_lock1 = False
                                cuo_lock2 = False
                                cuo_lock3 = False
                                self.putMessage2Queue(Message(0, "第三次取接地线", image))
                                choice = 0
                                choice_weight = 3
                                stage = 1
                                xu_yao_pan_duan = True
                                count.add(3)
                                count = set()
                                # count_image_num3_is = 0
                                polygon_last_num_is2 = True


                        elif kong_set == {1, 2}:
                            pass

                        else:
                            # count_image_num3_is -= 1
                            # count_image_num3_not += 1
                            if say_times3 == 0:
                                if not cuo_lock3:  # and count_image_num2_not == 6:
                                    say_thread = threading.Thread(target=self.say,
                                                                  args=("取接地线顺序错误,取接地线失败,请重新开始",))
                                    say_thread.start()
                                    say_times3 = 1
                                    say_times1 = 0
                                    say_times2 = 0
                                    cuo_lock1 = True
                                    cuo_lock2 = True
                                    # can_say3 = True
                                    # count_image_num3_not = 0
                                    count = set()
                                self.putMessage2Queue(Message(0, "取接地线顺序错误", image))

            self.putImg2Queue(image)
        if self.thread.isStop:
            return

    def say(self, content):
        time_difference = time.time() - self.pre_time
        if time_difference > 6 or self.say_first:
            print("时间差", time_difference, self.say_first)
            self.say_first = False
            self.pre_time = time.time()
            speaker.Speak(content)
            print("wancheng")

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


class DetectThread(Thread):
    def __init__(self, function, args):
        super(DetectThread, self).__init__()
        self.function = function
        self.args = args
        self.result = None
        print("检测线程已开始")

    def run(self):
        self.result = self.function(self.args)

    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            print("异常")
            return None


class DistanceThread3(Thread):
    def __init__(self, ser):
        super().__init__()
        self.distance = None
        self.ser = ser

    def run(self) -> None:
        while True:
            Hex_str = bytes.fromhex('03 03 00 00 00 04 45 EB')
            self.ser.write(Hex_str)
            distance_hex = self.ser.read(13).hex()
            self.distance = (distance_hex[6] * 16 ** 3 + distance_hex[7] * 16 ** 2 + distance_hex[8] * 16 +
                             distance_hex[9]) / 1000
            print("distance3", self.distance, "m")


class Say(Thread):
    pre_time = time.time()

    def __init__(self):
        super(Say, self).__init__()
        self.message = Queue(6)
        # print("怎么回事")

    def run(self) -> None:
        while self.message.not_empty:
            now_time = time.time()
            if now_time - self.pre_time > 6:
                self.pre_time = time.time()
                pt.say(self.message.get())
                pt.runAndWait()


class MySerial(object):
    def __init__(self):
        self.ser = serial.Serial()
        self.ser.baudrate = 9600
        self.ser.port = 'COM5'
        self.ser.bytesize = 8
        self.ser.stopbits = 1
        self.ser.parity = 'N'
        self.ser.open()
        self.distance1 = None
        self.distance2 = None
        self.distance3 = None

    def get_distance1(self):
        Hex_str = bytes.fromhex('01 03 00 00 00 04 44 09')
        self.ser.write(Hex_str)
        distance_hex = self.ser.read(13).hex()
        self.distance1 = (int(distance_hex[6], 16) * 16 ** 3 + int(distance_hex[7], 16) * 16 ** 2 + int(distance_hex[8],
                                                                                                        16) * 16 + int(
            distance_hex[9], 16)) / 1000
        print(self.distance1)

    def get_distance2(self):
        Hex_str = bytes.fromhex('02 03 00 00 00 04 44 3A')
        self.ser.write(Hex_str)
        distance_hex = self.ser.read(13).hex()

        self.distance2 = (int(distance_hex[6], 16) * 16 ** 3 + int(distance_hex[7], 16) * 16 ** 2 + int(distance_hex[8],
                                                                                                        16) * 16 + int(
            distance_hex[9], 16)) / 1000
        print(self.distance1)
    def get_distance3(self):
        Hex_str = bytes.fromhex('03 03 00 00 00 04 45 EB')
        self.ser.write(Hex_str)
        distance_hex = self.ser.read(13).hex()

        self.distance3 = (int(distance_hex[6], 16) * 16 ** 3 + int(distance_hex[7], 16) * 16 ** 2 + int(distance_hex[8],
                                                                                                        16) * 16 + int(
            distance_hex[9], 16)) / 1000

        print(self.distance1)
def change_base(distance):
    """
    将十六进制字符转换为10进制数字
    """
    list = []
    for i in distance:
        for j in range(48, 58):
            if i == j:
                flag = 1
                list.append(int(i))

        for j in range(61, 67):
            if i == j:
                list.append(int(i, 16))


def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
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
