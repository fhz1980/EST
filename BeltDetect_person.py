import os

import torch

from HKSdk.hkUtils.CameraUtils import HKCameraDataSet
from detector import Detector
from Core import Detect, Message
from utils.rtspdataset import LoadRTSPStreams
from tools import sendimgtype
import time
from utils.plots import plot_one_box

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BeltDetect_person(Detect):
    def __init__(self, thread, target):
        super().__init__(thread)

        self.target = target
        # self.weights2 = './yolov7.pt'


    # author zzl 2022.9.2 这个函数的作用判断安全带c1，c2，c3，c4，c5中心点是否全部在personbox中,在返回False，不在返回True
    def isAt(self, bboxes, personboxs):

        i = False
        if len(bboxes) == 0:
            i = True
        else:
            flag = [0 for x in range(len(personboxs))]
            for box in bboxes:
                for personindex in range(len(personboxs)):

                    x = (box[0] + box[2]) / 2  # x轴中心点
                    y = (box[1] + box[3]) / 2  # y轴中心点
                    if (personboxs[personindex][0] <= x <= personboxs[personindex][2]) and (
                            personboxs[personindex][1] <= y <= personboxs[personindex][3]): flag[personindex] = 1
            if 0 in flag: i = True

        return i

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

        detector = Detector(weights)
        # detector2 = Detector(self.weights2)
        dataset = HKCameraDataSet(source)  # 将视频资源获取
        print("__________________________________")

        # count = 0
        flag = []
        for path, image in dataset:  # author zzl 2022.9.2 datase中有视频的一帧信息
            im0 = image.copy()
            im0type = type(im0)
            img_mask = self.thread.mask.getOrginMask(scale=image.shape[1::-1])
            image = image & img_mask

            self.thread.mask.settingFence(im0, False, self.thread.mask.pointsList, (255, 0, 0))
            # bboxes2 = detector2.detect(image)
            bboxes = detector.detect(image)  # 如果画面中 有bbox//这个是检测出来的框
            # print(bboxes)  # author zzl 2022.9.1 这个就是一帧检测出来的东西
            # 先实现安全带在人身上
            pboxes =[]
            safeboxes = []#安全带的点
            for box in bboxes:
                if 'person' in box:
                    pboxes.append(box)
                else:
                    safeboxes.append(box)

            personboxs = []
            for box in pboxes:
                if 'person' in box:
                    personboxs.append(box)
                    if box[5] > 0.6:
                        # print("人的信息", box)
                        plot_one_box([box[0], box[1], box[2], box[3]], im0, label=box[4], line_thickness=line_thickness)

            if personboxs != []:
                if len(safeboxes) != 0:
                    for box in safeboxes:
                        if box[5] > 0.6:
                            # print("安全带的信息", box)
                            plot_one_box([box[0], box[1], box[2], box[3]], im0, label=box[4],
                                         line_thickness=line_thickness)
                if self.isAt(safeboxes, pboxes):
                    flag.append(im0)

                else:flag.append(1)


            if len(flag) == 20:#屯20帧检测一次，如果没有违规的照片小于10帧，那才判定检测不到，把最新的一帧违规照片上传
                x = 0
                for index in range(len(flag)):
                    if type(flag[index]) is int : x +=1

                if x <= 10:#x表示检测正常的帧
                    im0 = None
                    for index in range(len(flag)-1, -1, -1):
                        if type(flag[index]) is im0type:
                            im0 = flag[index]
                            break

                    if im0 is not None:
                        result = '请佩戴安全带'
                        print(result)
                        self.putMessage2Queue(Message(7, result, im0))  # Message是用来添加到语音播报队列的
                flag =[]

            self.putImg2Queue(im0)
            pass
        pass

        if self.thread.isStop:
            return

        '''
                if len(bboxes) == 0:  # author zzl 2022.9.2 有人没有安全带，警告佩戴安全带
                    count += 1
                    if (count >= 50):
                        result = '请佩戴安全带'
                        print(result)
                        self.putMessage2Queue(Message(7, result, im0))  # Message是用来添加到语音播报队列的
                        count = 0

                else:
                    for (x1, y1, x2, y2, lbl, conf) in bboxes:
                        if conf > 0.5:
                            print("安全带的信息", [x1, y1, x2, y2])
                            print(lbl)
                            plot_one_box([x1, y1, x2, y2], im0, label=lbl, line_thickness=line_thickness)  # 画框的方法

            self.putImg2Queue(im0)
            pass
        pass

        if self.thread.isStop:
            return
'''


'''
            for box, box2 in zip(bboxes, bboxes2):  # 安全带在人身上检测
                personbox = []
                if box2[4] == 'person':
                    personbox = box2
                    print(personbox)
                    if personbox[5] > 0.3:
                        print("人的信息", personbox)
                        plot_one_box([personbox[0], personbox[1], personbox[2],personbox[3]], im0, label=personbox[4], line_thickness=line_thickness)
                    if (len(bboxes) != 0):
                        if box[5] > 0.3:
                            print("安全带信息", box)
                            plot_one_box([box[0], box[1], box[2], box[3]], im0, label=box[4],
                                         line_thickness=line_thickness)

                    if self.isAt(bboxes, personbox):
                        count += 1
                        if (count >= 50):
                            result = '请佩戴安全带'
                            print(result)
                            self.putMessage2Queue(Message(7, result, im0))  # Message是用来添加到语音播报队列的
                            count = 0

            self.putImg2Queue(im0)
            pass
        pass

        if self.thread.isStop:
            return
'''
