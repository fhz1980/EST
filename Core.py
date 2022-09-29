import argparse
import threading
import time
import torch
import torch.backends.cudnn as cudnn
import cv2
import datetime
# import mediapipe as mp
from threading import Thread
from HKSdk.hkUtils.CameraUtils import HKCameraDataSet
from queue import Queue
from abc import ABCMeta, abstractmethod

from detector import Detector
from models.experimental import attempt_load
from tools import saveImg, shock, word2voice, sendimgtype, sleeptime
from utils.datasets import LoadStreams, LoadImages
from utils.rtspdataset import LoadRTSPStreams
from utils.general import check_img_size, non_max_suppression, \
    apply_classifier, scale_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class Detect:
    messages = Queue(20)

    def __init__(self, thread):
        self.images = Queue(5)
        self.thread = thread

    def putImg2Queue(self, im0):
        if self.images.full():  # Return True if the queue is full, False otherwise
            self.images.get()
        self.images.put(im0)  # Put an item into the queue.

    @classmethod
    def putMessage2Queue(cls, message):
        if cls.messages.full():
            cls.messages.get()
        cls.messages.put(message)

    @classmethod
    def getMessage(cls):
        if cls.messages.not_empty:
            return cls.messages.get()

    def getImage(self):
        if self.images.not_empty:
            return self.images.get()  # Return item in queue if the item exist

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

            # target=['person', 'worker', 'glove', 'shoes', 'helmet']
            if len(bboxes) > 0:
                for (x1, y1, x2, y2, lbl, conf) in bboxes:
                    if conf > 0.7:
                        plot_one_box([x1, y1, x2, y2], im0, label=lbl, line_thickness=line_thickness)

            self.putImg2Queue(im0)
            pass
        pass

        if self.thread.isStop:
            return

    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='{0}\\{1}.pt'.format(self.thread.modelDIR, self.thread.model),
                            help='model.pt path(s)')
        # parser.add_argument('--source', type=str, default=self.thread.url, help='source')
        parser.add_argument('--source',
                            type=str,
                            # default='rtsp://admin:jxlgust123@172.26.20.51:554/Streaming/Channels/302?transportmode=unicast',
                            default='{0}'.format(self.thread.url),
                            help='source')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=20, help='maximum detections per image')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        opt = parser.parse_args()

        return opt

    def execDetect(self, opt):
        print('CameraAddress:' + self.thread.url)
        self.run(**vars(opt))


class PersonDetect(Detect):
    def __init__(self, thread):
        super().__init__(thread)

    @torch.no_grad()
    def run(self,
            weights='yolov5s.pt.pt',  # model.pt path(s)
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
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        # Initialize
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        # Second-stage classifier
        classify = False
        modelc = None
        if classify:
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
        # Dataloader
        print('stride', stride)
        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        fenceTime = int(time.time())
        fallTime = int(time.time())
        for path, img, im0s, vid_cap in dataset:
            img_mask = self.thread.mask.getMask(scale=img.shape[3:1:-1])
            # img = img & img_mask
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # self.putImg2Queue(im0)            # Inference
            time_synchronized()
            pred = model(img, augment=augment, visualize=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            points = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                self.thread.mask.settingFence(im0, False, self.thread.mask.pointsList, (255, 0, 0))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    isFall = False
                    isInFence = False
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        if names[c] == 'person':
                            # print(label)
                            width, height = im0.shape[1], im0.shape[0]
                            p = (int(xyxy[0]) / width, int(xyxy[3]) / height)
                            points.append(p)
                            p1 = (int(xyxy[2]) / width, int(xyxy[3]) / height)
                            points.append(p1)
                            label = 'person'
                            # 跌倒判断
                            x1 = int(xyxy[0])
                            y1 = int(xyxy[1])
                            x2 = int(xyxy[2])
                            y2 = int(xyxy[3])
                            width1 = x2 - x1
                            height1 = y2 - y1
                            ratio = width1 / height1
                            if ratio > 2:
                                label = "fall"
                                isFall = True
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)

                    for j in points:
                        width, height = int(j[0] * img_mask.shape[3] - 1), int(j[1] * img_mask.shape[2] - 1)
                        if img_mask[0, 0, height, width] == 255:
                            isInFence = True
                            break

                    if isInFence:  # 判断电子围违规
                        fenceNow = int(time.time())
                        fenceTimePeriod = fenceNow - fenceTime
                        if fenceTimePeriod > 2:
                            # imgFilePath = saveImg('h:/voliateimg/', 1, im0)
                            self.putMessage2Queue(Message(1, '电子围栏违规', im0))  # Message是用来添加到语音播报队列的
                            sendimgtype(1, im0)
                            fenceTime = fenceNow

                    if isFall:
                        fallNow = int(time.time())
                        fallTimePeriod = fallNow - fallTime
                        if fallTimePeriod > 2:
                            # imgFilePath = saveImg('h:/voliateimg/', 4, im0)
                            self.putMessage2Queue(Message(4, '作业人员跌倒', im0))
                            # time.sleep(sleeptime(0, 0, 2))
                            sendimgtype(4, im0)
                            fallTime = fallNow

                self.putImg2Queue(im0)

            if self.thread.isStop:
                return


'''
class MediaPipeGloveDetect(Detect):
    def __init__(self, thread):
        super().__init__(thread)
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=10,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    def run(self,
            weights='yolov5s.pt.pt',  # model.pt path(s)
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
        dataset = LoadRTSPStreams(source)
        for path, image in dataset:
            img0=image.copy()
            img_mask = self.thread.mask.getOrginMask(scale=image.shape[1::-1])

            image = image & img_mask
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = self.hands.process(image)
            image_hight, image_width, _ = image.shape
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.thread.mask.settingFence(img0, False, self.thread.mask.pointsList, (255, 0, 0))
            
            if results.multi_hand_landmarks:
                # print(len(results.multi_hand_landmarks))
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    x_min = image_width;
                    y_min = image_hight;
                    x_max = 0;
                    y_max = 0;
                    avg = 0
                    for point_i in hand_landmarks.landmark:
                        if point_i.x > 1.0:
                            point_x = 1.0 * image_width
                        else:
                            point_x = point_i.x * image_width
                        if point_i.y > 1.0:
                            point_y = 1.0 * image_hight
                        else:
                            point_y = point_i.y * image_hight
                        # print("px",point_i.x)
                        # print("py", point_i.y)
                        if avg == 0:
                            pixpoint = image[(int)(point_y - 1), (int)(point_x - 1)]
                            avg = (pixpoint[0] * 117 + pixpoint[1] * 601 + pixpoint[2] * 306) >> 10
                        else:
                            pixpoint = image[(int)(point_y - 1), (int)(point_x - 1)]
                            tempgray = (pixpoint[0] * 117 + pixpoint[1] * 601 + pixpoint[2] * 306) >> 10
                            avg = (int)((avg + tempgray) / 2.0)
                        # print("gray",image[(int)(point_y),(int)(point_x)])
                        if point_x > 0 and point_x < x_min:
                            x_min = point_x
                        if point_x > 0 and point_x > x_max:
                            x_max = point_x

                        if point_y > 0 and point_y < y_min:
                            y_min = point_y
                        if point_y > 0 and point_y > y_max:
                            y_max = point_y
                    # print("avg", avg)
                    if int(x_min / 1.1) < 0:
                        x_min = 0
                    else:
                        x_min = int(x_min / 1.05)
                    if int(y_min / 1.1) < 0:
                        y_min = 0
                    else:
                        y_min = int(y_min / 1.05)
                    if int(x_max * 1.1) > image_width:
                        x_max = image_width
                    else:
                        x_max = int(x_max * 1.05)
                    if int(y_max * 1.1) > image_hight:
                        y_max = image_hight
                    else:
                        y_max = int(y_max * 1.05)
                    # path = "G:/dataset/hand_glove/validate/glove/"
                    # print(y_min, y_max)
                    # if k == ord("s"):
                    # cv2.imwrite(path + str(j) + ".jpg", image[y_min:y_max-1,x_min:x_max-1,:])

                    # print("kaishi")
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    type = ""
                    cv2.line(img0, (x_max, y_min), (x_min, y_min), (0, 0, 255))
                    cv2.line(img0, (x_max, y_min), (x_max, y_max), (0, 0, 255))
                    cv2.line(img0, (x_min, y_min), (x_min, y_max), (0, 0, 255))
                    cv2.line(img0, (x_max, y_max), (x_min, y_max), (0, 0, 255))
                    if avg > 58:
                        cv2.putText(img0, "hand", (x_min, y_min), font, 1, (0, 0, 255), 1, cv2.LINE_AA)  # 画文字
                        self.putMessage2Queue(Message(3, '佩戴绝缘手套违规', img0))
                    else:
                        cv2.putText(img0, "glove", (x_min, y_min), font, 1, (0, 0, 255), 1, cv2.LINE_AA)  # 画文字
            self.putImg2Queue(img0)
            if self.thread.isStop:
                return
                '''


class YoloGloveDetect(Detect):
    def __init__(self, thread):
        super().__init__(thread)

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
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Initialize
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        modelc = None
        if classify:
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(
                device).eval()

        # Dataloader
        print('stride', stride)
        if webcam:
            # print('stride',stride)
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        fenceTime = int(time.time())
        for path, img, im0s, vid_cap in dataset:
            img_mask = self.thread.mask.getMask(scale=img.shape[3:1:-1])
            img = img & img_mask
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            time_synchronized()
            pred = model(img, augment=augment, visualize=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            points = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                self.thread.mask.settingFence(im0, False, self.thread.mask.pointsList, (255, 0, 0))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    isFall = False
                    isInFence = False
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        if names[c] == 'hand':
                            centerx = int((int(xyxy[0]) + int(xyxy[2])) / 2)
                            centery = int((int(xyxy[3]) + int(xyxy[1])) / 2)
                            handwidth = int(xyxy[2] - xyxy[0])
                            handheight = int(xyxy[3] - xyxy[1])
                            # print(im0.shape)
                            pixpoint = im0[centery, centerx]
                            pixpoint1 = im0[centery, centerx - 1]
                            pixpoint2 = im0[centery - 1, centerx]
                            pixpoint3 = im0[centery + 1, centerx]
                            pixpoint4 = im0[centery, centerx + 1]
                            avg = (pixpoint[0] * 117 + pixpoint[1] * 601 + pixpoint[2] * 306) >> 10
                            gray1 = (pixpoint1[0] * 117 + pixpoint1[1] * 601 + pixpoint1[2] * 306) >> 10
                            gray2 = (pixpoint2[0] * 117 + pixpoint2[1] * 601 + pixpoint2[2] * 306) >> 10
                            gray3 = (pixpoint3[0] * 117 + pixpoint3[1] * 601 + pixpoint3[2] * 306) >> 10
                            gray4 = (pixpoint4[0] * 117 + pixpoint4[1] * 601 + pixpoint4[2] * 306) >> 10
                            avg = int((avg + gray1) / 2)
                            avg = int((avg + gray2) / 2)
                            avg = int((avg + gray3) / 2)
                            avg = int((avg + gray4) / 2)
                            # print("avg={0}".format(avg))
                            label = ''
                            if avg > 120 and handwidth > 40 and handheight > 40:
                                label = 'hand'
                                plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                             line_thickness=line_thickness)
                                fenceNow = int(time.time())
                                fenceTimePeriod = fenceNow - fenceTime
                                if fenceTimePeriod > 12:
                                    self.putMessage2Queue(Message(3, '佩戴绝缘手套违规', im0))
                                    sendimgtype(3, im0)
                                    fenceTime = fenceNow
                            else:
                                plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                             line_thickness=line_thickness)
                self.putImg2Queue(im0)

            if self.thread.isStop:
                return


class SmokeDetect(Detect):
    def __init__(self, thread):
        super().__init__(thread)

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
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        # Initialize
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        # Second-stage classifier
        classify = False
        modelc = None
        if classify:
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(
                device).eval()
        # Dataloader
        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        fenceTime = int(time.time())
        fallTime = int(time.time())
        for path, img, im0s, vid_cap in dataset:
            img_mask = self.thread.mask.getMask(scale=img.shape[3:1:-1])
            img = img & img_mask
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            time_synchronized()
            pred = model(img, augment=augment, visualize=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                self.thread.mask.settingFence(im0, False, self.thread.mask.pointsList, (255, 0, 0))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    isFall = False
                    isInFence = False
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = 'smoke'
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                     line_thickness=line_thickness)
                        fenceNow = int(time.time())
                        fenceTimePeriod = fenceNow - fenceTime
                        if fenceTimePeriod > 12:
                            # imgFilePath = saveImg('h:/voliateimg/', 5, im0)
                            self.putMessage2Queue(Message(6, '有人抽烟', im0))
                            # time.sleep(sleeptime(0, 0, 2))
                            sendimgtype(6, im0)
                            fenceTime = fenceNow
                self.putImg2Queue(im0)
            if self.thread.isStop:
                return


class ShoesDetect(Detect):
    def __init__(self, thread):
        super().__init__(thread)

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
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        # Initialize
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        # Second-stage classifier
        classify = False
        modelc = None
        if classify:
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(
                device).eval()
        # Dataloader
        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        fenceTime = int(time.time())
        fallTime = int(time.time())
        for path, img, im0s, vid_cap in dataset:
            img_mask = self.thread.mask.getMask(scale=img.shape[3:1:-1])
            img = img & img_mask
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            time_synchronized()
            pred = model(img, augment=augment, visualize=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                self.thread.mask.settingFence(im0, False, self.thread.mask.pointsList, (255, 0, 0))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    isFall = False
                    isInFence = False
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        if names[c] == 'hand':
                            centerx = int((int(xyxy[0]) + int(xyxy[2])) / 2)
                            centery = int((int(xyxy[3]) + int(xyxy[1])) / 2)
                            handwidth = int(xyxy[2] - xyxy[0])
                            handheight = int(xyxy[3] - xyxy[1])
                            # print(im0.shape)
                            pixpoint = im0[centery, centerx]
                            pixpoint1 = im0[centery, centerx - 1]
                            pixpoint2 = im0[centery - 1, centerx]
                            pixpoint3 = im0[centery + 1, centerx]
                            pixpoint4 = im0[centery, centerx + 1]
                            avg = (pixpoint[0] * 117 + pixpoint[1] * 601 + pixpoint[2] * 306) >> 10
                            gray1 = (pixpoint1[0] * 117 + pixpoint1[1] * 601 + pixpoint1[2] * 306) >> 10
                            gray2 = (pixpoint2[0] * 117 + pixpoint2[1] * 601 + pixpoint2[2] * 306) >> 10
                            gray3 = (pixpoint3[0] * 117 + pixpoint3[1] * 601 + pixpoint3[2] * 306) >> 10
                            gray4 = (pixpoint4[0] * 117 + pixpoint4[1] * 601 + pixpoint4[2] * 306) >> 10
                            avg = int((avg + gray1) / 2)
                            avg = int((avg + gray2) / 2)
                            avg = int((avg + gray3) / 2)
                            avg = int((avg + gray4) / 2)
                            # print("avg={0}".format(avg))
                            label = ''
                            if avg > 120 and handwidth > 40 and handheight > 40:
                                label = 'shoes'
                                plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                             line_thickness=line_thickness)
                                fenceNow = int(time.time())
                                fenceTimePeriod = fenceNow - fenceTime
                                if fenceTimePeriod > 12:
                                    self.putMessage2Queue(Message(5, '佩戴绝缘手套违规', im0))
                                    # time.sleep(sleeptime(0, 0, 2))
                                    sendimgtype(5, im0)
                                    fenceTime = fenceNow
                            else:
                                plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                             line_thickness=line_thickness)
                self.putImg2Queue(im0)
            if self.thread.isStop:
                return


class HelmetDetect(Detect):
    def __init__(self, thread):
        super().__init__(thread)

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
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        # Initialize
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        # Second-stage classifier
        classify = False
        modelc = None
        if classify:
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(
                device).eval()
        # Dataloader
        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        since = int(time.time())
        for path, img, im0s, vid_cap in dataset:
            img_mask = self.thread.mask.getMask(scale=img.shape[3:1:-1])
            img = img & img_mask
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            time_synchronized()
            pred = model(img, augment=augment, visualize=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                self.thread.mask.settingFence(im0, False, self.thread.mask.pointsList, (255, 0, 0))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    isFall = False
                    isInFence = False
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c]
                        if label == 'person':
                            isPerson = True
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                    if isPerson:  # 这里检测出是否安全帽违规
                        now = int(time.time())
                        timePeriod = now - since
                        if timePeriod > 2:
                            self.putMessage2Queue(Message(2, '佩戴安全帽违规', im0))
                            # time.sleep(sleeptime(0, 0, 2))
                            sendimgtype(2, im0)
                            since = now
                self.putImg2Queue(im0)
                if self.thread.isStop:
                    return


class Message:
    # 消息类型：
    #         1：进入危险区域
    #         2：安全帽违规
    #         3：绝缘手套违规
    #         4：作业人员跌倒
    #         5：绝缘靴违规
    #         6：抽烟

    def __init__(self, messageType, name, img):
        self.messageType = messageType
        self.name = name
        self.img = img


if __name__ == "__main__":
    print("Core module")
