import time

import numpy as np
import torch
import tracker
from detector import Detector
from Core import Detect
from utils.rtspdatasetfast import LoadRTSPStreamsFast
import cv2
import os
import queue
import threading
import torch.backends.cudnn as cudnn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PersonCountMT(Detect):
    def __init__(self, thread):  # thread
        super().__init__(thread)  # thread
        # 根据视频尺寸，填充一个polygon，供撞线计算使用
        mask_image_temp = np.zeros((720,1280), dtype=np.uint8)

        # 初始化2个撞线polygon
        list_pts_blue = [[0, 300], [1270, 300], [1270, 350], [0, 350]]
        ndarray_pts_blue = np.array(list_pts_blue, np.int32)
        polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
        polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

        # 填充第二个polygon
        mask_image_temp = np.zeros((720,1280), dtype=np.uint8)
        list_pts_yellow = [[0, 360], [1270, 360], [1270, 410], [0, 410]]
        ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
        polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
        polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

        # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
        polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

        # 缩小尺寸，1920x1080->960x540
        self.polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (1280, 720))

        # 蓝 色盘 b,g,r
        blue_color_plate = [255, 0, 0]
        # 蓝 polygon图片
        blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

        # 黄 色盘
        yellow_color_plate = [0, 255, 255]
        # 黄 polygon图片
        yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

        # 彩色图片（值范围 0-255）
        color_polygons_image = blue_image + yellow_image
        # 缩小尺寸，1920x1080->960x540
        self.color_polygons_image = cv2.resize(color_polygons_image, (1280, 720))

        # list 与蓝色polygon重叠
        self.list_overlapping_blue_polygon = []

        # list 与黄色polygon重叠
        self.list_overlapping_yellow_polygon = []

        # 进入数量
        self.down_count = 0
        # 离开数量
        self.up_count = 0

        self.font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
        self.draw_text_postion = (int(1280 * 0.01), int(720 * 0.05))
        self.q = queue.Queue(maxsize=20)
        # 初始化 yolov5
        self.detector = Detector("G:\\ZSL\\DLPX\\src\\models\\detect_model\\person.pt")
    @torch.no_grad()
    def run(self,
            weights='yolov5s.pt',  # model.pt path(s)
            source='0',  # file/dir/URL/glob, 0 for webcam
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
        # print("8888888888888")
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        # Initialize
        work1 = threading.Thread(target=self.imgop, args=())
        work2 = threading.Thread(target=self.imgop, args=())
        work1.start()
        work2.start()
        self.q.join()

        dataset = LoadRTSPStreamsFast(source)
        for path, im in dataset:
            if self.q.full():
                self.q.get()
                self.q.put(im)
            else:
                self.q.put(im)
        if self.thread.isStop:
            return

    def imgop(self):
        while True:
            time.sleep(0.02)
            # print(self.q.empty())
            if not self.q.empty():
                # print('7777777777777777')
                im=self.q.get()
                list_bboxs = []
                # start = time.time();
                bboxes = self.detector.detect(im)
                # print('detect', int((time.time() - start) * 1000))
                # 如果画面中 有bbox
                if len(bboxes) > 0:
                    # times = time.time();
                    list_bboxs = tracker.update(bboxes, im)
                    # print('tracker', int((time.time() - times) * 1000))
                    # 画框
                    # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                    output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
                    pass
                else:
                    # 如果画面中 没有bbox
                    output_image_frame = im
                pass
                output_image_frame = cv2.add(output_image_frame, self.color_polygons_image)
                if len(list_bboxs) > 0:
                    # ----------------------判断撞线----------------------
                    for item_bbox in list_bboxs:
                        x1, y1, x2, y2, label, track_id = item_bbox

                        # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                        y1_offset = int(y1 + ((y2 - y1) * 0.6))

                        # 撞线的点
                        y = y1_offset
                        x = x1

                        if self.polygon_mask_blue_and_yellow[y, x] == 1:
                            # 如果撞 蓝polygon
                            if track_id not in self.list_overlapping_blue_polygon:
                                self.list_overlapping_blue_polygon.append(track_id)
                            pass

                            # 判断 黄polygon list 里是否有此 track_id
                            # 有此 track_id，则 认为是 外出方向
                            if track_id in self.list_overlapping_yellow_polygon:
                                # 外出+1
                                self.up_count += 1

                                print(
                                    f'类别: {label} | id: {track_id} | 上行撞线 | 上行撞线总数: {self.up_count} | 上行id列表: {self.list_overlapping_yellow_polygon}')

                                # 删除 黄polygon list 中的此id
                                self.list_overlapping_yellow_polygon.remove(track_id)

                                pass
                            else:
                                # 无此 track_id，不做其他操作
                                pass

                        elif self.polygon_mask_blue_and_yellow[y, x] == 2:
                            # 如果撞 黄polygon
                            if track_id not in self.list_overlapping_yellow_polygon:
                                self.list_overlapping_yellow_polygon.append(track_id)
                            pass

                            # 判断 蓝polygon list 里是否有此 track_id
                            # 有此 track_id，则 认为是 进入方向
                            if track_id in self.list_overlapping_blue_polygon:
                                # 进入+1
                                self.down_count += 1

                                print(
                                    f'类别: {label} | id: {track_id} | 下行撞线 | 下行撞线总数: {self.down_count} | 下行id列表: {self.list_overlapping_blue_polygon}')

                                # 删除 蓝polygon list 中的此id
                                self.list_overlapping_blue_polygon.remove(track_id)

                                pass
                            else:
                                # 无此 track_id，不做其他操作
                                pass
                            pass
                        else:
                            pass
                        pass

                    pass

                    # ----------------------清除无用id----------------------
                    list_overlapping_all = self.list_overlapping_yellow_polygon + self.list_overlapping_blue_polygon
                    for id1 in list_overlapping_all:
                        is_found = False
                        for _, _, _, _, _, bbox_id in list_bboxs:
                            if bbox_id == id1:
                                is_found = True
                                break
                            pass
                        pass

                        if not is_found:
                            # 如果没找到，删除id
                            if id1 in self.list_overlapping_yellow_polygon:
                                self.list_overlapping_yellow_polygon.remove(id1)
                            pass
                            if id1 in self.list_overlapping_blue_polygon:
                                self.list_overlapping_blue_polygon.remove(id1)
                            pass
                        pass
                    list_overlapping_all.clear()
                    pass

                    # 清空list
                    list_bboxs.clear()

                    pass
                else:
                    # 如果图像中没有任何的bbox，则清空list
                    self.list_overlapping_blue_polygon.clear()
                    self.list_overlapping_yellow_polygon.clear()
                    pass
                pass

                text_draw = 'DOWN: ' + str(self.down_count) + \
                            ' , UP: ' + str(self.up_count)
                output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                                 org=self.draw_text_postion,
                                                 fontFace=self.font_draw_number,
                                                 fontScale=1, color=(0, 255, 0), thickness=2)

                self.putImg2Queue(output_image_frame)
                # cv2.imshow('demo', output_image_frame)
                # cv2.waitKey(1)

                pass

            pass


