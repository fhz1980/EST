import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
import time
import math
from threading import Thread
from collections import Iterable
from collections import Iterator


class getRealSenseStreams:
    def __init__(self, sources='rtsp://admin:jxlgust123@172.26.20.51:554/Streaming/Channels/301?transportmode=unicast'):

        self.sources = sources
        headers = {
            'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}

        # 开启流和配置相机参数

        self.imgs = None
        self.deep_image = None
        self.fps = 30

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Start streaming

        self.pipeline.start(config)  # 流程开始
        align_to = rs.stream.color  # 与color流对齐
        self.align = rs.align(align_to)

        self.thread = Thread(target=self.update, args=([True]), daemon=True)
        # print(f" success ({self.frames} frames {w}x{h} at {self.fps:.2f} FPS)")
        self.thread.start()

    def update(self, bo):
        while bo:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
            color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）

            depth_frame = frames.get_depth_frame()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            self.imgs = color_image
            self.deep_image = depth_image
            time.sleep(1 / self.fps)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not self.thread.is_alive() or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        img0 = None
        if self.imgs is not None:
            img0 = self.imgs

        return img0, self.deep_image

    def __len__(self):
        return 1

    def get_mid_pos(self, frame, box, depth_data, randnum):  # detect distance
        distance_list = []
        mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]  # 确定索引深度的中心像素位置
        # print(mid_pos)mid_pos里面的数据是浮点数，不是整数，cv2.circle中的坐标要整数

        min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1]))  # 确定深度搜索范围
        # print(box,)
        # input()
        """
        random.randint(参数1，参数2)

        参数1、参数2必须是整数
        函数返回参数1和参数2之间的任意整数， 闭区间
        """
        for i in range(randnum):
            bias = random.randint(-min_val // 4, min_val // 4)
            dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
            cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255, 0, 0), -1)
            # print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
            if dist:
                distance_list.append(dist)
        distance_list = np.array(distance_list)
        distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]  # 冒泡排序+中值滤波

        return np.mean(distance_list)

