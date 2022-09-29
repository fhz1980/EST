import cv2
import time
from threading import Thread


class LoadRTSPStreamsFast:      # multiple IP or RTSP cameras
    def __init__(self, sources='rtsp://admin:jxlgust123@172.26.20.51:554/Streaming/Channels/301?transportmode=unicast'):
        self.sources = sources
        self.centerPointMask = []
        cap = cv2.VideoCapture(sources)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)     # 设置 buffer 缓冲区只用 3 帧
        assert cap.isOpened(), f'Failed to open {sources}'
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0      # 30 FPS fallback
        self.frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')        # infinite stream fallback
        _, self.imgs = cap.read()       # guarante first frame
        self.thread = Thread(target=self.update, args=([cap]), daemon=True)
        # print(f" success ({self.frames} frames {w}x{h} at {self.fps:.2f} FPS)")
        self.thread.start()

    def update(self, cap):
        while cap.isOpened():
            success, im = cap.read()
            self.imgs = im if success else self.imgs * 0
            time.sleep(1 / (2*self.fps))  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not self.thread.is_alive()  or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        # Letterbox
        # self.imgs = cv2.resize(self.imgs, (640, 480), interpolation=cv2.INTER_AREA)
        img0 = None
        if self.imgs is not None:
            img0 = self.imgs.copy()
        return self.sources, img0

    def __len__(self):
        return 1        # 1E12 frames = 32 streams at 30 FPS for 30 years

    def setCenterPointMask(self, points):
        self.centerPointMask = points