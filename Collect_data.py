"""
海康威视的网络摄像头调用
"""
import cv2

cap = cv2.VideoCapture('rtsp://admin:jxlgust123@172.26.20.51:554/Streaming/Channels/301?transportmode=unicast')

i = 0
while cap.isOpened():

    ret, frame = cap.read()

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(f"dataset/image{i}.jpg", frame)
        print(f"第{i}张图片采集成功")
        i = i + 1
cv2.destroyAllWindows()
cap.release()
