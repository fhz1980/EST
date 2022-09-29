# import threading
#
# import serial
#
#
# class MySerial:
#     """
#     @c 激光雷达
#     @f 测距
#     """
#     def __init__(self):
#         self.ser = serial.Serial()
#         self.ser.baudrate = 9600
#         self.ser.port = 'COM5'
#         self.ser.bytesize = 8
#         self.ser.stopbits = 1
#         self.ser.parity = 'N'
#         self.ser.open()
#         self.distance1 = -1
#         self.distance2 = -1
#         self.distance3 = -1
#
#     def get_distance1(self):
#         while True:
#             Hex_str = bytes.fromhex('01 03 00 00 00 04 44 09')
#             self.ser.write(Hex_str)
#             distance_hex = self.ser.read(13).hex()
#             self.distance1 = (int(distance_hex[6], 16) * 16 ** 3 + int(distance_hex[7], 16) * 16 ** 2 + int(distance_hex[8],
#                                                                                                             16) * 16 + int(
#                 distance_hex[9], 16)) / 1000
#             print(self.distance1)
#
#     def get_distance2(self):
#         while True:
#             Hex_str = bytes.fromhex('02 03 00 00 00 04 44 3A')
#             self.ser.write(Hex_str)
#             distance_hex = self.ser.read(13).hex()
#
#             self.distance2 = (int(distance_hex[6], 16) * 16 ** 3 + int(distance_hex[7], 16) * 16 ** 2 + int(distance_hex[8],
#                                                                                                             16) * 16 + int(
#                 distance_hex[9], 16)) / 1000
#             print(self.distance2)
#     def get_distance3(self):
#         while True:
#             Hex_str = bytes.fromhex('03 03 00 00 00 04 45 EB')
#             self.ser.write(Hex_str)
#             distance_hex = self.ser.read(13).hex()
#
#             self.distance3 = (int(distance_hex[6], 16) * 16 ** 3 + int(distance_hex[7], 16) * 16 ** 2 + int(distance_hex[8],
#                                                                                                             16) * 16 + int(
#                 distance_hex[9], 16)) / 1000
#             print(self.distance3)
#
#
#
#
# class A:
#     def __init__(self):
#         self.a = 1
#
#     def p(self):
#         while(True):
#             print("jao")
#
# my_ser = MySerial()
# while True:
#     threading.Thread(target=my_ser.get_distance1()).start()
#     threading.Thread(target=my_ser.get_distance2()).start()
#     threading.Thread(target=my_ser.get_distance3()).start()


# dictt = {}
#
# dictt[0] = "val"
#
# print(dictt[0])
# import torch
# x = torch.arange(10).view(2, 5)
# print(x)
# x = torch.flip(x, dims=[1])
# print(x)
# from win32com.client import Dispatch
# speaker = Dispatch('SAPI.SpVoice')
#
# speaker.Speak("流程错误")
import queue
q = queue.Queue(3)
q.put(1)
q.put(1)
q.put(1)
if q.empty():
    q.put(3)
    print(1)