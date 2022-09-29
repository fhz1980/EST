import random
import re
import socket
from datetime import *

import cv2
import pymysql
import pyttsx3

from urllib3 import encode_multipart_formdata
import requests
import time


class Dao:
    #query, insert, update, insert to the database
    conn = None

    @classmethod
    def connectionDB(cls, host, user, password, db, autocommit=True):
        cls.conn = pymysql.connect(host=host,
                                   user=user,
                                   password=password,
                                   db=db,
                                   autocommit=autocommit)

    @classmethod
    def query(cls, url):
        reExp = r'select\s+.*from\s+.*'
        if re.search(reExp, url):
            cur = cls.conn.cursor()
            try:
                cur.execute(url)
            except Exception as e:
                print("Failed", e)
            else:
                tempList = []
                for i in cur:
                    tempList.append(i)
                return tempList
            finally:
                cur.close()
        else:
            print("查询语句有误！")

    @classmethod
    def insert(cls, url):
        reExp = r'insert\s+into\s+.*\s+value\(.*\).*'
        if re.search(reExp, url):
            cur = cls.conn.cursor()
            try:
                cur.execute(url)
            except Exception as e:
                print("Failed", e)
            finally:
                cur.close()
        else:
            print("插入语句有误！")

    @classmethod
    def update(cls, url):
        reExp = r'update\s+.*\s+set\s.*\swhere\s.*'
        if re.search(reExp, url):
            cur = cls.conn.cursor()
            try:
                cur.execute(url)
            except Exception as e:
                print("Failed", e)
            finally:
                cur.close()
        else:
            print("更新语句有误！")

    @classmethod
    def delete(cls, url):
        reExp = r'delete\s+from\s.*\swhere\s.*'
        if re.search(reExp, url):
            cur = cls.conn.cursor()
            try:
                cur.execute(url)
            except Exception as e:
                print("Failed", e)
            finally:
                cur.close()
        else:
            print("删除语句有误！")


engine = pyttsx3.init()
'''
HOST = '172.26.20.173'
PORT = 9527
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.connect((HOST, PORT))

except Exception as e:
    print("Server not found or not open #01")
    # sys.exit()
'''

def word2voice(string):
    content = string  # 语音播报内容
    try:
        engine.say(content)  # 设置要播报的Unicode字符串
        engine.runAndWait()  # 等待语音播报完毕
    except Exception as e:
        print(e)

'''
def videoSocket(vedio):
    s.sendall(vedio.encode())
'''

def shock(shockLevel):
    LOCAL_HOST = "127.0.0.1"
    LOCAL_PORT = 9527
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((LOCAL_HOST, LOCAL_PORT))
    except Exception as e:
        print('Server not found or not open #02', e)
        # sys.exit()
    level = f'20:20:12:17:00:20#{shockLevel:02d}'
    str2bytes = level.encode(encoding='UTF-8', errors='strict')
    s.sendall(str2bytes)
    s.close()


def generateFileName():
    nowTime = datetime.now().strftime("%Y%m%d%H%M%S")  # 生成当前的时间
    randomNum = random.randint(0, 100)  # 生成随机数n,其中0<=n<=100
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum)
    uniqueNum = str(nowTime) + str(randomNum)
    return uniqueNum


def add(saveType, imgpath):
    Url = f'insert into infraction (`infractionTypes`,`time`, `saveDIR`) value(\'{saveType}\',now(),\'{imgpath}\')'
    Dao.insert(Url)

#这个是读取图片路径的,因为self.putMessage2Queue(Message(2, '佩戴安全帽违规', imgFilePath))中要有imgFilePaths，不然会报错，
#不知道怎么解决，其实imgFilePaths好像没有什么具体作用

def saveImg(saveDir, saveType, img):#在core.py的isInFence那里使用

    imgName = saveDir + generateFileName() + str(saveType) + '.jpg'
    # cv2.imwrite(imgName, img)这里是为了不让照片保存到磁盘中而注释掉的
    add(saveType, imgName)#这个不知道有什么作用
    return imgName

#将违规照片上传至web
def sendimgtype(type,img):

    data = {}
    header = {}

    success, encoded_frame = cv2.imencode(".jpg", img)
    # 1：电子围栏违规，2：安全帽违规，3：绝缘手套违规，4：跌倒，5：绝缘靴违规，6：抽烟违规
    if(type == 1):params = {'violateType': "1"}
    elif(type == 2):params = {'violateType': "2"}
    elif(type == 3):params = {'violateType': "3"}
    elif(type == 4):params = {'violateType': "4"}
    elif(type == 5):params = {'violateType': "5"}
    elif (type == 6):params = {'violateType': "6"}
    # print(imgName)
    data['fileName'] = ("fileName", encoded_frame.tobytes())
    encode_data = encode_multipart_formdata(data)
    data = encode_data[0]

    header['Content-Type'] = encode_data[1]
    result = requests.post("http://jxust-db/common/sysFile/uploadViolate", headers=header, params=params,
                           data=data)
    # print(result.text)


def sleeptime(hour, min, sec):#传入时间间隔，在时间间隔后再运行代码
    return hour * 3600 + min * 60 + sec


def get_centerpoint(lis):
    area = 0.0
    x,y = 0.0,0.0

    a = len(lis)
    for i in range(a):
        lat = lis[i][0] #weidu
        lng = lis[i][1] #jingdu

        if i == 0:
            lat1 = lis[-1][0]
            lng1 = lis[-1][1]

        else:
            lat1 = lis[i-1][0]
            lng1 = lis[i-1][1]

        fg = (lat*lng1 - lng*lat1)/2.0

        area += fg
        x += fg*(lat+lat1)/3.0
        y += fg*(lng+lng1)/3.0

    x = x/area
    y = y/area

    x = int(x*640)
    y = int(y*480)

    return [x,y]
