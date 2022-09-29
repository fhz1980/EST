# -*- coding: utf-8 -*-
# xml解析包
import xml.etree.ElementTree as ET
import pickle
import os
# os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
from os import listdir, getcwd
from os.path import join
import shutil
import random
import math
# 进行归一化操作
def convert(size, box): # size:(原图w,原图h) , box:(xmin,xmax,ymin,ymax)
    dw = 1./size[0]     # 1/w
    dh = 1./size[1]     # 1/h
    x = (box[0] + box[1])/2.0   # 物体在图中的中心点x坐标
    y = (box[2] + box[3])/2.0   # 物体在图中的中心点y坐标
    w = box[1] - box[0]         # 物体实际像素宽度
    h = box[3] - box[2]         # 物体实际像素高度
    x = x*dw    # 物体中心点x的坐标比(相当于 x/原图w)
    w = w*dw    # 物体宽度的宽度比(相当于 w/原图w)
    y = y*dh    # 物体中心点y的坐标比(相当于 y/原图h)
    h = h*dh    # 物体宽度的宽度比(相当于 h/原图h)
    return (x, y, w, h)    # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]
# year ='2012', 对应图片的id（文件名）
def convert_annotation(anndir,textdir,image_id,classes):
    '''
    将对应文件名的xml文件转化为label文件，xml文件包含了对应的bunding框以及图片长款大小等信息，
    通过对其解析，然后进行归一化最终读到label文件中去，也就是说
    一张图片文件对应一个xml文件，然后通过解析和归一化，能够将对应的信息保存到唯一一个label文件中去
    labal文件中的格式：calss x y w h　　同时，一张图片对应的类别有多个，所以对应的ｂｕｎｄｉｎｇ的信息也有多个
    '''
    # 对应的通过year 找到相应的文件夹，并且打开相应image_id的xml文件，其对应bund文件
    in_file = open(anndir+'/%s.xml' % (image_id), encoding='utf-8')
    # 准备在对应的image_id 中写入对应的label，分别为
    # <object-class> <x> <y> <width> <height>
    out_file = open(textdir+'/%s.txt' % (image_id), 'w', encoding='utf-8')
    # 解析xml文件
    tree = ET.parse(in_file)
    # 获得对应的键值对
    root = tree.getroot()
    # 获得图片的尺寸大小
    size = root.find('size')
    # 如果xml内的标记为空，增加判断条件
    if size != None:
        # 获得宽
        w = int(size.find('width').text)
        # 获得高
        h = int(size.find('height').text)
        # 遍历目标obj
        for obj in root.iter('object'):
            # 获得difficult ？？
            difficult = obj.find('difficult').text
            # 获得类别 =string 类型
            cls = obj.find('name').text
            # 如果类别不是对应在我们预定好的class文件中，或difficult==1则跳过
            if cls not in classes or int(difficult) == 1:
                continue
            # 通过类别名称找到id
            cls_id = classes.index(cls)
            # 找到bndbox 对象
            xmlbox = obj.find('bndbox')
            # 获取对应的bndbox的数组 = ['xmin','xmax','ymin','ymax']
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            print(image_id, cls, b)
            # 带入进行归一化操作
            # w = 宽, h = 高， b= bndbox的数组 = ['xmin','xmax','ymin','ymax']
            bb = convert((w, h), b)
            # bb 对应的是归一化后的(x,y,w,h)
            # 生成 calss x y w h 在label文件中
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
# sourcefile 为存放样本的文本文件，
# fromimgdir 图片原本所在文件夹
# toimgdir 目的图片原本所在文件夹
# anndir anntation xml注解文件夹，
# txtfile 存放分类图片的文件夹，
# textdir 存放yolo训练标准的文件夹，
# classes 标注的类别
def generate(sourcefile,fromimgdir,toimgdir,anndir,txtfile,textdir,classes):

    # 先找labels文件夹如果不存在则创建
    if not os.path.exists(toimgdir):
        os.makedirs(toimgdir)
    if not os.path.exists(textdir):
        os.makedirs(textdir)
    # 读取在ImageSets/Main 中的train、test..等文件的内容
    # 包含对应的文件名称
    image_ids = open(sourcefile).read().strip().split()
    # 打开对应的2012_train.txt 文件对其进行写入准备
    list_file = open(txtfile, 'w')
    # 将对应的文件_id以及全路径写进去并换行
    for image_id in image_ids:
        list_file.write(toimgdir+'/%s.jpg\n' % (image_id))
        # list_file.write(toimgdir+'/%s.png\n' % (image_id))
        # 调用  year = 年份  image_id = 对应的文件名_id
        shutil.copy(fromimgdir+'/%s.jpg' % (image_id),toimgdir+'/%s.jpg' % (image_id))
        # shutil.copy(fromimgdir+'/%s.png' % (image_id),toimgdir+'/%s.png' % (image_id))
        convert_annotation(anndir,textdir,image_id,classes)
    # 关闭文件
    list_file.close()

#随机分配训练和测试样本
def ann_make_txt(anndir,imgdir):
    fo1 = open(imgdir+'/test.txt', 'w')
    fo2 = open(imgdir+'/trainval.txt', 'w')
    fo3 = open(imgdir+'/train.txt', 'w')
    fo4 = open(imgdir+'/val.txt', 'w')
    filepath = anndir
    filelist = os.listdir(filepath)
    file_name = []
    for w in filelist:
        file_name.append(w.replace('.xml', ''))
    file_num = len(file_name)
    trainval_num = 0.8
    train_num = 0.8
    # trainval
    trainval_list = random.sample(range(file_num), math.floor(trainval_num * file_num))
    # test
    test_list = (list(set(range(file_num)).difference(set(trainval_list))))
    random.shuffle(test_list)
    # train
    train_list = random.sample(trainval_list, math.floor(train_num * len(trainval_list)))
    # val
    val_list = list(set(trainval_list).difference(set(train_list)))
    random.shuffle(val_list)
    # put in txt
    for i in trainval_list:
        fo2.write(file_name[i] + '\n')
    for i in test_list:
        fo1.write(file_name[i] + '\n')
    for i in train_list:
        fo3.write(file_name[i] + '\n')
    for i in val_list:
        fo4.write(file_name[i] + '\n')
    fo1.close()
    fo2.close()
    fo3.close()
    fo4.close()
def main(imgspath,annpath,images,labels,classes):
    ann_make_txt(annpath,imgspath)
    sets = ['test', 'val', 'train', 'trainval']
    for image_set in sets:
        print(image_set)
        sourcefile = imgspath + '/' + image_set + '.txt'
        toimgdir = images + '/' + image_set
        txtfile = labels + '/' + image_set + '.txt'
        textdir = labels + '/' + image_set
        generate(sourcefile, imgspath, toimgdir, annpath, txtfile, textdir, classes)
if __name__ == "__main__":
    wd = getcwd()
    print(wd)
    imgspath=r'G:/dataset/smoking/smokingimages'# imgspath='G:/dataset/VOCdevkit-yolo/VOC2012/JPEGImages'#图片路径
    annpath=r'G:/dataset/smoking/Annotations'#annpath='G:/dataset/VOCdevkit-yolo/VOC2012/Annotations'#XML路径
    images=r'G:/dataset/smoking/images'#保存路径
    labels=r'G:/dataset/smoking/labels'#保存路径
    # classes = ["c1","c2","c3","c4","c5"]
    classes = ['smoke']
    #生成训练测试集
    main(imgspath, annpath, images, labels, classes)