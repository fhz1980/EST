import os
from PIL import Image


def renameFile(img_Dir):
    img_pathDir = os.listdir(img_Dir)  # 提取所有文件名，并存在列表中
    print("1", img_Dir)  # 输出文件路径
    print("2", img_pathDir)  # 输出文件名列表
    print(len(img_pathDir))  # 输出文件数
    for i in img_pathDir:
        img = Image.open(r"D:\datasets\Belt_person\Allimage" + "\\" + i)

        print("img", img)
        name = i.split(".")[0]
        number_name = r"D:\datasets\Belt_person\Allimage" + "\\" + name + '.png'  # 拼接路径
        print(number_name)

        img.save(number_name)  # 文件保存位置
    return


if __name__ == '__main__':
    img_Dir = 'D:\datasets\Belt_person\Allimage'  # 本地文件路径
    renameFile(img_Dir)
