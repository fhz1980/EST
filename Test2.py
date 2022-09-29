# a = r'D:\\code\\yolov7\\weights.Head.rod_head'
# weights = a.split(".")
# print(weights)
# weight1 = weights[0] + weights[1]
# weight2 = weights[0] + weights[2]
# print(weight1, weight2)




a= [[12,2],[34,2,5],[2,56],[2]]

'''
zip()循环次数以少的list为准
'''
# a = [1,2]
# b = [1,2,3]
# for x,y in zip(a,b):
#     print('x=',x,'y=',y)#

'''
不可以使用test.py这种命名,因为Anaconda自带的环境中有这个文件,会造成重定义问题

C:\ProgramData\Anaconda3\lib\test\__init__.py

ImportError: cannot import name 'test1' from 'test'  (C:\ProgramData\Anaconda3\lib\test\__init__.py)

解决方法:

将test.py重命名为test2.py
————————————————
版权声明：本文为CSDN博主「king52113141314」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/king52113141314/article/details/108363939
'''