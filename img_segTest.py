import matplotlib.pyplot as plt
import cv2
from FCMTEST import FCM
import numpy as np


def FCM_pic_cut0(img_path,gray=False,clustercenternum=5,iternum=10):
    if gray:
        img=cv2.imread(img_path,0)  #灰度图
        data=img.reshape(img.shape[0]*img.shape[1],1) #将图片拉成一列

    else:
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #转化为RGB，不然plt时图片会怪怪的
        data=img.reshape(-1,3)  # 将三维降成二维

    print("开始聚类")
    test=FCM(data,clustercenternum,iternum)
    #test.plot()
    cluster=test.label  # 聚类结果
    center=test.Clast # 聚类中心
    print("聚类结果",cluster.shape)
    print("聚类中心", center)
    print("聚类完成，开始生成图片")
    new_img=center[cluster] # 根据聚类结果和聚类中心构建新图像
    print(new_img.shape)
    new_img=np.reshape(new_img,img.shape) #矩阵转成原来图片的形状
    new_img=new_img.astype('uint8')  # 要变成图像得数据得转换成uint8
    if gray:
        plt.subplot(121), plt.imshow(img, cmap="gray"), plt.title("原图") # plt默认显示三通道，灰度图要加个cmap="gray"，不然图片绿绿的。。
        plt.subplot(122), plt.imshow(new_img, cmap="gray"), plt.title("FCM,%d个聚类中心"%clustercenternum)
    else :
        plt.subplot(121), plt.imshow(img), plt.title("原图")
        plt.subplot(122), plt.imshow(new_img), plt.title("FCM,%d个聚类中心"%clustercenternum)
    plt.show()
    #plt.imsave("cutgray.jpg",new_img) # 保存图片

if __name__ == '__main__':

    FCM_pic_cut0("Mai_sakurajima.jpg",gray=True,clustercenternum=2)



