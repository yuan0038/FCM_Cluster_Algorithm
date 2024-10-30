import random

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from sklearn.metrics import calinski_harabasz_score

m=2



class FCM:
    def __init__(self, data, clust_num,iter_num=10):
        self.data = data
        self.cnum = clust_num
        self.sample_num=data.shape[0]
        self.dim = data.shape[-1]  # 数据最后一维度数
        Jlist=[]   # 存储目标函数计算值的矩阵
        self.ch_score=[]
        U = self.Initial_U(self.sample_num, self.cnum)
        for i in range(0, iter_num): # 迭代次数默认为10
            C = self.Cen_Iter(self.data, U, self.cnum)
            U = self.U_Iter(U, C)
            print("第%d次迭代" %(i+1) ,end="")
            print("聚类中心",C)
            self.ch_score.append(calinski_harabasz_score(self.data, np.argmax(U, axis=0)))
            J = self.J_calcu(self.data, U, C)  # 计算目标函数
            Jlist = np.append(Jlist, J)
        self.label = np.argmax(U, axis=0)  # 所有样本的分类标签
        self.Clast = C    # 最后的类中心矩阵
        self.Jlist = Jlist  # 存储目标函数计算值的矩阵

    # 初始化隶属度矩阵U
    def Initial_U(self, sample_num, cluster_n):
        U = np.random.rand(sample_num, cluster_n)  # sample_num为样本个数, cluster_n为分类数
        row_sum = np.sum(U, axis=1)  # 按行求和 row_sum: sample_num*1
        row_sum = 1 / row_sum    # 该矩阵每个数取倒数
        U = np.multiply(U.T, row_sum)  # 确保U的每列和为1 (cluster_n*sample_num).*(sample_num*1)
        return U   # cluster_n*sample_num

    # 计算类中心
    def Cen_Iter(self, data, U, cluster_n):
        c_new = np.empty(shape=[0, self.dim])  # self.dim为样本矩阵的最后一维度
        for i in range(0, cluster_n):          # 如散点的dim为2，图片像素值的dim为1
            u_ij_m = U[i, :] ** m  # (sample_num,)
            sum_u = np.sum(u_ij_m)
            ux = np.dot(u_ij_m, data)  # (dim,)
            ux = np.reshape(ux, (1, self.dim))  # (1,dim)
            c_new = np.append(c_new, ux / sum_u, axis=0)   # 按列的方向添加类中心到类中心矩阵
        return c_new  # cluster_num*dim

    # 隶属度矩阵迭代
    def U_Iter(self, U, c):
        for i in range(0, self.cnum):
            for j in range(0, self.sample_num):
                sum = 0
                for k in range(0, self.cnum):
                    temp = (np.linalg.norm(self.data[j, :] - c[i, :]) /
                            np.linalg.norm(self.data[j, :] - c[k, :])) ** (
                                2 / (m - 1))
                    sum = temp + sum
                U[i, j] = 1 / sum

        return U

    # 计算目标函数值
    def J_calcu(self, data, U, c):
        temp1 = np.zeros(U.shape)
        for i in range(0, U.shape[0]):
            for j in range(0, U.shape[1]):
                temp1[i, j] = (np.linalg.norm(data[j, :] - c[i, :])) ** 2 * U[i, j] ** m

        J = np.sum(np.sum(temp1))
        print("目标函数值:%.2f" %J)
        return J


    # 打印聚类结果图
    def plot(self):

        mark = ['or', 'ob', 'og', 'om', 'oy', 'oc']  # 聚类点的颜色及形状

        if self.dim == 2:
            #第一张图
            plt.subplot(221)
            plt.plot(self.data[:, 0], self.data[:, 1],'ob',markersize=2)
            plt.title('未聚类前散点图')

            #第二张图
            plt.subplot(222)
            j = 0
            for i in self.label:
                plt.plot(self.data[j:j + 1, 0], self.data[j:j + 1, 1], mark[i],
                         markersize=2)
                j += 1

            plt.plot(self.Clast[:, 0], self.Clast[:, 1], 'k*', markersize=7)
            plt.title("聚类后结果")

            # 第三张图
            plt.subplot(212)
            plt.plot(self.Jlist, 'g-', )
            plt.title("目标函数变化图",)

            plt.show()
        elif self.dim==1:

            plt.subplot(221)
            plt.title("聚类前散点图")
            for j in range(0, self.data.shape[0]):
                plt.plot(self.data[j, 0], 'ob',markersize=3)  # 打印散点图

            plt.subplot(222)
            j = 0
            for i in self.label:
                plt.plot(self.data[j:j + 1, 0], mark[i], markersize=3)
                j += 1

            plt.plot([0]*self.Clast.shape[0],self.Clast[:, 0], 'k*',label='聚类中心',zorder=2)
            plt.title("聚类后结果图")
            plt.legend()
            # 第三张图
            plt.subplot(212)
            plt.plot(self.Jlist, 'g-', )
            plt.title("目标函数变化图", )
            plt.show()

        elif self.dim==3:
            # 第一张图
            fig = plt.figure()
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.scatter(self.data[:, 0], self.data[:, 1],self.data[:,2], "b")
            ax1.set_xlabel("X 轴")
            ax1.set_ylabel("Y 轴")
            ax1.set_zlabel("Z 轴")
            plt.title("未聚类前的图")

            # 第二张图
            ax2 = fig.add_subplot(222, projection='3d')

            j = 0

            for i in self.label:
                ax2.plot(self.data[j:j+1, 0], self.data[j:j+1, 1],self.data[j:j+1,2], mark[i],markersize=5)
                j += 1
            ax2.plot(self.Clast[:, 0], self.Clast[:, 1], self.Clast[:, 2], 'k*', label='聚类中心', markersize=8)

            plt.legend()


            ax2.set_xlabel("X 轴")
            ax2.set_ylabel("Y 轴")
            ax2.set_zlabel("Z 轴")
            plt.title("聚类后结果")
            # # 第三张图
            plt.subplot(212)
            plt.plot(self.Jlist, 'g-', )
            plt.title("目标函数变化图", )
            plt.show()



def example0():
    N=1000
    C=[[N/4,N/2,0,N/2],[N/2,N,0,N/2],[N/4,N/2,N/2,N],[N/2,N,N/2,N]]
    data=[]
    for i in range(4):
        center_datanum=random.randint(20,50)
        for j in range(center_datanum):
            change=random.randint(20,100)
            x=random.randint(C[i][0]+change,C[i][1]-change)
            y=random.randint(C[i][2]+change,C[i][3]-change)
            data.append([x,y])
    data=np.mat(data)
    a=FCM(data,4,20)

    print(a.ch_score)
    a.plot()
def example1():
    x1 = np.zeros((10, 1))
    x2 = np.zeros((10, 1))
    for i in range(0, 10):
        x1[i] = np.random.rand() * 5
        x2[i] = np.random.rand() * 5 + 5
        x = np.append(x1, x2, axis=0)


    a = FCM(x, 2,20)
    a.plot()
def example2():
    x1 = np.zeros((10, 1))
    y1 = np.zeros((10, 1))
    x2 = np.zeros((10, 1))
    y2 = np.zeros((10, 1))
    x3 = np.zeros((10, 1))
    y3 = np.zeros((10, 1))
    for i in range(0, 10):
        x1[i] = np.random.rand() * 5
        y1[i] = np.random.rand() * 5
        x2[i] = np.random.rand() * 5 + 5
        y2[i] = np.random.rand() * 5 + 5
        x3[i] = np.random.rand() * 0.5 + 1
        y3[i] = np.random.rand() * 0.5 + 1
    x = np.append(x1, x2, axis=0)
    x = np.append(x, x3, axis=0)
    y = np.append(y1, y2, axis=0)
    y = np.append(y, y3, axis=0)
    data = np.append(x, y, axis=1)

    a = FCM(data, 3,20)  # 将数据分为三类
    a.plot()    # 打印结果图
def example3():
    x1 = np.zeros((10, 1))
    y1 = np.zeros((10, 1))
    z1=  np.zeros((10, 1))
    x2 = np.zeros((10, 1))
    y2 = np.zeros((10, 1))
    z2 = np.zeros((10, 1))
    x3 = np.zeros((10, 1))
    y3 = np.zeros((10, 1))
    z3 = np.zeros((10, 1))
    for i in range(0, 10):
        x1[i] = np.random.rand() * 5
        y1[i] = np.random.rand() * 5
        z3[i] = np.random.rand() * 5
        x2[i] = np.random.rand() * 5 + 5
        y2[i] = np.random.rand() * 5 + 5
        z2[i] = np.random.rand() * 5+5
        x3[i] = np.random.rand() * 0.5 + 1
        y3[i] = np.random.rand() * 0.5 + 1
        z3[i] = np.random.rand() * 0.5 + 3

    x = np.append(x1, x2, axis=0)
    x = np.append(x, x3, axis=0)
    y = np.append(y1, y2, axis=0)
    y = np.append(y, y3, axis=0)
    z = np.append(z1, z2, axis=0)
    z = np.append(z, z3, axis=0)
    data = np.append(x, y, axis=1)
    print(data.shape)
    data=np.append(data,z,axis=1)

    a = FCM(data, 3,20)  # 将数据分为三类

    a.plot()  # 打印结果图


if __name__ == '__main__':
    example0()
    #example1()
    #example2()
    #example3()

