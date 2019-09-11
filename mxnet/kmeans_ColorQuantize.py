import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math


#用mu律做一些展缩优化看看，发现效果更差。这合理，因为图像
# 很多像素的绝对值是比较大的，不像音频很多样本的绝对值比较小，展缩就有帮助。
BASE = math.e
RATE=3 #大小决定曲线的弯曲度
SCALE = 255 # 范围
def mu_law(x):
    xx = x / SCALE
    return SCALE* math.log(1+RATE*xx, BASE) / math.log(1+RATE ,BASE)
def mu_law_back(y):
    yy = y / SCALE
    return (math.pow(BASE, yy * math.log(1+RATE,BASE))-1)/RATE * SCALE
######################################################################

def scatter(p): #三维散点图
    ax = plt.figure().add_subplot(111, projection='3d')
    xs = [x[0] for x in p]
    ys = [x[1] for x in p]
    zs = [x[2] for x in p]
    cs = np.array([x[3] for x in p])
    ax.scatter(xs, ys, zs, c=cs)

    # 设置坐标轴
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # 显示图像
    plt.show()



def compressImg(path:str):
    img = cv2.imread(path) # type:np.ndarray
    # img shape: (h, w, c) BGR
    print(type(img), img.shape, img.dtype)
    resizedImg = cv2.resize(img, (img.shape[1]//5, img.shape[0]//5)) #缩小图片，用它的像素来训练kmeans会快些
    print(type(resizedImg), resizedImg.shape)
    pixels = resizedImg.reshape((-1,3))
    km = KMeans(n_clusters=64)
    km.fit(pixels)
    print("kmeans fitting finished!")

    #用kmeans聚类出来的簇中心点的值作为调色板颜色，图片结果保存在img2
    img2 = np.zeros(img.shape, dtype=img.dtype)
    hist2 = np.zeros(img.shape[:2], dtype="uint32").reshape(-1) #直方图
    p2 = list() # 散点图
    cnt = 0
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            pallete = km.predict([img[h,w]])
            center = km.cluster_centers_[pallete[0]]
            img2[h,w] = center
            hist2[cnt] = center[0]*65536+center[1]*256+center[2]
            cnt += 1
            if random.randint(0, 1000) > 995:#抽样生成散点图，要不然点太多
                p2.append((img[h,w,0],img[h,w,1],img[h,w,2], [img2[h,w,2]/255,img2[h,w,1]/255,img2[h,w,0]/255]  ))


    #简单粗暴的对像素进行量化，结果保存在img3
    img3 = np.zeros(img.shape, dtype=img.dtype)
    hist3 = np.zeros(img.shape[:2], dtype="uint32").reshape(-1)
    cnt = 0
    p3 = list()
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            for c in range(img.shape[2]):
                img3[h,w,c] = img[h,w,c]//64 * 64
                #img3[h,w,c] = round(mu_law_back(round(mu_law(img[h,w,c])) // 64 *64))
            #hist3[cnt] = img3[h,w,0]*65536+img3[h,w,1]*256+img3[h,w,2]
            cnt += 1
            if random.randint(0, 1000) > 995:#抽样生成散点图，要不然点太多
                p3.append( (img[h, w, 0], img[h, w, 1], img[h, w, 2],  [img3[h, w, 2]/255, img3[h, w, 1]/255, img3[h, w, 0]/255] ))

    toshow=np.vstack((img2,img3))
    cv2.namedWindow("kmeans")
    cv2.imshow("kmeans", toshow)
    cv2.waitKey(0)

    #显示他们的直方图
    '''plt.figure()
    plt.subplot(1,2,1)
    plt.hist(hist2, bins=64,)
    plt.subplot(1,2,2)
    plt.hist(hist3, bins=64)
    plt.show()'''

    scatter(p2)
    scatter(p3)

compressImg("e:\\bazha.jpg")



