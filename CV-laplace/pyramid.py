import numpy as np
import cv2
from matplotlib import pyplot as plt

# 构造高斯金字塔和拉普拉斯金字塔
def pyra(img):
    # 高斯金字塔的底层是原图像，从前到后分辨率降低
    Gaussian_apple = [img]
    Gau = img.copy()
    # 构建五层高斯金字塔
    for i in range(4):
        # 逐步下采样
        Gau = cv2.pyrDown(Gau)
        Gaussian_apple.append(Gau)
    # 拉普拉斯金字塔的底层是高斯金字塔的顶层，从前到后分辨率升高
    Laplace_apple = [Gaussian_apple[4]]
    # 构建五层拉普拉斯金字塔
    for i in range(4, 0, -1):
        # 高斯金字塔上采样
        GausUp = cv2.pyrUp(Gaussian_apple[i])
        # 拉普拉斯金字塔中的第i层，等于高斯金字塔中的第i层与高斯金字塔中的第i+1层的向上采样结果之差
        Lap = cv2.subtract(Gaussian_apple[i-1], SameSize(GausUp, Gaussian_apple[i-1]))
        Laplace_apple.append(Lap)
    return Gaussian_apple, Laplace_apple

# 求取融合图像拉普拉斯金字塔，也即将两个拉普拉斯金字塔逐层按mask拼接
def lap_blend(Laplace_apple, Laplace_orange):
    # 构建融合的拉普拉斯金字塔
    Lap_b = []
    for la, lo in zip(Laplace_apple, Laplace_orange):
        rows, cols, deepth = la.shape
        # 这里可以理解为mask，由于左右拼接所以不必专门构建mask金字塔
        l_blend = np.hstack((la[:, 0:int(cols / 2)], lo[:, int(cols / 2):]))
        # 曾报错，slice即列表切片，所以意为切片应为整型或者none，列表加上int()
        Lap_b.append(l_blend)
    # 返回列表
    return Lap_b

# 把两个高斯金字塔顶端融合图作为起始，进行逐层上采样并与lap_b加和，最后得到原始分辨率融合图像
def blend(lap_b):
    img_blend = lap_b[0]
    # 这是起始图像，这是两个高斯金字塔的顶层的拼接
    for i in range(1, 5):
        img_blend = cv2.pyrUp(img_blend)
        # 上采样
        img_blend = cv2.add(SameSize(img_blend, lap_b[i]), lap_b[i])
        # 加上融合的拉普拉斯金字塔
    # 返回融合图像
    return img_blend

# 相同分辨率处理
def SameSize(img1, img2):
    # 将img1图像转化为img2的大小
    rows, cols, deepth = img2.shape
    dst = img1[:rows, :cols]
    return dst

# 画图
def draw(img_apple, img_orange, img_blend):
    # subplot括号的含义是1行3列 第1个
    plt.subplot(131), plt.imshow(cv2.cvtColor(img_apple, cv2.COLOR_BGR2RGB))
    plt.title("apple"), plt.xticks([]), plt.yticks([])
    # subplot括号的含义是1行3列 第2个
    plt.subplot(132), plt.imshow(cv2.cvtColor(img_orange, cv2.COLOR_BGR2RGB))
    plt.title("orange"), plt.xticks([]), plt.yticks([])
    # subplot括号的含义是1行3列 第3个
    plt.subplot(133), plt.imshow(cv2.cvtColor(img_blend, cv2.COLOR_BGR2RGB))
    plt.title("img_blend"), plt.xticks([]), plt.yticks([])
    plt.show()
def draw_gaus(Gaussian_apple):
    # 画高斯金字塔
    plt.subplot(131), plt.imshow(cv2.cvtColor(Gaussian_apple[1], cv2.COLOR_BGR2RGB))
    plt.title("Gaussian_apple[1]"), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(cv2.cvtColor(Gaussian_apple[2], cv2.COLOR_BGR2RGB))
    plt.title("Gaussian_apple[2]"), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(cv2.cvtColor(Gaussian_apple[3], cv2.COLOR_BGR2RGB))
    plt.title("Gaussian_apple[3]"), plt.xticks([]), plt.yticks([])
    plt.show()
def draw_lap(Laplace_apple):
    # 画拉普拉斯金字塔
    # print(Laplace_apple[1])
    plt.subplot(131), plt.imshow(cv2.cvtColor(Laplace_apple[1], cv2.COLOR_BGR2GRAY))
    # print((cv2.cvtColor(Laplace_apple[1], cv2.COLOR_BGR2RGB)).shape)
    # print(cv2.cvtColor(Laplace_apple[1], cv2.COLOR_BGR2RGB))
    plt.title("Laplace_apple[1]"), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(cv2.cvtColor(Laplace_apple[2], cv2.COLOR_BGR2GRAY))
    plt.title("Laplace_apple[2]"), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(cv2.cvtColor(Laplace_apple[3], cv2.COLOR_BGR2GRAY))
    plt.title("Laplace_apple[3]"), plt.xticks([]), plt.yticks([])
    plt.show()

# 读取图像，作相同分辨率处理
img_apple = cv2.imread('apple.png')
img_orange = cv2.imread('orange.png')
img_apple = SameSize(img_apple, img_orange)
# 求取高斯和拉普拉斯金字塔
Gaussian_apple, Laplace_apple = pyra(img_apple)
Gaussian_orange, Laplace_orange = pyra(img_orange)
# 求取融合图像拉普拉斯金字塔
lap_b = lap_blend(Laplace_apple, Laplace_orange)
# 把两个高斯金字塔顶端融合图作为起始，利用lap_b进行逐层上采样和加和，最后得到原始分辨率融合图像
img_blend = blend(lap_b)
# 图像可视化
cv2.imwrite('blend.png', img_blend)
draw(img_apple, img_orange, img_blend)
draw_gaus(Gaussian_apple)
draw_lap(Laplace_apple)