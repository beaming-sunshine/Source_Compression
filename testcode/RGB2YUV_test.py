import numpy as np
import cv2
import matplotlib.pyplot as plt

# 颜色模型转换与二次采样
def rgb2yuv(img, width, height):
    Y = [[0 for i in range(width)] for i in range(height)]
    U = [[0 for i in range((width - 1) // 2 + 1)] for i in range((height - 1) // 2 + 1)]
    V = [[0 for i in range((width - 1) // 2 + 1)] for i in range((height - 1) // 2 + 1)]
    WR = 0.299
    WG = 0.587
    WB = 0.114
    for i in range(height):
        flag = False
        if i % 2 == 1 or i == height - 1:
            flag = True
        for j in range(width):
            B = img[i][j][0]
            G = img[i][j][1]
            R = img[i][j][2]
            Y[i][j] = (WR * R + WG * G + WB * B)
            if i % 2 == 0 and j % 2 == 0:
                U[i // 2][j // 2] = 0.5 *(((-WR)/(1-WB)) * R + ((-WG)/(1-WB)) * G + B)
            if flag:
                V[i // 2][j // 2] = 0.5 *(((-WB)/(1-WR)) * B + ((-WG)/(1-WR)) * G + R)
    return Y, U, V 


def yuv2rgb(Y, U, V, width, height):
    WR = 0.299
    WG = 0.587
    WB = 0.114    
    img = np.zeros([height, width, 3], dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img[i][j][0] = np.clip(Y[i][j]+2*(1-WB) * (U[i//2][j//2]),0,255 )#  B
            img[i][j][1] = np.clip(Y[i][j]-(2*(1-WB)*WB/WG)*(U[i//2][j//2])-(2*(1-WR)*WR/WG)*(V[i//2][j//2] ),0,255)
            img[i][j][2] = np.clip(Y[i][j]+2*(1-WR) * (V[i//2][j//2]),0,255 )
    return img


if __name__ == "__main__":

    # 显示YUV图像
    img = cv2.imread("test.jpg")
    Y, U, V = rgb2yuv(img, img.shape[1], img.shape[0])
    imgY = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    imgU = np.zeros([img.shape[0] // 2, img.shape[1] // 2], dtype=np.uint8)
    imgV = np.zeros([img.shape[0] // 2, img.shape[1] // 2], dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imgY[i][j] = Y[i][j]
    for i in range(int(img.shape[0]/2)):
        for j in range(int(img.shape[1]/2)):
            imgU[i][j] = U[i][j] + 128    #  (U~(-128-127)) + 128 -> (0-255)
            imgV[i][j] = V[i][j] + 128
    cv2.imshow("created_imgY", imgY)
    cv2.imshow("created_imgU", imgU)
    cv2.imshow("created_imgV", imgV)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # RGB->YUV->RGB
    img = cv2.imread("test.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title("original")

    Y, U, V = rgb2yuv(img, img.shape[1], img.shape[0])
    img = yuv2rgb(Y, U, V, img.shape[1], img.shape[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.title("after trans")
    plt.show()
