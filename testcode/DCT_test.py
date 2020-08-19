import cv2
import time
import math
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 读取图片,(转为YUV),进行dct变换, 查看一个通道/RGB三个通道的频谱图
# 再逆dct变换,求梯度,输出原图图片

# 二维DCT变换
def dct(image):
    temp = torch.zeros(image.shape)
    m, n = image.shape
    N = n
    temp[0, :] = torch.sqrt(torch.tensor(1/N))
    for i in range(1, m):
        for j in range(n):
            temp[i, j] = torch.cos(torch.tensor(math.pi*i*(2*j+1))/torch.tensor((2*N)))* torch.sqrt(torch.tensor(2/N))
    return torch.mm(torch.mm(temp,image), temp.T) , temp 
        
def idct(img_dct,temp):
    return torch.mm(torch.mm(temp.T,img_dct), temp)

def RGB2YUV(image):
    WR = 0.299
    WG = 0.587
    WB = 0.114
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    y = WR*r + WG*g + WB*b      # Y
    u = 0.5 *(b-y) / (1-WB)     # U
    v = 0.5 *(r-y) / (1-WR)     # V
    image[:,:,0] = y      # Y
    image[:,:,1] = u + 128     # U~(-128-127) + 128 -> (0-255)
    image[:,:,2] = v + 128    # V~(-128-127) + 128 -> (0-255)
    return image

def fill(img):
    width = len(img[0])
    height = len(img)
    if height % 8 != 0:
        for i in range(8 - height % 8):
            img.append([0 for i in range(width)])
    if width % 8 != 0:
        for row in img:
            for i in range(8 - width % 8):
                row.append(0)
    return img

def split(img):
    width = len(img[0])
    height = len(img)
    blocks = []
    for i in range(height // 8):
        for j in range(width // 8):
            temp = [[0 for i in range(8)] for i in range(8)]
            for r in range(8):
                for c in range(8):
                    temp[r][c] = img[i * 8 + r][j * 8 + c]
            blocks.append(temp)
    return blocks
        
# main
if __name__ == "__main__":
    img_org = cv2.imread('./JPEG/test.jpeg') 
    img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img = torch.Tensor(img_org)  

    # 转为YUV
    # img = RGB2YUV(img)
    
    img_s = torch.zeros(img.shape)
    img_dct_log = torch.zeros(img.shape)
    img_recor = torch.zeros(img.shape)

    for k in range(3):
        image = img[:, :, k]      # 获取rgb通道 0代表R通道,1代表G通道,2代表B通道 

        # 切成8*8 验证DCT
        # Y= fill(image.detach().numpy()) 
        # blocks = split(Y)
        # image = torch.Tensor(blocks[0])

        img_dct,temp = dct(image)
        img_s[:,:,k] = img_dct
        img_dct_log[:,:,k]= np.log(abs(img_dct))  #进行log处理

        img_s_var = Variable(img_dct,requires_grad=True)
        temp_var= Variable(temp)
        img_idct = idct(img_s_var,temp)
        img_idct.sum().backward()
        ds = img_s_var.grad
        img_recor[:,:,k] = img_idct
    print(ds)

    plt.subplot(131)
    plt.imshow(img_org)
    plt.title('original image(RGB)')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132)
    # 查看一个通道或三个通道的RGB
    # img_dct_log = np.clip(abs(img_dct_log),0,1)
    # plt.imshow(img_dct_log)
    plt.imshow(img_dct_log[:,:,0])
    plt.title('DCT-R')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133)
    plt.imshow(img_recor.detach().numpy().astype(np.uint8))
    plt.title('IDCT')
    plt.xticks([]), plt.yticks([])
    plt.show()

    print('done!')