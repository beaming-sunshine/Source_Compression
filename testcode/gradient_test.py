import os
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_W(gradients):
    # 图片分别在三个通道的梯度均值
    grad_avg = torch.mean(gradients,dim=0)
    gB = grad_avg[0,:,:]
    gR = grad_avg[1,:,:]
    gG = grad_avg[2,:,:]
    # 计算G-YUV参数
    Z1 = (gB/gG).mean()
    Z2 = (gR/gG).mean()
    WR = Z2/(1+Z1+Z2)
    WG = 1/(1+Z1+Z2)
    WB = Z1/(1+Z1+Z2)
    return WR, WG, WB

def compute_gYUV(gradients,WR,WB,WG):

    # 图片分别在三个通道的梯度均值
    grad_avg = torch.mean(gradients,dim=0)
    gB = grad_avg[0,:,:]
    gG = grad_avg[1,:,:]
    gR = grad_avg[2,:,:]
    # 计算gYUV
    gY = gR+gG+gB
    gV = 2*(1-WB)*(gB-(WB/WG)*gG)
    gU = 2*(1-WR)*(gR-(WR/WG)*gG)

    return gY, gV, gU




if __name__ == "__main__":
    
    datasetdir = 'dataset'
    # ImageFolder 一个通用的数据加载器
    train_dataset = datasets.ImageFolder(
        os.path.join(datasetdir, 'train'),
        # 对数据进行预处理
        transforms.Compose([                        # 将几个transforms 组合在一起
            transforms.RandomSizedCrop(16),        # 随机切再resize成给定的size大小
            transforms.ToTensor(),       # 把一个取值范围是[0,255]或者shape为(H,W,C)的numpy.ndarray，
                                            # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],       
            #                             std=[0.229, 0.224, 0.225])
        ])
    )

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=500)
    resnet = models.resnet18(pretrained=True)
    resnet = torch.nn.DataParallel(resnet).cuda()
    lossfunc = torch.nn.CrossEntropyLoss()

    for i, (imgs,targets ) in enumerate(train_loader):
        input_var = torch.autograd.Variable(imgs,requires_grad=True)
        img_b = imgs[0,2,:,:]
        img_g = imgs[0,1,:,:]
        img_r = imgs[0,0,:,:]
        target_var = torch.autograd.Variable(targets).cuda()

    output = resnet(input_var)
    loss = lossfunc(output, target_var).cuda()  
    loss.backward()
    gradients = input_var.grad
    print(gradients)
    print(gradients[1,:,:,0])

    # # get_W函数计算参数
    # WR, WG, WB = get_W(gradients) 
    # gY, gV, gU = compute_gYUV(gradients,WR, WG, WB)

