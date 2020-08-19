import os, cv2, time
import math
import torch
import numpy as np
import seaborn as sns
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import JPEG16 as JPEG

class GRACE:
    def __init__(self,scale):
        self.__scale = scale
    # DCT
    def dct(self,image):
        temp = torch.zeros(image.shape)
        m, n = image.shape
        N = n
        temp[0, :] = torch.sqrt(torch.tensor(1/N))
        for i in range(1, m):
            for j in range(n):
                temp[i, j] = torch.cos(torch.tensor(math.pi*i*(2*j+1))/torch.tensor((2*N)))* torch.sqrt(torch.tensor(2/N))
        return torch.mm(torch.mm(temp,image), temp.T) , temp 
            
    def idct(self,img_dct):
        temp = torch.zeros(img_dct.shape)
        m, n = img_dct.shape
        N = n
        temp[0, :] = torch.sqrt(torch.tensor(1/N))
        for i in range(1, m):
            for j in range(n):
                temp[i, j] = torch.cos(torch.tensor(math.pi*i*(2*j+1))/torch.tensor((2*N)))* torch.sqrt(torch.tensor(2/N))
        return torch.mm(torch.mm(temp.T,img_dct), temp)

    def get_ds(self):
        scale = self.__scale
        img_dct = torch.randn(scale,scale)
        img_s_var = Variable(img_dct,requires_grad=True)
        img_idct = self.idct(img_s_var) # 使用idct获得img的原始图像 
        img_idct.sum().backward()
        ds = img_s_var.grad
        return ds

    def get_dx(self,datasetdir,model):
        scale = self.__scale
        # ImageFolder 一个通用的数据加载器
        train_dataset = datasets.ImageFolder(
            os.path.join(datasetdir, 'train'),
            # 对数据进行预处理
            transforms.Compose([      
                transforms.Resize((scale+10,scale+10)),        
                transforms.CenterCrop(scale),                     
                transforms.ToTensor(),       # 把一个取值范围是[0,255]或者shape为(H,W,C)的numpy.ndarray，
                                        # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=5)
        if model=='resnet':
            nnmodel = models.resnet18(pretrained=False)
            nnmodel.load_state_dict(torch.load('JPEG/dataset/resnet18-epoch100-acc70.316.pth'))
        elif model=='alexnet':
            nnmodel = models.alexnet(pretrained=False)
            nnmodel.load_state_dict(torch.load('JPEG/dataset/alexnet-owt-4df8aa71.pth'))
        elif model=='vgg':
            nnmodel = models.vgg11(pretrained=False)
            nnmodel.load_state_dict(torch.load('JPEG/dataset/resnet18-epoch100-acc70.316.pth'))
        elif model== 'squeezenet':
            nnmodel = models.squeezenet1_0(pretrained=False)
            nnmodel.load_state_dict(torch.load('JPEG/dataset/vgg11-bbd30ac9.pth'))
        nnmodel = torch.nn.DataParallel(nnmodel).cuda()
        lossfunc = torch.nn.CrossEntropyLoss()
        for i, (imgs,targets ) in enumerate(train_loader):
            input_var = torch.autograd.Variable(imgs,requires_grad=True)
            target_var = torch.autograd.Variable(targets).cuda()
        output = nnmodel(input_var)
        loss = lossfunc(output, target_var).cuda()  
        loss.backward()
        dx = input_var.grad
        return dx,imgs

    def compute_gradients(self, dx,ds):
        scale = self.__scale
        gradients = torch.Tensor(5,scale,scale,3)
        for k in range(5):
            for i in range(3):
                gradients[k,:,:,i] = torch.mul(dx[k,i,:,:] , ds[:,:])
        return gradients

    def get_W(self, gradients):
        # 图片分别在三个通道的梯度均值
        grad_abs = torch.abs(gradients)
        grad_sum = torch.sum(grad_abs,dim=0)
        gR = grad_sum[:,:,0].mean()
        gG = grad_sum[:,:,1].mean()
        gB = grad_sum[:,:,2].mean()
        # 计算G-YUV参数
        Z1 = torch.div(gB,gG)
        Z2 = torch.div(gR,gG)
        WR = Z2/(1+Z1+Z2)
        WG = 1/(1+Z1+Z2)
        WB = Z1/(1+Z1+Z2)
        return WR, WG, WB

    def compute_gYUV(self, gradients,WR,WB,WG):
        # 每张图片都转为YUV
        grad_yuv = gradients
        for i in range(5):
            gR = gradients[i,:,:,0]
            gG = gradients[i,:,:,1]
            gB = gradients[i,:,:,2]
            grad_yuv[i,:,:,0] = gR+gG+gB
            grad_yuv[i,:,:,1] = 2*(1-WB)*(gB-(WB/WG)*gG)
            grad_yuv[i,:,:,2] = 2*(1-WR)*(gR-(WR/WG)*gG)
        grad_abs = torch.abs(grad_yuv)
        grad_sum = torch.sum(grad_abs,dim=0)
        # 计算gYUV
        gY = grad_sum[:,:,0]
        gV = grad_sum[:,:,1]
        gU = grad_sum[:,:,2]
        return gY, gV, gU

    def RGB2YUV(self, img_rgb,WR, WG, WB):
        img_yuv = img_rgb
        img_r = img_rgb[0,:,:]
        img_g = img_rgb[1,:,:]
        img_b = img_rgb[2,:,:]
        y = WR*img_r + WG*img_g + WB*img_b
        u = 0.5 *(img_b-y) / (1-WB)
        v = 0.5 *(img_r-y) / (1-WR)  
        img_yuv[0,:,:] = y      # Y
        img_yuv[1,:,:] = u     # U
        img_yuv[2,:,:] = v     # V
        return img_yuv

    def get_Syuv(self, img_rgb,WR, WG, WB):
        scale = self.__scale
        img_s = torch.Tensor(5,3,scale,scale)
        for i in range(5):
            img = img_rgb[i,:,:,:]
            img_yuv = self.RGB2YUV(img,WR, WG, WB)
            for j in range(3):
                image = img_yuv[j, :, :]    # 获取yuv通道 2代表v通道,1代表u通道,0代表y通道 
                img_dct,temp = self.dct(image)          # 使用dct获得img的频域图像  
                img_s[i, j, :, :] =img_dct
        img_s_abs = torch.abs(img_s)
        img_s_sum = torch.sum(img_s_abs,dim=0)
        SY = img_s_sum[0,:,:]
        SU = img_s_sum[1,:,:]
        SV = img_s_sum[2,:,:]
        return SY, SU, SV

    def compute_T(self, grad, S, B):
        scale = self.__scale
        theta = torch.mul(grad,S)
        theta = theta.view(1,(scale*scale))
        grad = grad.view(1,(scale*scale))
        theta_new, indices = torch.sort(theta, descending=True)
        grad_new = grad.gather(dim = 1,index = indices)
        for k in range((scale*scale-1),-1,-1):
            index = torch.arange(k,(scale*scale)).view(1,(scale*scale)-k)
            dk = (B-torch.sum(theta_new.gather(dim=1,index=index)))/k
            if theta_new[0,k]<dk and theta_new[0,k-1]>=dk:
                K = k
                break
        dn = theta_new
        for i in range(K):
            dn[0,i] = dk
        qn = (2*dn)/grad_new
        qn = qn.gather(dim = 1,index = indices)
        T = qn.view(scale,scale)
        return T


# main
if __name__ == "__main__":

    ds = get_ds()
    dx,img_rgb = get_dx(datasetdir='JPEG/dataset',model='resnet')
    gradients = compute_gradients(dx,ds) 

    gR = gradients[0,:,:,0]
    gR = np.abs(gR.detach().numpy())
    # cmap = plt.cm.get_cmap('Spectral', 1000)
    # plt.imshow(gR,cmap=cmap)
    # sns.heatmap(gR)
    # plt.show()

    WR, WG, WB = get_W(gradients)

    gY, gV, gU = compute_gYUV(gradients,WR, WG, WB)
    SY, SU, SV = get_Syuv(img_rgb,WR, WG, WB)
    TY = compute_T(gY,SY,B=0.00015)
    TU = compute_T(gU,SU,B=0.00015)
    TV = compute_T(gV,SV,B=0.00015)
    TY = TY.detach().numpy().reshape(-1)
    TU = TU.detach().numpy().reshape(-1)
    TV = TV.detach().numpy().reshape(-1)
    kjpeg = JPEG.KJPEG(WR.item(), WG.item(), WB.item(), TY, TU, TV)
    y_code, u_code, v_code = kjpeg.Compress("test1.jpeg")
    print('编码成功！')
    img = kjpeg.Decompress(y_code, u_code, v_code)
    img.save("result_new2.jpg", "jpeg")
    print('图片保存成功！')


    print('done!')