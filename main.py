import os, time,cv2
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import GRACE,JPEG16,JPEG_new

def dir_name(file_dir):   
    D = []
    L = []
    for root, dirs, files in os.walk(file_dir):  
        for dir in dirs:
            D.append(dir)
        for file in files:
            L.append(file)
    return D,L


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(datasetdir,model):
    # ImageFolder 一个通用的数据加载器
    test_dataset = datasets.ImageFolder(
        datasetdir,
        transforms.Compose([                        
            transforms.Resize((256,256)),        
            transforms.CenterCrop(224),   
            transforms.ToTensor(),      # 转换成[C,H,W],[0,1.0]的torch.FloadTensor     
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    )
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=5)
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
    criterion = torch.nn.CrossEntropyLoss().cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    nnmodel.eval()
    end = time.time()
    for i, (input, target)  in enumerate(test_loader):
        target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        output = nnmodel(input_var)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1000 == 999:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i+1, len(test_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))



if __name__ == "__main__":
    
    root = r'H:\\imagenet\\val'
    des_path = r'H:\\imagenet\\val_compress2'
    # test(des_path,model='alexnet')
    # 原始8*8的压缩后验证
    '''
    kjpeg = JPEG_new.KJPEG()
    des_path = r'H:\\imagenet\\val_compress2'
    D, L = dir_name(root)
    for i in range(450,1000):
        for j in range(50):
            from_path = os.path.join(root,D[i])
            to_path = os.path.join(des_path,D[i])
            if not os.path.isdir(to_path):
                 os.makedirs(to_path)
            y_code, u_code, v_code = kjpeg.Compress(os.path.join(from_path,L[50*i+j]))
            image_compress = kjpeg.Decompress(y_code, u_code, v_code)
            print('编码完成！')
            image_compress.save(os.path.join(to_path,L[50*i+j]), "jpeg")
    # 验证模型准确率
    test(des_path,model='resnet')
    '''

    # 采用GRACE算法生成策略
    GRACE = GRACE.GRACE(scale=16)
    ds = GRACE.get_ds()
    dx,img_rgb = GRACE.get_dx(datasetdir='JPEG/dataset',model='resnet')
    gradients = GRACE.compute_gradients(dx,ds) 
    '''
    # 热度图
    gR = gradients[0,:,:,0]
    gR = np.abs(gR.detach().numpy())
    cmap = plt.cm.get_cmap('Spectral', 1000)
    plt.imshow(gR,cmap=cmap)
    sns.heatmap(gR)
    plt.show()
    '''
    WR, WG, WB = GRACE.get_W(gradients)
    gY, gV, gU = GRACE.compute_gYUV(gradients,WR, WG, WB)
    SY, SU, SV = GRACE.get_Syuv(img_rgb,WR, WG, WB)
    TY = GRACE.compute_T(gY,SY,B=0.00015)
    TU = GRACE.compute_T(gU,SU,B=0.00015)
    TV = GRACE.compute_T(gV,SV,B=0.00015)
    TY = TY.detach().numpy().reshape(-1)
    TU = TU.detach().numpy().reshape(-1)
    TV = TV.detach().numpy().reshape(-1)

    # 用新生成的策略生成对象并压缩
    kjpeg = JPEG16.KJPEG(224, WR.item(), WG.item(), WB.item(), TY, TU, TV)
    # y_code, u_code, v_code = kjpeg.Compress("../test1.jpeg")
    # img = kjpeg.Decompress(y_code, u_code, v_code)
    # img.save("result_new2.jpg", "jpeg")
    des_path = r'H:\\imagenet\\val_compress5'
    D, L = dir_name(root)
    for i in range(1000):
        for j in range(50):
            from_path = os.path.join(root,D[i])
            to_path = os.path.join(des_path,D[i])
            if not os.path.isdir(to_path):
                 os.makedirs(to_path)
            y_code, u_code, v_code = kjpeg.Compress(os.path.join(from_path,L[50*i+j]))
            image_compress = kjpeg.Decompress(y_code, u_code, v_code)
            image_compress.save(os.path.join(to_path,L[50*i+j]), "jpeg")
    # 测试新策略的准确率
    test(des_path,model='alexnet')

    print('done!')

    

    

