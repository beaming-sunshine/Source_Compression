import os, time
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class AverageMeter(object):
    """Computes and stores the average and current value"""
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
    """Computes the precision@k for the specified values of k"""
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

if __name__ == "__main__":
    
    datasetdir = 'H:\\imagenet'
    # ImageFolder 一个通用的数据加载器
    test_dataset = datasets.ImageFolder(
        os.path.join(datasetdir, 'val_compress1'),
        transforms.Compose([                        
            transforms.Resize((256,256)),        
            transforms.CenterCrop(224),
            transforms.ToTensor(),          # 转换成[C,H,W],[0,1.0]的torch.FloadTensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    )
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=5)
    resnet_model = models.resnet18(pretrained=False)
    resnet_model.load_state_dict(torch.load('H:\\imagenet\\resnet18-epoch100-acc70.316.pth'))
    resnet_model = torch.nn.DataParallel(resnet_model).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
 
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    resnet_model.eval()
    end = time.time()

    for i, (input, target)  in enumerate(test_loader):
        target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        output = resnet_model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 9:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    print('done!')