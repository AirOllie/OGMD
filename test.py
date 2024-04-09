import argparse
import os
import time
import torch
import torch.nn.parallel
from thop import profile
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import accuracy, ProgressMeter, AverageMeter
from models.se_resnet import *

device = torch.device('cuda:0')

model_dir = '/home/nanostring/OGMD/checkpoint/slam_map/se_resnet20/best_acc_ckpt.pth'
test_dir = '/home/nanostring/OGMD/data_map/test'
batch_size = 100

def test():
    model = se_resnet20().to(device)
    model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().to(device)

    print("=> loading checkpoint")
    checkpoint = torch.load(model_dir)
    ckpt = checkpoint['model']
    model.load_state_dict(ckpt)

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False, pin_memory=True)
    validate(testloader , model, criterion)

def validate(testloader, model, criterion):
    batch_time = AverageMeter('batch_time')
    losses = AverageMeter('losses')
    top1 = AverageMeter('top1')
    top5 = AverageMeter('top5')
    progress = ProgressMeter(
        len(testloader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(testloader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg

if __name__ == '__main__':
    test()