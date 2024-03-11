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

device = torch.device('cpu')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('mode', metavar='MODE', default='deploy', choices=['train', 'deploy'], help='train or deploy')
parser.add_argument('weights', metavar='WEIGHTS', default='/home/nanostring/OGMD/checkpoint/slam_map/se_resnet20/best_acc_ckpt.pth', help='path to the weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='se_resnet20')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: 100) for test')

def test():
    args = parser.parse_args()
    model = se_resnet20().to(device)
    # model = torch.nn.DataParallel(model).cuda()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        use_gpu = False
    else:
        model = model.cuda()
        use_gpu = True
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if os.path.isfile(args.weights):
        print("=> loading checkpoint '{}'".format(args.weights))
        checkpoint = torch.load(args.weights)
        ckpt = checkpoint['model']
        model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.weights))


    cudnn.benchmark = True

    # Data loading code
    test = args.data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    validate(testloader , model, criterion, use_gpu)

def validate(testloader, model, criterion, use_gpu):
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
            if use_gpu:
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

            if i % 10 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg

if __name__ == '__main__':
    test()