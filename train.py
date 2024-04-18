#!/usr/bin/env python
import os
import time
import argparse
import random
import warnings
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from utils import *
from models.se_resnet import *
from models.resnet_cbam import *

parse = argparse.ArgumentParser(description="PyTorch Training")
parse.add_argument('--lr', default=0.001, type=float, help='learning rate')
parse.add_argument('--epoch', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parse.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parse.add_argument('--netname', default='se_resnet20', type=str, help='train network name')
parse.add_argument('-j', '--workers', default=1, type=int, metavar='N',help='number of data loading workers (default: 4)')
parse.add_argument('--baseline', default=False, action='store_true', help='choose origin_net or senet')
parse.add_argument('--resume', '-r', default=False, action='store', help='resume from checkpoint')
parse.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parse.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parse.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parse.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parse.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# 硬件设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    args = parse.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args)


def main_worker(args):
    global best_acc

    model = se_resnet20(num_classes=1).to(device)
    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 16
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    total_epoch = args.start_epoch + args.epoch
    global_step = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    data_dir = 'data_map'
    train_folder = os.path.join(data_dir, "train")
    val_folder = os.path.join(data_dir, "val")


    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_set = datasets.ImageFolder(train_folder, transform_train)
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_set = datasets.ImageFolder(val_folder, transform_val)
    valloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    if args.evaluate:
        validate(valloader, model, criterion, args)
        return

    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    for epoch in range(args.start_epoch, total_epoch):
        for idx in range(steps):
            scheduler.step()
            print(scheduler.get_lr())
        print('Reset scheduler')

        # train for one epoch
        train(trainloader, model, epoch, total_epoch, criterion, optimizer,args)

        # evaluate on validation set
        acc = validate(valloader, model, criterion, args)

        best_acc = 0
        if acc > best_acc:
            best_acc = acc
            print("Saveing model...")
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'global_step': global_step,
                'optimizer': optimizer.state_dict(),
            }
            
            if not os.path.isdir('checkpoint/{}/{}'.format(args.netname, current_time)):
                os.makedirs('checkpoint/{}/{}'.format(args.netname, current_time))
            torch.save(state, ('checkpoint/{}/{}/Epoch{}_acc{:.2f}_ckpt.pth'.format(args.netname, current_time, epoch, acc)))
            torch.save(state, ('checkpoint/{}/{}/best_acc_ckpt.pth'.format(args.netname, current_time, epoch, acc)))

# train
def train(trainloader, model, epoch, total_epoch, criterion, optimizer,args):
    global global_step
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.2f')
    progress = ProgressMeter(
        len(trainloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    global_step = 0

    for batch_idx, (inputs, labels) in enumerate(trainloader):

        global_step += 1
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.float()

        # compute output
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        predictions = outputs > 0.5
        acc = (predictions == labels).sum().item() / labels.size(0)
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)

        if batch_idx % 50 == 0:
            print('Epoch: [{0}/{1}] [{2}/{3}]'
                'Loss(avg): {loss.val:.4f}({loss.avg:.4f}), '
                'Acc(avg): {acc:.3f}'.format(
                epoch,
                total_epoch,
                batch_idx,
                len(trainloader),
                loss=losses,
                acc=acc))
            print(' * Acc {acc:.3f}'.format(acc=acc))


# validate
def validate(valloader, model, criterion,args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    progress = ProgressMeter(
        len(valloader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, (inputs, labels) in enumerate(valloader):

            labels = labels.to(device)
            # compute output
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            predictions = outputs > 0.5
            acc = (predictions == labels).sum().item() / labels.size(0)
            losses.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % args.print_freq == 0:
                progress.display(batch_idx)

            if batch_idx % 50 == 0:
                print(' Epoch: [{0}/{1}]'
                      ' Loss(avg): {loss.val:.4f}({loss.avg:.4f}), '
                      ' Acc: {acc:.3f} '.format(
                    batch_idx, len(valloader), loss=losses, acc=acc))

        return acc



if __name__ == "__main__":
    main()
    #predict()

