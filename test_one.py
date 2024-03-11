import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from models.se_resnet import *
from PIL import Image

device = torch.device('cuda:0')


# test_dir = 'processed_map.png'
test_dir = 'my_map.png'
# test_dir = '0_0.png'
model_dir = '/home/nanostring/OGMD/checkpoint/slam_map/se_resnet20/best_acc_ckpt.pth'


def test():
    model = se_resnet20().to(device)
    model = torch.nn.DataParallel(model).cuda()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        use_gpu = False
    else:
        model = model.cuda()
        use_gpu = True


    print("=> loading checkpoint '{}'".format(model_dir))
    checkpoint = torch.load(model_dir)
    ckpt = checkpoint['model']
    model.load_state_dict(ckpt)


    cudnn.benchmark = True
    image = Image.open(test_dir)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    # Data loading code
    image = transform(image).unsqueeze(0)
    model.eval()

    with torch.no_grad():
        if use_gpu:
            image = image.cuda(non_blocking=True)

        # compute output
        output = model(image)[0]
        print("probability of abnormal: {}".format(output[0]/(output[0]+output[1])*100))
        if output[0]>output[1]:
            print('Abnormal')
        else:
            print('Normal')


if __name__ == '__main__':
    test()