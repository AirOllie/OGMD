import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from models.se_resnet import *
from PIL import Image


test_dir = 'my_map.png'
model_dir = '/home/nanostring/OGMD/checkpoint/slam_map/se_resnet20/best_acc_ckpt.pth'

def test():
    model = se_resnet20()

    print("=> loading checkpoint '{}'".format(model_dir))
    checkpoint = torch.load(model_dir)
    ckpt = checkpoint['model']
    model.load_state_dict(ckpt)

    image = Image.open(test_dir)
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


    # compute output
    output = model(image)[0]
    print("probability of abnormal: {}".format(output[0]/(output[0]+output[1])*100))
    if output[0]>output[1]:
        print('Abnormal')
    else:
        print('Normal')


if __name__ == '__main__':
    test()