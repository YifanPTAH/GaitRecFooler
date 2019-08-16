from torchfcn.models.fcn8s import FCN8s
from torchfcn.models.fcn16s import FCN16s
from torchfcn.models.fcn32s import  FCN32s
import torch.nn.functional as F
from distutils.version import LooseVersion
import PIL
import numpy as np
import torch
import os
from util.gif_creator import create_gif

# load segmentation network

experiment='experiment-3'

ori='./input-part2/'+experiment+'/origin'
fake='./input-part2/'+experiment+'/silh'
dst='./output-part2/'+experiment
dst_fake=dst+'/fake'
dst_noise=dst+'/noise'
if not os.path.exists(dst_fake):
    os.makedirs(dst_fake)

if not os.path.exists(dst_noise):
    os.makedirs(dst_noise)

seg_net = FCN8s()
'''seg_net = FCN16s()'''
'''seg_net = FCN32s())'''
seg_net.download()
seg_net.load_state_dict(torch.load(seg_net.pretrained_model))
epsilon=5

MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

# attack algorithm
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    #perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def target_transform(target_image):
    target=np.zeros((140,140,1))
    for i in range(0,140):
        for j in range(0,140):
            if target_image[i,j] == 255:
                target[i,j,0]=15
            else:
                target[i,j,0] = 0
    return torch.from_numpy(target).long()

def transform(img):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img -= MEAN_BGR
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

def untransform(img):
    img = img.detach().numpy()
    img = img.transpose(1, 2, 0)
    img += MEAN_BGR
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]
    return img

for filename in os.listdir(ori):
    img = PIL.Image.open(ori + '/' + filename)
    img = np.array(img)
    img_ori = np.array(PIL.Image.open(ori+'/'+filename))
    img = transform(img)
    img = img.unsqueeze(0)
    img.requires_grad = True
    result = seg_net.forward(img)

    img2=PIL.Image.open(fake+ '/' + filename)
    img2=np.array(img2)
    target=target_transform(img2)
    target=target.unsqueeze(0)

    while():
        loss=cross_entropy2d(result,target,size_average=False)
        seg_net.zero_grad()
        loss.backward()
        data_grad=img.grad.data
        img = fgsm_attack(img, epsilon, data_grad)
        result = seg_net.forward(img)


    fake_image = untransform(img[0])

    PIL.Image.fromarray(fake_image).save(os.path.join(dst_fake,filename))
    PIL.Image.fromarray(fake_image-img_ori).save(os.path.join(dst_noise,'noise-'+filename))

create_gif('./input-part2/'+experiment+'/origin','./input-part2/'+experiment+'/gif','origin.gif')
create_gif('./input-part2/'+experiment+'/target-origin','./input-part2/'+experiment+'/gif','target.gif')
create_gif('./output-part2/'+experiment+'/fake','./output-part2/'+experiment+'./gif','fake.gif')








