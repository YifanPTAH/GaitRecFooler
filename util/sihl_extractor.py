from torchfcn.models.fcn8s import FCN8s
from torchfcn.models.fcn16s import FCN16s
from torchfcn.models.fcn32s import  FCN32s
import PIL
import numpy as np
import torch
import os

model = FCN8s()
model.load_state_dict(torch.load(model.pretrained_model))
'''model = FCN16s()
model.load_state_dict(torch.load(model.pretrained_model))'''
'''model = FCN32s()
model.download()
model.load_state_dict(torch.load(model.pretrained_model))'''

src='./self-database/'
dst='./self-database-silh/'
MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])
for i in range(125,128):
    stri = ""
    if (i < 10):
        stri = "00" + str(i)
    elif (i < 100):
            stri = "0" + str(i)
    else:
        stri = str(i)
    for dir_name in os.listdir(src+stri):
        input_dir_temp=src+stri+'/'+dir_name+'/090'
        dst_dir_temp=dst+stri+'/'+dir_name
        for filename in os.listdir(input_dir_temp):
            img = PIL.Image.open(input_dir_temp+'/'+filename)
            img = np.array(img, dtype=np.uint8)
            img = img[:, :, ::-1]  # RGB -> BGR
            img = img.astype(np.float32)
            img -= MEAN_BGR
            img = img.transpose(2, 0, 1)  # H, W, C -> C, H, W
            img = torch.from_numpy(img).float()
            img = img.unsqueeze(0)
            result=model.forward(img)
            resimg=result[0].detach().numpy()
            finimage=np.zeros((140,140))
            for i in range(0,140):
                for j in range(0,140):
                    if np.exp(resimg[15,i,j])/np.sum(np.exp(resimg[:,i,j]))>0.48:
                        finimage[i,j]= 255
                    else:
                        finimage[i,j]=0

            if not os.path.exists(dst_dir_temp):
                os.makedirs(dst_dir_temp)
            PIL.Image.fromarray(finimage).convert('L').save(dst_dir_temp+'/'+filename)

