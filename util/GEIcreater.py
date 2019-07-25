import os
from PIL import Image
import numpy as np

def cut_image(path,cut_path,size):
    for (root,dirs,files) in os.walk(path):
        temp = root.replace(path,cut_path)
        if not os .path.exists(temp):
            os.makedirs(temp)
        for file in files:
            image,flag = cut(Image.open(os.path.join(root,file)))
            if not flag:
                Image.fromarray(image).convert('L').resize((size,size)).save(os.path.join(temp,file))
    pass

def cut(image):
    image = np.array(image)

    height_min = (image.sum(axis=1)!=0).argmax()
    height_max = ((image.sum(axis=1)!=0).cumsum()).argmax()
    width_min = (image.sum(axis=0)!=0).argmax()
    width_max = ((image.sum(axis=0)!=0).cumsum()).argmax()
    head_top = image[height_min,:].argmax()

    size = height_max-height_min
    temp = np.zeros((size,size))

    l1 = head_top-width_min
    r1 = width_max-head_top

    flag = False
    if size<= width_max-width_min or size//2<r1 or size//2<l1:
        flag = True
        return temp,flag

    temp[:,(size//2-l1):(size//2+r1)] = image[height_min:height_max,width_min:width_max]

    return temp, flag

def GEI(cut_path, data_path,size,filename):
    for (root,dirs,files) in os.walk(cut_path):
        temp = root.replace(cut_path,data_path)
        if not os.path.exists(temp):
            os.makedirs(temp)
        GEI = np.zeros([size,size])
        if len(files)!=0:
            for file in files:
                GEI += Image.open(os.path.join(root,file)).convert('L')
            GEI /= len(files)
    Image.fromarray(GEI).convert('L').resize((size,size)).save(os.path.join(temp,filename))


if __name__ == '__main__':
    motion_name = [
        'bg-01', 'bg-02', 'cl-01', 'cl-02', 'nm-01',
        'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06'
    ]

    src_dir = './GaitDatasetB-silh/'

    for i in range(1, 125):
        stri = ""
        if (i < 10):
            stri = "00" + str(i)
        elif (i < 100):
            stri = "0" + str(i)
        else:
            stri = str(i)
        for motion in motion_name:
            src = src_dir + stri + '/' + motion + '/' + '090'
            dst_dir = './cut90/' + stri+"/"+ motion
            target_dir='./gei90/'+stri+'/'
            filename = stri+'-'+motion+"-090.png"
            cut_image(src,dst_dir,140)
            GEI(dst_dir,target_dir,140,filename)