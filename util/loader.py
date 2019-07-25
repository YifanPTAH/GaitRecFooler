from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

def loader(input_dir):
    source_image=[]
    target_image=[]
    source_dir = input_dir+'/source'
    target_dir = input_dir+ '/target'

    input_list=os.listdir(source_dir)
    for name in input_list:
        img=load_img(source_dir+'/'+name,target_size=(140,140))
        x3d=img_to_array(img)
        x=np.expand_dims(x3d[:,:,0],axis=2)
        source_image.append(x)

    target_list=os.listdir(target_dir)
    for name in target_list:
        img=load_img(target_dir+'/'+name,target_size=(140,140))
        x3d=img_to_array(img)
        x=np.expand_dims(x3d[:,:,0],axis=2)
        target_image.append(x)

    return np.array(source_image), np.array(target_image)

def test_loader(input_dir, output_dir):
    origin_image = []
    fake_image= []

    origin_list=os.listdir(input_dir)
    for name in origin_list:
        img = load_img(input_dir + '/' + name, target_size=(140, 140))
        x3d = img_to_array(img)
        x = np.expand_dims(x3d[:, :, 0], axis=2)
        origin_image.append(x)

    fake_list=os.listdir(output_dir)
    for name in fake_list:
        img = load_img(output_dir + '/' + name, target_size=(140, 140))
        x3d = img_to_array(img)
        x = np.expand_dims(x3d[:, :, 0], axis=2)
        fake_image.append(x)
    return np.array(origin_image), np.array(fake_image)