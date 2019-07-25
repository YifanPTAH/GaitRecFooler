from keras.preprocessing.image import load_img,img_to_array,array_to_img
import numpy as np
import os

def save_image(hacked_image, r,temp,input_dir,output_dir):
    if temp:
        img = hacked_image[0]
        img *= 255
        im = array_to_img(img.astype('uint8'))
        if not os.path.isdir(output_dir+"/temp/gei"):
            os.makedirs(output_dir+"/temp/gei")
        im.save(output_dir+"/temp/gei/fake_temp.png")

        input_dir = input_dir+'/source-silh'
        input_list = os.listdir(input_dir)
        for image_name in input_list:
            img = load_img(input_dir + '/' + image_name, target_size=(140, 140))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:, :, 0], axis=2)
            x += (r[0] * 255)
            im = array_to_img(x.astype('uint8'))
            if not os.path.isdir(output_dir+"/temp/silh"):
                os.makedirs(output_dir+"/temp/silh")
            im.save(output_dir+"/temp/silh/silh_fake_" + image_name)
    else:
        img = hacked_image[0]
        img *= 255
        im = array_to_img(img.astype('uint8'))
        image_name='fake-gait-'+'gei'+'.png'
        if not os.path.isdir(output_dir+"/gei"):
            os.makedirs(output_dir+'/gei')
        im.save(output_dir+"/gei/"+image_name)

        input_dir = input_dir+'/source-silh'
        input_list = os.listdir(input_dir)
        for image_name in input_list:
            img = load_img(input_dir + '/' + image_name, target_size=(140, 140))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:, :, 0], axis=2)
            x += ((r[0]*255)/len(input_list))
            im = array_to_img(x.astype('uint8'))
            if not os.path.isdir(output_dir+"/silh"):
                os.makedirs(output_dir+"/silh")
            im.save(output_dir+"/silh/silh_fake_"+image_name)
    return