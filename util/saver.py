from keras.preprocessing.image import load_img,img_to_array,array_to_img
import os

def save_image(hacked_image, r,temp,input_dir,output_dir):
    if temp:
        img = hacked_image[0]
        img *= 255
        im = array_to_img(img.astype('uint8'))
        if not os.path.isdir(output_dir+"/temp/gei"):
            os.makedirs(output_dir+"/temp/gei")
        im.save(output_dir+"/temp/gei/fake_temp.png")

    else:
        img = hacked_image[0]
        img *= 255
        im = array_to_img(img.astype('uint8'))
        image_name='fake-gait-'+'gei'+'.png'
        if not os.path.isdir(output_dir+"/gei"):
            os.makedirs(output_dir+'/gei')
        im.save(output_dir+"/gei/"+image_name)

    return