import imageio
import os

def create_gif(source_path,target_path,savename):
    sihl_path=source_path
    images = []
    file_list = os.listdir(sihl_path)
    for filename in file_list:
        images.append(imageio.imread(sihl_path+"/"+filename))
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    imageio.mimsave(target_path+"/"+savename, images)
    return