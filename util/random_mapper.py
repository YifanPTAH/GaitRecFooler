import os
from PIL import Image
import numpy as np
import random
def ramdom_mapper(gei_ori,gei_fake,ori_silh_dir,output_dir):
    gei_ori=gei_ori[0]
    gei_ori*=255
    print(np.max(gei_ori))
    gei_fake=gei_fake[0]
    print(np.max(gei_fake))
    filename_list = os.listdir(ori_silh_dir)
    n = len(filename_list)
    r = gei_fake-gei_ori
    filelist_ori=list()
    filelist=list()
    noise = np.zeros((140,140))
    for filename in filename_list:
        filelist.append(np.array(Image.open(os.path.join(ori_silh_dir,filename))))
        filelist_ori.append(np.array(Image.open(os.path.join(ori_silh_dir,filename))))

    for i in range(0,140):
        for j in range(0,140):
            diff=r[i,j]
            frame_num=abs(diff*n//255)
            frame_num=min(n,frame_num)
            success=False
            index_list = None
            num=0
            while not success:
                num+=1
                success = True
                index_list=random.sample(range(0,n),int(frame_num))
                if frame_num == n:
                    break
                if num > 100:
                    break
                if diff < 0:
                    for index in index_list:
                        if filelist[index][i,j]!= 255:
                            success = False
                            break
                else:
                    for index in index_list:
                        if filelist[index][i,j]!=0:
                            success = False
                            break
            if diff < 0:
                noise[i,j]=255//2-diff
                for index in index_list:
                    filelist[index][i, j] = 0
            else:
                noise[i,j]=255//2+diff
                for index in index_list:
                    filelist[index][i, j] = 255

    silh_dir=output_dir+'/silh'
    if not os.path.exists(silh_dir):
        os.mkdir(silh_dir)
    for i, filename in enumerate(filename_list):
        Image.fromarray(filelist[i]).convert('L').save(os.path.join(silh_dir,filename))

    Image.fromarray(noise).convert('L').save(os.path.join(output_dir+'/gei','noise.png'))
    ode_noise=np.sqrt(np.sum(np.sum(noise**2)))/(140*140)

    average_frame_noise=0
    if not os.path.exists(output_dir+'/noise'):
        os.mkdir(output_dir+'/noise')
    for i, filename in enumerate(filename_list):
        frame_noise_temp= filelist[i]-filelist_ori[i]
        Image.fromarray(frame_noise_temp).convert('L').save(os.path.join(output_dir+'/noise','noise-'+filename))
        average_frame_noise += np.sqrt(np.sum(np.sum(frame_noise_temp**2)))/(140*140)
    average_frame_noise = average_frame_noise/i

    return ode_noise, average_frame_noise






