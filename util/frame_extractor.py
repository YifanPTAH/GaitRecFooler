import cv2
import os

def frame_extractor(input_path,output_path):
    cap = cv2.VideoCapture(input_path)
    i = 1
    while (cap.isOpened()):
        if (i < 10):
            stri = "00" + str(i)
        elif (i < 100):
            stri = "0" + str(i)
        else:
            stri = str(i)
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(output_path + stri + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()
    pass




# Opens the Video file
input_dir='./raw/'
output_dir='./test-database/'
for i in range(125,128):
    stri = ""
    if (i < 10):
        stri = "00" + str(i)
    elif (i < 100):
        stri = "0" + str(i)
    else:
        stri = str(i)
    filename_list=os.listdir(input_dir+stri)
    for j,filename in enumerate(filename_list):
        strj=''
        if (j<10):
            strj="0"+str(j+1)
        else:
            strj=str(j+1)
        output_dir_temp=output_dir+stri+'/'+'nm-'+strj+'/090'
        if not os.path.exists(output_dir_temp):
            os.makedirs(output_dir_temp)
        frame_extractor(input_dir+stri+'/'+filename,output_dir_temp+'/'+stri+'-nm-'+strj+'-090-')