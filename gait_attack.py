from keras import backend as K
from util.loader import loader
from util.saver import save_image
from util.gif_creator import create_gif
from util.random_mapper import ramdom_mapper
from GaitRecognizer.model_creator import create_model
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


experiment = 'experiment-3'

model=create_model()
model.load_weights("./GaitRecognizer/gait_recognizer_model_weight.h5")

input_image, source_image =loader('./input/'+experiment)
create_gif('./input/'+experiment+'/source-silh','./input/'+experiment+'/gif','source.gif')
input_image=input_image.astype('float32')
source_image=source_image.astype('float32')
input_image/=255
source_image/=255


model_input_layer = model.layers[0].input
model_output_layer = model.layers[-1].output

max_change_above = input_image + 0.05
max_change_below = input_image - 0.05

origin=np.argmax(model.predict(input_image))
gait_to_fake = np.argmax(model.predict(source_image))
hacked_image = np.copy(input_image)
learning_rate = 0.001

cost_function=model_output_layer[0,gait_to_fake]
gradient_function = K.gradients(cost_function,model_input_layer)[0]
grab_cost_and_gradients_from_model = K.function([model_input_layer,K.learning_phase()],[cost_function,gradient_function])
cost = 0.0
epoch=0

while np.argmax(model.predict(hacked_image)) != gait_to_fake:
    cost, gradients = grab_cost_and_gradients_from_model([hacked_image,0])
    hacked_image += gradients * learning_rate


    hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
    hacked_image = np.clip(hacked_image,0,1)

    r=hacked_image-input_image


    if epoch%1000==0:
        print(str(epoch)+" "+"origin:"+str(np.argmax(model.predict(input_image)))+"  fake:"+str(np.argmax(model.predict(hacked_image)))+"  target:"+str(gait_to_fake))
        save_image(hacked_image,r, True,'./input/'+experiment,'./output/'+experiment)

    epoch+=1

print("origin:"+str(np.argmax(model.predict(input_image)))+"  fake:"+str(np.argmax(model.predict(hacked_image)))+ "  target:"+str(gait_to_fake))
print("completed")
confidence_input=model.predict(input_image)[0,np.argmax(model.predict(input_image))]/np.sum(model.predict(input_image))
confidence_output = model.predict(hacked_image)[0,np.argmax(model.predict(hacked_image))]/np.sum(model.predict(hacked_image))
confidence_ori = model.predict(hacked_image)[0,np.argmax(model.predict(input_image))]/np.sum(model.predict(hacked_image))

save_image(hacked_image,r, False, './input/'+experiment,'output/'+experiment)
ode_noise,frame_noise=ramdom_mapper(input_image,hacked_image,'./input/'+experiment+'/source-silh','./output/'+experiment)
create_gif("./output/"+experiment+'/silh','./output/'+experiment+'/gif','fake.gif')
file = open('./output/'+experiment+'/log.txt', 'w')
file.write("input: " + str(origin))
file.write("\nconfidence: "+str(confidence_input))
file.write("\noutput: "+str(gait_to_fake))
file.write("\nconfidence: "+str(confidence_output))
file.write(".\nconfidence_origin:"+str(confidence_ori))
file.write("\nGEI-noise-norm2: "+str(ode_noise))
file.write("\nAverage-frame-noise-norm2: "+str(frame_noise))
file.close()
print("result saved!")


