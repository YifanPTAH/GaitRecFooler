from keras import backend as K
from util.loader import loader
from util.saver import save_image
from util.gif_creator import create_gif
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


experiment = 'experiment-1'

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

max_change_above = input_image + 0.02
max_change_below = input_image - 0.02

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
save_image(hacked_image,r, False, './input/'+experiment,'output/'+experiment)
create_gif("./output/"+experiment+'/silh','./output/'+experiment+'/gif','fake.gif')
print("result saved!")


