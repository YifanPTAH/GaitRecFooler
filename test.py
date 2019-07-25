from util.loader import test_loader
from GaitRecognizer.model_creator import create_model
import tensorflow as tf
from keras import backend as K
import numpy as np

config = tf.ConfigProto(intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=4,
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )

session = tf.Session(config=config)
K.set_session(session)


experiment='experiment-1'
model=create_model()
model.load_weights("./GaitRecognizer/gait_recognizer_model_weight.h5")

origin_image, fake_image =  test_loader('./input/'+experiment+'/source','./output/'+experiment+'/gei')
origin_image=origin_image.astype('float32')
fake_image=fake_image.astype('float32')
origin_image/=255
fake_image/=255
print('origin:'+str(np.argmax(model.predict(origin_image)))+'  fake:'+str(np.argmax(model.predict(fake_image))))

