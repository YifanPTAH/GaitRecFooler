from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout
from layers.Conv2D121 import Conv2D121

def create_model():
    num_classes = 124
    model = Sequential()
    model.add(Conv2D(8, (5, 5), padding='valid', input_shape=(140, 140, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
    model.add(Conv2D121(8, (5, 5), padding='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
    model.add(Conv2D121(8, (5, 5), padding='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
    model.add(Conv2D121(8, (5, 5), padding='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
    model.add(Flatten())
    model.add(Dense(num_classes, input_shape=(200,)))
    model.add(Activation('softmax'))

    model.load_weights('./GaitRecognizer/gait_recognizer_model_weight.h5')
    return model