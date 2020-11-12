import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

Input = keras.layers.Input
Model = keras.models.Model
Conv2D = keras.layers.Conv2D
BatchNormalization = keras.layers.BatchNormalization
Activation = keras.layers.Activation
MaxPooling2D = keras.layers.MaxPooling2D
Dropout = keras.layers.Dropout
Flatten = keras.layers.Flatten
Dense = keras.layers.Dense
Softmax = keras.layers.Softmax


def LiveIRNet_K(input_shape, input_tensor, output_size):
    # Stage 1 #
    img_input = Input(shape=input_shape, tensor=input_tensor)

    ## Block 1 ##
    x = Conv2D(12, (3 ,3), strides=(1 ,1), name='S1_conv1', padding='same')(img_input)
    x = Activation('relu', name='S1_relu_conv1')(x)
    x = MaxPooling2D(pool_size=(2 ,2), strides=(2 ,2), name='S1_pool1')(x)

    ## Block 2 ##
    x = Conv2D(12, (1 ,1), strides=(1 ,1), name='S1_conv2', padding='same')(x)
    x = Activation('relu', name='S1_relu_conv2')(x)
    x = Conv2D(12, (3 ,3), strides=(1 ,1), name='S1_conv3', padding='same')(x)
    x = Activation('relu', name='S1_relu_conv3')(x)
    x = MaxPooling2D(pool_size=(2 ,2), strides=(2 ,2), name='S1_pool2')(x)

    ## Block 3 ##
    x = Conv2D(24, (1 ,1), strides=(1 ,1), name='S1_conv4', padding='same')(x)
    x = Activation('relu', name='S1_relu_conv4')(x)
    x = Conv2D(24, (3 ,3), strides=(1 ,1), name='S1_conv5', padding='same')(x)
    x = Activation('relu', name='S1_relu_conv5')(x)
    x = MaxPooling2D(pool_size=(2 ,2), strides=(2 ,2), name='S1_pool3')(x)

    ## Block 4 ##
    x = Conv2D(32, (1 ,1), strides=(1 ,1), name='S1_conv6', padding='same')(x)
    x = Activation('relu', name='S1_relu_conv6')(x)
    x = Conv2D(32, (3 ,3), strides=(1 ,1), name='S1_conv7', padding='same')(x)
    x = Activation('relu', name='S1_relu_conv7')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='S1_pool4')(x)
    x = Flatten(name='S1_flatten')(x)

    ## Block 6 ##
    x = Dropout(0.7)(x)
    x = Dense(128, activation='relu', name='S1_fc1')(x)
    x = Dropout(0.6)(x)
    x = Dense(output_size, activation=None, name='output')(x)
    x = Softmax()(x)
    model = Model([img_input], x, name='facial_landmark_model')

    return model


def LiveIRNet_K0(input_shape, input_tensor, output_size):
    model = LiveIRNet_K(input_shape, input_tensor, output_size)
    return model