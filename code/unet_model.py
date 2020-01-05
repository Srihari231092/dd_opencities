# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

import keras
from keras.layers import Convolution2D, BatchNormalization, merge, Cropping2D, concatenate


smooth = 1e-12

num_mask_channels = 1


def get_unet0(n_channels, img_rows, img_cols, class_weights=[0.2, 0.3, 0.1, 0.1, 0.3]):

    activation_func = 'tanh'

    inputs = Input((img_rows, img_cols, n_channels))
    conv1 = Convolution2D(32, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(inputs)
    conv1 = BatchNormalization(mode=0, axis=2)(conv1)
    # conv1 = keras.layers.advanced_activations.ELU()(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(conv1)
    conv1 = BatchNormalization(mode=0, axis=2)(conv1)
    # conv1 = keras.layers.advanced_activations.ELU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="mp1")(conv1)

    conv2 = Convolution2D(64, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(pool1)
    conv2 = BatchNormalization(mode=0, axis=2)(conv2)
    # conv2 = keras.layers.advanced_activations.ELU()(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(conv2)
    conv2 = BatchNormalization(mode=0, axis=2)(conv2)
    # conv2 = keras.layers.advanced_activations.ELU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="mp2")(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(pool2)
    conv3 = BatchNormalization(mode=0, axis=2)(conv3)
    # conv3 = keras.layers.advanced_activations.ELU()(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(conv3)
    conv3 = BatchNormalization(mode=0, axis=2)(conv3)
    # conv3 = keras.layers.advanced_activations.ELU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name="mp3")(conv3)

    conv4 = Convolution2D(256, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(pool3)
    conv4 = BatchNormalization(mode=0, axis=2)(conv4)
    # conv4 = keras.layers.advanced_activations.ELU()(conv4)
    conv4 = Convolution2D(256, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(conv4)
    conv4 = BatchNormalization(mode=0, axis=2)(conv4)
    # conv4 = keras.layers.advanced_activations.ELU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(pool4)
    conv5 = BatchNormalization(mode=0, axis=2)(conv5)
    # conv5 = keras.layers.advanced_activations.ELU()(conv5)
    conv5 = Convolution2D(512, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(conv5)
    conv5 = BatchNormalization(mode=0, axis=2)(conv5)
    # conv5 = keras.layers.advanced_activations.ELU()(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Convolution2D(256, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(up6)
    conv6 = BatchNormalization(mode=0, axis=2)(conv6)
    # conv6 = keras.layers.advanced_activations.ELU()(conv6)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(conv6)
    conv6 = BatchNormalization(mode=0, axis=2)(conv6)
    # conv6 = keras.layers.advanced_activations.ELU()(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Convolution2D(128, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(up7)
    conv7 = BatchNormalization(mode=0, axis=2)(conv7)
    # conv7 = keras.layers.advanced_activations.ELU()(conv7)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(conv7)
    conv7 = BatchNormalization(mode=0, axis=2)(conv7)
    # conv7 = keras.layers.advanced_activations.ELU()(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Convolution2D(64, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(up8)
    conv8 = BatchNormalization(mode=0, axis=2)(conv8)
    # conv8 = keras.layers.advanced_activations.ELU()(conv8)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(conv8)
    conv8 = BatchNormalization(mode=0, axis=2)(conv8)
    # conv8 = keras.layers.advanced_activations.ELU()(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Convolution2D(32, 3, 3, border_mode='same',activation=activation_func,  init='glorot_uniform')(up9)
    conv9 = BatchNormalization(mode=0, axis=2)(conv9)
    # conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv9 = Convolution2D(32, 3, 3, border_mode='same', activation=activation_func, init='glorot_uniform')(conv9)
    # conv9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
    conv9 = BatchNormalization(mode=0, axis=2)(conv9)
    # conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv10 = Convolution2D(num_mask_channels, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    def weighted_binary_crossentropy(y_true, y_pred, class_weights):
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return K.sum(class_loglosses * K.constant(class_weights))

    model.compile(optimizer=Adam(), loss=weighted_binary_crossentropy)

    return model


def unet_model(n_classes=5, im_sz=160, n_channels=8, n_filters_start=32, growth_factor=2, upconv=True,
               class_weights=[0.2, 0.3, 0.1, 0.1, 0.3]):
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels))
    
    # activation_func = 'relu'
    activation_func = 'tanh'

    # First half of the U Net

    conv1 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    n_filters *= growth_factor
    conv2 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    n_filters *= growth_factor
    conv3 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    n_filters *= growth_factor
    conv4 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(pool3)
    conv4 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    n_filters *= growth_factor
    conv5 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(pool4)
    conv5 = Dropout(0.25)(conv5)
    conv5 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(conv5)
     
    # Second half of the U Net

    n_filters //= growth_factor
    if upconv:
        up6 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4])
    else:
        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(up6)
    conv6 = Dropout(0.25)(conv6)
    conv6 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(conv6)

    n_filters //= growth_factor
    if upconv:
        up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(conv7)

    n_filters //= growth_factor
    if upconv:
        up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(conv8)

    n_filters //= growth_factor
    if upconv:
        up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation=activation_func, padding='same')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    def weighted_binary_crossentropy(y_true, y_pred):
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return K.sum(class_loglosses * K.constant(class_weights))

    model.compile(optimizer=Adam(), loss=weighted_binary_crossentropy)
    return model


if __name__ == '__main__':
    model = unet_model()
    print(model.summary())
    plot_model(model, to_file='unet_model.png', show_shapes=True)
