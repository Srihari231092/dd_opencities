from unet_model import *
from gen_patches import *

import pickle

import time

import os.path
import numpy as np
import tifffile as tiff
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from sklearn.decomposition import PCA

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x


N_BANDS = 3
N_CLASSES = 5  # buildings, roads, trees, crops and water
# CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3]
# CLASS_WEIGHTS = [0.3, 0.1, 0.2, 0.3, 0.1]
CLASS_WEIGHTS = [0.1, 0.6, 0.1, 0.1, 0.1]
N_EPOCHS = 100
UPCONV = True
PATCH_SZ = 160   # should divide by 16
BATCH_SIZE = 200

TRAIN_SZ = 6000  # train size
VAL_SZ = 1200    # validation size

weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights2.hdf5'


#weights_path = 'weights_pca'
#if not os.path.exists(weights_path):
#    os.makedirs(weights_path)
#weights_path += '/unet_weights.hdf5'


def get_model(n_channels=N_BANDS):
    return unet_model(N_CLASSES, PATCH_SZ,
                      n_channels=n_channels,
                      upconv=UPCONV,
                      n_filters_start=32,
                      class_weights=CLASS_WEIGHTS)


trainIds = [str(i).zfill(2) for i in range(1, 24)]  # all availiable ids: from "01" to "24"


def old_main():

    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    train_path = "../data/mband/"
    mask_path = "../data/gt_mband/"

    # trainIds = os.listdir(train_path)
    # maskIds = [t.split(".")[0]+"_gt.tiff" for t in trainIds]

    print('Reading images')
    for img_id in trainIds:
        img_m = normalize(tiff.imread(train_path + '{}.tif'.format(img_id)).transpose([1, 2, 0]))
        # Subset the image
        img_m = img_m[:,:,[3, 4, 6]]
        mask = tiff.imread(mask_path + '{}.tif'.format(img_id)).transpose([1, 2, 0]) / 255
        train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        print(img_id + ' read')
    print('Images were read')

    def train_net():
        print("start train net")
        # Random sample patches
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
        # Create the model schema
        model = get_model()
        # Load pre-existing weights, if any
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        # Make model check point configuration
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        # Logger configuration
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        # Tensorboard logging configuration
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)

        # Fit the model on the sampled data
        # model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
        #           verbose=2, shuffle=True,
        #           callbacks=[model_checkpoint, csv_logger, tensorboard],
        #           validation_data=(x_val, y_val))

        # OR
        # Fit the model using ImageDataAugmentor from Keras
        # Keras documentation - https://keras.io/preprocessing/image/
        datagen = ImageDataGenerator(
           featurewise_center=True,
           featurewise_std_normalization=True,
           rotation_range=90,
           width_shift_range=0.2,
           height_shift_range=0.2,
           horizontal_flip=True,
           vertical_flip=True)
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(x_train)

        # fits the model on batches with real-time data augmentation:
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                           verbose=2,
                           shuffle=True,
                           callbacks=[model_checkpoint, csv_logger, tensorboard],
                           validation_data=(x_val, y_val),
                           epochs=N_EPOCHS)

        return model

    train_net()


def main():

    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    train_path = "../data/mband/"
    mask_path = "../data/gt_mband/"
    pca_path = "./pca.pkl"

    # open a file, where you stored the pickled data
    file = open(pca_path, 'rb')
    pca = pickle.load(file)
    file.close()

    # trainIds = os.listdir(train_path)
    # maskIds = [t.split(".")[0]+"_gt.tiff" for t in trainIds]

    print('Reading images')
    for img_id in trainIds:
        img_m = normalize(tiff.imread(train_path + '{}.tif'.format(img_id)).transpose([1, 2, 0]))

        # Apply PCA to the image
        reshaped_img_m = img_m.reshape((img_m.shape[0]*img_m.shape[1], img_m.shape[2]))
        reshaped_img_pca = pca.transform(reshaped_img_m)
        img_pca = reshaped_img_pca.reshape((img_m.shape[0], img_m.shape[1],
                                            reshaped_img_pca.shape[1]))
        img_m = img_pca

        mask = tiff.imread(mask_path + '{}.tif'.format(img_id)).transpose([1, 2, 0]) / 255
        train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        print(img_id + ' read')
    print('Images were read')

    def train_net(n_channels=N_BANDS):
        print("start train net")
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
        model = get_model(n_channels=n_channels)
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(x_val, y_val))
        return model

    train_net(n_channels=img_m.shape[2])


if __name__ == '__main__':

    start = time.time()
    old_main()
    stop = time.time()

    #    start = time.time()
    #    main()
    #    stop = time.time()

    print("Elapsed time :", np.round(abs(stop-start)/1e9, 2), "s")

