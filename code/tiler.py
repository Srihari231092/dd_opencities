# from unet_model import *
# from gen_patches import *

import pickle

import time

import os
from os.path import join as pjoin
import numpy as np
import tifffile as tiff
import rasterio
from rasterio.windows import Window


# from keras.callbacks import CSVLogger
# from keras.callbacks import TensorBoard
# from keras.callbacks import ModelCheckpoint
#
# from keras.preprocessing.image import ImageDataGenerator

# from sklearn.decomposition import PCA
# from argparse import ArgumentParser

# import random


# def normalize(img):
#     min = img.min()
#     max = img.max()
#     x = 2.0 * (img - min) / (max - min) - 1.0
#     return x
#
#
# # N_BANDS = 8
# # N_CLASSES = 5  # buildings, roads, trees, crops and water
# # CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3]
# # CLASS_WEIGHTS = [0.3, 0.1, 0.2, 0.3, 0.1]
# # CLASS_WEIGHTS = [0.6, 0.1, 0.1, 0.1, 0.1]
# # CLASS_WEIGHTS = [0.3, 0.3, 0.05, 0.05, 0.3]	# suppressed weights
# # N_EPOCHS = 200
# UPCONV = True
# # PATCH_SZ = 160   # should divide by 16
# # BATCH_SIZE = 200
# # TRAIN_SZ = 4000  # train size
# # VAL_SZ = 1000    # validation size
#
# # weights_path = 'weights'
# # if not os.path.exists(weights_path):
# #     os.makedirs(weights_path)
# # weights_path += '/unet_weights.hdf5'
#
#
# # weights_path = 'weights_pca'
# # if not os.path.exists(weights_path):
# #    os.makedirs(weights_path)
# # weights_path += '/unet_weights.hdf5'
#
#
# def get_model(n_classes, patch_size, n_channels, class_weights):
#     return unet_model(n_classes, patch_size,
#                       n_channels=n_channels,
#                       upconv=UPCONV,
#                       n_filters_start=32,
#                       class_weights=class_weights)
#
#
# def old_main(patch_size, train_size, val_size, input_weights_path, output_weights_path, batch_size, num_epochs,
#              bands, class_weights, n_classes):
#
#     X_DICT_TRAIN = dict()
#     Y_DICT_TRAIN = dict()
#     X_DICT_VALIDATION = dict()
#     Y_DICT_VALIDATION = dict()
#
#     train_path = "../data/mband/"
#     mask_path = "../data/gt_mband/"
#
#     # trainIds = os.listdir(train_path)
#     # maskIds = [t.split(".")[0]+"_gt.tiff" for t in trainIds]
#
#     print('Reading images')
#     for img_id in trainIds:
#         img_m = normalize(tiff.imread(train_path + '{}.tif'.format(img_id)).transpose([1, 2, 0]))
#         # Subset the image
#         if bands[0] != -1:
#             img_m = img_m[:, :, bands]
#         mask = tiff.imread(mask_path + '{}.tif'.format(img_id)).transpose([1, 2, 0]) / 255
#         train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
#         X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
#         Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
#         X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
#         Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
#         print(img_id + ' read')
#     print('Images were read')
#
#     def train_net():
#         print("start train net")
#         # Random sample patches
#         x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=train_size, sz=patch_size)
#         x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=val_size, sz=patch_size)
#         # Create the model schema
#         n_channels = len(bands)
#         if bands[0] == -1:
#             n_channels = 8
#         print("\t num_channels", n_channels)
#         model = get_model(n_classes, patch_size, n_channels, class_weights)
#         # Load pre-existing weights, if any
#         if os.path.isfile(input_weights_path):
#             model.load_weights(input_weights_path)
#         # Make model check point configuration
#         model_checkpoint = ModelCheckpoint(output_weights_path, monitor='val_loss', save_best_only=True)
#         # Logger configuration
#         csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
#         # Tensorboard logging configuration
#         tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
#
#         # Fit the model on the sampled data
#         # model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
#         #           verbose=2, shuffle=True,
#         #           callbacks=[model_checkpoint, csv_logger, tensorboard],
#         #           validation_data=(x_val, y_val))
#
#         # OR
#         # Fit the model using ImageDataAugmentor from Keras
#         # Keras documentation - https://keras.io/preprocessing/image/
#         datagen = ImageDataGenerator(
#            featurewise_center=True,
#            featurewise_std_normalization=True,
#            rotation_range=90,
#            width_shift_range=0.2,
#            height_shift_range=0.2,
#            horizontal_flip=True,
#            vertical_flip=True)
#         # compute quantities required for featurewise normalization
#         # (std, mean, and principal components if ZCA whitening is applied)
#         datagen.fit(x_train)
#
#         # fits the model on batches with real-time data augmentation:
#         model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size,
#                                          seed=int(time.time())),  # Change the random seed every time
#                            verbose=2,
#                            shuffle=True,
#                            callbacks=[model_checkpoint, csv_logger, tensorboard],
#                            validation_data=(x_val, y_val),
#                            epochs=num_epochs)
#
#         return model
#
#     train_net()
#
#



if __name__ == '__main__':

    data_path = "/home/sriharis/scratch-midway2/train_tier_1/"
    acc_data_path = pjoin(data_path, "acc")
    acc_out_path = pjoin(os.getcwd(), "out")
    if not os.path.exists(acc_out_path):
        os.mkdir(acc_out_path)
    scene_id = "665946"
    scene_dir_path = pjoin(acc_data_path, scene_id)
    scene_labels_path = pjoin(acc_data_path, scene_id + "_labels")
    scene_path = pjoin(scene_dir_path, scene_id + ".tif")

    # Read the
    input_source = rasterio.open(scene_path)
    input_source_lbl = rasterio.open(scene_labels_path)

    print(input_source.width, input_source.height)
    print(input_source_lbl.width, input_source_lbl.height)

    

