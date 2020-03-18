#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from osgeo import gdal_array
import gdal
from gdalconst import *
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss


from keras import losses
from keras.optimizers import Adam

import keras.backend as K
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from xnet import XNet

IMAGE_SIZE = 256


# Normalize pixel-value from -1 to 1 for original images

def new_normalize_x(image,min,max):

    image = 2 * (image - (min)) / (max - (min)) - 1 # test

    return image

##MAX 27.4614 MIN -22.3425  train
### MAX 19.8958 MIN -21.1946 test
def normalize_x(image):
    image2=image*1
    #image = 2*(image-(-22.34))/(27.46-(-22.34))-1  #train
    image = 2 * (image - (-21.19)) / (19.8958 - (-21.19)) - 1 # test
    return image


# Normalize pixel-value from 0 to 1 for label images
# assuming label image has 0-255
def normalize_y(image):
    image = image / 255
    return image


# Convert normalized value to 0 to 255
def denormalize_y(image):
    image = image * 255
    return image

def load_X(folder_path,bands):
    import os, cv2

    # image_files = os.listdir(folder_path)
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    image_files.sort()
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, bands), np.float32)
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, bands), np.float32)

    # Read tiff file via gdal and convert to cv array from gdal array
    for i in range(len(image_files)):
        image_file = image_files[i]
        g = gdal.Open(folder_path + os.sep + image_file, gdal.GA_ReadOnly)
        g_image = np.array([g.GetRasterBand(band+1).ReadAsArray() for band in range(bands)])


        #---------new ver ----------------
        b, h, w = g_image.shape
        g_image_v = np.zeros((h, w, b))
        g_image_v[:, :, 0] = g_image[0, :, :]
        g_image_v[:, :, 1] = g_image[1, :, :]
        g_image = g_image_v


        g_image = cv2.resize(g_image, (IMAGE_SIZE, IMAGE_SIZE))

        image = np.copy(g_image)
        print('min=', image.min(), ' max=', image.max(), ' avg=', image.mean())
        images[i] = new_normalize_x(image,image.min(),image.max())
        #print('-------------------')
    num_img=len(image_files)

    return images, image_files, num_img


# Load label images
def load_Y(folder_path,bands):
    import os, cv2

    image_files = os.listdir(folder_path)
    image_files.sort()
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, bands), np.float32)
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, bands), np.float32)

    for i in range(len(image_files)):
        image_file = image_files[i]
        g = gdal.Open(folder_path + os.sep + image_file, gdal.GA_ReadOnly)
        #img_cv = cv2.imread(folder_path + os.sep + image_file, -1) #max 255
        g_image = np.array([g.GetRasterBand(band+1).ReadAsArray() for band in range(bands)]) #max 1
        #print('shape_Y',g_image.shape)
        #plt.imshow(g_image[0,:,:])
        g_image=g_image*255

        b, h, w = g_image.shape
        g_image_v = np.zeros((h, w, b))
        g_image_v[:, :, 0] = g_image[0, :, :]
        #g_image_v[:, :, 1] = g_image[1, :, :]
        g_image = g_image_v



        # #print('g img before',g_image.shape,g_image)
        g_image = cv2.resize(g_image, (IMAGE_SIZE, IMAGE_SIZE))
        image = np.copy(g_image)
        #plt.imshow(g_image[:, :])
        image = image[:, :, np.newaxis]
        images[i] = normalize_y(image)

    return images

def weighted_binary_crossentropy2( y_true, y_pred ) :
    pos_weight= 181.5  # have to chaneg in train func loss=
    i=2.2250738585072014e-30

    logloss = y_true * -K.log(y_pred+i) * pos_weight + (1 - y_true) * -K.log(1 - y_pred+i)

    return K.mean( logloss, axis=-1)

def weighted_binary_crossentropy_mix( y_true, y_pred) :
    pos_weight= 181.5  # have to chaneg in train func loss=
    i=2.2250738585072014e-30
    y0_true = y_true[:,:,0]
    y1_true = y_true[:, :, 1]
    y0_pred = y_pred[:,:,0]
    y1_pred = y_pred[:, :, 0]

    logloss0 = y0_true * -K.log(y0_pred+i) * pos_weight + (1 - y0_true) * -K.log(1 - y0_pred+i)
    logloss1 = y1_true * -K.log(y1_pred + i) * pos_weight + (1 - y1_true) * -K.log(1 - y1_pred + i)
    logloss = (0.7*logloss0)+(0.3*logloss1)
    return K.mean( logloss, axis=-1)

def reverseinputlayer(input):
    c,h,w,b = input.shape
    temp =np.zeros((c,h, w, b))
    temp[:,:,:,0]=input[:,:,:,1]
    temp[:,:,:,1] = input[:,:, :, 0]

    return temp

# Compatible with tensorflow backend

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def train_unet(input_channel_count,images_dir_name,label_dir_name,save_model):
    input_dir_train = './test_datasets/data/trainData/'
    check_dir = './test_datasets/check_points/'

    # input_channel_count = 2
    output_channel_count = 1

    print("Data load...")
    # trainingDataフォルダ配下にleft_imagesフォルダを置いている
    #X_train, file_names = load_X(input_dir_train + os.sep + 'left_images', input_channel_count)
    X_train, file_names, num_img= load_X('datasets' + os.sep + images_dir_name, input_channel_count)
    X_train_R = reverseinputlayer(X_train)

    # trainingDataフォルダ配下にleft_groundTruthフォルダを置いている
    #Y_train = load_Y(input_dir_train + os.sep + 'left_groundTruth')
    Y_train = load_Y('datasets' + os.sep + label_dir_name,output_channel_count)

    print("Model define...")
    # number of filters for first layer
    first_layer_filter_count = 64

    #loss_func='weighted_binary_crossentropy2'+str(class_weight[0])+str(class_weight[1])
    loss_func = 'weighted_bce_mix' + str(181)
    # loss_func ='focal_loss'+'alpha25_gamma2'

    output_channel_count = 2
    # Generate U-Net
    network = XNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    #model.compile(loss=dice_coef_loss, optimizer=Adam(), metrics=[dice_coef])

    model.compile(loss=weighted_binary_crossentropy_mix, optimizer='adam', metrics=['accuracy'])
    # model.compile(loss=[focal_loss(alpha=.25, gamma=2)], optimizer='adam', metrics=['accuracy'])


    # Batch size
    BATCH_SIZE = 16
    # Epoch
    NUM_EPOCH = 10

    #Y_binary = to_categorical(Y_train)

    # Training

    history = model.fit([X_train,X_train_R], Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1, validation_split=0.10)
    #history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1, validation_split=0.10)

    # Save progress
    print("Save...")
    save_model = 'xnet' + loss_func + '_T' + str(num_img) + 'E' + str(NUM_EPOCH) + '.hdf5'
    model.save_weights(save_model)

    return save_model

def predict(input_channel_count,test_dir_name,use_model,wheretosave='datasets'+os.sep+'predictions',savename='pre_'):
    import cv2
    import gdal

    # Image sets: ./datasets/test_images/
    X_test, file_names,num_image = load_X('datasets' + os.sep + test_dir_name,input_channel_count)
    X_test_R = reverseinputlayer(X_test)

    #input_channel_count = 2
    output_channel_count = 1
    first_layer_filter_count = 64
    network = XNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    model.load_weights(use_model)

    # Batch size setting #
    BATCH_SIZE = 12

    # Prediction #
    Y_pred = model.predict([X_test, X_test_R], BATCH_SIZE)

    # Image sets: ./datasets/test_images/
    # Output: ./datasets/predictions/

    for i, y in enumerate(Y_pred):
        img = gdal.Open('datasets' + os.sep + test_dir_name + os.sep + file_names[i])
        print('filename','datasets' + os.sep + test_dir_name + os.sep + file_names[i])
        cols = img.RasterXSize
        rows = img.RasterYSize

        dataset =img

        width = dataset.RasterXSize
        height = dataset.RasterYSize
        gt = dataset.GetGeoTransform()
        minx = gt[0]  # top left x
        miny = gt[3] + width * gt[4] + height * gt[5]
        maxx = gt[0] + width * gt[1] + height * gt[2]
        maxy = gt[3]  # top left y
        Mres = gt[5]
        Pres = gt[1]
        bands=1

        y = cv2.resize(y, (cols, rows))
        ###### save geo corrodinate #######
        format = 'GTiff'
        driver = gdal.GetDriverByName(format)
        driver.Register()
        if not os.path.exists(wheretosave):
            os.makedirs(wheretosave)
        dst_ds = driver.Create(wheretosave + os.sep + savename + use_model + file_names[i],cols, rows, bands, GDT_Float32)
        dst_ds.SetGeoTransform(dataset.GetGeoTransform())
        dst_ds.SetProjection(dataset.GetProjection())
        dst_ds.GetRasterBand(1).WriteArray(y)
        
if __name__ == '__main__':

    # set number of input channels
    # for 3 channels
    #input_channel_count = 3
    # for 2 channels
    input_channel_count = 2

    # Train #
    images_dir_name = 'images_200'#'images' this pc
    label_dir_name = 'labels_200'#'labels' this_pc
    save_model ='unet_2x_gdal32binary_crossentropy_T2028E10.hdf5'

    print("Training...")
    model_name=train_unet(input_channel_count,images_dir_name,label_dir_name,save_model)

    # Prediction #
    test_dir_name = 'test_images_sar2band'

    use_model = 'xynet_7-3o_7-3r_in_half_add_r_sk_max_dec8_weighted_binary_crossentropy2181_T2028E10.hdf5' #main_corn
    prediction_output_path = 'datasets'+os.sep+'predictions_'+test_dir_name

    prediction_save_name = 'pre_'
    print("Prediction...")
    predict(input_channel_count,test_dir_name,use_model,prediction_output_path,prediction_save_name)



