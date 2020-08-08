import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, UpSampling2D, Concatenate
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D, AveragePooling2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, Nadam, SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img,save_img
import cv2
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard,  ReduceLROnPlateau, LearningRateScheduler
from keras.models import clone_model
from keras.utils import multi_gpu_model

from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model


def Conv2D_block(input_tensor, n_filters, name, kernel_size=3, batchnorm=True, kernel_initializer="he_uniform", dilation_rate=1):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same", kernel_initializer=kernel_initializer, dilation_rate=dilation_rate, name=name+"_1")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same", kernel_initializer=kernel_initializer, dilation_rate=dilation_rate, name=name+"_2")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def encoder(input_image, n_filters):
    
    c1 = Conv2D_block(input_image, n_filters, name="c1")
    mc1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D_block(mc1, n_filters*2, name="c2")
    mc2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D_block(mc2, n_filters*4, name="c3")
    mc3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D_block(mc3, n_filters*8, name="c4")
    mc4 = MaxPooling2D((2, 2))(c4)
    
    c5 = Conv2D_block(mc4, n_filters*16, name="c5")
    
    return c1, c2, c3, c4, c5


def decoder(c1, c2, c3, c4, c5, n_filters, name=""):
    
    #==================================================#

    d4_mp_1 = MaxPooling2D((8, 8))(c1)
    d4_mp_1_conv = Conv2D_block(d4_mp_1, n_filters, name="d4_mp_1_conv_" + name)
    
    d4_mp_2 = MaxPooling2D((4, 4))(c2)
    d4_mp_2_conv = Conv2D_block(d4_mp_2, n_filters, name="d4_mp_2_conv_" + name)
    
    d4_mp_3 = MaxPooling2D((2, 2))(c3)
    d4_mp_3_conv = Conv2D_block(d4_mp_3, n_filters, name="d4_mp_3_conv_" + name)
    
    d4_c_4 = Conv2D_block(c4, n_filters, name="d4_c_4_" + name)
    
    d4_up_5 = UpSampling2D((2, 2))(c5)
    d4_up_5_conv = Conv2D_block(d4_up_5, n_filters, name="d4_up_5_conv_" + name)
    
    d4_fusion = Concatenate()([d4_mp_1_conv, d4_mp_2_conv, d4_mp_3_conv, d4_c_4, d4_up_5_conv])
    d4_fusion = Conv2D_block(d4_fusion, n_filters, name="d4_fusion_" + name)
    
    #==================================================#
    
    d3_mp_1 = MaxPooling2D((4, 4))(c1)
    d3_mp_1_conv = Conv2D_block(d3_mp_1, n_filters, name="d3_mp_1_conv_" + name)
    
    d3_mp_2 = MaxPooling2D((2, 2))(c2)
    d3_mp_2_conv = Conv2D_block(d3_mp_2, n_filters, name="d3_mp_2_conv_" + name)
    
    d3_c_3 = Conv2D_block(c3, n_filters, name="d3_c_3_" + name)
    
    d3_up_4 = UpSampling2D((2, 2))(d4_fusion)
    d3_up_4_conv = Conv2D_block(d3_up_4, n_filters, name="d3_up_4_conv_" + name)
    
    d3_up_5 = UpSampling2D((4, 4))(c5)
    d3_up_5_conv = Conv2D_block(d3_up_5, n_filters, name="d3_up_5_conv_" + name)
    
    d3_fusion = Concatenate()([d3_mp_1_conv, d3_mp_2_conv, d3_c_3, d3_up_4_conv, d3_up_5_conv])
    d3_fusion = Conv2D_block(d3_fusion, n_filters, name="d3_fusion_" + name)
    
    #==================================================#
    
    d2_mp_1 = MaxPooling2D((2, 2))(c1)
    d2_mp_1_conv = Conv2D_block(d2_mp_1, n_filters, name="d2_mp_1_conv_" + name)
    
    d2_c_2 = Conv2D_block(c2, n_filters, name="d2_c_2_" + name)
    
    d2_up_3 = UpSampling2D((2, 2))(d3_fusion)
    d2_up_3_conv = Conv2D_block(d2_up_3, n_filters, name="d2_up_3_conv_" + name)
    
    d2_up_4 = UpSampling2D((4, 4))(d4_fusion)
    d2_up_4_conv = Conv2D_block(d2_up_4, n_filters, name="d2_up_4_conv_" + name)
    
    d2_up_5 = UpSampling2D((8, 8))(c5)
    d2_up_5_conv = Conv2D_block(d2_up_5, n_filters, name="d2_up_5_conv_" + name)
    
    d2_fusion = Concatenate()([d2_mp_1_conv, d2_c_2, d2_up_3_conv, d2_up_4_conv, d2_up_5_conv])
    d2_fusion = Conv2D_block(d2_fusion, n_filters, name="d2_fusion_" + name)
    
    #==================================================#
    
    d1_c_1 = Conv2D_block(c1, n_filters, name="d1_c_1_" + name)
    
    d1_up_2 = UpSampling2D((2, 2))(d2_fusion)
    d1_up_2_conv = Conv2D_block(d1_up_2, n_filters, name="d1_up_2_conv_" + name)
    
    d1_up_3 = UpSampling2D((4, 4))(d3_fusion)
    d1_up_3_conv = Conv2D_block(d1_up_3, n_filters, name="d1_up_3_conv_" + name)
    
    d1_up_4 = UpSampling2D((8, 8))(d4_fusion)
    d1_up_4_conv = Conv2D_block(d1_up_4, n_filters, name="d1_up_4_conv_" + name)
    
    d1_up_5 = UpSampling2D((16, 16))(c5)
    d1_up_5_conv = Conv2D_block(d1_up_5, n_filters, name="d1_up_5_conv_" + name)
    
    d1_fusion = Concatenate()([d1_c_1, d1_up_2_conv, d1_up_3_conv, d1_up_4_conv, d1_up_5_conv])
    d1_fusion = Conv2D_block(d1_fusion, n_filters, name="d1_fusion_" + name)
    
    #==================================================#
    
    return d1_fusion


def unet3plus(shape, n_classes, activation, n_filters=64):
    inputs = Input(shape, name='inp1')
    c1, c2, c3, c4, c5 = encoder(inputs, n_filters)
    class_output = decoder(c1, c2, c3, c4, c5, n_filters, "class")
    class_output = Conv2D(n_classes, (1, 1), activation=activation, name='unet_op_fin')(class_output)
    model = Model(inputs=inputs , outputs=class_output)
    return model
