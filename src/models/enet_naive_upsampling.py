# coding=utf-8

from keras.layers.advanced_activations import PReLU
from keras.layers.core import SpatialDropout2D, Permute
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
import keras.backend as K

##############################################
# ENCODER
##############################################

def encoder_initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):
    conv = Conv2D(nb_filter, (nb_row, nb_col), padding='same', strides=strides)(inp)
    max_pool = MaxPooling2D()(inp)
    merged = concatenate([conv, max_pool], axis=3)
    return merged


def encoder_bottleneck(inp, output, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = output / internal_scale
    encoder = inp

    # 1x1
    input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = Conv2D(internal, (input_stride, input_stride),
                     # padding='same',
                     strides=(input_stride, input_stride), use_bias=False)(encoder)
    # Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # conv
    if not asymmetric and not dilated:
        encoder = Conv2D(internal, (3, 3), padding='same')(encoder)
    elif asymmetric:
        encoder = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
        encoder = Conv2D(internal, (asymmetric, 1), padding='same')(encoder)
    elif dilated:
        encoder = Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(encoder)
    else:
        raise (Exception('You shouldn\'t be here'))

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # 1x1
    encoder = Conv2D(output, (1, 1), use_bias=False)(encoder)

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = inp
    # other branch
    if downsample:
        other = MaxPooling2D()(other)

        other = Permute((1, 3, 2))(other)
        pad_feature_maps = output - inp.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = Permute((1, 3, 2))(other)

    encoder = add([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)
    return encoder


def encoder_build(inp, dropout_rate=0.01):
    enet = encoder_initial_block(inp)
    enet = encoder_bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0

    for i in range(4):
        enet = encoder_bottleneck(enet, 64, dropout_rate=dropout_rate)  # bottleneck 1.i

    enet = encoder_bottleneck(enet, 128, downsample=True)  # bottleneck 2.0
    # bottleneck 2.x and 3.x
    for i in range(2):
        enet = encoder_bottleneck(enet, 128)  # bottleneck 2.1
        enet = encoder_bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
        enet = encoder_bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.3
        enet = encoder_bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
        enet = encoder_bottleneck(enet, 128)  # bottleneck 2.5
        enet = encoder_bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
        enet = encoder_bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.7
        enet = encoder_bottleneck(enet, 128, dilated=16)  # bottleneck 2.8

    return enet


##############################################
# DECODER
##############################################

def decoder_bottleneck(encoder, output, upsample=False, reverse_module=False):
    internal = output / 4

    x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    if not upsample:
        x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(output, (1, 1), padding='same', use_bias=False)(x)

    other = encoder
    if encoder.get_shape()[-1] != output or upsample:
        other = Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module:
            other = UpSampling2D(size=(2, 2))(other)

    if not upsample or reverse_module:
        x = BatchNormalization(momentum=0.1)(x)
    else:
        return x

    decoder = add([x, other])
    decoder = Activation('relu')(decoder)
    return decoder


def decoder_build(encoder, nc):
    enet = decoder_bottleneck(encoder, 64, upsample=True, reverse_module=True)  # bottleneck 4.0
    enet = decoder_bottleneck(enet, 64)  # bottleneck 4.1
    enet = decoder_bottleneck(enet, 64)  # bottleneck 4.2
    enet = decoder_bottleneck(enet, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
    enet = decoder_bottleneck(enet, 16)  # bottleneck 5.1

    enet = Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)
    return enet


##############################################
# ENET NETWORK
##############################################

def get_model(input_shape, num_classes, encoder_only=False):
    inputs = Input(shape=input_shape)

    enet = encoder_build(inputs)

    # If we are only building the encoder add a convolutional layer and a softmax
    # activation layer for classification purposes
    if encoder_only:
        # In order to avoid increasing the number of variables with a huge dense layer
        # use average pooling with a pool size of the previous layer's spatial
        # dimension
        pool_size = K.int_shape(enet)[1:3]
        enet = AveragePooling2D(pool_size=pool_size)(enet)
        enet = Flatten()(enet)
        enet = Dense(num_classes, activation='softmax')(enet)
    else:
        enet = decoder_build(enet, nc=num_classes)

    model = Model(inputs=inputs, outputs=enet, name='enet_naive_upsampling')

    return model