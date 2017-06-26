# coding=utf-8

from keras.layers import Input
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.core import SpatialDropout2D, Permute
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model

import keras.backend as K

from layers.pooling import MaxPoolingWithArgmax2D
from layers.pooling import MaxUnpooling2D

##############################################
# ENCODER
##############################################


def encoder_initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):
    conv = Conv2D(filters=nb_filter,
                  kernel_size=(nb_row, nb_col),
                  padding='same',
                  strides=strides,
                  name='input_block_conv2d')(inp)

    max_pool, indices = MaxPoolingWithArgmax2D(name='initial_block_pool2d')(inp)
    merged = concatenate([conv, max_pool], axis=3, name='initial_block_concat')
    return merged, indices


def encoder_bottleneck(inp,
                       output,
                       name_prefix,
                       internal_scale=4,
                       asymmetric=0,
                       dilated=0,
                       downsample=False,
                       dropout_rate=0.1):

    internal = output / internal_scale
    encoder = inp

    """
    Main branch
    """
    # 1x1 projection downwards to internal feature space
    # Note: The 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    input_stride = 2 if downsample else 1

    encoder = Conv2D(filters=internal,
                     kernel_size=(input_stride, input_stride),
                     strides=(input_stride, input_stride),
                     use_bias=False,
                     name='{}_proj_conv2d_1'.format(name_prefix))(encoder)

    # Batch normalization + PReLU
    # ENet uses momentum of 0.1, keras default is 0.99
    encoder = BatchNormalization(momentum=0.1, name='{}_bnorm_1'.format(name_prefix))(encoder)
    encoder = PReLU(shared_axes=[1, 2], name='{}_prelu_1'.format(name_prefix))(encoder)

    # Convolution block; either normal, asymmetric or dilated convolution
    if not asymmetric and not dilated:
        encoder = Conv2D(filters=internal,
                         kernel_size=(3, 3),
                         padding='same',
                         name='{}_conv2d_1'.format(name_prefix))(encoder)
    elif asymmetric:
        encoder = Conv2D(filters=internal,
                         kernel_size=(1, asymmetric),
                         padding='same',
                         use_bias=False,
                         name='{}_aconv2d_1'.format(name_prefix))(encoder)

        encoder = Conv2D(filters=internal,
                         kernel_size=(asymmetric, 1),
                         padding='same',
                         name='{}_aconv2d_2'.format(name_prefix))(encoder)
    elif dilated:
        encoder = Conv2D(filters=internal,
                         kernel_size=(3, 3),
                         dilation_rate=(dilated, dilated),
                         padding='same',
                         name='{}_dconv2d'.format(name_prefix))(encoder)
    else:
        raise RuntimeError('Invalid convolution options for encoder block')

    # ENet uses momentum of 0.1, keras default is 0.99
    encoder = BatchNormalization(momentum=0.1, name='{}_bnorm_2'.format(name_prefix))(encoder)
    encoder = PReLU(shared_axes=[1, 2], name='{}_prelu_2'.format(name_prefix))(encoder)

    # 1x1 projection upwards from internal to output feature space
    encoder = Conv2D(filters=output,
                     kernel_size=(1, 1),
                     use_bias=False,
                     name='{}_proj_conv2d_2'.format(name_prefix))(encoder)

    # ENet uses momentum of 0.1, keras default is 0.99
    encoder = BatchNormalization(momentum=0.1, name='{}_bnorm_3'.format(name_prefix))(encoder)
    encoder = SpatialDropout2D(rate=dropout_rate, name='{}_sdrop2d_1'.format(name_prefix))(encoder)

    """
    Other branch
    """
    other = inp

    if downsample:
        other, indices = MaxPoolingWithArgmax2D(name='{}_other_pool2d'.format(name_prefix))(other)
        other = Permute((1, 3, 2), name='{}_other_permute_1'.format(name_prefix))(other)

        pad_feature_maps = output - inp.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad), name='{}_other_zpad2d'.format(name_prefix))(other)
        other = Permute((1, 3, 2), name='{}_other_permute_2'.format(name_prefix))(other)

    """
    Merge branches
    """
    encoder = add([encoder, other], name='{}_add'.format(name_prefix))
    encoder = PReLU(shared_axes=[1, 2], name='{}_prelu_3'.format(name_prefix))(encoder)

    if downsample:
        return encoder, indices

    return encoder


def encoder_build(inp, dropout_rate=0.01):
    pooling_indices = []

    # Initial block
    enet, indices_single = encoder_initial_block(inp)
    pooling_indices.append(indices_single)

    # Bottleneck 1.0
    enet, indices_single = encoder_bottleneck(enet, 64, name_prefix='en_bn_1.0', downsample=True, dropout_rate=dropout_rate)
    pooling_indices.append(indices_single)

    # Bottleneck 1.i
    for i in range(4):
        name_prefix = 'en_bn_1.{}'.format(i+1)
        enet = encoder_bottleneck(enet, 64, name_prefix=name_prefix, dropout_rate=dropout_rate)

    # Bottleneck 2.0
    enet, indices_single = encoder_bottleneck(enet, 128, name_prefix='en_bn_2.0', downsample=True)
    pooling_indices.append(indices_single)

    # Bottleneck 2.x and 3.x
    for i in range(2):
        name_prefix = 'en_bn_{}.'.format(2+i)

        enet = encoder_bottleneck(enet, 128, name_prefix=name_prefix+'1')               # bottleneck 2.1
        enet = encoder_bottleneck(enet, 128, name_prefix=name_prefix+'2', dilated=2)    # bottleneck 2.2
        enet = encoder_bottleneck(enet, 128, name_prefix=name_prefix+'3', asymmetric=5) # bottleneck 2.3
        enet = encoder_bottleneck(enet, 128, name_prefix=name_prefix+'4', dilated=4)    # bottleneck 2.4
        enet = encoder_bottleneck(enet, 128, name_prefix=name_prefix+'5')               # bottleneck 2.5
        enet = encoder_bottleneck(enet, 128, name_prefix=name_prefix+'6', dilated=8)    # bottleneck 2.6
        enet = encoder_bottleneck(enet, 128, name_prefix=name_prefix+'7', asymmetric=5) # bottleneck 2.7
        enet = encoder_bottleneck(enet, 128, name_prefix=name_prefix+'8', dilated=16)   # bottleneck 2.8

    return enet, pooling_indices


##############################################
# DECODER
##############################################

def decoder_bottleneck(encoder,
                       output,
                       name_prefix,
                       upsample=False,
                       reverse_module=False):

    internal = output / 4

    """
    Main branch
    """
    # 1x1 projection downwards to internal feature space
    x = Conv2D(filters=internal,
               kernel_size=(1, 1),
               use_bias=False,
               name='{}_proj_conv2d_1'.format(name_prefix))(encoder)

    # ENet uses momentum of 0.1, keras default is 0.99
    x = BatchNormalization(momentum=0.1, name='{}_bnorm_1'.format(name_prefix))(x)
    x = Activation('relu', name='{}_relu_1'.format(name_prefix))(x)

    # Upsampling
    if not upsample:
        x = Conv2D(filters=internal,
                   kernel_size=(3, 3),
                   padding='same',
                   use_bias=True,
                   name='{}_conv2d_1'.format(name_prefix))(x)
    else:
        x = Conv2DTranspose(filters=internal,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='{}_tconv2d_1'.format(name_prefix))(x)


    # ENet uses momentum of 0.1 keras default is 0.99
    x = BatchNormalization(momentum=0.1, name='{}_bnorm_2'.format(name_prefix))(x)
    x = Activation('relu', name='{}_relu_2'.format(name_prefix))(x)

    # 1x1 projection upwards from internal to output feature space
    x = Conv2D(filters=output,
               kernel_size=(1, 1),
               padding='same',
               use_bias=False,
               name='{}_proj_conv2d_2'.format((name_prefix)))(x)

    """
    Other branch
    """
    other = encoder

    if encoder.get_shape()[-1] != output or upsample:
        other = Conv2D(filters=output,
                       kernel_size=(1, 1),
                       padding='same',
                       use_bias=False,
                       name='{}_other_conv2d'.format(name_prefix))(other)

        other = BatchNormalization(momentum=0.1, name='{}_other_bnorm_1'.format(name_prefix))(other)

        if upsample:
            mpool = MaxUnpooling2D(name='{}_other_unpool2d'.format(name_prefix))
            other = mpool([other, reverse_module])

    if upsample and reverse_module is False:
        return x
    else:
        x = BatchNormalization(momentum=0.1, name='{}_bnorm_3'.format(name_prefix))(x)

    """
    Merge branches
    """
    decoder = add([x, other], name='{}_add'.format(name_prefix))
    decoder = Activation('relu', name='{}_relu_3'.format(name_prefix))(decoder)

    return decoder


def decoder_build(encoder, index_stack, nc):
    enet = decoder_bottleneck(encoder, 64, name_prefix='de_bn_4.0',upsample=True, reverse_module=index_stack.pop())
    enet = decoder_bottleneck(enet, 64, name_prefix='de_bn_4.1')
    enet = decoder_bottleneck(enet, 64, name_prefix='de_bn_4.2')
    enet = decoder_bottleneck(enet, 16, name_prefix='de_bn_5.0', upsample=True, reverse_module=index_stack.pop())
    enet = decoder_bottleneck(enet, 16, name_prefix='de_bn_5.1')

    enet = Conv2DTranspose(filters=nc,
                           kernel_size=(2, 2),
                           strides=(2, 2),
                           padding='same',
                           name='de_tconv2d')(enet)

    return enet


##############################################
# ENET NETWORK
##############################################

def get_model(input_shape, num_classes, encoder_only=False):
    inputs = Input(shape=input_shape, name='input')

    enet, index_stack = encoder_build(inputs)
    name = 'enet_max_unpooling'

    if encoder_only:
        # In order to avoid increasing the number of variables with a huge dense layer
        # use average pooling with a pool size of the previous layer's spatial
        # dimension
        name = 'enet_max_unpooling_encoder_only'
        pool_size = K.int_shape(enet)[1:3]

        enet = AveragePooling2D(pool_size=pool_size, name='avg_pool2d')(enet)
        enet = Flatten(name='flatten')(enet)
        enet = Dense(num_classes, activation='softmax', name='fc1')(enet)
    else:
        enet = decoder_build(enet, index_stack=index_stack, nc=num_classes)

    model = Model(inputs=inputs, outputs=enet, name=name)
    return model
