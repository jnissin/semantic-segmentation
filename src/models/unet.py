# coding=utf-8

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Dropout, \
    concatenate, UpSampling2D, Lambda, SpatialDropout2D
from keras.layers.advanced_activations import LeakyReLU


##############################################
# ENCODER / DECODER BLOCKS
##############################################

def get_convolution_block(
        num_filters,
        input_layer,
        name,
        use_batch_normalization=True,
        use_activation=True,
        use_dropout=True,
        use_bias=True,
        kernel_size=(3, 3),
        padding='valid',
        conv2d_kernel_initializer='he_normal',
        conv2d_bias_initializer='zeros',
        relu_alpha=0.1,
        dropout_rate=0.1):
    conv = ZeroPadding2D(
        (1, 1),
        name='{}_padding'.format(name))(input_layer)

    conv = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding=padding,
        kernel_initializer=conv2d_kernel_initializer,
        bias_initializer=conv2d_bias_initializer,
        name=name,
        use_bias=use_bias)(conv)

    '''
    From a statistics point of view BN before activation does not make sense to me.
    BN is normalizing the distribution of features coming out of a convolution, some
    these features might be negative which will be truncated by a non-linearity like ReLU.
    If you normalize before activation you are including these negative values in the
    normalization immediately before culling them from the feature space. BN after
    activation will normalize the positive features without statistically biasing them
    with features that do not make it through to the next convolutional layer.
    '''
    if use_batch_normalization:
        conv = BatchNormalization(
            momentum=0.1,
            name='{}_normalization'.format(name))(conv)

    if use_activation:
        # With alpha=0.0 LeakyReLU is a ReLU
        conv = LeakyReLU(
            alpha=relu_alpha,
            name='{}_activation'.format(name))(conv)

    if use_dropout:
        conv = SpatialDropout2D(dropout_rate)(conv)

    return conv


def get_encoder_block(
        name_prefix,
        num_convolutions,
        num_filters,
        input_layer,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        strides=(2, 2)):
    previous_layer = input_layer

    # Add the convolution blocks
    for i in range(0, num_convolutions):
        conv = get_convolution_block(
            num_filters=num_filters,
            input_layer=previous_layer,
            kernel_size=kernel_size,
            name='{}_conv{}'.format(name_prefix, i + 1))

        previous_layer = conv

    # Add the pooling layer
    pool = MaxPooling2D(
        pool_size=pool_size,
        strides=strides,
        name='{}_pool'.format(name_prefix))(previous_layer)

    return previous_layer, pool


def get_decoder_block(
        name_prefix,
        num_convolutions,
        num_filters,
        input_layer,
        concat_layer=None,
        upsampling_size=(2, 2),
        kernel_size=(3, 3)):

    # Add upsampling layer
    up = UpSampling2D(size=upsampling_size)(input_layer)

    # Add concatenation layer to pass features from encoder path
    # to the decoder path
    previous_layer = None

    if concat_layer is not None:
        concat = concatenate([up, concat_layer], axis=-1)
        previous_layer = concat
    else:
        previous_layer = up

    for i in range(0, num_convolutions):
        conv = get_convolution_block(
            num_filters=num_filters,
            input_layer=previous_layer,
            kernel_size=kernel_size,
            name='{}_conv{}'.format(name_prefix, i + 1),
            use_bias=False,
            use_activation=False)

        previous_layer = conv

    return previous_layer


##############################################
# SEGNET
##############################################


def get_segnet(input_shape, num_classes):
    # H x W x CHANNELS
    # None, None - means we support variable sized images, however
    # each image within one minibatch during training has to have
    # the same dimensions
    inputs = Input(shape=input_shape)

    # SegNet-Basic

    """
    Encoder path
    """
    conv1, pool1 = get_encoder_block('encoder_block1', 2, 64, inputs)
    conv2, pool2 = get_encoder_block('encoder_block2', 2, 128, pool1)
    conv3, pool3 = get_encoder_block('encoder_block3', 3, 256, pool2)
    conv4, pool4 = get_encoder_block('encoder_block4', 3, 512, pool3)
    #conv5, pool5 = get_encoder_block('encoder_block5', 3, 1024, conv4)

    """
    Decoder path
    """
    #conv6 = get_decoder_block('decoder_block1', 3, 1024, conv5, conv5)
    conv7 = get_decoder_block('decoder_block2', 3, 512, conv4, conv4)
    conv8 = get_decoder_block('decoder_block3', 3, 256, conv7, conv3)
    conv9 = get_decoder_block('decoder_block4', 2, 128, conv8, conv2)
    conv10 = get_decoder_block('decoder_block5', 2, 64, conv9, conv1)

    """
    Last convolutional layer and softmax activation for
    per-pixel classification
    """
    conv11 = Conv2D(num_classes, (1, 1), name='fc1', kernel_initializer='he_normal', bias_initializer='zeros')(conv10)
    
    model = Model(inputs=inputs, outputs=conv11, name='SegNet-Basic')

    return model



##############################################
# UNET
##############################################

'''
The functions builds the U-net model presented in the paper:
https://arxiv.org/pdf/1505.04597.pdf
'''


def get_unet(input_shape, num_classes):
    # H x W x CHANNELS
    # None, None - means we support variable sized images, however
    # each image within one minibatch during training has to have
    # the same dimensions
    inputs = Input(shape=input_shape)

    '''
    Contracting path
    '''
    conv1, pool1 = get_encoder_block('encoder_block1', 2, 64, inputs)
    conv2, pool2 = get_encoder_block('encoder_block2', 2, 128, pool1)
    conv3, pool3 = get_encoder_block('encoder_block3', 2, 256, pool2)
    conv4, pool4 = get_encoder_block('encoder_block4', 2, 512, pool3)

    '''
    Connecting path
    '''
    conv5 = get_convolution_block(
        num_filters=1024,
        input_layer=pool4,
        name='connecting_block_conv1')

    conv5 = get_convolution_block(
        num_filters=1024,
        input_layer=conv5,
        name='connecting_block_conv2')

    '''
    Expansive path
    '''
    conv6 = get_decoder_block('decoder_block1', 2, 512, conv5, conv4)
    conv7 = get_decoder_block('decoder_block2', 2, 256, conv6, conv3)
    conv8 = get_decoder_block('decoder_block3', 2, 128, conv7, conv2)
    conv9 = get_decoder_block('decoder_block4', 2, 64, conv8, conv1)

    '''
    Last convolutional layer and softmax activation for
    per-pixel classification
    '''
    conv10 = Conv2D(num_classes, (1, 1), name='fc1', kernel_initializer='he_normal', bias_initializer='zeros')(conv9)

    model = Model(inputs=inputs, outputs=conv10, name='U-net')

    return model