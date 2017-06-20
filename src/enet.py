from keras.layers.advanced_activations import PReLU
from keras.layers.core import SpatialDropout2D, Permute
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D
from keras.layers.core import Activation
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input


##############################################
# ENCODER
##############################################

def encoder_initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):
    conv = Conv2D(nb_filter, (nb_row, nb_col), padding='same', strides=strides)(inp)
    max_pool = MaxPooling2D()(inp)
    merged = concatenate([conv, max_pool], axis=3)
    return merged


def encoder_bottleneck(input, num_output_filters, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = num_output_filters / internal_scale
    encoder = input

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
        raise (Exception('Invalid encoder bottleneck type'))

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # 1x1
    encoder = Conv2D(num_output_filters, (1, 1), use_bias=False)(encoder)

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = input
    # other branch
    if downsample:
        other = MaxPooling2D()(other)

        other = Permute((1, 3, 2))(other)
        pad_feature_maps = num_output_filters - input.get_shape().as_list()[3]
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

def decoder_bottleneck(input, num_output_filters, upsample=False):
    internal = num_output_filters / 4

    x = Conv2D(internal, (1, 1), use_bias=False)(input)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    if not upsample:
        x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(num_output_filters, (1, 1), padding='same', use_bias=False)(x)

    other = input
    if input.get_shape()[-1] != num_output_filters or upsample:
        other = Conv2D(num_output_filters, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample:
            other = UpSampling2D(size=(2, 2))(other)

    if not upsample:
        x = BatchNormalization(momentum=0.1)(x)
    else:
        return x

    decoder = add([x, other])
    decoder = Activation('relu')(decoder)
    return decoder


def decoder_build(encoder, nc):
    enet = decoder_bottleneck(encoder, 64, upsample=True)  # bottleneck 4.0
    enet = decoder_bottleneck(enet, 64)  # bottleneck 4.1
    enet = decoder_bottleneck(enet, 64)  # bottleneck 4.2
    enet = decoder_bottleneck(enet, 16, upsample=True)  # bottleneck 5.0
    enet = decoder_bottleneck(enet, 16)  # bottleneck 5.1

    enet = Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)
    return enet


##############################################
# ENET NETWORK
##############################################

def get_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    enet = encoder_build(inputs)
    enet = decoder_build(enet, nc=num_classes)

    model = Model(inputs=inputs, outputs=enet, name='enet_naive_upsampling')

    return model