from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Dropout, \
    concatenate, UpSampling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

##############################################
# METRICS
##############################################

"""
Calculates the accuracy for each class, then takes the mean of that.

# Arguments
    y_true: ground truth classification BATCH_SIZExHxWxNUM_CLASSES
    y_predicted: predicted classification BATCH_SIZExHxWxNUM_CLASSES

# Returns
    The mean class accuracy.
"""


def mean_per_class_accuracy(num_classes):
    def mpca(y_true, y_pred):
        labels = K.tf.cast(K.tf.argmax(y_true, axis=-1), K.tf.int32)
        predictions = K.tf.cast(K.tf.argmax(y_pred, axis=-1), K.tf.int32)

        result = K.tf.metrics.mean_per_class_accuracy(labels, predictions, num_classes)[0]

        # Make sure the Tensorflow local variables are initialized
        sess = K.get_session()
        init = K.tf.local_variables_initializer()
        sess.run(init)

        return result

    return mpca


"""
Calculates the mean intersection over union which is a popular measure
for segmentation accuracy.

# Arguments
    y_true: ground truth classification BATCH_SIZExHxWxNUM_CLASSES
    y_predicted: predicted classification BATCH_SIZExHxWxNUM_CLASSES

# Returns
    The mean of IoU metrics over all classes
"""


def mean_iou(num_classes):
    def miou(y_true, y_pred):
        labels = K.tf.cast(K.tf.argmax(y_true, axis=-1), K.tf.int32)
        predictions = K.tf.cast(K.tf.argmax(y_pred, axis=-1), K.tf.int32)

        # Make sure the Tensorflow local variables are initialized
        sess = K.get_session()
        init = K.tf.local_variables_initializer()
        sess.run(init)

        result = K.tf.metrics.mean_iou(labels, predictions, num_classes)[0]

        return result

    return miou


##############################################
# LOSS/ACTIVATION FUNCTIONS
##############################################

_EPSILON = 10e-8

"""
Convert the input `x` to a tensor of type `dtype`.

# Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
# Returns
    A tensor.
"""


def _to_tensor(x, dtype):
    x = K.tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = K.tf.cast(x, dtype)
    return x


"""
A per-pixel softmax i.e. each pixel is considered as a sample and the
class probabilities for each pixel sum to one.

# Background
    Keras softmax doesn't work for N-dimensional tensors. The function
    takes in a keras matrix of size HxWxNUM_CLASSES and applies
    'depth-wise' softmax to the matrix. The output is thus a matrix of
    size HxWxNUM_CLASSES where for each WxH entry the depth slice
    of NUM_CLASSES entries sum to 1.
# Arguments
    matrix: A tensor from a network layer with dimensions HxWxNUM_CLASSES
"""


def depth_softmax(matrix):
    sigmoid = lambda x: 1.0 / (1.0 + K.exp(-x))
    sigmoided_matrix = sigmoid(matrix)
    softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=-1, keepdims=True)
    return softmax_matrix


"""
Pixel-wise categorical crossentropy between an output
tensor and a target tensor.

# Arguments
    y_pred: A tensor resulting from a softmax.
    y_true: A tensor of the same shape as `output`.
# Returns
    Output tensor.
"""


def pixelwise_crossentropy(y_true, y_pred):
    labels = K.tf.cast(K.tf.argmax(y_true, axis=-1), K.tf.int32)
    return K.tf.reduce_sum(K.tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels))


# epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
# y_pred = K.tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
# return - K.tf.reduce_sum(y_true * tf.log(y_pred))


"""
Pixel-wise weighted categorical crossentropy between an
output tensor and a target tensor.

# Arguments
    y_pred: A tensor resulting from a softmax.
    y_true: A tensor of the same shape as `output`.
    class_weights: Weights for each class
# Returns
    Output tensor.
"""


def weighted_pixelwise_crossentropy(class_weights):
    def loss(y_true, y_pred):
        # Try to increase numerical stability by adding epsilon to predictions
        # to counter very small predictions
        epsilon = K.tf.constant(value=0.00001, shape=shape)
        y_pred = y_pred + epsilon

        # Calculate cross-entropy loss
        softmax = K.tf.nn.softmax(y_pred)
        xent = -K.tf.reduce_sum(K.tf.multiply(y_true * K.tf.log(softmax), class_weights))

        # NaN protection
        xent = K.tf.where(K.tf.is_nan(xent), K.tf.ones_like(xent) * _EPSILON, xent);

        return xent

    return loss


#        epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
#        y_pred = K.tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
#        return - K.tf.reduce_sum(K.tf.multiply(y_true * K.tf.log(y_pred), class_weights))
# labels = K.tf.cast(K.tf.argmax(y_true, axis=-1), K.tf.int32)


##############################################
# UNET
##############################################

def get_convolution_block(
        num_filters,
        input_layer,
        name,
        use_batch_normalization=True,
        use_activation=True,
        use_dropout=True,
        kernel_size=(3, 3),
        padding='valid',
        conv2d_kernel_initializer='he_normal',
        conv2d_bias_initializer='zeros',
        relu_alpha=0.01,
        dropout_rate=0.2):
    conv = ZeroPadding2D(
        (1, 1),
        name='{}_padding'.format(name))(input_layer)

    conv = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding=padding,
        kernel_initializer=conv2d_kernel_initializer,
        bias_initializer=conv2d_bias_initializer,
        name=name)(conv)

    if use_activation:
        # With alpha=0.0 LeakyReLU is a ReLU
        conv = LeakyReLU(
            alpha=relu_alpha,
            name='{}_activation'.format(name))(conv)

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
            name='{}_normalization'.format(name))(conv)

    if use_dropout:
        conv = Dropout(dropout_rate)(conv)

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
        concat_layer,
        upsampling_size=(2, 2),
        kernel_size=(3, 3),
        pool_size=(2, 2)):
    # Add upsampling layer
    up = UpSampling2D(size=upsampling_size)(input_layer)

    # Add concatenation layer to pass features from encoder path
    # to the decoder path
    concat = concatenate([up, concat_layer], axis=-1)

    # Add convolution layers
    previous_layer = concat

    for i in range(0, num_convolutions):
        conv = get_convolution_block(
            num_filters=num_filters,
            input_layer=previous_layer,
            kernel_size=kernel_size,
            name='{}_conv{}'.format(name_prefix, i + 1))

        previous_layer = conv

    return previous_layer


def get_segnet(input_shape, num_classes):
    # H x W x CHANNELS
    # None, None - means we support variable sized images, however
    # each image within one minibatch during training has to have
    # the same dimensions
    inputs = Input(shape=input_shape)

    """
    Encoder path
    """
    conv1, pool1 = get_encoder_block('encoder_block1', 2, 64, inputs)
    conv2, pool2 = get_encoder_block('encoder_block2', 2, 128, conv1)
    conv3, pool3 = get_encoder_block('encoder_block3', 3, 256, conv2)
    conv4, pool4 = get_encoder_block('encoder_block4', 3, 512, conv3)
    conv5, pool5 = get_encoder_block('encoder_block5', 3, 1024, conv4)

    """
    Decoder path
    """
    conv6 = get_decoder_block('decoder_block1', 3, 1024, conv5, conv5)
    conv7 = get_decoder_block('decoder_block2', 3, 512, conv6, conv4)
    conv8 = get_decoder_block('decoder_block3', 3, 256, conv7, conv3)
    conv9 = get_decoder_block('decoder_block4', 2, 128, conv8, conv2)
    conv10 = get_decoder_block('decoder_block5', 2, 64, conv9, conv1)

    """
    Last convolutional layer and softmax activation for
    per-pixel classification
    """
    conv11 = Conv2D(num_classes, (1, 1), name='fc1', kernel_initializer='he_normal', bias_initializer='zeros')(conv10)
    #softmax = Lambda(depth_softmax, name='softmax')(conv11)

    model = Model(inputs=inputs, outputs=conv11, name='SegNet')

    return model


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
    # softmax = Lambda(depth_softmax, name='softmax')(conv10)

    model = Model(inputs=inputs, outputs=conv10, name='U-net')

    return model