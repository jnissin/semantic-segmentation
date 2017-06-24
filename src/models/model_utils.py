# coding=utf-8

from keras import backend as K
from tensorflow.python.client import device_lib

import numpy as np

import unet
import enet_naive_upsampling
import enet_max_unpooling
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary


##############################################
# GLOBALS
##############################################

_EPSILON = 10e-8


##############################################
# UTILITIES
##############################################

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def _tf_filter_nans(t, epsilon):
    """
    Filter NaNs from a tensor 't' and replace with value epsilon

    # Arguments
        t: A tensor to filter
        epsilon: Value to replace NaNs with
    # Returns
        A tensor of same shape as t with NaN values replaced by epsilon.
    """

    return K.tf.where(K.tf.is_nan(t), K.tf.ones_like(t) * epsilon, t)


def _tf_clamp_to_min(t, epsilon):
    return K.tf.where(K.tf.less(t, epsilon), K.tf.ones_like(t) * epsilon, t)


def _tf_initialize_local_variables():
    """
    Initializes all the global and local variables of the Keras Tensorflow backend
    session.
    """
    sess = K.get_session()

    local_init = K.tf.local_variables_initializer()
    global_init = K.tf.global_variables_initializer()
    sess.run(local_init)
    sess.run(global_init)


def _to_tensor(x, dtype):
    """
    Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """

    x = K.tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = K.tf.cast(x, dtype)
    return x


def get_model(model_name, input_shape, num_classes):
    """
    Get the model by the model name.

    # Arguments
        model_name: name of the model
        input_shape: input shape to the model
        num_classes: number of classification classes
    # Returns
        The appropriate Keras model.
    """
    if model_name == 'unet':
        return unet.get_unet(input_shape, num_classes)
    elif model_name == 'enet-naive-upsampling':
        return enet_naive_upsampling.get_model(input_shape, num_classes)
    elif model_name == 'enet-max-unpooling':
        return enet_max_unpooling.get_model(input_shape, num_classes)
    else:
        raise ValueError('Unknown model name: {}'.format(model_name))

    return None

##############################################
# METRICS
##############################################


def mean_per_class_accuracy(num_classes):

    def mpca(y_true, y_pred):
        """
         Calculates the accuracy for each class, then takes the mean of that.

         # Arguments
             y_true: ground truth classification BATCH_SIZExHxWxNUM_CLASSES
             y_predicted: predicted classification BATCH_SIZExHxWxNUM_CLASSES

         # Returns
             The mean class accuracy.
         """

        labels = K.tf.cast(K.tf.argmax(y_true, axis=-1), K.tf.int32)
        predictions = K.tf.cast(K.tf.argmax(y_pred, axis=-1), K.tf.int32)
        cfm = K.tf.confusion_matrix(
            labels=K.flatten(labels), predictions=K.flatten(predictions), num_classes=num_classes)

        """
        Compute the mean per class accuracy via the confusion matrix.
        """

        per_row_sum = K.tf.to_float(K.tf.reduce_sum(cfm, 1))
        cm_diag = K.tf.to_float(K.tf.diag_part(cfm))
        denominator = per_row_sum

        # If the value of the denominator is 0, set it to 1 to avoid
        # zero division.
        denominator = K.tf.where(
            K.tf.greater(denominator, 0), denominator,
            K.tf.ones_like(denominator))
        accuracies = K.tf.div(cm_diag, denominator)
        result = K.tf.reduce_mean(accuracies)

        return result

    return mpca


def mean_iou(num_classes):
    def miou(y_true, y_pred):
        """
        Calculates the mean intersection over union which is a popular measure
        for segmentation accuracy.

        # Arguments
            y_true: ground truth classification BATCH_SIZExHxWxNUM_CLASSES
            y_predicted: predicted classification BATCH_SIZExHxWxNUM_CLASSES

        # Returns
            The mean of IoU metrics over all classes
        """

        labels = K.tf.cast(K.tf.argmax(y_true, axis=-1), K.tf.int32)
        predictions = K.tf.cast(K.tf.argmax(y_pred, axis=-1), K.tf.int32)
        cfm = K.tf.confusion_matrix(
            labels=K.flatten(labels), predictions=K.flatten(predictions), num_classes=num_classes)

        """
        Compute the mean intersection-over-union via the confusion matrix.
        """

        sum_over_row = K.tf.to_float(K.tf.reduce_sum(cfm, 0))
        sum_over_col = K.tf.to_float(K.tf.reduce_sum(cfm, 1))
        cm_diag = K.tf.to_float(K.tf.diag_part(cfm))
        denominator = sum_over_row + sum_over_col - cm_diag

        # If the value of the denominator is 0, set it to 1 to avoid
        # zero division.
        denominator = K.tf.where(
            K.tf.greater(denominator, 0),
            denominator,
            K.tf.ones_like(denominator))

        iou = K.tf.div(cm_diag, denominator)
        result = K.tf.reduce_mean(iou)

        return result

    return miou


##############################################
# LOSS/ACTIVATION FUNCTIONS
##############################################

def depth_softmax(matrix):
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
    sigmoid = lambda x: 1.0 / (1.0 + K.exp(-x))
    sigmoided_matrix = sigmoid(matrix)
    softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=-1, keepdims=True)
    return softmax_matrix


def pixelwise_crossentropy(y_true, y_pred):
    """
    Pixel-wise categorical cross-entropy between an output
    tensor and a target tensor.

    # Arguments
        y_pred: A tensor resulting from a softmax.
        y_true: A tensor of the same shape as `output`.
    # Returns
        Output tensor.
    """
    labels = K.tf.cast(K.tf.argmax(y_true, axis=-1), K.tf.int32)
    return K.tf.reduce_sum(K.tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels))


def weighted_pixelwise_crossentropy(class_weights):
    def loss(y_true, y_pred):
        """
        Pixel-wise weighted categorical cross-entropy between an
        output tensor and a target tensor.

        # Arguments
            y_pred: A tensor resulting from the last convolutional layer.
            y_true: A tensor of the same shape as `y_pred`.
            class_weights: Weights for each class
        # Returns
            Output tensor.
        """
        # Calculate cross-entropy loss
        epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)

        softmax = K.tf.nn.softmax(y_pred)
        softmax = K.tf.clip_by_value(softmax, epsilon, 1. - epsilon)
        softmax = _tf_filter_nans(softmax, epsilon)

        xent = K.tf.multiply(y_true * K.tf.log(softmax), class_weights)
        xent = _tf_filter_nans(xent, epsilon)
        xent = -K.tf.reduce_mean(K.tf.reduce_sum(xent, axis=(1, 2, 3)))

        return xent

    return loss


##############################################
# NUMPY DATA MANIPULATION
##############################################

def np_softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


##############################################
# DENSE CRF
##############################################

def get_dcrf(img, nlabels):
    width = img.shape[1]
    height = img.shape[0]
    img_shape = img.shape[:2]

    d = dcrf.DenseCRF2D(width, height, nlabels)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(5, 5), shape=img_shape)

    d.addPairwiseEnergy(feats,
                        compat=4,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(80, 80),
                                      schan=(20, 20, 20),
                                      img=img,
                                      chdim=2)

    d.addPairwiseEnergy(feats,
                        compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    return d
