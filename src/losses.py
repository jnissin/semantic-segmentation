# coding=utf-8

import numpy as np

from enum import Enum
from keras import backend as K

import settings


class ModelLambdaLossType(Enum):
    """
    These types are required in order to determine the necessary lambda layer
    for cost calculation (Keras doesn't support extra parameters to cost functions).
    """
    NONE = 0
    SEGMENTATION_CATEGORICAL_CROSS_ENTROPY = 1
    SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER = 2
    SEGMENTATION_SEMI_SUPERVISED_SUPERPIXEL = 3
    SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER_SUPERPIXEL = 4
    CLASSIFICATION_CATEGORICAL_CROSS_ENTROPY = 5
    CLASSIFICATION_SEMI_SUPERVISED_MEAN_TEACHER = 6


##############################################
# GLOBALS
##############################################

_EPSILON = 10e-7


##############################################
# UTILITY FUNCTIONS
##############################################

def _tf_filter_nans(t, replace=10e-7):
    """
    Filter NaNs from a tensor 't' and replace with value epsilon

    # Arguments
        :param t: A tensor to filter
        :param replace: Value to replace NaNs with
    # Returns
        :return: A tensor of same shape as t with NaN values replaced by epsilon.
    """

    return K.tf.where(K.tf.is_nan(t), K.tf.ones_like(t) * replace, t)


def _tf_filter_infs(t, replace=10e-7):
    """
    Filter infs from a tensor 't' and replace with value replace

    # Arguments
        :param t: A tensor to filter
        :param replace: Value to replace NaNs with
    # Returns
        :return: A tensor of same shape as t with NaN values replaced by epsilon.
    """

    return K.tf.where(K.tf.is_inf(t), K.tf.ones_like(t) * replace, t)


def _tf_filter_infinite(t, replace=10e-7):
    """
    Filters infs, ninfs and NaNs from tensor t and replaces with the given value.

    # Arguments
        :param t: tensor
        :param replace: replacement value for infinite values
    # Returns
        :return: tensor where infinite values have been replaced
    """
    return K.tf.where(K.tf.is_finite(t), t, K.tf.ones_like(t) * replace)


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


##############################################
# COST FUNCTIONS
##############################################

def _tf_unlabeled_superpixel_cost_internal(y_true_unlabeled, y_pred_unlabeled, num_classes, scale_factor):
    # Calculate the softmax of the predictions
    epsilon = _to_tensor(_EPSILON, y_pred_unlabeled.dtype.base_dtype)

    # Scale by scale factor
    scaled_size = K.tf.stop_gradient(K.tf.cast(K.tf.cast(K.tf.shape(y_pred_unlabeled)[1:-1], dtype=K.tf.float32) * scale_factor, dtype=K.tf.int32))
    y_pred_unlabeled = K.tf.image.resize_images(images=y_pred_unlabeled, size=scaled_size, method=K.tf.image.ResizeMethod.BILINEAR)
    y_true_unlabeled = K.tf.expand_dims(y_true_unlabeled, axis=-1)
    y_true_unlabeled = K.tf.image.resize_images(images=y_true_unlabeled, size=scaled_size, method=K.tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    y_true_unlabeled = K.tf.squeeze(y_true_unlabeled, axis=-1)

    # Take softmax of the unlabeled predictions
    y_pred_unlabeled_softmax = K.tf.nn.softmax(y_pred_unlabeled)

    # Calculate the gradients for the softmax output using a convolution with a Sobel mask
    Gx_kernel = K.tf.tile(K.tf.constant([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], shape=[3, 3, 1, 1], dtype=K.tf.float32), [1, 1, num_classes, 1])
    Gy_kernel = K.tf.transpose(Gx_kernel, [1, 0, 2, 3])

    Gx = K.tf.nn.depthwise_conv2d(y_pred_unlabeled_softmax, Gx_kernel, strides=[1, 1, 1, 1], padding='SAME')
    Gy = K.tf.nn.depthwise_conv2d(y_pred_unlabeled_softmax, Gy_kernel, strides=[1, 1, 1, 1], padding='SAME')

    # Calculate the gradient magnitude: sqrt(Gx^2 + Gy^2), B_SIZExHxW
    # Note: clip by epsilon to avoid NaN values due to: (small grad value)^2
    Gx = K.tf.clip_by_value(Gx, clip_value_min=epsilon, clip_value_max=K.tf.float32.max)
    Gy = K.tf.clip_by_value(Gy, clip_value_min=epsilon, clip_value_max=K.tf.float32.max)

    G_mag = K.tf.sqrt(K.tf.add(K.tf.pow(Gx, 2), K.tf.pow(Gy, 2)))
    G_mag = K.tf.reduce_mean(G_mag, axis=-1)

    # For each image - extract each superpixel and calculate the cost
    batch_entropy = K.tf.reduce_sum(G_mag * y_true_unlabeled)

    return batch_entropy

    # Take the mean over the batch images
    # Note: take the maximum between num_unlabeled and 1 to avoid dividing by 0
    #mean_image_entropy = K.tf.div(batch_entropy, K.tf.cast(K.tf.maximum(num_unlabeled, 1), K.tf.float32))

    #return K.tf.multiply(unlabeled_cost_coefficient, mean_image_entropy)


def _tf_unlabeled_superpixel_cost(y_true_unlabeled, y_pred_unlabeled, unlabeled_cost_coefficient, num_unlabeled, num_classes):
    """
    Calculates loss for a batch of unlabeled images. The function assumes that the
    ground truth labels are superpixel segmentations with superpixel areas marked as
    1 and superpixel borders marked as 0.

    # Arguments
        :param y_true_unlabeled: ground truth labels (index encoded) (dtype=int32)
        :param y_pred_unlabeled: predicted labels (index encoded) (dtype=int32)
        :param unlabeled_cost_coefficient: cost coefficient (dtype=float32)
        :param num_unlabeled: number of unlabeled images (dtype=int32)
        :param num_classes
    # Returns
        :return: the mean (image-level) unlabeled superpixel loss for the batch
    """

    # Ensure the y_true_unlabeled are binary masks with borders denoted by 0s and everything else as 1s
    y_true_unlabeled = K.tf.stop_gradient(K.tf.clip_by_value(y_true_unlabeled, clip_value_min=0, clip_value_max=1))
    y_true_unlabeled = K.tf.cast(y_true_unlabeled, dtype=K.tf.float32)

    batch_entropy_1x = _tf_unlabeled_superpixel_cost_internal(y_true_unlabeled=y_true_unlabeled,
                                                              y_pred_unlabeled=y_pred_unlabeled,
                                                              num_classes=num_classes,
                                                              scale_factor=1.0)

    batch_entropy_05x = _tf_unlabeled_superpixel_cost_internal(y_true_unlabeled=y_true_unlabeled,
                                                               y_pred_unlabeled=y_pred_unlabeled,
                                                               num_classes=num_classes,
                                                               scale_factor=0.5)

    batch_entropy_025x = _tf_unlabeled_superpixel_cost_internal(y_true_unlabeled=y_true_unlabeled,
                                                                y_pred_unlabeled=y_pred_unlabeled,
                                                                num_classes=num_classes,
                                                                scale_factor=0.25)

    batch_entropy = K.tf.div(K.tf.add(K.tf.add(batch_entropy_1x, batch_entropy_05x), batch_entropy_025x), 3.0)

    # Take the mean over the batch images
    # Note: take the maximum between num_unlabeled and 1 to avoid dividing by 0
    mean_image_entropy = K.tf.div(batch_entropy, K.tf.cast(K.tf.maximum(num_unlabeled, 1), K.tf.float32))

    return K.tf.multiply(unlabeled_cost_coefficient, mean_image_entropy)

    """
    # Calculate the softmax of the predictions
    epsilon = _to_tensor(_EPSILON, y_pred_unlabeled.dtype.base_dtype)
    y_pred_unlabeled_softmax = K.tf.nn.softmax(y_pred_unlabeled)

    # Calculate the gradients for the softmax output using a convolution with a Sobel mask
    Gx_kernel = K.tf.tile(K.tf.constant([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], shape=[3, 3, 1, 1], dtype=K.tf.float32), [1, 1, num_classes, 1])
    Gy_kernel = K.tf.transpose(Gx_kernel, [1, 0, 2, 3])

    Gx = K.tf.nn.depthwise_conv2d(y_pred_unlabeled_softmax, Gx_kernel, strides=[1, 1, 1, 1], padding='SAME')
    Gy = K.tf.nn.depthwise_conv2d(y_pred_unlabeled_softmax, Gy_kernel, strides=[1, 1, 1, 1], padding='SAME')

    # Calculate the gradient magnitude: sqrt(Gx^2 + Gy^2), B_SIZExHxW
    # Note: clip by epsilon to avoid NaN values due to: (small grad value)^2
    Gx = K.tf.clip_by_value(Gx, clip_value_min=epsilon, clip_value_max=K.tf.float32.max)
    Gy = K.tf.clip_by_value(Gy, clip_value_min=epsilon, clip_value_max=K.tf.float32.max)

    G_mag = K.tf.sqrt(K.tf.add(K.tf.pow(Gx, 2), K.tf.pow(Gy, 2)))
    G_mag = K.tf.reduce_mean(G_mag, axis=-1)

    # Ensure the y_true_unlabeled are binary masks with borders denoted by 0s and everything else as 1s
    y_true_unlabeled = K.tf.stop_gradient(K.tf.clip_by_value(y_true_unlabeled, clip_value_min=0, clip_value_max=1))

    # For each image - extract each superpixel and calculate the cost
    batch_entropy = K.tf.reduce_sum(G_mag * K.tf.cast(y_true_unlabeled, dtype=K.tf.float32))

    # Take the mean over the batch images
    # Note: take the maximum between num_unlabeled and 1 to avoid dividing by 0
    mean_image_entropy = K.tf.div(batch_entropy, K.tf.cast(K.tf.maximum(num_unlabeled, 1), K.tf.float32))

    return K.tf.multiply(unlabeled_cost_coefficient, mean_image_entropy)
    """

def _tf_mean_teacher_consistency_cost(y_pred, mt_pred, consistency_coefficient):
    """
    Calculates the consistency cost between mean teacher and student model
    predictions.

    # Arguments
        :param y_pred: student predictions from the network (logits)
        :param mt_pred: mean teacher predictions from the teacher network (logits)
        :param consistency_coefficient: the consistency coefficient for this batch
    # Returns
        :return: the consistency cost (mean for the batch)
    """

    student_softmax = K.tf.nn.softmax(y_pred, dim=-1)
    teacher_softmax = K.tf.nn.softmax(mt_pred, dim=-1)

    # Calculate the MSE between the softmax predictions
    mse_softmax = K.tf.reduce_mean(K.tf.square(K.tf.subtract(teacher_softmax, student_softmax)), axis=-1)

    # Output of the softmax is B_SIZExHxW
    # Sum the last two axes to get the total loss over all the pixels in the images per image
    mse_softmax = K.tf.reduce_sum(mse_softmax, axis=(1, 2))

    # Take the mean of the loss per image
    return K.tf.multiply(consistency_coefficient, K.tf.reduce_mean(mse_softmax))


##############################################
# LOSS FUNCTIONS
##############################################


def _weighted_pixelwise_crossentropy_loss(class_weights):
    def loss(y_true, y_pred):
        """
        Pixel-wise weighted categorical cross-entropy between an
        output tensor and a target tensor.

        # Arguments
            :param y_pred: A tensor resulting from the last convolutional layer.
            :param y_true: A tensor of the same shape as `y_pred`.
        # Returns
            :return: loss as a Tensorflow tensor.
        """
        # Calculate cross-entropy loss
        epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
        softmax = K.tf.nn.softmax(y_pred)

        # Before taking the log, make sure the softmax values are in range [epsilon, 1.0]
        softmax = K.tf.clip_by_value(softmax, epsilon, 1.0)

        xent = K.tf.multiply(y_true * K.tf.log(softmax), class_weights)
        xent = -K.tf.reduce_mean(K.tf.reduce_sum(xent, axis=(1, 2, 3)))
        return xent

    return loss


def _sparse_weighted_pixelwise_crossentropy_loss(y_true, y_pred, weights):
    """
    Calculates weighted pixelwise cross-entropy loss from sparse ground truth
    labels. All entries in labels must have values between [0, num classes).
    The shape of weights must be same as the y_true or broadcastable to the shape
    of y_true.

    # Arguments
        :param y_true: Ground truth labels (B_SIZExHxW)
        :param y_pred: Predictions from the network (unscaled logits)
        :param weights: Weights for each y_true (B_SIZExHxW)
    # Returns
        :return: The pixelwise cross-entropy loss
    """

    # Returns cross-entropy loss for each pixel, i.e. B_SIZExHxW, multiply by weights
    xent = K.tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    xent = K.tf.multiply(xent, weights)

    # Calculate the sum of pixel cross-entropies (HxW) for each image and take the mean of images in the batch
    loss = K.tf.reduce_mean(K.tf.reduce_sum(xent, axis=(1, 2)))

    return loss


def pixelwise_crossentropy_loss(class_weights):
    # type: (np.array) -> function[K.tf.Tensor, K.tf.Tensor]

    if class_weights is None:
        raise ValueError('Class weights is None. Use a numpy array of ones instead of None.')

    return _weighted_pixelwise_crossentropy_loss(K.constant(value=class_weights))


def dummy_loss(y_true, y_pred):
    # type: (K.tf.Tensor, K.tf.Tensor) -> K.tf.Tensor

    """
    A dummy loss function used in conjunction with semi-supervised models'
    loss lambda layers. Just returns the predictions which should be the output of the
    loss lambda layer.

    # Arguments
        :param y_true: doesn't really matter (same shape as y_pred)
        :param y_pred: the output of the previous loss lambda layer i.e. loss
    # Returns
        :return: y_pred
    """
    return y_pred


def _classification_weighted_categorical_crossentropy_loss(class_weights):
    def loss(y_true, y_pred):
        epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
        softmax = K.tf.nn.softmax(y_pred)
        softmax = K.tf.clip_by_value(softmax, epsilon, 1.0)
        xent = K.tf.multiply(y_true * K.tf.log(softmax), class_weights)
        xent = -K.tf.reduce_mean(K.tf.reduce_sum(xent, axis=-1))

        return xent

    return loss


def classification_weighted_categorical_crossentropy_loss(class_weights):
    # type: (np.array) -> function[K.tf.Tensor, K.tf.Tensor]

    if class_weights is None:
        raise ValueError('Class weights is None. Use a numpy array of ones instead of None.')

    return _classification_weighted_categorical_crossentropy_loss(K.constant(value=class_weights))


def _segmentation_sparse_weighted_categorical_cross_entropy(class_weights):

    def get_per_sample_weights(i, weights, y_true):
        class_weight_val = K.tf.ones_like(y_true, dtype=K.tf.float32) * class_weights[i]
        zero = K.tf.zeros_like(y_true, dtype=K.tf.float32)
        class_i_weights = K.tf.where(K.tf.equal(y_true, i), class_weight_val, zero)

        # Add the class weights to the weights array
        weights = K.tf.add(weights, class_i_weights)

        # Increase the loop variable
        i = K.tf.add(i, 1)

        return i, weights, y_true

    def loss(y_true, y_pred):
        # Sanity checks for argument ranks
        K.tf.assert_rank(y_pred, 4)
        K.tf.assert_rank(y_true, 4)

        # Squeeze the last dimension from the labels
        # It only exists so keras metrics won't whine about mismatching ranks of y_pred and y_true
        y_true = K.tf.stop_gradient(K.tf.cast(K.tf.squeeze(y_true, axis=-1), dtype=K.tf.int32))

        # Pre-process the weights, we need them separately for every y_true in the batch
        num_classes = K.tf.stop_gradient(K.tf.shape(y_pred)[-1])
        weights = K.tf.stop_gradient(K.tf.zeros_like(y_true, dtype=K.tf.float32))
        i = K.tf.stop_gradient(K.tf.constant(0, dtype=K.tf.int32))

        cond = lambda i, _, __: K.tf.less(i, num_classes)
        _, weights, __ = K.tf.while_loop(cond=cond, body=get_per_sample_weights, loop_vars=[i, weights, y_true], back_prop=False)

        # Returns cross-entropy loss for each pixel, i.e. B_SIZExHxW, multiply by weights
        xent = K.tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        xent = K.tf.multiply(xent, weights)

        # Calculate the sum of pixel cross-entropies (HxW) for each image and take the mean of images in the batch
        loss = K.tf.reduce_mean(K.tf.reduce_sum(xent, axis=(1, 2)))

        if settings.DEBUG:
            loss = K.tf.Print(loss, [loss], message="costs: ", summarize=24)

        return loss

    return loss


def segmentation_sparse_weighted_categorical_cross_entropy(class_weights):
    if class_weights is None:
        raise ValueError('Class weights is None. Use a numpy array of ones instead of None.')

    return _segmentation_sparse_weighted_categorical_cross_entropy(K.tf.constant(value=class_weights, dtype=K.tf.float32))


def segmentation_categorical_cross_entropy_lambda_loss(args):
    # type: (list[K.tf.Tensor]) -> K.tf.Tensor

    """
    Calculates the categorical cross entropy loss, assuming the
    data only contains labeled examples.

    The function is used in conjunction with a Lambda layer to create
    a layer which can calculate the loss. This is done because the
    parameters to the function change on each training step and thus
    need to be passed through the network as inputs.

    # Arguments
        :param args: a list of Tensorflow tensors, described below
            0: y_pred: predictions from the network (logits)
            1: y_true: ground truth labels in index encoded format
            2: weights: weights for every pixel in y_true labeled
            3: num_unlabeled: number of unlabeled - ignored in this loss function
    # Returns
        :return: the categorical cross entropy loss (1x1 Tensor)
    """

    # Extract arguments
    if len(args) != 4:
        raise ValueError('Expected 4 arguments (y_pred, y_true, weights, num_unlabeled), got: {} ({})'.format(len(args), args))

    y_pred, y_true, weights, _ = args

    # Sanity checks for argument ranks
    K.tf.assert_rank(y_pred, 4)
    K.tf.assert_rank(y_true, 3)
    K.tf.assert_rank(weights, K.tf.rank(y_true))

    """
    Classification cost calculation - only for labeled
    """
    classification_costs = _sparse_weighted_pixelwise_crossentropy_loss(y_true=K.tf.cast(y_true, dtype=K.tf.int32), y_pred=y_pred, weights=weights)

    if settings.DEBUG:
        classification_costs = K.tf.Print(classification_costs, [classification_costs], message="costs: ", summarize=24)

    return classification_costs


def segmentation_mean_teacher_lambda_loss(args):
    # type: (list[K.tf.Tensor]) -> K.tf.Tensor

    """
    Calculates the Mean Teacher loss function, which consists of
    classification cost and consistency cost as presented in:

        https://arxiv.org/pdf/1703.01780.pdf

    The function is used in conjunction with a Lambda layer to create
    a layer which can calculate the loss. This is done because the
    parameters to the function change on each training step and thus
    need to be passed through the network as inputs.

    # Arguments
        :param args: a list of Tensorflow tensors, described below
            0: y_pred: predictions from the network (logits)
            1: y_true: ground truth labels in index encoded format
            2: weights: weights for every pixel in y_true labeled
            3: num_unlabeled: number of unlabeled data
            4: mt_pred: mean teacher predictions from the teacher network (logits)
            5: cons_coefficient: consistency cost coefficient
    # Returns
        :return: the mean teacher loss (1x1 Tensor)
    """

    # TODO: Create option to apply/not apply consistency to labeled data
    # see: https://github.com/CuriousAI/mean-teacher/blob/master/mean_teacher/model.py#L410

    # Extract arguments
    if len(args) != 6:
        raise ValueError('Expected 6 arguments (y_pred, y_true, weights, num_unlabeled, mt_pred, cons_coefficient), got: {} ({})'.format(len(args), args))

    y_pred, y_true, weights, num_unlabeled, mt_pred, cons_coefficient = args

    # Sanity checks for argument ranks
    K.tf.assert_rank(y_pred, 4)
    K.tf.assert_rank(y_true, 3)
    K.tf.assert_rank(weights, K.tf.rank(y_true))
    K.tf.assert_rank(num_unlabeled, 2)
    K.tf.assert_rank(mt_pred, K.tf.rank(y_pred))
    K.tf.assert_rank(cons_coefficient, 2)

    # Stop gradient while parsing the necessary values
    num_unlabeled = K.tf.stop_gradient(K.tf.cast(K.tf.squeeze(num_unlabeled[0]), dtype=K.tf.int32))
    num_labeled = K.tf.stop_gradient(K.tf.shape(y_true)[0] - num_unlabeled)
    weights_labeled = K.tf.stop_gradient(weights[0:num_labeled])
    cons_coefficient = K.tf.stop_gradient(K.tf.squeeze(cons_coefficient[0]))

    y_pred_labeled = y_pred[0:num_labeled]
    y_true_labeled = K.tf.cast(y_true[0:num_labeled], dtype=K.tf.int32)

    """
    Classification cost calculation - only for labeled
    """
    classification_costs = _sparse_weighted_pixelwise_crossentropy_loss(y_true=y_true_labeled, y_pred=y_pred_labeled, weights=weights_labeled)

    """
    Consistency costs - for labeled and unlabeled
    """
    consistency_cost = K.tf.cond(cons_coefficient > 0, lambda: _tf_mean_teacher_consistency_cost(y_pred, mt_pred, cons_coefficient), lambda: 0.0)

    # Total cost
    total_costs = K.tf.add(classification_costs, consistency_cost)

    if settings.DEBUG:
        total_costs = K.tf.Print(total_costs, [consistency_cost, classification_costs, total_costs], message="costs: ", summarize=24)

    return total_costs


def segmentation_superpixel_lambda_loss(args):
    # type: (list[K.tf.Tensor]) -> K.tf.Tensor

    """
    Calculates semi-supervised segmentation loss. If the class weights are
    provided uses weighted pixelwise cross-entropy for the labeled data and
    uses unlabeled superpixel loss for the unlabeled data.

    The function assumes that the unlabeled labels are SLIC superpixel segmentations
    of their respective images.

    # Arguments
        :param args: Expects an array with 5 values

        0: y_pred: predictions from the network (logits)
        1: y_true: ground truth labels in index encoded format
        2: weights: weights for every pixel in y_true labeled
        3: num_unlabeled: number of unlabeled data
        4: unlabeled_cost_coefficient: unlabeled cost coefficient
    # Returns
        :return: the classification + superpixel loss (1x1 Tensor)
    """
    if len(args) != 5:
        raise ValueError('Expected 5 values (y_pred, y_true, weights, num_unlabeled, unlabeled_cost_coefficient), got: {} ({})'.format(len(args), args))

    y_pred, y_true, weights, num_unlabeled, unlabeled_cost_coefficient = args

    # Sanity checks for argument ranks
    K.tf.assert_rank(y_pred, 4)
    K.tf.assert_rank(y_true, 3)
    K.tf.assert_rank(weights, K.tf.rank(y_true))
    K.tf.assert_rank(num_unlabeled, 2)
    K.tf.assert_rank(unlabeled_cost_coefficient, 2)

    num_unlabeled = K.tf.stop_gradient(K.tf.cast(K.tf.squeeze(num_unlabeled[0]), dtype=K.tf.int32))
    num_labeled = K.tf.stop_gradient(K.tf.shape(y_true)[0] - num_unlabeled)
    num_classes = K.tf.stop_gradient(K.tf.shape(y_pred)[-1])
    weights_labeled = K.tf.stop_gradient(weights[0:num_labeled])
    unlabeled_cost_coefficient = K.tf.stop_gradient(K.tf.squeeze(unlabeled_cost_coefficient[0]))

    y_pred_labeled = y_pred[0:num_labeled]
    y_true_labeled = K.tf.cast(y_true[0:num_labeled], dtype=K.tf.int32)
    y_pred_unlabeled = y_pred[num_labeled:]
    y_true_unlabeled = K.tf.cast(y_true[num_labeled:], dtype=K.tf.int32)

    """
    Classification cost calculation - only for labeled
    """
    classification_costs = _sparse_weighted_pixelwise_crossentropy_loss(y_true=y_true_labeled, y_pred=y_pred_labeled, weights=weights_labeled)

    """
    Unlabeled classification costs (superpixel seg) - only for unlabeled
    """
    unlabeled_costs = K.tf.cond(unlabeled_cost_coefficient > 0.0, lambda: _tf_unlabeled_superpixel_cost(y_true_unlabeled, y_pred_unlabeled, unlabeled_cost_coefficient, num_unlabeled, num_classes), lambda: 0.0)

    # Total cost
    total_costs = K.tf.add(classification_costs, unlabeled_costs)

    if settings.DEBUG:
        total_costs = K.tf.Print(total_costs, [unlabeled_costs, classification_costs, total_costs], message="costs: ")

    return total_costs


def segmentation_mean_teacher_superpixel_lambda_loss(args):
    # type: (list[K.tf.Tensor]) -> K.tf.Tensor

    """
    Calculates the Mean Teacher loss function, which consists of
    classification cost and consistency cost as presented in.

        https://arxiv.org/pdf/1703.01780.pdf

    Additionally also uses the generated superpixel labels to calculate
    a semisupervised cost for the unlabeled.

    The function is used in conjunction with a Lambda layer to create
    a layer which can calculate the loss. This is done because the
    parameters to the function change on each training step and thus
    need to be passed through the network as inputs.

    # Arguments
        :param args: a list of Tensorflow tensors, described below
            0: y_pred: predictions from the network (logits)
            1: y_true: ground truth labels in index encoded format
            2: weights: weights for ground truth labels
            3: num_unlabeled: number of unlabeled data
            4: mt_pred: mean teacher predictions from the teacher network (logits)
            5: cons_coefficient: consistency cost coefficient
            6: unlabeled_cost_coefficient: coefficient for the unlabeled superpixel loss
    # Returns
        :return: the classification + mean teacher + superpixel loss (1x1 Tensor)
    """

    # TODO: Create option to apply/not apply consistency to labeled data
    # see: https://github.com/CuriousAI/mean-teacher/blob/master/mean_teacher/model.py#L410

    # Extract arguments
    if len(args) != 7:
        raise ValueError('Expected 7 arguments (y_pred, y_true, weights, num_unlabeled, mt_pred, cons_coefficient, unlabeled_cost_coefficient), got: {} ({})'.format(len(args), args))

    y_pred, y_true, weights, num_unlabeled, mt_pred, cons_coefficient, unlabeled_cost_coefficient = args

    # Sanity checks for argument ranks
    K.tf.assert_rank(y_pred, 4)
    K.tf.assert_rank(y_true, 3)
    K.tf.assert_rank(weights, K.tf.rank(y_true))
    K.tf.assert_rank(num_unlabeled, 2)
    K.tf.assert_rank(mt_pred, K.tf.rank(y_pred))
    K.tf.assert_rank(cons_coefficient, 2)
    K.tf.assert_rank(unlabeled_cost_coefficient, 2)

    num_unlabeled = K.tf.stop_gradient(K.tf.cast(K.tf.squeeze(num_unlabeled[0]), dtype=K.tf.int32))
    num_labeled = K.tf.stop_gradient(K.tf.shape(y_true)[0] - num_unlabeled)
    num_classes = K.tf.stop_gradient(K.tf.shape(y_pred)[-1])
    weights_labeled = K.tf.stop_gradient(weights[0:num_labeled])
    unlabeled_cost_coefficient = K.tf.stop_gradient(K.tf.squeeze(unlabeled_cost_coefficient[0]))
    cons_coefficient = K.tf.stop_gradient(K.tf.squeeze(cons_coefficient[0]))

    y_pred_labeled = y_pred[0:num_labeled]
    y_true_labeled = K.tf.cast(y_true[0:num_labeled], dtype=K.tf.int32)
    y_pred_unlabeled = y_pred[num_labeled:]
    y_true_unlabeled = K.tf.cast(y_true[num_labeled:], dtype=K.tf.int32)

    """
    Classification cost calculation - only for labeled
    """
    classification_costs = _sparse_weighted_pixelwise_crossentropy_loss(y_true=y_true_labeled, y_pred=y_pred_labeled, weights=weights_labeled)

    """
    Consistency costs - for labeled and unlabeled
    """
    consistency_costs = K.tf.cond(cons_coefficient > 0.0, lambda: _tf_mean_teacher_consistency_cost(y_pred, mt_pred, cons_coefficient), lambda: 0.0)

    """
    Unlabeled classification costs (superpixel seg) - only for unlabeled
    """
    unlabeled_costs = K.tf.cond(unlabeled_cost_coefficient > 0.0, lambda: _tf_unlabeled_superpixel_cost(y_true_unlabeled, y_pred_unlabeled, unlabeled_cost_coefficient, num_unlabeled, num_classes), lambda: 0.0)

    # Total cost
    total_costs = K.tf.add(K.tf.add(classification_costs, consistency_costs), unlabeled_costs)

    if settings.DEBUG:
        total_costs = K.tf.Print(total_costs, [consistency_costs, unlabeled_costs, classification_costs, total_costs], message="costs: ", summarize=24)

    return total_costs


def classification_categorical_cross_entropy(args):
    # type: (list[K.tf.Tensor]) -> K.tf.Tensor

    # Extract arguments
    if len(args) != 4:
        raise ValueError('Expected 4 arguments (y_pred, y_true, weights, num_unlabeled), got: {} ({})'.format(len(args), args))

    y_pred, y_true, weights, _ = args

    # Sanity checks for argument ranks
    K.tf.assert_rank(y_pred, 2)
    K.tf.assert_rank(y_true, 2)
    K.tf.assert_rank(weights, K.tf.rank(y_true))

    """
    Classification cost calculation - only for labeled
    """
    # Calculate cross-entropy loss
    epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    softmax = K.tf.nn.softmax(y_pred)

    # Before taking the log, make sure the softmax values are in range [epsilon, 1.0]
    softmax = K.tf.clip_by_value(softmax, epsilon, 1.0)

    xent = K.tf.multiply(y_true * K.tf.log(softmax), weights)
    classification_costs = -K.tf.reduce_mean(K.tf.reduce_sum(xent, axis=1))

    if settings.DEBUG:
        classification_costs = K.tf.Print(classification_costs, [classification_costs], message="costs: ", summarize=24)

    return classification_costs


def classification_mean_teacher_lambda_loss(args):
    # type: (list[K.tf.Tensor]) -> K.tf.Tensor

    """
    Calculates the Mean Teacher loss function, which consists of
    classification cost and consistency cost as presented in:

        https://arxiv.org/pdf/1703.01780.pdf

    The function is used in conjunction with a Lambda layer to create
    a layer which can calculate the loss. This is done because the
    parameters to the function change on each training step and thus
    need to be passed through the network as inputs.

    # Arguments
        :param args: a list of Tensorflow tensors, described below
            0: y_pred: predictions from the network (softmax) [B_SIZExN_CLASSES]
            1: y_true: ground truth labels in one-hot encoded format [B_SIZExN_CLASSES]
            2: weights: class weights [B_SIZExN_CLASSES]
            4: mt_pred: mean teacher predictions from the teacher network (softmax) [B_SIZExN_CLASSES]
            5: cons_coefficient: consistency cost coefficient [B_SIZExN_CLASSES]
    # Returns
        :return: the classification + mean teacher loss (1x1 Tensor)
    """

    # TODO: Create option to apply/not apply consistency to labeled data
    # see: https://github.com/CuriousAI/mean-teacher/blob/master/mean_teacher/model.py#L410

    # Extract arguments
    if len(args) != 6:
        raise ValueError('Expected 6 arguments (y_pred, y_true, weights, num_unlabeled, mt_pred, cons_coefficient), got: {} ({})'.format(len(args), args))

    y_pred, y_true, weights, num_unlabeled, mt_pred, cons_coefficient = args

    # Sanity checks for argument ranks
    K.tf.assert_rank(y_pred, 2)
    K.tf.assert_rank(y_true, 2)
    K.tf.assert_rank(cons_coefficient, 2)
    K.tf.assert_rank(num_unlabeled, 2)
    K.tf.assert_rank(mt_pred, K.tf.rank(y_pred))
    K.tf.assert_rank(weights, K.tf.rank(y_true))

    # Stop the gradient while parsing the necessary values
    num_unlabeled = K.tf.stop_gradient(K.tf.cast(K.tf.squeeze(num_unlabeled[0]), dtype=K.tf.int32))
    num_labeled = K.tf.stop_gradient(K.tf.shape(y_true)[0] - num_unlabeled)
    weights_labeled = K.tf.stop_gradient(weights[0:num_labeled])
    cons_coefficient = K.tf.stop_gradient(K.tf.squeeze(cons_coefficient[0]))

    y_pred_labeled = y_pred[0:num_labeled]
    y_true_labeled = y_true[0:num_labeled]

    """
    Classification cost calculation - only for labeled
    """
    # Calculate cross-entropy loss
    epsilon = _to_tensor(_EPSILON, y_pred_labeled.dtype.base_dtype)
    softmax = K.tf.nn.softmax(y_pred_labeled)

    # Before taking the log, make sure the softmax values are in range [epsilon, 1.0]
    softmax = K.tf.clip_by_value(softmax, epsilon, 1.0)

    xent = K.tf.multiply(y_true_labeled * K.tf.log(softmax), weights_labeled)
    classification_costs = -K.tf.reduce_mean(K.tf.reduce_sum(xent, axis=1))

    """
    Consistency costs - for labeled and unlabeled
    """
    student_softmax = K.tf.nn.softmax(y_pred, dim=-1)
    teacher_softmax = K.tf.nn.softmax(mt_pred, dim=-1)

    # Calculate the MSE between the softmax predictions
    mse_softmax = K.tf.reduce_mean(K.tf.square(K.tf.subtract(teacher_softmax, student_softmax)), axis=-1)

    # Take the mean of the loss per image
    consistency_cost = K.tf.multiply(cons_coefficient, K.tf.reduce_mean(mse_softmax))

    if settings.DEBUG:
        consistency_cost = K.tf.Print(consistency_cost, [consistency_cost, classification_costs], message="costs: ", summarize=24)

    # Total cost
    total_costs = K.tf.add(classification_costs, consistency_cost)
    return total_costs
