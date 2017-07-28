# coding=utf-8

import numpy as np
from keras import backend as K

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


def _tf_log2(x):
    # Note: returns inf when x -> 0
    numerator = K.tf.log(x)
    denominator = K.tf.log(K.tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


def _tf_softmax(y_pred, epsilon=None):
    epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)

    softmax = K.tf.nn.softmax(y_pred)
    softmax = K.tf.clip_by_value(softmax, epsilon, 1. - epsilon)
    softmax = _tf_filter_nans(softmax, epsilon)

    return softmax


def _tf_calculate_superpixel_entropy(j, img_y_true_unlabeled, img_y_pred_unlabeled, num_classes, image_entropy):
    # j is the superpixel label - select only the area of the unlabeled_true
    # image where the values match the superpixel index

    # A tensor of HxW
    superpixel = K.tf.cast(K.tf.equal(img_y_true_unlabeled, j), K.tf.int32)
    num_pixels = K.tf.cast(K.tf.reduce_sum(superpixel), dtype=K.tf.float32)
    rest = K.tf.cast(K.tf.not_equal(img_y_true_unlabeled, j), K.tf.int32)


    # Copy the predictions with the superpixel 'stencil' and encode everything else
    # as num_classes index
    superpixel = K.tf.multiply(img_y_pred_unlabeled, superpixel)
    rest = K.tf.multiply(rest, num_classes)
    superpixel = K.tf.add(superpixel, rest)

    # Count the occurrences of different class predictions within the superpixel and
    # get rid of the last index, which represents the image area outside the superpixel
    superpixel = K.tf.reshape(superpixel, [-1])
    class_occurrences = K.tf.cast(K.tf.unsorted_segment_sum(K.tf.ones_like(superpixel), superpixel, num_classes+1), dtype=K.tf.float32)
    class_occurrences = class_occurrences[:-1]

    # Calculate the entropy of this superpixel
    superpixel_entropy = K.tf.div(class_occurrences, num_pixels)

    # Log2 can easily have NaNs and infs when class frequencies are zero, filter them out
    log2_entropy = _tf_log2(superpixel_entropy)
    log2_entropy = _tf_filter_infinite(log2_entropy, 0)

    # Super pixel entropy can get quite small when multiplying, filter NaNs and replace
    # NaN values with epsilon
    superpixel_entropy = K.tf.multiply(superpixel_entropy, log2_entropy)
    superpixel_entropy = _tf_filter_infinite(superpixel_entropy, _EPSILON)
    #superpixel_entropy = K.tf.Print(superpixel_entropy, [j, num_pixels, class_occurrences, superpixel_entropy, log2_entropy], message="spixel: ", summarize=24)
    superpixel_entropy = -1 * K.tf.reduce_sum(superpixel_entropy)

    # Add to the image entropy accumulator and increase the loop variable
    image_entropy += superpixel_entropy
    j += 1

    return j, img_y_true_unlabeled, img_y_pred_unlabeled, num_classes, image_entropy


def _tf_calculate_image_entropy(i, y_true_unlabeled, y_pred_unlabeled, num_classes, batch_entropy):

    # Superpixel segment indices are continuous and start from index 0
    num_superpixels = K.tf.reduce_max(y_true_unlabeled[i])
    for_each_superpixel_cond = lambda j, p1, p2, p3, p4: K.tf.less(j, num_superpixels)
    j = K.tf.constant(0, dtype=K.tf.int32)
    image_entropy = K.tf.constant(0, dtype=K.tf.float32)

    img_y_true_unlabeled = y_true_unlabeled[i]
    img_y_pred_unlabeled = y_pred_unlabeled[i]

    #img_y_pred_unlabeled = K.tf.Print(img_y_pred_unlabeled, [i, img_y_pred_unlabeled], message="img y pred: ", summarize=100)
    #img_y_true_unlabeled = K.tf.Print(img_y_true_unlabeled, [i, img_y_true_unlabeled], message="img y true: ", summarize=100)

    j, _, _, _, image_entropy = \
        K.tf.while_loop(for_each_superpixel_cond, _tf_calculate_superpixel_entropy,
                        [j, img_y_true_unlabeled, img_y_pred_unlabeled, num_classes, image_entropy])

    # Add the image entropy to the batch entropy and increase the loop variable
    batch_entropy += image_entropy
    i += 1

    return i, y_true_unlabeled, y_pred_unlabeled, num_classes, batch_entropy


def _tf_unlabeled_superpixel_loss(y_true_unlabeled, y_pred_unlabeled, num_unlabeled, num_classes):
    """
    Calculates loss for a batch of unlabeled images. The function assumes that the
    ground truth labels are SLIC superpixel segmentations with index encoded superpixel
    boundaries.

    # Arguments
        :param y_true_unlabeled: ground truth labels (index encoded) (dtype=int32)
        :param y_pred_unlabeled: predicted labels (index encoded) (dtype=int32)
        :param num_unlabeled: number of unlabeled images (dtype=int32)
        :param num_classes: number of classes (dtype=int32)
    # Returns
        :return: the mean (image-level) unlabeled superpixel loss for the batch
    """

    for_each_unlabeled_image_cond = lambda i, p1, p2, p3, p4: K.tf.less(i, num_unlabeled)
    i = K.tf.constant(0, dtype=K.tf.int32)
    batch_entropy = K.tf.constant(0, dtype=K.tf.float32)

    i, _, _, _, batch_entropy =\
        K.tf.while_loop(for_each_unlabeled_image_cond, _tf_calculate_image_entropy,
                        [i, y_true_unlabeled, y_pred_unlabeled, num_classes, batch_entropy])

    # Take the mean over the batch images
    mean_image_entropy = K.tf.cond(num_unlabeled > 0,
                                   lambda: K.tf.div(batch_entropy, K.tf.cast(num_unlabeled, K.tf.float32)),
                                   lambda: 0.0)

    return mean_image_entropy


def _tf_mean_teacher_consistency_cost(y_pred, mt_pred, cons_coefficient):
    """
    Calculates the consistency cost between mean teacher and student model
    predictions.

    # Arguments
        :param y_pred: student predictions from the network (logits)
        :param mt_pred: mean teacher predictions from the teacher network (logits)
        :param cons_coefficient: consistency coefficient
    # Returns
        :return: the consistency cost (mean for the batch)
    """

    student_softmax = _tf_softmax(y_pred)
    teacher_softmax = _tf_softmax(mt_pred)

    # Calculate the L2 distance between the predictions (softmax)
    l2_softmax_dist = (student_softmax - teacher_softmax) ** 2

    # Output of the softmax is B_SIZExHxWxN_CLASSES
    # Sum the last three axes to get the total loss over images
    l2_softmax_dist = K.tf.reduce_sum(l2_softmax_dist, axis=(1, 2, 3))

    # Take the mean of the loss per image and multiply by the consistency coefficient
    consistency_cost = K.tf.reduce_mean(l2_softmax_dist) * cons_coefficient[0]
    return consistency_cost


##############################################
# ACTIVATION FUNCTIONS
##############################################

def _depth_softmax(matrix):
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
# LOSS FUNCTIONS
##############################################

def _pixelwise_crossentropy_loss(y_true, y_pred):
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

    # Cross-entropy is calculated for each pixel i.e. the xent shape is
    # B_SIZExHxW - calculate the sums for each image and take the mean for the
    # batch
    xent = K.tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels)
    xent = K.tf.reduce_sum(xent, axis=(1, 2))

    return K.tf.reduce_mean(xent)


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
        softmax = _tf_softmax(y_pred, epsilon)

        epsilon = _to_tensor(_EPSILON, softmax.dtype.base_dtype)
        xent = K.tf.multiply(y_true * K.tf.log(softmax), class_weights)
        xent = _tf_filter_nans(xent, epsilon)
        xent = -K.tf.reduce_mean(K.tf.reduce_sum(xent, axis=(1, 2, 3)))
        return xent

    return loss


def pixelwise_crossentropy_loss(class_weights=None):
    if class_weights is None:
        return _pixelwise_crossentropy_loss
    else:
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


def mean_teacher_lambda_loss(class_weights=None):
    def loss(args):
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
                2: num_unlabeled: number of unlabeled data
                3: mt_pred: mean teacher predictions from the teacher network (logits)
                4: cons_coefficient: consistency cost coefficient
        # Returns
            :return: the mean teacher loss (1x1 Tensor)
        """

        # TODO: Create option to apply/not apply consistency to labeled data
        # see: https://github.com/CuriousAI/mean-teacher/blob/master/mean_teacher/model.py#L410

        # Extract arguments
        if len(args) != 5:
            raise ValueError('Expected 5 arguments (y_pred, y_true, num_unlabeled, mt_pred, cons_coefficient), '
                             'got: {} ({})'.format(len(args), args))

        y_pred, y_true, num_unlabeled, mt_pred, cons_coefficient = args

        num_labeled = K.tf.squeeze(K.tf.subtract(K.tf.shape(y_pred)[0], K.tf.to_int32(num_unlabeled[0])))
        y_pred_labeled = y_pred[0:num_labeled]
        y_true_labeled = y_true[0:num_labeled]
        #y_pred_unlabeled = y_pred[num_unlabeled:]
        #y_true_unlabeled = y_true[num_unlabeled:]

        """
        Classification cost calculation - only for labeled
        """

        classification_costs = None

        if class_weights is not None:
            # Weighted pixelwise cross-entropy
            # The labels are index encoded - expand to one hot encoding for weighted_pixelwise_crossentropy calculation
            y_true_labeled = K.tf.cast(y_true_labeled, K.tf.int32)
            y_true_labeled = K.tf.one_hot(y_true_labeled, y_pred.shape[-1])
            classification_costs = _weighted_pixelwise_crossentropy_loss(class_weights)(y_true_labeled, y_pred_labeled)
        else:
            # Pixelwise cross-entropy
            y_true_labeled = K.tf.cast(y_true_labeled, K.tf.int32)
            xent = K.tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred_labeled, labels=y_true_labeled)
            # Returns cross-entropy loss for each pixel, i.e. B_SIZExHxW
            # calculate the sum of pixel cross-entropies for each image and take the mean of images in the batch
            xent = K.tf.reduce_sum(xent, axis=(1, 2))
            classification_costs = K.tf.reduce_mean(xent)

        """
        Consistency costs - for labeled and unlabeled
        """
        consistency_cost = _tf_mean_teacher_consistency_cost(y_pred, mt_pred, cons_coefficient[0])

        # Total cost
        total_costs = classification_costs + consistency_cost
        return total_costs

    return loss


def semisupervised_superpixel_lambda_loss(class_weights=None):
    def loss(args):
        """
        Calculates semi-supervised segmentation loss. If the class weights are
        provided uses weighted pixelwise cross-entropy for the labeled data and
        uses unlabeled superpixel loss for the unlabeled data.

        The function assumes that the unlabeled labels are SLIC superpixel segmentations
        of their respective images.

        # Arguments
            :param args: Expects an array with 4 values

            0: y_pred: predictions from the network (logits)
            1: y_true: ground truth labels in index encoded format
            2: num_unlabeled: number of unlabeled data
            3: unlabeled_cost_coefficient: unlabeled cost coefficient
        # Returns
            :return: the semi-supervised segmentation loss (1x1 Tensor)
        """
        if len(args) != 4:
            raise ValueError('Expected 4 values (y_pred, y_true, num_unlabeled, unlabeled_cost_coefficient), '
                             'got: {} ({})'.format(len(args), args))

        y_pred, y_true, num_unlabeled, unlabeled_cost_coefficient = args

        int_num_unlabeled = K.tf.squeeze(K.tf.to_int32(num_unlabeled[0]))
        int_num_labeled = K.tf.squeeze(K.tf.subtract(K.tf.shape(y_pred)[0], K.tf.to_int32(num_unlabeled[0])))
        int_num_classes = K.tf.shape(y_pred)[-1]

        y_pred_labeled = y_pred[0:int_num_labeled]
        y_true_labeled = y_true[0:int_num_labeled]

        """
        Labeled loss - pixelwise cross-entropy
        """
        labeled_loss = 0

        # Pixelwise cross-entropy
        if class_weights is not None:
            # Weighted pixelwise cross-entropy
            # The labels are index encoded - expand to one hot encoding for weighted_pixelwise_crossentropy calculation
            y_true_labeled = K.tf.cast(y_true_labeled, K.tf.int32)
            y_true_labeled = K.tf.one_hot(y_true_labeled, y_pred.shape[-1])
            labeled_loss = _weighted_pixelwise_crossentropy_loss(class_weights)(y_true_labeled, y_pred_labeled)
        else:
            y_true_labeled = K.tf.cast(y_true_labeled, K.tf.int32)
            xent = K.tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred_labeled, labels=y_true_labeled)
            # Returns cross-entropy loss for each pixel, i.e. B_SIZExHxW
            # calculate the sum of pixel cross-entropies for each image and take the mean of images in the batch
            xent = K.tf.reduce_sum(xent, axis=(1, 2))
            labeled_loss = K.tf.reduce_mean(xent)

        """
        Unlabeled superpixel loss - assumes the y_true_unlabeled have been made using SLIC
        """
        y_pred_unlabeled = K.tf.cast(K.tf.argmax(y_pred[int_num_labeled:], axis=-1), dtype=K.tf.int32)
        y_true_unlabeled = K.tf.cast(y_true[int_num_labeled:], dtype=K.tf.int32)
        unlabeled_loss = _tf_unlabeled_superpixel_loss(y_true_unlabeled, y_pred_unlabeled, int_num_unlabeled, int_num_classes)

        #unlabeled_loss = K.tf.Print(unlabeled_loss, [unlabeled_loss], message="Unlabeled loss: ")
        #labeled_loss = K.tf.Print(labeled_loss, [labeled_loss], message="Labeled loss: ")
        return labeled_loss + unlabeled_cost_coefficient[0] * unlabeled_loss

    return loss


def mean_teacher_superpixel_lambda_loss(class_weights=None):
    def loss(args):
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
                2: num_unlabeled: number of unlabeled data
                3: mt_pred: mean teacher predictions from the teacher network (logits)
                4: cons_coefficient: consistency cost coefficient
                5: unlabeled_cost_coefficient: coefficient for the unlabeled superpixel loss
        # Returns
            :return: the mean teacher loss (1x1 Tensor)
        """

        # TODO: Create option to apply/not apply consistency to labeled data
        # see: https://github.com/CuriousAI/mean-teacher/blob/master/mean_teacher/model.py#L410

        # Extract arguments
        if len(args) != 6:
            raise ValueError('Expected 6 arguments (y_pred, y_true, num_unlabeled, mt_pred, cons_coefficient, unlabeled_cost_coefficient), '
                             'got: {} ({})'.format(len(args), args))

        y_pred, y_true, num_unlabeled, mt_pred, cons_coefficient, unlabeled_cost_coefficient = args

        int_num_unlabeled = K.tf.squeeze(K.tf.to_int32(num_unlabeled[0]))
        int_num_labeled = K.tf.squeeze(K.tf.subtract(K.tf.shape(y_pred)[0], K.tf.to_int32(num_unlabeled[0])))
        int_num_classes = K.tf.shape(y_pred)[-1]

        y_pred_labeled = y_pred[0:int_num_labeled]
        y_true_labeled = y_true[0:int_num_labeled]
        y_pred_unlabeled = K.tf.cast(K.tf.argmax(y_pred[int_num_labeled:], axis=-1), dtype=K.tf.int32)
        y_true_unlabeled = K.tf.cast(y_true[int_num_labeled:], dtype=K.tf.int32)

        """
        Classification cost calculation - only for labeled
        """
        classification_costs = None

        if class_weights is not None:
            # Weighted pixelwise cross-entropy
            # The labels are index encoded - expand to one hot encoding for weighted_pixelwise_crossentropy calculation
            y_true_labeled = K.tf.cast(y_true_labeled, K.tf.int32)
            y_true_labeled = K.tf.one_hot(y_true_labeled, y_pred.shape[-1])
            classification_costs = _weighted_pixelwise_crossentropy_loss(class_weights)(y_true_labeled, y_pred_labeled)
        else:
            # Pixelwise cross-entropy
            y_true_labeled = K.tf.cast(y_true_labeled, K.tf.int32)
            xent = K.tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred_labeled, labels=y_true_labeled)
            # Returns cross-entropy loss for each pixel, i.e. B_SIZExHxW
            # calculate the sum of pixel cross-entropies for each image and take the mean of images in the batch
            xent = K.tf.reduce_sum(xent, axis=(1, 2))
            classification_costs = K.tf.reduce_mean(xent)

        #classification_costs = K.tf.Print(classification_costs, [classification_costs], message="clasf loss: ", summarize=24)

        """
        Unlabeled classification costs (superpixel seg) - only for unlabeled
        """
        unlabeled_classification_cost = _tf_unlabeled_superpixel_loss(y_true_unlabeled, y_pred_unlabeled, int_num_unlabeled, int_num_classes)
        unlabeled_classification_cost = unlabeled_cost_coefficient[0] * unlabeled_classification_cost
        #unlabeled_classification_cost = K.tf.Print(unlabeled_classification_cost, [unlabeled_classification_cost], message="spixel loss: ", summarize=24)

        """
        Consistency costs - for labeled and unlabeled
        """
        consistency_cost = _tf_mean_teacher_consistency_cost(y_pred, mt_pred, cons_coefficient[0])
        #consistency_cost = K.tf.Print(consistency_cost, [consistency_cost], message="mt loss: ", summarize=24)

        # Total cost
        total_costs = classification_costs + unlabeled_classification_cost + consistency_cost
        return total_costs

    return loss
