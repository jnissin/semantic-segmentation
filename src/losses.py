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
    return K.tf.div(numerator, denominator)


def _tf_softmax(y_pred):
    #epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)

    softmax = K.tf.nn.softmax(y_pred)
    #softmax = K.tf.clip_by_value(softmax, epsilon, 1. - epsilon)
    #softmax = _tf_filter_nans(softmax, epsilon)

    return softmax


def _tf_sobel(img):
    """
    Applies a Sobel filter to the parameter image.

    # Arguments
        :param img: Image tensor
    # Returns
        :return: Gx, Gy
    """

    sobel_x = K.tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], K.tf.float32)
    sobel_x_filter = K.tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = K.tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    # If the image is of form: HxW -> reshape to 1xHxWx1
    img_reshaped = K.tf.cond(img.rank == 2, lambda: K.tf.cast(K.tf.expand_dims(K.tf.expand_dims(img, 0), 3), K.tf.float32), img)
    # If the image is of form: HxWx1 -> reshape to 1xHxWx1
    img_reshaped = K.tf.cond(img.rank == 3, lambda: K.tf.cast(K.tf.expand_dims(img, 0), K.tf.float32), img_reshaped)

    filtered_x = K.tf.nn.conv2d(img_reshaped, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    filtered_x = K.tf.squeeze(filtered_x)
    filtered_y = K.tf.nn.conv2d(img_reshaped, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
    filtered_y = K.tf.squeeze(filtered_y)

    return filtered_x, filtered_y


def _tf_calculate_superpixel_entropy_with_gradients(j, img_y_true_unlabeled, img_y_pred_unlabeled, num_classes, image_entropy):
    # A tensor of HxW
    f_dtype = K.tf.float32

    superpixel_mask = K.tf.cast(K.tf.equal(img_y_true_unlabeled, j), dtype=f_dtype)
    #num_pixels = K.tf.count_nonzero(superpixel_mask, dtype=K.tf.float32)

    # Create the superpixel prediction mask
    superpixel_predictions = K.tf.multiply(superpixel_mask, K.tf.cast(img_y_pred_unlabeled, dtype=f_dtype))

    # Calculate the 2D, X and Y, gradient
    gx, gy = _tf_sobel(superpixel_predictions)
    gx = K.tf.multiply(superpixel_mask, K.tf.cast(gx, dtype=f_dtype))
    gy = K.tf.multiply(superpixel_mask, K.tf.cast(gy, dtype=f_dtype))
    g_mag_squared = K.tf.add(K.tf.square(gx), K.tf.square(gy))
    g_mag_squared = K.tf.reduce_sum(K.tf.sqrt(g_mag_squared))
    #g_mag_squared = K.tf.Print(g_mag_squared, [g_mag_squared], message='g_mag: ', summarize=24)

    # Add to the image entropy accumulator and increase the loop variable
    image_entropy = K.tf.add(image_entropy, g_mag_squared)
    image_entropy.set_shape(K.tf.TensorShape([]))
    j = K.tf.add(j, 1)

    return j, img_y_true_unlabeled, img_y_pred_unlabeled, num_classes, image_entropy


def _tf_calculate_superpixel_entropy(j, img_y_true_unlabeled, img_y_pred_unlabeled, num_classes, image_entropy):
    # j is the superpixel label - select only the area of the unlabeled_true
    # image where the values match the superpixel index

    # A tensor of HxW
    superpixel_mask = K.tf.cast(K.tf.equal(img_y_true_unlabeled, j), dtype=K.tf.uint8)
    num_pixels = K.tf.count_nonzero(superpixel_mask, dtype=K.tf.float32)

    # Count the frequencies of different class predictions within the superpixel and
    # get rid of the last index, which represents the image area outside the superpixel
    superpixel_mask = K.tf.reshape(superpixel_mask, [-1])
    segment_ids = K.tf.reshape(img_y_pred_unlabeled, [-1])
    class_frequencies = K.tf.cast(K.tf.unsorted_segment_sum(data=superpixel_mask, segment_ids=segment_ids, num_segments=num_classes), K.tf.float32)

    # Calculate the entropy of this superpixel
    # Log2 can easily have NaNs and infs when class frequencies are zero, filter them out
    p_k = K.tf.div(class_frequencies, num_pixels)
    log2_p_k = _tf_filter_infinite(_tf_log2(p_k), 0) # TODO: Get rid of this, instead replace zero frequencies with value 1 -> log2(1) == 0

    # Super pixel entropy can get quite small when multiplying, filter NaNs and replace
    # NaN values with epsilon
    superpixel_entropy = K.tf.multiply(p_k, log2_p_k)
    superpixel_entropy = _tf_filter_infinite(superpixel_entropy, _EPSILON) # TODO: Get rid of this, somehow? 10e-8*10e-8 is something very small.. Does it matter?
    superpixel_entropy = K.tf.multiply(-1.0, K.tf.reduce_sum(superpixel_entropy))

    # Add to the image entropy accumulator and increase the loop variable
    image_entropy = K.tf.add(image_entropy, superpixel_entropy)
    j = K.tf.add(j, 1)

    return j, img_y_true_unlabeled, img_y_pred_unlabeled, num_classes, image_entropy


def _tf_calculate_image_entropy(i, img_y_true_unlabeled, img_y_pred_unlabeled, num_classes, batch_entropy):
    """
    Calculates the superpixel entropy for one (unlabeled) image.

    # Arguments
        :param i: loop index
        :param img_y_true_unlabeled: index encoded superpixel image, HxW
        :param img_y_pred_unlabeled: index encoded predictions from the network for the image, HxW
        :param num_classes: number of classes
        :param batch_entropy: batch entropy accumulator
    # Returns
        :return: loop index, y true unlabeled, num classes, increased batch entropy
    """

    # Superpixel segment indices are continuous and start from index 0
    num_superpixels = K.tf.reduce_max(img_y_true_unlabeled)+1
    for_each_superpixel_cond = lambda j, p1, p2, p3, p4: K.tf.less(j, num_superpixels)
    j = K.tf.constant(0, dtype=K.tf.int32)
    image_entropy = K.tf.constant(0, dtype=K.tf.float32)

    j, _, _, _, image_entropy = \
        K.tf.while_loop(cond=for_each_superpixel_cond,
                        body=_tf_calculate_superpixel_entropy,
                        loop_vars=[j, img_y_true_unlabeled, img_y_pred_unlabeled, num_classes, image_entropy])

    # Add the image entropy to the batch entropy and increase the loop variable
    batch_entropy = K.tf.add(batch_entropy, image_entropy)
    i = K.tf.add(i, 1)

    return i, img_y_true_unlabeled, img_y_pred_unlabeled, num_classes, batch_entropy


def _tf_unlabeled_superpixel_loss(y_true_unlabeled, y_pred_unlabeled, unlabeled_cost_coefficient, num_unlabeled, num_classes):
    """
    Calculates loss for a batch of unlabeled images. The function assumes that the
    ground truth labels are SLIC superpixel segmentations with index encoded superpixel
    boundaries.

    # Arguments
        :param y_true_unlabeled: ground truth labels (index encoded) (dtype=int32)
        :param y_pred_unlabeled: predicted labels (index encoded) (dtype=int32)
        :param unlabeled_cost_coefficient: cost coefficient (dtype=float32)
        :param num_unlabeled: number of unlabeled images (dtype=int32)
        :param num_classes: number of classes (dtype=int32)
    # Returns
        :return: the mean (image-level) unlabeled superpixel loss for the batch
    """

    def _tf_unlabeled_superpixel_loss_internal(_y_true_unlabeled, _y_pred_unlabeled, _num_unlabeled, _num_classes):
        for_each_unlabeled_image_cond = lambda i, p1, p2, p3, p4: K.tf.less(i, _num_unlabeled)
        i = K.tf.constant(0, dtype=K.tf.int32)
        batch_entropy = K.tf.constant(0, dtype=K.tf.float32)

        i, _, _, _, batch_entropy =\
            K.tf.while_loop(cond=for_each_unlabeled_image_cond,
                            body=_tf_calculate_image_entropy,
                            loop_vars=[i, _y_true_unlabeled[i], _y_pred_unlabeled[i], _num_classes, batch_entropy])

        # Take the mean over the batch images
        return K.tf.div(batch_entropy, K.tf.cast(_num_unlabeled, K.tf.float32))

    mean_image_entropy = K.tf.cond(num_unlabeled > 0,
                                   lambda: _tf_unlabeled_superpixel_loss_internal(y_true_unlabeled, y_pred_unlabeled, num_unlabeled, num_classes),
                                   lambda: 0.0)

    return K.tf.multiply(unlabeled_cost_coefficient, mean_image_entropy)


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

    # Calculate the L2 distance between the predictions (softmax)
    l2_softmax_dist = K.tf.reduce_mean((teacher_softmax - student_softmax) ** 2, axis=-1)

    # Output of the softmax is B_SIZExHxW
    # Sum the last two axes to get the total loss over images
    l2_softmax_dist = K.tf.reduce_sum(l2_softmax_dist, axis=(1, 2))

    # Take the mean of the loss per image
    return K.tf.multiply(consistency_coefficient, K.tf.reduce_mean(l2_softmax_dist))


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
        softmax = _tf_softmax(y_pred)

        xent = K.tf.multiply(y_true * K.tf.log(softmax), class_weights)
        xent = _tf_filter_nans(xent, epsilon)
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
    # Calculate the sum of pixel cross-entropies for each image and take the mean of images in the batch
    loss = K.tf.reduce_mean(K.tf.reduce_sum(xent, axis=(1, 2)))

    return loss


def pixelwise_crossentropy_loss(class_weights):
    # type: (np.array) -> Callable[K.tf.Tensor, K.tf.Tensor]

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


def mean_teacher_lambda_loss(args):
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
    consistency_cost = _tf_mean_teacher_consistency_cost(y_pred, mt_pred, cons_coefficient)
    consistency_cost = K.tf.Print(consistency_cost, [consistency_cost, classification_costs], message="costs: ", summarize=24)

    # Total cost
    total_costs = K.tf.add(classification_costs, consistency_cost)
    return total_costs


def semisupervised_superpixel_lambda_loss(num_classes):

    tf_num_classes = K.tf.constant(num_classes, dtype=K.tf.int32)

    def loss(args):
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
            :return: the semi-supervised segmentation loss (1x1 Tensor)
        """
        if len(args) != 5:
            raise ValueError('Expected 5 values (y_pred, y_true, weights, num_unlabeled, unlabeled_cost_coefficient), got: {} ({})'.format(len(args), args))

        y_pred, y_true, weights, num_unlabeled, unlabeled_cost_coefficient = args

        num_unlabeled = K.tf.stop_gradient(K.tf.cast(K.tf.squeeze(num_unlabeled[0]), dtype=K.tf.int32))
        num_labeled = K.tf.stop_gradient(K.tf.shape(y_true)[0] - num_unlabeled)
        weights_labeled = K.tf.stop_gradient(weights[0:num_labeled])
        unlabeled_cost_coefficient = K.tf.stop_gradient(K.tf.squeeze(unlabeled_cost_coefficient[0]))

        y_pred_labeled = y_pred[0:num_labeled]
        y_true_labeled = K.tf.cast(y_true[:num_labeled], dtype=K.tf.int32)
        y_pred_unlabeled = K.tf.cast(K.tf.argmax(y_pred[num_labeled:], axis=-1), dtype=K.tf.int32)
        y_true_unlabeled = K.tf.cast(y_true[num_labeled:], dtype=K.tf.int32)

        """
        Classification cost calculation - only for labeled
        """
        classification_costs = _sparse_weighted_pixelwise_crossentropy_loss(y_true=y_true_labeled, y_pred=y_pred_labeled, weights=weights_labeled)

        """
        Unlabeled classification costs (superpixel seg) - only for unlabeled
        """
        unlabeled_costs = _tf_unlabeled_superpixel_loss(y_true_unlabeled, y_pred_unlabeled, unlabeled_cost_coefficient, num_unlabeled, tf_num_classes)
        unlabeled_costs = K.tf.Print(unlabeled_costs, [unlabeled_costs, classification_costs], message="costs: ")

        # Total cost
        total_costs = K.tf.add(classification_costs, unlabeled_costs)
        return total_costs

    return loss


def mean_teacher_superpixel_lambda_loss(num_classes):

    tf_num_classes = K.tf.constant(num_classes, dtype=K.tf.int32)

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
                2: weights: weights for ground truth labels
                3: num_unlabeled: number of unlabeled data
                4: mt_pred: mean teacher predictions from the teacher network (logits)
                5: cons_coefficient: consistency cost coefficient
                6: unlabeled_cost_coefficient: coefficient for the unlabeled superpixel loss
        # Returns
            :return: the mean teacher loss (1x1 Tensor)
        """

        # TODO: Create option to apply/not apply consistency to labeled data
        # see: https://github.com/CuriousAI/mean-teacher/blob/master/mean_teacher/model.py#L410

        # Extract arguments
        if len(args) != 7:
            raise ValueError('Expected 7 arguments (y_pred, y_true, weights, num_unlabeled, mt_pred, cons_coefficient, unlabeled_cost_coefficient), got: {} ({})'.format(len(args), args))

        y_pred, y_true, weights, num_unlabeled, mt_pred, cons_coefficient, unlabeled_cost_coefficient = args

        num_unlabeled = K.tf.stop_gradient(K.tf.cast(K.tf.squeeze(num_unlabeled[0]), dtype=K.tf.int32))
        num_labeled = K.tf.stop_gradient(K.tf.shape(y_true)[0] - num_unlabeled)
        weights_labeled = K.tf.stop_gradient(weights[0:num_labeled])
        unlabeled_cost_coefficient = K.tf.stop_gradient(K.tf.squeeze(unlabeled_cost_coefficient[0]))
        cons_coefficient = K.tf.stop_gradient(K.tf.squeeze(cons_coefficient[0]))

        y_pred_labeled = y_pred[:num_labeled]
        y_true_labeled = K.tf.cast(y_true[:num_labeled], dtype=K.tf.int32)
        y_pred_unlabeled = K.tf.cast(K.tf.argmax(y_pred[num_labeled:], axis=-1), dtype=K.tf.int32)
        y_true_unlabeled = K.tf.cast(y_true[num_labeled:], dtype=K.tf.int32)


        """
        Classification cost calculation - only for labeled
        """
        classification_costs = _sparse_weighted_pixelwise_crossentropy_loss(y_true=y_true_labeled, y_pred=y_pred_labeled, weights=weights_labeled)

        """
        Consistency costs - for labeled and unlabeled
        """
        consistency_costs = _tf_mean_teacher_consistency_cost(y_pred, mt_pred, cons_coefficient)

        """
        Unlabeled classification costs (superpixel seg) - only for unlabeled
        """
        unlabeled_costs = _tf_unlabeled_superpixel_loss(y_true_unlabeled, y_pred_unlabeled, unlabeled_cost_coefficient, num_unlabeled, tf_num_classes)
        unlabeled_costs = K.tf.Print(unlabeled_costs, [consistency_costs, unlabeled_costs, classification_costs], message="costs: ", summarize=24)

        # Total cost
        total_costs = K.tf.add(K.tf.add(classification_costs, consistency_costs), unlabeled_costs)
        return total_costs

    return loss
