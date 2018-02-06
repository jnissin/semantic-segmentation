# coding=utf-8

import keras.backend as K
from tensorflow.python.ops import variable_scope

_CFM_DTYPE = K.tf.int64


def _get_ignore_mask(labels, ignore_classes):
    """
    Creates a boolean mask where true marks ignored samples. Mask is the same shape
    as the labels parameter.

    # Arguments
        :param labels: labels (class encoding should match the ignore classes)
        :param ignore_classes: ignored classses as a rank 1 Tensor e.g. [0, 3, 5]
    # Returns
        :return: A boolean tensor fo same shape as labels true indicates ignored.
    """
    K.tf.assert_rank(ignore_classes, 1)

    ignore_mask = K.tf.foldl(fn=lambda a, v: a + K.tf.cast(K.tf.equal(labels, v), dtype=K.tf.int32),
                             initializer=K.tf.zeros_like(labels, dtype=K.tf.int32),
                             elems=ignore_classes)
    return K.tf.cast(ignore_mask, dtype=K.tf.bool)


def function_attributes(**kwargs):
    """
    Sets function attributes to the target function. Used as a decorator.

    # Arguments
        :param kwargs: attributes to be set
    # Returns
        :return: The function with the attributes set
    """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def _create_reset_metric(metric, scope='reset_metrics', **metric_args):
    """
    Creates a metric with limited scope and hooks to update and reset operations.

    Example:
        epoch_loss, epoch_loss_update, epoch_loss_reset = create_reset_metric(
                            tf.contrib.metrics.streaming_mean_squared_error, 'epoch_loss',
                            predictions=output, labels=target)

    # Arguments
        :param metric: the metric function
        :param scope: name of the scope
        :param metric_args: the arguments for the metric
    # Returns
        :return: A tuple with: (value, update_op, reset_op)
    """
    with K.tf.variable_scope(scope) as scope:
        value, update_op = metric(**metric_args)
        metric_vars = K.tf.contrib.framework.get_variables(scope.original_name_scope, collection=K.tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = K.tf.variables_initializer(metric_vars)
    return value, update_op, reset_op


def _create_local(name, shape, collections=None, validate_shape=True, dtype=K.tf.float32):
    """Creates a new local variable.
    Args:
        name: The name of the new or existing variable.
        shape: Shape of the new or existing variable.
        collections: A list of collection names to which the Variable will be added.
        validate_shape: Whether to validate the shape of the variable.
        dtype: Data type of the variables.
    Returns:
        The created variable.
    """
    # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
    collections = list(collections or [])
    collections += [K.tf.GraphKeys.LOCAL_VARIABLES]
    return variable_scope.variable(
        lambda: K.tf.zeros(shape, dtype=dtype),
        name=name,
        trainable=False,
        collections=collections,
        validate_shape=validate_shape)


def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None, dtype=K.tf.float64):
    """Calculate a streaming confusion matrix.
    Calculates a confusion matrix. For estimation over a stream of data,
    the function creates an  `update_op` operation.
    Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since a confusion matrix of
      dimension = [num_classes, num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    Returns:
    total_cm: A `Tensor` representing the confusion matrix.
    update_op: An operation that increments the confusion matrix.
    """
    # Local variable to accumulate the predictions in the confusion matrix.
    total_cm = _create_local(name='total_confusion_matrix', shape=[num_classes, num_classes], dtype=dtype)

    # Cast the type to int64 required by confusion_matrix_ops.
    predictions = K.tf.to_int64(predictions)
    labels = K.tf.to_int64(labels)
    num_classes = K.tf.to_int64(num_classes)

    # Flatten the input if its rank > 1.
    if predictions.get_shape().ndims > 1:
        predictions = K.tf.reshape(predictions, [-1])

    if labels.get_shape().ndims > 1:
        labels = K.tf.reshape(labels, [-1])

    if (weights is not None) and (weights.get_shape().ndims > 1):
        weights = K.tf.reshape(weights, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = K.tf.confusion_matrix(labels, predictions, num_classes, weights=weights, dtype=dtype)
    update_op = K.tf.assign_add(total_cm, current_cm)
    return total_cm, update_op


##############################################
# CLASSIFICATION METRICS
##############################################

def _preprocess_classification_data(y_true, y_pred, num_unlabeled, ignore_classes):
    K.tf.assert_rank(y_true, 2)  # BxC (one-hot encoded)
    K.tf.assert_rank(y_pred, 2)  # BxC (class probabilities)

    if ignore_classes is not None and len(ignore_classes) > 0:
        _ignore_classes = K.tf.constant(ignore_classes, dtype=K.tf.int32)
    else:
        _ignore_classes = K.tf.constant([-1], dtype=K.tf.int32)

    # Assumes that in evaluation/validation there are no unlabeled
    num_labeled = K.in_train_phase(K.tf.cast(K.tf.shape(y_true)[0] - num_unlabeled, dtype=K.tf.int32),
                                   K.tf.shape(y_true)[0])

    # Get flattened versions for labels, predictions and weights
    labels = K.tf.cast(K.tf.argmax(y_true[0:num_labeled], axis=-1), dtype=K.tf.int32)
    predictions = K.tf.cast(K.tf.argmax(y_pred[0:num_labeled], axis=-1), dtype=K.tf.int32)
    weights = K.tf.cast(K.tf.logical_not(_get_ignore_mask(labels, _ignore_classes)), dtype=K.tf.float32)

    K.tf.assert_rank(labels, 1)
    K.tf.assert_rank(predictions, 1)
    K.tf.assert_rank(weights, 1)

    return labels, predictions, weights


def classification_accuracy(num_unlabeled, ignore_classes=None):
    @function_attributes(reset_op=None, streaming=True)
    def acc(y_true, y_pred):
        # Get flattened versions for labels, predictions and weights
        labels, predictions, weights = _preprocess_classification_data(y_true=y_true,
                                                                       y_pred=y_pred,
                                                                       num_unlabeled=num_unlabeled,
                                                                       ignore_classes=ignore_classes)

        # Calculate accuracy using Tensorflow
        value, update_op, reset_op = _create_reset_metric(K.tf.metrics.accuracy,
                                                          'metrics_accuracy',
                                                          labels=labels,
                                                          predictions=predictions,
                                                          weights=weights)

        acc.reset_op = reset_op

        # Force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
        return value

    return acc


def classification_mean_per_class_accuracy(num_classes, num_unlabeled, ignore_classes=None):
    @function_attributes(reset_op=None, streaming=True)
    def mpca(y_true, y_pred):
        # Get flattened versions for labels, predictions and weights
        labels, predictions, weights = _preprocess_classification_data(y_true=y_true,
                                                                       y_pred=y_pred,
                                                                       num_unlabeled=num_unlabeled,
                                                                       ignore_classes=ignore_classes)

        # Calculate MPCA using Tensorflow
        value, update_op, reset_op = _create_reset_metric(K.tf.metrics.mean_per_class_accuracy,
                                                          'metrics_mpca',
                                                          labels=labels,
                                                          predictions=predictions,
                                                          num_classes=num_classes,
                                                          weights=weights)

        mpca.reset_op = reset_op

        # Force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)

        return value

    return mpca


def classification_confusion_matrix(num_classes, num_unlabeled, ignore_classes=None):
    @function_attributes(reset_op=None, streaming=True, hide_from_progbar=True, exclude_from_callbacks=True, cfm=True)
    def cfm(y_true, y_pred):
        # Get flattened versions for labels, predictions and weights
        labels, predictions, weights = _preprocess_classification_data(y_true=y_true,
                                                                       y_pred=y_pred,
                                                                       num_unlabeled=num_unlabeled,
                                                                       ignore_classes=ignore_classes)


        value, update_op, reset_op = _create_reset_metric(_streaming_confusion_matrix,
                                                          'metrics_cfm',
                                                          labels=labels,
                                                          predictions=predictions,
                                                          num_classes=num_classes,
                                                          weights=weights,
                                                          dtype=_CFM_DTYPE)
        cfm.reset_op = reset_op

        # Force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)

        with K.tf.control_dependencies([value]):
            return value

    return cfm

##############################################
# SEGMENTATION METRICS
##############################################


def _preprocess_segmentation_data(y_true, y_pred, num_unlabeled, ignore_classes):
    K.tf.assert_rank(y_true, 4)  # BxHxWx1
    K.tf.assert_rank(y_pred, 4)  # BxHxWxC

    if ignore_classes is not None and len(ignore_classes) > 0:
        _ignore_classes = K.tf.constant(ignore_classes, dtype=K.tf.int32)
    else:
        _ignore_classes = K.tf.constant([-1], dtype=K.tf.int32)

    # Assumes that in evaluation/validation there are no unlabeled
    num_labeled = K.in_train_phase(K.tf.cast(K.tf.shape(y_true)[0] - num_unlabeled, dtype=K.tf.int32),
                                   K.tf.shape(y_true)[0])

    labels = K.tf.cast(K.flatten(K.tf.squeeze(y_true[0:num_labeled], axis=-1)), dtype=K.tf.int32)
    predictions = K.flatten(K.tf.cast(K.tf.argmax(y_pred[0:num_labeled], axis=-1), K.tf.int32))
    ignore_mask = _get_ignore_mask(labels, _ignore_classes)
    weights = K.tf.cast(K.tf.logical_not(ignore_mask), dtype=K.tf.float32)

    K.tf.assert_rank(labels, 1)       # Flattened labels
    K.tf.assert_rank(predictions, 1)  # Flattened predictions
    K.tf.assert_rank(weights, 1)      # Flattened weights

    return labels, predictions, ignore_mask, weights


def segmentation_accuracy(num_unlabeled, ignore_classes=None):
    @function_attributes(reset_op=None, streaming=True)
    def acc(y_true, y_pred):
        labels, predictions, ignore_mask, weights = _preprocess_segmentation_data(y_true=y_true,
                                                                                  y_pred=y_pred,
                                                                                  num_unlabeled=num_unlabeled,
                                                                                  ignore_classes=ignore_classes)
        # Calculate accuracy using Tensorflow
        value, update_op, reset_op = _create_reset_metric(K.tf.metrics.accuracy,
                                                          'metrics_accuracy',
                                                          labels=labels,
                                                          predictions=predictions,
                                                          weights=weights)

        acc.reset_op = reset_op

        # Force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
        return value

    return acc


def segmentation_mean_per_class_accuracy(num_classes, num_unlabeled, ignore_classes=None):
    @function_attributes(reset_op=None, streaming=True)
    def mpca(y_true, y_pred):
        labels, predictions, ignore_mask, weights = _preprocess_segmentation_data(y_true=y_true,
                                                                                  y_pred=y_pred,
                                                                                  num_unlabeled=num_unlabeled,
                                                                                  ignore_classes=ignore_classes)

        # Calculate MPCA using Tensorflow
        value, update_op, reset_op = _create_reset_metric(K.tf.metrics.mean_per_class_accuracy,
                                                          'metrics_mpca',
                                                          labels=labels,
                                                          predictions=predictions,
                                                          num_classes=num_classes,
                                                          weights=weights)

        mpca.reset_op = reset_op

        # Force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
        return value

    return mpca


def segmentation_mean_iou(num_classes, num_unlabeled, ignore_classes=None):
    @function_attributes(reset_op=None, streaming=True)
    def miou(y_true, y_pred):
        labels, predictions, ignore_mask, weights = _preprocess_segmentation_data(y_true=y_true,
                                                                                  y_pred=y_pred,
                                                                                  num_unlabeled=num_unlabeled,
                                                                                  ignore_classes=ignore_classes)

        # Calculate MIoU using Tensorflow
        value, update_op, reset_op = _create_reset_metric(K.tf.metrics.mean_iou,
                                                          'metrics_miou',
                                                          labels=labels,
                                                          predictions=predictions,
                                                          num_classes=num_classes,
                                                          weights=weights)

        miou.reset_op = reset_op

        # Force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
        return value

    return miou


def segmentation_confusion_matrix(num_classes, num_unlabeled, ignore_classes=None):
    @function_attributes(reset_op=None, streaming=True, hide_from_progbar=True, exclude_from_callbacks=True, cfm=True)
    def cfm(y_true, y_pred):
        labels, predictions, ignore_mask, weights = _preprocess_segmentation_data(y_true=y_true,
                                                                                  y_pred=y_pred,
                                                                                  num_unlabeled=num_unlabeled,
                                                                                  ignore_classes=ignore_classes)

        value, update_op, reset_op = _create_reset_metric(_streaming_confusion_matrix,
                                                          'metrics_cfm',
                                                          labels=labels,
                                                          predictions=predictions,
                                                          num_classes=num_classes,
                                                          weights=weights,
                                                          dtype=_CFM_DTYPE)
        cfm.reset_op = reset_op

        # Force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
        return value

    return cfm
