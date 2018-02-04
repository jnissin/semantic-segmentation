# coding=utf-8

import keras.backend as K

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


def create_reset_metric(metric, scope='reset_metrics', **metric_args):
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
        v = K.tf.contrib.framework.get_variables(scope, collection=K.tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = K.tf.variables_initializer(v)
    return value, update_op, reset_op


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


def classification_mean_per_class_accuracy(num_classes, num_unlabeled, ignore_classes=None):
    def mpca(y_true, y_pred):
        # Get flattened versions for labels, predictions and weights
        labels, predictions, weights = _preprocess_classification_data(y_true=y_true,
                                                                       y_pred=y_pred,
                                                                       num_unlabeled=num_unlabeled,
                                                                       ignore_classes=ignore_classes)

        # Calculate MPCA using Tensorflow
        value, update_op = K.tf.metrics.mean_per_class_accuracy(name='mpca',
                                                                labels=labels,
                                                                predictions=predictions,
                                                                num_classes=num_classes,
                                                                weights=weights)

        K.get_session().run(K.tf.local_variables_initializer())

        # Force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
            return value

    return mpca


def classification_accuracy(num_unlabeled, ignore_classes=None):
    def acc(y_true, y_pred):
        # Get flattened versions for labels, predictions and weights
        labels, predictions, weights = _preprocess_classification_data(y_true=y_true,
                                                                       y_pred=y_pred,
                                                                       num_unlabeled=num_unlabeled,
                                                                       ignore_classes=ignore_classes)

        # Calculate accuracy using Tensorflow
        value, update_op = K.tf.metrics.accuracy(name='accuracy',
                                                 labels=labels,
                                                 predictions=predictions,
                                                 weights=weights)

        K.get_session().run(K.tf.local_variables_initializer())

        # Force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
            return value

    return acc


def classification_confusion_matrix(num_classes, num_unlabeled, ignore_classes=None):
    def cfm(y_true, y_pred):
        # Get flattened versions for labels, predictions and weights
        labels, predictions, weights = _preprocess_classification_data(y_true=y_true,
                                                                       y_pred=y_pred,
                                                                       num_unlabeled=num_unlabeled,
                                                                       ignore_classes=ignore_classes)

        confusion_matrix = K.tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_classes, weights=weights, dtype=_CFM_DTYPE)
        return confusion_matrix

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
    def acc(y_true, y_pred):
        labels, predictions, ignore_mask, weights = _preprocess_segmentation_data(y_true=y_true,
                                                                                  y_pred=y_pred,
                                                                                  num_unlabeled=num_unlabeled,
                                                                                  ignore_classes=ignore_classes)
        # Calculate accuracy using Tensorflow
        value, update_op, reset_op = create_reset_metric(K.tf.metrics.accuracy,
                                                         'metrics_accuracy',
                                                         labels=labels,
                                                         predictions=predictions,
                                                         weights=weights)
        K.get_session().run(K.tf.local_variables_initializer())
        setattr(acc, 'reset_op', reset_op)

        # Force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
            return value

    setattr(acc, 'streaming', True)
    return acc


def segmentation_mean_per_class_accuracy(num_classes, num_unlabeled, ignore_classes=None):
    def mpca(y_true, y_pred):
        labels, predictions, ignore_mask, weights = _preprocess_segmentation_data(y_true=y_true,
                                                                                  y_pred=y_pred,
                                                                                  num_unlabeled=num_unlabeled,
                                                                                  ignore_classes=ignore_classes)

        # Calculate MPCA using Tensorflow
        value, update_op, reset_op = create_reset_metric(K.tf.metrics.mean_per_class_accuracy,
                                                         'metrics_mpca',
                                                         labels=labels,
                                                         predictions=predictions,
                                                         num_classes=num_classes,
                                                         weights=weights)
        K.get_session().run(K.tf.local_variables_initializer())
        setattr(mpca, 'reset_op', reset_op)

        # Force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
            return value

    setattr(mpca, 'streaming', True)
    return mpca


def segmentation_mean_iou(num_classes, num_unlabeled, ignore_classes=None):
    def miou(y_true, y_pred):
        labels, predictions, ignore_mask, weights = _preprocess_segmentation_data(y_true=y_true,
                                                                                  y_pred=y_pred,
                                                                                  num_unlabeled=num_unlabeled,
                                                                                  ignore_classes=ignore_classes)

        # Calculate MIoU using Tensorflow
        value, update_op, reset_op = create_reset_metric(K.tf.metrics.mean_iou,
                                                         'metrics_miou',
                                                         labels=labels,
                                                         predictions=predictions,
                                                         num_classes=num_classes,
                                                         weights=weights)
        K.get_session().run(K.tf.local_variables_initializer())
        setattr(miou, 'reset_op', reset_op)

        # Force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
            return value

    setattr(miou, 'streaming', True)
    return miou


def segmentation_confusion_matrix(num_classes, num_unlabeled, ignore_classes=None):
    def cfm(y_true, y_pred):
        labels, predictions, ignore_mask, weights = _preprocess_segmentation_data(y_true=y_true,
                                                                                  y_pred=y_pred,
                                                                                  num_unlabeled=num_unlabeled,
                                                                                  ignore_classes=ignore_classes)

        confusion_matrix = K.tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_classes, weights=weights, dtype=_CFM_DTYPE)
        return confusion_matrix

    setattr(cfm, 'hidden', True)
    return cfm
