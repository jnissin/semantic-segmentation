# coding=utf-8

import keras.backend as K

##############################################
# METRICS
##############################################


def _miou(labels, predictions, num_classes, weights=None):
    cfm = K.tf.confusion_matrix(
        labels=labels, predictions=predictions, num_classes=num_classes, weights=weights)

    # Compute the mean intersection-over-union via the confusion matrix.
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


def _mpca(labels, predictions, num_classes, weights=None):
    cfm = K.tf.confusion_matrix(
        labels=labels, predictions=predictions, num_classes=num_classes, weights=weights)

    # Compute the mean per class accuracy via the confusion matrix.

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


def mean_per_class_accuracy(num_classes, ignore_class=-1):

    def mpca(y_true, y_pred):
        """
         Calculates the accuracy for each class, then takes the mean of that.

         # Arguments
             y_true: ground truth classification BATCH_SIZExHxWxNUM_CLASSES
             y_predicted: predicted classification BATCH_SIZExHxWxNUM_CLASSES

         # Returns
             The mean class accuracy.
         """

        labels = K.flatten(K.tf.cast(K.tf.argmax(y_true, axis=-1), K.tf.int32))
        predictions = K.flatten(K.tf.cast(K.tf.argmax(y_pred, axis=-1), K.tf.int32))
        weights = K.tf.where(K.tf.equal(predictions, ignore_class),
                             K.tf.zeros_like(predictions, dtype=K.tf.float32),
                             K.tf.ones_like(predictions, dtype=K.tf.float32))

        return _mpca(labels=labels, predictions=predictions, num_classes=num_classes)

    return mpca


def mean_iou(num_classes, ignore_class=-1):

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

        labels = K.flatten(K.tf.cast(K.tf.argmax(y_true, axis=-1), dtype=K.tf.int32))
        predictions = K.flatten(K.tf.cast(K.tf.argmax(y_pred, axis=-1), dtype=K.tf.int32))
        weights = K.tf.where(K.tf.equal(predictions, ignore_class),
                             K.tf.zeros_like(predictions, dtype=K.tf.float32),
                             K.tf.ones_like(predictions, dtype=K.tf.float32))

        return _miou(labels=labels, predictions=predictions, num_classes=num_classes)

    return miou


def semisupervised_mean_per_class_accuracy(num_classes, num_unlabeled, ignore_class=-1):

    def mpca(y_true, y_pred):
        num_labeled = K.tf.cast(K.tf.shape(y_true)[0] - num_unlabeled, dtype=K.tf.int32)
        labels = K.flatten(K.tf.squeeze(y_true[0:num_labeled]))
        predictions = K.flatten(K.tf.cast(K.tf.argmax(y_pred[0:num_labeled], axis=-1), K.tf.int32))
        weights = K.tf.where(K.tf.equal(predictions, ignore_class),
                             K.tf.zeros_like(predictions, dtype=K.tf.float32),
                             K.tf.ones_like(predictions, dtype=K.tf.float32))

        return _mpca(labels=labels, predictions=predictions, num_classes=num_classes, weights=weights)

    return mpca


def semisupervised_mean_iou(num_classes, num_unlabeled, ignore_class=-1):

    def miou(y_true, y_pred):
        num_labeled = K.tf.cast(K.tf.shape(y_true)[0] - num_unlabeled, dtype=K.tf.int32)
        labels = K.flatten(K.tf.squeeze(y_true[0:num_labeled]))
        predictions = K.flatten(K.tf.cast(K.tf.argmax(y_pred[0:num_labeled], axis=-1), dtype=K.tf.int32))
        weights = K.tf.where(K.tf.equal(predictions, ignore_class),
                             K.tf.zeros_like(predictions, dtype=K.tf.float32),
                             K.tf.ones_like(predictions, dtype=K.tf.float32))

        return _miou(labels=labels, predictions=predictions, num_classes=num_classes, weights=weights)

    return miou
