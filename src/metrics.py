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


##############################################
# CLASSIFICATION METRICS
##############################################


def classification_mean_per_class_accuracy(num_classes, num_unlabeled, ignore_classes=None):
    def mpca(y_true, y_pred):
        K.tf.assert_rank(y_true, 2)  # BxC (one-hot encoded)
        K.tf.assert_rank(y_pred, 2)  # BxC (class probabilities)

        if ignore_classes is not None and len(ignore_classes) > 0:
            _ignore_classes = K.tf.constant(ignore_classes, dtype=K.tf.int32)
        else:
            _ignore_classes = K.tf.constant([-1], dtype=K.tf.int32)

        # Assumes that in evaluation/validation there are no unlabeled
        num_labeled = K.in_train_phase(K.tf.cast(K.tf.shape(y_true)[0] - num_unlabeled, dtype=K.tf.int32), K.tf.shape(y_true)[0])
        labels = K.tf.cast(K.tf.argmax(y_true[0:num_labeled], axis=-1), dtype=K.tf.int32)
        predictions = K.tf.cast(K.tf.argmax(y_pred[0:num_labeled], axis=-1), dtype=K.tf.int32)
        mask_weights = K.tf.cast(K.tf.logical_not(_get_ignore_mask(labels, _ignore_classes)), dtype=K.tf.float32)

        cfm = K.tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_classes, weights=mask_weights)

        # Compute the mean per class accuracy via the confusion matrix.
        per_row_sum = K.tf.to_float(K.tf.reduce_sum(cfm, 1))
        cm_diag = K.tf.to_float(K.tf.diag_part(cfm))
        denominator = per_row_sum

        # If the value of the denominator is 0, set it to 1 to avoid
        # zero division.
        denominator = K.tf.where(K.tf.greater(denominator, 0), denominator, K.tf.ones_like(denominator))
        accuracies = K.tf.div(cm_diag, denominator)
        mpca = K.tf.reduce_mean(accuracies)

        return mpca

    return mpca


def classification_accuracy(num_unlabeled, ignore_classes=None):
    def acc(y_true, y_pred):
        K.tf.assert_rank(y_true, 2)  # BxC (one-hot encoded)
        K.tf.assert_rank(y_pred, 2)  # BxC (class probabilities)

        if ignore_classes is not None and len(ignore_classes) > 0:
            _ignore_classes = K.tf.constant(ignore_classes, dtype=K.tf.int32)
        else:
            _ignore_classes = K.tf.constant([-1], dtype=K.tf.int32)

        # Assumes that in evaluation/validation there are no unlabeled
        num_labeled = K.in_train_phase(K.tf.cast(K.tf.shape(y_true)[0] - num_unlabeled, dtype=K.tf.int32), K.tf.shape(y_true)[0])

        # Take the argmax of the labeled
        labels = K.tf.cast(K.tf.argmax(y_true[0:num_labeled], axis=-1), dtype=K.tf.int32)
        predictions = K.tf.cast(K.tf.argmax(y_pred[0:num_labeled], axis=-1), dtype=K.tf.int32)

        # Mark the ignored class samples as -1 to count them into incorrect
        ignored_samples = _get_ignore_mask(labels, _ignore_classes)
        predictions = K.tf.where(ignored_samples, K.tf.ones_like(predictions, dtype=K.tf.int32)*-1, predictions)

        # Now the accuracy is num_correct / num_samples - num_ignored_samples
        num_ignored_samples = K.tf.reduce_sum(K.tf.cast(ignored_samples, dtype=K.tf.float32))
        num_correct_samples = K.tf.reduce_sum(K.tf.cast(K.tf.equal(labels, predictions), dtype=K.tf.float32))
        num_incorrect_samples = K.tf.reduce_sum(K.tf.cast(K.tf.not_equal(labels, predictions), dtype=K.tf.float32))
        num_samples = num_correct_samples + num_incorrect_samples - num_ignored_samples
        return num_correct_samples / num_samples

    return acc


def classification_confusion_matrix(num_classes, num_unlabeled, ignore_classes=None):
    def cfm(y_true, y_pred):
        K.tf.assert_rank(y_true, 2)  # BxC (one-hot encoded)
        K.tf.assert_rank(y_pred, 2)  # BxC (class probabilities)

        if ignore_classes is not None and len(ignore_classes) > 0:
            _ignore_classes = K.tf.constant(ignore_classes, dtype=K.tf.int32)
        else:
            _ignore_classes = K.tf.constant([-1], dtype=K.tf.int32)

        num_labeled = K.in_train_phase(K.tf.cast(K.tf.shape(y_true)[0] - num_unlabeled, dtype=K.tf.int32), K.tf.shape(y_true)[0])
        labels = K.tf.cast(K.tf.argmax(y_true[0:num_labeled], axis=-1), dtype=K.tf.int32)
        predictions = K.tf.cast(K.tf.argmax(y_pred[0:num_labeled], axis=-1), dtype=K.tf.int32)
        mask_weights = K.tf.cast(K.tf.logical_not(_get_ignore_mask(labels, _ignore_classes)), dtype=K.tf.float32)

        confusion_matrix = K.tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_classes, weights=mask_weights, dtype=_CFM_DTYPE)
        return confusion_matrix

    return cfm

##############################################
# SEGMENTATION METRICS
##############################################


def _segmentation_miou(labels, predictions, num_classes, weights=None):
    K.tf.assert_rank(labels, 1)         # Flattened labels
    K.tf.assert_rank(predictions, 1)    # Flattened predictions

    cfm = K.tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_classes, weights=weights)

    # Compute the mean intersection-over-union via the confusion matrix.
    sum_over_row = K.tf.to_float(K.tf.reduce_sum(cfm, 0))
    sum_over_col = K.tf.to_float(K.tf.reduce_sum(cfm, 1))
    cm_diag = K.tf.to_float(K.tf.diag_part(cfm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = K.tf.where(K.tf.greater(denominator, 0), denominator, K.tf.ones_like(denominator))

    iou = K.tf.div(cm_diag, denominator)
    result = K.tf.reduce_mean(iou)

    return result


def _segmentation_mpca(labels, predictions, num_classes, weights=None):
    K.tf.assert_rank(labels, 1)         # Flattened labels
    K.tf.assert_rank(predictions, 1)    # Flattened predictions

    cfm = K.tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_classes, weights=weights)

    # Compute the mean per class accuracy via the confusion matrix.
    per_row_sum = K.tf.to_float(K.tf.reduce_sum(cfm, 1))
    cm_diag = K.tf.to_float(K.tf.diag_part(cfm))
    denominator = per_row_sum

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = K.tf.where(K.tf.greater(denominator, 0), denominator, K.tf.ones_like(denominator))
    accuracies = K.tf.div(cm_diag, denominator)
    result = K.tf.reduce_mean(accuracies)

    return result


def _segmentation_accuracy(labels, predictions, ignore_classes):
    K.tf.assert_rank(labels, 3)          # BxHxW
    K.tf.assert_rank(predictions, 3)     # BxHxW
    K.tf.assert_rank(ignore_classes, 1)  # N

    # Create a mask of true/false describing correct/false classification: [B_SIZE, H, W]
    prediction_results = K.tf.cast(K.equal(labels, predictions), dtype=K.tf.float32)

    # Create a mask of the ignored pixels: [B_SIZE, H, W]
    ignore_mask = _get_ignore_mask(labels, ignore_classes)

    # Calculate number of ignored pixels: [B_SIZE]
    ignored_pixels = K.tf.reduce_sum(K.tf.cast(ignore_mask, dtype=K.tf.float32), axis=(1, 2))

    # Calculate number of correct guesses - mark the ignored pixels as incorrect (0): [B_SIZE]
    prediction_results = K.tf.multiply(prediction_results, K.tf.cast(K.tf.logical_not(ignore_mask), dtype=K.tf.float32))
    prediction_results = K.tf.reduce_sum(prediction_results, axis=(1, 2))

    # Calculate the number of non-ignored pixels in the batch images: [B_SIZE]
    number_of_pixels = K.tf.multiply(K.tf.ones_like(prediction_results, dtype=K.tf.float32), K.tf.cast(K.tf.multiply(K.tf.shape(predictions)[1], K.tf.shape(predictions)[2]), dtype=K.tf.float32))
    number_of_pixels = K.tf.subtract(number_of_pixels, ignored_pixels)

    # If for some reason number of pixels is 0 - set it to 1 to avoid zero division
    number_of_pixels = K.tf.where(K.tf.greater(number_of_pixels, 0), number_of_pixels, K.tf.ones_like(number_of_pixels, dtype=K.tf.float32))

    # Calculate the proportion of true guesses of the non-ignored pixels in the images: [B_SIZE]
    prediction_results = K.tf.div(prediction_results, number_of_pixels)

    # Take the mean of the batch
    prediction_results = K.tf.reduce_mean(prediction_results)

    return K.tf.cond(K.tf.is_finite(prediction_results), lambda: prediction_results, lambda: -1.0)


def segmentation_mean_per_class_accuracy(num_classes, num_unlabeled, ignore_classes=None):
    def mpca(y_true, y_pred):
        K.tf.assert_rank(y_true, 4)  # BxHxWx1
        K.tf.assert_rank(y_pred, 4)  # BxHxWxC

        if ignore_classes is not None and len(ignore_classes) > 0:
            _ignore_classes = K.tf.constant(ignore_classes, dtype=K.tf.int32)
        else:
            _ignore_classes = K.tf.constant([-1], dtype=K.tf.int32)

        # Assumes that in evaluation/validation there are no unlabeled
        num_labeled = K.in_train_phase(K.tf.cast(K.tf.shape(y_true)[0] - num_unlabeled, dtype=K.tf.int32), K.tf.shape(y_true)[0])

        labels = K.tf.cast(K.flatten(K.tf.squeeze(y_true[0:num_labeled], axis=-1)), dtype=K.tf.int32)
        predictions = K.flatten(K.tf.cast(K.tf.argmax(y_pred[0:num_labeled], axis=-1), K.tf.int32))
        ignore_mask = _get_ignore_mask(labels, _ignore_classes)
        weights = K.tf.cast(K.tf.logical_not(ignore_mask), dtype=K.tf.float32)

        return _segmentation_mpca(labels=labels, predictions=predictions, num_classes=num_classes, weights=weights)

    return mpca


def segmentation_mean_iou(num_classes, num_unlabeled, ignore_classes=None):
    def miou(y_true, y_pred):
        K.tf.assert_rank(y_true, 4)  # BxHxWx1
        K.tf.assert_rank(y_pred, 4)  # BxHxWxC

        if ignore_classes is not None and len(ignore_classes) > 0:
            _ignore_classes = K.tf.constant(ignore_classes, dtype=K.tf.int32)
        else:
            _ignore_classes = K.tf.constant([-1], dtype=K.tf.int32)

        # Assumes that in evaluation/validation there are no unlabeled
        num_labeled = K.in_train_phase(K.tf.cast(K.tf.shape(y_true)[0] - num_unlabeled, dtype=K.tf.int32), K.tf.shape(y_true)[0])

        labels = K.tf.cast(K.flatten(K.tf.squeeze(y_true[0:num_labeled], axis=-1)), dtype=K.tf.int32)
        predictions = K.flatten(K.tf.cast(K.tf.argmax(y_pred[0:num_labeled], axis=-1), dtype=K.tf.int32))
        ignore_mask = _get_ignore_mask(labels, _ignore_classes)
        weights = K.tf.cast(K.tf.logical_not(ignore_mask), dtype=K.tf.float32)

        return _segmentation_miou(labels=labels, predictions=predictions, num_classes=num_classes, weights=weights)

    return miou


def segmentation_accuracy(num_unlabeled, ignore_classes=None):
    def acc(y_true, y_pred):
        K.tf.assert_rank(y_true, 4)  # BxHxWx1
        K.tf.assert_rank(y_pred, 4)  # BxHxWxC

        if ignore_classes is not None and len(ignore_classes) > 0:
            _ignore_classes = K.tf.constant(ignore_classes, dtype=K.tf.int32)
        else:
            _ignore_classes = K.tf.constant([-1], dtype=K.tf.int32)

        # Assumes that in evaluation/validation there are no unlabeled
        num_labeled = K.in_train_phase(K.tf.cast(K.tf.shape(y_true)[0] - num_unlabeled, dtype=K.tf.int32), K.tf.shape(y_true)[0])

        labels = K.tf.cast(K.tf.squeeze(y_true[0:num_labeled], axis=-1), dtype=K.tf.int32)
        predictions = K.tf.cast(K.tf.argmax(y_pred[0:num_labeled], axis=-1), dtype=K.tf.int32)
        return _segmentation_accuracy(labels, predictions, _ignore_classes)

    return acc


def segmentation_confusion_matrix(num_classes, num_unlabeled, ignore_classes=None):
    def cfm(y_true, y_pred):
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
        predictions = K.flatten(K.tf.cast(K.tf.argmax(y_pred[0:num_labeled], axis=-1), dtype=K.tf.int32))
        ignore_mask = _get_ignore_mask(labels, _ignore_classes)
        weights = K.tf.cast(K.tf.logical_not(ignore_mask), dtype=K.tf.float32)

        confusion_matrix = K.tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_classes, weights=weights, dtype=_CFM_DTYPE)
        return confusion_matrix

    return cfm
