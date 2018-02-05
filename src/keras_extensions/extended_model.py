# coding=utf-8

import time
import os

try:
    import queue
except ImportError:
    import Queue as queue

import warnings
import numpy as np

from keras.engine.training import Model
from keras.engine.training import GeneratorEnqueuer
from keras.engine.training import _standardize_input_data

from keras import backend as K
from keras import callbacks as cbks
from keras.legacy import interfaces
from keras import optimizers
from keras import losses
from keras import metrics as metrics_module

from keras.engine.training import _collect_metrics, _weighted_masked_objective

from utils.data_utils import Sequence
from utils.data_utils import OrderedEnqueuer
import extended_callbacks as extended_cbks

from ..logger import Logger

#########################################
# UTILITIES
#########################################


class ExtendedModel(Model):
    """
    This class provides extended capabilities to regular Keras model. Namely this
    was created to be able to inject mean teacher data within the fit generator
    method.
    """

    @interfaces.legacy_model_constructor_support
    def __init__(self, inputs, outputs, name=None):
        super(ExtendedModel, self).__init__(inputs=inputs, outputs=outputs, name=name)

        self.fit_generator_stopped = False
        self.process_clean_up_called = False
        self.training_enqueuer = None
        self._training_enqueuer_pre_created = False
        self.validation_enqueuer = None
        self._validation_enqueuer_pre_created = False

        try:
            self.logger = Logger.instance()
        except ValueError:
            self.logger = Logger(log_file_path=None, stdout_only=True)

    @property
    def training_enqueuer_pre_created(self):
        return self._training_enqueuer_pre_created and self.training_enqueuer is not None

    @property
    def validation_enqueuer_pre_created(self):
        return self._validation_enqueuer_pre_created and self.validation_enqueuer is not None

    def stop_training_loop(self):
        self.fit_generator_stopped = True

    def clean_up_processes(self):
        # This can sometimes throw some NoneType errors when stopping training - catch them
        try:
            if not self.process_clean_up_called:
                self.process_clean_up_called = True
                self.fit_generator_stopped = True

                if self.training_enqueuer is not None:
                    self.training_enqueuer.stop(timeout=5)

                # Note: Callback model stop_training flag is set in fit_generator
                if hasattr(self, 'callback_model') and self.callback_model:
                    callback_model = self.callback_model
                else:
                    callback_model = self

                callback_model.stop_training = True
        except (AttributeError, ValueError):
            pass

    def set_pre_created_training_enqueuer(self, enqueuer):
        # type: (SequenceEnqueuer) -> None

        # Pre-create validation enqueuer
        if self.training_enqueuer is not None:
            self.training_enqueuer.stop()
            self.training_enqueuer = None

        self.training_enqueuer = enqueuer
        self._training_enqueuer_pre_created = enqueuer is not None

    def set_pre_created_validation_enqueuer(self, enqueuer):
        # type: (SequenceEnqueuer) -> None

        # Pre-create validation enqueuer
        if self.validation_enqueuer is not None:
            self.validation_enqueuer.stop()
            self.validation_enqueuer = None

        self.validation_enqueuer = enqueuer
        self._validation_enqueuer_pre_created = enqueuer is not None

    def write_cfm_to_file(self, epoch, cfm_key, cfm, file_path=None):
        # type: (int, str, np.ndarray, str) -> None

        # If no file path - log to same folder as other logs
        if file_path is None:
            if self.logger.log_folder_path is None:
                self.logger.warn('Cannot write CFMs to file, logger has not log_folder_path')
                return

            _folder_path = os.path.join(self.logger.log_folder_path, 'cfms/')
            _file_path = os.path.join(_folder_path, '{}_{}.txt'.format(cfm_key, epoch))
        else:
            _file_path = file_path

        # Create directories if they don't exist
        if not os.path.exists(os.path.dirname(_file_path)):
            os.makedirs(os.path.dirname(_file_path))

        try:
            self.logger.log('Writing epoch {} confusion matrix {} to file: {}'.format(epoch, cfm_key, _file_path))
            np.savetxt(_file_path, X=cfm)
        except Exception as e:
            self.logger.warn('Failed to write CFM metrics to file, exception: {}'.format(e.message))

    def reset_metrics(self):

        def reset_metric(metric):
            if getattr(metric, 'reset_op', None) is not None:
                self.logger.debug_log('Resetting metric: {}'.format(metric.__name__))
                K.get_session().run(metric.reset_op)
            else:
                self.logger.debug_log('No reset_op found for metric: {}'.format(metric.__name__))

        # Reset any necessary values from the metrics at the beginning of an epoch
        if isinstance(self.metrics, dict):
            for output, metrics in self.metrics.items():
                for metric in metrics:
                    reset_metric(metric)
        elif isinstance(self.metrics, list):
            for metric in self.metrics:
                reset_metric(metric)
        else:
            self.logger.warn('Unknown format for metrics, expected list or dict, got: {}'.format(self.metrics))

    @property
    def using_cfm_metric(self):
        return self.metrics_cfm is not None and len(self.metrics_cfm) > 0

    @staticmethod
    def pre_create_training_enqueuer(generator,
                                     use_multiprocessing,
                                     shuffle,
                                     epochs,
                                     initial_epoch,
                                     workers,
                                     max_queue_size,
                                     random_seed):
        # type: (Sequence, bool, bool, int, int, int, int, int) -> SequenceEnqueuer
        wait_time = 0.01  # in seconds
        training_generator_is_sequence = isinstance(generator, Sequence)

        training_enqueuer = None

        try:
            if training_generator_is_sequence:
                training_enqueuer = OrderedEnqueuer(generator,
                                                    use_multiprocessing=use_multiprocessing,
                                                    shuffle=shuffle,
                                                    initial_epoch=initial_epoch,
                                                    max_epoch=epochs,
                                                    random_seed=random_seed)
            else:
                training_enqueuer = GeneratorEnqueuer(generator,
                                                      use_multiprocessing=use_multiprocessing,
                                                      wait_time=wait_time,
                                                      random_seed=random_seed)

            training_enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        except Exception as e:
            if training_enqueuer is not None:
                training_enqueuer.stop()

            raise e

        return training_enqueuer

    @staticmethod
    def pre_create_validation_enqueuer(generator,
                                       use_multiprocessing,
                                       workers,
                                       max_queue_size,
                                       random_seed):
        # type: (Sequence, bool, int, int, int) -> SequenceEnqueuer
        wait_time = 0.01  # in seconds
        validation_generator_is_sequence = isinstance(generator, Sequence)

        validation_enqueuer = None

        try:
            if validation_generator_is_sequence:
                validation_enqueuer = OrderedEnqueuer(generator,
                                                      use_multiprocessing=use_multiprocessing,
                                                      shuffle=False,
                                                      initial_epoch=0,
                                                      max_epoch=None,
                                                      random_seed=random_seed)
            else:
                validation_enqueuer = GeneratorEnqueuer(generator,
                                                        use_multiprocessing=use_multiprocessing,
                                                        wait_time=wait_time,
                                                        random_seed=random_seed)

            validation_enqueuer.start(workers=workers, max_queue_size=max_queue_size, start_paused=True)
        except Exception as e:
            if validation_enqueuer is not None:
                validation_enqueuer.stop()

            raise e

        return validation_enqueuer

    def _metrics_updates(self):

        update_ops = []

        for output, metrics in self.metrics.items():
            for metric in metrics:
                if getattr(metric, 'update_op', None) is not None:
                    update_ops.append(getattr(metric, 'update_op', None))

        return update_ops

    def _make_test_function(self):
        if not hasattr(self, 'test_function'):
            raise RuntimeError('You must compile your model before using it.')
        if self.test_function is None:
            inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs += [K.learning_phase()]
            # Return loss and metrics, no gradient updates.
            # Does update the network states.
            self.test_function = K.function(inputs,
                                            [self.total_loss] + self.metrics_tensors,
                                            updates=self.state_updates + self._metrics_updates,
                                            name='test_function',
                                            **self._function_kwargs)

    def compile(self, optimizer, loss, metrics=None, loss_weights=None,
                sample_weight_mode=None, weighted_metrics=None,
                target_tensors=None, **kwargs):
        """Configures the model for training.

        # Arguments
            optimizer: String (name of optimizer) or optimizer object.
                See [optimizers](/optimizers).
            loss: String (name of objective function) or objective function.
                See [losses](/losses).
                If the model has multiple outputs, you can use a different loss
                on each output by passing a dictionary or a list of losses.
                The loss value that will be minimized by the model
                will then be the sum of all individual losses.
            metrics: List of metrics to be evaluated by the model
                during training and testing.
                Typically you will use `metrics=['accuracy']`.
                To specify different metrics for different outputs of a
                multi-output model, you could also pass a dictionary,
                such as `metrics={'output_a': 'accuracy'}`.
            loss_weights: Optional list or dictionary specifying scalar
                coefficients (Python floats) to weight the loss contributions
                of different model outputs.
                The loss value that will be minimized by the model
                will then be the *weighted sum* of all individual losses,
                weighted by the `loss_weights` coefficients.
                If a list, it is expected to have a 1:1 mapping
                to the model's outputs. If a tensor, it is expected to map
                output names (strings) to scalar coefficients.
            sample_weight_mode: If you need to do timestep-wise
                sample weighting (2D weights), set this to `"temporal"`.
                `None` defaults to sample-wise weights (1D).
                If the model has multiple outputs, you can use a different
                `sample_weight_mode` on each output by passing a
                dictionary or a list of modes.
            weighted_metrics: List of metrics to be evaluated and weighted
                by sample_weight or class_weight during training and testing.
            target_tensors: By default, Keras will create placeholders for the
                model's target, which will be fed with the target data during
                training. If instead you would like to use your own
                target tensors (in turn, Keras will not expect external
                Numpy data for these targets at training time), you
                can specify them via the `target_tensors` argument. It can be
                a single tensor (for a single-output model), a list of tensors,
                or a dict mapping output names to target tensors.
            **kwargs: When using the Theano/CNTK backends, these arguments
                are passed into K.function. When using the TensorFlow backend,
                these arguments are passed into `tf.Session.run`.

        # Raises
            ValueError: In case of invalid arguments for
                `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
        """
        loss = loss or {}
        self.optimizer = optimizers.get(optimizer)
        self.sample_weight_mode = sample_weight_mode
        self.loss = loss
        self.loss_weights = loss_weights

        # Prepare loss functions.
        if isinstance(loss, dict):
            for name in loss:
                if name not in self.output_names:
                    raise ValueError('Unknown entry in loss '
                                     'dictionary: "' + name + '". '
                                     'Only expected the following keys: ' +
                                     str(self.output_names))
            loss_functions = []
            for name in self.output_names:
                if name not in loss:
                    warnings.warn('Output "' + name +
                                  '" missing from loss dictionary. '
                                  'We assume this was done on purpose, '
                                  'and we will not be expecting '
                                  'any data to be passed to "' + name +
                                  '" during training.', stacklevel=2)
                loss_functions.append(losses.get(loss.get(name)))
        elif isinstance(loss, list):
            if len(loss) != len(self.outputs):
                raise ValueError('When passing a list as loss, '
                                 'it should have one entry per model outputs. '
                                 'The model has ' + str(len(self.outputs)) +
                                 ' outputs, but you passed loss=' +
                                 str(loss))
            loss_functions = [losses.get(l) for l in loss]
        else:
            loss_function = losses.get(loss)
            loss_functions = [loss_function for _ in range(len(self.outputs))]
        self.loss_functions = loss_functions
        weighted_losses = [_weighted_masked_objective(fn) for fn in loss_functions]
        skip_target_indices = []
        skip_target_weighing_indices = []
        self._feed_outputs = []
        self._feed_output_names = []
        self._feed_output_shapes = []
        self._feed_loss_fns = []
        for i in range(len(weighted_losses)):
            if weighted_losses[i] is None:
                skip_target_indices.append(i)
                skip_target_weighing_indices.append(i)

        # Prepare output masks.
        masks = self.compute_mask(self.inputs, mask=None)
        if masks is None:
            masks = [None for _ in self.outputs]
        if not isinstance(masks, list):
            masks = [masks]

        # Prepare loss weights.
        if loss_weights is None:
            loss_weights_list = [1. for _ in range(len(self.outputs))]
        elif isinstance(loss_weights, dict):
            for name in loss_weights:
                if name not in self.output_names:
                    raise ValueError('Unknown entry in loss_weights '
                                     'dictionary: "' + name + '". '
                                     'Only expected the following keys: ' +
                                     str(self.output_names))
            loss_weights_list = []
            for name in self.output_names:
                loss_weights_list.append(loss_weights.get(name, 1.))
        elif isinstance(loss_weights, list):
            if len(loss_weights) != len(self.outputs):
                raise ValueError('When passing a list as loss_weights, '
                                 'it should have one entry per model outputs. '
                                 'The model has ' + str(len(self.outputs)) +
                                 ' outputs, but you passed loss_weights=' +
                                 str(loss_weights))
            loss_weights_list = loss_weights
        else:
            raise TypeError('Could not interpret loss_weights argument: ' +
                            str(loss_weights) +
                            ' - expected a list of dicts.')

        # Prepare targets of model.
        self.targets = []
        self._feed_targets = []
        if target_tensors is not None:
            if isinstance(target_tensors, list):
                if len(target_tensors) != len(self.outputs):
                    raise ValueError(
                        'When passing a list as `target_tensors`, '
                        'it should have one entry per model outputs. '
                        'The model has ' + str(len(self.outputs)) +
                        ' outputs, but you passed target_tensors=' +
                        str(target_tensors))
            elif isinstance(target_tensors, dict):
                for name in target_tensors:
                    if name not in self.output_names:
                        raise ValueError('Unknown entry in `target_tensors` '
                                         'dictionary: "' + name + '". '
                                         'Only expected the following keys: ' +
                                         str(self.output_names))
                _target_tensors = []
                for name in self.output_names:
                    _target_tensors.append(target_tensors.get(name, None))
                target_tensors = _target_tensors
            else:
                raise TypeError('Expected `target_tensors` to be '
                                'a list or dict, but got:', target_tensors)
        for i in range(len(self.outputs)):
            if i in skip_target_indices:
                self.targets.append(None)
            else:
                shape = self.internal_output_shapes[i]
                name = self.output_names[i]
                if target_tensors is not None:
                    target = target_tensors[i]
                else:
                    target = None
                if target is None or K.is_placeholder(target):
                    if target is None:
                        target = K.placeholder(ndim=len(shape),
                                               name=name + '_target',
                                               sparse=K.is_sparse(self.outputs[i]),
                                               dtype=K.dtype(self.outputs[i]))
                    self._feed_targets.append(target)
                    self._feed_outputs.append(self.outputs[i])
                    self._feed_output_names.append(name)
                    self._feed_output_shapes.append(shape)
                    self._feed_loss_fns.append(self.loss_functions[i])
                else:
                    skip_target_weighing_indices.append(i)
                self.targets.append(target)

        # Prepare sample weights.
        sample_weights = []
        sample_weight_modes = []
        if isinstance(sample_weight_mode, dict):
            for name in sample_weight_mode:
                if name not in self.output_names:
                    raise ValueError('Unknown entry in '
                                     'sample_weight_mode dictionary: "' +
                                     name + '". '
                                     'Only expected the following keys: ' +
                                     str(self.output_names))
            for i, name in enumerate(self.output_names):
                if i in skip_target_weighing_indices:
                    weight = None
                    sample_weight_modes.append(None)
                else:
                    if name not in sample_weight_mode:
                        raise ValueError('Output "' + name +
                                         '" missing from sample_weight_modes '
                                         'dictionary')
                    if sample_weight_mode.get(name) == 'temporal':
                        weight = K.placeholder(ndim=2,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append('temporal')
                    else:
                        weight = K.placeholder(ndim=1,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append(None)
                sample_weights.append(weight)
        elif isinstance(sample_weight_mode, list):
            if len(sample_weight_mode) != len(self.outputs):
                raise ValueError('When passing a list as sample_weight_mode, '
                                 'it should have one entry per model outputs. '
                                 'The model has ' + str(len(self.outputs)) +
                                 ' outputs, but you passed '
                                 'sample_weight_mode=' +
                                 str(sample_weight_mode))
            for i in range(len(self.output_names)):
                if i in skip_target_weighing_indices:
                    weight = None
                    sample_weight_modes.append(None)
                else:
                    mode = sample_weight_mode[i]
                    name = self.output_names[i]
                    if mode == 'temporal':
                        weight = K.placeholder(ndim=2,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append('temporal')
                    else:
                        weight = K.placeholder(ndim=1,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append(None)
                sample_weights.append(weight)
        else:
            for i, name in enumerate(self.output_names):
                if i in skip_target_weighing_indices:
                    sample_weight_modes.append(None)
                    sample_weights.append(None)
                else:
                    if sample_weight_mode == 'temporal':
                        sample_weights.append(
                            K.placeholder(ndim=2,
                                          name=name + '_sample_weights'))
                        sample_weight_modes.append('temporal')
                    else:
                        sample_weights.append(
                            K.placeholder(ndim=1,
                                          name=name + '_sample_weights'))
                        sample_weight_modes.append(None)
        self.sample_weight_modes = sample_weight_modes
        self._feed_sample_weight_modes = []
        for i in range(len(self.outputs)):
            if i not in skip_target_weighing_indices:
                self._feed_sample_weight_modes.append(self.sample_weight_modes[i])

        # Prepare metrics.
        self.metrics = metrics
        self.weighted_metrics = weighted_metrics
        self.metrics_names = ['loss']
        self.metrics_tensors = []
        self.metrics_hidden_from_progbar = set()
        self.metrics_excluded_from_callbacks = set()
        self.metrics_streaming = set()
        self.metrics_cfm = set()

        # Compute total loss.
        total_loss = None
        with K.name_scope('loss'):
            for i in range(len(self.outputs)):
                if i in skip_target_indices:
                    continue
                y_true = self.targets[i]
                y_pred = self.outputs[i]
                weighted_loss = weighted_losses[i]
                sample_weight = sample_weights[i]
                mask = masks[i]
                loss_weight = loss_weights_list[i]
                with K.name_scope(self.output_names[i] + '_loss'):
                    output_loss = weighted_loss(y_true, y_pred,
                                                sample_weight, mask)
                if len(self.outputs) > 1:
                    self.metrics_tensors.append(output_loss)
                    self.metrics_names.append(self.output_names[i] + '_loss')
                if total_loss is None:
                    total_loss = loss_weight * output_loss
                else:
                    total_loss += loss_weight * output_loss
            if total_loss is None:
                if not self.losses:
                    raise ValueError('The model cannot be compiled '
                                     'because it has no loss to optimize.')
                else:
                    total_loss = 0.

            # Add regularization penalties
            # and other layer-specific losses.
            for loss_tensor in self.losses:
                total_loss += loss_tensor

        # List of same size as output_names.
        # contains tuples (metrics for output, names of metrics).
        nested_metrics = _collect_metrics(metrics, self.output_names)
        nested_weighted_metrics = _collect_metrics(weighted_metrics, self.output_names)

        def append_metric(layer_index, metric_name, metric_tensor, is_hidden_from_progbar, is_excluded_from_callbacks, is_streaming, is_cfm):
            """Helper function used in loop below."""
            if len(self.output_names) > 1:
                metric_name = self.output_names[layer_index] + '_' + metric_name
            self.metrics_names.append(metric_name)
            self.metrics_tensors.append(metric_tensor)

            if is_hidden_from_progbar:
                self.metrics_hidden_from_progbar.add(metric_name)

            if is_excluded_from_callbacks:
                self.metrics_excluded_from_callbacks.add(metric_name)

            if is_streaming:
                self.metrics_streaming.add(metric_name)

            if is_cfm:
                self.metrics_cfm.add(metric_name)

        with K.name_scope('metrics'):
            for i in range(len(self.outputs)):
                if i in skip_target_indices:
                    continue

                y_true = self.targets[i]
                y_pred = self.outputs[i]
                weights = sample_weights[i]
                output_metrics = nested_metrics[i]
                output_weighted_metrics = nested_weighted_metrics[i]

                def handle_metrics(metrics, weights=None):
                    metric_name_prefix = 'weighted_' if weights is not None else ''

                    for metric in metrics:
                        if metric == 'accuracy' or metric == 'acc':
                            # custom handling of accuracy
                            # (because of class mode duality)
                            output_shape = self.internal_output_shapes[i]
                            if (output_shape[-1] == 1 or
                               self.loss_functions[i] == losses.binary_crossentropy):
                                # case: binary accuracy
                                acc_fn = metrics_module.binary_accuracy
                            elif self.loss_functions[i] == losses.sparse_categorical_crossentropy:
                                # case: categorical accuracy with sparse targets
                                acc_fn = metrics_module.sparse_categorical_accuracy
                            else:
                                acc_fn = metrics_module.categorical_accuracy

                            is_hidden_from_progbar = getattr(acc_fn, 'hide_from_progbar', False)
                            is_excluded_from_callbacks = getattr(acc_fn, 'exclude_from_callbacks', False)
                            is_streaming = getattr(acc_fn, 'streaming', False)
                            is_cfm = getattr(acc_fn, 'cfm', False)
                            weighted_metric_fn = _weighted_masked_objective(acc_fn)
                            metric_name = metric_name_prefix + 'acc'
                        else:
                            metric_fn = metrics_module.get(metric)
                            is_hidden_from_progbar = getattr(metric_fn, 'hide_from_progbar', False)
                            is_excluded_from_callbacks = getattr(metric_fn, 'exclude_from_callbacks', False)
                            is_streaming = getattr(metric_fn, 'streaming', False)
                            is_cfm = getattr(metric_fn, 'cfm', False)

                            # Streaming metrics and CFMs skip the weighted_masked_objective function
                            if is_streaming or is_cfm:
                                weighted_metric_fn = lambda y_true, y_pred, weights, mask: metric_fn(y_true, y_pred)
                            else:
                                weighted_metric_fn = _weighted_masked_objective(metric_fn)

                            metric_name = metric_name_prefix + metric_fn.__name__

                        with K.name_scope(metric_name):
                            metric_result = weighted_metric_fn(y_true, y_pred, weights=weights, mask=masks[i])

                        append_metric(i,
                                      metric_name,
                                      metric_result,
                                      is_hidden_from_progbar,
                                      is_excluded_from_callbacks,
                                      is_streaming,
                                      is_cfm)

                handle_metrics(output_metrics)
                handle_metrics(output_weighted_metrics, weights=weights)

        # Prepare gradient updates and state updates.
        self.total_loss = total_loss
        self.sample_weights = sample_weights
        self._feed_sample_weights = []
        for i in range(len(self.sample_weights)):
            if i not in skip_target_weighing_indices:
                self._feed_sample_weights.append(sample_weights[i])

        # Functions for train, test and predict will
        # be compiled lazily when required.
        # This saves time when the user is not using all functions.
        self._function_kwargs = kwargs

        self.train_function = None
        self.test_function = None
        self.predict_function = None

        # Collected trainable weights, sorted in topological order.
        trainable_weights = self.trainable_weights
        self._collected_trainable_weights = trainable_weights

    def predict_on_batch(self, x, use_training_phase_layers=False):
        """Returns predictions for a single batch of samples.

        # Arguments
            x: Input samples, as a Numpy array.
            use_training_phase_layers: A boolean value describing whether we should use training only layers
            such as GaussianNoise and Dropout during predictions.

        # Returns
            Numpy array(s) of predictions.
        """
        x = _standardize_input_data(x, self._feed_input_names, self._feed_input_shapes)

        if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
            if use_training_phase_layers:
                ins = x + [1.]
            else:
                ins = x + [0.]
        else:
            ins = x
        self._make_predict_function()
        outputs = self.predict_function(ins)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    @interfaces.legacy_generator_methods_support
    def fit_generator(self, generator,
                      steps_per_epoch,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_steps=None,
                      class_weight=None,
                      max_queue_size=10,
                      validation_max_queue_size=10,
                      workers=1,
                      validation_workers=1,
                      use_multiprocessing=False,
                      shuffle=True,
                      initial_epoch=0,
                      trainer=None,
                      random_seed=None):
        """Fits the model on data yielded batch-by-batch by a Python generator.

        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.

        The use of `keras.utils.Sequence` guarantees the ordering
        and guarantees the single use of every input per epoch when
        using `use_multiprocessing=True`.

        # Arguments
            generator: a generator or an instance of Sequence (keras.utils.Sequence)
                    object in order to avoid duplicate data
                    when using multiprocessing.
                The output of the generator must be either
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
                All arrays should contain the same number of samples.
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `steps_per_epoch`
                batches have been seen by the model.
            steps_per_epoch: Total number of steps (batches of samples)
                to yield from `generator` before declaring one epoch
                finished and starting the next epoch. It should typically
                be equal to the number of unique samples if your dataset
                divided by the batch size.
            epochs: integer, total number of iterations on the data.
            verbose: verbosity mode, 0, 1, or 2.
            callbacks: list of callbacks to be called during training.
            validation_data: this can be either
                - a generator for the validation data
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
            validation_steps: Only relevant if `validation_data`
                is a generator. Total number of steps (batches of samples)
                to yield from `generator` before stopping.
            class_weight: dictionary mapping class indices to a weight
                for the class.
            max_queue_size: maximum size for the generator queue
            validation_max_queue_size: maximum size for the generator queue for validation
            workers: maximum number of processes to spin up
                when using process based threading
            validation_workers: maximum number of processes to spin up
                when running validation
            use_multiprocessing: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
            shuffle: Whether to shuffle the data at the beginning of each
                epoch. Only used with instances of `Sequence` (
                keras.utils.Sequence).
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)
            trainer: reference to a TrainerBase inherited object
            random_seed: random seed
        # Returns
            A `History` object.

        # Example

        ```python
            def generate_arrays_from_file(path):
                while 1:
                    f = open(path)
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x1, x2, y = process_line(line)
                        yield ({'input_1': x1, 'input_2': x2}, {'output': y})
                    f.close()

            model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                                steps_per_epoch=10000, epochs=10)
        ```

        # Raises
            ValueError: In case the generator yields
                data in an invalid format.
        """
        self.fit_generator_stopped = False
        self.process_clean_up_called = False

        wait_time = 0.01  # in seconds
        epoch = initial_epoch

        do_validation = bool(validation_data)
        self._make_train_function()
        if do_validation:
            self._make_test_function()

        # python 2 has 'next', 3 has '__next__'
        # avoid any explicit version checks
        val_gen = (hasattr(validation_data, 'next') or
                   hasattr(validation_data, '__next__') or
                   isinstance(validation_data, Sequence))
        if val_gen and not validation_steps:
            raise ValueError('When using a generator for validation data, '
                             'you must specify a value for '
                             '`validation_steps`.')

        # Prepare display labels.
        out_labels = self._get_deduped_metrics_names()
        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        # prepare callbacks
        self.history = cbks.History()
        baselogger = extended_cbks.ExtendedBaseLogger()
        callbacks = [baselogger] + (callbacks or []) + [self.history]
        if verbose:
            callbacks += [extended_cbks.ExtendedProgbarLogger(count_mode='steps')]
        callbacks = cbks.CallbackList(callbacks)

        # it's possible to callback a different model than self:
        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self
        callbacks.set_model(callback_model)
        callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })
        callbacks.on_train_begin()

        if do_validation and not val_gen:
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError('`validation_data` should be a tuple '
                                 '`(val_x, val_y, val_sample_weight)` '
                                 'or `(val_x, val_y)`. Found: ' +
                                 str(validation_data))
            val_x, val_y, val_sample_weights = self._standardize_user_data(
                val_x, val_y, val_sample_weight)
            val_data = val_x + val_y + val_sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                val_data += [0.]
            for cbk in callbacks:
                cbk.validation_data = val_data

        if not self.training_enqueuer_pre_created and generator is None:
            raise ValueError('Invalid generator passed - must either pass a valid generator or pre create the enqueuer')

        enqueuer = None

        try:
            if not self.training_enqueuer_pre_created:
                is_sequence = isinstance(generator, Sequence)

                if not is_sequence and use_multiprocessing and workers > 1:
                    warnings.warn(
                        UserWarning('Using a generator with `use_multiprocessing=True`'
                                    ' and multiple workers may duplicate your data.'
                                    ' Please consider using the`keras.utils.Sequence'
                                    ' class.'))

                enqueuer = None

                if is_sequence:
                    enqueuer = OrderedEnqueuer(generator,
                                               use_multiprocessing=use_multiprocessing,
                                               shuffle=shuffle,
                                               initial_epoch=initial_epoch,
                                               max_epoch=epochs,
                                               random_seed=random_seed)
                else:
                    enqueuer = GeneratorEnqueuer(generator,
                                                 use_multiprocessing=use_multiprocessing,
                                                 wait_time=wait_time,
                                                 random_seed=random_seed)
                enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            else:
                enqueuer = self.training_enqueuer

            enqueuer.continue_run()
            output_generator = enqueuer.get()

            callback_model.stop_training = False
            while epoch < epochs:

                # Reset any necessary values from the metrics at the beginning of an epoch
                self.reset_metrics()

                epoch_s_time = time.time()
                callbacks.on_epoch_begin(epoch)
                steps_done = 0
                batch_index = 0
                while steps_done < steps_per_epoch:
                    generator_output = next(output_generator)

                    if not hasattr(generator_output, '__len__'):
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
                    if len(generator_output) == 2:
                        x, y = generator_output
                        sample_weight = None
                    elif len(generator_output) == 3:
                        x, y, sample_weight = generator_output
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
                    # build batch logs
                    batch_logs = {}
                    if isinstance(x, list):
                        batch_size = x[0].shape[0]
                    elif isinstance(x, dict):
                        batch_size = list(x.values())[0].shape[0]
                    else:
                        batch_size = x.shape[0]
                    batch_logs['batch'] = batch_index
                    batch_logs['size'] = batch_size
                    s_time = time.time()
                    callbacks.on_batch_begin(batch_index, batch_logs)
                    self.logger.debug_log('Callbacks on_batch_begin took: {}s'.format(time.time() - s_time))

                    step_index = steps_done + epoch*steps_per_epoch

                    # Extended functionality: notify trainer
                    if trainer is not None:
                        s_time = time.time()
                        x, y = trainer.modify_batch_data(step_index, x, y)
                        self.logger.debug_log('Trainer modify_batch_data took: {}s'.format(time.time() - s_time))

                    # Extended functionality: stop if early stopping has been initiated
                    if self.fit_generator_stopped:
                        return self.history

                    s_time = time.time()

                    outs = self.train_on_batch(x, y,
                                               sample_weight=sample_weight,
                                               class_weight=class_weight)

                    self.logger.debug_log('Train on batch took: {} s'.format(time.time() - s_time))

                    if not isinstance(outs, list):
                        outs = [outs]
                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o

                    s_time = time.time()
                    callbacks.on_batch_end(batch_index, batch_logs)
                    self.logger.debug_log('Callbacks on_batch_end took: {}s'.format(time.time() - s_time))

                    # Extended functionality: notify trainer
                    if trainer is not None:
                        s_time = time.time()
                        trainer.on_batch_end(step_index)
                        self.logger.debug_log('Trainer on_batch_end took: {}s'.format(time.time() - s_time))

                    # Construct epoch logs.
                    epoch_logs = {}
                    batch_index += 1
                    steps_done += 1

                    # Epoch finished.
                    if steps_done >= steps_per_epoch and do_validation:
                        self.logger.log('Epoch {} training took: {} s'.format(epoch, time.time()-epoch_s_time))

                        # Reset metrics before validation run
                        self.reset_metrics()

                        if val_gen:
                            # Extended functionality: pass trainer and validation flag
                            enqueuer.pause_run()

                            val_outs = self.evaluate_generator(
                                validation_data,
                                validation_steps,
                                max_queue_size=validation_max_queue_size,
                                workers=validation_workers,
                                trainer=trainer,
                                use_multiprocessing=use_multiprocessing,
                                validation=True)

                            enqueuer.continue_run()
                        else:
                            # No need for try/except because
                            # data has already been validated.
                            val_outs = self.evaluate(
                                val_x, val_y,
                                batch_size=batch_size,
                                sample_weight=val_sample_weights,
                                verbose=0)
                        if not isinstance(val_outs, list):
                            val_outs = [val_outs]
                        # Same labels assumed.
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o

                # Write CFMs to files
                if self.using_cfm_metric:

                    # Write training CFM to file
                    for cfm_metric_name in self.metrics_cfm:
                        # Write training confusion matrix to file
                        cfm = baselogger.get_metric_value(cfm_metric_name)

                        if cfm is None:
                            self.logger.warn('Could not get value from baselogger for CFM: {}'.format(cfm_metric_name))
                            continue

                        cfm = np.array(cfm, dtype=np.float64)
                        self.write_cfm_to_file(epoch, cfm_key=cfm_metric_name, cfm=cfm)

                        # Write corresponding validation CFM to file - same labels with 'val_' prefix assumed
                        val_cfm_metric_name = 'val_' + cfm_metric_name
                        if val_cfm_metric_name in epoch_logs:
                            val_cfm = epoch_logs[val_cfm_metric_name]
                            self.write_cfm_to_file(epoch, cfm_key=val_cfm_metric_name, cfm=val_cfm)

                callbacks.on_epoch_end(epoch, epoch_logs)

                # Reset metrics after epoch ends
                self.reset_metrics()

                # Extended functionality: notify trainer
                if trainer is not None:
                    trainer.on_epoch_end(epoch, (epoch+1)*steps_per_epoch, epoch_logs)

                epoch += 1
                self.logger.log('Epoch {} took in total: {} s'.format(epoch-1, time.time()-epoch_s_time))

                if callback_model.stop_training:
                    break
        finally:
            if enqueuer is not None:
                enqueuer.stop()

        # Extended functionality: notify trainer
        if trainer is not None:
            trainer.on_training_end()

        callbacks.on_train_end()
        return self.history

    @interfaces.legacy_generator_methods_support
    def evaluate_generator(self, generator, steps,
                           max_queue_size=10,
                           workers=1,
                           use_multiprocessing=False,
                           validation=False,
                           trainer=None,
                           random_seed=None):
        """Evaluates the model on a data generator.

        The generator should return the same kind of data
        as accepted by `test_on_batch`.

        # Arguments
            generator: Generator yielding tuples (inputs, targets)
                or (inputs, targets, sample_weights)
                or an instance of Sequence (keras.utils.Sequence)
                    object in order to avoid duplicate data
                    when using multiprocessing.
            steps: Total number of steps (batches of samples)
                to yield from `generator` before stopping.
            max_queue_size: maximum size for the generator queue
            workers: maximum number of processes to spin up
                when using process based threading
            trainer: a reference to TrainerBase inherited object to augment batch data if need be
            use_multiprocessing: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
            random_seed: random seed

        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.

        # Raises
            ValueError: In case the generator yields
                data in an invalid format.
        """
        self._make_test_function()

        steps_done = 0
        wait_time = 0.01
        all_outs = []
        batch_sizes = []

        if not self.training_enqueuer_pre_created and generator is None:
            raise ValueError('Invalid generator passed - must either pass a valid generator or pre create the enqueuer')

        enqueuer = None

        try:
            if not self.validation_enqueuer_pre_created or not validation:
                is_sequence = isinstance(generator, Sequence)
                if not is_sequence and use_multiprocessing and workers > 1:
                    warnings.warn(
                        UserWarning('Using a generator with `use_multiprocessing=True`'
                                    ' and multiple workers may duplicate your data.'
                                    ' Please consider using the`keras.utils.Sequence'
                                    ' class.'))
                enqueuer = None

                if is_sequence:
                    enqueuer = OrderedEnqueuer(generator,
                                               use_multiprocessing=use_multiprocessing,
                                               max_epoch=1,
                                               random_seed=random_seed)
                else:
                    enqueuer = GeneratorEnqueuer(generator,
                                                 use_multiprocessing=use_multiprocessing,
                                                 wait_time=wait_time,
                                                 random_seed=random_seed)
                enqueuer.start(workers=workers, max_queue_size=max_queue_size)
                self.logger.log('Created a new validation enqueuer')
            else:
                enqueuer = self.validation_enqueuer

            enqueuer.continue_run()
            output_generator = enqueuer.get()
            eval_s_time = time.time()

            while steps_done < steps:
                generator_output = next(output_generator)
                if not hasattr(generator_output, '__len__'):
                    raise ValueError('Output of generator should be a tuple '
                                     '(x, y, sample_weight) '
                                     'or (x, y). Found: ' +
                                     str(generator_output))
                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    raise ValueError('Output of generator should be a tuple '
                                     '(x, y, sample_weight) '
                                     'or (x, y). Found: ' +
                                     str(generator_output))

                if trainer is not None:
                    s_time = time.time()
                    x, y = trainer.modify_batch_data(steps_done, x, y, validation)
                    self.logger.debug_log('Call to modify_batch_data took: {} s'.format(time.time()-s_time))

                s_time = time.time()
                outs = self.test_on_batch(x, y, sample_weight=sample_weight)
                self.logger.debug_log('Test on batch took: {} s'.format(time.time()-s_time))

                if isinstance(x, list):
                    batch_size = len(x[0])
                elif isinstance(x, dict):
                    batch_size = len(list(x.values())[0])
                else:
                    batch_size = len(x)
                if batch_size == 0:
                    raise ValueError('Received an empty batch. '
                                     'Batches should at least contain one item.')
                all_outs.append(outs)

                steps_done += 1
                batch_sizes.append(batch_size)
        except Exception as e:
            if enqueuer is not None:
                enqueuer.stop()
            raise e
        finally:
            if validation and self.validation_enqueuer_pre_created:
                self.validation_enqueuer.pause_run()
            else:
                if enqueuer is not None:
                    enqueuer.stop()

        self.logger.log('Evaluation took: {} s'.format(time.time()-eval_s_time))

        if not isinstance(outs, list):
            return np.average(np.asarray(all_outs), weights=batch_sizes)
        else:
            averages = []

            for i in range(len(outs)):
                per_batch_metrics = np.array([out[i] for out in all_outs])
                metric_name = self.metrics_names[i]

                # If the metric is a streaming metric - only use the lasr value
                if metric_name in self.metrics_streaming:
                    averages.append(per_batch_metrics[-1])
                else:
                    # If the metric is a CFM - calculate the sum across batches instead of an average
                    if metric_name in self.metrics_cfm:
                        averages.append(np.sum(per_batch_metrics, axis=0))
                    else:
                        averages.append(np.average(per_batch_metrics, weights=batch_sizes))

            return averages
