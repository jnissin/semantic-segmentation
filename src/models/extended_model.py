# coding=utf-8

import time

try:
    import queue
except ImportError:
    import Queue as queue

import warnings
import numpy as np

from keras.utils import Sequence
from keras.utils import OrderedEnqueuer
from keras.engine.training import Model
from keras.engine.training import GeneratorEnqueuer

from keras import backend as K
from keras import callbacks as cbks
from keras.legacy import interfaces


#########################################
# UTILITIES
#########################################

class ExtendedModel(Model):
    """
    This class provides extended capabilities to regular Keras model. Namely this
    was created to be able to inject mean teacher data within the fit generator
    method.
    """

    fit_generator_stopped = False

    def stop_fit_generator(self):
        self.fit_generator_stopped = True

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
                      workers=1,
                      use_multiprocessing=False,
                      initial_epoch=0,
                      trainer=None,
                      debug=False):
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
            workers: maximum number of processes to spin up
                when using process based threading
            use_multiprocessing: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)
            trainer: reference to a TrainerBase inherited object
            debug: run in debug mode

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
        callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger(count_mode='steps')]
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
        is_sequence = isinstance(generator, Sequence)
        if not is_sequence and use_multiprocessing and workers > 1:
            warnings.warn(
                UserWarning('Using a generator with `use_multiprocessing=True`'
                            ' and multiple workers may duplicate your data.'
                            ' Please consider using the`keras.utils.Sequence'
                            ' class.'))
        enqueuer = None

        try:
            if is_sequence:
                enqueuer = OrderedEnqueuer(generator,
                                           use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(generator,
                                             use_multiprocessing=use_multiprocessing,
                                             wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()

            callback_model.stop_training = False
            while epoch < epochs:
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
                    callbacks.on_batch_begin(batch_index, batch_logs)

                    step_index = steps_done + epoch*steps_per_epoch

                    # Extended functionality: notify trainer
                    if trainer is not None:
                        x, y = trainer.modify_batch_data(step_index, x, y)

                    # Extended functionality: stop if early stopping has been initiated
                    if self.fit_generator_stopped:
                        self.fit_generator_stopped = False
                        return self.history

                    s_time = 0

                    if debug:
                        s_time = time.time()

                    outs = self.train_on_batch(x, y,
                                               sample_weight=sample_weight,
                                               class_weight=class_weight)

                    if debug:
                        print 'Train on batch took: {} s'.format(time.time()-s_time)

                    if not isinstance(outs, list):
                        outs = [outs]
                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o

                    callbacks.on_batch_end(batch_index, batch_logs)

                    # Extended functionality: notify trainer
                    if trainer is not None:
                        trainer.on_batch_end(step_index)

                    # Construct epoch logs.
                    epoch_logs = {}
                    batch_index += 1
                    steps_done += 1

                    # Epoch finished.
                    if steps_done >= steps_per_epoch and do_validation:
                        if val_gen:

                            if debug:
                                s_time = time.time()

                            # Extended functionality: pass trainer and validation flag
                            val_outs = self.evaluate_generator(
                                validation_data,
                                validation_steps,
                                max_queue_size=max_queue_size,
                                workers=workers,
                                trainer=trainer,
                                use_multiprocessing=use_multiprocessing,
                                validation=True)

                            if debug:
                                print 'Validation evaluation took: {}'.format(time.time()-s_time)
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

                callbacks.on_epoch_end(epoch, epoch_logs)

                # Extended functionality: notify trainer
                if trainer is not None:
                    trainer.on_epoch_end(epoch, (epoch+1)*steps_per_epoch, epoch_logs)

                epoch += 1
                if callback_model.stop_training:
                    break

        finally:
            if enqueuer is not None:
                enqueuer.stop()

        callbacks.on_train_end()
        return self.history

    @interfaces.legacy_generator_methods_support
    def evaluate_generator(self, generator, steps,
                           max_queue_size=10,
                           workers=1,
                           use_multiprocessing=False,
                           trainer=None,
                           validation=False):
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
            use_multiprocessing: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
            trainer: a reference to TrainerBase inherited object to augment batch data if need be
            validation: is this a validation run evaluation

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
        is_sequence = isinstance(generator, Sequence)
        if not is_sequence and use_multiprocessing and workers > 1:
            warnings.warn(
                UserWarning('Using a generator with `use_multiprocessing=True`'
                            ' and multiple workers may duplicate your data.'
                            ' Please consider using the`keras.utils.Sequence'
                            ' class.'))
        enqueuer = None

        try:
            if is_sequence:
                enqueuer = OrderedEnqueuer(generator,
                                           use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(generator,
                                             use_multiprocessing=use_multiprocessing,
                                             wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()

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
                    x, y = trainer.modify_batch_data(steps_done, x, y, validation)

                outs = self.test_on_batch(x, y, sample_weight=sample_weight)

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

        finally:
            if enqueuer is not None:
                enqueuer.stop()

        if not isinstance(outs, list):
            return np.average(np.asarray(all_outs),
                              weights=batch_sizes)
        else:
            averages = []
            for i in range(len(outs)):
                averages.append(np.average([out[i] for out in all_outs],
                                           weights=batch_sizes))
            return averages
