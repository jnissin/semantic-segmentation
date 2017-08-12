import keras
import keras.backend as K
import numpy as np


class StepwiseLearningRateScheduler(keras.callbacks.Callback):
    """
    Stepwise Learning rate scheduler.

    # Arguments
        schedule: a function that takes a step index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """

    def __init__(self, schedule, last_scheduled_step=None, verbose=False):
        super(StepwiseLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.step_index = 0
        self.last_scheduled_step = int(last_scheduled_step) if last_scheduled_step is not None else None
        self.stop_reported = True
        self.verbose = verbose

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        if self.last_scheduled_step is not None and self.step_index > self.last_scheduled_step:
            if not self.stop_reported:
                print 'Stop schedule limit reached - stopping scheduling at step: {}'.format(self.step_index)
                self.stop_reported = True
            return

        lr = self.schedule(self.step_index)

        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be float.')

        if self.verbose:
            print 'StepwiseLearningRateScheduler: Step {}, setting learning rate to: {}'.format(self.step_index, lr)

        K.set_value(self.model.optimizer.lr, lr)
        self.step_index += 1
