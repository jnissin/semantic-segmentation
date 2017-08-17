import keras
import keras.backend as K
import numpy as np


class StepwiseLearningRateScheduler(keras.callbacks.Callback):
    """
    Stepwise Learning rate scheduler.

    # Arguments
        lr_schedule: a function that takes a step index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
        b2_schedule: a function that takes a step index as input
            (integer, indexed from 0) and returns a new
            beta 2 value as output (float).
        last_scheduled_step: step index of the last step to be scheduled (integer)
        verbose: should we print the values on each update
    """

    def __init__(self, lr_schedule, b2_schedule=None, last_scheduled_step=None, verbose=False):
        assert(lr_schedule is not None)

        super(StepwiseLearningRateScheduler, self).__init__()
        self.lr_schedule = lr_schedule
        self.b2_schedule = b2_schedule
        self.step_index = 0
        self.last_scheduled_step = int(last_scheduled_step) if last_scheduled_step is not None else None
        self.stop_reported = False
        self.verbose = verbose

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Check whether the last scheduled step has been reached
        if self.last_scheduled_step is not None and self.step_index > self.last_scheduled_step:
            if not self.stop_reported:
                print 'Stop schedule limit reached, stopping scheduling at step: {}, current learning rate: {}'.format(self.step_index, K.get_value(self.model.optimizer.lr))
                self.stop_reported = True
            return

        lr = self.lr_schedule(self.step_index)

        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be float.')

        if self.verbose:
            print 'StepwiseLearningRateScheduler: Step {}, setting learning rate to: {}'.format(self.step_index, lr)

        # Set learning rate value
        K.set_value(self.model.optimizer.lr, lr)

        # Set Adam beta 2 value
        if self.b2_schedule and isinstance(self.model.optimizer, keras.optimizers.Adam):
            b2 = self.b2_schedule(self.step_index)
            K.set_value(self.model.optimizer.beta_2, b2)

            if self.verbose:
                print 'StepwiseLearningRateScheduler: Step {}, setting Adam beta 2 to: {}'.format(self.step_index, b2)

        # Increase the step index
        self.step_index += 1
