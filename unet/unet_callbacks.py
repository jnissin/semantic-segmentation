import keras
import os
import time

"""
This class prints information to a file after every batch. Useful
for observing progress somewhere where the standard output is not
readily available for observation such as the Triton cluster.
"""


class FileMonitor(keras.callbacks.Callback):
    """
    # Arguments
        log_file_path: Path to the log file that should be created
    """
    def __init__(self,
                 log_file_path,
                 steps_per_epoch,
                 buffer_size=3):
        self.log_file_path = log_file_path
        self.steps_per_epoch = steps_per_epoch
        self.buffer_size = 3

        self.buffer = []
        self.log_file = None
        self.last_batch_start_time = None

    def on_train_begin(self, logs=None):
        # Create and open the log file
        if not self.log_file:
            if self.log_file_path:
                dirname = os.path.dirname(self.log_file_path)
                if not os.path.exists(dirname) and dirname != '':
                    os.makedirs(dirname)

                self.log_file = open(self.log_file_path, 'w')

    def on_train_end(self, logs=None):
        # Close the log file
        if self.log_file is not None:
            self.log_file.close()

    def on_batch_begin(self, batch, logs=None):
        self.last_batch_start_time = time.time()

    def on_batch_end(self, batch, logs={}):

        acc = logs.get('acc', None)
        loss = logs.get('loss', None)
        batch_num = logs.get('batch', None)
        mpca = logs.get('mpca', None)
        miou = logs.get('miou', None)
        eta = int((time.time() - self.last_batch_start_time) * (self.steps_per_epoch - (batch_num+1)))

        log_str = "Batch: {}/{} - ETA: {}s - loss: {} - acc: {} - mIoU: {}, mPCA: {}\n"\
            .format(batch_num+1, self.steps_per_epoch, eta, loss, acc, miou, mpca)

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(log_str)
        else:
            self.buffer = self.buffer[1:] + [log_str]

        # Clear the contents of the log file and write the buffer
        self.log_file.seek(0)
        self.log_file.truncate()

        for line in self.buffer:
            self.log_file.write(line)

        self.log_file.flush()

