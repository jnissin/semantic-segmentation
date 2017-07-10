import keras
import os
import json


class OptimizerCheckpoint(keras.callbacks.Callback):
    """
    Logs the current configuration of the optimizer to the selected configuration
    (JSON) file at the beginning of every epoch. This ensures that we can continue
    with the same optimizer configuration in a different run.
    """

    def __init__(self, log_file_path):
        # type: (str) -> ()

        """
        # Arguments
           log_file_path: Path to the log file that should be created
        """
        super(OptimizerCheckpoint, self).__init__()

        self.log_file_path = log_file_path
        self.log_file = None

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

    def on_epoch_begin(self, epoch, logs=None):
        optimizer_config = self.model.optimizer.get_config()
        json_str = json.dumps(optimizer_config)

        # Clear the contents of the log file
        self.log_file.seek(0)
        self.log_file.truncate()

        # Write the JSON and flush
        self.log_file.write(json_str)
        self.log_file.flush()
