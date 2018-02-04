from keras.callbacks import Callback, ProgbarLogger


class ExtendedBaseLogger(Callback):
    """Callback that accumulates epoch averages of metrics.

    This callback is automatically applied to every Keras model.
    """

    def _is_streaming_metric(self, metric_name):
        if metric_name is not None and self.model is not None:
            return metric_name in self.model.metrics_streaming

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if self._is_streaming_metric(k):
                self.totals[k] = k
            else:
                if k in self.totals:
                    self.totals[k] += v * batch_size
                else:
                    self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    if self._is_streaming_metric(k):
                        logs[k] = self.totals[k]
                    else:
                        # Make value available to next callbacks.
                        logs[k] = self.totals[k] / self.seen


class ExtendedProgbarLogger(ProgbarLogger):
    def _filter_log_items(self, logs):
        filtered_logs = None

        # Filter any hidden metrics from the logs
        if logs is not None:
            filtered_logs = {}

            for k, v in logs.items():
                if k not in self.model.metrics_hidden:
                    filtered_logs[k] = v

        return filtered_logs

    def __init__(self, count_mode='samples'):
        self.log_item_key_to_metric_index = None
        super(ExtendedProgbarLogger, self).__init__(count_mode=count_mode)

    def on_train_begin(self, logs=None):
        super(ExtendedProgbarLogger, self).on_train_begin(logs=self._filter_log_items(logs))

    def on_epoch_begin(self, epoch, logs=None):
        super(ExtendedProgbarLogger, self).on_epoch_begin(epoch=epoch, logs=self._filter_log_items(logs))

    def on_batch_begin(self, batch, logs=None):
        super(ExtendedProgbarLogger, self).on_batch_begin(batch=batch, logs=self._filter_log_items(logs))

    def on_batch_end(self, batch, logs=None):
        super(ExtendedProgbarLogger, self).on_batch_end(batch=batch, logs=self._filter_log_items(logs))
