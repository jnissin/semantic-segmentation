from keras.callbacks import Callback, ProgbarLogger


class ExtendedBaseLogger(Callback):
    """Callback that accumulates epoch averages of metrics.

    This callback is automatically applied to every Keras model.
    """

    def _is_streaming_metric(self, metric_name):
        if metric_name is not None and self.model is not None:
            if metric_name.startswith('val_'):
                return (metric_name in self.model.metrics_streaming) or\
                       (metric_name[4:] in self.model.metrics_streaming)
            return metric_name in self.model.metrics_streaming

    def _is_excluded_from_callbacks(self, metric_name):
        if metric_name is not None and self.model is not None:
            if metric_name.startswith('val_'):
                return (metric_name in self.model.metrics_excluded_from_callbacks) or\
                       (metric_name[4:] in self.model.metrics_excluded_from_callbacks)
            return metric_name in self.model.metrics_excluded_from_callbacks

    def get_metric_value(self, k):
        if k in self.totals:
            # Streaming metrics are already averaged - return the value
            if self._is_streaming_metric(k):
                return self.totals[k]

            # Non-streaming metrics should be divided by the number of seen samples
            return self.totals[k] / self.seen

        return None

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if self._is_streaming_metric(k):
                self.totals[k] = v
            else:
                if k in self.totals:
                    self.totals[k] += v * batch_size
                else:
                    self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:

            for k in self.params['metrics']:
                if k in self.totals:
                    logs[k] = self.get_metric_value(k)

            # If this should be excluded from the rest of the callbacks - remove the key from the dict
            exclude_keys = []

            for k, v in logs.items():
                if self._is_excluded_from_callbacks(k):
                    exclude_keys.append(k)

            for k in exclude_keys:
                del logs[k]


class ExtendedProgbarLogger(ProgbarLogger):
    def _filter_log_items(self, logs):
        filtered_logs = None

        # Filter any hidden metrics from the logs
        if logs is not None:
            filtered_logs = {}

            for k, v in logs.items():
                if k.startswith('val_'):
                    if k[4:] not in self.model.metrics_hidden_from_progbar and k not in self.model.metrics_hidden_from_progbar:
                        filtered_logs[k] = v
                else:
                    if k not in self.model.metrics_hidden_from_progbar:
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
