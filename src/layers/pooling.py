# coding=utf-8

from keras import backend as K
from keras.layers import Layer


class MaxPoolingWithArgmax2D(Layer):

    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)

        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.running_on_gpu = False

        if K.backend() is not 'tensorflow':
            raise NotImplementedError('{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__))

        # Check whether we are running on GPU to decide which version of pooling to use
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        self.running_on_gpu = len(gpus) > 0

        if not self.running_on_gpu:
            raise NotImplementedError('MaxPoolingWithArgmax2D works only on GPU')

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(MaxPoolingWithArgmax2D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        pool_size = self.pool_size
        padding = self.padding
        strides = self.strides

        if K.backend() == 'tensorflow':
            # tf.nn.max_pool_with_argmax works only on GPU
            # See: https://stackoverflow.com/questions/39493229/how-to-use-tf-nn-max-pool-with-argmax-correctly
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)
        else:
            raise NotImplementedError('{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__))

        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim / ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):

    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size
        self.kwargs = kwargs

    def build(self, input_shape):
        super(MaxUnpooling2D, self).build(input_shape)

    def call(self, inputs, output_shape=None):
        """
        Seen on https://github.com/tensorflow/tensorflow/issues/2169
        Replace with unpool op when/if issue merged
        Add Theano backend
        """
        pool, ind = inputs[0], inputs[1]

        with tf.variable_scope(scope):
            input_shape = K.tf.shape(pool, out_type='int64')
            output_shape = [input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3]]

            flat_input_size = K.tf.reduce_prod(input_shape)
            flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

            pool_ = K.tf.reshape(pool, [flat_input_size])
            batch_range = K.tf.reshape(K.tf.range(tf.cast(output_shape[0], K.tf.int64), dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
            b = K.tf.ones_like(ind) * batch_range
            b1 = K.tf.reshape(b, [flat_input_size, 1])
            ind_ = K.tf.reshape(ind, [flat_input_size, 1])
            ind_ = K.tf.concat([b1, ind_], 1)

            ret = K.tf.scatter_nd(ind_, pool_, shape=K.tf.cast(flat_output_shape, K.tf.int64))
            ret = K.tf.reshape(ret, output_shape)

            set_input_shape = pool.get_shape()
            set_output_shape = [set_input_shape[0],
                                set_input_shape[1] * self.size[0],
                                set_input_shape[2] * self.size[1],
                                set_input_shape[3]]
            ret.set_shape(set_output_shape)
            return ret


    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        dim0 = mask_shape[0]
        dim1 = mask_shape[1] * self.size[0] if mask_shape[1] is not None else None
        dim2 = mask_shape[2] * self.size[1] if mask_shape[2] is not None else None
        dim3 = mask_shape[3]
        return dim0, dim1, dim2, dim3
