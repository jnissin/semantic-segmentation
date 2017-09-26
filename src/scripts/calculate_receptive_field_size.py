# coding=utf-8

import math
from abc import ABCMeta, abstractmethod


class ReceptiveField(object):

    def __init__(self, n, j, r, start):
        # type: (int, int, int, float) -> None

        self.n = n          # Number of features (dimension)
        self.j = j          # Jump (distance between two consecutive features)
        self.r = r          # Receptive field size
        self.start = start  # Center coordinate of the first feature


class Receptive(object):
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_receptive_field(self, rf_in):
        # type: (ReceptiveField) -> ReceptiveField
        raise NotImplementedError('Not implemented')


class Network(Receptive):
    def __init__(self, name, receptive_blocks):
        # type: (str, list[Receptive]) -> None
        super(Network, self).__init__(name)
        self.receptive_blocks = receptive_blocks

    def get_receptive_field(self, rf_in, verbose=False):
        rf = rf_in
        if verbose:
            print 'RF of {}: n: {}, j: {}, r: {}, start: {}'.format('input', rf.n, rf.j, rf.r, rf.start)

        for r in self.receptive_blocks:
            rf = r.get_receptive_field(rf)

            if verbose:
                print 'RF of {}: n: {}, j: {}, r: {}, start: {}'.format(r.name, rf.n, rf.j, rf.r, rf.start)

        return rf


class ResidualBlock(Receptive):

    def __init__(self, name, path_a, path_b):
        # type: (str, list[Receptive], list[Receptive]) -> None
        super(ResidualBlock, self).__init__(name)
        self.path_a = path_a
        self.path_b = path_b

    def get_receptive_field(self, rf_in):
        # type: (ReceptiveField) -> ReceptiveField
        path_a_rf = None
        path_b_rf = None

        for r in self.path_a:
            path_a_rf = r.get_receptive_field(rf_in if path_a_rf is None else path_a_rf)

        for r in self.path_b:
            path_b_rf = r.get_receptive_field(rf_in if path_b_rf is None else path_b_rf)

        path_a_rf = rf_in if path_a_rf is None else path_a_rf
        path_b_rf = rf_in if path_b_rf is None else path_b_rf

        # Select the maximum receptive field and return it since
        # residual blocks usually mix multiple receptive fields
        # ... "Stacking a convolution or an Inception module on top of that feature volume will blend
        # features with different receptive field sizes together. So, a quick solution is that we represent
        # an Inception module as a conv layer with same padding and the kernel = the largest kernel of its
        # conv components"
        if path_a_rf.r > path_b_rf.r:
            out = path_a_rf
        else:
            out = path_b_rf

        return out


class Convolution2DLayer(Receptive):

    def __init__(self, name, kernel_size, stride, padding, dilation=1):
        # type: (str, int, int, str, int) -> None
        super(Convolution2DLayer, self).__init__(name)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def get_receptive_field(self, rf_in):
        # type: (ReceptiveField) -> ReceptiveField
        k = self.effective_kernel_size
        p = self.padding_size
        s = self.stride

        n_out = int(math.floor((rf_in.n - k + 2 * p) / s) + 1)
        actualP = (n_out - 1) * s - rf_in.n + k
        pR = math.ceil(actualP / 2)
        pL = math.floor(actualP / 2)
        j_out = rf_in.j * s
        r_out = rf_in.r + (k - 1) * rf_in.j
        start_out = rf_in.start + ((k - 1) / 2 - pL) * rf_in.j

        return ReceptiveField(n=n_out, j=j_out, r=r_out, start=start_out)

    @property
    def effective_kernel_size(self):
        # Apply possible dilation to effective kernel size
        k = self.kernel_size
        d = self.dilation

        if self.dilation > 1:
            k = k + (k-1) * (d-1)

        return k

    @property
    def padding_size(self):
        # type: () -> int

        if self.padding == 'same':
            p = int(math.floor(self.effective_kernel_size/2))
        elif self.padding == 'valid':
            p = 0
        else:
            raise ValueError('Invalid padding type')

        return p


class TransposedConvolution2DLayer(Receptive):
    def __init__(self, name, kernel_size, stride, padding, dilation=1):
        # type: (str, int, int, str, int) -> None
        super(TransposedConvolution2DLayer, self).__init__(name)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def get_receptive_field(self, rf_in):
        # type: (ReceptiveField) -> ReceptiveField

        # According to https://arxiv.org/pdf/1603.07285.pdf it is always possible to emulate a transposed
        # convolution with a direct convolution

        # Applying a transposed convolution with kernel size k, stride s and padding size p is equivalent to
        # applying a convolution with kernel size k’ = k, on a stretched input obtained by adding s-1 zeros in
        # between each input unit, with stride s’ = 1, and padding p’ = k-p-1.

        # Calculate the number of zeros added to the top and right edges of the input
        a = (rf_in.n + 2*self.padding_size - self.effective_kernel_size) % self.stride

        # Add zeros between the input data and calculate new input size
        i_dot = rf_in.n + (self.stride-1)*(rf_in.n-1)
        k = self.effective_kernel_size
        s = 1
        p = self.effective_kernel_size - self.padding_size - 1

        # Calculate the output size
        n_out = s * (i_dot - 1) + a + k - 2*p

        actualP = (n_out - 1) * s - i_dot + k
        pR = math.ceil(actualP / 2)
        pL = math.floor(actualP / 2)
        j_out = rf_in.j * s
        r_out = rf_in.r + (k - 1) * rf_in.j
        start_out = rf_in.start + max(((k - 1) / 2 - pL), 0) * rf_in.j

        # The computed receptive field size has to be divided by s because of the stretching. That size is a fractional
        # number, and in some cases can be smaller than 1, especially when the kernel size k is small. It means one output
        # is affected by only one input.
        r_out = int(min(math.floor(r_out/self.stride), 1))

        return ReceptiveField(n=n_out, j=j_out, r=r_out, start=start_out)

    @property
    def effective_kernel_size(self):
        # Apply possible dilation to effective kernel size
        k = self.kernel_size
        d = self.dilation

        if self.dilation > 1:
            k = k + (k - 1) * (d - 1)

        return k

    @property
    def padding_size(self):
        # type: () -> int

        if self.padding == 'same':
            p = int(math.floor(self.effective_kernel_size / 2))
        elif self.padding == 'valid':
            p = 0
        else:
            raise ValueError('Invalid padding type')

        return p


class PoolingLayer(Receptive):

    def __init__(self, name, pool_size, stride, padding='same'):
        # type: (str, int, int, str) -> None
        super(PoolingLayer, self).__init__(name)

        self.pool_size = pool_size
        self.padding = padding
        self.stride = stride

    def get_receptive_field(self, rf_in):
        # type: (ReceptiveField) -> ReceptiveField
        k = self.pool_size
        s = self.stride
        p = self.get_padding_size(rf_in.n)

        n_out = rf_in.n/k
        actualP = (n_out - 1) * s - rf_in.n + k
        pR = math.ceil(actualP / 2)
        pL = math.floor(actualP / 2)
        j_out = rf_in.j * s
        r_out = k * rf_in.r
        start_out = rf_in.start + ((k - 1) / 2 - pL) * rf_in.j

        return ReceptiveField(n=n_out, j=j_out, r=r_out, start=start_out)

    def get_padding_size(self, input_size):
        if self.padding == 'same':
            p = input_size % self.pool_size
        elif self.padding == 'valid':
            p = 0

        return p


class UpsamplingLayer(Receptive):

    def __init__(self, name, upsampling_factor):
        # type: (str, int) -> None
        super(UpsamplingLayer, self).__init__(name)
        self.upsampling_factor = upsampling_factor

    def get_receptive_field(self, rf_in):
        # type: (ReceptiveField) -> ReceptiveField
        k = self.upsampling_factor

        n_out = rf_in.n * k
        j_out = rf_in.j
        r_out = rf_in.r/self.upsampling_factor
        start_out = rf_in.start + ((k - 1) / 2 - 0) * rf_in.j

        return ReceptiveField(n=n_out, j=j_out, r=r_out, start=start_out)


# Convolutional layers
conv_1x1_1 = Convolution2DLayer('conv1x1_1', 1, 1, 'same')
conv_1x1_2 = Convolution2DLayer('conv1x1_2', 1, 2, 'same')
conv_2x2 = Convolution2DLayer('conv2x2', 2, 2, 'same')
conv_3x3 = Convolution2DLayer('conv3x3', 3, 1, 'same')
conv_5x5 = Convolution2DLayer('conv5x5', 5, 1, 'same')
conv_3x3_dilated_2 = Convolution2DLayer('conv3x3', 3, 1, 'same', 2)
conv_3x3_dilated_4 = Convolution2DLayer('conv3x3', 3, 1, 'same', 4)
conv_3x3_dilated_8 = Convolution2DLayer('conv3x3', 3, 1, 'same', 8)
conv_3x3_dilated_16 = Convolution2DLayer('conv3x3', 3, 1, 'same', 16)

# Transposed convolutional ayers
fullconv = TransposedConvolution2DLayer('fullconv', 2, 2, 'same')
tconv_3x3 = TransposedConvolution2DLayer('tconv3x3', 3, 2, 'same')

# Pooling layers
pool_2x2 = PoolingLayer('pool2x2', 2, 2, 'valid')

# Upsampling layers
upsampling_2 = UpsamplingLayer('upsampling_2', 2)

# Initial block
initial_block = ResidualBlock('initial_block', [Convolution2DLayer('conv3x3', 3, 2, 'same')], [pool_2x2])

# Bottlenecks
bn_conv_downsampling = ResidualBlock('bn_conv_downsampling', [conv_1x1_2, conv_3x3, conv_1x1_1], [pool_2x2])
bn_conv_regular = ResidualBlock('bn_conv_regular', [conv_1x1_1, conv_3x3, conv_1x1_1], [])
bn_conv_asymmetric_5 = ResidualBlock('bn_conv_asymmetric_5', [conv_1x1_1, conv_5x5, conv_1x1_1], [])
bn_conv_dilated_2 = ResidualBlock('bn_conv_dilated_2', [conv_1x1_1, conv_3x3_dilated_2, conv_1x1_1], [])
bn_conv_dilated_4 = ResidualBlock('bn_conv_dilated_4', [conv_1x1_1, conv_3x3_dilated_4, conv_1x1_1], [])
bn_conv_dilated_8 = ResidualBlock('bn_conv_dilated_8', [conv_1x1_1, conv_3x3_dilated_8, conv_1x1_1], [])
bn_conv_dilated_16 = ResidualBlock('bn_conv_dilated_16', [conv_1x1_1, conv_3x3_dilated_16, conv_1x1_1], [])

de_bn_conv_upsampling = ResidualBlock('de_bn_conv_upsampling', [conv_1x1_1, tconv_3x3, conv_1x1_1], [upsampling_2])
de_bn_conv_regular = ResidualBlock('de_bn_conv_regular', [conv_1x1_1, conv_3x3, conv_1x1_1], [])

# Network
network = Network('ENet',
                  receptive_blocks=
                  [
                       # Encoder
                       initial_block,              # initial
                       bn_conv_downsampling,       # bn 1.0
                       bn_conv_regular,            # bn 1.1
                       bn_conv_regular,            # bn 1.2
                       bn_conv_regular,            # bn 1.3
                       bn_conv_regular,            # bn 1.4
                       bn_conv_downsampling,       # bn 2.0
                       bn_conv_regular,            # bn 2.1
                       bn_conv_dilated_2,          # bn 2.2
                       bn_conv_asymmetric_5,       # bn 2.3
                       bn_conv_dilated_4,          # bn 2.4
                       bn_conv_regular,            # bn 2.5
                       bn_conv_dilated_8,          # bn 2.6
                       bn_conv_asymmetric_5,       # bn 2.7
                       bn_conv_dilated_16,         # bn 2.8
                       bn_conv_regular,            # bn 3.0
                       bn_conv_dilated_2,          # bn 3.1
                       bn_conv_asymmetric_5,       # bn 3.2
                       bn_conv_dilated_4,          # bn 3.3
                       bn_conv_regular,            # bn 3.4
                       bn_conv_dilated_8,          # bn 3.5
                       bn_conv_asymmetric_5,       # bn 3.6
                       bn_conv_dilated_16,         # bn 3.7

                       # Decoder
                       de_bn_conv_upsampling,      # bn 4.0
                       de_bn_conv_regular,         # bn 4.1
                       de_bn_conv_regular,         # bn 4.2
                       de_bn_conv_upsampling,      # bn 5.0
                       de_bn_conv_regular,         # bn 5.1

                       # Full conv
                       fullconv                    # fullconv
                   ])

# Input data
n_0 = 512
r_0 = 1
j_0 = 1
start_0 = 0.5
input_layer = ReceptiveField(n_0, j_0, r_0, start_0)


def main():
    print 'Starting receptive field calculation'
    rf = network.get_receptive_field(rf_in=input_layer, verbose=True)
    print 'Receptive field calculation complete'
    print 'Theoretical receptive field of {}: n: {}, r: {}, j: {}, start: {}'.format(network.name, rf.n, rf.r, rf.j, rf.start)


if __name__ == '__main__':
    main()
