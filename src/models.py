# coding=utf-8

from abc import ABCMeta, abstractmethod

import keras.backend as K

from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Lambda
from keras.layers.advanced_activations import PReLU
from keras.layers.core import SpatialDropout2D, Permute
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers import Input

from layers.pooling import MaxPoolingWithArgmax2D
from layers.pooling import MaxUnpooling2D

from keras_extensions.extended_model import ExtendedModel
from logger import Logger

import losses
from losses import ModelLambdaLossType


##############################################
# UTILITY CLASSES
##############################################

class WeightTransferInformation(object):

    def __init__(self, num_transferred_layers, last_transferred_layer_name, num_frozen_layers=0, last_frozen_layer_name=None, lr_scalers=dict()):
        # type: (int, str, int, str, dict) -> None

        self.num_transferred_layers = num_transferred_layers
        self.last_transferred_layer_name = last_transferred_layer_name
        self.num_frozen_layers = num_frozen_layers
        self.last_frozen_layer_name = last_frozen_layer_name
        self.lr_scalers = lr_scalers

    @property
    def num_lr_scaling_trainable_weights(self):
        return len(self.lr_scalers)


#############################################
# UTILITY FUNCTIONS
#############################################


def get_model(model_name,
              input_shape,
              num_classes,
              model_lambda_loss_type=ModelLambdaLossType.NONE):
    # type: (str, tuple(int), int, Logger, ModelLambdaLossType) -> ModelBase

    """
    Get the model by the model name.

    # Arguments
        model_name: name of the model
        input_shape: input shape to the model
        num_classes: number of classification classes
        model_type: type of the model - affects loss calculation
    # Returns
        The appropriate ModelBase.
    """

    if model_name == 'unet':
        model_wrapper = UNetModel(
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type)
    elif model_name == 'enet-naive-upsampling':
        model_wrapper = ENetNaiveUpsampling(
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type,
            encoder_only=False)
    elif model_name == 'enet-naive-upsampling-encoder-only':
        model_wrapper = ENetNaiveUpsampling(
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type,
            encoder_only=True)
    elif model_name == 'enet-max-unpooling':
        model_wrapper = ENetMaxUnpooling(
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type,
            encoder_only=False)
    elif model_name == 'enet-max-unpooling-encoder-only':
        model_wrapper = ENetMaxUnpooling(
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type,
            encoder_only=True)
    elif model_name == 'segnet':
        model_wrapper = SegNetBasicModel(
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type,
            encoder_only=False)
    elif model_name == 'segnet-encoder-only':
        model_wrapper = SegNetBasicModel(
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type,
            encoder_only=True)
    elif model_name == 'enet-naive-upsampling-enhanced':
        model_wrapper = ENetNaiveUpsamplingEnhanced(
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type,
            encoder_only=False)
    elif model_name == 'enet-naive-upsampling-enhanced-encoder-only':
        model_wrapper = ENetNaiveUpsamplingEnhanced(
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type,
            encoder_only=True)
    else:
        raise ValueError('Unknown model name: {}'.format(model_name))

    return model_wrapper


def get_lambda_loss_function(model_lambda_loss_type):
    # type: (ModelLambdaLossType) -> function

    if model_lambda_loss_type == ModelLambdaLossType.NONE:
        return None
    elif model_lambda_loss_type == ModelLambdaLossType.SEGMENTATION_CATEGORICAL_CROSS_ENTROPY:
        return losses.segmentation_categorical_cross_entropy_lambda_loss
    elif model_lambda_loss_type == ModelLambdaLossType.SEGMENTATION_SEMI_SUPERVISED_SUPERPIXEL:
        return losses.segmentation_superpixel_lambda_loss
    elif model_lambda_loss_type == ModelLambdaLossType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER:
        return losses.segmentation_mean_teacher_lambda_loss
    elif model_lambda_loss_type == ModelLambdaLossType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER_SUPERPIXEL:
        return losses.segmentation_mean_teacher_superpixel_lambda_loss
    elif model_lambda_loss_type == ModelLambdaLossType.CLASSIFICATION_CATEGORICAL_CROSS_ENTROPY:
        return losses.classification_categorical_crossentropy_lambda_loss
    elif model_lambda_loss_type == ModelLambdaLossType.CLASSIFICATION_SEMI_SUPERVISED_MEAN_TEACHER:
        return losses.classification_mean_teacher_lambda_loss

    raise ValueError('Unknown model lambda loss type: {}'.format(model_lambda_loss_type))


##############################################
# MODEL BASE
##############################################


class ModelBase(object):

    __metaclass__ = ABCMeta

    def __init__(self,
                 name,
                 input_shape,
                 num_classes,
                 model_lambda_loss_type=ModelLambdaLossType.NONE):
        # type: (str, tuple[int], int, ModelLambdaLossType) -> None

        self.name = name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_lambda_loss_type = model_lambda_loss_type

        # Deduce the lambda loss function from the model type
        self.lambda_loss_function = get_lambda_loss_function(model_lambda_loss_type)

        # Configure inputs
        images = Input(name="images", shape=self.input_shape, dtype='float32')
        self.inputs = [images]

        # Build the model
        self.outputs = self._build_model(images)

        # If we are using the mean teacher method and this is the student model or
        # we are using a semisupervised model - add the custom lambda loss layer
        if self.model_lambda_loss_type == ModelLambdaLossType.SEGMENTATION_CATEGORICAL_CROSS_ENTROPY:
            self.name = self.name + '-CCE'
            self._model = self._get_segmentation_categorical_cross_entropy_lambda_loss_model()
        if self.model_lambda_loss_type == ModelLambdaLossType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER:
            self.name = self.name + '-MT'
            self._model = self._get_segmentation_semi_supervised_mean_teacher_lambda_loss_model()
        elif self.model_lambda_loss_type == ModelLambdaLossType.SEGMENTATION_SEMI_SUPERVISED_SUPERPIXEL:
            self.name = self.name + '-SP'
            self._model = self._get_segmentation_semi_supervised_superpixel_lambda_loss_model()
        elif self.model_lambda_loss_type == ModelLambdaLossType.SEGMENTATION_SEMI_SUPERVISED_MEAN_TEACHER_SUPERPIXEL:
            self.name = self.name + '-MT-SP'
            self._model = self._get_segmentation_semi_supervised_mean_teacher_superpixel_lambda_loss_model()
        elif self.model_lambda_loss_type == ModelLambdaLossType.CLASSIFICATION_CATEGORICAL_CROSS_ENTROPY:
            self.name = self.name + '-CCE'
            self._model = self._get_classification_categorical_cross_entropy_lambda_loss_model()
        elif self.model_lambda_loss_type == ModelLambdaLossType.CLASSIFICATION_SEMI_SUPERVISED_MEAN_TEACHER:
            self.name = self.name + '-MT'
            self._model = self._get_classification_semi_supervised_mean_teacher_lambda_loss_model()
        else:
            # Otherwise just return the model
            self._model = ExtendedModel(name=self.name, inputs=self.inputs, outputs=self.outputs)

    @property
    def model(self):
        return self._model

    def transfer_weights(self,
                         from_model,
                         from_layer_index,
                         to_layer_index,
                         freeze_from_layer_index=None,
                         freeze_to_layer_index=None,
                         scale_lr_from_layer_index=None,
                         scale_lr_to_layer_index=None,
                         scale_lr_factor=None):
        # type: (Model, int, int, int, int, int, int, float) -> (WeightTransferInformation)

        """
        Transfers weights of the given layer range from the parameter model to this model.
        Note: compile should be called on the model only after this function in order for
        the freezing to take effect.

        # Arguments
            :param from_model: model where to transfer weights from
            :param from_layer_index: first layer to transfer
            :param to_layer_index: end of the transferrable layers
            :param freeze_from_layer_index: the first layer index to freeze
            :param freeze_to_layer_index: end of the frozen layers index
            :param scale_lr_from_layer_index: the first layer where to apply the scale_lr_factor
            :param scale_lr_to_layer_index: end of the learning rate scaling layers
            :param scale_lr_factor: scaling factor for the learning rate of the specified layers
        # Returns
            :return: a WeightTransferInformation object
        """

        num_frozen_layers = 0
        freezing_layers = freeze_from_layer_index is not None and freeze_to_layer_index is not None
        scaling_lr_layers = scale_lr_from_layer_index is not None and scale_lr_to_layer_index is not None
        num_from_model_layers = len(from_model.layers)

        # Support negative indexing
        if from_layer_index < 0:
            from_layer_index += num_from_model_layers

        if to_layer_index < 0:
            to_layer_index += num_from_model_layers

        if freeze_from_layer_index is not None and freeze_from_layer_index < 0:
            freeze_from_layer_index += num_from_model_layers

        if freeze_to_layer_index is not None and freeze_to_layer_index < 0:
            freeze_to_layer_index += num_from_model_layers

        if scale_lr_from_layer_index is not None and scale_lr_from_layer_index < 0:
            scale_lr_from_layer_index += num_from_model_layers

        if scale_lr_to_layer_index is not None and scale_lr_to_layer_index < 0:
            scale_lr_to_layer_index += num_from_model_layers

        # Sanity checks
        assert(from_layer_index < to_layer_index)
        assert(not (freezing_layers and freeze_from_layer_index > freeze_to_layer_index))
        assert(not (scaling_lr_layers and scale_lr_from_layer_index > scale_lr_to_layer_index))
        assert(not (scaling_lr_layers and scale_lr_factor is None))

        # Assumes indexing is the same for both models for the specified
        # layer range
        to_model_weights = self._model.get_weights()
        from_model_weights = from_model.get_weights()

        # Transfer weights
        to_model_weights[from_layer_index:to_layer_index] = from_model_weights[from_layer_index:to_layer_index]
        num_transferred_layers = to_layer_index-from_layer_index
        self._model.set_weights(to_model_weights)

        # Freeze layers if any
        if freezing_layers:
            for i in range(freeze_from_layer_index, freeze_to_layer_index):
                self._model.layers[i].trainable = False
                num_frozen_layers += 1
        else:
            num_frozen_layers = 0

        # Create learning rate scaler params
        lr_scalers = dict()

        if scaling_lr_layers:
            for i in range(scale_lr_from_layer_index, scale_lr_to_layer_index):
                lr_scalers[i] = scale_lr_factor

        info = WeightTransferInformation(num_transferred_layers=num_transferred_layers,
                                         last_transferred_layer_name=from_model.layers[to_layer_index - 1].name,
                                         num_frozen_layers=num_frozen_layers,
                                         last_frozen_layer_name=from_model.layers[freeze_to_layer_index - 1].name if freezing_layers else None,
                                         lr_scalers=lr_scalers)

        return info

    def transfer_weights_from_file(self,
                                   filepath,
                                   from_layer_index,
                                   to_layer_index,
                                   freeze_from_layer_index=None,
                                   freeze_to_layer_index=None,
                                   scale_lr_from_layer_index=None,
                                   scale_lr_to_layer_index=None,
                                   scale_lr_factor=None):
        # type: (str, int, int, int, int, int, int, float) -> WeightTransferInformation

        """Implements topological (order-based) weight loading.

        # Arguments
            :param filepath: path to weights HDF5 file
            :param from_layer_index: first layer to transfer
            :param to_layer_index: end of the transferrable layers
            :param freeze_from_layer_index: the first layer index to freeze
            :param freeze_to_layer_index: end of the frozen layers index
            :param scale_lr_from_layer_index: the first layer where to apply the scale_lr_factor
            :param scale_lr_to_layer_index: end of the learning rate scaling layers
            :param scale_lr_factor: scaling factor for the learning rate of the specified layers
        # Raises
            ValueError: in case of mismatch between provided layers
                and weights file.
        """

        ###########################
        # Utility functions
        ###########################

        def normalize_layer_index(idx, num_layers):
            if idx is None:
                return None
            elif idx < 0:
                return idx + num_layers
            return idx

        def get_trainable_layer_names_from_hdf5(hdf5_file):
            # Get all the layers from the HDF5 file
            layer_names = [n.decode('utf8') for n in hdf5_file.attrs['layer_names']]

            # Filter the layers which have weights
            trainable_layer_names = []

            for name in layer_names:
                g = hdf5_file[name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                if weight_names:
                    trainable_layer_names.append(name)

            return trainable_layer_names

        def get_trainable_layers_from_model(model):

            # Filter the layers which have weights
            layers = model.layers
            layers_with_weights = []

            for layer in layers:
                weights = layer.weights
                if weights:
                    layers_with_weights.append(layer)

            return layers_with_weights

        ###########################
        # Imports
        ###########################

        try:
            import h5py
        except ImportError:
            h5py = None

        from keras.engine.topology import preprocess_weights_for_loading

        ###########################
        # Read HDF5 file
        ###########################

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')

        with h5py.File(filepath, mode='r') as f:

            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']

            ##########################
            # Load weights
            ##########################

            if 'keras_version' in f.attrs:
                original_keras_version = f.attrs['keras_version'].decode('utf8')
            else:
                original_keras_version = '1'
            if 'backend' in f.attrs:
                original_backend = f.attrs['backend'].decode('utf8')
            else:
                original_backend = None

            # Get trainable weights from source HDF5 file and target model
            source_model_trainable_layer_names = get_trainable_layer_names_from_hdf5(f)
            target_model_trainable_layers = get_trainable_layers_from_model(self.model)

            # Transform layer indices from possibly negative to positive range [0, N_TRAINABLE]
            source_num_trainable_layers = len(source_model_trainable_layer_names)
            from_layer_index = normalize_layer_index(from_layer_index, source_num_trainable_layers)
            to_layer_index = normalize_layer_index(to_layer_index, source_num_trainable_layers)
            freeze_from_layer_index = normalize_layer_index(freeze_from_layer_index, source_num_trainable_layers)
            freeze_to_layer_index = normalize_layer_index(freeze_to_layer_index, source_num_trainable_layers)
            scale_lr_from_layer_index = normalize_layer_index(scale_lr_from_layer_index, source_num_trainable_layers)
            scale_lr_to_layer_index = normalize_layer_index(scale_lr_to_layer_index, source_num_trainable_layers)
            freezing_layers = freeze_from_layer_index is not None and freeze_to_layer_index is not None
            scaling_lr_layers = scale_lr_from_layer_index is not None and scale_lr_to_layer_index is not None

            # Select the corresponding layers from the source and target
            transfer_source_layer_names = source_model_trainable_layer_names[from_layer_index:to_layer_index]
            transfer_target_layers = target_model_trainable_layers[from_layer_index:to_layer_index]

            # Sanity check
            if len(transfer_source_layer_names) != len(transfer_target_layers):
                raise ValueError('Mismatch between number of target and source layers: {} vs {}'.format(len(transfer_target_layers), len(transfer_source_layer_names)))

            if transfer_target_layers[0].name != transfer_source_layer_names[0] or transfer_target_layers[-1].name != transfer_source_layer_names[-1]:
                Logger.instance().warn('Mismatch in first/last transferred layer names between target and source models: first: {} vs {}, last: {} vs {}'
                                       .format(transfer_target_layers[0].name, transfer_source_layer_names[0], transfer_target_layers[-1].name, transfer_source_layer_names[-1]))

            # Extract info for return value
            last_transferred_layer_name = transfer_source_layer_names[-1]
            num_transferred_layers = len(transfer_source_layer_names)

            # We batch weight value assignments in a single backend call
            # which provides a speedup in TensorFlow.
            weight_value_tuples = []

            for k, name in enumerate(transfer_source_layer_names):
                g = f[name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                weight_values = [g[weight_name] for weight_name in weight_names]
                layer = transfer_target_layers[k]
                symbolic_weights = layer.weights
                weight_values = preprocess_weights_for_loading(layer,
                                                               weight_values,
                                                               original_keras_version,
                                                               original_backend)

                if len(weight_values) != len(symbolic_weights):
                    raise ValueError('Layer #' + str(k) +
                                     ' (named "' + layer.name +
                                     '" in the current model) was found to '
                                     'correspond to layer ' + name +
                                     ' in the save file. '
                                     'However the new layer ' + layer.name +
                                     ' expects ' + str(len(symbolic_weights)) +
                                     ' weights, but the saved weights have ' +
                                     str(len(weight_values)) +
                                     ' elements.')
                weight_value_tuples += zip(symbolic_weights, weight_values)

            K.batch_set_value(weight_value_tuples)

            ##############################
            # Freezing
            ##############################

            num_frozen_layers = 0
            last_frozen_layer_name = None

            if freezing_layers:
                freeze_target_layers = target_model_trainable_layers[freeze_from_layer_index:freeze_to_layer_index]
                last_frozen_layer_name = freeze_target_layers[-1].name
                num_frozen_layers = len(freeze_target_layers)

                for layer in freeze_target_layers:
                    layer.trainable = False

            ##############################
            # LR scaling
            ##############################

            lr_scalers = dict()

            if scaling_lr_layers:
                # The learning rate scalers must be mapped from layer indices to trainable weights indices which
                # are used in the optimizer
                scale_lr_target_layers = target_model_trainable_layers[scale_lr_from_layer_index:scale_lr_to_layer_index]

                scale_lr_from_trainable_weight = 0
                for i in range(0, scale_lr_from_layer_index):
                    scale_lr_from_trainable_weight += len(target_model_trainable_layers[i].trainable_weights)

                scale_lr_to_trainable_weight = scale_lr_from_trainable_weight
                for layer in scale_lr_target_layers:
                    scale_lr_to_trainable_weight += len(layer.trainable_weights)

                # Create a dictionary
                for i in range(scale_lr_from_trainable_weight, scale_lr_to_trainable_weight):
                    lr_scalers[i] = scale_lr_factor

            ###############################
            # Return params
            ###############################

            info = WeightTransferInformation(num_transferred_layers=num_transferred_layers,
                                             last_transferred_layer_name=last_transferred_layer_name,
                                             num_frozen_layers=num_frozen_layers,
                                             last_frozen_layer_name=last_frozen_layer_name,
                                             lr_scalers=lr_scalers)

            return info



    def load_optimizer_weights(self, weights_file_path):

        if self.model is None:
            raise ValueError('The internal model must be built before loading optimizer weights')

        if self.model.optimizer is None:
            raise ValueError('The model must be compiled with an optimizer before loading optimizer weights')

        ###########################
        # Imports
        ###########################

        try:
            import h5py
        except ImportError:
            h5py = None

        if h5py is None:
            raise ImportError('`load_optimizer_weights` requires h5py.')

        ###########################
        # Load weights
        ###########################

        with h5py.File(weights_file_path, mode='r') as f:
            # Set optimizer weights.
            if 'optimizer_weights' in f:
                # Build train function (to get weight updates).
                if isinstance(self.model, Sequential):
                    self.model.model._make_train_function()
                else:
                    self.model._make_train_function()

                optimizer_weights_group = f['optimizer_weights']
                optimizer_weight_names = [n.decode('utf8') for n in
                                          optimizer_weights_group.attrs['weight_names']]
                optimizer_weight_values = [optimizer_weights_group[n] for n in
                                           optimizer_weight_names]
                try:
                    self.model.optimizer.set_weights(optimizer_weight_values)
                except ValueError:
                    return False
            else:
                # No optimizer weights in the file
                return False

        return True

    @abstractmethod
    def _build_model(self, inputs):
        """
        Builds the model and returns the inputs and outputs as lists. This method does not
        take into account additional inputs due to being possible Mean Teacher student model.

        # Arguments
            :param inputs: The previous layer i.e. an input layer
        # Returns
            :return: inputs and outputs of the model as a tuple of two lists [inputs, outputs]
        """

        raise NotImplementedError('This method should be implemented in the class derived from ModelBase')

    def _get_segmentation_categorical_cross_entropy_lambda_loss_model(self):
        if self.inputs is None or self.outputs is None:
            raise RuntimeError('The model must be built by calling _build_model() first')
        if self.lambda_loss_function is None:
            raise ValueError('Categorical cross entropy models must be given a lambda loss function')

        labels_shape = (self.input_shape[0], self.input_shape[1])
        labels = Input(name="labels", shape=labels_shape, dtype='int32')
        self.inputs.append(labels)

        class_weights = Input(name="class_weights", shape=labels_shape, dtype='float32')
        self.inputs.append(class_weights)

        num_unlabeled = Input(name='num_unlabeled', shape=[1], dtype='int32')
        self.inputs.append(num_unlabeled)

        # Note: assumes there is only a single output, which is the last layer
        logits = self.outputs[0]
        lambda_inputs = [logits, labels, class_weights, num_unlabeled]
        loss_layer = Lambda(self.lambda_loss_function, output_shape=(1,), name='loss')(lambda_inputs)
        self.outputs = [loss_layer, logits]

        model = ExtendedModel(name=self.name, inputs=self.inputs, outputs=self.outputs)
        return model

    def _get_segmentation_semi_supervised_mean_teacher_lambda_loss_model(self):

        if self.inputs is None or self.outputs is None:
            raise RuntimeError('The model must be built by calling _build_model() first')
        if self.lambda_loss_function is None:
            raise ValueError('Mean teacher models must be given a lambda loss function')

        labels_shape = (self.input_shape[0], self.input_shape[1])
        logits_shape = (self.input_shape[0], self.input_shape[1], self.num_classes)

        labels = Input(name="labels", shape=labels_shape, dtype='int32')
        self.inputs.append(labels)

        class_weights = Input(name="class_weights", shape=labels_shape, dtype='float32')
        self.inputs.append(class_weights)

        num_unlabeled = Input(name='num_unlabeled', shape=[1], dtype='int32')
        self.inputs.append(num_unlabeled)

        mt_predictions = Input(name="mt_predictions", shape=logits_shape, dtype='float32')
        self.inputs.append(mt_predictions)

        consistency_cost = Input(name="consistency_cost", shape=[1], dtype='float32')
        self.inputs.append(consistency_cost)

        # Note: assumes there is only a single output, which is the last layer
        logits = self.outputs[0]
        lambda_inputs = [logits, labels, class_weights, num_unlabeled, mt_predictions, consistency_cost]
        loss_layer = Lambda(self.lambda_loss_function, output_shape=(1,), name='loss')(lambda_inputs)
        self.outputs = [loss_layer, logits]

        model = ExtendedModel(name=self.name, inputs=self.inputs, outputs=self.outputs)
        return model

    def _get_segmentation_semi_supervised_superpixel_lambda_loss_model(self):

        if self.inputs is None or self.outputs is None:
            raise RuntimeError('The model must be built by calling _build_model() first')
        if self.lambda_loss_function is None:
            raise ValueError('Semi-supervised models must be given a lambda loss function')

        labels_shape = (self.input_shape[0], self.input_shape[1])

        labels = Input(name="labels", shape=labels_shape, dtype='int32')
        self.inputs.append(labels)

        class_weights = Input(name="class_weights", shape=labels_shape, dtype='float32')
        self.inputs.append(class_weights)

        num_unlabeled = Input(name='num_unlabeled', shape=[1], dtype='int32')
        self.inputs.append(num_unlabeled)

        unlabeled_cost_coeff = Input(name='unlabeled_cost_coeff', shape=[1], dtype='float32')
        self.inputs.append(unlabeled_cost_coeff)

        # Note: assumes there is only a single output, which is the last layer
        logits = self.outputs[0]
        lambda_inputs = [logits, labels, class_weights, num_unlabeled, unlabeled_cost_coeff]
        loss_layer = Lambda(self.lambda_loss_function, output_shape=(1,), name='loss')(lambda_inputs)
        self.outputs = [loss_layer, logits]

        model = ExtendedModel(name=self.name, inputs=self.inputs, outputs=self.outputs)
        return model

    def _get_segmentation_semi_supervised_mean_teacher_superpixel_lambda_loss_model(self):

        if self.inputs is None or self.outputs is None:
            raise RuntimeError('The model must be built by calling _build_model() first')
        if self.lambda_loss_function is None:
            raise ValueError('Mean teacher models must be given a lambda loss function')

        labels_shape = (self.input_shape[0], self.input_shape[1])
        logits_shape = (self.input_shape[0], self.input_shape[1], self.num_classes)

        labels = Input(name="labels", shape=labels_shape, dtype='int32')
        self.inputs.append(labels)

        class_weights = Input(name="class_weights", shape=labels_shape, dtype='float32')
        self.inputs.append(class_weights)

        num_unlabeled = Input(name='num_unlabeled', shape=[1], dtype='int32')
        self.inputs.append(num_unlabeled)

        mt_predictions = Input(name="mt_predictions", shape=logits_shape, dtype='float32')
        self.inputs.append(mt_predictions)

        consistency_cost = Input(name="consistency_cost", shape=[1], dtype='float32')
        self.inputs.append(consistency_cost)

        unlabeled_cost_coeff = Input(name="unlabeled_cost_coeff", shape=[1], dtype='float32')
        self.inputs.append(unlabeled_cost_coeff)

        # Note: assumes there is only a single output, which is the last layer
        logits = self.outputs[0]
        lambda_inputs = [logits, labels, class_weights, num_unlabeled, mt_predictions, consistency_cost, unlabeled_cost_coeff]
        loss_layer = Lambda(self.lambda_loss_function, output_shape=(1,), name='loss')(lambda_inputs)
        self.outputs = [loss_layer, logits]

        model = ExtendedModel(name=self.name, inputs=self.inputs, outputs=self.outputs)
        return model

    def _get_classification_categorical_cross_entropy_lambda_loss_model(self):
        if self.inputs is None or self.outputs is None:
            raise RuntimeError('The model must be built by calling _build_model() first')
        if self.lambda_loss_function is None:
            raise ValueError('Categorical cross entropy models must be given a lambda loss function')

        labels_shape = [self.num_classes]
        labels = Input(name="labels", shape=labels_shape, dtype='float32')
        self.inputs.append(labels)

        class_weights = Input(name="class_weights", shape=labels_shape, dtype='float32')
        self.inputs.append(class_weights)

        num_unlabeled = Input(name='num_unlabeled', shape=[1], dtype='int32')
        self.inputs.append(num_unlabeled)

        # Note: assumes there is only a single output, which is the last layer
        logits = self.outputs[0]
        lambda_inputs = [logits, labels, class_weights, num_unlabeled]
        loss_layer = Lambda(self.lambda_loss_function, output_shape=(1,), name='loss')(lambda_inputs)
        self.outputs = [loss_layer, logits]

        model = ExtendedModel(name=self.name, inputs=self.inputs, outputs=self.outputs)
        return model

    def _get_classification_semi_supervised_mean_teacher_lambda_loss_model(self):

        if self.inputs is None or self.outputs is None:
            raise RuntimeError('The model must be built by calling _build_model() first')
        if self.lambda_loss_function is None:
            raise ValueError('Mean teacher models must be given a lambda loss function')

        labels_shape = [self.num_classes]
        logits_shape = [self.num_classes]

        labels = Input(name="labels", shape=labels_shape, dtype='float32')
        self.inputs.append(labels)

        class_weights = Input(name="class_weights", shape=labels_shape, dtype='float32')
        self.inputs.append(class_weights)

        num_unlabeled = Input(name='num_unlabeled', shape=[1], dtype='int32')
        self.inputs.append(num_unlabeled)

        mt_predictions = Input(name="mt_predictions", shape=logits_shape, dtype='float32')
        self.inputs.append(mt_predictions)

        consistency_cost = Input(name="consistency_cost", shape=[1], dtype='float32')
        self.inputs.append(consistency_cost)

        # Note: assumes there is only a single output, which is the last layer
        logits = self.outputs[0]
        lambda_inputs = [logits, labels, class_weights, num_unlabeled, mt_predictions, consistency_cost]
        loss_layer = Lambda(self.lambda_loss_function, output_shape=(1,), name='loss')(lambda_inputs)
        self.outputs = [loss_layer, logits]

        model = ExtendedModel(name=self.name, inputs=self.inputs, outputs=self.outputs)
        return model


##############################################
# UNET
##############################################


class UNetModel(ModelBase):
    """
    The UNet model presented in the paper:

    https://arxiv.org/pdf/1505.04597.pdf
    """

    def __init__(self,
                 input_shape,
                 num_classes,
                 model_lambda_loss_type=ModelLambdaLossType.NONE):

        super(UNetModel, self).__init__(
            name="UNet",
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type)

    def _build_model(self, inputs):
        '''
        Contracting path
        '''
        conv1, pool1 = UNetModel.get_encoder_block('encoder_block1', 2, 64, inputs)
        conv2, pool2 = UNetModel.get_encoder_block('encoder_block2', 2, 128, pool1)
        conv3, pool3 = UNetModel.get_encoder_block('encoder_block3', 2, 256, pool2)
        conv4, pool4 = UNetModel.get_encoder_block('encoder_block4', 2, 512, pool3)

        '''
        Connecting path
        '''
        conv5 = UNetModel.get_convolution_block(
            num_filters=1024,
            input_layer=pool4,
            name='connecting_block_conv1')

        conv5 = UNetModel.get_convolution_block(
            num_filters=1024,
            input_layer=conv5,
            name='connecting_block_conv2')

        '''
        Expansive path
        '''
        conv6 = UNetModel.get_decoder_block('decoder_block1', 2, 512, conv5, conv4)
        conv7 = UNetModel.get_decoder_block('decoder_block2', 2, 256, conv6, conv3)
        conv8 = UNetModel.get_decoder_block('decoder_block3', 2, 128, conv7, conv2)
        conv9 = UNetModel.get_decoder_block('decoder_block4', 2, 64, conv8, conv1)

        '''
        Last convolutional layer and softmax activation for
        per-pixel classification
        '''
        conv10 = Conv2D(self.num_classes, (1, 1), name='logits', kernel_initializer='he_normal', bias_initializer='zeros')(conv9)

        return [conv10]

    @staticmethod
    def get_convolution_block(
            num_filters,
            input_layer,
            name,
            use_batch_normalization=True,
            use_activation=True,
            use_dropout=True,
            use_bias=True,
            kernel_size=(3, 3),
            padding='valid',
            conv2d_kernel_initializer='he_normal',
            conv2d_bias_initializer='zeros',
            relu_alpha=0.1,
            dropout_rate=0.1):
        conv = ZeroPadding2D(
            (1, 1),
            name='{}_padding'.format(name))(input_layer)

        conv = Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=conv2d_kernel_initializer,
            bias_initializer=conv2d_bias_initializer,
            name=name,
            use_bias=use_bias)(conv)

        '''
        From a statistics point of view BN before activation does not make sense to me.
        BN is normalizing the distribution of features coming out of a convolution, some
        these features might be negative which will be truncated by a non-linearity like ReLU.
        If you normalize before activation you are including these negative values in the
        normalization immediately before culling them from the feature space. BN after
        activation will normalize the positive features without statistically biasing them
        with features that do not make it through to the next convolutional layer.
        '''
        if use_batch_normalization:
            conv = BatchNormalization(
                momentum=0.1,
                name='{}_normalization'.format(name))(conv)

        if use_activation:
            # With alpha=0.0 LeakyReLU is a ReLU
            conv = LeakyReLU(
                alpha=relu_alpha,
                name='{}_activation'.format(name))(conv)

        if use_dropout:
            conv = SpatialDropout2D(dropout_rate)(conv)

        return conv

    @staticmethod
    def get_encoder_block(
            name_prefix,
            num_convolutions,
            num_filters,
            input_layer,
            kernel_size=(3, 3),
            pool_size=(2, 2),
            strides=(2, 2)):
        previous_layer = input_layer

        # Add the convolution blocks
        for i in range(0, num_convolutions):
            conv = UNetModel.get_convolution_block(
                num_filters=num_filters,
                input_layer=previous_layer,
                kernel_size=kernel_size,
                name='{}_conv{}'.format(name_prefix, i + 1))

            previous_layer = conv

        # Add the pooling layer
        pool = MaxPooling2D(
            pool_size=pool_size,
            strides=strides,
            name='{}_pool'.format(name_prefix))(previous_layer)

        return previous_layer, pool

    @staticmethod
    def get_decoder_block(
            name_prefix,
            num_convolutions,
            num_filters,
            input_layer,
            concat_layer=None,
            upsampling_size=(2, 2),
            kernel_size=(3, 3)):

        # Add upsampling layer
        up = UpSampling2D(size=upsampling_size)(input_layer)

        # Add concatenation layer to pass features from encoder path
        # to the decoder path
        previous_layer = None

        if concat_layer is not None:
            concat = concatenate([up, concat_layer], axis=-1)
            previous_layer = concat
        else:
            previous_layer = up

        for i in range(0, num_convolutions):
            conv = UNetModel.get_convolution_block(
                num_filters=num_filters,
                input_layer=previous_layer,
                kernel_size=kernel_size,
                name='{}_conv{}'.format(name_prefix, i + 1),
                use_bias=False,
                use_activation=False)

            previous_layer = conv

        return previous_layer


##############################################
# SEGNET
##############################################


class SegNetBasicModel(ModelBase):

    """
    The SegNet-Basic model:

    https://arxiv.org/pdf/1511.00561.pdf
    """

    def __init__(self,
                 input_shape,
                 num_classes,
                 encoder_only=False,
                 model_lambda_loss_type=ModelLambdaLossType.NONE):

        self.encoder_only = encoder_only
        name = "SegNet-Basic" if not self.encoder_only else "SegNet-Basic-Encoder-Only"

        super(SegNetBasicModel, self).__init__(
            name=name,
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type)

    def _build_model(self, inputs):

        """
        Encoder path
        """
        conv1, pool1 = UNetModel.get_encoder_block('encoder_block1', 2, 64, inputs)
        conv2, pool2 = UNetModel.get_encoder_block('encoder_block2', 2, 128, pool1)
        conv3, pool3 = UNetModel.get_encoder_block('encoder_block3', 3, 256, pool2)
        conv4, pool4 = UNetModel.get_encoder_block('encoder_block4', 3, 512, pool3)
        #conv5, pool5 = UNetModel.get_encoder_block('encoder_block5', 3, 1024, conv4)

        if self.encoder_only:
            # In order to avoid increasing the number of variables with a huge dense layer
            # use average pooling with a pool size of the previous layer's spatial
            # dimension
            pool_size = K.int_shape(pool4)[1:3]

            encoder = AveragePooling2D(pool_size=pool_size, name='avg_pool2d')(pool4)
            encoder = Flatten(name='flatten')(encoder)
            encoder = Dense(self.num_classes, name='logits')(encoder)
            return [encoder]


        """
        Decoder path
        """
        #conv6 = UNetModel.get_decoder_block('decoder_block1', 3, 1024, conv5, conv5)
        conv5 = UNetModel.get_decoder_block('decoder_block2', 3, 512, pool4, conv4)
        conv6 = UNetModel.get_decoder_block('decoder_block3', 3, 256, conv5, conv3)
        conv7 = UNetModel.get_decoder_block('decoder_block4', 2, 128, conv6, conv2)
        conv8 = UNetModel.get_decoder_block('decoder_block5', 2, 64, conv7, conv1)

        """
        Last convolutional layer and softmax activation for
        per-pixel classification
        """
        conv9 = Conv2D(self.num_classes, (1, 1), name='logits', kernel_initializer='he_normal', bias_initializer='zeros')(conv8)

        return [conv9]


##############################################
# ENET NAIVE UPSAMPLING
##############################################


class ENetNaiveUpsampling(ModelBase):

    def __init__(self,
                 input_shape,
                 num_classes,
                 encoder_only=False,
                 model_lambda_loss_type=ModelLambdaLossType.NONE):

        self.encoder_only = encoder_only
        name = "ENet-Naive-Upsampling" if not self.encoder_only else "ENet-Naive-Upsampling-Encoder-Only"

        super(ENetNaiveUpsampling, self).__init__(
            name=name,
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type)

    def _build_model(self, inputs):
        enet = ENetNaiveUpsampling.encoder_build(inputs)

        if self.encoder_only:
            # In order to avoid increasing the number of variables with a huge dense layer
            # use average pooling with a pool size of the previous layer's spatial
            # dimension
            pool_size = K.int_shape(enet)[1:3]

            enet = AveragePooling2D(pool_size=pool_size, name='avg_pool2d')(enet)
            enet = Flatten(name='flatten')(enet)
            enet = Dense(self.num_classes, name='logits')(enet)
        else:
            enet = ENetNaiveUpsampling.decoder_build(enet, nc=self.num_classes)

        return [enet]

    @staticmethod
    def encoder_initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):
        conv = Conv2D(filters=nb_filter,
                      kernel_size=(nb_row, nb_col),
                      name="initial_block_conv2d",
                      padding='same',
                      strides=strides)(inp)

        max_pool = MaxPooling2D(name="initial_block_pool2d")(inp)
        merged = concatenate([conv, max_pool], axis=3, name="initial_block_concat")
        merged = BatchNormalization(momentum=0.1, name='initial_block_bnorm')(merged)
        merged = PReLU(shared_axes=[1, 2], name='initial_block_prelu')(merged)
        return merged

    @staticmethod
    def encoder_bottleneck(inp,
                           output,
                           name_prefix,
                           internal_scale=4,
                           asymmetric=0,
                           dilated=0,
                           downsample=False,
                           dropout_rate=0.1,
                           kernel_initializer='he_normal',
                           bias_initializer='zeros'):

        internal = output // internal_scale
        encoder = inp

        """
        Main branch
        """
        # 1x1 projection downwards to internal feature space
        # Note: The 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
        input_stride = 2 if downsample else 1

        encoder = Conv2D(filters=internal,
                         kernel_size=(input_stride, input_stride),
                         strides=(input_stride, input_stride),
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         use_bias=False,
                         name='{}_proj_conv2d_1'.format(name_prefix))(encoder)

        # Batch normalization + PReLU
        # ENet uses momentum of 0.1, keras default is 0.99
        encoder = BatchNormalization(momentum=0.1, name='{}_bnorm_1'.format(name_prefix))(encoder)
        encoder = PReLU(shared_axes=[1, 2], name='{}_prelu_1'.format(name_prefix))(encoder)

        # Convolution block; either normal, asymmetric or dilated convolution
        if not asymmetric and not dilated:
            encoder = Conv2D(filters=internal,
                             kernel_size=(3, 3),
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             padding='same',
                             name='{}_conv2d_1'.format(name_prefix))(encoder)
        elif asymmetric:
            encoder = Conv2D(filters=internal,
                             kernel_size=(1, asymmetric),
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             padding='same',
                             use_bias=False,
                             name='{}_aconv2d_1'.format(name_prefix))(encoder)

            encoder = Conv2D(filters=internal,
                             kernel_size=(asymmetric, 1),
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             padding='same',
                             name='{}_aconv2d_2'.format(name_prefix))(encoder)
        elif dilated:
            encoder = Conv2D(filters=internal,
                             kernel_size=(3, 3),
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             dilation_rate=(dilated, dilated),
                             padding='same',
                             name='{}_dconv2d'.format(name_prefix))(encoder)
        else:
            raise RuntimeError('Invalid convolution options for encoder block')

        # ENet uses momentum of 0.1, keras default is 0.99
        encoder = BatchNormalization(momentum=0.1, name='{}_bnorm_2'.format(name_prefix))(encoder)
        encoder = PReLU(shared_axes=[1, 2], name='{}_prelu_2'.format(name_prefix))(encoder)

        # 1x1 projection upwards from internal to output feature space
        encoder = Conv2D(filters=output,
                         kernel_size=(1, 1),
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         use_bias=False,
                         name='{}_proj_conv2d_2'.format(name_prefix))(encoder)

        # ENet uses momentum of 0.1, keras default is 0.99
        encoder = BatchNormalization(momentum=0.1, name='{}_bnorm_3'.format(name_prefix))(encoder)
        encoder = SpatialDropout2D(rate=dropout_rate, name='{}_sdrop2d_1'.format(name_prefix))(encoder)

        """
        Other branch
        """
        other = inp

        if downsample:
            other = MaxPooling2D(name='{}_other_pool2d'.format(name_prefix))(other)
            other = Permute((1, 3, 2), name='{}_other_permute_1'.format(name_prefix))(other)

            pad_feature_maps = output - inp.get_shape().as_list()[3]
            tb_pad = (0, 0)
            lr_pad = (0, pad_feature_maps)
            other = ZeroPadding2D(padding=(tb_pad, lr_pad), name='{}_other_zpad2d'.format(name_prefix))(other)
            other = Permute((1, 3, 2), name='{}_other_permute_2'.format(name_prefix))(other)

        """
        Merge branches
        """
        encoder = add([encoder, other], name='{}_add'.format(name_prefix))
        encoder = PReLU(shared_axes=[1, 2], name='{}_prelu_3'.format(name_prefix))(encoder)

        return encoder

    @staticmethod
    def encoder_build(inp, dropout_rate=0.01):
        # Initial block
        enet = ENetNaiveUpsampling.encoder_initial_block(inp)

        # Bottleneck 1.0
        enet = ENetNaiveUpsampling.encoder_bottleneck(enet, 64, name_prefix='en_bn_1.0', downsample=True, dropout_rate=dropout_rate)

        # Bottleneck 1.i
        for i in range(4):
            name_prefix = 'en_bn_1.{}'.format(i + 1)
            enet = ENetNaiveUpsampling.encoder_bottleneck(enet, 64, name_prefix, dropout_rate=dropout_rate)

        # Bottleneck 2.0
        enet = ENetNaiveUpsampling.encoder_bottleneck(enet, 128, name_prefix='en_bn_2.0', downsample=True)

        # Bottleneck 2.x and 3.x
        for i in range(2):
            name_prefix = 'en_bn_{}.'.format(2 + i)

            enet = ENetNaiveUpsampling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '1')  # bottleneck 2.1
            enet = ENetNaiveUpsampling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '2', dilated=2)  # bottleneck 2.2
            enet = ENetNaiveUpsampling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '3', asymmetric=5)  # bottleneck 2.3
            enet = ENetNaiveUpsampling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '4', dilated=4)  # bottleneck 2.4
            enet = ENetNaiveUpsampling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '5')  # bottleneck 2.5
            enet = ENetNaiveUpsampling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '6', dilated=8)  # bottleneck 2.6
            enet = ENetNaiveUpsampling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '7', asymmetric=5)  # bottleneck 2.7
            enet = ENetNaiveUpsampling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '8', dilated=16)  # bottleneck 2.8

        return enet

    @staticmethod
    def decoder_bottleneck(encoder,
                           output,
                           name_prefix,
                           upsample=False,
                           reverse_module=False,
                           kernel_initializer='he_normal',
                           bias_initializer='zeros'):

        internal = output / 4

        """
        Main branch
        """
        # 1x1 projection downwards to internal feature space
        x = Conv2D(filters=internal,
                   kernel_size=(1, 1),
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer,
                   use_bias=False,
                   name='{}_proj_conv2d_1'.format(name_prefix))(encoder)

        # ENet uses momentum of 0.1, keras default is 0.99
        x = BatchNormalization(momentum=0.1, name='{}_bnorm_1'.format(name_prefix))(x)
        x = PReLU(shared_axes=[1, 2], name='{}_prelu_1'.format(name_prefix))(x)
        # x = Activation('relu', name='{}_relu_1'.format(name_prefix))(x)

        # Upsampling
        if not upsample:
            x = Conv2D(filters=internal,
                       kernel_size=(3, 3),
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       padding='same',
                       use_bias=True,
                       name='{}_conv2d_1'.format(name_prefix))(x)
        else:
            x = Conv2DTranspose(filters=internal,
                                kernel_size=(3, 3),
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                strides=(2, 2),
                                padding='same',
                                name='{}_tconv2d_1'.format(name_prefix))(x)

        # ENet uses momentum of 0.1 keras default is 0.99
        x = BatchNormalization(momentum=0.1, name='{}_bnorm_2'.format(name_prefix))(x)
        x = PReLU(shared_axes=[1, 2], name='{}_prelu_1'.format(name_prefix))(x)
        # x = Activation('relu', name='{}_relu_2'.format(name_prefix))(x)

        # 1x1 projection upwards from internal to output feature space
        x = Conv2D(filters=output,
                   kernel_size=(1, 1),
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer,
                   padding='same',
                   use_bias=False,
                   name='{}_proj_conv2d_2'.format((name_prefix)))(x)
        x = BatchNormalization(momentum=0.1, name='{}_bnorm_3'.format(name_prefix))(x)
        # NOTE: Different than original ENet, original ENet omits dropout for decoder
        x = SpatialDropout2D(rate=0.1, name='{}_sdrop2d_1'.format(name_prefix))(x)


        """
        Other branch
        """
        other = encoder

        if encoder.get_shape()[-1] != output or upsample:
            other = Conv2D(filters=output,
                           kernel_size=(1, 1),
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           padding='same',
                           use_bias=False,
                           name='{}_other_conv2d'.format(name_prefix))(other)

            other = BatchNormalization(momentum=0.1, name='{}_other_bnorm_1'.format(name_prefix))(other)

            if upsample and reverse_module is not False:
                other = UpSampling2D(size=(2, 2), name='{}_other_usample2d'.format(name_prefix))(other)

        if upsample and reverse_module is False:
            decoder = x
        else:
            """
            Merge branches
            """
            decoder = add([x, other], name='{}_add'.format(name_prefix))
            decoder = PReLU(shared_axes=[1, 2], name='{}_prelu_3'.format(name_prefix))(decoder)
            # decoder = Activation('relu', name='{}_relu_3'.format(name_prefix))(decoder)

        return decoder

    @staticmethod
    def decoder_build(encoder, nc):
        enet = ENetNaiveUpsampling.decoder_bottleneck(encoder, 64, name_prefix='de_bn_4.0', upsample=True,
                                  reverse_module=True)  # bottleneck 4.0
        enet = ENetNaiveUpsampling.decoder_bottleneck(enet, 64, name_prefix='de_bn_4.1')  # bottleneck 4.1
        enet = ENetNaiveUpsampling.decoder_bottleneck(enet, 64, name_prefix='de_bn_4.2')  # bottleneck 4.2
        enet = ENetNaiveUpsampling.decoder_bottleneck(enet, 16, name_prefix='de_bn_5.0', upsample=True,
                                  reverse_module=True)  # bottleneck 5.0
        enet = ENetNaiveUpsampling.decoder_bottleneck(enet, 16, name_prefix='de_bn_5.1')  # bottleneck 5.1

        enet = Conv2DTranspose(filters=nc,
                               kernel_size=(2, 2),
                               kernel_initializer='he_normal',
                               bias_initializer='zeros',
                               strides=(2, 2),
                               padding='same',
                               name='logits')(enet)

        return enet


##############################################
# ENET MAX UNPOOLING
##############################################

class ENetMaxUnpooling(ModelBase):

    def __init__(self,
                 input_shape,
                 num_classes,
                 encoder_only=False,
                 model_lambda_loss_type=ModelLambdaLossType.NONE):

        self.encoder_only = encoder_only
        name = "ENet-Max-Unpooling" if not self.encoder_only else "ENet-Max-Unpooling-Encoder-Only"

        super(ENetMaxUnpooling, self).__init__(
            name=name,
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type)

    def _build_model(self, inputs):
        enet, pooling_indices = ENetMaxUnpooling.encoder_build(inputs)

        if self.encoder_only:
            # In order to avoid increasing the number of variables with a huge dense layer
            # use average pooling with a pool size of the previous layer's spatial
            # dimension
            pool_size = K.int_shape(enet)[1:3]

            enet = AveragePooling2D(pool_size=pool_size, name='avg_pool2d')(enet)
            enet = Flatten(name='flatten')(enet)
            enet = Dense(self.num_classes, name='logits')(enet)
        else:
            enet = ENetMaxUnpooling.decoder_build(enet, index_stack=pooling_indices, nc=self.num_classes)

        return [enet]

    @staticmethod
    def encoder_initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):
        conv = Conv2D(filters=nb_filter,
                      kernel_size=(nb_row, nb_col),
                      padding='same',
                      strides=strides,
                      name='initial_block_conv2d')(inp)

        max_pool, indices = MaxPoolingWithArgmax2D(name='initial_block_pool2d')(inp)
        merged = concatenate([conv, max_pool], axis=3, name='initial_block_concat')
        return merged, indices

    @staticmethod
    def encoder_bottleneck(inp,
                           output,
                           name_prefix,
                           internal_scale=4,
                           asymmetric=0,
                           dilated=0,
                           downsample=False,
                           dropout_rate=0.1,
                           kernel_initializer='he_normal',
                           bias_initializer='zeros'):

        internal = output // internal_scale
        encoder = inp

        """
        Main branch
        """
        # 1x1 projection downwards to internal feature space
        # Note: The 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
        input_stride = 2 if downsample else 1

        encoder = Conv2D(filters=internal,
                         kernel_size=(input_stride, input_stride),
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         strides=(input_stride, input_stride),
                         use_bias=False,
                         name='{}_proj_conv2d_1'.format(name_prefix))(encoder)

        # Batch normalization + PReLU
        # ENet uses momentum of 0.1, keras default is 0.99
        encoder = BatchNormalization(momentum=0.1, name='{}_bnorm_1'.format(name_prefix))(encoder)
        encoder = PReLU(shared_axes=[1, 2], name='{}_prelu_1'.format(name_prefix))(encoder)

        # Convolution block; either normal, asymmetric or dilated convolution
        if not asymmetric and not dilated:
            encoder = Conv2D(filters=internal,
                             kernel_size=(3, 3),
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             padding='same',
                             name='{}_conv2d_1'.format(name_prefix))(encoder)
        elif asymmetric:
            encoder = Conv2D(filters=internal,
                             kernel_size=(1, asymmetric),
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             padding='same',
                             use_bias=False,
                             name='{}_aconv2d_1'.format(name_prefix))(encoder)

            encoder = Conv2D(filters=internal,
                             kernel_size=(asymmetric, 1),
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             padding='same',
                             name='{}_aconv2d_2'.format(name_prefix))(encoder)
        elif dilated:
            encoder = Conv2D(filters=internal,
                             kernel_size=(3, 3),
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             dilation_rate=(dilated, dilated),
                             padding='same',
                             name='{}_dconv2d'.format(name_prefix))(encoder)
        else:
            raise RuntimeError('Invalid convolution options for encoder block')

        # ENet uses momentum of 0.1, keras default is 0.99
        encoder = BatchNormalization(momentum=0.1, name='{}_bnorm_2'.format(name_prefix))(encoder)
        encoder = PReLU(shared_axes=[1, 2], name='{}_prelu_2'.format(name_prefix))(encoder)

        # 1x1 projection upwards from internal to output feature space
        encoder = Conv2D(filters=output,
                         kernel_size=(1, 1),
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         use_bias=False,
                         name='{}_proj_conv2d_2'.format(name_prefix))(encoder)

        # ENet uses momentum of 0.1, keras default is 0.99
        encoder = BatchNormalization(momentum=0.1, name='{}_bnorm_3'.format(name_prefix))(encoder)
        encoder = SpatialDropout2D(rate=dropout_rate, name='{}_sdrop2d_1'.format(name_prefix))(encoder)

        """
        Other branch
        """
        other = inp

        if downsample:
            other, indices = MaxPoolingWithArgmax2D(name='{}_other_pool2d'.format(name_prefix))(other)
            other = Permute((1, 3, 2), name='{}_other_permute_1'.format(name_prefix))(other)

            pad_feature_maps = output - inp.get_shape().as_list()[3]
            tb_pad = (0, 0)
            lr_pad = (0, pad_feature_maps)
            other = ZeroPadding2D(padding=(tb_pad, lr_pad), name='{}_other_zpad2d'.format(name_prefix))(other)
            other = Permute((1, 3, 2), name='{}_other_permute_2'.format(name_prefix))(other)

        """
        Merge branches
        """
        encoder = add([encoder, other], name='{}_add'.format(name_prefix))
        encoder = PReLU(shared_axes=[1, 2], name='{}_prelu_3'.format(name_prefix))(encoder)

        if downsample:
            return encoder, indices

        return encoder

    @staticmethod
    def encoder_build(inp, dropout_rate=0.01):
        pooling_indices = []

        # Initial block
        enet, indices_single = ENetMaxUnpooling.encoder_initial_block(inp)
        pooling_indices.append(indices_single)

        # Bottleneck 1.0
        enet, indices_single = ENetMaxUnpooling.encoder_bottleneck(enet, 64, name_prefix='en_bn_1.0', downsample=True, dropout_rate=dropout_rate)
        pooling_indices.append(indices_single)

        # Bottleneck 1.i
        for i in range(4):
            name_prefix = 'en_bn_1.{}'.format(i + 1)
            enet = ENetMaxUnpooling.encoder_bottleneck(enet, 64, name_prefix=name_prefix, dropout_rate=dropout_rate)

        # Bottleneck 2.0
        enet, indices_single = ENetMaxUnpooling.encoder_bottleneck(enet, 128, name_prefix='en_bn_2.0', downsample=True)
        pooling_indices.append(indices_single)

        # Bottleneck 2.x and 3.x
        for i in range(2):
            name_prefix = 'en_bn_{}.'.format(2 + i)

            enet = ENetMaxUnpooling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '1')  # bottleneck 2.1
            enet = ENetMaxUnpooling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '2', dilated=2)  # bottleneck 2.2
            enet = ENetMaxUnpooling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '3', asymmetric=5)  # bottleneck 2.3
            enet = ENetMaxUnpooling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '4', dilated=4)  # bottleneck 2.4
            enet = ENetMaxUnpooling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '5')  # bottleneck 2.5
            enet = ENetMaxUnpooling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '6', dilated=8)  # bottleneck 2.6
            enet = ENetMaxUnpooling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '7', asymmetric=5)  # bottleneck 2.7
            enet = ENetMaxUnpooling.encoder_bottleneck(enet, 128, name_prefix=name_prefix + '8', dilated=16)  # bottleneck 2.8

        return enet, pooling_indices

    @staticmethod
    def decoder_bottleneck(encoder,
                           output,
                           name_prefix,
                           upsample=False,
                           reverse_module=False,
                           kernel_initializer='he_normal',
                           bias_initializer='zeros'):

        internal = output / 4

        """
        Main branch
        """
        # 1x1 projection downwards to internal feature space
        x = Conv2D(filters=internal,
                   kernel_size=(1, 1),
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer,
                   use_bias=False,
                   name='{}_proj_conv2d_1'.format(name_prefix))(encoder)

        # ENet uses momentum of 0.1, keras default is 0.99
        x = BatchNormalization(momentum=0.1, name='{}_bnorm_1'.format(name_prefix))(x)
        x = PReLU(shared_axes=[1, 2], name='{}_prelu_1'.format(name_prefix))(x)
        # x = Activation('relu', name='{}_relu_1'.format(name_prefix))(x)

        # Upsampling
        if not upsample:
            x = Conv2D(filters=internal,
                       kernel_size=(3, 3),
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       padding='same',
                       use_bias=True,
                       name='{}_conv2d_1'.format(name_prefix))(x)
        else:
            x = Conv2DTranspose(filters=internal,
                                kernel_size=(3, 3),
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                strides=(2, 2),
                                padding='same',
                                name='{}_tconv2d_1'.format(name_prefix))(x)

        # ENet uses momentum of 0.1 keras default is 0.99
        x = BatchNormalization(momentum=0.1, name='{}_bnorm_2'.format(name_prefix))(x)
        x = PReLU(shared_axes=[1, 2], name='{}_prelu_2'.format(name_prefix))(x)
        # x = Activation('relu', name='{}_relu_2'.format(name_prefix))(x)

        # 1x1 projection upwards from internal to output feature space
        x = Conv2D(filters=output,
                   kernel_size=(1, 1),
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer,
                   padding='same',
                   use_bias=False,
                   name='{}_proj_conv2d_2'.format((name_prefix)))(x)

        """
        Other branch
        """
        other = encoder

        if encoder.get_shape()[-1] != output or upsample:
            other = Conv2D(filters=output,
                           kernel_size=(1, 1),
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           padding='same',
                           use_bias=False,
                           name='{}_other_conv2d'.format(name_prefix))(other)

            other = BatchNormalization(momentum=0.1, name='{}_other_bnorm_1'.format(name_prefix))(other)

            if upsample and reverse_module is not False:
                mpool = MaxUnpooling2D(name='{}_other_unpool2d'.format(name_prefix))
                other = mpool([other, reverse_module])

        if upsample and reverse_module is False:
            decoder = x
        else:
            x = BatchNormalization(momentum=0.1, name='{}_bnorm_3'.format(name_prefix))(x)

            """
            Merge branches
            """
            decoder = add([x, other], name='{}_add'.format(name_prefix))
            decoder = PReLU(shared_axes=[1, 2], name='{}_prelu_3'.format(name_prefix))(decoder)
            # decoder = Activation('relu', name='{}_relu_3'.format(name_prefix))(decoder)

        return decoder

    @staticmethod
    def decoder_build(encoder, index_stack, nc):
        enet = ENetMaxUnpooling.decoder_bottleneck(encoder, 64, name_prefix='de_bn_4.0', upsample=True, reverse_module=index_stack.pop())
        enet = ENetMaxUnpooling.decoder_bottleneck(enet, 64, name_prefix='de_bn_4.1')
        enet = ENetMaxUnpooling.decoder_bottleneck(enet, 64, name_prefix='de_bn_4.2')
        enet = ENetMaxUnpooling.decoder_bottleneck(enet, 16, name_prefix='de_bn_5.0', upsample=True, reverse_module=index_stack.pop())
        enet = ENetMaxUnpooling.decoder_bottleneck(enet, 16, name_prefix='de_bn_5.1')

        enet = Conv2DTranspose(filters=nc,
                               kernel_size=(2, 2),
                               kernel_initializer='he_normal',
                               bias_initializer='zeros',
                               strides=(2, 2),
                               padding='same',
                               name='logits')(enet)

        return enet


##############################################
# ENET NAIVE UPSAMPLING - ENHANCED
##############################################

def l2_normalization(x):
    alpha = 40.0
    x = K.tf.div(x, K.tf.norm(x, axis=-1, ord='euclidean', keep_dims=True))
    x = K.tf.multiply(x, alpha)
    return x


class ENetNaiveUpsamplingEnhanced(ModelBase):

    def __init__(self,
                 input_shape,
                 num_classes,
                 encoder_only=False,
                 model_lambda_loss_type=ModelLambdaLossType.NONE):

        self.encoder_only = encoder_only
        name = "ENet-Naive-Upsampling-Enhanced" if not self.encoder_only else "ENet-Naive-Upsampling-Enhanced-Encoder-Only"

        super(ENetNaiveUpsamplingEnhanced, self).__init__(
            name=name,
            input_shape=input_shape,
            num_classes=num_classes,
            model_lambda_loss_type=model_lambda_loss_type)

    def _build_model(self, inputs):
        enet = ENetNaiveUpsampling.encoder_build(inputs)

        # In order to avoid increasing the number of variables with a huge dense layer
        # use average pooling with a pool size of the previous layer's spatial
        # dimension
        if self.encoder_only:
            enet = ENetNaiveUpsamplingEnhanced.decoder_build(enet, nc=self.num_classes, final_layer_name='decoder_logits')
            enet = GlobalAveragePooling2D(name='avg_pool2d')(enet)
            enet = Dense(self.num_classes, name='logits')(enet)
        else:
            enet = ENetNaiveUpsamplingEnhanced.decoder_build(enet, nc=self.num_classes)

        return [enet]

    @staticmethod
    def decoder_bottleneck(encoder,
                           output,
                           name_prefix,
                           upsample=False,
                           reverse_module=False,
                           kernel_initializer='he_normal',
                           bias_initializer='zeros'):

        internal = output / 4

        """
        Main branch
        """
        # 1x1 projection downwards to internal feature space
        x = Conv2D(filters=internal,
                   kernel_size=(1, 1),
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer,
                   use_bias=False,
                   name='{}_proj_conv2d_1'.format(name_prefix))(encoder)

        # ENet uses momentum of 0.1, keras default is 0.99
        x = BatchNormalization(momentum=0.1, name='{}_bnorm_1'.format(name_prefix))(x)
        x = PReLU(shared_axes=[1, 2], name='{}_prelu_1'.format(name_prefix))(x)
        # x = Activation('relu', name='{}_relu_1'.format(name_prefix))(x)

        # Upsampling
        if not upsample:
            x = Conv2D(filters=internal,
                       kernel_size=(3, 3),
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       padding='same',
                       use_bias=True,
                       name='{}_conv2d_1'.format(name_prefix))(x)
        else:
            # Unlike in regular ENet we will replace Conv2DTranspose with a Nearest-Neighbour
            # upsampling on each layer plus a Conv2D. This is called a NN-Resize convolution
            x = UpSampling2D(size=(2, 2), name='{}_nn_rs_conv_upsample2d'.format(name_prefix))(x)
            x = Conv2D(filters=internal,
                       kernel_size=(3, 3),
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       padding='same',
                       name='{}_nn_rs_conv_conv2d'.format(name_prefix))(x)

        # ENet uses momentum of 0.1 keras default is 0.99
        x = BatchNormalization(momentum=0.1, name='{}_bnorm_2'.format(name_prefix))(x)
        x = PReLU(shared_axes=[1, 2], name='{}_prelu_2'.format(name_prefix))(x)
        # x = Activation('relu', name='{}_relu_2'.format(name_prefix))(x)

        # 1x1 projection upwards from internal to output feature space
        x = Conv2D(filters=output,
                   kernel_size=(1, 1),
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer,
                   padding='same',
                   use_bias=False,
                   name='{}_proj_conv2d_2'.format((name_prefix)))(x)
        x = BatchNormalization(momentum=0.1, name='{}_bnorm_3'.format(name_prefix))(x)
        # NOTE: Different than original ENet, original ENet omits dropout for decoder
        x = SpatialDropout2D(rate=0.1, name='{}_sdrop2d_1'.format(name_prefix))(x)

        """
        Other branch
        """
        other = encoder

        if encoder.get_shape()[-1] != output or upsample:
            other = Conv2D(filters=output,
                           kernel_size=(1, 1),
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           padding='same',
                           use_bias=False,
                           name='{}_other_conv2d'.format(name_prefix))(other)

            other = BatchNormalization(momentum=0.1, name='{}_other_bnorm_1'.format(name_prefix))(other)

            if upsample and reverse_module is not False:
                other = UpSampling2D(size=(2, 2), name='{}_other_usample2d'.format(name_prefix))(other)

        if upsample and reverse_module is False:
            decoder = x
        else:
            """
            Merge branches
            """
            decoder = add([x, other], name='{}_add'.format(name_prefix))
            decoder = PReLU(shared_axes=[1, 2], name='{}_prelu_3'.format(name_prefix))(decoder)
            # decoder = Activation('relu', name='{}_relu_3'.format(name_prefix))(decoder)

        return decoder

    @staticmethod
    def decoder_build(encoder, nc, final_layer_name='logits'):
        enet = ENetNaiveUpsamplingEnhanced.decoder_bottleneck(encoder, 64, name_prefix='de_bn_4.0', upsample=True, reverse_module=True)  # bottleneck 4.0
        enet = ENetNaiveUpsamplingEnhanced.decoder_bottleneck(enet, 64, name_prefix='de_bn_4.1')  # bottleneck 4.1
        enet = ENetNaiveUpsamplingEnhanced.decoder_bottleneck(enet, 64, name_prefix='de_bn_4.2')  # bottleneck 4.2
        enet = ENetNaiveUpsamplingEnhanced.decoder_bottleneck(enet, 16, name_prefix='de_bn_5.0', upsample=True, reverse_module=True)  # bottleneck 5.0
        enet = ENetNaiveUpsamplingEnhanced.decoder_bottleneck(enet, 16, name_prefix='de_bn_5.1')  # bottleneck 5.1

        # Unlike in regular ENet we will replace Conv2DTranspose with a Nearest-Neighbour
        # upsampling on each layer plus a Conv2D. This is called a NN-Resize convolution (NNR Convolution)
        enet = UpSampling2D(size=(2, 2), name='final_nn_rs_conv_upsample2d')(enet)
        enet = Conv2D(filters=nc,
                      kernel_size=(3, 3),
                      kernel_initializer='he_normal',
                      bias_initializer='zeros',
                      padding='same',
                      name='final_nn_rs_conv_conv2d')(enet)
        enet = BatchNormalization(momentum=0.1, name='final_nn_rs_conv_bnorm')(enet)
        enet = PReLU(shared_axes=[1, 2], name='final_nn_rs_conv_prelu')(enet)

        enet = Conv2D(filters=nc,
                      kernel_size=(1, 1),
                      kernel_initializer='he_normal',
                      bias_initializer='zeros',
                      padding='same',
                      name=final_layer_name)(enet)

        return enet
