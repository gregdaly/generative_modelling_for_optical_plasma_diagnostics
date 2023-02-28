import tensorflow as tf
# import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras import backend as K

from scipy import signal

from tensorflow.keras import datasets, layers, models, regularizers, mixed_precision
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Dense, Conv1D, Conv2D, Flatten, Dropout, ZeroPadding2D, Cropping2D, MaxPooling1D, MaxPooling2D, Input, Activation, BatchNormalization, Concatenate, 
                                     SpatialDropout1D, Reshape, UpSampling2D, UpSampling1D, AveragePooling1D, AveragePooling2D, Multiply, Add, add, GlobalAveragePooling2D, 
                                     GlobalAveragePooling1D, Conv1DTranspose, Conv2DTranspose, GaussianNoise, LayerNormalization)#, DepthwiseConv1D)
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay, PolynomialDecay, LearningRateSchedule, CosineDecay
from tensorflow.keras.utils import get_custom_objects, plot_model
from tensorflow.keras.losses import MeanSquaredError, mse, BinaryCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

import math

class LayerScale(layers.Layer):
    """Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.
    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

#     @tf.function
    def build(self, input_shape):
        self.gamma = tf.Variable(
            self.init_values * tf.ones((self.projection_dim,))
        )

#     @tf.function
    def call(self, x):
        return tf.cast(x, tf.float32) * self.gamma

    # def call(self, x):
    #     return x * self.gamma

#     @tf.function
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config

# class StochasticDepth(layers.Layer):
#     """Stochastic Depth module.
#     It performs batch-wise dropping rather than sample-wise. In libraries like
#     `timm`, it's similar to `DropPath` layers that drops residual paths
#     sample-wise.
#     References:
#       - https://github.com/rwightman/pytorch-image-models
#     Args:
#       drop_path_rate (float): Probability of dropping paths. Should be within
#         [0, 1].
#     Returns:
#       Tensor either with the residual path dropped or kept.
#     """

#     def __init__(self, drop_path_rate, **kwargs):
#         super().__init__(**kwargs)
#         self.drop_path_rate = drop_path_rate

# #     @tf.function
#     def call(self, x, training=None):
#         if training:
#             keep_prob = 1 - self.drop_path_rate
#             shape = (tf.shape(x)[0],) + ((tf.rank(x) - 1),)
#             random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
#             random_tensor = tf.floor(random_tensor)
#             return (x / keep_prob) * random_tensor
#         return x

# #     @tf.function
#     def get_config(self):
#         config = super().get_config()
#         config.update({"drop_path_rate": self.drop_path_rate})
#         return config

def downsample_layer1D(filters, block_number):
    def apply(x):
        x = LayerNormalization(
                    epsilon=1e-6,
                    name = "spectra_encoder_downsampling_layernorm_" + str(block_number),
                )(x)
        x = Conv1D(
                    filters,
                    kernel_size=2,
                    strides=2,
                    name="spectra_encoder_downsampling_conv_" + str(block_number),
                )(x)
        return x
    return apply

def downsample_layer2D(filters, block_number):
    def apply(x):
        x = LayerNormalization(
                    epsilon=1e-6,
                    name = "image_encoder_downsampling_layernorm_" + str(block_number),
                )(x)
        x = Conv2D(
                    filters,
                    kernel_size=2,
                    strides=2,
                    name="image_encoder_downsampling_conv_" + str(block_number),
                )(x)
        return x
    return apply

# @tf.function
def upsample_layer1D(filters, block_number):
    def apply(x):
        x = LayerNormalization(
                    epsilon=1e-6,
                    name = "spectra_decoder_downsampling_layernorm_" + str(block_number),
                )(x)
        x = Conv1DTranspose(
                    filters,
                    kernel_size=2,
                    strides=2,
                    name="spectra_decoder_downsampling_conv_" + str(block_number),
                )(x)
        return x
    return apply

def upsample_layer2D(filters, block_number):
    def apply(x):
        x = LayerNormalization(
                    epsilon=1e-6,
                    name = "image_decoder_downsampling_layernorm_" + str(block_number),
                )(x)
        x = Conv2DTranspose(
                    filters,
                    kernel_size=2,
                    strides=2,
                    name="image_decoder_downsampling_conv_" + str(block_number),
                )(x)
        return x
    return apply

def ConvNeXTBlock1D(filters_spec, drop_path_rate=0.0, layer_scale_init_value=1e-6, name=None):

    """ConvNeXt block. - from https://github.com/keras-team/keras/blob/master/keras/applications/convnext.py
    References:
    - https://arxiv.org/abs/2201.03545
    - https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    Notes:
    In the original ConvNeXt implementation (linked above), the authors use
    `Dense` layers for pointwise convolutions for increased efficiency.
    Following that, this implementation also uses the same.
    Args:
    filters_spec (int): Number of filters for convolution layers. In the
    ConvNeXt paper, this is referred to as projection dimension.
    drop_path_rate (float): Probability of dropping paths. Should be within
    [0, 1].
    layer_scale_init_value (float): Layer scale value. Should be a small float
    number.
    name: name to path to the keras layer.
    Returns:
    A function representing a ConvNeXtBlock block.
    """

    def apply(inputs):

        x = inputs

        x = layers.Conv1D(
          filters=filters_spec,
          kernel_size=7,
          padding="same",
          groups=filters_spec,
          name=name + "_depthwise_conv",
        )(x)

        x = LayerNormalization(epsilon=1e-6, name=name + "_layernorm")(x)
        x = Dense(4 * filters_spec, name=name + "_pointwise_conv_1")(x)
        x = Activation("gelu", name=name + "_gelu")(x)
        x = Dense(filters_spec, name=name + "_pointwise_conv_2")(x)

        if layer_scale_init_value is not None:
            x = LayerScale(
              layer_scale_init_value,
              filters_spec,
              name=name + "_layer_scale",
            )(x)
        if drop_path_rate:
#             layer = StochasticDepth(
#               drop_path_rate, name=name + "_stochastic_depth"
#             )
            
            output = tfa.layers.StochasticDepth(drop_path_rate, name=name + "_stochastic_depth")([inputs, x])
        else:
            layer = layers.Activation("linear", name=name + "_identity")
            output = Add()([inputs, layer(x)])
        
        

#         return Add()([inputs, layer(x)])
        return output

    return apply

def ConvNeXTBlock2D(filters_spec, drop_path_rate=0.0, layer_scale_init_value=1e-6, name=None):

    """ConvNeXt block. - from https://github.com/keras-team/keras/blob/master/keras/applications/convnext.py
    References:
    - https://arxiv.org/abs/2201.03545
    - https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    Notes:
    In the original ConvNeXt implementation (linked above), the authors use
    `Dense` layers for pointwise convolutions for increased efficiency.
    Following that, this implementation also uses the same.
    Args:
    filters_spec (int): Number of filters for convolution layers. In the
    ConvNeXt paper, this is referred to as projection dimension.
    drop_path_rate (float): Probability of dropping paths. Should be within
    [0, 1].
    layer_scale_init_value (float): Layer scale value. Should be a small float
    number.
    name: name to path to the keras layer.
    Returns:
    A function representing a ConvNeXtBlock block.
    """

    def apply(inputs):

        x = inputs

        x = layers.Conv2D(
          filters=filters_spec,
          kernel_size=7,
          padding="same",
          groups=filters_spec,
          name=name + "_depthwise_conv",
        )(x)

        x = LayerNormalization(epsilon=1e-6, name=name + "_layernorm")(x)
        x = Dense(4 * filters_spec, name=name + "_pointwise_conv_1")(x)
        x = Activation("gelu", name=name + "_gelu")(x)
        x = Dense(filters_spec, name=name + "_pointwise_conv_2")(x)

        if layer_scale_init_value is not None:
            x = LayerScale(
              layer_scale_init_value,
              filters_spec,
              name=name + "_layer_scale",
            )(x)
        if drop_path_rate:
#             layer = StochasticDepth(
#               drop_path_rate, name=name + "_stochastic_depth"
#             )
            
            output = tfa.layers.StochasticDepth(drop_path_rate, name=name + "_stochastic_depth")([inputs, x])
        else:
            layer = layers.Activation("linear", name=name + "_identity")
            output = Add()([inputs, layer(x)])
        
        

#         return Add()([inputs, layer(x)])
        return output

    return apply

class CosineDecayWarmup(LearningRateSchedule):
  """A LearningRateSchedule that uses a cosine decay schedule.
  See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
  SGDR: Stochastic Gradient Descent with Warm Restarts.
  When training a model, it is often useful to lower the learning rate as
  the training progresses. This schedule applies a cosine decay function
  to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.
  The schedule is a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:
  ```python
  def decayed_learning_rate(step):
    step = min(step, decay_steps)
    cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_learning_rate * decayed
  ```
  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
      initial_learning_rate, decay_steps)
  ```
  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.
  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      warmup_steps,
      alpha=0.0,
      name=None):
    """Applies cosine decay to the learning rate.
    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a
        Python number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of initial_learning_rate.
      name: String. Optional name of the operation.  Defaults to 'CosineDecay'.
    """
    super(CosineDecayWarmup, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.warmup_steps = warmup_steps
    self.alpha = alpha
    self.name = name

  @tf.function
  def __call__(self, step):
    with tf.name_scope(self.name or "CosineDecay"):
      initial_learning_rate = tf.convert_to_tensor(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      decay_steps = tf.cast(self.decay_steps, dtype)
      warmup_steps = tf.cast(self.warmup_steps, dtype)

      global_step_recomp = tf.cast(step, dtype)
      global_step_recomp = tf.minimum(global_step_recomp, decay_steps)

      if global_step_recomp < warmup_steps:
        decayed = global_step_recomp / warmup_steps
      else:
        completed_fraction = (global_step_recomp - warmup_steps) / (decay_steps- warmup_steps)
        cosine_decayed = 0.5 * (1.0 + tf.cos(
            tf.constant(math.pi, dtype=dtype) * completed_fraction))

        decayed = (1 - self.alpha) * cosine_decayed + self.alpha
      return tf.multiply(initial_learning_rate, decayed)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "warmup_steps": self.warmup_steps,
        "alpha": self.alpha,
        "name": self.name
    }

