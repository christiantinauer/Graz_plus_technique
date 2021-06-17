from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import conv_utils

class Bias(Layer):
  def __init__(self,
               data_format=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               bias_constraint=None,
               **kwargs):
    super(Bias, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.bias_initializer = initializers.get(bias_initializer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)

  def build(self, input_shape):
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])

    self.bias = self.add_weight(name='bias',
                                shape=(input_dim,),
                                initializer=self.bias_initializer,
                                regularizer=self.bias_regularizer,
                                constraint=self.bias_constraint,
                                trainable=True,
                                dtype=self.dtype)
    self.built = True

  def call(self, inputs):
    return K.bias_add(inputs,
                      self.bias,
                      data_format=self.data_format)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
      'data_format': self.data_format,
      'bias_initializer': initializers.serialize(self.bias_initializer),
      'bias_regularizer': regularizers.serialize(self.bias_regularizer),
      'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(Bias, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
