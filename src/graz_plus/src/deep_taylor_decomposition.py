from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import conv3d_transpose

from layers import Bias

class Pass(layers.Layer):
  def __init__(self, for_layer: layers.Layer, stop_gradient=False, class_index=None, **kwargs):
    super(Pass, self).__init__(**kwargs)
    self._auto_track_sub_layers = False
    self.for_layer = for_layer
    self.stop_gradient = stop_gradient
    self.class_index = class_index

  def compute_output_shape(self, input_shape):
    return self.for_layer.input_shape

  def call(self, inputs, **kwargs):
    outputs = inputs
    if self.stop_gradient:
      outputs = K.stop_gradient(outputs)
    if self.class_index is not None:
      outputs = outputs * K.one_hot(self.class_index, self.for_layer.input_shape[-1])
    return outputs
  
  def get_config(self):
    config = {
      'stop_gradient': self.stop_gradient,
      'class_index': self.class_index,
    }
    base_config = super(Pass, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class ReshapeToInput(layers.Layer):
  def __init__(self, for_layer: layers.Layer, **kwargs):
    super(ReshapeToInput, self).__init__(**kwargs)
    self._auto_track_sub_layers = False
    self.for_layer = for_layer

  def compute_output_shape(self, input_shape):
    return self.for_layer.input_shape

  def call(self, inputs, **kwargs):
    return K.reshape(inputs, (-1,) + self.for_layer.input_shape[1:])

  def get_config(self):
    return super(ReshapeToInput, self).get_config()

class DenseDeepTaylorDecomposition(layers.Layer):
  def __init__(self, for_layer: layers.Dense, **kwargs):
    super(DenseDeepTaylorDecomposition, self).__init__(**kwargs)
    self._auto_track_sub_layers = False
    self.for_layer = for_layer

  def compute_output_shape(self, input_shape):
    return self.for_layer.input_shape

  def call(self, inputs, **kwargs):
    R = inputs[0]
    X = inputs[1]
    W = self.for_layer.kernel
    
    Wp = K.maximum(W, 0.)
    Zp = K.dot(X, Wp)
    S = tf.math.divide_no_nan(R, Zp)
    C = K.dot(S, K.transpose(Wp))
    
    result = X * C

    self.add_metric(K.sum(result, axis=[1]), name=self.name + '_sum')

    return result
  def get_config(self):
    return super(DenseDeepTaylorDecomposition, self).get_config()

class Conv3DDeepTaylorDecomposition(layers.Layer):
  def __init__(self, for_layer: layers.Conv3D, **kwargs):
    super(Conv3DDeepTaylorDecomposition, self).__init__(**kwargs)
    self._auto_track_sub_layers = False
    self.for_layer = for_layer

  def compute_output_shape(self, input_shape):
    return self.for_layer.input_shape

  def call(self, inputs, **kwargs):
    R = inputs[0]
    X = inputs[1]
    W = self.for_layer.kernel

    Wp = K.maximum(W, 0.)
    Zp = K.conv3d(X, Wp,
                  strides=self.for_layer.strides,
                  padding=self.for_layer.padding,
                  data_format=self.for_layer.data_format,
                  dilation_rate=self.for_layer.dilation_rate)
    S = tf.math.divide_no_nan(R, Zp)
    C = conv3d_transpose(S, Wp,
                         K.shape(X),
                         strides=self.for_layer.strides,
                         padding=self.for_layer.padding,
                         data_format=self.for_layer.data_format)
    result = X * C

    self.add_metric(K.sum(result, axis=[1, 2, 3, 4]), name=self.name + '_sum')

    return result

  def get_config(self):
    return super(Conv3DDeepTaylorDecomposition, self).get_config()

def add_deep_taylor_decomposition_to_model_output(output, class_index=None):
  current_input = output
  current_output = output

  previous_layer = None
  while True:
    layer, _, _ = current_input._keras_history
    if layer == previous_layer or isinstance(layer, layers.Lambda):
      break

    previous_layer = layer
    current_input = layer.get_input_at(0)
    
    if layer.name == 'output':
      current_output = Pass(layer, stop_gradient=False, class_index=class_index, name=layer.name + '_dtd')(current_output)
    elif isinstance(layer, (Bias, layers.Activation, layers.InputLayer)):
      current_output = Pass(layer, name=layer.name + '_dtd')(current_output)
    elif isinstance(layer, layers.Flatten):
      current_output = ReshapeToInput(layer, name=layer.name + '_dtd')(current_output)
    elif isinstance(layer, layers.Dense):
      current_output = DenseDeepTaylorDecomposition(layer, name=layer.name + '_dtd')([current_output, layer.input, layer.output])
    elif isinstance(layer, layers.Conv3D):
      current_output = Conv3DDeepTaylorDecomposition(layer, name=layer.name + '_dtd')([current_output, layer.input, layer.output])
    else:
      raise NotImplementedError()

  return current_output
