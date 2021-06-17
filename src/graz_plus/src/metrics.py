from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend as K
from tensorflow.python.keras.metrics import MeanMetricWrapper

def sum_relevance_inner_mask(y_true, y_pred):
  return K.sum(y_true * y_pred, axis=[1, 2, 3, 4])

class SumRelevanceInnerMask(MeanMetricWrapper):
  def __init__(self,
               name='sum_relevance_inner_mask',
               dtype=None):
    super(SumRelevanceInnerMask, self).__init__(sum_relevance_inner_mask, name, dtype=dtype)

def sum_relevance(y_true, y_pred):
  return K.sum(y_pred, axis=[1, 2, 3, 4])

class SumRelevance(MeanMetricWrapper):
  def __init__(self,
               name='sum_relevance',
               dtype=None):
    super(SumRelevance, self).__init__(sum_relevance, name, dtype=dtype)
