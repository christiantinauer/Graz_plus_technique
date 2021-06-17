from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend as K
from tensorflow.python.keras.losses import LossFunctionWrapper, losses_utils

def relevance_guided(y_true, y_pred):
  return -K.sum(y_true * y_pred, axis=[1, 2, 3, 4])

class RelevanceGuided(LossFunctionWrapper):
  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='relevance_guided'):
    super().__init__(relevance_guided, name=name, reduction=reduction)
