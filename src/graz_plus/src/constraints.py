from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.constraints import Constraint
from tensorflow.keras import backend as K

class Neg(Constraint):
  """Constrains the weights to be negative.
  """

  def __call__(self, w):
    return w * K.cast(K.less_equal(w, 0.), K.floatx())
