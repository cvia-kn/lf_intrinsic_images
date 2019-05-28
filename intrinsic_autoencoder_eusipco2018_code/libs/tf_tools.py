#
# some TensorFlow tools (mostly found online in threads, thanks to all sources)
#

import tensorflow as tf
import numpy as np


#
# this function replaces the stupid standard saver restore,
# it ignores missing variables in the save file.
#
# by RalphMao on GitHub
#
def optimistic_restore(session, save_file):

  reader = tf.train.NewCheckpointReader(save_file)
  saved_shapes = reader.get_variable_to_shape_map()
  var_names = sorted( [(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                      if var.name.split(':')[0] in saved_shapes])
  restore_vars = []
  name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

  with tf.variable_scope('', reuse=True):
    for var_name, saved_var_name in var_names:
      curr_var = name2var[saved_var_name]
      var_shape = curr_var.get_shape().as_list()
      if var_shape == saved_shapes[saved_var_name]:
        restore_vars.append(curr_var)

  saver = tf.train.Saver(restore_vars)
  saver.restore(session, save_file)






####################################################################################
# Class to handle fractional labels for continuous input
####################################################################################

class labelspace:

  # initialize with range and total number of labels
  def __init__( self, lmin=0.0, lmax=1.0, n=100 ):
    self._lmin    = lmin
    self._lmax    = lmax
    self._nlabels = n
   
  
  def labels_to_one_hot( self, labels_dense):

    """Convert class labels from scalars to one-hot vectors."""

    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * self._nlabels
    labels_one_hot = np.zeros(( num_labels, self._nlabels ))
    labels_dense = np.uint16( labels_dense.round() );
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


  def labels_to_fractional_one_hot( self, labels_dense ):

    """Convert class labels from scalars to fractional one-hot vectors.
    Actually, these are "two-hot", with relative weights denoting intermediate
    labels.
    """
    num_labels = labels_dense.shape[0]
    labels_dense = labels_dense.ravel()
    index_offset = np.arange(num_labels) * self._nlabels

    labels_one_hot = np.zeros((num_labels, self._nlabels ))
    labels_floor = np.minimum( self._nlabels-1, np.int16( np.floor( labels_dense ) ))
    labels_floor_p = np.minimum( self._nlabels-1, labels_floor+1 )
    labels_fraction = labels_dense - labels_floor
    labels_floor = np.maximum( 0, labels_floor )

    #code.interact( local = locals() )

    labels_one_hot.flat[index_offset + labels_floor] = 1.0 - labels_fraction
    labels_one_hot.flat[index_offset + labels_floor_p] = labels_fraction

    return labels_one_hot


  def fractional_one_hot_to_labels( self, v ):

    """ Convert a fractional one-hot vector to a disparity value
    in the given range.

    Strategy: find maximum response, then average over direct
    neighbours of the argmax.
    """

    # base offsets into flat array
    off = np.arange( v.shape[0] ) * v.shape[1]
    # neighbouring label offsets
    L = v.argmax(1)
    v = np.maximum( 0.0, v )
    Lp = np.minimum( v.shape[1]-1, L+1 )
    Lm = np.maximum( 0, L-1 )
    # readout values from one-hot vector
    vp = v.flat[ Lp + off ]
    vm = v.flat[ Lm + off ]
    v = v.flat[ L + off ]
    # ... and recompute maximum using local quadratic fit (cf SIFT)
    # TODO: only positive part?
    a = 0.5 * (vp + vm) - v
    b = vp - v - a
    b = b * (a != 0.0)
    a = (a==0.0) * 1e-10 + (a!=0.0) * a
    x = -b / (2.0*a)
    x = x * (x<1.0) * (x>-1.0)
    return L + x #((L+1) * vp + L * v + (L-1) * vm) / (vp + v + vm)


  def labels_to_value( self, labels ):
    """ Convert fractional label values to disparity score
    """
    return labels / ( self._nlabels-1 ) * (self._lmax-self._lmin) + self._lmin


  def value_to_labels( self, disp ):
    """ Convert disparity to fractional label
    """
    return ( disp - self._lmin ) / (self._lmax - self._lmin) * ( self._nlabels-1 )
