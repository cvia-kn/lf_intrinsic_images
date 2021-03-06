"""Activations for TensorFlow.
Parag K. Mital, Jan 2016."""
import tensorflow as tf


def lrelu( x, name="lrelu", leak=0.2 ):
    """Leaky rectifier.

    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.

    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def elu(x, name = "elu"):
    with tf.variable_scope(name):
        return tf.nn.elu(x)