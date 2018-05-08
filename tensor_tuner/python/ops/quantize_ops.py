r"""
Initialize and wrap quantization operations.
"""

import os

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.ops import add_arg_scope

LOCAL_DIR = os.path.dirname(__file__)
TENSOR_TUNER_LIB = os.path.join(
    LOCAL_DIR, '..', '..', '..', 'libtensor_tuner.so')
TENSOR_TUNER = tf.load_op_library(TENSOR_TUNER_LIB)


def _get_min_val(num_fb, num_ib, signed):
  if signed:
    return - float(1 << (num_fb + num_ib)) / (1 << num_fb)
  return 0.0


def _get_max_val(num_fb, num_ib, signed):
  if signed:
    return (float(1 << (num_fb + num_ib)) - 1) / (1 << num_fb)
  return float(1 << (num_fb + num_ib - 1)) / (1 << num_fb)


def _get_tensor_type(num_bits, signed):
  if num_bits == 16:
    return tf.int16 if signed else tf.uint16

  raise ValueError('Unrecognized num_bits and signed.')


@add_arg_scope
def fixed_point_quant(inputs, num_bits, num_fb, signed, scope=None):
  """
  Quantize the input tensor into fixed-point data type.

  Args:
    inputs: an input tensor
    num_bits: number of total bits
    num_fb: number of fraction bits
    signed: whether the fixed-point data type is signed or not
    scope: optional name_scope

  Returns:
    A fixed-point quantized tensor
  """

  with tf.name_scope(scope, 'FixedPointQuant', values=[inputs]):
    if signed:
      num_ib = num_bits - 1 - num_fb
    else:
      num_ib = num_bits - num_fb
    quant_type = _get_tensor_type(num_bits, signed)

    return TENSOR_TUNER.fixed_point_quant(
        inputs, quant_type=quant_type, num_fb=num_fb, num_ib=num_ib, signed=signed)


@ops.RegisterGradient("FixedPointQuant")
def _fixed_point_quant_grad(op, grad):
  """ The gradient for FixedPointQuant.

  We use straight-throught estimation (STE) to resolve the gradient of
  the quantization function.

  The input gradient should be a quantized value,
  and the output gradient will also be quantized.
  """

  input_tensor = op.inputs[0]
  shape = input_tensor.shape
  num_fb, num_ib, signed = [op.get_attr(x)
                            for x in ['num_fb', 'num_ib', 'signed']]

  # compute attribute values
  max_val = tf.constant(_get_max_val(num_fb, num_ib, signed),
                        tf.float32, name='MaxValue')
  min_val = tf.constant(_get_min_val(num_fb, num_ib, signed),
                        tf.float32, name='MinValue')

  # whether to pass through the gradient
  mask = tf.logical_and(tf.less_equal(input_tensor, max_val),
                        tf.greater_equal(input_tensor, min_val),
                        name='Mask')

  prop_grad = tf.where(mask, grad, tf.zeros(shape, dtype=grad.dtype))

  return [prop_grad]
