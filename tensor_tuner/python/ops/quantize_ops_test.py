r"""
Test quantization operations.
"""

import unittest
import numpy as np
import tensorflow as tf

from tensor_tuner.python.ops import quantize_ops


class TestFixedPointQuantForward(unittest.TestCase):
  """ Test the forward results from the FixedPointQuant operation. """

  def test_integer_constants(self):
    """ Test some manually selected integer constants. """
    with tf.Graph().as_default():
      inputs_val = [
          1.0, 2.0, 3.0,
      ]
      inputs = tf.constant(inputs_val, dtype=tf.float32)
      outputs = quantize_ops.fixed_point_quant(inputs, 16, 8, True)

      with tf.Session() as sess:
        outputs_val = sess.run(outputs)

        self.assertEqual(outputs_val.dtype, np.int16)
        self.assertEqual(outputs_val[0], int('0100', 16))
        self.assertEqual(outputs_val[1], int('0200', 16))
        self.assertEqual(outputs_val[2], int('0300', 16))

  def test_real_constants(self):
    """ Test some manually selected real constants. """
    with tf.Graph().as_default():
      inputs_val = [
          1.25, 1.26, 1.27
      ]
      inputs = tf.constant(inputs_val, dtype=tf.float32)
      outputs = quantize_ops.fixed_point_quant(inputs, 16, 8, True)

      with tf.Session() as sess:
        outputs_val = sess.run(outputs)

        self.assertEqual(outputs_val.dtype, np.int16)
        # show the round-to-nearest results
        self.assertEqual(outputs_val[0], int('0140', 16))
        self.assertEqual(outputs_val[1], int('0143', 16))
        self.assertEqual(outputs_val[2], int('0145', 16))


class TestFixedPointQuantBackward(unittest.TestCase):
  """ Test the backward quantization function """

  def test_integer_constants(self):
    """ Compute the backward gradient of the quantization operation.
    """

    with tf.Graph().as_default():
      g = tf.get_default_graph()

      inputs_val = [1.0, 1024.0]

      inputs = tf.constant(inputs_val, dtype=tf.float32)
      outputs = quantize_ops.fixed_point_quant(inputs, 16, 8, True)
      grad_ys = tf.constant([18, 42], dtype=tf.int16)
      grad = [tf.gradients(outputs[0], [inputs], grad_ys=grad_ys[0])[0][0],
              tf.gradients(outputs[1], [inputs], grad_ys=grad_ys[1])[0][1], ]

      min_val = g.get_tensor_by_name(
          'gradients/FixedPointQuant/FixedPointQuant_grad/MinValue:0')
      max_val = g.get_tensor_by_name(
          'gradients/FixedPointQuant/FixedPointQuant_grad/MaxValue:0')
      mask = g.get_tensor_by_name(
          'gradients/FixedPointQuant/FixedPointQuant_grad/Mask:0')

      with tf.Session() as sess:
        _, grad_val, _, _, _ = sess.run(
            [outputs, grad, min_val, max_val, mask])

        self.assertEqual(grad_val[0], 18)
        self.assertEqual(grad_val[1], 0)
