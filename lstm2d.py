# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A small library of functions dealing with LSTMs applied to images.

Tensors in this library generally have the shape (num_images, height, width,
depth).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import lstm1d

from tensorflow.contrib import rnn, slim


def _shape(tensor):
  """Get the shape of a tensor as an int list."""
  return tensor.get_shape().as_list()


def images_to_sequence(inputs):
  """Convert a batch of images into a batch of sequences.
  Args:
    inputs: a (num_images, height, width, depth) tensor
  Returns:
    (width, num_images*height, depth) sequence tensor
  """
  _, _, width, depth = _shape(inputs)
  s = tf.shape(inputs)
  batch_size, height = s[0], s[1]
  transposed = tf.transpose(inputs, [2, 0, 1, 3])
  return tf.reshape(transposed, [width, batch_size * height, depth])


def sequence_to_images(inputs, height):
  """Convert a batch of sequences into a batch of images.
  Args:
    inputs: (num_steps, num_batches, depth) sequence tensor
    height: the height of the images
      Currently supports `'channels_first'` and `'channels_last'`.
  Returns:
    A tensor representing the output of the operation.
  """
  width, num_batches, depth = _shape(inputs)
  if num_batches is None:
    num_batches = -1
  else:
    num_batches = num_batches // height
  reshaped = tf.reshape(inputs, [width, num_batches, height, depth])
  return tf.transpose(reshaped, [1, 2, 0, 3])


def _get_cell(num_units, cell_type='LSTM'):
  if cell_type == 'LSTM':
      return rnn.LSTMCell(num_units, initializer=slim.xavier_initializer())
  if cell_type == 'GLSTM':
      return rnn.GLSTMCell(num_units, initializer=slim.xavier_initializer())
  if cell_type == 'GRU':
      return rnn.GRUCell(num_units, kernel_initializer=slim.xavier_initializer())
  raise NotImplementedError(cell_type + " not supported by ndlstm.")


def horizontal_lstm(images, num_filters_out, cell_type='LSTM', scope=None):
  """Run an LSTM bidirectionally over all the rows of each image.

  Args:
    images: (num_images, height, width, depth) tensor
    num_filters_out: output depth
    scope: optional scope name
    cell_type: type of rnn cell to use.

  Returns:
    (num_images, height, width, num_filters_out) tensor, where
    num_steps is width and new num_batches is num_image_batches * height
  """
  with tf.variable_scope(scope, "HorizontalLstm", [images]):
    _, height, _, _ = _shape(images)
    sequence = images_to_sequence(images)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        _get_cell(num_filters_out // 2, cell_type),
        _get_cell(num_filters_out // 2, cell_type),
        sequence,
        dtype=sequence.dtype)
    output_sequence = tf.concat(outputs, 2)
    output = sequence_to_images(output_sequence, height)
    return output


def separable_lstm(inputs, num_filters_out, nhidden=None, cell_type='LSTM',
                   data_format='NHWC', scope=None):
  """Run bidirectional LSTMs first horizontally then vertically.

  Args:
    inputs: (batch_size, height, width, depth) tensor
    num_filters_out: output layer depth
    nhidden: hidden layer depth,
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    cell_type: type of rnn cell to use.
    scope: optional scope name
  Returns:
    (num_images, height, width, num_filters_out) tensor
  """
  with tf.variable_scope(scope, "SeparableLstm", [inputs]):
    if data_format not in ('NHWC', 'NCHW'):
      raise ValueError('data_format has to be either NCHW or NHWC.')
    df = ('channels_last'
          if data_format and data_format.startswith('NH') else 'channels_first')
    if df == 'channels_first':
      inputs = tf.transpose(inputs, [0, 2, 3, 1])
    if nhidden is None:
      nhidden = num_filters_out
    hidden = horizontal_lstm(inputs, nhidden, cell_type=cell_type)
    with tf.variable_scope("vertical"):
      transposed = tf.transpose(hidden, [0, 2, 1, 3])
      output_transposed = horizontal_lstm(transposed, num_filters_out)
    if df == 'channels_last':
      return tf.transpose(output_transposed, [0, 2, 1, 3])
    return tf.transpose(output_transposed, [0, 3, 2, 1])


def reduce_to_sequence(images, num_filters_out, scope=None):
  """Reduce an image to a sequence by scanning an LSTM vertically.

  Args:
    images: (num_images, height, width, depth) tensor
    num_filters_out: output layer depth
    scope: optional scope name

  Returns:
    A (width, num_images, num_filters_out) sequence.
  """
  with tf.variable_scope(scope, "ReduceToSequence", [images]):
    batch_size, height, width, depth = _shape(images)
    transposed = tf.transpose(images, [1, 0, 2, 3])
    reshaped = tf.reshape(transposed, [height, batch_size * width, depth])
    reduced = lstm1d.sequence_to_final(reshaped, num_filters_out)
    output = tf.reshape(reduced, [batch_size, width, num_filters_out])
    return output


def reduce_to_final(images, num_filters_out, nhidden=None, scope=None):
  """Reduce an image to a final state by running two LSTMs.

  Args:
    images: (num_images, height, width, depth) tensor
    num_filters_out: output layer depth
    nhidden: hidden layer depth (defaults to num_filters_out)
    scope: optional scope name

  Returns:
    A (num_images, num_filters_out) batch.
  """
  with tf.variable_scope(scope, "ReduceToFinal", [images]):
    nhidden = nhidden or num_filters_out
    batch_size, height, width, depth = _shape(images)
    transposed = tf.transpose(images, [1, 0, 2, 3])
    reshaped = tf.reshape(transposed, [height, batch_size * width, depth])
    with tf.variable_scope("reduce1"):
      reduced = lstm1d.sequence_to_final(reshaped, nhidden)
      transposed_hidden = tf.reshape(reduced, [batch_size, width, nhidden])
      hidden = tf.transpose(transposed_hidden, [1, 0, 2])
    with tf.variable_scope("reduce2"):
      output = lstm1d.sequence_to_final(hidden, num_filters_out)
    return output
