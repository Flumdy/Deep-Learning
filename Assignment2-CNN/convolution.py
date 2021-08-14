from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
  """
  Performs 2D convolution given 4D inputs and filter Tensors.
  :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
  :param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
  :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
  :param padding: either "SAME" or "VALID", capitalization matters
  :return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
  """

  assert inputs[3] == filters[3], "inputs 'in channels' is not equal to filters 'in channels"

  num_examples = inputs.shape[0]
  in_height = inputs.shape[1]
  in_width = inputs.shape[2]
  input_in_channels = inputs.shape[3]

  filter_height = filters.shape[0]
  filter_width = filters.shape[1]
  filter_in_channels = filters.shape[2]
  filter_out_channels = filters.shape[3]

  num_examples_stride = strides[0]
  strideY = strides[1]
  strideX = strides[2]
  channels_stride = strides[3]

  # Cleaning padding input
  padX = (filter_height - 1) / 2
  padY = (filter_width - 1) / 2
  inputs = inputs.pad(padX)

  # Calculate output dimensions
  ouput_array = np.array([num_examples, (in_height + 2 * pad_size - filter_height) / strideY + 1], [(in_width + 2 * pad_size - filter_width) / strideX + 1], in_channels)

  for example in range(num_examples): #for each image
    for i in range(filter_height): #for the layer of the filter
      for j in range(filter_width): #for the length of the filter
        for k in range(in_channels): # for the channel being used
          conv_area = inputs[example, i : i + filter_hieght, j : j + filter_width, k] #get each value under the filter
          output_array[example, i, j, k] = np.dot(conv_area, filters)



  return output_array


