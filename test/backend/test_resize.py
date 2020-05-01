from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from typing import Any, List, Callable, Union, Optional, Text
import tensorflow as tf


def cartesian(arrays, out=None):
    # type: (List[np.ndarray], np.ndarray) -> np.ndarray
    """
    From https://stackoverflow.com/a/1235363
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def interpolate_1d_with_x(data,                                             # type: np.ndarray
                          scale_factor,                                     # type: float
                          x,                                                # type: float
                          get_coeffs,                                       # type: Callable[[float], np.ndarray]
                          roi=None,                                         # type: np.ndarray
                          extrapolation_value=0.0,                          # type: float
                          coordinate_transformation_mode='half_pixel',      # type: Text
                          exclude_outside=False,                            # type: bool
                          ):                                                # type: (...) -> np.ndarray
    def get_neighbor_idxes(x, n, limit):  # type: (float, int, int) -> np.ndarray
        """
        Return the n nearest indexes to x among [0, limit), prefer the indexes smaller than x.
        As a result, the ratio must be in (0, 1]
        Examples:
        get_neighbor_idxes(4, 2, 10) == [3, 4]
        get_neighbor_idxes(4, 3, 10) == [3, 4, 5]
        get_neighbor_idxes(4.4, 3, 10) == [3, 4, 5]
        get_neighbor_idxes(4.5, 3, 10) == [3, 4, 5]
        get_neighbor_idxes(4.6, 3, 10) == [4, 5, 6]
        get_neighbor_idxes(4.4, 1, 10) == [4]
        get_neighbor_idxes(4.6, 1, 10) == [5]
        :param x:
        :param n: the number of the wanted indexes
        :param limit: the maximum value of index
        :return: An np.array containing n nearest indexes in ascending order
        """
        idxes = sorted(range(limit), key=lambda idx: (abs(x - idx), idx))[:n]
        idxes = sorted(idxes)
        return np.array(idxes)

    def get_neighbor(x, n, data):  # type: (float, int, np.ndarray) -> np.ndarray
        """
        Pad `data` in 'edge' mode, and get n nearest elements in the padded array and their indexes in the original
        array
        :param x: center index (in the unpadded coordinate system) of the found nearest elements.
        :param n: the number of neighbors.
        :param data: the array
        :return: A tuple containing the indexes of neighbor elements (the index can be smaller than 0 or higher than
        len(data)) and the value of these elements
        """
        pad_width = np.ceil(n / 2).astype(np.int)
        padded = np.pad(data, pad_width, mode='edge')
        x += pad_width

        idxes = get_neighbor_idxes(x, n, len(padded))
        ret = padded[idxes]
        return idxes - pad_width, ret

    input_width = len(data)
    output_width = scale_factor * input_width
    if coordinate_transformation_mode == 'align_corners':
        if output_width == 1:
            x_ori = 0.
        else:
            x_ori = x * (input_width - 1) / (output_width - 1)
    elif coordinate_transformation_mode == 'asymmetric':
        x_ori = x / scale_factor
    elif coordinate_transformation_mode == 'tf_crop_and_resize':
        if output_width == 1:
            x_ori = (roi[1] - roi[0]) * (input_width - 1) / 2
        else:
            x_ori = x * (roi[1] - roi[0]) * \
                (input_width - 1) / (output_width - 1)
        x_ori += (roi[0] * (input_width - 1))
        # Return extrapolation_value directly as what TF CropAndResize does
        if x_ori < 0 or x_ori > input_width - 1:
            return extrapolation_value
    elif coordinate_transformation_mode == 'tf_half_pixel_for_nn':
        x_ori = (x + 0.5) / scale_factor
    elif coordinate_transformation_mode == 'pytorch_half_pixel':
        if output_width == 1:
            x_ori = -0.5
        else:
            x_ori = (x + 0.5) / scale_factor - 0.5
    else:  # coordinate_transformation_mode == 'half_pixel'
        x_ori = (x + 0.5) / scale_factor - 0.5
    x_ori_int = np.floor(x_ori).astype(np.int).item()

    # ratio must be in (0, 1] since we prefer the pixel on the left of `x_ori`
    if x_ori.is_integer():
        ratio = 1
    else:
        ratio = x_ori - x_ori_int

    coeffs = get_coeffs(ratio)
    n = len(coeffs)

    idxes, points = get_neighbor(x_ori, n, data)

    if exclude_outside:
        for i, idx in enumerate(idxes):
            if idx < 0 or idx >= input_width:
                coeffs[i] = 0
        coeffs /= sum(coeffs)

    return np.dot(coeffs, points).item()


def interpolate_nd_with_x(data,                      # type: np.ndarray
                          n,                         # type: int
                          scale_factors,             # type: List[float]
                          x,                         # type: List[float]
                          get_coeffs,                # type: Callable[[float], np.ndarray]
                          roi=None,                  # type: np.ndarray
                          **kwargs                   # type: Any
                          ):                         # type: (...) -> np.ndarray
    if n == 1:
        return interpolate_1d_with_x(data, scale_factors[0], x[0], get_coeffs, roi=roi,
                                     **kwargs)
    return interpolate_1d_with_x(
        [interpolate_nd_with_x(data[i], n - 1, scale_factors[1:], x[1:], get_coeffs,
                               roi=None if roi is None else np.concatenate(
                                   [roi[1:n], roi[n + 1:]]),
                               **kwargs)
         for i in range(data.shape[0])], scale_factors[0], x[0], get_coeffs,
        roi=None if roi is None else [roi[0], roi[n]], **kwargs)


def interpolate_nd(data,                      # type: np.ndarray
                   get_coeffs,                # type: Callable[[float], np.ndarray]
                   output_size=None,          # type: Optional[List[int]]
                   scale_factors=None,        # type: Optional[List[float]]
                   roi=None,                  # type: np.ndarray
                   **kwargs                   # type: Any
                   ):                         # type: (...) -> np.ndarray
    def get_all_coords(data):   # type: (np.ndarray) -> np.ndarray
        return cartesian([list(range(data.shape[i])) for i in range(len(data.shape))])

    assert output_size is not None or scale_factors is not None
    if output_size is not None:
        scale_factors = np.array(output_size) / np.array(data.shape)
    else:
        output_size = (scale_factors * np.array(data.shape)).astype(np.int)
    assert scale_factors is not None

    ret = np.zeros(output_size)
    for x in get_all_coords(ret):
        ret[tuple(x)] = interpolate_nd_with_x(data, len(data.shape), scale_factors, x, get_coeffs, roi=roi,
                                              **kwargs)
    return ret


def cubic_coeffs(ratio, A=-0.75):   # type: (float, float) -> np.ndarray
    coeffs = [((A * (ratio + 1) - 5 * A) * (ratio + 1) + 8 * A) * (ratio + 1) - 4 * A,
              ((A + 2) * ratio - (A + 3)) * ratio * ratio + 1,
              ((A + 2) * (1 - ratio) - (A + 3)) * (1 - ratio) * (1 - ratio) + 1,
              ((A * ((1 - ratio) + 1) - 5 * A) * ((1 - ratio) + 1) + 8 * A) * ((1 - ratio) + 1) - 4 * A]

    return np.array(coeffs)


def linear_coeffs(ratio):           # type: (float) -> np.ndarray
    return np.array([1 - ratio, ratio])


def nearest_coeffs(ratio, mode='round_prefer_floor'):          # type: (float, Text) -> np.ndarray
    if type(ratio) == int or ratio.is_integer():
        return np.array([0, 1])
    elif mode == 'round_prefer_floor':
        return np.array([ratio <= 0.5, ratio > 0.5])
    elif mode == 'round_prefer_ceil':
        return np.array([ratio < 0.5, ratio >= 0.5])
    elif mode == 'floor':
        return np.array([1, 0])
    elif mode == 'ceil':
        return np.array([0, 1])

#########################
# Test cases start here #
#########################

def export_resize_upsample_scales_nearest():  # type: () -> None
  print('#######################################')  
  print('# test_resize_upsample_scales_nearest #')
  print('#######################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='nearest',
  )

  data = np.array([[[
      [1, 2],
      [3, 4],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
#  scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)
  scales = np.array([1.0, 1.0, 2.7, 3.9], dtype=np.float32)
  print('data = ')
  print(data)
  print('scales = ', scales)

  # [[[[1. 1. 1. 2. 2. 2.]
  #    [1. 1. 1. 2. 2. 2.]
  #    [3. 3. 3. 4. 4. 4.]
  #    [3. 3. 3. 4. 4. 4.]]]]
  expected = interpolate_nd(
      data, lambda x: nearest_coeffs(x, mode='floor'), scale_factors=scales, coordinate_transformation_mode='tf_half_pixel_for_nn').astype(np.float32)
  print('expected = ')
  print(expected)

  h_w_scale = scales[2:]
  h_w_shape = list(data.shape)[2:]
  new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data, 
                           size=new_h_w_shape, 
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_nearest_neighbor(data, new_h_w_shape, half_pixel_centers=False)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_nearest_neighbor(data, new_h_w_shape, half_pixel_centers=False)')
  print('output = ', output2)
  try :
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_downsample_scales_nearest():  # type: () -> None
  print('#########################################')
  print('# test_resize_downsample_scales_nearest #')
  print('#########################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='nearest',
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
  print('data = ')
  print(data)
  print('scales = ', scales)

  # [[[[1. 3.]]]]
  expected = interpolate_nd(
      data, nearest_coeffs, scale_factors=scales).astype(np.float32)
  print('expected = ')
  print(expected)

  h_w_scale = scales[2:]
  h_w_shape = list(data.shape)[2:]
  new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output = tf.image.resize(data,
                           size=new_h_w_shape,
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output = tf.transpose(output, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)')
  print('output = ', output)
  output = tf.compat.v1.image.resize_nearest_neighbor(data, new_h_w_shape)
  output = tf.transpose(output, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_nearest_neighbor(data, new_h_w_shape)')
  print('output = ', output)
  np.testing.assert_almost_equal(output, expected)

def export_resize_upsample_sizes_nearest():  # type: () -> None
  print('######################################')
  print('# test_resize_upsample_sizes_nearest #')
  print('######################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='nearest',
  )

  data = np.array([[[
      [1, 2],
      [3, 4],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 7, 8], dtype=np.int64)
  print('data = ')
  print(data)
  print('sizes = ', sizes)

  # [[[[1. 1. 1. 1. 2. 2. 2. 2.]
  #    [1. 1. 1. 1. 2. 2. 2. 2.]
  #    [1. 1. 1. 1. 2. 2. 2. 2.]
  #    [1. 1. 1. 1. 2. 2. 2. 2.]
  #    [3. 3. 3. 3. 4. 4. 4. 4.]
  #    [3. 3. 3. 3. 4. 4. 4. 4.]
  #    [3. 3. 3. 3. 4. 4. 4. 4.]]]]
  expected = interpolate_nd(
      data, nearest_coeffs, output_size=sizes).astype(np.float32)
  print('expected = ')
  print(expected)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output = tf.image.resize(data,
                           size=sizes[2:],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output = tf.transpose(output, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=sizes[2:], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)')
  print('output = ', output)
  output = tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:])
  output = tf.transpose(output, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:])')
  print("output = ", output)
  np.testing.assert_almost_equal(output, expected)

def export_resize_downsample_sizes_nearest():  # type: () -> None
  print('########################################')  
  print('# test_resize_downsample_sizes_nearest #')
  print('########################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='nearest',
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 1, 3], dtype=np.int64)
  print('data = ')
  print(data)
  print('sizes = ', sizes)

  # [[[[1. 3.]]]]
  expected = interpolate_nd(
      data, nearest_coeffs, output_size=sizes).astype(np.float32)
  print('expected = ')
  print(expected)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=sizes[2:],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=sizes[2:], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:])
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:])')
  print('output = ', output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_upsample_scales_linear():  # type: () -> None
  print('######################################')
  print('# test_resize_upsample_scales_linear #')
  print('######################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='linear',
  )

  data = np.array([[[
      [1, 2],
      [3, 4],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
#  scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
  scales = np.array([1.0, 1.0, 2.5, 2.9], dtype=np.float32)
  print('data = ')
  print(data)
  print('scales = ', scales)

  # [[[[1.   1.25 1.75 2.  ]
  #    [1.5  1.75 2.25 2.5 ]
  #    [2.5  2.75 3.25 3.5 ]
  #    [3.   3.25 3.75 4.  ]]]]
  expected = interpolate_nd(
      data, linear_coeffs, scale_factors=scales).astype(np.float32)
  print('expected = ')
  print(expected)

  h_w_scale = scales[2:]
  h_w_shape = list(data.shape)[2:]
  new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                           size=new_h_w_shape,
                           method=tf.image.ResizeMethod.BILINEAR)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.BILINEAR)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_bilinear(data, new_h_w_shape)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bilinear(data, new_h_w_shape)')
  print('output = ', output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_upsample_scales_linear_align_corners():  # type: () -> None
  print('####################################################')
  print('# test_resize_upsample_scales_linear_align_corners #')
  print('####################################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='linear',
      coordinate_transformation_mode='align_corners'
  )

  data = np.array([[[
      [1, 2],
      [3, 4],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
  print('data = ')
  print(data)
  print('scales = ', scales)

  # [[[[1.         1.33333333 1.66666667 2.        ]
  #    [1.66666667 2.         2.33333333 2.66666667]
  #    [2.33333333 2.66666667 3.         3.33333333]
  #    [3.         3.33333333 3.66666667 4.        ]]]]
  expected = interpolate_nd(
      data, linear_coeffs, scale_factors=scales, coordinate_transformation_mode='align_corners').astype(np.float32)
  print('expected = ')
  print(expected)

  h_w_scale = scales[2:]
  h_w_shape = list(data.shape)[2:]
  new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=new_h_w_shape,
                            method=tf.image.ResizeMethod.BILINEAR)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.BILINEAR)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_bilinear(data, new_h_w_shape, align_corners=True)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bilinear(data, new_h_w_shape, align_corners=True)')
  print('output = ', output2)
  output3 = tf.image.resize_with_pad(data, 
                                     target_height=new_h_w_shape[0], 
                                     target_width=new_h_w_shape[1], 
                                     method=tf.image.ResizeMethod.BILINEAR)
  output3 = tf.transpose(output3, perm=[0, 3, 1, 2])
  print('tf.image.resize_with_pad(data, target_height=new_h_w_shape[0], target_width=new_h_w_shape[1], method=tf.image.ResizeMethod.BILINEAR)')
  print('output3 = ', output3)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output3, expected)
  except AssertionError as err:
    print(format(err))  

def export_resize_downsample_scales_linear():  # type: () -> None
  print('########################################')
  print('# test_resize_downsample_scales_linear #')
  print('########################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='linear',
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
  print('data = ')
  print(data)
  print('scales = ', scales)

  # [[[[2.6666665 4.3333331]]]]
  expected = interpolate_nd(
      data, linear_coeffs, scale_factors=scales).astype(np.float32)
  print('expected = ')
  print(expected)

  h_w_scale = scales[2:]
  h_w_shape = list(data.shape)[2:]
  new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=new_h_w_shape,
                            method=tf.image.ResizeMethod.BILINEAR)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.BILINEAR)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_bilinear(data, new_h_w_shape)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bilinear(data, new_h_w_shape)')
  print('output = ', output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_downsample_scales_linear_align_corners():  # type: () -> None
  print('######################################################')  
  print('# test_resize_downsample_scales_linear_align_corners #')
  print('######################################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='linear',
      coordinate_transformation_mode='align_corners'
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
  print('data = ')
  print(data)
  print('scales = ', scales)

  # [[[[1.       3.142857]]]]
  expected = interpolate_nd(
      data, linear_coeffs, scale_factors=scales, coordinate_transformation_mode='align_corners').astype(np.float32)
  print('expected = ')
  print(expected)

  h_w_scale = scales[2:]
  h_w_shape = list(data.shape)[2:]
  new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=new_h_w_shape,
                            method=tf.image.ResizeMethod.BILINEAR)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.BILINEAR')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_bilinear(data, new_h_w_shape, align_corners=True)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bilinear(data, new_h_w_shape, align_corners=True)')
  print('output = ', output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_upsample_scales_cubic():  # type: () -> None
  print('#####################################')
  print('# test_resize_upsample_scales_cubic #')
  print('#####################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='cubic',
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
  print('data = ')
  print(data)
  print('scales = ', scales)

  # [[[[ 0.47265625  0.76953125  1.24609375  1.875       2.28125
  #      2.91015625  3.38671875  3.68359375]
  #    [ 1.66015625  1.95703125  2.43359375  3.0625      3.46875
  #      4.09765625  4.57421875  4.87109375]
  #    [ 3.56640625  3.86328125  4.33984375  4.96875     5.375
  #      6.00390625  6.48046875  6.77734375]
  #    [ 6.08203125  6.37890625  6.85546875  7.484375    7.890625
  #      8.51953125  8.99609375  9.29296875]
  #    [ 7.70703125  8.00390625  8.48046875  9.109375    9.515625
  #     10.14453125 10.62109375 10.91796875]
  #    [10.22265625 10.51953125 10.99609375 11.625      12.03125
  #     12.66015625 13.13671875 13.43359375]
  #    [12.12890625 12.42578125 12.90234375 13.53125    13.9375
  #     14.56640625 15.04296875 15.33984375]
  #    [13.31640625 13.61328125 14.08984375 14.71875    15.125
  #     15.75390625 16.23046875 16.52734375]]]]
  expected = interpolate_nd(
      data, cubic_coeffs, scale_factors=scales).astype(np.float32)
  print('expected = ') 
  print(expected)

  h_w_scale = scales[2:]
  h_w_shape = list(data.shape)[2:]
  new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=new_h_w_shape,
                            method=tf.image.ResizeMethod.BICUBIC)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.BICUBIC)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_bicubic(data, new_h_w_shape)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bicubic(data, new_h_w_shape)')
  print('output = ', output2)
  output3 = tf.image.resize_with_pad(data, 
                                     target_height=new_h_w_shape[0], 
                                     target_width=new_h_w_shape[1], 
                                     method=tf.image.ResizeMethod.BICUBIC)
  output3 = tf.transpose(output3, perm=[0, 3, 1, 2])
  print('tf.image.resize_with_pad(data, target_height=new_h_w_shape[0], target_width=new_h_w_shape[1], method=tf.image.ResizeMethod.BICUBIC)')
  print('output = ', output3)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output3, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_upsample_scales_cubic_align_corners():  # type: () -> None
  print('###################################################')  
  print('# test_resize_upsample_scales_cubic_align_corners #')
  print('###################################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='cubic',
      coordinate_transformation_mode='align_corners'
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
  print('data = ')
  print(data)
  print('scales = ', scales)

  # [[[[ 1.          1.34110787  1.80029155  2.32944606  2.67055394
  #      3.19970845  3.65889213  4.        ]
  #    [ 2.36443149  2.70553936  3.16472303  3.69387755  4.03498542
  #      4.56413994  5.02332362  5.36443149]
  #    [ 4.20116618  4.54227405  5.00145773  5.53061224  5.87172012
  #      6.40087464  6.86005831  7.20116618]
  #    [ 6.31778426  6.65889213  7.1180758   7.64723032  7.98833819
  #      8.51749271  8.97667638  9.31778426]
  #    [ 7.68221574  8.02332362  8.48250729  9.01166181  9.35276968
  #      9.8819242  10.34110787 10.68221574]
  #    [ 9.79883382 10.13994169 10.59912536 11.12827988 11.46938776
  #     11.99854227 12.45772595 12.79883382]
  #    [11.63556851 11.97667638 12.43586006 12.96501458 13.30612245
  #     13.83527697 14.29446064 14.63556851]
  #    [13.         13.34110787 13.80029155 14.32944606 14.67055394
  #     15.19970845 15.65889213 16.        ]]]]
  expected = interpolate_nd(
      data, cubic_coeffs, scale_factors=scales, coordinate_transformation_mode='align_corners').astype(np.float32)
  print('expected = ')
  print(expected)

  h_w_scale = scales[2:]
  h_w_shape = list(data.shape)[2:]
  new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=new_h_w_shape,
                            method=tf.image.ResizeMethod.BICUBIC)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.BICUBIC)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_bicubic(data, new_h_w_shape, align_corners=True)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bicubic(data, new_h_w_shape, align_corners=True)')
  print("output = ", output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_downsample_scales_cubic():  # type: () -> None
  print('#######################################')
  print('# test_resize_downsample_scales_cubic #')
  print('#######################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='cubic',
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
  print('data = ')
  print(data)
  print('scales = ', scales)

  # [[[[ 1.47119141  2.78125     4.08251953]
  #    [ 6.71142578  8.02148438  9.32275391]
  #    [11.91650391 13.2265625  14.52783203]]]]
  expected = interpolate_nd(
      data, cubic_coeffs, scale_factors=scales).astype(np.float32)
  print('expected = ')
  print(expected)

  h_w_scale = scales[2:]
  h_w_shape = list(data.shape)[2:]
  new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=new_h_w_shape,
                            method=tf.image.ResizeMethod.BICUBIC)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.BICUBIC)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_bicubic(data, new_h_w_shape)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bicubic(data, new_h_w_shape)')
  print('output = ', output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_downsample_scales_cubic_align_corners():  # type: () -> None
  print('#####################################################')  
  print('# test_resize_downsample_scales_cubic_align_corners #')
  print('#####################################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='cubic',
      coordinate_transformation_mode='align_corners'
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
  print('data = ')
  print(data)
  print('scales = ', scales)

  # [[[[ 1.          2.39519159  3.79038317]
  #    [ 6.58076634  7.97595793  9.37114951]
  #    [12.16153268 13.55672427 14.95191585]]]]
  expected = interpolate_nd(
      data, cubic_coeffs, scale_factors=scales, coordinate_transformation_mode='align_corners').astype(np.float32)
  print('expected = ')
  print(expected)

  h_w_scale = scales[2:]
  h_w_shape = list(data.shape)[2:]
  new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=new_h_w_shape,
                            method=tf.image.ResizeMethod.BICUBIC)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.BICUBIC)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_bicubic(data, new_h_w_shape, align_corners=True)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bicubic(data, new_h_w_shape, align_corners=True)')
  print('output = ', output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_upsample_sizes_cubic():  # type: () -> None
  print('####################################')
  print('# test_resize_upsample_sizes_cubic #')
  print('####################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='cubic',
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 9, 10], dtype=np.int64)
  print('data = ')
  print(data)
  print('sizes = ', sizes)

  # [[[[ 0.45507922  0.64057922  0.97157922  1.42257922  1.90732922
  #      2.22332922  2.70807922  3.15907922  3.49007922  3.67557922]
  #    [ 1.39437963  1.57987963  1.91087963  2.36187963  2.84662963
  #      3.16262963  3.64737963  4.09837963  4.42937963  4.61487963]
  #    [ 2.95130693  3.13680693  3.46780693  3.91880693  4.40355693
  #      4.71955693  5.20430693  5.65530693  5.98630693  6.17180693]
  #    [ 5.20525069  5.39075069  5.72175069  6.17275069  6.65750069
  #      6.97350069  7.45825069  7.90925069  8.24025069  8.42575069]
  #    [ 6.88975     7.07525     7.40625     7.85725     8.342
  #      8.658       9.14275     9.59375     9.92475    10.11025   ]
  #    [ 8.57424931  8.75974931  9.09074931  9.54174931 10.02649931
  #     10.34249931 10.82724931 11.27824931 11.60924931 11.79474931]
  #    [10.82819307 11.01369307 11.34469307 11.79569307 12.28044307
  #     12.59644307 13.08119307 13.53219307 13.86319307 14.04869307]
  #    [12.38512037 12.57062037 12.90162037 13.35262037 13.83737037
  #     14.15337037 14.63812037 15.08912037 15.42012037 15.60562037]
  #    [13.32442078 13.50992078 13.84092078 14.29192078 14.77667078
  #     15.09267078 15.57742078 16.02842078 16.35942078 16.54492078]]]]
  expected = interpolate_nd(
      data, cubic_coeffs, output_size=sizes).astype(np.float32)
  print('expected = ')
  print(expected)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=sizes[2:],
                            method=tf.image.ResizeMethod.BICUBIC)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.BICUBIC)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_bicubic(data, sizes[2:])
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bicubic(data, sizes[2:])')
  print('output = ', output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_downsample_sizes_cubic():  # type: () -> None
  print('######################################')
  print('# test_resize_downsample_sizes_cubic #')
  print('######################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='cubic',
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 3, 3], dtype=np.int64)
  print('data = ')
  print(data)
  print('sizes = ', sizes)

  # [[[[ 1.63078704  3.00462963  4.37847222]
  #    [ 7.12615741  8.5         9.87384259]
  #    [12.62152778 13.99537037 15.36921296]]]]
  expected = interpolate_nd(
      data, cubic_coeffs, output_size=sizes).astype(np.float32)
  print('expected = ')
  print(expected)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=sizes[2:],
                            method=tf.image.ResizeMethod.BICUBIC)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.BICUBIC)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_bicubic(data, sizes[2:])
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bicubic(data, sizes[2:])')
  print('output = ', output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_upsample_scales_cubic_A_n0p5_exclude_outside():  # type: () -> None
  print('############################################################')
  print('# test_resize_upsample_scales_cubic_A_n0p5_exclude_outside #')
  print('############################################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='cubic',
      cubic_coeff_a=-0.5,
      exclude_outside=True
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
  print('data = ')
  print(data)
  print('scales = ', scales)

  # [[[[ 0.55882353  0.81494204  1.35698249  1.89705882  2.39705882
  #      2.93713516  3.47917561  3.73529412]
  #    [ 1.58329755  1.83941606  2.38145651  2.92153285  3.42153285
  #      3.96160918  4.50364964  4.75976814]
  #    [ 3.75145936  4.00757787  4.54961832  5.08969466  5.58969466
  #      6.12977099  6.67181144  6.92792995]
  #    [ 5.91176471  6.16788321  6.70992366  7.25        7.75
  #      8.29007634  8.83211679  9.08823529]
  #    [ 7.91176471  8.16788321  8.70992366  9.25        9.75
  #     10.29007634 10.83211679 11.08823529]
  #    [10.07207005 10.32818856 10.87022901 11.41030534 11.91030534
  #     12.45038168 12.99242213 13.24854064]
  #    [12.24023186 12.49635036 13.03839082 13.57846715 14.07846715
  #     14.61854349 15.16058394 15.41670245]
  #    [13.26470588 13.52082439 14.06286484 14.60294118 15.10294118
  #     15.64301751 16.18505796 16.44117647]]]]
  expected = interpolate_nd(data, lambda x: cubic_coeffs(x, A=-0.5), scale_factors=scales,
      exclude_outside=True).astype(np.float32)
  print('expected = ')
  print(expected)

  h_w_scale = scales[2:]
  h_w_shape = list(data.shape)[2:]
  new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=new_h_w_shape,
                            method=tf.image.ResizeMethod.BICUBIC)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.BICUBIC)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_bicubic(data, new_h_w_shape, half_pixel_centers=True)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bicubic(data, new_h_w_shape, half_pixel_centers=True)')
  print('output = ', output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_downsample_scales_cubic_A_n0p5_exclude_outside():  # type: () -> None
  print('##############################################################')
  print('# test_resize_downsample_scales_cubic_A_n0p5_exclude_outside #')
  print('##############################################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='cubic',
      cubic_coeff_a=-0.5,
      exclude_outside=True
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
  sizes = np.array([1, 1, 3, 3], dtype=np.int64)
  print('data = ')
  print(data)
  print('scales = ', scales)

  # [[[[ 1.36812675  2.6695014   4.0133367 ]
  #    [ 6.57362535  7.875       9.2188353 ]
  #    [11.94896657 13.25034122 14.59417652]]]]
  expected = interpolate_nd(data, lambda x: cubic_coeffs(x, A=-0.5), scale_factors=scales,
      exclude_outside=True).astype(np.float32)
  print('using scales = ', scales)
  print('expected = ')
  print(expected)
  expected = interpolate_nd(data, lambda x: cubic_coeffs(x, A=-0.5), output_size=sizes,
      exclude_outside=True).astype(np.float32)
  print('using sizes = ', sizes)
  print('expected = ') 
  print(expected)

  h_w_scale = scales[2:]
  h_w_shape = list(data.shape)[2:]
  new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=new_h_w_shape,
                            method=tf.image.ResizeMethod.BICUBIC)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.BICUBIC)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_bicubic(data, new_h_w_shape, half_pixel_centers=True)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bicubic(data, new_h_w_shape, half_pixel_centers=True)')
  print('output = ', output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_upsample_scales_cubic_asymmetric():  # type: () -> None
  print('################################################')  
  print('# test_resize_upsample_scales_cubic_asymmetric #')
  print('################################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='cubic',
      coordinate_transformation_mode='asymmetric'
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
  roi = np.array([], dtype=np.float32)
  print('data = ')
  print(data)
  print('scales = ', scales)

  # [[[[ 1.       1.40625  2.       2.5      3.       3.59375  4.
  #      4.09375]
  #    [ 2.625    3.03125  3.625    4.125    4.625    5.21875  5.625
  #      5.71875]
  #    [ 5.       5.40625  6.       6.5      7.       7.59375  8.
  #      8.09375]
  #    [ 7.       7.40625  8.       8.5      9.       9.59375 10.
  #     10.09375]
  #    [ 9.       9.40625 10.      10.5     11.      11.59375 12.
  #     12.09375]
  #    [11.375   11.78125 12.375   12.875   13.375   13.96875 14.375
  #     14.46875]
  #    [13.      13.40625 14.      14.5     15.      15.59375 16.
  #     16.09375]
  #    [13.375   13.78125 14.375   14.875   15.375   15.96875 16.375
  #     16.46875]]]]
  expected = interpolate_nd(data, lambda x: cubic_coeffs(x, A=-0.75), scale_factors=scales,
      coordinate_transformation_mode='asymmetric').astype(np.float32)
  print('expected = ')
  print(expected)

  h_w_scale = scales[2:]
  h_w_shape = list(data.shape)[2:]
  new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=new_h_w_shape,
                            method=tf.image.ResizeMethod.BICUBIC)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=new_h_w_shape, method=tf.image.ResizeMethod.BICUBIC)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_bicubic(data, new_h_w_shape, half_pixel_centers=False)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bicubic(data, new_h_w_shape, half_pixel_centers=False)')
  print('output = ', output2)
#  try:
#    np.testing.assert_almost_equal(output1, expected)
#  except AssertionError as err:
#    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_tf_crop_and_resize():  # type: () -> None
  print('##################################')
  print('# test_resize_tf_crop_and_resize #')
  print('##################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='linear',
      coordinate_transformation_mode='tf_crop_and_resize'
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  # Note: for some rois, the result may be different with that of TF for inaccurate floating point
  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 3, 3], dtype=np.int64)
  print('data = ')
  print(data)
  print('roi = ', roi)
  print('sizes = ', sizes)

  # [[[[ 7.6000004  7.9        8.2      ]
  #    [ 8.8        9.1        9.400001 ]
  #    [10.        10.3       10.6      ]]]]
  expected = interpolate_nd(data, linear_coeffs, output_size=sizes, roi=roi,
      coordinate_transformation_mode='tf_crop_and_resize').astype(np.float32)
  print('expected = ') 
  print(expected)

  indices = []
  for i in range(2, (len(data.shape))):
    indices.insert(i-2, i)
    indices.insert(i, i + len(data.shape))
  boxes = tf.expand_dims(tf.gather(roi, indices, axis=0), 0)
  box_indices = tf.range(0, tf.shape(data)[0])
  crop_size = sizes[2:]
  method = tf.image.ResizeMethod.BILINEAR
  print('boxes = ', boxes)
  print('box_indices = ', box_indices)
  print('crop_size = ', crop_size)
  print('method = ', method)
  box_indices = np.array([0], np.int32)
  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output = tf.image.crop_and_resize(data, boxes, box_indices, crop_size, method)   
  output = tf.transpose(output, perm=[0, 3, 1, 2])
  print('tf.image.crop_and_resize(data, boxes, box_indices, crop_size, method)')
  print('output = ', output)
  try:
    np.testing.assert_almost_equal(output, expected)
  except AssertionError as err:
    print(format(err))
    
def export_resize_tf_crop_and_resize_extrapolation_value():  # type: () -> None
  print('##################################')
  print('# test_resize_tf_crop_and_resize #')
  print('##################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='linear',
      coordinate_transformation_mode='tf_crop_and_resize',
      extrapolation_value=10.0
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  # Note: for some rois, the result may be different with that of TF for inaccurate floating point
  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 3, 3], dtype=np.int64)
  print('data = ')
  print(data)
  print('roi = ', roi)
  print('sizes = ', sizes)

  # [[[[ 7.6000004 10.        10.       ]
  #    [12.400001  10.        10.       ]
  #    [10.        10.        10.       ]]]]
  expected = interpolate_nd(data, linear_coeffs, output_size=sizes, roi=roi,
      coordinate_transformation_mode='tf_crop_and_resize', extrapolation_value=10.0).astype(np.float32)
  print('expected = ')
  print(expected)

  indices = []
  for i in range(2, (len(data.shape))):
    indices.insert(i-2, i)
    indices.insert(i, i + len(data.shape))
  boxes = tf.expand_dims(tf.gather(roi, indices, axis=0), 0)
  box_indices = tf.range(0, tf.shape(data)[0])
  crop_size = sizes[2:]
  method = tf.image.ResizeMethod.BILINEAR
  extrapolation_value = 10.0
  print('boxes = ', boxes)
  print('box_indices = ', box_indices)
  print('crop_size = ', crop_size)
  print('method = ', method)
  print('extrapolation_value = ', extrapolation_value)
  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output = tf.image.crop_and_resize(data, boxes, box_indices, crop_size, method, extrapolation_value)
  output = tf.transpose(output, perm=[0, 3, 1, 2])
  print('tf.image.crop_and_resize(data, boxes, box_indices, crop_size, method, extrapolation_value)')
  print('output = ', output)
  np.testing.assert_almost_equal(output, expected) 

def export_resize_downsample_sizes_nearest_tf_half_pixel_for_nn():  # type: () -> None
  print('#############################################################')  
  print('# test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn #')
  print('#############################################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='nearest',
      coordinate_transformation_mode='tf_half_pixel_for_nn'
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 3, 2], dtype=np.int64)
  print('data = ')
  print(data)
  print('sizes = ', sizes)

  # [[[[ 6.  8.]
  #    [10. 12.]
  #    [14. 16.]]]]
  expected = interpolate_nd(
      data, nearest_coeffs, output_size=sizes, coordinate_transformation_mode='tf_half_pixel_for_nn').astype(np.float32)
  print('expected = ')
  print(expected)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output = tf.image.resize(data,
                           size=sizes[2:],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output = tf.transpose(output, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=sizes[2:], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)')
  print('output = ', output)
  output = tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:], half_pixel_centers=True)
  output = tf.transpose(output, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:], half_pixel_centers=True)')
  print("output = ", output)
  try:
    np.testing.assert_almost_equal(output, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_downsample_sizes_linear_pytorch_half_pixel():  # type: () -> None
  print('##########################################################')  
  print('# test_resize_downsample_sizes_linear_pytorch_half_pixel #')
  print('##########################################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='linear',
      coordinate_transformation_mode='pytorch_half_pixel'
  )

  data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 3, 1], dtype=np.int64)
  print('data = ')
  print(data)
  print('sizes = ', sizes)

  # [[[[ 1.6666666]
  #    [ 7.       ]
  #    [12.333333 ]]]]
  expected = interpolate_nd(
      data, linear_coeffs, output_size=sizes, coordinate_transformation_mode='pytorch_half_pixel').astype(np.float32)
  print('expected = ')
  print(expected)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output = tf.image.resize(data,
                           size=sizes[2:],
                           method=tf.image.ResizeMethod.BILINEAR)
  output = tf.transpose(output, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=sizes[2:], method=tf.image.ResizeMethod.BILINEAR)')
  print('output = ', output)
  output = tf.compat.v1.image.resize_bilinear(data, sizes[2:], half_pixel_centers=True)
  output = tf.transpose(output, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_bilinear(data, sizes[2:], half_pixel_centers=True)')
  print("output = ", output)
  try:
    np.testing.assert_almost_equal(output, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_upsample_sizes_nearest_floor_align_corners():  # type: () -> None
  print('##########################################################')
  print('# test_resize_upsample_sizes_nearest_floor_align_corners #')
  print('##########################################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='nearest',
      coordinate_transformation_mode='align_corners',
      nearest_mode='floor'
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 8, 8], dtype=np.int64)
  print('data = ')
  print(data)
  print('sizes = ', sizes)

  # [[[[ 1.  1.  1.  2.  2.  3.  3.  4.]
  #    [ 1.  1.  1.  2.  2.  3.  3.  4.]
  #    [ 1.  1.  1.  2.  2.  3.  3.  4.]
  #    [ 5.  5.  5.  6.  6.  7.  7.  8.]
  #    [ 5.  5.  5.  6.  6.  7.  7.  8.]
  #    [ 9.  9.  9. 10. 10. 11. 11. 12.]
  #    [ 9.  9.  9. 10. 10. 11. 11. 12.]
  #    [13. 13. 13. 14. 14. 15. 15. 16.]]]]
  expected = interpolate_nd(
      data, lambda x: nearest_coeffs(x, mode='floor'), output_size=sizes, coordinate_transformation_mode='align_corners').astype(np.float32)
  print('expected = ')
  print(expected)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=sizes[2:],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=sizes[2:], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:], half_pixel_centers=False, align_corners=True)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:], half_pixel_centers=False, align_corners=True)')
  print('output = ', output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric():  # type: () -> None
  print('###################################################################')  
  print('# test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric #')
  print('###################################################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='nearest',
      coordinate_transformation_mode='asymmetric',
      nearest_mode='round_prefer_ceil'
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 8, 8], dtype=np.int64)
  print('data = ')
  print(data)
  print('sizes = ', sizes)

  # [[[[ 1.  2.  2.  3.  3.  4.  4.  4.]
  #    [ 5.  6.  6.  7.  7.  8.  8.  8.]
  #    [ 5.  6.  6.  7.  7.  8.  8.  8.]
  #    [ 9. 10. 10. 11. 11. 12. 12. 12.]
  #    [ 9. 10. 10. 11. 11. 12. 12. 12.]
  #    [13. 14. 14. 15. 15. 16. 16. 16.]
  #    [13. 14. 14. 15. 15. 16. 16. 16.]
  #    [13. 14. 14. 15. 15. 16. 16. 16.]]]]
  expected = interpolate_nd(
      data, lambda x: nearest_coeffs(x, mode='round_prefer_ceil'),
      output_size=sizes, coordinate_transformation_mode='asymmetric').astype(np.float32)
  print('expected = ')
  print(expected)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=sizes[2:],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=sizes[2:], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:], half_pixel_centers=False, align_corners=False)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:]')
  print('output = ', output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def export_resize_upsample_sizes_nearest_ceil_half_pixel():  # type: () -> None
  print('######################################################')
  print('# test_resize_upsample_sizes_nearest_ceil_half_pixel #')
  print('######################################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='nearest',
      coordinate_transformation_mode='half_pixel',
      nearest_mode='ceil'
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 8, 8], dtype=np.int64)
  print('data = ')
  print(data)
  print('sizes = ', sizes)

  # [[[[ 1.  2.  2.  3.  3.  4.  4.  4.]
  #    [ 5.  6.  6.  7.  7.  8.  8.  8.]
  #    [ 5.  6.  6.  7.  7.  8.  8.  8.]
  #    [ 9. 10. 10. 11. 11. 12. 12. 12.]
  #    [ 9. 10. 10. 11. 11. 12. 12. 12.]
  #    [13. 14. 14. 15. 15. 16. 16. 16.]
  #    [13. 14. 14. 15. 15. 16. 16. 16.]
  #    [13. 14. 14. 15. 15. 16. 16. 16.]]]]
  expected = interpolate_nd(
      data, lambda x: nearest_coeffs(x, mode='ceil'), output_size=sizes).astype(np.float32)
  print('expected = ')
  print(expected)

  data = tf.transpose(data, perm=[0, 2, 3, 1])
  output1 = tf.image.resize(data,
                            size=sizes[2:],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=sizes[2:], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:], half_pixel_centers=True, align_corners=False)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:], half_pixel_centers=True)')
  print('output = ', output2)
  try:
    np.testing.assert_almost_equal(output1, expected)
  except AssertionError as err:
    print(format(err))
  try:
    np.testing.assert_almost_equal(output2, expected)
  except AssertionError as err:
    print(format(err))

def test_export_resize_upsample_sizes_nearest():  # type: () -> None
  print('#######################################')
  print('# test_resize_upsample_sizes_nearest #')
  print('#######################################')
  node = onnx.helper.make_node(
      'Resize',
      inputs=['X', 'roi', 'sizes'],
      outputs=['Y'],
      mode='nearest',
  )

  data = np.array([[[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16]
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 7, 7], dtype=np.int64)
  print('data = ')
  print(data)
  print('sizes = ', sizes)

  # [[[[1. 1. 1. 2. 2. 2.]
  #    [1. 1. 1. 2. 2. 2.]
  #    [3. 3. 3. 4. 4. 4.]
  #    [3. 3. 3. 4. 4. 4.]]]]
  expected = interpolate_nd(
      data, nearest_coeffs, output_size=sizes).astype(np.float32)
  print('expected = ')
  print(expected)

  data = tf.transpose(data, perm=[0, 2, 3, 1])  
  output1 = tf.image.resize(data,
                            size=sizes[2:],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output1 = tf.transpose(output1, perm=[0, 3, 1, 2])
  print('tf.image.resize(data, size=sizes[2:], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)')
  print('output = ', output1)
  output2 = tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:], half_pixel_centers=False, align_corners=False)
  output2 = tf.transpose(output2, perm=[0, 3, 1, 2])
  print('tf.compat.v1.image.resize_nearest_neighbor(data, sizes[2:], half_pixel_centers=False, align_corners=False)')
  print('output = ', output2)
  np.testing.assert_almost_equal(output2, expected)

def main():                                                          #___________________________________________________________________________________________________________
# mode = nearest                                                     # resize_images_v2            | resize_nearest_neighbor       
                                                                     #_____________________________|_____________________________________________________________________________
#  export_resize_upsample_scales_nearest()                            # Pass      |                 |                                     |
#  export_resize_downsample_scales_nearest()                          #           | Fail            | Pass(half_pixel_centers=False)      |
#  export_resize_upsample_sizes_nearest()                             #           | Fail            | Pass(half_pixel_centers=False)      |
#  export_resize_downsample_sizes_nearest()                           #           | Fail            |                                     | Fail(half_pixel_centers=False) (closer)
#  export_resize_downsample_sizes_nearest_tf_half_pixel_for_nn() #tf  #           | Fail            |                                     | Fail(half_pixel_centers=True)
#  export_resize_upsample_sizes_nearest_floor_align_corners()         #           | Fail            |                                     | Fail(align_corners=True)
#  export_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric()#           | Fail            |                                     | Fail
#  export_resize_upsample_sizes_nearest_ceil_half_pixel()             #           | Fail            |                                     | Fail(half_pixel_centers=True)
                                                                     #_____________________________|_____________________________________|________________________________________
# mode = linear                                                      # resize_images_v2            | resize_bilinear
                                                                     #_____________________________|______________________________________________________________________________
#  export_resize_upsample_scales_linear()                             # Pass      |                 |                                     |  
#  export_resize_upsample_scales_linear_align_corners()               #           | Fail            |                                     | Fail(align_corners=True) (closer)
#  export_resize_downsample_scales_linear()                           #           | Fail            |                                     | Fail
#  export_resize_downsample_scales_linear_align_corners()             #           | Fail            |                                     | Fail(align_corners=True) (closer)
#  export_resize_downsample_sizes_linear_pytorch_half_pixel() #pytorch#           | Fail            |                                     | Fail(half_pixel_centers=True)
                                                                     #_____________________________|______________________________________________________________________________
# mode = cubic                                                       # resize_images_v2            | resize_bicubic
                                                                     #_____________________________|______________________________________________________________________________
#  export_resize_upsample_scales_cubic() #pytorch                     #           | Fail            |                                     | Fail (worse than resize_images_v2)
#  export_resize_upsample_scales_cubic_align_corners() #pytorch       #           | Fail            |                                     | Fail(align_corners=True) (closer) 
#  export_resize_downsample_scales_cubic() #pytorch                   #           | Fail            |                                     | Fail (close)
#  export_resize_downsample_scales_cubic_align_corners() #pytorch     #           | Fail            |                                     | Fail
#  export_resize_upsample_sizes_cubic() #pytorch                      #           | Fail            |                                     | Fail (worse than resize_images_v2)
#  export_resize_downsample_sizes_cubic() #pytorch                    #           | Fail            |                                     | Fail (worse than resize_images_v2)
#  export_resize_upsample_scales_cubic_A_n0p5_exclude_outside() #tf   #           | Fail(very close)|                                     | Fail (very close)
#  export_resize_downsample_scales_cubic_A_n0p5_exclude_outside() #tf #           | Fail            |                                     | Fail
#  export_resize_upsample_scales_cubic_asymmetric() #tf               #           | Fail            | Pass(half_pixel_centers=False)      |
                                                                     #_____________________________|_____________________________________|_______________________________________
# mode = linear                                                      #                             |
# coordinate_transformation_mode='tf_crop_and_resize                 # crop_and_resize_v2          | crop_and_resize_v1
                                                                     #_____________________________|_____________________________________________________________________________
  export_resize_tf_crop_and_resize() #tf                             # Pass      |                 |                                     |
  export_resize_tf_crop_and_resize_extrapolation_value() #tf         # Pass      |                 |                                     | 


if __name__ == '__main__':
  main()
