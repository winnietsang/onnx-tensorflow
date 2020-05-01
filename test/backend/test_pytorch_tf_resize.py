import numpy as np  # type: ignore

from typing import List, Callable, Union, Optional, Text


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


def interpolate_1d_with_x(data,                      # type: np.ndarray
                          scale_factor,              # type: float
                          x,                         # type: float
                          get_coeffs,                # type: Callable[[float], np.ndarray]
                          roi=None,                  # type: np.ndarray
                          extrapolation_value=0.0,   # type: float
                          scaler='half_pixel',       # type: Text
                          exclude_outside=False,     # type: bool
                          ):                         # type: (...) -> np.ndarray
    def get_neighbor_idxes(x, n, limit):  # type: (float, int, int) -> np.ndarray
        """
        Return the n nearest indexes, prefer the indexes smaller than x
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
        :param x:
        :param n:  the number of the wanted elements
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
    if scaler == 'align_corners':
        if output_width == 1:
            x_ori = 0.
        else:
            x_ori = x * (input_width - 1) / (output_width - 1)
    elif scaler == 'asymmetric':
        x_ori = x / scale_factor
    elif scaler == 'tf_crop_and_resize':
        if output_width == 1:
            x_ori = (roi[1] - roi[0]) * (input_width - 1) / 2
        else:
            x_ori = x * (roi[1] - roi[0]) * \
                (input_width - 1) / (output_width - 1)
        x_ori += (roi[0] * (input_width - 1))
        # Return extrapolation_value directly as what TF CropAndResize does
        if x_ori < 0 or x_ori > input_width - 1:
            return extrapolation_value
    elif scaler == 'tf_half_pixel_for_nn':
        x_ori = (x + 0.5) / scale_factor
    elif scaler == 'pytorch_half_pixel':
        if output_width == 1:
            x_ori = -0.5
        else:
            x_ori = (x + 0.5) / scale_factor - 0.5
    else:  # scaler == 'half_pixel'
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
                          **kwargs
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
                   **kwargs
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


data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]], dtype=np.float32)


#def test_pytorch():
#    import torch
#    import torch.nn.functional as F
#    for h in range(1, 20):
#        for w in range(1, 20):
#            for resize in ['nearest', 'linear', 'cubic']:
#                scalers = ['asymmetric'] if resize == 'nearest' else ['align_corners', 'pytorch_half_pixel']
#                for scaler in scalers:
#                    sizes = np.array([1, 1, h, w], dtype=np.int64)
#
#                    coeffs_dict = {'nearest': lambda x: nearest_coeffs(x, mode='floor'),
#                                   'linear': linear_coeffs,
#                                   'cubic': cubic_coeffs}
#                    coeffs = coeffs_dict[resize]
#
#                    onnx_output = interpolate_nd(data, coeffs, output_size=sizes, scaler=scaler).astype(np.float32)
#
#                    pytorch_resize_mode_dict = {'nearest': 'nearest',
#                                                'linear': 'bilinear',
#                                                'cubic': 'bicubic'
#                                                }
#                    pytorch_resize_mode = pytorch_resize_mode_dict[resize]
#                    tensor = torch.tensor(data)
#                    if resize != 'nearest':
#                        pytorch_output = F.interpolate(tensor, size=(h, w), mode=pytorch_resize_mode,
#                                                       align_corners=(scaler == 'align_corners'))
#                    else:
#                        pytorch_output = F.interpolate(tensor, size=(h, w), mode=pytorch_resize_mode)
#                    pytorch_output = pytorch_output.numpy()
#
#                    if not np.allclose(onnx_output, pytorch_output):
#                        print("Different results when resize={}, h={}, w={}, scaler={}".format(resize, h, w, scaler))
#                        print("onnx: ")
#                        print(onnx_output)
#                        print("pytorch: ")
#                        print(pytorch_output)
#                        print("max difference: ")
#                        print(np.max(np.abs(onnx_output - pytorch_output)))


def test_tf():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    for h in range(1, 20):
        for w in range(1, 20):
            for resize in ['nearest', 'linear', 'cubic']:
                if resize == 'nearest':
                    scalers = ['asymmetric', 'align_corners', 'tf_half_pixel_for_nn']
                else:
                    scalers = ['asymmetric', 'align_corners', 'half_pixel']
                for scaler in scalers:
                    sizes = np.array([1, 1, h, w], dtype=np.int64)

                    if resize == 'cubic' and scaler == 'half_pixel':
                        coeffs = lambda x: cubic_coeffs(x, A=-0.5)
                    elif resize == 'nearest' and scaler != 'align_corners':
                        coeffs = lambda x: nearest_coeffs(x, mode='floor')
                    else:
                        coeffs_dict = {'nearest': lambda x: nearest_coeffs(x, mode='round_prefer_ceil'),
                                       'linear': linear_coeffs,
                                       'cubic': cubic_coeffs}
                        coeffs = coeffs_dict[resize]

                    exclude_outside = resize == 'cubic' and scaler == 'half_pixel'
                    onnx_output = interpolate_nd(data, coeffs, output_size=sizes, exclude_outside=exclude_outside,
                                                 scaler=scaler).astype(np.float32)

                    tf_resize_dict = {'nearest': tf.compat.v1.image.resize_nearest_neighbor,
                                      'linear': tf.compat.v1.image.resize_bilinear,
                                      'cubic': tf.compat.v1.image.resize_bicubic}
                    tf_resize = tf_resize_dict[resize]
                    input = tf.constant(np.moveaxis(data, 1, -1))
                    if scaler == 'asymmetric':
                        res = tf_resize(input, size=(h, w))
                    elif scaler == 'align_corners':
                        res = tf_resize(input, size=(h, w), align_corners=True)
                    elif scaler == 'half_pixel' or scaler == 'tf_half_pixel_for_nn':
                        res = tf_resize(input, size=(h, w), half_pixel_centers=True)
                    sess = tf.compat.v1.Session()
                    tf_output = sess.run(res)

                    tf_output = np.moveaxis(tf_output, -1, 1)
                    print("data = ", data)
                    print("sizes = ", sizes)
                    print("resize = ", resize)
                    print("scaler = ", scaler)
                    print("exclude_outside = ", exclude_outside)

                    print("onnx_output = ", onnx_output)
                    print("tf_output = ", tf_output)

                    atol = 1e-3 if resize == 'cubic' else 1e-8
                    rtol = 5e-4 if resize == 'cubic' else 1e-5
                    if not np.allclose(onnx_output, tf_output, atol=atol, rtol=rtol):
                        print("Different results when resize={}, h={}, w={}, scaler={}".format(resize, h, w, scaler))
                        print("onnx: ")
                        print(onnx_output)
                        print("tf: ")
                        print(tf_output)
                        print("max difference: ")
                        print(np.max(np.abs(onnx_output - tf_output)))


#print("PyTorch")
#test_pytorch()
print("TF")
test_tf()
