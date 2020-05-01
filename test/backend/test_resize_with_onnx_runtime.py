import onnxruntime
import numpy as np
import onnx_tf
import tensorflow as tf

import onnxruntime.backend
from onnx import helper
from onnx import TensorProto

#tf.enable_eager_execution()

#data = np.array([[[
#    [1, 2, 3, 4],
#    [5, 6, 7, 8],
#    [9, 10, 11, 12],
#    [13, 14, 15, 16],
#]]], dtype=np.float32)

data = np.array([[[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
]]], dtype=np.float32)

data = np.array([[[
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
    [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
    [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
    [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
]]], dtype=np.float32)

def nearest_tf_half_pixel_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      coordinate_transformation_mode='tf_half_pixel_for_nn',
      mode='nearest',
      nearest_mode='floor'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = nearest')
  print('coordinate_transformation_mode = tf_half_pixel_for_nn')
  print('nearest_mode = floor')
  print('data =')
  print(data)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])

#  x = tf.constant(data, tf.bfloat16)
#  x_t = tf.transpose(x, perm=[0, 2, 3, 1])
#  new_size = tf.cast(scales[2:] * tf.cast(x.get_shape()[2:], scales.dtype), tf.int32)
#  y = tf.image.resize(x_t, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#  y = tf.transpose(y, perm=[0, 3, 1, 2])
#  print("x = ", x)
#  print("tf.image.resize(x_t, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)")
#  print(y)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output=')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print(format(err)) 

def nearest_tf_half_pixel_sizes():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      coordinate_transformation_mode='tf_half_pixel_for_nn',
      mode='nearest',
      nearest_mode='floor'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 7, 7], dtype=np.int64)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = nearest')
  print('coordinate_transformation_mode = tf_half_pixel_for_nn')
  print('nearest_mode = floor')
  print('data = ')
  print(data)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])

  np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])

def nearest_half_pixel_sizes():
  node_def = helper.make_node(
      "Resize",
  #    inputs=['X', 'roi', 'scales', 'sizes'],
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      coordinate_transformation_mode='half_pixel',
      mode='nearest',
      nearest_mode='floor'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
 #         helper.make_tensor_value_info('sizes', TensorProto.INT64,
 #             [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
#  scales = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)
#  sizes = np.array([1, 1, 3, 3], dtype=np.int64)

#  new_size = np.floor(sizes[2:] - 0.5)
  data_shape = data.shape
#  new_size = (scales[2:]) * data_shape[2:] 
#  print('new_size = ', new_size)
#  data_t = tf.transpose(data, perm=[0, 2, 3, 1])
#  y = tf.image.resize(data_t, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#  tf_output = tf.transpose(y, perm=[0, 3, 1, 2])
#  print(tf_output)
#  tf_output_t = tf.transpose(tf_output, perm=[0, 2, 3, 1])
#  new_size = scales[2:] * data_shape[2:]
#  print('new_size = ', new_size)
#  y = tf.image.resize(data_t, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#  tf_output = tf.transpose(y, perm=[0, 3, 1, 2])
#  print(tf_output)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = nearest')
  print('coordinate_transformation_mode = half_pixel')
  print('nearest_mode = floor')
  print('data = ')
  print(data)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])
#  print(tf_output)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])

  np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])

def nearest_round_prefer_ceil_align_corners_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      coordinate_transformation_mode='align_corners',
      mode='nearest',
      nearest_mode='round_prefer_ceil'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.9, 0.9], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})

  print('-----------------------------------------------------------')
  print('mode = nearest')
  print('coordinate_transformation_mode = align_corners')
  print('nearest_mode = round_prefer_ceil')
  print('data = ')
  print(data)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])

  np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])

def nearest_round_prefer_ceil_align_corners_sizes():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      coordinate_transformation_mode='align_corners',
      mode='nearest',
      nearest_mode='round_prefer_ceil'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 7, 7], dtype=np.int64)

  data = np.array([[[
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
      [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
      [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
      [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
      [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
      [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
      [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
      [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
  ]]], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})

  x = tf.constant(data, tf.bfloat16)
  x_t = tf.transpose(x, perm=[0, 2, 3, 1])
#  y_v2 = tf.image.resize(x_t, size=[3,3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#  y_v2_t = tf.transpose(y_v2, perm=[0, 3, 1, 2])
  y_v1 = tf.compat.v1.image.resize_nearest_neighbor(x_t, size=[7,7], align_corners=True, half_pixel_centers=False)
  y_v1_t = tf.transpose(y_v1, perm=[0, 3, 1, 2])
  print("x = ", x)
  print("tf.compat.v1.image.resize_nearest_neighbor(x_t, size=[7,7], align_corners=True, half_pixel_centers=False)")
  print(y_v1_t)

  print('-----------------------------------------------------------')
  print('mode = nearest')
  print('coordinate_transformation_mode = align_corners')
  print('nearest_mode = round_prefer_ceil')
  print('data = ')
  print(data)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])
#  print("tf.compat.v1.image.resize_nearest_neighbor = ")
#  print(y_v1_t)
#  print('tf.image.resize(data_t, size=[3,3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) =')
#  print(y_v2_t)
#  print('tf.compat.v1.image.resize_nearest_neighbor(data_t, size=[3,3], align_corners=False, half_pixel_centers=False) =')
#  print(y_v1_t)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])    

  np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])

def nearest_floor_asymmetric_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      coordinate_transformation_mode='asymmetric',
      mode='nearest',
      nearest_mode='floor'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

#  data = np.array([[[
#      [1, 2, 3, 4, 5],
#      [6, 7, 8, 9, 10],
#      [11, 12, 13, 14, 15],
#      [16, 17, 18, 19, 20],
#      [21, 22, 23, 24, 25]
#  ]]], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
#  data_t = tf.transpose(data, perm=[0, 2, 3, 1])
#  y_v2 = tf.image.resize(data_t, size=[4,4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#  y_v2_t = tf.transpose(y_v2, perm=[0, 3, 1, 2])
#  y_v1 = tf.compat.v1.image.resize(data_t, size=[4,4], method=tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR)
#  y_v1 = tf.compat.v1.image.resize_nearest_neighbor(data_t, size=[4,4], align_corners=False, half_pixel_centers=False)
#  y_v1_t = tf.transpose(y_v1, perm=[0, 3, 1, 2])

  print('-----------------------------------------------------------')
  print('mode = nearest')
  print('coordinate_transformation_mode = asymmetric')
  print('nearest_mode = floor')
  print('data = ')
  print(data)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])
#  print('tf.image.resize(data_t, size=[4,4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) =')
#  print(y_v2_t)
#  print('tf.compat.v1.image.resize_nearest_neighbor(data_t, size=[4,4], align_corners=False, half_pixel_centers=False) =')
#  print(y_v1_t)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])

  np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])

def nearest_floor_asymmetric_size():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      coordinate_transformation_mode='asymmetric',
      mode='nearest',
      nearest_mode='floor'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  data = np.array([[[
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
      [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
      [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
      [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
      [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
      [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
      [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
      [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 7, 7], dtype=np.int64)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
#  data_t = tf.transpose(data, perm=[0, 2, 3, 1])
#  y_v2 = tf.image.resize(data_t, size=[3,3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#  y_v2_t = tf.transpose(y_v2, perm=[0, 3, 1, 2])
#  y_v1 = tf.compat.v1.image.resize_nearest_neighbor(data_t, size=[7,7], align_corners=False, half_pixel_centers=False)
#  y_v1_t = tf.transpose(y_v1, perm=[0, 3, 1, 2])

  print('-----------------------------------------------------------')
  print('mode = nearest')
  print('coordinate_transformation_mode = asymmetric')
  print('nearest_mode = floor')
  print('data = ')
  print(data)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])
#  print('tf.image.resize(data_t, size=[3,3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) =') 
#  print(y_v2_t)
#  print('tf.compat.v1.image.resize_nearest_neighbor(data_t, size=[7,7], align_corners=False, half_pixel_centers=False) =')
#  print(y_v1_t)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])  
  
  np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])

def nearest_crop_and_resize_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      coordinate_transformation_mode='tf_crop_and_resize',
      mode='nearest',
      nearest_mode='round_prefer_ceil',
      extrapolation_value=-20.0
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

#  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = nearest')
  print('nearest_mode = round_prefer_ceil')
  print('coordinate_transformation_mode = tf_crop_and_resize')
  print('exclude_outside = 0')
  print('extrapolation_value = -20.0') 
  print('data =')
  print(data)
  print('roi = ', roi)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])
  print('shape = ', tf_output['Y'].shape)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])
  print('shape = ', rt_output[0].shape)

  np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])

def nearest_crop_and_resize_sizes():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      coordinate_transformation_mode='tf_crop_and_resize',
      mode='nearest',
      nearest_mode='round_prefer_ceil',
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

#  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 7, 7], dtype=np.int64)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = nearest')
  print('nearest_mode = round_prefer_ceil')
  print('coordinate_transformation_mode = tf_crop_and_resize')
  print('exclude_outside = 0')
  print('extrapolation_value = 00.0')
  print('data =')
  print(data)
  print('roi = ', roi)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])
  print('shape = ', tf_output['Y'].shape)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])
  print('shape = ', rt_output[0].shape)

  np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])

def linear_align_corner_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      coordinate_transformation_mode='align_corners',
      mode='linear'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = linear')
  print('coordinate_transformation_mode = align_corners')
  print('exclude_outside = 0')
  print('data =')
  print(data)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in linear_allign_corners_scales')
    print(format(err))
  np.testing.assert_allclose(tf_output['Y'], rt_output[0], rtol=1e-6, atol=1e-6)

def linear_align_corner_sizes():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      coordinate_transformation_mode='align_corners',
      mode='linear'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  data = np.array([[[
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
      [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
      [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
      [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
      [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
      [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
      [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
      [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 7, 7], dtype=np.int64)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = linear')
  print('coordinate_transformation_mode = align_corners')
  print('exclude_outside = 0')
  print('data =')
  print(data)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])

#  x = tf.constant(data, tf.bfloat16)
#  x_t = tf.transpose(x, perm=[0, 2, 3, 1])
#  new_size = tf.cast(sizes[2:], tf.int32)
#  y = tf.compat.v1.image.resize_bilinear(x, size=new_size, align_corners=True, half_pixel_centers=False)
#  y = tf.transpose(y, perm=[0, 3, 1, 2])
#  print("x = ", x)
#  print("tf.compat.v1.image.resize_bilinear(x, size=new_size, align_corners=True, half_pixel_centers=False)")
#  print("y = ")
#  print(y)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])

  np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])

def linear_asymmetric_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      coordinate_transformation_mode='asymmetric',
      mode='linear'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1, 1, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = linear')
  print('coordinate_transformation_mode = asymmetric')
  print('exclude_outside = 0')
  print('data =')
  print(data)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])

  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in linear_asymmetric_size')
    print(format(err))
  np.testing.assert_allclose(tf_output['Y'], rt_output[0], rtol=1e-6, atol=1e-6)

def linear_asymmetric_sizes():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      coordinate_transformation_mode='asymmetric',
      mode='linear'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  data = np.array([[[
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
      [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
      [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
      [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
      [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
      [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
      [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
      [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 7, 7], dtype=np.int64)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = linear')
  print('coordinate_transformation_mode = asymmetric')
  print('exclude_outside = 0')
  print('data =')
  print(data)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])

  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in linear_asymmetric_size')
    print(format(err))
  np.testing.assert_allclose(tf_output['Y'], rt_output[0], rtol=1e-6, atol=1e-6) 
  
def linear_half_pixel_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='linear'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = linear')
  print('coordinate_transformation_mode = half_pixel')
  print('exclude_outside = 0')
  print('data =')
  print(data)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print(format(err))

def linear_half_pixel_sizes():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='linear'
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 7, 7], dtype=np.int64)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = linear')
  print('coordinate_transformation_mode = half_pixel')
  print('exclude_outside = 0')
  print('data =')
  print(data)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])

#  x = tf.constant(data, tf.bfloat16)
#  x_t = tf.transpose(x, perm=[0, 2, 3, 1])
#  new_size = tf.cast(sizes[2:], tf.int32)
#  y = tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.BILINEAR)
#  y = tf.transpose(y, perm=[0, 3, 1, 2])
#  print("x = ", x)
#  print("tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.BILINEAR)")
#  print("y = ")
#  print(y)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in linear_half_pixel_sizes')
    print(format(err))
  np.testing.assert_allclose(tf_output['Y'], rt_output[0], rtol=1e-6, atol=1e-6)

def linear_crop_and_resize_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='linear',
      coordinate_transformation_mode='tf_crop_and_resize',
      extrapolation_value=20.0
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

#  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = linear')
  print('coordinate_transformation_mode = tf_crop_and_resize')
  print('exclude_outside = 0')
  print('data =')
  print(data)
  print('roi = ', roi)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])
  print('shape = ', tf_output['Y'].shape)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])
  print('shape = ', rt_output[0].shape)

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in linear_crop_and_resize_half_pixel_scales')
    print(format(err))

def linear_crop_and_resize_sizes():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      mode='linear',
      coordinate_transformation_mode='tf_crop_and_resize',
      extrapolation_value=50.0
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

#  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
  roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 7, 7], dtype=np.int64)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = linear')
  print('coordinate_transformation_mode = tf_crop_and_resize')
  print('exclude_outside = 0')
  print('data =')
  print(data)
  print('roi = ', roi)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])
  print('shape = ', tf_output['Y'].shape)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])
  print('shape = ', rt_output[0].shape)

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in linear_crop_and_resize_half_pixel_sizes')
    print(format(err))

  np.testing.assert_allclose(tf_output['Y'], rt_output[0], rtol=1e-6, atol=1e-6)  

def cubic_align_corners_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      coordinate_transformation_mode='align_corners',
      mode='cubic',
      cubic_coeff_a=-0.5,
      exclude_outside=True
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 11)])

  roi = np.array([], dtype=np.float32)
  scales = np.array([1, 1, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = cubic')
  print('coordinate_transformation_mode = align_corners')
  print('cubic_coeff_a = -0.5')
  print('exclude_outside = 1')
  print('data =')
  print(data)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in cubic_align_corners_size')
    print(format(err))
  np.testing.assert_allclose(tf_output['Y'], rt_output[0], rtol=1e-1, atol=1e-6)

def cubic_align_corners_sizes():
  cast_node = helper.make_node(
      "Cast",
      inputs=['data'],
      outputs=['X'],
      to=TensorProto.BFLOAT16
  )    
  resize_node = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      coordinate_transformation_mode='align_corners',
      mode='cubic',
      cubic_coeff_a=-0.5,
      exclude_outside=True
  )
  graph_def = helper.make_graph(
      [cast_node, resize_node],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('data', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)
#  model_def = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 11)])

  data = np.array([[[
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
      [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
      [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
      [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
      [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
      [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
      [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
      [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 7, 7], dtype=np.int64)

  tf_rep = onnx_tf.backend.prepare(model_def)
#  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  tf_output = tf_rep.run({'data': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = cubic')
  print('coordinate_transformation_mode = align_corners')
  print('cubic_coeff_a = -0.5')
  print('exclude_outside = 1')
  print('data =')
  print(data)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])

#  x = tf.constant(data, tf.bfloat16)
#  tf_output = tf_rep.run({'X': x, 'roi': roi, 'scales': scales, 'sizes': sizes})
#  print('x = ')
#  print(x)
#  print('onnx_tf output =')
#  print(tf_output['Y'])
##  x_t = tf.transpose(x, perm=[0, 2, 3, 1])
##  new_size = tf.cast(sizes[2:], tf.int32)
##  y = tf.compat.v1.image.resize_bicubic(x, size=new_size, align_corners=True, half_pixel_centers=False)
##  y = tf.transpose(y, perm=[0, 3, 1, 2])
#  print("x = ", x)
#  print("tf.compat.v1.image.resize_bicubic(x, size=new_size, align_corners=True, half_pixel_centers=False)")
#  print("y = ")
#  print(y)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in cubic_align_corners_size')
    print(format(err))

def cubic_asymmetric_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      coordinate_transformation_mode='asymmetric',
      mode='cubic',
      cubic_coeff_a=-0.5,
      exclude_outside=True
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )

  model_def = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 11)])

  roi = np.array([], dtype=np.float32)
  scales = np.array([1, 1, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = cubic')
  print('coordinate_transformation_mode = asymmetric')
  print('cubic_coeff_a = -0.5')
  print('exclude_outside = 1')
  print('data =')
  print(data)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in cubic_asymmetric_scales')
    print(format(err))
  np.testing.assert_allclose(tf_output['Y'], rt_output[0], rtol=1e-1, atol=1e-6)

def cubic_asymmetric_sizes():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'],
      coordinate_transformation_mode='asymmetric',
      mode='cubic',
      cubic_coeff_a=-0.5,
      exclude_outside=True
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )

  model_def = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 11)])

  data = np.array([[[
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
      [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
      [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
      [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
      [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
      [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
      [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
      [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
  ]]], dtype=np.float32)

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 7, 7], dtype=np.int64)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = cubic')
  print('coordinate_transformation_mode = asymmetric')
  print('cubic_coeff_a = -0.5')
  print('exclude_outside = 1')
  print('data =')
  print(data)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in cubic_asymmetric_sizes')
    print(format(err))
  np.testing.assert_allclose(tf_output['Y'], rt_output[0], rtol=1e-1, atol=1e-6)

def cubic_half_pixel_scales():
  node_def = helper.make_node(
      "Resize",
      inputs=['X', 'roi', 'scales'],
      outputs=['Y'],
      mode='cubic',
      cubic_coeff_a=-0.5,
      exclude_outside=True
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def)

  roi = np.array([], dtype=np.float32)
  scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales})
  print('-----------------------------------------------------------')
  print('mode = cubic')
  print('coordinate_transformation_mode = half_pixel')
  print('cubic_coeff_a = -0.5')
  print('exclude_outside = 1')
  print('data =')
  print(data)
  print('scales = ', scales)
  print('onnx_tf output =')
  print(tf_output['Y'])

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales])
  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print(format(err))

def cubic_half_pixel_sizes():
  node_def = helper.make_node(
      "Resize", 
      inputs=['X', 'roi', 'scales', 'sizes'],
      outputs=['Y'], 
      mode='cubic',
      cubic_coeff_a=-0.5,
      exclude_outside=True
  )
  graph_def = helper.make_graph(
      [node_def],
      name='test_resize',
      inputs=[
          helper.make_tensor_value_info('X', TensorProto.FLOAT,
              [None, None, None, None]),
          helper.make_tensor_value_info('roi', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('scales', TensorProto.FLOAT,
              [None]),
          helper.make_tensor_value_info('sizes', TensorProto.INT64,
              [None])
      ],
      outputs=[
          helper.make_tensor_value_info('Y', TensorProto.FLOAT,
              [None, None, None, None])
      ]
  )
  model_def = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 11)])

  roi = np.array([], dtype=np.float32)
  scales = np.array([], dtype=np.float32)
  sizes = np.array([1, 1, 7, 7], dtype=np.int64)
  
  tf_rep = onnx_tf.backend.prepare(model_def)
  tf_output = tf_rep.run({'X': data, 'roi': roi, 'scales': scales, 'sizes': sizes})
  print('-----------------------------------------------------------')
  print('mode = cubic')
  print('coordinate_transformation_mode = half_pixel')
  print('cubic_coeff_a = -0.5')
  print('exclude_outside = 1')
  print('data =')
  print(data)
  print('sizes = ', sizes)
  print('onnx_tf output =')
  print(tf_output['Y'])

  x = tf.constant(data, tf.bfloat16)
  x_t = tf.transpose(x, perm=[0, 2, 3, 1])
  new_size = tf.cast(sizes[2:], tf.int32)
  y = tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.BICUBIC)
  y = tf.transpose(y, perm=[0, 3, 1, 2])
  print("x = ", x)
  print("tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.BICUBIC)")
  print("y = ")
  print(y)

  rt_rep = onnxruntime.backend.prepare(model_def)
  rt_output = rt_rep.run([data, roi, scales, sizes])
  print('onnxruntime output =')
  print(rt_output[0])

  try:
    np.testing.assert_almost_equal(tf_output['Y'], rt_output[0])
  except AssertionError as err:
    print('mismatch in cubic_half_pixel_size')
    print(format(err))

  np.testing.assert_allclose(tf_output['Y'], rt_output[0], rtol=1e-2, atol=1e-6)
  
  

def main():

#  nearest_round_prefer_ceil_align_corners_scales()
#  nearest_round_prefer_ceil_align_corners_sizes()
#  nearest_floor_asymmetric_scales()
#  nearest_floor_asymmetric_size()
#  nearest_tf_half_pixel_scales()
#  nearest_tf_half_pixel_sizes()
##  nearest_half_pixel_sizes()

#  linear_align_corner_scales()
#  linear_align_corner_sizes()
#  linear_asymmetric_scales()
#  linear_asymmetric_sizes()
#  linear_half_pixel_scales()
#  linear_half_pixel_sizes()

#  cubic_align_corners_scales()
  cubic_align_corners_sizes()
#  cubic_asymmetric_scales()
#  cubic_asymmetric_sizes()
#  cubic_half_pixel_scales()
#  cubic_half_pixel_sizes()

#  nearest_crop_and_resize_scales()
#  nearest_crop_and_resize_sizes()
#  linear_crop_and_resize_scales()
#  linear_crop_and_resize_sizes()

if __name__ == '__main__':
  main()
