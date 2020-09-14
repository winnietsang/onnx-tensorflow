from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import math
import unittest

from onnx import helper
from onnx import TensorProto
from onnx import defs
import numpy as np
import tensorflow as tf

from onnx_tf.backend import onnx_graph_to_tensorflow_rep
from onnx_tf.backend import run_node
from onnx_tf.common import supports_device
from onnx_tf.common.legacy import legacy_onnx_pre_ver, legacy_opset_pre_ver
from onnx_tf.common.pooling_helper import py_pool


class TestNode(unittest.TestCase):
  """ Tests for nodes
  """

  def _get_rnd_float32(self, low=-1.0, high=1.0, shape=None):
    output = np.random.uniform(low, high, shape)
    if shape is None:
      return np.float32(output)
    else:
      return output.astype(np.float32)

  def _get_rnd_int(self, low, high=None, shape=None, dtype=np.int32):
    return np.random.randint(low, high, size=shape, dtype=dtype)

  def test_for_loop(self):
    add_node = helper.make_node('Add', inputs=['x', 'x'], outputs=['sum'])
    sub_node = helper.make_node('Sub',
                                 inputs=['y', 'sum'],
                                 outputs=['diff'])
    mul_node = helper.make_node('Mul',
                                 inputs=['sum', 'diff'],
                                 outputs=['prod'])
    less_node = helper.make_node('Less',
                                 inputs=['sum', 'diff'],
                                 outputs=['new_cond'])
    greater_node = helper.make_node('Greater',
                                    inputs=['sum', 'diff'],
                                    outputs=['new_cond'])

    iter_count_in = helper.make_tensor_value_info('iter_count', TensorProto.INT64, [])
    cond_in = helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
    cond_int_in = helper.make_tensor_value_info('cond', TensorProto.INT32, [])
    x_in = helper.make_tensor_value_info('x', TensorProto.INT32, [None])
    y_in = helper.make_tensor_value_info('y', TensorProto.INT32, [None])

    cond_out = helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
    new_cond_out = helper.make_tensor_value_info('new_cond', TensorProto.BOOL,
                                                 [])
    sum_out = helper.make_tensor_value_info('sum', TensorProto.INT32, [None])
    diff_out = helper.make_tensor_value_info('diff', TensorProto.INT32, [None])
    prod_out = helper.make_tensor_value_info('prod', TensorProto.INT32, [None])

    v1_initial = np.array([1, 1], dtype=np.int32)
    v2_initial = np.array([100, 100], dtype=np.int32)

    # test for loop that doesn't run at all (M = 0)
    M = np.int64(3)
    cond = True  # value will be ignore because optional "cond" input will be skip
    graph = helper.make_graph(nodes=[add_node, sub_node, mul_node],
                              name="for_loop_graph",
                              inputs=[iter_count_in, cond_in, x_in, y_in],
                              outputs=[cond_out, sum_out, diff_out, prod_out])
    node_def = helper.make_node('Loop', ['M', '', 'v1_initial', 'v2_initial'],
                                ['v1_final', 'v2_final', 'scan_output'],
                                body=graph)
    output = run_node(node_def, [M, cond, v1_initial, v2_initial])
    v1_final = np.array([1, 1], dtype=np.int32)
    v2_final = np.array([100, 100], dtype=np.int32)
    scan_output = np.array([], dtype=np.int32).reshape([0, 0])
#    np.testing.assert_almost_equal(output['v1_final'], v1_final)
#    np.testing.assert_almost_equal(output['v2_final'], v2_final)
#    np.testing.assert_almost_equal(output['scan_output'], scan_output)

    print('M = ', M)
    print('v1_initial = ', v1_initial)
    print('v2_initial = ', v2_initial)
    print('output[v1_final] = ', output['v1_final'])
    print('output[v2_final] = ', output['v2_final'])
    print('output[scan_output] = ', output['scan_output'])

  def test_for_loop_2(self):
    v1_initial = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int32)
    v3_initial = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int32)
    add_node = helper.make_node('Add', inputs=['x', 'x'], outputs=['sum'])
    matmul_node = helper.make_node('MatMul',
                                   inputs=['x', 'z'],
                                   outputs=['product'])
    iter_count_in = helper.make_tensor_value_info('iter_count', TensorProto.INT64, [])
    cond_in = helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
    x_in = helper.make_tensor_value_info('x', TensorProto.INT32, [None, None])
    z_in = helper.make_tensor_value_info('z', TensorProto.INT32, [None, None])
    cond_out = helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
    sum_out = helper.make_tensor_value_info('sum', TensorProto.INT32,
                                            [None, None])
    z_out = helper.make_tensor_value_info('z', TensorProto.INT32, [None, None])
    product_out = helper.make_tensor_value_info('product', TensorProto.INT32,
                                                [None, None])

    M = np.array(0, dtype=np.int64)
    cond = np.array(
        True, dtype=np.bool
    )  # value will be ignore because optional "cond" input will be skip
    graph = helper.make_graph(nodes=[add_node, matmul_node],
                              name="for_loop_graph",
                              inputs=[iter_count_in, cond_in, x_in, z_in],
                              outputs=[cond_out, sum_out, z_out, product_out])
    node_def = helper.make_node('Loop', ['M', '', 'v1_initial', 'v3_initial'],
                                ['v1_final', 'v3_final', 'scan_output'],
                                body=graph)
    scan_output = np.array([], dtype=np.int32).reshape([0, 0])
    output = run_node(node_def, [M, cond, v1_initial, v3_initial])
#    np.testing.assert_almost_equal(output['v1_final'], v1_initial)
#    np.testing.assert_almost_equal(output['v3_final'], v3_initial)
#    np.testing.assert_almost_equal(output['scan_output'], scan_output)
    print('test for loop do not run and the shape of scan_output is not the same as the input')
    print('M = ', M)
    print('output[v1_final] = ', output['v1_final'])
    print('output[v3_final] = ', output['v3_final'])
    print('output[scan_output] =', output['scan_output'])

if __name__ == '__main__':
  unittest.main()
