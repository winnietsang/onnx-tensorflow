import onnxruntime
import numpy as np
import onnx_tf
import tensorflow as tf

import onnxruntime.backend
from onnx import helper
from onnx import TensorProto

def test_loop():
    # here is the loop testcase
    # while x < y:
    #   x = x + x (v_1)
    #   y = y - x (v_2)
    #   z = x * y (scan_output)
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

    # test for loop and while loop conbine
    M = np.array(1, dtype=np.int64)
    cond_initial = np.array(np.all(v1_initial < v2_initial), dtype=np.bool)
    body_graph = helper.make_graph(
        nodes=[add_node, sub_node, mul_node, less_node],
        name="for_and_while_loop_graph",
        inputs=[iter_count_in, cond_in, x_in, y_in],
        outputs=[new_cond_out, sum_out, diff_out, prod_out])
    node_def = helper.make_node('Loop',
                                ['M', 'cond_initial', 'v1_initial', 'v2_initial'],
                                ['v1_final', 'v2_final', 'scan_output'],
                                body=body_graph)
    graph_def = helper.make_graph(
        [node_def],
        name='test_loop',
        inputs=[
            helper.make_tensor_value_info('M', TensorProto.INT64, []),
            helper.make_tensor_value_info('cond_initial', TensorProto.BOOL, []),
            helper.make_tensor_value_info('v1_initial', TensorProto.INT32,
                [None]),
            helper.make_tensor_value_info('v2_initial', TensorProto.INT32,
                [None])
        ],
        outputs=[
            helper.make_tensor_value_info('v1_final', TensorProto.INT32,
                [None]),
            helper.make_tensor_value_info('v2_final', TensorProto.INT32,
                [None]),
            helper.make_tensor_value_info('scan_output', TensorProto.INT32,
                [None])
        ]
    )
    model_def = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 11)])
  
    rt_rep = onnxruntime.backend.prepare(model_def)
    rt_output = rt_rep.run([M, cond_initial, v1_initial, v2_initial])
    print('rt_output = ', rt_output)
    print('rt_output[v1_final] = ', rt_output[0])
    print('rt_output[v2_final] = ', rt_output[1])
    print('rt_output[scan_output] = ', rt_output[2])

    tf_rep = onnx_tf.backend.prepare(model_def)
    tf_output = tf_rep.run({'M': M, 'cond_initial': cond_initial, 'v1_initial': v1_initial, 'v2_initial': v2_initial })
    print('tf_output = ', tf_output)
    print('tf_output[v1_final] = ', tf_output['v1_final'])
    print('tf_output[v2_final] = ', tf_output['v2_final'])
    print('tf_output[scan_output] = ', tf_output['scan_output'])

    # test for loop that doesn't run at all (M = 0)
    # and the scan_outputs shape is not the same as the inputs
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
    cond_initial = np.array(True, dtype=np.bool)  # value will be ignore because optional "cond" input will be skip
    body_graph = helper.make_graph(nodes=[add_node, matmul_node],
                              name="for_loop_graph",
                              inputs=[iter_count_in, cond_in, x_in, z_in],
                              outputs=[cond_out, sum_out, z_out, product_out])
    node_def = helper.make_node('Loop', ['M', '', 'v1_initial', 'v3_initial'],
                                ['v1_final', 'v3_final', 'scan_output'],
                                body=body_graph)
    graph_def = helper.make_graph(
        [node_def],
        name='test_loop',
        inputs=[
            helper.make_tensor_value_info('M', TensorProto.INT64, []),
            helper.make_tensor_value_info('cond_initial', TensorProto.BOOL, []),
            helper.make_tensor_value_info('v1_initial', TensorProto.INT32,
                [None, None]),
            helper.make_tensor_value_info('v3_initial', TensorProto.INT32,
                [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info('v1_final', TensorProto.INT32,
                [None, None]),
            helper.make_tensor_value_info('v3_final', TensorProto.INT32,
                [None, None]),
            helper.make_tensor_value_info('scan_output', TensorProto.INT32,
                [None, None])
        ]
    )
    model_def = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 11)])

    # in loop node_def set cond input name as '' will casue ort to seg fault
#    rt_rep = onnxruntime.backend.prepare(model_def)
#    rt_output = rt_rep.run([M, cond_initial, v1_initial, v3_initial])
#    print('rt_output = ', rt_output)
#    print('rt_output[v1_final] = ', rt_output[0])
#    print('rt_output[v3_final] = ', rt_output[1])
#    print('rt_output[scan_output] = ', rt_output[2])

    tf_rep = onnx_tf.backend.prepare(model_def)
    tf_output = tf_rep.run({'M': M, 'cond_initial': cond_initial, 'v1_initial': v1_initial, 'v3_initial': v3_initial})
    print('tf_output = ', tf_output)
    print('tf_output[v1_final] = ', tf_output['v1_final'])
    print('tf_output[v3_final] = ', tf_output['v3_final'])
    print('tf_output[scan_output] = ', tf_output['scan_output'])

def main():
    test_loop()

if __name__ == '__main__':
  main()
