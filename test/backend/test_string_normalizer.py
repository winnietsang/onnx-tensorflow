from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import unittest
import numpy as np
import tensorflow as tf
from onnx_tf.backend import run_node
from onnx_tf.common import supports_device
from onnx_tf.common.legacy import legacy_onnx_pre_ver, legacy_opset_pre_ver
from onnx import helper
from onnx import TensorProto
from onnx import defs


class TestNode(unittest.TestCase):
  """ Tests for nodes
  """

  def test_string_normalizer(self):
    stopwords = [u'monday']
    node_def = helper.make_node(
        "StringNormalizer", 
        ["X"], 
        ["Y"],
        case_change_action='LOWER',
        is_case_sensitive=1,
        stopwords=stopwords)
    x = np.array([u'monday', u'tuesday', u'wednesday', u'thursday']).astype(np.object)
    y = np.array([u'tuesday', u'wednesday', u'thursday']).astype(np.object)
    output = run_node(node_def, [x])
    np.testing.assert_equal(output["Y"], y)


if __name__ == '__main__':
  unittest.main()
