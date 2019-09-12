import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("StringNormalizer")
class StringNormalizer(BackendHandler):

  @classmethod
  def version_10(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_rank = len(x.get_shape())
    case_change_action = node.attrs.get("case_change_action", "NONE")
    is_case_sensitive = node.attrs.get("is_case_sensitive", 0)
    stopwords = node.attrs.get("stopwords", [])

    # since x must be in [C] or [1, C] shape tensor, 
    # squeeze the 0 dimension can simpify the following process 
    if x_rank > 1:
      x = tf.squeeze(x, 0)
      
    # remove stopwords
    for stopword in stopwords:
      pattern = stopword if is_case_sensitive == 1 else '(?i)' + stopword
      for stopword in stopwords:
        x = tf.regex_replace(x, pattern, '')

    # remove '' inserted by tf.regex_replace
    mask = tf.map_fn(
        lambda e: tf.where(tf.equal(e, ""), False, True), x, dtype=tf.bool)
    x = tf.boolean_mask(x, mask)
    # insert '' if the tensor is empty
    if x.get_shape().as_list()[0] == 0:
        x = tf.constant([''], dtype=tf.string)

    if case_change_action == "LOWER":
      x = tf.strings.lower(x)
    elif case_change_action == "UPPER":
      x = tf.strings.upper(x)

    # add the 0 dimension back
    if x_rank > 1:
      x = tf.expand_dim(x, 0)

    # here x is in [b'tuesday', b'wednesday', b'thursday'] format
    # but the expected value should be ['tuesday', 'wednesday', 'thursday']
    # will try to support this after migrate to tensorflow 2.0 using Tensorflow Text
    # https://github.com/tensorflow/text

    return [x]
