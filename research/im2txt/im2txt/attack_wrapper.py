## Copyright (C) 2017, Hongge Chen <chenhg@mit.edu>.
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.

"""Model wrapper class for performing attack with a ShowAndTellModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import math

FLAGS = tf.flags.FLAGS

from im2txt import configuration
from im2txt import show_and_tell_model
from im2txt.inference_utils import inference_wrapper_base

class AttackWrapper(inference_wrapper_base.InferenceWrapperBase):
  """Model wrapper class for performing attack with a ShowAndTellModel."""

  def __init__(self):
    super(AttackWrapper, self).__init__()
    # TODO: change this based on configuration
    self.image_size = 299
    self.num_channels = 3

  def build_model(self, model_config, image_raw_feed = None, input_feed = None, mask_feed = None):
    model = show_and_tell_model.ShowAndTellModel(model_config, mode="attack")
    model.build(image_raw_feed, input_feed, mask_feed)
    self.model = model
    return model

  def feed_image(self, sess, encoded_image):
    initial_state = sess.run(fetches="lstm/initial_state:0",
                             feed_dict={"image_feed:0": encoded_image})
                             #feed_dict={"image_raw_feed:0": encoded_image})
    return initial_state

  # input feed, mask_feed and image_feed are numpy arrays
  # returns a number
  def attack_step(self, sess, input_feed, mask_feed, image_raw_feed):
    grad_op = tf.gradients(self.model.target_cross_entropy_losses, self.model.images)
    grads, target_cross_entropy_losses = sess.run(
        fetches=[grad_op, self.model.target_cross_entropy_losses],
        feed_dict={
            "input_feed:0": input_feed,
            "input_mask:0": mask_feed,
            "image_raw_feed:0": image_raw_feed
        })
    print(grads)
    return math.exp(-np.sum(target_cross_entropy_losses))

  # input feed, mask_feed and image_feed are tensors
  # returns a tensor
  def predict(self, sess, image_raw_feed, input_feed, mask_feed):
    tf.logging.info("Building model.")
    start_vars = set(x.name for x in tf.global_variables())
    self.build_model(configuration.ModelConfig(), image_raw_feed, input_feed, mask_feed)
    end_vars = tf.global_variables()
    restore_vars = [x for x in end_vars if x.name not in start_vars]
    saver = tf.train.Saver(var_list = restore_vars)
    restore_fn = self._create_restore_fn(FLAGS.checkpoint_path, saver)
    restore_fn(sess)
    sum_log_probs = sess.graph.get_tensor_by_name("batch_loss:0")
    logits = self.model.logits
    softmax = sess.graph.get_tensor_by_name("softmax:0")
    # return sum_log_probs, logits, softmax
    return sum_log_probs, softmax, logits

