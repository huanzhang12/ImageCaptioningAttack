# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from l2_attack import CarliniL2
import math
import os
import re
import numpy as np

import tensorflow as tf

from im2txt import configuration
from im2txt import attack_wrapper
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

from PIL import Image

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)

def show(img, name = "output.png"):
    """
    Show MNSIT digits in the console.
    """
    np.save(name, img)
    fig = np.around((img + 1.0) / 2.0 * 255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(name)

def main(_):
  '''
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    
    model = attack_wrapper.AttackWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  # g.finalize()

  # Create the vocabulary.
  '''
  tf.set_random_seed(1234)
  model = attack_wrapper.AttackWrapper()
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
  sess = tf.Session()
  attack = CarliniL2(sess, model, targeted = True, batch_size=1, initial_const = 10.0, max_iterations=1000, print_every=1, confidence=0, use_log=False, abort_early=False, learning_rate=0.001)
  
  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  # Prepare the caption generator. Here we are implicitly using the default
  # beam search parameters. See caption_generator.py for a description of the
  # available beam search parameters.
  
  # preprocessing compute graph
  image_placeholder = tf.placeholder(dtype=tf.string, shape=[])
  preprocessor = model.model.process_image(image_placeholder)
  for filename in filenames:

    with tf.gfile.GFile(filename, "rb") as f:
      image = f.read()
    
    # preprocess image
    # testing computation graph
    raw_image = sess.run(preprocessor, feed_dict = {image_placeholder: image})
    print('raw image size:', raw_image.shape)

    new_sentence = "kite"
    new_sentence = "a man on a surfboard riding a wave ."
    new_sentence = "a dog riding a bike on a road ."
    new_sentence = "a group of giraffe standing next to each other ." # success, p=0.016556
    new_sentence = "a person skiing down a snow covered slope ." # success, p=0.021917
    new_sentence = "a person on a beach flying a kite ." # success, p=0.019417
    new_sentence = "a black and white photo of a train on a track ." # success, p=0.006146
    new_sentence = new_sentence.split()
    print("My new sentence:", new_sentence)
    max_caption_length = 20
    new_caption = [vocab.start_id]+[vocab.word_to_id(w) for w in new_sentence] + [vocab.end_id]
    true_cap_len = len(new_caption)
    new_caption = new_caption + [vocab.end_id]*(max_caption_length-true_cap_len)

    new_caption = [new_caption]
    print("My new id:", new_caption)
    new_mask = np.append(np.ones(true_cap_len),np.zeros(max_caption_length-true_cap_len))
    # print("Probability by attack_step:", model.attack_step(sess, new_caption, new_mask, raw_image))
    
    adv = attack.attack(np.array([raw_image]), new_caption, [new_mask])
    l2_distortion = np.sum((adv - raw_image)**2)**.5
    print("L2 distortion is", l2_distortion)
    show(raw_image, "original.png")
    show(adv, "adversarial.png")
    show(adv - raw_image, "diff.png")
    
      
if __name__ == "__main__":
  tf.app.run()
