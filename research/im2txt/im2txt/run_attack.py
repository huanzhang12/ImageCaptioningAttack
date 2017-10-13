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
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
  attack_graph = tf.Graph()
  # TODO: build the inference graph
  inference_graph = tf.Graph()
  with attack_graph.as_default():
    model = attack_wrapper.AttackWrapper()
    sess = tf.Session()
    # build the attacker graph
    attack = CarliniL2(sess, attack_graph, inference_graph, model, targeted = True, batch_size=1, initial_const = 1.0, max_iterations=1000, print_every=1, confidence=0, use_log=False, abort_early=False, learning_rate=0.001)
    # compute graph for preprocessing
    image_placeholder = tf.placeholder(dtype=tf.string, shape=[])
    preprocessor = model.model.process_image(image_placeholder)
    
  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  for filename in filenames:

    with tf.gfile.GFile(filename, "rb") as f:
      image = f.read()
    
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
    
    # adv = attack.attack(np.array([raw_image]), new_caption, [new_mask])
    # key_words = [vocab.word_to_id("surfboard"),vocab.word_to_id("riding"),vocab.word_to_id("man"),vocab.word_to_id("wave"),vocab.word_to_id("dog"),vocab.word_to_id("water"),vocab.word_to_id("woman"),vocab.word_to_id("surfer"),vocab.word_to_id("ocean"),vocab.word_to_id("frisbee")]
    key_words = [vocab.word_to_id("surfboard"), vocab.word_to_id("man"), vocab.word_to_id("wave"), vocab.word_to_id("riding"), vocab.word_to_id("water")]
    key_words = key_words + [vocab.end_id]*(max_caption_length-len(key_words))
    key_words_mask = np.append(np.ones(len(key_words)),np.zeros(max_caption_length-len(key_words)))
    adv = attack.attack(np.array([raw_image]), sess, model, vocab, key_words, key_words_mask, iter_per_sentence=1)
    l2_distortion = np.sum((adv - raw_image)**2)**.5
    print("L2 distortion is", l2_distortion)
    show(raw_image, "original.png")
    show(adv, "adversarial.png")
    show(adv - raw_image, "diff.png")
    
      
if __name__ == "__main__":
  tf.app.run()
