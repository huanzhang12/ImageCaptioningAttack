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
"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from l2_attack import CarliniL2
import math
import nltk
import os
import csv
import re
import numpy as np
import json
import random

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
tf.flags.DEFINE_string("image_directory", "",
                       "directory of the images ")
tf.flags.DEFINE_bool("use_keywords", False,
                       "Use keywords based attack instead of exact attack")
tf.flags.DEFINE_bool("targeted", True,
                       "Use targeted attack")
tf.flags.DEFINE_bool("use_logits", True,
                       "Use logits as loss")
tf.flags.DEFINE_string("norm", "inf",
                        "norm to use: inf or l2")
tf.flags.DEFINE_integer("exp_num", 10,
                        "number of experiments")
tf.flags.DEFINE_string("result_directory", "./experiments_records/",
                        "the directory to save results")
tf.flags.DEFINE_integer("seed", 8,
                        "random seed")
tf.flags.DEFINE_integer("offset", 0,
                        "attack start offset")
tf.flags.DEFINE_float("C", 1,
                        "initial constant multiplied with loss1")
tf.flags.DEFINE_integer("iters", 1000,
                        "number of iterations")
tf.flags.DEFINE_integer("C_search_times", 5,
                        "try how many times for C")
tf.flags.DEFINE_integer("infer_per_iter", 5,
                        "number of iterations before inference again (valid for keywords attack)")
tf.flags.DEFINE_string("caption_file","","human caption file")

tf.flags.DEFINE_string("input_feed", "",
                       "keywords or caption input")
'''
tf.flags.DEFINE_string("keywords_POS_num", "1 0 0 0",
                       "number of keywords from different part-of-speech (POS), e.g. \"w x y z\" means w noun, x verb, y adjective and z adverb.")
'''

tf.flags.DEFINE_integer("keywords_num", 1,
                        "number of keywords")

tf.logging.set_verbosity(tf.logging.INFO)

def show(img, path, name = "output.png"):

    np.save(path+name, img)
    fig = np.around((img + 1.0) / 2.0 * 255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(path+name)

def main(_):

  tf.set_random_seed(FLAGS.seed)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  

  record_path = FLAGS.result_directory
  # we should use os.path.join!
  if record_path[-1] != "/":
      record_path += "/"

  with open(FLAGS.caption_file) as data_file:
    caption_file = json.load(data_file)
  caption_info = caption_file['annotations']

  print("using " + FLAGS.norm +" for attack")
  if FLAGS.use_keywords:
    keywords_num = FLAGS.keywords_num
    header = ("attack filename",\
      "L2 distortion","L_inf distortion","loss","loss1","loss2",\
      "optimal C","attack successful?",)
    header += tuple(["keywords"] * keywords_num)
    header += ("human caption",\
      "caption before attack 1","caption before attack 1 probability","caption before attack 2","caption before attack 2 probability",\
      "caption before attack 3","caption before attack 3 probability","caption before attack 4","caption before attack 4 probability",\
      "caption before attack 5","caption before attack 5 probability","caption after attack 1","caption after attack 1 probability",\
      "caption after attack 2","caption after attack 2 probability","caption after attack 3","caption after attack 3 probability",\
      "caption after attack 4","caption after attack 4 probability","caption after attack 5","caption after attack 5 probability")
    with open('wordPOS/noun.txt') as noun_file:
      noun = noun_file.read().split()
    with open('wordPOS/verb.txt') as verb_file:
      verb = verb_file.read().split()
    with open('wordPOS/adjective.txt') as adjective_file:
      adjective = adjective_file.read().split()
    with open('wordPOS/adverb.txt') as adverb_file:
      adverb = adverb_file.read().split()
    good_words = set(noun+verb+adjective+adverb)
  else:
    header = ("target filename","attack filename","L2 distortion","L_inf distortion","loss","loss1","loss2",\
      "optimal C","attack successful?","target caption 1","target caption 1 probability",\
      "target caption 2","target caption 2 probability","target caption 3","target caption 3 probability",\
      "target caption 4","target caption 4 probability","target caption 5","target caption 5 probability",\
      "human caption",\
      "caption before attack 1","caption before attack 1 probability","caption before attack 2","caption before attack 2 probability",\
      "caption before attack 3","caption before attack 3 probability","caption before attack 4","caption before attack 4 probability",\
      "caption before attack 5","caption before attack 5 probability","caption after attack 1","caption after attack 1 probability",\
      "caption after attack 2","caption after attack 2 probability","caption after attack 3","caption after attack 3 probability",\
      "caption after attack 4","caption after attack 4 probability","caption after attack 5","caption after attack 5 probability")
  os.system("mkdir -p {}".format(os.path.join(record_path, "fail_log")))
  record = open(os.path.join(record_path, "record_"+str(FLAGS.offset)+".csv"),"a+")
  writer = csv.writer(record)
  writer.writerow(header)
  record.close()

  fail_log = open(os.path.join(record_path, "fail_log/record_"+str(FLAGS.offset)+".csv"),"a+")
  fail_log_writer = csv.writer(fail_log)
  fail_log_writer.writerow(header)
  fail_log.close()

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
  config=tf.ConfigProto(gpu_options=gpu_options)
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
  
  
  inference_graph = tf.Graph()
  with inference_graph.as_default():
    inf_model = inference_wrapper.InferenceWrapper()
    inf_restore_fn = inf_model.build_graph_from_config(configuration.ModelConfig(),FLAGS.checkpoint_path)               
    inf_image_placeholder = tf.placeholder(dtype=tf.string, shape=[])
    inf_preprocessor = inf_model.model.process_image(inf_image_placeholder)
  inference_graph.finalize()
  inf_sess = tf.Session(graph=inference_graph, config=config)
  # Load the model from checkpoint.
  inf_restore_fn(inf_sess)
  inf_generator = caption_generator.CaptionGenerator(inf_model, vocab, beam_size=5)

  if FLAGS.targeted or FLAGS.use_keywords:
    target_g = tf.Graph()
    with target_g.as_default():
      target_model = inference_wrapper.InferenceWrapper()
      target_restore_fn = target_model.build_graph_from_config(configuration.ModelConfig(),FLAGS.checkpoint_path)
      target_image_placeholder = tf.placeholder(dtype=tf.string, shape=[])
      target_preprocessor = target_model.model.process_image(target_image_placeholder)
    target_g.finalize()
    target_sess = tf.Session(graph=target_g, config=config)
    target_restore_fn(target_sess)
    target_generator = caption_generator.CaptionGenerator(target_model, vocab, beam_size=5)

  
  attack_graph = tf.Graph()
  with attack_graph.as_default():
    model = attack_wrapper.AttackWrapper()
    sess = tf.Session(config=config)
    # build the attacker graph
    print("target:",FLAGS.targeted)
    attack = CarliniL2(sess, inf_sess, attack_graph, inference_graph, model, inf_model, targeted = FLAGS.targeted, use_keywords = FLAGS.use_keywords, use_logits = FLAGS.use_logits, batch_size=1, initial_const = FLAGS.C, max_iterations=FLAGS.iters, print_every=1, confidence=1, use_log=False, norm=FLAGS.norm, abort_early=False, learning_rate=0.005)
    # compute graph for preprocessing
    image_placeholder = tf.placeholder(dtype=tf.string, shape=[])
    preprocessor = model.model.process_image(image_placeholder)
  

  # get all the files in the directory
  image_directory = FLAGS.image_directory
  filenames = [file for file in os.listdir(image_directory)]
  filenames.sort()
  random.shuffle(filenames)


  for j in range(FLAGS.exp_num):

    if FLAGS.targeted or FLAGS.use_keywords:
      target_filename = filenames[j+FLAGS.offset]
      print("Captions for target image %s:" % os.path.basename(target_filename))
      with tf.gfile.GFile(image_directory+target_filename, "rb") as f:
        target_image = f.read()
        target_image = target_sess.run(target_preprocessor, {target_image_placeholder: target_image})
      target_captions = target_generator.beam_search(target_sess, target_image)
      target_sentences = []
      target_probs = []
      for indx, target_caption in enumerate(target_captions):
        target_sentence = [vocab.id_to_word(w) for w in target_caption.sentence[1:-1]]
        target_sentence = " ".join(target_sentence)
        print("  %d) %s (p=%f)" % (1, target_sentence, math.exp(target_caption.logprob)))
        target_sentences = target_sentences + [target_sentence]
        target_probs = target_probs + [math.exp(target_caption.logprob)]


    attack_filename = filenames[len(filenames)-1-j-FLAGS.offset]
    attack_image_id = int(re.match(r"^.*\_(.*)\..*$",attack_filename).group(1))
    human_cap = next((item for item in caption_info if item["image_id"] == attack_image_id))
    # human_cap = filter(lambda item: item["image_id"] == attack_image_id, caption_info)[0]
    
    human_cap = human_cap['caption']
    print("attack filename:",attack_filename)
    print("human's caption:", human_cap)
    with tf.gfile.GFile(image_directory+attack_filename, "rb") as f:
      image = f.read()
    raw_image = sess.run(preprocessor, feed_dict = {image_placeholder: image})


    show(raw_image, record_path, "original_"+attack_filename.replace(".jpg",".png"))
    raw_filename = record_path+"original_"+attack_filename.replace(".jpg",".png.npy")
    # raw_image = np.squeeze(np.load(raw_filename))
    raw_captions = inf_generator.beam_search(inf_sess, raw_image)
    print("Captions for original image %s:" % os.path.basename(raw_filename))
    raw_sentences = []
    raw_probs = []
    for indx, raw_caption in enumerate(raw_captions):
      raw_sentence = [vocab.id_to_word(w) for w in raw_caption.sentence[1:-1]]
      raw_sentence = " ".join(raw_sentence)
      print("  %d) %s (p=%f)" % (1, raw_sentence, math.exp(raw_caption.logprob)))
      raw_sentences = raw_sentences + [raw_sentence]
      raw_probs = raw_probs + [math.exp(raw_caption.logprob)]

    
    if FLAGS.use_keywords:
      if FLAGS.input_feed:
        words = FLAGS.input_feed.split()
      else:
        target_sentences_words = set([item for sublist in target_sentences for item in sublist.split()])
        raw_sentences_words = set([item for sublist in raw_sentences for item in sublist.split()])
        word_candidates = list((target_sentences_words & good_words) - raw_sentences_words)
        word_candidates.sort()
        words = list(np.random.choice(word_candidates, keywords_num, replace=False))
    

    if not FLAGS.targeted and not FLAGS.use_keywords:
        target_sentences = raw_sentences
        target_probs = raw_probs
        target_filename = attack_filename

    # run multiple attacks
    success = []
    C_val = [FLAGS.C]
    best_adv = None
    best_loss, best_loss1, best_loss2 = None, None, None
    l2_distortion_log = []
    linf_distortion_log = []
    best_l2_distortion = 1e10
    best_linf_distortion = 1e10
    adv_log = []
    loss1_log = []
    loss2_log = []
    loss_log = []
    for try_index in range(FLAGS.C_search_times):
      

      attack_const = C_val[try_index]
      

      max_caption_length = 20

      if FLAGS.use_keywords:
        # keywords based attack
        key_words = [vocab.word_to_id(word) for word in words]
        print("My key words are: ", words)
        key_words_mask = np.append(np.ones(len(key_words)),np.zeros(max_caption_length-len(key_words)))
        key_words = key_words + [vocab.end_id]*(max_caption_length-len(key_words))
        adv, loss, loss1, loss2, _ = attack.attack(np.array([raw_image]), sess, inf_sess, model, inf_model, vocab, key_words, key_words_mask, j, try_index, FLAGS.infer_per_iter, attack_const = attack_const)
      else:
        # exact attack
        if FLAGS.targeted:
          new_sentence = target_sentences[0]
        else:
          new_sentence = raw_sentences[0]
        # new_sentence = "a black and white photo of a train on a track ."
        new_sentence = new_sentence.split()
        print("My target sentence:", new_sentence)
        new_caption = [vocab.start_id]+[vocab.word_to_id(w) for w in new_sentence] + [vocab.end_id]
        true_cap_len = len(new_caption)
        new_caption = new_caption + [vocab.end_id]*(max_caption_length-true_cap_len)
        print("My target id:", new_caption)
        new_mask = np.append(np.ones(true_cap_len),np.zeros(max_caption_length-true_cap_len))
        adv, loss, loss1, loss2, _ = attack.attack(np.array([raw_image]), sess, inf_sess,model, inf_model, vocab, new_caption, new_mask, j, try_index, 1, attack_const = attack_const)

      # save information of this image to log array
      adv_log += [adv]
      loss_log += [loss]
      loss1_log += [loss1]
      loss2_log += [loss2]


      adv_captions = inf_generator.beam_search(inf_sess, np.squeeze(adv))
      print("Captions after this attempt:")
      adv_caption = adv_captions[0]
      adv_sentence = [vocab.id_to_word(w) for w in adv_caption.sentence[1:-1]]
      adv_sentence = " ".join(adv_sentence)
      print("  %d) %s (p=%f)" % (1, adv_sentence, math.exp(adv_caption.logprob)))

      if FLAGS.use_keywords:
        success += [set(words)<set(adv_sentence.split())]
      else:
        if FLAGS.targeted:
          success += [(adv_sentence==target_sentences[0])]
        else:
          '''
          raw_split = [item.split() for item in raw_sentences]
          nltk_BLEU = nltk.translate.bleu_score.sentence_bleu(raw_split, adv_sentence.split())
          print("BLEU by nltk is:", nltk_BLEU)
          success += [nltk_BLEU<0.5]
          '''
          success += [False]

      print("Attack with this C is successful?", success[try_index])

      l2_distortion = np.sum((adv - raw_image)**2)**.5
      linf_distortion = np.max(np.abs(adv - raw_image))
      l2_distortion_log += [l2_distortion]
      linf_distortion_log += [linf_distortion]
      print("L2 distortion is", l2_distortion)
      print("L_inf distortion is", linf_distortion)
      if success[try_index]:
        if FLAGS.norm == "l2":
          if l2_distortion<best_l2_distortion: 
            best_adv = adv
            best_loss, best_loss1, best_loss2 = loss, loss1, loss2
            best_l2_distortion = l2_distortion
            best_linf_distortion = linf_distortion
            final_C = C_val[try_index]
        elif FLAGS.norm == "inf":
          if linf_distortion<best_linf_distortion:
            best_adv = adv
            best_loss, best_loss1, best_loss2 = loss, loss1, loss2
            best_l2_distortion = l2_distortion
            best_linf_distortion = linf_distortion
            final_C = C_val[try_index]
        else:
          raise ValueError("unsupported distance metric:" + FLAGS.norm)
      if FLAGS.targeted:
        if try_index + 1 < FLAGS.C_search_times:
          if success[try_index]:
            if any(not _ for _ in success):
              last_false = len(success) - success[::-1].index(False) - 1
              C_val += [0.5 * (C_val[try_index] + C_val[last_false])]
            else:
              C_val += [C_val[try_index] * 0.5]
          else:
            if any(_ for _ in success):
              last_true = len(success) - success[::-1].index(True) - 1
              C_val += [0.5 * (C_val[try_index] + C_val[last_true])]
            else:
              C_val += [C_val[try_index] * 10.0]
      else:
        C_val += [C_val[try_index] * 10.0]

    print("results of each attempt:", success)
    print("C values of each attempt:", C_val)
    print("L2 distortion log is", l2_distortion_log)
    print("L_inf distortion log is", linf_distortion_log)
    final_success = any(_ for _ in success)
    
    if not final_success:
      final_C = C_val[-1]
      best_adv = adv
      best_loss, best_loss1, best_loss2 = loss, loss1, loss2
      if FLAGS.use_keywords:
        target_info = {'noun_keywords': noun_keywords, 'verb_keywords': verb_keywords, 'adjective_keywords':adjective_keywords, 'adverb_keywords':adverb_keywords}
      else:
        target_info = {'target_filename':target_filename, "target_sentences":target_sentences,"target_probs":target_probs}
      save_fail_log(adv_log, loss_log, loss1_log, loss2_log, l2_distortion_log, linf_distortion_log, success, C_val, record_path, attack_filename, raw_image, human_cap,\
        raw_sentences, raw_probs, inf_sess, inf_generator, vocab, target_info)

    
    show(best_adv, record_path, "adversarial_"+attack_filename.replace(".jpg",".png"))
    show(best_adv - raw_image, record_path, "diff_"+attack_filename.replace(".jpg",".png"))

    
    best_l2_distortion = np.sum((best_adv - raw_image)**2)**.5
    best_linf_distortion = np.max(np.abs(best_adv - raw_image))
    print("best L2 distortion is", best_l2_distortion)
    print("best L_inf distortion is", best_linf_distortion)

    adv_filename = record_path+"adversarial_"+attack_filename.replace(".jpg",".png.npy")
    adv_image = np.squeeze(np.load(adv_filename))
    adv_captions = inf_generator.beam_search(inf_sess, adv_image)
    print("Captions for adversarial image %s:" % os.path.basename(adv_filename))
    adv_sentences = []
    adv_probs = []
    for indx, adv_caption in enumerate(adv_captions):
      adv_sentence = [vocab.id_to_word(w) for w in adv_caption.sentence[1:-1]]
      adv_sentence = " ".join(adv_sentence)
      print("  %d) %s (p=%f)" % (1, adv_sentence, math.exp(adv_caption.logprob)))
      adv_sentences = adv_sentences + [adv_sentence]
      adv_probs = adv_probs + [math.exp(adv_caption.logprob)]

    record = open(record_path + "record_"+str(FLAGS.offset)+".csv","a+")
    writer = csv.writer(record)
    if FLAGS.use_keywords:
      row = (attack_filename, best_l2_distortion,best_linf_distortion,\
        best_loss,best_loss1,best_loss2,final_C,str(final_success))
      row +=  tuple(noun_keywords)
      row += tuple(verb_keywords)
      row += tuple(adjective_keywords)
      row += tuple(adverb_keywords)
      row += (human_cap,raw_sentences[0],str(raw_probs[0]),raw_sentences[1],str(raw_probs[1]),raw_sentences[2],str(raw_probs[2]),\
        raw_sentences[3],str(raw_probs[3]),raw_sentences[4],str(raw_probs[4]),\
        adv_sentences[0],str(adv_probs[0]),adv_sentences[1],str(adv_probs[1]),adv_sentences[2],str(adv_probs[2]),\
        adv_sentences[3],str(adv_probs[3]),adv_sentences[4],str(adv_probs[4]))
      writer.writerow(row)
    else:

      writer.writerow( (target_filename,attack_filename,\
        best_l2_distortion,best_linf_distortion,best_loss,best_loss1,best_loss2,\
        final_C,str(final_success),\
        target_sentences[0],str(target_probs[0]),target_sentences[1],str(target_probs[1]),target_sentences[2],str(target_probs[2]),\
        target_sentences[3],str(target_probs[3]),target_sentences[4],str(target_probs[4]),human_cap,\
        raw_sentences[0],str(raw_probs[0]),raw_sentences[1],str(raw_probs[1]),raw_sentences[2],str(raw_probs[2]),\
        raw_sentences[3],str(raw_probs[3]),raw_sentences[4],str(raw_probs[4]),\
        adv_sentences[0],str(adv_probs[0]),adv_sentences[1],str(adv_probs[1]),adv_sentences[2],str(adv_probs[2]),\
        adv_sentences[3],str(adv_probs[3]),adv_sentences[4],str(adv_probs[4])))
    record.close()
    print("****************************** END OF THIS ATTACK ***********************************")

  inf_sess.close()
  if not FLAGS.use_keywords and not FLAGS.targeted:
    target_sess.close()
  
  sess.close()



def save_fail_log(adv_log, loss_log, loss1_log, loss2_log, l2_distortion_log, linf_distortion_log, success, C_val, record_path, attack_filename, raw_image,human_cap,\
  raw_sentences, raw_probs, inf_sess,inf_generator, vocab, target_info): 
  for i in range(len(adv_log)):
    show(adv_log[i], record_path+"fail_log/", "fail_adversarial_C_"+str(C_val[i])+attack_filename.replace(".jpg",".png"))
    show(adv_log[i] - raw_image, record_path+"fail_log/", "fail_diff_C_"+str(C_val[i])+attack_filename.replace(".jpg",".png"))

    fail_log = open(record_path + "fail_log/record_"+str(FLAGS.offset)+".csv","a+")
    fail_log_writer = csv.writer(fail_log)
    adv_captions = inf_generator.beam_search(inf_sess, np.squeeze(adv_log[i]))
    adv_sentences = []
    adv_probs = []
    if FLAGS.use_keywords:
      noun_keywords = target_info['noun_keywords']
      verb_keywords = target_info['verb_keywords']
      adjective_keywords = target_info['adjective_keywords']
      adverb_keywords = target_info['adverb_keywords']
      for indx, adv_caption in enumerate(adv_captions):
        adv_sentence = [vocab.id_to_word(w) for w in adv_caption.sentence[1:-1]]
        adv_sentence = " ".join(adv_sentence)
        adv_sentences = adv_sentences + [adv_sentence]
        adv_probs = adv_probs + [math.exp(adv_caption.logprob)]
      row = (attack_filename, l2_distortion_log[i],linf_distortion_log[i],\
        loss_log[i],loss1_log[i],loss2_log[i],C_val[i],success[i],)
      row +=  tuple(noun_keywords)
      row += tuple(verb_keywords)
      row += tuple(adjective_keywords)
      row += tuple(adverb_keywords)
      row += (human_cap,\
        raw_sentences[0],str(raw_probs[0]),raw_sentences[1],str(raw_probs[1]),raw_sentences[2],str(raw_probs[2]),\
        raw_sentences[3],str(raw_probs[3]),raw_sentences[4],str(raw_probs[4]),adv_sentences[0],str(adv_probs[0]),\
        adv_sentences[1],str(adv_probs[1]),adv_sentences[2],str(adv_probs[2]),adv_sentences[3],str(adv_probs[3]),\
        adv_sentences[4],str(adv_probs[4]))
      fail_log_writer.writerow(row)
    else:
      target_sentences = target_info['target_sentences']
      target_filename = target_info['target_filename']
      target_probs = target_info['target_probs']
      for indx, adv_caption in enumerate(adv_captions):
        adv_sentence = [vocab.id_to_word(w) for w in adv_caption.sentence[1:-1]]
        adv_sentence = " ".join(adv_sentence)
        adv_sentences = adv_sentences + [adv_sentence]
        adv_probs = adv_probs + [math.exp(adv_caption.logprob)]
      fail_log_writer.writerow( (target_filename,attack_filename,\
        l2_distortion_log[i],linf_distortion_log[i],loss_log[i],loss1_log[i],loss2_log[i],\
        C_val[i],success[i],\
        target_sentences[0],str(target_probs[0]),target_sentences[1],str(target_probs[1]),\
        target_sentences[2],str(target_probs[2]),target_sentences[3],str(target_probs[3]),\
        target_sentences[4],str(target_probs[4]),\
        human_cap,\
        raw_sentences[0],str(raw_probs[0]),raw_sentences[1],str(raw_probs[1]),raw_sentences[2],str(raw_probs[2]),\
        raw_sentences[3],str(raw_probs[3]),raw_sentences[4],str(raw_probs[4]),adv_sentences[0],str(adv_probs[0]),\
        adv_sentences[1],str(adv_probs[1]),adv_sentences[2],str(adv_probs[2]),adv_sentences[3],str(adv_probs[3]),\
        adv_sentences[4],str(adv_probs[4])))
    fail_log.close()

if __name__ == "__main__":
  tf.app.run()
