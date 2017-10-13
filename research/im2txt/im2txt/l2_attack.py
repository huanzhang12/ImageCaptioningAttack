## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np
import time
import timeit
from im2txt.inference_utils import vocabulary
from im2txt.inference_utils import caption_generator

BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 2e-3     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1     # the initial constant c to pick as a first guess

class CarliniL2:
    def __init__(self, sess, attack_graph, inference_graph, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS, print_every = 100, early_stop_iters = 0,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 use_log = False, adam_beta1 = 0.9, adam_beta2 = 0.999):
        """
        The L_2 optimized attack. 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        """

        # image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        image_size, num_channels = model.image_size, model.num_channels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.print_every = print_every
        self.early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iterations // 10
        print("early stop:", self.early_stop_iters)
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        max_caption_length = 20
        self.repeat = binary_search_steps >= 10
        # store the two graphs
        self.attack_graph = attack_graph
        self.inference_graph = inference_graph
        # make sure we are building the attack graph
        assert sess.graph is attack_graph

        shape = (batch_size,image_size,image_size,num_channels)
        
        # the variable we're going to optimize over
        self.modifier = tf.Variable(np.zeros(shape,dtype=np.float32), name="modifier_var")
        # self.modifier = tf.Variable(np.load('black_iter_350.npy').astype(np.float32).reshape(shape))

        # these are variables to be more efficient in sending data to tf
        max_caption_length = 20

        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32, name="timg_var")
        # self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32, name="const_var")
        self.input_feed = tf.Variable(np.zeros((batch_size, max_caption_length)), dtype=tf.int64, name="input_feed_var")
        self.input_mask = tf.Variable(np.zeros((batch_size, max_caption_length)), dtype=tf.int64, name="input_mask_var")
        self.key_words = tf.Variable(np.zeros(max_caption_length), dtype=tf.int64, name="key_words_var")
        self.key_words_mask = tf.Variable(np.zeros(max_caption_length), dtype=tf.int64, name="key_words_mask_var")
        # TODO: add keywords input, same as self.timg

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        # self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        self.assign_input_feed = tf.placeholder(tf.int64, (batch_size, max_caption_length), name="input_feed")
        self.assign_input_mask = tf.placeholder(tf.int64, (batch_size, max_caption_length), name="input_mask")
        self.assign_key_words = tf.placeholder(tf.int64, [max_caption_length], name="key_words")
        self.assign_key_words_mask = tf.placeholder(tf.int64, [max_caption_length], name="key_words_mask")
        # the resulting image, tanh'd to keep bounded from -0.5 to 0.5
        # self.newimg = tf.tanh(self.modifier + self.timg)/2
        self.newimg = tf.tanh(self.modifier + self.timg)
        
        # prediction BEFORE-SOFTMAX of the model
        self.output, self.logits = model.predict(self.sess, self.newimg, self.input_feed, self.input_mask)
        
        # distance to the input data
        # self.l2dist = tf.reduce_sum(tf.square(self.newimg-tf.tanh(self.timg)/2),[1,2,3])
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-tf.tanh(self.timg)),[1,2,3])

        '''
        # compute the probability of the label class versus the maximum other
        self.real = tf.reduce_sum((self.tlab)*self.output,1)
        self.other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)
        
        if self.TARGETED:
            if use_log:
                # loss1 = tf.maximum(- tf.log(self.other), - tf.log(self.real))
                # loss1 = - tf.log(self.real)
                loss1 = tf.maximum(0.0, tf.log(self.other + 1e-30) - tf.log(self.real + 1e-30))
            else:
                # if targetted, optimize for making the other class most likely
                loss1 = tf.maximum(0.0, self.other-self.real+self.CONFIDENCE)
        else:
            if use_log:
                # loss1 = tf.log(self.real)
                loss1 = tf.maximum(0.0, tf.log(self.real + 1e-30) - tf.log(self.other + 1e-30))
            else:
            # if untargeted, optimize for making this class least likely.
                loss1 = tf.maximum(0.0, self.real-self.other+self.CONFIDENCE)
        '''

        # loss1 = - self.output
        # TODO: use a new loss
        # loss1 = self.output
        # loss1 = - tf.sum([tf.log(tf.reduce_maximum(self.logits,axis=1)[self.key_word]) for keyword in self.key_words[0]])
        # loss1 = tf.reduce_sum(tf.log(tf.reduce_max(tf.gather(self.logits, self.key_words, axis=2), axis=1)) * self.key_words_mask, axis=1)
        # loss1 = tf.reduce_sum(tf.log(tf.reduce_max(tf.gather(self.logits, self.key_words, axis=1), axis=1)) * tf.cast(self.key_words_mask, tf.float32))
        # t=tf.log(tf.reduce_max(tf.gather(self.logits, self.key_words, axis=1), axis=1)) * tf.cast(self.key_words_mask, tf.float32)
        # print(t.get_shape())
        # loss1 = tf.reduce_sum(tf.log(tf.reduce_max(tf.gather(tf.transpose(self.logits, perm=[1,0]), self.key_words), axis=1)+ 1e-30) * tf.cast(self.key_words_mask, tf.float32))
        # print(self.logits.get_shape())
        # print("logits:", self.logits.get_shape())
        # print("tf.gather(self.logits, self.key_words, axis=1):", tf.gather(self.logits, self.key_words, axis=1))
        # print("key_words_mask:", tf.cast(self.key_words_mask, tf.float32).get_shape())
        # print(tf.reduce_max(tf.gather(self.logits, self.key_words, axis=1), axis=0).get_shape())
        self.keywords_probs = tf.reduce_max(tf.gather(self.logits, self.key_words, axis=1), axis=0)
        loss1 = tf.reduce_sum(tf.log(self.keywords_probs) * tf.cast(self.key_words_mask, tf.float32))

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        # self.loss2 = tf.constant(0.0)
        self.loss1 = tf.reduce_sum(self.const*loss1)
        # self.loss = self.loss1+self.loss2
        self.loss = self.loss1
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        # optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        # optimizer = tf.train.MomentumOptimizer(self.LEARNING_RATE, 0.99, use_nesterov = True)
        # optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE, centered = True, momentum = 0.9)
        # optimizer = tf.train.AdadeltaOptimizer(self.LEARNING_RATE)
        # optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE, adam_beta1, adam_beta2)
        # self.train = optimizer.minimize(self.loss, var_list=[self.modifier])
        self.train = self.adam_optimizer_tf(self.loss, self.modifier)
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        # TODO: add keywords input
        self.setup.append(self.key_words.assign(self.assign_key_words))
        self.setup.append(self.key_words_mask.assign(self.assign_key_words_mask))
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.input_feed.assign(self.assign_input_feed))
        self.setup.append(self.input_mask.assign(self.assign_input_mask))
        self.setup.append(self.const.assign(self.assign_const))
        # self.grad_op = tf.gradients(self.loss, self.modifier)
        
        self.reset_input = []
        self.reset_input.append(self.input_feed.assign(self.assign_input_feed))
        self.reset_input.append(self.input_mask.assign(self.assign_input_mask))

        self.init = tf.variables_initializer(var_list=[self.modifier]+new_vars)


    def adam_optimizer_tf(self, loss, var):
        with tf.name_scope("adam_optimier"):
            self.grad = tf.gradients(loss, var)[0]
            self.noise = tf.random_normal(self.grad.shape, 0.0, 1.0)
            self.beta1 = tf.constant(0.9)
            self.beta2 = tf.constant(0.999)
            self.lr = tf.constant(self.LEARNING_RATE)
            self.epsilon = 1e-8
            self.epoch = tf.Variable(0, dtype = tf.float32)
            self.mt = tf.Variable(np.zeros(var.shape), dtype = tf.float32)
            self.vt = tf.Variable(np.zeros(var.shape), dtype = tf.float32)

            new_mt = self.beta1 * self.mt + (1 - self.beta1) * self.grad
            new_vt = self.beta2 * self.vt + (1 - self.beta2) * tf.square(self.grad)
            corr = (tf.sqrt(1 - tf.pow(self.beta2, self.epoch))) / (1 - tf.pow(self.beta1, self.epoch))
            # delta = self.lr * corr * (new_mt / (tf.sqrt(new_vt) + self.epsilon))
            delta = self.lr * corr * ((new_mt / tf.sqrt(new_vt + self.epsilon)) + self.noise / tf.sqrt(self.epoch + 1))
            # delta = self.lr * (self.grad + self.noise)

            assign_var = tf.assign_sub(var, delta)
            assign_mt = tf.assign(self.mt, new_mt)
            assign_vt = tf.assign(self.vt, new_vt)
            assign_epoch = tf.assign_add(self.epoch, 1)
            return tf.group(assign_var, assign_mt, assign_vt, assign_epoch)

    # def attack(self, imgs, targets):
    def attack(self, imgs, sess, model, vocab, cap_key_words, cap_key_words_mask, iter_per_sentence=1):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], sess, model, vocab, cap_key_words, cap_key_words_mask, iter_per_sentence)[0])
        return np.array(r)

    def attack_batch(self, imgs, sess, model, vocab, key_words, key_words_mask, iter_per_sentence):
   
        batch_size = self.batch_size

        # convert to tanh-space
        imgs = np.arctanh(imgs)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10
        # completely reset adam's internal state.
        self.sess.run(self.init)
        batch = imgs[:batch_size]
        # batchkeywords = key_words[:batch_size]
        # batchseqs = input_seqs[:batch_size]
        # batchmasks = input_masks[:batch_size]

        bestl2 = [1e10]*batch_size

        # The last iteration (if we run many steps) repeat the search once.
        if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
            CONST = upper_bound

        # set the variables so that we don't have to send them over again
        # TODO: set batchmask and batchseq to inference result
       
        infer_caption = [1, 0, 11, 46, 0, 195, 4, 33, 5, 0, 155, 3, 2]
        true_infer_cap_len = len(infer_caption)
        max_caption_length = 20
        infer_caption = infer_caption + [vocab.end_id]*(max_caption_length-true_infer_cap_len)
        infer_mask = np.append(np.ones(true_infer_cap_len),np.zeros(max_caption_length-true_infer_cap_len))
 		
 		
        self.sess.run(self.setup, {self.assign_timg: batch,
                                   self.assign_input_mask: [infer_mask],
                                   self.assign_input_feed: [infer_caption],
                                   self.assign_key_words: key_words,
                                   self.assign_key_words_mask: key_words_mask,
                                   self.assign_const: CONST})
        
        prev = 1e6
        train_timer = 0.0
        for iteration in range(self.MAX_ITERATIONS):
            start = time.time()
            # print out the losses every 10%
            # if iteration%(self.MAX_ITERATIONS//self.print_every) == 0:
            if True:
                # print(iteration,self.sess.run((self.loss,self.real,self.other,self.loss1,self.loss2)))
                # grad = self.sess.run(self.grad_op)
                # old_modifier = self.sess.run(self.modifier)
                # np.save('white_iter_{}'.format(iteration), modifier)
                loss, loss1, loss2 = self.sess.run((self.loss,self.loss1,self.loss2))
                print("[STATS][L2] iter = {}, time = {:.8f}, loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}".format(iteration, train_timer, loss, loss1, loss2))
                sys.stdout.flush()

            attack_begin_time = time.time()
            # perform the attack 
            _, l, l2s, scores, logits, nimg = self.sess.run([self.train, self.loss, 
                                                     self.l2dist, self.output, 
                                                     self.logits, self.newimg])

            # update logits using the latest variable
            keywords_probs, logits = self.sess.run([self.keywords_probs, self.logits])
            print("keywords probs:", keywords_probs[:5])

            # TODO: according to logits find the "top-1" prediction and generate batchseqs and batchmasks
            if iteration % iter_per_sentence == 0:
                infer_caption = np.argmax(np.array(logits), axis=1)
                infer_caption = np.append([vocab.start_id], infer_caption).tolist()

                sentence = [vocab.id_to_word(w) for w in infer_caption]

                print("max likelihood sentence found:",sentence)
                # print("max likelihood id array found:", infer_caption)
                if 2 in infer_caption:
                	true_infer_cap_len = infer_caption.index(2)+1
                else:
                	true_infer_cap_len = 15 
                infer_caption = infer_caption[0:true_infer_cap_len] + [vocab.end_id]*(max_caption_length-true_infer_cap_len)
                infer_mask = np.append(np.ones(true_infer_cap_len),np.zeros(max_caption_length-true_infer_cap_len))
                # print("input id array for next iteration:", infer_caption)
                # print("input mask for next iteration:", infer_mask)
                # TODO: reset batchmask and batchseq
                # print caption in each iteration
                self.sess.run(self.reset_input, {self.assign_input_mask: [infer_mask],
                                           self.assign_input_feed: [infer_caption]})
            # print(grad[0].reshape(-1))
            # print((old_modifier - new_modifier).reshape(-1))

            # check if we should abort search if we're getting nowhere.
            if self.ABORT_EARLY and iteration % self.early_stop_iters == 0:
                if l > prev*.9999:
                    print("Early stopping because there is no improvement")
                    break
                prev = l
            end = time.time()
            print("time of this iteration:", end - start)

        return np.array(nimg), CONST

