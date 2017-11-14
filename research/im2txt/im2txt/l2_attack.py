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
    def __init__(self, sess, inf_sess, attack_graph, inference_graph, model, inf_model, use_keywords = True, use_logits = True, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS, print_every = 100, early_stop_iters = 0,
                 abort_early = ABORT_EARLY, 
                 initial_const = -1,
                 use_log = False, norm = "inf", adam_beta1 = 0.9, adam_beta2 = 0.999):
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

        if initial_const != -1:
            print("WARNING: initial_const in l2_attack has no effect!")
            print("WARNING: initial_const in l2_attack has no effect!")

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
        self.batch_size = batch_size
        self.repeat = binary_search_steps >= 10
        self.use_keywords = use_keywords
        self.use_logits = use_logits
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
        self.output, self.softmax, self.logits = model.predict(self.sess, self.newimg, self.input_feed, self.input_mask)
        
        # distance to the input data
        # self.lpdist = tf.reduce_sum(tf.square(self.newimg-tf.tanh(self.timg)/2),[1,2,3])
        if norm == "l2":
            self.lpdist = tf.reduce_sum(tf.square(self.newimg-tf.tanh(self.timg)),[1,2,3])
        elif norm == "inf":
            self.lpdist = tf.reduce_max(tf.abs(self.newimg-tf.tanh(self.timg)),[1,2,3])
        else:
            raise ValueError("unsupported distance metric:" + norm)

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
        # loss1 = self.output
        # loss1 = - tf.sum([tf.log(tf.reduce_maximum(self.softmax,axis=1)[self.key_word]) for keyword in self.key_words[0]])
        # loss1 = tf.reduce_sum(tf.log(tf.reduce_max(tf.gather(self.softmax, self.key_words, axis=2), axis=1)) * self.key_words_mask, axis=1)
        # loss1 = tf.reduce_sum(tf.log(tf.reduce_max(tf.gather(self.softmax, self.key_words, axis=1), axis=1)) * tf.cast(self.key_words_mask, tf.float32))
        # t=tf.log(tf.reduce_max(tf.gather(self.softmax, self.key_words, axis=1), axis=1)) * tf.cast(self.key_words_mask, tf.float32)
        # print(t.get_shape())
        # loss1 = tf.reduce_sum(tf.log(tf.reduce_max(tf.gather(tf.transpose(self.softmax, perm=[1,0]), self.key_words), axis=1)+ 1e-30) * tf.cast(self.key_words_mask, tf.float32))
        # print(self.softmax.get_shape())
        # print("softmax:", self.softmax.get_shape())
        # print("tf.gather(self.softmax, self.key_words, axis=1):", tf.gather(self.softmax, self.key_words, axis=1))
        # print("key_words_mask:", tf.cast(self.key_words_mask, tf.float32).get_shape())
        # print(tf.reduce_max(tf.gather(self.softmax, self.key_words, axis=1), axis=0).get_shape())


        # self.keywords_probs = tf.reduce_max(tf.gather(self.softmax, self.key_words, axis=1), axis=0)
        # loss1 = tf.reduce_sum(tf.log(self.keywords_probs) * tf.cast(self.key_words_mask, tf.float32))


        if self.use_keywords:
            # use the keywords loss
            # these are the true lenghth of logits and keywords without masked words
            true_logits_len = tf.cast(tf.reduce_sum(self.input_mask), tf.int32) - 1 
            true_keywords_len = tf.cast(tf.reduce_sum(self.key_words_mask), tf.int32)

            # generate masks for masking the position where a keyword is already top-1
            # reshape logits to true size
            self.logits = self.logits[:true_logits_len]
            print(self.logits.shape)

            # current top-1 prediction probability
            self.top1_probs = tf.reduce_max(self.logits, axis=1)

            # select the keywords probability from all
            self.keywords_probs = tf.gather(self.logits, self.key_words[:true_keywords_len], axis=1)
            print(self.keywords_probs.shape) # 19 * 20, or true_logits_len * true_keywords_len
            # evalute each word position, and find the maximum probability keyword at each position
            self.max_keywords_args = tf.cast(tf.argmax(self.keywords_probs, axis = 1), tf.int32)
            # largest probability among all keywords
            self.max_keywords_vals = tf.reduce_max(self.keywords_probs, axis = 1)

            # if the top-1 word is a keyword, then decrease all other keywords probability at this position!
            self.is_top1_keyword = tf.expand_dims(tf.cast(tf.equal(self.top1_probs, self.max_keywords_vals), tf.float32), axis=1)
            # generate the indices for decreasing 10000 (sparse index), by combining a range(3) with the max keyword location at each word
            self.keywords_indices_to_dec = tf.concat([tf.expand_dims(tf.range(true_logits_len), axis=1), tf.expand_dims(self.max_keywords_args, axis=1)], axis=1)
            # convert to dense array. This array indicaties the location of top-1 keywords
            self.keywords_mask = tf.scatter_nd(self.keywords_indices_to_dec, tf.ones(tf.expand_dims(true_logits_len, axis=0)), shape=(true_logits_len, true_keywords_len))
            # disable the keywords on the positions where top-1 is already a keyword
            self.disabled_mask = self.is_top1_keyword * (tf.ones_like(self.keywords_probs) - self.keywords_mask)

            if self.use_logits:
                
                """
                # generate the (sparse) indices for the maximum keyw
                self.keywords_indices_to_dec = tf.concat([tf.expand_dims(tf.range(true_logits_len), axis=1), tf.expand_dims(self.max_keywords_args, axis=1)], axis=1)
                # convert to dense array
                self.keywords_mask = tf.scatter_nd(self.keywords_indices_to_dec, tf.ones(tf.expand_dims(true_logits_len, axis=0)), shape=(true_logits_len, true_keywords_len))
                # new keywords probability with the largest masked
                self.masked_keywords_probs = self.keywords_probs - 10000 * self.keywords_mask
                # extract the second largest keyword probability
                self.top2_keyword = tf.reduce_max(self.masked_keywords_probs, axis = 1)
                # extract the top-2 loss
                self.top2_dis = tf.maximum(5 - (self.top1_probs - self.top2_keyword), 0)
                """

                # disable them! (at each position, K-1 keywords)
                self.masked_keywords_probs = self.keywords_probs - 10000 * self.disabled_mask

                
                # get the key word IDs for each position
                self.key_words_to_dec = tf.cast(tf.gather(self.key_words, self.max_keywords_args), tf.int32)
                # generate 2-D indices
                self.indices_to_dec = tf.concat([tf.expand_dims(tf.range(true_logits_len), axis=1), tf.expand_dims(self.key_words_to_dec, axis=1)], axis=1)
                # generate a mask for the maximum key word probability at each word position
                self.logits_mask = tf.scatter_nd(self.indices_to_dec, tf.ones(tf.expand_dims(true_logits_len, axis=0)), shape=(true_logits_len, int(self.logits.shape[1])))
                # modify the logits, add a large negative number to the corresponding max keyword
                self.modified_logits = self.logits - 10000 * self.logits_mask

                self.max_probs = tf.reduce_max(self.modified_logits, axis=1)
                print(self.max_probs.shape) # 19

                self.diff_probs = tf.maximum(tf.tile(tf.expand_dims(self.max_probs,1),[1,true_keywords_len]) - self.masked_keywords_probs, - self.CONFIDENCE)
                print(self.diff_probs.shape)

                self.min_diff_probs = tf.reduce_min(self.diff_probs, axis = 0)
                
                # loss1 = tf.reduce_sum(self.min_diff_probs) + tf.reduce_sum(self.top2_dis)
                loss1 = tf.reduce_sum(self.min_diff_probs)
                self.loss1 = tf.reduce_sum(self.const*loss1)
                print(loss1.shape)
            else:
                # reshape softmax to true size
                self.softmax = self.softmax[:true_logits_len]
                self.top1_softmax = tf.reduce_max(self.softmax, axis=1)

                # gather the probability of keywords at each position
                self.keywords_softmax = tf.gather(self.softmax, self.key_words[:true_keywords_len], axis=1)

                # disable them! (at each position, K-1 keywords)
                self.masked_keywords_softmax = self.keywords_softmax * (1 - self.disabled_mask)

                self.max_probs = tf.reduce_max(self.masked_keywords_softmax, axis=0)
                loss1 = tf.log(self.max_probs + 1e-30)
                self.loss1 = - tf.reduce_sum(self.const*loss1)
                # print(t.get_shape())
                # loss1 = tf.reduce_sum(tf.log(tf.reduce_max(tf.gather(tf.transpose(self.softmax, perm=[1,0]), self.key_words), axis=1)+ 1e-30)
                # print(self.softmax.get_shape())
                # print("softmax:", self.softmax.get_shape())
                # print("tf.gather(self.softmax, self.key_words, axis=1):", tf.gather(self.softmax, self.key_words, axis=1))
                # print("key_words_mask:", tf.cast(self.key_words_mask, tf.float32).get_shape())
                # print(tf.reduce_max(tf.gather(self.softmax, self.key_words, axis=1), axis=0).get_shape())
                
            if self.TARGETED:
                self.loss = self.loss1 #increase the probability of keywords
            else:
                self.loss = - self.loss1 #decrease the probability of keywords
        else:
            # use a new caption
            if self.use_logits:
                true_cap_len = tf.cast(tf.reduce_sum(self.input_mask), tf.int32)
                true_logits_len = tf.cast(tf.reduce_sum(self.input_mask), tf.int32) - 1 
                self.logits = self.logits[:true_logits_len]
                print("input_feed shape:", self.input_feed.shape)
                print("logits shape:", self.logits.shape)
                self.true_input_feed = tf.cast(self.input_feed[0][:true_cap_len],dtype=tf.int32)
                print("true_input_feed shape:", self.true_input_feed.shape)
                self.cap_indices = tf.transpose(tf.concat([[tf.range(true_logits_len)],[self.true_input_feed[1:]]], axis=0))
                print("cap_indices shape:", self.cap_indices.shape)
                self.cap_logits = tf.gather_nd(self.logits, self.cap_indices) 
                print("cap_logits shape:",self.cap_logits.shape)
                self.logits_mask = tf.scatter_nd(self.cap_indices, tf.ones([true_logits_len]),shape=(true_logits_len, int(self.logits.shape[1])))
                self.modified_logits = self.logits - 10000 * self.logits_mask
                self.max_probs = tf.reduce_max(self.modified_logits, axis=1)
                self.original_max_prob = tf.reduce_max(self.logits, axis=1)
                if self.TARGETED:
                    self.diff_probs = tf.maximum(self.max_probs - self.cap_logits, -self.CONFIDENCE)
                else:
                    self.diff_probs = tf.maximum(self.cap_logits - self.max_probs, -self.CONFIDENCE)
                print("max_probs shape:",self.max_probs.shape)
                loss1 = tf.reduce_sum(self.diff_probs)
                self.loss1 = tf.reduce_sum(self.const*loss1)
                self.loss = self.loss1
                
            else:
                # use the output probability directly
                if self.TARGETED:
                    self.loss1 = tf.reduce_sum(self.const*self.output)

                else:
                    self.loss1 = - tf.reduce_sum(self.const*self.output)
                self.loss = self.loss1

        # self.loss2 = tf.constant(0.0)
        
        # self.loss = self.loss1+self.loss2

        # regularization loss
        self.loss2 = tf.reduce_sum(self.lpdist)

        self.loss += self.loss2
        
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
        self.assign_const_op = self.const.assign(self.assign_const)
        self.setup.append(self.key_words.assign(self.assign_key_words))
        self.setup.append(self.key_words_mask.assign(self.assign_key_words_mask))
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.input_feed.assign(self.assign_input_feed))
        self.setup.append(self.input_mask.assign(self.assign_input_mask))
        self.setup.append(self.assign_const_op)
        # self.grad_op = tf.gradients(self.loss, self.modifier)
        
        self.reset_input = []
        self.reset_input.append(self.input_feed.assign(self.assign_input_feed))
        self.reset_input.append(self.input_mask.assign(self.assign_input_mask))

        self.init = tf.variables_initializer(var_list=[self.modifier]+new_vars)

    def adam_optimizer_tf(self, loss, var):
        with tf.name_scope("adam_optimier"):
            self.grad = tf.gradients(loss, var)[0]
            self.grad_norm = tf.norm(self.grad)
            # self.noise = tf.random_normal(self.grad.shape, 0.0, 1.0)
            self.noise = 0
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
            
            self.new_var = var - delta
            self.updated_newimg = tf.tanh(self.new_var + self.timg)
            assign_var = tf.assign_sub(var, delta)
            assign_mt = tf.assign(self.mt, new_mt)
            assign_vt = tf.assign(self.vt, new_vt)
            assign_epoch = tf.assign_add(self.epoch, 1)
            return tf.group(assign_var, assign_mt, assign_vt, assign_epoch)

    def attack(self, imgs, sess, inf_sess, model, inf_model, vocab, cap_key_words, cap_key_words_mask, attackid, try_id, iter_per_sentence=1, attack_const = 1.0):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            t = self.attack_batch(imgs[i:i+self.batch_size], sess, inf_sess, model, inf_model, vocab, cap_key_words, cap_key_words_mask, iter_per_sentence, attackid, try_id, attack_const)
            # r.extend(t[0])
        return t
        # return np.array(r)

    def attack_batch(self, imgs, sess, inf_sess, model, inf_model, vocab, key_words, key_words_mask, iter_per_sentence, attackid, try_id, attack_const = 1.0):
        max_caption_length = 20
        batch_size = self.batch_size

        # convert to tanh-space
        imgs = np.arctanh(imgs*0.9999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * attack_const
        upper_bound = np.ones(batch_size)*1e10
        # completely reset adam's internal state.
        self.sess.run(self.init)
        batch = imgs[:batch_size]
        # batchkeywords = key_words[:batch_size]
        # batchseqs = input_seqs[:batch_size]
        # batchmasks = input_masks[:batch_size]

        np.set_printoptions(precision=3, linewidth=150)

        # set the variables so that we don't have to send them over again
        if self.use_keywords:
            # TODO: use inference mode here
            generator = caption_generator.CaptionGenerator(inf_model, vocab, beam_size = 3)
            captions = generator.beam_search(inf_sess, imgs[0])
            infer_caption = captions[0].sentence
            # infer_caption = [1, 0, 11, 46, 0, 195, 4, 33, 5, 0, 155, 3, 2]
            true_infer_cap_len = len(infer_caption)
            
            infer_caption = infer_caption + [vocab.end_id]*(max_caption_length-true_infer_cap_len)
            infer_mask = np.append(np.ones(true_infer_cap_len),np.zeros(max_caption_length-true_infer_cap_len))
        else:
            # if not using keywords, it is the exact sentence
            infer_caption = key_words
            infer_mask = key_words_mask

 		
        self.sess.run(self.setup, {self.assign_timg: batch,
                                   self.assign_input_mask: [infer_mask],
                                   self.assign_input_feed: [infer_caption],
                                   self.assign_key_words: np.squeeze(key_words),
                                   self.assign_key_words_mask: np.squeeze(key_words_mask),
                                   self.assign_const: CONST})
        
        prev = 1e6
        train_timer = 0.0
        best_lp = 1e10
        best_img = None
        best_loss1 = 1e10
        best_loss2 = 1e10
        best_loss = 1e10
        for iteration in range(self.MAX_ITERATIONS):
            attack_begin_time = time.time()
            # perform the attack 
            if self.use_logits:
                if self.use_keywords:
                    l, l1, l2, lps, grad_norm, nimg, _, keywords_probs, top1_probs, disabled_mask, masked_keywords_probs, max_probs, diff_probs, min_diff_probs, last_input, softmax = self.sess.run([self.loss, self.loss1, self.loss2, self.lpdist, self.grad_norm, self.updated_newimg, self.train, self.keywords_probs, self.top1_probs, self.disabled_mask, self.masked_keywords_probs, self.max_probs, self.diff_probs, self.min_diff_probs, self.input_feed, self.softmax])
                else:
                    l, l1, l2, lps, grad_norm, nimg, _, cap_logits, diff_probs, original_max_prob = self.sess.run([self.loss, self.loss1, self.loss2, self.lpdist, self.grad_norm, self.newimg, self.train, self.cap_logits, self.diff_probs, self.original_max_prob])
            else:
                if self.use_keywords:
                    l, l1, l2, lps, grad_norm, nimg, _, disabled_mask, keywords_softmax, masked_keywords_softmax, top1_probs, max_probs, last_input, softmax = self.sess.run([self.loss, self.loss1, self.loss2, self.lpdist, self.grad_norm, self.updated_newimg, self.train, self.disabled_mask, self.keywords_softmax, self.masked_keywords_softmax, self.top1_softmax, self.max_probs, self.input_feed, self.softmax])
                else:
                    l, l1, l2, lps, grad_norm, nimg, _ = self.sess.run([self.loss, self.loss1, self.loss2, self.lpdist, self.grad_norm, self.newimg, self.train])
            
            print("[attack No.{}] [try No.{}] [C={:.5g}] iter = {}, time = {:.8f}, grad_norm = {:.5g}, loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}, best_lp = {:.5g}".format(attackid, try_id, CONST[0], iteration, train_timer, grad_norm, l, l1, l2, best_lp))
            sys.stdout.flush()
            
            # update softmax using the latest variable
            if self.use_logits:
                if self.use_keywords:
                    # keywords_probs, top1_probs, disabled_mask, masked_keywords_probs, max_probs, top2_keyword, top2_dis, diff_probs, min_diff_probs, last_input, softmax, logits, modified_logits = self.sess.run([self.keywords_probs, self.top1_probs, self.disabled_mask, self.masked_keywords_probs, self.max_probs, self.top2_keyword, self.top2_dis, self.diff_probs, self.min_diff_probs, self.input_feed, self.softmax, self.logits, self.modified_logits])
                    # keywords_probs, top1_probs, disabled_mask, masked_keywords_probs, max_probs, diff_probs, min_diff_probs, last_input, softmax, logits, modified_logits = self.sess.run([self.keywords_probs, self.top1_probs, self.disabled_mask, self.masked_keywords_probs, self.max_probs, self.diff_probs, self.min_diff_probs, self.input_feed, self.softmax, self.logits, self.modified_logits])
                    # print("keywords probs:", keywords_probs[:int(np.sum(key_words_mask))])
                    print("keywords probs:\n", keywords_probs.T)
                    print("disabled mask:\n", disabled_mask.T)
                    print("masked keywords probs:\n", masked_keywords_probs.T)
                    # print("top2 keyword prob:\n", top2_keyword)
                    print("top1 probs:\n", top1_probs)
                    print("max probs (after -10000):\n", max_probs)
                    # print("top2 distance:\n", top2_dis)
                    print("diff probs:\n", diff_probs.T)
                    print("min diff probs:\n", min_diff_probs)
                else:

                    # cap_logits, diff_probs, original_max_prob= self.sess.run([self.cap_logits, self.diff_probs, self.original_max_prob])
                    # print("keywords probs:", keywords_probs[:int(np.sum(key_words_mask))])
                    print("cap_logits:\n", cap_logits)
                    print("diff_probs:\n", diff_probs)
                    print("original_max_prob:\n", original_max_prob)
            else:
                if self.use_keywords:
                    print("keywords probs:\n", keywords_softmax.T)
                    print("top1 probs:\n", top1_probs)
                    print("disabled mask:\n", disabled_mask.T)
                    print("masked keywords probs:\n", masked_keywords_softmax.T)
                    print("max probs:\n", max_probs)
            
            if self.use_keywords:
                last_input = [vocab.id_to_word(s) for s in last_input[0]]
                top1_sentence = []
                for word in softmax:
                    top1_sentence.append(vocab.id_to_word(np.argmax(word)))
                print("top 1 pred:", list(zip(last_input, top1_sentence)))
            
            # use beam search in inference mode here if we do key_words based attack
            if iteration % iter_per_sentence == 0:
                # infer_caption = np.argmax(np.array(softmax), axis=1)
                # infer_caption = np.append([vocab.start_id], infer_caption).tolist()
                # generator = caption_generator.CaptionGenerator(inf_model, vocab)
                # print(nimg[0].shape)
                if self.use_keywords:
                    captions = generator.beam_search(inf_sess, nimg[0])
                    infer_caption = captions[0].sentence
                    print("current sentence:")
                    for new_caption in captions:
                        sentence = [vocab.id_to_word(w) for w in new_caption.sentence]
                        print(sentence, "p =", np.exp(new_caption.logprob))

                true_key_words = key_words[:int(np.sum(key_words_mask))]
                if self.use_keywords:
                    if self.TARGETED and set(true_key_words).issubset(infer_caption):
                        l, l1, l2, lps = self.sess.run([self.loss, self.loss1, self.loss2, self.lpdist])
                        if lps[0] < best_lp:
                            best_img = np.array(nimg)
                            best_loss = l
                            best_loss1 = l1
                            best_loss2 = l2
                            best_lp = lps[0]
                        print("<<<<<<<<<<<<<< a valid attack is found, lp =", lps[0], ", best =", best_lp, ">>>>>>>>>>>>>>>>>>>")
                else:
                    if l < best_loss:
                        best_img = np.array(nimg)
                        best_loss1 = l1
                        best_loss2 = l2
                        best_lp = lps[0]
                        best_loss = l
                        
                    
                if self.use_keywords:
                    # print("max likelihood id array found:", infer_caption)
                    true_infer_cap_len = len(infer_caption)
                    infer_caption = infer_caption[0:true_infer_cap_len] + [vocab.end_id]*(max_caption_length-true_infer_cap_len)
                    infer_mask = np.append(np.ones(true_infer_cap_len),np.zeros(max_caption_length-true_infer_cap_len))
                    # print("input id array for next iteration:", infer_caption)
                    # print("input mask for next iteration:", infer_mask)
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
            train_timer += time.time() - attack_begin_time

        # no successful attack is found ,return the last iteration image
        if best_img is None:
            return np.array(nimg), best_loss, best_loss1, best_loss2, CONST
        else:
            return best_img, best_loss, best_loss1, best_loss2, CONST

