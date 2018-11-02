#!/anaconda3/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - songheqi <songheqi1996@gmail.com>

import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode

class SharedModel:
    """The Shared Network of NER models, includes embedding and RNN(LSTM/GRU)s"""

    reuse = False
    
    def __init__(self, args, vocab_size):
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.dropout_rate = args.dropout
        self.vocab_size = vocab_size
        self.name = 'shared_part'      
        
    
    def add_placeholders(self):
        with tf.variable_scope(self.name, reuse=SharedModel.reuse):
            self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='inputs')
            self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='targets')
            self.dropout_pl = tf.placeholder(dtype=tf.float32, name='dropout')
            self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_lenth')
    

    def embedding_layer(self):
        with tf.variable_scope("embedding", reuse=SharedModel.reuse), tf.device('/cpu:0'):
            self._embedding = tf.get_variable(name='_embedding',
                                               shape=[self.vocab_size, self.embedding_size],
                                               trainable=False)
            embedding = tf.nn.embedding_lookup(params=self._embedding,
                                               ids=self.inputs,
                                               name="embeddings")
        self.embeddings = tf.nn.dropout(embedding, self.dropout_pl)
    
    def biLSTM_layer(self):
        with tf.variable_scope("bi-lstm", reuse=SharedModel.reuse):
            cell_fw = LSTMCell(self.hidden_size)
            cell_bw = LSTMCell(self.hidden_size)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)
    
        self.lstm_outputs = output
    
    def build(self):
        self.add_placeholders()
        self.embedding_layer()
        self.biLSTM_layer()
        SharedModel.reuse = True

class SpecModel():
    """The Special part of each domain ner task"""

    def __init__(self, args, num_tags, vocab_size, name):
        self.args = args
        self.PROJ = args.PROJ
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.num_tags = num_tags
        self.dropout_rate = args.dropout
        self.lr = args.lr
        self.grad_clip = args.grad_clip
        self.project_size = args.project_size
        self.vocab_size = vocab_size
        self.name = name
    

    def shared_layer_op(self):
        self.shared_layers = SharedModel(self.args, self.vocab_size)
        self.shared_layers.build()


    def get_shared_params(self):
        self.inputs = self.shared_layers.inputs
        self.targets = self.shared_layers.targets
        self.dropout_pl = self.shared_layers.dropout_pl
        self.sequence_lengths = self.shared_layers.sequence_lengths
        self.batch_size = tf.shape(self.inputs)[0]
        self.time_steps = tf.shape(self.inputs)[-1]
        self.lstm_outputs = self.shared_layers.lstm_outputs

    
    def project_layer(self):
        with tf.variable_scope(self.name + "proj"):
            with tf.variable_scope('hidden'):
                w = tf.get_variable('w',
                                    shape=[self.hidden_size * 2, self.project_size],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b',
                                    shape=[self.project_size],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.lstm_outputs, shape=[-1, self.hidden_size * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, w, b))

            with tf.variable_scope('logits'):
                w = tf.get_variable('w',
                                    shape=[self.project_size, self.num_tags],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b',
                                    shape=[self.num_tags],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                pred = tf.nn.xw_plus_b(hidden, w, b)

        self.logits = tf.reshape(pred, [-1, self.time_steps, self.num_tags])
    
    
    def crf_layer(self):
        with tf.variable_scope(self.name + 'crf'):
            if self.PROJ == "Linear":
                log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                            tag_indices=self.targets,
                                                                            sequence_lengths=self.sequence_lengths)
                self.loss = -tf.reduce_mean(log_likelihood)

            else:
                """About Domain Masks see Multi-task Domain Adaptation for Sequence Tagging"""
                small = -1000.0
                start_logits = tf.concat(
                    [small * tf.ones(shape=[self.batch_size, 1, self.num_tags + 1]),
                    tf.zeros(shape=[self.batch_size, 1, 1])],
                    axis=-1
                )
                end_logits = tf.concat(
                    [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]),
                    tf.zeros(shape=[self.batch_size, 1, 1]),
                    small * tf.ones(shape=[self.batch_size, 1, 1])],
                    axis=-1
                )
                pad_logits = tf.cast(small * tf.ones([self.batch_size, self.time_steps, 2]), tf.float32)
                logits = tf.concat([self.logits, pad_logits], axis=-1)
                logits = tf.concat([start_logits, logits, end_logits], axis=1)
                targets = tf.concat(
                    [tf.cast((self.num_tags + 1) * tf.ones([self.batch_size, 1]), tf.int32),
                    self.targets,
                    tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32)],
                    axis=-1
                )
                log_likelihood, self.transition_params = crf_log_likelihood(
                    inputs=logits,
                    tag_indices=targets,
                    sequence_lengths=self.sequence_lengths + 2
                )
                self.loss = -tf.reduce_mean(log_likelihood)

    
    def optimize(self):
        with tf.variable_scope(self.name + "optimizer"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            # solute "None values not supported."
            grads_and_vars_clip = [[g, v] if g is None else \
                            [tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v] for g, v in grads_and_vars]
            self.train_op = self.optimizer.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    
    def build(self):
        self.shared_layer_op()
        self.get_shared_params()
        self.project_layer()
        self.crf_layer()
        self.optimize()
        
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def _create_feed_dict(self, is_train, batch):
        _, ids, tags, lengths = batch
        feed_dict = {
            self.inputs: np.asarray(ids),
            self.sequence_lengths: np.asarray(lengths),
            self.dropout_pl: 1.0
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout_pl] = 0.5
        return feed_dict

    def get_embeddings(self):
        return self.shared_layers._embedding

    def run_one_step(self, sess, is_train, batch):
        feed_dict = self._create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run([self.global_step, self.loss, self.train_op],
                                            feed_dict)
            return global_step, loss
        else:
            logits = sess.run([self.logits],
                              feed_dict)
            return logits[0]
    
    def decode(self, logits, seq_len_list, matrix):
        if self.PROJ == "Linear":
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                logit = logit[:seq_len]
                viterbi_seq, _ = viterbi_decode(logit, matrix)
                label_list.append(viterbi_seq)
        else:
            label_list = []
            small = -1000.0
            start = np.asarray([[small] * self.num_tags + [small] + [0]])
            end = np.asarray([[small] * self.num_tags + [0] + [small]])
            for logit, seq_len in zip(logits, seq_len_list):
                logit = logit[:seq_len]
                pad = small * np.ones([seq_len, 2])
                logit = np.concatenate([logit, pad], axis=1)
                logit = np.concatenate([start, logit, end], axis=0)
                path, _ = viterbi_decode(logit, matrix)
                label_list.append(path[1:-1])
        return label_list

    def evaluate_line(self, sess, inputs, id2tag):
        lengths = inputs[-1]
        trans = self.transition_params.eval()
        logits = self.run_one_step(sess, False, inputs)
        paths = self.decode(logits, lengths, trans)
        tags = [id2tag[idx] for idx in paths[0]]
        return tags

    def evaluate(self, sess, data_manger, id2tag):
        results = []
        trans = self.transition_params.eval()
        for batch in data_manger.iter_batch():
            chars, _, tags, lengths = batch
            logits = self.run_one_step(sess, False, batch)
            paths = self.decode(logits, lengths, trans)
            for i in range(len(chars)):
                result = []
                string = chars[i][:lengths[i]]
                lab = [id2tag[x] for x in tags[i][:lengths[i]]]
                pred = [id2tag[x] for x in paths[i][:lengths[i]]]
                for char, lab, pred in zip(string, lab, pred):
                    result.append(" ".join([char, lab, pred]))
                results.append(result)
        return results