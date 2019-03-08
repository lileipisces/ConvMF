import tensorflow as tf
import numpy as np
import random


class CNN(object):
    def __init__(self, item2word_list, seq_max_len, word_emb, output_dim=50, batch_size=128, learning_rate=0.001, dropout_ratio=0.8):

        self.item2word_list = item2word_list
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.dropout_ratio = dropout_ratio
        embed_size = word_emb.shape[1]

        g = tf.Graph()
        with g.as_default():
            # input
            self.text_i = tf.placeholder(tf.int32, shape=[None, seq_max_len], name='text_i')
            self.vector_i = tf.placeholder(tf.float32, shape=[None, output_dim], name='vector_i')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

            # embedding
            # with tf.device(device_name_or_function='/cpu:0'):
            embeddings = tf.get_variable(name='embeddings', initializer=word_emb)
            word_feature = tf.nn.embedding_lookup(params=embeddings, ids=self.text_i)  # (batch_size, seq_max_len, embed_size)
            lookup_result = tf.expand_dims(word_feature, -1)  # (batch_size, seq_max_len, embed_size, 1)

            # convolution
            filter_num = 100
            filter_sizes = [3, 4, 5]
            pooled_outputs = []
            for filter_size in filter_sizes:
                with tf.name_scope(name='conv-maxpool-{}'.format(filter_size)):
                    filter_shape = [filter_size, embed_size, 1, filter_num]
                    W = tf.get_variable(name='conv-W{}'.format(filter_size), shape=filter_shape)  # xavier_initializer by default
                    b = tf.get_variable(name='conv-b{}'.format(filter_size), shape=[filter_num], initializer=tf.constant_initializer(0.1))
                    conv = tf.nn.conv2d(input=lookup_result, filter=W, strides=[1, 1, 1, 1], padding='VALID')
                    h = tf.nn.relu(conv + b)
                    pooled = tf.nn.max_pool(value=h, ksize=[1, seq_max_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')  # (batch_size, 1, 1, filter_num)
                    pooled_outputs.append(pooled)

            # concatenate results of max pooling
            filter_total_num = filter_num * len(filter_sizes)
            h_pool = tf.concat(values=pooled_outputs, axis=-1)
            h_pool_flat = tf.reshape(tensor=h_pool, shape=[-1, filter_total_num])

            # fully connected layers
            middle_dim = 200
            hidden = tf.layers.dense(inputs=h_pool_flat, units=middle_dim, activation=tf.nn.tanh,
                                     kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                     bias_initializer=tf.constant_initializer(0.1))
            dropped = tf.nn.dropout(hidden, self.dropout_keep_prob)
            self.output_vec = tf.layers.dense(inputs=dropped, units=output_dim, activation=tf.nn.tanh,
                                              kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                              bias_initializer=tf.constant_initializer(0.1))

            self.loss = tf.losses.mean_squared_error(self.vector_i, self.output_vec)
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            init = tf.global_variables_initializer()

        self.sess = tf.Session(graph=g)
        self.sess.run(init)

    def train_one_epoch(self, M):
        '''
        :param M: shape = (item_num, dimension)
        '''
        sample_num = len(self.item2word_list)
        index_list = list(range(sample_num))
        random.shuffle(index_list)

        offset = 0
        while offset < sample_num:
            start = offset
            offset += self.batch_size
            offset = min(offset, sample_num)

            index = index_list[start:offset]
            text_i = self.item2word_list[index]
            vector_i = M[index]

            feed_dict = {self.text_i: text_i,
                         self.vector_i: vector_i,
                         self.dropout_keep_prob: self.dropout_ratio}
            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

    def get_latent_factor(self):
        sample_num = len(self.item2word_list)
        M = np.zeros((sample_num, self.output_dim), dtype=np.float32)

        offset = 0
        while offset < sample_num:
            start = offset
            offset += self.batch_size
            offset = min(offset, sample_num)

            text_i = self.item2word_list[start:offset]

            feed_dict = {self.text_i: text_i, self.dropout_keep_prob: 1.0}
            output_vec = self.sess.run(self.output_vec, feed_dict=feed_dict)
            M[start:offset] = output_vec

        return M
