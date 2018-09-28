# -*- coding:utf-8 -*-

import tensorflow as tf


class hCNNConfig(object):
    """CNN配置参数"""
    model_name = 'hCNN'
    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    sentence_len = 30
    doc_len = 20
    num_classes = 3  # 类别数
    num_filters = 256  # 卷积核数目
    sent_filter_sizes = [2, 3, 4, 5]
    doc_filter_sizes = [2, 3, 4]
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class hCNN(object):
    """
    title: inputs->textcnn->output_title
    content: inputs->hcnn->output_content
    concat[output_title, output_content] -> fc+bn+relu -> sigmoid_entropy.
    """

    def __init__(self, config):
        self.config = config

        self.model_name = config.model_name
        self.sent_len = config.sentence_len
        self.doc_len = config.doc_len
        self.sent_filter_sizes = config.sent_filter_sizes
        self.doc_filter_sizes = config.doc_filter_sizes
        self.n_filter = config.num_filters
        self.n_class = config.num_classes
        self.fc_hidden_size = config.hidden_dim
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        # placeholders
        self._tst = tf.placeholder(tf.bool)

        with tf.name_scope('Inputs'):
            self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            # self.batch_size = config.batch_size
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')

        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedding_size = config.embedding_dim

        with tf.variable_scope('hcnn_content'):
            output_content = self.hcnn_inference(self.embedding_inputs)

        with tf.variable_scope('fc-bn-layer'):
            output = output_content
            output_size = self.n_filter * (len(self.doc_filter_sizes))
            W_fc = self.weight_variable([output_size, self.fc_hidden_size], name='Weight_fc')
            tf.summary.histogram('W_fc', W_fc)
            h_fc = tf.matmul(output, W_fc, name='h_fc')
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.fc_hidden_size], name="beta_fc"))
            tf.summary.histogram('beta_fc', beta_fc)
            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")
            fc_bn_drop = tf.nn.dropout(self.fc_bn_relu, self.keep_prob)

        with tf.variable_scope('out_layer'):
            W_out = self.weight_variable([self.fc_hidden_size, self.n_class], name='Weight_out')
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.n_class], name='bias_out')
            tf.summary.histogram('bias_out', b_out)
            self._y_pred = tf.nn.xw_plus_b(fc_bn_drop, W_out, b_out, name='y_pred')  # 每个类别的分数 scores
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.y_pred), 1)  # 预测类别

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self._y_pred, labels=self.input_y))
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.saver = tf.train.Saver(max_to_keep=2)

        with tf.name_scope("optimize"):
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)


    @property
    def tst(self):
        return self._tst

    @property
    def global_step(self):
        return self._global_step

    @property
    def y_pred(self):
        return self._y_pred

    def weight_variable(self, shape, name):
        """Create a weight variable with appropriate initialization."""
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def batchnorm(self, Ylogits, offset, convolutional=False):
        """batchnormalization.
        Args:
            Ylogits: 1D向量或者是3D的卷积结果。
            num_updates: 迭代的global_step
            offset：表示beta，全局均值；在 RELU 激活中一般初始化为 0.1。
            scale：表示lambda，全局方差；在 sigmoid 激活中需要，这 RELU 激活中作用不大。
            m: 表示batch均值；v:表示batch方差。
            bnepsilon：一个很小的浮点数，防止除以 0.
        Returns:
            Ybn: 和 Ylogits 的维度一样，就是经过 Batch Normalization 处理的结果。
            update_moving_everages：更新mean和variance，主要是给最后的 test 使用。
        """
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,
                                                           self._global_step)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(self.tst, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(self.tst, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    def textcnn(self, X_inputs, n_step, filter_sizes, embed_size):
        """build the TextCNN network.
        n_step: the sentence len."""
        inputs = tf.expand_dims(X_inputs, -1)
        pooled_outputs = list()
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embed_size, 1, self.n_filter]
                W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_filter")
                beta = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.n_filter], name="beta"))
                tf.summary.histogram('beta', beta)
                conv = tf.nn.conv2d(inputs, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                conv_bn, update_ema = self.batchnorm(conv, beta, convolutional=True)  # 在激活层前面加 BN
                # Apply nonlinearity, batch norm scaling is not useful with relus
                # batch norm offsets are used instead of biases,使用 BN 层的 offset，不要 biases
                h = tf.nn.relu(conv_bn, name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, n_step - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
                self.update_emas.append(update_ema)
        h_pool = tf.concat(pooled_outputs, 3)
        n_filter_total = self.n_filter * len(filter_sizes)
        h_pool_flat = tf.reshape(h_pool, [-1, n_filter_total])
        return h_pool_flat  # shape = [-1, n_filter_total]

    def hcnn_inference(self, X_inputs):
        sent_inputs = tf.reshape(X_inputs, [self.batch_size * self.doc_len, self.sent_len,
                                            self.embedding_size])  # [batch_size*doc_len, sent_len, embedding_size]
        with tf.variable_scope('sentence_encoder'):  # 生成句向量
            sent_outputs = self.textcnn(sent_inputs, self.sent_len, self.sent_filter_sizes, self.embedding_size)
        with tf.variable_scope('doc_encoder'):  # 生成文档向量
            doc_inputs = tf.reshape(sent_outputs, [self.batch_size, self.doc_len, self.n_filter * len(
                self.sent_filter_sizes)])  # [batch_size, doc_len, n_filter*len(filter_sizes_sent)]
            doc_outputs = self.textcnn(doc_inputs, self.doc_len, self.doc_filter_sizes, self.n_filter * len(
                self.sent_filter_sizes))  # [batch_size, doc_len, n_filter*filter_num_doc]
        return doc_outputs  # [batch_size,  n_filter*len(doc_filter_sizes)]
