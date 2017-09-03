# NOAA SEALION COUNTING CHALLENGE COMPETITION CODE 58th/385
# Author : Young-chul Yoon

import tensorflow as tf

############################### CNN materials ###########################################
def weight_variable(shape, w_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=w_name)
def bias_variable(shape, b_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = b_name)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#def batch_normalization(x, w, b, is_training):
    #z = conv2d(x, w) + b
    #return tf.layers.batch_normalization(z, center=False, scale=False, training=is_training)

def fc_mul(flat_input, input_size, fc_size, w_name, b_name):
    W_fc = weight_variable([input_size, fc_size], w_name)
    b_fc = bias_variable([fc_size], b_name)
    fc_out = tf.matmul(flat_input, W_fc) + b_fc
    return fc_out

def fc_drop(flat_input, input_size, fc_size, keep_prob, w_name, b_name):
    fc_out = fc_mul(flat_input, input_size, fc_size, w_name, b_name)
    fc_relu = tf.nn.relu(fc_out)
    fc_drop = tf.nn.dropout(fc_relu, keep_prob)
    return fc_drop

def batch_normalization(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
##########################################################################################

def featureExtract(img, ft_sz, is_training, sz):
    W_conv1 = weight_variable([ft_sz, ft_sz, 3, 64], sz + "_det_w1")  # patch size, in_channel, out_channel
    b_conv1 = bias_variable([64], sz + "_det_b1")  # bias vector with a component for each out_channel
    out_conv1_1 = conv2d(img, W_conv1) + b_conv1
    batch1_1 = batch_normalization(out_conv1_1, 64, is_training)
    relu_out1_1 = tf.nn.relu(batch1_1)

    W_re1 = weight_variable([ft_sz, ft_sz, 64, 64], sz + "_det_w_re1")  #
    b_re1 = bias_variable([64], sz + "_det_b_re1")
    out_conv1_2 = conv2d(relu_out1_1, W_re1) + b_re1
    batch1_2 = batch_normalization(out_conv1_2, 64, is_training)
    relu_out1_2 = tf.nn.relu(batch1_2)  #
    h_pool1 = max_pool_2x2(relu_out1_2)  #

    out_size = 8 * 8 * 64

    h_pool2_flat = tf.reshape(h_pool1, [-1, out_size])

    return h_pool2_flat, out_size

def deepDetect(img, ft_sz, is_training):

    W_conv1 = weight_variable([ft_sz, ft_sz, 3, 64], "det_w1")  # patch size, in_channel, out_channel
    b_conv1 = bias_variable([64], "det_b1")  # bias vector with a component for each out_channel
    out_conv1_1 = conv2d(img, W_conv1) + b_conv1
    batch1_1 = batch_normalization(out_conv1_1, 64, is_training)
    relu_out1_1 = tf.nn.relu(batch1_1)

    W_re1 = weight_variable([ft_sz, ft_sz, 64, 64], "det_w_re1")  #
    b_re1 = bias_variable([64], "det_b_re1")
    out_conv1_2 = conv2d(relu_out1_1, W_re1) + b_re1
    batch1_2 = batch_normalization(out_conv1_2, 64, is_training)
    relu_out1_2 = tf.nn.relu(batch1_2)  #
    h_pool1 = max_pool_2x2(relu_out1_2)  #

    W_conv2 = weight_variable([ft_sz, ft_sz, 64, 128], "det_w2")
    b_conv2 = bias_variable([128], "det_b2")
    out_conv2_1 = conv2d(h_pool1, W_conv2) + b_conv2
    batch2_1 = batch_normalization(out_conv2_1, 128, is_training)
    relu_out2_1 = tf.nn.relu(batch2_1)

    W_re2 = weight_variable([ft_sz, ft_sz, 128, 128], "det_w_re2")  #
    b_re2 = bias_variable([128], "det_b_re2")
    out_conv2_2 = conv2d(relu_out2_1, W_re2) + b_re2
    batch2_2 = batch_normalization(out_conv2_2, 128, is_training)
    relu_out2_2 = tf.nn.relu(batch2_2)  #
    h_pool2 = max_pool_2x2(relu_out2_2)  #

    W_conv3 = weight_variable([ft_sz, ft_sz, 128, 256], "det_w3")
    b_conv3 = bias_variable([256], "det_b3")
    out_conv3_1 = conv2d(h_pool2, W_conv3) + b_conv3
    batch3_1 = batch_normalization(out_conv3_1, 256, is_training)
    relu_out3_1 = tf.nn.relu(batch3_1)

    W_re3 = weight_variable([ft_sz, ft_sz, 256, 256], "det_w_re3")  #
    b_re3 = bias_variable([256], "det_b_re3")
    out_conv3_2 = conv2d(relu_out3_1, W_re3) + b_re3
    batch3_2 = batch_normalization(out_conv3_2, 256, is_training)
    relu_out3_2 = tf.nn.relu(batch3_2)  #
    W_rere3 = weight_variable([ft_sz, ft_sz, 256, 256], "det_w_rere3")
    b_rere3 = bias_variable([256], "det_b_rere3")
    out_conv3_3 = conv2d(relu_out3_2, W_rere3) + b_rere3
    batch3_3 = batch_normalization(out_conv3_3, 256, is_training)
    relu_out3_3 = tf.nn.relu(batch3_3)  #
    h_pool3 = max_pool_2x2(relu_out3_3)  #

    out_size = 4*4*256

    h_pool3_flat = tf.reshape(h_pool3, [-1, out_size])

    return h_pool3_flat, out_size

def deepClass(img, ft_sz, is_training):

    W_conv1 = weight_variable([ft_sz, ft_sz, 3, 64], "cls_w1")  # patch size, in_channel, out_channel
    b_conv1 = bias_variable([64], "cls_b1")  # bias vector with a component for each out_channel
    out_conv1_1 = conv2d(img, W_conv1) + b_conv1
    batch1_1 = batch_normalization(out_conv1_1, 64, is_training)
    relu_out1_1 = tf.nn.relu(batch1_1)

    W_re1 = weight_variable([ft_sz, ft_sz, 64, 64], "cls_w_re1")  #
    b_re1 = bias_variable([64], "cls_b_re1")
    out_conv1_2 = conv2d(relu_out1_1, W_re1) + b_re1
    batch1_2 = batch_normalization(out_conv1_2, 64, is_training)
    relu_out1_2 = tf.nn.relu(batch1_2)  #
    h_pool1 = max_pool_2x2(relu_out1_2)  #

    W_conv2 = weight_variable([ft_sz, ft_sz, 64, 128], "cls_w2")
    b_conv2 = bias_variable([128], "cls_b2")
    out_conv2_1 = conv2d(h_pool1, W_conv2) + b_conv2
    batch2_1 = batch_normalization(out_conv2_1, 128, is_training)
    relu_out2_1 = tf.nn.relu(batch2_1)

    W_re2 = weight_variable([ft_sz, ft_sz, 128, 128], "cls_w_re2")  #
    b_re2 = bias_variable([128], "cls_b_re2")
    out_conv2_2 = conv2d(relu_out2_1, W_re2) + b_re2
    batch2_2 = batch_normalization(out_conv2_2, 128, is_training)
    relu_out2_2 = tf.nn.relu(batch2_2)  #
    h_pool2 = max_pool_2x2(relu_out2_2)  #

    W_conv3 = weight_variable([ft_sz, ft_sz, 128, 256], "cls_w3")
    b_conv3 = bias_variable([256], "cls_b3")
    out_conv3_1 = conv2d(h_pool2, W_conv3) + b_conv3
    batch3_1 = batch_normalization(out_conv3_1, 256, is_training)
    relu_out3_1 = tf.nn.relu(batch3_1)

    W_re3 = weight_variable([ft_sz, ft_sz, 256, 256], "cls_w_re3")  #
    b_re3 = bias_variable([256], "cls_b_re3")
    out_conv3_2 = conv2d(relu_out3_1, W_re3) + b_re3
    batch3_2 = batch_normalization(out_conv3_2, 256, is_training)
    relu_out3_2 = tf.nn.relu(batch3_2)  #
    W_rere3 = weight_variable([ft_sz, ft_sz, 256, 256], "cls_w_rere3")
    b_rere3 = bias_variable([256], "cls_b_rere3")
    out_conv3_3 = conv2d(relu_out3_2, W_rere3) + b_rere3
    batch3_3 = batch_normalization(out_conv3_3, 256, is_training)
    relu_out3_3 = tf.nn.relu(batch3_3)  #
    h_pool3 = max_pool_2x2(relu_out3_3)  #

    out_size = 4 * 4 * 256

    h_pool3_flat = tf.reshape(h_pool3, [-1, out_size])

    return h_pool3_flat, out_size

def deepDetectNet(batch, class_num, keep_prob, is_training):
    conv_out, out_size = deepDetect(batch, 3, is_training)
    fc_drop1 = fc_drop(conv_out, out_size, 2048, keep_prob, "det_fc_w1", "det_fc_b1")
    fc_drop2 = fc_drop(fc_drop1, 2048, 1024, keep_prob, "det_fc_w2", "det_fc_b2")
    y_conv = fc_mul(fc_drop2, 1024, class_num, "det_fc_w3", "det_fc_b3")
    soft_out = tf.nn.softmax(y_conv)
    return y_conv, soft_out

def deepClassNet(batch, class_num, keep_prob, is_training):
    conv_out, out_size = deepClass(batch, 3, is_training)
    fc_drop1 = fc_drop(conv_out, out_size, 2048, keep_prob, "cls_fc_w1", "det_fc_b1")
    fc_drop2 = fc_drop(fc_drop1, 2048, 1024, keep_prob, "cls_fc_w2", "cls_fc_b2")
    y_conv = fc_mul(fc_drop2, 1024, class_num, "cls_fc_w3", "cls_fc_b3")
    soft_out = tf.nn.softmax(y_conv)
    return y_conv, soft_out

def deepConcatNet(s_batch, m_batch, b_batch, class_num, keep_prob, is_training):
    small_out, small_size = featureExtract(s_batch, 3, is_training, "small")
    med_out, med_size = featureExtract(m_batch, 3, is_training, "med")
    big_out, big_size = featureExtract(b_batch, 3, is_training, "big")

    tot_sz = small_size + med_size + big_size
    concat_out = tf.concat([small_out,med_out, big_out], 1)

    fc_drop1 = fc_drop(concat_out, tot_sz, 1024, keep_prob, "concat_fc_w1", "concat_fc_w2")
    y_conv = fc_mul(fc_drop1, 1024, class_num, "concat_fc_w2", "concat_fc_b2")
    soft_out = tf.nn.softmax(y_conv)
    return y_conv, soft_out

def trainStep(y_conv, gt):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    return train_step

def accuracy(y_conv, gt):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(gt, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy