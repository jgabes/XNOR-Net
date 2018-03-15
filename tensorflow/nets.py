import tensorflow as tf

import config


@tf.RegisterGradient("CustomClipGrad")
def _clip_grad(op, grad):
    return tf.clip_by_value(grad, -1, 1)




def quantize_w(w):
    b = BinActiveZ(w)
    b[b == 0] = 1
    a = 1 / config.batch_size * tf.norm(w, ord=1)
    W = a * b
    return W

def quantize_b(b):
    b = BinActiveZ(b)
    b[b == 0] = 1
    return b

def quantize(X):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "CustomClipGrad"}):
        output = tf.sign(X)
        return output

def BinActiveZ(X):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "CustomClipGrad"}):
        output = tf.sign(X)
        return output


def BinConvolution(X, dOut, kW, kH, dW, dH):
    X = tf.layers.batch_normalization(X)
    X = BinActiveZ(X)
    X = tf.layers.conv2d(X, dOut, (kW, kH), (dW, dH), padding='same')
    return X


def BinMaxConvolution(X, dOut, kW, kH, dW, dH, mW, mH):
    X = tf.layers.batch_normalization(X)
    X = BinActiveZ(X)
    X = tf.layers.conv2d(X, dOut, (kW, kH), (dW, dH), padding='same')
    X = tf.layers.max_pooling2d(X, (mW, mH), 2, padding='same')
    return X


def BinFC(X, dOut):
    X = tf.layers.batch_normalization(X)
    X = BinActiveZ(X)
    X = tf.layers.dense(X, dOut)
    return X


def mnist_xnor(X):
    X = tf.reshape(X, [-1, 28, 28, 1])
    X = tf.layers.conv2d(X, 32, 5, 1, 'same')
    X = tf.layers.batch_normalization(X)
    X = tf.nn.relu(X)
    X = tf.layers.max_pooling2d(X, 2, 2, 'same')

    X = BinMaxConvolution(X, 64, 5, 5, 1, 1, 2, 2)
    X = BinMaxConvolution(X, 128, 5, 5, 1, 1, 2, 2)

    X = tf.reshape(X, [-1, int(X.shape[2] * X.shape[2] * X.shape[3])])
    X = BinFC(X, 128)
    X = BinFC(X, config.num_classes)
    X = tf.reshape(X, [-1, config.num_classes])
    X = tf.nn.softmax(X)
    return X


def jamie_xnor(X):
    X = tf.reshape(X, [-1, 28, 28, 1])
    X = tf.layers.conv2d(X, 32, 5, 1, 'same')
    X = tf.layers.batch_normalization(X)
    X = tf.nn.relu(X)
    X = tf.layers.max_pooling2d(X, 2, 2, 'same')

    weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.03), name='conv1_W')
    bias = tf.Variable(tf.truncated_normal([64]), name='conv1_b')
    w_q = quantize(weights)
    b_q = quantize(bias)

    X = tf.layers.batch_normalization(X)
    X = BinActiveZ(X)
    X = tf.nn.conv2d(X, w_q, [1, 1, 1, 1], padding='SAME')
    X += b_q
    X=tf.layers.max_pooling2d(X, 2, 2, padding='SAME')

    weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.03), name='conv2_W')
    bias = tf.Variable(tf.truncated_normal([128]), name='conv2_b')
    w_q = quantize(weights)
    b_q = quantize(bias)
    X = tf.layers.batch_normalization(X)
    X = BinActiveZ(X)
    X = tf.nn.conv2d(X, w_q, [1, 1, 1, 1], padding='SAME')
    X += b_q
    X = tf.layers.max_pooling2d(X, 2, 2, padding='SAME')

    numel = int(X.shape[2] * X.shape[2] * X.shape[3])
    X = tf.reshape(X, [-1, numel])
    wd1 = tf.Variable(tf.truncated_normal([numel, 128], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([128], stddev=0.01), name='bd1')
    w1_q = quantize(wd1)
    b1_q = quantize(bd1)
    X = tf.matmul(X, w1_q) + b1_q
    X = tf.nn.relu(X)

    wd1 = tf.Variable(tf.truncated_normal([128, config.num_classes], stddev=0.03), name='wd2')
    bd1 = tf.Variable(tf.truncated_normal([config.num_classes], stddev=0.01), name='bd2')
    X = tf.matmul(X, wd1) + bd1
    X = tf.reshape(X, [-1, config.num_classes])
    X = tf.nn.softmax(X)
    return X


def mnist_norm(X):
    X = tf.reshape(X, [-1, 28, 28, 1])
    X = tf.layers.conv2d(X, 32, 5, 1, 'same', activation=tf.nn.relu)
    X = tf.layers.batch_normalization(X)
    X = tf.layers.max_pooling2d(X, 2, 2, 'same')

    X = tf.layers.conv2d(X, 64, 3, 1, 'same', activation=tf.nn.relu)
    X = tf.layers.batch_normalization(X)
    X = tf.layers.max_pooling2d(X, 2, 2, 'same')

    X = tf.reshape(X, [-1, int(X.shape[2] * X.shape[2] * X.shape[3])])
    X = tf.layers.dense(X, 128, activation=tf.nn.relu)
    X = tf.layers.dense(X, config.num_classes)
    X = tf.reshape(X, [-1, config.num_classes])
    X = tf.nn.softmax(X)
    return X
