import tensorflow as tf

import config

strides = [1, 2, 2, 1]


@tf.RegisterGradient("CustomClipGrad")
def _clip_grad(op, grad):
    return tf.clip_by_value(grad, -1, 1)


def quantize_w(w):
    b = BinActiveZ(w)
    # [b == 0] = 1
    a = 1 / config.batch_size * tf.norm(w, ord=1)
    W = a * b
    return W


def quantize_b(b):
    # b = BinActiveZ(b)
    # b[b == 0] = 1
    return BinActiveZ(b)


def quantize(X):
    return BinActiveZ(X)


def BinActiveZ(X):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "CustomClipGrad"}):
        return tf.sign(X)


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
    X = tf.layers.conv2d(X, 32, 5, 1, 'same', name='conv1')
    X = tf.layers.batch_normalization(X)
    X = tf.nn.relu(X, name="relu1")
    X = tf.layers.max_pooling2d(X, 2, 2, 'same', name='cpool1')

    weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.2, mean=0.5), name='conv2_W')
    bias = tf.Variable(tf.truncated_normal([64], stddev=0.2, mean=0.0), name='conv2_b')
    w_q = quantize(weights)
    b_q = quantize(bias)

    X = tf.layers.batch_normalization(X)
    X = BinActiveZ(X)
    X = tf.nn.conv2d(X, w_q, strides=strides, padding='SAME', name="conv2")
    X = X + b_q
    X = tf.layers.max_pooling2d(X, 2, 2, padding='SAME')

    weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.2, mean=0.5), name='conv3_W')
    bias = tf.Variable(tf.truncated_normal([128], stddev=0.2, mean=0.0), name='conv3_b')
    w_q = quantize(weights)
    b_q = quantize(bias)
    X = tf.layers.batch_normalization(X)
    X = BinActiveZ(X)
    X = tf.nn.conv2d(X, w_q, strides=strides, padding='SAME', name='conv3')
    X = X + b_q
    X = tf.layers.max_pooling2d(X, 2, 2, padding='SAME', name='cpool3')

    numel = int(X.shape[2] * X.shape[2] * X.shape[3])
    X = tf.reshape(X, [-1, numel])
    wd1 = tf.Variable(tf.truncated_normal([numel, 128], stddev=0.2, mean=0.5), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([128], stddev=0.2, mean=0.0), name='bd1')
    w1_q = quantize(wd1)
    b1_q = quantize(bd1)
    X = tf.matmul(X, w1_q) + b1_q
    X = tf.nn.relu(X)

    wd2 = tf.Variable(tf.truncated_normal([128, config.num_classes], stddev=0.2, mean=0.5), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([config.num_classes], stddev=0.2, mean=0.0), name='bd2')
    X = tf.matmul(X, wd2) + bd2
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


def pong_norm(X):
    X = tf.layers.conv2d(X, filters=config.conv1, kernel_size=5, padding='same', activation=tf.nn.relu)
    X = tf.layers.batch_normalization(X)
    X = tf.layers.max_pooling2d(X, 2, 2, 'same')

    X = tf.layers.conv2d(X, filters=config.conv2, kernel_size=3, padding='same', activation=tf.nn.relu)
    X = tf.layers.batch_normalization(X)
    X = tf.layers.max_pooling2d(X, 2, 2, 'same')

    X = tf.reshape(X, [-1, int(X.shape[2] * X.shape[2] * X.shape[3])])
    X = tf.layers.dense(X, units=config.FC1, activation=tf.nn.relu)
    X = tf.layers.dense(X, units=1)
    X = tf.reshape(X, [-1, 1])
    return X
