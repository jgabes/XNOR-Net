import tensorflow as tf
import config

@tf.RegisterGradient("CustomClipGrad")
def _clip_grad(op, grad):
    return tf.clip_by_value(grad, -1, 1)


def BinActiveZ(X):
    g = tf.get_default_graph()
    with g.gradient_override_map({"BAZ": "CustomClipGrad"}):
        output = tf.sign(X, name="BAZ")
        return output


def BinConvolution(X, dOut, kW, kH, dW, dH):
    X = tf.layers.batch_normalization(X)
    X = BinActiveZ(X)
    X = tf.layers.conv2d(X, dOut, (kW, kH), (dW, dH), padding='same', activation=tf.nn.relu)
    return X


def BinMaxConvolution(X, dOut, kW, kH, dW, dH, mW, mH):
    X = tf.layers.batch_normalization(X)
    X = BinActiveZ(X)
    X = tf.layers.conv2d(X, dOut, (kW, kH), (dW, dH), padding='same', activation=tf.nn.relu)
    X = tf.layers.max_pooling2d(X, (mW, mH), 2, padding='same')
    return X

def mnist_xnor(X):
    X = tf.reshape(X, [-1,28,28,1])
    X = tf.layers.conv2d(X, 32, 5, 1, 'same')
    X = tf.layers.batch_normalization(X)
    X = tf.nn.relu(X)
    X = tf.layers.max_pooling2d(X, 2, 2, 'same')

    X = BinMaxConvolution(X, 64, 5, 5, 1, 1, 2, 2)
    X = BinConvolution(X, 128, 3, 3, 1, 1)
    X = BinConvolution(X, 128, 3, 3, 1, 1)
    X = BinMaxConvolution(X, 64, 3, 3, 1, 1, 1, 1)
    X = BinConvolution(X, 512, 3, 3, 1, 1)
    X = BinConvolution(X, 512, 3, 3, 1, 1)
    X = tf.layers.average_pooling2d(X, 4,1)

    X = tf.layers.batch_normalization(X)
    X = tf.nn.relu(X)
    X = tf.layers.conv2d(X, config.num_classes, 1, 1,activation=tf.nn.relu)
    X = tf.reshape(X, [-1, config.num_classes])
    X = tf.nn.softmax(X)
    return X


def mnist_norm(X):
    X = tf.reshape(X, [-1,28,28,1])
    X = tf.layers.conv2d(X, 32, 5, 1, 'same')
    X = tf.layers.batch_normalization(X)
    X = tf.nn.relu(X)
    X = tf.layers.max_pooling2d(X, 2, 2, 'same')

    X = tf.layers.conv2d(X, 64, 3, 1, 'same')
    X = tf.layers.batch_normalization(X)
    X = tf.nn.relu(X)
    X = tf.layers.max_pooling2d(X, 2, 2, 'same')

    X = tf.layers.conv2d(X, 64, 3, 1, 'same')
    X = tf.layers.conv2d(X, 128, 3, 1, 'same')
    X = tf.layers.conv2d(X, 128, 3, 1, 'same')

    X = tf.layers.conv2d(X, 512, 3, 1, 'same')
    X = tf.layers.average_pooling2d(X, 4,1)


    X = tf.layers.batch_normalization(X)
    X = tf.nn.relu(X)
    X = tf.layers.conv2d(X, config.num_classes, 1, 1,activation=tf.nn.relu)
    X = tf.reshape(X, [-1, config.num_classes])
    X = tf.nn.softmax(X)
    return X
