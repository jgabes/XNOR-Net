import tensorflow as tf


@tf.RegisterGradient("CustomClipGrad")
def _clip_grad(op, grad):
  return tf.clip_by_value(grad, -1, 1)

def BinActiveZ(X):
    g=tf.get_default_graph()
    with g.gradient_override_map({"BAZ": "CustomClipGrad"}):
        output = tf.sign(X, name="BAZ")
        return output

def BinConvolution(X, dOut, kW, kH, dW, dH):
     X=tf.layers.batch_normalization(X)
     X=BinActiveZ(X)
     X=tf.layers.conv2d(X, dOut, (kW, kH), (dW, dH), padding = 'same')
     return X

def BinMaxConvolution(X, dOut,kW, kH, dW, dH, mW, mH):
    X = tf.layers.batch_normalization(X)
    X = BinActiveZ(X)
    X = tf.layers.conv2d(X, dOut, (kW, kH), (dW, dH), padding='same')
    X = tf.layers.max_pooling2d(X,(mW, mH), 2, padding='same')