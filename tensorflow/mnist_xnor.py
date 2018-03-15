import numpy as np
import tensorflow as tf

import config
import nets


def train():
    with tf.Session() as sess:
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images  # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images  # Returns np.array
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
        input_batch = tf.placeholder(tf.float32, [config.batch_size, 28 * 28])
        label_batch = tf.placeholder(tf.int32, [config.batch_size])
        logit_batch = nets.mnist_xnor(input_batch)
        #logit_batch = nets.mnist_norm(input_batch)
        #logit_batch = nets.jamie_xnor(input_batch)
        error = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(label_batch, 10), logits=logit_batch)
        optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
        train_op = optimizer.minimize(error)
        prediction = tf.argmax(logit_batch, axis=1)
        correct_prediction = tf.equal(prediction, tf.cast(label_batch, tf.int64))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for e in range(config.num_epochs):
            print("Staring Epoch: ",e)
            for n in range(train_labels.size // config.batch_size):
                _= sess.run([train_op], feed_dict={
                    input_batch: train_data[n * config.batch_size:(n + 1) * config.batch_size, :],
                    label_batch: train_labels[n * config.batch_size:(n + 1) * config.batch_size]})
                if n%50==0:
                    print("batch: ", n)
                acc = np.zeros([eval_labels.size//config.batch_size])
                if n%300==0:
                    for j in range(eval_labels.size//config.batch_size):
                        a = sess.run([accuracy], feed_dict={
                              input_batch: eval_data[j * config.batch_size:(j + 1) * config.batch_size, :],
                            label_batch: eval_labels[j * config.batch_size:(j + 1) * config.batch_size]})
                        acc[j] = a[0]
                    print("mean test acc", np.mean(acc))


if __name__ == '__main__':
    train()
