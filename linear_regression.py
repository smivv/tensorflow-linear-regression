import sys
import numpy as np
import tensorflow as tf
from random import randint


# Show debugging output
tf.logging.set_verbosity(tf.logging.INFO)

""" ---------------------------------- Flags ---------------------------------- """

tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')

tf.app.flags.DEFINE_integer('display_step', 100, 'Display logs per step.')

tf.app.flags.DEFINE_integer('max_steps', 100, 'Maximum steps.')

tf.app.flags.DEFINE_integer('N', 5, 'Number of rows.')

tf.app.flags.DEFINE_integer('K', 2, 'Number of columns')

FLAGS = tf.app.flags.FLAGS


def run_training(train_x, train_y):

    #   Y   =     W  *  X   +   b
    # [1*K] =   [1*N]*[N*K] + [1*K]

    X = tf.placeholder(tf.float32, [FLAGS.N, FLAGS.K])
    Y = tf.placeholder(tf.float32, [1, FLAGS.K])

    # weights
    W = tf.Variable(tf.zeros([1, FLAGS.N], dtype=np.float32), name="weight")
    b = tf.Variable(tf.zeros([1, FLAGS.K], dtype=np.float32), name="bias")

    # linear model
    Y_ = tf.add(tf.matmul(W, X), b)

    cost = tf.reduce_sum(tf.square(Y_ - Y)) / (2 * FLAGS.max_steps)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(FLAGS.max_steps):

            sess.run(optimizer, feed_dict={X: np.asarray(train_x[step]), Y: np.asarray(train_y[step])})

            if step % FLAGS.display_step == 0:
                c = sess.run(cost, feed_dict={X: np.asarray(train_x[step]), Y: np.asarray(train_y[step])})
                tf.logging.info("Step: %d, Cost: %f" % (step+1, c))
                tf.logging.debug("W = ", sess.run(W))
                tf.logging.debug("b = ", sess.run(b))

        tf.logging.info("Optimization Finished!")

        tf.logging.info("Random prediction:")

        predict_x = tf.cast(train_x[FLAGS.max_steps // 2], dtype=tf.float32)

        tf.logging.debug("X =", sess.run(predict_x))

        predict_x = (predict_x - mean) / std

        predict_y = tf.add(tf.matmul(W, predict_x), b)

        cost = tf.reduce_mean(tf.squared_difference(predict_y, train_y))

        tf.logging.debug("W = ", sess.run(W))

        tf.logging.debug("b = ", sess.run(b))

        tf.logging.debug("Y =", sess.run(predict_y))

        tf.logging.info("Cost: %f" % sess.run(cost))


def read_data():

    train_x = np.random.rand(FLAGS.max_steps, FLAGS.N, FLAGS.K)
    train_y = np.random.rand(FLAGS.max_steps, 1, FLAGS.K)

    return train_x, train_y


def feature_normalize(train_x):
    global mean, std
    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)
    return (train_x - mean) / std


def main(_):
    
    train_x, train_y = read_data()
    train_x = feature_normalize(train_x)
    run_training(train_x, train_y)


if __name__ == '__main__':

    tf.logging.info('Processing started..')

    tf.app.run()
