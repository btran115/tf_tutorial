from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

import tensorflow as tf
# none allows for any number of images
x = tf.placeholder(tf.float32, [None, 784])
# 10 outputs for each digit 0-9
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#network output
y = tf.nn.softmax(tf.matmul(x,W)+b)
#target labels
y_ = tf.placeholder(tf.float32, [None, 10])
#definng our loss function
#reduce_mean takes mean over batches, reduce_sum sums over second col of y
#recall y here is number of inputs, 10 slots for each output
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#picks step according to grad desc using cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
#initializer variables (to 0)
tf.global_variables_initializer().run()
#take minibatch of samples, compute step, move, and repeat
for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict = {x:batch_xs, y_: batch_ys})
#argmax to pull estimated label for each y, gives boolean of whether we match
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy)
print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))

