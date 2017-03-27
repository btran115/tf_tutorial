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
#compute accuracy on test data
print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))

#initialize weights and bias with noise
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

#we implement convolution and pooling here
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding = 'SAME')
#here ksize specifices size of window for each dimension of the input
#middle two dimensions are x and y axis of image
#stride is similar
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

#first layer consisting of convolution followed by max pooling. 32 features for each 
#5x5 patch. first two dims are patch size, then num input channels, then num output
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

#reshape x so it fits the pool functions
#-1 is used so that total size remains constant
x_image = tf.reshape(x, [-1, 28, 28, 1])
# compute middle layers by first convolution and then pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


