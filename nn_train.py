import tensorflow as tf
import batch_generator as gen

batch_size = 100
learning_rate = 0.001
nb_epochs = 1000

# Placeholders
x = tf.placeholder(tf.float32, [None, 784], 'X')
y_ = tf.placeholder(tf.float32, [None, 10], 'Y_')

# Variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#Initializing the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#Defining the cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))

#Determining the accuracy of the parameters
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Training Operation
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

saver = tf.train.Saver(max_to_keep=1000)
with tf.Session() as sess:
    version = 0
    sess.run(tf.global_variables_initializer())
    for batch_x, batch_y, epoch in gen.batch_sequencer(nb_epochs, batch_size):
        sess.run([train_op], feed_dict={x: batch_x, y_: batch_y})

        #Print accuracy of the model
        if epoch % 2 == 0:
            print("Epoch: ", epoch)
            print("Accuracy: ", accuracy.eval(feed_dict={x: gen.mnist.test.images, y_: gen.mnist.test.labels}))
            print("Model Execution Complete")

        #Save checkpoints
        if epoch % 100 == 0:
            saved_file = saver.save(sess, 'checkpoints/nn_train_' + str(version))
            version += 1
            print("Saved file: " + saved_file)