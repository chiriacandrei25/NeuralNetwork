import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
LAYER_SIZE = 500
CLASS_SIZE = 10
BATCH_SIZE = 100
PHOTO_SIZE = 784
learning_rate = 0.001
NUM_OF_EPOCHS = 10

# Building the model graph
with tf.name_scope("Data") as scope:
    X = tf.placeholder(tf.float32, [None, PHOTO_SIZE], name="X")
    Y = tf.placeholder(tf.float32, [None, CLASS_SIZE], name="Y")

with tf.name_scope("Graph") as scope:
    hidden_layer1 = {"Weights": tf.Variable(tf.random_normal([PHOTO_SIZE, LAYER_SIZE])),
                     "Biases": tf.Variable(tf.random_normal([LAYER_SIZE]))}

    hidden_layer2 = {"Weights": tf.Variable(tf.random_normal([LAYER_SIZE, LAYER_SIZE])),
                     "Biases": tf.Variable(tf.random_normal([LAYER_SIZE]))}

    hidden_layer3 = {"Weights": tf.Variable(tf.random_normal([LAYER_SIZE, LAYER_SIZE])),
                     "Biases": tf.Variable(tf.random_normal([LAYER_SIZE]))}

    output_layer = {"Weights": tf.Variable(tf.random_normal([LAYER_SIZE, CLASS_SIZE])),
                    "Biases": tf.Variable(tf.random_normal([CLASS_SIZE]))}

    l1 = tf.nn.relu(tf.add(tf.matmul(X, hidden_layer1['Weights']), hidden_layer1['Biases']))
    l2 = tf.nn.relu(tf.add(tf.matmul(l1, hidden_layer2['Weights']), hidden_layer2['Biases']))
    l3 = tf.nn.relu(tf.add(tf.matmul(l2, hidden_layer3['Weights']), hidden_layer3['Biases']))

    output = tf.add(tf.matmul(l3, output_layer['Weights']), output_layer['Biases'])

with tf.name_scope("Optimizer") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(NUM_OF_EPOCHS):
        epoch_cost = 0
        num_of_batches = int(mnist.train.num_examples/BATCH_SIZE)
        for batch in range(num_of_batches):
            data_x, data_y = mnist.train.next_batch(BATCH_SIZE)
            _, batch_cost = sess.run([optimizer, cost], feed_dict={X: data_x, Y: data_y})
            epoch_cost += batch_cost
        epoch_cost /= num_of_batches
        print("Epoch", epoch, "finished with cost:", epoch_cost)

    prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print("Accuracy:", accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

