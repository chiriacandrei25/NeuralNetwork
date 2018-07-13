from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("model_data/", one_hot=True)

def batch_sequencer(nb_epochs, batch_size):
    for epoch in range(nb_epochs):
        nb_batches = int(mnist.train.num_examples/batch_size)
        for batch in range(nb_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            yield batch_x, batch_y, epoch
