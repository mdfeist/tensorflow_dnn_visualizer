import shutil
import os
import numpy as np
import scipy.misc
#import cv2
import tensorflow as tf
import dnn_visualizer.visualizer_saver as dv

# Set Globals
TMPDIR = 'tmp/'
LOGDIR = 'tmp/mnist_conv_net/'
GIST_URL = 'https://gist.githubusercontent.com/dandelionmane/4f02ab8f1451e276fea1f165a20336f1/raw/a20c87f5e1f176e9abf677b46a74c6f2581c7bd8/'

learning_rate = 1E-4
hparam = "lr_%.0E" % (learning_rate)

# Load mnist data
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(
    train_dir=TMPDIR + 'data', one_hot=True)

# Clean Log Directory
if os.path.exists(LOGDIR):
    shutil.rmtree(LOGDIR)

def images_to_sprite(data, invert_color=False):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    if invert_color:
        data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)

    return data

def conv_layer(x, size_in, size_out, name="conv"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="weights")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="biases")
    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act

def max_pool_2x2(x, name="pool"):
    with tf.name_scope(name):
        return tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME")

def fc_layer(x, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="weights")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="biases")
    act = tf.nn.relu(tf.matmul(x, w) + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act


tf.reset_default_graph()
sess = tf.Session()

# Create visualizer
dv_saver = dv.Visualizer_Saver(LOGDIR)

 # Setup placeholders, and reshape the data
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)

y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

conv1 = conv_layer(x_image, 1, 32, "conv1")
pool1 = max_pool_2x2(conv1, "pool1")
conv2 = conv_layer(pool1, 32, 64, "conv2")
pool2 = max_pool_2x2(conv2, "pool2")

flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])
fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
embedding_input = fc1
embedding_size = 1024

keep_prob = tf.placeholder(tf.float32)
fc1_drop = tf.nn.dropout(fc1, keep_prob)

logits = fc_layer(fc1_drop, 1024, 10, "logits")

# Add layers to visualizer
dv_saver.add_layer(dv.Visualizer_Layer(x_image, 'input'))
dv_saver.add_layer(dv.Visualizer_Layer(conv1, 'conv1'))
dv_saver.add_layer(dv.Visualizer_Layer(pool1, 'pool1'))
dv_saver.add_layer(dv.Visualizer_Layer(conv2, 'conv2'))
dv_saver.add_layer(dv.Visualizer_Layer(pool2, 'pool2'))
dv_saver.add_layer(dv.Visualizer_Layer(fc1, 'fc1'))
dv_saver.add_layer(dv.Visualizer_Layer(logits, 'logits'))

with tf.name_scope("xent"):
    xent = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="xent")
    tf.summary.scalar("xent", xent)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

summ = tf.summary.merge_all()

embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
assignment = embedding.assign(embedding_input)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(LOGDIR + hparam)
writer.add_graph(sess.graph)

# Create Embedding Sprite and labels
sprite = images_to_sprite(mnist.test.images[:1024].reshape(1024, 28, 28), True)
scipy.misc.imsave(os.path.join(LOGDIR, 'sprite_1024.png'), sprite)

with open(os.path.join(LOGDIR, 'labels_1024.tsv'), 'w') as f:
    for label in mnist.test.labels[:1024]:
        f.write(str(np.argmax(label)) + '\n')

config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
embedding_config = config.embeddings.add()
embedding_config.tensor_name = embedding.name
embedding_config.sprite.image_path = LOGDIR + 'sprite_1024.png'
embedding_config.metadata_path = LOGDIR + 'labels_1024.tsv'
# Specify the width and height of a single thumbnail.
embedding_config.sprite.single_image_dim.extend([28, 28])
tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

print "Starting Training ..."

for i in range(2001):
    batch = mnist.train.next_batch(100)
    if i % 10 == 0:
      [train_accuracy, s] = sess.run([accuracy, summ],
                                        feed_dict={
                                            x: batch[0],
                                            y: batch[1],
                                            keep_prob: 1.0})
      writer.add_summary(s, i)
    if i % 500 == 0:
      sess.run(assignment,
        feed_dict={
            x: mnist.test.images[:1024],
            y: mnist.test.labels[:1024],
            keep_prob: 1.0})
      saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
    sess.run(train_step,
        feed_dict={
            x: batch[0],
            y: batch[1],
            keep_prob: 0.5})

print "Save Visualization ..."

# Save Visualization of CNN
num_of_samples = 1024
dv_saver.save_network(sess)
dv_saver.save_activations(sess,
    feed_dict={
        x: mnist.test.images[:num_of_samples],
        y: mnist.test.labels[:num_of_samples],
        keep_prob: 1.0},
    x=mnist.test.images[:num_of_samples],
    y=mnist.test.labels[:num_of_samples])
