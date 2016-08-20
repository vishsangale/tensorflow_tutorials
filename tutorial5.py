import pandas as pd
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 28
NUMBER_OF_LABELS = 10
LEARNING_RATE = 1e-4
TRAINING_ITERATIONS = 20000
DROPOUT = 0.27
BATCH_SIZE = 50
VALIDATION_SIZE = 2000
data = pd.read_csv('./data/train.csv')

images = data.iloc[:, 1:].values
training_images = images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
training_images = np.multiply(training_images, 1.0 / 255.0)
labels = data[[0]].values.ravel()


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# print labels.shape[0]
training_labels = dense_to_one_hot(labels, np.unique(labels).shape[0])
training_labels = training_labels.astype(np.uint8)

validation_images = training_images[:VALIDATION_SIZE]
validation_labels = training_labels[:VALIDATION_SIZE]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder('float', shape=[None, IMAGE_SIZE * IMAGE_SIZE])

y_true = tf.placeholder('float', shape=[None, NUMBER_OF_LABELS])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, NUMBER_OF_LABELS])
b_fc2 = bias_variable([NUMBER_OF_LABELS])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_conv), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs_completed = 0
index_in_epoch = 0
num_examples = training_images.shape[0]

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)


def next_batch(batch_size):
    global training_images
    global training_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        epochs_completed += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        training_images = training_images[perm]
        training_labels = training_labels[perm]
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return training_images[start:end], training_labels[start:end]


train_accuracies = []
validation_accuracies = []

display_step = 1

for i in range(TRAINING_ITERATIONS):
    batch_xs, batch_ys = next_batch(BATCH_SIZE)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 1.0})
        validation_accuracy = accuracy.eval(feed_dict={x: validation_images, y_true: validation_labels,
                                                       keep_prob: 1.0})
        print('training_accuracy / validation_accuracy => %.4f/%.4f  for step %d' % (train_accuracy,
                                                                                     validation_accuracy, i))
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
    sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: DROPOUT})


test_images = pd.read_csv('./data/test.csv').values
test_images = test_images.astype(np.float)

test_images = np.multiply(test_images, 1.0 / 255.0)
predict = tf.argmax(y_conv, 1)

predicted_labels = np.zeros(test_images.shape[0])
for i in range(0, test_images.shape[0] // BATCH_SIZE):
    predicted_labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i * BATCH_SIZE:
    (i + 1) * BATCH_SIZE], keep_prob: 1.0})

np.savetxt('submission.csv', np.c_[range(1, len(test_images) + 1), predicted_labels], delimiter=',',
           header='ImageId,Label', comments='', fmt='%d')
