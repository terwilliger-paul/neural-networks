from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import time

EPOCHS = 20
BATCH_SIZE = 1024
NODES = 512
ADAM = 0.0002

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.reshape(60000, 28*28) / 255.0, x_test.reshape(10000, 28*28) / 255.0

size = NODES

'''
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(size, activation=tf.nn.relu),
  tf.keras.layers.Dense(size, activation=tf.nn.relu),
  tf.keras.layers.Dense(size, activation=tf.nn.relu),
  tf.keras.layers.Dense(size, activation=tf.nn.relu),
  tf.keras.layers.Dense(size, activation=tf.nn.relu),
  tf.keras.layers.Dense(size, activation=tf.nn.relu),
  tf.keras.layers.Dense(size, activation=tf.nn.relu),
  tf.keras.layers.Dense(size, activation=tf.nn.relu),
  tf.keras.layers.Dense(size, activation=tf.nn.relu),
  tf.keras.layers.Dense(size, activation=tf.nn.relu),
  tf.keras.layers.Dense(size, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(ADAM),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
model.evaluate(x_test, y_test)
'''

# Declare constants

vector_placeholder = tf.placeholder(tf.float32, shape=[None, 28*28])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices((vector_placeholder, labels_placeholder))
dataset = dataset.shuffle(100000).batch(BATCH_SIZE)
iter = dataset.make_initializable_iterator()
input_vector, input_labels = iter.get_next()

h = tf.layers.Dense(NODES, activation=tf.nn.relu)(input_vector)
hx = tf.layers.Dense(NODES, activation=tf.nn.relu)(h)
h = tf.layers.Dense(NODES, activation=tf.nn.relu)(hx)
hx = tf.layers.Dense(NODES, activation=tf.nn.relu)(h) + hx
h = tf.layers.Dense(NODES, activation=tf.nn.relu)(hx)
hx = tf.layers.Dense(NODES, activation=tf.nn.relu)(h) + hx
h = tf.layers.Dense(NODES, activation=tf.nn.relu)(hx)
hx = tf.layers.Dense(NODES, activation=tf.nn.relu)(h) + hx
h = tf.layers.Dense(NODES, activation=tf.nn.relu)(hx)
hx = tf.layers.Dense(NODES, activation=tf.nn.relu)(h) + hx
h = tf.layers.Dense(NODES, activation=tf.nn.relu)(hx)
hx = tf.layers.Dense(NODES, activation=tf.nn.relu)(h) + hx
output_vector = tf.layers.Dense(10)(hx)

loss = tf.losses.sparse_softmax_cross_entropy(labels=input_labels, logits=output_vector)
train = tf.train.AdamOptimizer(ADAM).minimize(loss)
accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(output_vector, axis=1), input_labels), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        sess.run(iter.initializer, feed_dict={vector_placeholder: x_train,
                                              labels_placeholder: y_train})
        start = time.time()
        loss_list = []
        while True:
            try:
                _, loss_output = sess.run([train, loss])
                loss_list.append(loss_output)
            except tf.errors.OutOfRangeError:
                break
        end = time.time() - start
        sess.run(iter.initializer, feed_dict={vector_placeholder: x_test,
                                              labels_placeholder: y_test})
        accuracy_value = 0
        while True:
            try:
                accuracy_value += sess.run(accuracy)
            except tf.errors.OutOfRangeError:
                break
        loss_value = np.mean(loss_list)
        print(f"Iter: {i}, Loss: {loss_value}, Accuracy: {accuracy_value/10000}, Time: {end}")

#########################

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28*28])

    dense = tf.layers.dense(inputs=input_layer, units=NODES, activation=tf.nn.relu)
    dense = tf.layers.dense(inputs=dense, units=NODES, activation=tf.nn.relu)

    # Logits Layer
    logits = tf.layers.dense(inputs=dense, units=10)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(ADAM)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
            )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn)

    # Set up logging for predictions
    """
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=49)
    """

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        #steps=20000,
        #hooks=[logging_hook]
        )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
        )
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()
