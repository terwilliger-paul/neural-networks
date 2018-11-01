import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=[None, 1])
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

x_cpu = np.array([[1], [2], [3], [4]], dtype=np.float32)
sess = tf.Session()
sess.run(init)
for i in range(100):
  _, loss_value = sess.run((train, loss), feed_dict={x: x_cpu})
  print(loss_value)

print(sess.run(y_pred, feed_dict={x: x_cpu}))

new_x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
new_y_pred = linear_model(new_x)

print('new')
print(sess.run(new_y_pred))

'''
def foo(name):
    with tf.variable_scope(name):
        x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
        y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
        linear_model = tf.layers.Dense(units=1)

        y_pred = linear_model(x)
        loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)

    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return train, y_pred, loss, params

old_train, old_y_pred, old_loss, old_params = foo("old")
train, y_pred, loss, params = foo("current")

to_update = [oldp.assign(p) for p, oldp in zip(params, old_params)]
print(to_update)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
sess.run(to_update)


print(sess.run(y_pred))
print(sess.run(old_y_pred))
_, loss_value = sess.run((train, loss))
print()
print(sess.run(y_pred))
print(sess.run(old_y_pred))

sess.run(to_update)
_, loss_value = sess.run((train, loss))
print()
print(sess.run(y_pred))
print(sess.run(old_y_pred))
'''

'''
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

print(sess.run(y_pred))
'''
