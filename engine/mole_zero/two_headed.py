import tensorflow as tf
import numpy as np
import gym
import time

TRUNK_NODES = 32
EPSILON = .2 # Clip value
ENTROPY_COEF = 0.01
VALUE_COEF = 1.
ADAM = 0.0001
GAMMA = 0.99

def two_headed_model_fn(features, labels, mode):
    """Model for mole

    labels should have three possible keys:
        policy,
        value,
        old_policy,
    """

    # Trunk of the hydra
    trunk = tf.layers.dense(inputs=features, units=32, activation=tf.nn.relu)
    trunk = tf.layers.dense(inputs=trunk, units=32, activation=tf.nn.relu)
    trunk = tf.layers.dense(inputs=trunk, units=32, activation=tf.nn.relu)

    # Policy head
    policy = tf.layers.dense(inputs=trunk, units=32, activation=tf.nn.relu)
    policy = tf.layers.dense(inputs=policy, units=32, activation=tf.nn.relu)
    policy = tf.layers.dense(inputs=policy, units=32, activation=tf.nn.relu)
    policy = tf.layers.dense(inputs=policy, units=2, activation=tf.nn.softmax)

    # Value head
    value = tf.layers.dense(inputs=trunk, units=32, activation=tf.nn.relu)
    value = tf.layers.dense(inputs=value, units=32, activation=tf.nn.relu)
    value = tf.layers.dense(inputs=value, units=32, activation=tf.nn.relu)
    value = tf.layers.dense(inputs=value, units=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"policy": policy, "value": value}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Define constants
    old_policy = labels["old_policy"]
    target_value = labels["target_value"]
    advantage = target_value - value

    # Calculate policy loss
    ratio = tf.div(policy, old_policy)
    clipped_ratio = tf.clip_by_value(ratio,
                                     clip_value_min = 1-EPSILON,
                                     clip_value_max = 1+EPSILON)
    inside_E_t = tf.minimum(tf.multiply(advantage, ratio),
                            tf.multiply(advantage, clipped_ratio))
    policy_loss = -tf.reduce_mean(inside_E_t)

    # Calculate value loss
    squared_diff = tf.squared_difference(value, target_value)
    value_loss = tf.reduce_mean(squared_diff)

    # Calculate entropy?
    entropy = tf.reduce_sum(policy - tf.log(policy))

    # Calculate loss
    loss = (VALUE_COEF*value_loss) + policy_loss + (ENTROPY_COEF*entropy)

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdamOptimizer(ADAM)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
            )
        return tf.estimator.EstimatorSpec(
                            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={})

ppo = tf.estimator.Estimator(model_fn=two_headed_model_fn)
test = np.random.randn(2, 4)

predict_fn = tf.estimator.inputs.numpy_input_fn(
    x=test,
    y=None,
    batch_size=2,
    num_epochs=1,
    shuffle=False,
    )
print(list(ppo.predict(predict_fn)))
print(type(ppo))

foo = ppo.replicate_model_fn()
predict_fn = tf.estimator.inputs.numpy_input_fn(
    x=test,
    y=None,
    batch_size=2,
    num_epochs=1,
    shuffle=False,
    )
print(type(foo))
print(list(foo.predict(predict_fn)))


'''
env = gym.make('CartPole-v0')

print(env.action_space)
print(env.action_space.shape)
#> Discrete(2)
print(env.observation_space)
print(env.observation_space.shape)
#> Box(4,)

for i_episode in range(20):
    state = env.reset()

    done = False
    states = []
    rewards = []
    values = []
    while not done:
        env.render()

        # Neural network goes here and plays 1 or 0
        predict_fn = tf.estimator.inputs.numpy_input_fn(
            x=state, batch_size=1, num_epochs=1, shuffle=False)
        evaluation = next(ppo.predict(predict_fn))
        policy = evaluation['policy']
        value = evaluation['value']
        action = np.random.choice(np.arange(2), p=policy)

        # Make move
        state, reward, done, info = env.step(action)

        # Collect environment states
        states.append(state)
        rewards.append(reward)
        values.append(value)

    # Generate target values
    target_values = []
    for i in range(len(states)):
        if i+1 < len(states):
            next_value = values[i+1]
            next_rewards = np.array(rewards[i+1:])
            delays = np.array([GAMMA**(j+1) for j in range(len(next_rewards))])
            target_value = next_value + np.sum(next_rewards*delays)
        else:
            target_value = np.sum(rewards)
        target_values.append(target_value)

    # Generate old policy values

    #print("Episode finished after {} timesteps".format(t+1))
'''
