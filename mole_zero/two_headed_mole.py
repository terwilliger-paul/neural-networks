import tensorflow as tf
import numpy as np
import gym
import time

np.set_printoptions(precision=5)

EPSILON = .2 # Clip value
ENTROPY_COEF = .1
VALUE_COEF = 1.
ADAM = 1e-4
GAMMA = 0.95
EPOCHS = 10
BATCH_SIZE = 2048
ER_SIZE = 100000

class PPO:

    def __init__(self):

        # Declare constants
        self.PO = 2#1792 # length of policy vector
        STATE_NUM = 4#781 # length of input vector
        self.VN = 32 # Value nodes
        self.TN = 32 # Trunk nodes
        self.PN = 32 # Policy nodes
        self.TP = 1. # Temperature

        # Declare the neural net
        self.state = tf.placeholder(tf.float32, shape=[None, STATE_NUM])
        self.action = tf.placeholder(tf.int32, shape=[None])
        self.advantages = tf.placeholder(tf.float32, shape=[None])
        self.target_value = tf.placeholder(tf.float32, shape=[None])
        self.old_policy_v = tf.placeholder(tf.float32, shape=[None, self.PO])

        # Declare layers for neural net
        self.t1 = None
        self.t2 = None
        self.t3 = None

        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.p4 = None

        self.v1 = tf.layers.Dense(self.VN, tf.nn.relu)
        self.v2 = tf.layers.Dense(self.VN, tf.nn.relu)
        self.v3 = tf.layers.Dense(self.VN, tf.nn.relu)
        self.v4 = tf.layers.Dense(1, tf.nn.relu)

        # Declare trunk and policy nets
        old_params, old_trunk, old_policy = self._build_net("old_policy")
        params, trunk, policy = self._build_net("policy")
        to_update = [old_p.assign(p) for p, old_p in zip(params, old_params)]

        # Value head
        value = self.v4(self.v3(self.v2(self.v1(trunk))))

        # Calculate loss
        #advantage = self.target_value - value
        advantage = self.advantages

        # Calculate policy loss
        action_hot = tf.one_hot(self.action, self.PO)
        policy_a = tf.reduce_sum(policy*action_hot, axis=1)
        old_policy_a = tf.reduce_sum(self.old_policy_v*action_hot, axis=1)
        clip_old_policy_a = tf.clip_by_value(old_policy_a, 1e-10, 1.)
        ratio = tf.div(policy_a, tf.clip_by_value(old_policy_a, 1e-10, 1.))
        clipped_ratio = tf.clip_by_value(ratio,
                                         clip_value_min = 1-EPSILON,
                                         clip_value_max = 1+EPSILON)
        inside_E_t = tf.minimum(tf.multiply(advantage, ratio),
                                tf.multiply(advantage, clipped_ratio))
        policy_loss = -tf.reduce_mean(inside_E_t)

        # Calculate value loss
        squared_diff = tf.squared_difference(value, self.target_value)
        value_loss = tf.reduce_mean(squared_diff)

        # Calculate entropy?
        clip_policy = tf.clip_by_value(policy, 1e-10, 1.)
        to_mean = tf.reduce_sum(policy*tf.log(clip_policy), axis=1)
        entropy = tf.reduce_mean(to_mean)

        # Calculate loss
        loss = (VALUE_COEF*value_loss) + policy_loss + (ENTROPY_COEF*entropy)

        train = tf.train.AdamOptimizer(ADAM).minimize(loss)

        # Declare class-wide variables
        self.policy = policy
        self.old_policy = old_policy
        self.to_update = to_update
        self.value = value
        self.loss = loss
        self.train = train
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def self_train(self, state, action, target_value, advantages):

        # Gather old policy values
        old_policy_vector = self.sess.run(self.old_policy,
                                          feed_dict={self.state: state})

        # Update old policy
        self.sess.run(self.to_update)

        # Run self.train
        train_feed_dict = {self.action: action,
                           self.state: state,
                           self.target_value: target_value,
                           self.old_policy_v: old_policy_vector,
                           self.advantages: advantages}
        self.sess.run(self.train, feed_dict=train_feed_dict)
        loss = self.sess.run(self.loss, feed_dict=train_feed_dict)

        return loss

    def predict_old_policy(self, state):
        return self.sess.run(self.old_policy, feed_dict={self.state: state})

    def predict_policy(self, state):
        return self.sess.run(self.policy, feed_dict={self.state: state})

    def predict_value(self, state):
        return self.sess.run(self.value, feed_dict={self.state: state})

    def tensor_predict(self, inp_state):
        pass

    def _build_net(self, name):

        with tf.variable_scope(name):
            # Trunk of the hydra
            self.t1 = tf.layers.Dense(self.TN, tf.nn.relu)
            self.t2 = tf.layers.Dense(self.TN, tf.nn.relu)
            self.t3 = tf.layers.Dense(self.TN, tf.nn.relu)

            # Policy head
            self.p1 = tf.layers.Dense(self.PN, tf.nn.relu)
            self.p2 = tf.layers.Dense(self.PN, tf.nn.relu)
            self.p3 = tf.layers.Dense(self.PN, tf.nn.relu)
            self.p4 = tf.layers.Dense(self.PO, tf.nn.softmax)

            trunk = self.t3(self.t2(self.t1(self.state)))
            policy = self.p4(self.p3(self.p2(self.p1(trunk))))

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return params, trunk, policy


ppo = PPO()

'''
trials = 1
test = np.random.randn(1, 4).astype(np.float32)
start = time.time()
for _ in range(trials):
    foo = ppo.predict_policy(test)
    #bar = ppo.tensor_predict(test)
print(f"time: {time.time() - start}, {foo}, {foo.max()}")
print(foo.mean(), foo.max(), foo.std(), foo.min())
print(bar.mean(), bar.max(), bar.std(), bar.min())

print()

print(ppo.self_train(np.random.randn(20, 4).astype(np.float32),
               np.random.choice([0, 1, 2, 3], (20)).astype(np.int32),
               np.abs(np.random.randn(20).astype(np.float32)),
               np.random.randn(20)))
print("after training")
print(ppo.predict_old_policy(test))
print(ppo.predict_policy(test))
print()

print(ppo.self_train(np.random.randn(20, 4).astype(np.float32),
               np.random.choice([0, 1, 2, 3], (20)).astype(np.int32),
               np.abs(np.random.randn(20).astype(np.float32)),
               np.random.randn(20)))
print("after training")
print(ppo.predict_old_policy(test))
print(ppo.predict_policy(test))
print()

print(ppo.self_train(np.random.randn(20, 4).astype(np.float32),
               np.random.choice([0, 1, 2, 3], (20)).astype(np.int32),
               np.abs(np.random.randn(20).astype(np.float32)),
               np.random.randn(20)))
print("after training")
print(ppo.predict_old_policy(test))
print(ppo.predict_policy(test))
print()
'''

env = gym.make('CartPole-v1')

big_states = []
big_actions = []
big_rewards = []
big_done = []
trials = 0
while True:
    trials += 1
    state = env.reset()

    done = False
    states = []
    rewards = []
    actions = []
    values = []
    dones = []
    t = 0
    while not done:
        t += 1
        env.render()

        # Neural network goes here and plays 1 or 0
        policy = ppo.predict_policy(state[None])
        value = ppo.predict_value(state[None])
        action = np.random.choice(np.arange(2), p=policy[0])

        # Make move
        new_state, reward, done, info = env.step(action)

        #print(reward, end=", ")

        # Collect environment states
        states.append(state)
        actions.append(action)
        values.append(value)
        if done:
            reward = 0
        rewards.append(reward)
        dones.append(not done)
        state = new_state

    def slope(xs):
        xs = np.array(xs)
        ys = np.arange(len(xs))
        m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
             ((np.mean(xs)**2) - np.mean(xs**2)))
        return m

    print(f"{trials}. timesteps={t}, r={np.sum(rewards)}, mean={np.mean(values)}", end="")
    print(f", std={np.std(values)}, slope={slope(values)}", end="")

    '''
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float32)
    for _ in range(EPOCHS):
        predicted = ppo.predict_value(states)
        rolled = np.roll(predicted.T[0], -1)
        rolled[-1] = 0
        target_values = np.array((GAMMA*rolled) + rewards).astype(np.float32)
        advantages = target_values - predicted.T[0]
        gaes = [r_t + GAMMA * v_next - v for r_t, v_next, v
                in zip(rewards, rolled, predicted.T[0])]
        # is T-1, where T is time step which run policy
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + GAMMA * gaes[t + 1]
        gaes = np.array(gaes).astype(dtype=np.float32)
        gaes = (gaes - gaes.mean()) / gaes.std()

        shuffler = np.arange(len(states))
        np.random.shuffle(shuffler)
        loss = ppo.self_train(states[shuffler],
                              actions[shuffler],
                              target_values[shuffler],
                              gaes[shuffler])
    '''


    # Generate target values
    big_states += states
    big_actions += actions
    big_rewards += rewards
    big_done += dones

    def minibatch(mat, batches):
        return np.array_split(mat, int(mat.shape[0]/batches))

    big_states = np.array(big_states, dtype=np.float32)
    big_actions = np.array(big_actions, dtype=np.int32)
    big_rewards = np.array(big_rewards, dtype=np.float32)
    big_done = np.array(big_done, dtype=np.int32)

    if len(big_states) > BATCH_SIZE:
        for _ in range(EPOCHS):
            # Generate target values
            predicted = ppo.predict_value(big_states).T[0]
            p_values = np.roll(predicted, -1)*big_done
            big_target_values = np.array(
                        (GAMMA*p_values) + big_rewards).astype(np.float32)
            advantages = big_target_values - p_values
            # GAES IS NOT CALCULATED CORRECTLY HERE
            ###############
            gaes = [r_t + GAMMA * v_next - v for r_t, v_next, v
                    in zip(big_rewards, p_values, predicted)]
            # is T-1, where T is time step which run policy
            for t in reversed(range(len(gaes) - 1)):
                if big_done[t] == 1:
                    gaes[t] = gaes[t] + GAMMA * gaes[t + 1]
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()

            shuffler = np.arange(len(big_states))
            np.random.shuffle(shuffler)
            for i in range(len(minibatch(big_states, BATCH_SIZE))):
                loss = ppo.self_train(
                        minibatch(big_states[shuffler], BATCH_SIZE)[i],
                        minibatch(big_actions[shuffler], BATCH_SIZE)[i],
                        minibatch(big_target_values[shuffler], BATCH_SIZE)[i],
                         minibatch(gaes[shuffler], BATCH_SIZE)[i])
        print(f", loss: {loss}")

    big_states = big_states.tolist()[-ER_SIZE:]
    big_actions = big_actions.tolist()[-ER_SIZE:]
    big_rewards = big_rewards.tolist()[-ER_SIZE:]
    big_done = big_done.tolist()[-ER_SIZE:]
    '''
        big_states = []
        big_actions = []
        big_rewards = []
    '''
