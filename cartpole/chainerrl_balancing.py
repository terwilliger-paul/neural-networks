import numpy as np
import cupy as cp
import chainer
from chainer import cuda, Function, gradient_check
from chainer import report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import time
import chainerrl
from chainerrl.experiments import LinearInterpolationHook
import gym

TOTAL_STEPS = 40000

class QFunction(chainer.Chain):

    def __init__(self, obs_size, n_actions, n_hidden_channels=256):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))


env = gym.make('CartPole-v1')
'''
print('observation space:', env.observation_space)
print('action space:', env.action_space)

obs = env.reset()
env.render()
print('initial observation:', obs)

action = env.action_space.sample()
obs, r, done, info = env.step(action)
print('next observation:', obs)
print('reward:', r)
print('done:', done)
print('info:', info)
'''

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)

# Uncomment to use CUDA
q_func.to_gpu(0)

# Set up decaying learning rate
def lr_setter(env, agent, value):
    agent.optimizer.lr = value

decay_hook = LinearInterpolationHook(TOTAL_STEPS, 1e-2, 1e-6, lr_setter)

# Use Adam to optimize q_func. eps=1e-2 is for stability.
#optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer = chainer.optimizers.SMORMS3(eps=1e-2)
optimizer.setup(q_func)
#optimizer.add_hook(decay_hook)
#optimizer.add_hook(chainer.optimizer.Lasso(.0001))
#optimizer.add_hook(chainer.optimizer.WeightDecay(.0001))

# Set the discount factor that discounts future rewards.
gamma = .95

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.,
    end_epsilon=.0001,
    decay_steps=TOTAL_STEPS,
    random_action_func=env.action_space.sample)
'''
explorer = chainerrl.explorers.Gaussian()
'''

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10000)

# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
#agent = chainerrl.agents.DoubleDQN(
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=32,
    update_interval=1, target_update_interval=100,
    phi=phi)

n_episodes = 40
max_episode_len = 10000
'''
for i in range(1, n_episodes + 1):
    obs = env.reset()
    reward = 0
    done = False
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while not done and t < max_episode_len:
        # Uncomment to watch the behaviour
        # env.render()
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
    if i % 10 == 0:
        print('episode:', i,
              'R:', R,
              'statistics:', agent.get_statistics())
    agent.stop_episode_and_train(obs, reward, done)
print('Finished.')
'''

# Set up the logger to print info messages for understandability.
import logging
import sys
gym.undo_logger_setup()  # Turn off gym's default logger settings
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

chainerrl.experiments.train_agent_with_evaluation(
    agent, env,
    steps=TOTAL_STEPS,           # Train the agent for 2000 steps
    eval_n_runs=10,       # 10 episodes are sampled for each evaluation
    max_episode_len=max_episode_len,  # Maximum length of each episodes
    eval_interval=5000,   # Evaluate the agent after every 1000 steps
    outdir='result',      # Save everything to 'result' directory
    step_hooks=[decay_hook],
    )

for i in range(1000):
    obs = env.reset()
    done = False
    R = 0
    t = 0
    while not done and t < max_episode_len:
        env.render()
        action = agent.act(obs)
        obs, r, done, _ = env.step(action)
        R += r
        t += 1
    print('test episode:', i, 'R:', R)
    agent.stop_episode()

# Save an agent to the 'agent' directory
agent.save('agent')

# Uncomment to load an agent from the 'agent' directory
# agent.load('agent')
