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
import gym

DTYPE = np.float32
GAMMA = .9

np.set_printoptions(precision=2)

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, new_qs):
        qs = self.predictor(x)
        loss = .5 * F.mean_squared_error(qs, new_qs)
        report({'loss': loss}, self)
        return loss

class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h = F.leaky_relu(self.l1(x))
        h = F.leaky_relu(self.l2(h))
        y = self.l3(h)
        return y

env = gym.make('CartPole-v1')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = MLP(obs_size, n_actions)
q_func.to_gpu(0)

model = Classifier(q_func)
optimizer = optimizers.SMORMS3()
optimizer.setup(model)
#optimizer.add_hook(chainer.optimizer.WeightDecay(0.001))
