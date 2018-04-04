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
GAMMA = .999
TRAINING_EPISODES = 10000
EPS = .3
REPLAY_MAX = 16348
CHAINER_DTYPE = np.float64
BATCH_SIZE = 128
MAX_REWARD = 500.
END_REWARD = 200
END_TRUE = False

np.set_printoptions(precision=4)

class QClassifier(Chain):
    def __init__(self, predictor):
        super(QClassifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = .5 * F.mean_squared_error(y, t)
        report({'loss': loss}, self)
        return loss

class SClassifier(Chain):
    def __init__(self, predictor):
        super(SClassifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        report({'loss': loss}, self)
        return loss

class MLPD(Chain):
    def __init__(self, n_units, n_out):
        super(MLPD, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l0 = L.Linear(None, n_units)  # n_in -> n_units
            self.l1 = L.Linear(None, n_units)  # n_units -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            '''
            self.l3 = L.Linear(None, n_units)  # n_units -> n_units
            self.l4 = L.Linear(None, n_units)  # n_units -> n_units
            '''
            self.l9 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):

        h = self.activation(self.l0(x))
        h = self.activation(self.l1(x))
        h = self.activation(self.l2(h))
        '''
        h = self.activation(self.l3(h))
        h = self.activation(self.l4(h))
        '''
        y = F.softmax(self.l9(h))
        return y

    def activation(self, x):
        #return F.leaky_relu(x)
        return x * F.sigmoid(x)

class MLPB(Chain):
    def __init__(self, n_units, n_out):
        super(MLPB, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l0 = L.Linear(None, n_units)  # n_in -> n_units
            self.l1 = L.Linear(None, n_units)  # n_units -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            '''
            self.l3 = L.Linear(None, n_units)  # n_units -> n_units
            self.l4 = L.Linear(None, n_units)  # n_units -> n_units
            '''
            self.l9 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):

        h = self.activation(self.l0(x))
        h = self.activation(self.l1(x))
        h = self.activation(self.l2(h))
        '''
        h = self.activation(self.l3(h))
        h = self.activation(self.l4(h))
        '''
        y = self.l9(h)
        return y

    def activation(self, x):
        #return F.leaky_relu(x)
        return x * F.sigmoid(x)

class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l0 = L.Linear(None, n_units)  # n_in -> n_units
            self.l1 = L.Linear(None, n_units)  # n_units -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            '''
            self.l3 = L.Linear(None, n_units)  # n_units -> n_units
            self.l4 = L.Linear(None, n_units)  # n_units -> n_units
            self.l5 = L.Linear(None, n_units)  # n_units -> n_units
            self.l6 = L.Linear(None, n_units)  # n_units -> n_units
            self.l7 = L.Linear(None, n_units)  # n_units -> n_units
            self.l8 = L.Linear(None, n_units)  # n_units -> n_units
            '''
            self.l9 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h = self.activation(self.l0(x))
        h = self.activation(self.l1(x))
        h = self.activation(self.l2(h))
        '''
        h = self.activation(self.l3(h))
        h = self.activation(self.l4(h))
        h = self.activation(self.l5(h))
        h = self.activation(self.l6(h))
        h = self.activation(self.l7(h))
        h = self.activation(self.l8(h))
        '''
        y = self.l9(h)
        return y

    def activation(self, x):
        #return F.leaky_relu(x)
        return x * F.sigmoid(x)

def eps_prob(q_func, env, state, n_actions, eps=.3):
    '''Not done yet'''
    choice = np.random.rand()
    if choice < eps:
        action = env.action_space.sample()
        return action
        #return np.random.randint(0, n_actions)
    else:
        o = cp.array(state[None].astype(DTYPE))
        choices = q_func(o)[0].data.get()
        softmax = (choices / np.abs(choices.sum()))
        if np.any(softmax < 0):
            softmax += 1
        print(choices, softmax)
        output = np.argmax(q_func(o)[0].data.get())
        return output


def eps_greedy(q_func, env, state, n_actions, eps=.3):
    choice = np.random.rand()
    if choice < eps:
        action = env.action_space.sample()
        return action
        #return np.random.randint(0, n_actions)
    else:
        o = cp.array(state[None].astype(DTYPE))
        output = np.argmax(q_func(o)[0].data.get())
        return output

def gen_train(q_func, s0, a0, r1, s1, done):
    train = []
    gamma = GAMMA

    t = q_func(s0).data
    t1 = q_func(s1).data
    t1[done, :] = 0
    tt = r1 + (gamma * cp.max(t1, axis=1))
    t[a0[:, None] == cp.arange(t.shape[1])] = tt
    #foo[ind[:,None] == range(foo.shape[1])] = bar
    train = [(s0[i], t[i]) for i in range(t.shape[0])]
    return train

env = gym.make('Acrobot-v1')
state_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = MLP(64, n_actions)
q_func.to_gpu(0)

model = QClassifier(q_func)
optimizer = optimizers.SMORMS3(1e-2)
optimizer.setup(model)
#optimizer.add_hook(chainer.optimizer.WeightDecay(0.001))
#optimizer.add_hook(chainer.optimizer.GradientNoise(0.001))

h_func = MLPB(256, state_size)
h_model = QClassifier(h_func)
h_optimizer = optimizers.SMORMS3()
h_optimizer.setup(h_model)
#h_optimizer.add_hook(chainer.optimizer.WeightDecay(0.001))

done_func = MLPD(256, 2)
done_model = SClassifier(done_func)
done_optimizer = optimizers.SMORMS3()
done_optimizer.setup(done_model)
#done_optimizer.add_hook(chainer.optimizer.Lasso(0.001))

replay_buffer = []
s0_stack = cp.array([], dtype=cp.float32).reshape(0, state_size)
a0_stack = cp.array([], dtype=cp.int32)
r1_stack = cp.array([], dtype=cp.float32)
s1_stack = cp.array([], dtype=cp.float32).reshape(0, state_size)
done_stack = cp.array([], dtype=cp.bool_)

for i in range(1, TRAINING_EPISODES+1):
    s0 = env.reset()
    s0 = cp.array(s0, dtype=DTYPE)
    r0 = 0
    done = False
    while not done:
        #env.render()
        a0 = eps_greedy(q_func, env, s0, n_actions, eps=EPS)
        s1, reward, done, _ = env.step(a0)
        r1 = r0 + (reward / MAX_REWARD)
        if END_TRUE and done and r1 > -1:
            r1 += END_REWARD
            print('----------------------------------victory')

        s1 = cp.array(s1, dtype=cp.float32)

        s0_stack = cp.vstack([s0_stack, s0[None]])
        a0_stack = cp.concatenate([a0_stack,
                                   cp.array([a0], dtype=cp.int32)])
        r1_stack = cp.concatenate([r1_stack,
                                   cp.array([r1], dtype=cp.float32)])
        s1_stack = cp.vstack([s1_stack, s1[None]])
        done_stack = cp.concatenate([done_stack,
                                     cp.array([done], dtype=cp.bool_)])

        r0 = r1
        s0 = s1

    print(
          'steps', s0_stack.shape[0],
          'episode:', i,
          'r:', r0,
          )
    if i % 5 == 0:
        # Train observer
        h_x = cp.hstack([s0_stack,
                         a0_stack[None].T,
                         #r1_stack[None].T,
                         ]).astype(cp.float32)
        h_y = cp.hstack([s1_stack,
                         #done_stack[None].T,
                         ]).astype(cp.float32)
        h_train = [(h_x[i], h_y[i]) for i in range(h_x.shape[0])]
        train_iter = iterators.SerialIterator(h_train, batch_size=BATCH_SIZE,
                                              shuffle=True)
        updater = training.StandardUpdater(train_iter, h_optimizer, device=0)
        trainer = training.Trainer(updater, (30, 'epoch'), out='result')
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(['epoch',
                                               'main/loss',
                                               'elapsed_time',
                                               ]))
        trainer.extend(extensions.ProgressBar())
        trainer.run()

        d_x = cp.hstack([s1_stack,
                         r1_stack[None].T,
                         ]).astype(cp.float32)
        d_y = cp.hstack([
                         done_stack,
                         #(-1)*(done_stack[None].T.astype(cp.float32) - 1),
                         ]).astype(cp.int32)
        d_train = [(d_x[i], d_y[i]) for i in range(d_x.shape[0])]
        train_iter = iterators.SerialIterator(d_train, batch_size=BATCH_SIZE,
                                              shuffle=True)
        updater = training.StandardUpdater(train_iter, done_optimizer, device=0)
        trainer = training.Trainer(updater, (30, 'epoch'), out='result')
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(['epoch',
                                               'main/loss',
                                               'elapsed_time',
                                               'observe_lr']))
        trainer.extend(extensions.ProgressBar())
        trainer.run()

        s0 = env.reset()
        s0 = cp.array(s0, dtype=DTYPE)
        r0 = 0
        done = False
        while not done:
            env.render()
            a0 = eps_greedy(q_func, env, s0, n_actions, eps=0)
            s1, reward, done, _ = env.step(a0)
            r1 = r0 + (reward / MAX_REWARD)
            if END_TRUE and done and r1 > -1:
                r1 += END_REWARD
                print('----------------------------------victory')
            print("X: ", s0, a0, r0)
            print("Y: ", s1, done)
            s0_cp = cp.array(s0)[None].astype(cp.float32)
            s1_cp = cp.array(s1)[None].astype(cp.float32)
            a0_cp = cp.array([[a0]]).astype(cp.float32)
            r1_cp = cp.array([[r1]]).astype(cp.float32)
            output = h_func(cp.hstack([s0_cp,
                                       a0_cp,
                                       ]).astype(cp.float32)).data.get()[0]
            is_done = done_func(cp.hstack([
                                           s1_cp,
                                           r1_cp,
                                           ]).astype(cp.float32)).data.get()[0]
            print("G: ", output, is_done)
            print("")

            s1 = cp.array(s1, dtype=cp.float32)

            s0_stack = cp.vstack([s0_stack, s0[None]])
            a0_stack = cp.concatenate([a0_stack,
                                       cp.array([a0], dtype=cp.int32)])
            r1_stack = cp.concatenate([r1_stack,
                                       cp.array([r1], dtype=cp.float32)])
            s1_stack = cp.vstack([s1_stack, s1[None]])
            done_stack = cp.concatenate([done_stack,
                                         cp.array([done], dtype=cp.bool_)])

            r0 = r1
            s0 = s1

        print(
              'steps', s0_stack.shape[0],
              'episode:', i,
              'R:', r0, "----------test----------", r0)

    s0_stack = s0_stack[-REPLAY_MAX:, :]
    a0_stack = a0_stack[-REPLAY_MAX:]
    r1_stack = r1_stack[-REPLAY_MAX:]
    s1_stack = s1_stack[-REPLAY_MAX:, :]
    done_stack = done_stack[-REPLAY_MAX:]

    if a0_stack.shape[0] > BATCH_SIZE:
        for _ in range(1):
            train = gen_train(q_func,
                              s0_stack,
                              a0_stack,
                              r1_stack,
                              s1_stack,
                              done_stack)
            train_iter = iterators.SerialIterator(train, batch_size=BATCH_SIZE,
                                                  shuffle=True)
            updater = training.StandardUpdater(train_iter, optimizer, device=0)
            trainer = training.Trainer(updater, (1, 'epoch'), out='result')
            '''
            trainer.extend(extensions.LogReport())
            trainer.extend(extensions.PrintReport(['epoch',
                                                   'main/loss',
                                                   'elapsed_time',
                                                   'observe_lr']))
            trainer.extend(extensions.ProgressBar())
            '''
            trainer.run()
