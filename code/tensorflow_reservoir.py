import tensorflow as tf
from IPython import display
from d2l import tensorflow as d2l
import time
from scipy.sparse import random, linalg
import numpy as np
import matplotlib.pyplot as plt
import sys
from numba import njit, objmode
#sys.path.insert(1, 'C:/Users/user/Dropbox/AlexanderWikner_1/UMD_Grad_School/res-noise-stabilization/helpers')
from code.lorenzrungekutta_numba import *


def input_mat_init(shape, input_weight = 1.0, seed = 1, dtype = None):
    #print(shape)
    tf.random.set_seed(seed)
    Win = np.zeros(shape)
    q = int(shape[1]//shape[0])
    for i in range(shape[0]):
        Win[i,q*i:q*(i+1)] = (np.random.rand(q)*2-1)*input_weight
    leftover = shape[1] - q*shape[0]
    for i in range(leftover):
        Win[np.random.randint(shape[0]), shape[1] - leftover + i] = (np.random.rand()*2-1)*input_weight
    return tf.convert_to_tensor(Win)

def adjacency_init(shape, spectral_radius = 0.6, avg_degree = 3, seed = 1, dtype = None):
    density = avg_degree/shape[1]
    unnormalized_W = random(shape[1], shape[0], density=density, random_state = seed)
    max_eig = linalg.eigs(unnormalized_W, k=1, return_eigenvectors = False, maxiter = 10**5)
    W = unnormalized_W * spectral_radius / np.abs(max_eig[0])
    return tf.convert_to_tensor(W.toarray())

def res_mat_init(shape, spectral_radius = 0.6, avg_degree = 3, input_weight = 0.6, seed = 1, dtype = None):
    W_shape = (shape[1], shape[1])
    W = adjacency_init(W_shape, spectral_radius, avg_degree, seed, dtype)
    Win_shape = (shape[0] - shape[1], shape[1])
    Win = input_mat_init(Win_shape, input_weight, seed, dtype)
    out = tf.concat([Win, W], 0)
    out = tf.concat([tf.eye(shape[0], shape[0] - shape[1], dtype = tf.float64), out], 1)
    return out

def input_pass_leaky_tanh(x, num_nodes, leakage = 1.0):
    return tf.concat([x[:,:-num_nodes], leaky_tanh(x[:,-num_nodes:],leakage)], 1)

def leaky_tanh(x, leakage = 1.0):
    return (1.-leakage)*x + leakage*tf.tanh(x)

class random_bias(tf.keras.initializers.Initializer):
    def __init__(self, num_nodes, minval, maxval, seed):
        super().__init__()
        self.num_nodes = num_nodes
        self.minval = minval
        self.maxval = maxval
        self.seed   = seed
        #print('initialized')
        
    def __call__(self, shape, dtype = None, **kwargs):
        return tf.concat([tf.zeros(shape[0] - self.num_nodes, dtype = dtype), tf.random.uniform([self.num_nodes],\
            minval = self.minval, maxval = self.maxval, seed = self.seed, dtype = dtype)], axis = 0)

class Reservoir(tf.keras.Model):
    def __init__(self, num_outputs, num_hidden, num_steps = 1, spectral_radius = 0.9, \
                avg_degree = 3, input_weight = 1.0, leakage = 1.0, seed = 1, out_reg = 0., dtype = None):
        super().__init__()
        self.num_steps = num_steps
        self.num_nodes = num_hidden
        self.num_outputs = num_outputs
        self.res_layer = tf.keras.layers.Dense(\
            num_hidden + num_outputs,\
            activation = lambda x: input_pass_leaky_tanh(x,self.num_nodes,leakage),\
            kernel_initializer = lambda shape, dtype: res_mat_init(\
                (shape[0], shape[1] - self.num_outputs), spectral_radius, avg_degree, input_weight, seed, dtype),\
            bias_initializer = random_bias(self.num_nodes, -input_weight, input_weight, seed),\
            trainable = False,\
            dtype = tf.float64)
        self.output_layer = tf.keras.layers.Dense(\
            num_outputs,\
            kernel_regularizer = tf.keras.regularizers.L2(out_reg),\
            bias_regularizer   = tf.keras.regularizers.L2(out_reg),\
            dtype = tf.float64)
        
    def call(self, inputs, pred_len = 1, sync_len = 1, training = None):
        if training:
            u = self.call_training(inputs)
        else:
            u = self.call_prediction(inputs, pred_len, sync_len)
        return u
    
    def call_training(self, inputs):
        input_recurrent = tf.identity(inputs)
        u_out = tf.zeros((inputs.shape[0],0), dtype = tf.float64)
        for i in range(self.num_steps):
            r = self.res_layer(input_recurrent)
            #print(r[:,:7])
            u = self.output_layer(r)
            u_out = tf.concat([u_out, u], 1)
            if i != self.num_steps - 1:
                input_recurrent = tf.concat([u, r[:,-self.num_nodes:]], 1)
        return u_out
    
    def call_prediction(self, inputs, pred_len, sync_len):
        if sync_len != 0:
            r_in = self.synchronize(inputs[:sync_len])
        else:
            r_in = tf.cast(tf.reshape(tf.identity(inputs[sync_len]), shape = (1,-1)), dtype = tf.float64)
        u_out = tf.cast(tf.reshape(tf.identity(inputs[sync_len,:-self.num_nodes]), shape = (1,-1)), dtype = tf.float64)
        for i in range(pred_len):
            r = self.res_layer(r_in)
            u = self.output_layer(r)
            u_out = tf.concat([u_out, tf.cast(u, u_out.dtype)], 0)
            if i != pred_len - 1:
                r_in = tf.reshape(tf.concat([u_out[-1], r[0,-self.num_nodes:]],0),(1,-1))
            
        return u_out
    
    def synchronize(self, input_0):
        res_out = tf.reshape(tf.identity(input_0[0,:]), (1,-1))
        #res_out = tf.zeros(shape=(1, input_0.shape[1]), dtype = tf.float64)
        for i in range(input_0.shape[0]-1):
            res_in  = tf.reshape(tf.concat([input_0[i,:-self.num_nodes], res_out[i,-self.num_nodes:]], 0), shape = (1,-1))
            r       = self.res_layer(res_in)
            r_sync  = tf.reshape(tf.concat([input_0[i+1,:-self.num_nodes], r[0,-self.num_nodes:]], 0), (1,-1))
            res_out = tf.concat([res_out, tf.cast(r_sync, res_out.dtype)], 0)
        return res_out

def evaluate_preds(Res, test_iter, loss, cutoff_vt = 0.5, cutoff_var = [0.95, 1.1], cutoff_map = 1e-2):
    metric = d2l.Accumulator(2)
    for (k, (X, y)) in enumerate(test_iter):
        y_hat     = Res(X, sync_len = 0, pred_len = X.shape[0])
        y_compare = y[:,:Res.num_outputs]
        y_rmse    = tf.math.sqrt(tf.reduce_mean(tf.math.square(y_hat[1:] - y_compare), 1))
        valid_time = 0
        for i in range(y_hat.shape[0]):
            if y_rmse[i] < cutoff_vt:
                valid_time += 1
            elif i == y_hat.shape[0] - 1:
                print('Maximum valid time achieved.')
            else:
                break
        metric.add(valid_time, 1)
        if k == len(test_iter) - 1:
            x_plot = tf.concat([tf.reshape(y_hat[1:,0], shape = (-1,1)),\
                                tf.reshape(y_compare[:,0], shape = (-1,1))], axis = 1)
            y_plot = tf.concat([tf.reshape(y_hat[1:,1], shape = (-1,1)),\
                                tf.reshape(y_compare[:,1], shape = (-1,1))], axis = 1)
            z_plot = tf.concat([tf.reshape(y_hat[1:,2], shape = (-1,1)),\
                                tf.reshape(y_compare[:,2], shape = (-1,1))], axis = 1)
    return metric[0] / metric[1], x_plot, y_plot, z_plot
    
@njit(fastmath = True)
def generate_lorenz_data(num_steps, data_seed = 1, train_samples = 1000,\
                         num_tests = 50, test_samples = 100, transient = 100, sync_len = 50, h = 0.01, tau = 0.1):
    np.random.seed(data_seed)
    u0 = np.random.rand(3)*2-1
    with objmode(int_steps = 'int64'):
        int_steps = int(tau/h)
    test_start = transient+1+train_samples+sync_len+num_steps
    test_len   = test_samples+sync_len+num_steps
    u_base  = rungekutta(x0 = u0[0], y0 = u0[1], z0 = u0[2], h = h, tau = tau, \
                         T = test_start - 1 + num_tests*test_len)[:,::int_steps]
    u_base[0] = u_base[0]/7.929788629895004
    u_base[1] = u_base[1]/8.9932616136662
    u_base[2] = (u_base[2]-23.596294463016896)/8.575917849311919
    u_train = np.ascontiguousarray(u_base[:,transient+1:transient+1+train_samples+sync_len+num_steps].T)
    u_test  = np.zeros((num_tests, sync_len+num_steps+test_samples, 3))
    for i in range(num_tests):
        u_test[i]  = np.ascontiguousarray(u_base[:,test_start+i*(test_len):test_start+(i+1)*test_len].T)
    return u_train, u_test

@njit(fastmath = True)
def lorenz_predict_onestep(y_in, h = 0.01, tau = 0.1):
    y_in[:,0] = y_in[:,0]*7.929788629895004
    y_in[:,1] = y_in[:,1]*8.9932616136662
    y_in[:,2] = y_in[:,2]*8.575917849311919+23.596294463016896
    y_onestep = np.zeros(y_in.shape)
    for i in range(y_in.shape[0]):
        y_onestep[i] = rungekutta(x0 = y_in[i,0], y0 = y_in[i,1], z0 = y_in[i,2], h = h, tau = tau,\
                                 T = 1)[:,-1]
    y_onestep[:,0] = y_onestep[:,0]/7.929788629895004
    y_onestep[:,1] = y_onestep[:,1]/8.9932616136662
    y_onestep[:,2] = (y_onestep[:,2] - 23.596294463016896)/8.575917849311919
    return y_onestep

def get_features_targets(Reservoir, u_train, u_test, batch_size, sync_len = 50):
    num_train  = u_train.shape[0] - Reservoir.num_steps - sync_len
    num_test   = u_test.shape[1]  - Reservoir.num_steps - sync_len
    num_inputs = u_train.shape[1]
    train_features, train_targets = get_synchronized_res_states(Reservoir, u_train, \
                                        sync_len, num_inputs, num_train)
    #print(train_features.shape)
    #print(train_targets.shape)
    #print(train_features[:10,:3])
    #print(train_targets[:10,:3])
    test_features = tf.zeros([0,train_features.shape[1]], dtype = train_features.dtype)
    test_targets  = tf.zeros([0,train_targets.shape[1]], dtype = train_targets.dtype)
    for i in range(u_test.shape[0]):
        test_features_i, test_targets_i = get_synchronized_res_states(Reservoir, u_test[i], \
                                        sync_len, num_inputs, num_test)
        test_features = tf.concat([test_features, test_features_i], 0)
        test_targets  = tf.concat([test_targets, test_targets_i], 0)
    
    #print(test_features.shape)
    #print(test_targets.shape)
    return (tf.data.Dataset.from_tensor_slices((train_features, train_targets)).batch(
        batch_size).shuffle(len(train_features)), tf.data.Dataset.from_tensor_slices(
        (test_features, test_targets)).batch(num_test))

def get_synchronized_res_states(Reservoir, u_in, sync_len, num_inputs, data_len):
    u_tf = tf.cast(tf.convert_to_tensor(np.concatenate((u_in, \
        np.zeros((u_in.shape[0], Reservoir.num_nodes))), axis = 1)), tf.double)
    res_states = Reservoir.synchronize(u_tf)
    features = res_states[sync_len:-Reservoir.num_steps]
    targets  = tf.zeros([features.shape[0],0], dtype = tf.double)
    for i in range(1,Reservoir.num_steps+1):
        targets = tf.concat([targets, u_tf[sync_len+i:sync_len+data_len+i,:num_inputs]], axis = 1)
    #print(features[0:3,:-num_nodes])
    #print(targets[0:3,:])
    return features, targets

def train_epoch_reservoir(Res, train_iter, loss, updater):
    metric = d2l.Accumulator(2)
    for X, y in train_iter:
        with tf.GradientTape() as tape:
            y_hat = Res(X, training = True)
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = Res.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        
        l_sum = l * float(tf.size(y)) if isinstance(loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, tf.size(y))
    return metric[0] / metric[1]

def evaluate_loss(Res, data_iter, loss):
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        y_hat = Res(X, training = True)
        if isinstance(loss, tf.keras.losses.Loss):
            metric.add(loss(y, y_hat) * float(tf.size(y)), tf.size(y))
        else:
            metric.add(tf.reduce_sum(loss(y_hat, y)), tf.size(y))
    return metric[0] / metric[1]

class Animator_lorenz:  #@save
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:', 'k--', 'm:'), nrows=4, ncols=1,
                 figsize=(6, 12)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize,\
            gridspec_kw= {'height_ratios': [3,1,1,1]})
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        self.x, self.y, self.z    = None, None, None

    def add(self, x, y, x_plot, y_plot, z_plot):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        self.axes[1].clear()
        self.axes[1].plot(x_plot)
        self.axes[1].set_ylim(-2.5, 2.5)
        self.axes[2].clear()
        self.axes[2].plot(y_plot)
        self.axes[2].set_ylim(-2.5, 2.5)
        self.axes[3].clear()
        self.axes[3].plot(z_plot)
        self.axes[3].set_ylim(-2.5, 2.5)
        display.display(self.fig)
        display.clear_output(wait=True)

def train_reservoir_lorenz(Res, train_iter, test_iter, loss, num_epochs, updater, ylim = [8e-2, 1e2]):
    animator = Animator_lorenz(xlabel='epoch', xlim=[1, num_epochs], ylim=ylim,
                        yscale = 'log', legend=['train loss', 'test loss', 'test valid time'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_reservoir(Res, train_iter, loss, updater)
        test_loss     = evaluate_loss(Res, test_iter, loss)
        if epoch % 20 == 0:
            test_valid_time, x_plot, y_plot, z_plot = evaluate_preds(Res, test_iter, loss)
        animator.add(epoch+1, (train_metrics, test_loss, test_valid_time), x_plot, y_plot, z_plot)
    train_loss = train_metrics
    
def train_reservoir(Res, train_iter, test_iter, loss, num_epochs, updater, ylim = [8e-2, 1e2]):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=ylim,
                        yscale = 'log', legend=['train loss', 'test loss', 'test valid time'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_reservoir(Res, train_iter, loss, updater)
        test_loss     = evaluate_loss(Res, test_iter, loss)
        if epoch % 20 == 0:
            test_valid_time, x_plot, y_plot, z_plot = evaluate_preds(Res, test_iter, loss)
        animator.add(epoch+1, (train_metrics, test_loss, test_valid_time))
    train_loss = train_metrics
