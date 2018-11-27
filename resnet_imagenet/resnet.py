import torch
import tensorflow as tf
import numpy as np
import re

def Resnet(inputs, params):
    '''Bottleneck WRN-50-2 model definition
    '''
    def __init__(params, hp, images, labels, global_step):
        self._params = {k: tf.constant(tr(v)) for k, v in params.items()}
        self._hp = hp 
        self._images = images 
        self._labels = labels
        self._global_step = global_step

    def tr(self, v):
        if v.ndim == 4:
            return v.transpose(2,3,1,0)
        elif v.ndim == 2:
            return v.transpose()
        return v
    
    def conv2d(self, x,  name, stride=1, padding=0):
        x = tf.pad(x, [[0,0],[padding,padding],[padding,padding],[0,0]])
        z = tf.nn.conv2d(x, self._params['%s.weight'%name], [1,stride,stride,1],
                         padding='VALID')
        if '%s.bias'%name in self._params:
            return tf.nn.bias_add(z, self._params['%s.bias'%name])
        else:
            return z
    
    def group(self, input, base, stride, n):
        o = input
        for i in range(0,n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o = conv2d(x, b_base + '0')
            o = tf.nn.relu(o)
            o = conv2d(o, b_base + '1', stride=i==0 and stride or 1, padding=1)
            o = tf.nn.relu(o)
            o = conv2d(o, b_base + '2')
            if i == 0:
                o += conv2d(x, b_base + '_dim', stride=stride)
            else:
                o += x
            o = tf.nn.relu(o)

        return o

    def build_train_op(self):
        # Add l2 loss
        with tf.variable_scope('l2_loss'):
            costs = [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
            l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
        self._total_loss = self.loss + l2_loss
        
        # Learning rate
        self.lr = tf.train.exponential_decay(self._hp.initial_lr, self._global_step,
                                        self._hp.decay_step, self._hp.lr_decay, staircase=True)
        tf.summary.scalar('learing_rate', self.lr)
        
        # Gradient descent step
        opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        grads_and_vars = opt.compute_gradients(self._total_loss, tf.trainable_variables())
        apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)
        self.train_op = apply_grad_op

    def build_model(self):
        # determine network size by parameters
        blocks = [sum([re.match('group%d.block\d+.conv0.weight'%j, k) is not None
                       for k in self._params.keys()]) for j in range(4)]
    
        o = conv2d(inputs, 'conv0', 2, 3)
        o = tf.nn.relu(o)
        o = tf.pad(o, [[0,0], [1,1], [1,1], [0,0]])
        o = tf.nn.max_pool(o, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
        o_g0 = group(o, 'group0', 1, blocks[0])
        o_g1 = group(o_g0, 'group1', 2, blocks[1])
        o_g2 = group(o_g1, 'group2', 2, blocks[2])
        o_g3 = group(o_g2, 'group3', 2, blocks[3])
        o = tf.nn.avg_pool(o_g3, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID')
        o = tf.reshape(o, [-1, 2048])
        o = tf.matmul(o, self._params['fc.weight']) + self._params['fc.bias']
        
        self._logits = o
        # Probs & preds & acc
        self.probs = tf.nn.softmax(o, name='probs')
        self.preds = tf.to_int32(tf.argmax(self._logits, 1, name='preds'))
        ones = tf.constant(np.ones([self._hp.batch_size]), dtype=tf.float32)
        zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
        correct = tf.where(tf.equal(self.preds, self._labels), ones, zeros)
        self.acc = tf.reduce_mean(correct, name='acc')
        tf.summary.scalar('accuracy', self.acc)
        # Loss & acc
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=self._labels)
        self.loss = tf.reduce_mean(loss, name='cross_entropy')
        tf.summary.scalar('cross_entropy', self.loss)

        def count_trainable_params(self):
            total_parameters = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                variable_parametes = 1
                for dim in shape:
                    variable_parametes *= dim.value
                total_parameters += variable_parametes
            print("Total training params: {0}".format(total_parameters)) 