from collections import namedtuple
import tensorflow as tf
import numpy as np
import re


HParams = namedtuple('HParams',
                    'batch_size, num_classes,'
                    'weight_decay, initial_lr, decay_step, lr_decay, '
                    'momentum')

class ResNet():
    '''Bottleneck WRN-50-2 model definition
    '''
    def __init__(self, params, hp, images, labels, global_step):
        self._params = {k: self.tr(v) for k, v in params.items()}
        self._hp = hp 
        self._images = images 
        self._labels = labels
        self._global_step = global_step

    def tr(self, v):
        if len(v) == 2:
            if v[1]:
                v = v[0]
            else:
                return v[0]
        if v.ndim == 4:
            return v.transpose(2,3,1,0)
        elif v.ndim == 2:
            return v.transpose()
        return v

    def init_variable(self, param, name):
        #variable = tf.constant(param)
        variable = tf.get_variable(name, initializer=np.float32(param))
        return variable
    
    def conv2d(self, x,  name, stride=1, padding=0):
        with tf.variable_scope(name) as scope:
            x = tf.pad(x, [[0,0],[padding,padding],[padding,padding],[0,0]])
            kernal = self.init_variable(self._params['%s.weight'%name], 'kernel')
            z = tf.nn.conv2d(x, kernal, [1,stride,stride,1], padding='VALID')
            if '%s.bias'%name in self._params:
                bias = self.init_variable(self._params['%s.bias'%name], 'bias')
                return tf.nn.bias_add(z, bias)
            else:
                return z

    def group(self, input, base, stride, n):
        with tf.variable_scope(base) as scope:
            o = input
            for i in range(0,n):
                b_base = ('%s.block%d.conv') % (base, i)
                x = o
                o = self.conv2d(x, b_base + '0')
                o = tf.nn.relu(o)
                o = self.conv2d(o, b_base + '1', stride=i==0 and stride or 1, padding=1)
                o = tf.nn.relu(o)
                o = self.conv2d(o, b_base + '2')
                if i == 0:
                    o += self.conv2d(x, b_base + '_dim', stride=stride)
                else:
                    o += x
                o = tf.nn.relu(o)
            return o

    def build_model(self):
        # determine network size by parameters
        self.blocks = [sum([re.match('group%d.block\d+.conv0.weight'%j, k) is not None
                       for k in self._params.keys()]) for j in range(4)]
        
        o = self.conv2d(self._images, 'conv0', 2, 3)
        o = tf.nn.relu(o)
        o = tf.pad(o, [[0,0], [1,1], [1,1], [0,0]])
        o = tf.nn.max_pool(o, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
        o_g0 = self.group(o, 'group0', 1, self.blocks[0])
        o_g1 = self.group(o_g0, 'group1', 2, self.blocks[1])
        o_g2 = self.group(o_g1, 'group2', 2, self.blocks[2])
        o_g3 = self.group(o_g2, 'group3', 2, self.blocks[3])
        o_4 = tf.nn.avg_pool(o_g3, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID')
        o = tf.reshape(o_4, [-1, 2048])
        fc_weights = self.init_variable(self._params['fc.weight'], 'fc.weight')
        fc_bias = self.init_variable(self._params['fc.bias'], 'fc.bias')
        o = tf.add(tf.matmul(o, fc_weights), fc_bias)
        
        self._logits = o
        # Probs & preds & acc
        self.preds = tf.to_int32(tf.argmax(self._logits, 1, name='preds'))
        ones = tf.constant(np.ones([self._hp.batch_size]), dtype=tf.float32)
        zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
        correct = tf.where(tf.equal(self.preds, self._labels), ones, zeros)
        self.acc = tf.reduce_mean(correct, name='acc')
        tf.summary.scalar('accuracy', self.acc)
        # Loss & acc
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=o, labels=self._labels)
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
        return total_parameters


class MultiResNet():

    def __init__(self, params, hp, images, labels, num_gpus, global_step):
        self._params = params
        self._hp = hp 
        self._images = images 
        self._labels = labels
        self._global_step = global_step
        self._num_gpus = num_gpus
        
        self.lr = tf.train.exponential_decay(self._hp.initial_lr, self._global_step,
                                    self._hp.decay_step, self._hp.lr_decay, staircase=True)
        tf.summary.scalar('learing_rate', self.lr)
        # Gradient descent step
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
    
    def average_gradients(self, tower_grads):
          average_grads = []
          for grad_and_vars in zip(*tower_grads):
              # Note that each grad_and_vars looks like the following:
              #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
              grads = []
              for g, _ in grad_and_vars:
                  # Add 0 dimension to the gradients to represent the tower.
                  expanded_g = tf.expand_dims(g, 0)
          
                  # Append on a 'tower' dimension which we will average over below.
                  grads.append(expanded_g)
          
              # Average over the 'tower' dimension.
              grad = tf.concat(axis= 0, values= grads)
              grad = tf.reduce_mean(grad, 0)
      
              # Keep in mind that the Variables are redundant because they are shared
              # across towers. So .. we will just return the first tower's pointer to
              # the Variable.
              v = grad_and_vars[0][1]
              grad_and_var = (grad, v)
              average_grads.append(grad_and_var)
          return average_grads

    def get_grads(self, device):
        with tf.device(device):
            model = ResNet(self._params, self._hp, self._images, self._labels, self._global_step)
            model.build_model()

            # Add l2 loss
            with tf.variable_scope('l2_loss'):
                costs = [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
                l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
            self._total_loss = model.loss + l2_loss
            
            grads = self.optimizer.compute_gradients(self._total_loss)
        
        return grads, model.loss, model.acc

    def multigpu_grads(self):
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
          for i in range(self._num_gpus):
              with tf.name_scope('Tower_%d' % i) as scope:
                # Calculate the loss for one tower. This function
                # constructs the entire model but shares the variables across
                # all towers.
                grads, cross_entropy_mean, top1acc = self.get_grads('/gpu:%d' % i)
        
                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()
        
                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        
                # Keep track of the gradients across all towers.
                tower_grads.append(grads)
        #average graidents blah blah blah
        return self.average_gradients(tower_grads), cross_entropy_mean, top1acc

    def build_train_op(self):
        grads, self.loss, self.acc = self.multigpu_grads()
        self.train_op = self.optimizer.apply_gradients(grads)

    def count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: {0}".format(total_parameters)) 
        return total_parameters