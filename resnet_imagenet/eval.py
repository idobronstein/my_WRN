#!/usr/bin/env python

import os
from datetime import datetime
import time

import torch
import tensorflow as tf
import numpy as np

import dataset
import image_processing
import resnet
import re
import pickle

import sys

# Dataset Configuration
tf.app.flags.DEFINE_string('param_dir', './wide-resnet-50-2-export.pth', """Resnet-50-2-bottelneck pre-train""")
tf.app.flags.DEFINE_integer('num_classes', 1000, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_train_instance', 50000, """Number of training images.""")
tf.app.flags.DEFINE_integer('num_test_instance', 10000, """Number of test images.""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_step_epoch', 100.0, """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('test_interval', 1000, """Number of iterations to run a test""")
tf.app.flags.DEFINE_integer('test_iter', 195, """Number of iterations during a test""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 10000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

FLAGS = tf.app.flags.FLAGS

def g(inputs, params):
    '''Bottleneck WRN-50-2 model definition
    '''
    def tr(v):
        if v.ndim == 4:
            return v.transpose(2,3,1,0)
        elif v.ndim == 2:
            return v.transpose()
        return v
    params = {k: tf.constant(tr(v)) for k, v in params.items()}
    
    def conv2d(x, params, name, stride=1, padding=0):
        x = tf.pad(x, [[0,0],[padding,padding],[padding,padding],[0,0]])
        z = tf.nn.conv2d(x, params['%s.weight'%name], [1,stride,stride,1],
                         padding='VALID')
        if '%s.bias'%name in params:
            return tf.nn.bias_add(z, params['%s.bias'%name])
        else:
            return z
    
    def group(input, params, base, stride, n):
        o = input
        for i in range(0,n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o = conv2d(x, params, b_base + '0')
            o = tf.nn.relu(o)
            o = conv2d(o, params, b_base + '1', stride=i==0 and stride or 1, padding=1)
            o = tf.nn.relu(o)
            o = conv2d(o, params, b_base + '2')
            if i == 0:
                o += conv2d(x, params, b_base + '_dim', stride=stride)
            else:
                o += x
            o = tf.nn.relu(o)
        return o
    
    # determine network size by parameters
    blocks = [sum([re.match('group%d.block\d+.conv0.weight'%j, k) is not None
                   for k in params.keys()]) for j in range(4)]

    o = conv2d(inputs, params, 'conv0', 2, 3)
    o = tf.nn.relu(o)
    o = tf.pad(o, [[0,0], [1,1], [1,1], [0,0]])
    o = tf.nn.max_pool(o, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
    o_g0 = group(o, params, 'group0', 1, blocks[0])
    o_g1 = group(o_g0, params, 'group1', 2, blocks[1])
    o_g2 = group(o_g1, params, 'group2', 2, blocks[2])
    o_g3 = group(o_g2, params, 'group3', 2, blocks[3])
    o = tf.nn.avg_pool(o_g3, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID')
    o = tf.reshape(o, [-1, 2048])
    o = tf.matmul(o, params['fc.weight']) + params['fc.bias']
    return o

def train():
    print('[Dataset Configuration]')
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of training images: %d' % FLAGS.num_train_instance)
    print('\tNumber of test images: %d' % FLAGS.num_test_instance)

    print('[Optimization Configuration]')
    print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    print('\tEpochs per lr step: %f' % FLAGS.lr_step_epoch)
    print('\tLearning rate decay: %f' % FLAGS.lr_decay)

    print('[Training Configuration]')
    print('\tTrain dir: %s' % FLAGS.train_dir)
    print('\tTraining max steps: %d' % FLAGS.max_steps)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tSteps per testing: %d' % FLAGS.test_interval)
    print('\tSteps during testing: %d' % FLAGS.test_iter)
    print('\tSteps per saving checkpoints: %d' % FLAGS.checkpoint_interval)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)

    sys.stdout.flush()

    assert FLAGS.image_size == 224

    with tf.Graph().as_default():

        # Build a Graph that computes the predictions from the inference model.
        images = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size])

        # Build model
        decay_step = FLAGS.lr_step_epoch * FLAGS.num_train_instance / FLAGS.batch_size
        hp = resnet.HParams(batch_size=FLAGS.batch_size,
                            num_classes=FLAGS.num_classes,
                            weight_decay=FLAGS.l2_weight,
                            initial_lr=FLAGS.initial_lr,
                            decay_step=decay_step,
                            lr_decay=FLAGS.lr_decay,
                            momentum=FLAGS.momentum)
        params = {k: v.numpy() for k,v in torch.load(FLAGS.param_dir).items()}
        '''network = resnet.ResNet(params, hp, images, labels, None)
        network.build_model()
        network.count_trainable_params()
        '''  
        network = g(images, params)  
        # Summaries(training)
        train_summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)


        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)
        if not os.path.exists(FLAGS.train_dir):
            os.mkdir(FLAGS.train_dir)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        # Testing!
        result_ll = [[0, 0] for _ in range(FLAGS.num_classes)] # Correct/wrong counts for each class
        test_loss = 0.0, 0.0
        for i in range(FLAGS.test_iter):
            print("Step number: {0}".format(i))
            with open('/specific/netapp5_2/gamir/idobronstein/checkouts/my_WRN/resnet_imagenet/images/image_{0}'.format(i), 'rb') as f:
                    test_images_val, test_labels_val = pickle.load(f)
            b, c, h, w = test_images_val.shape
            test_images_val = np.reshape(test_images_val, [b, h, w, c])
            preds_val, loss_value, acc_value = sess.run([network.preds, network.loss, network.acc],
                        feed_dict={ images:test_images_val, labels:test_labels_val})
            test_loss += loss_value
            for j in range(FLAGS.batch_size):
                correct = 0 if test_labels_val[j] == preds_val[j] else 1
                result_ll[test_labels_val[j] % FLAGS.num_classes][correct] += 1
        test_loss /= FLAGS.test_iter
        # Summary display & output
        acc_list = [float(r[0])/float(r[0]+r[1]) for r in result_ll if r[0]+r[1] > 0]
        result_total = np.sum(np.array(result_ll), axis=0)
        acc_total = float(result_total[0])/np.sum(result_total)
        format_str = '%-31s %7d %7d %.5f'
        print(format_str % ('(Total)', result_total[0], result_total[1], acc_total))

def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
