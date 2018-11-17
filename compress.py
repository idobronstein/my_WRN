#!/usr/bin/env python

import sys
import os
from datetime import datetime
import time
import re 
import pickle

import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans

import cifar100 as data_input
import resnet


UPDATE_PARAM_REGEX = re.compile('(unit_)(\d_\d|last)(/)(bn|conv)(_\d)?(/)(kernel|beta|gamma|mu|sigma)(:0)')
SKIP_PARAM_REGEX =re.compile('(unit_)(1_0/)(bn)(_1)(/)(beta|gamma|mu|sigma)(:0)')
BATCH_NORM_PARAM_NUM = 4
BATCH_NORM_PARAN_NAMES = ['mu', 'sigma', 'beta', 'gamma']


# Dataset Configuration
tf.app.flags.DEFINE_string('data_dir', './cifar-100-binary', """Path to the CIFAR-100 binary data.""")
tf.app.flags.DEFINE_integer('num_classes', 100, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_test_instance', 10000, """Number of test images.""")
tf.app.flags.DEFINE_integer('num_train_instance', 50000, """Number of training images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 100, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_residual_units', 2, """Number of residual block per group.
                                                Total number of conv layers will be 6n+4""")
tf.app.flags.DEFINE_integer('k', 2, """Network width multiplier""")

# Testing Configuration
tf.app.flags.DEFINE_string('ckpt_path', '', """Path to the checkpoint or dir.""")
tf.app.flags.DEFINE_bool('train_data', False, """Whether to test over training set.""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of iterations during a test""")
tf.app.flags.DEFINE_string('output', '', """Path to the output txt.""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

# cluster params
tf.app.flags.DEFINE_integer('new_k', 1, """New Network width multiplier""")

# Other Configuration(not needed for testing, but required fields in
# build_model())
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_step_epoch', 100.0, """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")

FLAGS = tf.app.flags.FLAGS


def get_kernel(block_num, unit_num, conv_num, graph, sess):
    name = 'unit_{block_num}_{unit_num}/conv_{conv_num}/kernel:0'.format(
            block_num=block_num, unit_num=unit_num, conv_num=conv_num)
    kernel_tensor = graph.get_tensor_by_name(name)
    kernel_vector = sess.run(kernel_tensor)
    return kernel_vector

def get_batch_norm(block_num, unit_num, bn_num, graph, sess):
    name = 'unit_{block_num}_{unit_num}/bn_{bn_num}/{param}:0'
    batch_param = []
    for param in BATCH_NORM_PARAN_NAMES:
        param_tensor = graph.get_tensor_by_name(name.format(block_num=block_num, unit_num=unit_num, bn_num=bn_num, param=param))
        param_vector = sess.run(param_tensor)
        batch_param.append(param_vector)
    return batch_param

def get_last_batch_norm(graph, sess):
    name_last = 'unit_last/bn/{param}:0'
    batch_param = []
    for param in BATCH_NORM_PARAN_NAMES:
        param_tensor = graph.get_tensor_by_name(name_last.format(param=param))
        param_vector = sess.run(param_tensor)
        batch_param.append(param_vector)
    return batch_param

def cluster_kernel(kernel, cluster_num):
    k_means =  KMeans(n_clusters=cluster_num, algorithm="full", random_state=0)
    h, w, i, o = kernel.shape
    kernel_shift = np.moveaxis(kernel, -1, 0)
    kernel_reshape = np.reshape(kernel_shift, [o, h*w*i])
    k_meas_res = k_means.fit(kernel_reshape)
    cluster_indices = k_meas_res.labels_
    cluster_centers = k_meas_res.cluster_centers_
    cluster_centers = [np.reshape(cluster_centers[k], [h, w, i]) for k in range(cluster_num)]
    cluster_centers = np.moveaxis(cluster_centers, 0, 3)
    return cluster_centers, cluster_indices

def cluster_batch_norm(batch_norm, cluster_indices, cluster_num):
    clusters_batch_norm = np.zeros([BATCH_NORM_PARAM_NUM, cluster_num])
    mean = batch_norm[0]
    variance = batch_norm[1]
    sqaured_mean = [v + m**2 for v, m in zip(variance, mean)]
    for param_index in range(BATCH_NORM_PARAM_NUM):
        for cluster in range(cluster_num):
            cluster_size = 0
            cluster_sum = 0
            for i in range(len(cluster_indices)):
                if cluster_indices[i] == cluster:
                    cluster_size += 1
                    cluster_sum += batch_norm[param_index][i]
            if param_index == 1:
                clusters_batch_norm[param_index][cluster] = cluster_sum / (cluster_size ** 2)
            else:
                clusters_batch_norm[param_index][cluster] = cluster_sum / cluster_size
    return clusters_batch_norm

def train():
    print('[Dataset Configuration]')
    print('\tCIFAR-100 dir: %s' % FLAGS.data_dir)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of test images: %d' % FLAGS.num_test_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tResidual blocks per group: %d' % FLAGS.num_residual_units)
    print('\tNetwork width multiplier: %d' % FLAGS.k)

    print('[Testing Configuration]')
    print('\tCheckpoint path: %s' % FLAGS.ckpt_path)
    print('\tDataset: %s' % ('Training' if FLAGS.train_data else 'Test'))
    print('\tNumber of testing iterations: %d' % FLAGS.test_iter)
    print('\tOutput path: %s' % FLAGS.output)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)


    with tf.Graph().as_default():

        # Build a Graph that computes the predictions from the inference model.
        images = tf.placeholder(tf.float32, [FLAGS.batch_size, data_input.HEIGHT, data_input.WIDTH, 3])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size])

        # Build model
        decay_step = FLAGS.lr_step_epoch * FLAGS.num_train_instance / FLAGS.batch_size
        hp = resnet.HParams(batch_size=FLAGS.batch_size,
                            num_classes=FLAGS.num_classes,
                            num_residual_units=FLAGS.num_residual_units,
                            k=FLAGS.k,
                            weight_decay=FLAGS.l2_weight,
                            initial_lr=FLAGS.initial_lr,
                            decay_step=decay_step,
                            lr_decay=FLAGS.lr_decay,
                            momentum=FLAGS.momentum)
        network = resnet.ResNet(hp, images, labels, None)
        network.build_model()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
        if os.path.isdir(FLAGS.ckpt_path):
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
            # Restores from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
               print('\tRestore from %s' % ckpt.model_checkpoint_path)
               saver.restore(sess, ckpt.model_checkpoint_path)
            else:
               print('No checkpoint file found in the dir [%s]' % FLAGS.ckpt_path)
               sys.exit(1)
        elif os.path.isfile(FLAGS.ckpt_path):
            print('\tRestore from %s' % FLAGS.ckpt_path)
            saver.restore(sess, FLAGS.ckpt_path)
        else:
            print('No checkpoint file found in the path [%s]' % FLAGS.ckpt_path)
            sys.exit(1)
        
        graph = tf.get_default_graph()
        block_num = 3
        conv_num = 2
        old_kernels = []
        old_batch_norm = []     
        for i in range(1, block_num + 1):
            for j in range(FLAGS.num_residual_units):
                for k in range(1, conv_num + 1):
                    old_kernels.append(get_kernel(i, j, k, graph, sess))
                    old_batch_norm.append(get_batch_norm(i, j, k, graph, sess))
        old_batch_norm = old_batch_norm[1:]
        old_batch_norm.append(get_last_batch_norm(graph, sess))

        new_params = []
        new_width = [16, 16 * FLAGS.new_k, 32 * FLAGS.new_k, 64 * FLAGS.new_k]
        for i in range(len(old_batch_norm)):
            cluster_kernels, cluster_indices = cluster_kernel(old_kernels[i], new_width[int(i / 8) + 1])
            cluster_batchs_norm = cluster_batch_norm(old_batch_norm[i], cluster_indices, new_width[int(i / 8) + 1])
            output_size = old_kernels[i].shape[-1]
            new_kernel = np.zeros(old_kernels[i].shape)
            for l in range(output_size):
                new_kernel[:, :, :, l] = cluster_kernels[:, :, :, cluster_indices[l]]
            new_params.append(new_kernel)
            for p in range(BATCH_NORM_PARAM_NUM):
                new_batch_norm_param = np.zeros(output_size)
                for l in range(output_size):
                    new_batch_norm_param[l] = cluster_batchs_norm[p][cluster_indices[l]]
                new_params.append(new_batch_norm_param)
    
        '''
        f = open('new_params.pkl', 'rb')
        new_params = pickle.load(f)
        '''
        # save variables
        init_params = []
        new_param_index = 0
        for var in tf.global_variables():
            update_match = UPDATE_PARAM_REGEX.match(var.name)
            skip_match = SKIP_PARAM_REGEX.match(var.name)
            skip_first_bn = 0
            if update_match and not skip_match:
                print("update {}".format(var.name))
                init_params.append((new_params[new_param_index], var.name))
                new_param_index += 1
            else:
                print("not update {}".format(var.name))
                var_vector = sess.run(var)
                init_params.append((var_vector, var.name))

        #close old graph
        sess.close()
    tf.reset_default_graph()

    # build new graph and eval
    with tf.Graph().as_default():
        # The CIFAR-100 dataset
        with tf.variable_scope('test_image'):
            test_images, test_labels = data_input.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=FLAGS.train_data, num_threads=1)

        # The class labels
        with open(os.path.join(FLAGS.data_dir, 'fine_label_names.txt')) as fd:
            classes = [temp.strip() for temp in fd.readlines()]

        images = tf.placeholder(tf.float32, [FLAGS.batch_size, data_input.HEIGHT, data_input.WIDTH, 3])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size])

        new_network = resnet.ResNet(hp, images, labels, None, init_params)
        new_network.build_model()

        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Testing!
        result_ll = [[0, 0] for _ in range(FLAGS.num_classes)] # Correct/wrong counts for each class
        test_loss = 0.0, 0.0
        for i in range(FLAGS.test_iter):
            test_images_val, test_labels_val = sess.run([test_images, test_labels])
            preds_val, loss_value, acc_value = sess.run([new_network.preds, new_network.loss, new_network.acc],
                        feed_dict={new_network.is_train:False, images:test_images_val, labels:test_labels_val})
            test_loss += loss_value
            for j in range(FLAGS.batch_size):
                correct = 0 if test_labels_val[j] == preds_val[j] else 1
                result_ll[test_labels_val[j] % FLAGS.num_classes][correct] += 1
        test_loss /= FLAGS.test_iter

        # Summary display & output
        acc_list = [float(r[0])/float(r[0]+r[1]) for r in result_ll]
        result_total = np.sum(np.array(result_ll), axis=0)
        acc_total = float(result_total[0])/np.sum(result_total)

        print('Class    \t\t\tT\tF\tAcc.')
        format_str = '%-31s %7d %7d %.5f'
        for i in range(FLAGS.num_classes):
            print(format_str % (classes[i], result_ll[i][0], result_ll[i][1], acc_list[i]))
        print(format_str % ('(Total)', result_total[0], result_total[1], acc_total))

        # Output to file(if specified)
        if FLAGS.output.strip():
            with open(FLAGS.output, 'w') as fd:
                fd.write('Class    \t\t\tT\tF\tAcc.\n')
                format_str = '%-31s %7d %7d %.5f'
                for i in range(FLAGS.num_classes):
                    t, f = result_ll[i]
                    format_str = '%-31s %7d %7d %.5f\n'
                    fd.write(format_str % (classes[i].replace(' ', '-'), t, f, acc_list[i]))
                fd.write(format_str % ('(Total)', result_total[0], result_total[1], acc_total))


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
