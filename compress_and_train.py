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


UPDATE_PARAM_REGEX = re.compile('(unit_)(\d_\d)(/)(bn|conv)(_\d)(/)(kernel|beta|gamma|mu|sigma)(:0)')
SKIP_PARAM_REGEX =re.compile('(unit_)(\d_\d)/(bn)(_1)(/)(beta|gamma|mu|sigma)(:0)')
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
tf.app.flags.DEFINE_float('k', 2, """Network width multiplier""")

# Testing Configuration
tf.app.flags.DEFINE_string('ckpt_path', '', """Path to the checkpoint or dir.""")
tf.app.flags.DEFINE_bool('train_data', False, """Whether to test over training set.""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of iterations during a test""")
tf.app.flags.DEFINE_string('output', '', """Path to the output txt.""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

# cluster params
tf.app.flags.DEFINE_float('new_k', 1, """New Network width multiplier""")

# Other Configuration(not needed for testing, but required fields in
# build_model())
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_step_epoch', 100.0, """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 50000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 10000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_integer('test_interval', 1000, """Number of iterations to run a test""")

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

def add_kernel(kernel, cluster_indices, cluster_num):
    h, w, i, o = kernel.shape
    add_kernels = np.zeros([h, w, cluster_num, o])
    for cluster in range(cluster_num):
        cluster_sum = 0
        for i in range(len(cluster_indices)):
            if cluster_indices[i] == cluster:
                cluster_sum += kernel[:, :, i, :]
        add_kernels[:, :, cluster, :] = cluster_sum
    return add_kernels

def cluster_batch_norm(batch_norm, cluster_indices, cluster_num):
    clusters_batch_norm = np.zeros([BATCH_NORM_PARAM_NUM, cluster_num])
    for param_index in range(BATCH_NORM_PARAM_NUM):
        for cluster in range(cluster_num):
            cluster_size = 0
            cluster_sum = 0
            for i in range(len(cluster_indices)):
                if cluster_indices[i] == cluster:
                    cluster_size += 1
                    cluster_sum += batch_norm[param_index][i]
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
        old_kernels_to_cluster = []
        old_kernels_to_add = []
        old_batch_norm = []     
        for i in range(1, block_num + 1):
            for j in range(FLAGS.num_residual_units):
                    old_kernels_to_cluster.append(get_kernel(i, j, 1, graph, sess))
                    old_kernels_to_add.append(get_kernel(i, j, 2, graph, sess))
                    old_batch_norm.append(get_batch_norm(i, j, 2, graph, sess))
        #old_batch_norm = old_batch_norm[1:]
        #old_batch_norm.append(get_last_batch_norm(graph, sess))

        new_params = []
        new_width = [16, int(16 * FLAGS.new_k), int(32 * FLAGS.new_k), int(64 * FLAGS.new_k)]
        for i in range(len(old_batch_norm)):
            cluster_num = new_width[int(i / 4) + 1]
            cluster_kernels, cluster_indices = cluster_kernel(old_kernels_to_cluster[i], cluster_num)
            add_kernels = add_kernel(old_kernels_to_add[i], cluster_indices, cluster_num)
            cluster_batchs_norm = cluster_batch_norm(old_batch_norm[i], cluster_indices, cluster_num)
            new_params.append(cluster_kernels)
            for p in range(BATCH_NORM_PARAM_NUM):
                new_params.append(cluster_batchs_norm[p])
            new_params.append(add_kernels)
    
        # save variables
        init_params = []
        new_param_index = 0
        for var in tf.global_variables():
            update_match = UPDATE_PARAM_REGEX.match(var.name)
            skip_match = SKIP_PARAM_REGEX.match(var.name)
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
        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels of CIFAR-100
        with tf.variable_scope('train_image'):
            train_images, train_labels = data_input.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=True)
        with tf.variable_scope('test_image'):
            test_images, test_labels = data_input.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=False)

        # The class labels
        with open(os.path.join(FLAGS.data_dir, 'fine_label_names.txt')) as fd:
            classes = [temp.strip() for temp in fd.readlines()]

        images = tf.placeholder(tf.float32, [FLAGS.batch_size, data_input.HEIGHT, data_input.WIDTH, 3])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size])

        new_network = resnet.ResNet(hp, images, labels, global_step, init_params, FLAGS.new_k)
        new_network.build_model()
        new_network.build_train_op()

        train_summary_op = tf.summary.merge_all()

        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
           print('\tRestore from %s' % ckpt.model_checkpoint_path)
           # Restores from checkpoint
           saver.restore(sess, ckpt.model_checkpoint_path)
           init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
           print('No checkpoint file found. Start from the scratch.')
        sys.stdout.flush()
        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)
        if not os.path.exists(FLAGS.train_dir):
            os.mkdir(FLAGS.train_dir)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        # Training!
        test_best_acc = 0.0
        for step in range(init_step, FLAGS.max_steps):
            # Test
            if step % FLAGS.test_interval == 0:
                test_loss, test_acc = 0.0, 0.0
                for i in range(FLAGS.test_iter):
                    test_images_val, test_labels_val = sess.run([test_images, test_labels])
                    loss_value, acc_value = sess.run([new_network.loss, new_network.acc],
                                feed_dict={new_network.is_train:False, images:test_images_val, labels:test_labels_val})
                    test_loss += loss_value
                    test_acc += acc_value
                test_loss /= FLAGS.test_iter
                test_acc /= FLAGS.test_iter
                test_best_acc = max(test_best_acc, test_acc)
                format_str = ('%s: (Test)     step %d, loss=%.4f, acc=%.4f')
                print(format_str % (datetime.now(), step, test_loss, test_acc))
                sys.stdout.flush()
                test_summary = tf.Summary()
                test_summary.value.add(tag='test/loss', simple_value=test_loss)
                test_summary.value.add(tag='test/acc', simple_value=test_acc)
                test_summary.value.add(tag='test/best_acc', simple_value=test_best_acc)
                summary_writer.add_summary(test_summary, step)
                summary_writer.flush()
            # Train
            start_time = time.time()
            train_images_val, train_labels_val = sess.run([train_images, train_labels])
            _, lr_value, loss_value, acc_value, train_summary_str = \
                    sess.run([new_network.train_op, new_network.lr, new_network.loss, new_network.acc, train_summary_op],
                        feed_dict={new_network.is_train:True, images:train_images_val, labels:train_labels_val})
            duration = time.time() - start_time
            assert not np.isnan(loss_value)
            # Display & Summary(training)
            if step % FLAGS.display == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: (Training) step %d, loss=%.4f, acc=%.4f, lr=%f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, acc_value, lr_value,
                                     examples_per_sec, sec_per_batch))
                sys.stdout.flush()
                summary_writer.add_summary(train_summary_str, step)
            # Save the model checkpoint periodically.
            if (step > init_step and step % FLAGS.checkpoint_interval == 0) or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  assert FLAGS.ckpt_path != FLAGS.train_dir
  train()


if __name__ == '__main__':
  tf.app.run()
