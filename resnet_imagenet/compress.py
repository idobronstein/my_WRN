#!/usr/bin/env python

import sys
import os
from datetime import datetime
import time
import re 
import pickle

import torch
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans

import dataset
import image_processing
import resnet


UPDATE_PARAM_REGEX = re.compile('(unit_)(\d_\d)(/)(bn|conv)(_\d)(/)(kernel|beta|gamma|mu|sigma)(:0)')
SKIP_PARAM_REGEX =re.compile('(unit_)(\d_\d)/(bn)(_1)(/)(beta|gamma|mu|sigma)(:0)')
BATCH_NORM_PARAM_NUM = 4
BATCH_NORM_PARAN_NAMES = ['mu', 'sigma', 'beta', 'gamma']


# Dataset Configuration
tf.app.flags.DEFINE_string('param_dir', './wide-resnet-50-2-export.pth', """Resnet-50-2-bottelneck pre-train""")
tf.app.flags.DEFINE_integer('num_classes', 1000, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_test_instance', 10000, """Number of test images.""")

# Testing Configuration
tf.app.flags.DEFINE_bool('train_data', False, """Whether to test over training set.""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of iterations during a test""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

# cluster params
tf.app.flags.DEFINE_integer('new_k', 1, """New Network width multiplier""")

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
    k_means =  KMeans(n_clusters==cluster_num, algorithm="full", random_state=0)
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

def compress():

    assert FLAGS.image_size == 224

    with tf.Graph().as_default():

        # Build a Graph that computes the predictions from the inference model.
        images = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size])

        # Build model
        hp = resnet.HParams(batch_size=FLAGS.batch_size,
                            num_classes=FLAGS.num_classes,
                            weight_decay=None,
                            initial_lr=None,
                            decay_step=None,
                            lr_decay=None,
                            momentum=None)
        params = {k: v.numpy() for k,v in torch.load(FLAGS.param_dir).items()}
        network = resnet.ResNet(params, hp, images, labels, None)
        network.build_model()
        network.count_trainable_params()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        
        graph = tf.get_default_graph()
        import ipdb; ipdb.set_trace()
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
        # The CIFAR-100 dataset
        with tf.variable_scope('test_image'):
            test_images, test_labels = data_input.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=FLAGS.train_data, num_threads=1)

        # The class labels
        with open(os.path.join(FLAGS.data_dir, 'fine_label_names.txt')) as fd:
            classes = [temp.strip() for temp in fd.readlines()]

        images = tf.placeholder(tf.float32, [FLAGS.batch_size, data_input.HEIGHT, data_input.WIDTH, 3])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size])

        new_network = resnet.ResNet(hp, images, labels, None, init_params, FLAGS.new_k)
        new_network.build_model()

        init = tf.initialize_all_variables()
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            log_device_placement=FLAGS.log_device_placement))
        #sess.run(init)


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
            assert b % FLAGS.batch_size == 0
            for j in range(int(b / FLAGS.batch_size)):
                batch_images_val =  np.moveaxis(test_images_val[j : j + FLAGS.batch_size], 1, -1)
                batch_labels_val = test_labels_val[j : j + FLAGS.batch_size]
                preds_val, loss_value, acc_value = sess.run([network.preds, network.loss, network.acc],
                            feed_dict={ images:batch_images_val, labels:batch_labels_val})
                print('acc: ', acc_value)
                test_loss += loss_value
                for k in range(FLAGS.batch_size):
                    correct = 0 if test_labels_val[k] ==    [k] else 1
                    result_ll[test_labels_val[k] % FLAGS.num_classes][correct] += 1
        test_loss /= FLAGS.test_iter
        # Summary display & output
        acc_list = [float(r[0])/float(r[0]+r[1]) for r in result_ll if r[0]+r[1] > 0]
        result_total = np.sum(np.array(result_ll), axis=0)
        acc_total = float(result_total[0])/np.sum(result_total)
        format_str = '%-31s %7d %7d %.5f'
        print(format_str % ('(Total)', result_total[0], result_total[1], acc_total))


def main(argv=None):  # pylint: disable=unused-argument
  compress()


if __name__ == '__main__':
  tf.app.run()
