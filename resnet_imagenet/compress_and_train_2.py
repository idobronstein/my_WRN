#!/usr/bin/env python

import sys
import os
from datetime import datetime
import time
import re 
import pickle
import random
import torch
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from time import sleep

import dataset
import image_processing
import resnet

import cv2
import hickle as hkl
import torch.utils.data
import torchnet as tnt
import cvtransforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.autograd import Variable

UPDATE_PARAM_REGEX = '(group)({0})(/group{0}.block)({1})(.conv1/kernel:0)'
CONV1_KERNEL1_NAME = 'group{group_num}.block{block_num}.conv1.weight'
CONV1_KERNEL2_NAME = 'group{group_num}.block{block_num}.conv2.weight'
CONV1_BIAS_NAME = 'group{group_num}.block{block_num}.conv1.bias'


# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0005, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.001, """Initial learning rate""")
tf.app.flags.DEFINE_float('initial_lr_batchnorm', 0.001, """Initial learning rate fpr last retrain""")
tf.app.flags.DEFINE_float('lr_step_epoch', 3.0, """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")
tf.app.flags.DEFINE_float('decay_step', 5000, """steps between decay""")
tf.app.flags.DEFINE_integer('num_train_instance', 1000000, """Number of training images.""")
tf.app.flags.DEFINE_integer('num_test_instance', 49920, """Number of test images.""")


# Dataset Configuration
tf.app.flags.DEFINE_string('param_dir', './wide-resnet-50-2-export.pth', """Resnet-50-2-bottelneck pre-train""")
tf.app.flags.DEFINE_integer('num_classes', 1000, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_string('imagenetpath', '/specific/netapp5_2/gamir/datasets/imagenet/',  help='path to dataset')

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('test_interval', 500, """Number of iterations to run a test""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of iterations during a test""")
tf.app.flags.DEFINE_integer('final_test_iter', 4000, """Number of iterations during a test""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 10000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_integer('num_gpus', 4, """Number of gpu to use""")
tf.app.flags.DEFINE_integer('num_workers', 8, """Number of workers to get data""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")


# cluster params
tf.app.flags.DEFINE_float('compression_rate', 0.5, """New Network width multiplier""")
tf.app.flags.DEFINE_integer('block_to_compress', 3, """The block to compress""")


FLAGS = tf.app.flags.FLAGS

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

def sum_bias(bias, cluster_indices, cluster_num):
    new_bias = np.zeros([cluster_num])
    for cluster in range(cluster_num):
        cluster_sum = 0
        for i in range(len(cluster_indices)):
            if cluster_indices[i] == cluster:
                cluster_sum += bias[i]
        new_bias[cluster] = cluster_sum
    return new_bias

def sum_kernel(kernel, cluster_indices, cluster_num):
    h, w, i, o = kernel.shape
    add_kernels = np.zeros([h, w, cluster_num, o])
    for cluster in range(cluster_num):
        cluster_sum = 0
        for i in range(len(cluster_indices)):
            if cluster_indices[i] == cluster:
                cluster_sum += kernel[:, :, i, :]
        add_kernels[:, :, cluster, :] = cluster_sum
    return add_kernels

def get_image_file(image_path, is_np=True):
    with open(image_path, 'rb') as f:
            test_images_val, test_labels_val = pickle.load(f)
    if not is_np:
        test_images_val = test_images_val.numpy()
        test_labels_val = test_labels_val.numpy()
    test_images_val = np.moveaxis(test_images_val, 1, -1)
    return test_images_val, test_labels_val

def cvload(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_data_loder(data_set_type, suffle):
    datadir = os.path.join(FLAGS.imagenetpath, data_set_type)
    ds = datasets.ImageFolder(datadir, tnt.transform.compose([
            cvtransforms.Scale(256),
            cvtransforms.CenterCrop(224),
            lambda x: x.astype(np.float32) / 255.0,
            cvtransforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            lambda x: x.transpose(2,0,1).astype(np.float32),
            torch.from_numpy,
            ]), loader = cvload)
    print('{0} has {1} images'.format(datadir, len(ds)))
    train_loader = torch.utils.data.DataLoader(ds,
            batch_size=FLAGS.batch_size, shuffle=suffle, drop_last=True,
            num_workers=FLAGS.num_workers, pin_memory=False)
    return train_loader

def get_enumerate(loader):
    for sample in loader:
        test_images_val, test_labels_vals = sample
        test_images_val = sample[0].numpy()
        test_images_val = np.moveaxis(test_images_val, 1, -1)
        test_labels_val = sample[1].numpy()
        yield test_images_val, test_labels_val

def get_next_batch(enum, data_set_type, suffle):
    batch = next(enum, None)
    new_enum = None
    if batch is None:
        del enum
        new_enum = get_enumerate(get_data_loder(data_set_type, suffle))
        batch = next(new_enum, None)
    return batch[0], batch[1], new_enum

def compress():

    assert FLAGS.image_size == 224

    # set up data loader
    print("| setting up data loader...")
    train_loader = get_enumerate(get_data_loder('train', True))
    test_loader = get_enumerate(get_data_loder('val', True))

    params = {k: v.numpy() for k,v in torch.load(FLAGS.param_dir).items()}
    max_steps = FLAGS.max_steps
    init_step = 0
    restore_flag = True
    just_compress = True
    for layer_num in range(4):
        compress_layer = re.compile(UPDATE_PARAM_REGEX.format(FLAGS.block_to_compress, layer_num))
        if layer_num != 3:
            batch_norm = False
            initial_lr = FLAGS.initial_lr
            with tf.Graph().as_default():
        
                # Build a Graph that computes the predictions from the inference model.
                images = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
                labels = tf.placeholder(tf.int32, [None])
                images_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=images)
                labels_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=labels)
                is_training = tf.placeholder(tf.bool, shape=[])
        
                # Build model
                hp = resnet.HParams(batch_size=int(FLAGS.batch_size / FLAGS.num_gpus),
                                    num_classes=FLAGS.num_classes,
                                    weight_decay=None,
                                    initial_lr=None,
                                    decay_step=None,
                                    lr_decay=None,
                                    momentum=None)
                
                network = resnet.ResNet(params, hp, images_splits[0], labels_splits[0], None, is_training, False)
                network.build_model()
                if layer_num == 0:
                    old_param_num = network.count_trainable_params()
        
                # Build an initialization operation to run below.
                init = tf.initialize_all_variables()
        
                # Start running operations on the Graph.
                sess = tf.Session(config=tf.ConfigProto(
                    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
                    log_device_placement=FLAGS.log_device_placement, allow_soft_placement=True))
                sess.run(init)
                
                graph = tf.get_default_graph()
                flag1 = False
                flag2 = False
                new_params = {}
                for var in tf.trainable_variables():
                    var_vec = sess.run(var)
                    match = compress_layer.match(var.name)
                    if match:
                        print("compress: ", var.name)
                        sys.stdout.flush()
                        group_num = int(match.groups()[1])
                        block_num = int(match.groups()[3])
                        cluster_num = int(int(var.shape[-1]) * FLAGS.compression_rate)
                        cluster_centers, cluster_indices = cluster_kernel(var_vec, cluster_num)
                        new_params[CONV1_KERNEL1_NAME.format(group_num=group_num, block_num=block_num)] = (cluster_centers, False)
                        flag1 = True
                    elif flag1:
                        new_bias = sum_bias(var_vec, cluster_indices, cluster_num)
                        new_params[CONV1_BIAS_NAME.format(group_num=group_num, block_num=block_num)] = (new_bias ,False)
                        flag1 = False
                        flag2 = True
                    elif flag2:
                        new_kernel = sum_kernel(var_vec, cluster_indices, cluster_num)
                        new_params[CONV1_KERNEL2_NAME.format(group_num=group_num, block_num=block_num)] = (new_kernel ,False)
                        flag2 = False
    
                for k, v in params.items():
                        if k not in new_params:
                            if len(v) == 1:
                                new_params[k] = (v, True)
                            else:
                                new_params[k] = v
                #close old graph
                sess.close()
            tf.reset_default_graph()
        else:
            batch_norm = True
            new_params = params
            initial_lr = FLAGS.initial_lr_batchnorm
    
        if just_compress:
            just_compress = False
            # build new graph and eval
            with tf.Graph().as_default():
                global_step = tf.Variable(0, trainable=False, name='global_step')
        
                images = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
                labels = tf.placeholder(tf.int32, [None])
                images_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=images)
                labels_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=labels)
                is_training = tf.placeholder(tf.bool, shape=[])
                
                hp = resnet.HParams(batch_size=int(FLAGS.batch_size / FLAGS.num_gpus),
                            num_classes=FLAGS.num_classes,
                            weight_decay=FLAGS.l2_weight,
                            initial_lr=initial_lr,
                            decay_step=FLAGS.decay_step,
                            lr_decay=FLAGS.lr_decay,
                            momentum=FLAGS.momentum)
                new_network = resnet.MultiResNet(new_params, hp, images_splits, labels_splits, FLAGS.num_gpus, global_step, is_training, batch_norm)
                new_network.build_train_op()
                new_param_num = new_network.count_trainable_params()
                print("compression rate: ", 100 - new_param_num / old_param_num * 100, " %")
        
                # Summaries(training)
                train_summary_op = tf.summary.merge_all()
        
                init = tf.initialize_all_variables()
                # Start running operations on the Graph.
                sess = tf.Session(config=tf.ConfigProto(
                    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
                    log_device_placement=FLAGS.log_device_placement, allow_soft_placement=True))
                sess.run(init)
        
                # Create a saver.
                saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
                if restore_flag:
                    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        print('\tRestore from %s' % ckpt.model_checkpoint_path)
                        # Restores from checkpoint
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                    else:
                       print('No checkpoint file found. Start from the scratch.')
                    restore_flag = False
                sys.stdout.flush()
        
                # Start queue runners & summary_writer
                tf.train.start_queue_runners(sess=sess)
                if not os.path.exists(FLAGS.train_dir):
                    os.mkdir(FLAGS.train_dir)
                summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        
                # Training!
                test_best_acc = 0.0
                image_train_file = 0
                index_train_file = 0
                for step in range(init_step, max_steps):
                    # Test
                    if step % FLAGS.test_interval == 0:
                        test_loss, test_acc = 0.0, 0.0
                        for i in  range(FLAGS.test_iter):
                            test_images_val, test_labels_val, new_test_loader = get_next_batch(test_loader, 'val', True)
                            if new_test_loader is not None:
                                test_loader = new_test_loader
                            loss_value, acc_value = sess.run([new_network.loss, new_network.acc],
                                        feed_dict={images:test_images_val, labels:test_labels_val, is_training:False})
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
                    image_batch, labels_batch, new_train_loader = get_next_batch(train_loader, 'train', True)
                    if new_train_loader is not None:
                        train_loader = new_train_loader
                    _, lr_value, loss_value, acc_value, train_summary_str = \
                            sess.run([new_network.train_op, new_network.lr, new_network.loss, new_network.acc, train_summary_op],
                                feed_dict={images:image_batch, labels:labels_batch, is_training:True})
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
                test_loss, test_acc = 0.0, 0.0
                for i in  range(FLAGS.final_test_iter):
                    test_images_val, test_labels_val, new_test_loader = get_next_batch(test_loader, 'val', True)
                    if new_test_loader is not None:
                        test_loader = new_test_loader
                    loss_value, acc_value = sess.run([new_network.loss, new_network.acc],
                                feed_dict={images:test_images_val, labels:test_labels_val, is_training:False})
                    test_loss += loss_value
                    test_acc += acc_value
                test_loss /= FLAGS.final_test_iter
                test_acc /= FLAGS.final_test_iter
                test_best_acc = max(test_best_acc, test_acc)
                format_str = ('%s: (Final Test)     step %d, loss=%.4f, acc=%.4f')
                print(format_str % (datetime.now(), step, test_loss, test_acc))
                sys.stdout.flush()
                sess.close()
            tf.reset_default_graph()        
            params = new_params
            init_step = max_steps - 1
            max_steps += FLAGS.max_steps
        

def main(argv=None):  # pylint: disable=unused-argument
        try:
            compress()
        except Exception as e:
            print(e)
            sys.stdout.flush()

if __name__ == '__main__':
  tf.app.run()
