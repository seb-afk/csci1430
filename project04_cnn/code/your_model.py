#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: your_model.py
# Brown CSCI 1430 assignment
# Created by Aaron Gokaslan

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.tower import get_current_tower_context
import tensorflow as tf
import hyperparameters as hp

class YourModel(ModelDesc):

    def __init__(self):
        super(YourModel, self).__init__()
        self.use_bias = True

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, hp.img_size, hp.img_size, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs

        #####################################################################
        # TASK 1: Change architecture (to try to improve performance)
        # TASK 1: Add dropout regularization                                  
        
        # Declare convolutional layers
        #
        # TensorPack: Convolutional layer
        # 10 filters (out_channel), 9x9 (kernel_shape), 
        # no padding, stride 1 (default settings)
        # with ReLU non-linear activation function.
        logits = Conv2D('conv0', image, 10, (9,9), padding='valid', stride=(1,1), nl=tf.nn.relu)
        #
        # TensorPack: Max pooling layer
        # Chain layers together using reference 'logits'
        # 7x7 max pool, stride = none (defaults to same as shape), padding = valid
        logits = MaxPooling('pool0', logits, (7,7), stride=None, padding='valid')
        #
        # TensorPack: Fully connected layer
        # number of outputs = number of categories (the 15 scenes in our case)
        #logits = FullyConnected('fc0', logits, hp.category_num, nl=tf.identity)
        logits = FullyConnected('fc0', logits, hp.category_num, nl=tf.nn.relu)
        #####################################################################

        # Add a loss function based on our network output (logits) and the ground truth labels
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)

        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))


        #####################################################################
        # TASK 1: If you like, you can add other kinds of regularization, 
        # e.g., weight penalization, or weight decay


        #####################################################################


        # Set costs and monitor them for TensorBoard
        add_moving_summary(cost)
        add_param_summary(('.*/kernel', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost], name='cost')


    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', hp.learning_rate, summary=True)

        # Use gradient descent as our optimizer
        opt = tf.train.GradientDescentOptimizer(lr)

        # There are many other optimizers - https://www.tensorflow.org/api_guides/python/train#Optimizers
        # Including the momentum-based gradient descent discussed in class.
        
        return opt