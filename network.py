#!/usr/bin/env python
"""
CNN layer wrapper.
"""

from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'






def layer(op):
    """Decorator for composable network layers."""

    def layer_decorated(self, *args, **kwargs):
        """Layer decoration."""
        # We allow to construct low-level layers instead of high-level networks.
        if self.inputs is None or (len(args) > 0 and isinstance(args[0], tf.Tensor)):
            layer_output = op(self, *args, **kwargs)
            return layer_output
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if not self.terminals:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    """Class NetWork."""

    def __init__(self, inputs, is_training,
                 dropout_rate=0.5, seed=None, epsilon=1e-5, reuse=False, fcn=True, regularize=True,
                 **kwargs):
        # The input nodes for this network
        self.inputs = inputs
        # If true, the resulting variables are set as trainable
        self.trainable = is_training if isinstance(is_training, bool) else True
        # If true, variables are shared between feature towers
        self.reuse = reuse
        # If true, layers like batch normalization or dropout are working in training mode
        self.training = is_training
        # Dropout rate
        self.dropout_rate = dropout_rate
        # Seed for randomness
        self.seed = seed
        # Add regularizer for parameters.
        self.regularizer = tf.contrib.layers.l2_regularizer(1.0) if regularize else None
        # The epsilon paramater in BN layer.
        self.bn_epsilon = epsilon
        self.extra_args = kwargs
        if inputs is not None:
            # The current list of terminal nodes
            self.terminals = []
            # Mapping from layer names to layers
            self.layers = dict(inputs)
            # If true, dense layers will be omitted in network construction
            self.fcn = fcn
            self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False, exclude_var=None):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        if exclude_var is not None:
            keyword = exclude_var.split(',')
        assign_op = []
        for op_name in data_dict:
            if exclude_var is not None:
                find_keyword = False
                for tmp_keyword in keyword:
                    if op_name.find(tmp_keyword) >= 0:
                        find_keyword = True
                if find_keyword:
                    continue

            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():

                    try:
                        var = tf.get_variable(param_name)
                        assign_op.append(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise
                        else:
                            print(Notify.WARNING, ':'.join(
                                [op_name, param_name]), "is omitted.", Notify.ENDC)
        session.run(assign_op)

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert args
        self.terminals = []
        for fed_layer in args:
            #if isinstance(fed_layer, basestring):
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_output_by_name(self, layer_name):
        '''
        Get graph node by layer name
        :param layer_name: layer name string
        :return: tf node
        '''
        return self.layers[layer_name]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def change_inputs(self, input_tensors):
        assert len(input_tensors) == 1
        for key in input_tensors:
            self.layers[key] = input_tensors[key]
  

    @layer
    def conv(self,
             input_tensor,
             kernel_size,
             filters,
             strides,
             name,
             relu=True,
             dilation_rate=1,
             padding=DEFAULT_PADDING,
             biased=True,
             reuse=False,
             separable=False):
        """2D/3D convolution."""
        kwargs = {'filters': filters,
                  'kernel_size': kernel_size,
                  'strides': strides,
                  'activation': tf.nn.relu if relu else None,
                  'use_bias': biased,
                  'dilation_rate': dilation_rate,
                  'trainable': self.trainable,
                  'reuse': self.reuse or reuse,
                  'bias_regularizer': self.regularizer if biased else None,
                  'name': name,
                  'padding': padding}

        if separable:
            kwargs['depthwise_regularizer'] = self.regularizer
            kwargs['pointwise_regularizer'] = self.regularizer
        else:
            kwargs['kernel_regularizer'] = self.regularizer

        if len(input_tensor.get_shape()) == 4:
            if not separable:
                return tf.layers.conv2d(input_tensor, **kwargs)
            else:
                return tf.layers.separable_conv2d(input_tensor, **kwargs)
        elif len(input_tensor.get_shape()) == 5:
            if not separable:
                return tf.layers.conv3d(input_tensor, **kwargs)
            else:
                raise NotImplementedError('No official implementation for separable_conv3d')
        else:
            raise ValueError('Improper input rank for layer: ' + name)

    @layer
    def conv_gn(self,
                input_tensor,
                kernel_size,
                filters,
                strides,
                name,
                relu=True,
                center=False,
                scale=False,
                dilation_rate=1,
                channel_wise=True,
                group=32,
                group_channel=8,
                padding=DEFAULT_PADDING,
                biased=False,
                separable=False):
        assert len(input_tensor.get_shape()) == 4
        conv = self.conv(input_tensor, kernel_size, filters, strides, name, relu=False,
                         dilation_rate=dilation_rate, padding=padding,
                         biased=biased, reuse=self.reuse, separable=separable)

        # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
        x = tf.transpose(conv, [0, 3, 1, 2])
        shape = tf.shape(x)
        N = shape[0]
        C = int(x.get_shape()[1])
        H = shape[2]
        W = shape[3]
        if channel_wise:
            G = int(max(1, C / group_channel))
        else:
            G = int(min(group, C))

        # normalization 
        x = tf.reshape(x, [N, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + self.bn_epsilon)

        # per channel scale and bias (gamma and beta)
        with tf.variable_scope(name + '/gn', reuse=self.reuse):
            if scale:
                gamma = tf.get_variable('gamma', [C], dtype=tf.float32, initializer=tf.ones_initializer())
            else:
                gamma = tf.constant(1.0, shape=[C])
            if center:
                beta = tf.get_variable('beta', [C], dtype=tf.float32, initializer=tf.zeros_initializer())
            else:
                beta = tf.constant(0.0, shape=[C])
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])
        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta

        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])
        if relu:
            output = self.relu(output, name + '/relu')
        return output

    @layer
    def conv_bn(self,
                input_tensor,
                kernel_size,
                filters,
                strides,
                name,
                relu=True,
                center=False,
                scale=False,
                dilation_rate=1,
                padding=DEFAULT_PADDING,
                biased=False,
                separable=False,
                reuse=False):
        conv = self.conv(input_tensor, kernel_size, filters, strides, name, relu=False,
                         dilation_rate=dilation_rate, padding=padding,
                         biased=biased, reuse=reuse, separable=separable)
        conv_bn = self.batch_normalization(conv, name + '/bn',
                                           center=center, scale=scale, relu=relu, reuse=reuse)
        return conv_bn

    @layer
    def deconv(self,
               input_tensor,
               kernel_size,
               filters,
               strides,
               name,
               relu=True,
               padding=DEFAULT_PADDING,
               biased=True,
               reuse=False):
        """2D/3D deconvolution."""
        kwargs = {'filters': filters,
                  'kernel_size': kernel_size,
                  'strides': strides,
                  'activation': tf.nn.relu if relu else None,
                  'use_bias': biased,
                  'trainable': self.trainable,
                  'reuse': self.reuse or reuse,
                  'kernel_regularizer': self.regularizer,
                  'bias_regularizer': self.regularizer if biased else None,
                  'name': name,
                  'padding': padding}

        if len(input_tensor.get_shape()) == 4:
            return tf.layers.conv2d_transpose(input_tensor, **kwargs)
        elif len(input_tensor.get_shape()) == 5:
            return tf.layers.conv3d_transpose(input_tensor, **kwargs)
        else:
            raise ValueError('Improper input rank for layer: ' + name)

    @layer
    def deconv_bn(self,
                  input_tensor,
                  kernel_size,
                  filters,
                  strides,
                  name,
                  relu=True,
                  center=False,
                  scale=False,
                  padding=DEFAULT_PADDING,
                  biased=False,
                  reuse=False):
        deconv = self.deconv(input_tensor, kernel_size, filters, strides, name,
                             relu=False, padding=padding, biased=biased, reuse=reuse)
        deconv_bn = self.batch_normalization(deconv, name + '/bn',
                                             center=center, scale=scale, relu=relu, reuse=reuse)
        return deconv_bn

    @layer
    def deconv_gn(self,
                  input_tensor,
                  kernel_size,
                  filters,
                  strides,
                  name,
                  relu=False,
                  center=False,
                  scale=False,
                  channel_wise=True,
                  group=32,
                  group_channel=8,
                  padding=DEFAULT_PADDING,
                  biased=False):
        assert len(input_tensor.get_shape()) == 4

        # deconvolution
        deconv = self.deconv(input_tensor, kernel_size, filters, strides, name,
                             relu=False, padding=padding, biased=biased, reuse=self.reuse)

        # group normalization
        x = tf.transpose(deconv, [0, 3, 1, 2])
        shape = tf.shape(x)
        N = shape[0]
        C = int(x.get_shape()[1])
        H = shape[2]
        W = shape[3]
        if channel_wise:
            G = int(max(1, C / group_channel))
        else:
            G = int(min(group, C))

        # normalization 
        x = tf.reshape(x, [N, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + self.bn_epsilon)

        # per channel scale and bias (gamma and beta)
        with tf.variable_scope(name + '/gn', reuse=self.reuse):
            if scale:
                gamma = tf.get_variable('gamma', [C], dtype=tf.float32, initializer=tf.ones_initializer())
            else:
                gamma = tf.constant(1.0, shape=[C])
            if center:
                beta = tf.get_variable('beta', [C], dtype=tf.float32, initializer=tf.zeros_initializer())
            else:
                beta = tf.constant(0.0, shape=[C])
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])
        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta

        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])

        if relu:
            output = self.relu(output, name + '/relu')
        return output

    @layer
    def relu(self, input_tensor, name=None):
        """ReLu activation."""
        return tf.nn.relu(input_tensor, name=name)

    @layer
    def max_pool(self, input_tensor, pool_size, strides, name, padding=DEFAULT_PADDING):
        """Max pooling."""
        return tf.layers.max_pooling2d(input_tensor,
                                       pool_size=pool_size,
                                       strides=strides,
                                       padding=padding,
                                       name=name)

    @layer
    def avg_pool(self, input_tensor, pool_size, strides, name, padding=DEFAULT_PADDING):
        """"Average pooling."""
        return tf.layers.average_pooling2d(input_tensor,
                                           pool_size=pool_size,
                                           strides=strides,
                                           padding=padding,
                                           name=name)

    @layer
    def l2_pool(self, input_tensor, pool_size, strides, name, padding=DEFAULT_PADDING):
        """L2 pooling."""
        return tf.sqrt(tf.layers.average_pooling2d(
            tf.square(input_tensor),
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            name=name) + 1e-6)

    @layer
    def lrn(self, input_tensor, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input_tensor,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, input_tensors, axis, name):
        return tf.concat(values=input_tensors, axis=axis, name=name)

    @layer
    def add(self, input_tensors, name):
        return tf.add_n(input_tensors, name=name)

    @layer
    def fc(self, input_tensor, num_out, name, biased=True, relu=True, flatten=True, reuse=False):
        # To behave same to Caffe.
        if flatten:
            flatten_tensor = tf.layers.flatten(input_tensor)
        else:
            flatten_tensor = input_tensor
        return tf.layers.dense(flatten_tensor,
                               units=num_out,
                               use_bias=biased,
                               activation=tf.nn.relu if relu else None,
                               trainable=self.trainable,
                               reuse=self.reuse or reuse,
                               kernel_regularizer=self.regularizer,
                               bias_regularizer=self.regularizer if biased else None,
                               name=name)

    @layer
    def fc_bn(self, input_tensor, num_out, name,
              biased=False, relu=True, center=False, scale=False, flatten=True, reuse=False):
        # To behave same to Caffe.
        fc = self.fc(input_tensor, num_out, name, relu=False,
                     biased=biased, flatten=flatten, reuse=reuse)
        fc_bn = self.batch_normalization(fc, name + '/bn',
                                         center=center, scale=scale, relu=relu, reuse=reuse)
        return fc_bn

    @layer
    def softmax(self, input_tensor, name, dim=-1):
        return tf.nn.softmax(input_tensor, dim=dim, name=name)

    @layer
    def batch_normalization(self, input_tensor, name,
                            center=False, scale=False, relu=False, reuse=False):
        """Batch normalization."""
        output = tf.layers.batch_normalization(input_tensor,
                                               center=center,
                                               scale=scale,
                                               fused=True,
                                               training=self.training,
                                               trainable=self.trainable,
                                               reuse=self.reuse or reuse,
                                               epsilon=self.bn_epsilon,
                                               gamma_regularizer=None,  # self.regularizer if scale else None,
                                               beta_regularizer=None,  # self.regularizer if center else None,
                                               name=name)
        if relu:
            output = self.relu(output, name + '/relu')
        return output

    @layer
    def dropout(self, input_tensor, name):
        return tf.layers.dropout(input_tensor,
                                 rate=self.dropout_rate,
                                 training=self.training,
                                 seed=self.seed,
                                 name=name)

    @layer
    def l2norm(self, input_tensor, name, dim=-1):
        return tf.nn.l2_normalize(input_tensor, dim=dim, name=name)

    @layer
    def squeeze(self, input_tensor, axis=None, name=None):
        return tf.squeeze(input_tensor, axis=axis, name=name)

    @layer
    def reshape(self, input_tensor, shape, name=None):
        return tf.reshape(input_tensor, shape, name=name)

    @layer
    def flatten(self, input_tensor, name=None):
        return tf.layers.flatten(input_tensor, name=name)

    @layer
    def tanh(self, input_tensor, name=None):
        return tf.tanh(input_tensor, name=name)


    @layer
    def NonLocalBlock(self, input_tensor, name=None):

        shape = tf.shape(input_tensor)
        N = shape[0]
        H = shape[1]
        W = shape[2]
        C = int(input_tensor.get_shape()[3])

        block1 = self.conv(input_tensor, 1, C // 2, 1, name='Block1')
        block2 = self.conv(input_tensor, 1, C // 2, 1, name='Block2')
        block3 = self.conv(input_tensor, 1, C // 2, 1, name='Block3')

        block1_reshape = tf.reshape(block1, [N, -1, C // 2], name='reshape1')
        block2_reshape = tf.reshape(block2, [N, -1, C // 2], name='reshape2')
        block3_reshape = tf.reshape(block3, [N, -1, C // 2], name='reshape3')

        block2_trans = tf.transpose(block2_reshape, [0, 2, 1],name='trans2')  # (B, C, H * W)
        f = tf.matmul(block1_reshape, block2_trans)                           # (B, H * W, H * W)
        fs = self.softmax(f, name='softmaxfs')                                # (B, H * W, H * W)

        y = tf.matmul(fs, block3_reshape)
        ys = tf.reshape(y, [N, H, W, C // 2])
        yconv = self.conv(ys, 1, C, 1, name='yconv')

        result = tf.add(input_tensor, yconv)

        return result


    @layer
    def simple_NonLocalBlock(self, input_tensor, name=None):

        shape = tf.shape(input_tensor)
        N = shape[0]
        H = shape[1]
        W = shape[2]
        C = int(input_tensor.get_shape()[3])

        block1 = self.conv(input_tensor, 1, 1, 1, name='Block1')             # (B, H, W, 1)
        block1_reshape = tf.reshape(block1, [N, -1, 1], name='reshape1')     # (B, H * W, 1)
        block3 = self.conv(input_tensor, 1, C, 1, name='Block3')             # (B, H, W, C)
        block3_reshape = tf.reshape(block3, [N, -1, C], name='reshape3')     # (B, H * W, C)

        block3_trans = tf.transpose(block3_reshape, [0, 2, 1], name='trans3')  # (B, C, H * W)

        fs = self.softmax(block1_reshape, name='softmaxfs')                   # (B, H * W, 1)
        f = tf.matmul(block3_trans, fs)                                       # (B, C, 1)
        f = tf.expand_dims(f, axis=3)                                         # (B, C, 1, 1)
        ft = tf.transpose(f, [0, 2, 3, 1], name='trans3')                      # (B, 1, 1, C)

        result = tf.add(input_tensor, ft)

        return result



def group_norm(input_tensor,
               name,
               channel_wise=True,
               group=32,
               group_channel=8,
               is_reuse=False):
    x = tf.transpose(input_tensor, [0, 3, 1, 2])

    # shapes and groups
    shape = tf.shape(x)
    N = shape[0]
    C = int(x.get_shape()[1])
    H = shape[2]
    W = shape[3]
    if channel_wise:
        G = int(max(1, C / group_channel))
    else:
        G = int(min(group, C))

    # use layer normalization to simplify operations
    if G == 1:
        return tf.contrib.layers.layer_norm(input_tensor)

    # use instance normalization to simplify operations
    elif G >= C:
        return tf.contrib.layers.instance_norm(input_tensor)

    # group normalization as suggested in the paper
    else:
        x = tf.reshape(x, [N, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + 1e-5)

        # per channel scale and bias (gamma and beta)
        with tf.variable_scope(name + '/gn', reuse=is_reuse):
            gamma = tf.get_variable('gamma', [C], dtype=tf.float32, initializer=tf.ones_initializer())
            beta = tf.get_variable('beta', [C], dtype=tf.float32, initializer=tf.zeros_initializer())

        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])
        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        output = tf.transpose(output, [0, 2, 3, 1])
        return output


class ConvGRUCell(tf.contrib.rnn.RNNCell):
    """A GRU cell with convolutions instead of multiplications."""

    def __init__(self,
                 shape,
                 filters,
                 kernel,
                 initializer=None,
                 activation=tf.tanh,
                 normalize=True):
        self._filters = filters
        self._kernel = kernel
        self._initializer = initializer
        self._activation = activation
        self._size = tf.TensorShape(shape + [self._filters])
        self._normalize = normalize
        self._feature_axis = self._size.ndims

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def __call__(self, x, h, scope=None):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Gates'):
                # concatenation
                inputs = tf.concat([x, h], axis=self._feature_axis)

                # convolution
                conv = tf.layers.conv2d(
                    inputs, 2 * self._filters, self._kernel, padding='same', name='conv')
                reset_gate, update_gate = tf.split(conv, 2, axis=self._feature_axis)

                # group normalization, actually is 'instance normalization' as to save GPU memory
                reset_gate = group_norm(reset_gate, 'reset_norm', group_channel=16)
                update_gate = group_norm(update_gate, 'update_norm', group_channel=16)

                # activation
                reset_gate = tf.sigmoid(reset_gate)
                update_gate = tf.sigmoid(update_gate)

            with tf.variable_scope('Output'):
                # concatenation
                inputs = tf.concat([x, reset_gate * h], axis=self._feature_axis)

                # convolution
                conv = tf.layers.conv2d(
                    inputs, self._filters, self._kernel, padding='same', name='output_conv')

                # group normalization
                conv = group_norm(conv, 'output_norm', group_channel=16)

                # activation
                y = self._activation(conv)

                # soft update
                output = update_gate * h + (1 - update_gate) * y

            return output, output




########################################################################################
############################# 2D feature extraction nework #############################
########################################################################################

class UNetDS2(Network):
    """2D U-Net with group normalization."""

    def setup(self):
        print('2D UNet with 16 channel output')
        base_filter = 8
        (self.feed('data')
         .conv_gn(3, base_filter * 2, 2, center=True, scale=True, name='2dconv1_0')
         .conv_gn(3, base_filter * 4, 2, center=True, scale=True, name='2dconv2_0')
         .conv_gn(3, base_filter * 8, 2, center=True, scale=True, name='2dconv3_0')
         .conv_gn(3, base_filter * 16, 2, center=True, scale=True, name='2dconv4_0'))
        """
        (self.feed('data')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv0_1')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv0_2'))
        """

        (self.feed('2dconv1_0')
         .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv1_1')
         .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv1_2'))

        (self.feed('2dconv2_0')
         .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv2_1')
         .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv2_2'))

        (self.feed('2dconv3_0')
         .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv3_1')
         .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv3_2'))

        (self.feed('2dconv4_0')
         .conv_gn(3, base_filter * 16, 1, center=True, scale=True, name='2dconv4_1')
         .conv_gn(3, base_filter * 16, 1, center=True, scale=True, name='2dconv4_2')
         .deconv_gn(3, base_filter * 8, 2, center=True, scale=True, name='2dconv5_0'))

        (self.feed('2dconv5_0', '2dconv3_2')
         .concat(axis=-1, name='2dconcat5_0')
         .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv5_1')
         .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv5_2')
         .deconv_gn(3, base_filter * 4, 2, center=True, scale=True, name='2dconv6_0'))

        (self.feed('2dconv6_0', '2dconv2_2')
         .concat(axis=-1, name='2dconcat6_0')
         .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv6_1')
         .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv6_2')
         .deconv_gn(3, base_filter * 2, 2, center=True, scale=True, name='2dconv7_0'))

        (self.feed('2dconv7_0', '2dconv1_2')
         .concat(axis=-1, name='2dconcat7_0')
         .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv7_1')
         .conv(3, base_filter * 2, 1, biased=False, relu=False, name='2dconv7_2'))



class UniNetDS(Network):
    """Simple UniNet, as described in the paper."""

    def setup(self):
        print ('2D with 32 filters')
        base_filter = 8
        (self.feed('data')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='conv0_0')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='conv0_1')
        .conv_gn(5, base_filter * 2, 2, center=True, scale=True, name='conv1_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='conv1_1')
        .conv(3, base_filter * 2, 1, biased=False, relu=False, name='conv1_2'))
