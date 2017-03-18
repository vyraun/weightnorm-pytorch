"""
Various tensorflow utilities
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

def get_var_maybe_avg(namespace, var_name, training, polyak_decay):
    ''' utility for retrieving polyak averaged params '''
    
    # Update average
    v = getattr(namespace, var_name)
    v_avg = getattr(namespace, var_name + '_avg')
    # if v_avg.sum() == 0:
    #     v_avg.copy_(v.data)
    # else:
    v_avg -= (1 - polyak_decay) * (v_avg - v.data)

    if training:
        return v
    else:
        return Variable(v_avg)

def get_vars_maybe_avg(namespace, var_names, training, polyak_decay):
    ''' utility for retrieving polyak averaged params '''
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(namespace, vn, training, polyak_decay))
    return vars

# def WN_Linear(nn.Module):
#     def __init__(self, in_features, out_features, init_scale=1., polyak_decay=0.9995):
#         super(Linear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         self.V = Parameter(torch.Tensor(out_features, in_features))
#         self.g = Parameter(torch.Tensor(out_features))
#         self.b = Parameter(torch.Tensor(out_features))

#         self.register_buffer('V_avg', torch.zeros(out_features, in_features))
#         self.register_buffer('g_avg', torch.zeros(out_features))
#         self.register_buffer('b_avg', torch.zeros(out_features))

#         self.init_scale = init_scale
#         self.polyak_decay = polyak_decay
#         self.reset_parameters()

#     def reset_parameters(self):
#         return
#         #stdv = 1. / math.sqrt(self.weight.size(1))
#         #self.V.data.copy_(torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05) 
#         #if self.bias is not None:
#         #    self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, x, init=False):
#         if init == True:
#             self.V.data.copy_(torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05)
#             V_norm = self.V.data / self.V.data.norm(2, 1).expandas(self.V.data) # in_features * out_features
#             x_init = F.linear(x, Variable(V_norm)).data # batch_size * out_features
#             m_init, v_init = x_init.mean(0).squeeze(0), x_init.var(0).squeeze(0) # out_features
#             scale_init = self.init_scale / torch.sqrt(v_init + 1e-10) # out_features
#             self.g.data.copy_(scale_init)
#             self.b.data.copy_(-m_init*scale_init)
#             x_init = scale_init.view(1, -1).expand_as(x_init)*(x_init-m_init.view(1, -1).expand_as(x_init))
#             return Variable(x_init)
#         else:
#             V,g,b = get_vars_maybe_avg(self, ['V','g','b'], self.training, polyak_decay=self.polyak_decay)
#             x = F.linear(x, V) # batch_size * out_features
#             scalar = g / torch.norm(V, 2, 1).squeeze(1)
#             x = scalar.view(1,-1).expand_as(x)*x + b.view(1,-1).expand_as(x)
#             return x


# def dense(x, num_units, nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
#     ''' fully connected layer '''
#     name = get_name('dense', counters)
#     with tf.variable_scope(name):
#         if init:
#             # data based initialization of parameters
#             V = tf.get_variable('V', [int(x.get_shape()[1]),num_units], tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
#             V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
#             x_init = tf.matmul(x, V_norm)
#             m_init, v_init = tf.nn.moments(x_init, [0])
#             scale_init = init_scale/tf.sqrt(v_init + 1e-10)
#             g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init, trainable=True)
#             b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init*scale_init, trainable=True)
#             x_init = tf.reshape(scale_init,[1,num_units])*(x_init-tf.reshape(m_init,[1,num_units]))
#             if nonlinearity is not None:
#                 x_init = nonlinearity(x_init)
#             return x_init

#         else:
#             V,g,b = get_vars_maybe_avg(['V','g','b'], ema)
#             tf.assert_variables_initialized([V,g,b])

#             # use weight normalization (Salimans & Kingma, 2016)
#             x = tf.matmul(x, V)
#             scaler = g/tf.sqrt(tf.reduce_sum(tf.square(V),[0]))
#             x = tf.reshape(scaler,[1,num_units])*x + tf.reshape(b,[1,num_units])

#             # apply nonlinearity
#             if nonlinearity is not None:
#                 x = nonlinearity(x)
#             return x

class WN_Linear(nn.Linear):
    def __init__(self, in_features, out_features, init_scale=1., polyak_decay=0.9995):
        super(WN_Linear, self).__init__(in_features, out_features, bias=True)

        self.V = self.weight
        self.g = Parameter(torch.Tensor(out_features))
        self.b = self.bias

        self.register_buffer('V_avg', torch.zeros(out_features, in_features))
        self.register_buffer('g_avg', torch.zeros(out_features))
        self.register_buffer('b_avg', torch.zeros(out_features))

        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.V.data.copy_(torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05) 
        #if self.bias is not None:
        #    self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, init=False):
        if init == True:
            self.V.data.copy_(torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05) # out_features * in_features
            V_norm = self.V.data / self.V.data.norm(2, 1).expand_as(self.V.data) # norm is out_features * 1
            x_init = F.linear(x, Variable(V_norm)).data # batch_size * out_features
            m_init, v_init = x_init.mean(0).squeeze(0), x_init.var(0).squeeze(0) # out_features
            scale_init = self.init_scale / torch.sqrt(v_init + 1e-10) # out_features
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init*scale_init)
            x_init = scale_init.view(1, -1).expand_as(x_init)*(x_init-m_init.view(1, -1).expand_as(x_init))
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return Variable(x_init)
        else:
            V,g,b = get_vars_maybe_avg(self, ['V','g','b'], self.training, polyak_decay=self.polyak_decay)
            x = F.linear(x, V) # batch_size * out_features
            scalar = g / torch.norm(V, 2, 1).squeeze(1)
            x = scalar.view(1,-1).expand_as(x)*x + b.view(1,-1).expand_as(x)
            return x

class WN_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, init_scale=1., polyak_decay=0.9995):
        super(WN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)

        self.V = self.weight # out_channels, in_channels // groups, *kernel_size
        self.g = Parameter(torch.Tensor(out_channels))
        self.b = self.bias

        self.register_buffer('V_avg', torch.zeros(self.V.size()))
        self.register_buffer('g_avg', torch.zeros(out_channels))
        self.register_buffer('b_avg', torch.zeros(out_channels))

        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.V.data.copy_(torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05) 
        #if self.bias is not None:
        #    self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, init=False):
        if init == True:
            self.V.data.copy_(torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05) # out_channels, in_channels // groups, *kernel_size
            V_norm = self.V.data / self.V.data.view(self.out_channels, -1).norm(2, 1).view(self.out_channels, *([1] * (len(self.kernel_size)+1))).expand_as(self.V.data) # norm is out_channels * 1
            x_init = F.conv2d(x, Variable(V_norm), None, self.stride,
                        self.padding, self.dilation, self.groups).data # batch_size * out_features
            t_x_init = x_init.transpose(0, 1).contiguous().view(self.out_channels, -1) # self.out_channels, 1
            m_init, v_init = t_x_init.mean(1).squeeze(1), t_x_init.var(1).squeeze(1) # out_features
            scale_init = self.init_scale / torch.sqrt(v_init + 1e-10) # out_features
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init*scale_init)
            x_init = scale_init.view(1, self.out_channels, *([1] * (len(x_init.size()) - 2))).expand_as(x_init)\
                *(x_init-m_init.view(1, self.out_channels, *([1] * (len(x_init.size()) - 2))).expand_as(x_init))
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return Variable(x_init)
        else:
            V,g,b = get_vars_maybe_avg(self, ['V','g','b'], self.training, polyak_decay=self.polyak_decay)
            # use weight normalization (Salimans & Kingma, 2016)
            scalar = g / torch.norm(V.view(self.out_channels, -1), 2, 1).squeeze(1)
            W = scalar.view(self.out_channels, *([1] * (len(V.size()) - 1))).expand_as(V)*V

            x = F.conv2d(x, W, b, self.stride,\
                        self.padding, self.dilation, self.groups)
            return x

# @add_arg_scope
# def conv2d(x, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
#     ''' convolutional layer '''
#     name = get_name('conv2d', counters)
#     with tf.variable_scope(name):
#         if init:
#             # data based initialization of parameters
#             V = tf.get_variable('V', filter_size+[int(x.get_shape()[-1]),num_filters], tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
#             V_norm = tf.nn.l2_normalize(V.initialized_value(), [0,1,2])
#             x_init = tf.nn.conv2d(x, V_norm, [1]+stride+[1], pad)
#             m_init, v_init = tf.nn.moments(x_init, [0,1,2])
#             scale_init = init_scale/tf.sqrt(v_init + 1e-8)
#             g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init, trainable=True)
#             b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init*scale_init, trainable=True)
#             x_init = tf.reshape(scale_init,[1,1,1,num_filters])*(x_init-tf.reshape(m_init,[1,1,1,num_filters]))
#             if nonlinearity is not None:
#                 x_init = nonlinearity(x_init)
#             return x_init

#         else:
#             V, g, b = get_vars_maybe_avg(['V', 'g', 'b'], ema)
#             tf.assert_variables_initialized([V,g,b])

#             # use weight normalization (Salimans & Kingma, 2016)
#             W = tf.reshape(g,[1,1,1,num_filters])*tf.nn.l2_normalize(V,[0,1,2])

#             # calculate convolutional layer output
#             x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1]+stride+[1], pad), b)

#             # apply nonlinearity
#             if nonlinearity is not None:
#                 x = nonlinearity(x)
#             return x

class WN_ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, init_scale=1., polyak_decay=0.9995):
        super(WN_ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups)

        self.V = self.weight # in_channels, out_channels, *kernel_size
        self.g = Parameter(torch.Tensor(out_channels))
        self.b = self.bias

        self.register_buffer('V_avg', torch.zeros(self.V.size()))
        self.register_buffer('g_avg', torch.zeros(out_channels))
        self.register_buffer('b_avg', torch.zeros(out_channels))

        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.V.data.copy_(torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05) 
        #if self.bias is not None:
        #    self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, init=False):
        if init == True:
            self.V.data.copy_(torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05) # in_channels, out_channels, *kernel_size
            V_norm = self.V.data / self.V.data.transpose(0,1).contiguous().view(self.out_channels, -1).norm(2, 1).view(self.in_channels, self.out_channels, *([1] * len(self.kernel_size))).expand_as(self.V.data) # norm is out_channels * 1
            x_init = F.conv_transpose2d(x, Variable(V_norm), None, self.stride,
                        self.padding, self.output_padding, self.groups).data # batch_size * out_features
            t_x_init = x_init.tranpose(0, 1).contiguous().view(self.out_channels, -1) # self.out_channels, 1
            m_init, v_init = t_x_init.mean(1).squeeze(1), t_x_init.var(1).squeeze(1) # out_features
            scale_init = self.init_scale / torch.sqrt(v_init + 1e-10) # out_features
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init*scale_init)
            x_init = scale_init.view(1, self.out_channels, *([1] * (len(x_init.size()) - 2))).expand_as(x_init)\
                *(x_init-m_init.view(1, self.out_channels, *([1] * (len(x_init.size()) - 2))).expand_as(x_init))
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return Variable(x_init)
        else:
            V,g,b = get_vars_maybe_avg(self, ['V','g','b'], self.training, polyak_decay=self.polyak_decay)
            # use weight normalization (Salimans & Kingma, 2016)
            scalar = g / torch.norm(V.transpose(0,1).contiguous().view(self.out_channels, -1), 2, 1).squeeze(1)
            W = scalar.view(self.in_channels, self.out_channels, *([1] * (len(V.size()) - 2))).expand_as(V)*V

            x = F.conv_transpose2d(x, W, b, self.stride,\
                        self.padding, self.output_padding, self.groups)
            return x

# @add_arg_scope
# def deconv2d(x, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
#     ''' transposed convolutional layer '''
#     name = get_name('deconv2d', counters)
#     xs = int_shape(x)
#     if pad=='SAME':
#         target_shape = [xs[0], xs[1]*stride[0], xs[2]*stride[1], num_filters]
#     else:
#         target_shape = [xs[0], xs[1]*stride[0] + filter_size[0]-1, xs[2]*stride[1] + filter_size[1]-1, num_filters]
#     with tf.variable_scope(name):
#         if init:
#             # data based initialization of parameters
#             V = tf.get_variable('V', filter_size+[num_filters,int(x.get_shape()[-1])], tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
#             V_norm = tf.nn.l2_normalize(V.initialized_value(), [0,1,3])
#             x_init = tf.nn.conv2d_transpose(x, V_norm, target_shape, [1]+stride+[1], padding=pad)
#             m_init, v_init = tf.nn.moments(x_init, [0,1,2])
#             scale_init = init_scale/tf.sqrt(v_init + 1e-8)
#             g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init, trainable=True)
#             b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init*scale_init, trainable=True)
#             x_init = tf.reshape(scale_init,[1,1,1,num_filters])*(x_init-tf.reshape(m_init,[1,1,1,num_filters]))
#             if nonlinearity is not None:
#                 x_init = nonlinearity(x_init)
#             return x_init

#         else:
#             V, g, b = get_vars_maybe_avg(['V', 'g', 'b'], ema)
#             tf.assert_variables_initialized([V,g,b])

#             # use weight normalization (Salimans & Kingma, 2016)
#             W = tf.reshape(g,[1,1,num_filters,1])*tf.nn.l2_normalize(V,[0,1,3])

#             # calculate convolutional layer output
#             x = tf.nn.conv2d_transpose(x, W, target_shape, [1]+stride+[1], padding=pad)
#             x = tf.nn.bias_add(x, b)

#             # apply nonlinearity
#             if nonlinearity is not None:
#                 x = nonlinearity(x)
#             return x
