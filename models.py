import numpy as np
import tensorflow as tf
import cv2
import os
import time
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output

# Using instancenormalization:
'''
https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py
'''
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K


class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


### Resnet block
def resnet_block(inp_layer,filters):
    c1 = Conv2D(filters=filters, kernel_size=3, strides=1,padding='same')(inp_layer)
    c1 = InstanceNormalization(axis=-1)(c1)
    c1 = Activation('relu')(c1)

    c2 = Conv2D(filters=filters, kernel_size=3, strides=1,padding='same')(c1)
    c2 = InstanceNormalization(axis=-1)(c2)
    c2 = concatenate([c2,inp_layer])
    return c2

#############################################
#############################################

def Generator():
    input_layer = Input(shape=[128,128,3])

    c = Conv2D(filters=64, kernel_size=7, strides=1,padding='same')(input_layer) #(bs, 128, 128, 64)
    bat_norm = InstanceNormalization(axis=-1)(c)
    leaky_relu = Activation('relu')(bat_norm)

    c = Conv2D(128,(3,3),strides=(2,2),padding='same')(leaky_relu)
    inst_norm = InstanceNormalization(axis=-1)(c)
    ac = Activation('relu')(inst_norm)

    c = Conv2D(256, (3,3),strides=(2,2),padding='same')(ac)
    inst_norm = InstanceNormalization(axis=-1)(c)
    c = Activation('relu')(inst_norm)

    #Resnet
    for i in range(6):
        c = resnet_block(c,256)

    ct = Conv2DTranspose(128, (3,3), strides=(2,2),padding='same')(c)
    inst_norm = InstanceNormalization(axis=-1)(ct)
    c = Activation('relu')(inst_norm)

    ct = Conv2DTranspose(64, (3,3), strides=(2,2),padding='same')(c)
    inst_norm = InstanceNormalization(axis=-1)(ct)
    c = Activation('relu')(inst_norm)

    last = Conv2DTranspose(filters=3,kernel_size=(7,7), padding='same')(c)
    inst_norm = InstanceNormalization(axis=-1)(last)
    c = Activation('tanh')(inst_norm)

    return tf.keras.models.Model(inputs = input_layer,outputs=c)


###########################################
def Discriminator():
    input_layer = Input(shape=[128,128,3])
    # downsampling
    conv1 = Conv2D(filters=64, kernel_size=4, strides=2,padding='same',activation='relu')(input_layer) # (bs, 64, 64, 64)
    leaky1 = LeakyReLU()(conv1)

    conv2 = Conv2D(filters=128,kernel_size=4, strides=2,activation='relu',padding='same')(leaky1) #(bs, 32, 32, 128)
    bat_norm = InstanceNormalization(axis=-1)(conv2)
    leaky2 = LeakyReLU()(bat_norm)

    conv3 = Conv2D(filters=256,kernel_size=4, strides=2,activation='relu',padding='same')(leaky2) #(bs, 16, 16, 256)
    bat_norm = InstanceNormalization(axis=-1)(conv3)
    leaky3 = LeakyReLU()(bat_norm)

    zero_pad1 = ZeroPadding2D()(leaky3)
    conv = Conv2D(filters=512, kernel_size=4, strides=1, use_bias=False)(zero_pad1)

    batch_norm = InstanceNormalization(axis=-1)(conv)

    leaky_relu = LeakyReLU()(batch_norm)

    zero_pad2 = ZeroPadding2D()(leaky_relu)

    last = Conv2D(1, 4, strides=1)(zero_pad2)

    return tf.keras.Model(inputs=input_layer, outputs=last)
