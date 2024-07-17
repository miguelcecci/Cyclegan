"""Modified version of pix2pix 
generator/discriminator models for 128x128 
images.

source code of original pix2pix models: 
    https://github.com/tensorflow/examples/
    blob/master/tensorflow_examples/models/
    pix2pix/pix2pix.py
"""

import tensorflow as tf


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def _upsample(filters, size, norm_type='batchnorm', dropout=False):

    """Upsamples an input.

        Conv2DTranspose => Batchnorm => Dropout => Relu

        Args:
            filters: number of filters
            size: filter size
            norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
            dropout: If True, adds the dropout layer

        Returns:
            Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
            tf.keras.layers.Conv2DTranspose(
                filters,
                size,
                strides=2,
                padding='same',
                kernel_initializer=initializer,
                use_bias=False))


    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def _downsample(filters, size, norm_type='batchnorm', apply_norm=True):

    """Downsamples an input.

        Conv2D => Batchnorm => LeakyRelu

        Args:
            filters: number of filters
            size: filter size
            norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
            apply_norm: If True, adds the batchnorm layer

        Returns:
            Downsample Sequential Model

    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
            tf.keras.layers.Conv2D(
                filters,
                size,
                strides=2,
                padding='same',
                kernel_initializer=initializer,
                use_bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def generator(norm_type='batchnorm'):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    down_stack = [
        _downsample(64, 4, norm_type=norm_type, apply_norm=False),  
        _downsample(128, 4, norm_type=norm_type),  
        _downsample(256, 4, norm_type=norm_type),  
        _downsample(512, 4, norm_type=norm_type),  
        _downsample(512, 4, norm_type=norm_type),  
        _downsample(512, 4, norm_type=norm_type),  
        _downsample(512, 4, norm_type=norm_type),  
    ]

    up_stack = [
        _upsample(512, 4, norm_type=norm_type, dropout=True),  
        _upsample(512, 4, norm_type=norm_type, dropout=True),  
        _upsample(512, 4, norm_type=norm_type, dropout=True),  
        _upsample(256, 4, norm_type=norm_type),  
        _upsample(128, 4, norm_type=norm_type),  
        _upsample(64, 4, norm_type=norm_type),  
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
            3, # number of channels
            4,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='tanh'
    )  

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    padding = [[14, 14], [4, 4], [0, 0]]

    zero_matrix = tf.zeros((100, 120, 3), dtype=tf.float32)
    padded_matrix = tf.pad(zero_matrix, padding, mode='CONSTANT', constant_values=1.0)
    
    ones_matrix = tf.ones((100, 120, 3), dtype=tf.float32)
    inv_padded_matrix = tf.pad(ones_matrix, padding, mode='CONSTANT', constant_values=0.0)

    ones_matrix = tf.keras.layers.Lambda(lambda inputs: inputs[0] * inputs[1])([inputs, padded_matrix])
    inv_padded_matrix = tf.keras.layers.Lambda(lambda inputs: inputs[0] * inputs[1])([x, inv_padded_matrix])

    x = tf.keras.layers.Add()([ones_matrix, inv_padded_matrix])

    return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator(norm_type='batchnorm', target=True):

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[128, 128, 3], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[128, 128, 3], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])  

    down1 = _downsample(64, 4, norm_type=norm_type, apply_norm=False)(x)  
    down2 = _downsample(128, 4, norm_type=norm_type)(down1)  

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2)  
    conv = tf.keras.layers.Conv2D(
        256, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)  

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)  

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)
