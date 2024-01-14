import tensorflow as tf
from tensorflow.keras import layers

class GaussianBlur(layers.Layer):
    def __init__(self, filter_size=3, sigma=1., **kwargs):
        super().__init__(**kwargs)
        self.filter_size = filter_size
        self.sigma = sigma

    def build(self, input_shape):
        # Create a 2D Gaussian filter.
        x = tf.range(-self.filter_size // 2, self.filter_size // 2 + 1, dtype=tf.float32)
        x = tf.exp(-0.5 * (x / self.sigma) ** 2)
        x /= tf.reduce_sum(x)
        gaussian_filter = tf.tensordot(x, x, axes=0)
        gaussian_filter = tf.reshape(gaussian_filter, (*gaussian_filter.shape, 1, 1))
        n_channels = input_shape[-1]
        self.gaussian_filter = tf.repeat(gaussian_filter, n_channels, axis=-2)

    def call(self, inputs):
        return tf.nn.depthwise_conv2d(inputs, self.gaussian_filter, strides=[1, 1, 1, 1], padding='SAME')