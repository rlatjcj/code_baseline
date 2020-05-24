import math
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LayerNormalization
from tensorflow_addons.layers import InstanceNormalization
from tensorflow_addons.layers import GroupNormalization
from tensorflow_addons.layers import GELU

if float(tf.__version__[:3]) == 2.2:
    from tensorflow.keras.layers.experimental import SyncBatchNormalization

def _normalization(inputs, norm='bn', name=None):
    if norm == 'bn':
        return BatchNormalization(name=name)(inputs)
    elif norm == 'syncbn':
        return SyncBatchNormalization(name=name)(inputs)
    elif norm == 'in':
        return InstanceNormalization(name=name)(inputs)
    elif norm == 'ln':
        return LayerNormalization(name=name)(inputs)
    elif 'gn' in norm:
        if len(norm) == 2:
            return GroupNormalization(name=name)(inputs)
        else:
            return GroupNormalization(groups=int(norm[2:]), name=name)(inputs)
        

def _activation(inputs, activation='relu', name=None):
    if activation == 'sigmoid':
        return Activation(activation, name=name)(inputs)
    elif activation == 'leakyrelu':
        return LeakyReLU(alpha=.3, name=name)(inputs)
    elif activation == 'gelu':
        return GELU(name=name)(inputs)
    else:
        return Activation(eval('tf.nn.{}'.format(activation)), name=name)(inputs)


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.w = self.add_weight(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        # loss
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')
        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        logists = tf.where(labels == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')
        logists = tf.nn.softmax(logists)

        return logists