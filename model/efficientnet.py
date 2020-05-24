"""EfficientNet models for Keras.

# Reference paper

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks]
  (https://arxiv.org/abs/1905.11946) (ICML 2019)

# Reference implementation

- [TensorFlow]
  (https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
"""

import math
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model

from model.layers import _activation
from model.layers import _normalization
from model.layers import ArcMarginPenaltyLogists
from model.attention import _se_block
from model.attention import _cbam_block


DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1}
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim+2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def block(args, inputs, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True):

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = Conv2D(filters, 1, padding='same', use_bias=False,
                   kernel_initializer=CONV_KERNEL_INITIALIZER, 
                   name=name+'expand_conv')(inputs)
        x = _normalization(x, norm=args.norm, name=name+'expand_norm')
        x = _activation(x, activation=args.activation, name=name+'expand_acti')
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = ZeroPadding2D(padding=correct_pad(x, kernel_size), name=name+'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'

    x = DepthwiseConv2D(kernel_size, strides=strides, padding=conv_pad, use_bias=False,
                        depthwise_initializer=CONV_KERNEL_INITIALIZER,
                        name=name+'dwconv')(x)
    x = _normalization(x, norm=args.norm, name=name+'norm')
    x = _activation(x, activation=args.activation, name=name+'acti')

    if args.attention == 'se':
        x = _se_block(x, name=name+'se')
    elif args.attention == 'cbam':
        x = _cbam_block(x, name=name+'cbam')

    # Output phase
    x = Conv2D(filters_out, 1, padding='same', use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               name=name+'project_conv')(x)
    x = _normalization(x, norm=args.norm, name=name+'project_norm')
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            x = Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name+'drop')(x)
        x = Add(name=name+'add')([x, inputs])

    return x


def EfficientNet(args, 
                 width_coefficient,
                 depth_coefficient,
                 default_size,
                 dropout_rate,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 **kwargs):
                 
    img_input = x = Input(shape=(args.img_size, args.img_size, args.img_channel), name='main_input')

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters+divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    x = ZeroPadding2D(padding=correct_pad(x, 3), name='stem_conv_pad')(x)
    x = Conv2D(round_filters(32), 3, strides=2, padding='valid', use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER, name='stem_conv')(x)
    x = _normalization(x, norm=args.norm, name='stem_norm')
    x = _activation(x, activation=args.activation, name='stem_acti')

    # Build blocks
    from copy import deepcopy
    blocks_args = deepcopy(blocks_args)

    b = 0
    blocks = float(sum(ba['repeats'] for ba in blocks_args))
    for (i, ba) in enumerate(blocks_args):
        assert ba['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        ba['filters_in'] = round_filters(ba['filters_in'])
        ba['filters_out'] = round_filters(ba['filters_out'])

        for j in range(round_repeats(ba.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                ba['strides'] = 1
                ba['filters_in'] = ba['filters_out']
            x = block(args, x, drop_connect_rate*b/blocks,
                      name='block{}{}_'.format(i+1, chr(j+97)), **ba)
            b += 1

    # Build top
    x = Conv2D(round_filters(1280), 1, padding='same', use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER, name='top_conv')(x)
    x = _normalization(x, norm=args.norm, name='top_norm')
    x = _activation(x, activation=args.activation, name='top_acti')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name='top_dropout')(x)

    if args.embedding == 'softmax':
        x = Dense(args.classes, activation='softmax' if args.classes > 1 else 'sigmoid', name='main_output')(x)
        model_input = [img_input]
        model_output = [x]

    elif args.embedding == 'arcface':
        # x = _normalization(x, norm=args.norm, name='avg_pool_norm')
        x = Dense(args.embd_shape, name='fc2')(x)
        x = _normalization(x, norm=args.norm, name='fc2_norm')

        label = Input(shape=(args.classes,), name='arcface_input')
        x = ArcMarginPenaltyLogists(num_classes=args.classes, margin=args.margin, logist_scale=args.logist_scale, name='arcface_output')(x, label)
        model_input = [img_input, label]
        model_output = [x]

    elif args.embedding == 'dual':
        # bengali 8th model
        # x = _normalization(x, norm=args.norm, name='avg_pool_norm')
        x = Dense(args.embd_shape, name='fc2')(x)
        x = _normalization(x, norm=args.norm, name='fc2_norm')

        x1 = _activation(x, activation=args.activation, name='fc2_acti')
        x1 = Dense(args.classes, activation='softmax' if args.classes > 1 else 'sigmoid', name='main_output')(x1)

        label = Input(shape=(args.classes,), name='arcface_input')
        x2 = ArcMarginPenaltyLogists(num_classes=args.classes, margin=args.margin, logist_scale=args.logist_scale, name='arcface_output')(x, label)
        model_input = [img_input, label]
        model_output = [x1, x2]

    else:
        raise ValueError()

    # Create model.
    model = Model(model_input, model_output, name='{}_{}'.format(args.backbone, args.embedding))

    return model


def EfficientNetB0(args, **kwargs):
    return EfficientNet(args, 1.0, 1.0, 224, 0.2, **kwargs)


def EfficientNetB1(args, **kwargs):
    return EfficientNet(args, 1.0, 1.1, 240, 0.2, **kwargs)


def EfficientNetB2(args, **kwargs):
    return EfficientNet(args, 1.1, 1.2, 260, 0.3, **kwargs)


def EfficientNetB3(args, **kwargs):
    return EfficientNet(args, 1.2, 1.4, 300, 0.3, **kwargs)


def EfficientNetB4(args, **kwargs):
    return EfficientNet(args, 1.4, 1.8, 380, 0.4, **kwargs)


def EfficientNetB5(args, **kwargs):
    return EfficientNet(args, 1.6, 2.2, 456, 0.4, **kwargs)


def EfficientNetB6(args, **kwargs):
    return EfficientNet(args, 1.8, 2.6, 528, 0.5, **kwargs)


def EfficientNetB7(args, **kwargs):
    return EfficientNet(args, 2.0, 3.1, 600, 0.5, **kwargs)

def build_EfficientNet(args, **kwargs):
    return eval('{}(args, **kwargs)'.format(
        args.backbone.replace('efficientnetb', 'EfficientNetB')))