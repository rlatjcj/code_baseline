import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.models import Model

from model.layers import _activation
from model.layers import _normalization
from model.layers import ArcMarginPenaltyLogists
from model.attention import _se_block
from model.attention import _cbam_block


def block1(args, x, filters, kernel_size=3, stride=1, conv_shortcut=True, attention='no', name=None):
    if conv_shortcut is True:
        shortcut = Conv2D(4 * filters, 1, strides=stride, name=name+'_0_conv')(x)
        shortcut = _normalization(shortcut, norm=args.norm, name=name+'_0_norm')
    else:
        shortcut = x

    x = Conv2D(filters, 1, strides=stride, name=name+'_1_conv')(x)
    x = _normalization(x, norm=args.norm, name=name+'_1_norm')
    x = _activation(x, activation=args.activation, name=name+'_1_acti')

    x = Conv2D(filters, kernel_size, padding='same', name=name+'_2_conv')(x)
    x = _normalization(x, norm=args.norm, name=name+'_2_norm')
    x = _activation(x, activation=args.activation, name=name+'_2_acti')

    x = Conv2D(4 * filters, 1, name=name+'_3_conv')(x)
    x = _normalization(x, norm=args.norm, name=name+'_3_norm')
    if args.attention == 'se':
        x = _se_block(x, name=name+'_3_se')
    elif args.attention == 'cbam':
        x = _cbam_block(x, name=name+'_3_cbam')
    
    x = Add(name=name+'_add')([shortcut, x])
    x = _activation(x, activation=args.activation, name=name+'_3_acti')
    return x


def stack1(args, x, filters, blocks, stride1=2, name=None):
    x = block1(args, x, filters, stride=stride1, attention=args.attention, name=name+'_block1')
    for i in range(2, blocks + 1):
        x = block1(args, x, filters, conv_shortcut=False, attention=args.attention, name=name+'_block' + str(i))
    return x


def block2(args, x, filters, kernel_size=3, stride=1, conv_shortcut=False, attention='no', name=None):
    preact = _normalization(x, norm=args.norm, name=name+'_pre_norm')
    preact = _activation(preact, activation=args.activation, name=name+'_pre_acti')

    if conv_shortcut is True:
        shortcut = Conv2D(4 * filters, 1, strides=stride, name=name+'_0_conv')(preact)
    else:
        shortcut = MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = Conv2D(filters, 1, strides=1, use_bias=False, name=name+'_1_conv')(preact)
    x = _normalization(x, norm=args.norm, name=name+'_1_norm')
    x = _activation(x, activation=args.activation, name=name+'_1_acti')

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name+'_2_pad')(x)
    x = Conv2D(filters, kernel_size, strides=stride, use_bias=False, name=name+'_2_conv')(x)
    x = _normalization(x, norm=args.norm, name=name+'_2_norm')
    x = _activation(x, activation=args.activation, name=name+'_2_acti')

    x = Conv2D(4 * filters, 1, name=name+'_3_conv')(x)
    if args.attention == 'se':
        x = _se_block(x, name=name+'_3_se')
    elif args.attention == 'cbam':
        x = _cbam_block(x, name=name+'_3_cbam')
        
    x = Add(name=name+'_out')([shortcut, x])
    return x


def stack2(args, x, filters, blocks, stride1=2, name=None):
    x = block2(args, x, filters, conv_shortcut=True, attention=args.attention, name=name+'_block1')
    for i in range(2, blocks):
        x = block2(args, x, filters, attention=args.attention, name=name+'_block' + str(i))
    x = block2(args, x, filters, stride=stride1, attention=args.attention, name=name+'_block' + str(blocks))
    return x


def block3(args, x, filters, kernel_size=3, stride=1, groups=32, conv_shortcut=True, attention='no', name=None):
    if conv_shortcut is True:
        shortcut = Conv2D((64 // groups) * filters, 1, strides=stride, use_bias=False, name=name+'_0_conv')(x)
        shortcut = _normalization(shortcut, norm=args.norm, name=name+'_0_norm')
    else:
        shortcut = x

    x = Conv2D(filters, 1, use_bias=False, name=name+'_1_conv')(x)
    x = _normalization(x, norm=args.norm, name=name+'_1_norm')
    x = _activation(x, activation=args.activation, name=name+'_1_acti')

    c = filters // groups
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name+'_2_pad')(x)
    x = DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=c, use_bias=False, name=name+'_2_conv')(x)
    kernel = np.zeros((1, 1, filters * c, filters), dtype=np.float32)
    for i in range(filters):
        start = (i // c) * c * c + i % c
        end = start + c * c
        kernel[:, :, start:end:c, i] = 1.
        
    x = Conv2D(filters, 1, use_bias=False, trainable=False,
               kernel_initializer={
                   'class_name': 'Constant',
                   'config': {'value': kernel}
                },
               name=name+'_2_gconv')(x)
    x = _normalization(x, norm=args.norm, name=name+'_2_norm')
    x = _activation(x, activation=args.activation, name=name+'_2_acti')

    x = Conv2D((64 // groups) * filters, 1, use_bias=False, name=name+'_3_conv')(x)
    x = _normalization(x, norm=args.norm, name=name+'_3_norm')
    if args.attention == 'se':
        x = _se_block(x, name=name+'_3_se')
    elif args.attention == 'cbam':
        x = _cbam_block(x, name=name+'_3_cbam')

    x = Add(name=name+'_add')([shortcut, x])
    x = _activation(x, activation=args.activation, name=name+'_3_acti')
    return x


def stack3(args, x, filters, blocks, stride1=2, groups=32, name=None):
    x = block3(args, x, filters, stride=stride1, groups=groups, attention=args.attention, name=name+'_block1')
    for i in range(2, blocks + 1):
        x = block3(args, x, filters, groups=groups, conv_shortcut=False, attention=args.attention, name=name+'_block' + str(i))
    return x


def ResNet(args, embd_shape, logist_scale, stack_fn, preact, use_bias, **kwargs):
    img_input = x = Input(shape=(args.img_size, args.img_size, args.img_channel), name='main_input')

    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
    x = Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if preact is False:
        x = _normalization(x, norm=args.norm, name='conv1_norm')
        x = _activation(x, activation=args.activation, name='conv1_acti')

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact is True:
        x = _normalization(x, norm=args.norm, name='post_norm')
        x = _activation(x, activation=args.activation, name='post_acti')

    x = GlobalAveragePooling2D(name='avg_pool')(x)

    if args.embedding == 'softmax':
        x = Dense(args.classes, activation='softmax' if args.classes > 1 else 'sigmoid', name='main_output')(x)
        model_input = [img_input]
        model_output = [x]

    elif args.embedding == 'arcface':
        x = _normalization(x, norm=args.norm, name='avg_pool_norm')
        x = Flatten(name='flatten')(x)
        x = Dense(args.embd_shape, kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='fc2')(x)
        x = _normalization(x, norm=args.norm, name='fc2_norm')

        label = Input(shape=(args.classes,), name='arcface_input')
        x = ArcMarginPenaltyLogists(num_classes=args.classes, margin=args.margin, logist_scale=args.logist_scale, name='arcface_output')(x, label)
        model_input = [img_input, label]
        model_output = [x]

    elif args.embedding == 'dual':
        # bengali 8th model
        x = Dense(args.embd_shape, name='fc2')(x)
        x = _activation(x, activation=args.activation, name='avg_pool_acti')
        
        x1 = Dense(args.classes, activation='softmax' if args.classes > 1 else 'sigmoid', name='main_output')(x)

        label = Input(shape=(args.classes,), name='arcface_input')
        x2 = ArcMarginPenaltyLogists(num_classes=args.classes, margin=args.margin, logist_scale=args.logist_scale, name='arcface_output')(x, label)
        model_input = [img_input, label]
        model_output = [x1, x2]

    else:
        raise ValueError()

    model = Model(model_input, model_output, name='{}_{}'.format(args.backbone, args.embedding))

    return model


def ResNet50(args, embd_shape, logist_scale, **kwargs):
    def stack_fn(x):
        x = stack1(args, x, 64, 3, stride1=1, name='conv2')
        x = stack1(args, x, 128, 4, name='conv3')
        x = stack1(args, x, 256, 6, name='conv4')
        x = stack1(args, x, 512, 3, name='conv5')
        return x
    return ResNet(args, embd_shape, logist_scale, stack_fn, False, True, **kwargs)


def ResNet101(args, embd_shape, logist_scale, **kwargs):
    def stack_fn(x):
        x = stack1(args, x, 64, 3, stride1=1, name='conv2')
        x = stack1(args, x, 128, 4, name='conv3')
        x = stack1(args, x, 256, 23, name='conv4')
        x = stack1(args, x, 512, 3, name='conv5')
        return x
    return ResNet(args, embd_shape, logist_scale, stack_fn, False, True, **kwargs)


def ResNet152(args, embd_shape, logist_scale, **kwargs):
    def stack_fn(x):
        x = stack1(args, x, 64, 3, stride1=1, name='conv2')
        x = stack1(args, x, 128, 8, name='conv3')
        x = stack1(args, x, 256, 36, name='conv4')
        x = stack1(args, x, 512, 3, name='conv5')
        return x
    return ResNet(args, embd_shape, logist_scale, stack_fn, False, True, **kwargs)


def ResNet50V2(args, embd_shape, logist_scale, **kwargs):
    def stack_fn(x):
        x = stack2(args, x, 64, 3, name='conv2')
        x = stack2(args, x, 128, 4, name='conv3')
        x = stack2(args, x, 256, 6, name='conv4')
        x = stack2(args, x, 512, 3, stride1=1, name='conv5')
        return x
    return ResNet(args, embd_shape, logist_scale, stack_fn, True, True, **kwargs)


def ResNet101V2(args, embd_shape, logist_scale, **kwargs):
    def stack_fn(x):
        x = stack2(args, x, 64, 3, name='conv2')
        x = stack2(args, x, 128, 4, name='conv3')
        x = stack2(args, x, 256, 23, name='conv4')
        x = stack2(args, x, 512, 3, stride1=1, name='conv5')
        return x
    return ResNet(args, embd_shape, logist_scale, stack_fn, True, True, **kwargs)


def ResNet152V2(args, embd_shape, logist_scale, **kwargs):
    def stack_fn(x):
        x = stack2(args, x, 64, 3, name='conv2')
        x = stack2(args, x, 128, 8, name='conv3')
        x = stack2(args, x, 256, 36, name='conv4')
        x = stack2(args, x, 512, 3, stride1=1, name='conv5')
        return x
    return ResNet(args, embd_shape, logist_scale, stack_fn, True, True, **kwargs)


def ResNeXt50(args, embd_shape, logist_scale, **kwargs):
    def stack_fn(x):
        x = stack3(args, x, 128, 3, stride1=1, name='conv2')
        x = stack3(args, x, 256, 4, name='conv3')
        x = stack3(args, x, 512, 6, name='conv4')
        x = stack3(args, x, 1024, 3, name='conv5')
        return x
    return ResNet(args, embd_shape, logist_scale, stack_fn, False, False, **kwargs)


def ResNeXt101(args, embd_shape, logist_scale, **kwargs):
    def stack_fn(x):
        x = stack3(args, x, 128, 3, stride1=1, name='conv2')
        x = stack3(args, x, 256, 4, name='conv3')
        x = stack3(args, x, 512, 23, name='conv4')
        x = stack3(args, x, 1024, 3, name='conv5')
        return x
    return ResNet(args, embd_shape, logist_scale, stack_fn, False, False, **kwargs)


def build_ResNet(args, embd_shape=512, logist_scale=64, **kwargs):
    if 'resnet' in args.backbone:
        if 'v2' in args.backbone:
            return eval('{}(args, embd_shape, logist_scale, **kwargs)'.format(
                args.backbone.replace('resnet', 'ResNet').replace('v2', 'V2')))
        else:
            return eval('{}(args, embd_shape, logist_scale, **kwargs)'.format(
                args.backbone.replace('resnet', 'ResNet')))
    else:
        return eval('{}(args, embd_shape, logist_scale, **kwargs)'.format(
            args.backbone.replace('resnext', 'ResNeXt')))