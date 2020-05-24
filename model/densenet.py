import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model

from model.layers import _activation
from model.layers import _normalization
from model.layers import ArcMarginPenaltyLogists
from model.attention import _se_block
from model.attention import _cbam_block


def dense_block(args, x, blocks, name):
    for i in range(blocks):
        x = conv_block(args, x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(args, x, reduction, name):
    channel = K.int_shape(x)[-1]
    x = _normalization(x, norm=args.norm, name=name+'_norm')
    x = _activation(x, activation=args.activation, name=name+'_acti')
    x = Conv2D(int(channel*reduction), 1, use_bias=False, name=name+'_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name+'_pool')(x)
    return x


def conv_block(args, x, growth_rate, name):
    x1 = _normalization(x, norm=args.norm, name=name+'_0_norm')
    x1 = _activation(x, activation=args.activation, name=name+'_0_acti')
    x1 = Conv2D(4*growth_rate, 1, use_bias=False, name=name+'_1_conv')(x1)
    x1 = _normalization(x1, norm=args.norm, name=name+'_1_norm')
    x1 = _activation(x1, activation=args.activation, name=name+'_1_acti')
    if args.attention == 'se':
        x1 = _se_block(x1, name=name + '_1_se')
    elif args.attention == 'cbam':
        x1 = _cbam_block(x1, name=name + '_1_cbam')

    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name+'_2_conv')(x1)
    x = Concatenate(name=name+'_concat')([x, x1])
    return x


def DenseNet(blocks, args, **kwargs):
    img_input = x = Input(shape=(args.img_size, args.img_size, args.img_channel), name='main_input')

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = _normalization(x, norm=args.norm, name='conv1/norm')
    x = _activation(x, activation=args.activation, name='conv1/acti')
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)
    
    x = dense_block(args, x, blocks[0], name='conv2')
    x = transition_block(args, x, 0.5, name='pool2')
    x = dense_block(args, x, blocks[1], name='conv3')
    x = transition_block(args, x, 0.5, name='pool3')
    x = dense_block(args, x, blocks[2], name='conv4')
    x = transition_block(args, x, 0.5, name='pool4')
    x = dense_block(args, x, blocks[3], name='conv5')

    x = _normalization(x, norm=args.norm, name='norm')
    x = _activation(x, activation=args.activation, name='acti')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    if args.embedding == 'softmax':
        x = Dense(args.classes, activation='softmax' if args.classes > 1 else 'sigmoid', name='main_output')(x)
        
        model_input = [img_input]
        model_output = [x]

    elif args.embedding == 'arcface':
        x = Dense(args.embd_shape, name='fc2')(x)
        x = _normalization(x, norm=args.norm, name='fc2_norm')

        label = Input(shape=(args.classes,), name='arcface_input')
        x = ArcMarginPenaltyLogists(num_classes=args.classes, margin=args.margin, logist_scale=args.logist_scale, name='arcface_output')(x, label)

        model_input = [img_input, label]
        model_output = [x]

    elif args.embedding == 'dual':
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

    model = Model(model_input, model_output, name='{}_{}'.format(args.backbone, args.embedding))

    return model


def DenseNet121(args, **kwargs):
    return DenseNet([6, 12, 24, 16], args, **kwargs)


def DenseNet169(args, **kwargs):
    return DenseNet([6, 12, 32, 32], args, **kwargs)


def DenseNet201(args, **kwargs):
    return DenseNet([6, 12, 48, 32], args, **kwargs)


def build_DenseNet(args, embd_shape=512, logist_scale=64, **kwargs):
    return eval('{}(args, **kwargs)'.format(
        args.backbone.replace('densenet', 'DenseNet')))