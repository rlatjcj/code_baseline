import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from model.layers import _activation
from model.layers import _normalization
from model.layers import ArcMarginPenaltyLogists
from model.attention import _se_block
from model.attention import _cbam_block


def VGG(args, **kwargs):
    total_layers = int(args.backbone[-2:])
    num_layers = {
        11: [1, 1, 2, 2, 2],
        13: [2, 2, 2, 2, 2],
        16: [2, 2, 3, 3, 3],
        19: [2, 2, 4, 4, 4]
    }

    filters = [1, 2, 4, 8, 8]

    img_input = x = Input(shape=(args.img_size, args.img_size, args.img_channel), name='main_input')
    for i, layers in enumerate(num_layers[total_layers]):
        for layer in range(layers):
            x = Conv2D(64*filters[i], (3, 3), padding='same', name='block{}_conv{}'.format(i+1, layer+1))(x)
            x = _normalization(x, norm=args.norm, name='block{}_norm{}'.format(i+1, layer+1))
            if layer == layers-1:
                if args.attention == 'se':
                    x = _se_block(x, name='block{}_se{}'.format(i+1, layer+1))
                elif args.attention == 'cbam':
                    x = _cbam_block(x, name='block{}_cbam{}'.format(i+1, layer+1))

            x = _activation(x, activation=args.activation, name='block{}_acti{}'.format(i+1, layer+1))

        x = MaxPooling2D((2, 2), name='block{}_pool'.format(i+1))(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, name='fc1')(x)
    x = _normalization(x, norm=args.norm, name='fc1_norm')
    x = _activation(x, activation=args.activation, name='fc1_acti')

    if args.embedding == 'softmax':
        x = Dense(4096, name='fc2')(x)
        x = _normalization(x, norm=args.norm, name='fc2_norm')
        x = _activation(x, activation=args.activation, name='fc2_acti')
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
