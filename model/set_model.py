import tensorflow as tf
from tensorflow.keras.utils import get_file

from model.vgg import VGG
from model.resnet import build_ResNet
from model.densenet import build_DenseNet
from model.efficientnet import build_EfficientNet


WEIGHTS_HASHES = {
    'vgg16'             : '6d6bbae143d832006294945121d1f1fc',
    'vgg19'             : '253f8cb515780f3b799900260a226db6',

    'resnet50'          : '4d473c1dd8becc155b73f8504c6f6626',
    'resnet101'         : '88cf7a10940856eca736dc7b7e228a21',
    'resnet152'         : 'ee4c566cf9a93f14d82f913c2dc6dd0c',
    'resnet50v2'        : 'fac2f116257151a9d068a22e544a4917',
    'resnet101v2'       : 'c0ed64b8031c3730f411d2eb4eea35b5',
    'resnet152v2'       : 'ed17cf2e0169df9d443503ef94b23b33',
    'resnext50'         : '62527c363bdd9ec598bed41947b379fc',
    'resnext101'        : '0f678c91647380debd923963594981b3',

    'densenet121'       : '30ee3e1110167f948a6b9946edeeb738',
    'densenet169'       : 'b8c4d4c20dd625c148057b9ff1c1176b',
    'densenet201'       : 'c13680b51ded0fb44dff2d8f86ac8bb1',

    'efficientnetb0'    : '345255ed8048c2f22c793070a9c1a130',
    'efficientnetb1'    : 'b20160ab7b79b7a92897fcb33d52cc61',
    'efficientnetb2'    : 'c6e46333e8cddfa702f4d8b8b6340d70',
    'efficientnetb3'    : 'e0cf8654fad9d3625190e30d70d0c17d',
    'efficientnetb4'    : 'b46702e4754d2022d62897e0618edc7b',
    'efficientnetb5'    : '0a839ac36e46552a881f2975aaab442f',
    'efficientnetb6'    : '375a35c17ef70d46f9c664b03b4437f2',
    'efficientnetb7'    : 'd55674cc46b805f4382d18bc08ed43c1',
}


def download_imagenet(args):
    if 'vgg' in args.backbone:
        BASE_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/'

    elif 'res' in args.backbone:
        BASE_WEIGHTS_PATH = 'https://github.com/keras-team/keras-applications/releases/download/resnet/'

    elif 'dense' in args.backbone:
        BASE_WEIGHTS_PATH = 'https://github.com/keras-team/keras-applications/releases/download/densenet/'

    elif 'efficient' in args.backbone:
        BASE_WEIGHTS_PATH = 'https://github.com/Callidior/keras-applications/releases/download/efficientnet/'

    else:
        raise ValueError("Backbone '{}' not recognized.".format(args.backbone))

    if 'efficient' in args.backbone:
        file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(args.backbone.replace('b', '-b'))
    else:
        file_name = '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(args.backbone)

    return get_file(
        file_name,
        BASE_WEIGHTS_PATH + file_name,
        cache_subdir='models',
        file_hash=WEIGHTS_HASHES[args.backbone]
    )


def build_model(args):
    if 'vgg' in args.backbone:
        return VGG(args)

    elif 'res' in args.backbone:
        return build_ResNet(args)
    
    elif 'dense' in args.backbone:
        return build_DenseNet(args)

    elif 'efficient' in args.backbone:
        return build_EfficientNet(args)


def Backbone(args, logger):
    model = build_model(args)

    if args.snapshot:
        model.load_weights(args.snapshot)
        logger.info('Load weights at {}!'.format(args.snapshot))

    elif args.weights == 'imagenet':
        assert args.img_channel == 3, 'Pretrained imagenet weights must have 3 input channels.'

        model.load_weights(download_imagenet(args), by_name=True)
        logger.info('Load pretrained imagenet weights!')

    elif args.weights == 'best':
        # TODO:
        # 1. Find the stamp that have same configuration with this training
        # 2. Find the best weights
        # 3. Set the initial epoch to the last epoch in the previous training
        pass


    if args.freeze_backbone:
        for layer in model.layers:
            try:
                if int(layer.name.split('_')[0][-1]) <= args.freeze_backbone:
                    layer.trainable = False
                    logger.info('{} is freeze.'.format(layer.name))
            except:
                pass

    return model