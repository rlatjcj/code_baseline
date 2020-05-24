import os
import sys
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
SEED = 414
np.random.seed(SEED)
tf.random.set_seed(SEED)


def check_arguments(args):
    assert args.src_path is not None, 'src_path must be entered.'
    assert args.data_path is not None, 'data_path must be entered.'
    assert args.result_path is not None, 'result_path must be entered.'

    return args

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",           type=str,       default='supervised',   choices=['supervised', 'fixmatch'])
    parser.add_argument("--backbone",       type=str,       default=None)
    parser.add_argument("--weights",        type=str,       default=None)
    parser.add_argument('--embedding',      type=str,       default='softmax',      choices=['softmax', 'arcface', 'dual'])
    parser.add_argument('--attention',      type=str,       default='no',           choices=['no', 'se', 'cbam'])
    parser.add_argument('--norm',           type=str,       default=None)
    parser.add_argument('--activation',     type=str,       default=None)
    parser.add_argument('--freeze-backbone',type=int,       default=0)

    # arcface
    parser.add_argument('--margin',         type=float,     default=.5)
    parser.add_argument('--embd-shape',     type=int,       default=512)
    parser.add_argument('--logist-scale',   type=int,       default=64)

    # dual
    parser.add_argument('--loss-weights',   type=float,     default=.5)
    
    # hyperparameters
    parser.add_argument("--batch-size",     type=int,       default=4)
    parser.add_argument("--classes",        type=int,       default=200)
    parser.add_argument("--img-size",       type=int,       default=None)
    parser.add_argument("--img-channel",    type=int,       default=3)
    parser.add_argument("--steps",          type=int,       default=0)
    parser.add_argument("--epochs",         type=int,       default=100)
    parser.add_argument("--lr",             type=float,     default=.0001)
    parser.add_argument("--loss",           type=str,       default='crossentropy')
    parser.add_argument("--class-weight",   action='store_true')

    # preprocessing
    parser.add_argument("--standardize",    type=str,       default='minmax1',      choices=['minmax1', 'minmax2', 'norm', 'eachnorm'])
    parser.add_argument("--ignore-search",  type=str,       default='')
    parser.add_argument("--noise",          type=float,     default=0.)
    parser.add_argument("--angle",          type=int,       default=15)
    parser.add_argument("--crop",           type=int,       default=0,              choices=[0, 1, 2, 3])
    parser.add_argument("--contrast",       action='store_true')
    parser.add_argument("--distortion",     action='store_true')
    parser.add_argument("--vflip",          action='store_true')
    parser.add_argument("--hflip",          action='store_true')
    parser.add_argument("--mixup",          action='store_true')

    #callback
    parser.add_argument("--evaluate",       type=int,       default=0,              choices=[0, 1, 2])
    parser.add_argument("--checkpoint",     action='store_true')
    parser.add_argument("--history",        action='store_true')
    parser.add_argument("--tensorboard",    action='store_true')
    parser.add_argument("--lr-scheduler",   action='store_true')
    parser.add_argument("--lr-mode",        type=str,       default='constant')
    parser.add_argument("--lr-value",       type=float,     default=.1)
    parser.add_argument("--lr-interval",    type=str,       default='20,50,80')
    parser.add_argument("--lr-warmup",      type=int,       default=0)

    # etc
    parser.add_argument("--summary",        action='store_true')
    parser.add_argument('--src-path',       type=str,       default=None)
    parser.add_argument('--data-path',      type=str,       default=None)
    parser.add_argument('--result-path',    type=str,       default=None)
    parser.add_argument('--snapshot',       type=str,       default=None)
    parser.add_argument("--version",        type=int,       default=1,              choices=[1, 2, 3, 4, 5])
    parser.add_argument("--multi-gpu",      action='store_true')
    parser.add_argument("--gpus",           type=str,       default=-1)
    parser.add_argument("--verbose",        type=int,       default=0)

    return check_arguments(parser.parse_args())

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)
    return logger

def get_session(args):
    assert int(tf.__version__.split('.')[0]) >= 2.0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.gpus != -1:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

def set_default(args):
    if args.norm is None:
        args.norm = 'bn'

    if args.activation is None:
        if 'efficient' in args.backbone:
            args.activation = 'swish'
        else:
            args.activation = 'relu'

    if args.img_size is None:
        if 'efficient' in args.backbone:
            if args.backbone == 'efficientnetb0':
                args.img_size = 224
            elif args.backbone == 'efficientnetb1':
                args.img_size = 240
            elif args.backbone == 'efficientnetb2':
                args.img_size = 260
            elif args.backbone == 'efficientnetb3':
                args.img_size = 300
            elif args.backbone == 'efficientnetb4':
                args.img_size = 380
            elif args.backbone == 'efficientnetb5':
                args.img_size = 456
            elif args.backbone == 'efficientnetb6':
                args.img_size = 528
            elif args.backbone == 'efficientnetb7':
                args.img_size = 600
            else:
                raise ValueError()
            
        else:
            args.img_size = 224

    return args

def search_same(args):
    search_ignore = ['checkpoint', 'history', 'snapshot', 'src_path', 'data_path', 'result_path', 'verbose', 'ignore_search']
    if len(args.ignore_search) > 0:
        search_ignore += args.ignore_search.split(',')

    initial_epoch = 0
    stamps = os.listdir(args.result_path)
    for stamp in stamps:
        desc = yaml.full_load(open(os.path.join(args.result_path, '{}/model_desc.yml'.format(stamp))))
        flag = True
        for k, v in vars(args).items():
            if k in search_ignore:
                continue

            if v != desc[k]:
                flag = False
                break

        if flag:
            args.stamp = stamp
            df = pd.read_csv(os.path.join(args.result_path, '{}/history/epoch.csv'.format(args.stamp)))

            if len(df) > 0:
                if df['epoch'].values[-1]+1 == args.epochs:
                    print('{} Training already finished!!!'.format(stamp))
                    return args, -1

                else:
                    ckpt_list = sorted([d for d in os.listdir(os.path.join(args.result_path, '{}/checkpoint'.format(stamp))) if 'h5' in d],
                                        key=lambda x: int(x.split('_')[0]))
                    args.snapshot = os.path.join(args.result_path, '{}/checkpoint/{}'.format(args.stamp, ckpt_list[-1]))
                    initial_epoch = int(ckpt_list[-1].split('_')[0])

            break
    
    return args, initial_epoch