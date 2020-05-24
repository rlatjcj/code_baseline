import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tqdm
import yaml
import argparse
import numpy as np
import pandas as pd

from common import get_logger
from common import get_session
from model import set_model
from generator.dataloader import set_dataset
from generator.dataloader import create_generator

import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp",  type=str,   default=None)
    parser.add_argument("--epoch",  type=int,   default=0)
    parser.add_argument("--gpus",   type=str,   default='-1')
    
    return parser.parse_args()


def main():
    args = get_argument()
    assert args.stamp is not None

    new_args = yaml.full_load(open(os.path.join('./result/{}/model_desc.yml'.format(args.stamp))))
    for k, v in new_args.items():
        if k == 'gpus':
            continue
        setattr(args, k, v)

    get_session(args)

    logger = get_logger('MyLogger')
    for k, v in vars(args).items():
        logger.info('{} : {}'.format(k, v))

    ##########################
    # Model
    ##########################
    model = set_model.Backbone(args, logger)

    ##########################
    # Generator
    ##########################
    trainset, valset = set_dataset(args, logger)
    generator1 = create_generator(args, trainset, 'test', 1)
    generator2 = create_generator(args, valset, 'test', 1)

    os.makedirs('./inference_result/{}'.format(args.stamp), exist_ok=True)
    ckpt_list = sorted([d for d in os.listdir('./result/{}/checkpoint'.format(args.stamp)) if 'h5' in d],
                        key=lambda x: int(x.split('_')[0]))

    for_save = []
    # for ckpt in ckpt_list:
    ckpt = ckpt_list[args.epoch]
    logger.info('{}_{}'.format(args.stamp, ckpt))
    # if os.path.isfile('./inference_result/{}/{}/{}_train_label.npy'.format(args.task, args.stamp, ckpt.split('_')[0])):
    #     continue

    model.load_weights('./result/{}/checkpoint/{}'.format(args.stamp, ckpt), by_name=True)

    output_name = 'fc2' if args.embedding == 'arcface' else 'main_output'    
    result_function = K.function([model.get_layer(name='main_input').input], [model.get_layer(name=output_name).output])
    feature_function = K.function([model.get_layer(name='main_input').input], 
                                  [model.get_layer(name='avg_pool' if args.embedding == 'softmax' else 'fc2').output])

    train_feature = []
    train_result = []
    train_label = []
    train_patient = []
    for g in tqdm.tqdm(generator1.take(len(trainset)), total=len(trainset)):
        result_input = [g[0]['main_input']]
        feature = feature_function(result_input)
        result = result_function(result_input)

        train_feature.append(feature[0].tolist())
        train_result.append(result[0].tolist())
        train_label.append(g[1][output_name].numpy().tolist())
        train_patient.append(g[2]['patient'].numpy()[0,0].decode('utf-8'))

    if args.embedding == 'arcface':
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(np.squeeze(np.array(train_feature)), np.squeeze(np.array(train_label)))
        knn_result = neigh.predict_proba(np.squeeze(np.array(train_feature)))
        train_result = []
        for i in range(len(knn_result[0])):
            train_result.append([knn_result[j][i, 1] for j in range(len(knn_result))])

    np.save('./inference_result/{}/{}_train_feature.npy'.format(args.stamp, ckpt.split('_')[0]), np.array(train_feature))
    np.save('./inference_result/{}/{}_train_result.npy'.format(args.stamp, ckpt.split('_')[0]), np.array(train_result))
    np.save('./inference_result/{}/{}_train_label.npy'.format(args.stamp, ckpt.split('_')[0]), np.array(train_label))
    pd.DataFrame(data=train_patient).to_csv(
        './inference_result/{}/{}_train_patient.csv'.format(args.stamp, ckpt.split('_')[0]), index=False)
    
    val_feature = []
    val_result = []
    val_label = []
    val_patient = []
    for g in tqdm.tqdm(generator2.take(len(valset)), total=len(valset)):
        result_input = [g[0]['main_input']]
        feature = feature_function(result_input)
        result = result_function(result_input)

        val_feature.append(feature[0].tolist())
        val_result.append(result[0].tolist())
        val_label.append(g[1][output_name].numpy().tolist())
        val_patient.append(g[2]['patient'].numpy()[0,0].decode('utf-8'))

    if args.embedding == 'arcface':
        knn_result = neigh.predict_proba(np.squeeze(np.array(val_feature)))
        val_result = []
        for i in range(len(knn_result[0])):
            val_result.append([knn_result[j][i, 1] for j in range(len(knn_result))])


    np.save('./inference_result/{}/{}_val_feature.npy'.format(args.stamp, ckpt.split('_')[0]), np.array(val_feature))
    np.save('./inference_result/{}/{}_val_result.npy'.format(args.stamp, ckpt.split('_')[0]), np.array(val_result))
    np.save('./inference_result/{}/{}_val_label.npy'.format(args.stamp, ckpt.split('_')[0]), np.array(val_label))
    pd.DataFrame(data=val_patient).to_csv(
        './inference_result/{}/{}_val_patient.csv'.format(args.stamp, ckpt.split('_')[0]), index=False)
    
    val_result = np.squeeze(np.array(val_result))
    val_result_acc = np.eye(args.classes)[val_result.argmax(axis=1)]
    val_label = np.squeeze(np.array(val_label))

    val_result_acc = val_result.argmax(axis=1)
    val_label_acc = val_label.argmax(axis=1)
    
    loss = tf.keras.losses.categorical_crossentropy(val_label, val_result).numpy().mean()
    acc = [((val_label_acc == i) * (val_result_acc == i)).sum()/(val_label_acc == i).sum() for i in range(args.classes)]
    auc = [roc_auc_score(val_label[:,i], val_result[:,i]) for i in range(args.classes)]
    tp = [((val_label[:,i] >= .5) & (val_result[:,i] >= .5)).sum() for i in range(args.classes)]
    fp = [((val_label[:,i] < .5) & (val_result[:,i] >= .5)).sum() for i in range(args.classes)]
    fn = [((val_label[:,i] >= .5) & (val_result[:,i] < .5)).sum() for i in range(args.classes)]
    tn = [((val_label[:,i] < .5) & (val_result[:,i] < .5)).sum() for i in range(args.classes)]
    sen = [tp[i]/(tp[i]+fn[i]) for i in range(args.classes)]
    spe = [tn[i]/(tn[i]+fp[i]) for i in range(args.classes)]

    print_acc = "{:.4f} (".format(sum(acc) / args.classes)
    print_auc = "{:.4f} (".format(sum(auc) / args.classes)
    print_sen = "{:.4f} (".format(sum(sen) / args.classes)
    print_spe = "{:.4f} (".format(sum(spe) / args.classes)
    for i in range(args.classes):
        print_acc += "{:.4f}".format(acc[i])
        print_auc += "{:.4f}".format(auc[i])
        print_sen += "{:.4f}".format(sen[i])
        print_spe += "{:.4f}".format(spe[i])

        if i < args.classes - 1:
            print_acc += " "
            print_auc += " "
            print_sen += " "
            print_spe += " "

    print()
    print(
        "Epoch {:04d}\n\tLoss : {:.4f}\n\tAccuracy : {})\n\tAUC : {})\n\tSensitivity : {})\n\tSpecificity : {})".format(
            args.epoch + 1, loss, print_acc, print_auc, print_sen, print_spe
        )
    )

    for_save.append([int(ckpt.split('_')[0]), loss]+acc+auc+sen+spe)

    if os.path.isfile('./inference_result/{}/score.csv'.format(args.stamp)):
        df = pd.read_csv('./inference_result/{}/score.csv'.format(args.stamp))
        df = df.append(pd.DataFrame(data=for_save, columns=df.columns))
        df.sort_values('epoch', inplace=True)
        df.reset_index(drop=True, inplace=True)

    else:
        df = pd.DataFrame(
            data=for_save, 
            columns=['epoch', 'loss'] + \
                ['acc_{}'.format(i+1) for i in range(200)] + \
                    ['auc_{}'.format(i+1) for i in range(200)] + \
                        ['sen_{}'.format(i+1) for i in range(200)] + \
                            ['spe_{}'.format(i+1) for i in range(200)])

    df.to_csv('./inference_result/{}/score.csv'.format(args.stamp), index=False)


if __name__ == "__main__":
    main()