import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import yaml
import tqdm
import numpy as np
import pandas as pd
from datetime import datetime

from common import get_logger
from common import set_default
from common import search_same
from common import get_session
from common import get_argument
from generator.dataloader import set_dataset
from generator.dataloader import create_generator
from generator.dataloader_semi import dataloader
from model import set_model
from callback_tf import create_callbacks

import tensorflow as tf

def main():
    args = set_default(get_argument())
    get_session(args)

    args, initial_epoch = search_same(args)
    if initial_epoch == -1:
        return

    if args.snapshot is None:
        weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        temp = datetime.now()
        args.stamp = "{:02d}{:02d}{:02d}_{}_{:02d}_{:02d}_{:02d}".format(
            temp.year // 100,
            temp.month,
            temp.day,
            weekday[temp.weekday()],
            temp.hour,
            temp.minute,
            temp.second,
        )

    logger = get_logger("MyLogger")
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))

    ##########################
    # Generator
    ##########################
    trainset, valset, unlabset = set_dataset(args, logger)

    train_generator = create_generator(args, trainset, "train", args.batch_size)
    val_generator = create_generator(args, valset, "val", args.batch_size)
    unlabel_generator = dataloader(args, unlabset)

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== trainset ==========")
    steps_per_epoch = args.steps or len(trainset) // args.batch_size
    logger.info("    --> {}".format(steps_per_epoch))
    # logger.info("    --> {}".format(trainset[:, 2:].sum(axis=0)))
    # logger.info("    --> {}".format(class_weight))

    logger.info("=========== valset ===========")
    validation_steps = len(valset) // args.batch_size
    logger.info("    --> {}".format(validation_steps))
    # logger.info("    --> {}".format(valset[:, 2:].sum(axis=0)))

    ##########################
    # Model
    ##########################
    if args.multi_gpu:
        raise ValueError()
        # assert float(tf.__version__[:3]) == 2.2
        
        # strategy = tf.distribute.MirroredStrategy()
        # # strategy = tf.distribute.experimental.CentralStorageStrategy() # over 2.1
        # with strategy.scope():
        #     model = set_model.Backbone(args, logger)
    else:
        model = set_model.Backbone(args, logger)

    if args.summary:
        model.summary()
        print(model.inputs[0])
        print(model.get_layer(name="fc2"))
        return

    logger.info("Build model!")

    ##########################
    # Metric
    ##########################
    metrics = {
        'loss'    :   tf.keras.metrics.Mean('loss', dtype=tf.float32),
        'loss_xe' :   tf.keras.metrics.Mean('loss_xe', dtype=tf.float32),
        'loss_kl' :   tf.keras.metrics.Mean('loss_kl', dtype=tf.float32),
        'acc'     :   tf.keras.metrics.CategoricalAccuracy('acc', dtype=tf.float32),
        'val_loss':   tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
        'val_acc' :   tf.keras.metrics.CategoricalAccuracy('val_acc', dtype=tf.float32),
    }

    lr_scheduler = create_callbacks(args, steps_per_epoch, metrics)
    optimizer = tf.keras.optimizers.Adam(lr_scheduler)

    ##########################
    # Train
    ##########################
    # steps_per_epoch = 10
    # validation_steps = 10
    train_iterator = iter(train_generator)
    val_iterator = iter(val_generator)
    unlabel_iterator = iter(unlabel_generator)

    progress_desc_train = 'Train : Loss {:.4f} | XE {:.4f} | KL {:.4f} | Acc {:.4f}'
    progress_desc_val = 'Val : Loss {:.4f} | Acc {:.4f}'

    # @tf.function
    # def train_step():
        

    # @tf.function
    # def val_step():
        


    for epoch in range(args.epochs):
        print('\nEpoch {}/{}'.format(epoch+1, args.epochs))
        print('Learning Rate : {}'.format(optimizer.learning_rate(optimizer.iterations)))

        progressbar_train = tqdm.tqdm(
            tf.range(steps_per_epoch), 
            desc=progress_desc_train.format(0, 0, 0, 0), 
            leave=True)
        for step in progressbar_train:
            sup_inputs = next(train_iterator)
            unup_inputs = next(unlabel_iterator)
            sup_img = sup_inputs[0]['main_input']
            sup_label = sup_inputs[1]['main_output']
            unsup_img = unup_inputs['img']
            unsup_augimg = unup_inputs['augimg']

            with tf.GradientTape() as tape:
                img = tf.concat([sup_img, unsup_img, unsup_augimg], axis=0)
                logits = tf.cast(model(img, training=True), tf.float32)

                logit_sup, logit_unsup, logit_aug_unsup = tf.split(
                    logits,
                    [int(sup_img.shape[0]),
                        int(unsup_img.shape[0]),
                        int(unsup_augimg.shape[0])])

                loss_xe = tf.keras.losses.categorical_crossentropy(sup_label, logit_sup)
                loss_xe = tf.reduce_mean(loss_xe)
                loss_kl = tf.keras.losses.KLD(logit_unsup, logit_aug_unsup)

                total_loss = loss_xe + args.klloss_weight * loss_kl

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            metrics['loss'].update_state(total_loss)
            metrics['loss_xe'].update_state(loss_xe)
            metrics['loss_kl'].update_state(loss_kl)
            metrics['acc'].update_state(sup_label, logit_sup)

            progressbar_train.set_description(
                progress_desc_train.format(
                    metrics['loss'].result(),
                    metrics['loss_xe'].result(),
                    metrics['loss_kl'].result(),
                    metrics['acc'].result()))
            progressbar_train.refresh()
        
        
        progressbar_val = tqdm.tqdm(
            tf.range(validation_steps), 
            desc=progress_desc_val.format(0, 0), 
            leave=True)
        for step in progressbar_val:
            val_inputs = next(val_iterator)
            val_img = val_inputs[0]['main_input']
            val_label = val_inputs[1]['main_output']
            val_logits = tf.cast(model(val_img, training=False), tf.float32)
            
            val_loss = tf.keras.losses.categorical_crossentropy(val_label, val_logits)
            val_loss = tf.reduce_mean(val_loss)

            metrics['val_loss'].update_state(val_loss)
            metrics['val_acc'].update_state(val_label, val_logits)

            progressbar_val.set_description(
                progress_desc_val.format(
                    metrics['val_loss'].result(),
                    metrics['val_acc'].result()))
            progressbar_val.refresh()


        logs = {k: v.result().numpy() for k, v in metrics.items()}
        logs['epoch'] = epoch + 1

        if args.checkpoint:
            model.save_weights(
                os.path.join(
                    args.result_path, 
                    '{}/checkpoint/{:04d}_{:.4f}_{:.4f}.h5'.format(
                        args.stamp, epoch+1, logs['val_acc'], logs['val_loss'])))

            print('\nSaved at {}'.format(
                os.path.join(
                    args.result_path, 
                    '{}/checkpoint/{:04d}_{:.4f}_{:.4f}.h5'.format(
                        args.stamp, epoch+1, logs['val_acc'], logs['val_loss']))))

        if args.history:
            csvlogger = csvlogger.append(logs, ignore_index=True)
            csvlogger.to_csv(os.path.join(args.result_path, '{}/history/epoch.csv'.format(args.stamp)), index=False)
        
        for v in metrics.values:
            v.reset_states()
        # metrics['loss'].reset_states()
        # metrics['loss_xe'].reset_states()
        # metrics['loss_kl'].reset_states()
        # metrics['acc'].reset_states()
        # metrics['val_loss'].reset_states()
        # metrics['val_acc'].reset_states()


if __name__ == '__main__':
    main()