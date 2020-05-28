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

    if initial_epoch == -1:
        # training was already finished!
        return

    elif initial_epoch == 0:
        # first training or training with snapshot
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
    trainset, valset, testset = set_dataset(args, logger)
    train_generator = dataloader(args, trainset, 'train')
    val_generator = dataloader(args, valset, 'val', False)

    # for t in train_generator:
    #     print(t[0]['main_input'].shape, t[0]['main_input'].numpy().min(), t[0]['main_input'].numpy().max(), 
    #           t[1]['main_output'].shape, t[1]['main_output'].numpy().min(), t[1]['main_output'].numpy().max(), 
    #           t[1]['main_output'].numpy().argmax())
    #     print()

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== trainset ==========")
    logger.info("    --> {}".format(len(trainset)))
    steps_per_epoch = args.steps or len(trainset) // args.batch_size
    logger.info("    --> {}".format(steps_per_epoch))

    logger.info("=========== valset ===========")
    validation_steps = len(valset) // args.batch_size
    logger.info("    --> {}".format(len(valset)))
    logger.info("    --> {}".format(validation_steps))

    ##########################
    # Metric
    ##########################
    metrics = {
        'loss'    :   tf.keras.metrics.Mean('loss', dtype=tf.float32),
        'acc'     :   tf.keras.metrics.CategoricalAccuracy('acc', dtype=tf.float32),
        'val_loss':   tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
        'val_acc' :   tf.keras.metrics.CategoricalAccuracy('val_acc', dtype=tf.float32),
    }

    csvlogger, lr_scheduler = create_callbacks(args, steps_per_epoch, metrics)
    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr_scheduler, momentum=.9, decay=.0001)
    elif args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr_scheduler)

    ##########################
    # Train
    ##########################
    # steps_per_epoch = 10
    # validation_steps = 10
    train_iterator = iter(train_generator)
    val_iterator = iter(val_generator)

    progress_desc_train = 'Train : Loss {:.4f} | Acc {:.4f}'
    progress_desc_val = 'Val : Loss {:.4f} | Acc {:.4f}'

    for epoch in range(initial_epoch, args.epochs):
        print('\nEpoch {}/{}'.format(epoch+1, args.epochs))
        print('Learning Rate : {}'.format(optimizer.learning_rate(optimizer.iterations)))

        progressbar_train = tqdm.tqdm(
            tf.range(steps_per_epoch), 
            desc=progress_desc_train.format(0, 0, 0, 0), 
            leave=True)
        for step in progressbar_train:
            inputs = next(train_iterator)
            img = inputs[0]['main_input']
            label = inputs[1]['main_output']
            with tf.GradientTape() as tape:
                logits = tf.cast(model(img, training=True), tf.float32)
                loss = tf.keras.losses.categorical_crossentropy(label, logits)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            metrics['loss'].update_state(loss)
            metrics['acc'].update_state(label, logits)

            progressbar_train.set_description(
                progress_desc_train.format(
                    metrics['loss'].result(),
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
        
        for k, v in metrics.items():
            v.reset_states()


if __name__ == '__main__':
    main()