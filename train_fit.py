import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from datetime import datetime

from common import get_logger
from common import set_default
from common import search_same
from common import get_session
from common import get_argument
from model import set_model
from generator.dataloader import set_dataset
from generator.dataloader import create_generator
from callback import create_callbacks


def create_model(args, logger):
    if args.multi_gpu:
        import tensorflow as tf
        assert float(tf.__version__[:3]) == 2.2
        
        strategy = tf.distribute.MirroredStrategy()
        # strategy = tf.distribute.experimental.CentralStorageStrategy() # over 2.1
        with strategy.scope():
            model = set_model.Backbone(args, logger)
    else:
        model = set_model.Backbone(args, logger)
    
    return model

def compile_model(args, model, steps_per_epoch):
    import loss
    import metric
    from callback import OptionalLearningRateSchedule
    
    optimizer = tf.keras.optimizers.Adam(OptionalLearningRateSchedule(args, steps_per_epoch))

    if args.embedding == 'dual':
        if args.loss == 'crossentropy':
            losses = {'main_output': loss.crossentropy(args), 'arcface_output': loss.crossentropy(args)}
        metrics = {'main_output': 'acc', 'arcface_output': 'acc'}
        loss_weights = {'main_output': args.loss_weights, 'arcface_output': 1-args.loss_weights}
    else:
        if args.loss == 'crossentropy':
            losses = loss.crossentropy(args)
        metrics = ['acc']
        loss_weights = None

    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics,
        loss_weights=loss_weights
    )
    
    return model



def main():
    args = set_default(get_argument())
    args, initial_epoch = search_same(args)
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

    get_session(args)
    logger = get_logger("MyLogger")
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))

    ##########################
    # Generator
    ##########################
    trainset, valset = set_dataset(args, logger)

    train_generator = create_generator(args, trainset, "train", args.batch_size)
    # for t in train_generator:
        # print(sorted(t[1]['main_output'].numpy().argmax(axis=0)))
    #     print(t[0]['main_input'].shape, t[0]['main_input'].numpy().min(), t[0]['main_input'].numpy().max(), t[1]['main_output'])
    val_generator = create_generator(args, valset, "val", args.batch_size)
    # for t in val_generator:
    #     print(t[0][0].shape, t[0][1], t[1])
    test_generator1 = create_generator(args, trainset, "val", 1)
    test_generator2 = create_generator(args, valset, "val", 1)
    # for t in test_generator2:
    # print(t[0]['main_input'].shape, t[0]['arcface_input'])

    if args.class_weight:
        assert args.classes > 1
        from sklearn.utils.class_weight import compute_class_weight

        train_label = trainset[:, 1:].astype(np.int).argmax(axis=1)
        class_weight = compute_class_weight(
            class_weight="balanced", classes=np.unique(train_label), y=train_label
        )

    else:
        class_weight = None

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
    model = create_model(args, logger)
    if args.summary:
        model.summary()
        print(model.inputs[0])
        print(model.get_layer(name="fc2"))
        return

    model = compile_model(args, model, steps_per_epoch)
    logger.info("Build model!")

    ##########################
    # Callbacks
    ##########################
    callbacks = create_callbacks(args, test_generator1, test_generator2, trainset, valset)
    logger.info("Build callbacks!")

    ##########################
    # Train
    ##########################
    model.fit(
        x=train_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        class_weight=class_weight,
        initial_epoch=initial_epoch,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
