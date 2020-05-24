import os
import tqdm
import yaml
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TerminateOnNaN

from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score


class Evaluate(Callback):
    def __init__(
        self, args, generator1, generator2, trainset, valset, classes, verbose=1, **kwargs
    ):

        super(Evaluate, self).__init__()
        self.args = args
        self.generator1 = generator1
        self.generator2 = generator2
        self.trainset = trainset
        self.valset = valset
        self.classes = classes
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if self.args.embedding == "arcface":
            result_function = K.function(
                [self.model.get_layer(name="main_input").input],
                [self.model.get_layer(name="fc2").output],
            )

            total_train_result = []
            label_train_result = []
            for g in tqdm.tqdm(self.generator1.take(len(self.trainset)), total=len(self.trainset)):
                result_input = [g[0]["main_input"]]
                result = result_function(result_input)
                total_train_result.append(result[0].tolist())
                label_train_result.append(g[1]["arcface_output"].numpy().tolist())

            total_train_result /= np.linalg.norm(np.array(total_train_result), axis=1, keepdims=True)

            total_result = []
            label_result = []
            for g in tqdm.tqdm(self.generator2.take(len(self.valset)), total=len(self.valset)):
                result_input = [g[0]["main_input"]]
                result = result_function(result_input)
                total_result.append(result[0].tolist())
                label_result.append(g[1]["arcface_output"].numpy().tolist())

            total_result /= np.linalg.norm(np.array(total_result), axis=1, keepdims=True)

            neigh = KNeighborsClassifier(n_neighbors=5)
            neigh.fit(
                np.squeeze(total_train_result), np.squeeze(np.array(label_train_result))
            )
            knn_result = neigh.predict_proba(np.squeeze(total_result))

            pred_result = []
            for i in range(len(knn_result[0])):
                pred_result.append([knn_result[j][i, 1] for j in range(len(knn_result))])

        else:
            result_function = K.function(
                [self.model.get_layer(name="main_input").input],
                [self.model.get_layer(name="main_output").output],
            )

            pred_result = []
            label_result = []
            for g in tqdm.tqdm(self.generator2.take(len(self.valset)), total=len(self.valset)):
                result_input = [g[0]["main_input"]]
                result = result_function(result_input)

                pred_result.append(result[0].tolist())
                label_result.append(g[1]["main_output"].numpy().tolist())

        pred_result = np.squeeze(np.array(pred_result))
        label_result = np.squeeze(np.array(label_result))

        pred_result_acc = pred_result.argmax(axis=1)
        label_result_acc = label_result.argmax(axis=1)

        self.loss = (
            tf.keras.losses.categorical_crossentropy(label_result, pred_result).numpy().mean()
        )
        self.acc = [
            ((label_result_acc == i) * (pred_result_acc == i)).sum()/(label_result_acc == i).sum() for i in range(self.classes)
        ]
        self.auc = [
            roc_auc_score(label_result[:, i], pred_result[:, i]) for i in range(self.classes)
        ]
        tp = [
            ((label_result[:, i] >= 0.5) & (pred_result[:, i] >= 0.5)).sum()
            for i in range(self.classes)
        ]
        fp = [
            ((label_result[:, i] < 0.5) & (pred_result[:, i] >= 0.5)).sum()
            for i in range(self.classes)
        ]
        fn = [
            ((label_result[:, i] >= 0.5) & (pred_result[:, i] < 0.5)).sum()
            for i in range(self.classes)
        ]
        tn = [
            ((label_result[:, i] < 0.5) & (pred_result[:, i] < 0.5)).sum()
            for i in range(self.classes)
        ]
        self.sen = [tp[i] / (tp[i] + fn[i]) for i in range(self.classes)]
        self.spe = [tn[i] / (tn[i] + fp[i]) for i in range(self.classes)]

        # for saving
        logs = logs or {}

        if self.verbose == 1:
            print_acc = "{:.4f} (".format(sum(self.acc) / self.classes)
            print_auc = "{:.4f} (".format(sum(self.auc) / self.classes)
            print_sen = "{:.4f} (".format(sum(self.sen) / self.classes)
            print_spe = "{:.4f} (".format(sum(self.spe) / self.classes)
            for i in range(self.classes):
                print_acc += "{:.4f}".format(self.acc[i])
                print_auc += "{:.4f}".format(self.auc[i])
                print_sen += "{:.4f}".format(self.sen[i])
                print_spe += "{:.4f}".format(self.spe[i])

                if i < self.classes - 1:
                    print_acc += " "
                    print_auc += " "
                    print_sen += " "
                    print_spe += " "

            print()
            print(
                "Epoch {:04d}\n\tLoss : {:.4f}\n\tAccuracy : {})\n\tAUC : {})\n\tSensitivity : {})\n\tSpecificity : {})".format(
                    epoch + 1, self.loss, print_acc, print_auc, print_sen, print_spe
                )
            )

        logs["eval_loss"] = self.loss
        logs["eval_acc"] = sum(self.acc) / self.classes
        logs["eval_auc"] = sum(self.auc) / self.classes
        logs["eval_sen"] = sum(self.sen) / self.classes
        logs["eval_spe"] = sum(self.spe) / self.classes
        for i in range(self.classes):
            logs["eval_acc{}".format(i + 1)] = self.acc[i]
            logs["eval_auc{}".format(i + 1)] = self.auc[i]
            logs["eval_sen{}".format(i + 1)] = self.sen[i]
            logs["eval_spe{}".format(i + 1)] = self.spe[i]


class TerminateOnNaN_Epoch(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("val_loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print("Epoch %d: Invalid loss, terminating training" % (epoch))
                self.model.stop_training = True


class TerminateOnNoSensitivity(Callback):
      def on_epoch_end(self, epoch, logs=None):
            if epoch >= 10:
                logs = logs or {}
                monitor = 0.
                for m in [k for k in logs.keys() if 'sen' in k]:
                    monitor += logs.get(m)
                if monitor == 0.:
                    print('Epoch %d: No sensitivity, terminating training' % (epoch+1))
                    self.model.stop_training = True


class OptionalLearningRateSchedule(LearningRateSchedule):
    def __init__(self, args, steps_per_epoch):
        super(OptionalLearningRateSchedule, self).__init__()
        self.args = args
        self.steps_per_epoch = steps_per_epoch

        if self.args.lr_mode == 'exponential':
            decay_epochs = [int(e) for e in self.args.lr_interval.split(',')]
            lr_values = [self.args.lr * (self.args.lr_value ** k)
                         for k in range(len(decay_epochs) + 1)]
            self.lr_scheduler = \
                tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    decay_epochs, lr_values)

        elif self.args.lr_mode == 'cosine':
            self.lr_scheduler = \
                tf.keras.optimizers.schedules.CosineDecay(
                    self.args.lr, self.args.epochs
                )
            

    def get_config(self):
        return {
            'steps_per_epoch': self.steps_per_epoch,
            'init_lr': self.args.lr,
            'lr_mode': self.args.lr_mode,
            'lr_warmup': self.args.lr_warmup,
            'lr_value': self.args.lr_value,
            'lr_interval': self.args.lr_interval,
        }

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr_epoch = step / self.steps_per_epoch
        if self.args.lr_warmup:
            total_lr_warmup_step = self.args.lr_warmup * self.steps_per_epoch
            return tf.cond(lr_epoch < self.args.lr_warmup,
                           lambda: step / total_lr_warmup_step * self.args.lr,
                           lambda: self.lr_scheduler(lr_epoch-self.args.lr_warmup))
        else:
            if self.args.lr_mode == 'constant':
                return self.args.lr
            else:
                return self.lr_scheduler(lr_epoch)


#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
def callback_checkpoint(filepath, monitor, verbose, mode, save_best_only, save_weights_only):
    return ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        verbose=verbose,
        mode=mode,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
    )


def callback_epochlogger(filename, separator, append):
    return CSVLogger(filename=filename, separator=separator, append=append)


def callback_tensorboard(log_dir, batch_size):
    return TensorBoard(log_dir=log_dir, batch_size=batch_size)


def callback_learningrate(initlr, warmup, mode, value, interval, total_epoch):
    def scheduler(initlr, warmup, mode, value, interval, total_epoch):
        def _cosine_anneal_schedule(epoch, total_epoch):
            return float((1 + np.cos((np.pi * (epoch % total_epoch)) / total_epoch)) * initlr / 2)

        def _exponential(epoch, value, interval):
            if interval == "0" or len(interval) == 0:
                return initlr

            interval = np.array([int(i) for i in interval.split(",")])
            return initlr * (value ** (interval < epoch).sum())

        def _schedule(epoch, lr):
            if warmup and epoch < warmup:
                return initlr * (epoch + 1) / warmup

            if mode == "constant":
                return lr
            elif mode == "exponential":
                return _exponential(epoch - warmup, value, interval)
            elif mode == "cosine":
                return _cosine_anneal_schedule(epoch - warmup, total_epoch - warmup)

        return _schedule

    return LearningRateScheduler(
        schedule=scheduler(
            initlr=initlr,
            warmup=warmup,
            mode=mode,
            value=value,
            interval=interval,
            total_epoch=total_epoch,),
        verbose=1,)


def callback_terminateonnan():
    return TerminateOnNaN_Epoch()


def create_callbacks(args, test_generator1, test_generator2, trainset, valset):
    callback_list = []
    if args.evaluate:
        callback_list.append(
            Evaluate(
                args=args,
                generator1=test_generator1,
                generator2=test_generator2,
                trainset=trainset,
                valset=valset,
                classes=args.classes,
                verbose=args.evaluate,
            )
        )
        # callback_list.append(TerminateOnNoSensitivity())

    if args.snapshot is None:
        if args.checkpoint or args.history or args.tensorboard:
            os.makedirs(os.path.join(args.result_path, args.stamp), exist_ok=True)
            with open(
                os.path.join(args.result_path, args.stamp, "model_desc.yml"), "w"
            ) as f:
                yaml.dump(vars(args), f, default_flow_style=False)

    if args.checkpoint:
        os.makedirs(
            os.path.join(args.result_path, "{}/checkpoint".format(args.stamp)),
            exist_ok=True,
        )
        callback_list.append(
            callback_checkpoint(
                filepath=os.path.join(
                    args.result_path,
                    "{}/checkpoint/".format(args.stamp)
                    + "{epoch:04d}_{eval_acc:.4f}_{eval_loss:.4f}.h5",
                ),
                monitor="val_loss",
                verbose=1,
                mode="min",
                save_best_only=False,
                save_weights_only=True,
            )
        )

    if args.history:
        os.makedirs(
            os.path.join(args.result_path, "{}/history".format(args.stamp)),
            exist_ok=True,
        )
        callback_list.append(
            callback_epochlogger(
                filename=os.path.join(
                    args.result_path, "{}/history/epoch.csv".format(args.stamp)
                ),
                separator=",",
                append=True,
            )
        )

    if args.tensorboard:
        os.makedirs(
            os.path.join(args.result_path, "{}/logs".format(args.stamp)),
            exist_ok=True,
        )
        callback_list.append(
            callback_tensorboard(
                log_dir=os.path.join(args.result_path, "{}/logs".format(args.stamp)),
                batch_size=1,
            )
        )

    callback_list.append(callback_terminateonnan())

    return callback_list
