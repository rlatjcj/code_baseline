import os
import yaml
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import roc_auc_score

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
                tf.keras.experimental.CosineDecay(
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


def evaluate(args, model, epoch, generator1=None, generator2=None, trainset=None, valset=None, verbose=1):
    if args.embedding == 'arcface':
        raise
    elif args.embedding in ['softmax', 'dual']:
        result_func = K.function(
            [model.get_layer(name="main_input").input],
            [model.get_layer(name="main_output").output]
        )

        pred_result = []
        label_result = []
        for g in tqdm.tqdm(generator2.take(len(valset)), total=len(valset)):
            result_input = [g[0]["main_input"]]
            result = result_function(result_input)

            pred_result.append(result[0].tolist())
            label_result.append(g[1]["main_output"].numpy().tolist())

    pred_result = np.squeeze(np.array(pred_result))
    label_result = np.squeeze(np.array(label_result))

    pred_result_acc = pred_result.argmax(axis=1)
    label_result_acc = label_result.argmax(axis=1)

    loss = tf.keras.losses.categorical_crossentropy(label_result, pred_result).numpy().mean()
    acc = [((label_result_acc == i) * (pred_result_acc == i)).sum()/(label_result_acc == i).sum() for i in range(args.classes)]
    auc = [roc_auc_score(label_result[:, i], pred_result[:, i]) for i in range(args.classes)]
    tp = [((label_result[:, i] >= 0.5) & (pred_result[:, i] >= 0.5)).sum() for i in range(args.classes)]
    fp = [((label_result[:, i] < 0.5) & (pred_result[:, i] >= 0.5)).sum() for i in range(args.classes)]
    fn = [((label_result[:, i] >= 0.5) & (pred_result[:, i] < 0.5)).sum() for i in range(args.classes)]
    tn = [((label_result[:, i] < 0.5) & (pred_result[:, i] < 0.5)).sum() for i in range(args.classes)]
    sen = [tp[i] / (tp[i] + fn[i]) for i in range(args.classes)]
    spe = [tn[i] / (tn[i] + fp[i]) for i in range(args.classes)]

    # for saving
    logs = logs or {}

    if verbose == 1:
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
                epoch + 1, loss, print_acc, print_auc, print_sen, print_spe
            )
        )

    logs["eval_loss"] = loss
    logs["eval_acc"] = sum(acc) / args.classes
    logs["eval_auc"] = sum(auc) / args.classes
    logs["eval_sen"] = sum(sen) / args.classes
    logs["eval_spe"] = sum(spe) / args.classes
    for i in range(args.classes):
        logs["eval_acc{}".format(i + 1)] = acc[i]
        logs["eval_auc{}".format(i + 1)] = auc[i]
        logs["eval_sen{}".format(i + 1)] = sen[i]
        logs["eval_spe{}".format(i + 1)] = spe[i]

    return logs

def create_callbacks(args, steps_per_epoch, metrics):
    if args.snapshot is None:
        if args.checkpoint or args.history or args.tensorboard:
            os.makedirs(os.path.join(args.result_path, args.stamp), exist_ok=True)
            yaml.dump(
                vars(args), 
                open(os.path.join(args.result_path, args.stamp, "model_desc.yml"), "w"), 
                default_flow_style=False)

    if args.checkpoint:
        os.makedirs(os.path.join(args.result_path, '{}/checkpoint'.format(args.stamp)), exist_ok=True)

    if args.history:
        os.makedirs(os.path.join(args.result_path, '{}/history'.format(args.stamp)), exist_ok=True)
        csvlogger = pd.DataFrame(columns=['epoch']+list(metrics.keys()))
        if os.path.isfile(os.path.join(args.result_path, '{}/history/epoch.csv'.format(args.stamp))):
            csvlogger = pd.read_csv(os.path.join(args.result_path, '{}/history/epoch.csv'.format(args.stamp)))
        else:
            csvlogger.to_csv(os.path.join(args.result_path, '{}/history/epoch.csv'.format(args.stamp)), index=False)
    else:
        csvlogger = None

    lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch)

    return csvlogger, lr_scheduler