import mxnet as mx
import logging
from symbol.resnetEx import *
from DataIter import faceRecordIter, Multi_iterator
import numpy as np
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)
def get_iterators(batch_size, data_shape=(3, 112, 112)):
    train = mx.io.ImageRecordIter(
        path_imgrec='/home/sbb/data/record/train.rec',
        batch_size=batch_size,
        data_shape=data_shape,
        shuffle=True,
        mean_r=127.5,
        mean_g=127.5,
        mean_b=127.5,
        label_width=2
    )
    val = mx.io.ImageRecordIter(
        path_imgrec='/home/sbb/data/record/val.rec',
        batch_size=batch_size,
        data_shape=data_shape,
        mean_r=127.5,
        mean_g=127.5,
        mean_b=127.5,
        label_width=2
    )
    return (train, val)


class Multi_Loss(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""
    def __init__(self, num=None):
        super(Multi_Loss, self).__init__('multi-Multi_Loss', num)
        self.num = num
    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)
        if self.num != None:
            assert len(labels) == self.num

        for i in range(len(labels)):
            pred_label = mx.nd.argmax_channel(preds[i]).asnumpy().astype('float32')
            labels = labels[i].asnumpy().astype('float32')

            for label, pred in zip(labels, pred_label):
                label = label.asnumpy()
                pred = pred.asnumpy()

                if len(label.shape) == 1:
                    label = label.reshape(label.shape[0], 1)
                if len(pred.shape) == 1:
                    pred = pred.reshape(pred.shape[0], 1)
                self.sum_metric[i] += np.abs(label - pred).mean()
                self.num_inst[i] += 1  # numpy.prod(label.shape)


def multi_factor_scheduler(begin_epoch, epoch_size, step=[30, 60, 90, 95, 115, 120], factor=0.1):
    step_ = [epoch_size * (x - begin_epoch) for x in step if x - begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None


def get_symbol_resnetBin():
    net = resnetBin()
    return net


def fit(symbol, train, val, batch_size, num_gpus):
    # devs = [mx.gpu(i) for i in range(num_gpus)]
    devs = [mx.gpu(3)]
    mod = mx.mod.Module(symbol, context=devs,
                        label_names=['softmax1_label', 'softmax2_label'])
    # save_prefix = "./model-sq/squeeze"
    prefix = "./models/resnetBin"
    lr_sch = mx.lr_scheduler.FactorScheduler(step=3000, factor=0.9)
    mod.fit(train, val,
            num_epoch=100,
            allow_missing=True,
            batch_end_callback=mx.callback.Speedometer(batch_size, 100),
            epoch_end_callback=mx.callback.do_checkpoint(prefix),
            kvstore='device',
            optimizer='sgd',
            optimizer_params={'learning_rate': 0.08, 'lr_scheduler': lr_sch},
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric=Multi_Loss)


if __name__ == "__main__":
    prefix = "./model/resnetBin"
    net = get_symbol_resnetBin()
    batch_size = 100
    num_gpus = 1
    # get_iterators(batch_size)
    if not os.path.exists("./log"):
        os.mkdir("./log")
    hdlr = logging.FileHandler('./log/log-patchNet-{}-{}.log'.format("msu", 16))
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    # logging.info(args)
    train, val = get_iterators(batch_size)
    train = Multi_iterator(train)
    val = Multi_iterator(val)
    fit(net, train, val, batch_size, num_gpus)
