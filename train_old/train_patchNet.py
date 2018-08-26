import mxnet as mx
import logging
from symbol_patchNet import patchNet,get_symbol_patchNet
from DataIter import MsuRecordIter
import os
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

def multi_factor_scheduler(begin_epoch, epoch_size, step=[30, 60, 90, 95, 115, 120], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None


def get_iterators(batch_size, data_shape=(3, 96, 96)):
    train_imgrec = '/home/sbb/data/antiSpoof/MSU_AUG/msu_train.rec'
    val_imgrec = '/home/sbb/data/antiSpoof/MSU_AUG/msu_val.rec'
    mean_pixels = [123,104,117]
    label_pad_width = 1
    train_list = '/home/sbb/data/antiSpoof/MSU_AUG/msu_train.lst'
    val_list = '/home/sbb/data/antiSpoof/MSU_AUG/msu_val.lst'
    train_flag = 1
    # is_train =True
    train_iter = MsuRecordIter(train_imgrec, batch_size, data_shape, mean_pixels=mean_pixels,
                               label_pad_width=1, path_imglist=train_list)
    val_iter = MsuRecordIter(val_imgrec, batch_size, data_shape, mean_pixels=mean_pixels,
                               label_pad_width=1, path_imglist=train_list)
    return (train_iter, val_iter)

def get_symbol_sqNet():
    num_classes = 2
    net = patchNet(num_classes)
    return net

def fit(symbol, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    # devs = [mx.gpu(1)]
    mod = mx.mod.Module(symbol, context=devs)
    # save_prefix = "./model-sq/squeeze"
    prefix ="./tt/patchNet"
    lr_sch = mx.lr_scheduler.FactorScheduler(step=3000, factor=0.9)
    mod.fit(train, val,
        num_epoch=100,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 500),
        epoch_end_callback = mx.callback.do_checkpoint(prefix),
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate':0.08, 'lr_scheduler': lr_sch},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)

if __name__ == "__main__":
    prefix = "/home/sbb/start/Antispot/model/squeeze"
    net = get_symbol_patchNet()
    batch_size =100
    num_gpus = 1
    # get_iterators(batch_size)
    if not os.path.exists("./log"):
        os.mkdir("./log")
    hdlr = logging.FileHandler('./log/log-patchNet-{}-{}.log'.format("msu", 16))
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    # logging.info(args)
    (train, val) = get_iterators(batch_size)
    fit(net,train,val,batch_size,num_gpus)
