import mxnet as mx
#fine-tune an model
import logging
import os
from symbol import  fmobilenet
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

def get_iterators(batch_size, data_shape=(3, 128, 128)):
    train = mx.io.ImageRecordIter_v1(
        path_imgrec         = '/home/sbb/start/Antispot/record/data_train.rec',
        data_name           = 'data',
        batch_size          = batch_size,
        data_shape          = data_shape,
        mean_r              = 123,
        mean_g              = 117,
        mean_b              = 104,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True)
    val = mx.io.ImageRecordIter_v1(
        path_imgrec         = '/home/sbb/start/Antispot/record/data_val.rec',
        data_name           = 'data',
        batch_size          = batch_size,
        data_shape          = data_shape,
        mean_r              = 123,
        mean_g              = 117,
        mean_b              = 104,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True)
    return (train, val)

def get_symbol_mobileNet(num_classes=2):
    net = fmobilenet.get_symbol(num_classes)
    return net

def fit(symbol, train, val, batch_size, num_gpus):
    devs = [mx.gpu(5),mx.gpu(6)]
    mod = mx.mod.Module(symbol, context=devs)
    prefix="./tt/mobile-RGB"
    lr_sch = mx.lr_scheduler.FactorScheduler(step=30000, factor=0.8)
    mod.fit(train, val,
        num_epoch=200,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 1000),
        epoch_end_callback = mx.callback.do_checkpoint(prefix),
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate':0.05, 'lr_scheduler': lr_sch},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc')
    metric = mx.metric.Accuracy()
    # mod.save_checkpoint(prefix="./modelzoo/squeezenet",epoch=31,save_optimizer_states=False)
    return mod.score(val, metric)


if __name__=="__main__":
    print "success"
    num_classes = 2
    batch_size = 128
    # batch_per_gpu = 16
    num_gpus = 2
    #load model
    # sym, arg_params, aux_params = mx.model.load_checkpoint('tt/lmobileE',2)
    net= get_symbol_mobileNet(num_classes)
    if not os.path.exists("./log"):
        os.mkdir("./log")
    hdlr = logging.FileHandler('./log/log-mob-{}-{}.log'.format("data", 128))
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    (train, val) = get_iterators(batch_size,data_shape=(3, 128, 128))
    fit(net,train,val,batch_size,num_gpus)

