import mxnet as mx
#fine-tune an model
import logging
import os
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

def get_iterators(batch_size, data_shape=(3, 224, 224)):
    train = mx.io.ImageRecordIter_v1(
        path_imgrec         = '/home/sbb/start/Antispot/record/data_aug_train.rec',
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
        path_imgrec         = '/home/sbb/start/Antispot/record/data_aug_val.rec',
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


def get_fine_tune_sqeezenet_model(symbol, arg_params, num_classes, layer_name='drop9'):
    """
    symbol: the pretrained network symbol
    arg_params: the argument parameters of the pretrained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    #net = all_layers[layer_name]
    conv10=mx.symbol.Convolution(data=net, num_filter=num_classes, kernel=(1,1), stride=(1,1), name="conv10")
    relu_conv10=mx.symbol.Activation(data = conv10, act_type="relu", attr={})
    pool10=mx.symbol.Pooling(data=relu_conv10, kernel=(13, 13), pool_type='avg', attr={})
    flatten = mx.symbol.Flatten(data=pool10, name='flatten')
    net = mx.symbol.SoftmaxOutput(data=flatten, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'conv10' not in k})
    return (net, new_args)


def get_fine_tune_mobile_model(symbol, arg_params, num_classes, layer_name='dropout0'):
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc8')
    bn1 = mx.symbol.BatchNorm(data=net)
    net = mx.symbol.SoftmaxOutput(data=bn1, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc7' not in k})
    return (net, new_args)


import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

def fit(symbol, arg_params, aux_params, train, val, batch_size):
    devs = [mx.gpu(3),mx.gpu(4)]
    mod = mx.mod.Module(symbol, context=devs)
    prefix="./tt/squeezeNet-224-dataaugpooling"
    lr_sch = mx.lr_scheduler.FactorScheduler(step=30000, factor=0.8)

    mod.fit(train, val,
        num_epoch=200,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 1000),
        epoch_end_callback = mx.callback.do_checkpoint(prefix),
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate':0.0008, 'lr_scheduler': lr_sch},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc')
    metric = mx.metric.Accuracy()
    # mod.save_checkpoint(prefix="./modelzoo/squeezenet",epoch=31,save_optimizer_states=False)
    return mod.score(val, metric)


if __name__=="__main__":
    print "success"
    num_classes = 2
    batch_size = 512
    # batch_per_gpu = 16
    shape = 224
    #load model
    sym, arg_params, aux_params = mx.model.load_checkpoint('tt/squeezeNet-224-dataaug',53)
    (new_sym, new_args) = get_fine_tune_sqeezenet_model(sym, arg_params, num_classes)
    # (new_sym, new_args) = get_fine_tune_mobile_model(sym, arg_params, num_classes)
    if not os.path.exists("./log"):
        os.mkdir("./log")
    hdlr = logging.FileHandler('./log/log-seq-{}-{}.log'.format("224-dataaug", shape))
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    (train, val) = get_iterators(batch_size,data_shape=(3, shape, shape))
    mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size)
    assert mod_score > 0.77, "Low training accuracy."

