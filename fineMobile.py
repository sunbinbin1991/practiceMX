import mxnet as mx
#fine-tune an model
def get_iterators(batch_size, data_shape=(3, 128, 128)):
    train = mx.io.ImageRecordIter_v1(
        path_imgrec         = './record/spott_train.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        # mean_r              = 123,
        # mean_g              = 117,
        # mean_b              = 104,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True)
    val = mx.io.ImageRecordIter_v1(
        path_imgrec         = './record/spott_val.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        # mean_r              = 123,
        # mean_g              = 117,
        # mean_b              = 104,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True)
    return (train, val)

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='drop7'):
    """
    symbol: the pretrained network symbol
    arg_params: the argument parameters of the pretrained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    #net = all_layers[layer_name]
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc8')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc8' not in k})
    return (net, new_args)

def get_fine_tune_sqeezenet_model(symbol, arg_params, num_classes, layer_name='drop9'):
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

def get_fine_tune_mobile_model(symbol, arg_params, num_classes, layer_name='pooling0'):
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    # pool6 = mx.symbol.Pooling(name='pool6', data=net , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
    fc7 = mx.symbol.Convolution(name='fc7', data=net , num_filter=num_classes, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
    flatten = mx.symbol.Flatten(data=fc7, name='flatten')
    net = mx.symbol.SoftmaxOutput(data=flatten, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc7' not in k})
    return (net, new_args)


import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol, context=devs)
    save_prefix = "./modelzoor/mobilenet-0.5-128-E"
    mod.fit(train, val,
        num_epoch=30,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 1000),
        epoch_end_callback = mx.callback.do_checkpoint(save_prefix),
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate':0.001},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)


if __name__=="__main__":
    print "success"
    num_classes = 2
    batch_size = 512
    # batch_per_gpu = 16
    num_gpus = 2
    #load model
    prefix = "./modelzoor/mobilenet-0.5-128-D"
    # print'%s-symbol.json' % prefix
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix,26)
    (new_sym, new_args) = get_fine_tune_mobile_model(sym, arg_params, num_classes)

    (train, val) = get_iterators(batch_size)
    mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus)
    assert mod_score > 0.77, "Low training accuracy."

