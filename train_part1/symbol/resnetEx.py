import mxnet as mx
from resnetEx import *

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_batchnorm0')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_activation0')
    conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                  no_bias=False, workspace=workspace, name=name + '_conv0')
    # drop = mx.symbol.Dropout(data=conv1, p=0.3)
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_batchnorm1')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_activation1')
    conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                  no_bias=False, workspace=workspace, name=name + '_conv1')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=False,
                                        workspace=workspace, name=name+'_conv2')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv2 + shortcut

def residual_unit_stage2(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_batchnorm2')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_activation2')
    conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                  no_bias=False, workspace=workspace, name=name + '_conv3')
    # drop = mx.symbol.Dropout(data=conv1, p=0.3)
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_batchnorm3')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_activation3')
    conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                  no_bias=False, workspace=workspace, name=name + '_conv4')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=False,
                                        workspace=workspace, name=name+'_conv2')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv2 + shortcut

def resnetBin(): # 0719 replace all pooling to conv
    filter_list =  [32, 64]
    num_stages = 1
    units = [2, 2, 2, 2]
    bn_mom = 0.9
    workspace = 256
    bottle_neck = False
    memonger = False
    data = mx.symbol.Variable(name="data")
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='hybridsequential0_batchnorm0')
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                              no_bias=True, name="hybridsequential0_conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='hybridsequential0_batchnorm1')
    body = mx.sym.Activation(data=body, act_type='relu', name='hybridsequential0_relu0')
    body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type='max',pooling_convention='full',name='hybridsequential0_pool0')
    for i in range(num_stages):
        body = residual_unit(body, filter_list[i + 1], (2, 2), False,
                             name='hybridsequential0_stage%d' % (i + 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i] - 1):
            body = residual_unit_stage2(body, filter_list[i + 1], (1,1), True, name='hybridsequential0_stage%d' % (i + 1),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)

    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='hybridsequential0_batchnorm2')
    # elu1 = mx.symbol.LeakyReLU(data=bn1, act_type='elu', name='elu1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='hybridsequential0_relu1')
    convadd = mx.sym.Convolution(data=relu1, num_filter=32, kernel=(3, 3), stride=(2, 2), pad=(0, 0),
                               no_bias=False, name="hybridsequential0_conv1", workspace=workspace)
    pool1 = mx.sym.Pooling(data=convadd, global_pool=True, kernel=(7, 7), pool_type='avg', name='hybridsequential0_pool1')
    Conv1 = mx.sym.Convolution(data=pool1, num_filter=1, kernel=(1, 1), no_bias=False, name="hybridsequential0_conv2", workspace=workspace)
    Conv2 = mx.sym.Convolution(data=pool1, num_filter=1, kernel=(1, 1), no_bias=False, name="hybridsequential0_conv3", workspace=workspace)
    # fc1 = mx.symbol.FullyConnected(data=Conv1, num_hidden=1, name='fc1')
    # fc2 = mx.symbol.FullyConnected(data=Conv2, num_hidden=1, name='fc2')
    out1= mx.symbol.SoftmaxOutput(data =Conv1,name = "softmax1")
    # out2= mx.symbol.SoftmaxOutput(data =Conv2,name="softmax2")
    # out1= mx.symbol.softmax(data =Conv1,name = "softmax1")
    # out2= mx.symbol.softmax(data =Conv2,name="softmax2")
    # net = mx.symbol.Group([fc1,fc2])
    net = mx.symbol.Group([out1])
    return net
