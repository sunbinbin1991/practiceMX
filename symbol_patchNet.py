import mxnet as mx

def patchNet(num_classes,**kwargs):
    data=mx.sym.Variable(name='data')
    conv1=mx.symbol.Convolution(data=data, num_filter=50, kernel=(5,5), stride=(3,3), pad=(2,2))
    bn1 = mx.symbol.BatchNorm(data=conv1)
    act1 = mx.symbol.Activation(data=bn1, act_type='relu')
    pool1 = mx.symbol.Pooling(data=act1, pool_type="max", pooling_convention="full", kernel=(2, 2), stride=(2, 2), name="pool1")
    # 48*48*50

    conv2=mx.symbol.Convolution(data=pool1, num_filter=100, kernel=(3,3), stride=(1,1),pad=(1,1))
    bn2 = mx.symbol.BatchNorm(data=conv2)
    act2 = mx.symbol.Activation(data=bn2, act_type='relu')
    pool2 = mx.symbol.Pooling(data=act2, pool_type="max", pooling_convention="full", kernel=(2, 2), stride=(2, 2), name="pool2")
    # 24*24*100

    conv3=mx.symbol.Convolution(data=pool2, num_filter=150, kernel=(3,3), stride=(1,1),pad=(1,1))
    bn3 = mx.symbol.BatchNorm(data=conv3)
    act3 = mx.symbol.Activation(data=bn3, act_type='relu')
    pool3 = mx.symbol.Pooling(data=act3, pool_type="max", pooling_convention="full", kernel=(2, 2), stride=(2, 2), name="pool3")
    # 12*12*150

    conv4=mx.symbol.Convolution(data=pool3, num_filter=200, kernel=(3,3), stride=(1,1),pad=(1,1))
    bn4 = mx.symbol.BatchNorm(data=conv4)
    act4 = mx.symbol.Activation(data=bn4, act_type='relu')
    pool4 = mx.symbol.Pooling(data=act4, pool_type="max", pooling_convention="full", kernel=(2, 2), stride=(2, 2), name="pool4")
    # 6*6*200

    conv5=mx.symbol.Convolution(data=pool4, num_filter=250, kernel=(3,3), stride=(1,1),pad=(1,1))
    bn5 = mx.symbol.BatchNorm(data=conv5)
    act5 = mx.symbol.Activation(data=bn5, act_type='relu')
    pool5 = mx.symbol.Pooling(data=act5, pool_type="max", pooling_convention="full", kernel=(2, 2), stride=(2, 2), name="pool5")
    # 3*3*250

    fc_1 = mx.symbol.FullyConnected(data=pool5, num_hidden=1000, name='fc_1')
    bn6 = mx.symbol.BatchNorm(data=fc_1)
    drop = mx.symbol.Dropout(data = bn6, p = 0.5)
    # 1*1*1000

    fc_2 = mx.symbol.FullyConnected(data=drop, num_hidden=400, name='fc_2')
    bn7 = mx.symbol.BatchNorm(data=fc_2)
    fc_3 = mx.symbol.FullyConnected(data=bn7, num_hidden=num_classes, name='fc_3')
    softmax = mx.symbol.SoftmaxOutput(data=fc_3, name='softmax')
    return softmax

def get_symbol_patchNet():
    num_classes = 2
    net = patchNet(num_classes)
    return net