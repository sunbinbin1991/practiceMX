import mxnet as mx
import cv2
import cPickle as pickle
import random
def PKL2rRec():
    celebAPath = '/home/sbb/data/pkl/celebA_Alllabel_268489.pkl'
    # celebAPath = '/home/sbb/data/pkl/celebA_112x112_4268.pkl'
    fid = open(celebAPath, 'rb')
    celebApkltemp = pickle.load(fid)
    result = []
    record_train = mx.recordio.MXRecordIO('/home/sbb/data/record/train.rec', 'w')
    record_val = mx.recordio.MXRecordIO('/home/sbb/data/record/val.rec', 'w')
    # print celebApkltemp
    valratio = 0.9
    random.shuffle(celebApkltemp)
    for item, value in enumerate(celebApkltemp):
        img = value['data']
        print item
        img_resize = cv2.resize(img, (112, 112))
        label = [value['illumination'], value['blur']]
        header = mx.recordio.IRHeader(0, label, item, 0)
        packed_s = mx.recordio.pack_img(header, img_resize)
        if item<268489*valratio:
            record_train.write(packed_s)
        else:
            record_val.write(packed_s)
        # cv2.waitKey(0)
    record_train.close()
    record_val.close()
    # outPut = open('/home/sbb/data/pkl/celebA_112x112_%d.pkl' % (result.__len__()), 'wb')
    # pickle.dump(result, outPut, protocol=2)
    # outPut.close()

def readRec():
    # record = mx.recordio.MXRecordIO('tmp.rec', 'w')
    # for i in range(5):
    #     img = cv2.imread('test.jpg')
    #     label = [i,i+1] # label can also be a 1-D array, for example: label = [1,2,3]
    #     id = 2574
    #     header = mx.recordio.IRHeader(0, label, id,0)
    #     packed_s = mx.recordio.pack_img(header, img)
    #     record.write(packed_s)
    # record.close()
    record = mx.recordio.MXRecordIO('/home/sbb/data/record/val.rec', 'r')
    while True:
        item = record.read()
        header, img = mx.recordio.unpack_img(item)
        print header
        cv2.imshow("img",img)
        cv2.waitKey(0)
    # for i in range(5):
    #     item = record.read()
    #     print(item)

readRec()