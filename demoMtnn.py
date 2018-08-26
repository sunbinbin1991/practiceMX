import mxnet as mx
from anti.mtcnn_detector import MtcnnDetector

detector = MtcnnDetector(model_folder='model', ctx=mx.gpu(0), num_worker=4, accurate_landmark=False)