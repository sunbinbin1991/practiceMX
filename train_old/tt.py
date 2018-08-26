import os
import mxnet as mx
print "hehe"
# train_list = "train_val.lst"
# with open(train_list) as f:
#     cou = 0
#     data_dict = {}
#     label_dict ={}
#     while True:
#         line =  f.readline()
#         # print line
#         if not line:
#             break
#         index = line.split("\t")[0]
#         label = line.split("\t")[1]
#         path = line.split("\t")[2]
#         img = mx.img.imdecode(path)
#         data_dict[index] = img
#         label_dict[index] = label
#         # print dict
#         cou +=1
#     for k, v in data_dict.items():
#         print k,v,type(k),int(k)
#     # print cou