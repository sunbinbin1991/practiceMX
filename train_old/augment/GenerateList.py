import os
import random

def list_image(root, recursive):
    i = 0
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                if(i%10000==0):
                    print "=======  Process recursive list ========= list Num = %d"%i
                fpath = os.path.join(path, fname)
                if os.path.isfile(fpath):
                    if path not in cat:
                        cat[path] = len(cat)
                    yield (i, os.path.relpath(fpath, root), cat[path])
                    i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, root), v)

    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath):
                yield (i, os.path.relpath(fpath, root), 0)
                i += 1

def write_list(path_out, image_list,prefix,useperfix):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            print item
            path =item[1].replace("\\","/")
            if useperfix == True:
                path = prefix+"/"+ path
            line = '%d\t' % item[0]#num
            line += '%s\t' % item[2]#label
            line += '%s\n' % path
            # print line
            fout.write(line)

def write_list_celebA(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            path = "facegender_train/"+item[1]
            # print item
            Label = item[1].split("_")[0].split("/")[0]
            if Label =="female":
                res = 0
            if Label =="male":
                res = 1
            line = '%d\t' % i
            line += '%s\t' % res
            line += '%s\n' % path
            print i,Label,path
            fout.write(line)

def make_list(root,recursive,shuffle,train_ratio,prefix,usePrefix):
    test_ratio = 0.1
    val_ratio = 1 - train_ratio - test_ratio
    image_list=list_image(root,recursive)
    image_list = list(image_list)
    if shuffle is True:
        random.shuffle(image_list)
    N = len(image_list)
    chunk_size = N
    chunk = image_list
    sep_test = int(chunk_size * test_ratio)
    sep_val = int(chunk_size * val_ratio)
    if train_ratio == 1:
        write_list(prefix  + '.lst', chunk,prefix,usePrefix)
    else:
        if test_ratio:
            write_list(prefix  + '_test.lst', chunk[:sep_test])
        if val_ratio:
            write_list(prefix  + '_val.lst', chunk[sep_test:sep_test+sep_val])
        if train_ratio:
            write_list(prefix + '_train.lst', chunk[sep_test+sep_val:])

def read_list(path_in):
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split()]
            line_len = len(line)
            # if line_len < 3:
            #     print('lst should at least has three parts, but only has %s parts for %s' %(line_len, line))
            #     continue
            # try:
            item = [int(line[0])] + [float(i) for i in line[1:-1]] + [line[-1]]
            # print line
            # item = [int(line[0])] + [line[-1]]
            # except Exception, e:
            #     print('Parsing lst met error for %s, detail: %s' %(line, e))
            #     continue
            yield item
