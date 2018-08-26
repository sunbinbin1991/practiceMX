import cv2
import os
def list_image(root, recursive):
    i = 0
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                if fname.endswith(".jpg") or fname.endswith(".bmp"):
                    if(i%1000==0):
                        print i
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

if __name__=="__main__":
    list_image()
    workingdir= "/home/sbb/data/antiSpoof/NIR-VIS/alive/"
    save_dir = "/home/sbb/data/antiSpoof/Train/"
    pass