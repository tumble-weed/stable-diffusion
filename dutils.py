# import model.utils 
import os
my_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(my_dir,'debug-results')
import os
import skimage.io
from matplotlib import pyplot as plt
def img_save_(im, path):
    dir = os.path.splitext(os.path.dirname(path))[0]
    # if not os.path.isdir(dir):
    #     os.mkdir(dir)
    os.makedirs(dir,exist_ok=True)
    # imsave(path, img_as_ubyte(im))
    if im.max() > 1:
        im = im/im.max()
    skimage.io.imsave(path,im)
img_save = lambda im,filename,ROOT_DIR=ROOT_DIR:img_save_(im,os.path.join(ROOT_DIR,filename))
def save_plot_(y,title,filename):
    plt.figure()
    plt.plot(y)
    plt.title(title)
    plt.draw()
    plt.savefig(filename)
    plt.close()
    pass
save_plot = lambda y,title,filename,ROOT_DIR=ROOT_DIR:save_plot_(y,title,os.path.join(ROOT_DIR,filename))
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
def array_info(*args):
    for ar in args:
        try:
            print(ar.__class__)
            print(ar.shape)
            print(ar.min())
            print(ar.max())
            print('*'*10)
        except Exception as e:
            print('exception')
    pass

def show_pyramids(gpnn_inpainting,pi):
    import inspect
    # v =  inspect.currentframe().f_back.f_locals
    # gpnn_inpainting = v['gpnn_inpainting']
    try:
        img_save(tensor_to_numpy(gpnn_inpainting.mask_pyramid[pi][0,:,:,0]),'mask_pyramid.png')
        img_save(tensor_to_numpy(gpnn_inpainting.x_pyramid[pi][0,:,:,0]),'x_pyramid.png')
        img_save(tensor_to_numpy(gpnn_inpainting.y_pyramid[pi][...,:3][0]),'y_pyramid.png')
    except Exception as e:
        print(e)

def UPDATE_PAPER(d):
    pass
#========================================
import IPython.core.ultratb
import sys
def e():
    tb = IPython.core.ultratb.VerboseTB()
    print(tb.text(*sys.exc_info()))
#========================================
def list_info(*args):
    for ar in args:
        try:
            print(ar.__len__())
            if len(ar):
                print(ar[0].__class__)
        except Exception as e:
            print('exception')
    pass

import atexit
def place_process_lock(filename):
    pid = os.getpid()
    lockname = filename + f'{pid}.lock'
    dircontents = os.listdir(os.path.dirname(filename))
    for el in dircontents:
        if el.startswith(filename) and el.endswith('.lock'):
            owner = el[len(filename):]
            owner = el[-len('.lock'):]
            owner = int(owner)
            if pid != owner:
                assert False, f'{owner} owns the file. i am {pid}'
    os.system(f'touch {lockname}')
    def delete_file():
        os.remove(lockname)
    atexit.register(delete_file)
    
