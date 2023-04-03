import os
import skimage.io
import numpy as np
DEBUG_DIR = "debugging"
if not os.path.isdir(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)
def save_img(img_np,savename,savefolder=DEBUG_DIR,sync=False):
    full_savename = os.path.join(savefolder,savename)
    skimage.io.imsave(full_savename,img_np)
    if sync:
        os.system(f'rclone sync -Pv {full_savename} aniket-gdrive:stable-diffusion-experiments/debugging')
    pass
def save_results(impath,results_root,
                 gen_image_np,repasted_np,inp_image_np,extras,save_prefix='generated_all'):
    rootname = os.path.basename(impath).split('.')[0]
    if False and "make-subfolder":
        savefolder  = os.path.join(results_root,rootname)
        try:
            os.makedirs(savefolder)
        except FileExistsError:
            pass
    else:
        savefolder  = results_root
    if False and "save individual images":
        skimage.io.imsave(os.path.join(savefolder,'generated.png'),gen_image_np)
        skimage.io.imsave(os.path.join(savefolder,'repasted.png'),repasted_np)
        skimage.io.imsave(os.path.join(savefolder,'holefilled.png'),extras['holefilled'])
        skimage.io.imsave(os.path.join(savefolder,'initial_repasted.png'),extras['initial_repasted'])
        
    if "save grid layout image":
        
        final_sz = gen_image_np.shape
        np_final = np.zeros((2*final_sz[0], 2*final_sz[1], final_sz[2]))
        #np_final = np.zeros((2*gen_img_np.shape[0], 2*gen_image_np.shape[1], gen_image_np.shape[2]))
        np_final[:final_sz[0],:final_sz[1],:] = gen_image_np
        np_final[final_sz[0]:,:final_sz[1],:] = repasted_np
        np_final[:final_sz[0],final_sz[1]:,:] = extras['holefilled']
        np_final[final_sz[0]:,final_sz[1]:,:] = inp_image_np
        skimage.io.imsave(os.path.join(savefolder,f'{rootname}_{save_prefix}.png'),np_final)

class ChangeDir():
    def __init__(self, new_path):
        self.new_path = new_path
        self.saved_path = os.getcwd()
    def __enter__(self):
        os.chdir(self.new_path)
    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)
        
def cipdb(flag,val='1'):
    if os.environ.get(flag,False) == val:
        import ipdb; ipdb.set_trace()
    pass
