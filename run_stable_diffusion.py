import setup
from diffusers import StableDiffusionInpaintPipeline
import torch
import matplotlib.pyplot as plt 
import numpy
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import skimage.io
import skimage.transform
from rembg import remove
import dutils
import copy
import glob
import colorful
import debug
from utils import save_results
import utils
from bg_removal_clip import bgremove_clip
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
HARMONIZE = True
USE_CLIPSEG = False
if os.environ.get('USE_CLIPSEG',None) == "1":
    USE_CLIPSEG = True
debug.DEBUG_HARMONIZATION = True
####################################################################
# flags for holefilling
os.environ['DBG_IGNORE_RESIZE']='1'
os.environ['PNN_XRANGE_ERROR']='0'
os.environ['DBG_NO_UPSIZE']='0'
def init_sd(device = "cuda"):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", # 'runwayml/stable-diffusion-inpainting'
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    return pipe
####################################################################

def prepare_image(impath):
    # import ipdb; ipdb.set_trace()
    orig_img_pil = Image.open(impath).convert('RGB')
    width = orig_img_pil.size[0]
    height = orig_img_pil.size[1]
    orig_img_np = np.array(orig_img_pil)
    if orig_img_np.max() > 1:
        orig_img_np = orig_img_np/255.
    if orig_img_np.shape[-1] == 4:
        orig_img_np = orig_img_np[...,:3]
    if orig_img_np.dtype == np.uint8:
        orig_img_np = orig_img_np.astype(np.float32)
    if False and 'hardcode for bisleri':
        coords="362,884,377,947,419,975,464,986,514,983,547,976,597,962,612,956,637,884,637,833,636,776,638,717,637,678,629,660,632,423,636,405,643,353,640,306,628,263,616,230,589,179,557,143,550,116,561,110,549,97,563,91,553,35,433,38,426,44,426,96,433,102,421,112,433,116,433,140,409,168,389,194,374,228,355,272,355,289,354,336,354,406,358,426,360,648"
        coords = coords.split(',')
        poly_coord = []
        for i in coords: 
            poly_coord.append(int(i))
        
        polygon = poly_coord #[(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
        mask_img_pil = Image.new('L', (width, height), 0)
        ImageDraw.Draw(mask_img_pil).polygon(polygon, outline=1, fill=1)
        mask_np = numpy.array(mask_img_pil)
    else:
        if True:
            # import ipdb; ipdb.set_trace()            
            if False and USE_CLIPSEG:
                mask_np = bgremove_clip(orig_img_np,["flask"],max_val=255)
                # (mask_np == 255).astype(np.uint8).sum()
                utils.cipdb('DBG_CLIPSEG')
            else:
                mask_img_pil = remove(orig_img_pil) # remove background
                mask_np = np.array(mask_img_pil).astype(np.float32)        
                mask_np = mask_np[...,-1]
            mask_np = np.stack([mask_np,mask_np,mask_np],axis=-1)
            if mask_np.max() > 1:
                mask_np = mask_np/255.
            mask_np = (mask_np > 0.5).astype(np.float32)
        else:
            mask_np = create_mask(orig_img_np)
            mask_np = (mask_np > 0.5).astype(np.float32)
            # import ipdb; ipdb.set_trace()
        # mask_img_pil = Image.fromarray((mask_np*255.).astype(np.uint8)).convert('L')
        # import ipdb;ipdb.set_trace()
    # return orig_img_pil,mask_np,mask_img_pil
    
    return orig_img_np,mask_np
####################################################################

def place_in_large_image(orig_img_np,orig_mask_np):
    # target_height,target_width = 256,256
    max_height, max_width = 312, 312
    current_height,current_width =  orig_img_np.shape[:2]
    if current_height > current_width:
        target_height = max_height
        target_width = (current_width/current_height) * target_height
        target_width = int(target_width)
    else:
        target_width = max_width
        target_height = (current_height/current_width) * target_width
        target_height = int(target_height)        
    # import pdb; pdb.set_trace()
    if False:
        obj_height = orig_mask_np[...,0].sum(axis=0).max()
        obj_width = orig_mask_np[...,0].sum(axis=1).max()
        obj_height_ratio = obj_height/orig_mask_np.shape[0]
        obj_width_ratio = obj_width/orig_mask_np.shape[1]
        target_obj_height_ratio = 0.5
        target_obj_width_ratio = 0.2
        #.....................................
        # get 2 possible target height,wdth pairs, and choose the larger one
        # target_obj_height_ratio = (obj_height_ratio * target_height)/512
        target_height1 = obj_height_ratio/(512*target_obj_height_ratio)
        # target_height/target_width = orig_mask_np.shape[0]/orig_mask_np.shape[1]
        target_width1 = target_height1/(orig_mask_np.shape[0]/orig_mask_np.shape[1])
        target_width2 = obj_width_ratio/(512/target_obj_width_ratio)
        target_height2 = target_width2/(orig_mask_np.shape[1]/orig_mask_np.shape[0])
        if target_width2 > target_width1:
            target_height,target_width = target_height2,target_width2
        else:
            target_height,target_width = target_height1,target_width1
    # import ipdb; ipdb.set_trace()
    #.....................................
    # small_res_np = np.array(orig_img_pil.resize((256,256)))
    small_res_np = skimage.transform.resize(orig_img_np,(target_height,target_width))
    small_canvas_np = np.full((target_height,target_width,3), 1.)
    # small_mask_img_np = np.array(orig_mask_pil.resize((256,256))) # mask of the object
    small_mask_np = skimage.transform.resize(orig_mask_np,(target_height,target_width))
    assert small_mask_np.max() <= 1
    max_val = 1
    # small_out_np = np.where(small_mask_np == (max_val,max_val,max_val), small_res_np, small_canvas_np)
    small_out_np = small_res_np * small_mask_np +  small_canvas_np * (1- small_mask_np)
    print(small_out_np.shape)
    #==============================================================
    
    large_img_np = np.full((512,512,3), 1.)
    # x_offset=y_offset=128
    y_offset = (512 - target_height)//2
    x_offset = (512 - target_width)//2
    large_img_np[y_offset:y_offset+small_out_np.shape[0], x_offset:x_offset+small_out_np.shape[1]] = small_out_np

    large_mask_np = np.full((512,512,3), 0,dtype=np.float32)
    large_mask_np[y_offset:y_offset+small_out_np.shape[0], x_offset:x_offset+small_out_np.shape[1]] = small_mask_np
    #==============================================================
    # pdb.set_trace()
    # import ipdb; ipdb.set_trace()
    return large_img_np,large_mask_np

def create_mask_(img_np):
    # import ipdb; ipdb.set_trace()
    assert img_np.max() <= 1
    if USE_CLIPSEG:        
        mask_np = bgremove_clip(img_np,["flask"],max_val=1)        
        mask_np = np.stack([mask_np,mask_np,mask_np],axis=-1)
        utils.cipdb('DBG_CLIPSEG')
    else:
        img_pil = Image.fromarray((img_np*255.).astype(np.uint8)).convert('L')
        mask_pil = remove(img_pil) # remove background
        mask_np = np.array(mask_pil)
        assert mask_np.ndim == 3
        assert mask_np.shape[-1] == 4
        mask_np = mask_np[...,-1]
        if mask_np.max() > 1:
            mask_np = mask_np/255.
        mask_np = np.stack([mask_np,mask_np,mask_np],axis=-1)
    return mask_np
def create_mask(img_np):
    '''
    will run create_mask_ at multiple scales and aggregate the results
    '''
    print(colorful.green("TODO: determine min and max scale according to the relative size of the object segmented at scale 1."))
    min_scale,max_scale = 0.75,1.5
    n_scales = 5
    scales = np.linspace(min_scale,max_scale,n_scales)
    mask_np = 0
    for s in scales:
        if img_np.ndim ==2:
            img_s = skimage.transform.rescale(img_np,s)
        else:
            img_s = np.stack([skimage.transform.rescale(img_np[...,i],s) for i in range(img_np.shape[-1])],axis=-1)
        #import ipdb; ipdb.set_trace()
        mask_s = create_mask_(img_s)
        if mask_s.ndim ==2:
            mask_s = skimage.transform.rescale(mask_s,1/s)
        else:
            mask_s = np.stack([skimage.transform.rescale(mask_s[...,i],1/s) for i in range(mask_s.shape[-1])],axis=-1)
        mask_np = np.maximum(mask_np,mask_s)
    # mask_np = mask_np/n_scales
    return mask_np
def repaste_after_downshift(holefilled_np,large_mask_np_float,large_image_np,gen_mask_np,large_mask_np):
    # shift the object in large_image downwards
    Y,X = np.meshgrid(np.arange(gen_mask_np.shape[0]),np.arange(gen_mask_np.shape[1]),indexing='ij')
    YX = np.stack([Y,X],axis=-1)
    lowest_gen = (gen_mask_np.mean(axis=-1) * Y).argmax(axis=0).max()
    lowest_large = (large_mask_np.mean(axis=-1) * Y).argmax(axis=0).max()
    to_move = 0
    if lowest_gen > lowest_large:
        to_move = lowest_gen - lowest_large
        # if to_move > 30:
        #     os.environ['shifted_down'] = '1'
        new_large_image_np = np.zeros(large_image_np.shape)
        new_large_image_np[to_move:,...] = large_image_np[:-to_move]
        
        new_large_mask_np_float = np.zeros(large_mask_np.shape)
        new_large_mask_np_float[to_move:,...] = large_mask_np_float[:-to_move]
    else:
        new_large_mask_np_float = large_mask_np_float
        new_large_image_np = large_image_np
    repasted = holefilled_np*(1-new_large_mask_np_float) + (new_large_image_np* new_large_mask_np_float)
    if HARMONIZE:
        from harmonize import harmonize
        repasted_pre = repasted
        debug.repasted_pre = repasted_pre
        if new_large_mask_np_float.ndim == 3:
            new_large_mask_np_float_2d = new_large_mask_np_float[...,:1]
        elif new_large_mask_np_float.ndim == 2:
            new_large_mask_np_float_2d = new_large_mask_np_float[...,None]
        repasted = harmonize(repasted_pre,
        new_large_mask_np_float_2d,device=None)
        # import ipdb; ipdb.set_trace()
    return repasted

def holefill_and_repaste(large_image_np,gen_image_np,gen_mask_np,large_mask_np):
    large_mask_np_float = large_mask_np
    # NOTE: use threshold of 0 here to make a large mask to fill
    large_mask_np = (large_mask_np_float > 0.).astype(np.float32)
    gen_mask_np = (gen_mask_np > 0.).astype(np.float32)
    
    worst_mask_np = np.maximum(gen_mask_np[...,-1].astype(np.float32),
                                large_mask_np[...,-1].astype(np.float32))

    if worst_mask_np.max() > 1:
        assert False
        worst_mask_np /= 255.
    worst_mask_np = (worst_mask_np>0.).astype(np.float32)


    assert gen_image_np.max() <= 1
    if False and 'gpnn':
        config = {
            'out_dir':'stable-diffusion/output',
            'iters':10,
            # 'iters':1,#10
            'coarse_dim':14,#
            # 'coarse_dim':28,
            # 'coarse_dim':100,#
            'out_size':0,
            'patch_size':7,
            # 'patch_size':15,
            'stride':1,
            'pyramid_ratio':4/3,
            # 'pyramid_ratio':2,
            'faiss':True,
            # 'faiss':False,
            'no_cuda':False,
            #---------------------------------------------
            'in':None,
            'sigma':4*0.75,
            # 'sigma':0.3*0.75,
            'alpha':0.005,
            'task':'inpainting',
            #---------------------------------------------
            # 'input_img':original_imname,
            # 'input_img':tensor_to_numpy(denormalize_imagenet(img.unsqueeze(0)).permute(0,2,3,1)[0]),
            # 'input_img':skimage.transform.resize(np.array(image)/255.,(256,256)),
            # 'input_img':(np.array(image)/255.),
            'input_img':gen_image_np,
            # NOTE: the mask arrives with 0's at holes. need to flip it
            # 'mask':tensor_to_numpy(1 - mask),
            'mask':worst_mask_np,
            # 'mask':(skimage.transform.resize(mask.astype(np.float32),(256,256))>0.5).astype(np.float32),
            'batch_size':10,
            #---------------------------------------------
            'implementation':'gpnn',#'efficient-gpnn','gpnn'
            'init_from':'zeros',#'zeros','target'
            'keys_type':'single-resolution',#'multi-resolution','single-resolution'
            #---------------------------------------------
            'use_pca':True,
            'n_pca_components':30,
            #---------------------------------------------
            'patch_aggregation':'distance-weighted',#'uniform','distance-weighted','median'
            # 'imagenet_target':imagenet_target
            #---------------------------------------------
            'index_type':'simple',
            'use_xy':True,
            }
        # from model.my_gpnn_inpainting2 import gpnn
        from model.my_gpnn_inpainting_for_sd import gpnn
        # import ipdb; ipdb.set_trace()
        gpnn_inpainting = gpnn(config)
        holefilled,holefilling_results = gpnn_inpainting.run(to_save=False)
    else:
        from hole_filling import lama_holefill
        holefilled = lama_holefill(gen_image_np, worst_mask_np)
    
    # repasted = copy.deepcopy(tensor_to_numpy(holefilled[:,:3].permute(0,2,3,1)[0]))
    holefilled_np = copy.deepcopy(tensor_to_numpy(holefilled[:,:3].permute(0,2,3,1)[0]))
    # import ipdb; ipdb.set_trace()
    if False:
        repasted = holefilled_np*(1-large_mask_np_float) + (large_image_np* large_mask_np_float)
    else:
        repasted = repaste_after_downshift(holefilled_np,large_mask_np_float,large_image_np,gen_mask_np,large_mask_np)

            
        
        
    # import ipdb; ipdb.set_trace()
    
    return repasted,holefilled
def vae_encode_decode(repasted_np,pipe):
    assert repasted_np.min() >= 0. 
    assert repasted_np.max() <= 1.
    device = 'cuda'
    repasted_th = torch.tensor(repasted_np,device=device).half().unsqueeze(0).permute(0,3,1,2)
    repasted_th = (repasted_th - 0.5)*2
    repasted_latents = pipe.vae.encode(repasted_th).latent_dist.sample(generator=None)
    # repasted_latents = 1 / pipe.vae.config.scaling_factor * repasted_latents
    # repasted_latents = pipe.vae.config.scaling_factor * repasted_latents
    repasted2 = pipe.vae.decode(repasted_latents).sample
    repasted2 = (repasted2 / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    repasted2_np = tensor_to_numpy(repasted2.permute(0, 2, 3, 1))[0]
    return repasted2_np
# assert False
def run(impath='flask3.jpg'):
    
    orig_img_np,orig_mask_np = prepare_image(impath)

    # import ipdb; ipdb.set_trace()
    
    large_image_np,large_mask_np = place_in_large_image(orig_img_np,orig_mask_np)
    inverted_large_mask_np = 1 - large_mask_np
    GOOD = False
    while not GOOD:
        ## Testing the Stable Diffusion: 
        prompt = "hyperrealistic photo of the object lying on a table in front of nigara falls. HD, 4K, 8K, render, lens flare, product photo, generate new prospective"
        neg_prompt = 'ugly, black and white, blurr, oversaturated, 3d, render, cartoon, fusioned, deformed, mutant, bad anatomy, extra hands and fingers'

        #new_image = pipe(prompt=prompt, image=image, mask_image=inverted_mask, negative_prompt=neg_prompt, num_inference_steps=50,
                        #guidance_scale=8).images[0]
        
        pipe = init_sd(device = "cuda")                
        inverted_large_mask_pil = Image.fromarray((inverted_large_mask_np*255.).astype(np.uint8)).convert('RGB')
        large_image_pil = Image.fromarray((large_image_np*255.).astype(np.uint8)).convert('RGB')
        # import ipdb; ipdb.set_trace()
        num_inference_steps = 50
        if os.environ.get('DBG_FAST_SD',False):
            num_inference_steps = int(os.environ['DBG_FAST_SD'])
        new_image_obj = pipe(prompt=prompt, image=large_image_pil, mask_image=inverted_large_mask_pil, negative_prompt=neg_prompt, num_inference_steps=num_inference_steps,
                        guidance_scale=8)
        gen_image_pil = new_image_obj.images[0]
        gen_image_np = np.array(gen_image_pil)/255.
        skimage.io.imsave('gen_image_np.png',gen_image_np)
        # orig_img_pil = gen_image_pil
        if False and 'mask using polygon':
            ####################################################################
            coords2 = "191,104,190,389,308,387,308,106"
            coords2 = coords2.split(',')
            poly_coord = []
            for i in coords2: 
                poly_coord.append(int(i))
            ####################################################################
            
            polygon = poly_coord #[(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
            width = gen_image_pil.size[0]
            height = gen_image_pil.size[1]

            img_pil = Image.new('L', (width, height), 0)
            ImageDraw.Draw(img_pil).polygon(polygon, outline=1, fill=1)
            mask_np = numpy.array(img_pil)
            # assert False
        
        # large_img_mask_np,gen_mask_np,gen_image_np = create_masks(orig_img_pil,gen_image_np)
        gen_mask_np = create_mask(gen_image_np)
        assert gen_mask_np.max() <= 1
        # if large_img_mask_np.max() >1:
        #     large_img_mask_np = large_img_mask_np/255.
        # if gen_mask_np.max() >1:
        #     gen_mask_np = gen_mask_np/255.
        large_mask_np_float = large_mask_np
        large_mask_np = (large_mask_np>0.5).astype(np.float32)
        
        area_discrepancy = np.abs(gen_mask_np[:,:,-1] - large_mask_np[:,:,-1]).sum()/np.abs(large_mask_np[:,:,-1]).sum()
        # area_discrepancy2 = worst_mask_np[:,:].sum()/large_mask_np[:,:,-1].sum()
        print(colorful.red("remove images that have large holes"))
        print(area_discrepancy)
        if area_discrepancy < 0.2:
            GOOD = True
        else:
            print(colorful.yellow("too large a difference in sizes of generated object and original object, redoing"))
        # import ipdb; ipdb.set_trace()        
        # mask_np = (gen_mask_np>0.5).astype(np.float32)
        

    
    repasted_np,holefilled = holefill_and_repaste(large_image_np,gen_image_np,gen_mask_np,large_mask_np_float)    
    
    holefilled_np = tensor_to_numpy(holefilled[:,:3].permute(0,2,3,1)[0])
    extras = {'holefilled':holefilled_np,'initial_repasted':repasted_np}
    # import ipdb; ipdb.set_trace()
    #============================================================
    # vae fine tuning
    # prev_pil = Image.fromarray((repasted_np*255.).astype(np.uint8)).convert('L')
    # prev_mask_pil = remove(prev_pil)
    # prev_mask_np = np.array(prev_mask_pil).astype(np.float32)        
    repasted2_np = repasted_np
    if False:
        for i in range(1):
            repasted2_np = vae_encode_decode(repasted2_np,pipe)
            # repasted2_pil = Image.fromarray((repasted2_np*255.).astype(np.uint8)).convert('L')
            # repasted2_mask_pil = remove(repasted2_pil)
            # repasted2_mask_np = np.array(repasted2_mask_pil).astype(np.float32)   
            # repasted2_mask_np = repasted2_mask_np[...,-1]    
            # if repasted2_mask_np.max() > 1:
            #     repasted2_mask_np = repasted2_mask_np/255.        
            repasted2_mask_np = create_mask(repasted2_np)
            repasted2_np,_ = holefill_and_repaste(large_image_np,repasted2_np,repasted2_mask_np,large_mask_np)    
        #============================================================
        
    # import ipdb; ipdb.set_trace()
    # return     
    return gen_image_np,repasted2_np,extras, large_image_np

def run_on_folder(folder,results_root='results'):
    if not os.path.isdir(results_root):
        os.makedirs(results_root)
    for impath in glob.glob(os.path.join(folder,'*')):
        print(impath)
        gen_image_np,repasted_np,extras, inp_image_np = run(impath)

        save_results(impath,results_root,
                     gen_image_np,repasted_np,inp_image_np,extras)
        if os.environ.get('DEBUG_HARMONIZATION',False) == '1':
            save_results(impath,results_root,
                     gen_image_np,debug.repasted_pre,inp_image_np,extras,save_prefix='generated_no_harmonization')
            
        # break
        if os.environ.get('DBG_SHIFTED_DOWN',False) == '1':
            import sys
            sys.exit()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync',action="store_true",default=False)
    parser.add_argument('--folder',type=str,default=None)
    parser.add_argument('--results_root',type=str,default='results')
    args = parser.parse_args()
    if False and 'run on single image':
        impath = 'flask3.jpg'
        gen_image_np,repasted_np,extras = run(impath)
    if True and 'folder':
        folder = args.folder
        results_folder = args.results_root
        if True:
            run_on_folder(folder,results_root=results_folder)
        #--------------------------------------------
        if args.sync:
            os.system(f'rclone sync -Pv {results_folder} aniket-gdrive:stable-diffusion-experiments/{results_folder}')
