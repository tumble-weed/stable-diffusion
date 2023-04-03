import torch
import numpy as np
import os
import sys
#================================================================
mydir = os.path.dirname(os.path.abspath(__file__))
lamadir = os.path.join(mydir,'lama')
sys.path.append(lamadir)
#================================================================
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
#================================================================
from omegaconf import OmegaConf
# import torch
import yaml
import sys
# import os
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
"""
def holefill():
    pass
def gpnn_holefill():
    if True and 'gpnn':
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
        import os
        # import ipdb; ipdb.set_trace()
        gpnn_inpainting = gpnn(config)
        holefilled,holefilling_results = gpnn_inpainting.run(to_save=False)
"""
def lama_holefill(gen_image_np, worst_mask_np):
    mydir = os.path.dirname(os.path.abspath(__file__))
    predict_config = {'indir': os.path.join(mydir,'lama/LaMa_test_images'), 'outdir': os.path.join(mydir,'lama/output'), 'model': {'path': os.path.join(mydir,'lama/big-lama'), 'checkpoint': 'best.ckpt'}, 'dataset': {'kind': 'default', 'img_suffix': '.png', 'pad_out_to_modulo': 8}, 'device': 'cuda', 'out_key': 'inpainted', 'refine': True, 'refiner': {'gpu_ids': '0', 'modulo': 8, 'n_iters': 15, 'lr': 0.002, 'min_side': 32, 'max_scales': 3, 'px_budget': 1800000}}


    # predict_config = OmegaConf(predict_config)

    # device = torch.device(predict_config['device'])
    # device = 'cuda:0'
    device = 'cuda'
    
    train_config_path = os.path.join(predict_config["model"]['path'], 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    # out_ext = predict_config.get('out_ext', '.png')

    checkpoint_path = os.path.join(predict_config["model"]["path"], 
                                    'models', 
                                    predict_config["model"]["checkpoint"])
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    # if not predict_config.get('refine', False):
    model.to(device)
        

    #==================================================
    #from torch.utils.data._utils.collate import default_collate
    #dataset = make_default_val_dataset(predict_config["indir"], **predict_config["dataset"])


    #mask_fname = dataset.mask_filenames[0]
    #batch = default_collate([dataset[0]])
    #==================================================
    
    gen_image_th = torch.tensor(gen_image_np).float().to(device).permute(2,0,1).unsqueeze(0)
    # import ipdb; ipdb.set_trace()
    # worst_mask_th = torch.tensor(worst_mask_np).float().to(device).permute(2,0,1).unsqueeze(0)
    worst_mask_th = torch.tensor(worst_mask_np).float().to(device).unsqueeze(0).unsqueeze(0)
    worst_mask_th = worst_mask_th[:,:1]

    # batch_for_lama = {'image': gen_image_th, 'mask': mask_th,'unpad_to_size':[[gen_image_th.shape[-2]],[gen_image_th.shape[-1]]]}
    # cur_res = refine_predict(batch_for_lama, model, **predict_config["refiner"])
    #=================================================
    # cur_res = refine_predict(batch, model, **predict_config.refiner)
    # cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
    # import ipdb; ipdb.set_trace()
    assert worst_mask_th.unique().shape[0] == 2
    batch_for_lama = {'image': gen_image_th, 'mask': worst_mask_th}
    if False:    
        # using unrefined
        # cur_res = refine_predict(batch_for_lama, model, **predict_config["refiner"])
        batch_for_lama_out = model(batch_for_lama)                    
        holefilled = batch_for_lama_out['inpainted']        
    else:
        # using refinement
        from saicinpainting.evaluation.data import pad_img_to_modulo
        pad_out_to_modulo = 8
        # if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
        batch_for_lama['unpad_to_size'] = torch.tensor(batch_for_lama['image'].shape[2:]).unsqueeze(1)
        image_np = tensor_to_numpy(batch_for_lama['image'])
        mask_np = tensor_to_numpy(batch_for_lama['mask'])
        batch_for_lama['image'] = np.stack([pad_img_to_modulo(image_np[i], pad_out_to_modulo) for i in range(image_np.shape[0])],axis=0)
        batch_for_lama['mask'] = np.stack([pad_img_to_modulo(mask_np[i], pad_out_to_modulo) for i in range(mask_np.shape[0])],axis=0)        
        batch_for_lama['image'] = torch.tensor(batch_for_lama['image']).float().to(device)
        batch_for_lama['mask'] = torch.tensor(batch_for_lama['mask']).float().to(device)
        holefilled = refine_predict(batch_for_lama, model, **predict_config['refiner'])
        # import ipdb; ipdb.set_trace()

    return holefilled
