import tqdm
import numpy as np
import torch
from skimage.transform.pyramids import pyramid_gaussian
from skimage.transform import rescale, resize
from torch.nn.functional import fold, unfold
from .utils import *
from .pca import PCA
import faiss.contrib.torch_utils
import kornia as K
# from kornia.geometry.transform.pyramid import 
from pyrdown import pyrdown
from eutils.NN_modules import PytorchNNLowMemory
from aggregation import combine_patches
import debug
from collections import defaultdict
from model.nearest_neighbors import create_index,get_nearest_neighbors_of_subset
import dutils
from collections import OrderedDict
from termcolor import colored

tensor_to_numpy = lambda t:t.detach().cpu().numpy()
TODO = None
def cat_lr_flip(t):
    print('appending lr flips')
    assert t.shape[-1] == 7
    t = torch.cat([t,t.flip((-1,))],dim=0)
    # import pdb;pdb.set_trace()
    return t
def resize_bhwc(bhwc,size):
    bchw = bhwc.permute(0,3,1,2)
    bchw_r = torch.nn.functional.interpolate(bchw,size=size,mode='bilinear')
    bhwc_r = bchw_r.permute((0,2,3,1))
    return bhwc_r
class gpnn:
    def __init__(self, config):
        print('HIIIIIIIIII!!!!!!!!')
        # import ipdb;ipdb.set_trace()
        self.IMPLEMENTATION = config['implementation']
        self.INIT_FROM = config['init_from']
        self.KEYS_TYPE = config['keys_type']
        # pca settings
        self.N_PCA_COMPONENTS = config['n_pca_components']
        self.USE_PCA = config['use_pca']
        # general settings
        self.T = config['iters']
        self.PATCH_SIZE = (config['patch_size'], config['patch_size'])
        self.COARSE_DIM = (config['coarse_dim'], config['coarse_dim'])
        self.TASK = config['task']
        assert config['task'] == 'inpainting'
        if True:
            '''
            mask = img_read(config['mask'])
            mask_patch_ratio = np.max(np.sum(mask, axis=0), axis=0) // self.PATCH_SIZE
            coarse_dim = mask.shape[0] / mask_patch_ratio
            self.COARSE_DIM = (coarse_dim, coarse_dim)
            '''
            print('from the documentation: **the mask is in the following format - ones in the pixels where the inpainted area is, and zeros elsewhere.** ')
            mask = config['mask']
            self.mask = mask
            # mask_patch_ratio = np.max(np.sum(mask, axis=0), axis=0) // self.PATCH_SIZE[0]
            print('TODO: new mask_patch_ratio to prevent ~1 ratios')
            mask_patch_ratio = np.sqrt(mask.sum())//(self.PATCH_SIZE[0])
            if mask_patch_ratio <= 1:
                import pdb;pdb.set_trace()
            coarse_dim = mask.shape[0] / mask_patch_ratio
            self.COARSE_DIM = (coarse_dim, coarse_dim)
            
        self.STRIDE = (config['stride'], config['stride'])
        self.R = config['pyramid_ratio']
        self.ALPHA = config['alpha']
        self.PATCH_AGGREGATION = config['patch_aggregation']
        self.INDEX_TYPE = config['index_type']
        self.USE_XY = config['use_xy']
        # cuda init
        global device
        print('see device setting');
        # import ipdb;ipdb.set_trace()
        if config.get('device',None) is None:
            if config['no_cuda']:
                device = torch.device('cpu')
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if torch.cuda.is_available():
                    print('cuda initialized!')
        else:
            device = config['device']

        # faiss init
        self.is_faiss = config['faiss']
        if self.is_faiss:
            global faiss, res
            import faiss
            res = faiss.StandardGpuResources()
            print('faiss initialized!')

        # input image
        import os
        img_path = None
        if True:
            if isinstance(config['input_img'],str):
                img_path = config['input_img']
            else:
                self.input_img = config['input_img']
        
        if isinstance(config['input_img'],str):
            self.input_img = img_read(img_path)
        # self.input_img = self.input_img[:100,:100]
        # import pdb;pdb.set_trace()
        if 'float' not in str(self.input_img.dtype):
            self.input_img = self.input_img/255.
        if isinstance(self.input_img,torch.Tensor):

            self.input_img_tensor = self.input_img
            if False:
                assert self.input_img_tensor.shape[-1] in [1,3,4],'shape of input_img_tensor should be BCHW'
                self.input_img = tensor_to_numpy(self.input_img_tensor)[0]
                self.input_img_tensor = self.input_img_tensor.permute(0,3,1,2)
                
            assert self.input_img_tensor.shape[1] in [1,3,4],'shape of input_img_tensor should be BCHW'
            self.input_img = tensor_to_numpy(self.input_img_tensor.permute(0,2,3,1))[0]
            self.input_img_tensor = self.input_img_tensor        
        else:
            if config['out_size'] != 0:
                if self.input_img.shape[0] > config['out_size']:
                    self.input_img = rescale(self.input_img, config['out_size'] / self.input_img.shape[0], multichannel=True)
            self.input_img_tensor = torch.tensor(self.input_img).float().to(device).permute(2,0,1).unsqueeze(0)
        if config['task'] == 'inpainting':
            self.mask_tensor = torch.tensor(self.mask).float().to(device).unsqueeze(0).unsqueeze(1)
            assert self.mask_tensor.ndim == 4
        # pyramids
        print(self.input_img.shape)
        pyramid_depth = np.log(min(self.input_img.shape[:2]) / min(self.COARSE_DIM)) / np.log(self.R)
        self.add_base_level = True if np.ceil(pyramid_depth) > pyramid_depth else False
        pyramid_depth = int(np.ceil(pyramid_depth))
        print('NOTE:making pyramid depth at least 1')
        pyramid_depth = max(pyramid_depth,1)
        '''
        self.x_pyramid = [pyrdown(self.input_img_tensor, 
                                  mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R ** i).permute(0,2,3,1) for i in range(1,pyramid_depth)]
        mask_tensor = torch.tensor( (1 - self.mask),device=device)[None,None,...]
        self.x_pyramid.insert(0,(self.input_img_tensor*mask_tensor).permute(0,2,3,1))
        import ipdb;ipdb.set_trace()
        '''
        #============================================
        if config['task'] == 'inpainting':
            from model import mask_pyramid
            import importlib
            
            if False:
                self.mask_pyramid = [pyrdown(self.mask_tensor, 
                                    border_type = 'reflect', 
                                    align_corners = True, 
                                    factor = self.R ** i).permute(0,2,3,1) for i in range(1,pyramid_depth)]
                self.mask_pyramid.insert(0,self.mask_tensor.permute(0,2,3,1))            
                self.key_pyramid = []
                self.query_pyramid = []
                for i in range(len(self.mask_pyramid)):
                    # import pdb;pdb.set_trace()
                    mask = self.mask_pyramid[i]
                    mask = extract_patches(mask, self.PATCH_SIZE, self.STRIDE)
                    # if self.input_img_tensor.shape[2] > 1:
                    #     mask = torch.all(mask, dim=3)
                    
                    if True:
                        # # check if all values are ON in the patch
                        # mask = torch.all(mask, dim=-1)
                        # mask = torch.all(mask, dim=-1)
                        
                        # check if any values are ON in the patch
                        # mask = torch.any(mask, dim=-1)
                        # mask = torch.any(mask, dim=-1)           
                        #========================================
                        if 'dual pyramids':
                            if False and 'key:any':
                                mask_key = torch.any(mask, dim=-1)
                                mask_key = torch.any(mask_key, dim=-1)           
                                # mask_key = ~mask_key
                                
                            else:
                                mask_key = (mask.mean(dim=(-1,-2)) > 0.5)
                            self.key_pyramid.append(mask_key)
                            #========================================
                            if False:
                                mask_query = torch.all(mask, dim=-1)
                                mask_query = torch.all(mask_query, dim=-1)           
                            elif False:
                                mask_query = torch.any(mask, dim=-1)
                                mask_query = torch.any(mask_query, dim=-1)           
                            else:
                                
                                mask_query = torch.any((mask > 0.75), dim=-1)
                                mask_query = torch.any(mask_query, dim=-1)                                       
                                
                            self.query_pyramid.append(mask_query)                             
                    else:
                        # import ipdb;ipdb.set_trace()
                        mask = (mask.mean(dim=(-1,-2)) > 0.5)
                        if False:
                            print('TODO:hard coding mask to half true')
                            mask[:mask.shape[0]//2] = True
                            mask[mask.shape[0]//2:] = False
                    if False:
                        assert mask.ndim == 2
                        assert mask.shape[-1] == 1
                    else:
                        assert mask_key.ndim == 2
                        assert mask_key.shape[-1] == 1                    
                    self.mask_pyramid[i] = mask
            '''
            self.mask_pyramid,self.query_pyramid,self.key_pyramid = mask_pyramid.create_pyramid(self.mask_tensor,self.R,self.PATCH_SIZE,self.STRIDE,pyramid_depth)
            '''

            def create_pyramid2(mask_tensor,R,PATCH_SIZE,STRIDE):
                mask_pyramid = []
                i = 1
                query_pyramid = []
                key_pyramid = []
                query_ratios = OrderedDict()
                key_ratios = OrderedDict()
                #=================================================
                while True:
                    complement_smaller = pyrdown(
                        (1 - mask_tensor), 
                        mask = (1-mask_tensor),
                        border_type = 'reflect', 
                        align_corners = True, 
                        factor = R ** i).permute(0,2,3,1)
                    smaller = 1 - complement_smaller
                    smaller = (smaller != 0.).float();print(colored('setting the pyramid level to binary eary','yellow'))
                    if min(smaller.shape[1:3]) < max(PATCH_SIZE):
                        break
                    mask_pyramid.append(smaller)
                    if smaller.sum() == 0:
                        break
                    #====================================================
                    # # standard pyramid:
                    # patch_size = PATCH_SIZE
                    # min_dim = min(smaller.shape[1:3])
                    # if max(patch_size) > min_dim:
                    #     patch_size = (min_dim,min_dim)
                    # patches = extract_patches(smaller, patch_size, STRIDE)
                    # mask_query = torch.all(patches, dim=-1)
                    # mask_query = torch.all(mask_query, dim=-1)           
                    # mask_key = mask_query
                    # key_pyramid.append(mask_key)
                    # query_pyramid.append(mask_query)
                    #====================================================
                    i += 1
                # import ipdb;ipdb.set_trace()
                #=================================================
                mask_pyramid.insert(0,mask_tensor.permute(0,2,3,1))
                
                print(colored('TODO:cutting off mask_pyramid at 8','red'))
                mask_pyramid = mask_pyramid[:8]
                n_levels = len(mask_pyramid)
                #import ipdb;ipdb.set_trace()
                mask_pyramid2 = []
                for mask in mask_pyramid:
                    pass
                for mask in ((mask_pyramid)):
                    #====================================================
                    # standard pyramid:
                    # mask = (mask == 1.).float()
                    if False:
                        mask = (mask != 0.).float()
                    patch_size = PATCH_SIZE
                    min_dim = min(mask.shape[1:3])
                    if max(patch_size) > min_dim:
                        patch_size = (min_dim,min_dim)
                    patches = extract_patches(mask, patch_size, STRIDE)
                    mask_query = torch.any(patches, dim=-1)
                    mask_query = torch.any(mask_query, dim=-1)           
                    #=========================================
                    # mask_key = mask_query
                    mask_key = torch.any(patches, dim=-1)
                    mask_key = torch.any(mask_key, dim=-1)                      
                    #=========================================         
                    key_pyramid.append(mask_key)
                    query_pyramid.append(mask_query)
                    #====================================================  
                    available_area = np.prod(mask.shape[1:3])
                    query_ratio  = (mask_query).sum()/available_area
                    key_ratio = (mask_key).sum()/available_area
                    query_ratios[available_area] = query_ratio.item()
                    key_ratios[available_area] = key_ratio.item()
                    print('see if ratios are fine')
                    # import ipdb;ipdb.set_trace()
                    #====================================================  
                print('TODO: will key query and mask pyramids be same?')
                print('TODO: mask_pyramid will have float, so cant be same')
                return mask_pyramid,query_pyramid,key_pyramid
            self.mask_pyramid,self.query_pyramid,self.key_pyramid = create_pyramid2(self.mask_tensor,self.R,self.PATCH_SIZE,self.STRIDE)
            # self.query_ratios = OrderedDict()
            # self.key_ratios = OrderedDict()
            # pyramid_depth = max(pyramid_depth,1)
            pyramid_depth = len(self.mask_pyramid)
            # self.x_pyramid0 = list(
            #     tuple(pyramid_gaussian(self.input_img, pyramid_depth, downscale=self.R, multichannel=True)))
            
            self.x_pyramid = [pyrdown(self.input_img_tensor, 
                                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                    border_type = 'reflect', 
                                    align_corners = True, 
                                    factor = self.R ** i).permute(0,2,3,1) for i in range(1,pyramid_depth)]
            mask_tensor = torch.tensor( (1 - self.mask),device=device)[None,None,...]
            self.x_pyramid.insert(0,(self.input_img_tensor*mask_tensor).permute(0,2,3,1))
            # import ipdb;ipdb.set_trace()                
        if os.environ.get('BREAK_AFTER_PYRDOWN',False) == '1':
            import ipdb;ipdb.set_trace()
        #============================================
        #============================================================
        # import ipdb;ipdb.set_trace()
        # if os.environ.get('USE_XY',False) == '1':
        if self.USE_XY:
            for i_x,x in enumerate(self.x_pyramid):
                Y,X = torch.meshgrid(torch.linspace(0,1,x.shape[1],device=device),
                               torch.linspace(0,1,x.shape[2],device=device),
                               indexing = 'ij')
                spatial_mag = 0.3
                x = torch.cat([x,spatial_mag*Y[None,:,:,None],spatial_mag*X[None,:,:,None]],dim=-1)
                self.x_pyramid[i_x] = x
                # import ipdb;ipdb.set_trace()
        #============================================================
        # import pdb;pdb.set_trace()
        self.add_base_level = False
        print('setting self.add_base_level to False')
        if self.add_base_level is True:
            # self.x_pyramid[-1] = resize(self.x_pyramid[-2], self.COARSE_DIM)
            self.x_pyramid[-1] = resize_bhwc(self.x_pyramid[-2], self.COARSE_DIM)
            print('TODO:is this correct')
            self.mask_pyramid[-1] = resize_bhwc(self.mask_pyramid[-2], self.COARSE_DIM)
        # import ipdb;ipdb.set_trace()
        self.y_pyramid = [0] * (pyramid_depth + 1)
        #============================================
        if config['task'] == 'inpainting':
            # import ipdb;ipdb.set_trace()
            if False and 'cut-off zero query':
                mask_pyramid.cutoff_pyramid_at_zero(self.query_pyramid,self.x_pyramid,self.key_pyramid)   
            if True and 'cut-off zero key':
                mask_pyramid.cutoff_pyramid_at_zero([~l for l in self.key_pyramid],self.mask_pyramid,self.key_pyramid,self.x_pyramid,self.query_pyramid)        
            # import ipdb;ipdb.set_trace()
            if False:
                max_i = len(self.mask_pyramid) - 1
                for i in reversed(range(len(self.mask_pyramid))):
                    if True in self.mask_pyramid[i].unique():
                        print('some queries still to be addressed')
                        break
                    print('no queries at this level')
                    max_i -= 1
                # max_i = 1;print(f'setting max_i to {max_i}')
                self.x_pyramid = self.x_pyramid[:max_i]
                self.mask_pyramid = self.mask_pyramid[:max_i]
                self.y_pyramid = self.y_pyramid[:max_i]
        #============================================
        # out_file
        # filename = os.path.splitext(os.path.basename(img_path))[0]
        filename = 'out_img'
        self.out_file = os.path.join(config['out_dir'], "%s_%s.png" % (filename, config['task']))
        self.batch_size = config['batch_size']
        # coarse settings
        # import ipdb;ipdb.set_trace()
        '''
        if config['task'] == 'random_sample':
            if isinstance(self.x_pyramid[-1],np.ndarray):
                noise = np.random.normal(0, config['sigma'], (self.batch_size,)+ self.COARSE_DIM)[..., np.newaxis]
                self.coarse_img = noise

                if self.INIT_FROM == 'target':
                    self.coarse_img = self.coarse_img + noise
            else:
                assert len(self.x_pyramid[-1].shape) == 4
                noise = config['sigma']*torch.randn((self.batch_size,)+ self.x_pyramid[-1].shape[1:3])[..., np.newaxis]
                noise = noise.to(device)
                self.coarse_img = noise
                if self.INIT_FROM == 'target':
                    # assert False
                    self.coarse_img = self.x_pyramid[-1] + noise
                elif self.INIT_FROM == 'zeros':
                    self.coarse_img = torch.zeros_like(self.x_pyramid[-1]) + noise
        elif config['task'] == 'structural_analogies':
            assert False,'not implemented'
            self.coarse_img = img_read(config['img_b'])
            self.coarse_img = resize(self.coarse_img, self.x_pyramid[-1].shape)
        '''
        if config['task'] == 'inpainting':
            self.coarse_img = self.x_pyramid[-1]
        assert len(self.coarse_img.shape) == 4
        self.running_keys = None
        self.running_values = None
        self.resolution =None
        self.n_keys = {}
        self.trends = defaultdict(list)
        # self.keys_to_keep = None
        print('init done')
        if False and 'limit_pyramids':
            till = 3
            
            print(colored(f'limiting pyramids to {till}','red'))
            self.x_pyramid = self.x_pyramid[:till]
            self.mask_pyramid = self.mask_pyramid[:till]
            self.key_pyramid = self.key_pyramid[:till]
            self.query_pyramid = self.query_pyramid[:till]

    def run(self, to_save=True):
        if self.IMPLEMENTATION == 'efficient-gpnn':

            from model.efficient_gpnn import generate
            # nn_module
            nn_module = PytorchNNLowMemory(alpha=self.ALPHA, use_gpu=True)
            assert self.input_img_tensor.max() <= 1
            scales = [int(min(self.input_img_tensor.shape[-2:])*((3/4)**factor)) for factor in range(11)]
            scales = list(reversed(scales))
            scales = [s for s in scales if s>= 14]
            # scales = [128]
            # import pdb;pdb.set_trace()
            augmentations,I =  generate(
                # self.x_pyramid,
                # self.input_img,
                self.input_img_tensor,
                # 2*self.input_img_tensor - 1,
                nn_module,
                patch_size=7,
                stride=1,
                init_from =  self.INIT_FROM,#'target',#
                pyramid_scales=scales,
                #  pyramid_scales=(14,25,45,60,81),
                aspect_ratio=(1, 1),
                additive_noise_sigma=1*0.75,
                num_iters = 10,
                initial_level_num_iters = 1,
                keys_blur_factor=1,
                device=torch.device("cuda"),
                debug_dir=None)
            print('W not implemented for efficient-gpnn')
            return augmentations,I,None
        # for i in tqdm.tqdm_notebook(reversed(range(len(self.x_pyramid)))):
        resolution_results = {}
        for i in tqdm.tqdm(reversed(range(len(self.x_pyramid)))):
            # import ipdb;ipdb.set_trace()
            # if i < 2:
            #         self.STRIDE = (2,2)            
            query_mask = None
            del resolution_results
            # if i == 9:
            #     I = None
            #     print('breaking early')
            #     break
            if  False and i ==1:
                torch.cuda.empty_cache()
            if i == len(self.x_pyramid) - 1:
                # queries = self.coarse_img.flip(-1);print('flipping initial image')
                queries = self.coarse_img
                keys = self.x_pyramid[i]
                # import ipdb;ipdb.set_trace()
            else:
                # queries = np.array([resize(yp, self.x_pyramid[i].shape) for yp in self.y_pyramid[i + 1]])
                # queries = torch.stack([torch.nn.functional.interpolate(yp,self.x_pyramid[i].shape[:2]) for yp in self.y_pyramid[i + 1]],dim=0)
                
                if False and 'adding per level noise':
                    noise = 0.1*0.5*config['sigma']*torch.randn((self.batch_size,)+ self.x_pyramid[i].shape[1:3])[..., np.newaxis]
                    noise = noise.to(device)
                    
                    queries = resize_bhwc(self.y_pyramid[i + 1],self.x_pyramid[i].shape[1:3]) + noise
                else:
                    queries = resize_bhwc(self.y_pyramid[i + 1],self.x_pyramid[i].shape[1:3])
                    if False and self.TASK == 'inpainting':
                        print('TODO:copy from pnn_faiss')
                        import ipdb;ipdb.set_trace()
                        queries_patches = extract_patches(queries, self.PATCH_SIZE, self.STRIDE)
                        gt_patches = extract_patches(resize_bhwc(self.x_pyramid[i + 1],self.x_pyramid[i].shape[1:3]),self.PATCH_SIZE, self.STRIDE)
                        assert queries_patches.ndim == 4
                        assert queries_patches.shape[0] == self.query_pyramid[i].shape[0]
                        query_holes_at = self.query_pyramid[i][:,None,None].float()
                        # anchored_query_patches = queries_patches * query_holes_at + \
                        # (1 - query_holes_at)* gt_patches
                        anchored_query_patches = (1 - query_holes_at)* gt_patches
                        print('TODO:copy from pnn_faiss')
                        queries = combine_patches(anchored_query_patches, self.PATCH_SIZE, self.STRIDE, queries.shape[1:],
                        as_np=False,
                        patch_aggregation=self.PATCH_AGGREGATION,
                        distances=None,I=None)['combined']
                        assert queries.ndim == 3
                        queries = queries.unsqueeze(0)
                # import pdb;pdb.set_trace()
                # keys = resize(self.x_pyramid[i + 1], self.x_pyramid[i].shape)
                keys = resize_bhwc(self.x_pyramid[i + 1],self.x_pyramid[i].shape[1:3])
            new_keys = True
            # for j in tqdm.tqdm_notebook(range(self.T)):
            D,I = None,None
            if i == len(self.x_pyramid) - 1:
                if self.query_pyramid[i].any():
                    # run road for the smallest level
                    print('TODO: since we are running road at the smallest level, put pyramid_level back in create_mask_pyramid2 (no need to keep going down the scale space)')
                    print('TODO: what is the implication of the still having queries to solve at the bottom')
                    print('TODO:is there any chance that keys can be None?')
                    print('TODO:why add noise in the noisy linear imputer')
                    print('TODO:simpler way of setting the initial values than NoisyLinearImputer (because it will take up time)?')
                    from model.road_for_gpnn import NoisyLinearImputer
                    imputer = NoisyLinearImputer(noise=0.)
                    im_for_road = self.x_pyramid[i].cpu().permute(0,3,1,2)[0]
                    mask_for_road = 1 - (self.mask_pyramid[i].permute(0,3,1,2)[0,0]!=0.).float().cpu()
                    imputed = imputer(im_for_road,mask_for_road).unsqueeze(0).permute(0,2,3,1).to(device)
                    if os.environ.get('DEBUG_REPRODUCIBILITY',False) == '1':
                        dutils.road_imputed = imputed
                    assert not imputed.isnan().any()
                    self.y_pyramid[i] = imputed
                    # queries = self.y_pyramid[i] 
                    keys = self.x_pyramid[i]      
                    resolution_results = None     
                    import ipdb;ipdb.set_trace() 
                else:
                    self.y_pyramid[i] = self.x_pyramid[i]
                    keys = self.x_pyramid[i]      
                    resolution_results = None     
                # import ipdb;ipdb.set_trace()
            else:
                for j in tqdm.tqdm(range(self.T)):                
                    if self.is_faiss:
                        # self.y_pyramid[i],I,W 
                        
                        resolution_results = self.PNN_faiss(self.x_pyramid[i], keys, queries, self.PATCH_SIZE, self.STRIDE,
                                                        self.ALPHA, 
                                                        mask= (self.mask_pyramid[i] if self.TASK == 'inpainting' else None), 
                                                        mask_key= (self.key_pyramid[i] if self.TASK == 'inpainting' else None), 
                                                        mask_query= (self.query_pyramid[i] if self.TASK == 'inpainting' else None), 
                                                        new_keys=new_keys,
                                                        other_x = self.x_pyramid[i],query_mask=query_mask,
                                                        Dprev=D,Iprev=I,
                                                        index_type = self.INDEX_TYPE,
                                                        #    keys_to_keep=self.keys_to_keep
                                                        )
                        
                        if j ==0:
                            if False and 'find small distance':
                                # import pdb;pdb.set_trace()
                                # import IPython;IPython.embed()
                                n_random = 31
                                TODO = None
                                keys_patches = extract_patches(keys, self.PATCH_SIZE, self.STRIDE)
                                selected_patches = keys_patches[np.random.choice(keys_patches.shape[0],n_random)]
                                # selected_patches = selected_patches.flatten(start_dim=0,end_dim=1)
                                
                                selected_patches = self.get_feats(selected_patches,init=False)
                                
                                # as we need take square distance for query_mask
                                distances = torch.pdist(selected_patches)**2 
                                distances = distances[distances!=0]
                                # so log doesnt give inf
                                # distances[torch.arange(n_random,device=device),torch.arange(n_random,device=device)] = torch.ones_like(distances.diag()) 
                                log_distances = torch.log(distances)
                                # assert distances.shape == N*N-1
                                mean_log_distance = log_distances.mean()
                                mean_square_log_distance = (log_distances**2).mean()
                                var_log_distance = mean_square_log_distance - mean_log_distance**2
                                assert var_log_distance > 0
                                std_log_distance = var_log_distance.sqrt()
                                _3_sigma_distance = torch.exp(mean_log_distance - 6*std_log_distance)
                                assert _3_sigma_distance > 0
                                
                            
                        change_in_I = 1
                        if False and 'trends':
                            if j>0:
                                
                                change_per_pixel = (resolution_results['combined'] - self.y_pyramid[i]).pow(2)
                                
                                # import pdb;pdb.set_trace()
                                channels = change_per_pixel.shape[-1]
                                change_per_patch = torch.nn.functional.conv2d(
                                    change_per_pixel.permute(0,3,1,2), 
                                    torch.ones(1,channels,self.PATCH_SIZE[0],self.PATCH_SIZE[1],device=device).float()
                                )
                                
                                change_per_patch = change_per_patch.view(change_per_patch.shape[0],-1)
                                query_mask = (change_per_patch > _3_sigma_distance).flatten()
                                # if not query_mask.all():
                                #     import pdb;pdb.set_trace()
                                # query_mask =None;print('setting query mask to None')
                                # assert np.prod(change_per_patch.shape[-2]) == 
                                
                                # change_in_image_ = change_in_image.item()
                                # import pdb;pdb.set_trace()
                                change_in_image = change_per_pixel.mean(dim=0).sum()
                                self.trends[('change_in_image',i)].append(change_in_image.item())
                            D = resolution_results['D']
                            D_ = tensor_to_numpy(D.squeeze())
                            total_distance = D_.mean()
                            
                            self.trends[('D',i)].append(total_distance)
                            
                            I = resolution_results['I']
                            I_ = I.squeeze()
                            if isinstance(I,torch.Tensor):
                                I_ = tensor_to_numpy(I_)
                            # import pdb;pdb.set_trace()
                            I_per_image = I_.reshape((self.batch_size,-1))
                            diversity_per_image = np.array([np.unique(Ii).shape[0] for Ii in I_per_image])*1./I_per_image.shape[-1]
                            self.trends[('diversity',i)].append(diversity_per_image.mean())
                            if j > 0:
                                # import pdb;pdb.set_trace()
                                change_in_I = np.float32(Iprev != I_).mean()
                                self.trends[('change_in_I',i)].append(change_in_I)
                            Iprev = I_
                            # import pdb;pdb.set_trace()

                        self.y_pyramid[i] = resolution_results['combined']
                        # if i != 0:
                        #     del I
                    else:
                        assert False, 'not returning patch_aggregation style results'
                        self.y_pyramid[i],I = self.PNN(self.x_pyramid[i], keys, queries, self.PATCH_SIZE, self.STRIDE,
                                                    self.ALPHA)
                    # if (query_mask is not None) and not query_mask.any():
                    #     break
                    print('TODO:diagnose incresing memory')
                    # import pdb;pdb.set_trace()
                    queries = self.y_pyramid[i] 
                    keys = self.x_pyramid[i]
                    if j > 1:
                        new_keys = False
                    if change_in_I < 0.01:
                        break
                # import pdb;pdb.set_trace()
            last = i
        # import ipdb;ipdb.set_trace()

                
        #==========================================================
        if False:
            assert False,'untested for running-values'
            # for the masks
            mask = torch.zeros(1,*self.y_pyramid[0].shape[1:3],1).to(device)
            mask[:,:,torch.arange(mask.shape[-2]).to(device),:] = torch.linspace(0,1,mask.shape[-2]).to(device)[None,None,:,None]
            mask_values = extract_patches(mask, self.PATCH_SIZE, self.STRIDE)
            # import pdb;pdb.set_trace()
            # mask_keys_flat = mask_keys.reshape((mask_keys.shape[0], -1)).contiguous()
            mask_values = mask_values[I.T]
            mask_values = mask_values.squeeze(0)
            mask_values = mask_values.reshape(self.y_pyramid[0].shape[0],
                                    mask_values.shape[0]//self.y_pyramid[0].shape[0],*mask_values.shape[1:])
            assert mask_values.ndim == 5,'1,npatches,nchan,7,7'
            masks = torch.stack([combine_patches(v, self.PATCH_SIZE, self.STRIDE, mask.shape[1:3]+(3,),patch_aggregation=self.PATCH_AGGREGATION,as_np=False)['combined'] for v in mask_values],dim=0)
        #==========================================================
        '''
        if to_save and False:
            # if self.batch_size > 1:
            for ii,yi in enumerate(self.y_pyramid[0]):
                # yi = (yi - yi.min())/(yi.max()-yi.min())
                assert yi.shape[-1] == 3
                img_save(tensor_to_numpy(yi), self.out_file[:-len('.png')] + str(ii) + '.png' )
                # mi = masks[i]
                # img_save(tensor_to_numpy(mi), 'mask'+self.out_file[:-len('.png')] + str(ii) + '.png' )
        # import pdb;pdb.set_trace()
        '''

        #================================================================  
        self.query_pyramid1 = []
        self.key_pyramid1 = []
        for iq,_ in enumerate(self.query_pyramid):
            m = self.mask_pyramid[iq]
            q0 = torch.zeros_like(m)
            k0 = torch.zeros_like(m)
            qpatches = extract_patches(q0, self.PATCH_SIZE, self.STRIDE)
            kpatches = extract_patches(k0,self.PATCH_SIZE,self.STRIDE)
            q = self.query_pyramid[iq]
            k = self.key_pyramid[iq]
            print(q.shape)
            qpatches[q.squeeze(1)] = 1
            kpatches[k.squeeze(1)] = 1
            q = combine_patches(qpatches, self.PATCH_SIZE, self.STRIDE, m.shape[1:],
                            as_np=False,
                            patch_aggregation='uniform',
                            distances=None,I=None)['combined']
            k = combine_patches(kpatches, self.PATCH_SIZE, self.STRIDE, m.shape[1:],
                            as_np=False,
                            patch_aggregation='uniform',
                            distances=None,I=None)['combined']
            self.query_pyramid1.append(q[...,0])
            self.key_pyramid1.append(k[...,0])        

        #================================================================
        if os.environ.get('DBG_BREAK_NEAR_END',False) == '1':
            import ipdb;ipdb.set_trace()        
        import ipdb;ipdb.set_trace()        
        return self.y_pyramid[last].permute(0,3,1,2),resolution_results
        # return self.y_pyramid[0],I
    
    '''
    def PNN(self, x, x_scaled, y_scaled, patch_size, stride, alpha, mask=None):
        # queries = extract_patches(y_scaled, patch_size, stride)
        queries = torch.stack([extract_patches(ys, patch_size, stride) for ys in y_scaled],dim=0)
        keys = extract_patches(x_scaled, patch_size, stride)
        values = extract_patches(x, patch_size, stride)
        
        if mask is None:
            assert queries.ndim == 5
            dist = torch.cdist(queries.view(queries.shape[0]*queries.shape[1], -1), keys.view(len(keys), -1))
        else:
            assert False,'not implemented'
            m_queries, m_keys = queries[mask], keys[~mask]
            dist = torch.cdist(m_queries.view(len(m_queries), -1), m_keys.view(len(m_keys), -1))
        norm_dist = (dist / (torch.min(dist, dim=0)[0] + alpha))  # compute_normalized_scores
        NNs = torch.argmin(norm_dist, dim=1)  # find_NNs
        if mask is None:
            values = values[NNs]
        else:
            values[mask] = values[~mask][NNs]
            # O = values[NNs]  # replace_NNs(values, NNs)
        # assert values.ndim == 5
        # assert values.shape[0] == 1
        # values = values.squeeze()
        values = values.reshape(queries.shape[0],values.shape[0]//queries.shape[0],*values.shape[1:])
        #====================================================================  assert False,'old style combine_patches'
        y = combine_patches(values, patch_size, stride, x_scaled.shape)
        assert len(x_scaled.shape) == 4
        # NNs = torch.atleast_2d(NNs)
        if NNs.ndim == 1:
            NNs = NNs[:,None]
        return y,NNs,w
    '''
    print('find the part about keys, and check if task is inpainting')
    def PNN_faiss(self, x, x_scaled, y_scaled, patch_size, stride, alpha, mask=None, mask_key = None,mask_query = None,
                  new_keys=True,
        other_x=None,extra_return={},query_mask=None,
        Dprev=None,Iprev=None,keys_to_keep=None,index_type='simple'):
        if not mask_query.any():
            import ipdb;ipdb.set_trace()
            print('TODO: will need to remove similar if clause from later on')
            patch_aggregation_results = dict( 
                I = torch.arange(mask_query.shape[0],device=device)[:,None],
                D = torch.zeros(mask_query.shape[0],device=device),
                combined = x,
                patch_aggregation = self.PATCH_AGGREGATION,
            )
            return patch_aggregation_results            
        #============================================================
        min_dim = min(x.shape[1:3])
        if max(patch_size) > min_dim:
            patch_size = (min_dim,min_dim)
        #============================================================
        if self.resolution == None:
            self.resolution = x.shape
        if os.environ.get('PNN_XRANGE_ERROR','1') != '0':
            assert ((1. - x.max()) >= -2e-7) and (x.min() >= -2e-7)
            assert ((1. - x_scaled.max()) >= -2e-7) and (x_scaled.min() >= -2e-7)
            # assert (x_scaled.max() <= 1.) and (x_scaled.min() >= 0.)
        # y_scaled has noise added, so will have >1 and <0 values
        # assert (y_scaled.max() <= 1.) and (y_scaled.min() >= 0.)
        print('using faiss')
        print('this shouldnt be np.array but also work for tensor')
        assert y_scaled[0].shape[-1] in [3,4,5]
        queries = torch.stack([extract_patches(ys, patch_size, stride) for ys in y_scaled],dim=0)
        assert queries.ndim == 5 , 'shape from extract_patches is 1,169,3,7,7'

        from model.hog import gradient_histogram
        
        # import pdb;pdb.set_trace()
        # queries = queries[...,::2,::2]
        print('extracted query',queries.shape)
        keys = extract_patches(x_scaled, patch_size, stride)
        #====================================================
        # query_ratio  = (mask_query).sum()/queries.shape[1]
        # key_ratio = (mask_key).sum()/keys.shape[0]
        # self.query_ratios[queries.shape[1]] = query_ratio.item()
        # self.key_ratios[keys.shape[0]] = key_ratio.item()
        #====================================================
        
        use_lr_flip = False
        if use_lr_flip:
            keys = cat_lr_flip(keys)
        # keys = keys[...,::2,::2]
        print('extracted keys')
        if x.shape not in self.n_keys:
            self.n_keys[x.shape] = keys.shape[0]
        if True:
            values = extract_patches(x, patch_size, stride)
            if use_lr_flip:
                values = cat_lr_flip(values)
            if self.KEYS_TYPE == 'multi-resolution':
                if new_keys:
                    if self.running_values is None:
                        self.running_values = values
                    else:
                        if self.running_values.shape[0] == sum(self.n_keys.values()):
                            self.running_values[-self.n_keys[x.shape]:] = values
                        else:
                            self.running_values = torch.cat([self.running_values,values],dim=0)

        print('extracted values')
        if mask_query is not None:
            assert mask_query.shape[-1] == 1
            queries = queries[:,mask_query[:,0]]
            
            '''
            we are going to keep all keys, and then use only valid ones in
            nearest_neighbors
            '''
            if False:
                keys = keys[~mask[:,0]]
        
        #====================================================================
        assert queries.ndim == 5
        queries_flat_batch = queries.flatten(start_dim=0,end_dim=1).contiguous()
        if queries_flat_batch.shape[0] == 0 :
            import ipdb;ipdb.set_trace()
            patch_aggregation_results = dict(
                I = torch.arange(mask_query.shape[0],device=device)[:,None],
                D = torch.zeros(mask_query.shape[0],device=device),
                combined = x,
                patch_aggregation = self.PATCH_AGGREGATION,
            )
            return patch_aggregation_results
        '''
        if query_mask is not None:
            queries_flat_batch = queries_flat_batch[query_mask]        
        '''
        assert keys.ndim == 4
        if new_keys:
            reproducible_pca = False
            if os.environ.get('DEBUG_REPRODUCIBILITY',False) == '1' or False:
                reproducible_pca = True
            keys_proj = self.get_feats(keys,init=True,reproducible=reproducible_pca)
            #==================================================
            
            n_patches = keys_proj.shape[0]
            print(n_patches)
            keys_for_index = keys_proj
            
            if self.TASK == 'inpainting':
                assert mask_key is not None                
                assert mask_key.ndim == 2
                assert mask_key.shape[-1] == 1
                all_I = torch.arange(mask_key.shape[0],device=device)
                # import pdb;ipdb.set_trace()
                assert mask_key.dtype == torch.bool
                complement_keys_at = mask_key[:,0]
                keys_to_keep = all_I[~complement_keys_at]
                if not (keys[keys_to_keep].abs().sum(dim=(1,2,3)) == 0).sum() == 0:
                    import ipdb;ipdb.set_trace()
            print('Check keys to keep here')
            #==================================================================
            if index_type == 'ivf':
                print('using simple index if keys_for_index < 15000')
                if len(keys_for_index) > 15000:
                    self.index = create_index(keys_for_index,index_type='ivf',index_options={'nlist':200},keys_to_keep=keys_to_keep)           
                else:
                    self.index = create_index(keys_for_index,index_type='simple',keys_to_keep=keys_to_keep)             
            else:
                self.index = create_index(keys_for_index,index_type=index_type,keys_to_keep=keys_to_keep)             
                
        queries_proj = self.get_feats(queries_flat_batch,init=False)

        print('searching')

        nearest_neighbor_results = get_nearest_neighbors_of_subset(queries_proj,query_mask,None,index=self.index,Ddtype='float',Idtype='long',max_batch_size=62496,D=Dprev,I=Iprev)

        D,I = nearest_neighbor_results['D'],nearest_neighbor_results['I']

        if os.environ.get('DEBUG_REPRODUCIBILITY',False) == '1':
            '''
            dI = dutils.__dict__.get('I',None)
            dutils.I = [] if dI is None else dI
            dutils.I.append(I)
            '''
            dq = dutils.__dict__.get('q',None)
            dutils.q = [] if dq is None else dq
            # dutils.q.append(queries_proj)
            dutils.q.append(queries_flat_batch)
            
            if new_keys:
                dk = dutils.__dict__.get('k',None)
                dutils.k = [] if dk is None else dk
                # dutils.k.append(keys_for_index)
                dutils.k.append(keys)

                
        if self.TASK == 'inpainting':
            print('reinsert the partial I back into fullI')
            partial_I,partial_D = I,D
            I = torch.arange(mask_query.shape[0],device=device)[:,None]
            assert D.ndim == 2
            assert D.shape[-1] == 1
            D = torch.zeros(I.shape,device=device).float()
            # import ipdb;ipdb.set_trace()
            queries_at = mask_query[:,0]
            all_I = torch.arange(mask_query.shape[0],device=device)
            I_holes = all_I[queries_at]
            I[I_holes] = partial_I
            D[I_holes] = partial_D                
            print(queries_proj.shape)
            assert mask_query.shape[0] == I.shape[0], 'partial_I should have been reinserted within I'
        print('now no need for mask, we can use standard handling of the values')
        if self.KEYS_TYPE == 'single-resolution':
            assert I.shape[-1] == 1
            # import ipdb;ipdb.set_trace()
            print('hello-2')
            values = torch.index_select(values,0,I.squeeze()).unsqueeze(0)
            
            # assert (values[I.T] == values1).all()
            # import sys;sys.exit()
        
        assert values.ndim == 5
        assert values.shape[0] == 1

        values = values.squeeze()
        # import pdb;pdb.set_trace()
        values = values.reshape(queries.shape[0],values.shape[0]//queries.shape[0],*values.shape[1:])
        distances = D.reshape(queries.shape[0],D.shape[0]//queries.shape[0],1,1,1)
        debug.distances = distances
        if 'check' and False:
            chosen_keys = keys[I.T]
            chosen_keys = chosen_keys.squeeze()
            d0 = (chosen_keys[keys.shape[0]*0:keys.shape[0]*(0+1)] - keys).abs()
            diffs = [(chosen_keys[keys.shape[0]*ii:keys.shape[0]*(ii+1)] - keys).abs().sum() for ii in range(self.batch_size) ]
            flags = [torch.isclose(d,torch.zeros_like(d)) for d in diffs]
            assert all(flags)
        #====================================================================
        assert len(x_scaled.shape) == 4

        if True and '1 image at a time':
            y = []
            w = []
            from collections import defaultdict
            patch_aggregation_results = defaultdict(list)
            
            for ii,(v,d) in enumerate(zip(values,distances)):            
                
                debug.I = I[ii*distances.shape[1]:(ii+1)*distances.shape[1]]
                # print('TODO:debug.I should trigger error in distance-weighted aggregation')

                result = combine_patches(v, patch_size, stride, x_scaled.shape[1:3]+(v.shape[1],),as_np=False,
                patch_aggregation=self.PATCH_AGGREGATION,
                distances=d,I=I)
                # import ipdb;ipdb.set_trace()
                for k in result:
                    if k not in ['patch_aggregation']:
                        # import pdb;pdb.set_trace()
                        patch_aggregation_results[k].append(result[k])
                # yi,wi = result['combined'],result['weights']
                # y.append(yi)
                # w.append(wi)
            # To remove defaultdict
            patch_aggregation_results = {k:v for k,v in patch_aggregation_results.items()}
            patch_aggregation_results['patch_aggregation']=result['patch_aggregation']
            for k in result:
                if k not in ['patch_aggregation']:
                    patch_aggregation_results[k] = torch.stack(patch_aggregation_results[k],dim=0)
                    # .append(torch.stack(patch_aggregation_results[k],dim=0))        
        elif  'multiple images':
            patch_aggregation_results = combine_patches(values, patch_size, stride, x_scaled.shape[1:3]+(values.shape[1],),as_np=False,
                patch_aggregation=self.PATCH_AGGREGATION,
                distances=distances,I=I)
            
        patch_aggregation_results['D'], patch_aggregation_results['I'] = D,I

        print('combined')
        # import ipdb;ipdb.set_trace()
        if patch_aggregation_results['combined'].shape[-1] not in [3,4,5]:
            import pdb;pdb.set_trace()
        if 1 in patch_aggregation_results['combined'].shape[1:3]:
            import pdb;pdb.set_trace()
        # return y,I,w
        if True:
            patch_aggregation_results['combined']  = patch_aggregation_results['combined'] * mask  + (1-mask) * x
            # import ipdb;ipdb.set_trace()
        return patch_aggregation_results
        END_PNN_faiss = None
    
    def get_pca_feats(self,input,init=False,reproducible=False):
        if input.ndim > 2:
            input = input.reshape(input.shape[0],-1).contiguous()
        # import pdb;pdb.set_trace()
        if init:
            self.pca = PCA(self.N_PCA_COMPONENTS,reproducible=reproducible)
            self.pca.fit(input)
            if os.environ.get('DEBUG_REPRODUCIBILITY',False) == '1':
                dpca = dutils.__dict__.get('pca',None)
                dutils.pca = [] if dpca is None else dpca
                # dutils.k.append(keys_for_index)
                dutils.pca.append(self.pca.components_)
        input_proj = self.pca.transform(input)
        input_proj = (input_proj).contiguous()    
        print('USING PCA!!!!!!!!!')
        return input_proj        
    def get_feats(self,input,init=False,reproducible=False):
        '''
        b,c,h,w = input.shape[0],3,7,7
        from hog import gradient_histogram
        return gradient_histogram(input.view(-1,3,7,7), 8).view(b,-1)
        '''
        if False and 'hog features':
            from model.hog import gradient_histogram
            input_hog = gradient_histogram(input, 10)
            input_hog = input_hog.flatten(start_dim=-2,end_dim=-1)
            return input_hog
        # import ipdb;ipdb.set_trace()
        if self.USE_PCA:
            if init: assert (self.N_PCA_COMPONENTS) and (self.N_PCA_COMPONENTS > 0) 
            # import pdb;pdb.set_trace()
            return self.get_pca_feats(input,init=init,reproducible=reproducible)
        if input.ndim > 2:
            input = input.reshape(input.shape[0],-1).contiguous()        
        return input
        

def extract_patches(src_img, patch_size, stride):
    channels = src_img.shape[-1]
    assert channels in [1,3,4,5]
    if not isinstance(src_img,torch.Tensor) and not len(src_img.shape) == 4:
        img = torch.from_numpy(src_img).to(device).unsqueeze(0).permute(0, 3, 1, 2)
    else:
        img = src_img
        if src_img.ndim == 3:
            img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)

    return torch.nn.functional.unfold(img, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0)) \
        .squeeze(dim=0).permute((1, 0)).reshape(-1, channels, patch_size[0], patch_size[1])


def compute_distances(queries, keys):
    dist_mat = torch.zeros((queries.shape[0], keys.shape[0]), dtype=torch.float16, device=device)
    for i in range(len(queries)):
        dist_mat[i] = torch.mean((queries[i] - keys) ** 2, dim=(1, 2, 3))
    return dist_mat

