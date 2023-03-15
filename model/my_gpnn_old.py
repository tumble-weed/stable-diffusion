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
from kornia.geometry.transform.pyramid import pyrdown
from eutils.NN_modules import PytorchNNLowMemory
from aggregation import combine_patches
import debug
from collections import defaultdict
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
    bchw_r = torch.nn.functional.interpolate(bchw,size)
    bhwc_r = bchw_r.permute((0,2,3,1))
    return bhwc_r
class gpnn:
    def __init__(self, config):
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
        if config['task'] == 'inpainting':
            mask = img_read(config['mask'])
            mask_patch_ratio = np.max(np.sum(mask, axis=0), axis=0) // self.PATCH_SIZE
            coarse_dim = mask.shape[0] / mask_patch_ratio
            self.COARSE_DIM = (coarse_dim, coarse_dim)
        self.STRIDE = (config['stride'], config['stride'])
        self.R = config['pyramid_ratio']
        self.ALPHA = config['alpha']
        self.PATCH_AGGREGATION = config['patch_aggregation']

        # cuda init
        global device
        if config['no_cuda']:
            device = torch.device('cpu')
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                print('cuda initialized!')

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
        if config['task'] == 'structural_analogies':
            img_path = config['img_a']
        else:
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

        
        
        if config['out_size'] != 0:
            if self.input_img.shape[0] > config['out_size']:
                self.input_img = rescale(self.input_img, config['out_size'] / self.input_img.shape[0], multichannel=True)
        self.input_img_tensor = torch.tensor(self.input_img).float().to(device).permute(2,0,1).unsqueeze(0)
        # pyramids
        pyramid_depth = np.log(min(self.input_img.shape[:2]) / min(self.COARSE_DIM)) / np.log(self.R)
        self.add_base_level = True if np.ceil(pyramid_depth) > pyramid_depth else False
        pyramid_depth = int(np.ceil(pyramid_depth))

        # self.x_pyramid0 = list(
        #     tuple(pyramid_gaussian(self.input_img, pyramid_depth, downscale=self.R, multichannel=True)))
        
        self.x_pyramid = [pyrdown(self.input_img_tensor, 
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R ** i).permute(0,2,3,1) for i in range(1,pyramid_depth)]
        self.x_pyramid.insert(0,self.input_img_tensor.permute(0,2,3,1))
        # import pdb;pdb.set_trace()
        #============================================
        self.other_x = torch.ones(1,1,*self.input_img.shape[:2]).float().to(device)
        self.other_x_pyramid = list(
            tuple(pyramid_gaussian(self.input_img, pyramid_depth, downscale=self.R, multichannel=True)))
        #============================================
        # import pdb;pdb.set_trace()
        if self.add_base_level is True:
            # self.x_pyramid[-1] = resize(self.x_pyramid[-2], self.COARSE_DIM)
            self.x_pyramid[-1] = resize_bhwc(self.x_pyramid[-2], self.COARSE_DIM)
        self.y_pyramid = [0] * (pyramid_depth + 1)

        # out_file
        # filename = os.path.splitext(os.path.basename(img_path))[0]
        filename = 'out_img'
        self.out_file = os.path.join(config['out_dir'], "%s_%s.png" % (filename, config['task']))
        self.batch_size = config['batch_size']
        # coarse settings
        if config['task'] == 'random_sample':
            if isinstance(self.x_pyramid[-1],np.ndarray):
                noise = np.random.normal(0, config['sigma'], (self.batch_size,)+ self.COARSE_DIM)[..., np.newaxis]
                self.coarse_img = noise

                if self.INIT_FROM == 'target':
                    self.coarse_img = self.coarse_img + noise
                
            else:
                assert len(self.x_pyramid[-1].shape) == 4
                noise = config['sigma']*torch.randn((self.batch_size,)+ self.COARSE_DIM)[..., np.newaxis]
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
        elif config['task'] == 'inpainting':
            assert False,'not implemented'
            self.coarse_img = self.x_pyramid[-1]
            mask_img = img_read(config['mask'])
            self.mask_pyramid = [0] * len(self.x_pyramid)
            for i in range(len(self.mask_pyramid)):
                mask = resize(mask_img, self.x_pyramid[i].shape) != 0
                mask = extract_patches(mask, self.PATCH_SIZE, self.STRIDE)
                if self.input_img.shape[2] > 1:
                    mask = torch.all(mask, dim=3)
                mask = torch.all(mask, dim=2)
                mask = torch.all(mask, dim=1)
                self.mask_pyramid[i] = mask
        assert len(self.coarse_img.shape) == 4

        self.running_keys = None
        self.running_values = None
        self.resolution =None
        self.n_keys = {}
        self.trends = defaultdict(list)
        print('init done')

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
            # if i < 2:
            #         self.STRIDE = (2,2)            
            query_mask = None
            del resolution_results
            # if i == 9:
            #     I = None
            #     print('breaking early')
            #     break
            if  i ==1:
                torch.cuda.empty_cache()
            if i == len(self.x_pyramid) - 1:
                print('flipping initial image')
                queries = self.coarse_img.flip(-1)
                keys = self.x_pyramid[i]
            else:
                # queries = np.array([resize(yp, self.x_pyramid[i].shape) for yp in self.y_pyramid[i + 1]])
                # queries = torch.stack([torch.nn.functional.interpolate(yp,self.x_pyramid[i].shape[:2]) for yp in self.y_pyramid[i + 1]],dim=0)
                
                if False and 'adding per level noise':
                    noise = 0.1*0.5*config['sigma']*torch.randn((self.batch_size,)+ self.x_pyramid[i].shape[1:3])[..., np.newaxis]
                    noise = noise.to(device)
                    
                    queries = resize_bhwc(self.y_pyramid[i + 1],self.x_pyramid[i].shape[1:3]) + noise
                else:
                    queries = resize_bhwc(self.y_pyramid[i + 1],self.x_pyramid[i].shape[1:3])
                # import pdb;pdb.set_trace()
                # keys = resize(self.x_pyramid[i + 1], self.x_pyramid[i].shape)
                keys = resize_bhwc(self.x_pyramid[i + 1],self.x_pyramid[i].shape[1:3])
            new_keys = True
            # for j in tqdm.tqdm_notebook(range(self.T)):
            D,I = None,None
                
            for j in tqdm.tqdm(range(self.T)):                
                if self.is_faiss:
                    # self.y_pyramid[i],I,W 

                    resolution_results = self.PNN_faiss(self.x_pyramid[i], keys, queries, self.PATCH_SIZE, self.STRIDE,
                                                       self.ALPHA, mask=None, new_keys=new_keys,
                                                       other_x = self.x_pyramid[i],query_mask=query_mask,
                                                       Dprev=D,Iprev=I
                                                       )
                    if j ==0:
                        if True and 'find small distance':
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
                if (query_mask is not None) and not query_mask.any():
                    break
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
    def PNN_faiss(self, x, x_scaled, y_scaled, patch_size, stride, alpha, mask=None, new_keys=True,
        other_x=None,extra_return={},query_mask=None,
        Dprev=None,Iprev=None):
        if self.resolution == None:
            self.resolution = x.shape
        assert (x.max() <= 1.) and (x.min() >= 0.)
        assert (x_scaled.max() <= 1.) and (x_scaled.min() >= 0.)
        # y_scaled has noise added, so will have >1 and <0 values
        # assert (y_scaled.max() <= 1.) and (y_scaled.min() >= 0.)
        other_x = None;print('setting other_x to None forcefully')
        print('using faiss')
        print('this shouldnt be np.array but also work for tensor')
        assert y_scaled[0].shape[-1] == 3
        queries = torch.stack([extract_patches(ys, patch_size, stride) for ys in y_scaled],dim=0)

            
        from model.hog import gradient_histogram
        
        # import pdb;pdb.set_trace()
        # queries = queries[...,::2,::2]
        print('extracted query',queries.shape)
        keys = extract_patches(x_scaled, patch_size, stride)
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

                
            # import pdb;pdb.set_trace()
            if self.KEYS_TYPE == 'multi-resolution':
                if new_keys:
                    if self.running_values is None:
                        self.running_values = values
                    else:
                        if self.running_values.shape[0] == sum(self.n_keys.values()):
                            self.running_values[-self.n_keys[x.shape]:] = values
                        else:
                            self.running_values = torch.cat([self.running_values,values],dim=0)
                        # self.running_values = self.running_values#[-values.shape[0]:]
        else:
            print('using laplacian pyramid')
            x_high = x - x_scaled
            values = extract_patches(x_high, patch_size, stride)
        if other_x is not None:
            other_values = extract_patches(other_x,patch_size, stride)
        print('extracted values')
        if mask is not None:
            assert False,'not implemented for 4d query'
            queries = queries[mask]
            keys = keys[~mask]
        #====================================================================

        # queries_flat = np.ascontiguousarray(queries.reshape((queries.shape[0]*queries.shape[1], -1)).cpu().numpy(), dtype='float32')
        assert queries.ndim == 5
        # queries_flat = queries.reshape((queries.shape[0]*queries.shape[1], -1)).contiguous()
        queries_flat_batch = queries.flatten(start_dim=0,end_dim=1).contiguous()
        if query_mask is not None:
            # import pdb;pdb.set_trace()
            queries_flat_batch = queries_flat_batch[query_mask]        

        # keys_flat = np.ascontiguousarray(keys.reshape((keys.shape[0], -1)).cpu().numpy(), dtype='float33142')
        assert keys.ndim == 4
        # keys_flat = keys.reshape((keys.shape[0], -1)).contiguous()
        # import pdb;pdb.set_trace()

        if new_keys:
            # keys_proj = keys_flat
            keys_proj = self.get_feats(keys,init=True)
            # import pdb;pdb.set_trace()
            '''
            if self.USE_PCA:
                # keys_proj = self.fit_transform_pca(self.N_PCA_COMPONENTS,keys_proj)
                keys_proj = self.get_pca_feats(keys_proj,n_pca_components = self.N_PCA_COMPONENTS,init=True)
            '''
            #==================================================
            if self.KEYS_TYPE == 'multi-resolution':
                if self.running_keys is not None:
                    # assert False
                    if self.running_keys.shape[0] == sum(self.n_keys.values()):
                        self.running_keys[-self.n_keys[x.shape]:] = keys_proj
                    else:
                        self.running_keys = torch.cat([self.running_keys,keys_proj],dim=0)
                    # self.running_keys =self.running_keys[-keys_proj.shape[0]:]
                else:
                    self.running_keys = keys_proj
                # import pdb;pdb.set_trace()
            #==================================================
            
            n_patches = keys_proj.shape[0]
            print(n_patches)
            # import pdb;pdb.set_trace()
            if False and 'simple':
                self.index = faiss.IndexFlatL2(keys_proj.shape[-1])
            elif True and 'ivf':
                print('using ivf')
                nlist = min(200,keys_proj.shape[0])
                print(keys_proj.shape)
                if self.KEYS_TYPE == 'single-resolution':
                    quantizer = faiss.IndexFlatL2(keys_proj.shape[-1])  # the other index
                    # import pdb;pdb.set_trace()
                    self.index = faiss.IndexIVFFlat(quantizer, keys_proj.shape[-1], nlist)
                    self.index.nprobe = 3
                elif self.KEYS_TYPE == 'multi-resolution':
                    print('using running keys')
                    quantizer = faiss.IndexFlatL2(self.running_keys.shape[-1])  # the other index
                    self.index = faiss.IndexIVFFlat(quantizer, self.running_keys.shape[-1], nlist)
                    self.index.nprobe = 1
            else:
                print('using smaller code length')
                self.index = faiss.IndexFlatL2(10)
            # import pdb;pdb.set_trace()
            print('created index')
            # import pdb;pdb.set_trace()
            if torch.cuda.is_available():
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print('pushed index to gpu')
            if self.KEYS_TYPE ==  'single-resolution':
                self.index.train(keys_proj)
                self.index.add(keys_proj)
            elif self.KEYS_TYPE ==  'multi-resolution':
                self.index.train(self.running_keys)
                self.index.add(self.running_keys)                
        # import pdb;pdb.set_trace()
        '''
        queries_proj = queries_flat
        queries_proj = self.get_feats(queries_proj,init=False)
        '''
        queries_proj = self.get_feats(queries_flat_batch,init=False)
        # import pdb;pdb.set_trace()
        print('searching')
        # import pdb;pdb.set_trace()
        print(queries_proj.shape)
        if queries_proj.shape[0] > 62496:
            batch_size = 62496
            D = torch.zeros(queries_proj.shape[0],1,device=device).half()
            I = torch.zeros(queries_proj.shape[0],1,device=device)
            I = I.long()
            if True and keys.shape[0] < 2147483647:
                I =I.int()
            else:
                I =I.long()
            nbatches = (queries_proj.shape[0]+batch_size - 1)//batch_size
            for i in range(nbatches):
                queries_proji = queries_proj[i*batch_size:(i+1)*batch_size]
                Di, Ii = self.index.search(queries_proji, 1)
                D[i*batch_size:(i)*batch_size + Di.shape[0]] = Di
                I[i*batch_size:(i)*batch_size + Ii.shape[0]] = Ii
                # import pdb;pdb.set_trace()
        else:
            D, I = self.index.search(queries_proj, 1)
            D = D.half()
            if True and keys.shape[0] < 2147483647:
                I = I.int()
        if False and keys.shape[0] > 100000:
            print(keys.shape[0],I.shape,Ii.shape)
            import time;time.sleep(5)
            # import pdb;pdb.set_trace()
        # D1, I1 = self.index.search(queries_proj.cpu().numpy(), 1)
        if query_mask is not None:
            # import pdb;pdb.set_trace()
            Dprev[query_mask] = D
            Iprev[query_mask] = I
            D = Dprev
            I = Iprev
        if mask is not None:
            assert False,'not implemented'
            values[mask] = values[~mask][I.T]
        else:
            # import pdb;pdb.set_trace()
            if self.KEYS_TYPE == 'single-resolution':
                # import pdb;pdb.set_trace()
                if False:
                    values = values[I.T]
                else:
                    assert I.shape[-1] == 1
                    values = torch.index_select(values,0,I.squeeze()).unsqueeze(0)
                    # assert (values[I.T] == values1).all()
                    # import sys;sys.exit()
            elif self.KEYS_TYPE == 'multi-resolution':
                print('using running values')
                values = self.running_values[I.T]
                assert False,'index_select not implemented'
                # import pdb;pdb.set_trace()
            #O = values[I.T]
        # print('see D shape')
        # import pdb;pdb.set_trace()        
        
        assert values.ndim == 5
        assert values.shape[0] == 1

        values = values.squeeze()
        # import pdb;pdb.set_trace()
        if 'hardcoded' and False:
            values = torch.tile(keys,(100,1,1,1));
            print('hardcoding values')
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
        # import pdb;pdb.set_trace()
        assert len(x_scaled.shape) == 4
        # import pdb;pdb.set_trace()
        # import IPython;IPython.embed()
        '''
        y = torch.stack([combine_patches(v, patch_size, stride, x_scaled.shape[1:3]+(3,),as_np=False,
                                            patch_aggregation=self.PATCH_AGGREGATION,
                                            distances=d)['combined'] for v,d in zip(values,distances)],dim=0)
        '''
        if True and '1 image at a time':
            y = []
            w = []
            from collections import defaultdict
            patch_aggregation_results = defaultdict(list)
            
            for ii,(v,d) in enumerate(zip(values,distances)):            
                debug.I = I[ii*distances.shape[1]:(ii+1)*distances.shape[1]]
                result = combine_patches(v, patch_size, stride, x_scaled.shape[1:3]+(3,),as_np=False,
                patch_aggregation=self.PATCH_AGGREGATION,
                distances=d)
                for k in result:
                    if k not in ['patch_aggregation']:
                        # import pdb;pdb.set_trace()
                        patch_aggregation_results[k].append(result[k])
                # yi,wi = result['combined'],result['weights']
                # y.append(yi)
                # w.append(wi)
            patch_aggregation_results = {k:v for k,v in patch_aggregation_results.items()}
            patch_aggregation_results['patch_aggregation']=result['patch_aggregation']
            for k in result:
                if k not in ['patch_aggregation']:
                    patch_aggregation_results[k] = torch.stack(patch_aggregation_results[k],dim=0)
                    # .append(torch.stack(patch_aggregation_results[k],dim=0))        
        elif  'multiple images':
            patch_aggregation_results = combine_patches(values, patch_size, stride, x_scaled.shape[1:3]+(3,),as_np=False,
                patch_aggregation=self.PATCH_AGGREGATION,
                distances=distances)
            
        patch_aggregation_results['D'], patch_aggregation_results['I'] = D,I
        if False:
            # y = torch.clamp(y + y_scaled,0,1)            
            y = (y + x_scaled); print('using laplacian pyramid')
            # y = (y -y.min())/(y.max() -y.min())
        if other_x is not None:
            assert False,'shouldnt be here'
            # assert isinstance(other_x,torch.Tensor)
            other_y = combine_patches(other_values, patch_size, stride, 
                                    x_scaled.shape,patch_aggregation=self.PATCH_AGGREGATION,as_np=False)
            extra_return['other_y']  = other_y
        print('combined')
        if patch_aggregation_results['combined'].shape[-1] !=3:
            import pdb;pdb.set_trace()
        if 1 in patch_aggregation_results['combined'].shape[1:3]:
            import pdb;pdb.set_trace()
        # return y,I,w
        return patch_aggregation_results

    def get_pca_feats(self,input,init=False):
        if input.ndim > 2:
            input = input.reshape(input.shape[0],-1).contiguous()
        # import pdb;pdb.set_trace()
        if init:
            self.pca = PCA(self.N_PCA_COMPONENTS)
            self.pca.fit(input)
        input_proj = self.pca.transform(input)
        input_proj = (input_proj).contiguous()    
        print('USING PCA!!!!!!!!!')
        return input_proj        
    def get_feats(self,input,init=False):
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
        if self.USE_PCA:
            if init: assert (self.N_PCA_COMPONENTS) and (self.N_PCA_COMPONENTS > 0) 
            # import pdb;pdb.set_trace()
            return self.get_pca_feats(input,init=init)
        if input.ndim > 2:
            input = input.reshape(input.shape[0],-1).contiguous()        
        return input
        

def extract_patches(src_img, patch_size, stride):
    channels = src_img.shape[-1]
    assert channels in [1,3]
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

