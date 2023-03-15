import numpy as np
import torch
from skimage.transform.pyramids import pyramid_gaussian
from skimage.transform import rescale, resize
from torch.nn.functional import fold, unfold
from .utils import *
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
# import kornia.geometry.transform
import model.kornia_utils
class gpnn:
    def __init__(self, config):
        # general settings
        self.T = config['iters']
        self.PATCH_SIZE = (config['patch_size'], config['patch_size'])
        self.COARSE_DIM = (config['coarse_dim'], config['coarse_dim'])
        if config['task'] == 'inpainting':
            mask = img_read(config['mask'])
            mask_patch_ratio = np.max(np.sum(mask, axis=0), axis=0) // self.PATCH_SIZE

            print('from the documentation: **the mask is in the following format - ones in the pixels where the inpainted area is, and zeros elsewhere.** ')
            coarse_dim = mask.shape[0] / mask_patch_ratio
            self.COARSE_DIM = (coarse_dim, coarse_dim)
        self.STRIDE = (config['stride'], config['stride'])
        self.R = config['pyramid_ratio']
        self.ALPHA = config['alpha']

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
        if config['task'] == 'structural_analogies':
            img_path = config['img_a']
        else:
            img_path = config['input_img']
        self.input_img = img_read(img_path)
        self.input_img = rescale(self.input_img,(0.5,0.5,1)); print('rescaling input image to 0.5')
        assert self.input_img.shape[-1] == 3,'channels reduced during rescaling?'
        # self.imput
        if not isinstance(self.input_img,torch.Tensor):
            print('warning input_img is not a tensor')
        input_img_3chan = self.input_img if self.input_img.ndim == 3 else self.input_img[...,None]
        self.input_img_t = torch.tensor(input_img_3chan).permute(2,0,1).unsqueeze(0).to(device)
        if self.input_img_t.dtype == torch.uint8:
            self.input_img_t = self.input_img_t.float() / 255.0
            
        if config['out_size'] != 0:
            if self.input_img.shape[0] > config['out_size']:
                self.input_img = rescale(self.input_img, config['out_size'] / self.input_img.shape[0], multichannel=True)
        self.saliency = torch.tensor(
			np.random.random(size = (1,1) + self.input_img.shape[:2])).float().to(device).requires_grad_(True)
        assert self.saliency.max() <= 1
        assert self.saliency.min() >= 0
        # pyramids
        pyramid_depth = np.log(min(self.input_img.shape[:2]) / min(self.COARSE_DIM)) / np.log(self.R)
        self.add_base_level = True if np.ceil(pyramid_depth) > pyramid_depth else False
        pyramid_depth = int(np.ceil(pyramid_depth))

        '''
        assert False,'we need to represent saliency pyramid using kornia'
        self.x_pyramid = list(
            tuple(pyramid_gaussian(self.input_img, pyramid_depth, downscale=self.R, multichannel=True)))
            
        self.saliency_pyramid = list(
            tuple(pyramid_gaussian(self.saliency, pyramid_depth, downscale=self.R, multichannel=True)))

        '''
        self.x_pyramid = model.kornia_utils.build_pyramid(self.input_img_t, pyramid_depth,downscale=self.R)
        
        self.saliency_pyramid = model.kornia_utils.build_pyramid(self.saliency, pyramid_depth,downscale=self.R)
                
        if self.add_base_level is True:
            # self.x_pyramid[-1] = resize(self.x_pyramid[-2], self.COARSE_DIM)
            self.x_pyramid[-1] = torch.nn.functional.interpolate(self.x_pyramid[-2], self.COARSE_DIM)
        self.y_pyramid = [0] * (pyramid_depth + 1)

        # out_file
        filename = os.path.splitext(os.path.basename(img_path))[0]
        self.out_file = os.path.join(config['out_dir'], "%s_%s.png" % (filename, config['task']))

        # coarse settings
        if config['task'] == 'random_sample':
            noise = np.random.normal(0, config['sigma'], self.COARSE_DIM)[..., np.newaxis]
            device = self.x_pyramid[-1]
            noise = torch.tensor(np.transpose(noise, (2, 0, 1))).to(device).float().unsqueeze(0)
            self.coarse_img = self.x_pyramid[-1] + noise
        elif config['task'] == 'structural_analogies':
            self.coarse_img = img_read(config['img_b'])
            self.coarse_img = resize(self.coarse_img, self.x_pyramid[-1].shape)
        elif config['task'] == 'inpainting':
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

    def run(self, to_save=True):
        for i in reversed(range(len(self.x_pyramid))):
            assert isinstance(self.x_pyramid[i],torch.Tensor)
            # assert isinstance(self.y_pyramid[i],torch.Tensor)
            if i == len(self.x_pyramid) - 1:
                queries = self.coarse_img
                keys = self.x_pyramid[i]
            else:
                '''
                queries = resize(self.y_pyramid[i + 1], self.x_pyramid[i].shape)
                keys = resize(self.x_pyramid[i + 1], self.x_pyramid[i].shape)
                '''
                queries = torch.nn.functional.interpolate(self.y_pyramid[i + 1], self.x_pyramid[i].shape)
                keys = torch.nn.functional.interpolate(self.x_pyramid[i + 1], self.x_pyramid[i].shape)
            new_keys = True
            for j in range(self.T):
                if self.is_faiss:
                    self.y_pyramid[i] = self.PNN_faiss(self.x_pyramid[i], keys, queries, self.PATCH_SIZE, self.STRIDE,
                                                       self.ALPHA, mask=None, new_keys=new_keys,saliency=self.saliency)
                else:
                    assert False,'not compatible with tensors'
                    self.y_pyramid[i] = self.PNN(self.x_pyramid[i], keys, queries, self.PATCH_SIZE, self.STRIDE,
                                                 self.ALPHA)
                queries = self.y_pyramid[i]
                keys = self.x_pyramid[i]
                # if j > 1:
                if j > 0:
                    new_keys = False
        if to_save:
            img_save(self.y_pyramid[0], self.out_file)
            as_np = tensor_to_numpy(self.y_pyramid[0])[0]
            as_np = torch.transpose(as_np,(1,2,0))
            img_save(as_np, self.out_file)
        else:
            return self.y_pyramid[0]

    def PNN(self, x, x_scaled, y_scaled, patch_size, stride, alpha, mask=None):
        # assert False,'not implemented for saliency'
        queries = extract_patches(y_scaled, patch_size, stride)
        keys = extract_patches(x_scaled, patch_size, stride)
        values = extract_patches(x, patch_size, stride)
        if mask is None:
            dist = torch.cdist(queries.view(len(queries), -1), keys.view(len(keys), -1))
        else:
            m_queries, m_keys = queries[mask], keys[~mask]
            dist = torch.cdist(m_queries.view(len(m_queries), -1), m_keys.view(len(m_keys), -1))
        norm_dist = (dist / (torch.min(dist, dim=0)[0] + alpha))  # compute_normalized_scores
        NNs = torch.argmin(norm_dist, dim=1)  # find_NNs
        if mask is None:
            values = values[NNs]
        else:
            values[mask] = values[~mask][NNs]
            # O = values[NNs]  # replace_NNs(values, NNs)
        
        # y = combine_patches(values, patch_size, stride, x_scaled.shape)
        y = combine_patches(values, patch_size, stride, x_scaled.shape)
        assert y.ndim == 4
        y = y.permute(0,2,3,1)[0]
        y = tensor_to_numpy(y)
        return y

    def PNN_faiss(self, x, x_scaled, y_scaled, patch_size, stride, alpha, mask=None, new_keys=True,saliency=None):
        # assert x.ndim == 3
        # assert x_scaled.ndim == 3
        queries = extract_patches(y_scaled, patch_size, stride)
        keys = extract_patches(x_scaled, patch_size, stride)
        values = extract_patches(x, patch_size, stride)
        saliency_values = extract_patches(saliency, patch_size, stride)
        if mask is not None:
            assert False,'not compatible with saliency'
            queries = queries[mask]
            keys = keys[~mask]
        queries_flat = np.ascontiguousarray(queries.reshape((queries.shape[0], -1)).cpu().numpy(), dtype='float32')
        keys_flat = np.ascontiguousarray(keys.reshape((keys.shape[0], -1)).cpu().numpy(), dtype='float32')

        if new_keys:
            self.index = faiss.IndexFlatL2(keys_flat.shape[-1])
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            self.index.add(keys_flat)
        print('see if any problems with faiss');import pdb;pdb.set_trace()
        D, I = self.index.search(queries_flat, 1)
        print('see if any problems with faiss');import pdb;pdb.set_trace()
        if mask is not None:
            assert False,'not compatible with saliency'
            values[mask] = values[~mask][I.T]
        else:
            values = values[I.T]
            saliency_values = saliency_values[I.T]
            #O = values[I.T]
        x = torch.tensor(x,device=values.device)
        x_scaled = torch.tensor(x_scaled,device=values.device)
        x = x.permute(2,0,1)[None,...]
        x_scaled = x_scaled.permute(2,0,1)[None,...]
        y = combine_patches(values, patch_size, stride, x_scaled.shape)
        y_saliency = combine_patches(values, patch_size, stride, x_scaled.shape)
        # print('view y');import pdb;pdb.set_trace()
        return y


def extract_patches(src_img, patch_size, stride):
    
    if isinstance(src_img,torch.Tensor):
        img = src_img
        channels = src_img.shape[1]    
    else:
        if src_img.ndim == 3:
            assert src_img.shape[-1] == 3
            channels = 3
        else:
            src_img = src_img[...,None]
            channels = 1
        img = torch.from_numpy(src_img).to(device).unsqueeze(0).permute(0, 3, 1, 2)
        

    return torch.nn.functional.unfold(img, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0)) \
        .squeeze(dim=0).permute((1, 0)).reshape(-1, channels, patch_size[0], patch_size[1])


def compute_distances(queries, keys):
    dist_mat = torch.zeros((queries.shape[0], keys.shape[0]), dtype=torch.float16, device=device)
    for i in range(len(queries)):
        dist_mat[i] = torch.mean((queries[i] - keys) ** 2, dim=(1, 2, 3))
    return dist_mat

'''
def combine_patches(O, patch_size, stride, img_shape):
    channels = 3
    O = O.permute(1, 0, 2, 3).unsqueeze(0)
    patches = O.contiguous().view(O.shape[0], O.shape[1], O.shape[2], -1) \
        .permute(0, 1, 3, 2) \
        .contiguous().view(1, channels * patch_size[0] * patch_size[0], -1)
    combined = fold(patches, output_size=img_shape[:2], kernel_size=patch_size, stride=stride)

    # normal fold matrix
    input_ones = torch.ones((1, img_shape[2], img_shape[0], img_shape[1]), dtype=O.dtype, device=device)
    divisor = unfold(input_ones, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
    divisor = fold(divisor, output_size=img_shape[:2], kernel_size=patch_size, stride=stride)

    divisor[divisor == 0] = 1.0
    return (combined / divisor).squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
'''
def combine_patches(O, patch_size, stride, img_shape):
    assert img_shape.__len__() == 4
    assert len(patch_size) == 2
    if len(O.shape) == 6:
        # O: (batch_size,chan,n_patches,n_patches,patch_size,patch_size) -> (batch_size,n_patches,n_patches,chan,patch_size,patch_size)
        O = O.permute((0,2,3,1,4,5))
        O = O.contiguous()
        O = O.view(-1,*O.shape[-3:])
        assert O.shape[-2:] == patch_size

    
    

    # O.shape == (-1,channels,patch_size,patch_size)
    channels = 3
    O = O.permute(1, 0, 2, 3).unsqueeze(0) # chan,batch_size,patch_size,patch_size
    patches = O.contiguous().view(O.shape[0], O.shape[1], O.shape[2], -1) \
        .permute(0, 1, 3, 2) \
        .contiguous().view(1, channels * patch_size[0] * patch_size[0], -1)
    
    # batch_size,chan,Ypatch_size,-1
    # batch_size,chan,Xpatch_size,Ypatch_size
    #  -> 1, channels*Xpatch_size*Ypatch_size, H*W
    # chan,batch_size,patch_size,patch_size
    combined = fold(patches, output_size=img_shape[-2:], kernel_size=patch_size, stride=stride)

    # normal fold matrix
    input_ones = torch.ones((1, img_shape[1], img_shape[-2], img_shape[-1]), dtype=O.dtype, device=device)
    divisor = unfold(input_ones, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
    divisor = fold(divisor, output_size=img_shape[-2:], kernel_size=patch_size, stride=stride)

    divisor[divisor == 0] = 1.0
    combined =  (combined / divisor).squeeze(dim=0).permute(1, 2, 0)
    # convert from hwc to bchw format
    combined = combined.permute(2,0,1)[None,...]
    return combined