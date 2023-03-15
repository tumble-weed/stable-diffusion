import numpy as np
import torch
import kornia as K
from kornia.geometry.transform.pyramid import pyrdown

def extract_patches(src_img, patch_size, stride):
    channels = src_img.shape[-1]
    assert channels in [1,3]
    if not isinstance(src_img,torch.Tensor) and not len(src_img.shape) == 4:
        device = src_img.device
        img = torch.from_numpy(src_img).to(device).unsqueeze(0).permute(0, 3, 1, 2)
    else:
        img = src_img
        if src_img.ndim == 3:
            img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)

    return torch.nn.functional.unfold(img, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0)) \
        .squeeze(dim=0).permute((1, 0)).reshape(-1, channels, patch_size[0], patch_size[1])
def create_pyramid(mask_tensor,R,PATCH_SIZE,STRIDE,pyramid_depth):
    mask_pyramid = [pyrdown(mask_tensor, 
                        border_type = 'reflect', 
                        align_corners = True, 
                        factor = R ** i).permute(0,2,3,1) for i in range(1,pyramid_depth)]
    mask_pyramid.insert(0,mask_tensor.permute(0,2,3,1))         
    # import dutils
    # import ipdb;ipdb.set_trace()   
    key_pyramid = []
    query_pyramid = []
    for i in range(len(mask_pyramid)):
        # import pdb;pdb.set_trace()
        mask = mask_pyramid[i]
        mask = extract_patches(mask, PATCH_SIZE, STRIDE)
        # if self.input_img_tensor.shape[2] > 1:
        #     mask = torch.all(mask, dim=3)
        
        if True:
            def thresholded_mean_query_and_thresholded_mean_key(mask,query_threshold=0.5,key_threshold=0.5):
                mean_mask = mask.mean(dim=(-1,-2))
                mask_query = (mean_mask > query_threshold)
                mask_key = ~(mean_mask < key_threshold)
                return mask_query,mask_key
            
            def thresholded_query_and_thresholded_key(mask,query_threshold=0.75,key_threshold=0.25):
                mask_query = (mask > query_threshold).any(dim =-1)
                mask_query = mask_query.any(dim=-1)
                mask_key = (mask < key_threshold).any(dim=-1)
                mask_key = mask_key.any(dim=-1)
                return mask_query,mask_key

            def eroded_query_and_eroded_key(mask):
                #=========================================
                mask_query = torch.all(mask, dim=-1)
                mask_query = torch.all(mask_query, dim=-1)           
                #=========================================
                anti_mask = 1 - mask
                mask_key = torch.all(anti_mask, dim=-1)
                mask_key = torch.all(mask_key, dim=-1)                
                # again complement it because the mask is stored as 1 for holes
                mask_key = ~mask_key
                return mask_query,mask_key
            #========================================
            def eroded_query_and_dilated_key(mask):
                mask_key = torch.any(mask, dim=-1)
                mask_key = torch.any(mask_key, dim=-1)           
                mask_query = torch.all(mask, dim=-1)
                mask_query = torch.all(mask_query, dim=-1)           
                return mask_query,mask_key                
            def dilated_query_and_avg_key(mask):
                mask_key = (mask.mean(dim=(-1,-2)) > 0.5)
                mask_query = torch.any((mask > 0.75), dim=-1)
                mask_query = torch.any(mask_query, dim=-1)                                       
                return mask_query,mask_key                
            def standard(mask):

                mask_query = torch.all(mask, dim=-1)
                mask_query = torch.all(mask_query, dim=-1)           
                mask_key = mask_query
                return mask_query,mask_key                            
            #=======================================================
            if 'dual pyramids':
                if False and 'thresholded mean query and thresholded mean key':
                    mask_query,mask_key = thresholded_mean_query_and_thresholded_mean_key(mask)
                    key_pyramid.append(mask_key)
                    query_pyramid.append(mask_query)
                if False and 'thresholded query and thresholded key':
                    mask_query,mask_key = thresholded_query_and_thresholded_key(mask)
                    key_pyramid.append(mask_key)
                    query_pyramid.append(mask_query)

                if False and 'dilated query, avg_key':
                    mask_query,mask_key = dilated_query_and_avg_key(mask)
                    key_pyramid.append(mask_key)
                    query_pyramid.append(mask_query)
                elif False and 'eroded key, eroded query':
                    mask_query,mask_key = eroded_query_and_eroded_key(mask)
                    key_pyramid.append(mask_key)
                    query_pyramid.append(mask_query)
                elif False and 'dilated key eroded query':
                    mask_query,mask_key = eroded_query_and_dilated_key(mask)
                    key_pyramid.append(mask_key)
                    query_pyramid.append(mask_query)                                
                elif True and 'standard':
                    mask_query,mask_key = standard(mask)
                    key_pyramid.append(mask_key)
                    query_pyramid.append(mask_query)            
            # import ipdb;ipdb.set_trace()
            #=======================================================
        else:
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
        mask_pyramid[i] = mask
    '''
    # mask_pyramid = [pyrdown(mask_tensor, 
    #                     border_type = 'reflect', 
    #                     align_corners = True, 
    #                     factor = R ** i).permute(0,2,3,1) for i in range(1,pyramid_depth)]
    # mask_pyramid.insert(0,mask_tensor.permute(0,2,3,1))                
    import ipdb;ipdb.set_trace()
    importlib.reload(mask_pyramid)
    for ii,p in enumerate(self.mask_pyramid):
        skimage.io.imsave('mask_pyramid{ii}.png',tensor_to_numpy(p))
    for ii,p in enumerate(self.query_pyramid):
        skimage.io.imsave('query_pyramid{ii}.png',p)                
    for ii,p in enumerate(self.key_pyramid):
        skimage.io.imsave('key_pyramid{ii}.png',p)                                
    '''        
    return mask_pyramid,query_pyramid,key_pyramid


def cutoff_pyramid_at_zero(based_on,*others):
    L = len(based_on)
    max_i = L
    for i in reversed(range(L)):
        if based_on[i].sum() > 0:
            break
        max_i -= 1
    del based_on[max_i:]
    for l in others:
        del l[max_i:]
    # if max_i < L:
    #     import ipdb;ipdb.set_trace()

    
