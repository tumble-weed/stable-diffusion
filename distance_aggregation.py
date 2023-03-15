import torch
from torch.nn.functional import fold, unfold
import debug
def get_weights_from_distance(unfolded_patches,patch_size,distances=None):
    assert distances is not None
    assert distances.shape[-2:] == (1,1)
    distances = distances.permute(1, 0, 2, 3).unsqueeze(0)
    distances = distances * torch.ones_like(unfolded_patches[:,:1])
    '''
    # print('is this correct? distances = unfolded_patches[:,:1]*distances')
    # distances = unfolded_patches[:,:1]*distances
    '''
    # 1,3,64,7,7
    '''
    patches = unfolded_patches.contiguous().view(unfolded_patches.shape[0], unfolded_patches.shape[1], unfolded_patches.shape[2], -1) \
        .permute(0, 1, 3, 2) \
        .contiguous().view(1, channels * patch_size[0] * patch_size[0], -1)
    '''
    distances = distances.contiguous().view(distances.shape[0], distances.shape[1], distances.shape[2], -1) \
        .permute(0, 1, 3, 2) \
        .contiguous().view(1, 1 * patch_size[0] * patch_size[0], -1)
    if False:
        weights = torch.exp(-distances/(1e-4 + 10*distances.mean(dim=1,keepdim=True)))
    else:
        # import pdb;pdb.set_trace()
        print('dividing by distances.std for weights')
        sigma = distances.std()
        # if sigma > 0:
        #     print('is sigma ever > 0')
        #     import pdb;pdb.set_trace()
        distance_min = distances.min()
        distance_min = distance_min  + (distance_min == 0).float()
        sigma = sigma + (sigma==0.).float() * distance_min
        weights = torch.exp(-(distances/sigma))
    return weights
def normalize_weights_(weights,img_shape,patch_size,stride):
    # fold the weights to get the aggregate weights
    '''
    approach:
    - get first divisor to see if all weights are 0 at a location
    - make those weights non zero to a fixed value like 1e-8
    - recompute the divisor with weights changed
    '''
    divisor = fold(weights, output_size=img_shape[:2], 
                    kernel_size=patch_size, stride=stride)                
    divisor = unfold(divisor, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
    assert not divisor.isnan().any()
    weights[divisor==0]  = 1e-8
    divisor = fold(weights, output_size=img_shape[:2], 
                    kernel_size=patch_size, stride=stride)                
    divisor = unfold(divisor, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
    assert not divisor.isnan().any() or (divisor == 0).any()
    weights = weights/divisor
    if True and 'just checking correctness':                    
        weights_f = fold(weights, output_size=img_shape[:2], kernel_size=patch_size, stride=stride)
        # weights_u = unfold(weights, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
        assert torch.allclose(weights_f,torch.ones_like(weights_f))
    return weights                
def weighted_fold(unfolded,weights,img_shape,stride):
    assert unfolded.ndim == 5
    channels = unfolded.shape[1]
    patch_size = unfolded.shape[-2:]
    patches = unfolded.contiguous().view(unfolded.shape[0], unfolded.shape[1], unfolded.shape[2], -1) \
        .permute(0, 1, 3, 2) \
        .contiguous().view(1, channels * patch_size[0] * patch_size[0], -1)                    
    weighted_patches = torch.tile(weights,(1,channels,1)) * patches
    # 1,49,64
    combined = fold(weighted_patches, output_size=img_shape[:2], kernel_size=patch_size, stride=stride)
    assert not combined.isnan().any()
    return combined

def aggregate(O, patch_size, stride, img_shape,as_np = False,use_divisor=True,patch_aggregation=None,distances=None,weights=None,other=None,I=None):
    device = O.device
    channels = O.shape[1]
    assert channels in [5,4,3,1]
    if weights is None:
        #************************************************************
        weights = get_weights_from_distance((O),
                                  patch_size,
                                  distances=(distances))
        #************************************************************
        if False:
            from termcolor import colored 
            print(colored('using other to add an extra weight in distance_aggregation','yellow'))
            other = debug.cam0;print('setting other to debug.cam0')
            CAM0HACK = False
            if True and other is not None:
                if weights.shape[1:] == other.shape[1:]:
                    # CAM0HACK = True
                    # import pdb;pdb.set_trace()
                    assert I is not None
                    other = torch.index_select(other,-1,debug.I.squeeze())
                    # weights = other;print('setting weights to "other"')
                    weights = (weights + other)
            # weights = torch.exp(-(distances/100))
        weights = normalize_weights_((weights),img_shape,patch_size,stride)

    else:
        # import pdb;pdb.set_trace()
        if debug.__dict__.get('stop_at_combine',False):
            import pdb;pdb.set_trace()

    combined = weighted_fold( O,weights.detach(),img_shape,stride)
    if as_np:
        combined = (combined).squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
    else:
        combined = (combined).squeeze(dim=0).permute(1, 2, 0)
        print(combined.max())
    return dict(combined=combined,weights=weights,patch_aggregation=patch_aggregation)                