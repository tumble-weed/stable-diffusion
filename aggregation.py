from torch.nn.functional import fold, unfold
import torch
import debug
# from typing import Dict
from typing import TypedDict
class combine_patches_out(TypedDict):
    combined:torch.Tensor
    patch_aggregation:str

def combine_patches(O, patch_size, stride, img_shape,as_np = False,use_divisor=True,patch_aggregation=None,distances=None,weights=None,other = None,I=None) -> combine_patches_out:
    assert len(img_shape) == 3,'H,W,3'
    # patch_aggregation='uniform';print(f'hardcoding patch_aggregation to {patch_aggregation}')
    # channels = 3
    # assert patch_aggregation=='median'
    device = O.device
    channels = O.shape[1]
    O = O.permute(1, 0, 2, 3).unsqueeze(0)
    # O[:,0] = 1
    # O[:,1] = 2
    # O[:,2] = 3

    if patch_aggregation == 'uniform':
        # assert False,'shouldnt be here'
        
        patches = O.contiguous().view(O.shape[0], O.shape[1], O.shape[2], -1) \
        .permute(0, 1, 3, 2) \
        .contiguous().view(1, channels * patch_size[0] * patch_size[0], -1)
        combined = fold(patches, output_size=img_shape[:2], kernel_size=patch_size, stride=stride)
        # import pdb;pdb.set_trace()
        # normal fold matrix
        input_ones = torch.ones((1, img_shape[2], img_shape[0], img_shape[1]), dtype=O.dtype, device=device)
        # '''
        divisor = unfold(input_ones, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
        divisor = fold(divisor, output_size=img_shape[:2], kernel_size=patch_size, stride=stride)

        divisor[divisor == 0] = 1.0
        combined = (combined / divisor)
        # '''
        #================================================
        if as_np:
            combined = (combined).squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
        else:
            combined = (combined).squeeze(dim=0).permute(1, 2, 0)
            print(combined.max())
        
        return dict(combined=combined,patch_aggregation=patch_aggregation)        

    elif patch_aggregation == 'distance-weighted':
        import distance_aggregation
        return distance_aggregation.aggregate(O, patch_size, stride, img_shape,as_np = as_np,use_divisor=use_divisor,patch_aggregation=patch_aggregation,distances=distances,weights=weights,other = other,I=I)

        # import pdb;pdb.set_trace()
    elif patch_aggregation == 'median':
        from median_aggregation import embed_in_larger,get_median,overlap_form
        b,c,hw,ph,pw = O.shape
        patches = O.view(b,c,(img_shape[0] - 2*(patch_size[0]//2) + stride[0] - 1)//stride[0],(img_shape[1] - 2*(patch_size[1]//2) + stride[1] - 1)//stride[1],ph,pw)
        # patches_grey = patches.mean(dim=1,keepdim=True)
        embedded_patches = torch.cat([embed_in_larger(patches[:,[ci]],img_shape[:2],patch_size,stride=stride) for ci in range(c)],dim=1)
        combined,median_at,sort_order = get_median(embedded_patches.mean(dim=1,keepdim=True),img_shape[:2],patch_size)
        '''
        def aggregate_using_median(img_shape,patch_size,
                                     median_at,sort_order,
                                     patches = None,
                                     embedded_patches=None):
            print('TODO: move me to median_aggregation')
            if embedded_patches is None:
                assert patches is not None
                assert False
            I1,I2 = torch.meshgrid(torch.arange(img_shape[0],device=device),torch.arange(img_shape[1],device=device),indexing='ij')
            combined = torch.stack([overlap_form(embedded_patches[:,[ci]],img_shape[:2],patch_size,sort_order=sort_order)[0][I1,I2,median_at] for ci in range(c)],dim=0)
            return combined
        # from median_aggregation import aggregate_using_median
        '''
        
        I1,I2 = torch.meshgrid(torch.arange(img_shape[0],device=device),torch.arange(img_shape[1],device=device),indexing='ij')
        combined = torch.stack([overlap_form(embedded_patches[:,[ci]],img_shape[:2],patch_size,sort_order=sort_order)[0][I1,I2,median_at] for ci in range(c)],dim=0)
        combined = combined.unsqueeze(0)
        # import pdb;pdb.set_trace()
        weights = DUMMY_WEIGHTS = median_at
        out = (combined).squeeze(dim=0).permute(1, 2, 0)
        return dict(combined = out,median_at=median_at,sort_order=sort_order,patch_aggregation=patch_aggregation)
    '''
    if not use_divisor:
        assert False,'this might be wrong'
        divisor = torch.ones_like(divisor)
    if as_np:
        # return (combined / divisor).squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
        return (combined).squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
    else:
        # import pdb;pdb.set_trace()
        # out = (combined / divisor).squeeze(dim=0).permute(1, 2, 0)
        out = (combined).squeeze(dim=0).permute(1, 2, 0)
        print(out.max())
        return out,weights
    '''    
#******************************************************
from collections import defaultdict
def combine_patches_multiple_images(values, 
    patch_size, stride, img_shape,
    as_np=False,
    patch_aggregation=None,
    distances=None,I=None):
    assert len(img_shape) == 3,'H,W,3'
    print('TODO:this is untested')
    # import pdb;pdb.set_trace()
    patch_aggregation_results = defaultdict(list)
    
    for ii,(v) in enumerate(values):            
        # debug.I = I[ii*distances.shape[1]:(ii+1)*distances.shape[1]]
        print('TODO:debug.I should trigger error in distance-weighted aggregation')
        d = None
        if distances is not None:
            d = distances[ii]
        result = combine_patches(v, patch_size, stride, img_shape,as_np=as_np,
        patch_aggregation=patch_aggregation,
        distances=d,I=I)
        
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
    return patch_aggregation_results
#******************************************************