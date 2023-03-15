import torch
import math
def gradient_histogram(patch, k):
    # Calculate the gradient of the patch using PyTorch's built-in gradient function
    b,c = patch.shape[:2]
    patch_x = patch.clone()
    patch_y = patch.clone()
    patch_x[..., :, :-1] = patch[..., :, 1:] - patch[..., :, :-1]
    patch_y[..., :-1, :] = patch[..., 1:, :] - patch[..., :-1, :]
    # import pdb;pdb.set_trace()
    grad_magnitude = torch.sqrt(patch_x**2 + patch_y**2)
    grad_orientation = torch.atan2(patch_y, patch_x)
    # import pdb;pdb.set_trace()
    grad_orientation[grad_orientation<0] = 2*math.pi + grad_orientation[grad_orientation<0]

    # Bin the gradients by orientation
    histogram = torch.zeros(*grad_orientation.shape[:-2] + (k,))
    bin_size = 2*math.pi/k
    for i in range(k):
        # import pdb;pdb.set_trace()
        histogram[..., i] = torch.sum(torch.where(
            (grad_orientation >= i*bin_size) & (grad_orientation < (i+1)*bin_size),
            grad_magnitude ,
            torch.zeros_like(grad_magnitude)
        ), dim=(-2,-1))
    # 
    return histogram


'''
def circular_convolution(inputs, weights):
    # Get the batch size and input length
    batch_size = inputs.size(0)
    input_len = inputs.size(1)

    # Pad the inputs with k-1 zeros at the end
    inputs_padded = torch.cat((inputs, torch.zeros(batch_size, k-1, 3)), dim=1)

    # Perform the convolution
    outputs = torch.zeros(batch_size, input_len)
    for i in range(input_len):
        outputs[:, i] = torch.sum(inputs_padded[:, i:i+k] * weights, dim=(1, 2))

    return outputs
'''

'''
import torch.nn.functional as F
F.conv1d(F.pad(input, 
               pad=(kernel.shape[-2]//2,kernel.shape[-2]//2,kernel.shape[-1]//2,kernel.shape[-1]//2), 
               mode='circular'), kernel, padding=0) 
'''