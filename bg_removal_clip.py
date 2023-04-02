import sys
import os
sys.path.append('clipseg')
mydir = os.path.dirname(os.path.abspath(__file__))
import torch
import requests
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import colorful
# a context manager to change the directory and change it back on exit
class ChangeDir():
    def __init__(self, new_path):
        self.new_path = new_path
        self.saved_path = os.getcwd()
    def __enter__(self):
        os.chdir(self.new_path)
    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)
    
if False and 'old weights':
    # load model
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval();
else:
    # new weights
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    model.load_state_dict(torch.load(os.path.join(mydir,'clipseg','weights/rd64-uni-refined.pth')), strict=False)

# non-strict, because we only stored decoder weights (not CLIP weights)
model.load_state_dict(torch.load(os.path.join(mydir,'clipseg','weights/rd64-uni.pth'), map_location=torch.device('cpu')), strict=False);

import skimage.transform
import numpy as np
def bgremove_clip(img_np,prompts,max_val=1):
    # img_np  = (600, 224, 3)
    # img_np range (0,1)
    # import ipdb;ipdb.set_trace()
    # make shorter side 352:
    original_shape = img_np.shape[:-1]
    # figure out new shape, where the shorter side is 352
    new_shape0 = (352, int(352*original_shape[1]/original_shape[0]))
    new_shape1 = (int(352*original_shape[0]/original_shape[1]), 352)
    new_shape = new_shape0 if np.prod(new_shape0) < np.prod(new_shape1) else new_shape1
    img_np_small = skimage.transform.resize(img_np, new_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
    if 'embed in larger image' and True:
        img_np_small2 = np.ones((352,352,3),dtype=np.float32)
        # paste into center
        # fix ValueError: could not broadcast input array from shape (352,131,3) into shape (352,130,3)
        margin = (352-img_np_small.shape[0])//2,(352-img_np_small.shape[1])//2
        img_np_small2[margin[0]:margin[0]+img_np_small.shape[0],margin[1]:margin[1]+img_np_small.shape[1],:] = img_np_small
    else:
        img_np_small2 = img_np_small
    img_small = torch.from_numpy(img_np_small2).unsqueeze(0).permute(0,3,1,2).float().contiguous()
    # normalize by the same mean and std as the training data
    img_small = (img_small - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    assert hasattr(prompts,'__len__') and not isinstance(prompts,str)
    n_prompts = len(prompts)
    assert n_prompts == 1
    preds_small = model(img_small.repeat(n_prompts,1,1,1), prompts)[0]
    import utils
    utils.cipdb('DBG_CLIPSEG')
    if preds_small.shape[-2:] != img_small.shape[-2:]:
        preds_small = torch.nn.functional.interpolate(preds_small, size=img_small.shape[-2:], mode='bilinear', align_corners=False)
    preds_small = torch.sigmoid(preds_small)
    preds_small_np = preds_small.detach().cpu().numpy()[0,0]
    preds_np = skimage.transform.resize(preds_small_np, original_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
    print(colorful.red("clipseg:scaling the clipseg mask by its max. This is a hack, in the hope that there is a valid object in the image"))
    preds_np = preds_np/preds_np.max()
    print(colorful.red("clipseg:hack to give binary masks"))
    preds_np = (preds_np > 0.5).astype(np.float32)
    preds_np = max_val * preds_np
    return preds_np
def test():
    with ChangeDir(os.path.join(mydir,'clipseg')):
        
        # load and normalize image
        input_image = Image.open('example_image.jpg')

        # or load from URL...
        # image_url = 'https://farm5.staticflickr.com/4141/4856248695_03475782dc_z.jpg'
        # input_image = Image.open(requests.get(image_url, stream=True).raw)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352)),
        ])
        img = transform(input_image).unsqueeze(0)



        prompts = ['a glass', 'something to fill', 'wood', 'a jar']

        # predict
        with torch.no_grad():
            # img.shape = (1, 3, 352, 352)
            # img.max() = 2.64
            # img.min() = -2.1179
            preds = model(img.repeat(4,1,1,1), prompts)[0]
            # preds.shape = (4, 1, 352, 352)
            # preds.max() > 0,  preds.min() < 0


        # visualize prediction
        import ipdb;ipdb.set_trace()
        _, ax = plt.subplots(1, 5, figsize=(15, 4))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(input_image)
        [ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(4)];
        [ax[i+1].text(0, -15, prompts[i]) for i in range(4)];

if __name__ == '__main__':
    test()