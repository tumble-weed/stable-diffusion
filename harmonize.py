import os
import sys

#================================================================
mydir = os.path.dirname(os.path.abspath(__file__))
harmonizerdir = os.path.join(mydir,'Harmonizer')
sys.path.append(harmonizerdir)
#================================================================
import torch
from src import model
import numpy as np
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
global harmonizer
def init(device):
    if 'harmonizer' not in globals():
        global harmonizer
        pretrained_harmonizer = "Harmonizer/pretrained/harmonizer.pth"
        harmonizer = model.Harmonizer()
        harmonizer.to(device)
        harmonizer.load_state_dict(torch.load(pretrained_harmonizer), strict=True)
        harmonizer.eval()

def harmonize(pasted_np,
pasted_mask_np,device=None):
    init(device)
    global harmonizer
    pasted = torch.tensor(pasted_np).float().unsqueeze(0).permute(0,3,1,2).to(device)
    pasted_mask = torch.tensor(pasted_mask_np).float().unsqueeze(0).permute(0,3,1,2).to(device)
    arguments = harmonizer.predict_arguments(pasted, pasted_mask)
    harmonized = harmonizer.restore_image(pasted, pasted_mask, arguments)[-1]
    harmonized_np = tensor_to_numpy(harmonized.permute(0, 2, 3, 1))[0]
    return harmonized_np
def test():
    pasted_np = np.random.uniform(size=(256, 256, 3))
    pasted_mask_np = np.zeros((256, 256, 1))
    harmonized_np = harmonize(pasted_np,
pasted_mask_np,device=None)
    pass
if __name__ == '__main__':
    test()