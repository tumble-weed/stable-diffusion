from collections import defaultdict
from matplotlib import pyplot as plt
import torch
TODO = None
def combine_ift_grad(self,g):
    raise NotImplementedError
    return g
def run(f,trends,grad_hook):
    print('TODO: set up the initial optimizer,x etc')
    print('TODO: add all the input arguments')
    print('TODO: define the ift_grad function')
    print('TODO: rederive the ift grad to see if it is correct')
    print('TODO: start from the same x_')
    nthresh=TODO
    ndim = 2
    device = 'cuda'
    nepochs = 10
    x = torch.randn(ndim).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([x],lr=0.1)
    print('STAGE:setup done')
    for i in range(nepochs):
        print('copy this from tmp_saliency4.py for finding the nearest thresh')
        thresh = TODO
        thresh = [0,1]
        print('DEBUG:hardcoding the thresh')
        x_tiled = torch.tile(x,[nthresh,1])
        import pdb;pdb.set_trace()

        x_tiled.register_hook(grad_hook)
        mask = torch.sigmoid(x_tiled-thresh)
        out = f(mask)
        loss = - out.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trends['loss'].append(loss.item())
        print('TODO: track cchange in thresh etc in trends')
        trends['']
if __name__ == '__main__':
    f = TODO
    trends_ift=defaultdict(list)
    trends_simple=defaultdict(list)
    run(f,trends_ift)
    run(f,trends_simple)

    if 'plot':
        plt.figure()
        
        plt.plot(trends_ift['loss'],label='with ift thresh')
        plt.plot(trends_simple['loss'],label='without ift thresh')
        plt.legend()
        plt.show()