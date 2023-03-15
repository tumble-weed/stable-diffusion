TODO = None
import faiss
import torch
from termcolor import colored
faiss_assets = {}
class nn_debug:
    pass
dummy = 'hi'
def initialize_faiss(faiss_assets=faiss_assets):
    # global faiss, res
    if 'res' not in faiss_assets:
        res = faiss.StandardGpuResources()
        faiss_assets['res'] = res
        
def create_index(keys,index_type='ivf',index_options={'nlist':200},keys_to_keep=None):  
    #=====================================
    nn_debug.keys = keys
    nn_debug.keys_to_keep = keys_to_keep
    # import ipdb;ipdb.set_trace()
    #=====================================
    # import ipdb;ipdb.set_trace()  
    # self = TODO
    # self.index = TODO
    initialize_faiss()
    device = keys.device
    if keys_to_keep is None:
        keys_to_keep = torch.arange(keys.shape[0],device=device)
    assert keys[keys_to_keep].shape[0] > 0
    if index_type == 'simple':
        index = faiss.IndexFlatL2(keys.shape[-1])
    elif index_type == 'ivf':
        print('using ivf')
        # nlist = min(index_options['nlist'],keys.shape[0])
        print('setting nlist to a smaller number of keys_to_keep')
        nlist = min(index_options['nlist'],len(keys_to_keep)//10)
        if  nlist < index_options['nlist']:
            print('# of keys < nlist')
        nn_debug.nlist = nlist
        print(keys.shape)
        quantizer = faiss.IndexFlatL2(keys.shape[-1])  # the other index
        
        index = faiss.IndexIVFFlat(quantizer, keys.shape[-1], nlist)
        index.nprobe = min(3,len(keys_to_keep))

        '''
        if self.KEYS_TYPE == 'single-resolution':
            quantizer = faiss.IndexFlatL2(keys.shape[-1])  # the other index
            # import pdb;pdb.set_trace()
            self.index = faiss.IndexIVFFlat(quantizer, keys.shape[-1], nlist)
            self.index.nprobe = 3
        elif self.KEYS_TYPE == 'multi-resolution':
            print('using running keys')
            quantizer = faiss.IndexFlatL2(self.running_keys.shape[-1])  # the other index
            self.index = faiss.IndexIVFFlat(quantizer, self.running_keys.shape[-1], nlist)
            self.index.nprobe = 1
        '''
    elif index_type == 'brute-force':
        class BruteForceIndex():
            def __init__(self,):
                pass
            def search(self,queries,k):
                assert k == 1, 'not implemented for k>1'
                assert queries.ndim ==2
                d = torch.cdist(queries[None,...],self.keys[None,...])
                D,I = d.min(dim =-1)
                D,I = D.squeeze(0).unsqueeze(-1),I.squeeze(0).unsqueeze(-1)
                D = D**2 # for consistency with faiss
                # print(colored('D might be unsquared distance unlike faiss index','yellow'))
                # import ipdb;ipdb.set_trace()
                return D,I
            def add(self,keys):
                self.keys = keys
                assert self.keys.ndim ==2   
                pass
        index =  BruteForceIndex()
    print('created index')
    # import ipdb;ipdb.set_trace()
    if not 'cuda' in str(device):
        print('nearest neighbors:CUDA not found in device')
    if 'cuda' in str(device) and index_type != 'brute-force':
        index = faiss.index_cpu_to_gpu(faiss_assets['res'], 0, index)
    print('pushed index to gpu')
    if index_type == 'ivf':
        index.train(keys[keys_to_keep])
        print('TODO: keys_to_keep as long or int?')
        index.add_with_ids(keys[keys_to_keep],torch.tensor(keys_to_keep).long().to(device))
    else:
        # print('!!add_with_ids not implemented fo simple indexes')
        index.add(keys[keys_to_keep])
    if False:
        D,I = index.search(keys[keys_to_keep],1)
        if (I.min() < 0) or (I.max() > len(keys_to_keep)):
            import ipdb;ipdb.set_trace()
    index.keys_to_keep = keys_to_keep
    index.index_type = index_type
    # import ipdb;ipdb.set_trace()
    return index
def get_nearest_neighbors(queries,keys=None,index=None,Ddtype='float',Idtype='int'):
    if index is None:
        # print('TODO:creating hardcoded index')
        assert keys is not None
        index = create_index(keys,index_type='ivf',index_options={'nlist':200})
    D, I = index.search(queries, 1)
    where_bad = I.squeeze(-1) < 0
    if where_bad.sum() > 0:
        nprobe_old = index.nprobe
        index.nprobe = 100
        D_,I_ = index.search(queries[where_bad], 1)
        D[where_bad] = D_
        I[where_bad] = I_
        index.nprobe = nprobe_old
    if I.min() < 0:
        import ipdb;ipdb.set_trace()
    if index.index_type in ['simple','brute-force']:
        print('!!add_with_ids not implemented fo simple indexes')
        I = index.keys_to_keep[I]
    #===============================================================
    if False:
        print('understand subset indexed search')
        import numpy as np
        sel = faiss.IDSelectorArray(10,np.arange(10).astype('int64'))
        reduced_search_params = faiss.SearchParametersIVF(
            sel = faiss.IDSelectorArray(range(10))
        )
        D2,I2 = index.search(queries,1,params=reduced_search_params)
        import pdb;pdb.set_trace()
    #===============================================================
    if Ddtype == 'half':
        D = D.half()
    elif Idtype == 'int':
        I = I.int()
    out = dict(
        D=D,
        I=I,
        index=index,
    )
    return out
torch_dtypes = dict(
    half = torch.float16,
    int = torch.int,
    float = torch.float32,
    long = torch.long,
)
def get_nearest_neighbors_batched(queries,keys=None,index=None,Ddtype='float',Idtype='int',max_batch_size=62496):
    device = queries.device

    # I = I.long()
    # if True and keys.shape[0] < 2147483647:
    #     I =I.int()
    # else:
    #     I =I.long()

    nbatches = (queries.shape[0]+max_batch_size - 1)//max_batch_size
    for i in range(nbatches):
        queriesi = queries[i*max_batch_size:(i+1)*max_batch_size]
        outi = get_nearest_neighbors(queriesi,keys,index=index,Ddtype=Ddtype,Idtype=Idtype)
        Di,Ii = outi['D'],outi['I']
        if i == 0:
            if Idtype == 'int':
                index = outi['index']
                # assert keys.shape[0] < 2147483647,'int cant encode large indexes'
                assert index.quantizer.ntotal < 2147483647,'int cant encode large indexes'
                Idtype = 'long'
            
            if Ddtype:
                'half'
            D = torch.zeros(queries.shape[0],1,device=device,dtype=torch_dtypes[Ddtype])
            I = torch.zeros(queries.shape[0],1,device=device,dtype=torch_dtypes[Idtype])            
        # Di, Ii = self.index.search(queries_proji, 1)
        D[i*max_batch_size:(i)*max_batch_size + queriesi.shape[0]] = Di
        I[i*max_batch_size:(i)*max_batch_size + queriesi.shape[0]] = Ii
        # import pdb;pdb.set_trace()    
    # index = outi['index']
    out = dict(
        D = D,
        I = I,
        index = outi['index'],
    )
    return out
def get_nearest_neighbors_of_subset(queries,query_mask,keys=None,index=None,Ddtype='float',Idtype='int',max_batch_size=62496,D=None,I=None):
    # assert False,'make keys default to None'
    queries_subset = queries
    if query_mask is not None:
        assert not any([D is None,I is None])
        queries_subset =  queries[query_mask]
    out_subset = get_nearest_neighbors_batched(queries_subset,keys,index=index,Ddtype=Ddtype,Idtype=Idtype,max_batch_size=max_batch_size)
    if query_mask is not None:
        D[query_mask] = out_subset['D']
        I[query_mask] = out_subset['I']
    else:
        D,I = out_subset['D'],out_subset['I']
    out =  dict(
        D = D,
        I = I,
        index = out_subset['index'],
    )
    return out