import torch
import numpy
import sklearn.cluster
import torch.nn as nn
from chofer_torchex.utils.functional import collection_cascade
from chofer_torchex.utils.data.collate import dict_sample_target_iter_concat
from chofer_torchex.nn.slayer import SLayer
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from collections import Counter


def numpy_to_torch_cascade(input):
    def numpy_to_torch(array):
        return_value = None
        try:
            return_value = torch.from_numpy(array)
        except Exception as ex:
            if len(array) == 0:
                return_value = torch.Tensor()
            else:
                raise ex

        return return_value.float()

    return collection_cascade(input,
                              stop_predicate=lambda x: isinstance(x, numpy.ndarray),
                              function_to_apply=numpy_to_torch)


def bar_code_slayer_collate_fn(sample_target_iter, cuda=False):
    x, y = dict_sample_target_iter_concat(sample_target_iter)
    x = collection_cascade(x, stop_predicate=lambda x: isinstance(x, list),
                           function_to_apply=lambda x: SLayer.prepare_batch(x, 2))

    y = torch.LongTensor(y)
    if cuda:
        # Shifting the necessary parts of the prepared batch to the cuda
        x = {k: collection_cascade(v,
                                   lambda x: isinstance(x, tuple),
                                   lambda x: (x[0].cuda(), x[1].cuda(), x[2], x[3]))
             for k, v in x.items()}

        y = y.cuda()

    return x, y


def get_train_test_sampler(dataset, train_size, stratified=False):
    
    splitter_type = ShuffleSplit if not stratified else StratifiedShuffleSplit
        
    splitter = splitter_type(n_splits=1, train_size=train_size, test_size=1-train_size)
    split = list(splitter.split(X=dataset.labels, y=dataset.labels))[0]
    split = [x.tolist() for x in split]
    
    train_sampler, test_sampler = split
    print("Generated training and testing split:")
    print("Train:", Counter([dataset[i][1] for i in train_sampler]))
    print("Test:", Counter([dataset[i][1] for i in test_sampler]))
    
    return train_sampler, test_sampler


def k_means_center_init(sample_target_iter: dict, n_centers: int):
    samples_by_view, _ = dict_sample_target_iter_concat(sample_target_iter)
    
    points_by_view = {}
    for k, v in samples_by_view.items():
        points_by_view[k] = torch.cat(v, dim=0).numpy()
    
    k_means = {k: sklearn.cluster.KMeans(n_clusters=n_centers, init='k-means++', n_init=10)
               for k in points_by_view.keys()}
    
    center_inits_by_view = {}
    for k in points_by_view.keys():
        centers = k_means[k].fit(points_by_view[k]).cluster_centers_
        centers = torch.from_numpy(centers)
        center_inits_by_view[k] = centers
        
    return center_inits_by_view  


def adapt_lr(optimizer, changer):
    for para_group in optimizer.param_groups:
        para_group['lr'] = changer(para_group['lr'])
        
        
class ModuleDict(nn.Module):
    def __init__(self):
        super().__init__()
        
    def __setitem__(self, key, item):
        setattr(self, key, item)
        
    def __getitem__(self, key):
        return getattr(self, key)
    
    
class UnitSGD(torch.optim.SGD):
    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            loss = closure()
        
        norm = 0
        for group in self.param_groups:
                for p in group['params']:
                    norm += float(p.grad.norm(1))
                    
        for group in self.param_groups:
            for p in group['params']:
                p.grad.data = p.grad.data/norm                 
        
        super(UnitSGD, self).step()  
        
        return loss
