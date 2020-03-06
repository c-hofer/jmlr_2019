import torch
import numpy
import sklearn.cluster
import torch.nn as nn
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from collections import Counter, defaultdict


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
        
    splitter = splitter_type(n_splits=1, train_size=train_size, test_size=1-train_size, random_state=1234)
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
    
    k_means = {k: sklearn.cluster.KMeans(n_clusters=n_centers, init='k-means++', n_init=10, random_state=123)
               for k in points_by_view.keys()}
    
    center_inits_by_view = {}
    for k in points_by_view.keys():
        centers = k_means[k].fit(points_by_view[k]).cluster_centers_
        centers = torch.from_numpy(centers)
        center_inits_by_view[k] = centers
        
    return center_inits_by_view  

def min_max_random_init(sample_target_iter: dict, n_centers: int):
    samples_by_view, _ = dict_sample_target_iter_concat(sample_target_iter)
    
    points_by_view = {}
    for k, v in samples_by_view.items():
        points_by_view[k] = torch.cat(v, dim=0).numpy()
        
    center_inits_by_view = {}
    for k, points in points_by_view.items():
        x = points[:, 0]
        x_max = x.max()
        x_min = x.min()
        x_init = torch.zeros(n_centers).uniform_(float(x_min), float(x_max))
        
        y = points[:, 1]
        y_max = y.max()
        y_min = y.min()
        y_init = torch.zeros(n_centers).uniform_(float(y_min), float(y_max))
        
        c = torch.stack([x_init, y_init], dim=1)
        center_inits_by_view[k] = c
        
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


def dict_sample_target_iter_concat(sample_target_iter: iter):
    """
    Gets an sample target iterator of dict samples. Returns
    a concatenation of the samples based on each key and the
    target list.

    Example:
    ```
    sample_target_iter = [({'a': 'a1', 'b': 'b1'}, 0), ({'a': 'a2', 'b': 'b2'}, 1)]
    x = dict_sample_iter_concat([({'a': 'a1', 'b': 'b1'}, 0), ({'a': 'a2', 'b': 'b2'}, 1)])
    print(x)
    ({'a': ['a1', 'a2'], 'b': ['b1', 'b2']}, [0, 1])
    ```

    :param sample_target_iter:
    :return:
    """

    samples = defaultdict(list)
    targets = []

    for sample_dict, y in sample_target_iter:
        for k, v in sample_dict.items():
            samples[k].append(v)

        targets.append(y)

    samples = dict(samples)

    length = len(samples[next(iter(samples))])
    assert all(len(samples[k]) == length for k in samples)

    return samples, targets


def collection_cascade(input, stop_predicate: callable, function_to_apply: callable):
    if stop_predicate(input):
        return function_to_apply(input)
    elif isinstance(input, list or tuple):
        return [collection_cascade(x,
                                   stop_predicate=stop_predicate,
                                   function_to_apply=function_to_apply) for x in input]
    elif isinstance(input, dict):
        return {k: collection_cascade(v,
                                      stop_predicate=stop_predicate,
                                      function_to_apply=function_to_apply) for k, v in input.items()}
    else:
        raise ValueError('Unknown type collection type. Expected list, tuple, dict but got {}'
                         .format(type(input)))


def cuda_cascade(input, **kwargs):
    return collection_cascade(input,
                              stop_predicate=lambda x: isinstance(x, torch._TensorBase),
                              function_to_apply=lambda x: x.cuda(**kwargs))


def histogram_intersection_loss(input: torch.Tensor,
                                target: torch.Tensor,
                                size_average: bool = True,
                                reduce: bool = True,
                                symetric_version: bool = True) -> torch.Tensor:
    r"""
    This loss function is based on the `Histogram Intersection` score. 
    The output is the *negative* Histogram Intersection Score.
    Args:
        input (Tensor): :math:`(N, B)` where `N = batch size` and `B = number of classes`
        target (Tensor): :math:`(N, B)` where `N = batch size` and `B = number of classes`
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                :attr:`size_average` is set to ``False``, the losses are instead summed
                for each minibatch. Ignored if :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional):
        symetric_version (bool, optional): By default, the symetric version of histogram intersection
                is used. If false the asymetric version is used. Default: ``True``
    Returns: Tensor.
    """
    assert input.size() == target.size(), \
        "input.size() != target.size(): {} != {}!".format(input.size(), target.size())
    assert input.dim() == target.dim() == 2, \
        "input, target must be 2 dimensional. Got dim {} resp. {}".format(
            input.dim(), target.dim())

    minima = input.min(target)
    summed_minima = minima.sum(dim=1)

    if symetric_version:
        normalization_factor = (input.sum(dim=1)).max(target.sum(dim=1))
    else:
        normalization_factor = target.sum(dim=1)

    loss = summed_minima / normalization_factor

    if reduce:
        loss = sum(loss)

        if size_average:
            loss = loss / input.size(0)

    return -loss
