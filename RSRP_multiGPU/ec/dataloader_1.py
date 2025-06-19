import jax
import jax.numpy as jnp
import numpy as np

from torch.utils.data import DataLoader, Subset
from torch.utils.data import SequentialSampler, BatchSampler,RandomSampler
from torch import Tensor as TorchTensor

from ec import EvoConfig

from functools import partial

class NumpyDataLoader(DataLoader):

    """
    Adopted from Torch Dataloader, with numpy output. Designed to run on CPU only.
    Data is batched in shape (Local_device_count, Sub_Pop_num, Batch_Size, ..).
    Drop_last = True, to make sure each batch is full.
    TODO: Warning: Multi-process behavoir UNDEFINED.
    """

    def __init__(self, dataset, evo_conf: EvoConfig, *args, **kwargs):
        super().__init__(dataset, 
                         batch_size=evo_conf.pop_batch_size, 
                         drop_last=True, # Make sure each batch is full
                         *args, **kwargs)
        
        if self.batch_size > len(dataset):
            raise Exception(f"Batch size {self.batch_size} > Dataset size {len(dataset)}")

        self.dataiter = iter(self)
        self.batch_shape = evo_conf.pop_batch_shape
        self.data_shape = [np.array(i).shape for i in dataset[0]]

    def __next__(self):
        try:
            batch = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self)  # reset the iterator
            batch = next(self.dataiter)
        
        finally:
            def _body(x):
                x = x.numpy()
                shape = x.shape
                x = x.reshape((*self.batch_shape, *shape[1:]))
                return x
            
            def _is_leaf(x):
                return isinstance(x, TorchTensor)

            return jax.tree_map(_body, batch.data, is_leaf = _is_leaf)

    def get_batch(self):
        return self.__next__()


class PreAllocatedDataLoader(DataLoader):

    """
    Fully load a PyTorch dataset, and distribute it into different devices.
    Each device only have a subset of dataset. Mini-batches are generated per device.
    When you have a smaller dataset, please use NumpyDataLoader.
    TODO: Only supports dataset that organized in lists. Should add support for every type of PyTrees
    """

    def __init__(self, dataset, evo_conf: EvoConfig, train: bool, *args, **kwargs) -> None:
        
        ps_idx = jax.process_index()
        self.data_per_process = len(dataset) // jax.process_count() # num of data for each device
        self.data_per_device = len(dataset) // jax.device_count() # num of data for this process

        l_idx = self._data_idx(ps_idx, self.data_per_process)
        r_idx = self._data_idx(ps_idx+1, self.data_per_process)

        subset = Subset(dataset, range(l_idx, r_idx))
        super().__init__(subset,
                         shuffle=True, 
                         batch_size=len(subset), 
                        #  num_workers = 8,
                         *args, **kwargs)
        torch_data = next(iter(self))
        np_data = []
        for i in torch_data:
            x = i.numpy()
            x = x.reshape((jax.local_device_count(), self.data_per_device, *x.shape[1:]))
            np_data.append(x)
        
        self.alloc_data = jax.pmap(lambda x:x) (np_data)
        self.batchs_size = evo_conf.batch_size
        self.subpop_size = evo_conf.subpop_size
        self.data_shape = [np.array(i).shape for i in subset[0]]

        self.train = train
        self.seq_sampler = list(BatchSampler(RandomSampler(range((self.data_per_device))), batch_size=evo_conf.batch_size, drop_last=True))
        self.iterator = iter(SequentialSampler(self.seq_sampler))


    def _data_idx(self, p_idx:int, share:int) -> int:

        return p_idx * share
    
    def reset(self):
        self.seq_sampler = list(BatchSampler(RandomSampler(range((self.data_per_device))), batch_size=self.batchs_size, drop_last=True))
        self.iterator = iter(SequentialSampler(self.seq_sampler))
    
    def next_indexs(self):
        x = next(self.iterator, 'end')
        if x == 'end':
            self.reset()
            x = next(self.iterator, 'end')
        return self.seq_sampler[x]
    
    def __next__(self):
        if self.train == True:
            choices = self.next_indexs()
            batch = _sample_batch(self.alloc_data, self.subpop_size, choices)
        else:
            batch = _sample_batch_test(self.alloc_data, self.batchs_size)
        return batch

    def get_batch(self):
        return self.__next__()


@partial(jax.pmap, in_axes = (0, None, None,), static_broadcasted_argnums = (1, ))
def _sample_batch(data, subpop_size, choices):
    choices = jnp.array(choices)
    def _body(x):
        x = x[choices]
        x = jnp.expand_dims(x, axis=0).repeat(subpop_size, axis=0)
        return x

    batch = jax.tree_map(_body, data)

    return batch

@partial(jax.pmap, in_axes = (0, None,), static_broadcasted_argnums = (1, ))
def _sample_batch_test(data, batch_size):
    def _body(x):
        x = x[0:batch_size]
        return x

    batch = jax.tree_map(_body, data)

    return batch