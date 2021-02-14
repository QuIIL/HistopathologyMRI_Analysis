import numpy as np
from mxnet import nd, context


class BatchifyFn:
    """"""

    def __init__(self, batch_size):
        self._batch_size = batch_size

    def batchify_fn(self, data):
        """Collate data into batch. Split into supervised and unsupervised data accordingly if necessary"""
        if isinstance(data[0], nd.NDArray):
            out = nd.empty((len(data),) + data[0].shape, dtype=data[0].dtype,
                           ctx=context.Context('cpu_shared', 0))
            return nd.stack(*data, out=out)
        elif isinstance(data[0], tuple):
            if data.__len__() > self._batch_size:
                len_list = np.unique([len(d) for d in data])
                sup_len, unsup_len = np.max(len_list), np.min(len_list)
                data_sup = zip(*[_data for _data in data if len(_data) == sup_len])
                data_unsup = zip(*[_data for _data in data if len(_data) == unsup_len])
                return (
                    [self.batchify_fn(i) for i in data_sup],
                    [self.batchify_fn(i) for i in data_unsup]
                )
            else:
                data = zip(*data)
                return [self.batchify_fn(i) for i in data]
        else:
            if isinstance(data[0][0], np.ndarray):
                data = np.asarray(data).astype('float32')
                return nd.array(data, dtype=data.dtype,
                                ctx=context.Context('cpu_shared', 0))
            return data  # augmentation sequences

