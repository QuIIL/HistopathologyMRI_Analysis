from mxnet.gluon.data.dataset import Dataset


class RadPath(Dataset):
    def __init__(self, *data, transform=None, is_val=False, input_size,
                 not_augment_values=False, density_range=None, margin=0,
                 batch_size=None,
                 batch_size_unsup=None,
                 ):
        self._transform = transform
        self._data = data
        self._is_val = is_val
        self._input_size = input_size
        self._not_augment_values = not_augment_values
        self._density_range = density_range
        self._margin = margin
        self._batch_size = batch_size
        self._batch_size_unsup = batch_size_unsup
        self._count_instance = 0
        self._current_it = 0

    def __len__(self):
        return len(self._data[0])

    def __getitem__(self, idx):
        if self._transform is not None:
            get_unsup = False
            data = self._data[:5]
            if idx < 0:
                get_unsup = True
                idx = -idx - 1
                data = self._data[5:]

            aug_data = self._transform(
                tuple([d[idx] for d in data]),
                is_val=self._is_val,
                input_size=self._input_size,
                not_augment_values=self._not_augment_values,
                density_range=self._density_range,
                margin=self._margin,
                get_unsup=get_unsup,
                current_it=self._current_it,
            )
            return tuple([d for d in aug_data])
        else:
            return tuple([d[idx] for d in self._data])


class RadPathV1(Dataset):
    def __init__(self, *data, transform=None, is_val=False, input_size,
                 not_augment_values=False, density_range=None, margin=0,
                 batch_size=None,
                 batch_size_unsup=None,
                 ):
        self._transform = transform
        self._data = data
        self._is_val = is_val
        self._input_size = input_size
        self._not_augment_values = not_augment_values
        self._density_range = density_range
        self._margin = margin
        self._batch_size = batch_size
        self._batch_size_unsup = batch_size_unsup
        self._count_instance = 0
        self._current_it = 0

    def __len__(self):
        return len(self._data[0])

    def __getitem__(self, idx):
        if self._transform is not None:
            get_unsup = False
            data = self._data[:4]
            if idx < 0:
                get_unsup = True
                idx = -idx - 1
                data = self._data[4:]

            aug_data = self._transform(
                tuple([d[idx] for d in data]),
                is_val=self._is_val,
                input_size=self._input_size,
                not_augment_values=self._not_augment_values,
                density_range=self._density_range,
                margin=self._margin,
                get_unsup=get_unsup,
                current_it=self._current_it,
            )
            return tuple([d for d in aug_data])
        else:
            return tuple([d[idx] for d in self._data])
