import re
from glob import glob
import os
import numpy as np

DENSITY_CHANNELS_MAPPING = {
    'EPI': 2,
    'ENUC': 3,
    'STR': 4,
    'LUM': 5
}


class Generate2D:
    def __init__(self, density_type=None, dataset_dir='data/preproc_EESL', dataset_file='', train_ratio=.6):
        """

        :param density_type: a str or a list/tuple of density types (default: all keys of DENSITY_CHANNELS_MAPPING)
        :param dataset_file:
        :param train_ratio: the percentage of data for training
        """
        self.train_ratio = train_ratio
        if density_type is None:
            self.density_type = list(DENSITY_CHANNELS_MAPPING.keys())
        else:
            self.density_type = [density_type] if isinstance(density_type, str) else density_type
        self.dataset_file = os.path.join(dataset_dir, dataset_file)

    def gen_2d(self, dir_out, get_unsup_data=False):
        """"""
        print('Loading data...')
        supervised_list = [glob(os.path.join(self.dataset_file, '*.npy'))][0]
        case_id = []
        supervised_data = []
        density_channels = [DENSITY_CHANNELS_MAPPING[tis_type] for tis_type in self.density_type]
        for i, l in enumerate(supervised_list):
            supervised_data.append(self.extract_slices(np.load(l)))
            case_id.append(np.tile(i, (1, len(supervised_data[i]))))
        supervised_data = np.concatenate(supervised_data)
        supervised_data = supervised_data[..., [0, 1] + density_channels + [-2, -1]].transpose([1, 2, 0, 3])
        print(supervised_data.shape)

        case_id = np.concatenate(case_id, axis=1)[0]
        cut_idx = round(len(np.unique(case_id)) * self.train_ratio)
        case_id = case_id >= cut_idx
        # case_id = case_id > self.train_thr
        print('Saving data...')
        suffix = ''.join(s[0] for s in self.density_type)
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)
        np.save(f'{dir_out}/mri_density_%dchannels_%s.npy' % (supervised_data.shape[-1], suffix),
                supervised_data.astype('float32'))
        np.save(f'{dir_out}/caseID_by_time_0.60.npy', case_id)

    def print_slice_id(self, get_unsup_data=False):
        """"""
        supervised_list = [glob(os.path.join(self.dataset_file, '*.npy'))][0]
        count = 0
        for l in supervised_list:
            for k in self.find_density_slice(np.load(l))[0]:
                mr_id = re.findall('MRI\d+', l)[0]
                tis_id = re.findall('PATH\d+', l)[0]
                print(f'{count} {mr_id} {tis_id} {k}')
                count += 1

    @staticmethod
    def find_density_slice(x):
        return np.where(x[..., -1].sum(axis=(1, 2)))

    def extract_slices(self, x):
        """Extract slices having the density maps"""
        x = np.asarray([x[i] for i in self.find_density_slice(x)])[0]
        x[..., -1:] = (x[..., -1:] > 0).astype(x.dtype)  # Binarize the ROIs and WP masks
        return x


if __name__ == "__main__":
    generator = Generate2D(None)
    # generator = Generate2D('EPI')
    generator.gen_2d(dir_out=r'../inputs', get_unsup_data=False)
    # generator.print_slice_id()
