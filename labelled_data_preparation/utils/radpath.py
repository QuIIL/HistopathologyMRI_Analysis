from scipy.io import loadmat
import numpy as np
import pylab as plt
from skimage.util import montage
from os.path import join as join_pth
import os
from glob import glob
import pydicom
import re
from matplotlib.path import Path
import xml.etree.ElementTree as ET


class ROIOnSlice:
    """

    """

    def __init__(self):
        """"""
        self.slice_idx, self.mask, self.pth, self.roi_type = (None,) * 4


class RadPath:
    """3D RadPath Class"""
    path_pth = [
        r'F:\Workspace\RadPath_1\densityestimate',
        r'F:\Workspace\RadPath_2',
    ]
    roi_pth = r'F:\Workspace\RadPath_ROIs'
    density_ROIs_mapping = {
        'cg-left': 1, 'cg-right': 1, 'pz-left': 2, 'pz-right': 2, 'pz-right2': 2,
        'cg-tumor': 4, 'pz-tumor': 4, 'cg-stroma': 3,  # 'tumor': 6
    }

    path_suffice = '-mr-cell-density04.mat'
    # tissue_types = ('EPI', 'NUC', 'STR', 'LUM')
    tissue_types = ('EPI', 'ENUC', 'STR', 'LUM')

    def __init__(self, mr_id=None, mr_pth=None, tis_id=None, load_data=True, _load_density=True):
        self.crop_loc = None
        self.mr_id = mr_id
        self.mr_path = mr_pth
        self.density = {}
        self.tis_id = list(tis_id.keys())[0] if tis_id is not None else None  # may or may not have tissue data
        self.slices_idx = list(tis_id.values())[0] if tis_id is not None else None  # may or may not have tissue data
        self.T2, self.ADC, self.B2000, self.WP = (None,) * 4
        self.d, self.h, self.w = (None,) * 3
        self.COM = None
        self.seq = []
        self.roi_meta = {}
        self.density_ROIs = None

        if load_data:
            self.load_mr()
            self.find_com()
            self.find_com_per_slice()
            if _load_density:
                if tis_id is not None:
                    self.load_density(self.tissue_types)

    @staticmethod
    def threshold_img(x, mask):
        x_max, x_min = x[mask > 0].max(), x[mask > 0].min()
        x[x > x_max] = x_max
        return x

    @staticmethod
    def norm_01(x, mask=None, mask_input=False):
        x = x.astype('float32')
        mask = np.ones_like(x) if mask is None else mask
        x_max, x_min = x[mask > 0].max(), x[mask > 0].min()
        x[x > x_max] = x_max
        if mask_input:
            return mask * (x - x_min) / (x_max - x_min)
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def get_wp_contained_slices(mask):
        return np.where(mask.sum(axis=(tuple(range(1, np.ndim(mask))))) > 0)

    def rm_non_prostate(self, x, mask):
        """
        :param mask: binary array
        :param x: target array
        :return:
        """
        return x[self.get_wp_contained_slices(mask)]

    def pad_non_prostate(self, x, mask):
        """
        Pad the volume with zeros slices to match with the original image
        :param mask:
        :param x:
        :return:
        """
        z = np.zeros(((self.d,) + x.shape[1:]), dtype='float32')
        z[self.get_wp_contained_slices(mask)] = x
        return z

    def find_com(self):
        """Compute the center of mass (COM)"""
        mip = self.WP.sum(axis=0) > 0
        self.COM = np.round(np.mean(np.where(mip), axis=1)).astype(int)

    def find_com_per_slice(self):
        """Compute the center of mass (COM) for each slice separately"""
        pseudo_WP = [x if x.sum() > 0 else np.ones_like(x) for x in self.WP]
        self.COM_per_slice = [np.round(np.mean(np.where(x), axis=1)).astype(int) for x in pseudo_WP]

    def com_crop(self, x, crop_size=256):
        """Crop at the center of mass (of the mask)"""
        self.crop_loc = self.COM - int(crop_size / 2) + 1
        return x[..., self.crop_loc[0]: self.crop_loc[0] + crop_size, self.crop_loc[1]: self.crop_loc[1] + crop_size]

    def com_crop_per_slice(self, x, crop_size=256):
        """

        :param x:
        :param crop_size:
        :return:
        """
        x_cropped = np.zeros((x.shape[0], crop_size, crop_size), dtype=x.dtype)
        self.crop_loc_per_slice = self.COM_per_slice.copy()
        for i, COM in enumerate(self.COM_per_slice):
            crop_loc = COM - int(crop_size / 2)
            x_cropped[i] = x[i, crop_loc[0]: crop_loc[0] + crop_size, crop_loc[1]: crop_loc[1] + crop_size]  # +1 for the attempt to match the COM with matlab version
            self.crop_loc_per_slice[i] = crop_loc.copy()
        return x_cropped

    def com_paste(self, x):
        """
        Paste x at the center of mass
        :param x: array to be pasted (DHW array)
        :return:
        """
        z = np.zeros((x.shape[0], self.h, self.w), dtype='float32')
        z[:, self.crop_loc[0]: self.crop_loc[0] + x.shape[1], self.crop_loc[1]: self.crop_loc[1] + x.shape[2]] = x
        return z

    def save_mr_fig(self, fig_size_unit=7e-3, all_slices=False, com_crop=True, show_im=False, prefix=''):
        """
        Show all MR sequences with the prostate mask
        :param show_im: whether show the figure or not
        :param com_crop: whether crop the image at the COM or not
        :param fig_size_unit: size of the short edge
        :param all_slices: whether show slices without the prostate or not
        :return:
        """
        seq = self.seq
        mask = np.concatenate([self.com_crop(self.WP) for s in seq], axis=1)
        im = np.concatenate(
            [self.norm_01(self.com_crop(self.__getattribute__(s)), self.com_crop(self.WP)) for s in seq], axis=1)
        if self.density:
            im = np.concatenate([im] + [self.com_crop(self.density[tt]) for tt in self.tissue_types], axis=1)
            mask = np.concatenate([mask] + [self.com_crop(self.WP) for s in self.tissue_types], axis=1)
            # Assume that every density map has ROIs
            ROIs = np.concatenate([self.com_crop(self.density_ROIs) for s in seq], axis=1)
            ROIs = np.concatenate([ROIs] + [self.com_crop(self.density_ROIs) for s in self.tissue_types], axis=1)
        if not all_slices:
            im = self.rm_non_prostate(im, mask)
            if self.density:
                mask += (ROIs * mask)
            mask = self.rm_non_prostate(mask, mask)  # This has to be called last

        im_mt = montage(im, grid_shape=(1, im.shape[0]))
        mask_mt = montage(mask, grid_shape=(1, im.shape[0]))
        fig_size = (fig_size_unit * im_mt.shape[1], fig_size_unit * im_mt.shape[0])
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        ax.imshow(im_mt, cmap='gray')
        ax.contour(mask_mt, levels=np.arange(0, 5))
        plt.axis('off')
        plt.savefig(
            r'E:\Minh\projects\RadPath\RadPath_PixUDA\data_preparation\NIH\figures\radpath_overview/%s_%s_%s.png' % (
                prefix, self.mr_id, self.tis_id))
        if show_im:
            plt.show()
        plt.close(fig)
        # exit()

    def get_sub_dir(self, current_dir=None):
        """Get a list of sub-directories of current_dir"""
        current_dir = os.path.dirname(self.mr_path) if current_dir is None else current_dir
        return [s for s in os.listdir(current_dir) if
                (os.path.isdir(os.path.join(current_dir, s)) and (s not in ['fplbp', 'MRICAD', 'slic'])
                 and ('target_lesions' not in s))]

    def load_acq_date(self, target_path=None, force_read=False):
        """"""
        target_path = self.mr_path if target_path is None else target_path
        current_dir = os.path.dirname(target_path)
        sub_dir = self.get_sub_dir()
        current_dir = join_pth(current_dir, sub_dir[0])
        assert len(sub_dir) == 1
        sub_dir = self.get_sub_dir(current_dir)
        assert len(sub_dir) > 0
        current_dir = join_pth(current_dir, sub_dir[0])
        try:
            dicom = pydicom.dcmread(glob(f'{current_dir}/*')[0], force=force_read)
            return dicom.AcquisitionDate
        except:
            return -1

    def load_mr(self):
        """
        Load all mr sequences and prostate mask.
        This function assumes that the prostate mask is always available for each patient
        :return:
        """
        expected_seq = ['T2', 'ADC', 'B2000']
        mr = loadmat(self.mr_path)

        [self.seq.append(eseq) for eseq in expected_seq if ((eseq in mr.keys()) and (len(mr[eseq]) != 0))]
        if len(mr['T2']) == 1:  # The dimension of the field might be (1, 26) or (26, 1)
            [self.__setattr__(s, np.concatenate([m[np.newaxis] for m in mr[s][0]]).astype('float32')) for s in self.seq]
            self.d, self.h, self.w = self.T2.shape

            self.WP = np.concatenate(
                [m[np.newaxis] if np.all(m.shape) else np.zeros((1, self.h, self.w)) for m in mr['ROI'][0]]).astype(
                'float32')
        else:
            [self.__setattr__(s, np.concatenate([m[0][np.newaxis] for m in mr[s]])) for s in self.seq]
            self.d, self.h, self.w = self.T2.shape
            self.WP = np.concatenate(
                [m[0][np.newaxis] if np.all(m[0].shape) else np.zeros((1, self.h, self.w)) for m in mr['ROI']])
        # In some MR, ROI may not have the same number of cells with other fields (last empty cells were discarded)
        if self.WP.shape[0] < self.d:
            self.WP = np.pad(self.WP, pad_width=((0, self.d - self.WP.shape[0]), (0, 0), (0, 0)), mode='constant')

    def get_roi_files(self, roi_char):
        """

        :param roi_char:
        :return:
        """
        roi_files = glob(os.path.join(self.roi_pth, self.tis_id, f'ROI_{roi_char}', '*.xml'))
        roi_files = [roi_file for roi_file in roi_files if 'old' not in roi_file]
        roi_files = [roi_file for roi_file in roi_files if
                     roi_file.split('\\')[-1][4:-4] in self.density_ROIs_mapping.keys()]
        return roi_files

    def xml_to_mask(self, roi_file):
        """

        :param roi_file:
        :return:
        """
        tree = ET.parse(roi_file)
        root = tree.getroot()
        poly_verts = [subelem.text.split(',') for elem in root for subelem in elem]
        poly_verts_idx = [i + 1 for i in range(len(poly_verts)) if len(poly_verts[i]) == 1] + [len(poly_verts) + 1]
        nx, ny = self.h, self.w
        grid = np.zeros((ny, nx))
        for i in range(len(poly_verts_idx) - 1):
            _poly_verts = [(int(float(poly_vert[0])), int(float(poly_vert[1]))) for poly_vert in
                           poly_verts[poly_verts_idx[i]:poly_verts_idx[i + 1] - 1]]
            x, y = np.meshgrid(np.arange(nx), np.arange(ny))
            x, y = x.flatten(), y.flatten()

            points = np.vstack((x, y)).T

            path = Path(_poly_verts)
            grid += path.contains_points(points).reshape((ny, nx))
        return grid

    def load_density(self, tissue_types=('EPI',)):
        """Load tissue density images"""
        tmp0 = {}
        for tt in self.tissue_types:
            tmp0[tt] = np.zeros(self.T2.shape)
        for sl in self.slices_idx:
            pth = []
            for ppth in self.path_pth:
                pth = glob(join_pth(ppth, f'{self.tis_id}-*{sl[-1]}{self.path_suffice}'))
                if pth:  # if density file found
                    break
            if not pth:  # if there is no density file
                continue
            else:
                sl_idx_str = re.search('\d*', sl).group()
                sl_idx = int(sl_idx_str) - 1
                roi_char = sl[len(str(sl_idx_str)):]
                for tt in tissue_types:
                    tmp1 = loadmat(pth[0])['area' + tt]
                    # Remove NaN values
                    tmp1[np.isnan(tmp1)] = 0
                    tmp0[tt][sl_idx] = tmp1
                self.roi_meta[sl_idx] = {}
                for roi_file in self.get_roi_files(roi_char):
                    roi_name = roi_file.split('\\')[-1][4:-4]
                    self.roi_meta[sl_idx][roi_name] = {}
                    self.roi_meta[sl_idx][roi_name]['bin'] = self.xml_to_mask(roi_file)
                    self.roi_meta[sl_idx][roi_name]['path'] = roi_file
        self.density = tmp0 if tmp0 else self.density
        self.generate_density_ROI_mask()

    def generate_density_ROI_mask(self):
        """Generate a multi-level density ROI mask from the roi-meta"""
        self.density_ROIs = np.zeros_like(self.WP)
        for sl_idx in list(self.roi_meta.keys()):
            this_slice = self.roi_meta[sl_idx]
            for roi_name in list(this_slice.keys()):
                locs = np.where(this_slice[roi_name]['bin'] == 1)
                self.density_ROIs[sl_idx, locs[0], locs[1]] = self.density_ROIs_mapping[roi_name]


if __name__ == '__main__':
    from data_preparation.NIH.s00_load_dat import read_excel
    mri_pth = read_excel('mri_list_sorted_by_acq_time')
    mr_id = 'MRI442'
    d = RadPath(mr_id=mr_id, mr_pth=mri_pth[mr_id], tis_id=None, load_data=True, _load_density=True)
    print()
