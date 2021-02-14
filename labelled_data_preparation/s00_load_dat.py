import pandas as pd
import re
import os
from data_preparation.utils.radpath import RadPath
import numpy as np


EXCLUSIONS = {
    'MRI367': -10,
    'MRI609': -5,
    'MRI611': -4,
    'MRI446': -3,
}


def check_RadPath_in_use(_MR2Path_map, _mri_pth):
    """Print MRI ID - Tissue ID - MR path
    Copy the below for quick usage:
    check_RadPath_in_use(MR2Path_map, mri_pth)
    """
    for i, (_mr_id, _tis_id) in enumerate(_MR2Path_map.items()):
        print(i, _tis_id, _mr_id, _mri_pth[_mr_id])


def read_excel(sheet_name='RadPath_IDmap'):
    """A convenience function to read information from info.xlsx"""
    filename = os.path.join(os.path.dirname(__file__), 'info.xlsx')
    df = pd.read_excel(filename, sheet_name)
    if sheet_name == 'RadPath_IDmap':

        _Path2MR_map = {}
        _MR2Path_map = {}
        for i in range(len(df.TissueID)):
            _Path2MR_map[df.TissueID[i]] = {df.MR_Slice[i].split(' ')[0]: df.MR_Slice[i].split(' ')[1:]}
            _MR2Path_map[df.MR_Slice[i].split(' ')[0]] = {df.TissueID[i]: df.MR_Slice[i].split(' ')[1:]}
        return _Path2MR_map, _MR2Path_map
    if 'mri_list' in sheet_name:
        _mri_list = {}
        for pth in df.paths:
            id = re.findall('MRI[0-9]+', pth)
            assert (len(id) == 1)
            _mri_list[id[0]] = pth
        return _mri_list
    if sheet_name == 'RadPath_IDs_InUse':
        return df
    if sheet_name == '200Unsup':
        return df


if __name__ == "__main__":
    dir_out = 'data/preproc_EESL/'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    ids_in_use = read_excel('RadPath_IDs_InUse')
    mri_pth = read_excel('mri_list_sorted_by_acq_time')
    Path2MR_map, MR2Path_map = read_excel('RadPath_IDmap')

    rp = {}
    for i, (mr_id, tis_id) in enumerate(MR2Path_map.items()):
        # if mr_id != 'MRI367':
        #     continue
        rp = RadPath(mr_id, mri_pth[mr_id], tis_id, load_data=True)
        print(i, rp.mr_id, rp.slices_idx, rp.load_acq_date())
        # print(rp[mr_id].load_acq_date())
        # continue
        # print(rp[mr_id].mr_path)

        """Image preprocessing"""
        # Crop images
        wp = rp.com_crop_per_slice(rp.WP)  # wp has to be crop first
        T2 = rp.threshold_img(rp.com_crop_per_slice(rp.T2), wp)
        ADC = rp.threshold_img(rp.com_crop_per_slice(rp.ADC), wp)
        EPI = rp.com_crop_per_slice(rp.density['EPI'])
        ENUC = rp.com_crop_per_slice(rp.density['ENUC'])
        STR = rp.com_crop_per_slice(rp.density['STR'])
        LUM = rp.com_crop_per_slice(rp.density['LUM'])
        # EPI = np.concatenate([blur_density_map(epi, _wp, [0, 1])[np.newaxis] for epi, _wp in zip(EPI, wp)])
        ROIs = rp.com_crop_per_slice(rp.density_ROIs)
        # Keep slices containing the prostate
        EPI, ENUC, STR, LUM, ROIs, wp, T2, ADC = \
            rp.rm_non_prostate(EPI, wp)[..., np.newaxis], \
            rp.rm_non_prostate(ENUC, wp)[..., np.newaxis], \
            rp.rm_non_prostate(STR, wp)[..., np.newaxis], \
            rp.rm_non_prostate(LUM, wp)[..., np.newaxis], \
            rp.rm_non_prostate(ROIs, wp)[..., np.newaxis], \
            rp.rm_non_prostate(wp, wp)[..., np.newaxis], \
            rp.rm_non_prostate(T2, wp)[..., np.newaxis], \
            rp.rm_non_prostate(ADC, wp)[..., np.newaxis]

        im = np.concatenate([T2, ADC, EPI, ENUC, STR, LUM, wp, ROIs], axis=-1)
        im = im[:EXCLUSIONS[rp.mr_id]] if rp.mr_id in EXCLUSIONS.keys() else im
        np.save('%s/%03d_%s_%s.npy' % (dir_out, i, rp.mr_id, rp.tis_id), im.astype('float32'))
        # rp.save_mr_fig(prefix='%03d' % i)
