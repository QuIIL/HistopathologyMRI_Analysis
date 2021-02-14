import os
from os.path import join as join_pth
import re
from glob import glob

import pydicom
import pandas as pd
from scipy.io import loadmat

PARTITIONS = ['sup', 'unsup', 'extra', 'sextant']


def get_intersection(list_A, list_B):
    return list(set(list_A).intersection(list_B))


def get_mri_path():
    _df = pd.read_excel(r'E:\Minh\projects\RadPath\RadPath_PixUDA\data_preparation\NIH/info.xlsx',
                        'mri_list_sorted_by_acq_time')
    _mri_list = {}
    for pth in _df.paths:
        id = re.findall('MRI[0-9]+', pth)
        assert (len(id) == 1)
        _mri_list[id[0]] = pth
    return _mri_list


def get_sub_dir(current_dir):
    """Get a list of sub-directories of current_dir"""
    return [s for s in os.listdir(current_dir) if
            (os.path.isdir(os.path.join(current_dir, s)) and (s not in ['fplbp', 'MRICAD', 'slic'])
             and ('target_lesions' not in s))]


def get_header(target_path, force_read=False):
    """"""
    current_dir = os.path.dirname(target_path)
    sub_dir = get_sub_dir(current_dir)
    current_dir = join_pth(current_dir, sub_dir[0])
    assert len(sub_dir) == 1
    sub_dir = get_sub_dir(current_dir)
    assert len(sub_dir) > 0
    current_dir = join_pth(current_dir, sub_dir[0])
    try:
        header = pydicom.dcmread(glob(f'{current_dir}/*')[0], force=force_read)
        return header
    except:
        return -1


def get_acq_date(header):
    return header.AcquisitionDate


def get_info():
    df = pd.read_excel('spread_sheets/RadPath_AveragedDensity_EESL.xlsx', '1mm')
    df_sup = pd.read_excel(r'E:\Minh\projects\RadPath\RadPath_PixUDA\data_preparation\NIH/info.xlsx', 'RadPath_IDmap')
    sup_list = list([df_sup.MR_Slice[i].split(' ')[0] for i in range(len(df_sup.TissueID))])

    PID = {
        'all': df.PatientID.unique(),
        'unsup': df[df.Partition == 1].PatientID.unique(),
        'extra': df[df.Partition == 2].PatientID.unique(),
        'sextant': df[df.Partition == 3].PatientID.unique(),
    }

    PID['common_unsup_extra'] = get_intersection(PID['unsup'], PID['extra'])
    PID['common_unsup_sextant'] = get_intersection(PID['unsup'], PID['sextant'])
    PID['common_extra_sextant'] = get_intersection(PID['extra'], PID['sextant'])

    print('\nNumber of patients:')
    for k in PID.keys():
        print(f'{k}: {len(PID[k])}')

    print(get_intersection(sup_list, PID['all']))

    partitions = PARTITIONS[1:]

    print('\nNumber of ROIs:')
    for i, partition in enumerate([1, 2, 3]):
        print(partitions[i], len(df[df.Partition == partition]))

    mri_paths = get_mri_path()
    PID['sup'] = sup_list
    #     mri_path = mri_paths[mr_id]
    # for mr_id in sup_list:
    #     header = get_header(mri_path)
    #     print(mr_id, get_acq_date(header))
    for mr_id in PID['sup']:
        mri_path = mri_paths[mr_id]
        print(mr_id, get_acq_date(get_header(mri_path)))
    exit()

    for partition in partitions:
        for mr_id in PID[partition]:
            mri_path = mri_paths[mr_id]
            try:
                mat = loadmat(mri_path)
                info = mat['T2info'][0][0]
                weight = info['PatientWeight'][0][0][0][0]
                birthdate = int(info['PatientBirthDate'][0][0][0][:4])
                acquisition_date = int(info['AcquisitionDate'][0][0][0][:4])
                age = acquisition_date - birthdate
                print(mr_id, partition, weight, age)
            except:
                print(mr_id, partition)


def summarize_info():
    from scipy.stats import iqr
    df = pd.read_excel(r'E:\Minh\projects\RadPath\RadPath_PixUDA\data_preparation\NIH/info.xlsx',
                       'demographics_Radiology')

    part = df[df.Partition == 'sup']
    print(f'Median [Labelled]: {int(part.Weight.median()):02d} {int(part.Age.median()):02d} ')
    print(part.describe())

    part = df[df.Partition != 'sup']
    part.drop('Partition', axis=1, inplace=True)
    part.drop_duplicates(inplace=True)
    print(f'Median [Unlabelled]: {int(part.Weight.median()):02d} {int(part.Age.median()):02d} ')
    print(part.describe())

    part = df
    part.drop('Partition', axis=1, inplace=True)
    part.drop_duplicates(inplace=True)
    print(f'Median: {int(part.Weight.median()):02d} {int(part.Age.median()):02d} ')
    print(part.describe())


if __name__ == '__main__':
    # get_info()
    summarize_info()
