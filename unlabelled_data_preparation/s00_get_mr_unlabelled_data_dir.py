import pickle
from glob import glob
import os
import pandas as pd
import re

DATA_DIR = r'F:\Workspace\NIH\MRI'


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


def read_txt(_filename):
    with open(_filename, 'rb') as fp:
        f = fp.readlines()
    f = [str(s).replace("b'", '').replace("\\r\\n'", '') for s in f]
    return f


def get_extra_dir():
    filename_unsup = 'data/subject_list_unsup200.txt'
    data_dir_exact = []
    mr_sup_list = list(read_excel('RadPath_IDmap')[1].keys())
    mr_unsup_list = read_txt(filename_unsup)
    mr_list = [mr.split('\\')[-2] for mr in glob(f'{DATA_DIR}/**/MRI*/MRIs*.mat')]
    mr_list_refined = [mr for mr in mr_list if mr not in (mr_sup_list + mr_unsup_list)]
    [data_dir_exact.append(glob(f'{DATA_DIR}/**/{sl}')[0]) for sl in mr_list_refined]
    with open('data/subject_list_extra_dir.txt', 'wb') as fp:
        pickle.dump(data_dir_exact, fp)
    print(data_dir_exact)
    print('Done!')


def get_unsup_dir():
    filename = 'data/subject_list_unsup200.txt'
    data_dir_exact = []
    subject_list = read_txt(filename)
    [data_dir_exact.append(glob(f'{DATA_DIR}/**/{sl}')[0]) for sl in subject_list]
    with open('data/subject_list_unsup200_dir.txt', 'wb') as fp:
        pickle.dump(data_dir_exact, fp)
    print(data_dir_exact)
    print('Done!')


def get_sextant_dir():
    filename = 'data/bx_sextant.txt'
    data_dir_exact = []
    subject_list = read_txt(filename)
    for sl in subject_list:
        pth = glob(f'{DATA_DIR}/**/{sl}')
        if len(pth) > 0:
            data_dir_exact.append(pth[0])
    with open('data/subject_list_sextant_dir.txt', 'wb') as fp:
        pickle.dump(data_dir_exact, fp)
    for i, _dir_ in enumerate(data_dir_exact):
        print(i, _dir_)
    print('Done!')


if __name__ == '__main__':
    get_unsup_dir()
    get_sextant_dir()
    get_extra_dir()
