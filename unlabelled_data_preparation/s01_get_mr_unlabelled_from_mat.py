import pickle
from glob import glob
from scipy.io import loadmat
import numpy as np
import os
import pandas as pd

font = {'family': 'consolas',
        'color': 'lightgreen',
        'weight': 'normal',
        'size': 14,
        }


def mkdir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def read_txt(_filename):
    try:
        with open(_filename, 'rb') as fp:
            f = fp.readlines()
        f = [str(s).replace("b'", '').replace("\\r\\n'", '').replace("\\t\\t# ", ' ') for s in f]
        return f
    except:
        return


def norm_01(x):
    return (x - x.min()) / (x.max() - x.min())


def center_crop(img, com, sz=256):
    start_x, start_y = com[1] - int(sz / 2), com[0] - int(sz / 2)
    return img[..., start_y:start_y + sz, start_x: start_x + sz]


def get_y(x, a, b, h, k):
    """For ellipse"""
    tmp = b ** 2 - (b / a) ** 2 * (x - h) ** 2
    tmp = 0 if tmp < 0 else tmp  # In some case tmp is a negative number approaching 0
    y1 = -np.sqrt(tmp) + k
    y2 = +np.sqrt(tmp) + k
    return y1, y2


def get_x(y, a, b, h, k):
    """For ellipse"""
    tmp = a ** 2 - (a / b) ** 2 * (y - k) ** 2
    tmp = 0 if tmp < 0 else tmp  # In some case tmp is a negative number approaching 0

    x1 = -np.sqrt(tmp) + h
    x2 = +np.sqrt(tmp) + h
    return x1, x2


if __name__ == "__main__":
    """This code selects and store data in each of the unlabelled set in the Numpy format after extracting them from the 
    .mat file. The stored data will be used as inputs for deep learning model."""

    for unlabelled_set in ['unsup200', 'sextant', 'extra']:
        print(unlabelled_set, '\n', '-----------------------')
        dir_npy = f'MRI_Numpy/inputs_{unlabelled_set}/'
        if not os.path.exists(dir_npy):
            mkdir(dir_npy)
        image_folder_suffix = ''
        subject_list_file = f'subject_list_{unlabelled_set}_dir'  # subject_list_unsup200_dir
        if 'sextant' in subject_list_file:
            f_sextant = pd.read_excel(r'F:\Minh\projects\RadPath\matlab/MRICAD_PatientList_SextantBx.xlsx')
            image_folder_suffix = '_sextant'
        with open('data/%s.txt' % subject_list_file, 'rb') as fp:
            subject_dirs = pickle.load(fp)
        for i_sub, sub_dir in enumerate(subject_dirs):
            sub_id = sub_dir.split('\\')[-1]
            mat_file = glob(f'{sub_dir}/MRIs*.mat')[0]
            numpy_filename = f'{dir_npy}/{sub_id}.npy'
            if os.path.exists(numpy_filename):
                continue
            if 'sextant' in subject_list_file:
                coor_list = f_sextant[f_sextant.MRN == sub_id]
                roi_loc = {}
                for i in range(len(coor_list)):
                    roi_loc[coor_list.T2.values[i] - 1] = [coor_list.TargetY.values[i], coor_list.TargetX.values[i]]
            else:
                roi_file = f'{sub_dir}/targets.voi'
                roi_str = read_txt(roi_file)
                if not roi_str:
                    print(i_sub, sub_dir, 'VOI missing')
                    continue
            print(i_sub, sub_dir)

            try:
                mat = loadmat(mat_file)
                T2 = np.concatenate([m[np.newaxis] for m in mat['T2'][0]])
                ADC = np.concatenate([m[np.newaxis] for m in mat['ADC'][0]])
                wp = np.zeros_like(T2)

                for (i, m) in enumerate(mat['ROI'][0]):
                    if m.shape != (1, 0):
                        wp[i] = m
            except Exception:
                print('Cannot read file or missing modality or whole prostate segmentation')
                continue

            # create img with 4 channels, with the 3rd channel acts as a place holder for ROI mask
            img = np.concatenate((T2[np.newaxis], ADC[np.newaxis], np.zeros_like(wp)[np.newaxis], wp[np.newaxis]))
            img *= img[-1]

            com = [int(round(x.mean())) for x in np.where(img[-1].sum(axis=0))]
            img = center_crop(img, com)
            np.save(numpy_filename, img.transpose(2, 3, 1, 0))
        print('\n')
