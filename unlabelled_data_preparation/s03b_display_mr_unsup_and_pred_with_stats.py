import pandas as pd
import pickle
from glob import glob
import numpy as np
from scipy.io import loadmat
from skimage.util import montage
import pylab as plt
import os
from numpy.ma import masked_array

font = {'family': 'consolas',
        'color': 'lightgreen',
        'weight': 'normal',
        'size': 14,
        }
font_label = {'family': 'consolas',
              'color': 'red',
              'weight': 'normal',
              'size': 14,
              }

radius_dict = {
    # 'point': -1,
    '1mm': 2,
    '3mm': 5,
    '5mm': 9,
    '7mm': 13,  # = (7/0.27 -> 26)
}


def read_txt(_filename):
    try:
        with open(_filename, 'rb') as fp:
            f = fp.readlines()
        f = [str(s).replace("b'", '').replace("\\r\\n'", '').replace("\\t\\t# ", ' ') for s in f]
        return f
    except:
        return


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


def center_crop(img, com, sz=256):
    start_x, start_y = com[1] - int(sz / 2), com[0] - int(sz / 2)
    return img[..., start_y:start_y + sz, start_x: start_x + sz]


def gen_bin_mask(_shape, rx, ry, cx, cy):
    _roi = np.zeros(_shape)
    if (rx == -1) | (ry == -1):
        _roi[cy, cx] = 1
        return _roi
    if rx >= ry:
        for x in np.arange(cx - rx, cx + rx, 1e-3):
            y1, y2 = get_y(x, rx, ry, cx, cy)
            if np.isnan(y1) or np.isnan(y1):
                print(x, rx, ry, cx, cy)
            _roi[int(round(y1)):int(round(y2)), int(round(x))] = 1
    else:
        for y in np.arange(cy - ry, cy + ry, 1e-3):
            x1, x2 = get_x(y, rx, ry, cx, cy)
            if np.isnan(x1) or np.isnan(x2):
                print(y, rx, ry, cx, cy)
            _roi[int(round(y)), int(round(x1)):int(round(x2))] = 1
    return _roi


def mkdir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def norm_01(x):
    return (x - x.min()) / (x.max() - x.min())


def show_targets(img, _all_sli, _labels, img_name, figsize_unit=2, suffix='', tissue_type='EPI'):
    """"""
    img[-2] = img[-2] * img[-1]
    mean_density = [_pred[_roi > 0].mean() for (_pred, _roi) in zip(img[2], img[-2])]
    # mean_T2 = [_pred[_roi > 0].mean() for (_pred, _roi) in zip(norm_01(img[0]), img[-2])]
    # mean_ADC = [_pred[_roi > 0].mean() for (_pred, _roi) in zip(norm_01(img[1]), img[-2])]
    mean_T2 = [_pred[_roi > 0].mean() for (_pred, _roi) in zip(img[0], img[-2])]
    mean_ADC = [_pred[_roi > 0].mean() for (_pred, _roi) in zip(img[1], img[-2])]

    t2_fig = norm_01(img[0, :])
    adc_fig = norm_01(img[1, :])
    pred_fig = img[2, :]
    roi_fig = img[-2, :]

    # plt.imshow(gray_fig_masked, cmap='gray')
    # plt.imshow(montage(jet_fig_masked, grid_shape=grid_shape), cmap='jet', vmax=.7)

    fig = np.vstack((t2_fig, adc_fig, pred_fig))
    roi_fig = np.vstack((roi_fig, roi_fig, roi_fig))

    grid_shape = (3, len(t2_fig))
    plt.figure(1, (figsize_unit * grid_shape[1], figsize_unit * grid_shape[0]))

    jet_fig = montage(np.vstack((np.zeros_like(t2_fig), np.zeros_like(t2_fig), img[-1, :])),
                      grid_shape=grid_shape)
    jet_fig_masked = masked_array(montage(fig, grid_shape=grid_shape), jet_fig == 0)
    # gray_fig_masked = masked_array(montage(fig), jet_fig == 1)

    plt.imshow(montage(fig, grid_shape=grid_shape), cmap='gray', vmax=1)
    plt.imshow(jet_fig_masked, cmap='jet', vmin=0, vmax=.7)
    plt.contour(montage(roi_fig, grid_shape=grid_shape), cmap='gray', linewidths=.3)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.axis('off')
    first_text_x = img[0].shape[1] / 2
    # [plt.text(first_text_x + img[0].shape[1] * i, 40, _all_sli[i], fontdict=font) for i in range(len(all_sli))]
    [plt.text(first_text_x + img[0].shape[1] * i - 30, img[0].shape[2] * 1 - 30, '%.3f' % mean_T2[i],
              fontdict=font) for i in range(len(_all_sli))]
    [plt.text(first_text_x + img[0].shape[1] * i - 30, img[0].shape[2] * 2 - 30, '%.3f' % mean_ADC[i],
              fontdict=font) for i in range(len(_all_sli))]
    [plt.text(first_text_x + img[0].shape[1] * i - 30, img[0].shape[2] * 3 - 30, '%.3f' % mean_density[i],
              fontdict=font) for i in range(len(_all_sli))]
    [plt.text(first_text_x + img[0].shape[1] * i - 45, img[0].shape[2] * 3 - 10, '%s' % _labels[i],
              fontdict=font_label) for i in range(len(_all_sli))]

    dir_fig = 'MRIs/targets_with_%sDensityPrediction_withStats' % (tissue_type,) + suffix
    mkdir(dir_fig)
    plt.savefig(f'{dir_fig}/{img_name}.png', dpi=300)
    plt.close()
    return mean_density, mean_T2, mean_ADC


def get_mean_data(d):
    # for tissue_type in ['NUC']:
    _mean_data = []
    _columns = []

    rx, ry = radius_dict[d], radius_dict[d]
    measure_mr_intensity = True
    pred_density_collection, intensity_T2_collection, intensity_ADC_collection = {}, {}, {}
    for m_idx in range(len(tissue_types)):
        pred_density_collection[tissue_types[m_idx]] = []
        intensity_T2_collection = []
        intensity_ADC_collection = []

    with open('data/subject_list_unsup200_dir.txt', 'rb') as fp:
        subject_dirs = pickle.load(fp)
    subject_dirs_dict = {sd.split('\\')[-1]: sd for sd in subject_dirs}
    f = pd.read_excel('200Unsup_Path_Report.xlsx', '200UnsupPathReportV1_QC_mGG')

    all_ids = f.PatientID.unique()
    for i_sub, sub_id in enumerate(all_ids):
        # if i_sub > 3:
        # continue
        # break
        cs = f[f.PatientID == sub_id]  # current subject
        sub_dir = subject_dirs_dict[sub_id]

        mat_file = glob(f'{sub_dir}/MRIs*.mat')[0]
        roi_file = f'{sub_dir}/targets.voi'
        roi_str = read_txt(roi_file)
        # print(i_sub, sub_dir)

        with open('%s/%s.npy' % (dir_pred, sub_id), 'rb') as fp:
            mr = pickle.load(fp)

        mat = loadmat(mat_file)
        T2 = np.concatenate([m[np.newaxis] for m in mat['T2'][0]])
        ADC = np.concatenate([m[np.newaxis] for m in mat['ADC'][0]])
        wp = np.zeros_like(T2)
        for (i, m) in enumerate(mat['ROI'][0]):
            if m.shape != (1, 0):
                wp[i] = m
        roi = np.zeros_like(wp)

        img = np.concatenate((T2[np.newaxis], ADC[np.newaxis], roi[np.newaxis], wp[np.newaxis]))
        img *= img[-1]
        com = [int(round(x.mean())) for x in np.where(img[-1].sum(axis=0))]

        slice_idx_report = [int(roi_str[i].replace(' slice number', '')) for i in range(len(roi_str)) if
                            'slice number' in roi_str[i]]
        coor_report = [roi_str[i + 3] for i in range(len(roi_str)) if 'slice number' in roi_str[i]]

        all_sli = [sli for sli in cs.SliceIdx]
        # Extract target labels
        all_labels = ['cancer' if con == 1 else 'benign' for con in cs.Condition]
        all_grades = [' (%s)' % g if g != 0 else '' for g in cs.GleasonGrade]
        all_labels = [l + g for (l, g) in zip(all_labels, all_grades)]
        img = img[:, all_sli]  # number of slices = len of sli

        for (img_i, sli) in enumerate(all_sli):
            sli_loc = np.where(np.array(slice_idx_report) == sli)[0][0]
            current_coor = coor_report[sli_loc]
            current_coor = [round(float(x)) for x in current_coor.split()]

            # Remove the chosen loc
            coor_report.pop(sli_loc)
            slice_idx_report.pop(sli_loc)

            cx, cy = current_coor
            img[-2, img_i] = gen_bin_mask(roi.shape[1:], rx, ry, cx, cy)

        img = center_crop(img, com).astype('float32')
        for m_idx in range(mr[1].shape[1]):
            pred = mr[1][:, m_idx][np.newaxis]  # shape: NCHW

            _img = np.insert(img, 2, pred[:, all_sli], axis=0)
            densities, intensity_T2, intensity_ADC = show_targets(_img, all_sli, all_labels, img_name=sub_id,
                                                                  suffix=f'_{d}', tissue_type=tissue_types[m_idx])
            pred_density_collection[tissue_types[m_idx]] += densities

            if m_idx == 0:
                intensity_T2_collection += intensity_T2
                intensity_ADC_collection += intensity_ADC
    _mean_data = [*[pred_density_collection[tt] for tt in tissue_types],
                  intensity_T2_collection, intensity_ADC_collection]
    _columns = [*[f'pred_{tt}_{d}' for tt in tissue_types], f'T2_{d}', f'ADC_{d}']
    # print(_mean_data, _columns)
    # _mean_data.extend([*[pred_density_collection[tt] for tt in tissue_types],
    #                   intensity_T2_collection, intensity_ADC_collection])
    # _columns.extend([*[f'pred_{tt}_{d}' for tt in tissue_types], f'T2_{d}', f'ADC_{d}'])

    return _mean_data, _columns


if __name__ == "__main__":
    """Load inputs, predictions, and create a target mask for each target
    This code tries to deal with a situation in which a slice contains multiple targets
    """
    tissue_types = ['EPI', 'NUC', 'STR', 'LUM']

    experiment_name = 'ImprovedSemi_PostMICCAI_5'
    # rid = 'drnnGR4_lCL0_ENSL_lUS1_lC0_l2_nc8_stsa0.9_sd11_normal_v2b_check_last_iter_issues_TSA0.90'
    rid = 'drnnGR4_lCL0_EESL_lUS1_lC0_l2_nc8_stsa0.9_r1_sd63_normal_v2b_switch_to_ENUC_v4_TSA0.90'
    it = 1999  # 2499
    dir_pred = r'../../results\%s\%s\inference\matrix\iter_%d' % (experiment_name, rid, it)
    dir_inputs = r'MRIs/inputs_unsup200/'

    # diameters = ['1mm', '3mm', '5mm', '7mm']
    diameters = ['7mm', ]
    mean_data = {}
    columns = {}
    from concurrent.futures import ThreadPoolExecutor, wait, as_completed

    # max_workers = len(diameters)
    max_workers = 4
    futures = []
    with ThreadPoolExecutor(max_workers) as pool:
        for diameter in diameters:  # point
            futures.append(pool.submit(get_mean_data, diameter))

    for future, diameter in zip(as_completed(futures), diameters):
        mean_data[diameter], columns[diameter] = future.result()

    mean_data_combined = []
    columns_combined = []
    for diameter in diameters:
        mean_data_combined.extend(mean_data[diameter])
        columns_combined.extend(columns[diameter])

    # df = pd.DataFrame(np.array(mean_data_combined).T, columns=columns_combined)
    # with pd.ExcelWriter('spread_sheets/RadPath_Mean_Values_200Unsup.xlsx', engine="openpyxl", mode="a") as writer:
    #     df.to_excel(writer, sheet_name='200Unsup')
