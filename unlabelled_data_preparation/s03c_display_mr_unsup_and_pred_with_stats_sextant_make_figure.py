import cv2
import pandas as pd
import pickle
from glob import glob
import numpy as np
from scipy.io import loadmat
from skimage.util import montage
import pylab as plt
import os
from numpy.ma import masked_array
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from pylab import cm
import matplotlib as mpl
from skimage.filters import sobel

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

groups = ['MR-', 'MR+', 'G3', 'G4']


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


def show_targets(img, _all_sli, _labels, img_name, figsize_unit=2, suffix='', tissue_type='EPI', density_range=(0, 1)):
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
    plt.imshow(jet_fig_masked, cmap='jet', vmin=density_range[0], vmax=density_range[1])
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


def get_color():
    """
    :return: a colormap jet
    """
    cm_jet = np.reshape(np.concatenate([cm.jet(i) for i in range(255)], axis=0), (255, 4))
    return np.vstack((np.array([0, 0, 0]), cm_jet[:, :3]))


JET = get_color()
JET = mpl.colors.ListedColormap(JET, name='my_jet', N=JET.shape[0])


def draw_contour(im, ctr, is_pred=False):
    ctr_color = (255, 255, 255)
    if not is_pred:
        ctr_color = (255, 255, 255)
        im = (im * 255).astype('uint8')
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    return cv2.drawContours(im, ctr, -1, ctr_color, 2)


def show_targets_separate(img, _all_sli, img_name, figsize_unit=2, suffix='', tissue_type='EPI',
                          density_range=(0, 1)):
    """"""
    group = groups[0]

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

    pred_fig = (pred_fig * 255).astype('uint8')

    for sl in range(img.shape[1]):
        edged = (sobel(roi_fig[sl]) > 0).astype('uint8')
        # edged = cv2.Canny(roi_fig[sl].astype('uint8'), 0.1, 2)
        contour, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        dir_fig = 'MRIs/targets_with_DensityPrediction_withStats' + suffix + '/' + group
        pred_name = f'{img_name}_{sl}_{tissue_type}_{mean_density[sl]:.3f}'
        if tissue_type == 'EPI':
            plt.imsave(f'{dir_fig}/{img_name}_{sl}_T2_{mean_T2[sl]:.0f}.png',
                       draw_contour(t2_fig[sl], contour), cmap='gray', vmax=1)
            plt.imsave(f'{dir_fig}/{img_name}_{sl}_ADC_{mean_ADC[sl]:.0f}.png',
                       draw_contour(adc_fig[sl], contour), cmap='gray', vmax=1)
        plt.imsave(f'{dir_fig}/{pred_name}.png', pred_fig[sl],
                   cmap=JET, vmin=int(density_range[0] * 255), vmax=int(density_range[1] * 255))
        pred_fig_ = cv2.cvtColor(cv2.imread(f'{dir_fig}/{pred_name}.png'), cv2.COLOR_BGR2RGB)
        plt.imsave(f'{dir_fig}/{pred_name}.png', draw_contour(pred_fig_, contour, is_pred=True),
                   cmap=JET)
    return mean_density, mean_T2, mean_ADC


def get_mean_data(d):
    _mean_data = []
    _columns = []
    all_sli_len = 0

    rx, ry = radius_dict[d], radius_dict[d]
    measure_mr_intensity = True

    pred_density_collection, intensity_T2_collection, intensity_ADC_collection = {}, {}, {}
    for m_idx in range(len(tissue_types)):
        pred_density_collection[tissue_types[m_idx]] = []
        intensity_T2_collection = []
        intensity_ADC_collection = []

    file = 'subject_list_sextant_dir.txt'
    with open('data/%s' % file, 'rb') as fp:
        subject_dirs = pickle.load(fp)
    subject_dirs_dict = {sd.split('\\')[-1]: sd for sd in subject_dirs}
    f = pd.read_excel(r'F:\Minh\projects\RadPath\matlab/MRICAD_PatientList_SextantBx.xlsx')
    # f = pd.read_excel('117Sextant_Report.xlsx', 'Report_Filter')

    suffix = f'_{d}'
    for group in groups:
        dir_fig = 'MRIs/targets_with_DensityPrediction_withStats' + suffix + '/' + group
        mkdir(dir_fig)

    def process(sub_id):
        nonlocal intensity_T2_collection, intensity_ADC_collection, all_sli_len
        # if i_sub != 0:
        #     continue
        # cs = f[f.PatientID == sub_id]  # current subject
        cs = f[f.MRN == sub_id]  # current subject
        sub_dir = subject_dirs_dict[sub_id]

        mat_file = glob(f'{sub_dir}/MRIs*.mat')[0]
        roi_file = f'{sub_dir}/targets.voi'
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

        all_sli = [sli - 1 for sli in cs.T2]
        # all_sli = [sli - 1 for sli in cs.SliceIdx]
        slice_idx_report = all_sli.copy()
        coor_report = []
        for i in range(len(cs)):
            coor_report.append([cs.TargetY.values[i], cs.TargetX.values[i]])
            # coor_report.append([cs.MR_Y.values[i], cs.MR_X.values[i]])
        import copy
        coors = copy.deepcopy(coor_report)
        # Extract target labels
        img = img[:, all_sli]  # number of slices = len of sli
        # [print(sub_id, list(cs.Location)[i], list(cs.T2)[i]) for i in range(len(coors))]

        for (img_i, sli) in enumerate(all_sli):
            sli_loc = np.where(np.array(slice_idx_report) == sli)[0][0]
            current_coor = coor_report[sli_loc]
            current_coor = [round(float(x)) for x in current_coor]
            # print(sub_id, sli_loc, current_coor[0], current_coor[1])

            # Remove the chosen loc
            coor_report.pop(sli_loc)
            slice_idx_report.pop(sli_loc)

            cx, cy = current_coor
            img[-2, img_i] = gen_bin_mask(roi.shape[1:], rx, ry, cx, cy)
        all_sli_len += len(all_sli)

        img = center_crop(img, com).astype('float32')
        coef = .8
        density_ranges = [(0.0, 0.61 * coef), (0.0, 0.1934 * 2.5), (0.43, 0.921 * coef), (0.0, 0.174 * 3.5)]

        for m_idx in range(mr[1].shape[1]):
            pred = mr[1][:, m_idx][np.newaxis]  # shape: NCHW
            _img = np.insert(img, 2, pred[:, all_sli], axis=0)
            densities, intensity_T2, intensity_ADC = show_targets_separate(
                _img, all_sli, img_name=sub_id, suffix=f'_{d}', tissue_type=tissue_types[m_idx],
                density_range=density_ranges[m_idx]
            )
            pred_density_collection[tissue_types[m_idx]] += densities
            if m_idx == 0:
                intensity_T2_collection += intensity_T2
                intensity_ADC_collection += intensity_ADC

    with ThreadPoolExecutor(8) as executor:
        executor.map(process, all_ids)

    _mean_data = [*[pred_density_collection[tt] for tt in tissue_types],
                  intensity_T2_collection, intensity_ADC_collection]
    _columns = [*[f'pred_{tt}_{d}' for tt in tissue_types], f'T2_{d}', f'ADC_{d}']
    # print(all_sli_len)
    return _mean_data, _columns


if __name__ == "__main__":
    """Load inputs, predictions, and create a target mask for each target
    This code tries to deal with a situation in which a slice contains multiple targets
    """
    tissue_types = ['EPI', 'NUC', 'STR', 'LUM']
    filename = 'data/bx_sextant.txt'
    all_ids = read_txt(filename)

    experiment_name = 'ImprovedSemi_PostMICCAI_5'
    # rid = 'drnnGR4_lCL0_ENSL_lUS1_lC0_l2_nc8_stsa0.9_sd11_normal_v2b_check_last_iter_issues_TSA0.90'
    rid = 'drnnGR4_lCL0_EESL_lUS1_lC0_l2_nc8_stsa0.9_r1_sd63_normal_v2b_switch_to_ENUC_v4_TSA0.90'
    it = 1999  # 2499
    dir_pred = r'../../results\%s\%s\inference\matrix\iter_%d' % (experiment_name, rid, it)

    # diameters = ['1mm', '3mm', '5mm', '7mm']
    diameters = ['7mm', ]

    for diameter in diameters:
        get_mean_data(diameter)
