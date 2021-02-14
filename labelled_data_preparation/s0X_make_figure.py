import numpy as np
import pylab as plt
from skimage.util import montage
import matplotlib as mpl
from matplotlib import cm
from scipy.ndimage.filters import median_filter
from PIL import Image
import cv2
import imgaug.augmenters as iaa


def norm_01(a: np.ndarray):
    return (a - a.min()) / (a.max() - a.min())


def blur_density_maps(density):
    """"""
    seq = iaa.Sequential([
        iaa.AverageBlur(k=18, random_state=256),  # blur images with a sigma of 0 to 3.0
    ]).to_deterministic()
    return seq(image=density)


def blur_density_map(_map, mask, d_range):
    tmp = _map.copy()
    tmp[mask == 0] = 0
    blurred_pre_corrected = blur_density_maps(tmp)
    corrector = blur_density_maps(mask)
    corrector[mask == 0] = 1
    blurred = blurred_pre_corrected / corrector
    blurred[mask == 0] = d_range[0]  # masking with whole prostate mask
    return blurred


def get_color():
    """
    :return: a colormap jet
    """
    cm_jet = np.reshape(np.concatenate([cm.jet(i) for i in range(255)], axis=0), (255, 4))
    return np.vstack((np.array([0, 0, 0]), cm_jet[:, :3]))


JET = get_color()
JET = mpl.colors.ListedColormap(JET, name='my_jet', N=JET.shape[0])

coef = .8
density_ranges_pred = [(0.0, 0.61 * coef), (0.0, 0.1934 * 2.5), (0.43, 0.921 * coef), (0.0, 0.174 * 3.5)]
density_ranges = [(0.0, .5), (0.0, .15), (0, .7), (0.0, .7)]
density_ranges_pred = [(0.0, .5 * 1.5), (0.0, .15 * 1.5), (0, .6), (0.0, 1)]

# x = np.load(r'E:\Minh\projects\RadPath\RadPath_PixUDA\data_preparation\NIH\data\preproc_EESL\026_MRI442_PATH36.npy')
x = np.load(r'E:\Minh\projects\RadPath\RadPath_PixUDA\data_preparation\NIH\data\preproc_EESL\035_MRI620_PATH50.npy')
x = x[6].transpose([2, 0, 1])
x[0] = norm_01(x[0])
x[1] = norm_01(x[1])
# plt.imshow(montage(x), cmap='gray', vmin=0, vmax=1)
# for i, dr in enumerate(density_ranges):
#     xx = blur_density_map(x[i+2], x[-2], [0, 1]) * x[-2]
#     z = blur_density_maps(x[i+2]) * x[-2]
#     plt.imshow(np.hstack((x[i+2], xx, z)), cmap=JET, vmin=dr[0], vmax=dr[1])
#     plt.show()

modalities = ['T2', 'ADC']
tissue_types = ['EPI', 'ENUC', 'STR', 'LUM']
names = modalities + tissue_types
save_dir = r'C:\Users\Minh\Google Drive\Writing\RadPath\Analysis\raw_images\method'

sz = 5
show_overlay = False

if show_overlay:
    for i, dr in enumerate(density_ranges):
        plt.figure(figsize=(sz, sz))
        # x[i+2] = blur_density_map(x[i+2], x[-2], [0, 1]) * x[-2]
        plt.imshow(x[0], cmap='gray')
        plt.imshow(x[i + 2], cmap=JET, vmin=dr[0], vmax=dr[1], alpha=.5)
        plt.axis('off')
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        plt.savefig(f'{save_dir}/overlay_nb_{tissue_types[i]}.png', dpi=300)
        plt.close()
    exit()

# for i, (dr, drp) in enumerate(zip(density_ranges, density_ranges_pred)):
#     z = blur_density_map(x[i+2], x[-2], [0, 1]) * x[-2]
#     # plt.figure(figsize=(sz, sz))
#     # plt.imshow(z, cmap=JET, vmin=dr[0], vmax=dr[1])
#     # plt.contour(x[-1], linewidths=.3, colors='white')
#     # plt.axis('off')
#     # plt.subplots_adjust(0, 0, 1, 1, 0, 0)
#     # plt.savefig(f'{save_dir}/gt_ctr_{tissue_types[i]}.png', dpi=300)
#     # plt.close()
#
#     plt.figure(figsize=(sz, sz))
#     plt.imshow(z, cmap=JET, vmin=drp[0], vmax=drp[1])
#     plt.contour(x[-1], linewidths=1, colors='white')
#     plt.axis('off')
#     plt.subplots_adjust(0, 0, 1, 1, 0, 0)
#     plt.savefig(f'{save_dir}/pred_ctr_{tissue_types[i]}.png', dpi=300)
#     plt.close()

for i in range(2):
    plt.figure(figsize=(sz, sz))
    plt.imshow(x[i] * x[-2], cmap='gray')
    # plt.contour(x[-1], linewidths=.3, colors='white')
    plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.savefig(f'{save_dir}/{modalities[i]}_ctr_2.png', dpi=300)
    plt.close()

# for i in range(2):
#     plt.figure(figsize=(sz, sz))
#     plt.imshow(x[i], cmap='gray')
#     plt.contour(x[-2], linewidths=1, colors='red')
#     plt.axis('off')
#     plt.subplots_adjust(0, 0, 1, 1, 0, 0)
#     plt.savefig(f'{save_dir}/{modalities[i]}_raw_ctr.png', dpi=300)
#     plt.close()

# for i in range(2):
#     plt.figure(figsize=(sz, sz))
#     plt.imshow(x[i], cmap='gray')
#     plt.axis('off')
#     plt.subplots_adjust(0, 0, 1, 1, 0, 0)
#     plt.savefig(f'{save_dir}/{modalities[i]}_raw.png', dpi=300)
#     plt.close()
