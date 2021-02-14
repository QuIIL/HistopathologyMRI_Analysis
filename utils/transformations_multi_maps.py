import imgaug.augmenters as iaa
import numpy as np
import mxnet as mx
from imgaug.random import seed
import os

seed(0)
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from scipy.ndimage.filters import gaussian_filter, uniform_filter
import pylab as plt


def view_image(*args):
    """

    :param args: sequences of images
    The dimension should be HWC
    :return:
    """
    num_col = max([x.shape[-1] for x in args])
    num_row = len(args)
    if (num_col == 1) and (num_col == 1):
        plt.imshow(np.squeeze(args[0]))
        plt.axis('off')
        return
    fig, ax = plt.subplots(num_row, num_col,
                           figsize=(args[0].shape[1] * .01 * num_col, args[0].shape[1] * .01 * num_row))
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    v_range = (min([x.min() for x in args]), max([x.max() for x in args]))
    for i in range(num_row):
        for j in range(num_col):
            if j < args[i].shape[-1]:
                ax[i, j].imshow(args[i][..., j], cmap='gray', vmin=v_range[0], vmax=v_range[1])
            ax[i, j].axis('off')
    plt.show()


def just_crop(joint, input_size=256):
    seq = iaa.CropToFixedSize(position='center', width=input_size, height=input_size)
    return np.asarray([seq.augment_image(joint[idx]) for idx in range(joint.shape[0])])


def blur_density_maps(density):
    """"""
    seq = iaa.Sequential([
        iaa.AverageBlur(k=18, random_state=256),  # blur images with a sigma of 0 to 3.0
    ]).to_deterministic()
    return seq(image=density)


STATE = None


class Augmenter:
    """Define augmentation sequences"""

    def __init__(self):
        """Input shape always stay the same after the augmentation, while value be change for a same Augmenter object"""
        self.seq_shape = self.get_seq_shape().to_deterministic()  # iaa.Noop()
        self.seq_val = self.get_seq_val()  # iaa.Noop()
        self.seq_val1 = self.get_seq_val()
        self.seq_val2 = self.get_seq_val()
        self.seq_noop = iaa.Sequential([iaa.Noop(), iaa.Noop()])

    def get_seq_combined(self, no_shape_augment=False, no_val_augment=False):
        """Same shape & same value augmentations every time"""
        seq = iaa.Sequential([
            self.seq_noop if no_shape_augment else self.seq_shape,
            self.seq_noop if no_val_augment else self.seq_val,
        ]).to_deterministic()
        return seq

    @staticmethod
    def get_seq_shape():
        sometimes = lambda aug: iaa.Sometimes(0.5, aug, random_state=STATE, )
        seq_shape = iaa.Sequential([
            # sometimes(iaa.Crop(percent=(0, .1))),  # crop images from each side by 0 to 16px (randomly chosen)
            sometimes(iaa.Fliplr(0.5, random_state=STATE, )),  # horizontally flip 50% of the images

            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                rotate=(-25, 25),
                shear=(-8, 8),
                random_state=STATE,
            ),
            # In some images distort local areas with varying strength.
            # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
            # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.10))),
            # iaa.PiecewiseAffine(scale=(0.01, 0.05)),
            iaa.PerspectiveTransform(scale=(0.01, 0.10), random_state=STATE, ),
        ], random_order=True)
        return seq_shape

    @staticmethod
    def get_seq_val():
        sometimes = lambda aug: iaa.Sometimes(0.5, aug, random_state=STATE, )
        seq_val = iaa.Sequential([
            # iaa.CoarseDropout((0.1, 0.3), size_percent=(0.0, 0.2)),
            # In some images move pixels locally around (with random strengths).
            iaa.OneOf([
                sometimes(iaa.GaussianBlur(sigma=(0.1, 1), random_state=STATE, )),
                # blur images with a sigma of 0 to 3.0
                # iaa.Sometimes(.5, iaa.AverageBlur(k=(3, 7))),
                # iaa.Sometimes(.5, iaa.MotionBlur(k=(3, 7))),
                # iaa.Sometimes(.5, iaa.AveragePooling((2, 8))),
                # iaa.Sometimes(.5, iaa.MaxPooling((2, 8))),
                # iaa.Sometimes(.5, iaa.MedianPooling((2, 8))),
                # iaa.Sometimes(.5, iaa.MinPooling((2, 8))),
            ]),
            # ciaa.CoarseSaltAndPepper(p=.1, size_percent=(.01, .1)),

            # iaa.OneOf([
            #     ciaa.CoarseSaltAndPepper(p=.1, size_percent=(.01, .1)),
            #     ciaa.CoarseSaltAndPepper(p=.2, size_percent=(.4, .6)),
            # ]),

            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((.75, 1.5), random_state=STATE, ),
            # iaa.LinearContrast((.25, 1.75), random_state=STATE, ),
            # iaa.Multiply((0.8, 1.2)),
        ], random_order=True)
        return seq_val

    @staticmethod
    def get_seq_val1():
        seq_val = iaa.Sequential([
            # Strengthen or weaken the contrast in each image.
            iaa.OneOf([
                iaa.Sometimes(.5, iaa.GaussianBlur(sigma=(0.1, 1), random_state=STATE)),
                # blur images with a sigma of 0 to 3.0
                # iaa.Sometimes(.5, iaa.AverageBlur(k=(3, 7))),
                # iaa.Sometimes(.5, iaa.MotionBlur(k=(3, 7))),
                # iaa.Sometimes(.5, iaa.AveragePooling((2, 8))),
                # iaa.Sometimes(.5, iaa.MaxPooling((2, 8))),
                # iaa.Sometimes(.5, iaa.MedianPooling((2, 8))),
                # iaa.Sometimes(.5, iaa.MinPooling((2, 8))),
            ]),
            iaa.LinearContrast((1.1, 1.5), random_state=STATE),
            # iaa.Multiply((1.1, 1.3)),
        ], random_order=True)
        return seq_val

    @staticmethod
    def get_seq_val2():
        seq_val = iaa.Sequential([
            # Strengthen or weaken the contrast in each image.
            iaa.OneOf([
                iaa.Sometimes(.5, iaa.GaussianBlur(sigma=(0.1, 1), random_state=STATE)),
                # blur images with a sigma of 0 to 3.0
                # iaa.Sometimes(.5, iaa.AverageBlur(k=(3, 7))),
                # iaa.Sometimes(.5, iaa.MotionBlur(k=(3, 7))),
                # iaa.Sometimes(.5, iaa.AveragePooling((2, 8))),
                # iaa.Sometimes(.5, iaa.MaxPooling((2, 8))),
                # iaa.Sometimes(.5, iaa.MedianPooling((2, 8))),
                # iaa.Sometimes(.5, iaa.MinPooling((2, 8))),
            ]),
            iaa.LinearContrast((.5, .9), random_state=STATE),
            # iaa.Multiply((0.7, .9)),
        ], random_order=True)
        return seq_val


def blur_density_map(_map, mask, d_range):
    tmp = _map.copy()
    tmp = tmp[..., np.newaxis] if tmp.ndim == 2 else tmp
    tmp[mask == 0] = 0
    # blurred_pre_corrected = gaussian_filter(tmp, sigma=9,
    #                                         truncate=1)  # gaussian_filter(joint[-3], sigma=2, truncate=4.5)
    blurred_pre_corrected = blur_density_maps(tmp)
    # corrector = gaussian_filter(mask, sigma=9, truncate=1)
    corrector = blur_density_maps(mask)
    corrector[mask == 0] = 1
    blurred = blurred_pre_corrected / corrector
    blurred[mask == 0] = d_range[0]  # masking with whole prostate mask
    if _map.ndim == 2:
        return blurred.squeeze()
    return blurred


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def transform_sup(arrays, is_val=False, input_size=256, not_augment_values=False,
                  density_range=None, margin=.0, current_it=0):
    """"""
    arrays = list(arrays)  # 5 first arrays

    density_range = [0, 1] if density_range is None else density_range
    num_input_channels = arrays[0].shape[-1]

    if arrays[0].ndim == 4:
        if is_val:
            sl_idx = 1
            margin = 0
        else:
            sl_idx = np.random.randint(3)
            margin += 1e-3 if sl_idx != 1 else 0
        arrays[0] = arrays[0][..., sl_idx, :]

    # First, split images with more than 1 channels into separate images
    _arrays = []
    for i in range(len(arrays)):
        for j in range(arrays[i].shape[-1]):
            _arrays.append(arrays[i][..., j:j + 1])
    joint = np.asarray(_arrays)

    # Crop input to expected input size
    joint = just_crop(joint, input_size=input_size)

    if not is_val:
        image_aug = np.zeros(shape=(joint[0].shape[0], joint[0].shape[1], num_input_channels), dtype='float32')

        # Combine wp and ROI masks
        segmap = (joint[-2] + joint[-1]).astype(np.int8)
        segmap = SegmentationMapsOnImage(segmap, shape=joint[0].shape)

        # Density map
        # heatmap = blur_density_map(joint[-3], joint[-1], density_range) if joint[-3].min() < 2 else joint[-3] - 3  # -3 for pseudo-label
        heatmaps = []
        for i in range(2, len(joint) - 2):
            heatmap = blur_density_map(joint[i], joint[-1], density_range)
            heatmap = HeatmapsOnImage(
                arr=heatmap,
                shape=joint[0].shape,
                min_value=density_range[0],
                max_value=density_range[1]
            )
            heatmaps.append(heatmap)
        # Create augmentation sequences (shape + value)
        augmenter = Augmenter()  # always create the Augmenter object first
        seq_shape = augmenter.seq_shape
        seq_val = augmenter.seq_val

        # Augmentation (1st input channel + masks + density map)
        image_aug_shape = image_aug.copy()
        image_aug_shape[..., 0:1], segmap_aug = seq_shape(image=joint[0], segmentation_maps=segmap, )
        for i, heatmap in enumerate(heatmaps):
            heatmaps[i] = seq_shape(heatmaps=heatmap)
        image_aug[..., 0:1] = seq_val(image=image_aug_shape[..., 0:1])

        # Augment all extra input channels
        for i in range(1, image_aug.shape[-1]):
            image_aug_shape[..., i:i + 1] = seq_shape(image=joint[i])
            image_aug[..., i:i + 1] = seq_val(image=image_aug_shape[..., i:i + 1])

        # Retrieve masks
        m = (segmap_aug.get_arr() == 2).astype(int).astype('float32')
        wp = (segmap_aug.get_arr() > 0).astype(int).astype('float32')

        # Masking augmented density map
        heatmap_aug = np.asarray([heatmap_aug.get_arr() for heatmap_aug in heatmaps])
        heatmap_aug = heatmap_aug.squeeze().transpose([1, 2, 0])
        heatmap_aug[np.tile(wp, heatmap_aug.shape[-1]) == 0] = density_range[0]

        # # For classification loss
        # qm = np.ones_like(heatmap_aug)
        # ql = np.concatenate((np.arange(-1, 1, .125), np.array([1.])))[np.newaxis, np.newaxis,]
        # qml = qm * ql
        # qml_gt = np.abs(qml - heatmap_aug).argmin(axis=2, )[..., np.newaxis] * wp

        """For checking qml and qml_gt"""
        # fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        # ax[0].imshow(heatmap_aug[..., 0], cmap='gray', vmin=-1, vmax=1)
        # ax[0].set_title('Ground Truth (EPI)')
        # ax[0].contour(m[..., 0], colors='r')
        # ax[1].imshow(qml_gt[..., 0], cmap='gray')
        # ax[1].set_title('Mapped Ground Truth (EPI)')
        # ax[1].contour(m[..., 0], colors='r')
        # plt.subplots_adjust(0, 0, 1, 1, 0, 0)

        # randomly remove (zero out) one channel of image_aug (only if the image has all three channels)
        # if np.random.rand() > .5:
        # if (np.all(np.sum(image_aug, axis=(0, 1)))) and (image_aug.shape[-1] == 3):
        #     image_aug[..., np.random.random_integers(image_aug.shape[-1]) - 1] = np.zeros_like(
        #         image_aug[..., np.random.random_integers(image_aug.shape[-1]) - 1])

        # exp_idx = 1. * (current_it // 400)
        # bins = np.concatenate((np.arange(-1, 1, .125 * (2 ** -exp_idx)), np.array([1.])))
        # heatmap_aug = bins[np.digitize(heatmap_aug, bins, right=True)]

        # show_all(arrays[0], image_aug, segmap, segmap_aug, heatmap, heatmap_aug)
        return (
            image_aug_shape,  # input A
            heatmap_aug,  # ground truth of A
            m,  # ROI mask
            wp,  # whole prostate mask
            image_aug,  # augmented input A
            np.array(margin)[np.newaxis, np.newaxis, np.newaxis],
        )
    else:
        # In case of validation, only need to blur the density map & double the input arrays
        arrays = list(arrays)
        # tmp1 = arrays[-3].copy()  # Have to create a copy, otherwise the ground truth will be overwritten
        # for i in range(arrays[-3].shape[-1]):
        #     tmp1[..., i] = blur_density_map(tmp1[..., i], joint[-1], density_range)
        # arrays[-3] = tmp1.copy()

        arrays[-1] = np.tile(arrays[-1], arrays[-3].shape[-1])

        # print(arrays[1].sum(axis=(0, 1)))
        return tuple(arrays) + tuple(arrays[0][np.newaxis])


def transform_unsup(arrays, input_size=256, margin=.0, num_channels_out=4, **kwargs):
    """"""
    num_input_channels = arrays[0].shape[-1]
    # First, split images with more than 1 channels into separate images
    _arrays = []
    for i in range(len(arrays)):
        for j in range(arrays[i].shape[-1]):
            _arrays.append(arrays[i][..., j:j + 1])
    joint = np.asarray(_arrays)

    # Crop input to expected input size
    joint = just_crop(joint, input_size=input_size)

    # Combine wp and ROI masks
    segmap = SegmentationMapsOnImage(joint[-1].astype(np.int8), shape=joint[0].shape)

    """2 cases of augmentation:
        1/ No shape augmentation
        2/ Shape augmentation + Value augmentation
        non_val_aug: image with shape augmentation only
        val_aug: non_val_aug with value augmentation only
    """
    non_val_aug = np.zeros(shape=(joint[0].shape[0], joint[0].shape[1], num_input_channels), dtype='float32')
    val_aug1 = np.zeros(shape=(joint[0].shape[0], joint[0].shape[1], num_input_channels), dtype='float32')
    val_aug2 = np.zeros_like(val_aug1)

    # Create augmentation sequence
    # case_prob = np.random.rand()
    augmenter = Augmenter()  # Create Augmenter object (including separate sequences of shape and value augmentations)
    seq_shape = augmenter.seq_shape  # shared shape augmenter
    seq_val1 = augmenter.seq_val1  # specific val augmenter
    seq_val2 = Augmenter().seq_val2  # specific val augmenter

    # Augmentation (1st input channel + masks)
    non_val_aug[..., 0:1], segmap_aug = seq_shape(image=joint[0], segmentation_maps=segmap)
    val_aug1[..., 0:1] = seq_val1(image=non_val_aug[..., 0:1].astype('float32'))
    val_aug2[..., 0:1] = seq_val2(image=non_val_aug[..., 0:1].astype('float32'))

    # Augment all extra input channels
    for i in range(1, non_val_aug.shape[-1]):
        non_val_aug[..., i:i + 1] = seq_shape(image=joint[i])
        val_aug1[..., i:i + 1] = seq_val1(image=non_val_aug[..., i:i + 1].astype('float32'))
        val_aug2[..., i:i + 1] = seq_val2(image=non_val_aug[..., i:i + 1].astype('float32'))

    # Retrieve masks
    wp_aug = (segmap_aug.get_arr() > 0).astype(int).astype('float32')
    # view_image(val_aug1, val_aug2, wp_aug)

    return (
        val_aug1,  # input A
        np.fliplr(val_aug2),  # augmented input A
        np.tile(wp_aug, num_channels_out),  # augmented whole prostate mask
        np.array(margin)[np.newaxis, np.newaxis, np.newaxis],
        Augmenter().seq_noop,
    )


def transform_unsup_shape_diff(arrays, input_size=256, margin=.0, num_channels_out=4, **kwargs):
    """"""
    num_input_channels = arrays[0].shape[-1]
    # First, split images with more than 1 channels into separate images
    _arrays = []
    for i in range(len(arrays)):
        for j in range(arrays[i].shape[-1]):
            _arrays.append(arrays[i][..., j:j + 1])
    joint = np.asarray(_arrays)

    # Crop input to expected input size
    joint = just_crop(joint, input_size=input_size)

    # Combine wp and ROI masks
    segmap = SegmentationMapsOnImage(joint[-1].astype(np.int8), shape=joint[0].shape)

    """2 cases of augmentation:
        1/ No shape augmentation
        2/ Shape augmentation + Value augmentation
        non_val_aug: image with shape augmentation only
        val_aug: non_val_aug with value augmentation only
    """
    non_val_aug = np.zeros(shape=(joint[0].shape[0], joint[0].shape[1], num_input_channels), dtype='float32')
    val_aug1 = np.zeros(shape=(joint[0].shape[0], joint[0].shape[1], num_input_channels), dtype='float32')
    val_aug2 = np.zeros_like(val_aug1)

    # Create augmentation sequence
    # case_prob = np.random.rand()
    augmenter = Augmenter()  # Create Augmenter object (including separate sequences of shape and value augmentations)
    seq_shape = augmenter.seq_shape  # shared shape augmenter
    seq_val1 = augmenter.seq_val1  # specific val augmenter
    seq_val2 = Augmenter().seq_val2  # specific val augmenter

    # Augmentation (1st input channel + masks)
    non_val_aug[..., 0:1], segmap_aug = seq_shape(image=joint[0], segmentation_maps=segmap)
    val_aug1[..., 0:1] = seq_val1(image=non_val_aug[..., 0:1].astype('float32'))
    val_aug2[..., 0:1] = seq_val2(image=joint[0].astype('float32'))  # pre - shape-augmentation

    # Augment all extra input channels
    for i in range(1, non_val_aug.shape[-1]):
        non_val_aug[..., i:i + 1] = seq_shape(image=joint[i])
        val_aug1[..., i:i + 1] = seq_val1(image=non_val_aug[..., i:i + 1].astype('float32'))
        val_aug2[..., i:i + 1] = seq_val2(image=joint[i].astype('float32'))

    # Retrieve masks
    wp_aug = (segmap_aug.get_arr() > 0).astype(int).astype('float32')

    # view_image(val_aug1, val_aug2, wp_aug)
    return (
        val_aug1,  # input A
        np.fliplr(val_aug2),  # augmented input A
        np.tile(wp_aug, num_channels_out),  # augmented whole prostate mask
        np.array(margin)[np.newaxis, np.newaxis, np.newaxis],
        seq_shape,
    )


def transform_unsup_shape_diff_for_embedding(arrays, input_size=256, margin=.0, **kwargs):
    """"""
    num_input_channels = arrays[0].shape[-1]
    # First, split images with more than 1 channels into separate images
    _arrays = []
    for i in range(len(arrays)):
        for j in range(arrays[i].shape[-1]):
            _arrays.append(arrays[i][..., j:j + 1])
    joint = np.asarray(_arrays)

    # Crop input to expected input size
    joint = just_crop(joint, input_size=input_size)

    # Combine wp and ROI masks
    segmap = SegmentationMapsOnImage(joint[-1].astype(np.int8), shape=joint[0].shape)

    """2 cases of augmentation:
        1/ No shape augmentation
        2/ Shape augmentation + Value augmentation
        non_val_aug: image with shape augmentation only
        val_aug: non_val_aug with value augmentation only
    """
    non_val_aug1 = np.zeros(shape=(joint[0].shape[0], joint[0].shape[1], num_input_channels), dtype='float32')
    non_val_aug2 = np.zeros_like(non_val_aug1)
    val_aug1 = np.zeros(shape=(joint[0].shape[0], joint[0].shape[1], num_input_channels), dtype='float32')
    val_aug2 = np.zeros_like(val_aug1)

    # Create augmentation sequence
    # case_prob = np.random.rand()
    seq_shape1 = Augmenter().seq_shape  # independent shape augmenter
    seq_shape2 = Augmenter().seq_shape  # independent shape augmenter
    seq_val1 = Augmenter().seq_val1  # independent val augmenter
    seq_val2 = Augmenter().seq_val2  # independent val augmenter

    # Augmentation (1st input channel + masks)
    non_val_aug1[..., 0:1], segmap_aug = seq_shape1(image=joint[0], segmentation_maps=segmap)
    non_val_aug2[..., 0:1], segmap_aug = seq_shape2(image=joint[0], segmentation_maps=segmap)
    val_aug1[..., 0:1] = seq_val1(image=non_val_aug1[..., 0:1].astype('float32'))
    val_aug2[..., 0:1] = seq_val2(image=non_val_aug2[..., 0:1].astype('float32'))  # pre - shape-augmentation

    # Augment all extra input channels
    for i in range(1, non_val_aug1.shape[-1]):
        non_val_aug1[..., i:i + 1] = seq_shape1(image=joint[i])
        non_val_aug2[..., i:i + 1] = seq_shape2(image=joint[i])
        val_aug1[..., i:i + 1] = seq_val1(image=non_val_aug1[..., i:i + 1].astype('float32'))
        val_aug2[..., i:i + 1] = seq_val2(image=non_val_aug2[..., i:i + 1].astype('float32'))

    # Retrieve masks
    wp_aug = (segmap_aug.get_arr() > 0).astype(int).astype('float32')

    # view_image(val_aug1, val_aug2, wp_aug)
    return (
        val_aug1,  # input A
        np.fliplr(val_aug2),  # augmented input A
        wp_aug,  # augmented whole prostate mask
        np.array(margin)[np.newaxis, np.newaxis, np.newaxis],
        seq_shape1,
    )


def joint_transform(arrays, is_val=False, input_size=256, not_augment_values=False, density_range=None, margin=0,
                    get_unsup=False, current_it=0):
    """"""
    if get_unsup:
        # return transform_unsup(arrays, input_size=input_size, margin=0)
        # if np.random.rand() >= .5:
        #     return transform_unsup(arrays, input_size=input_size, margin=0)
        # else:
        return transform_unsup_shape_diff(arrays, input_size=input_size, margin=0)
        # return transform_unsup_shape_diff_for_embedding(arrays, input_size=input_size, margin=0)
    else:
        return transform_sup(arrays, is_val=is_val, input_size=input_size, not_augment_values=not_augment_values,
                             density_range=density_range,
                             margin=margin, current_it=current_it)


def show_all(image, image_aug, segmap, segmap_aug, heatmap, heatmap_aug):
    plt.close('all')
    ia.imshow(
        np.vstack((
            np.vstack((
                [*[np.hstack((
                    image[..., i],
                    image_aug[..., i],
                )) for i in range(image.shape[-1])]])),
            np.hstack((
                segmap.get_arr(),
                segmap_aug.get_arr(),
            )),
            np.hstack((
                heatmap.get_arr()[..., 0],
                heatmap_aug[..., 0],
            )),
        ))
    )


def show_pair(non_val_aug, val_aug, channel=0):
    ia.imshow(np.hstack((
        non_val_aug[..., channel],
        val_aug[..., channel]
    )))
