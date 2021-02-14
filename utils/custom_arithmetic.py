import numpy as np
from imgaug.augmenters import meta
from imgaug import dtypes as iadt
from imgaug import parameters as iap
import imgaug as ia
from skimage.measure import regionprops
import pylab as plt
import cv2


class ReplaceElementwise(meta.Augmenter):
    """
    Replace pixels in an image with new values.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: no (1)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no
        * ``bool``: yes; tested

        - (1) uint64 is currently not supported, because iadt.clip_to_dtype_value_range_() does not
              support it, which again is because numpy.clip() seems to not support it.

    Parameters
    ----------
    mask : float or tuple of float or list of float or imgaug.parameters.StochasticParameter
        Mask that indicates the pixels that are supposed to be replaced.
        The mask will be thresholded with 0.5. A value of 1 then indicates a
        pixel that is supposed to be replaced.

            * If this is a float, then that value will be used as the
              probability of being a 1 per pixel.
            * If a tuple ``(a, b)``, then the probability will be sampled per image
              from the range ``a <= x <= b``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then this parameter will be used to
              sample a mask.

    replacement : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        The replacement to use at all locations that are marked as `1` in the mask.

            * If this is a number, then that value will always be used as the
              replacement.
            * If a tuple ``(a, b)``, then the replacement will be sampled pixelwise
              from the range ``a <= x <= b``.
            * If a list of number, then a random value will be picked from
              that list as the replacement per pixel.
            * If a StochasticParameter, then this parameter will be used sample
              pixelwise replacement values.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    # >>> aug = ReplaceElementwise(0.05, [0, 255])

    Replace 5 percent of all pixels in each image by either 0 or 255.

    """

    def __init__(self, mask, replacement, per_channel=False, name=None, deterministic=False, random_state=None, use_roi=False, bbox_shrink_dim=10):
        super(ReplaceElementwise, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.mask = iap.handle_probability_param(mask, "mask", tuple_to_uniform=True, list_to_choice=True)
        self.replacement = iap.handle_continuous_param(replacement, "replacement")
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")
        self.use_roi = use_roi
        self.bbox_shrink_dim = bbox_shrink_dim

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["bool", "uint8", "uint16", "uint32", "int8", "int16", "int32", "int64",
                                  "float16", "float32", "float64"],
                         disallowed=["uint64", "uint128", "uint256", "int64", "int128", "int256",
                                     "float96", "float128", "float256"],
                         augmenter=self)

        nb_images = len(images)
        rss = ia.derive_random_states(random_state, 2*nb_images+1)
        per_channel_samples = self.per_channel.draw_samples((nb_images,), random_state=rss[-1])

        gen = zip(images, per_channel_samples, rss[:-1:2], rss[1:-1:2])
        for image, per_channel_i, rs_mask, rs_replacement in gen:
            if self.use_roi:
                # Modify the original code by replacing the original sampling
                # shape (image shape), with prostate bounding box shape
                image_bin = (image[..., 0] > 0).astype(int)
                meas = regionprops(image_bin)
                bbox = np.array(meas[0].bbox)
                # Slightly shrink the bbox
                bbox[0:2] += self.bbox_shrink_dim
                bbox[-2:] -= self.bbox_shrink_dim
                # Create the sample mask
                height, width, nb_channels = bbox[-2] - bbox[0], bbox[-1] - bbox[1], image.shape[-1]
                sampling_shape = (height, width, nb_channels if per_channel_i > 0.5 else 1)
                mask_samples = self.mask.draw_samples(sampling_shape, random_state=rs_mask)
                # Paste the mask_samples to the ROI area
                mask_samples_roi = mask_samples.copy()
                mask_samples = np.zeros(image.shape, dtype=mask_samples.dtype)
                mask_samples[bbox[0]: bbox[-2], bbox[1]: bbox[-1], :] = mask_samples_roi
                mask_samples *= image_bin[..., np.newaxis]
            else:
                height, width, nb_channels = image.shape
                sampling_shape = (height, width, nb_channels if per_channel_i > 0.5 else 1)
                mask_samples = self.mask.draw_samples(sampling_shape, random_state=rs_mask)

            # This is slightly faster (~20%) for masks that are True at many locations, but slower (~50%) for masks
            # with few Trues, which is probably the more common use-case:
            # replacement_samples = self.replacement.draw_samples(sampling_shape, random_state=rs_replacement)
            #
            # # round, this makes 0.2 e.g. become 0 in case of boolean image (otherwise replacing values with 0.2 would
            # # lead to True instead of False).
            # if image.dtype.kind in ["i", "u", "b"] and replacement_samples.dtype.kind == "f":
            #     replacement_samples = np.round(replacement_samples)
            #
            # replacement_samples = iadt.clip_to_dtype_value_range_(replacement_samples, image.dtype, validate=False)
            # replacement_samples = replacement_samples.astype(image.dtype, copy=False)
            #
            # if sampling_shape[2] == 1:
            #     mask_samples = np.tile(mask_samples, (1, 1, nb_channels))
            #     replacement_samples = np.tile(replacement_samples, (1, 1, nb_channels))
            # mask_thresh = mask_samples > 0.5
            # image[mask_thresh] = replacement_samples[mask_thresh]

            if sampling_shape[2] == 1:
                mask_samples = np.tile(mask_samples, (1, 1, nb_channels))
            mask_thresh = mask_samples > 0.5
            replacement_samples = self.replacement.draw_samples((int(np.sum(mask_thresh)),),
                                                                random_state=rs_replacement)

            # round, this makes 0.2 e.g. become 0 in case of boolean image (otherwise replacing values with 0.2 would
            # lead to True instead of False).
            if image.dtype.kind in ["i", "u", "b"] and replacement_samples.dtype.kind == "f":
                replacement_samples = np.round(replacement_samples)

            replacement_samples = iadt.clip_to_dtype_value_range_(replacement_samples, image.dtype, validate=False)
            replacement_samples = replacement_samples.astype(image.dtype, copy=False)

            image[mask_thresh] = replacement_samples

        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.mask, self.replacement, self.per_channel]


# TODO merge with Multiply
class MultiplyElementwise(meta.Augmenter):
    """
    Multiply values of pixels with possibly different values for neighbouring pixels.

    While the Multiply Augmenter uses a constant multiplier per image,
    this one can use different multipliers per pixel.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: no
        * ``int64``: no
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: no
        * ``float128``: no
        * ``bool``: yes; tested

        Note: tests were only conducted for rather small multipliers, around -10.0 to +10.0.

        In general, the multipliers sampled from `mul` must be in a value range that corresponds to
        the input image's dtype. E.g. if the input image has dtype uint16 and the samples generated
        from `mul` are float64, this augmenter will still force all samples to be within the value
        range of float16, as it has the same number of bytes (two) as uint16. This is done to
        make overflows less likely to occur.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        The value by which to multiply the pixel values in the image.

            * If a number, then that value will always be used.
            * If a tuple ``(a, b)``, then a value from the range ``a <= x <= b`` will
              be sampled per image and pixel.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then that parameter will be used to
              sample a new value per image and pixel.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> from imgaug import augmenters as iaa
    >>> aug = iaa.MultiplyElementwise(2.0)

    multiply all images by a factor of 2.0, making them significantly
    bighter.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5))

    samples per pixel a value from the range ``0.5 <= x <= 1.5`` and
    multiplies the pixel with that value.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=True)

    samples per pixel *and channel* a value from the range
    ``0.5 <= x <= 1.5`` ands multiplies the pixel by that value. Therefore,
    added multipliers may differ between channels of the same pixel.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """

    def __init__(self, mul=1.0, per_channel=False, name=None, deterministic=False, random_state=None, bbox_shrink_dim=2,
                 mask=None):
        super(MultiplyElementwise, self).__init__(name=name, deterministic=deterministic, random_state=random_state,)

        self.mul = iap.handle_continuous_param(mul, "mul", value_range=None, tuple_to_uniform=True,
                                               list_to_choice=True)
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")
        self.bbox_shrink_dim = bbox_shrink_dim
        self.mask = mask

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["bool", "uint8", "uint16", "int8", "int16", "float16", "float32"],
                         disallowed=["uint32", "uint64", "uint128", "uint256", "int32", "int64", "int128", "int256",
                                     "float64", "float96", "float128", "float256"],
                         augmenter=self)

        input_dtypes = iadt.copy_dtypes_for_restore(images, force_list=True)

        nb_images = len(images)
        rss = ia.derive_random_states(random_state, nb_images+1)
        per_channel_samples = self.per_channel.draw_samples((nb_images,), random_state=rss[-1])
        is_mul_binomial = isinstance(self.mul, iap.Binomial) or (
            isinstance(self.mul, iap.FromLowerResolution) and isinstance(self.mul.other_param, iap.Binomial)
        )

        gen = enumerate(zip(images, per_channel_samples, rss[:-1], input_dtypes))
        for i, (image, per_channel_samples_i, rs, input_dtype) in gen:
            if self.mask is None:
                height, width, nb_channels = image.shape
                sample_shape = (height, width, nb_channels if per_channel_samples_i > 0.5 else 1)
                mul = self.mul.draw_samples(sample_shape, random_state=rs)
            else:
                ###############################################################
                # Modify the original code by replacing the original sampling
                # shape (image shape), with prostate bounding box shape
                image_bin = (self.mask[..., 0] > 0).astype(int)
                meas = regionprops(image_bin)
                bbox = np.array(meas[0].bbox)
                # Slightly shrink the bbox
                bbox[0:2] += self.bbox_shrink_dim
                bbox[-2:] -= self.bbox_shrink_dim
                # Create the sample mask
                height, width, nb_channels = bbox[-2] - bbox[0], bbox[-1] - bbox[1], image.shape[-1]
                sampling_shape = (height, width, nb_channels if per_channel_samples_i > 0.5 else 1)
                mul = self.mul.draw_samples(sampling_shape, random_state=rs)
                # Paste the mask_samples to the ROI area
                mask_samples_roi = mul.copy()
                mul = np.zeros(image.shape, dtype=mul.dtype)
                mul[bbox[0]: bbox[-2], bbox[1]: bbox[-1], :] = mask_samples_roi
                mul *= image_bin[..., np.newaxis]
                ###############################################################

            # TODO let Binomial return boolean mask directly instead of [0, 1] integers?
            # hack to improve performance for Dropout and CoarseDropout
            # converts mul samples to mask if mul is binomial
            if mul.dtype.kind != "b" and is_mul_binomial:
                mul = mul.astype(bool, copy=False)

            if mul.dtype.kind == "b":
                images[i] *= mul
            elif image.dtype.name == "uint8":
                # This special uint8 block is around 60-100% faster than the else-block further below (more speedup
                # for larger images).
                #
                if mul.dtype.kind == "f":
                    # interestingly, float32 is here significantly faster than float16
                    # TODO is that system dependent?
                    # TODO does that affect int8-int32 too?
                    mul = mul.astype(np.float32, copy=False)
                    image_aug = image.astype(np.float32)
                else:
                    mul = mul.astype(np.int16, copy=False)
                    image_aug = image.astype(np.int16)

                image_aug = np.multiply(image_aug, mul, casting="no", out=image_aug)
                images[i] = iadt.restore_dtypes_(image_aug, np.uint8, round=False)
            else:
                # TODO maybe introduce to stochastic parameters some way to get the possible min/max values,
                # could make things faster for dropout to get 0/1 min/max from the binomial
                mul_min = np.min(mul)
                mul_max = np.max(mul)
                is_not_increasing_value_range = (-1 <= mul_min <= 1) and (-1 <= mul_max <= 1)

                # We limit here the value range of the mul parameter to the bytes in the image's dtype.
                # This prevents overflow problems and makes it less likely that the image has to be up-casted, which
                # again improves performance and saves memory. Note that this also enables more dtypes for image inputs.
                # The downside is that the mul parameter is limited in its value range.
                itemsize = max(image.dtype.itemsize, 2 if mul.dtype.kind == "f" else 1)  # float min itemsize is 2
                dtype_target = np.dtype("%s%d" % (mul.dtype.kind, itemsize))
                mul = iadt.clip_to_dtype_value_range_(mul, dtype_target, validate=True,
                                                      validate_values=(mul_min, mul_max))

                if mul.shape[2] == 1:
                    mul = np.tile(mul, (1, 1, nb_channels))

                image, mul = iadt.promote_array_dtypes_(
                    [image, mul],
                    dtypes=[image, dtype_target],
                    increase_itemsize_factor=1 if is_not_increasing_value_range else 2)
                image = np.multiply(image, mul, out=image, casting="no")
                image = iadt.restore_dtypes_(image, input_dtype)
                images[i] = image

        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.mul, self.per_channel]