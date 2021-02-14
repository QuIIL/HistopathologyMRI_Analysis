from imgaug import parameters as iap
import imgaug as ia
from utils.custom_arithmetic import ReplaceElementwise, MultiplyElementwise


def CoarseSaltAndPepper(p=0, size_px=None, size_percent=None, per_channel=False, min_size=4, name=None, deterministic=False, random_state=None):
    """
    Adds coarse salt and pepper noise to an image, i.e. rectangles that contain noisy white-ish and black-ish pixels.

    TODO replace dtype support with uint8 only, because replacement is geared towards that value range

    dtype support::

        See ``imgaug.augmenters.arithmetic.ReplaceElementwise``.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of changing a pixel to salt/pepper noise.

            * If a float, then that value will be used for all images as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled per image
              from the range ``a <= x <= b``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then this parameter will be used as
              the *mask*, i.e. it is expected to contain values between
              0.0 and 1.0, where 1.0 means that salt/pepper is to be added
              at that location.

    size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the noise
        mask in absolute pixel dimensions.

            * If an integer, then that size will be used for both height and
              width. E.g. a value of 3 would lead to a ``3x3`` mask, which is then
              upsampled to ``HxW``, where ``H`` is the image size and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be sampled from the
              range ``[a..b]`` and the mask will be generated at size ``MxN``, then
              upsampled to ``HxW``.
            * If a StochasticParameter, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the noise
        mask *in percent* of the input image.

            * If a float, then that value will be used as the percentage of the
              height and width (relative to the original size). E.g. for value
              p, the mask will be sampled from ``(p*H)x(p*W)`` and later upsampled
              to ``HxW.``
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be sampled from the
              interval ``(a, b)`` and used as the percentages, i.e the mask size
              will be ``(m*H)x(n*W)``.
            * If a StochasticParameter, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float, optional
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    min_size : int, optional
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a 1x1 low resolution mask, leading easily
        to the whole image being replaced.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    aug = iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1))

    Replaces 5 percent of all pixels with salt/pepper in an image that has
    1 to 10 percent of the input image size, then upscales the results
    to the input image size, leading to large rectangular areas being replaced.

    """
    mask = iap.handle_probability_param(p, "p", tuple_to_uniform=True, list_to_choice=True)

    if size_px is not None:
        mask_low = iap.FromLowerResolution(other_param=mask, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        mask_low = iap.FromLowerResolution(other_param=mask, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    replacement = iap.Beta(0.5, 0.5) * 1

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return ReplaceElementwise(
        mask=mask_low,
        replacement=replacement,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state,
        use_roi=True
    )


def Dropout(p=0, per_channel=False, name=None, deterministic=False, random_state=None):
    """
    Augmenter that sets a certain fraction of pixels in images to zero.

    dtype support::

        See ``imgaug.augmenters.arithmetic.MultiplyElementwise``.

    Parameters
    ----------
    p : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The probability of any pixel being dropped (i.e. set to zero).

            * If a float, then that value will be used for all images. A value
              of 1.0 would mean that all pixels will be dropped and 0.0 that
              no pixels would be dropped. A value of 0.05 corresponds to 5
              percent of all pixels dropped.
            * If a tuple ``(a, b)``, then a value p will be sampled from the
              range ``a <= p <= b`` per image and be used as the pixel's dropout
              probability.
            * If a StochasticParameter, then this parameter will be used to
              determine per pixel whether it should be dropped (sampled value
              of 0) or shouldn't (sampled value of 1).
              If you instead want to provide the probability as a stochastic
              parameter, you can usually do ``imgaug.parameters.Binomial(1-p)``
              to convert parameter `p` to a 0/1 representation.

    per_channel : bool or float, optional
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.Dropout(0.02)

    drops 2 percent of all pixels.

    >>> aug = iaa.Dropout((0.0, 0.05))

    drops in each image a random fraction of all pixels, where the fraction
    is in the range ``0.0 <= x <= 0.05``.

    >>> aug = iaa.Dropout(0.02, per_channel=True)

    drops 2 percent of all pixels in a channel-wise fashion, i.e. it is unlikely
    for any pixel to have all channels set to zero (black pixels).

    >>> aug = iaa.Dropout(0.02, per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """
    if ia.is_single_number(p):
        p2 = iap.Binomial(1 - p)
    elif ia.is_iterable(p):
        ia.do_assert(len(p) == 2)
        ia.do_assert(p[0] < p[1])
        ia.do_assert(0 <= p[0] <= 1.0)
        ia.do_assert(0 <= p[1] <= 1.0)
        p2 = iap.Binomial(iap.Uniform(1 - p[1], 1 - p[0]))
    elif isinstance(p, iap.StochasticParameter):
        p2 = p
    else:
        raise Exception("Expected p to be float or int or StochasticParameter, got %s." % (type(p),))

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return MultiplyElementwise(p2, per_channel=per_channel, name=name, deterministic=deterministic,
                               random_state=random_state)


def CoarseDropout(p=0, size_px=None, size_percent=None, per_channel=False, min_size=4, name=None, deterministic=False,
                  random_state=None, mask=None):
    """
    Augmenter that sets rectangular areas within images to zero.

    In contrast to Dropout, these areas can have larger sizes.
    (E.g. you might end up with three large black rectangles in an image.)
    Note that the current implementation leads to correlated sizes,
    so when there is one large area that is dropped, there is a high likelihood
    that all other dropped areas are also large.

    This method is implemented by generating the dropout mask at a
    lower resolution (than the image has) and then upsampling the mask
    before dropping the pixels.

    dtype support::

        See ``imgaug.augmenters.arithmetic.MultiplyElementwise``.

    Parameters
    ----------
    p : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The probability of any pixel being dropped (i.e. set to zero).

            * If a float, then that value will be used for all pixels. A value
              of 1.0 would mean, that all pixels will be dropped. A value of
              0.0 would lead to no pixels being dropped.
            * If a tuple ``(a, b)``, then a value p will be sampled from the
              range ``a <= p <= b`` per image and be used as the pixel's dropout
              probability.
            * If a StochasticParameter, then this parameter will be used to
              determine per pixel whether it should be dropped (sampled value
              of 0) or shouldn't (sampled value of 1).

    size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the dropout
        mask in absolute pixel dimensions.

            * If an integer, then that size will be used for both height and
              width. E.g. a value of 3 would lead to a ``3x3`` mask, which is then
              upsampled to ``HxW``, where ``H`` is the image size and W the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be sampled from the
              range ``[a..b]`` and the mask will be generated at size ``MxN``, then
              upsampled to ``HxW``.
            * If a StochasticParameter, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the dropout
        mask *in percent* of the input image.

            * If a float, then that value will be used as the percentage of the
              height and width (relative to the original size). E.g. for value
              p, the mask will be sampled from ``(p*H)x(p*W)`` and later upsampled
              to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be sampled from the
              interval ``(a, b)`` and used as the percentages, i.e the mask size
              will be ``(m*H)x(n*W)``.
            * If a StochasticParameter, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float, optional
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    min_size : int, optional
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a ``1x1`` low resolution mask, leading easily
        to the whole image being dropped.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.CoarseDropout(0.02, size_percent=0.5)

    drops 2 percent of all pixels on an lower-resolution image that has
    50 percent of the original image's size, leading to dropped areas that
    have roughly 2x2 pixels size.


    >>> aug = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.05, 0.5))

    generates a dropout mask at 5 to 50 percent of image's size. In that mask,
    0 to 5 percent of all pixels are dropped (random per image).

    >>> aug = iaa.CoarseDropout((0.0, 0.05), size_px=(2, 16))

    same as previous example, but the lower resolution image has 2 to 16 pixels
    size.

    >>> aug = iaa.CoarseDropout(0.02, size_percent=0.5, per_channel=True)

    drops 2 percent of all pixels at 50 percent resolution (2x2 sizes)
    in a channel-wise fashion, i.e. it is unlikely
    for any pixel to have all channels set to zero (black pixels).

    >>> aug = iaa.CoarseDropout(0.02, size_percent=0.5, per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """
    if ia.is_single_number(p):
        p2 = iap.Binomial(1 - p)
    elif ia.is_iterable(p):
        ia.do_assert(len(p) == 2)
        ia.do_assert(p[0] < p[1])
        ia.do_assert(0 <= p[0] <= 1.0)
        ia.do_assert(0 <= p[1] <= 1.0)
        p2 = iap.Binomial(iap.Uniform(1 - p[1], 1 - p[0]))
    elif isinstance(p, iap.StochasticParameter):
        p2 = p
    else:
        raise Exception("Expected p to be float or int or StochasticParameter, got %s." % (type(p),))

    if size_px is not None:
        p3 = iap.FromLowerResolution(other_param=p2, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        p3 = iap.FromLowerResolution(other_param=p2, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return MultiplyElementwise(p3, per_channel=per_channel, name=name, deterministic=deterministic,
                               random_state=random_state, mask=mask)