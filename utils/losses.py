# coding=utf-8
import numpy as np
from mxnet import gluon, nd


def compute_ssim(x, y, m, l=2):
    """Compute Structural Similarity Index"""
    C1 = (0.01 * l) ** 2
    C2 = (0.03 * l) ** 2
    mux = x.sum(axis=0, exclude=True, keepdims=True) / m.sum(axis=0, exclude=True, keepdims=True)
    muy = y.sum(axis=0, exclude=True, keepdims=True) / m.sum(axis=0, exclude=True, keepdims=True)
    sigx = nd.sqrt(
        nd.where(m > 0, nd.square(x - mux), m).sum(axis=0, exclude=True, keepdims=True) /
        m.sum(axis=0, exclude=True, keepdims=True))
    sigy = nd.sqrt(
        nd.where(m > 0, nd.square(y - muy), m).sum(axis=0, exclude=True, keepdims=True) /
        m.sum(axis=0, exclude=True, keepdims=True))
    sigxy = nd.sqrt(sigx ** 2 * sigy ** 2)
    num = (2 * mux * muy + C1) * (2 * sigxy + C2)
    den = (mux ** 2 + muy ** 2 + C1) * (sigx ** 2 + sigy ** 2 + C2)
    return num / den


class SoftmaxEmbeddingLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, temperature=0.1, **kwargs):
        super(SoftmaxEmbeddingLoss, self).__init__(weight, batch_axis, **kwargs)
        self.temperature = temperature

    def hybrid_forward(self, F, emb, emb_aug, **kwargs):
        emb = F.L2Normalization(emb)
        emb_aug = F.L2Normalization(emb_aug)
        emb_mat = F.concat(emb, emb_aug, dim=0)
        dp = F.exp(F.dot(emb_mat, emb_mat.T) / self.temperature)  # dot product

        concentrated_loss = 0
        for i in range(emb.shape[0]):
            numerator = dp[i, i + emb.shape[0]]
            denominator = F.concat(*[dp[k, i + emb.shape[0]] for k in range(len(dp))], dim=0).sum()
            concentrated_loss = concentrated_loss + F.log(numerator / denominator)

        spread_loss = 0
        for i in range(emb.shape[0]):
            for j in range(len(dp)):
                if (j == i) or (j == (i + emb.shape[0])):
                    continue
                numerator = dp[i, j]
                denominator = F.concat(*[dp[k, j] for k in range(len(dp))], dim=0).sum()
                spread_loss = spread_loss + F.log(1 - numerator / denominator)
        return -(concentrated_loss + spread_loss)


class L1Loss_v2(gluon.loss.Loss):
    r"""Calculates the mean absolute error between `pred` and `label`.

    .. math:: L = \sum_i \vert {pred}_i - {label}_i \vert.

    `pred` and `label` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, with_Charbonnier=False, with_DepthAware=False, eps=1e-8,
                 scale_invar=False, smooth_l1=False, **kwargs):
        super(L1Loss_v2, self).__init__(weight, batch_axis, **kwargs)
        self.with_Charbonnier = with_Charbonnier
        self.with_DepthAware = with_DepthAware
        self.eps = eps
        self.smooth_l1 = smooth_l1

    def hybrid_forward(self, F, pred, label, sample_weight=None, _margin=0):
        label = gluon.loss._reshape_like(F, label, pred)
        loss = F.abs(pred - label)

        if self.with_Charbonnier:
            loss = nd.sqrt(F.square(loss) + self.eps ** 2)
        if self.with_DepthAware:
            c = 1
            smooth_factor = 1e-4
            offset = 2 + smooth_factor
            lamda = 1 - nd.broadcast_minimum(nd.log((pred + offset) * c),
                                             nd.log((label + offset + self.eps) * c)) / nd.broadcast_maximum(
                nd.log((pred + offset + self.eps) * c), nd.log((label + offset + self.eps) * c))
            alpha = ((label + 1) / 2).copy() if label.min() < 0 else label.copy()
            loss = gluon.loss._apply_weighting(F, loss, self._weight, (lamda + alpha))
        # nd.smooth_l1()

        if _margin is not None:
            loss = F.relu(loss - _margin)

        n_sample = sample_weight.sum(axis=self._batch_axis, exclude=True)
        loss = gluon.loss._apply_weighting(F, loss, self._weight, sample_weight)
        loss = nd.smooth_l1(loss) if self.smooth_l1 else loss

        loss = loss.sum(axis=self._batch_axis, exclude=True) / n_sample
        return loss.mean()


class L2Loss_v2(gluon.loss.Loss):
    r"""Calculates the mean squared error between `pred` and `label`.

    .. math:: L = \frac{1}{2} \sum_i \vert {pred}_i - {label}_i \vert^2.

    `pred` and `label` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, with_Charbonnier=False, with_DepthAware=False, eps=1e-8,
                 scale_invar=False, **kwargs):
        super(L2Loss_v2, self).__init__(weight, batch_axis, **kwargs)
        self.with_Charbonnier = with_Charbonnier
        self.with_DepthAware = with_DepthAware
        self.eps = eps

    def hybrid_forward(self, F, pred, label, sample_weight=None, _margin=None):
        label = gluon.loss._reshape_like(F, label, pred)
        loss = F.square(pred - label)

        if self.with_Charbonnier:
            loss = nd.sqrt(F.square(loss) + self.eps ** 2)
        if self.with_DepthAware:
            c = 1
            smooth_factor = 1e-4
            offset = 2 + smooth_factor
            lamda = 1 - nd.broadcast_minimum(nd.log((pred + offset) * c),
                                             nd.log((label + offset + self.eps) * c)) / nd.broadcast_maximum(
                nd.log((pred + offset + self.eps) * c), nd.log((label + offset + self.eps) * c))
            alpha = ((label + 1) / 2).copy() if label.min() < 0 else label.copy()
            loss = gluon.loss._apply_weighting(F, loss, self._weight, (lamda + alpha))

        if _margin is not None:
            loss = F.relu(loss - _margin ** 2)

        # n_sample = (sample_weight > 0).sum(axis=self._batch_axis, exclude=True) * loss.shape[1]  # number of density types
        # loss = gluon.loss._apply_weighting(F, loss, self._weight, sample_weight)
        # loss = loss.sum(axis=self._batch_axis, exclude=True) / (n_sample + 1)
        # # return loss.mean()
        # # print(loss)
        # return loss.sum() / max((n_sample > 0).sum(), 1)

        n_sample = (sample_weight > 0).sum(axis=self._batch_axis, exclude=True)
        loss = gluon.loss._apply_weighting(F, loss, self._weight, sample_weight)
        loss = loss.sum(axis=self._batch_axis, exclude=True) / (n_sample + 1e-8)
        return loss.mean()


class L2LogLoss_(gluon.loss.Loss):
    r"""Calculates the mean squared error between `pred` and `label`.

    .. math:: L = \frac{1}{2} \sum_i \vert {pred}_i - {label}_i \vert^2.

    `pred` and `label` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, with_Charbonnier=True, with_DepthAware=False, eps=1e-8,
                 scale_invar=True,
                 **kwargs):
        super(L2LogLoss, self).__init__(weight, batch_axis, **kwargs)
        self.with_Charbonnier = with_Charbonnier
        self.with_DepthAware = with_DepthAware
        self.eps = eps
        self.scale_invar = scale_invar

    def hybrid_forward(self, F, pred, label, sample_weight=None, _margin=None):
        label = gluon.loss._reshape_like(F, label, pred)

        def compute(_alpha=None):
            _alpha = 0 if _alpha is None else nd.reshape(_alpha, [_alpha.shape[0], 1, 1, 1])
            loss = F.square(F.log(pred + 2) - F.log(label + 2) + _alpha)
            # loss = F.square(F.log(pred+1+1e-8) - F.log(label+1+1e-8) + _alpha)

            if self.with_Charbonnier:
                loss = nd.sqrt(F.square(loss) + self.eps)
            if _margin is not None:
                loss = F.relu(loss - F.square(_margin))

            n_sample = sample_weight.sum(axis=self._batch_axis, exclude=True)
            loss = gluon.loss._apply_weighting(F, loss, self._weight, sample_weight)
            loss = loss.sum(axis=self._batch_axis, exclude=True) / n_sample
            return loss

        if self.scale_invar:
            return compute(compute()).mean()
        else:
            return compute().mean()


class L2LogLoss(gluon.loss.Loss):
    r"""Calculates the mean squared error between `pred` and `label`.

    .. math:: L = \frac{1}{2} \sum_i \vert {pred}_i - {label}_i \vert^2.

    `pred` and `label` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, with_Charbonnier=False, with_DepthAware=False, eps=1e-8,
                 scale_invar=False, **kwargs):
        super(L2LogLoss, self).__init__(weight, batch_axis, **kwargs)
        self.with_Charbonnier = with_Charbonnier
        self.with_DepthAware = with_DepthAware
        self.eps = eps
        self.scale_invar = scale_invar

    def hybrid_forward(self, F, pred, label, sample_weight=None, _margin=None):
        label = gluon.loss._reshape_like(F, label, pred)

        def compute(_alpha=None):
            _alpha = 0 if _alpha is None else nd.reshape(_alpha, [_alpha.shape[0], 1, 1, 1])

            loss = F.square(F.log(pred + 2) - F.log(label + 2) + _alpha)  # *.5 baseline_v10

            if self.with_Charbonnier:
                loss = nd.sqrt(F.square(loss) + self.eps)
            if self.with_DepthAware:
                c = 1
                smooth_factor = 1e-4
                offset = 2 + smooth_factor
                lamda = 1 - nd.broadcast_minimum(nd.log((pred + offset) * c),
                                                 nd.log((label + offset + self.eps) * c)) / nd.broadcast_maximum(
                    nd.log((pred + offset + self.eps) * c), nd.log((label + offset + self.eps) * c))
                alpha = ((label + 1) / 2).copy() if label.min() < 0 else label.copy()
                loss = gluon.loss._apply_weighting(F, loss, self._weight, (lamda + alpha))
            # if _margin is not None:
            #     loss = F.relu(loss - _margin)

            n_sample = sample_weight.sum(axis=self._batch_axis, exclude=True)
            # n_sample = sample_weight.sum(axis=self._batch_axis, exclude=False)
            loss = gluon.loss._apply_weighting(F, loss, self._weight, sample_weight)
            loss = loss.sum(axis=self._batch_axis, exclude=True) / n_sample
            # loss = loss.sum(axis=self._batch_axis, exclude=False) / n_sample
            return loss

        if self.scale_invar:
            return compute(compute()).mean()
        else:
            return compute().mean()


class L1LogLoss(gluon.loss.Loss):
    r"""Calculates the mean squared error between `pred` and `label`.

    .. math:: L = \frac{1}{2} \sum_i \vert {pred}_i - {label}_i \vert^2.

    `pred` and `label` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, with_Charbonnier=True, with_DepthAware=False, eps=1e-8,
                 scale_invar=False,
                 **kwargs):
        super(L1LogLoss, self).__init__(weight, batch_axis, **kwargs)
        self.with_Charbonnier = with_Charbonnier
        self.with_DepthAware = with_DepthAware
        self.eps = eps
        self.scale_invar = scale_invar

    def hybrid_forward(self, F, pred, label, sample_weight=None, _margin=None):
        label = gluon.loss._reshape_like(F, label, pred)

        def compute(_alpha=None):
            _alpha = 0 if _alpha is None else nd.reshape(_alpha, [_alpha.shape[0], 1, 1, 1])
            loss = F.log(pred + 2) - F.log(label + 2) + .1 * _alpha

            if self.with_Charbonnier:
                loss = nd.sqrt(F.square(loss) + self.eps)
            if self.with_DepthAware:
                c = 100
                offset = 2
                lamda = 1 - nd.broadcast_minimum(nd.log((pred + offset) * c),
                                                 nd.log((label + offset) * c)) / nd.broadcast_maximum(
                    nd.log((pred + offset) * c), nd.log((label + offset) * c))
                alpha = ((label + 1) / 1).copy() if label.min() == -1 else label.copy()
                loss = gluon.loss._apply_weighting(F, loss, self._weight, (lamda + alpha))
            if _margin is not None:
                loss = F.relu(loss - _margin)

            n_sample = sample_weight.sum(axis=self._batch_axis, exclude=True)
            loss = gluon.loss._apply_weighting(F, loss, self._weight, sample_weight)
            loss = loss.sum(axis=self._batch_axis, exclude=True) / n_sample
            return loss

        if self.scale_invar:
            return compute(compute()).mean()
        else:
            return compute().mean()


class LogCoshLoss(gluon.loss.Loss):
    r"""Calculates the mean squared error between `pred` and `label`.

    .. math:: L = \frac{1}{2} \sum_i \vert {pred}_i - {label}_i \vert^2.

    `pred` and `label` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, with_Charbonnier=False, with_DepthAware=False, eps=1e-8,
                 scale_invar=False, **kwargs):
        super(LogCoshLoss, self).__init__(weight, batch_axis, **kwargs)
        self.with_Charbonnier = with_Charbonnier
        self.with_DepthAware = with_DepthAware
        self.eps = eps
        self.scale_invar = scale_invar

    def hybrid_forward(self, F, pred, label, sample_weight=None, _margin=None):
        label = gluon.loss._reshape_like(F, label, pred)
        pred = pred

        def compute(_alpha=None):
            _alpha = 0 if _alpha is None else nd.reshape(_alpha, [_alpha.shape[0], 1, 1, 1])

            loss = F.log(F.cosh(pred - label))

            if self.with_Charbonnier:
                loss = nd.sqrt(F.square(loss) + self.eps)
            if self.with_DepthAware:
                c = 1
                smooth_factor = 1e-4
                offset = 2 + smooth_factor
                lamda = 1 - nd.broadcast_minimum(nd.log((pred + offset) * c),
                                                 nd.log((label + offset + self.eps) * c)) / nd.broadcast_maximum(
                    nd.log((pred + offset + self.eps) * c), nd.log((label + offset + self.eps) * c))
                alpha = ((label + 1) / 2).copy() if label.min() < 0 else label.copy()
                loss = gluon.loss._apply_weighting(F, loss, self._weight, (lamda + alpha))
            if _margin is not None:
                loss = F.relu(loss - F.square(_margin))

            n_sample = sample_weight.sum(axis=self._batch_axis, exclude=True)
            # n_sample = sample_weight.sum(axis=self._batch_axis, exclude=False)
            loss = gluon.loss._apply_weighting(F, loss, self._weight, sample_weight)
            loss = loss.sum(axis=self._batch_axis, exclude=True) / n_sample
            # loss = loss.sum(axis=self._batch_axis, exclude=False) / n_sample
            return loss

        if self.scale_invar:
            return compute(compute()).mean()
        else:
            return compute().mean()


class L1Loss(gluon.loss.Loss):
    """SOMETHING IS WRONG WITH THIS FUNCTION!!!"""
    """manually implemented L1 loss"""

    def __init__(self, weight=None, with_Charbonnier=True, eps=1e-12, f=None, bins=None, is_train=True,
                 with_DepthAware=True, **kwargs):
        super(L1Loss, self).__init__(weight, batch_axis=0, **kwargs)
        self.with_DepthAware = with_DepthAware
        self.with_Charbonnier = with_Charbonnier
        self.eps = eps
        self.bins = bins
        self.f = f
        self.is_train = is_train

    def hybrid_forward(self, F, pred, label, sample_weight=None, distribution_weighting=None):
        """Forward"""
        if self.is_train:
            sample_weight = sample_weight.reshape((sample_weight.shape[0], -1)).astype('float32')
            pred = pred.reshape((pred.shape[0], -1))
            label = label.reshape((label.shape[0], -1))

            """sample_weight2 is the weighting by data distribution"""
            if (self.f is not None) and distribution_weighting:
                x_np = label.asnumpy()
                sample_weight2 = np.zeros(x_np.shape)
                for ibin in range(self.bins.__len__() - 1):
                    sample_weight2[np.logical_and(x_np <= self.bins[ibin + 1], x_np > self.bins[ibin])] = self.f[ibin]
                sample_weight2 = sample_weight2.reshape((label.shape[0], -1))
                # sample_weight2 /= sample_weight2.max()
            else:
                sample_weight2 = np.ones(pred.shape, dtype='float32')

            U = pred - label
            if self.with_Charbonnier:
                U = nd.sqrt(U ** 2 + self.eps ** 2)
            if self.with_DepthAware:
                lamda = 1 - nd.broadcast_minimum(nd.log(pred * 100 + 1e-10),
                                                 nd.log(label * 100 + 1e-10)) / nd.broadcast_maximum(
                    nd.log(pred * 100 + 1e-10), nd.log(label * 100 + 1e-10))
                alpha = label
                U = (lamda + alpha) * U

            loss = F.abs(U) * sample_weight.astype('float32') * nd.array(sample_weight2, ctx=pred.context)
            # In case there is no ROI
            return F.sum(loss, axis=self._batch_axis, exclude=True) / (
                    F.sum(sample_weight, axis=self._batch_axis, exclude=True) + 1e-8)
        else:
            dif = pred - label
            dif = dif * sample_weight
            return dif.abs().sum(axis=self._batch_axis, exclude=True) / (
                    sample_weight.sum(axis=self._batch_axis, exclude=True) + 1e-8)


class L1Loss_scalar(gluon.loss.Loss):
    """manually implemented L1 loss"""

    def __init__(self, weight=None, with_Charbonnier=True, eps=1e-12, f=None, bins=None, is_train=True, **kwargs):
        super(L1Loss_scalar, self).__init__(weight, batch_axis=0, **kwargs)
        self.with_Charbonnier = with_Charbonnier
        self.eps = eps
        self.bins = bins
        self.f = f
        self.is_train = is_train

    def hybrid_forward(self, F, pred, label, sample_weight=None, distribution_weighting=None):
        """Forward"""
        if self.is_train:
            if (self.f is not None) and distribution_weighting:
                x_np = label.asnumpy()
                sample_weight2 = np.zeros(x_np.shape)
                for ibin in range(self.bins.__len__() - 1):
                    sample_weight2[np.logical_and(x_np <= self.bins[ibin + 1], x_np > self.bins[ibin])] = self.f[ibin]
                sample_weight2 = sample_weight2.reshape((label.shape[0], -1))
                # sample_weight2 /= sample_weight2.max()
            else:
                sample_weight2 = np.ones(pred.shape, dtype='float32')

            U = pred - label
            if self.with_Charbonnier:
                U = nd.sqrt(U ** 2 + self.eps ** 2)
            loss = F.abs(U) * sample_weight.astype('float32') * nd.array(sample_weight2, ctx=pred.context)
            return (F.sum(loss, axis=self._batch_axis, exclude=True) / F.sum(sample_weight, axis=self._batch_axis,
                                                                             exclude=True)).mean()
        else:
            return ((pred - label).abs().sum(axis=self._batch_axis, exclude=True) /
                    (sample_weight.sum(axis=self._batch_axis, exclude=True))).mean()


class L2Loss(gluon.loss.Loss):
    def __init__(self, weight=None, **kwargs):
        super(L2Loss, self).__init__(weight, batch_axis=0, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None, distribution_weighting=None):
        is_train = False  # should be True (False for experiment)
        if sample_weight is None:
            sample_weight = pred[:, 1]
            pred = pred[:, 0]
            is_train = False

        sample_weight = sample_weight.reshape((sample_weight.shape[0], -1)).astype('float32')
        pred = pred.reshape((pred.shape[0], -1))
        label = label.reshape((label.shape[0], -1))

        if is_train and distribution_weighting:
            x_np = label.asnumpy()
            x = label.asnumpy().flatten()
            xf = x[sample_weight.asnumpy().flatten() == 1]
            bins = np.linspace(0, xf.max(), 10)
            f = 1 / (np.histogram(xf, bins)[0] / xf.shape[0])

            sample_weight2 = np.zeros(x_np.shape)
            for ibin in range(bins.__len__() - 1):
                sample_weight2[np.logical_and(x_np <= bins[ibin + 1], x_np > bins[ibin])] = f[ibin]
            sample_weight2 = sample_weight2.reshape((label.shape[0], -1))
            sample_weight2 /= sample_weight2.max()
        else:
            sample_weight2 = np.ones(pred.shape)

        loss = (pred - label) ** 2 * sample_weight.astype('float32') * nd.array(sample_weight2, ctx=pred.context)
        return (F.sum(loss, axis=self._batch_axis, exclude=True) / F.sum(sample_weight, axis=self._batch_axis,
                                                                         exclude=True)).mean()


class corrcoefLoss(gluon.loss.Loss):
    """correlation loss"""

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(corrcoefLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None, distribution_weighting=None, **kwargs):
        if sample_weight is None:
            sample_weight = nd.ones_like(pred)
        # sample_weight = nd.round(sample_weight)

        # r = np.zeros((pred.shape[0],))
        # for k in range(pred.shape[0]):
        #     m = sample_weight[k, 0].asnumpy()
        #     if m.sum() > 1:
        #         r[k] = np.corrcoef(pred[k, 0].asnumpy()[m == 1], label[k, 0].asnumpy()[m == 1])[0, 1]
        #     else:
        #         r[k] = 1
        # r[np.isnan(r)] = -1

        # sample_weight = sample_weight.reshape((-1, 1)).astype('float32')
        # # a = mx.nd.array(pred.reshape((-1, 1)).asnumpy()[sample_weight == 1], ctx=pred.context)
        # # b = mx.nd.array(label.reshape((-1, 1)).asnumpy()[sample_weight == 1], ctx=pred.context)

        # sample_weight = sample_weight.reshape((-1, 1)).astype('float32')
        # a = pred.reshape((-1, 1)) * sample_weight
        # b = label.reshape((-1, 1)) * sample_weight
        # n = sample_weight.sum()

        if sample_weight.sum() == 0:
            return nd.zeros(1, ).as_in_context(sample_weight.context)

        sample_weight = nd.array(np.where(sample_weight.asnumpy() == 1), ctx=sample_weight.context)

        a = nd.gather_nd(pred, sample_weight)
        b = nd.gather_nd(label, sample_weight)
        # sample_weight = sample_weight.reshape((pred.shape[0], -1)).astype('float32')
        n = a.__len__()

        r = ((n * (nd.sum(a * b)) - (nd.sum(a) * nd.sum(b))) / \
             nd.sqrt((n * nd.sum(a ** 2) - nd.sum(a) ** 2) * (n * nd.sum(b ** 2) - nd.sum(b) ** 2)))
        # r = nd.maximum(1e-1, r)

        # return nd.array(10 * (-r + 1))
        # return nd.array(-np.log(0.1 * (r + 1 + 1e-2)))
        # return nd.array(-nd.exp(r) + nd.exp(nd.ones_like(r)))
        # return -nd.square(r)  #* 8e-3
        return -(r ** 3)  # * 8e-3


class HuberLoss(gluon.loss.Loss):
    r"""Calculates smoothed L1 loss that is equal to L1 loss if absolute error
    exceeds rho but is equal to L2 loss otherwise. Also called SmoothedL1 loss.

    .. math::
        L = \sum_i \begin{cases} \frac{1}{2 {rho}} ({label}_i - {pred}_i)^2 &
                           \text{ if } |{label}_i - {pred}_i| < {rho} \\
                           |{label}_i - {pred}_i| - \frac{{rho}}{2} &
                           \text{ otherwise }
            \end{cases}

    `label` and `pred` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    rho : float, default 1
        Threshold for trimmed mean estimator.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, rho=1, weight=None, batch_axis=0, **kwargs):
        super(HuberLoss, self).__init__(weight, batch_axis, **kwargs)
        self._rho = rho

    def hybrid_forward(self, F, pred, label, sample_weight=None, _margin=None):
        label = gluon.loss._reshape_like(F, label, pred)
        loss = F.abs(label - pred)
        loss = F.where(loss > self._rho, loss - 0.5 * self._rho,
                       (0.5 / self._rho) * F.square(loss))
        n_sample = sample_weight.sum(axis=self._batch_axis, exclude=True)
        loss = gluon.loss._apply_weighting(F, loss, self._weight, sample_weight)
        loss = loss.sum(axis=self._batch_axis, exclude=True) / n_sample
        return loss.mean()
        # return F.mean(loss, axis=self._batch_axis, exclude=True)


class BerHuLoss(gluon.loss.Loss):
    """Loss that takes advantages from both L2 norm and L1 norm --> accelerated optimization and detailed structure,
    followed 'https://arxiv.org/pdf/1803.10039.pdf'"""

    def __init__(self, weight=None, f=None, bins=None, **kwargs):
        super(BerHuLoss, self).__init__(weight, batch_axis=0, **kwargs)
        self.bins = bins
        self.f = f

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        """Forward"""
        if sample_weight is None:
            sample_weight = pred[:, 1]
            pred = pred[:, 0]
            is_train = False
        else:
            is_train = True  # should be True (False for experiment)

        sample_weight = sample_weight.reshape((sample_weight.shape[0], -1)).astype('float32')
        pred = pred.reshape((pred.shape[0], -1))
        label = label.reshape((label.shape[0], -1))

        if is_train:
            x_np = label.asnumpy()
            sample_weight2 = np.zeros(x_np.shape)
            for ibin in range(self.bins.__len__() - 1):
                sample_weight2[np.logical_and(x_np <= self.bins[ibin + 1], x_np > self.bins[ibin])] = self.f[ibin]
            sample_weight2 = sample_weight2.reshape((label.shape[0], -1))
            # sample_weight2 /= sample_weight2.max()
        else:
            sample_weight2 = np.ones(pred.shape, dtype='float32')

        D = (pred - label) * sample_weight.astype('float32') * nd.array(sample_weight2,
                                                                        ctx=pred.context)  # weighted difference & masked with prostate ROI
        c = 0.05 * F.abs(D).max()
        l1 = F.sum(F.abs(D), axis=self._batch_axis, exclude=True) / F.sum(sample_weight, axis=self._batch_axis,
                                                                          exclude=True)
        l2 = F.sum((D ** 2 + c ** 2) / (2 * c), axis=self._batch_axis, exclude=True) / F.sum(sample_weight,
                                                                                             axis=self._batch_axis,
                                                                                             exclude=True)
        l = 0
        for i in range(l1.shape[0]):
            if l1[i] <= c:
                l = l + l1[i]
            else:
                l = l + l2[i]
        return l / l1.shape[0]


class CharbonnierLoss(gluon.loss.Loss):
    """define Charbonnier loss"""

    def __init__(self, batch_axis=0, eps=1e-8, **kwargs):
        super(CharbonnierLoss, self).__init__(None, batch_axis, **kwargs)
        self.eps = eps

    def hybrid_forward(self, F, output, label, mask=None):
        """forward"""
        if mask is None:
            mask = nd.ones(output.shape, dtype='float32')
        loss = F.sqrt((output - label) ** 2 + self.eps ** 2) * mask.astype('float32')
        return F.mean(loss, self._batch_axis, exclude=True).mean()


class SharpnessLoss(gluon.loss.Loss):
    """define constrast loss"""

    def __init__(self, batch_axis=0, **kwargs):
        super(SharpnessLoss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, *args, **kwargs):
        output_sharpness = self.compute_sharpness(output)
        label_sharpness = self.compute_sharpness(label)
        return output_sharpness ** 2 - label_sharpness ** 2

    @staticmethod
    def gradient(im):
        gx = nd.Convolution(im, kernel=(1, 2,), weight=nd.reshape(nd.array([-1, 1], ctx=im.ctx), (1, 1, 1, -1)),
                            num_filter=1, no_bias=True, pad=(0, 0))
        gx = nd.Convolution(gx, kernel=(1, 2,), weight=nd.reshape(nd.array([1, 1], ctx=im.ctx), (1, 1, 1, -1)),
                            num_filter=1, no_bias=True, pad=(0, 1))
        gy = nd.Convolution(im, kernel=(2, 1,), weight=nd.reshape(nd.array([-1, 1], ctx=im.ctx), (1, 1, -1, 1)),
                            num_filter=1, no_bias=True, pad=(0, 0))
        gy = nd.Convolution(gy, kernel=(2, 1,), weight=nd.reshape(nd.array([1, 1], ctx=im.ctx), (1, 1, -1, 1)),
                            num_filter=1, no_bias=True, pad=(1, 0))
        return gx, gy

    def compute_sharpness(self, im):
        gx, gy = self.gradient(im)
        mag = nd.sqrt(gx ** 2 + gy ** 2)
        return mag.mean()


class DiceLoss(gluon.loss.Loss):
    """correlation loss"""

    def __init__(self, axis=[0, 1], weight=1., batch_axis=0, **kwargs):
        super(DiceLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._batch_axis = batch_axis
        self.smooth = 1.

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        """Forward"""
        label = nd.one_hot(label.squeeze(), depth=2).transpose((0, 3, 1, 2))
        intersection = F.sum(label * pred, axis=self._axis, exclude=True)
        union = F.sum(label + pred, axis=self._axis, exclude=True)
        dice = (2.0 * F.sum(intersection, axis=1) + self.smooth) / (F.sum(union, axis=1) + self.smooth)
        # return F.log(1 - dice)
        print(dice.mean())
        return 1 - dice
        # return F.exp(-dice)
        # return 1 - dice.sum() / np.prod(dice.shape)
        # return -dice


class PhotometricLoss(gluon.loss.Loss):
    r"""
    `pred` and `label` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, with_Charbonnier=False, with_DepthAware=False, eps=1e-8,
                 scale_invar=False, alpha=.85, **kwargs):
        super(PhotometricLoss, self).__init__(weight, batch_axis, **kwargs)
        self.with_Charbonnier = with_Charbonnier
        self.with_DepthAware = with_DepthAware
        self.eps = eps
        self.alpha = alpha

    def hybrid_forward(self, F, pred, label, sample_weight=None, _margin=None):
        label = gluon.loss._reshape_like(F, label, pred)
        loss = F.abs(pred - label)
        ssim = compute_ssim(pred, label, sample_weight)

        if self.with_Charbonnier:
            loss = nd.sqrt(F.square(loss) + self.eps ** 2)
        if self.with_DepthAware:
            c = 1
            smooth_factor = 1e-4
            offset = 2 + smooth_factor
            lamda = 1 - nd.broadcast_minimum(nd.log((pred + offset) * c),
                                             nd.log((label + offset + self.eps) * c)) / nd.broadcast_maximum(
                nd.log((pred + offset + self.eps) * c), nd.log((label + offset + self.eps) * c))
            alpha = ((label + 1) / 2).copy() if label.min() < 0 else label.copy()
            loss = gluon.loss._apply_weighting(F, loss, self._weight, (lamda + alpha))

        if _margin is not None:
            loss = F.relu(loss - _margin)

        n_sample = sample_weight.sum(axis=self._batch_axis, exclude=True, keepdims=True)
        loss = gluon.loss._apply_weighting(F, loss, self._weight, sample_weight)
        loss = loss.sum(axis=self._batch_axis, exclude=True, keepdims=True) / n_sample
        loss = 0.5 * self.alpha * (1 - ssim) + (1 - self.alpha) * loss
        return loss.mean()
