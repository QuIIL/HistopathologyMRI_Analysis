import numpy as np
from skimage.measure import compare_ssim as _ssim
from sklearn.metrics import normalized_mutual_info_score as _nmi
from sklearn import metrics as mtrc


def corrcoef(a, b, m=None):
    """Compute correlation coefficient between a and b
    Shapes of a, b, and m has to be (N, C, H, W), with C = 1
    """
    a_flat, b_flat = flatten(a, b, mask=m)
    r = np.asarray([np.corrcoef(x, y)[0, 1] for x, y in zip(a_flat, b_flat)])
    return r


def ssim(a, b, m=None):
    """Compute ssim between a and b
    Shapes of a, b, and m has to be (N, C, H, W), with C = 1
    NOTE: data_range is currently set to 2, which may not hold true for future usage
    """
    if (a.shape[1] == 1) or (a.ndim == 3):
        a_flat, b_flat = flatten(a, b, mask=m)
        return np.asarray([_ssim(x, y, data_range=2) for x, y in zip(a_flat, b_flat)])
    return [ssim(a[:, k], b[:, k], m[:, 0]) for k in range(a.shape[1])]


def nmi(a, b, m=None):
    """Compute ssim between a and b
    Shapes of a, b, and m has to be (N, C, H, W), with C = 1
    """
    a_flat, b_flat = flatten(a, b, mask=m)
    r = np.asarray([_nmi(x, y, 'arithmetic') for x, y in zip(a_flat, b_flat)])
    return r


def l1_error(a, b, m=None):
    """Compute L1 loss between a and b
    Shapes of a, b, and m has to be (N, C, H, W), with C = 1
    """
    if (a.shape[1] == 1) or (a.ndim == 3):
        a_flat, b_flat = flatten(a, b, mask=m)
        l1 = [(np.abs(x - y)).sum() / x.shape[0] for x, y in zip(a_flat, b_flat)]
        return l1
    return [l1_error(a[:, k], b[:, k], m[:, 0]) for k in range(a.shape[1])]


def corrcoef_whole_v0(a, b, m=None):
    """Compute correlation coefficient between a and b
    Shapes of a, b, and m has to be (N, C, H, W), with C = 1
    """
    a, b = a[m == 1], b[m == 1]
    r = np.corrcoef(a, b)[0, 1]
    return r


def corrcoef_whole(a, b, m=None):
    """R^2 (coefficient of determination) regression score function
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    """
    if (a.shape[1] == 1) or (a.ndim == 3):
        r = mtrc.r2_score(y_true=b.flatten(), y_pred=a.flatten(), sample_weight=m.flatten())
        return r
    return [corrcoef_whole(a[:, k], b[:, k], m[:, 0]) for k in range(a.shape[1])]


def l1_error_whole(a, b, m=None):
    """Compute L1 loss between a and b
    Shapes of a, b, and m has to be (N, C, H, W), with C = 1
    """
    if (a.shape[1] == 1) or (a.ndim == 3):
        a, b = a[m == 1], b[m == 1]
        l1 = (np.abs(a - b)).sum() / b.shape[0]
        # mtrc.mean_absolute_error(a.flatten(), b.flatten(), m.flatten())  # same results
        return l1
    l1 = []
    for k in range(a.shape[1]):
        l1.append(l1_error_whole(a[:, k], b[:, k], m[:, 0]))
    return l1


def abs_rel_diff(a, b, m=None):
    """Compute absolute relative difference
    Shapes of a, b, and m has to be (N, C, H, W), with C = 1
        """
    a, b = a[m == 1] * 100 + 1, b[m == 1] * 100 + 1  # 1 is the smooth factor
    return np.sum(np.abs(a - b) / b) / b.shape[0]


def sqr_rel_diff(a, b, m=None):
    """Compute absolute relative difference
    Shapes of a, b, and m has to be (N, C, H, W), with C = 1
        """
    a, b = a[m == 1] * 100 + 1, b[m == 1] * 100 + 1  # 1 is the smooth factor
    return np.sum((a - b) ** 2 / b) / b.shape[0]


class Threshold:
    """Compute the accuracy of the regression, where it counts as a correct if the difference
    between a and b smaller than delta ^ gamma"""

    def __init__(self, delta=1.25, gamma=1, from_11=False):
        self.delta = delta
        self.gamma = gamma
        self.from_11 = from_11

    def __call__(self, a, b, m=None):
        a = a[m == 1]
        b = b[m == 1]
        smooth_factor = 1e-6
        thr = self.delta ** self.gamma
        ratio = (a + smooth_factor) / (b + smooth_factor)
        return (np.maximum(ratio, 1 / ratio) < thr).sum() / b.shape[0]


def rmse_whole(a, b, m=None):
    """Compute root-mean-square-error
    Shapes of a, b, and m has to be (N, C, H, W), with C = 1
    """
    return mtrc.mean_squared_error(a.flatten(), b.flatten(), m.flatten())


def rmse_log_whole(a, b, m=None):
    """Compute root-mean-square-log-error
    Shapes of a, b, and m has to be (N, C, H, W), with C = 1
    """
    return mtrc.mean_squared_log_error(a.flatten(), b.flatten(), m.flatten())


def ssim_whole(a, b, m=None):
    """Compute ssim between a and b
    Shapes of a, b, and m has to be (N, C, H, W), with C = 1
    NOTE: data_range is currently set to 2, which may not hold true for future usage
    """
    if (a.shape[1] == 1) or (a.ndim == 3):
        a = a[m == 1]
        b = b[m == 1]
        return _ssim(a, b, data_range=2)
    return [ssim_whole(a[:, k], b[:, k], m[:, 0]) for k in range(a.shape[1])]


def nmi_whole(a, b, m=None):
    """Compute ssim between a and b
    Shapes of a, b, and m has to be (N, C, H, W), with C = 1
    NOTE: data_range is currently set to 2, which may not hold true for future usage
    """
    a = a[m == 1]
    b = b[m == 1]
    return _nmi(a, b, average_method='arithmetic')


class ThresholdedAccuracy:
    """Compute the accuracy of the regression, where it counts as a correct if the difference
    between a and b smaller than delta ^ gamma"""

    def __init__(self, delta=.1, gamma=1.0, from_11=False):
        self.delta = delta
        self.gamma = gamma
        self.from_11 = from_11

    def __call__(self, a, b, m=None):
        a = a[m == 1]
        b = b[m == 1]
        if self.from_11:
            a, b = scale_01(a), scale_01(b)
        thr = self.delta * self.gamma
        d = np.abs(a - b) < thr
        return d.sum() / m.sum()


def scale_01(x, vmin=-1, vmax=1):
    """scale to 0 - 1"""
    return (x - vmin) / (vmax - vmin)


def linear_scale(x, vmin=-1, vmax=1, tmin=0, tmax=1):
    return ((x - vmin) / (vmax - vmin)) * (tmax - tmin) + tmin


def flatten(*matrices, mask=None):
    """Flatten images. Each flatten output is masked (so may have a different size) and stored in a list
    Shapes of each matrix has to be (N, C, H, W), with C = 1
    """
    if mask is None:
        mask = np.ones_like(matrices[0])
    matrices_flatten = []
    for matrix in matrices:
        matrices_flatten.append([
            matrix[i, 0].flatten()[np.where(mask[i, 0].flatten() == 1)[0]] for i in range(matrix.shape[0])]
        )
    return [*matrices_flatten]


def dice_wp(preds, labels):
    """Compute Dice per case between prediction and label"""
    smooth = 1.
    labels = labels.squeeze()
    if preds.shape != labels.shape:
        preds = preds.argmax(axis=1)
    dice = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        dice[i] = ((2 * (labels[i] * preds[i]).sum()) + smooth) / (labels[i].sum() + preds[i].sum() + smooth)
    return dice


def update_mxboard_metric(sw, data, global_step, metric_names='r', prefix=None, ):
    """Log metrics to mxboard"""
    metric_names = list(metric_names) if not isinstance(metric_names, list) else metric_names
    metrics = {
        'r': ('correlation_coefficient', corrcoef),
        'l1': ('l1_loss', l1_error),
    }
    metric_list = {}
    for metric_name in metric_names:
        fn_name = metrics[metric_name][0]
        fn = metrics[metric_name][1]
        metric = np.asarray(fn(data[1], data[2], data[3]))  # pred, ground truth and mask
        sw.add_scalar('%smean_%s' % (prefix, fn_name), ['Val', metric.mean()], global_step=global_step)
        # [sw.add_scalar('%s%ss' % (prefix, fn_name), ['Val_%d' % i, metric[i]], global_step=global_step) for i in range(metric.shape[0])]

        metric_list[metric_name] = metric
    return metric_list


metrics = {
    'r': ('correlation_coefficient', corrcoef),
    'l1': ('l1_loss', l1_error),
    'ssim': ('ssim', ssim),
    'nmi': ('normalized_mutual_information', nmi),
    'r_whole': ('correlation_coefficient_whole', corrcoef_whole),
    'l1_whole': ('l1_whole', l1_error_whole),
    'rmse_whole': ('rmse_whole', rmse_whole),
    'rmse_log_whole': ('rmse_log_whole', rmse_log_whole),
    'ssim_whole': ('ssim_whole', ssim_whole),
    'nmi_whole': ('normalized_mutual_information_whole', nmi_whole),
    'ta1': ('thesholded_accuracy_1', ThresholdedAccuracy(gamma=1)),
    'ta2': ('thesholded_accuracy_2', ThresholdedAccuracy(gamma=0.5)),
    't1': ('threshold1', Threshold(gamma=1)),
    't2': ('threshold2', Threshold(gamma=2)),
    't3': ('threshold3', Threshold(gamma=3)),
    'abs_rel_diff': ('absolute_relative_difference', abs_rel_diff),
    'sqr_rel_diff': ('squared_relative_difference', sqr_rel_diff),
}
out_channels_names = ['EPI', 'NUC', 'STR', 'LUM']


def update_mxboard_metric_v1(sw, data, global_step, metric_names='r', prefix=None, num_input_channels=1,
                             not_write_to_mxboard=False, c_thr=1, density_range=[-1, 1], root=1):
    """Log metrics to mxboard
    Compared to original update_mxboard_metric method, thes v1 method does not show metrics of individuals on the tensorboard"""

    metric_names = list(metric_names) if not isinstance(metric_names, list) else metric_names
    metric_list = {}
    data_scaled = []
    # 0: input, 1: pred, 2: gt, 3: ROIs, 4: mask
    data_scaled.append(linear_scale(data[1], vmin=density_range[0], vmax=density_range[1], tmin=0, tmax=c_thr))
    data_scaled.append(linear_scale(data[2], vmin=density_range[0], vmax=density_range[1], tmin=0, tmax=c_thr))
    data_scaled[0] **= root
    data_scaled[1] **= root
    for metric_name in metric_names:
        fn_name = metrics[metric_name][0]
        fn = metrics[metric_name][1]
        metric = np.asarray(fn(
            data_scaled[0],  # the index is not affected by number of input channels
            data_scaled[1],
            data[3]
        ))  # pred, ground truth and mask
        if not not_write_to_mxboard:
            sw.add_scalar('metrics/%smean_%s' % (prefix, fn_name), metric.mean(), global_step=global_step)
        # [sw.add_scalar('%s%ss' % (prefix, fn_name), ['Val_%d' % i, metric[i]], global_step=global_step) for i in range(metric.shape[0])]

        metric_list[metric_name] = metric
    return metric_list


def update_mxboard_metric_multi_maps(sw, data, global_step, metric_names='r', prefix=None, num_input_channels=1,
                                     not_write_to_mxboard=False, c_thr=1, density_range=[-1, 1], root=1):
    """Log metrics to mxboard
    Compared to original update_mxboard_metric method, thes v1 method does not show metrics of individuals on the tensorboard"""

    metric_names = list(metric_names) if not isinstance(metric_names, list) else metric_names
    metric_list = {}
    data_scaled = []
    # 0: input, 1: pred, 2: gt, 3: ROIs, 4: mask
    data_scaled.append(linear_scale(data[1], vmin=density_range[0], vmax=density_range[1], tmin=0, tmax=c_thr))
    data_scaled.append(linear_scale(data[2], vmin=density_range[0], vmax=density_range[1], tmin=0, tmax=c_thr))
    data_scaled[0] **= root
    data_scaled[1] **= root
    for metric_name in metric_names:
        fn_name = metrics[metric_name][0]
        fn = metrics[metric_name][1]
        metric = np.asarray(fn(
            data_scaled[0],  # the index is not affected by number of input channels
            data_scaled[1],
            data[3]
        ))  # pred, ground truth and mask
        if not not_write_to_mxboard:
            if len(metric) == 1:
                sw.add_scalar('metrics/%smean_%s' % (prefix, fn_name), metric.mean(), global_step=global_step)
            else:
                for i in range(len(metric)):
                    sw.add_scalar('metrics/%smean_%s' % (prefix, fn_name + '_' + out_channels_names[i]),
                                  metric[i].mean(),
                                  global_step=global_step)
                sw.add_scalar('metrics/%smean_%s' % (prefix, fn_name), metric.mean(), global_step=global_step)
        # [sw.add_scalar('%s%ss' % (prefix, fn_name), ['Val_%d' % i, metric[i]], global_step=global_step) for i in range(metric.shape[0])]
        if len(metric) == 1:
            metric_list[metric_name] = metric
        else:
            for i in range(len(metric)):
                metric_list[metric_name + '_' + out_channels_names[i]] = metric[i].mean()
            metric_list[metric_name] = metric.mean()
    return metric_list
