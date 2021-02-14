import numpy as np
from sklearn.preprocessing import RobustScaler


def normalize(_A, mask=None, norm_0mean=False):
    """Norm A (MRI-T2): filtering top 0.1% values by assigning them to the top_thr (the value at the 99th percentage)
    then map values to [0 1] range by dividing by the max intensity within the prostate for each slide"""
    thr = .01  # .01
    mask = np.ones_like(_A) if mask is None else mask
    if not norm_0mean:
        x = np.zeros_like(_A)
        for c in range(_A.shape[-1]):
            for i in range(_A.shape[0]):
                tmp = _A[i, ..., c][mask[i, ..., 0] > 0].reshape((-1, 1))
                tmp_n = RobustScaler().fit_transform(X=tmp)[..., 0]
                tmp_n1 = x[i, ..., c]
                tmp_n1[np.where(mask[i, ..., 0] == 1)] = tmp_n
                x[i, ..., c] = tmp_n1
        _A = x.copy()
    else:
        x = np.zeros_like(_A)
        for c in range(_A.shape[-1]):
            mu = np.asarray([_A[i, ..., c][mask[i, ..., 0] == 1].mean() for i in range(_A.shape[0])])
            sigma = np.asarray([_A[i, ..., c][mask[i, ..., 0] == 1].std() for i in range(_A.shape[0])])
            _A[..., c] = ((_A[..., c] - mu[..., np.newaxis, np.newaxis]) / sigma[..., np.newaxis, np.newaxis]) * \
                         mask[..., 0]
    return _A