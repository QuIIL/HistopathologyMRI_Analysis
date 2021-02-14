from typing import Optional
import mxnet as mx


class WeightNormalization:
    """
    Implements Weight Normalization, see Salimans & Kingma 2016 (https://arxiv.org/abs/1602.07868).
    For a given tensor the normalization is done per hidden dimension.
    :param weight: Weight tensor of shape: (num_hidden, d1, d2, ...).
    :param num_hidden: Size of the first dimension.
    :param ndim: The total number of dimensions of the weight tensor.
    :param prefix: The prefix used for naming.
    """

    def __init__(self, weight, num_hidden, ndim=2, prefix: str = '') -> None:
        self.prefix = prefix
        self.weight = weight
        self.num_hidden = num_hidden
        self.scale = mx.sym.Variable("%swn_scale" % prefix,
                                     shape=tuple([num_hidden] + [1] * (ndim - 1)),
                                     init=mx.init.Constant(value=1.0))

    def __call__(self, weight: Optional[mx.nd.NDArray] = None, scale: Optional[mx.nd.NDArray] = None) -> mx.sym.Symbol:
        """
        Normalize each hidden dimension and scale afterwards
        :return: A weight normalized weight tensor.
        """
        if weight is None and scale is None:
            return mx.sym.broadcast_mul(lhs=mx.sym.L2Normalization(self.weight, mode='instance'),
                                        rhs=self.scale, name="%swn_scale" % self.prefix)
        else:
            assert isinstance(weight, mx.nd.NDArray)
            assert isinstance(scale, mx.nd.NDArray)
            return mx.nd.broadcast_mul(lhs=mx.nd.L2Normalization(weight, mode='instance'), rhs=scale)


# MXNet Implementation of Mish Activation Function.
class Mish(mx.gluon.HybridBlock):
    """ Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input."""

    def __init__(self):
        super(Mish, self).__init__()

    def hybrid_forward(self, F, x):
        return x * F.tanh(F.Activation(data=x, act_type='softrelu'))

