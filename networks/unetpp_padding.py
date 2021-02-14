from mxnet import nd, gpu, init, cpu
from mxnet.gluon import nn
from mxnet import gluon
from mxnet.optimizer import Adam
from networks.backbones.construct_backbones import Backbones

norm_layer = nn.BatchNorm
# norm_layer = gluon.contrib.nn.SyncBatchNorm


def conv_factory(ksz=3, stride=1, pad=1, channels=64, act='relu', bn=True, bn_mom=.9, bn_eps=1e-5):
    """Convolution + Activation + Batchnorm"""
    block = nn.HybridSequential()
    block.add(nn.Conv2D(kernel_size=ksz, strides=stride, padding=pad, channels=channels, use_bias=False))
    block.add(nn.Activation(act)) if act is not None else None
    block.add(norm_layer(momentum=bn_mom, epsilon=bn_eps)) if bn else None
    return block


def deconv_factory(ksz=3, stride=1, pad=1, channels=64, act='relu', bn=True, bn_mom=.9, bn_eps=1e-5):
    """Transpose convolution + Activation + Batchnorm"""
    block = nn.HybridSequential()
    block.add(nn.Conv2DTranspose(channels=channels, kernel_size=ksz, strides=stride, padding=pad, use_bias=False))
    block.add(norm_layer(momentum=bn_mom, epsilon=bn_eps)) if bn else None
    block.add(nn.Activation(act)) if act is not None else None
    return block


def process_layer(channels=None, bn_mom=.9, bn_eps=1e-5):
    """Feature extraction layers for each resolution"""
    layer = nn.HybridSequential()
    layer.add(
        conv_factory(channels=channels, bn_mom=bn_mom, bn_eps=bn_eps),
        conv_factory(channels=channels, bn_mom=bn_mom, bn_eps=bn_eps),
    )
    return layer


class CenteredLayer(nn.HybridBlock):
    """A layer that does center cropping"""

    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x, ref):
        # nd.slice(x, begin=)
        size_dif = int((x.shape[-1] - ref.shape[-1]) / 2)
        return nd.slice(x,
                        begin=(0, 0, size_dif, size_dif),
                        end=(x.shape[0], x.shape[1], x.shape[-2] - size_dif, x.shape[-1] - size_dif))


class UNet(nn.HybridBlock):
    """Build UNet"""

    def __init__(self, num_ds=4, add_scale_layer='tanh', bn_mom=.9, bn_eps=1e-5, base_channel=16, backbone_name='vgg'):
        super(UNet, self).__init__()
        self.backbone_name = backbone_name
        self.num_ds = num_ds
        assert num_ds < 10
        self.layer_names_ds = ['layer%d_ds' % i for i in range(num_ds + 1)]
        self.layer_names_us = ['layer%d%d_us' % (i, k) for i in range(num_ds - 1, -1, -1) for k in
                               range(1, num_ds - i + 1)]

        if self.backbone_name != 'vgg':
            layers_ds = Backbones(self.backbone_name, self.num_ds).separate_layers()

        with self.name_scope():
            for i, layer in enumerate(self.layer_names_ds):
                if self.backbone_name == 'vgg':
                    self.__setattr__(layer, process_layer(channels=(base_channel * (2 ** i)), bn_mom=bn_mom, bn_eps=bn_eps))
                else:
                    self.__setattr__(layer, layers_ds[layer])

            for layer in self.layer_names_us:
                i = int(layer[5:7][0])  # TODO: replace by some characater searching functions
                self.__setattr__(layer, process_layer(channels=(base_channel * (2 ** i)), bn_mom=bn_mom, bn_eps=bn_eps))
            self.downsample = nn.MaxPool2D()
            self.upsamples = []

            for i in range(self.num_ds - 1, -1, -1):
                for k in range(1, num_ds - i + 1):
                    self.__setattr__('upsample%d%d' % (i, k),
                                     deconv_factory(ksz=2, stride=2, pad=0, channels=base_channel * (2 ** i), act='relu', bn=True, bn_mom=.9, bn_eps=1e-5))

            self.center_crop = CenteredLayer()
            self.last_conv = nn.Conv2D(kernel_size=1, channels=1, use_bias=False)
            # self.last_bn = norm_layer(momentum=bn_mom, epsilon=bn_eps)
            self.to_density = nn.Activation(add_scale_layer)

    def hybrid_forward(self, F, x, *args, **kwargs):
        """"""
        # UNet++ and UNet has the same down-sampling process
        outs_us, outs_ds = {}, {}

        if self.backbone_name == 'vgg':
            outs_ds = {'out0': self.__getattribute__(self.layer_names_ds[0])(x)}
            for l in range(1, self.num_ds + 1):
                outs_ds['out%d' % l] = self.__getattribute__(self.layer_names_ds[l])(
                    nn.MaxPool2D()(outs_ds['out%d' % (l - 1)]))
        else:
            outs_ds = {'out0': self.__getattribute__(self.layer_names_ds[0])(x)}
            for l in range(1, self.num_ds + 1):
                outs_ds['out%d' % l] = self.__getattribute__(self.layer_names_ds[l])(
                    (outs_ds['out%d' % (l - 1)]))

        # for k in outs_ds.keys():
        #     print(outs_ds[k].shape)

        for current_num_ds in range(1, self.num_ds + 1):
            # print('upsample%d%d' % (current_num_ds - 1, 1))
            upsampled = self.__getattribute__('upsample%d%d' % (current_num_ds - 1, 1))(outs_ds['out%d' % current_num_ds])
            up_cropped_concat = F.concat(outs_ds['out%d' % (current_num_ds - 1)], upsampled, dim=1)
            outs_us = {'out%d' % (current_num_ds - 1):
                           self.__getattribute__('layer%d%d_us' % (current_num_ds - 1, 1))(up_cropped_concat)}
            outs_ds['out%d' % (current_num_ds - 1)] = outs_us['out%d' % (current_num_ds - 1)]

            if current_num_ds == 1:
                continue

            for (col, l) in enumerate(range(current_num_ds - 2, -1, -1)):
                # print('upsample%d%d' % (l, col + 2))
                upsampled = self.__getattribute__('upsample%d%d' % (l, col+2))(outs_us['out%d' % (l + 1)])
                up_cropped_concat = F.concat(outs_ds['out%d' % l], upsampled, dim=1)
                outs_us['out%d' % l] = self.__getattribute__('layer%d%d_us' % (l, col+2))(up_cropped_concat)
                outs_ds['out%d' % l] = outs_us['out%d' % l]

        # return self.to_density(self.last_bn(self.last_conv(outs_us['out0'])))
        # for k in outs_us.keys():
        #     print(outs_us[k].shape)
        return self.to_density(self.last_conv(outs_us['out0']))
        # return self.last_conv(outs_us['out0'])


if __name__ == "__main__":
    from mxnet.visualization import plot_network
    from mxnet import sym

    ctx = cpu()

    net = UNet(num_ds=4, base_channel=32, backbone_name='resnet')  # densenet resnet vgg
    net.initialize(ctx=ctx, init=init.Xavier(magnitude=2.2))
    optimizer = Adam(learning_rate=1e-3)
    trainer = gluon.Trainer(params=net.collect_params(), optimizer=optimizer)
    # print(net)
    x = nd.random.normal(0, 1, shape=(2, 2, 256, 256), ctx=ctx)
    outs_us = net(x)
    print(net.summary(x))
    print(outs_us.shape)

    # for out in outs_us.keys():
    #    print(outs_us[out].shape)\

    # net = UNet(num_ds=4)
    # x = sym.Variable('data')
    # outs_us = net(x)
    # plot_network(outs_us, shape={'data': (2, 2, 256, 256)}, title='UNet++_padding',
    #              node_attrs={"shape": "oval", "fixedsize": "false"}).view('UNet++_padding')
