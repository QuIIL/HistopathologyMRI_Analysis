import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from numpy import floor

from networks import dmnet_2d_init2 as net_init
from networks.dmnet_2d_init2 import ResDBlock
from utils.learning_rate_scheduler import OneCycleSchedule, CyclicalSchedule, TriangularSchedule
from networks.layers import Mish

# from gluoncv.nn import GroupNorm


use_global_stats = False


def set_trainer(model, net_):
    """Setup training policy"""
    if model.lr_scheduler:
        if model.lr_scheduler == 'cycle':
            schedule = OneCycleSchedule(start_lr=model.start_lr, max_lr=model.max_lr, cycle_length=model.cycle_length,
                                        cooldown_length=model.cooldown_length, finish_lr=model.finish_lr)
        elif model.lr_scheduler == 'cycles':
            schedule = CyclicalSchedule(TriangularSchedule, min_lr=model.start_lr, max_lr=model.max_lr,
                                        cycle_length=model.cycle_length)

        trainer = gluon.Trainer(net_.collect_params(), model.optimizer,
                                {'learning_rate': .001, 'lr_scheduler': schedule, 'wd': model.wd})
    else:
        trainer = gluon.Trainer(net_.collect_params(), model.optimizer, {'learning_rate': model.base_lr,
                                                                         'wd': model.wd})
    return trainer


def review_net(net_, ctx, check_symbol=True, check_nd=False):
    if check_nd:
        x = mx.nd.random.normal(shape=(8, 4, 96, 96), ctx=ctx)
        y = net_(x)
        print(y.shape)
    if check_symbol:
        x = mx.sym.Variable('data')
        y = net_(x)
        mx.viz.plot_network(y, shape={'data': (8, 4, 96, 96)}).view()


def get_num_filters_list(opts, num_stage):
    num_filters_list = []
    num_filters = opts.init_channels
    for st in range(num_stage):
        num_filters += opts.units[st] * opts.growth_rate
        num_filters = int(floor(num_filters * opts.reduction))
        num_filters_list.append(num_filters)
    return num_filters_list


def build_to_density(opts, num_filters, num_group_norm=4):
    to_density = nn.HybridSequential()
    to_density.add(ResDBlock(opts, num_filters=num_filters))
    if opts.norm_type == 'batch':
        to_density.add(nn.BatchNorm())
    elif opts.norm_type == 'group':
        to_density.add(net_init.GroupNorm(num_group=num_group_norm))
    elif opts.norm_type == 'instance':
        to_density.add(nn.InstanceNorm())
    elif opts.norm_type == 'layer':
        to_density.add(nn.LayerNorm())
    if opts.activation == 'leaky':
        to_density.add(nn.LeakyReLU(opts.alpha))
    if opts.activation == 'mish':
        to_density.add(Mish())
    else:
        to_density.add(nn.Activation(opts.activation))
    to_density.add(nn.Conv2D(kernel_size=(1, 1), channels=1, use_bias=opts.use_bias))

    # if opts.norm_type is 'batch':
    #     to_density.add(nn.BatchNorm())
    # elif opts.norm_type is 'group':
    #     to_density.add(net_init.GroupNorm(num_group=num_group_norm))
    # elif opts.norm_type is 'instance':
    #     to_density.add(nn.InstanceNorm())

    # to_density.add(nn.Activation('tanh'))

    to_density.add(nn.Activation(opts.scale_layer)) if opts.scale_layer != 'none' else None

    return to_density


class MutltitaskNet(nn.HybridBlock):
    """Multitasking network"""

    def __init__(self, current_net, density_branch):
        super(MutltitaskNet, self).__init__()
        self.shared_net = nn.HybridSequential()
        self.aux_branch = nn.HybridSequential()
        self.density_branch = nn.HybridSequential()

        with self.name_scope():
            # Segmentation branch
            for i in range(2, current_net.net.__len__()):
                self.aux_branch.add(current_net.net.__getitem__(i))
            # Regression branch
            self.density_branch.add(density_branch)
            # Share
            self.shared_net.add(current_net.net.__getitem__(0))
            self.shared_net.add(current_net.net.__getitem__(1))

    def forward(self, x, *args):
        """"""
        feat = self.shared_net(x)
        den = self.density_branch(feat)
        seg = self.aux_branch(feat)
        return den, seg


def pretrained_net(dir_model=r"F:\Minh\projects\NIH\prostateSegmentation\outputs\run0", activation='relu',
                   num_extracted_encoder=3, to_review_net=False, model=None, norm_type='batch', num_group_norm=4,
                   initializer='none', num_fpg=8, growth_rate=4):
    # if int(dir_model.split('\\')[-1][3:]) == 0:
    #     num_stage = 4
    # else:
    #     num_stage = 3

    num_stage = model.num_downs
    opts = net_init.Init(
        activation=activation,
        num_stage=num_stage,
        scale_layer=model.scale_layer,
        norm_type=norm_type,
        num_group_norm=num_group_norm,
        num_fpg=num_fpg,
        growth_rate=growth_rate,
    )
    # opts.scale_layer = model.scale_layer
    # opts.description()
    net = net_init.DenseMultipathNet(opts)

    # Adding layers on top
    num_filters_list = get_num_filters_list(opts, num_stage)

    # Commented out on Mar 12th, 2019 by MinhTo, to test prediction on the original scale instead of the downsampled scale
    ####################################################################################################################
    # to_density_net = build_to_density(opts, num_filters_list[num_extracted_encoder-1])
    to_density_net = build_to_density(opts, num_filters_list[0], num_group_norm=num_group_norm)
    ####################################################################################################################

    # Commented out on Mar 11th, 2019 by MinhTo, to try training on the full network,
    # instead of gradually growing the network
    # Extracting Encoder
    # new_net = nn.Sequential()
    # new_net.add(net.net.__getitem__(0))
    #
    # tmp = net
    # for stage in range(num_extracted_encoder):
    #     tmp = tmp.net.__getitem__(1)
    #     new_net.add(tmp.net.__getitem__(0))

    # To plug in a decoder
    # new_net.add(tmp.net.__getitem__(1))

    # Added on Mar 11th, 2019 by MinhTo, to try training on the full network
    ########################################################################

    # init = inits[initializer]
    if model.lambda_aux == 0:
        new_net = nn.HybridSequential()
        new_net.add(net.net.__getitem__(0))
        new_net.add(net.net.__getitem__(1))
        new_net.add(to_density_net)
    else:
        new_net = MutltitaskNet(net, to_density_net)

    # new_net.collect_params().initialize(init=init, ctx=model.ctx)

    if to_review_net:
        review_net(new_net, model.ctx)

    if model.freeze_pretrained_net:
        trainer = set_trainer(model, to_density_net)
        return new_net, trainer
    else:
        return new_net


def extract_encoder(net, num_downs):
    """Extract the encoder part of DMNet"""

    def get_encoder_block(_enc, _net, _num_downs, current_level):
        """Extract encoder block of each level"""
        if current_level < _num_downs:
            _enc.add(_net.net[0])
            _net = _net.net[1]
            current_level += 1
            get_encoder_block(_enc, _net, _num_downs, current_level)
        return _enc

    enc = nn.HybridSequential()
    enc.add(net[0])
    enc = get_encoder_block(enc, net[1], num_downs, 0)
    enc.add(nn.GlobalAvgPool2D())
    return enc
