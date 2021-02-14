import csv
import os
import mxnet as mx
import numpy as np
import pylab as plt
from matplotlib import ticker
from mxnet import nd, gluon, autograd
# from mxnet.contrib import amp
from mxnet.gluon.nn import HybridBlock
from mxnet.gluon.utils import split_and_load as sal
from numpy.ma import masked_array
from skimage.measure import regionprops
from skimage.transform import resize
from skimage.util import montage
from sklearn.preprocessing import RobustScaler

from networks.discriminators import Discriminator
from utils import metrics
from utils.dataloader import DataLoader
from utils.batchify_fn import BatchifyFn
from utils.datasets import RadPath
from utils.learning_rate_scheduler import *
from utils.load_pretrained_net import pretrained_net, extract_encoder
from utils.losses import L1Loss_v2, L2Loss_v2, DiceLoss, L2LogLoss, L1LogLoss, LogCoshLoss, PhotometricLoss, \
    corrcoefLoss, HuberLoss
from utils.optimizers import get_optimizer_dict
from utils.sampler import BatchSampler, RandomSamplerStratify2Sets
from utils.transformations import joint_transform
from utils.loss_margin import LossMargin
from collections import OrderedDict
from pydicom.dataset import Dataset, FileDataset
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from skimage.color import label2rgb

# from utils import lookahead_optimizer
# amp.init()

filename_unsup_pred = 'dummy/analyze_unsup_pred/data/pred_unsup'
# filename_pretrained_weights = 'dummy/analyze_unsup_pred/netG.params'
filename_pretrained_weights = 'results/ImprovedSemi_PostMICCAI_5\drnnGR4_lCL0_ENSL_lUS1_lC0_l2_nc8_stsa0.9_sd11_normal_v2b_check_last_iter_issues_TSA0.90\checkpoints\iter_2499'

# Refer: https://mxnet.incubator.apache.org/versions/master/tutorials/amp/amp_tutorial.html

inits = {
    'none': mx.init.Uniform(),
    'normal': mx.init.Normal(.05),
    'xavier': mx.init.Xavier(magnitude=2.2),
    'he': mx.init.MSRAPrelu(),
}


class Init:
    """Initialize training parameters and directories"""

    def __init__(self, args):
        self.__dict__.update(args.__dict__)
        if isinstance(self.run_id, int):
            self.result_folder = 'results/%s/run_%03d/' % (self.experiment_name, self.run_id)
        else:
            self.result_folder = 'results/%s/%s/' % (self.experiment_name, self.run_id)
        self.result_folder_checkpoint = '%s/checkpoints' % self.result_folder
        self.result_folder_figure_train = '%s/figures/train' % self.result_folder
        self.result_folder_figure_train_unsup = '%s/figures/train_unsup' % self.result_folder
        self.result_folder_figure_val = '%s/figures/val' % self.result_folder
        self.result_folder_figure_test = '%s/figures/test' % self.result_folder
        self.result_folder_logs = '%s/logs' % self.result_folder
        # suffice = '_adjacent' if self.use_adjacent else ''
        suffice = '_adjacent_%s' % self.density_type  # input file always has the '_adjacent_' suffix
        self.dataset_file = r'inputs/%s%s.npy' % (self.dataset_file, suffice)
        self.dataset_file_org = r'inputs/%s.npy' % self.dataset_file_org
        folders = [field for field in list(self.__dict__.keys()) if 'folder' in field]
        for folder in folders:
            if not os.path.exists(self.__getattribute__(folder)):
                os.makedirs(self.__getattribute__(folder))

        self.ctx = [mx.gpu(int(i)) for i in self.gpu_id.split(',')]
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        self.batch_size = args.batch_size
        self.save_setting()
        self.image_pool = ImagePool(args.pool_size)
        self.density_range = [-1, 1]

        Aus, wp_us = self.load_data()
        Aus = self.normalize(Aus, mask=wp_us)
        val_dataset = RadPath(
            Aus, Aus, wp_us, wp_us, wp_us,  # duplicate to simplify the modifications
            transform=joint_transform,
            is_val=True,
            input_size=self.input_size,
            density_range=self.density_range,
        )

        self.val_iter = DataLoader(val_dataset,
                                   batch_size=self.batch_size,
                                   num_workers=self.num_workers,
                                   last_batch='keep',
                                   shuffle=False,
                                   thread_pool=False,
                                   prefetch=None,
                                   )

    def load_data(self):
        x = np.load(r'%s' % self.mr_file).transpose(
            (2, 0, 1, 3)).astype('float32')[..., [0, 1, 3]]  # exclude the target ROI
        x[np.isnan(x)] = 0
        A = x[..., :2].astype(self.dtype)
        wp = x[..., 2:].astype(self.dtype)
        return A, wp

    def save_setting(self):
        """save input setting into a csv file"""
        with open('%s/parameters.csv' % self.result_folder, 'w') as f:
            w = csv.writer(f)
            for key, val in self.__dict__.items():
                w.writerow([key, val])

    @staticmethod
    def normalize(_A, _B=None, _C=None, mask=None, to_11=False, norm_0mean=False, train_idx_A=None, outlier_thr=1,
                  root=1):
        """Norm A (MRI-T2): filtering top 0.1% values by assigning them to the top_thr (the value at the 99th percentage)
        then map values to [0 1] range by dividing by the max intensity within the prostate for each slide"""
        thr = .01  # .01
        mask = np.ones_like(_A) if mask is None else mask
        if not norm_0mean:
            x = np.zeros_like(_A)
            for c in range(_A.shape[-1]):
                for i in range(_A.shape[0]):
                    if mask[i, ..., 0].sum() == 0:
                        continue
                    tmp = _A[i, ..., c][mask[i, ..., 0] == 1].reshape((-1, 1))
                    tmp_n = RobustScaler().fit_transform(X=tmp)[..., 0]
                    tmp_n1 = x[i, ..., c]
                    tmp_n1[np.where(mask[i, ..., 0] == 1)] = tmp_n
                    x[i, ..., c] = tmp_n1
            _A = x.copy()

        def find_threshold(_x=None, _mask=None, _outlier_thr=1):  # .999
            _x = _x[_mask == 1] if _mask is not None else _x
            _x = _x.flatten() if _x.ndim > 1 else _x
            _x = np.sort(_x)
            thr_val = _x[int(_outlier_thr * _x.__len__())]
            return thr_val

        if (_B is not None) and (_C is not None):
            """Norm B & C (Density maps): simply map density values to [0 1] range by dividing by the max intensity within the prostate, and mask for each slide, respectively"""
            thr = 1 if outlier_thr == 1 else find_threshold(_C[train_idx_A], mask[train_idx_A], outlier_thr)
            thr = np.round(thr, 5)
            _C **= (1 / root)
            _B[_B > thr] = thr
            _C[_C > thr] = thr
            # _C_tmp = _C[train_idx_A]
            # _C_tmp[_C_tmp > thr] = thr
            # _C[train_idx_A] = _C_tmp
            # print(thr)
            if to_11:
                # original_density_range = [_C[mask == 1].min(), _C[mask == 1].max()]
                _B = (_B - _B.min()) / (_B.max() / 2 - _B.min()) - 1
                _C = (_C - _C.min()) / (_C.max() / 2 - _C.min()) - 1  # _C[train_idx_A].max() will be 1
                # _C = (_C - _C[mask == 1].min()) / (_C[mask == 1].max() - _C[mask == 1].min())
                density_range = [-1, 1]
            else:
                # original_density_range = [_C[mask == 1].min(), _C[mask == 1].max()]
                _B = (_B - _B.min()) / (_B.max() - _B.min())
                # _C = (_C - _C.min()) / (_C.max() - _C.min())
                _C = (_C - _C[mask == 1].min()) / (_C[mask == 1].max() - _C[mask == 1].min())
                density_range = [0, 1]
            # print(thr)
            # _C[train_idx_A == False][mask[train_idx_A == False] > 0][100]
            return _A, _B, _C, density_range, thr
        else:
            return _A


def replace_conv2D(net, first_conv):
    """Replace the first convolution layer by a layer having the same in_channels with the number of input channels"""
    for key, layer in net._children.items():
        if isinstance(layer, gluon.nn.Conv2D):
            with net.name_scope():
                net.register_child(first_conv, key)


class FeatureComparator(HybridBlock):
    """Generate features from a given image
    Modify intermediate layers following "https://discuss.mxnet.io/t/modify-structure-of-loaded-network/1537"
    """

    def __init__(self, in_channels=1, ctx=None):
        super(FeatureComparator, self).__init__()
        from mxnet.gluon.model_zoo import vision
        from mxnet.initializer import Constant

        self.net = vision.resnet18_v1(pretrained=True, ctx=ctx)
        first_conv = gluon.nn.Conv2D(64, kernel_size=7, strides=2,
                                     padding=3, in_channels=in_channels)
        first_conv.initialize(init=Constant(self.net.features[0].weight.data(ctx=ctx[0])[:, 0:in_channels]),
                              force_reinit=True, ctx=ctx)
        replace_conv2D(self.net.features, first_conv)

    def forward(self, x, *args):
        return self.net(x)


class PixUDA(Init):
    def __init__(self, args):
        super(PixUDA, self).__init__(args=args)
        self.set_lr_scheduler()
        self.set_networks()
        self.def_loss()
        self.set_metric()

    def network_init(self, net):
        for param in net.collect_params().values():
            self.param_init(param, self.ctx)

    def set_networks(self):
        n_in = 2 if '5channels' in self.dataset_file else 1

        if self.true_density_generator == 'unet':
            from networks.unet_padding import UNet
            self.netG = UNet(base_channel=self.base_channel_unet, backbone_name=self.backbone_name)
        elif self.true_density_generator == 'unetpp':
            from networks.unetpp_padding import UNet
            self.netG = UNet(base_channel=self.base_channel_unet, backbone_name=self.backbone_name)
        elif self.true_density_generator == 'drnn':
            from networks.drnn import DenseMultipathNet, Init as init_net_params
            opts = init_net_params(num_fpg=self.num_fpg, growth_rate=self.growth_rate,
                                   init_channels=self.base_channel_drnn,
                                   num_channels_out=self.num_channels_out)
            self.netG = DenseMultipathNet(opts)
        elif self.true_density_generator == 'deeplabv3':
            from networks.deeplabv3 import DeepLabV3
            self.netG = DeepLabV3(1, backbone='resnet50', pretrained_base=False,
                                  ctx=self.ctx)
        if self.true_density_generator == 'deeplabv3plus':
            from networks.deeplabv3b_plus import DeepLabWV3Plus
            self.netG = DeepLabWV3Plus(1, backbone='wideresnet', ctx=self.ctx, base_size=self.input_size,
                                       crop_size=self.input_size)

        self.netG.initialize(inits[self.initializer], ctx=self.ctx, force_reinit=True)
        if self.resumed_it > -1:
            self.load_checkpoints(prefix=self.checkpoint_prefix)
        elif self.use_pretrained:
            self.load_checkpoints(pretrained_dir=filename_pretrained_weights)

        self.use_l_coefs = False
        coefs = []
        for l in ['0', '_aux', '_C', '_consistency', '_D', '_unsup']:
            if self.__getattribute__('lambda%s' % l) > 0:
                self.use_l_coefs = True
                self.netG.__setattr__('coef%s' % l,
                                      gluon.Parameter('coef%s' % l, shape=1, init=mx.init.Constant(.6), lr_mult=1))
                self.netG.__getattribute__('coef%s' % l).initialize(ctx=self.ctx)
                coefs.append(self.netG.__getattribute__('coef%s' % l))

        if self.use_l_coefs:
            self.netG.coef_G = gluon.Parameter('coefG', shape=1, init=mx.init.Constant(.6), lr_mult=1)
            self.netG.coef_G.initialize(ctx=self.ctx)
            coefs.append(self.netG.coef_G)
            self.trainer_coefs = gluon.Trainer(coefs,
                                               optimizer=self.optimizer,
                                               optimizer_params=get_optimizer_dict(
                                                   self.optimizer,
                                                   lr=self.base_lr,
                                                   lr_scheduler=self._lr_scheduler,
                                                   wd=self.wd,
                                                   beta1=self.beta1,
                                               ),
                                               # update_on_kvstore=False,
                                               )

        self.trainerG = gluon.Trainer(self.netG.collect_params(),
                                      optimizer=self.optimizer,
                                      optimizer_params=get_optimizer_dict(
                                          self.optimizer,
                                          lr=self.base_lr,
                                          lr_scheduler=self._lr_scheduler,
                                          wd=self.wd,
                                          beta1=self.beta1,
                                      ),
                                      # update_on_kvstore=False,
                                      )

        # amp.init_trainer(self.trainerG)  # automatic mixed precision
        largest_batch_size = int(np.ceil(self.batch_size / len(self.gpu_id.split(','))))
        largest_batch_size *= self.unsup_ratio if self.lambda_unsup > 0 else 1
        if self.show_generator_summary:
            [self.netG.summary(
                nd.random.normal(0, 1, shape=(largest_batch_size, n_in, self.input_size, self.input_size), ctx=ctx)) for
                ctx
                in self.ctx]
        self.D_features = FeatureComparator(in_channels=1, ctx=self.ctx) if self.lambda_D > 0 else None

        self.netGE = extract_encoder(self.netG if self.lambda_aux <= 0 else self.netG.shared_net,
                                     self.num_downs) if (
                (self.lambda_unsup > 0) and (self.compare_embedding_unsup > 0)) else None

        # GAN discriminator
        if self.lambda0 > 0:
            self.netD = Discriminator(in_channels=self.n_A_channel_idx + 1)
            # Initialize parameters
            self.network_init(self.netD)
            self.trainerD = gluon.Trainer(self.netD.collect_params(), self.optimizer,
                                          {'learning_rate': self.base_lr, 'beta1': self.beta1, 'wd': 1e-5},
                                          update_on_kvstore=False)
            # amp.init_trainer(self.trainerD)  # automatic mixed precision

    def def_loss(self):
        # Loss
        self.criterionGAN = gluon.loss.SigmoidBinaryCrossEntropyLoss()
        # self.trueDensity_train = gluon.loss.L1Loss()
        # self.trueDensity_val = gluon.loss.L1Loss()
        loss_fn = {
            'l1': L1Loss_v2,
            'l2': L2Loss_v2,
            'huber': HuberLoss,
            'l2log': L2LogLoss,
            'l1log': L1LogLoss,
            'logcosh': LogCoshLoss,
            'photomtrc': PhotometricLoss,
            'l2org': mx.gluon.loss.L2Loss,
            'rloss': corrcoefLoss,
        }

        self.density_corr = loss_fn['rloss']()
        # self.trueDensity_train = loss_fn[self.l_type](with_DepthAware=self.with_DepthAware)
        self.trueDensity_train = loss_fn[self.l_type]()
        self.trueDensity_val = loss_fn[self.l_type]()
        self.feature_difference = gluon.loss.CosineEmbeddingLoss() if self.lambda_D > 0 else None
        if self.compare_embedding_unsup:
            self.density_unsup = gluon.loss.CosineEmbeddingLoss() if self.lambda_unsup > 0 else None
        else:
            # self.density_unsup = loss_fn[self.l_type](
            #     with_DepthAware=self.with_DepthAware, scale_invar=False) if self.lambda_unsup > 0 else None
            self.density_unsup = loss_fn[self.l_type]()
            # self.density_unsup = loss_fn['photomtrc'](
            #     with_DepthAware=self.with_DepthAware, scale_invar=False) if self.lambda_unsup > 0 else None
        if self.lambda_aux > 0:
            self.aux_fn = loss_fn[self.l_type]() if self.reconstruct_input else DiceLoss()

    def set_metric(self):
        self.metric = mx.metric.CustomMetric(self.facc)

    def set_inputs(self, **kwargs):
        trp = [0, 3, 1, 2]
        for key, value in kwargs.items():
            # self.__setattr__(key, value.transpose(trp).as_in_context(self.ctx).astype(self.dtype))
            self.__setattr__(key, sal(value.transpose(trp), ctx_list=self.ctx, even_split=False))

    def set_lr_scheduler(self):
        """Setup learning rate scheduler"""
        self.lr_steps = [int(lr) for lr in self.lr_steps.split(',')]
        schedules = {
            'one_cycle': OneCycleSchedule(
                start_lr=self.min_lr, max_lr=self.max_lr, cycle_length=self.cycle_length,
                cooldown_length=self.cooldown_length, finish_lr=self.finish_lr, inc_fraction=self.inc_fraction,
            ),
            'triangular': TriangularSchedule(
                min_lr=self.min_lr, max_lr=self.max_lr, cycle_length=self.cycle_length, inc_fraction=self.inc_fraction,
            ),
            'factor': mx.lr_scheduler.FactorScheduler(
                step=self.lr_step, factor=self.lr_factor, warmup_mode=self.warmup_mode,
                warmup_steps=self.warmup_steps, warmup_begin_lr=self.warmup_begin_lr, base_lr=self.base_lr,
            ),
            'multifactor': mx.lr_scheduler.MultiFactorScheduler(
                step=self.lr_steps, factor=self.lr_factor, base_lr=self.base_lr, warmup_mode=self.warmup_mode,
                warmup_begin_lr=self.warmup_begin_lr, warmup_steps=self.warmup_steps,
            ),
            'poly': mx.lr_scheduler.PolyScheduler(
                max_update=self.cycle_length, base_lr=self.base_lr, pwr=2, final_lr=self.min_lr,
            ),
            'cycle': CyclicalSchedule(
                TriangularSchedule, min_lr=self.min_lr, max_lr=self.max_lr, cycle_length=self.cycle_length,
                inc_fraction=self.inc_fraction,
                cycle_length_decay=self.cycle_length_decay,
                cycle_magnitude_decay=self.cycle_magnitude_decay,
                # stop_decay_iter=self.stop_decay_iter,
                final_drop_iter=self.final_drop_iter,
            ),
            'cosine': LinearWarmUp(
                OneCycleSchedule(start_lr=self.min_lr, max_lr=self.max_lr, cycle_length=self.cycle_length,
                                 cooldown_length=self.cooldown_length, finish_lr=self.finish_lr),
                start_lr=self.warmup_begin_lr,
                length=self.warmup_steps,
            )
        }
        self._lr_scheduler = schedules[self.lr_scheduler]

    def compare_unsup(self):
        """Get unsupervised loss"""
        if self.compare_embedding_unsup:
            self.fake_out_unsup = [nd.squeeze(self.netGE(A_unsup)) for A_unsup in self.A_unsup]
            self.fake_out_unsup_aug = [nd.squeeze(self.netGE(A_rp_unsup)) for A_rp_unsup in self.A_rp_unsup]
            if self.lambda_aux > 0:
                self.fake_out_unsup = [fake_out_unsup[0] for fake_out_unsup in self.fake_out_unsup]
                self.fake_out_unsup_aug = [fake_out_unsup_aug[0] for fake_out_unsup_aug in self.fake_out_unsup_aug]
            self.unsup_loss = [
                self.density_unsup(
                    fake_out_unsup,
                    fake_out_unsup_aug,
                    nd.ones(fake_out_unsup.shape[0], ctx=fake_out_unsup.context),
                )
                for fake_out_unsup, fake_out_unsup_aug
                in zip(self.fake_out_unsup, self.fake_out_unsup_aug, )]
        else:
            self.fake_out_unsup = [self.netG(A_unsup) for A_unsup in self.A_unsup]
            self.fake_out_unsup_aug = [nd.flip(self.netG(A_rp_unsup), 3) for A_rp_unsup in self.A_rp_unsup]
            if self.lambda_aux > 0:
                self.fake_out_unsup = [fake_out_unsup[0] for fake_out_unsup in self.fake_out_unsup]
                self.fake_out_unsup_aug = [fake_out_unsup_aug[0] for fake_out_unsup_aug in self.fake_out_unsup_aug]

            self.fake_out_unsup = [nd.where(wp_unsup, fake_out_unsup, wp_unsup - 1) for wp_unsup, fake_out_unsup in
                                   zip(self.wp_unsup, self.fake_out_unsup)]

            self.fake_out_unsup_aug = [nd.where(wp_unsup, fake_out_unsup_aug, wp_unsup - 1) for
                                       wp_unsup, fake_out_unsup_aug in zip(self.wp_unsup, self.fake_out_unsup_aug)]

            self.unsup_loss = [
                self.density_unsup(
                    fake_out_unsup,
                    fake_out_unsup_aug,
                    wp_unsup,
                    # _margin_unsup / self.C_thr,
                    None,
                )
                for fake_out_unsup, fake_out_unsup_aug, wp_unsup, _margin_unsup
                in zip(self.fake_out_unsup, self.fake_out_unsup_aug, self.wp_unsup, self._margin_unsup)]
            if self.monitor_unsup_outputs:
                im = np.hstack(
                    (montage(self.fake_out_unsup[0].asnumpy()[:9, 0]),
                     montage(self.fake_out_unsup_aug[0].asnumpy()[:9, 0]),
                     montage(
                         np.abs(
                             self.fake_out_unsup[0].asnumpy()[:9, 0] - self.fake_out_unsup_aug[0].asnumpy()[:9, 0]))))

                [plt.imsave('%s/ep%04d_%02d_%d' % (
                    self.result_folder_figure_train_unsup, self.current_epoch, self.current_it, i), im) for i in
                 range(1)]

    def optimize_D(self):
        if hasattr(self, 'A_rp_unsup'):  # choose unsup data if avail.
            tmp_input = self.A_rp_unsup
        else:
            tmp_input = self.A_rp
        fake_out = [self.netG(A_rp) for A_rp in tmp_input]
        fake_out = [fo[0] if self.lambda_aux > 0 else fo for fo in fake_out]
        if hasattr(self, 'wp_unsup'):
            tmp_wp = self.wp_unsup
        else:
            tmp_wp = self.wp
        fake_out = [nd.where(wp, fo, wp - 1) for wp, fo in zip(tmp_wp, fake_out)]
        fake_concat = [self.image_pool.query(nd.concat(A_rp, fo, dim=1)) for A_rp, fo in zip(self.A_rp, fake_out)]
        with autograd.record():
            # Train with fake image
            # Use image pooling to utilize history images
            output = [self.netD(fc) for fc in fake_concat]
            fake_label = [nd.zeros_like(op) for op in output]
            err_DB_fake = [self.criterionGAN(op, fl) for op, fl in zip(output, fake_label)]
            [self.metric.update([fl, ], [op, ]) for fl, op in zip(fake_label, output)]
            # self.metric.update([fake_label[0], ], [output[0], ])

            # Train with real image
            real_concat = [nd.concat(A_rp, _C, dim=1) for A_rp, _C in zip(self.A_rp, self.C)]
            output = [self.netD(rc) for rc in real_concat]
            real_label = [nd.ones_like(op) for op in output]
            err_DB_real = [self.criterionGAN(op, rl) for op, rl in zip(output, real_label)]
            self.err_DB = [(edb + edf) * 0.5 for edb, edf in zip(err_DB_real, err_DB_fake)]
            [self.metric.update([rl, ], [op, ]) for rl, op in zip(real_label, output)]

        for err_DB in self.err_DB:
            err_DB.backward()
        # with amp.scale_loss(self.err_DB, self.trainerD) as scaled_loss:
        #     autograd.backward(scaled_loss)

        self.trainerD.step(self.batch_size)

    def create_net(self, upscale_factor=1):
        from mxnet.gluon import nn
        import mxnet.gluon.contrib.nn as contrib_nn

        def conv_factory(opts, num_filters, kernel_size, stride=1, group=1):
            """A convenience function for convolution with BatchNorm & activation"""
            pad = int((kernel_size - 1) / 2)
            out = nn.HybridSequential()
            out.add(nn.BatchNorm())
            if opts.activation == 'leaky':
                out.add(nn.LeakyReLU(opts.alpha))
            else:
                out.add(nn.Activation(opts.activation))

            out.add(nn.Conv2D(channels=num_filters, kernel_size=(kernel_size, kernel_size),
                              strides=(stride, stride), use_bias=opts.use_bias,
                              padding=(pad, pad), groups=group))
            return out

        class Options:
            """"""

            def __init__(self):
                super(Options, self).__init__()
                self.activation = 'relu'
                self.use_bias = False

        class SuperResolutionNet(gluon.HybridBlock):
            def __init__(self, upscale_factor, opts):
                super(SuperResolutionNet, self).__init__()
                with self.name_scope():
                    self.conv1 = conv_factory(opts, num_filters=64, kernel_size=5, stride=1)
                    self.conv2 = conv_factory(opts, num_filters=64, kernel_size=3, stride=1)
                    self.conv3 = conv_factory(opts, num_filters=32, kernel_size=3, stride=1)
                    self.conv4 = conv_factory(opts, num_filters=upscale_factor ** 2, kernel_size=3, stride=1)
                    self.pxshuf = contrib_nn.PixelShuffle2D(upscale_factor)

            def hybrid_forward(self, F, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = F.tanh(self.pxshuf(x))
                return x

        return SuperResolutionNet(upscale_factor, opts=Options())

    def optimize_G(self):
        """Optimize generator"""
        if np.array([self.lambda_C, self.lambda_D, self.lambda_consistency, self.lambda_unsup, self.lambda0,
                     self.lambda_aux]).sum() == 0:  # No extra loss
            with autograd.record():
                self.fake_out = [self.netG(A_rp) for A_rp in self.A_rp]
                self.loss_true_density_train = [self.trueDensity_train(fake_out, C, m, margin) for
                                                C, fake_out, m, margin in
                                                zip(self.C, self.fake_out, self.m, self._margin)]
                self.loss_G = self.loss_true_density_train
                [loss_G.backward() for loss_G in self.loss_G]
        else:
            with autograd.record():
                self.fake_out = [self.netG(A_rp) for A_rp in self.A_rp]
                # Supervised learning
                self.var0 = [nd.square(coef) for coef in self.netG.coef_G._data]
                self.loss_true_density_train = [self.trueDensity_train(fake_out, C, m, margin) for
                                                fake_out, C, m, margin in
                                                zip(self.fake_out, self.C, self.m, self._margin)]
                self.loss_G = [((1 / var) * l + nd.log(var)) for l, var in
                               zip(self.loss_true_density_train, self.var0)]
                ############################### Consistency Loss ###############################
                if self.lambda_consistency > 0:
                    fake_out_T2 = [self.netG(A_rp) for A_rp in
                                   [nd.concat(A_rp[:, 0:1], nd.zeros_like(A_rp[:, 0:1]), dim=1) for A_rp in
                                    self.A_rp]]  # masked out ADC channel
                    fake_out_ADC = [self.netG(A_rp) for A_rp in
                                    [nd.concat(nd.zeros_like(A_rp[:, 1:2]), A_rp[:, 1:2], dim=1) for A_rp in
                                     self.A_rp]]  # masked out T2 channel
                    self.loss_consistency_train = [self.density_corr(_fake_out_T2, _fake_out_ADC, wp) for
                                                   _fake_out_T2, _fake_out_ADC, wp in
                                                   zip(fake_out_T2, fake_out_ADC, self.wp)]
                    self.var1 = [nd.square(coef) for coef in self.netG.coef_consistency._data]
                    self.loss_G = [l0 + ((1 / var) * l1 + nd.log(var)) for l0, l1, var in
                                   zip(self.loss_G, self.loss_consistency_train, self.var1)]
                ############################### Correlation Loss ###############################
                if self.lambda_C > 0:
                    self.var2 = [nd.square(coef) for coef in self.netG.coef_C._data]
                    self.loss_corr_train = [self.density_corr(fake_out, C, m) for
                                            C, fake_out, m in
                                            zip(self.C, self.fake_out, self.m)]
                    self.loss_G = [l0 + ((1 / var) * l1 + nd.log(var)) for l0, l1, var in
                                   zip(self.loss_G, self.loss_corr_train, self.var2)]
                ############################### Unsupervised learning ###############################
                if self.lambda_unsup > 0:
                    self.compare_unsup()
                    self.var3 = [nd.square(coef) for coef in self.netG.coef_unsup._data]
                    self.loss_G = [l0 + ((1 / var) * l1 + nd.log(var)) for l0, l1, var in
                                   zip(self.loss_G, self.unsup_loss, self.var3)]
                ############################## Feature Comparision ###############################
                if self.lambda_D > 0:
                    self.var4 = [nd.square(coef) for coef in self.netG.coef_D._data]
                    self.loss_features = [self.feature_difference(
                        self.D_features(nd.where(m, C, m - 1)),
                        self.D_features(nd.where(m, fake_out, m - 1)),
                        nd.ones((C.shape[0]), ctx=C.context)
                    ).mean() for m, C, fake_out in zip(self.m, self.C, self.fake_out)]
                    self.loss_G = [l0 + ((1 / var) * l1 * .1 + nd.log(var)) for l0, l1, var in
                                   zip(self.loss_G, self.loss_features, self.var4)]

                [loss_G.backward() for loss_G in self.loss_G]

        self.trainerG.step(1, ignore_stale_grad=False)
        if self.use_l_coefs:
            self.trainer_coefs.step(1, ignore_stale_grad=False)

        [self.save_training_outputs(self.A_rp[i], self.fake_out[i], self.C[i], self.m[i], prefix='',
                                    suffix='_%02d_%d' % (self.current_it, i)) if self.monitor_training_outputs else None
         for i in range(len(self.ctx))]

    def update_running_loss(self, first_iter=False, num_batch=None):
        """Compute running loss"""
        if num_batch is None:
            if first_iter:
                loss_fields = [field for field in self.__dict__.keys() if ('loss' in field) or ('err' in field)]
                self.running_loss_fields = ['running_' + field for field in loss_fields]
                [self.__setattr__(field, 0.) for field in self.running_loss_fields]
            for loss_field in self.running_loss_fields:
                _loss = nd.concatenate(list(self.__getattribute__(loss_field.replace('running_', ''))))
                self.__setattr__(loss_field, (self.__getattribute__(loss_field) + _loss.mean().asscalar()))
        else:
            for loss_field in self.running_loss_fields:
                self.__setattr__(loss_field, (self.__getattribute__(loss_field) / num_batch))

    def update_mxboard(self, sw, epoch, val_data=None):
        """ SHOW STATS AND IMAGES ON TENSORBOARD. THIS SHOULD BE RUN AFTER RUnNNING UPDATE_RUNNING_LOSS """
        for loss_field in self.running_loss_fields:
            _loss = self.__getattribute__(loss_field)
            _loss = _loss.mean().asscalar() if isinstance(_loss, nd.NDArray) else _loss.mean()
            if 'loss_true_density' in loss_field:  # True density
                sw.add_scalar('loss/true_density_loss', _loss, global_step=epoch)
            else:  # GAN loss
                loss_type = loss_field.split('_')[0] + '_' + \
                            loss_field.split('_')[1] + '_' + \
                            loss_field.split('_')[2]
                # sw.add_scalar('loss/' + loss_type, {loss_field: _loss}, global_step=epoch)
                sw.add_scalar('loss/' + loss_type, _loss, global_step=epoch)
        if hasattr(self, 'running_loss_true_density_val'):
            sw.add_scalar('loss/true_density_loss_val', self.running_loss_true_density_val, global_step=epoch)

        metric_list = metrics.update_mxboard_metric_v1(sw, data=val_data, global_step=epoch,
                                                       metric_names=[
                                                           'r_whole', 'l1_whole', 'ssim_whole',
                                                           'rmse_whole', 'rmse_log_whole',
                                                           't1', 't2', 't3',
                                                           'abs_rel_diff', 'sqr_rel_diff',
                                                           'ta1', 'ta2',
                                                       ],
                                                       prefix='validation_',
                                                       num_input_channels=self.n_A_channel_idx,
                                                       c_thr=self.C_thr,
                                                       density_range=self.density_range,
                                                       root=self.root)  # 'r', 'l1', 'ssim', 'nmi',
        # if hasattr(self, 'current_margin'):
        sw.add_scalar('loss_margin', self.current_margin, global_step=epoch)
        #######################################
        # Map input data to 0 - 1
        for c in range(val_data[0].shape[1]):
            val_data[0][:, c] = (val_data[0][:, c] - val_data[0][:, c].min()) / (
                    val_data[0][:, c].max() - val_data[0][:, c].min()) * val_data[4][:, 0]
        """ MULTIPLE CHANNELS OF EACH IMAGE ARE SPLIT INTO SEPARATE IMAGES """
        _val_data = []
        for i in range(len(val_data)):
            for j in range(val_data[i].shape[1]):
                _val_data.append(val_data[i][:, j:j + 1])
        #######################################
        """ NORM TO 0-1 RANGE IF NECESSARY """
        if self.to_11:  # Normalize image from [-1, 1] to [0, 1]
            for i in range(-4, -2):  # prediction and label
                _val_data[i] = self.normalize_01(_val_data[i], [-1, 1]) * _val_data[-1]
        #######################################
        """ SAVE FIRST IMAGE TO FOLDER & UPDATE BEST METRICS """
        to_save_montage = self.update_best_metrics(metric_list)
        print(self.best_metrics)
        if to_save_montage:
            self.save_montage_im(_val_data)
        #######################################
        """ DROP LAST CHANNEL (WP) IN _val_data BECAUSE IT IS NO LONGER NECESSARY """
        _val_data = _val_data[:-1]
        #######################################
        return metric_list

    @staticmethod
    def linear_scale(x, vmin=-1, vmax=1, tmin=0, tmax=1):
        return ((x - vmin) / (vmax - vmin)) * (tmax - tmin) + tmin

    def _gen_unsup_pred(self):
        """Generate predictions for unsupvised data"""
        input_list, pred_list, wp_list = [], [], []

        for i, (_, _, C, m, wp, A_rp) in enumerate(self.val_iter):
            # Inputs to GPUs (or CPUs)
            self.set_inputs(A_rp_val=A_rp, C_val=C, m_val=m, wp_val=wp)
            pred = nd.concatenate([self.netG(A_rp_val) for A_rp_val in self.A_rp_val])

            # merge data across all used GPUs
            self.C_val, self.m_val, self.A_rp_val, self.wp_val = [
                nd.concatenate(list(x)) for x in [self.C_val,
                                                  self.m_val,
                                                  self.A_rp_val,
                                                  self.wp_val]
            ]
            wp_val = nd.tile(self.wp_val, (1, pred.shape[1], 1, 1))
            pred = nd.where(wp_val, pred, wp_val - 1)  # wp-masked

            input_list.append(self.A_rp_val.asnumpy())
            pred_list.append(self.linear_scale(pred.asnumpy()))
            wp_list.append(self.wp_val.asnumpy())

        return np.concatenate(input_list), \
               np.concatenate([*pred_list]), \
               np.concatenate([*wp_list]),  # duplicate to simplify the modifications

    def save_montage_im(self, im, prefix=''):
        _im = np.squeeze(im)[:-2]
        _im_contour = np.tile(np.squeeze(im)[-2], (len(im) - 2, 1, 1, 1))
        _im_wp = np.tile(np.squeeze(im)[-1], (len(im) - 2, 1, 1, 1))
        _im_wp[_im_wp == 1] = 2
        for i in range(self.n_A_channel_idx):
            _im_wp[i] = _im[i]

        _im_wp = montage(np.concatenate(_im_wp, axis=2))
        _im = montage(np.concatenate(_im, axis=2))
        _im_wp = masked_array(_im_wp, _im_wp == 2)
        plt.imshow(_im, cmap='jet', vmin=0, vmax=.7, interpolation='nearest')
        plt.imshow(_im_wp, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        plt.contour((montage(np.concatenate(_im_contour, axis=2))).astype(int), linewidths=.14, colors='white')
        self.save_fig(folder=self.result_folder_figure_test) if self.test_mode else self.save_fig(
            folder=self.result_folder_figure_val)

    def save_fig(self, folder, prefix='', suffix='', dpi=500):
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        plt.savefig(
            '%s/%sep%04d%s.png' % (folder, prefix, self.current_epoch, suffix),
            pad_inches=0,
            bbox_inches='tight',
            transparent=True,
            dpi=dpi,
        )
        plt.close('all')

    def update_best_metrics(self, metric_list):
        to_save_montage = False
        """update current best metrics"""
        if metric_list['r_whole'].mean() > self.best_metrics['r_whole_best']:
            self.best_metrics['r_whole_best'] = metric_list['r_whole'].mean()
            to_save_montage = True

        if metric_list['l1_whole'].mean() < self.best_metrics['l1_whole_best']:
            self.best_metrics['l1_whole_best'] = metric_list['l1_whole'].mean()
            to_save_montage = True

        if to_save_montage:
            if self.best_metrics['r_whole_best'] < .3:
                to_save_montage = False
        if self.current_epoch == 0:
            to_save_montage = True

        to_save_montage = False
        if self.current_it == 1999:
            to_save_montage = True

        return to_save_montage

    def save_checkpoints(self):
        """Saved parameters"""
        self.result_folder_checkpoint_current_iter = '%s/iter_%04d' % (
            self.result_folder_checkpoint, self.current_it)
        os.makedirs(self.result_folder_checkpoint_current_iter) if not os.path.exists(
            self.result_folder_checkpoint_current_iter) else None

        self.netG_filename = '%s/netG.params' % (self.result_folder_checkpoint_current_iter,)
        self.netG.save_parameters(self.netG_filename)

    def load_checkpoints(self, prefix='best_', pretrained_dir=None):
        if pretrained_dir:
            self.netG.load_parameters(pretrained_dir, ctx=self.ctx,
                                      ignore_extra=True)
        else:
            self.result_folder_checkpoint_iter = '%s/iter_%04d' % (
                self.result_folder_checkpoint, self.resumed_it) if self.resumed_it > -1 else '%s/%scheckpoints' % (
                self.result_folder_checkpoint, prefix)

            self.netG_filename = '%s/netG.params' % (self.result_folder_checkpoint_iter,)

            """Load parameters from checkpoints"""
            self.netG.load_parameters(self.netG_filename, ctx=self.ctx,
                                      ignore_extra=True)

    def hybridize_networks(self):
        if self.lambda0 > 0:
            self.netD.hybridize(static_alloc=True, static_shape=True)
        if self.lambda_D > 0:
            self.D_features.hybridize(static_alloc=True, static_shape=True)
        # self.D_features_unsup.hybridize(static_alloc=True, static_shape=True)
        self.netG.hybridize(static_alloc=True, static_shape=True)

    @staticmethod
    def param_init(param, ctx):
        """Initialize discriminator parameters"""
        if param.name.find('conv') != -1:
            if param.name.find('weight') != -1:
                param.initialize(init=mx.init.Normal(0.02), ctx=ctx)
            else:
                param.initialize(init=mx.init.Zero(), ctx=ctx)
        elif param.name.find('batchnorm') != -1:
            param.initialize(init=mx.init.Zero(), ctx=ctx)
            # Initialize gamma from normal distribution with mean 1 and std 0.02
            if param.name.find('gamma') != -1:
                for _ctx in ctx:
                    param.set_data(nd.random_normal(1, 0.02, param.data(ctx=_ctx).shape))

    @staticmethod
    def chunks(l, n):
        n = max(1, n)
        return (l[i:i + n] for i in range(0, len(l), n))

    @staticmethod
    def normalize_01(img, predef_range=None, ):
        if predef_range is None:
            return (img - img.min()) / (img.max() - img.min())
        else:
            return (img - predef_range[0]) / (predef_range[1] - predef_range[0])

    @staticmethod
    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()

    @staticmethod
    def resize_wp(wp, ref):
        wp = resize(wp.asnumpy(), (wp.shape[0], wp.shape[1], ref.shape[-2], ref.shape[-1]), order=0,
                    anti_aliasing=False)
        return nd.array(wp, ctx=ref.context)

    @staticmethod
    def show_im(im, to_show=True):
        im = im.asnumpy()[:, 0]
        plt.imshow(montage(im), cmap='gray')
        plt.show() if to_show else None

    @staticmethod
    def create_concat_image(_val_data):
        for i in range(_val_data[0].shape[0]):
            info = regionprops(_val_data[-1][i, 0].astype(int))
            # masking predictions and ground truth images
            _val_data[-2][i] *= _val_data[-1][i]
            _val_data[-3][i] *= _val_data[-1][i]
            val_data_cropped = []
            for j in range(len(_val_data) - 1):
                val_data_cropped.append(
                    _val_data[j][i, :, info[0].bbox[0]: info[0].bbox[2], info[0].bbox[1]:info[0].bbox[3]])
                val_data_cropped_rsz = resize(np.asarray(val_data_cropped),
                                              (val_data_cropped.__len__(), _val_data[-3].shape[1], 200, 200))
                tmp = np.concatenate([val_data_cropped_rsz[k, 0] for k in range(val_data_cropped_rsz.shape[0])], axis=1)
            img_concat = tmp if i == 0 else np.concatenate([img_concat, tmp], axis=0)

        img_concat[img_concat < 0] = 0
        img_concat[img_concat > 1] = 1
        return img_concat

    def save_training_outputs(self, img, pred, label, roi, prefix, suffix):
        def my_concat(im1, im2, im3):
            im_cc = nd.concatenate([im1, im2, im3], axis=1)
            im_cc = nd.concatenate([*nd.concatenate([*nd.transpose(im_cc, (1, 0, 2, 3))], axis=2)], axis=0).asnumpy()
            return im_cc

        a = my_concat(img, pred, label)
        roi_masked = my_concat(nd.ones_like(img), roi * -1 + 1, roi * -1 + 1)
        roi_masked = masked_array(a, roi_masked == 1)
        img_masked = my_concat(img, nd.ones_like(roi), nd.ones_like(roi))
        img_masked = masked_array(a, img_masked == 1)
        plt.imshow(a, cmap='gray', vmin=self.density_range[0], vmax=self.density_range[1])
        plt.imshow(img_masked, cmap='gray', vmin=0, vmax=1)
        plt.imshow(roi_masked, cmap='jet', vmin=self.density_range[0], vmax=self.density_range[1])
        plt.axis('off')

        self.save_fig(folder=self.result_folder_figure_train, prefix=prefix, suffix=suffix, dpi=500)

    def generate_test_figures(self, val_data):
        """ MULTIPLE CHANNELS OF EACH IMAGE ARE SPLIT INTO SEPARATE IMAGES """
        _val_data = []
        for i in range(len(val_data)):
            for j in range(val_data[i].shape[1]):
                _val_data.append(val_data[i][:, j:j + 1])
        #######################################
        """ NORM TO 0-1 RANGE IF NECESSARY """
        if self.to_11:  # Normalize image from [-1, 1] to [0, 1]
            for i in range(-4, -2):  # prediction and label
                _val_data[i] = self.normalize_01(_val_data[i], [-1, 1]) * _val_data[-1]

        if self.norm_0mean:  # norm inputs
            for i in range(val_data.__len__() - 4):  # excludes (pred, label, ROIs and wp)
                _val_data[i] = self.normalize_01(_val_data[i]) * _val_data[-1]
        #######################################
        """ SAVE FIRST IMAGE TO FOLDER & UPDATE BEST METRICS """
        self.save_montage_im(_val_data)

    def decay_loss_margin(self, margin):
        """Get loss margin by iteration index"""
        self.current_margin = self.lss_margin.get_margin(self.current_it)
        for _mg in margin:
            _mg = self.current_margin
        return margin

    def fade_signal(self, m):
        """Remove training signal with respect to the current training iteration"""
        num_signal = int(
            min(np.floor(self.current_it / (self.total_iter * (9 / 10) / m.shape[0])) + 1, m.shape[0]))
        if num_signal == self.batch_size:
            return m
        else:
            signal = np.zeros(self.batch_size)[:, np.newaxis, np.newaxis, np.newaxis]
            signal[np.random.permutation(self.batch_size)[:num_signal]] = 1
            return m * nd.array(signal)

    def expand_dataset(self):
        """Gradually Expand dataset with respect to the current training iteration"""
        if self.num_expand_level <= 1:
            return
        current_it = self.trainerG.optimizer.num_update
        interval = int((self.total_iter * (9 / 10)) / self.num_expand_level)
        self.train_iter._batch_sampler._sampler._length1 = int(min(
            self.data_inc_unit * (np.floor(current_it / interval) + 1),
            self.train_iter._dataset.__len__()))


class ImagePool:
    """Pooling"""

    def __init__(self, pool_size):
        """Initialization"""
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        ret_imgs = []
        for i in range(images.shape[0]):
            image = nd.expand_dims(images[i], axis=0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                ret_imgs.append(image)
            else:
                p = nd.random_uniform(0, 1, shape=(1,)).asscalar()
                if p > 0.5:
                    random_id = nd.random_uniform(0, self.pool_size - 1, shape=(1,)).astype(np.uint8).asscalar()
                    tmp = self.images[random_id].copy()
                    self.images[random_id] = image
                    ret_imgs.append(tmp.as_in_context(image.context))
                else:
                    ret_imgs.append(image)
        ret_imgs = nd.concat(*ret_imgs, dim=0)
        return ret_imgs


def get_color():
    """
    :return: a colormap jet
    """
    from pylab import cm
    cm_jet = np.reshape(np.concatenate([cm.jet(i) for i in range(255)], axis=0), (255, 4))
    return cm_jet[:, :3]


JET = get_color()


class ColorBar:
    def __init__(self, height, width):
        """"""
        bar_tmp = np.zeros((101, 10))
        for i in range(100):
            bar_tmp[i] = 99 - i + 1
        # bar position
        self.end_bar_x = int(width - 3)
        self.start_bar_x = int(self.end_bar_x - 10 + 1)
        self.start_bar_y = int(np.floor((height - 101) / 2))
        self.end_bar_y = int(self.start_bar_y + 101 - 1)
        # Index Image generation
        self.bar = bar_tmp * 2.551  # colorbar scale from 0 to 255
        self.numimg = np.load(r'extra_data/mr_collateral_numing.npy') / 255

    def insert_num_board(self, rgbIMG, maxIMG, maxWWL):
        """
        :param indIMG: a 2D image
        :return: add min - max numbers to the colorbar
        """
        # insert min number
        for ch in range(rgbIMG.shape[-1]):
            rgbIMG[self.end_bar_y:self.end_bar_y + 9, self.end_bar_x - 6:self.end_bar_x, ch] = self.numimg[..., 0, 0]

        # insert max number
        max_num_str = str((maxIMG * maxWWL * 255).astype('uint8'))
        str_length = len(max_num_str)
        num_board = np.zeros((9, str_length * 6, 3))
        for i in range(str_length):
            selected_num = int(max_num_str[i])
            num_board[:, i * 6:(i + 1) * 6, 0] = self.numimg[:, :, 0, selected_num]
            num_board[:, i * 6:(i + 1) * 6, 1] = self.numimg[:, :, 0, selected_num]
            num_board[:, i * 6:(i + 1) * 6, 2] = self.numimg[:, :, 0, selected_num]
        rgbIMG[self.start_bar_y - 9:self.start_bar_y, self.end_bar_x - str_length * 6 + 1:self.end_bar_x + 1] = \
            num_board
        return rgbIMG

    def insert_color_bar(self, indIMG):
        """
        :param indIMG: a 2D image
        :return: an image with a color bar on the right side
        """
        # insert bar
        indIMG[self.start_bar_y:self.end_bar_y + 1, self.start_bar_x:self.end_bar_x + 1] = self.bar
        return indIMG


def gen_dicoms(save_dir_path, density, mri, mask, suffix='', insert_bar=True):
    # Populate required values for file meta information (pseudo-values)
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"
    hdr = FileDataset('collateral_phase_maps', {}, file_meta=file_meta, preamble=b"\0" * 128)

    print('Generating DICOM files...')
    density = density.transpose([1, 0, 2, 3])
    mask = mask.transpose([1, 0, 2, 3])
    T2 = normalize_01(mri.transpose([1, 0, 2, 3])[0:1]) * mask
    ADC = normalize_01(mri.transpose([1, 0, 2, 3])[1:2]) * mask
    combined = np.concatenate((T2, ADC, density), 0)

    number_of_slice = combined.shape[1]

    outlier = [0, 0, 0]
    col_sf = 1
    density_dir = 'DICOM' + suffix
    presentations = ['GrayScale', 'Color']
    types = ['T2', 'ADC', 'EPI-Density']
    density_dir_org = save_dir_path
    density_dir_paths = {presentations[0]: {}, presentations[1]: {}}
    for presentation in presentations:
        for k in range(combined.shape[0]):
            density_dir_paths[presentation][k] = density_dir_org + '/' + presentation + '/' + '%d_%s' % (k, types[k])
            if not os.path.exists(density_dir_paths[presentation][k]):
                os.makedirs(density_dir_paths[presentation][k])

    new_sn_gray = 10000 + 1
    new_sn_color = 15000 + 1

    color_bar = ColorBar(*density.shape[-2:]) if insert_bar else None
    # Save Dicom Files
    [save_color_dcm(combined[k, slice_loop], hdr, new_sn_color, k, slice_loop, outlier, density_dir_paths['Color'][k],
                    mask[0, slice_loop], color_bar, col_sf,
                    rescaling_first=False)
     for k in [len(combined) - 1, ] for slice_loop in range(number_of_slice)]

    [save_grayscale_dcm(combined[k, slice_loop], hdr, new_sn_gray, k, slice_loop, outlier,
                        density_dir_paths['GrayScale'][k],
                        mask[0, slice_loop], col_sf)
     for k in range(len(combined) - 1) for slice_loop in range(number_of_slice)]


def save_color_dcm(phase_map, ds, new_sn, k, slice_loop, outlier, col_dir_path, mask, color_bar=None, col_sf=1,
                   num_bits=8, rescaling_first=False):
    """
    Save a phase map into a gray scale dicom series
    :param num_bits:
    :param rescaling_first:
    :param color_bar:
    :param phase_map: 2D phase map
    :param ds: pydicom dataset instance
    :param new_sn: new serial number
    :param k: phase index
    :param slice_loop: slice index
    :param outlier: outlier list
    :param col_dir_path: destination directory storing phase maps
    :param mask: a brain mask
    :param col_sf: *
    """
    SeriesDescription = [
        'T2', 'ADC', 'Density-EPI'
    ]
    phase_map = mr_collateral_gen_color_image(phase_map, outlier, mask, rescaling_first, color_bar)
    phase_map = (phase_map * (2 ** num_bits - 1)).astype('uint%d' % num_bits)
    ds.SeriesDescription = SeriesDescription[k]
    ds.SeriesInstanceUID = str(new_sn + k)
    ds.SeriesNumber = new_sn + k
    ds.AcquisitionNumber = slice_loop
    ds.InstanceNumber = slice_loop
    ds.PixelSpacing = [1, 1]
    ds.PixelData = phase_map.tostring()
    ds.Rows, ds.Columns = phase_map.shape[:2]
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = 'RGB'
    ds.BitsAllocated = num_bits
    ds.BitsStored = num_bits
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SmallestImagePixelValue = 0
    ds.LargestImagePixelValue = 255
    ds.WindowCenter = 128
    ds.WindowWidth = 255
    path = "%s/%s_%03d.dcm" % (col_dir_path, ds.SeriesDescription, slice_loop)
    ds.save_as(path)


def mr_collateral_gen_color_image(inIMG, outlier, mask, rescaling_first=False, color_bar=None):
    """

    :param inIMG:
    :param outlier:
    :param mask:
    :param rescaling_first:
    :param color_bar:
    :return:
    """
    if mask.sum() == 0:
        rgb = np.zeros((inIMG.shape + (3,)))
        return rgb

    minIMG, maxIMG = inIMG[mask > 0].min(), inIMG[mask > 0].max()
    if rescaling_first:
        if mask is None:
            mask = np.ones_like(inIMG)
        # Image Signal Normalization
        nIMG = inIMG / maxIMG
        outlier_low = (outlier / 100) / 2
        outlier_high = 1 - ((outlier / 100) / 2)
        WWL = stretch_lim(nIMG, [outlier_low, outlier_high])  # auto window width / level

        minWWL, maxWWL = WWL.min(), WWL.max()
        # Rescaled Image
        # rsIMG = imadjust(nIMG, WWL, [])
    else:
        rsIMG = inIMG
        maxWWL = 1

    indIMG = rsIMG * 255
    indIMG = color_bar.insert_color_bar(indIMG) if color_bar else indIMG
    rgb = label2rgb(indIMG.astype('uint8'), bg_label=0, colors=JET)
    # TODO: I cannot understand why the intensity of the bar is higher than the brain intensity, regarding 'indIMG', but seems to be the equal regarding 'rgb'
    rgb = color_bar.insert_num_board(rgbIMG=rgb, maxIMG=maxIMG, maxWWL=maxWWL) if color_bar else rgb
    return rgb


def stretch_lim(img, tol):
    """
    Mimic the stretchlim function in MATLAB
    :param img:
    :param tol:
    :return:
    """
    nbins = 65536
    tol_low = tol[0]
    tol_high = tol[1]
    N = np.histogram(img, nbins)[0]
    cdf = np.cumsum(N) / sum(N)  # cumulative distribution function
    ilow = np.where(cdf > tol_low)[0][0]
    ihigh = np.where(cdf >= tol_high)[0][0]
    if ilow == ihigh:  # this could happen if img is flat
        ilowhigh = np.array([1, nbins])
    else:
        ilowhigh = np.array([ilow, ihigh])
    lowhigh = ilowhigh / (nbins - 1)  # convert to range [0 1]
    return lowhigh


def save_grayscale_dcm(phase_map, ds, new_sn, k, slice_loop, outlier, col_dir_path, mask, col_sf=1, num_bits=16):
    """
    Save a phase map into a gray scale dicom series
    :param num_bits:
    :param phase_map: 2D phase map
    :param ds: pydicom dataset instance
    :param new_sn: new serial number
    :param k: phase index
    :param slice_loop: slice index
    :param outlier: outlier list
    :param col_dir_path: destination directory storing phase maps
    :param mask: a brain mask
    :param col_sf: *
    """
    SeriesDescription = [
        'T2', 'ADC', 'Density-EPI'
    ]
    phase_map = (phase_map * (2 ** num_bits - 1)).astype('uint%d' % num_bits)
    MIN, MAX, WW, WL = mr_collateral_find_WWL(phase_map * col_sf, mask, outlier[k])
    ds.SeriesDescription = SeriesDescription[k]
    ds.SeriesInstanceUID = str(new_sn + k)
    ds.SeriesNumber = new_sn + k
    ds.AcquisitionNumber = slice_loop
    ds.InstanceNumber = slice_loop
    ds.PixelSpacing = [1, 1]
    ds.PixelData = phase_map.tostring()
    ds.Rows, ds.Columns = phase_map.shape[:2]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = num_bits
    ds.BitsStored = num_bits
    ds.SmallestImagePixelValue = int(MIN)
    ds.LargestImagePixelValue = int(MAX)
    ds.WindowCenter = int(WL)
    ds.WindowWidth = int(WW)
    ds.PixelRepresentation = 0
    path = "%s/%s_%03d.dcm" % (col_dir_path, ds.SeriesDescription, slice_loop)
    ds.save_as(path)


def mr_collateral_find_WWL(inIMG, mask, outlier):
    """

    :param inIMG:
    :param mask:
    :param outlier:
    :return:
    """
    if mask.sum() == 0:
        return 0, 0, 0, 0
    MIN = inIMG.min()
    MAX = inIMG.max()

    # Image Signal Normalization
    nIMG = inIMG / MAX
    outlier_low = (outlier / 100) / 2
    outlier_high = 1 - ((outlier / 100) / 2)
    WWL = stretch_lim(nIMG[mask > 0], [outlier_low, outlier_high])  # auto window width / level
    minWWL = WWL.min()
    maxWWL = WWL.max()

    # Window width / level calculation
    WW = np.floor((MAX * maxWWL) - (MAX * minWWL))
    WL = np.floor(MAX * minWWL) + np.floor(WW / 2)
    return MIN, MAX, WW, WL


def normalize_01(img, predef_range=None, ):
    if predef_range is None:
        return (img - img.min()) / (img.max() - img.min())
    else:
        return (img - predef_range[0]) / (predef_range[1] - predef_range[0])
