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
from utils.datasets import RadPathV1 as RadPath
from utils.learning_rate_scheduler import *
from utils.load_pretrained_net import pretrained_net, extract_encoder
from utils.losses import L1Loss_v2, L2Loss_v2, DiceLoss, L2LogLoss, L1LogLoss, LogCoshLoss, PhotometricLoss, \
    corrcoefLoss, HuberLoss, SoftmaxEmbeddingLoss
from utils.optimizers import get_optimizer_dict
from utils.sampler import BatchSampler, RandomSamplerStratify2Sets
from utils.transformations_multi_maps import joint_transform
from utils.loss_margin import LossMargin
import re

from imgaug.augmentables.heatmaps import HeatmapsOnImage

filename_unsup_pred = 'results/ImprovedSemi_PostMICCAI_4//drnnGR4_lCL0_EPI_lD0_lUS1_lEUS0_lC0_l2_nc5_normal_sd1216_lrscycle_CLD.93_CMD.95_CL100_mxLR1e-3_mnLR1e-4_CDL600_fl1e-7_bs8_us2_trp0.60_use200_nel1_bLR1e-4_r1_v1aEmbeddingUnsup_ShapeAugmentBoth_1_TSA0.90//inference/iter_2499/pred_unsup'
filename_pretrained_weights = 'results/ByTime_Calibrate_Seeds//drnnGR4_mrg0_mdr0_lD0_lUS1_lC0_l2_nc5_normal_sd114_lrscycle_CLD.9_CMD.95_CL100_mxLR1e-3_mnLR1e-4_CDL100_fl1e-6_bs8_us2_trp0.60_use200_v1_tsa_9by10_longerNoLabelRelaxation_SingleGPUs//checkpoints/iter_1799/netG.params'

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
        # self.dataset_root = 'datasets/%s/' % self.dataset_name
        self.run_id += '_TSA%.2f' % self.stsa_rat if (self.stsa_rat != 0) else '_NoTSA'
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
        # suffice = '_adjacent_%s' % self.density_type  # input file always has the '_adjacent_' suffix
        suffice = '_%s' % self.density_type
        self.dataset_file = r'inputs/%s%s.npy' % (self.dataset_file, suffice)
        self.dataset_file_org = r'inputs/%s.npy' % self.dataset_file_org
        self.ENUC_max = None
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

        # Load data
        A, C, mask, wp, caseID_A, caseID_B = self.load_data()

        # Condition to split dataset
        train_idx_A = caseID_A < self.train_threshold
        train_idx_B = caseID_B < self.train_threshold
        # val_idx_A = caseID_A == self.train_threshold
        # val_idx_B = caseID_B == self.train_threshold
        # test_idx_A = caseID_A == (self.train_threshold + 1)
        # test_idx_B = caseID_B == (self.train_threshold + 1)

        val_idx_A = caseID_A >= self.train_threshold  # train_idx_A
        val_idx_B = caseID_B >= self.train_threshold  # train_idx_B  #

        # Blur the test set, blurring for the training set will be done on-the-fly
        from utils.transformations import blur_density_map
        for i in range(C.shape[0]):
            if not train_idx_A[i]:
                for j in range(C.shape[-1]):
                    C[i, :, :, j] = blur_density_map(C[i, :, :, j], wp[i, :, :, 0], d_range=[0, 1])

        if 'EESL' in self.dataset_file:
            C[..., 1] = self.map_ENUC(C[..., 1], mask[..., 0], train_idx_A)
        # Normalization
        A, C, self.density_range, self.C_thr = \
            self.normalize(A, C,
                           mask=wp,
                           to_11=self.to_11,
                           norm_0mean=self.norm_0mean,
                           train_idx_A=train_idx_A,
                           outlier_thr=self.density_outlier_thr,
                           root=self.root,
                           )
        wmask = self.compute_weight_mask(C, mask, train_idx_A) if self.weighting_density else None

        # prefetch = 0 if self.lambda_unsup > 0 else None
        self.filenames = [self.filenames[i] for i in np.where(val_idx_A == True)[0]]
        prefetch = 0

        if self.use_pseudo_labels:
            import pickle
            print('Loading and concatenating pseudo labeled data...')
            with open(filename_unsup_pred, 'rb') as fp:
                unsup_pred = pickle.load(fp)
                num_unsup_pred_used = int(np.floor(
                    unsup_pred[0].shape[0] / 1))  # Try if training on half of unsup pred would yield a same performance
                train_idx_A = np.concatenate((train_idx_A,
                                              np.ones(num_unsup_pred_used).astype(bool)))
                val_idx_A = np.concatenate((val_idx_A,
                                            np.zeros(num_unsup_pred_used).astype(bool)))
                A = np.concatenate((A, unsup_pred[0][:num_unsup_pred_used].transpose([0, 2, 3, 1])))
                # # "+3" is a signal to signal the transformation the presence of pseudo-label
                # C = np.concatenate((C, unsup_pred[1][:num_unsup_pred_used].transpose([0, 2, 3, 1]) + 3))
                C = np.concatenate((C, unsup_pred[1][:num_unsup_pred_used].transpose([0, 2, 3, 1])))
                mask = np.concatenate((mask, unsup_pred[2][:num_unsup_pred_used].transpose([0, 2, 3, 1])))
                wp = np.concatenate((wp, unsup_pred[2][:num_unsup_pred_used].transpose([0, 2, 3, 1])))
            print('Done!')

        if (self.lambda_unsup > 0) or (self.lambda_aux > 0) or (self.lambda_embedding_unsup > 0):
            Aus, wp_us = self.load_data(get_unsup_data=True)
            Aus = self.normalize(Aus, mask=wp_us)
            combined_batch_size = self.batch_size * (self.unsup_ratio + 1)
            sampler = RandomSamplerStratify2Sets(A[train_idx_A].shape[0], Aus.shape[0], self.batch_size,
                                                 self.batch_size * self.unsup_ratio)
            batch_sampler = BatchSampler(sampler, combined_batch_size, last_batch='discard')
            train_dataset = RadPath(
                A[train_idx_A], C[train_idx_A],
                mask[train_idx_A] if wmask is None else wmask[train_idx_A],
                wp[train_idx_A], Aus, wp_us,
                transform=joint_transform,
                is_val=self.no_augmentation,
                input_size=self.input_size,
                not_augment_values=self.not_augment_values,
                density_range=self.density_range,
                margin=self.initial_margin,
                batch_size=self.batch_size,
                batch_size_unsup=self.batch_size * self.unsup_ratio,
            )
        else:
            train_dataset = RadPath(
                A[train_idx_A], C[train_idx_A],
                mask[train_idx_A],
                wp[train_idx_A],
                transform=joint_transform,
                is_val=self.no_augmentation,
                input_size=self.input_size,
                not_augment_values=self.not_augment_values,
                density_range=self.density_range,
                margin=self.initial_margin,
                batch_size=self.batch_size,
                batch_size_unsup=self.batch_size * self.unsup_ratio,
            )
            sampler = RandomSamplerStratify2Sets(A[train_idx_A].shape[0], 0, self.batch_size, 0)
            batch_sampler = BatchSampler(sampler, self.batch_size, last_batch='discard')

        batch_size, shuffle, last_batch = None, None, None
        self.train_iter = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            last_batch=last_batch,
            shuffle=shuffle,
            thread_pool=False,
            batchify_fn=BatchifyFn(batch_size=self.batch_size).batchify_fn,
            prefetch=prefetch,
            batch_sampler=batch_sampler
        )

        if self.test_mode:
            val_dataset = RadPath(A[val_idx_A], C[val_idx_A], mask[val_idx_A], wp[val_idx_A],
                                  transform=joint_transform,
                                  is_val=True,
                                  input_size=self.input_size,
                                  density_range=self.density_range,
                                  )

        if self.gen_unsup_pred:
            val_dataset = RadPath(
                Aus, Aus, wp_us, wp_us, wp_us,  # duplicate to simplify the modifications
                transform=joint_transform,
                is_val=True,
                input_size=self.input_size,
                density_range=self.density_range,
            )

        else:
            val_dataset = RadPath(A[val_idx_A], C[val_idx_A], mask[val_idx_A], wp[val_idx_A],
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
        self.best_metrics = {
            'r_whole_best': 0,
            'l1_whole_best': 999,
        }

        self.lambda_A = 30
        self.lambda_B = 30
        self.lambda_idt = .5 if A.shape[-1] == 1 else 0
        self.lss_margin = LossMargin(total_iter=self.total_iter,
                                     margin_decay_rate=self.margin_decay_rate,
                                     initial_margin=self.initial_margin)

    def save_setting(self):
        """save input setting into a csv file"""
        with open('%s/parameters.csv' % self.result_folder, 'w') as f:
            w = csv.writer(f)
            for key, val in self.__dict__.items():
                w.writerow([key, val])

    def map_ENUC(self, C, m, train_idx):
        """

        :param C: NHWC (number of channel should be one)
        :param m: NHWC
        :param train_idx:
        :return:
        """
        self.ENUC_max = C[train_idx][m[train_idx]==1].max()
        C = C / self.ENUC_max
        C[C > 1] = 1
        return C

    @staticmethod
    def normalize(_A, _C=None, mask=None, to_11=False, norm_0mean=False, train_idx_A=None, outlier_thr=1,
                  root=1, ):
        """Norm A (MRI-T2): filtering top 0.1% values by assigning them to the top_thr (the value at the 99th percentage)
        then map values to [0 1] range by dividing by the max intensity within the prostate for each slide"""
        thr = .01  # .01
        mask = np.ones_like(_A) if mask is None else mask
        if not norm_0mean:
            # for (i, (__A, __m)) in enumerate(zip(_A, mask)):
            #     m = __m[..., 0]
            #     for j in range(__A.shape[-1]):
            #         a = __A[..., j]
            #         if a.ndim == 3:  # when use_adjacent is True
            #             for k in range(a.shape[-1]):
            #                 _a = a[..., k].copy()
            #                 tmp = np.sort(_a[m == 1])
            #                 top_thr = tmp[int((1 - thr) * tmp.__len__())]
            #                 _a[_a > top_thr] = top_thr
            #                 _A[i, ..., k, j] = (_a - _a.min()) / (_a.max() - _a.min()) * m
            #         else:
            #             tmp = np.sort(a[m == 1])
            #             top_thr = tmp[int((1 - thr) * tmp.__len__())]
            #             a[a > top_thr] = top_thr
            #             _A[i, ..., j] = (a - a.min()) / (a.max() - a.min()) * m
            x = np.zeros_like(_A)
            for c in range(_A.shape[-1]):
                for i in range(_A.shape[0]):
                    tmp = _A[i, ..., c][mask[i, ..., 0] == 1].reshape((-1, 1))
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

        def find_threshold(_x=None, _mask=None, _outlier_thr=1):  # .999
            _x = _x[_mask == 1] if _mask is not None else _x
            _x = _x.flatten() if _x.ndim > 1 else _x
            _x = np.sort(_x)
            thr_val = _x[int(_outlier_thr * _x.__len__())]
            return thr_val

        if _C is not None:
            """Norm C (Density maps): simply map density values to [0 1] range by dividing by the max intensity within the prostate, and mask for each slide, respectively"""
            thr = 1 if outlier_thr == 1 else find_threshold(_C[train_idx_A], mask[train_idx_A], outlier_thr)
            thr = np.round(thr, 5)
            _c = _C.transpose([3, 0, 1, 2])
            for i, _C in enumerate(_c):
                _C **= (1 / root)
                _C[_C > thr] = thr
                if to_11:
                    _C = _C * 2 - 1 if to_11 else _C  # the global range of density is [0, 1]
                    density_range = [-1, 1]
                else:
                    _C = (_C - _C[mask == 1].min()) / (_C[mask == 1].max() - _C[mask == 1].min())
                    density_range = [0, 1]
                _c[i] = _C.copy()
            _C = _c.transpose([1, 2, 3, 0])
            return _A, _C, density_range, thr
        else:
            return _A

    @staticmethod
    def compute_weight_mask(_C, mask=None, train_idx=None):
        mask = np.ones_like(_C) if mask is None else mask
        train_idx = np.ones(_C.shape[0]) if train_idx is None else train_idx
        tr = _C[train_idx == 1][mask[train_idx == 1] == 1]
        trhist, bins = np.histogram(tr, 50)
        # t = _C[train_idx_A == 0][mask[train_idx_A == 0] == 1]
        # thist = np.histogram(t, 50)
        thrhist_inv = 1 / (trhist / trhist.sum())
        wmask = np.zeros_like(mask)
        for k in range(len(bins) - 1):
            wmask[np.where((_C >= bins[k]) & (_C < bins[k + 1]))] = thrhist_inv[k]
        return wmask

    def load_data(self, get_unsup_data=False):
        """Load all Numpy"""
        if '6channels' in self.dataset_file:
            n_A_channel_idx = 3
        else:
            n_A_channel_idx = 2

        if not get_unsup_data:
            transpose_dims = (2, 0, 1, 3) if not self.use_adjacent else (2, 0, 1, 3, 4)
            print('Loading input file...')
            print(self.dataset_file)
            x = np.load('%s' % self.dataset_file)
            print('Done!')
            filenames = open('inputs/filenames_pre_excluded.txt', 'r')
            self.filenames = [f.replace('\n', '') for f in filenames.readlines()]
            filenames.close()
            if (not self.use_adjacent) and (x.ndim == 5):
                x[..., 0, 0:1] = x[..., 1, 0:1].copy()
                x = x[..., 0, :]
            x = x.transpose(transpose_dims)
            # Binarize the WP and ROIs
            x[..., -1] = (x[..., -1] > 0).astype(x.dtype)
            x[..., -2] = (x[..., -2] > 0).astype(x.dtype)
            # Remove NaN value (especially for density maps)
            x[np.isnan(x)] = 0
            if self.lambda_aux == 0:  # masking inputs with wp
                if not self.use_adjacent:
                    x *= x[..., -2:-1]
                else:
                    x *= x[..., 0:1, -2:-1]

            caseID = np.load('inputs/%s.npy' % self.caseID_file).squeeze()

            # Discard cases with #pixels in each roi_mask < 100
            if not self.use_adjacent:
                n_pix_mask = x[..., -1].sum(axis=(1, 2))
            else:
                n_pix_mask = x[..., 0, -1].sum(axis=(1, 2))

            caseID = caseID[np.where(n_pix_mask > 100)[0]]
            x = x[np.where(n_pix_mask > 100)[0]]
            self.filenames = [self.filenames[i] for i in np.where(n_pix_mask > 100)[0]]

            # idx_shuffle_A = np.random.permutation(x.shape[0])
            idx_shuffle_A = np.arange(0, x.shape[0])  # no shuffling in fact
            caseID_A = caseID[idx_shuffle_A].copy()

            # Number of channels of input A
            A = x[idx_shuffle_A, ..., :n_A_channel_idx].astype(self.dtype)
            C = x[idx_shuffle_A, ..., n_A_channel_idx:-2].astype(self.dtype)
            mask = x[idx_shuffle_A, ..., -1:]
            wp = x[idx_shuffle_A, ..., -2:-1]
            self.n_A_channel_idx = n_A_channel_idx
            self.num_channels_out = C.shape[-1]

            idx_shuffle_B = np.random.permutation(x.shape[0])
            caseID_B = caseID[idx_shuffle_B].copy()

            if self.use_adjacent:
                C, mask, wp = C[..., 0, :], mask[..., 0, :], wp[..., 0, :]
            return A, C, mask, wp, caseID_A, caseID_B
        else:

            if self.num_unsup > 0:
                print('Loading unsupervised data...')
                self.dataset_file_unsup = (self.dataset_file_org.replace('_adjacent', '')).replace('.npy',
                                                                                                   '_unsup_%dsubs_cleaned.npy' % self.num_unsup)
                self.dataset_file_unsup = self.dataset_file_unsup.replace(
                    re.findall('\d+channels', self.dataset_file_unsup)[0], '5channels')
                x = np.load(self.dataset_file_unsup).transpose((2, 0, 1, 3)).astype('float32')
                x[np.isnan(x)] = 0
                if (self.lambda_aux == 0) or self.reconstruct_input:
                    x = x * x[..., n_A_channel_idx:]  # masking inputs with wp
                idx_chosen = np.ones(x.shape[0]).astype(bool)
                if self.use_pseudo_labels:
                    num_unused_unsup = int(np.floor(x.shape[0] / 1))
                    idx_chosen[:num_unused_unsup] = False
                print('Done!')
            else:
                x = np.load(self.dataset_file_org.replace('.npy', '_unsup.npy')).transpose((2, 0, 1, 3)).astype(
                    'float32')
                x[np.isnan(x)] = 0
                if (self.lambda_aux == 0) or self.reconstruct_input:
                    x = x * x[..., n_A_channel_idx:]  # masking inputs with wp
                # Filter visually
                idx_chosen = np.ones(x.shape[0]).astype(bool)
                idx_chosen[1006:1013] = False
                idx_chosen[1227:1233] = False
                idx_chosen[1835:1846] = False
                idx_chosen[2268:2277] = False
                idx_chosen[3182:3188] = False
                idx_chosen[3477:3491] = False
                idx_chosen[3591:3598] = False
                idx_chosen[3610:3616] = False
                idx_chosen[3683:3691] = False
                idx_chosen[3873:3879] = False
                idx_chosen[4255:4265] = False
                idx_chosen[4448:4454] = False
                idx_chosen[4970:4973] = False
                # Filter by size
                idx_chosen[(x[..., 1] > 0).sum(axis=(1, 2)) < 1000] = False  # Remove slices with bad ADC
            A = x[idx_chosen, ..., :n_A_channel_idx].astype(self.dtype)
            wp = x[idx_chosen, ..., n_A_channel_idx:].astype(self.dtype)
            return A, wp


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
        # Pixel2pixel networks
        n_in = 2 if '5channels' in self.dataset_file else 1
        #
        # if self.true_density_generator == 'unet':  # UNet not working yet
        #     self.netG = UNet(num_ds=self.num_downs, add_scale_layer=self.scale_layer)
        #     self.netG.collect_params().initialize(inits[self.initializer], ctx=self.ctx)
        #     # self.netG.collect_params().initialize(ctx=self.ctx)
        # elif self.true_density_generator == 'dmnet':
        #     self.netG = pretrained_net(dir_model=self.dir_model, model=self, num_extracted_encoder=None,
        #                                initializer=self.initializer, num_fpg=self.num_fpg, growth_rate=self.growth_rate,
        #                                num_group_norm=self.ngroups_norm, norm_type=self.norm_type,
        #                                activation=self.activation)
        #     self.netG.collect_params().initialize(init=inits[self.initializer], ctx=self.ctx)
        # if self.resumed_epoch > -1:
        #     self.load_checkpoints(prefix=self.checkpoint_prefix)
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

        # x = nd.random_normal(0.1, 0.02, shape=(1, 2, 256, 256), ctx=mx.cpu())
        # self.netG.initialize(mx.initializer.Xavier(magnitude=2), ctx=mx.cpu())
        # print(self.netG.summary(x))
        # exit()

        self.netG.initialize(inits[self.initializer], ctx=self.ctx, force_reinit=True)
        if self.resumed_it > -1:
            self.load_checkpoints(prefix=self.checkpoint_prefix)
        elif self.use_pretrained:
            self.load_checkpoints(pretrained_dir=filename_pretrained_weights)

        self.use_l_coefs = False
        coefs = []
        for l in ['0', '_aux', '_C', '_consistency', '_D', '_unsup', '_CL', '_embedding_unsup']:
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
        largest_batch_size *= self.unsup_ratio if ((self.lambda_unsup > 0) or (self.lambda_embedding_unsup > 0)) else 1
        if self.show_generator_summary:
            [self.netG.summary(
                nd.random.normal(0, 1, shape=(largest_batch_size, n_in, self.input_size, self.input_size), ctx=ctx)) for
                ctx
                in self.ctx]
        self.D_features = FeatureComparator(in_channels=1, ctx=self.ctx) if self.lambda_D > 0 else None

        self.netGE = extract_encoder(self.netG.net if self.lambda_aux <= 0 else self.netG.shared_net,
                                     self.num_downs) if (self.lambda_embedding_unsup > 0) else None

        # self.D_features.collect_params().initialize(inits['xavier'], ctx=self.ctx, force_reinit=True)

        # Unsupervised features extractor
        # self.D_features_unsup = FeatureComparator(in_channels=1)  # n_in
        # self.D_features_unsup.collect_params().initialize(inits['xavier'], ctx=self.ctx, force_reinit=True)

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
            'cross_entropy': mx.gluon.loss.SoftmaxCrossEntropyLoss,
            'softmax_embedding': SoftmaxEmbeddingLoss,
        }

        self.cross_entropy = loss_fn['cross_entropy'](axis=1)
        self.density_corr = loss_fn['rloss']()
        # self.trueDensity_train = loss_fn[self.l_type](with_DepthAware=self.with_DepthAware)
        self.trueDensity_train = loss_fn[self.l_type]()
        self.trueDensity_val = loss_fn[self.l_type]()
        self.feature_difference = gluon.loss.CosineEmbeddingLoss()
        self.density_embedding_unsup = loss_fn['softmax_embedding']()
        self.density_unsup = loss_fn[self.l_type]()
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

    def compare_embedding_unsup(self):
        self.fake_out_unsup = [nd.squeeze(self.netGE(A_unsup)) for A_unsup in self.A_unsup]
        self.fake_out_unsup_aug = [nd.squeeze(self.netGE(A_rp_unsup)) for A_rp_unsup in self.A_rp_unsup]
        self.unsup_embedding_loss = [
            self.density_embedding_unsup(fake_out_unsup, fake_out_unsup_aug)
            for fake_out_unsup, fake_out_unsup_aug
            in zip(self.fake_out_unsup, self.fake_out_unsup_aug, )]

    def compare_unsup(self):
        """Get unsupervised loss"""
        self.fake_out_unsup = [self.netG(A_unsup) for A_unsup in self.A_unsup]
        self.fake_out_unsup_aug = [nd.flip(self.netG(A_rp_unsup), 3) for A_rp_unsup in self.A_rp_unsup]
        if self.lambda_aux > 0:
            self.fake_out_unsup = [fake_out_unsup[0] for fake_out_unsup in self.fake_out_unsup]
            self.fake_out_unsup_aug = [fake_out_unsup_aug[0] for fake_out_unsup_aug in self.fake_out_unsup_aug]

        self.fake_out_unsup = [nd.where(wp_unsup, fake_out_unsup, wp_unsup - 1) for wp_unsup, fake_out_unsup in
                               zip(self.wp_unsup, self.fake_out_unsup)]

        tmp = self.fake_out_unsup_aug.copy()
        self.shape_seq = list(self.chunks(self.shape_seq, tmp[0].shape[0]))
        self.fake_out_unsup_aug = []
        for (arr, shape_seq) in zip(tmp, self.shape_seq):
            tmp_sub = arr.asnumpy().transpose([0, 2, 3, 1])
            for (j, seq) in enumerate(shape_seq):
                heatmap = HeatmapsOnImage(
                    arr=tmp_sub[j].astype('float32'),
                    shape=tmp_sub[j].shape,
                    min_value=self.density_range[0],
                    max_value=self.density_range[1]
                )
                tmp_sub[j] = seq(heatmaps=heatmap).get_arr()
            self.fake_out_unsup_aug.append(nd.array(tmp_sub.transpose([0, 3, 1, 2]), ctx=arr.context))

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
        if np.array(
                [self.lambda_C, self.lambda_D, self.lambda_consistency, self.lambda_unsup, self.lambda_embedding_unsup,
                 self.lambda0, self.lambda_CL,
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
                        nd.L2Normalization(self.D_features(nd.where(m, C, m - 1))),
                        nd.L2Normalization(self.D_features(nd.where(m, fake_out, m - 1))),
                        nd.ones((C.shape[0]), ctx=C.context)
                    ).mean() for m, C, fake_out in zip(self.m, self.C, self.fake_out)]
                    self.loss_G = [l0 + ((1 / var) * l1 * .1 + nd.log(var)) for l0, l1, var in
                                   zip(self.loss_G, self.loss_features, self.var4)]
                ############################### Classification Loss ###############################
                if self.lambda_CL > 0:
                    qml_pred = [-nd.abs((fake_out - qml) * m) for fake_out, qml, m in
                                zip(self.fake_out, self.qml, self.m)]
                    self.var5 = [nd.square(coef) for coef in self.netG.coef_CL._data]
                    self.loss_CL = [self.cross_entropy(_qml_pred, qml_gt[:, 0], m) for
                                    _qml_pred, qml_gt, m in
                                    zip(qml_pred, self.qml_gt, self.m)]
                    self.loss_G = [l0 + ((1 / var) * l1 + nd.log(var)) for l0, l1, var in
                                   zip(self.loss_G, self.loss_CL, self.var5)]
                ############################### Unsupervised embedding learning ###############################
                if self.lambda_embedding_unsup > 0:
                    self.compare_embedding_unsup()
                    self.var6 = [nd.square(coef) for coef in self.netG.coef_embedding_unsup._data]
                    self.loss_G = [l0 + ((1 / var) * l1 + nd.log(var)) for l0, l1, var in
                                   zip(self.loss_G, self.unsup_embedding_loss, self.var6)]
                ################################### GAN ###################################
                # if self.lambda0 > 0:
                #     fake_concat = [nd.concat(A_rp, fake_out, dim=1) for A_rp, fake_out in zip(self.A_rp, self.fake_out)]
                #     output = [self.netD(_fake_concat) for _fake_concat in fake_concat]
                #     self.wp_rz = [self.resize_wp(wp, _output) for wp, _output in zip(self.wp, output)]
                #     real_label = [nd.ones(op.shape, ctx=_ctx) for op, _ctx in zip(output, self.ctx)]
                #     self.realistic_loss = [self.criterionGAN(_output, _real_label, wp_rz) for _output, _real_label, wp_rz in
                #                            zip(output, real_label, self.wp_rz)]
                # else:
                #     self.realistic_loss = [nd.zeros_like(l) for l in self.loss_true_density_train]
                #
                # self.loss_G = [realistic_loss * self.lambda0 + loss_true_density_train * self.lambda1 for
                #                realistic_loss, loss_true_density_train in
                #                zip(self.realistic_loss, self.loss_true_density_train)]
                ############################### Aux Out ###############################
                # if self.lambda_aux > 0:
                #     aux_lbs = self.A_unsup if self.reconstruct_input else self.wp
                #     self.aux_loss = [self.aux_fn(aux_out, aux_lb, wp_unsup) for aux_out, aux_lb, wp_unsup in
                #                      zip(self.aux_out, aux_lbs, self.wp_unsup)]
                # else:
                #     self.aux_loss = [nd.zeros_like(loss_G) for loss_G in self.loss_G]

                # Final generator loss

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

        metric_list = metrics.update_mxboard_metric_multi_maps(
            sw, data=val_data, global_step=epoch,
            metric_names=['r_whole', 'l1_whole', 'ssim_whole', ],
            prefix='validation_',
            num_input_channels=self.n_A_channel_idx, c_thr=self.C_thr,
            density_range=self.density_range, root=self.root)  # 'r', 'l1', 'ssim', 'nmi',
        # if hasattr(self, 'current_margin'):
        # if self.current_it >= (self.total_iter * .5):
        #     if metric_list['l1_whole_EPI'] >= .07:
        #         exit()
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
            for i in range(self.n_A_channel_idx,
                           self.n_A_channel_idx + self.num_channels_out * 2):  # prediction and label
                _val_data[i] = self.normalize_01(_val_data[i], [-1, 1]) * _val_data[-1]
        #######################################
        """ SAVE FIRST IMAGE TO FOLDER & UPDATE BEST METRICS """
        to_save_montage = self.update_best_metrics(metric_list)
        print(self.best_metrics)
        if to_save_montage:
            np.save(f'{self.result_folder_figure_val}_data_{self.current_it}.npy', _val_data)
            self.save_montage_im(_val_data)
        #######################################
        """ DROP LAST CHANNEL (WP) IN _val_data BECAUSE IT IS NO LONGER NECESSARY """
        _val_data = _val_data[:-1]
        #######################################
        """ SAVE SECOND IMAGE TO TENSORBOARD """
        # if to_save_montage:  # same condition with when saving the first image
        #     img_concat = self.create_concat_image(_val_data)
        #     sw.add_image('validation_results', img_concat, global_step=epoch)
        #######################################
        return metric_list

    @staticmethod
    def _debug_show_val_data(_val_data):
        ca, t = None, None
        fig, ax = plt.subplots(1, 1)
        for i in range(len(_val_data)):
            _img_ = montage(_val_data[i].squeeze())
            if i == 0:
                ca = ax.imshow(_img_, cmap='gray', vmin=0, vmax=1)
                t = ax.text(-10, -10, [i, _img_.max()])
            else:
                ca.set_data(_img_)
                t.set_text([i, _img_.max()])
            plt.pause(2)

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
            pred = nd.where(self.wp_val, pred, self.wp_val - 1)  # wp-masked

            input_list.append(self.A_rp_val.asnumpy())
            pred_list.append(pred.asnumpy())
            wp_list.append(self.wp_val.asnumpy())

        return np.concatenate(input_list), \
               np.concatenate([*pred_list]), \
               np.concatenate([*wp_list]),  # duplicate to simplify the modifications

    def validate(self):
        """Perform validation"""
        l = []
        input_list, pred_list, mask_list, gt_list, wp_list = [], [], [], [], []
        aux_list = [] if self.lambda_aux > 0 else None

        for i, batch in enumerate(self.val_iter):
            _, C, m, wp, A_rp = batch
            # Inputs to GPUs (or CPUs)
            self.set_inputs(A_rp_val=A_rp, C_val=C, m_val=m, wp_val=wp)
            pred = [self.netG(A_rp_val) for A_rp_val in self.A_rp_val]
            # Split segmentation and regression outputs if multitask learning is used
            if self.lambda_aux > 0:
                aux_out = [_pred[1] for _pred in pred]
                aux_out = nd.concatenate(aux_out)
                pred = [_pred[0] for _pred in pred]
            pred = nd.concatenate(pred)

            # Crop image to smaller size if the generator is UNet
            if self.C_val[0][0].shape != pred[0].shape:
                cc = self.netG.center_crop
                self.C_val, self.m_val, self.A_rp_val, self.wp_val = \
                    [cc(nd.concatenate(list(x)), pred) for x in [self.C_val,
                                                                 self.m_val,
                                                                 self.A_rp_val,
                                                                 self.wp_val]]

            # merge data across all used GPUs
            self.C_val, self.m_val, self.A_rp_val, self.wp_val = [
                nd.concatenate(list(x)) for x in [self.C_val,
                                                  self.m_val,
                                                  self.A_rp_val,
                                                  self.wp_val]
            ]
            pred = nd.where(self.wp_val, pred, self.wp_val - 1)  # wp-masked

            input_list.append(self.A_rp_val.asnumpy())
            pred_list.append(pred.asnumpy())
            gt_list.append(self.C_val.asnumpy())
            mask_list.append(self.m_val.asnumpy())
            wp_list.append(self.wp_val.asnumpy())
            aux_list.append(aux_out.asnumpy()) if self.lambda_aux > 0 else None

            l.append(self.trueDensity_val(self.C_val, pred, self.m_val).asnumpy())

        if (self.lambda_aux > 0) and (not self.reconstruct_input):
            val_dice = metrics.dice_wp(np.concatenate(aux_list), np.concatenate(wp_list)).mean()
            print("Val Dice: %.3f" % val_dice)
            if val_dice < 0.01:
                exit()

        self.running_loss_true_density_val = np.concatenate([*l]).mean()

        # c = np.concatenate([*gt_list])
        # print([c[i].sum() for i in range(c.shape[1])])
        # exit()

        return np.concatenate(input_list), \
               np.concatenate([*pred_list]), \
               np.concatenate([*gt_list]), \
               np.concatenate([*mask_list]), \
               np.concatenate([*wp_list]),

    def save_montage_im(self, IM, prefix=''):
        for k in range(self.num_channels_out):
            im = [IM[0], IM[1], IM[2 + k], IM[2 + self.num_channels_out + k],
                  IM[2 + self.num_channels_out * 2], IM[-1]]
            _im = np.squeeze(im)[:-2]
            _im_contour = np.tile(np.squeeze(im)[-2], (len(im) - 2, 1, 1, 1))
            _im_wp = np.tile(np.squeeze(im)[-1], (len(im) - 2, 1, 1, 1))
            _im_wp[_im_wp == 1] = 2
            for i in range(self.n_A_channel_idx):
                _im_wp[i] = _im[i]

            _im_wp = montage(np.concatenate(_im_wp, axis=2))
            _im = montage(np.concatenate(_im, axis=2))
            _im_wp = masked_array(_im_wp, _im_wp == 2)
            plt.imshow(_im, cmap='jet', vmin=0, vmax=.7)  # , interpolation='nearest'
            plt.imshow(_im_wp, cmap='gray', vmin=0, vmax=1)  # , interpolation='nearest'
            plt.contour((montage(np.concatenate(_im_contour, axis=2))).astype(int), linewidths=.14, colors='white')
            self.save_fig(folder=self.result_folder_figure_test) if self.test_mode else self.save_fig(
                folder=self.result_folder_figure_val, suffix=str(k))

    def save_fig(self, folder, prefix='', suffix='', dpi=500):
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        plt.savefig(
            '%s/%sep%04d_%04d_%s.png' % (folder, prefix, self.current_it, self.current_epoch, suffix),
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
        to_save_montage = False

        # if self.current_it == 9:
        #     to_save_montage = True

        # if self.current_it > 2000:
        #     to_save_montage = True
        if divmod(self.current_it + 1, 500)[1] == 0:
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

    def fade_signal(self, m, stsa_rat=.9):
        """Remove training signal with respect to the current training iteration"""
        num_signal = int(
            min(np.floor(self.current_it / (self.total_iter * stsa_rat / m.shape[0])) + 1, m.shape[0]))
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
