import os

try:
    SEED = int(os.getenv('SEED'))
except:
    SEED = 0

os.urandom(SEED)
import random
import numpy as np

os.environ['MXNET_ENFORCE_DETERMINISM'] = '1'
# os.environ['MXNET_BACKWARD_DO_MIRROR'] = '0'  # trade-off memory usage -  speed
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

# os.environ['MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION'] = '1'

from gluoncv.utils.random import seed

seed(SEED)

import mxnet as mx

[mx.random.seed(SEED, ctx=mx.gpu(i)) for i in range(mx.context.num_gpus())]
mx.random.seed(SEED, ctx=mx.cpu())

# seed(100)
from mxnet import ndarray as nd
from argparse import ArgumentParser
from PixUDA_MultiMaps import PixUDA
from datetime import datetime
import time
from mxboard import SummaryWriter
from utils import metrics
# import logging
import warnings

warnings.filterwarnings("ignore")
import pickle


def parse_args():
    """Get commandline parameters"""
    parser = ArgumentParser('RadPath Arguments')
    parser.add_argument('--dataset_file', type=str,
                        default='mri_density_5channels')
    parser.add_argument('--caseID_file', type=str, default=r'caseID_10_splits')
    parser.add_argument('-expn', '--experiment_name', type=str, default='pix2pix_uda_v1')
    parser.add_argument('-rid', '--run_id', type=str, default='999')
    parser.add_argument('-gid', '--gpu_id', type=str, default='0')
    parser.add_argument('-ngpu', '--num_gpus', type=int, default=0)
    parser.add_argument('-ep', '--epochs', type=int, default=400)
    parser.add_argument('--total_iter', type=int, default=1000)
    parser.add_argument('--checkpoint_iter', type=int, default=1999)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('-re', '--resumed_epoch', type=int,
                        default=-2)  # <-1: initialize the network from scratch, -1: load best epoch, otherwise for any specific epochs
    parser.add_argument('-ri', '--resumed_it', type=int,
                        default=-2)  # <-1: initialize the network from scratch, -1: load best epoch, otherwise for any specific epochs
    parser.add_argument('-tthr', '--train_threshold', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lambda1', type=float, default=1)
    parser.add_argument('--lambda0', type=float, default=0)
    parser.add_argument('--lambda_CL', type=float, default=0)
    parser.add_argument('--lambda_C', type=float, default=0, help='> 0 for correlation loss')
    parser.add_argument('--lambda_consistency', type=float, default=0, help='> 0 for consistency loss')
    parser.add_argument('--lambda_D', type=float, default=0, help='> 0 for feature camparison')
    parser.add_argument('--lambda_aux', type=float, default=0, help='> 0 for prostate segmentation')
    parser.add_argument('--pool_size', type=int, default=0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_downs', type=int, default=4)
    parser.add_argument("--input_size", type=int, default=256,
                        help='image size, would be automatically set to 252 if use U-Net')
    parser.add_argument('--l_type', type=str, default='l2')
    parser.add_argument('--scale_layer', type=str, default='tanh')
    parser.add_argument("--dir_model", type=str, default=r"F:\Minh\projects\NIH\prostateSegmentation\outputs\run10")
    parser.add_argument("--true_density_generator", type=str, default="dmnet")
    parser.add_argument('--use_gan', action='store_true')
    parser.add_argument('--freeze_pretrained_net', action='store_true')
    parser.add_argument('--no_augmentation', action='store_true')
    parser.add_argument('--validation_only', action='store_true')
    parser.add_argument('--with_DepthAware', action='store_true')
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--validate_only', action='store_true')
    parser.add_argument('--not_augment_values', action='store_true')
    parser.add_argument("--norm_0mean", action='store_true')
    parser.add_argument("--initializer", type=str, default='none')
    parser.add_argument("--dtype", type=str, default='float32')
    parser.add_argument("--num_fpg", type=int, default=8)
    parser.add_argument("--growth_rate", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--validation_start", type=int, default=0)
    parser.add_argument('--monitor_training_outputs', action='store_true')
    # UDA parameters
    parser.add_argument('--unsup_ratio', type=int, default=2)
    parser.add_argument('--lambda_unsup', type=float, default=0, help='> 0 for unsupervised learning')
    parser.add_argument('--lambda_embedding_unsup', type=float, default=0,
                        help='> 0 for unsupervised embedding learning')
    # LR scheduler parameters
    parser.add_argument('--lr_scheduler', type=str, default='factor')
    parser.add_argument('--warmup_mode', type=str, default='linear')
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--max_lr', type=float, default=1e-2)
    parser.add_argument('--lr_step', type=float, default=1, help='For factor learning rate scheduler')
    parser.add_argument('--lr_steps', type=str, default='1', help='For multifactor learning rate scheduler')
    parser.add_argument('--lr_factor', type=float, default=1)
    parser.add_argument('--finish_lr', type=float, default=None)
    parser.add_argument('--cycle_length', type=int, default=1000)
    parser.add_argument('--stop_decay_iter', type=int, default=None)
    parser.add_argument('--final_drop_iter', type=int, default=None)
    parser.add_argument('--cooldown_length', type=int, default=5000)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--warmup_begin_lr', type=float, default=1e-5)
    parser.add_argument('--inc_fraction', type=float, default=0.9)
    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--cycle_length_decay', type=float, default=.95)
    parser.add_argument('--cycle_magnitude_decay', type=float, default=.98)
    parser.add_argument('--show_generator_summary', action='store_true')
    parser.add_argument('--reconstruct_input', action='store_true')
    parser.add_argument('--discriminator_update_interval', type=int, default=1)
    parser.add_argument('--monitor_unsup_outputs', action='store_true')
    parser.add_argument('--checkpoint_prefix', type=str, default='')
    parser.add_argument('--initial_margin', type=float, default=0., help='margin in loss function')
    parser.add_argument('--fold_idx', type=int, default=0, help='index of training fold')
    parser.add_argument('--use_adjacent', action='store_true')
    parser.add_argument('--norm_type', type=str, default='batch', help='batch | group | instance')
    parser.add_argument('--ngroups_norm', type=int, default=4)
    parser.add_argument('--density_type', type=str, default='EPI')
    parser.add_argument('--margin_decay_rate', type=float, default=1)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--weighting_density', action='store_true')
    parser.add_argument('--iter_activate_unsup', type=int, default=0)
    parser.add_argument('--num_unsup', type=int, default=0)
    parser.add_argument('--backbone_name', type=str, default='vgg')
    parser.add_argument('--stsa_rat', type=float, default=0.9)
    parser.add_argument('--root', type=int, default=1,
                        help='take root of the ground truth density to achieve a more normal distribution'
                             'worked for the right-skewed distribution')
    parser.add_argument("--base_channel_unet", type=int, default=16,
                        help='number of channels in the first layer of UNet')
    parser.add_argument("--base_channel_drnn", type=int, default=8,
                        help='number of channels in the first layer of DRNN')
    parser.add_argument('--gen_unsup_pred', action='store_true')
    parser.add_argument('--use_pseudo_labels', action='store_true')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('-dot', '--density_outlier_thr', type=float, default=1.0)
    parser.add_argument('--num_expand_level', type=int, default=1,
                        help='number of levels during dataset expansion, must be greate than 0')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    args.dataset_file_org = '' + args.dataset_file
    if args.lambda_aux > 0:
        if not args.reconstruct_input:
            args.dataset_file = args.dataset_file + '_non_masked'
        args.true_density_generator = 'dmnet'
    args.fold_idx = 0 if args.caseID_file == 'caseID_by_time' else args.fold_idx

    args.to_11 = True if args.scale_layer in ['tanh', 'softsign'] else False

    model = PixUDA(args)
    sw = SummaryWriter(logdir='%s' % model.result_folder_logs, flush_secs=5)
    # print(sw.get_logdir())
    first_iter = True  # A trick to use running losses
    best_score = -9999
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    # logging.basicConfig(level=print)

    if args.gen_unsup_pred:
        unsup_pred = model._gen_unsup_pred()
        model.current_epoch = args.resumed_epoch
        model.result_folder_checkpoint_iter = '%s/iter_%04d' % (
            model.result_folder_checkpoint, model.resumed_it)
        model.result_folder_inference = model.result_folder_checkpoint_iter.replace('checkpoints', 'inference')
        if not os.path.exists(model.result_folder_inference):
            os.makedirs(model.result_folder_inference)
        with open('%s/pred_unsup' % model.result_folder_inference, 'wb') as fp:
            pickle.dump(unsup_pred, fp)
        print(model.result_folder_inference)
        exit()

    if args.validate_only:
        test_data = model.validate()
        model.current_epoch = args.resumed_epoch
        metric_list = metrics.update_mxboard_metric_v1(sw, data=test_data, global_step=None,
                                                       metric_names=['r', 'r_whole', 'l1_whole'],
                                                       prefix='validation_', num_input_channels=model.n_A_channel_idx,
                                                       not_write_to_mxboard=True)
        print(metric_list)
        print('Average metrics: r: %.3f, l1: %.3f' % (metric_list['r'].mean(), metric_list['l1_whole'].mean()))
        model.generate_test_figures(test_data)
        model.result_folder_checkpoint_iter = '%s/iter_%04d' % (
            model.result_folder_checkpoint, model.resumed_it)
        model.result_folder_inference = model.result_folder_checkpoint_iter.replace('checkpoints', 'inference')
        if not os.path.exists(model.result_folder_inference):
            os.makedirs(model.result_folder_inference)
        with open('%s/test_data' % model.result_folder_inference, 'wb') as fp:
            pickle.dump(test_data, fp)
        model.generate_test_figures(test_data)
        exit()

    tic = time.time()
    btic = time.time()
    count = 0
    # model.create_net()
    model.train_iter._current_it = 0
    # Calculate the number of images increasing every level
    model.data_inc_unit = int(model.train_iter._dataset.__len__() / model.num_expand_level)

    for epoch in range(args.epochs):
        model.current_epoch = epoch
        model.expand_dataset()

        for i, batch in enumerate(model.train_iter):
            model.current_it = model.trainerG.optimizer.num_update
            model.train_iter._current_it = model.current_it

            # if epoch >= 3:
            #     exit()
            # continue

            if (model.lambda_unsup > 0) or (model.lambda_aux > 0) or (model.lambda_embedding_unsup > 0):
                (_, C, m, wp, A_rp, margin), (
                    A_unsup, A_rp_unsup, wp_unsup, margin_unsup, model.shape_seq) = batch
                model.set_inputs(wp_unsup=wp_unsup, A_unsup=A_unsup, A_rp_unsup=A_rp_unsup, _margin_unsup=margin_unsup)
            else:
                _, C, m, wp, A_rp, margin = batch
                # Reduce weights of the loss on the pseudo-data
                # for _i in range(m.shape[0]):
                #     m[_i] = m[_i] / 5 if ((m[_i] - wp[_i]).sum() == 0) else m[_i]

            m = model.fade_signal(m, stsa_rat=args.stsa_rat) if (args.stsa_rat != 0) else m

            model.set_inputs(A_rp=A_rp, C=C, m=m, wp=wp, _margin=model.decay_loss_margin(margin))

            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            if (model.lambda0 > 0) and ((i + 1) % model.discriminator_update_interval == 0):
                model.optimize_D()
            else:
                if (epoch + i) == 0:
                    model.err_DB = nd.zeros(shape=(1,))

            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            # count += 1
            # with open('dummy/dummy1/m%d' % count, 'wb') as fp:
            #     pickle.dump(val_data, fp)
            # continue

            model.optimize_G()

            # Compute running loss
            model.update_running_loss(
                first_iter=first_iter)  # running_loss attributes will be created in the first iter
            first_iter = False

            # Print log infomation every ten batches
            if (model.current_it + 1) % args.log_interval == 0:
                name, acc = model.metric.get()
                print('speed: {} samples/s'.format(args.batch_size / ((time.time() - btic) / args.log_interval)))
                print('discriminator loss = %.4f, generator loss = %.5f, binary training acc = %f at iter %d epoch %d'
                      '[current_lr=%.8f, it=%d]'
                      % (nd.mean(nd.concatenate(list(model.err_DB))).asscalar(),
                         nd.mean(nd.concatenate(list(model.loss_G))).asscalar(), acc, model.current_it, epoch,
                         model.trainerG.learning_rate, model.trainerG.optimizer.num_update))
            btic = time.time()
            sw.add_scalar('learning_rate', model.trainerG.learning_rate,
                          global_step=model.trainerG.optimizer.num_update)
            # Hybridize networks to speed-up computation
            if (i + epoch) == 0:
                model.hybridize_networks()

            if ((model.current_it + 1) % model.val_interval == 0) & (model.current_it >= args.validation_start):
                val_data = model.validate()
                # Visualize generated images
                # model.visualize_epoch_outputs()
                print('          [Validation] loss_to_ground_truth: %.4f' % model.running_loss_true_density_val)
                # Update mxboard
                metric_list = model.update_mxboard(sw=sw, epoch=model.current_it, val_data=val_data)
                score = -metric_list['l1_whole'].mean()
                print('Current score (L1Loss): %.4f' % -score)
                # Save models after each epoch
                if score > best_score:
                    best_score = score
                    # model.save_checkpoints()
                print('time: %4f' % (time.time() - tic))
                tic = time.time()

            model.update_running_loss(num_batch=model.current_it + 1)

            if model.current_it == args.checkpoint_iter:
                model.save_checkpoints()
                model.result_folder_checkpoint_iter = '%s/iter_%04d' % (
                    model.result_folder_checkpoint, model.current_it)
                model.result_folder_inference = model.result_folder_checkpoint_iter.replace('checkpoints', 'inference')
                if not os.path.exists(model.result_folder_inference):
                    os.makedirs(model.result_folder_inference)
                with open('%s/test_data' % model.result_folder_inference, 'wb') as fp:
                    pickle.dump(val_data, fp)
                model.generate_test_figures(val_data)
            if model.current_it >= args.total_iter:
                exit()
        name, acc = model.metric.get()
        model.metric.reset()
        # print('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
