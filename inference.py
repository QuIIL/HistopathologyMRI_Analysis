import os

try:
    SEED = int(os.getenv('SEED'))
except:
    SEED = 0

os.urandom(SEED)
os.environ['MXNET_ENFORCE_DETERMINISM'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

from gluoncv.utils.random import seed

seed(SEED)

import mxnet as mx

[mx.random.seed(SEED, ctx=mx.gpu(i)) for i in range(mx.context.num_gpus())]
mx.random.seed(SEED, ctx=mx.cpu())

from argparse import ArgumentParser
from inference_model import PixUDA
from datetime import datetime
from mxboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore")
import pickle
import glob


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
    parser.add_argument('--compare_embedding_unsup', action='store_true')
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
    parser.add_argument('--root', type=int, default=1,
                        help='take root of the ground truth density to achieve a more normal distribution'
                             'worked for the right-skewed distribution')
    parser.add_argument("--base_channel_unet", type=int, default=16,
                        help='number of channels in the first layer of UNet')
    parser.add_argument("--base_channel_drnn", type=int, default=8,
                        help='number of channels in the first layer of DRNN')
    parser.add_argument('--use_tsa', action='store_true')
    parser.add_argument('--gen_unsup_pred', action='store_true')
    parser.add_argument('--use_pseudo_labels', action='store_true')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--num_channels_out', type=int, default=4)
    parser.add_argument('-dot', '--density_outlier_thr', type=float, default=1.0)
    parser.add_argument('--num_expand_level', type=int, default=1,
                        help='number of levels during dataset expansion, must be greate than 0')
    parser.add_argument('--mr_input_folder', type=str, default='inputs_unsup200')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.use_pretrained = True
    args.gen_unsup_pred = True

    args.dataset_file_org = '' + args.dataset_file
    if args.lambda_aux > 0:
        if not args.reconstruct_input:
            args.dataset_file = args.dataset_file + '_non_masked'
        args.true_density_generator = 'dmnet'
    args.fold_idx = 0 if args.caseID_file == 'caseID_by_time' else args.fold_idx

    args.to_11 = True if args.scale_layer in ['tanh', 'softsign'] else False

    mr_files = glob.glob('unlabelled_data_preparation/MRI_Numpy/%s/*.npy' % args.mr_input_folder)
    for mr_file in mr_files:
        print(mr_file)
        mr_file = mr_file.replace('\\', '/')
        patient_id = mr_file.split('/')[-1]
        args.mr_file = mr_file
        model = PixUDA(args)
        sw = SummaryWriter(logdir='%s' % model.result_folder_logs, flush_secs=5)
        first_iter = True  # A trick to use running losses
        best_score = -9999
        stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

        unsup_pred = model._gen_unsup_pred()

        model.current_epoch = args.resumed_epoch
        model.result_folder_checkpoint_iter = '%s/iter_%04d' % (
            model.result_folder_checkpoint, model.resumed_it)
        model.result_folder_inference_matrix = model.result_folder_checkpoint_iter.replace('checkpoints',
                                                                                           'inference/matrix')
        for folder in [model.result_folder_inference_matrix, ]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        with open('%s/%s' % (model.result_folder_inference_matrix, patient_id), 'wb') as fp:
            pickle.dump(unsup_pred, fp)
        exit()
