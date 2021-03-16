#!/usr/bin/env bash
export MXNET_ENFORCE_DETERMINISM=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export CUDA_VISIBLE_DEVICES=0,1

# This block of parameters are not be used by the python program, but still are important for the code to run
density_type=EPI
warmup_begin_lr=1e-5
warmup_steps=200
cooldown_length=600
finish_lr=1e-7
lr_scheduler=cycle
max_lr=1e-3
min_lr=1e-4
total_iter=2508
inc_fraction=.9
cycle_length_decay=.9
cycle_magnitude_decay=.95
cycle_length=100
base_lr=1e-4
lr_factor=.1
lr_step=9999
optimizer=adam
stop_decay_iter=10000
margin_decay_rate=0
margin=0
init=normal
checkpoint_iter=$((total_iter - 9))
train_threshold=1
iter_activate_unsup=0
loss_type=l2
train_perc=0.60

gid=0
n_channels_raw_inp=5
folder=ImprovedSemi_PostMICCAI_5
num_workers=0
lambda0=0
lambda_D=0
margin=0
lambda_C=0
growth_rate=4
batch_size=8
root=1
num_unsup=200
network=drnn
lambda_unsup=1
if [[ ${num_unsup} == 0 ]]
then
    lambda_unsup=0
fi
if [[ ${lambda_unsup} == 0 ]]
then
    unsup_ratio=0
else
    unsup_ratio=$(( num_unsup / 100 ))
fi

for mr_input_folder in inputs_sextant #inputs_unsup200 inputs_extra
do
  echo ${mr_input_folder}
  for rid in  drnnGR4_lCL0_EESL_lUS1_lC0_l2_nc8_stsa0.9_r1_sd63_normal_v2b_switch_to_ENUC_v4_TSA0.90
  do
      cmd=`echo inference.py -rid ${rid} --experiment_name ${folder}/ -gid $gid --lambda1 1 --pool_size 20 --batch_size ${batch_size} --l_type ${loss_type} --true_density_generator ${network} --num_downs 4 --base_lr ${base_lr} -ep 99999 --dataset_file mri_density_${n_channels_raw_inp}channels --scale_layer tanh --num_fpg -1 --growth_rate ${growth_rate} --beta1 0.9 --validation_start 0 --unsup_ratio ${unsup_ratio} --num_workers ${num_workers} --lambda0 ${lambda0} --lambda_aux 0 --lambda_D ${lambda_D} --lambda_unsup ${lambda_unsup} --initial_margin ${margin} --margin_decay_rate ${margin_decay_rate} --initializer ${init} --caseID_file caseID_by_time_${train_perc} --min_lr ${min_lr} --max_lr ${max_lr} --cycle_length ${cycle_length} --lr_scheduler ${lr_scheduler} --val_interval 10 --log_interval 10 --total_iter ${total_iter} --final_drop_iter 11000 --density_type ${density_type} --optimizer ${optimizer} --stop_decay_iter ${stop_decay_iter} --cycle_length_decay ${cycle_length_decay} --cycle_magnitude_decay ${cycle_magnitude_decay} --inc_fraction ${inc_fraction} --finish_lr ${finish_lr} --checkpoint_iter ${checkpoint_iter} --train_threshold ${train_threshold} --cooldown_length ${cooldown_length} --warmup_begin_lr ${warmup_begin_lr} --warmup_steps ${warmup_steps} --iter_activate_unsup ${iter_activate_unsup} --num_unsup ${num_unsup} --lr_factor ${lr_factor} --lr_step ${lr_step} --fold_idx 0 --lambda_C ${lambda_C} --root ${root} --use_tsa --use_pretrained --gen_unsup_pred --resumed_it 1999 --mr_input_folder ${mr_input_folder}`
      echo ${cmd}
      python ${cmd}
  done
done
