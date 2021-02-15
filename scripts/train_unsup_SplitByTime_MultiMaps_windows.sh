#!/usr/bin/env bash
export MXNET_ENFORCE_DETERMINISM=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export CUDA_VISIBLE_DEVICES=${1}

suffix=switch_to_ENUC_v4b
batch_size=8

#lr_scheduler=cosine
max_lr=1e-3
min_lr=7e-4
warmup_begin_lr=1e-5
warmup_steps=200
cooldown_length=200

lr_scheduler=one_cycle
max_lr=1e-3
min_lr=1e-4 # 1e-5
#inc_fraction=.5
cycle_length=400
cooldown_length=600
finish_lr=1e-7
total_iter=$((warmup_steps + cycle_length + cooldown_length + 108))

lr_scheduler=cycle
max_lr=1e-3 #1e-3
min_lr=1e-4 #1e-5
total_iter=2008 #2508  # 2508 # 1808 #6508 #908 # 2008

inc_fraction=.9
cycle_length_decay=.9  # .93
cycle_magnitude_decay=.95

base_lr=1e-4
lr_factor=.1
lr_step=9999

optimizer=adam
gid=0,1
stop_decay_iter=10000
runtime=0
margin_decay_rate=0
margin=0
num_workers=0
init=normal

checkpoint_iter=$((total_iter - 9))
lambda0=0
train_threshold=1

iter_activate_unsup=0

growth_rate=4
loss_type=l2
folder=ImprovedSemi_PostMICCAI_5
runtime=2d

n_channels_raw_inp=8
train_perc=0.60
dataset_file=mri_density_${n_channels_raw_inp}channels
caseID_file=caseID_by_time_${train_perc}

network=drnn  # drnn unetpp unet # deeplabv3plus
cycle_length=100
lambda_D=0
margin=0
lambda_C=0
growth_rate=4
batch_size=8
root=1
margin=0
lambda_CL=0
lambda_D=0
stsa_rat_int=9
for network in ${2}
do
    for density_type in EESL # ENSL
        do
        for lambda_embedding_unsup in 0 #1
        do
            for lambda_C in 0 #1
            do
                for num_unsup in 200  # 200
#                for num_unsup in 0 200 #100 200
                do
                    for seed in {100..150} # 63
                    # 128 135 140 for NUC
#                    for seed in 234 #213 219 227 247 114 234
                    do
                        stsa_rat=`echo "print('%.1f' % (${stsa_rat_int}/10))" | python`
                        lambda_unsup=1
                        prefix=${network}
                        if [[ ${num_unsup} == 0 ]]
                        then
                            lambda_unsup=0
                        fi
                        if [[ ${network} == drnn ]]
                        then
                            prefix=${prefix}GR${growth_rate}
                        fi
                        export SEED=${seed}
                        if [[ ${lambda_unsup} == 0 ]] && [[ ${lambda_embedding_unsup} == 0 ]]
                        then
                            unsup_ratio=0
                        else
                            unsup_ratio=$(( num_unsup / 100 ))
#                            unsup_ratio=1
                        fi
                        for num_expand_level in 1
                        do
                            cmd=`echo train_PixUDA_MultiMaps.py -rid ${prefix}_lCL${lambda_CL}_${density_type}_lUS${lambda_unsup}_lC${lambda_C}_${loss_type}_nc${n_channels_raw_inp}_stsa${stsa_rat}_r${root}_sd${seed}_${init}_v${runtime}_${suffix} --experiment_name ${folder}/ -gid $gid --lambda1 1 --pool_size 20 --batch_size ${batch_size} --l_type ${loss_type} --true_density_generator ${network} --num_downs 4 --base_lr ${base_lr} -ep 99999 --dataset_file ${dataset_file} --scale_layer tanh --num_fpg -1 --growth_rate ${growth_rate} --beta1 0.9 --validation_start 0 --unsup_ratio ${unsup_ratio} --num_workers ${num_workers} --lambda0 ${lambda0} --lambda_aux 0 --lambda_D ${lambda_D} --lambda_unsup ${lambda_unsup} --initial_margin ${margin} --margin_decay_rate ${margin_decay_rate} --initializer ${init} --caseID_file ${caseID_file} --min_lr ${min_lr} --max_lr ${max_lr} --cycle_length ${cycle_length} --lr_scheduler ${lr_scheduler} --val_interval 10 --log_interval 10 --total_iter ${total_iter} --final_drop_iter 11000 --density_type ${density_type} --optimizer ${optimizer} --stop_decay_iter ${stop_decay_iter} --cycle_length_decay ${cycle_length_decay} --cycle_magnitude_decay ${cycle_magnitude_decay} --inc_fraction ${inc_fraction} --finish_lr ${finish_lr} --checkpoint_iter ${checkpoint_iter} --train_threshold ${train_threshold} --cooldown_length ${cooldown_length} --warmup_begin_lr ${warmup_begin_lr} --warmup_steps ${warmup_steps} --iter_activate_unsup ${iter_activate_unsup} --num_unsup ${num_unsup} --lr_factor ${lr_factor} --lr_step ${lr_step} --fold_idx 0 --lambda_C ${lambda_C} --root ${root} --lambda_CL ${lambda_CL} --stsa_rat ${stsa_rat} --lambda_embedding_unsup ${lambda_embedding_unsup}` #--with_DepthAware  --use_tsa --num_expand_level ${num_expand_level} --use_pretrained --use_pseudo_labels --use_pseudo_labels --gen_unsup_pred --resumed_it 2499
                            echo ${cmd}
                            python ${cmd}
                        done
                    done
                done
            done
        done
    done
done


