#!/usr/bin/env bash
#cd /mnt/minh/projects/RadPath_Pix2Pix/results/ISBI/
#cd /mnt/minh/projects/RadPath_Pix2Pix/results/BaseLineDRNN/
#cd /mnt/minh/projects/RadPath_Pix2Pix/results/ISBI_ForFigures/
#cd /mnt/minh/projects/RadPath_Pix2Pix/results/UnSupDRNN_v0_Lcos_Margin_SuperSplit1/
#cd /mnt/minh/projects/RadPath_Pix2Pix/results/UnSupDRNN_v0_SuperSplit1/
#cd /mnt/minh/projects/RadPath_Pix2Pix/results/UnSupDRNN_v0_SuperSplit1/
# ByTime_architectures ByTime_Calibrate_Seeds_2
cd /media/data1/minh/projects/RadPath_RemoteControl/results/ImprovedSemi_PostMICCAI_7a
conda activate tensorflow36
tensorboard --logdir=. --port=6008 --host=127.0.0.1

# minh@211.180.114.161:
