# A non-invasive and quantitative histopathology analysis of prostate cancer in multiparametric magnetic resonance imaging
This repository contains the codes for 
- Extracting data and ROIs from the raw dataset
- Training a semi-supervised deep learning models to predict different types of tissue density from multi-parametric prostate MRI
- Visually analyzing and performing statistical tests on the predicted density maps

## Dependencies 
MXNet is used for training deep learning models. Tensorflow is also used with tensorboard to visualize the training process. However, as there are potential conflicts between the two frameworks, it is recommended to create two separate environments for deep learning / image analysis and training visualization.
### Deep Learning and Image Analysis
```
gluoncv==0.9.2
graphviz==0.8.4
imageio==2.9.0
imgaug==0.4.0
ipython==7.16.1
ipywidgets==7.5.1
jupytor==1.0.0
matplotlib==3.3.2
mxboard==0.1.0
mxnet-cu102==1.7.0.post1
numpy==1.19.5
opencv-python==4.5.1.48
pandas==1.1.2
Pillow==8.1.0
scikit-image==0.17.2
scikit-learn==0.23.2
scikit-optimize==0.8.1
scikit-posthocs=0.6.5
scipy==1.5.2
seaborn==0.11.0
```
### Tensorboard
```
tensorboard==2.1.0
tensorflow==1.14.0
```

## Usage Guideline

- `scripts/train.sh` will call `train_PixUDA_MultiMaps.py` to train a semi-supervised deep learning model using `mri_density_8channels_EESL.npy` and `mri_density_5channels_unsup_200subs_cleaned.npy` in `inputs` folder (available offline - backup disk and server 63@SejongUniversity).
- `scripts/predict.sh` will call `inference.py` to predict density maps for data in `unlabelled_data_preparation/MRI_Numpy` folder (available offline - backup disk and server 63@SejongUniversity). Predicted outputs will be stored in `results/run_id/experiment_name` in Numpy format.
- `labelled_data_preparation` contains the codes for extracting MRI images/density maps/prostate mask/ROIs from raw data
- `unlabelled_data_preparation` contains the codes for extracting unlabelled MRI/prostate mask/Biopsy targets from matlab/VOI files. Additionally, the codes with prefix 's03' in this folder are used to extract the averaged density at multiple sizes of ROIs and store them in excel files. Merging those files will result in `RadPath_AveragedDensity_EESL.xlsx`.
- `analyze_report_intensity_density_V3.ipynb` plots and analyzes `RadPath_AveragedDensity_EESL.xlsx`. `utils_analysis.py` contains the codes for statistical analysis.

## Citation

If any part of this code is used, please give appropriate citation to our paper.

## Authors

* [Nguyen Nhat Minh To](https://github.com/minhto2802)

## Acknowledgement

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details
