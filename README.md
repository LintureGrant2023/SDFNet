# Exploring Spatial Frequency Information for Enhanced Video Prediction Quality
![GitHub stars](https://img.shields.io/github/stars/LintureGrant2023/SDFNet)  ![GitHub forks](https://img.shields.io/github/forks/LintureGrant2023/SDFNet?color=green) 

This repository contains the implementation code for the paper:

__Exploring Spatial Frequency Information for Enhanced Video Prediction Quality__

## Introduction
The architecture of SDFNet:
![SDFNet](/figures/SDFNet.png "The overall framework of SDFNet")



## Overview

* `API/` contains dataloaders and metrics.
* `cls/` contains the implement of FATranslator.
* `model_build.py` contains the SDFNet model.
* `run.py` is the executable python file with possible arguments.
* `experiment_cfg.py` is the core file for model training, validating, and testing. 
* `configs.py` is the parameter configuration.
* `TDFL.py` contains the implement of 3DFL.
##  Metric Validation:
We firstly introduce a novel objective metric called 3D frequency loss (3DFL) based on 3D fast Fourier transform (3DFFT). As a DNN model-free and training-free metric, 3DFL provides an objective and rational approach to evaluate the similarity and absolute distance between videos. We provide visual comparisons of several traditional metrics on the KTH dataset. 

<!-- ![](/Multimedia_Files/1Metric_Vaildation/metric1.gif "") <img src="" width="50%" height="50%"> -->
<img src="/Multimedia_Files/1Metric_Vaildation/metric1.gif" width = 60%>

This validation demonstrates that 3DFL, as a metric measuring absolute errors, has perceptual capabilities similar to LPIPS for assessing natural video quality, confirming its effectiveness as a reliable metric for evaluating video prediction performance. You can find more examples in the `Multimedia_Files/1Metric_Vaildation` folder.

## Model Preparation

### 1. Environment install
We provide the environment requirements file for easy reproduction:
```
  conda create -n SDFNet python=3.7
  conda activate SDFNet

  pip install -r requirements.txt
```
### 2. Dataset download

Our model has been experimented on the following four datasets:
* [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/)
* [KTH](https://www.csc.kth.se/cvap/actions/)
* [Human3.6M](http://vision.imar.ro/human3.6m/description.php) 


We provide a download script for the Moving MNIST dataset:

```
  cd ./data/moving_mnist
  bash download_mmnist.sh 
```

### 3. Model traning

This example provide the detail implementation on Moving MNIST, you can easily reproduce our work using the following command:

```
conda activate SDFNet
python run.py             
```
Please note that __the model training must strictly adhere to the hyperparameter settings provided in our paper__; otherwise, reproducibility may not be guaranteed.

## Result:

SDFNet predicts more accurate actions with less motion blurring compared to other models. Here are some qualitative visualization examples:

### KTH:
More examples are available at `Multimedia_Files/2KTH_Visualization` folder.
![](/Multimedia_Files/2KTH_Visualization/KTH_example1.gif "")

### Human3.6M:
More examples are available at `Multimedia_Files/3Human_Visualization` folder.
![](/Multimedia_Files/3Human_Visualization/Human_example1.gif "")


