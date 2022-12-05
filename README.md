# NeckFrac: Deep Learning Powered Cervical Spine Fracture Detector 
## Quicker, better, more accurate diagnosis to save lives.
### Fengyao Luo, Weijia Li, Jane Hung, Minjie Xu
"NeckFrac", a deep learning model trained on over 2000 patients Cervical Spine CT scans, aimed to quickly detect and determine the location of vertebral fractures, which is essential to prevent neurologic deterioration and paralysis after trauma.

[Presentation] 

[Website](https://groups.ischool.berkeley.edu/NeckFrac/)

[Demo]

## Goal
- Build an end to end data pipeline to identify cervical spine fracture probabality
- Train models with over 2000 patients CT scans (343GB)

## Abstract

NeckFrac is an intricate deep learning model dedicated to cervical spine fracture detection, trained on over 2000 patents,  and ensembled the EffNet and 2.5D UNet + bi-GRU model to predict the probability of bone fracture. The training pipeline starts with loading data, feature engineering: Hounsfield units (HU) rescale, pixel normalization, image augmentation (flip, rotate, noise, affine), train models, evaluation performance, and inference. After iteration of training, our model achieve accuracy **%, recall **%, and precision **%. As the false negative is more fatal to patients, we prioritize the recall over precision, thus we introduce the F2 metric to evaluate the performance, which is  **%. 

## Dataset

**Public data provided by RSNA (Radiological Society of North America)** [LINK](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection)

* Collected imaging data sourced from 12 sites on six continents.

* Includes over 2,000 CT studies with diagnosed cervical spine fractures and an approximately equal number of negative studies.

* Data size (343 GB), each patient has around 300 images for CT scans.

* Experts labeled the annotation to indicate the presence, vertebral level and location of any cervical spine fractures.

## Pipeline

## Model 1:

## Model 2:

## Evaluation Result:




