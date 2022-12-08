# NeckFrac: Deep Learning Powered Cervical Spine Fracture Detector 
## Quicker, better, more accurate diagnosis to save lives.
### Fengyao Luo, Weijia Li, Jane Hung, Minjie Xu
"NeckFrac", a deep learning model trained on over 2000 patients Cervical Spine CT scans, aimed to quickly detect and determine the location of vertebral fractures, which is essential to prevent neurologic deterioration and paralysis after trauma.

Presentation 
* [Research Problem & EDA] 
* [Image Preprocess & Augmentation] 
* Final Ensembled Model

[Website](https://groups.ischool.berkeley.edu/NeckFrac/)

[Demo] 

Please refer to this [repo](https://github.com/janehung04/neckfrac-streamlit-app) for app development code.

## Goal
- Build an end to end data pipeline to identify cervical spine fracture probabality
- Train models with over 2000 patients CT scans (343GB)

## Abstract

NeckFrac is an intricate deep learning model dedicated to cervical spine fracture detection, trained on over 2000 patents,  and ensembled the EffNet and 2.5D UNet + bi-GRU model to predict the probability of bone fracture. The training pipeline starts with loading data, feature engineering: Hounsfield units (HU) rescale, pixel normalization, image augmentation (flip, rotate, noise, affine), train models, evaluation performance, and inference. After iteration of training, our model achieved sensitivity 98%. As the false negative is more fatal to patients, we prioritize the recall over precision, thus we introduce the F2 metric to evaluate the performance, which is 80%. For the business impact,  our model reached higher (+5%) sensitivity compared to radiologists and higher (+22%) sensitivity compared to the AIdoc cervical spine fracture detector model. Most importantly, our model can return the result within 2 mins while the AIdoc model needs 3-8 mins, which beats the average in the industry. 

## Dataset

![Image of Dataset](https://github.com/janehung04/UCB-w210-2022-fall-neckfrac/blob/master/Image/dataset.jpg)

**Public data provided by RSNA (Radiological Society of North America)** [LINK](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection)

* Collected imaging data sourced from 12 sites on six continents.

* Includes over 2,000 CT studies with diagnosed cervical spine fractures and an approximately equal number of negative studies.

* Data size (343 GB), each patient has around 300 images for CT scans.

* Experts labeled the annotation to indicate the presence, vertebral level and location of any cervical spine fractures.

## Pipeline

## Model 1 

![Model1](https://github.com/janehung04/UCB-w210-2022-fall-neckfrac/blob/master/Image/model_1.jpg)

## Model 2

![Model2](https://github.com/janehung04/UCB-w210-2022-fall-neckfrac/blob/master/Image/model_2.jpg)

## Evaluation Result


## Business Impact


|  | Recall / Sensitivity | Inference Time |
|:-----|:--------:|:------:|
| Radiologist   | 93% | 33-43 min |
|  AIDOC model  | 76%   |   3-8 min |
| NeckFrac   | 98% |    2 min |

Compared to Small et al. (2021), we outperform radiologists and an industry competitor (AIDOC) on recall/sensitivity and inference time. Our model would be best utilized in situations where the radiologist would like to sift through predicted negatives quickly and focus on predicted positives.

