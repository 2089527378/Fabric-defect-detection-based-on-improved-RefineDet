
# Fabric defect detection based on improved RefineDet
### Table of Contents
- <a href='#introduction'>Introduction</a>
- <a href='#data augmentation'>Data Preparation</a>
- <a href='#installation'>Installation</a>
- <a href='#training'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#test result'> Test results</a>
- <a href='#future work'>Future work</a>


&nbsp;
&nbsp;
&nbsp;
&nbsp;
## Introduction
The original RefineDet mainly based on [https://github.com/luuuyi/RefineDet.PyTorch](https://github.com/luuuyi/RefineDet.PyTorch)

For the actual problems in the field of fabric defect detection, we have made the following improvements:
-  Various attention mechanism blocks
   * [x] CAM (Optional)
   * [x] SAM (Optional)
   * [x] CBAM (Optional)
   * [x] SE (Optional)
-  Mosaic data augmentation 
-  Bottom-up augmentation TCB
-  Cosine annealing scheduler
-  Various activation function
   * [x] Mish (Optional)
   * [x] Swish (Optional)
- Various regression loss function
   * [x] GIoU loss (Optional)
   * [x] DIoU loss (Optional)
   * [x] CIoU loss (Optional)
- Label smoothing (Optional)

## Data Preparation

##### TILDA dataset 
download from [ https://lmb.informatik.uni-freiburg.de/resources/datasets/tilda.en.html](https://lmb.informatik.uni-freiburg.de/resources/datasets/tilda.en.html)
##### Hong Kong dataset
download from [ https://ytngan.wordpress.com/codes/]( https://ytngan.wordpress.com/codes/)
##### DAGM2007 dataset
download from [ https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learningindustrial-optical-inspection]( https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learningindustrial-optical-inspection)

## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
  * Note: You should use at least PyTorch 0.4.0
- Clone this repository.
  * Note: Only support Python 3+.
- Annotate defective data set data and Generate VOC format (/data/XXXXdatasets/VOCdevkit/VOC2007/JPEGImages and /data/XXXXdatasets/VOCdevkit/VOC2007/Annotations)
  * Note: Install [labelImg](https://github.com/tzutalin/labelImg) to label data
- Divide training set and data set (/data/XXXXdatasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt and /data/XXXXdatasets/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt)
  * Note: The ratio of training set to test set can be from 6:4 to 8:2

## Training
- Download the VGG-16 pretrained weight file from [ https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth]( https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth)
- Edit the key ##num_class and ##max_iter in data/config.py
- Edit the path of dataset and all name of defect class  in data/voc0712.py
- Run train.py 

## Evaluation
- Edit the file path of trained weight model in test.py
- Run test.py

## Test result
See file "test results" of my experiments for detail

## Future work
- Still to come:
  * [ ] Weakly supervised object detection branch
  * [ ] MixUp, CutOut and CutMix data augmentation method
