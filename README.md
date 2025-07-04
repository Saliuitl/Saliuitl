## OVERVIEW (YouTube)
[![Watch the video, width=10](https://img.youtube.com/vi/cuay8lF8CCw/maxresdefault.jpg)](https://www.youtube.com/watch?v=cuay8lF8CCw)
## DESCRIPTION
This repository contains the PyTorch implementation of Saliuitl, in particular it follows the proposed DBSCAN-based implementation in the paper, it is possible to customize the code to change the different components of Saliuitl, we will add examples in the future. Note that for simplicity, the attacked subsets of Pascal VOC and INRIA in this repository use the patch attack used during training in the paper. We will add code to create the patch scenarios using the attack we use for evaluation (or any other) soon. 

![saliu_image](https://github.com/user-attachments/assets/e52f6886-8af2-4c42-8dc9-e5e0216a5861)

Saliuitl means "ideological shift" in Nahuatl. The name is inspired by the shift of attrbiutes across ensembles which we use to detect adversarial patches.

## STRUCTURE
The code is organized as follows:

directories:
cfg: contains cfg files for possible object detection victim models (YOLOv2 by default).

checkpoints: contains the weights for the AD attack detector net. The weights for ResNet50 on CIFAR-10 should be placed in this folder.

data: contains the folder structure required to run Saliuitl on the datasets introduced in the paper. Due to the file size, only a small amount of examples are included for each dataset.

nets: torch models for AD and ResNet50.

utils: contains utils.py, a file with helper functions of object detection.

weights: the weights for YOLOv2 should be placed in this folder.

cfg.py, darknet.py, region_loss.py: files required to run YOLOv2 using PyTorch.
helper.py: various helper functions to perform object detection.
saliuitl.py: main code to run and evaluate attack detection and/or recovery using Saliuitl

Our code is based on the following publicly available repositories:

https://github.com/Zhang-Jack/adversarial_yolo2

https://github.com/inspire-group/PatchGuard/tree/master

To run attacks on CIFAR-10 it is necessary to download the resnet50_192_cifar.pth file from https://github.com/inspire-group/PatchGuard/tree/master and place it in the checkpoints folder.

To run attacks on INRIA and Pascal VOC it is necessary to follow the instructions on https://github.com/Zhang-Jack/adversarial_yolo2 to download the yolo.weights file into the weights folder.

## EXAMPLE COMMANDS
Run Saliuitl on effective rectangular single-patch attacks on INRIA using default settings (the "unsuccesful attacks" in the output refer to the recovery rate):
```
python saliuitl.py --inpaint biharmonic --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --inpainting_step 5 --effective_files effective_1p.npy --n_patches 1
```

For double patches:
```
python saliuitl.py --inpaint biharmonic --imgdir data/inria/clean --patch_imgdir data/inria/2p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --inpainting_step 5 --effective_files effective_2p.npy --n_patches 2
```

For triangular patches:
```
python saliuitl.py --inpaint biharmonic --imgdir data/inria/clean --patch_imgdir data/inria/trig --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --inpainting_step 5 --effective_files effective_1p.npy --n_patches 1
```

For multi-object patches (the "n_patches" argument is just relevant to name saved results):
```
python saliuitl.py --inpaint biharmonic --imgdir data/inria/clean --patch_imgdir data/inria/mo --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --inpainting_step 5 --effective_files effective_mop.npy --n_patches 7
```

To evaluate lost predictions, use the clean counterpart of each attack by adding the "--clean" flag to the above commands (the "succesful attacks" in the output refer to the lost prediction rate), for example:
```
python saliuitl.py --inpaint biharmonic --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --inpainting_step 5 --effective_files effective_1p.npy --n_patches 1 --clean
```

To evaluate inflicted attacks, use ineffective attacks by adding the "--uneffective" flag to the above commands (the "succesful attacks" in the output refer to the inflicted attack rate), for example:
```
python saliuitl.py --inpaint biharmonic --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --inpainting_step 5 --effective_files effective_1p.npy --n_patches 1 --uneffective
```

"--ensemble_step" and "--inpainting_step" refer to the size of the set of saliency thresholds used for attack detection and recovery, respectively. The maximum size of the set in the code is 100, which corresponds to a step of 1 (hence these parameters should be in the range 1-100).
Thus, a step 5 corresponds to the default 20 threshold set in the paper. To use, e.g., a set of size 50 run:
```
python saliuitl.py --inpaint biharmonic --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_2_atk_det.pth --det_net 2dcnn_raw --ensemble_step 2 --inpainting_step 2 --effective_files effective_1p.npy --n_patches 1
```

Different ensemble sizes can be used for detection and recovery. For example, now using the size 50 only for recovery, and a size of 10 for detection run:
```
python saliuitl.py --inpaint biharmonic --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_10_atk_det.pth --det_net 2dcnn_raw --ensemble_step 10 --inpainting_step 2 --effective_files effective_1p.npy --n_patches 1
```
Note that the "ensemble_step" determines the size of the ensmeble used for detection, and thus it also indicates which saved weightfile must be used on the attack detector AD.
The weightfile of AD will also depend on the dataset/task, for example, to run the Saliuitl configuration from the command above on ImageNet, for attacks with four rectangular patches:
```
python saliuitl.py --inpaint biharmonic --imgdir data/imagenet/clean --patch_imgdir data/imagenet/4p --dataset imagenet --det_net_path checkpoints/final_classification/2dcnn_raw_imagenet_10_atk_det.pth --det_net 2dcnn_raw --ensemble_step 10 --inpainting_step 2 --effective_files effective_1p.npy --n_patches 1
```

Refer to saliuitl.py for further customization options.

## TRAINING
While we include the weights for trained instances of AD on all datasets, we also provide the training set feature maps we use to train AD. Note these can be used to also train alternative models, and one may also extract alternative attributes for the ensemble attribute vector. We exclude the Pascal VOC feature maps due to file size, but one may extract such feature maps for any dataset based on the main code for Saliuitl, since we adapt vicitm models to return the feature maps we use for detection and recovery.

Example commands to train AD using the default ensemble size of 20 and the DBSCAN-based attributes used for our evaluation are added below:

Train AD for INRIA (YOLOv2 victim model):
```
python train_attack_detector.py --feature_maps net_train_data/inria/1p_train_fms.npy --adv_feature_maps net_train_data/inria/1p_train_pfms.npy
```
Train AD for ImageNet (ResNet-50 victim model):
```
python train_attack_detector.py --feature_maps net_train_data/imagenet/1p_train_fms.npy --adv_feature_maps net_train_data/imagenet/1p_train_pfms.npy
```
Train AD for CIFAR-10 (ResNet-50 victim model):
```
python train_attack_detector.py --feature_maps net_train_data/cifar/1p_train_fms.npy --adv_feature_maps net_train_data/cifar/1p_train_pfms.npy
```

Weights will be saved to the checkpoints folder.
