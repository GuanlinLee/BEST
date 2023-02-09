# Official Codes for Extracting Robust Models with Uncertain Examples (ICLR2023) [[pdf](https://openreview.net/forum?id=cMAjKYftNwx)]


### Requirements
1. pytorch >= 1.9.0
2. torchvision
3. numpy
4. tqdm
5. torchattacks

### Prepare Dataset

``python split_set.py --dataset [cifar10, cifar100] 
--num 5000 --class_num [10, 100]``


### Victim Model and Pretrained Model

put your victim model into folder 
``./models/[cifar10, cifar100]/``

put and rename your pretrained model into folder 
``./pretrained/tiny/[mobilenet, resnet, vgg, wrn]_pretrained/[mobilenet, resnet, vgg, wrn].pkl``

We provide some pre-trained model on Tiny-ImageNet, you can find them in this [[repo](https://github.com/GuanlinLee/Tiny-ImageNet-Pretrained-Model)].

### Run Model Extraction Attack

``python extraction_attack.py --arch the architecture of the victim model
-- ext the architecture of the pretrained model
--dataset [cifar10, cifar100] --num 5000 --class_num [10, 100]
--save1 the checkpoint name for the victim model
--save the name you want to save your extracted model
--exp experiment name
--method BEST
--aug [0,1]``


### Citation

If you find our work useful, please cite it:
```
@inproceedings{
li2023extracting,
title={Extracting Robust Models with Uncertain Examples},
author={Guanlin Li and Guowen Xu and Shangwei Guo and Han Qiu and Jiwei Li and Tianwei Zhang},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=cMAjKYftNwx}
}
```



